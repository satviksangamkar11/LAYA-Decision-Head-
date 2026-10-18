"""[EL-BANK-LOCAL] Bank execution engine over the SANDBOX PROVIDER PORT (evidence_ladder/controller/sandbox_provider.py: make_sandbox(...).run(workspace, cmd, persist, timeout)), so the local bank run
uses the new MXC sandbox (Windows processcontainer) instead of WSL bubblewrap. Same contract and result shapes as SandboxExecutionBackend (run_public / run_experiment / backend_info); the verdict logic
(engine._parse / judge / judge_each / observe, strict comparisons) is the SAME code, only the process launch is replaced.

Per call: a private scratch workspace holds the candidate bytes (cand.py), a portable runner (runner.py: no `resource`/stdin, inputs come from inputs.jsonl) and nothing else; the command is
`python3 runner.py <output cap>` (MXC dialect: one direct python command). The runner writes its result lines to a randomly named out_<uuid>.jsonl, hashes what it wrote, and prints one marker line on
the real stdout; the parent persists the run workspace back, requires EXACTLY ONE out_*.jsonl whose sha256 / line count equal the marker, and builds the same RunResult the Linux engine would.
Anything else (no marker, two out files, hash mismatch, unreadable file) is a sandbox error, never a pass and never a candidate failure. A killed run (timeout) has no persisted files -> every example times out.
Residual risk, stated: as in the receipted Linux runner_v2.py the candidate shares the runner's process, so deliberately hostile code could forge the result file; MXC contains it (no network, no host files,
Job Object limits) but cannot separate it from its own runner. Model-written candidates are not adversarial; red-team results of the Linux jail do NOT transfer to this engine.

FAIL CLOSED: every provider exception (SandboxUnavailable, UnsupportedCommand, UnsupportedPolicy, SandboxIntegrityError, RuntimeError, OSError, TimeoutExpired) is re-raised as ProviderEngineError
(`infra = True`); Java is NOT runnable here (no JDK reachable inside the container, dialect = python only) and raises JavaNotRunnable; nothing falls back to WSL or to the host."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidence_ladder.common.interfaces import Artifact, CheckOutcome, TaskSpec

from . import config as K
from . import engine
from .artifacts import ArtifactIntegrityError, ArtifactStore
from .backend import EXPERIMENT_CHECK, MAX_PAYLOAD_CHARS, PARSE_CHECK, PUBLIC_CHECK, SUPPORTED_EXPERIMENTS, SandboxExecutionBackend, _js

MARK = "@@OUT@@"
PORTABLE_RUNNER = '''import hashlib, importlib.util, json, os, sys, uuid
cap = int(sys.argv[1])
name = "out_" + uuid.uuid4().hex + ".jsonl"
out = open(name, "w", encoding="utf-8", newline="\\n")
real_stdout = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr
h, n, size = hashlib.sha256(), 0, 0
def emit(obj):
    global n, size
    s = json.dumps(obj, ensure_ascii=False) + "\\n"
    b = s.encode("utf-8")
    if size + len(b) > cap:
        real_stdout.write("@@OVERFLOW@@\\n"); real_stdout.flush(); out.close(); os._exit(3)
    out.write(s); h.update(b); n += 1; size += len(b)
try:
    spec = importlib.util.spec_from_file_location("candidate", "cand.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.solve
except BaseException as e:
    emit({"fatal": type(e).__name__, "message": str(e)[:200]})
    out.close()
    real_stdout.write("@@OUT@@ %s %d %s\\n" % (name, n, h.hexdigest())); real_stdout.flush()
    os._exit(1)
for line in open("inputs.jsonl", encoding="ascii"):
    if not line.strip():
        continue
    try:
        emit({"ok": True, "value": fn(json.loads(line))})
    except BaseException as e:
        emit({"ok": False, "error": type(e).__name__})
out.close()
real_stdout.write("@@OUT@@ %s %d %s\\n" % (name, n, h.hexdigest())); real_stdout.flush()
os._exit(0)
'''


class ProviderEngineError(RuntimeError):
    """The sandbox provider / engine is unavailable or broke (infrastructure failure). Never a verdict on a candidate."""
    infra = True


class JavaNotRunnable(ProviderEngineError):
    """No JDK is reachable inside the MXC container (native Windows executables only; the dialect runs `python3 <args>` only)."""


def java_capability(backend: str = "mxc", which=shutil.which, environ=os.environ) -> Dict[str, Any]:
    """Explicit capability record (never a silent move to another sandbox). mxc: Java is NOT runnable; the reason says which of the two blockers applies."""
    host_jdk = bool(which("javac") or which("java") or environ.get("JAVA_HOME"))
    if backend == "mxc":
        return {"runnable": False, "status": "NOT_TESTED_NO_JDK", "backend": "mxc",
                "reason": ("no JDK on the Windows host (java / javac not found, JAVA_HOME unset) and the MXC container runs native Windows executables only" if not host_jdk
                           else "a host JDK exists but the MXC dialect runs `python3|python <args>` only, so javac / java cannot be started inside the container (UnsupportedCommand)")}
    return {"runnable": True, "status": "RUNNABLE", "backend": backend, "reason": ""}


class ProviderExecutionBackend(SandboxExecutionBackend):
    stub = False

    def __init__(self, sandbox: Any = None, *, backend: str = "mxc", scratch: Optional[Path] = None, store: Optional[ArtifactStore] = None, config: Optional[K.SandboxConfig] = None):
        super().__init__(store, config)
        self.backend_name = backend
        self.scratch = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="el-provider-engine-"))
        self.scratch.mkdir(parents=True, exist_ok=True)
        self._sandbox, self.calls, self.seconds = sandbox, 0, 0.0
        self.python_seconds: List[float] = []
        self.exec_log: List[Dict[str, Any]] = []       # one raw sandbox result (the 8 contract fields) per sandbox.run call; drained per task by the runner

    @property
    def sandbox(self):
        if self._sandbox is None:
            from evidence_ladder.controller.sandbox_provider import make_sandbox
            try:
                self._sandbox = make_sandbox(self.scratch / "_sandbox", [], backend=self.backend_name)      # raises SandboxUnavailable / UnsupportedPolicy: fail closed, no fallback
            except Exception as e:      # noqa: BLE001
                raise ProviderEngineError("sandbox backend %r unavailable: %s: %s" % (self.backend_name, type(e).__name__, str(e)[:300])) from e
        return self._sandbox

    def start(self) -> Dict[str, Any]:
        sb = self.sandbox
        return {"engine": getattr(sb, "env_label", self.backend_name), "backend": self.backend_name}

    # ---------------------------------------------------------------- the one place a process is launched
    def _engine_run(self, language: str, raw: bytes, inputs: List[Any]) -> engine.RunResult:
        if language != "python":
            raise JavaNotRunnable("language %r cannot run on backend %r: %s" % (language, self.backend_name, java_capability(self.backend_name)["reason"]))
        cfg = self.cfg
        if len(raw) > cfg.max_source_bytes:
            return engine.RunResult("oversized", detail="source %d bytes > %d" % (len(raw), cfg.max_source_bytes))
        wd = self.scratch / "calls" / uuid.uuid4().hex
        wd.mkdir(parents=True)
        try:
            (wd / "cand.py").write_bytes(raw)
            (wd / "runner.py").write_text(PORTABLE_RUNNER, encoding="utf-8")
            (wd / "inputs.jsonl").write_bytes("".join(json.dumps(x, ensure_ascii=True) + "\n" for x in inputs).encode("ascii"))
            t0 = time.time()
            try:
                r = self.sandbox.run(str(wd), "python3 runner.py %d" % cfg.max_output_bytes, persist=True, timeout=int(max(1, round(cfg.py_timeout))))
            except ProviderEngineError:
                raise
            except BaseException as e:      # noqa: BLE001  KeyboardInterrupt is re-raised below; everything else is an infrastructure failure
                if isinstance(e, KeyboardInterrupt): raise
                raise ProviderEngineError("sandbox.run failed: %s: %s" % (type(e).__name__, str(e)[:300])) from e
            dt = time.time() - t0
            self.exec_log.append({k: r[k] for k in ('exit_code', 'killed', 'stdout', 'stderr', 'protected_unchanged', 'pre_tree_sha256', 'post_tree_sha256', 'persisted')})
            self.calls += 1; self.seconds += dt; self.python_seconds.append(round(dt, 3))
            out = engine.RunResult("ran", returncode=int(r["exit_code"]), stderr_tail=str(r.get("stderr", ""))[-1500:], duration=dt)
            if r["killed"]:
                out.phase_status, out.timed_out = "timeout", True
                return out
            so = str(r.get("stdout", ""))
            if "@@OVERFLOW@@" in so:
                out.overflow = True
                return out
            marker = next((l for l in reversed(so.splitlines()) if l.startswith(MARK)), None)
            files = sorted(wd.glob("out_*.jsonl"))
            if marker is None and int(r["exit_code"]) != 0:
                return out                       # the candidate process died without reaching the runner's end (hard crash): an ordinary runtime failure with no output lines
            if marker is None or len(files) != 1 or not r.get("persisted"):
                out.phase_status, out.detail = "sandbox_error", "no result marker / ambiguous result file (exit %s, files %d, persisted %s)" % (r["exit_code"], len(files), r.get("persisted"))
                return out
            _, name, n, digest = marker.split()
            data = files[0].read_bytes()
            if files[0].name != name or hashlib.sha256(data).hexdigest() != digest or data.count(b"\n") != int(n):
                out.phase_status, out.detail = "sandbox_error", "result file does not match the runner's marker (tampered or truncated)"
                return out
            try:
                out.stdout = data.decode("utf-8")
            except UnicodeDecodeError:
                out.bad_encoding = True
            return out
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    # ---------------------------------------------------------------- the same two methods as SandboxExecutionBackend, with _engine_run instead of engine.run
    def run_public(self, artifact: Artifact, task: TaskSpec) -> List[CheckOutcome]:
        if artifact.language != task.language or artifact.task_id != task.task_id:
            return [{"check_id": PUBLIC_CHECK, "status": "invalid_spec", "detail": "artifact/task language or task_id mismatch", "failure_class": ""}]
        cases = list(task.public_tests)
        try:
            raw = self._bytes(artifact)
        except ArtifactIntegrityError as e:
            msg = "integrity: " + str(e)
            return [self._err(PARSE_CHECK, msg)] + [{**self._err(PUBLIC_CHECK, msg), "example_index": i} for i in range(len(cases))]
        if not cases:
            return [{"check_id": PUBLIC_CHECK, "status": "missing_evidence", "detail": "task has no public tests", "failure_class": ""}]
        res = self._engine_run(artifact.language, raw, [c["input"] for c in cases])
        v, each = engine.judge_each(artifact.language, res, cases, self.cfg)
        cap = self.cfg.max_observed_chars
        if v.failure_class == "compile_error":
            parse = {"check_id": PARSE_CHECK, "status": "fail", "detail": v.detail, "failure_class": "compile_error", "stderr_tail": v.stderr_tail}
        elif v.failure_class == "sandbox_error":
            raise ProviderEngineError("the sandbox failed to run a candidate: %s" % (res.detail or v.detail))      # infrastructure, not a verdict
        elif v.detail == "compile timeout":
            parse = {"check_id": PARSE_CHECK, "status": "timeout", "detail": v.detail, "failure_class": "timeout"}
        elif res.phase_status == "oversized":
            parse = {"check_id": PARSE_CHECK, "status": "fail", "detail": v.detail, "failure_class": "invalid_output"}
        else:
            parse = {"check_id": PARSE_CHECK, "status": "pass", "detail": "source loaded in the %s sandbox" % self.backend_name, "failure_class": ""}
        parse["duration_s"] = v.duration
        outs: List[CheckOutcome] = [parse]
        for i, (st, cls, obs) in enumerate(each):
            o: CheckOutcome = {"check_id": PUBLIC_CHECK, "example_index": i, "duration_s": v.duration, "stderr_tail": v.stderr_tail}
            if parse["status"] != "pass":
                o.update(status="timeout" if parse["status"] == "timeout" else "missing_evidence", failure_class=parse["failure_class"], detail="not run: " + parse["detail"], observed="")
            else:
                o.update(status=st, failure_class=cls, observed=obs, detail="example %d: %s" % (i, "matched" if st == "pass" else (cls or v.detail)))
                if st != "pass" and obs:
                    c = cases[i]
                    o["counterexample"] = {"input": _js(c["input"], cap), "expected": _js(c["expected"], cap), "observed": obs}
            outs.append(o)
        return outs

    def run_experiment(self, artifact: Artifact, experiment: Dict[str, Any]) -> CheckOutcome:
        def bad(msg: str) -> CheckOutcome:
            return {"check_id": EXPERIMENT_CHECK, "status": "invalid_spec", "detail": msg, "failure_class": ""}
        if not isinstance(experiment, dict):
            return bad("experiment is not an object")
        kind, payload = experiment.get("kind"), experiment.get("payload")
        if kind not in SUPPORTED_EXPERIMENTS:
            return bad("experiment kind %r is not supported in stage 1 (allowed: %s)" % (kind, ", ".join(SUPPORTED_EXPERIMENTS)))
        if not isinstance(payload, str) or len(payload) > MAX_PAYLOAD_CHARS:
            return bad("payload must be a string of at most %d characters" % MAX_PAYLOAD_CHARS)
        try:
            obj = json.loads(payload)
        except ValueError as e:
            return bad("payload is not valid JSON: " + str(e)[:100])
        if not isinstance(obj, dict):
            return bad("payload must be a JSON object (solve takes an object)")
        try:
            raw = self._bytes(artifact)
        except ArtifactIntegrityError as e:
            return self._err(EXPERIMENT_CHECK, "integrity: " + str(e))
        res = self._engine_run(artifact.language, raw, [obj])
        status, cls, observed, detail = engine.observe(artifact.language, res, self.cfg)
        if cls == "sandbox_error":
            raise ProviderEngineError("the sandbox failed to run an experiment: %s" % (res.detail or detail))
        out: CheckOutcome = {"check_id": EXPERIMENT_CHECK, "status": status, "detail": detail, "failure_class": cls, "duration_s": res.duration, "stderr_tail": res.stderr_tail}
        if status in ("pass", "fail"):
            cap = self.cfg.max_observed_chars
            s = json.dumps(observed, ensure_ascii=False)
            out["observed"] = observed if len(s) <= cap else s[:cap] + "...[truncated]"
        return out

    def backend_info(self) -> Dict[str, Any]:
        sb = self._sandbox
        ps = sorted(self.python_seconds)
        return {"engine": getattr(sb, "env_label", self.backend_name) if sb is not None else self.backend_name, "backend": self.backend_name, "calls": self.calls, "sandbox_seconds_total": round(self.seconds, 2),
                "seconds_per_call": {"n": len(ps), "p50": ps[len(ps) // 2] if ps else None, "max": ps[-1] if ps else None},
                "limits_config": {k: getattr(self.cfg, k) for k in ("py_timeout", "max_source_bytes", "max_output_bytes", "max_observed_chars")}, "java": java_capability(self.backend_name),
                "diagnostics": getattr(sb, "last_diagnostics", None),
                "note": "candidate and runner share one process (as in runner_v2.py); Linux red-team results do not transfer; MXC containment + Job Object limits apply"}

    def drain_exec_log(self) -> List[Dict[str, Any]]:
        out, self.exec_log = self.exec_log, []
        return out

    def close(self) -> None:
        c = getattr(self._sandbox, "close", None)
        if c: c()
