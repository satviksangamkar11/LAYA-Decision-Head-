"""[EL-ARCH-1 F2d / Phase 2] TRUSTED VERIFIER: verdict admission outside the candidate's process. Engine version label `trusted_verify_v1` (new; the old runner path and every frozen hash / replay of old
runs are untouched: this module is used only when a backend is constructed with it).

The weakness it removes (provider_engine.py header): the portable runner imported the candidate into ITS OWN process and read results from a file / marker the same process wrote, so hostile code could
forge the result file, the marker and its checksum. Design here:

  * candidate code runs ONLY inside the untrusted sandbox, ONE FRESH PROCESS PER INPUT (python: `python3 wrap.py` on one staged input file; java: one `java -cp . Runner < in.json` per input file, stdin
    staged as a file). No state, file or handle is shared between inputs, and no later input can see or change an earlier one.
  * the supervisor (this module, host side) NEVER trusts a file the candidate's workspace could have written: python runs use persist=False (nothing is read back from the workspace); java reads class
    files back only after a compile step and re-stages them into fresh workspaces next to host-compiled Runner / Json classes. What it reads is the sandbox port's own capture of the process: raw exit
    status, killed flag, captured stdout / stderr.
  * pass / fail is decided HERE, host side: the candidate only emits a raw OUTPUT value for an input; the supervisor builds the same per-case records the old runner produced and hands them to the
    unchanged engine.judge_each / engine.observe, which compare with the expected values. The expected values (and any grading logic) are never passed to this module, never staged, never on disk
    inside a sandbox workspace: run_python / run_java take INPUTS only.
  * a candidate-written PASS, result marker, JSON report, checksum or exit message is never proof: exit status != 0 or killed overrides any claimed output; a claimed output is only a value that is
    compared with the expected value; more than one result frame, a missing frame or oversized output is invalid_output, never a pass.
  * compile errors and ordinary test failures stay valid candidate outcomes (compile_error / wrong_answer / runtime_error); sandbox infrastructure failures (the sandbox port raises, or the toolchain
    itself cannot start) are raised as TrustedVerifyInfraError (`infra = True`), never a pass and never a candidate failure.
  * every sandbox call's raw record (the 8 contract fields) is kept for the receipts (`drain_exec_records`), plus per-input observations (exit code, killed, output sha256 and length, seconds).

Residual limits, stated: a hostile candidate can still emit ANY value for the input it is running (that is its answer; the expected value is not visible to it, but public examples are in its prompt by
design); the sandbox boundary (no network, no host files, Job Object / cgroup limits) is the backend's, tested by the backend security gates; one launch per input costs time.

Call site (provider_engine.ProviderExecutionBackend._engine_run):
    if self.trusted_verifier is not None:                       # engine version label TrustedVerifier.VERSION goes into backend_info / receipts
        res = self.trusted_verifier.run(language, raw, inputs)  # -> engine.RunResult; verdict code (judge_each / observe) is unchanged
        self.exec_log.extend(self.trusted_verifier.drain_exec_records(...))
        return res
"""
from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import config as K
from . import engine

VERSION = "trusted_verify_v1"
TAG = "@@TV@@"          # result frame prefix on stdout; NOT a secret and NOT proof: every line is just a claim of the candidate's process
RESULT_KEYS = ("exit_code", "killed", "stdout", "stderr", "protected_unchanged", "pre_tree_sha256", "post_tree_sha256", "persisted")

PY_WRAPPER = '''import importlib.util, json, sys
TAG = sys.argv[1]
try:
    spec = importlib.util.spec_from_file_location("candidate", "cand.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.solve
except BaseException as e:
    sys.stdout.write("\\n" + TAG + " F " + json.dumps({"fatal": type(e).__name__, "message": str(e)[:200]}) + "\\n")
    sys.stdout.flush()
    sys.exit(1)
try:
    data = json.load(open("in.json", encoding="utf-8"))
    value = fn(data)
    s = json.dumps(value, ensure_ascii=False)
except BaseException as e:
    sys.stdout.write("\\n" + TAG + " E " + type(e).__name__ + "\\n")
    sys.stdout.flush()
    sys.exit(2)
sys.stdout.write("\\n" + TAG + " V " + s + "\\n")
sys.stdout.flush()
sys.exit(0)
'''


STUB_SOLUTION = "import java.util.*;\npublic final class Solution {\n    // HOST-STUB: only so the host's own Runner compiles; never used\n    public static Object solve(Map<String,Object> data) { return null; }\n}\n"


class TrustedVerifyInfraError(RuntimeError):
    """the sandbox port failed (not the candidate): never a verdict on a candidate"""
    infra = True


@dataclass
class JavaCommands:
    """the per-call command strings; the backend dialect may override them (WSLC: sh available). Workspace = cwd."""
    compile: str = "javac -proc:none -encoding UTF-8 -d . Json.java Runner.java Solution.java"
    compile_trusted: str = "javac -proc:none -encoding UTF-8 -d . Json.java Runner.java Solution.java"      # Runner references Solution: a host-written stub is compiled with them, its class is discarded
    run: str = "sh -c {script}"      # {script} = quoted `java ... -cp . Runner < in.json`


class TrustedVerifier:
    VERSION = VERSION

    def __init__(self, run: Callable[..., Dict[str, Any]], cfg: Optional[K.SandboxConfig] = None, *, scratch: Optional[Path] = None, java: Optional[JavaCommands] = None,
                 per_input_timeout: Optional[float] = None, total_timeout: Optional[float] = None, java_support: Optional[Path] = None):
        """run(workspace: str, cmd: str, persist: bool = False, timeout: int = 60) -> dict with the 8 contract keys (the sandbox port). Expected values are never an argument anywhere."""
        self._run = run
        self.cfg = cfg or K.SandboxConfig()
        self.scratch = Path(scratch or tempfile.mkdtemp(prefix="el-trusted-")).resolve()
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.java = java or JavaCommands()
        self.java_support = Path(java_support or K.JAVA_SUPPORT)
        self.per_input_timeout = float(per_input_timeout or self.cfg.py_timeout)
        self.total_timeout = float(total_timeout or self.cfg.py_timeout * 2)
        self.records: List[Dict[str, Any]] = []         # raw sandbox results (8 keys) + what the supervisor staged
        self.observations: List[Dict[str, Any]] = []    # per input: exit code, killed, output sha256 / length, seconds
        self._trusted_java: Optional[Dict[str, bytes]] = None
        self.wrapper_sha256 = hashlib.sha256(PY_WRAPPER.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- plumbing
    def _call(self, wd: Path, cmd: str, persist: bool, timeout: float, role: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            r = self._run(str(wd.resolve()), cmd, persist=persist, timeout=int(max(1, round(timeout))))
        except KeyboardInterrupt:
            raise
        except BaseException as e:      # noqa: BLE001  the port raises on infrastructure failure
            raise TrustedVerifyInfraError("sandbox.run failed (%s): %s: %s" % (role, type(e).__name__, str(e)[:300])) from e
        miss = [k for k in RESULT_KEYS if k not in r]
        if miss:
            raise TrustedVerifyInfraError("sandbox result lacks the contract fields %s" % miss)
        self.records.append({k: r[k] for k in RESULT_KEYS} | {"role": role, "engine_version": VERSION, "seconds": round(time.time() - t0, 3)})
        return r

    def drain_exec_records(self) -> List[Dict[str, Any]]:
        out, self.records = self.records, []
        return out

    def drain_observations(self) -> List[Dict[str, Any]]:
        out, self.observations = self.observations, []
        return out

    def run(self, language: str, raw: bytes, inputs: Sequence[Any]) -> engine.RunResult:
        return self.run_python(raw, inputs) if language == "python" else self.run_java(raw, inputs)

    def _obs(self, i: int, r: Dict[str, Any], dt: float, note: str = "") -> None:
        so = str(r.get("stdout", ""))
        self.observations.append({"input": i, "exit_code": r["exit_code"], "killed": bool(r["killed"]), "stdout_len": len(so.encode("utf-8", "replace")),
                                  "stdout_sha256": hashlib.sha256(so.encode("utf-8", "replace")).hexdigest(), "stderr_tail_sha256": hashlib.sha256(str(r.get("stderr", ""))[-1500:].encode("utf-8", "replace")).hexdigest(),
                                  "seconds": round(dt, 3), "note": note})

    # ---------------------------------------------------------------- frame parsing: every line is a claim; only the count and the exit status are structural
    @staticmethod
    def _frames(stdout: str) -> List[str]:
        return [l[len(TAG) + 1:] for l in stdout.splitlines() if l.startswith(TAG + " ")]

    # ---------------------------------------------------------------- python
    def run_python(self, raw: bytes, inputs: Sequence[Any]) -> engine.RunResult:
        cfg = self.cfg
        if len(raw) > cfg.max_source_bytes:
            return engine.RunResult("oversized", detail="source %d bytes > %d" % (len(raw), cfg.max_source_bytes))
        lines: List[str] = []
        res = engine.RunResult("ran", returncode=0)
        t_all = time.time()
        stderr_tail = ""
        for i, inp in enumerate(inputs):
            if time.time() - t_all > self.total_timeout:
                res.phase_status, res.timed_out = "timeout", True
                res.detail = "total wall budget %.0fs used before input %d" % (self.total_timeout, i)
                break
            wd = self.scratch / "py" / uuid.uuid4().hex
            wd.mkdir(parents=True)
            try:
                (wd / "cand.py").write_bytes(raw)
                (wd / "wrap.py").write_bytes(PY_WRAPPER.encode("utf-8"))                          # fresh host-written copy for EVERY input: editing it only hurts the candidate's own workspace
                (wd / "in.json").write_bytes(json.dumps(inp, ensure_ascii=True).encode("ascii"))
                t0 = time.time()
                r = self._call(wd, "python3 wrap.py %s" % TAG, False, self.per_input_timeout, "python_input_%d" % i)          # persist=False: NOTHING is read back from the workspace
                dt = time.time() - t0
            finally:
                shutil.rmtree(wd, ignore_errors=True)
            self._obs(i, r, dt)
            stderr_tail = str(r.get("stderr", ""))[-1500:]
            res.duration += dt
            so = str(r.get("stdout", ""))
            if r["killed"]:
                res.phase_status, res.timed_out, res.returncode, res.stderr_tail = "timeout", True, r["exit_code"], stderr_tail
                break
            if len(so.encode("utf-8", "replace")) > cfg.max_output_bytes:
                res.overflow, res.returncode, res.stderr_tail = True, r["exit_code"], stderr_tail
                break
            fr = self._frames(so)
            code = int(r["exit_code"])
            if len(fr) != 1:                                  # no frame (candidate exited early / crashed) or several (forgery attempt / ambiguity): never a value
                res.returncode, res.stderr_tail = code if code != 0 else 1, stderr_tail
                if len(fr) > 1:
                    lines.append("not-json: ambiguous result frames")      # invalid_line -> invalid_output
                break
            kind, _, body = fr[0].partition(" ")
            if kind == "F":
                lines.append(body)                                 # {"fatal": name, "message": ...}: SyntaxError -> compile_error, others -> runtime_error (judge decides; never a pass)
                res.returncode, res.stderr_tail = code if code != 0 else 1, stderr_tail
                break
            if code != 0 and kind != "E":                         # nonzero exit overrides any claimed value
                res.returncode, res.stderr_tail = code, stderr_tail
                break
            if kind == "E":
                lines.append(json.dumps({"ok": False, "error": body.strip()[:80]}))
                continue
            if kind == "V" and code == 0:
                try:
                    value = json.loads(body)
                except ValueError:
                    lines.append("not-json: unparsable result frame")
                    break
                lines.append(json.dumps({"ok": True, "value": value}, ensure_ascii=False))
                continue
            lines.append("not-json: unknown result frame")
            break
        res.stdout = "".join(l + "\n" for l in lines)
        res.stderr_tail = res.stderr_tail or stderr_tail
        return res

    # ---------------------------------------------------------------- java
    def _trusted_classes(self) -> Dict[str, bytes]:
        """compile the host's own Json.java / Runner.java ONCE in a candidate-free workspace; these class bytes are what every run stage uses (the candidate never gets to influence them)"""
        if self._trusted_java is not None:
            return self._trusted_java
        wd = self.scratch / "javaprep" / uuid.uuid4().hex
        wd.mkdir(parents=True)
        try:
            for f in ("Json.java", "Runner.java"):
                shutil.copyfile(self.java_support / f, wd / f)
            (wd / "Solution.java").write_text(STUB_SOLUTION, encoding="utf-8")
            r = self._call(wd, self.java.compile_trusted, True, self.cfg.java_compile_timeout, "java_prep")
            if r["killed"] or int(r["exit_code"]) != 0 or not r.get("persisted"):
                raise TrustedVerifyInfraError("the toolchain cannot compile the host's own support classes (exit %s, killed %s): %s" % (r["exit_code"], r["killed"], str(r.get("stderr", ""))[-300:]))
            classes = {p.name: p.read_bytes() for p in wd.glob("*.class") if not p.name.startswith("Solution")}      # the stub is never used
            if "Runner.class" not in classes:
                raise TrustedVerifyInfraError("javac produced no Runner.class for the support sources")
            self._trusted_java = classes
            return classes
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    def run_java(self, raw: bytes, inputs: Sequence[Any]) -> engine.RunResult:
        cfg = self.cfg
        if len(raw) > cfg.max_source_bytes:
            return engine.RunResult("oversized", detail="source %d bytes > %d" % (len(raw), cfg.max_source_bytes))
        trusted = self._trusted_classes()
        # 1. compile the candidate (raw exit status and compiler output come from the sandbox port's capture of the javac process)
        wd = self.scratch / "java" / uuid.uuid4().hex
        wd.mkdir(parents=True)
        cand_classes: Dict[str, bytes] = {}
        try:
            (wd / "Solution.java").write_bytes(raw)
            for f in ("Json.java", "Runner.java"):
                shutil.copyfile(self.java_support / f, wd / f)
            t0 = time.time()
            r = self._call(wd, self.java.compile, True, cfg.java_compile_timeout, "java_compile")
            dt = time.time() - t0
            self._obs(-1, r, dt, "compile")
            res = engine.RunResult("ran", returncode=int(r["exit_code"]), stderr_tail=str(r.get("stderr", ""))[-1500:], duration=dt)
            if r["killed"]:
                res.phase_status, res.timed_out, res.detail = "timeout", True, "compile timeout"
                return res
            if int(r["exit_code"]) != 0:
                res.phase_status = "compile_error"                  # an ordinary candidate outcome: javac said no
                return res
            if not r.get("persisted"):
                res.phase_status, res.detail = "sandbox_error", "the compile step persisted nothing (class files unavailable)"
                return res
            cand_classes = {p.name: p.read_bytes() for p in wd.glob("*.class") if not (p.name.startswith("Runner") or p.name.startswith("Json"))}
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        if not cand_classes:
            return engine.RunResult("compile_error", detail="javac exited 0 but produced no Solution class", returncode=0)
        # 2. one process per input; stdin staged as a file; host-compiled Runner / Json classes re-staged fresh next to the candidate's classes
        lines: List[str] = []
        res = engine.RunResult("ran", returncode=0)
        t_all = time.time()
        script = "java %s -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Dstdin.encoding=UTF-8 -cp . Runner < in.json" % cfg.java_xmx
        for i, inp in enumerate(inputs):
            if time.time() - t_all > max(self.total_timeout, cfg.java_run_timeout * 2):
                res.phase_status, res.timed_out, res.detail = "timeout", True, "total wall budget used before input %d" % i
                break
            wd = self.scratch / "java" / uuid.uuid4().hex
            wd.mkdir(parents=True)
            try:
                for n, b in trusted.items():
                    (wd / n).write_bytes(b)
                for n, b in cand_classes.items():
                    (wd / n).write_bytes(b)
                (wd / "in.json").write_bytes((json.dumps(inp, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8", "replace"))
                t0 = time.time()
                r = self._call(wd, self.java.run.format(script=shlex.quote(script)), False, cfg.java_run_timeout, "java_input_%d" % i)
                dt = time.time() - t0
            finally:
                shutil.rmtree(wd, ignore_errors=True)
            self._obs(i, r, dt)
            res.duration += dt
            so = str(r.get("stdout", ""))
            res.stderr_tail = str(r.get("stderr", ""))[-1500:]
            if r["killed"]:
                res.phase_status, res.timed_out, res.returncode = "timeout", True, r["exit_code"]
                break
            if len(so.encode("utf-8", "replace")) > cfg.max_output_bytes:
                res.overflow, res.returncode = True, r["exit_code"]
                break
            body = [l for l in so.splitlines() if l.strip()]
            code = int(r["exit_code"])
            if code != 0:                                           # nonzero exit overrides whatever was printed
                res.returncode = code
                break
            if len(body) != 1:
                res.returncode = 1
                lines.append("not-json: expected exactly one output line, got %d" % len(body))
                break
            lines.append(body[0])
        res.stdout = "".join(l + "\n" for l in lines)
        return res
