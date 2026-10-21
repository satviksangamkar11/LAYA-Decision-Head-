"""[MULTILANG 2026-10-21] Trusted-verifier path for C, C++, JavaScript, Go and Rust (stdin -> stdout programs). ADDITIVE: trusted_verify.py dispatches here ONLY for these five languages; the python and java
paths are untouched. Same guarantees as trusted_verify.py:

  * the candidate runs ONLY inside the untrusted sandbox, ONE FRESH PROCESS PER INPUT. Compiled languages are compiled once (one sandbox call, persist=True, ONLY the named output `prog` is read
    back, never anything else); every input then runs in a fresh workspace that holds the host-written launcher, the staged stdin file `in.txt` and the compiled program (or the JS source).
    Nothing runs the supervisor's logic inside the candidate's process: the launcher replaces itself with the candidate (`execvp`), so the exit status the port reports IS the candidate's.
  * pass / fail is decided HERE, host side. The candidate's stdout is just a value; there are no result frames to forge (a printed `@@TV@@ ...` line is part of the stdout and makes it differ from the
    expected text). A non-zero exit, a kill, an oversized or possibly truncated output is never a pass: claim-then-crash fails on the exit status.
  * the expected values never reach this module's sandbox calls: `run` / `grade` take INPUTS only; `LcbRunner` keeps the expected text on the host.
  * compile errors and wrong answers are candidate outcomes; a toolchain that cannot start in the sandbox (exit 127, command not found) or a port failure is INFRASTRUCTURE and raises.

Input -> stdin: a str input is the stdin text; any other JSON value is staged as its JSON text. Output of `run()` (java-style lines for engine.judge_each): one JSON string per input, the program's stdout.
Comparison for `LcbRunner` is LiveCodeBench's (lcb_runner.stdio_equal); engine.judge_each compares the stdout text exactly (engine.same on strings).

Stated limits: the supervisor does not see more of stdout than the port captures (`capture_limit`: WSL helper and MXC keep the last 6000 characters, the Kaggle jail 1 MiB); an output that reaches the
limit is reported as invalid_output ("possibly truncated"), never compared. Resource limits come from the backend (cgroup / Job Object / jail) plus the launcher's rlimits (cpu, fsize, nofile, nproc;
address space only for C/C++, where it is safe)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import config as K
from . import engine
from . import toolchains as TC

VERSION = "trusted_native_v1"
NATIVE_LANGUAGES = ("c", "cpp", "javascript", "go", "rust")
DEFAULT_CAPTURE_LIMIT = 6000          # WSL helper / MXC / WSLC keep the last 6000 characters of stdout; a port with a bigger capture sets tv.capture_limit
MAX_BINARY_BYTES = 64 << 20

LAUNCHER = '''import os, sys
a = sys.argv[1:]
lim = {}
while a and a[0] != "--":
    k, _, v = a.pop(0).partition("=")
    lim[k] = int(v)
a = a[1:]
if not a:
    sys.exit(0)
try:
    import resource
    for k, r in (("cpu", resource.RLIMIT_CPU), ("fsize", resource.RLIMIT_FSIZE), ("nofile", resource.RLIMIT_NOFILE), ("nproc", resource.RLIMIT_NPROC), ("as", resource.RLIMIT_AS), ("core", resource.RLIMIT_CORE)):
        if k in lim:
            resource.setrlimit(r, (lim[k], lim[k]))
except ImportError:
    pass
fd = os.open("in.txt", os.O_RDONLY)
if os.name == "posix":
    if "/" in a[0]:
        try:
            os.chmod(a[0], 0o755)
        except OSError:
            pass
    os.dup2(fd, 0)
    os.close(fd)
    try:
        os.execvp(a[0], a)
    except OSError as e:
        sys.stderr.write("launcher: exec failed: %s\\n" % e)
        sys.exit(127)
import subprocess
try:
    rc = subprocess.call(a, stdin=os.fdopen(fd, "rb"))
except OSError as e:
    sys.stderr.write("launcher: start failed: %s\\n" % e)
    sys.exit(127)
if rc:
    sys.stderr.write("launcher: child exit %d (0x%08X)\\n" % (rc, rc & 0xFFFFFFFF))
sys.exit(rc if 0 <= rc < 256 else 1)
'''
LAUNCHER_SHA256 = hashlib.sha256(LAUNCHER.encode("utf-8")).hexdigest()
RLIMITS = {"cpu": 30, "fsize": 64 << 20, "nofile": 256, "nproc": 256, "core": 0}
RLIMITS_AS = {"c": 2 << 30, "cpp": 2 << 30}           # address space only where the runtime does not reserve huge virtual ranges (Go, Rust, node do)

PROBE = {"c": "gcc --version", "cpp": "g++ --version", "go": "go version", "rust": "rustc --version", "javascript": "python3 launch.py -- node --version"}


class NativeInfraError(RuntimeError):
    infra = True


def probe_plan(language: str) -> Tuple[Dict[str, bytes], List[str]]:
    """-> (files to stage in the probe workspace, probe commands). JavaScript is probed THROUGH the launcher, the way it runs (the MXC process container starts `python3 <args>` only)."""
    files = {"launch.py": LAUNCHER.encode("utf-8"), "in.txt": b""} if language == "javascript" else {}
    return files, [PROBE[language]]


def _argv(language: str, name: str) -> str:
    return "node %s" % name if language == "javascript" else "./%s" % name


def _limits(language: str) -> str:
    lim = dict(RLIMITS)
    if language in RLIMITS_AS:
        lim["as"] = RLIMITS_AS[language]
    return " ".join("%s=%d" % kv for kv in lim.items())


def stdin_text(inp: Any) -> str:
    return inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False)


@dataclass
class Compiled:
    status: str                            # "ok" | "compile_error" | "timeout" | "oversized"
    name: str = ""                         # the file the run step stages: prog (binary) or cand.js
    data: bytes = b""
    stderr: str = ""
    seconds: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class NativeCore:
    """Both users (TrustedVerifier and LcbRunner) share this. port(wd, cmd, persist, timeout, role) -> the 8-key sandbox result dict; it raises its own infrastructure error."""
    def __init__(self, port: Callable[..., Dict[str, Any]], scratch: Path, cfg: Optional[K.SandboxConfig] = None, infra: type = NativeInfraError):
        self.port, self.scratch, self.cfg, self.infra = port, Path(scratch).resolve(), cfg or K.SandboxConfig(), infra
        self.scratch.mkdir(parents=True, exist_ok=True)

    def _call(self, files: Dict[str, bytes], cmd: str, persist: bool, timeout: float, role: str, executable: Sequence[str] = (), want: Sequence[str] = ()):
        wd = self.scratch / uuid.uuid4().hex
        wd.mkdir(parents=True)
        try:
            for n, b in files.items():
                (wd / n).write_bytes(b)
                if n in executable:
                    try:
                        os.chmod(wd / n, 0o755)
                    except OSError:
                        pass
            r = self.port(wd, cmd, persist, timeout, role)
            got: Dict[str, bytes] = {}
            if persist and r.get("persisted"):
                for n in want:
                    p = wd / n
                    if p.is_file() and not p.is_symlink() and p.stat().st_size <= MAX_BINARY_BYTES:
                        got[n] = p.read_bytes()
            return r, got
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    def compile(self, language: str, raw: bytes) -> Compiled:
        tc = TC.get(language)
        if tc is None or language not in NATIVE_LANGUAGES:
            raise self.infra("language %r has no native runner" % language)
        if len(raw) > self.cfg.max_source_bytes:
            return Compiled("oversized", stderr="source %d bytes > %d" % (len(raw), self.cfg.max_source_bytes))
        src = "cand" + tc.extension
        if tc.compile is None:                                     # javascript: the source itself is staged for every run
            return Compiled("ok", src, raw)
        t0 = time.time()
        r, got = self._call({src: raw}, tc.compile.format(src=src, out="prog"), True, self.cfg.java_compile_timeout, "compile_" + language, want=("prog",))
        dt = time.time() - t0
        err = str(r.get("stderr", ""))[-1500:] + str(r.get("stdout", ""))[-500:]
        if r["killed"]:
            return Compiled("timeout", stderr="compile timeout", seconds=dt, raw=r)
        if int(r["exit_code"]) == 127:
            raise self.infra("toolchain for %r cannot start in this sandbox (exit 127): %s" % (language, err[-200:]))
        if int(r["exit_code"]) != 0:
            return Compiled("compile_error", stderr=err, seconds=dt, raw=r)
        if "prog" not in got:
            if not r.get("persisted"):
                raise self.infra("the compile step persisted nothing (binary unavailable)")
            return Compiled("compile_error", stderr="the compiler exited 0 but produced no program\n" + err, seconds=dt, raw=r)
        return Compiled("ok", "prog", got["prog"], err, dt, r)

    def run_one(self, language: str, c: Compiled, stdin: str, kill: float, role: str) -> Dict[str, Any]:
        """one fresh process: -> the raw port result plus `seconds` (wall, including launch)"""
        files = {"launch.py": LAUNCHER.encode("utf-8"), "in.txt": stdin.encode("utf-8", "replace"), c.name: c.data}
        t0 = time.time()
        r, _ = self._call(files, "python3 launch.py %s -- %s" % (_limits(language), _argv(language, c.name)), False, kill, role, executable=(c.name,) if c.name == "prog" else ())
        r = dict(r)
        r["seconds"] = time.time() - t0
        return r

    def calibrate(self) -> float:
        """seconds of a trivial launcher start (python only, no candidate): the minimum of 2, so candidate time is never over-subtracted"""
        ts = []
        for k in range(2):
            t0 = time.time()
            r, _ = self._call({"launch.py": LAUNCHER.encode("utf-8"), "in.txt": b""}, "python3 launch.py --", False, 120, "calibrate_native_%d" % k)
            ts.append(time.time() - t0)
            if r["killed"] or int(r["exit_code"]) != 0:
                raise self.infra("native launch-overhead calibration failed (exit %s, killed %s)" % (r["exit_code"], r["killed"]))
        return min(ts)


def classify(r: Dict[str, Any], cap: int, limit: int) -> Tuple[str, str]:
    """-> (kind, detail). kind: ok | timeout | crash | overflow | maybe_truncated. Exit status and the kill flag always override the output."""
    so = str(r.get("stdout", ""))
    if r["killed"]:
        return "timeout", "killed"
    if len(so.encode("utf-8", "replace")) > cap:
        return "overflow", "stdout > %d bytes" % cap
    if int(r["exit_code"]) != 0:
        return "crash", "exit %s" % r["exit_code"]
    if len(so) >= limit:
        return "maybe_truncated", "stdout reached the port capture limit (%d)" % limit
    return "ok", ""


# ---------------------------------------------------------------------------------------------- TrustedVerifier path
def run(tv, language: str, raw: bytes, inputs: Sequence[Any]) -> engine.RunResult:
    """TrustedVerifier.run for the five native languages; mirrors TrustedVerifier.run_python's time accounting and result shapes."""
    from .trusted_verify import TrustedVerifyInfraError
    cfg = tv.cfg
    core = NativeCore(lambda wd, cmd, persist, timeout, role: tv._call(wd, cmd, persist, timeout, role), tv.scratch / "native", cfg, TrustedVerifyInfraError)
    comp = core.compile(language, raw)
    if comp.raw is not None:
        tv._obs(-1, comp.raw, comp.seconds, "compile")
    if comp.status == "oversized":
        return engine.RunResult("oversized", detail=comp.stderr)
    if comp.status == "timeout":
        return engine.RunResult("timeout", timed_out=True, detail="compile timeout", duration=comp.seconds)
    if comp.status == "compile_error":
        return engine.RunResult("compile_error", returncode=1, stderr_tail=comp.stderr[-1500:], duration=comp.seconds, detail="compile failed")
    if language not in tv.launch_overhead:
        tv.launch_overhead[language] = core.calibrate()
    oh = tv.launch_overhead[language]
    kill = tv.per_input_timeout + oh
    harness = tv.total_timeout if tv.total_timeout else len(inputs) * (kill + 1.0)
    tv.accounting = {"language": language, "inputs": len(inputs), "launch_overhead_s": round(oh, 3), "per_input_kill_s": round(kill, 3), "candidate_limit_s": cfg.py_timeout,
                     "harness_budget_s": round(harness, 3), "candidate_s": 0.0, "harness_s": 0.0, "outcome": "ran", "engine": VERSION}
    acc = tv.accounting
    limit = int(getattr(tv, "capture_limit", None) or DEFAULT_CAPTURE_LIMIT)
    lines: List[str] = []
    res = engine.RunResult("ran", returncode=0)
    t_all = time.time()
    for i, inp in enumerate(inputs):
        if time.time() - t_all > harness:
            res.phase_status, res.detail, acc["outcome"] = "sandbox_error", "harness budget %.0fs used before input %d (infrastructure, not a candidate verdict)" % (harness, i), "harness_budget"
            break
        if acc["candidate_s"] > cfg.py_timeout:
            res.phase_status, res.timed_out, acc["outcome"] = "timeout", True, "candidate_total_timeout"
            res.detail = "candidate compute %.1fs > %.0fs before input %d" % (acc["candidate_s"], cfg.py_timeout, i)
            break
        r = core.run_one(language, comp, stdin_text(inp), kill, "%s_input_%d" % (language, i))
        dt = r["seconds"]
        tv._obs(i, r, dt)
        acc["candidate_s"] = round(acc["candidate_s"] + max(0.0, dt - oh), 3)
        acc["harness_s"] = round(time.time() - t_all, 3)
        res.duration += dt
        res.stderr_tail = str(r.get("stderr", ""))[-1500:]
        kind, detail = classify(r, cfg.max_output_bytes, limit)
        if kind == "timeout":
            res.phase_status, res.timed_out, res.returncode, res.detail, acc["outcome"] = "timeout", True, r["exit_code"], "per-input timeout on input %d" % i, "per_input_timeout"
            break
        if kind == "overflow":
            res.overflow, res.returncode = True, r["exit_code"]
            break
        if kind == "crash":                                        # a non-zero exit overrides whatever was printed (claim-then-crash)
            res.returncode = int(r["exit_code"])
            break
        if kind == "maybe_truncated":
            res.returncode = 0
            lines.append("not-json: " + detail)                       # invalid_line -> invalid_output
            break
        lines.append(json.dumps(str(r.get("stdout", "")), ensure_ascii=False))
    res.stdout = "".join(l + "\n" for l in lines)
    return res
