"""Jailed execution engine. Reuses the EXISTING receipted command builders (python: bwrap_cmd + runner_v2.py; java: jail() + the worker Json/Runner support files) loaded by
file path and never edited; adds per-case output capture, output caps and verdict classification. No protected-test data is reachable from this module: callers pass
the inputs and expected values in (public tests from TaskSpec, or the protected grader's own data).
Trusted-parent facts: candidate bytes are copied into a fresh temp dir per run (no shared writable state); the jail mounts nothing from the host except ro /usr, the JDK
and that run's own files; the process group is always SIGKILLed afterwards."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from . import config as K

_MODS: dict = {}


def _load(path: Path, name: str):
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)     # these scripts only define functions at import (main guarded)
        if os.environ.get("EL_JAIL") == "kaggle":      # FC-1: no bwrap on Kaggle; swap in the chroot+uid+seccomp jail (kaggle_jail.py)
            from . import kaggle_jail
            if hasattr(m, "bwrap_cmd"):
                m.bwrap_cmd = kaggle_jail.bwrap_cmd
            if hasattr(m, "jail"):
                m.jail = lambda w, cmd, rw=True: kaggle_jail.java_jail(w, cmd, rw)
        _MODS[name] = m
    return _MODS[name]


def py_mod():
    return _load(K.PY_GRADE_SCRIPT, "_el_py_sandbox")


def java_mod():
    return _load(K.JAVA_GRADE_SCRIPT, "_el_java_sandbox")


@dataclass
class RunResult:
    phase_status: str                       # "ran" | "compile_error" | "timeout" | "sandbox_error" | "oversized"
    stdout: str = ""
    returncode: Optional[int] = None
    stderr_tail: str = ""
    timed_out: bool = False
    overflow: bool = False
    bad_encoding: bool = False
    duration: float = 0.0
    detail: str = ""


def _kill(p) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def spawn(cmd: list, stdin_bytes: bytes, timeout: float, max_out: int) -> RunResult:
    t0 = time.monotonic()
    # minimal env: bwrap is pid 1 of the jail and its /proc/1/environ is readable by the candidate's uid, so the parent's environment (API keys, tokens) must never reach it
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"})
    out, err, st = bytearray(), bytearray(), {"over": False}

    def pump_out():
        while True:
            try:
                c = os.read(p.stdout.fileno(), 65536)
            except OSError:
                break
            if not c:
                break
            if len(out) < max_out:
                out.extend(c[: max_out - len(out)])
            if len(out) >= max_out and not st["over"]:
                st["over"] = True
                _kill(p)

    def pump_err():
        while True:
            try:
                c = os.read(p.stderr.fileno(), 65536)
            except OSError:
                break
            if not c:
                break
            err.extend(c)
            del err[:-65536]        # rolling tail

    def feed():
        try:
            p.stdin.write(stdin_bytes)
            p.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    ths = [threading.Thread(target=f, daemon=True) for f in (pump_out, pump_err, feed)]
    for t in ths:
        t.start()
    timed_out = False
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(p)
        rc = p.wait()
    _kill(p)                         # reap any stragglers of the group (bwrap is pid 1 of its pid namespace and dies with it)
    for t in ths:
        t.join(5)
    for s in (p.stdout, p.stderr):
        try:
            s.close()
        except OSError:
            pass
    try:
        text, bad = bytes(out).decode("utf-8"), False
    except UnicodeDecodeError:
        text, bad = bytes(out).decode("utf-8", "replace"), True
    return RunResult("ran", text, rc, bytes(err).decode("utf-8", "replace")[-2000:], timed_out, st["over"], bad, round(time.monotonic() - t0, 3))


def _sandbox_err(r: RunResult) -> bool:
    return r.returncode not in (0, None) and not r.timed_out and not r.stdout and r.stderr_tail.lstrip().startswith(("bwrap:", "runuser:"))


# ------------------------------------------------------------------ python
def run_python(raw: bytes, inputs: Sequence[Any], cfg: K.SandboxConfig) -> RunResult:
    if len(raw) > cfg.max_source_bytes:
        return RunResult("oversized", detail=f"source {len(raw)} bytes > {cfg.max_source_bytes}")
    mod = py_mod()
    jd = Path(tempfile.mkdtemp(prefix="jail-"))
    try:
        os.chmod(jd, 0o755)
        cand, runner = jd / "cand.py", jd / "runner.py"
        cand.write_bytes(raw)
        shutil.copyfile(K.PY_RUNNER, runner)
        os.chmod(cand, 0o644)
        os.chmod(runner, 0o644)
        stdin = "".join(json.dumps(x, ensure_ascii=True) + "\n" for x in inputs).encode("ascii")
        r = spawn(mod.bwrap_cmd(cand, runner), stdin, cfg.py_timeout, cfg.max_output_bytes)
    finally:
        shutil.rmtree(jd, ignore_errors=True)
    if _sandbox_err(r):
        r.phase_status = "sandbox_error"
    elif r.timed_out:
        r.phase_status = "timeout"
    return r


# ------------------------------------------------------------------ java
def run_java(raw: bytes, inputs: Sequence[Any], cfg: K.SandboxConfig) -> RunResult:
    if len(raw) > cfg.max_source_bytes:
        return RunResult("oversized", detail=f"source {len(raw)} bytes > {cfg.max_source_bytes}")
    mod = java_mod()
    mod.JDK = cfg.jdk               # ponytail: the existing jail() reads a module global; single-threaded use only
    jdk = cfg.jdk
    w = Path(tempfile.mkdtemp(prefix="jjail-"))
    try:
        os.chmod(w, 0o777)
        (w / "Solution.java").write_bytes(raw)
        for f in ("Json.java", "Runner.java"):
            shutil.copyfile(K.JAVA_SUPPORT / f, w / f)
        for f in w.iterdir():
            os.chmod(f, 0o644)
        compile_cmd = mod.jail(w, ["/usr/bin/prlimit", *cfg.java_compile_prlimit, jdk + "/bin/javac", "-proc:none", "-encoding", "UTF-8", "-d", "/work",
                                   "/work/Json.java", "/work/Runner.java", "/work/Solution.java"])
        c = spawn(compile_cmd, b"", cfg.java_compile_timeout, 1 << 20)
        if c.timed_out:
            c.phase_status = "timeout"; c.detail = "compile timeout"
            return c
        if _sandbox_err(c):
            c.phase_status = "sandbox_error"
            return c
        if c.returncode != 0:
            c.phase_status = "compile_error"
            return c
        stdin = ("\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in inputs) + "\n").encode("utf-8", "replace")
        run_cmd = mod.jail(w, ["/usr/bin/prlimit", *cfg.java_run_prlimit, jdk + "/bin/java", cfg.java_xmx, "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8",
                               "-Dstdin.encoding=UTF-8", "-cp", "/work", "Runner"], rw=False)
        r = spawn(run_cmd, stdin, cfg.java_run_timeout, cfg.max_output_bytes)
    finally:
        shutil.rmtree(w, ignore_errors=True)
    if _sandbox_err(r):
        r.phase_status = "sandbox_error"
    elif r.timed_out:
        r.phase_status = "timeout"
    return r


def run(language: str, raw: bytes, inputs: Sequence[Any], cfg: K.SandboxConfig) -> RunResult:
    return run_python(raw, inputs, cfg) if language == "python" else run_java(raw, inputs, cfg)


# ------------------------------------------------------------------ parsing and verdicts
def _parse(language: str, r: RunResult):
    """-> (items, term). items[i] = (ok, value_or_error_name) for case i. term = why output stopped: None (clean), 'timeout', 'overflow', 'invalid_line', 'crash', 'fatal'."""
    items: list = []
    term: Optional[str] = None
    if r.bad_encoding:
        return items, "invalid_line"
    for line in r.stdout.splitlines():
        try:
            v = json.loads(line)
        except ValueError:
            return items, "invalid_line"
        if language == "python":
            if not isinstance(v, dict):
                return items, "invalid_line"
            if "fatal" in v:
                return items, "fatal:" + str(v.get("fatal")) + ":" + str(v.get("message", ""))
            if v.get("ok") is True and "value" in v:
                items.append((True, v["value"]))
            elif v.get("ok") is False and "error" in v:
                items.append((False, str(v["error"])))
            else:
                return items, "invalid_line"
        else:
            items.append((True, v))
    if r.overflow:
        term = "overflow"
    elif r.timed_out:
        term = "timeout"
    elif r.returncode != 0:
        term = "crash"
    return items, term


_TERM_CLASS = {"timeout": "timeout", "overflow": "invalid_output", "invalid_line": "invalid_output", "crash": "runtime_error", None: "missing_output"}


@dataclass
class Verdict:
    status: str                  # pass | fail | timeout | error
    failure_class: str           # "" | compile_error | runtime_error | timeout | sandbox_error | wrong_answer | missing_output | invalid_output
    passed: int = 0
    total: int = 0
    first_fail: Optional[int] = None
    observed: Optional[str] = None
    detail: str = ""
    stderr_tail: str = ""
    duration: float = 0.0


def _status(cls: str) -> str:
    return "timeout" if cls == "timeout" else "error" if cls == "sandbox_error" else "fail"


def _cmp(language: str):
    m = py_mod() if language == "python" else java_mod()
    return m.strict_equal if language == "python" else m.same


def judge(language: str, r: RunResult, cases: Sequence[dict], cfg: K.SandboxConfig) -> Verdict:
    n, tail = len(cases), r.stderr_tail
    base = dict(total=n, stderr_tail=tail, duration=r.duration)
    if r.phase_status == "sandbox_error":
        return Verdict("error", "sandbox_error", detail="sandbox failed to start: " + tail[:200], **base)
    if r.phase_status == "oversized":
        return Verdict("fail", "invalid_output", detail=r.detail, **base)
    if r.phase_status == "compile_error":
        return Verdict("fail", "compile_error", detail="javac failed", **base)
    if r.phase_status == "timeout" and r.detail == "compile timeout":
        return Verdict("timeout", "timeout", detail="compile timeout", **base)
    items, term = _parse(language, r)
    if term and term.startswith("fatal:"):
        name, _, msg = term[6:].partition(":")
        cls = "compile_error" if name in ("SyntaxError", "IndentationError", "TabError") else "runtime_error"
        return Verdict("fail", cls, detail=f"module failed to load: {name}: {msg[:200]}", **base)
    cmp = _cmp(language)
    npass, first, fcls, obs = 0, None, "", None
    for i, c in enumerate(cases):
        if i < len(items):
            ok, v = items[i]
            if ok and cmp(v, c["expected"]):
                npass += 1
                continue
            cls = "wrong_answer" if ok else "runtime_error"
            o = json.dumps(v, ensure_ascii=False) if ok else f"<error: {v}>"
        else:
            cls = _TERM_CLASS.get(term, "runtime_error")
            o = f"<{cls}>"
        if first is None:
            first, fcls, obs = i, cls, o[: cfg.max_observed_chars]
    if first is None:
        if len(items) > n:
            return Verdict("fail", "invalid_output", n, n, None, None, "surplus output lines", tail, r.duration)
        if term in ("timeout", "overflow", "invalid_line", "crash"):
            cls = _TERM_CLASS[term]
            return Verdict(_status(cls), cls, n, n, None, None, f"all cases matched but run ended abnormally ({term})", tail, r.duration)
        return Verdict("pass", "", n, n, None, None, "all cases matched", tail, r.duration)
    return Verdict(_status(fcls), fcls, npass, n, first, obs, f"case {first}: {fcls}", tail, r.duration)


def observe(language: str, r: RunResult, cfg: K.SandboxConfig):
    """Single-input experiment result -> (status, failure_class, observed, detail)."""
    v = judge(language, r, [{"input": None, "expected": object()}], cfg)      # expected is a sentinel that never equals anything
    if v.failure_class in ("wrong_answer", ""):
        items, _ = _parse(language, r)
        if items and items[0][0]:
            return "pass", "", items[0][1], "experiment executed; the output is an observation, not a correctness verdict"
    if v.failure_class == "runtime_error" and v.observed:
        return "fail", "runtime_error", v.observed, "candidate raised on this input"
    return v.status, v.failure_class, v.observed, v.detail


def judge_each(language: str, r: RunResult, cases: Sequence[dict], cfg: K.SandboxConfig):
    """-> (overall Verdict, [(status, failure_class, observed_str)] one per case, in order). Overall abnormal outcomes (compile/sandbox/oversized/timeout-at-compile,
    module load failure, or a run that ended abnormally after matching every case) make EVERY example non-pass."""
    v = judge(language, r, cases, cfg)
    n, cap = len(cases), cfg.max_observed_chars
    if v.status != "pass" and (v.first_fail is None):
        return v, [(v.status, v.failure_class, "") for _ in range(n)]
    items, term = _parse(language, r)
    cmp = _cmp(language)
    out = []
    for i, c in enumerate(cases):
        if i < len(items):
            ok, val = items[i]
            if ok:
                s = json.dumps(val, ensure_ascii=False)[:cap]
                out.append(("pass", "", s) if cmp(val, c["expected"]) else ("fail", "wrong_answer", s))
            else:
                out.append(("fail", "runtime_error", f"<error: {val}>"))
        else:
            cls = _TERM_CLASS.get(term, "runtime_error")
            out.append((_status(cls), cls, f"<{cls}>"))
    if v.status == "pass":
        assert all(o[0] == "pass" for o in out)
    return v, out
