"""[EL-BANK-LOCAL] WSL execution adapter: lets a Windows process use the bank execution engine (evidence_ladder/execution/{engine,backend}.py: bubblewrap python/java sandboxes, unmodified) by
forwarding each call as JSON to a small WSL-side server that runs the SAME SandboxExecutionBackend with the SAME isolation (bwrap --unshare-all, no network, read-only binds, timeouts, prlimit).

  Windows side   WslExecutionBackend(transport=None)       same interface as SandboxExecutionBackend: run_public(artifact, task), run_experiment(artifact, experiment), backend_info()
  WSL side       python3 -m evidence_ladder.execution.wsl_engine --serve          one JSON object per line on stdin -> one JSON object per line on stdout (stdout carries ONLY the protocol)
  Transport      WslProcessTransport (one persistent wsl.exe process, restart-once-and-retry on a dead pipe) or any object with .call(request: dict) -> dict (tests inject a fake)

FAIL CLOSED. A transport / WSL / bwrap / protocol failure RAISES WslEngineError; it never returns a result shape, never falls back to running the candidate on Windows or without bwrap. (A candidate that
fails inside the jail is an ordinary outcome of the engine, exactly as on Linux; only INFRASTRUCTURE failure raises.) The handshake (`hello`) must report bwrap, python3, the JDK and a jail probe that
ran; the engine label recorded in provenance is `wsl-bwrap`. Pure standard library on the Windows side; the engine modules are imported only on the WSL side."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ENGINE_LABEL = "wsl-bwrap"
DEFAULT_DISTRO = "Ubuntu-24.04"
JDK_DEFAULT = "/opt/toolchains/jdk-26.0.1"
CALL_TIMEOUT_S = 400.0               # one request: python batch <= 10 s, java compile <= 60 s + run <= 10 s per case batch; this is the transport's outer bound


class WslEngineError(RuntimeError):
    """The WSL execution engine is unavailable or broke (infrastructure failure). Never a verdict on a candidate."""


# ====================================================================== WSL side (also unit-testable on any OS with a fake backend)
def _artifact(d: Dict[str, Any]):
    from evidence_ladder.common.interfaces import Artifact
    return Artifact(**d)


def _task(d: Dict[str, Any]):
    from evidence_ladder.common.interfaces import TaskSpec
    return TaskSpec(**d)


def handle(backend: Any, req: Dict[str, Any], hello_fn: Optional[Callable[[], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One request -> one response. {"op": "hello"|"run_public"|"run_experiment"|"info"}; any exception becomes {"ok": False, "error": ...} (the Windows side raises it as WslEngineError)."""
    try:
        op = req.get("op")
        if op == "hello":
            return {"ok": True, "result": (hello_fn or (lambda: {}))()}
        if op == "run_public":
            return {"ok": True, "result": backend.run_public(_artifact(req["artifact"]), _task(req["task"]))}
        if op == "run_experiment":
            return {"ok": True, "result": backend.run_experiment(_artifact(req["artifact"]), req["experiment"])}
        if op == "info":
            return {"ok": True, "result": backend.backend_info()}
        return {"ok": False, "error": "unknown op %r" % (op,)}
    except BaseException as e:      # noqa: BLE001  never let one request kill the server silently; the caller decides
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:500])}


def _hello() -> Dict[str, Any]:
    import platform
    import shutil
    from evidence_ladder.execution import config as K, engine
    rec: Dict[str, Any] = {"engine": ENGINE_LABEL, "os": platform.system(), "python": sys.version.split()[0], "uid": os.getuid() if hasattr(os, "getuid") else None}
    rec["bwrap_path"] = shutil.which("bwrap")
    if rec["bwrap_path"]:
        rec["bwrap_version"] = subprocess.run(["bwrap", "--version"], capture_output=True, text=True).stdout.strip()
    jdk = K.SandboxConfig().jdk
    rec["jdk"] = jdk
    rec["java_exists"] = os.path.isfile(jdk + "/bin/java") and os.path.isfile(jdk + "/bin/javac")
    r = subprocess.run([jdk + "/bin/java", "-version"], capture_output=True, text=True) if rec["java_exists"] else None
    rec["java_version"] = (r.stderr or r.stdout).strip().splitlines()[0] if r is not None and r.returncode == 0 else None
    rec["prlimit"] = shutil.which("prlimit")
    from evidence_ladder.execution.receipts import isolation_receipt
    rec["isolation_receipt"] = isolation_receipt(K.SandboxConfig())
    pj = rec["isolation_receipt"].get("python_jail", {})
    rec["jail_probe_ran"] = isinstance(pj, dict) and "error" not in pj
    rec["ok"] = bool(rec["bwrap_path"] and rec["java_exists"] and rec["prlimit"] and rec["jail_probe_ran"] and rec["os"] == "Linux")
    return rec


def serve(stdin=None, stdout=None) -> int:
    """Line protocol server. stdout (fd 1) is duplicated for the protocol and the Python-level stdout is pointed at stderr so nothing in the engine can corrupt the stream."""
    from evidence_ladder.execution.artifacts import ArtifactStore
    from evidence_ladder.execution.backend import SandboxExecutionBackend
    import tempfile
    proto = stdout or os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    sys.stdout = sys.stderr
    backend = SandboxExecutionBackend(ArtifactStore(tempfile.mkdtemp(prefix="el-wsl-artifacts-")))
    for line in (stdin or sys.stdin):
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            resp = {"ok": False, "error": "bad json: %s" % str(e)[:100]}
        else:
            resp = handle(backend, req, _hello)
            resp["id"] = req.get("id")
        proto.write(json.dumps(resp, ensure_ascii=False, default=str) + "\n")
        proto.flush()
    return 0


# ====================================================================== Windows side
def wsl_path(p: Path) -> str:
    """D:\\local model -> /mnt/d/local model (the mechanism of evidence_ladder/execution/tests/wsl_env.sh runs from the same /mnt path)."""
    s = str(Path(p).resolve())
    if len(s) > 1 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")


class WslProcessTransport:
    """One persistent wsl.exe running the server. call() writes a request line and reads the response line, with an outer timeout; on a dead pipe it restarts the server ONCE and retries the same
    (idempotent) request; a second failure raises WslEngineError. The server's stderr goes to `stderr_log` (kept for the receipt)."""

    def __init__(self, repo: Path, distro: str = DEFAULT_DISTRO, stderr_log: Optional[Path] = None, call_timeout_s: float = CALL_TIMEOUT_S, popen=subprocess.Popen):
        self.repo, self.distro, self.stderr_log, self.timeout, self._popen = Path(repo), distro, stderr_log, call_timeout_s, popen
        self.proc = None
        self.restarts = 0
        self._lock = threading.Lock()
        self._n = 0

    def _spawn(self) -> None:
        cmd = ["wsl.exe", "-d", self.distro, "-u", "root", "-e", "bash", "-c",
               'cd "%s" && export PATH=/opt/toolchains/bin:$PATH && exec python3 -u -m evidence_ladder.execution.wsl_engine --serve' % wsl_path(self.repo)]
        err = open(self.stderr_log, "ab") if self.stderr_log else subprocess.DEVNULL
        try:
            self.proc = self._popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, bufsize=0)
        except OSError as e:
            raise WslEngineError("cannot start wsl.exe (is WSL installed?): %s" % e) from e

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _roundtrip(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if not self._alive():
            self._spawn()
        self._n += 1
        req = dict(req, id=self._n)
        data = (json.dumps(req, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        box: Dict[str, Any] = {}

        def work():
            try:
                self.proc.stdin.write(data); self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                box["line"] = line
            except BaseException as e:      # noqa: BLE001
                box["exc"] = e
        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(self.timeout)
        if th.is_alive():
            self.close(kill=True)
            raise WslEngineError("WSL engine did not answer within %.0f s (killed)" % self.timeout)
        if "exc" in box or not box.get("line"):
            raise WslEngineError("WSL engine pipe closed: %s" % (box.get("exc") or "EOF"))
        try:
            resp = json.loads(box["line"].decode("utf-8"))
        except ValueError as e:
            raise WslEngineError("WSL engine sent a non-JSON line: %s" % e) from e
        if resp.get("id") != self._n:
            raise WslEngineError("WSL engine answered a different request (id %r != %r)" % (resp.get("id"), self._n))
        return resp

    def call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            try:
                return self._roundtrip(req)
            except WslEngineError:
                self.close(kill=True)
                self.restarts += 1
                if self.restarts > 3:
                    raise
                return self._roundtrip(req)         # restart once for this request; a second failure propagates

    def close(self, kill: bool = False) -> None:
        p, self.proc = self.proc, None
        if p is not None:
            try:
                if kill: p.kill()
                else: p.stdin.close()
            except Exception:      # noqa: BLE001
                pass


class WslExecutionBackend:
    """ExecutionBackend (run_public / run_experiment / backend_info) executed inside WSL. `start()` runs the handshake and raises WslEngineError unless bwrap, python3, the JDK, prlimit and a jail probe
    are all there. No method ever executes anything on the calling OS."""
    stub = False

    def __init__(self, transport: Optional[Any] = None, repo: Optional[Path] = None, distro: str = DEFAULT_DISTRO, stderr_log: Optional[Path] = None):
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
        self.transport = transport or WslProcessTransport(self.repo, distro, stderr_log)
        self.distro, self.calls, self.hello = distro, 0, None

    def _call(self, req: Dict[str, Any]) -> Any:
        self.calls += 1
        resp = self.transport.call(req)
        if not isinstance(resp, dict) or not resp.get("ok"):
            raise WslEngineError("WSL engine failure on %s: %s" % (req.get("op"), (resp or {}).get("error", "no answer") if isinstance(resp, dict) else "bad response"))
        return resp["result"]

    def start(self) -> Dict[str, Any]:
        h = self._call({"op": "hello"})
        self.hello = h
        if not (isinstance(h, dict) and h.get("ok") and h.get("engine") == ENGINE_LABEL):
            miss = {k: h.get(k) for k in ("os", "bwrap_path", "java_exists", "prlimit", "jail_probe_ran")} if isinstance(h, dict) else h
            raise WslEngineError("WSL engine handshake failed (need Linux + bwrap + JDK + prlimit + a running jail probe): %s" % json.dumps(miss, default=str)[:400])
        return h

    @staticmethod
    def _plain(x: Any) -> Dict[str, Any]:
        return dataclasses.asdict(x)

    def run_public(self, artifact, task):
        t = self._plain(task)
        return self._call({"op": "run_public", "artifact": self._plain(artifact), "task": t})

    def run_experiment(self, artifact, experiment):
        return self._call({"op": "run_experiment", "artifact": self._plain(artifact), "experiment": experiment})

    def backend_info(self) -> Dict[str, Any]:
        h = self.hello or {}
        return {"engine": ENGINE_LABEL, "distro": self.distro, "calls": self.calls, "restarts": getattr(self.transport, "restarts", 0),
                "wsl": {k: h.get(k) for k in ("os", "python", "uid", "bwrap_version", "jdk", "java_version")}, "isolation_receipt": h.get("isolation_receipt")}

    def close(self) -> None:
        c = getattr(self.transport, "close", None)
        if c: c()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        sys.exit(serve())
    sys.exit("usage: python3 -m evidence_ladder.execution.wsl_engine --serve")
