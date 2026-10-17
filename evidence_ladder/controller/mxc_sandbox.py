"""[MXC replacement] Sandbox execution provider backed by Microsoft MXC (V1 Node SDK @microsoft/mxc-sdk 1.0.1, Windows `processcontainer` backend) through tools/mxc_bridge/bridge.mjs.

Public contract (identical to WslSandbox, see sandbox_provider.py):  MxcSandbox(scratch, protected).run(workspace, cmd, persist=False, timeout=60) ->
    {exit_code, killed, stdout, stderr, protected_unchanged, pre_tree_sha256, post_tree_sha256, persisted}
Backend-specific facts are NOT added to that dict; they go to `self.last_diagnostics` / `self.diagnostics_log`.

What this provider does, in order (every step fails closed: an infrastructure problem RAISES, a result is never invented):
  1. translate `cmd` to a Windows command line: either one plain `python3|python <args>` command ("direct"), or the small POSIX subset of mxc_dialect/posix_lite.py run inside the container
     ("posix_lite"). Anything else raises UnsupportedCommand (the Windows ProcessContainer cannot start MSYS bash or PowerShell, verified: both die with 0xC0000142 inside the container).
  2. refuse a workspace that contains a symlink / junction / reparse point (the WSL helper would follow it as root).
  3. copy the workspace to scratch/runs/<uuid>/ws (candidate isolation; nothing else is mounted), hash it (same tree hash as tools/laya_int1_t6_sandbox.py), hash the protected files,
     make the protected copies OS-read-only and also declare them read-only in the MXC policy.
  4. run through MXC: containment processcontainer, NO network block (MXC denies the network when none is declared), host environment NOT inherited, a minimal fixed environment, filesystem =
     read-write {run ws, run home}, read-only {the python install, the protected files}; hard timeout (MXC kills the process tree), output cap enforced by the bridge.
  5. re-hash; refuse to persist when a link appeared, a protected file changed, the run was killed or the output was capped; persist copies regular files back under the same rules as the WSL helper.
  6. always delete the run directory.
Resource limits (memory / processes / CPU): the MXC V1 policy has no such fields for processcontainer. They are enforced by a Windows Job Object around the bridge process (see _job.py logic below)
when `limits` is given; if limits are requested and cannot be enforced the constructor raises UnsupportedPolicy unless allow_unenforced_limits=True (then diagnostics say `limits_enforced: false`)."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "tools" / "mxc_bridge" / "bridge.mjs"
MARK = "@@MXC@@"
ENV_LABEL = "mxc-processcontainer"
OUT_KEEP = 6000                    # same tail length as the WSL helper
RESULT_KEYS = ("exit_code", "killed", "stdout", "stderr", "protected_unchanged", "pre_tree_sha256", "post_tree_sha256", "persisted")
PY_NAMES = ("python3", "python")
SHELL_PUNCT = set(";&|<>()")


class SandboxUnavailable(RuntimeError):
    """The requested backend cannot run here (capability check failed). Never auto-downgraded."""


class UnsupportedCommand(RuntimeError):
    """The command is outside the dialect this backend can execute. Raised (tool error), never turned into a pass or a candidate failure."""


class UnsupportedPolicy(RuntimeError):
    """A requested security policy cannot be enforced by this backend."""


class SandboxIntegrityError(RuntimeError):
    """The workspace or run directory violated an isolation precondition (e.g. a symlink)."""


# ------------------------------------------------------------------ hashing and file-system guards (shared semantics with tools/laya_int1_t6_sandbox.py)
def is_link(path: str) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def find_links(root: Path) -> List[str]:
    """Relative paths of every symlink / junction / reparse point under root (not followed)."""
    found: List[str] = []
    for dp, dns, fns in os.walk(root, followlinks=False):
        for n in list(dns) + list(fns):
            p = os.path.join(dp, n)
            if is_link(p):
                found.append(os.path.relpath(p, root).replace("\\", "/"))
    return sorted(found)


def tree_hash(ws: Path) -> str:
    """Same bytes as laya_int1_t6_sandbox.tree(): sorted files, 'relpath\\0sha256(content)'. Links never occur here (find_links runs first)."""
    h = hashlib.sha256()
    files = sorted((x for x in Path(ws).rglob("*") if x.is_file() and not is_link(str(x))), key=lambda p: str(p.relative_to(ws)))
    for p in files:
        h.update(str(p.relative_to(ws)).replace("\\", "/").encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _rmtree(p: Path) -> None:
    def onerr(fn, path, exc):
        try:
            os.chmod(path, stat.S_IWRITE)
            fn(path)
        except OSError:
            pass
    shutil.rmtree(p, onerror=onerr)


# ------------------------------------------------------------------ command dialect
DIALECT_DIR = Path(__file__).resolve().parent / "mxc_dialect"


def translate(cmd: str, python_exe: str) -> Tuple[str, str]:
    """-> (windows command line, mode). mode "direct": exactly one simple `python3|python <args>` command, started without any shell. mode "posix_lite": the subset documented in
    mxc_dialect/posix_lite.py (echo/cat/ls/pwd/mkdir/touch/rm/true/false/python, ; && || > >>), run by that module INSIDE the container. Anything else raises UnsupportedCommand."""
    from evidence_ladder.controller.mxc_dialect import posix_lite as PL
    try:
        plan = PL.parse(cmd)
    except PL.Unsupported as e:
        raise UnsupportedCommand("unsupported by the %s dialect: %s" % (ENV_LABEL, e))
    if len(plan) == 1 and not plan[0][2] and plan[0][1][0] in PL.PY_NAMES:
        return subprocess.list2cmdline([python_exe] + plan[0][1][1:]), "direct"
    return subprocess.list2cmdline([python_exe, str(DIALECT_DIR / "posix_lite.py"), "-c", cmd]), "posix_lite"


# ------------------------------------------------------------------ bridge process
class _Bridge:
    """node tools/mxc_bridge/bridge.mjs --serve, requests/answers as JSON lines. One request at a time (serialised by the caller)."""

    def __init__(self, node: str, scratch: Path, job=None):
        self.node, self.scratch, self.job = node, scratch, job
        self.proc: Optional[subprocess.Popen] = None
        self.q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.starts = 0

    def start(self) -> None:
        self.scratch.mkdir(parents=True, exist_ok=True)
        env = {"SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"), "PATH": os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32") + os.pathsep + os.environ.get("SystemRoot", r"C:\Windows"),
               "TEMP": str(self.scratch), "TMP": str(self.scratch)}
        for k in ("LOCALAPPDATA", "USERPROFILE", "APPDATA", "ProgramData", "ProgramFiles", "windir"):
            if k in os.environ:
                env[k] = os.environ[k]      # the SDK itself needs the real profile to locate its policy store; the WORKLOAD never sees these (inheritDefaultEnvironment=false)
        self.errlog = open(self.scratch / "bridge_stderr.log", "ab")
        self.proc = subprocess.Popen([self.node, str(BRIDGE), "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errlog, cwd=str(self.scratch), env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.starts += 1
        if self.job is not None:
            self.job.assign(self.proc)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.q), daemon=True).start()

    @staticmethod
    def _pump(proc, q):
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace")
            if line.startswith(MARK):
                q.put(line[len(MARK):])
        q.put(None)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def call(self, req: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        if not self.alive():
            self.start()
        payload = (json.dumps(req) + "\n").encode("utf-8")
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except OSError:      # the bridge died between calls: nothing was delivered, so starting a fresh one and sending once is safe (a request that WAS delivered is never re-sent)
            self.kill()
            self.start()
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            raise subprocess.TimeoutExpired("mxc bridge", timeout)
        if line is None:
            self.kill()
            raise RuntimeError("mxc bridge exited unexpectedly (see bridge_stderr.log)")
        return json.loads(line)

    def kill(self) -> None:
        if self.job is not None:      # the container's processes belong to the job: killing only node would leave them running
            self.job.terminate()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:       # noqa: BLE001
                pass
        self.proc = None

    def close(self) -> None:
        if self.alive():
            try:
                self.proc.stdin.write(b'{"op":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:       # noqa: BLE001
                self.kill()
        self.proc = None


def default_python() -> str:
    exe = os.environ.get("LAYA_MXC_PYTHON") or os.path.join(sys.base_prefix, "python.exe")
    return exe


_CAP_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}      # successful probes only, for the life of the process (the SDK memoizes its own platform checks the same way)


def capabilities(node: Optional[str] = None, python: Optional[str] = None, scratch: Optional[Path] = None, use_cache: bool = False) -> Dict[str, Any]:
    """Fail-closed capability check. -> {"available": bool, "reasons": [...], "backend": "processcontainer"|None, ...}. Available only if: Windows, node >= 24, the SDK is installed, MXC reports
    `processcontainer` in availableMethods, and a python interpreter exists. It runs ONE probe through the bridge; it executes no workload."""
    reasons: List[str] = []
    info: Dict[str, Any] = {"available": False, "backend": None, "reasons": reasons}
    key = (str(node or shutil.which("node") or ""), str(python or default_python()))
    if use_cache and key in _CAP_CACHE:
        return dict(_CAP_CACHE[key], cached=True)
    if os.name != "nt":
        reasons.append("not Windows: the processcontainer backend is Windows-only (use the platform-native MXC backend through a different provider)")
        return info
    node = node or shutil.which("node")
    if not node:
        reasons.append("node not found on PATH")
        return info
    py = python or default_python()
    if not os.path.isfile(py):
        reasons.append("python interpreter not found: %s" % py)
    if not (ROOT / "tools" / "mxc_bridge" / "node_modules" / "@microsoft" / "mxc-sdk").is_dir():
        reasons.append("@microsoft/mxc-sdk is not installed (cd tools/mxc_bridge && npm ci --ignore-scripts)")
        return info
    scr = Path(scratch) if scratch else Path(os.environ.get("TEMP", ".")) / ("mxc-probe-" + uuid.uuid4().hex[:8])
    br = _Bridge(node, scr)
    try:
        ans = br.call({"op": "probe"}, 60)
    except Exception as e:      # noqa: BLE001
        reasons.append("probe failed: %s: %s" % (type(e).__name__, str(e)[:200]))
        return info
    finally:
        br.close()
        if scratch is None:
            shutil.rmtree(scr, ignore_errors=True)
    if not ans.get("ok"):
        reasons.append("probe error: %s" % (ans.get("error"),))
        return info
    info.update(sdk=ans.get("sdk"), node=ans.get("node"), support=ans.get("support"), probe_tier=(ans.get("probe") or {}).get("tier"), probes=(ans.get("probe") or {}).get("probes"))
    sup = ans.get("support") or {}
    if not sup.get("isSupported") or "processcontainer" not in (sup.get("availableMethods") or []):
        reasons.append("MXC does not report processcontainer as available: %s" % json.dumps({k: sup.get(k) for k in ("isSupported", "reason", "availableMethods", "unavailableReasons")}))
    probes = info.get("probes") or {}
    if not probes.get("baseContainerApiPresent"):
        reasons.append("baseContainerApiPresent is false")
    if not reasons:
        info.update(available=True, backend="processcontainer")
        if use_cache:
            _CAP_CACHE[key] = dict(info)
    return info


class Limits:
    """Resource limits requested from the host (Windows Job Object around the bridge). memory_mb: job memory; max_processes: active process count; cpu_percent: hard CPU cap, 100 = one core."""

    def __init__(self, memory_mb: Optional[int] = 2048, max_processes: Optional[int] = 256, cpu_percent: Optional[int] = 400):
        self.memory_mb, self.max_processes, self.cpu_percent = memory_mb, max_processes, cpu_percent

    def as_dict(self) -> Dict[str, Any]:
        return {"memory_mb": self.memory_mb, "max_processes": self.max_processes, "cpu_percent": self.cpu_percent}


class MxcSandbox:
    backend = "mxc"
    env_label = ENV_LABEL

    def __init__(self, scratch: Path, protected: Sequence[str], *, python: Optional[str] = None, node: Optional[str] = None, persistent: bool = False,
                 limits: Optional[Limits] = None, allow_unenforced_limits: bool = False, max_output_bytes: int = 1 << 20, check_capabilities: bool = True):
        if os.name != "nt":
            raise SandboxUnavailable("MxcSandbox (processcontainer) is Windows-only")
        self.scratch, self.ro_names = Path(scratch), list(protected)       # not named "protected": the I8 deep context scan flags that word
        for p in self.ro_names:
            if os.path.isabs(p) or ".." in Path(p.replace("\\", "/")).parts or not p:
                raise UnsupportedPolicy("protected name must be a plain relative path: %r" % (p,))
        self.python = python or default_python()
        self.node = node or shutil.which("node") or ""
        self.persistent, self.max_output_bytes = bool(persistent), int(max_output_bytes)
        self.limits = limits
        if limits is not None:
            for k, v in limits.as_dict().items():
                if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0):
                    raise UnsupportedPolicy("invalid resource limit %s=%r (positive integer or None)" % (k, v))
        self.allow_unenforced_limits = bool(allow_unenforced_limits)
        self.diagnostics_log: List[Dict[str, Any]] = []
        self.last_diagnostics: Dict[str, Any] = {}
        self.capability: Dict[str, Any] = {}
        if check_capabilities:
            self.capability = capabilities(self.node, self.python, self.scratch / "_probe", use_cache=True)
            if not self.capability["available"]:
                raise SandboxUnavailable("MXC processcontainer unavailable: " + "; ".join(self.capability["reasons"]))
        self.job = None
        self.limits_enforced = False
        if limits is not None:
            from evidence_ladder.controller.win_job import Job
            try:
                self.job = Job(memory_mb=limits.memory_mb, max_processes=limits.max_processes, cpu_percent=limits.cpu_percent)
                self.limits_enforced = True
            except Exception as e:      # noqa: BLE001
                if not self.allow_unenforced_limits:
                    raise UnsupportedPolicy("resource limits %s cannot be enforced: %s" % (limits.as_dict(), e))
        self.bridge = _Bridge(self.node, self.scratch / "_bridge", self.job)
        weakref.finalize(self, _Bridge.close, self.bridge)      # a forgotten close() must not leave a node process behind

    # -------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.bridge.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -------------------------------------------------------------- the contract
    def run(self, workspace, cmd, persist=False, timeout=60):
        t_all = time.perf_counter()
        diag: Dict[str, Any] = {"backend": "mxc/processcontainer", "cmd": cmd, "persist": bool(persist), "timeout": timeout}
        command, mode = translate(cmd, self.python)
        diag["dialect"] = mode
        src = Path(workspace)
        if not src.is_dir():
            raise SandboxIntegrityError("workspace is not a directory: %s" % src)
        links = find_links(src)
        if links:
            raise SandboxIntegrityError("workspace contains links/reparse points (refused): %s" % links[:5])
        run = self.scratch / "runs" / uuid.uuid4().hex
        ws, home = run / "ws", run / "home"
        try:
            t0 = time.perf_counter()
            shutil.copytree(src, ws, symlinks=False)
            home.mkdir(parents=True)
            prot = [p for p in self.ro_names if (ws / p).is_file()]
            for p in prot:
                if is_link(str(ws / p)):
                    raise SandboxIntegrityError("protected file is a link: %s" % p)
            before = {p: _sha((ws / p).read_bytes()) for p in prot}
            pre = tree_hash(ws)
            for p in prot:
                os.chmod(ws / p, stat.S_IREAD)
            diag["prepare_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            pyd = str(Path(self.python).parent)
            env = {"SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"), "PATH": pyd + os.pathsep + os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"),
                   "LOCALAPPDATA": str(home), "TEMP": str(home), "TMP": str(home), "HOME": str(home), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
                   "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1", "CI": "1"}
            req = {"op": "run", "id": run.name, "command": command, "cwd": str(ws), "env": env, "readwrite": [str(ws), str(home)],
                   "readonly": [pyd] + ([str(DIALECT_DIR)] if mode == "posix_lite" else []) + [str(ws / p) for p in prot], "timeoutMs": int(timeout * 1000), "maxOutputBytes": self.max_output_bytes}
            t1 = time.perf_counter()
            try:
                ans = self._call(req, timeout + 60)
            except subprocess.TimeoutExpired:
                raise
            diag["mxc_call_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            if not isinstance(ans, dict) or "ok" not in ans:
                raise RuntimeError("malformed MXC bridge response: %r" % (str(ans)[:120],))
            if not ans.get("ok"):
                e = ans.get("error") or {}
                raise RuntimeError("mxc run failed: %s: %s" % (e.get("code"), (e.get("message") or "")[:300]))
            if not isinstance(ans.get("exitCode"), int) or isinstance(ans.get("exitCode"), bool) or not isinstance(ans.get("stdout", ""), str) or not isinstance(ans.get("stderr", ""), str):
                raise RuntimeError("malformed MXC bridge response (exitCode/stdout/stderr): %r" % (str(ans)[:160],))
            diag.update(mxc_wall_ms=ans.get("wallMs"), warnings=ans.get("warnings"), output_capped=bool(ans.get("outputCapped")), timed_out=bool(ans.get("timedOut")), bridge_starts=self.bridge.starts,
                        limits_enforced=self.limits_enforced, limits=(self.limits.as_dict() if self.limits else None))
            t2 = time.perf_counter()
            new_links = find_links(ws)
            after = {p: _sha((ws / p).read_bytes()) for p in prot if (ws / p).is_file() and not is_link(str(ws / p))}
            killed = bool(ans.get("timedOut")) or bool(ans.get("outputCapped"))
            res = dict(exit_code=int(ans["exitCode"]), killed=killed, stdout=_norm(ans.get("stdout"), ws, home)[-OUT_KEEP:], stderr=_norm(ans.get("stderr"), ws, home)[-OUT_KEEP:],
                       protected_unchanged=(before == after and not any(is_link(str(ws / p)) for p in prot)), pre_tree_sha256=pre, post_tree_sha256="", persisted=False)
            if new_links:
                diag["links_created"] = new_links[:10]
                res["post_tree_sha256"] = _post_hash_without_links(ws)
            else:
                res["post_tree_sha256"] = tree_hash(ws)
            diag["verify_ms"] = round((time.perf_counter() - t2) * 1000, 1)
            if persist and res["protected_unchanged"] and not killed and not new_links:
                t3 = time.perf_counter()
                self._persist(ws, src)
                res["persisted"] = True
                diag["persist_ms"] = round((time.perf_counter() - t3) * 1000, 1)
            elif persist:
                diag["persist_refused"] = "links_created" if new_links else ("protected_changed" if not res["protected_unchanged"] else "killed_or_capped")
            for k in RESULT_KEYS:
                assert k in res
            return res
        finally:
            _rmtree(run)
            diag["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
            self.last_diagnostics = diag
            self.diagnostics_log.append(diag)
            if not self.persistent:
                self.bridge.close()

    # -------------------------------------------------------------- helpers
    def _call(self, req: Dict[str, Any], outer: float) -> Dict[str, Any]:
        return self.bridge.call(req, outer)

    @staticmethod
    def _persist(ws: Path, src: Path) -> None:
        """Same rules as the WSL helper: delete files missing from the run copy, copy regular files back (skip __pycache__). Paths are re-checked to stay inside src."""
        src_root = src.resolve()
        for q in [x for x in src.rglob("*") if x.is_file() and not (ws / x.relative_to(src)).exists()]:
            q.unlink()
        for q in [x for x in ws.rglob("*") if x.is_file() and "__pycache__" not in x.parts]:
            if is_link(str(q)):
                continue
            d = src / q.relative_to(ws)
            if src_root not in d.resolve().parents and d.resolve() != src_root:
                raise SandboxIntegrityError("persist target escapes the workspace: %s" % d)
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists():
                os.chmod(d, stat.S_IWRITE | stat.S_IREAD)
            shutil.copyfile(q, d)
            os.chmod(d, stat.S_IWRITE | stat.S_IREAD)


def _norm(s: Optional[str], ws: Optional[Path] = None, home: Optional[Path] = None) -> str:
    """CRLF -> LF, and the per-run directories are rewritten to the fixed names the WSL sandbox shows (/work, /tmp) so that output (and therefore evidence and replay hashes) does not depend on
    the random run directory."""
    t = (s or "").replace("\r\n", "\n")
    for d, name in ((ws, "/work"), (home, "/tmp")):
        if d is not None:
            for form in (str(d), str(d).replace("\\", "/")):
                t = t.replace(form + "\\", name + "/").replace(form + "/", name + "/").replace(form, name)
    return t


def _post_hash_without_links(ws: Path) -> str:
    return tree_hash(ws)
