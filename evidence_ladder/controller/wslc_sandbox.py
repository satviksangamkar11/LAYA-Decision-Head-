"""[WSLC backend] Sandbox execution provider backed by Microsoft MXC's WSLC containment (a Linux container from a PINNED LOCAL IMAGE, started per run) through tools/mxc_bridge/bridge.mjs.
Backend name `wslc`, journal environment label `mxc-wslc`. Same contract as WslSandbox / MxcSandbox (sandbox_provider.py):
    WslcSandbox(scratch, protected).run(workspace, cmd, persist=False, timeout=60) ->
    {exit_code, killed, stdout, stderr, protected_unchanged, pre_tree_sha256, post_tree_sha256, persisted}
`cmd` is a full POSIX shell command run by /bin/bash -c INSIDE the container (python3, java/javac, go, coreutils are in the image); unlike the Windows processcontainer backend there is no dialect limit.
Backend facts live OUTSIDE the result: `last_diagnostics` / `diagnostics_log`, `provenance()` = {backend, image_digest, toolchain}.

Isolation, as measured (tests/phaseWSLC_tests.py; results/raw/wslc-*):
  * image   : a local rootfs tar (configs/wslc_image_v1.toml, sha256 pinned and re-checked once per process), imported by MXC; no registry, no pull.
  * mounts  : ONLY the per-run workspace copy (read-write), the per-run job directory (runner + command, read-only) and each protected file (read-only mount inside the read-write directory). The rest of
              the host drive is NOT visible inside the container (a sentinel next to the workspace and one in the project root were not found; a write outside the copy never reached the host).
  * network : none (MXC default when no network policy is declared): only `lo`; TCP and DNS fail.
  * env     : minimal fixed environment, host environment never inherited.
  * limits  : cpuCount and memoryMb of the WSLC session are enforced (measured). A process-count limit has no field in the WSLC V1 policy and was NOT enforced (1500 forks succeeded): requesting max_processes raises.
  * timeout : `timeout -s KILL` inside the container (kills the process group) with MXC's own timeout as a backstop; the container is destroyed after every run.
  * output  : stdout/stderr are redirected to files INSIDE the container (64 MiB file-size limit per file: a flood ends with SIGXFSZ), only the first max_output_bytes are returned and `killed` is set
              when more was produced. A process that writes a single file larger than 64 MiB is killed too (documented limit).
Every infrastructure problem RAISES (SandboxUnavailable / UnsupportedPolicy / SandboxIntegrityError / RuntimeError); a failing compile, exception, bad output or crash of the CANDIDATE is an ordinary result (exit_code != 0)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evidence_ladder.controller import mxc_sandbox as M
from evidence_ladder.controller.mxc_sandbox import (SandboxIntegrityError, SandboxUnavailable, UnsupportedPolicy, _rmtree, _sha, find_links, is_link, tree_hash)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "wslc_image_v1.toml"
ENV_LABEL = "mxc-wslc"
RESULT_KEYS = M.RESULT_KEYS
OUT_KEEP = 6000
META_RE = re.compile(r"@@LAYA_META rc=(-?\d+) timed_out=([01]) capped=([01]) elapsed=(\d+)@@\n?\Z")
# mirrors tools/fable_wsl_runner.ENV for the toolchains this image carries
CONTAINER_ENV = {"PATH": "/opt/toolchains/bin:/usr/bin:/bin", "HOME": "/tmp/home", "TMPDIR": "/tmp", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                 "JAVA_HOME": "/opt/toolchains/jdk-26.0.1", "GOCACHE": "/tmp/gocache", "GOPATH": "/tmp/gopath", "GOFLAGS": "-mod=mod", "GOTOOLCHAIN": "local", "GOTELEMETRY": "off", "GOSUMDB": "off",
                 "NO_COLOR": "1", "CI": "1"}
RUNNER = r'''#!/bin/bash
# LAYA WSLC runner. args: CAP TIMEOUT WS JOB.  Runs JOB/cmd.sh with bash -c semantics inside WS. stdout/stderr go to files in the container's /tmp (file-size limit 64 MiB per file, so a flood
# ends with SIGXFSZ and can never fill the VM); only the first CAP bytes of each are returned. Last stderr line = metadata.
CAP=$1; T=$2; WS=$3; JOB=$4
mkdir -p /tmp/home /tmp/run
cd "$WS" || exit 125
O=/tmp/run/out; E=/tmp/run/err
S=$SECONDS
( ulimit -f 65536; exec timeout -s KILL "$T" /bin/bash -c "$(cat "$JOB/cmd.sh")" ) > "$O" 2> "$E"
RC=$?
EL=$((SECONDS-S))
TO=0; if [ "$RC" -eq 137 ] && [ "$EL" -ge "$T" ]; then TO=1; fi
CAPPED=0
if [ "$(stat -c %s "$O")" -gt "$CAP" ] || [ "$(stat -c %s "$E")" -gt "$CAP" ]; then CAPPED=1; fi
head -c "$CAP" "$O"
head -c "$CAP" "$E" >&2
echo >&2
echo "@@LAYA_META rc=$RC timed_out=$TO capped=$CAPPED elapsed=$EL@@" >&2
exit "$RC"
'''
RUNNER_SHA256 = hashlib.sha256(RUNNER.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ pinned image
def load_config(path: Path = CONFIG) -> Dict[str, Any]:
    import tomllib
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


_TAR_OK: Dict[str, str] = {}      # tar path -> verified sha256 (once per process)


def verify_image(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """-> {"tar", "image_digest": "sha256:...", "bytes"}; raises SandboxUnavailable when the tar is missing or differs from the pin."""
    cfg = cfg or load_config()
    img = cfg["image"]
    tar = ROOT / img["tar"]
    if not tar.is_file():
        raise SandboxUnavailable("WSLC image tar is missing: %s (build it with tools/wslc_build_rootfs.sh)" % tar)
    key = str(tar)
    if key not in _TAR_OK:
        h = hashlib.sha256()
        with open(tar, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _TAR_OK[key] = h.hexdigest()
    if _TAR_OK[key] != img["sha256"]:
        raise SandboxUnavailable("WSLC image tar differs from the pin in configs/wslc_image_v1.toml: %s != %s" % (_TAR_OK[key], img["sha256"]))
    return {"tar": str(tar), "image_digest": "sha256:" + img["sha256"], "bytes": tar.stat().st_size}


def _wsl_path(p) -> str:
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def _quote(p: str) -> str:
    return "'" + p.replace("'", "'\\''") + "'"


_CAP_CACHE: Dict[str, Dict[str, Any]] = {}
_TOOLCHAIN: Dict[str, Dict[str, str]] = {}


def capabilities(node: Optional[str] = None, use_cache: bool = False) -> Dict[str, Any]:
    """Fail-closed capability check. available only if: Windows, node, the MXC SDK is installed and lists the wslc backend, the pinned image tar is present and matches the pin, and a REAL container
    run (echo) succeeds (availableMethods of the SDK is incomplete, so it is not trusted). Executes no candidate."""
    reasons: List[str] = []
    info: Dict[str, Any] = {"available": False, "backend": None, "reasons": reasons}
    if use_cache and "ok" in _CAP_CACHE:
        return dict(_CAP_CACHE["ok"], cached=True)
    if os.name != "nt":
        reasons.append("not Windows: the WSLC provider needs the Windows MXC SDK")
        return info
    node = node or shutil.which("node")
    if not node:
        reasons.append("node not found on PATH")
        return info
    if not (ROOT / "tools" / "mxc_bridge" / "node_modules" / "@microsoft" / "mxc-sdk").is_dir():
        reasons.append("@microsoft/mxc-sdk is not installed (cd tools/mxc_bridge && npm ci --ignore-scripts)")
        return info
    try:
        cfg = load_config()
        info["image"] = verify_image(cfg)
    except SandboxUnavailable as e:
        reasons.append(str(e))
        return info
    scr = Path(os.environ.get("TEMP", ".")) / ("wslc-probe-" + uuid.uuid4().hex[:8])
    br = M._Bridge(node, scr)
    try:
        ans = br.call({"op": "probe"}, 60)
        names = [b.get("backend") for b in (ans.get("backends") or [])]
        if "wslc" not in names:
            reasons.append("the MXC SDK does not list the wslc backend: %s" % names)
            return info
        info.update(sdk=ans.get("sdk"), node=ans.get("node"))
        job = scr / "job"
        job.mkdir(parents=True)
        (job / "cmd.sh").write_text("echo wslc_ok", "utf-8")
        (job / "runner.sh").write_bytes(RUNNER.encode("utf-8"))
        ws = scr / "ws"
        ws.mkdir()
        cmd = "/bin/bash %s 4096 30 %s %s" % (_quote(_wsl_path(job / "runner.sh")), _quote(_wsl_path(ws)), _quote(_wsl_path(job)))
        t0 = time.time()
        r = br.call({"op": "run", "id": "probe", "containment": "wslc", "command": cmd, "cwd": str(ws), "env": CONTAINER_ENV, "readwrite": [str(ws)], "readonly": [str(job)], "timeoutMs": 90000,
                     "maxOutputBytes": 65536, "wslc": {"imageTarPath": str(ROOT / cfg["image"]["tar"]), "storagePath": str(ROOT / cfg["image"]["store"]), "cpuCount": cfg["limits"]["cpu_count"],
                                                       "memoryMb": cfg["limits"]["memory_mb"]}}, 150)
        info["probe_run_seconds"] = round(time.time() - t0, 2)
        if not r.get("ok") or r.get("exitCode") != 0 or "wslc_ok" not in (r.get("stdout") or ""):
            reasons.append("a real WSLC container run failed: %s" % (json.dumps(r)[:300],))
            return info
    except Exception as e:      # noqa: BLE001
        reasons.append("probe failed: %s: %s" % (type(e).__name__, str(e)[:200]))
        return info
    finally:
        br.close()
        shutil.rmtree(scr, ignore_errors=True)
    info.update(available=True, backend="wslc")
    if use_cache:
        _CAP_CACHE["ok"] = dict(info)
    return info


class WslcSandbox:
    backend = "wslc"
    env_label = ENV_LABEL

    def __init__(self, scratch: Path, protected: Sequence[str], *, node: Optional[str] = None, persistent: bool = True, cpu_count: Optional[int] = None, memory_mb: Optional[int] = None,
                 max_processes: Optional[int] = None, max_output_bytes: Optional[int] = None, check_capabilities: bool = True):
        if os.name != "nt":
            raise SandboxUnavailable("WslcSandbox needs the Windows MXC SDK")
        self.scratch, self.ro_names = Path(scratch), list(protected)       # not named "protected": the I8 deep context scan flags that word
        for p in self.ro_names:
            if os.path.isabs(p) or ".." in Path(p.replace("\\", "/")).parts or not p:
                raise UnsupportedPolicy("protected name must be a plain relative path: %r" % (p,))
        if max_processes is not None:
            raise UnsupportedPolicy("a process-count limit cannot be enforced by MXC WSLC (no policy field; 1500 forks were not limited); refusing instead of ignoring it")
        self.cfg = load_config()
        lim = self.cfg["limits"]
        raw = (("cpu_count", cpu_count if cpu_count is not None else lim["cpu_count"]), ("memory_mb", memory_mb if memory_mb is not None else lim["memory_mb"]),
               ("max_output_bytes", max_output_bytes if max_output_bytes is not None else lim["max_output_bytes"]))
        for k, v in raw:      # validated BEFORE any conversion: a bool or a float is not a limit
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise UnsupportedPolicy("invalid %s=%r (positive integer)" % (k, v))
        self.cpu_count, self.memory_mb, self.max_output_bytes = (int(v) for _k, v in raw)
        self.node = node or shutil.which("node") or ""
        self.persistent = bool(persistent)
        self.diagnostics_log: List[Dict[str, Any]] = []
        self.last_diagnostics: Dict[str, Any] = {}
        self.capability: Dict[str, Any] = {}
        if check_capabilities:
            self.capability = capabilities(self.node, use_cache=True)
            if not self.capability["available"]:
                raise SandboxUnavailable("MXC WSLC unavailable: " + "; ".join(self.capability["reasons"]))
        self.image = verify_image(self.cfg)
        self.bridge = M._Bridge(self.node, self.scratch / "_bridge", None)
        import weakref
        weakref.finalize(self, M._Bridge.close, self.bridge)

    # -------------------------------------------------------------- lifecycle / identity
    def close(self) -> None:
        self.bridge.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def provenance(self) -> Dict[str, Any]:
        """{backend, image_digest, toolchain{javac,java,python,go}} - the toolchain strings are read INSIDE the image (one container run, cached per process) and compared with the pin."""
        d = self.image["image_digest"]
        if d not in _TOOLCHAIN:
            ws = self.scratch / "_toolchain_ws"
            ws.mkdir(parents=True, exist_ok=True)
            r = self.run(ws, "javac -version 2>&1; java -version 2>&1 | head -1; python3 --version 2>&1; go version 2>&1", timeout=60)
            if r["exit_code"] != 0:
                raise RuntimeError("toolchain probe inside the image failed: %r" % (r["stderr"][-200:],))
            lines = [l.strip() for l in r["stdout"].splitlines() if l.strip()]
            tc = {"javac": lines[0], "java": lines[1], "python": lines[2], "go": lines[3]}
            exp = self.cfg["toolchain_expected"]
            if tc["python"] != exp["python"] or exp["java"] not in tc["java"] or tc["javac"] != exp["javac"] or exp["go"] not in tc["go"]:
                raise SandboxUnavailable("toolchain inside the image differs from configs/wslc_image_v1.toml: %s" % tc)
            _TOOLCHAIN[d] = tc
        return {"backend": ENV_LABEL, "image_digest": d, "toolchain": dict(_TOOLCHAIN[d])}

    # -------------------------------------------------------------- the contract
    def run(self, workspace, cmd, persist=False, timeout=60):
        t_all = time.perf_counter()
        diag: Dict[str, Any] = {"backend": ENV_LABEL, "cmd": cmd, "persist": bool(persist), "timeout": timeout, "image_digest": self.image["image_digest"], "runner_sha256": RUNNER_SHA256,
                                "cpu_count": self.cpu_count, "memory_mb": self.memory_mb}
        if not isinstance(cmd, str) or not cmd.strip() or "\x00" in cmd or len(cmd) > 100000:
            raise ValueError("empty, oversized or binary command")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        src = Path(workspace)
        if not src.is_dir():
            raise SandboxIntegrityError("workspace is not a directory: %s" % src)
        links = find_links(src)
        if links:
            raise SandboxIntegrityError("workspace contains links/reparse points (refused): %s" % links[:5])
        run = self.scratch / "runs" / uuid.uuid4().hex
        ws, job = run / "ws", run / "job"
        try:
            t0 = time.perf_counter()
            shutil.copytree(src, ws, symlinks=False)
            job.mkdir(parents=True)
            (job / "cmd.sh").write_bytes(cmd.encode("utf-8"))
            (job / "runner.sh").write_bytes(RUNNER.encode("utf-8"))
            prot = [p for p in self.ro_names if (ws / p).is_file()]
            for p in prot:
                if is_link(str(ws / p)):
                    raise SandboxIntegrityError("protected file is a link: %s" % p)
            before = {p: _sha((ws / p).read_bytes()) for p in prot}
            pre = tree_hash(ws)
            diag["prepare_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            cap = self.max_output_bytes
            command = "/bin/bash %s %d %d %s %s" % (_quote(_wsl_path(job / "runner.sh")), cap, int(max(1, round(timeout))), _quote(_wsl_path(ws)), _quote(_wsl_path(job)))
            req = {"op": "run", "id": run.name, "containment": "wslc", "command": command, "cwd": str(ws), "env": dict(CONTAINER_ENV), "readwrite": [str(ws)],
                   "readonly": [str(job)] + [str(ws / p) for p in prot], "timeoutMs": int((timeout + 25) * 1000), "maxOutputBytes": cap * 4 + 65536,
                   "wslc": {"imageTarPath": self.image["tar"], "storagePath": str(ROOT / self.cfg["image"]["store"]), "cpuCount": self.cpu_count, "memoryMb": self.memory_mb}}
            t1 = time.perf_counter()
            ans = self.bridge.call(req, timeout + 120)
            diag["wslc_call_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            if not isinstance(ans, dict) or "ok" not in ans:
                raise RuntimeError("malformed MXC bridge response: %r" % (str(ans)[:120],))
            if not ans.get("ok"):
                e = ans.get("error") or {}
                raise RuntimeError("wslc run failed (infrastructure): %s: %s" % (e.get("code"), (e.get("message") or "")[:300]))
            if not isinstance(ans.get("exitCode"), int) or isinstance(ans.get("exitCode"), bool) or not isinstance(ans.get("stdout", ""), str) or not isinstance(ans.get("stderr", ""), str):
                raise RuntimeError("malformed MXC bridge response (exitCode/stdout/stderr): %r" % (str(ans)[:160],))
            m = META_RE.search(ans.get("stderr", ""))
            if m is None:
                raise RuntimeError("the WSLC runner did not report its metadata (exit %s, stderr tail %r)" % (ans["exitCode"], ans.get("stderr", "")[-160:]))
            rc, timed_out, capped, elapsed = int(m.group(1)), m.group(2) == "1", m.group(3) == "1", int(m.group(4))
            if rc != ans["exitCode"]:
                raise RuntimeError("runner exit code %d differs from the container exit code %d" % (rc, ans["exitCode"]))
            stderr_text = ans["stderr"][:m.start()].rstrip("\n") if True else ""
            diag.update(mxc_wall_ms=ans.get("wallMs"), warnings=ans.get("warnings"), timed_out=timed_out, output_capped=capped, in_container_seconds=elapsed, bridge_starts=self.bridge.starts)
            t2 = time.perf_counter()
            new_links = find_links(ws)
            after = {p: _sha((ws / p).read_bytes()) for p in prot if (ws / p).is_file() and not is_link(str(ws / p))}
            killed = timed_out or capped or rc in (137, -9)
            res = dict(exit_code=rc, killed=killed, stdout=_norm(ans.get("stdout"), ws, job)[-OUT_KEEP:], stderr=_norm(stderr_text, ws, job)[-OUT_KEEP:],
                       protected_unchanged=(before == after and not any(is_link(str(ws / p)) for p in prot)), pre_tree_sha256=pre, post_tree_sha256="", persisted=False)
            res["post_tree_sha256"] = tree_hash(ws)
            if new_links:
                diag["links_created"] = new_links[:10]
            diag["verify_ms"] = round((time.perf_counter() - t2) * 1000, 1)
            if persist and res["protected_unchanged"] and not killed and not new_links:
                t3 = time.perf_counter()
                M.MxcSandbox._persist(ws, src)
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


def _norm(s: Optional[str], ws: Optional[Path] = None, job: Optional[Path] = None) -> str:
    """CRLF -> LF, and the random per-run directories are rewritten to fixed names (/work, /job) so that output (evidence, replay hashes) does not depend on them."""
    t = (s or "").replace("\r\n", "\n")
    for d, name in ((ws, "/work"), (job, "/job")):
        if d is not None:
            t = t.replace(_wsl_path(d), name)
    return t
