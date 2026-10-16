"""Isolation receipt (what the jail actually looks like from inside, measured by running probe code in it) and replay of the pre-existing sandbox receipts
(pure file reads: sha256 + summary fields; nothing is executed). Pinned hashes are over LF-normalized bytes (so CRLF/LF checkouts agree), re-pinned by the coordinator from the main checkout on 2026-10-16."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from . import config as K
from . import engine

PROBE = r'''
import os, sys, socket, resource
def attempt(fn):
    try:
        fn(); return "ALLOWED"
    except BaseException as e:
        return "blocked:" + type(e).__name__
def solve(d):
    ns = {n: os.readlink("/proc/self/ns/" + n) for n in ("user", "pid", "net", "ipc", "uts", "mnt")}
    lim = {n: list(resource.getrlimit(getattr(resource, "RLIMIT_" + n))) for n in ("AS", "FSIZE", "NOFILE", "NPROC", "CORE")}
    return {"uid": os.getuid(), "gid": os.getgid(), "namespaces": ns, "rlimits": lim, "env_keys": sorted(os.environ), "python": sys.version.split()[0],
            "write_usr": attempt(lambda: open("/usr/x", "w")), "write_etc": attempt(lambda: open("/etc/x", "w")), "write_tmp": attempt(lambda: open("/tmp/x", "w")),
            "network": attempt(lambda: socket.create_connection(("1.1.1.1", 80), timeout=2)),
            "host_mnt_listing": attempt(lambda: [p for p in os.listdir("/mnt") if True][0]), "host_home_listing": attempt(lambda: os.listdir("/home")[0])}
'''

KPROBE = r'''
import os, sys, socket, resource, ctypes, signal
def attempt(fn):
    try:
        fn(); return "ALLOWED"
    except BaseException as e:
        return "blocked:" + type(e).__name__ + ":" + str(getattr(e, "errno", ""))
def x32():
    libc = ctypes.CDLL(None, use_errno=True)
    pid = os.fork()
    if pid == 0:
        libc.syscall(0x40000000 + 39)      # getpid through the x32 ABI number range
        os._exit(0)
    _, st = os.waitpid(pid, 0)
    return os.WTERMSIG(st) if os.WIFSIGNALED(st) else -1
def solve(d):
    lim = {n: list(resource.getrlimit(getattr(resource, "RLIMIT_" + n))) for n in ("AS", "FSIZE", "NOFILE", "NPROC", "CORE")}
    return {"uid": os.getuid(), "euid": os.geteuid(), "gid": os.getgid(), "groups": os.getgroups(), "rlimits": lim, "env_keys": sorted(os.environ), "python": sys.version.split()[0],
            "root_listing": sorted(os.listdir("/")), "proc_visible": attempt(lambda: os.listdir("/proc")), "kaggle_visible": attempt(lambda: os.listdir("/kaggle")),
            "write_usr": attempt(lambda: open("/usr/x", "w")), "write_etc": attempt(lambda: open("/etc/x", "w")), "write_tmp": attempt(lambda: open("/tmp/x", "w")),
            "network": attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)), "unix_socket": attempt(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)),
            "setuid0": attempt(lambda: os.setuid(0)), "chroot_escape": attempt(lambda: os.chroot("/")), "x32_syscall_killed_by_signal": x32()}
'''


def _host_ns() -> Dict[str, str]:
    try:
        return {n: os.readlink("/proc/self/ns/" + n) for n in ("user", "pid", "net", "ipc", "uts", "mnt")}
    except OSError:
        return {}


def isolation_receipt(cfg: K.SandboxConfig = K.SandboxConfig()) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"backend": "bubblewrap (existing receipted python/java sandboxes, unmodified)", "host_uid": os.getuid(),
                           "jail_user": "nobody via runuser when host uid is 0 (rlimits only bind a non-root uid)",
                           "limits_config": {k: getattr(cfg, k) for k in ("py_timeout", "java_run_timeout", "java_compile_timeout", "max_source_bytes", "max_output_bytes", "jdk", "java_xmx",
                                                                            "java_compile_prlimit", "java_run_prlimit")},
                           "kaggle": "NOT verified: FC-1 (uid drop / bwrap on Kaggle) is a separate unrun feasibility test"}
    kaggle = os.environ.get("EL_JAIL") == "kaggle"
    r = engine.run_python((KPROBE if kaggle else PROBE).encode(), [{}], cfg)
    items, term = engine._parse("python", r)
    if kaggle and r.phase_status == "ran" and items and items[0][0]:
        rec["python_jail"] = items[0][1]
        rec["kaggle_mechanisms"] = {"filesystem": "chroot into a root-owned rootfs (no /proc, no host paths); read-only by uid/permissions", "privileges": "setgroups([]), setgid/setuid 65534; capabilities lost on setuid",
                                    "network": "seccomp-bpf: socket() of every family returns EPERM; kernel internet off", "syscall_abi": "seccomp-bpf: non-x86_64 arch or x32 syscall numbers kill with SIGSYS",
                                    "limits": "rlimits AS/FSIZE/NOFILE/NPROC/CORE/CPU + parent wall-clock process-group kill", "tmp": "per-run private /tmp (symlink to a fresh nobody-owned 0700 dir, removed after the run)",
                                    "not_provided": "pid/net/mount/ipc/uts namespaces (seccomp on the Kaggle host denies unshare)"}
    elif r.phase_status == "ran" and items and items[0][0]:
        p = items[0][1]
        host = _host_ns()
        p["namespaces_differ_from_host"] = {n: (host.get(n) != v) for n, v in p["namespaces"].items()} if host else "host namespaces unreadable"
        rec["python_jail"] = p
    else:
        rec["python_jail"] = {"error": r.phase_status, "term": term, "stderr_tail": r.stderr_tail}
    mod = engine.java_mod()
    mod.JDK = cfg.jdk
    w = Path(tempfile.mkdtemp(prefix="jprobe-"))
    try:
        os.chmod(w, 0o755)
        ver = {}
        for tool in ("java", "javac"):
            x = engine.spawn(mod.jail(w, [cfg.jdk + "/bin/" + tool, "-version"], rw=False), b"", 30, 1 << 16)
            ver[tool] = (x.stderr_tail + x.stdout).strip().splitlines()[:1] if x.returncode == 0 else {"error": x.stderr_tail[:200]}
        rec["java_toolchain"] = ver
    finally:
        import shutil
        shutil.rmtree(w, ignore_errors=True)
    rec["java_encoding_pins"] = ["LC_ALL=C.UTF-8", "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-Dstdin.encoding=UTF-8", "javac -encoding UTF-8"]
    return rec


# ---------------------------------------------------------------- replay of the existing receipts
_P2 = "results/raw/codex-pilot1-20261016/fixed_v2/"
_J1 = "results/raw/codex-pilot1-java-20261016/"
PINNED = {
    _P2 + "reports/redteam_sandbox.json": "7016c03b9cf9c0c13500fc95d2c9116455c86d446ccceda25dca899552dc7eb0",
    _P2 + "reports/sandbox_reference_grade.json": "f79f68b630d48441c32664fc39970e110a1d3f7b5673abe50e9da0d7b7c1e6b1",
    _P2 + "reports/wrong_after.json": "553d8adfbb52a2b087619528bdc4e28d8367edaaa3ac5f357d335308e8b2bc93",
    _J1 + "fixed_v1/reports/java_sandbox_reference.json": "e19ce82e5c18fdb1f0e7c46da2d8cbbf42205b476e63284804e5922d192ff39b",
    _J1 + "fixed_v1/reports/java_redteam_sandbox.json": "f454edd4613a47a6123ce49ef5c173e44416c795edc7e68c4dc6b6c3f50226b2",
    # the scripts those receipts came from: they must stay byte-identical
    _P2 + "grader/sandbox/grade_sandboxed_v2.py": "9162aaa7a79b645c7b89eae0a9fa4d01562726692ff2d1a32cff7d1ee7b7e0b0",
    _P2 + "grader/sandbox/redteam_run_v2.py": "595fdd307ad135f6c395446a96b9de7337d41a0b4f8eb39d7a09ae3e5f46f09b",
    _P2 + "grader/sandbox/runner_v2.py": "5771953e35f42099e865a999a27668979d0c775c9537f8e6ac292687dab10941",
    _P2 + "grader/oracles.py": "7b0a497d3acf4f867b09e855b5b8491e0e28404550a10f6545575d0edcc747e8",
    _P2 + "grader/wrong_solutions_v2.py": "80deeb93f73d6727f247145bf6d54e1f74c84fd7ed8695da0965f1913994b4fb",
    _J1 + "fixed_v1/grader/grade_sandboxed_java_v1.py": "d98515efd29f13b2c573627e39c4ca89696e0318292f10e332a24087d69621e2",
    _J1 + "kit/Pilot1_Java/worker/support/Json.java": "e4374d910dbde3020604a6a2797a08c4118ea7680e8f0f885cdae74e7feacdbd",
    _J1 + "kit/Pilot1_Java/worker/support/Runner.java": "201e5a44a1d5d8759edd022972add3299f411a14e20d78bae3a3a27d97f3b316",
    _J1 + "kit/Pilot1_Java/grader/ReferenceSolutions.java": "0f9a179b3c98156ab74f9c407c4f9014c6c171fada406667388c03577042f912",
}


def _load_json(rel: str):
    return json.loads((K.REPO / rel).read_text(encoding="utf-8"))


def replay_existing_receipts() -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for rel, want in PINNED.items():
        p = K.REPO / rel
        got = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if p.is_file() else None  # LF-normalized: robust to checkout line endings
        files[rel] = {"sha256": got, "expected": want, "match": got == want}
    summ: Dict[str, Any] = {}
    try:
        rt = _load_json(_P2 + "reports/redteam_sandbox.json")
        summ["python_redteam"] = {"all_pass": rt["all_pass"], "tests": {k: v["pass"] for k, v in rt["tests"].items()}}
        ref = _load_json(_P2 + "reports/sandbox_reference_grade.json")
        summ["python_reference"] = {"solved": ref["solved"], "tasks": ref["tasks"]}
        wr = _load_json(_P2 + "reports/wrong_after.json")
        summ["python_wrong_solutions"] = {"caught": wr["caught"], "wrong_solutions": wr["wrong_solutions"], "escaped": wr["escaped"]}
        jr = _load_json(_J1 + "fixed_v1/reports/java_sandbox_reference.json")
        summ["java_reference"] = {"solved": jr["solved"], "tasks": jr["tasks"]}
        jt = _load_json(_J1 + "fixed_v1/reports/java_redteam_sandbox.json")
        tail = jt["results"][0].get("stderr_tail", "")
        summ["java_redteam"] = {"hostile_candidate_solved": jt["solved"], "status": jt["results"][0]["status"],
                                "escaped_markers_in_its_own_log": [l for l in tail.splitlines() if l.startswith("ESCAPED")]}
        ok = (rt["all_pass"] and ref["solved"] == ref["tasks"] == 40 and wr["caught"] == wr["wrong_solutions"] and not wr["escaped"]
              and jr["solved"] == jr["tasks"] == 40 and jt["solved"] == 0)
    except (OSError, KeyError, ValueError, IndexError) as e:
        summ["error"] = repr(e)
        ok = False
    return {"intact": bool(ok and all(f["match"] for f in files.values())), "files": files, "summaries": summ}
