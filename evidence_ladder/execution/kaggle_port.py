"""[P5 Kaggle] Sandbox PORT over the existing Kaggle jail (execution/kaggle_jail.py, unchanged): run(workspace, cmd, persist=False, timeout) -> the 8 contract fields.
Each call: the jail launcher copies the workspace's regular files into its private /work, runs `cmd` there inside the chroot as uid 65534 with NO_NEW_PRIVS and the
seccomp filter (socket() of every family -> EPERM, ptrace denied, x32/foreign arch killed), and (persist=True only) copies regular files back. The launcher is
serialised by flock and reaps every leftover uid-65534 process before and after each run. This lets the TRUSTED VERIFIER (trusted_verify.py, LcbRunner) run on Kaggle
exactly as on MXC: one fresh process per input, verdicts on the host. Infrastructure failures (launcher setup, exit 125) RAISE. Not provided by the jail (stated):
pid/net/ipc namespaces (network is blocked by seccomp + the kernel's internet setting) and a memory rlimit in this mode (wall-clock kill + reap are the limits)."""
from __future__ import annotations

import hashlib
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict

from . import kaggle_jail as KJ

MAX_CAPTURE = 1 << 20


def _tree(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(x for x in Path(p).rglob("*") if x.is_file()):
        h.update(f.relative_to(p).as_posix().encode()); h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


class KaggleJailPort:
    env_label = "kaggle-jail"

    def __init__(self, jdk: str = ""):
        self.jdk = jdk or os.environ.get("EL_KJAIL_JDK", "")
        self.calls = 0

    def provenance(self) -> Dict[str, Any]:
        return {"backend": "kaggle-jail", "rootfs": str(KJ.ROOTFS)}

    def run(self, workspace, cmd: str, persist: bool = False, timeout: int = 60) -> Dict[str, Any]:
        ws = Path(workspace)
        pre = _tree(ws)
        argv = KJ.java_jail(ws, shlex.split(cmd), rw=persist, jdk=self.jdk)
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        killed = False
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            killed = True
            os.killpg(p.pid, signal.SIGKILL)          # the launcher and its jailed child share the session; the next launch reaps any uid-65534 leftover
            out, err = p.communicate()
        self.calls += 1
        if not killed and p.returncode == 125 and b"kjail" in err:
            raise RuntimeError("kaggle jail setup failed: %s" % err.decode("utf-8", "replace")[-300:])
        return {"exit_code": -9 if killed else p.returncode, "killed": killed, "stdout": out[:MAX_CAPTURE].decode("utf-8", "replace"),
                "stderr": err[-65536:].decode("utf-8", "replace"), "protected_unchanged": True, "pre_tree_sha256": pre, "post_tree_sha256": _tree(ws), "persisted": bool(persist)}
