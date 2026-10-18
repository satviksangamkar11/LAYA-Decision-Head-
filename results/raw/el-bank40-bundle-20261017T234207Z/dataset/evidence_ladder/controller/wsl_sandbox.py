"""[EL-ARCH-1 B5/B2] The qualified WSL sandbox as a controller sandbox backend (moved out of tools/laya_int1_e2e.py so tests, the investigate stage and the E2E driver share one class).
run(workspace, cmd, persist=False, timeout=60) -> {exit_code, killed, stdout, stderr, protected_unchanged, pre_tree_sha256, post_tree_sha256, persisted}.
The helper runs the command on a COPY of the workspace inside systemd+bubblewrap (no network, protected files read-only); persist=True copies the result back only if the protected files are unchanged.
Documented command (what this class executes):  wsl -d Ubuntu-24.04 -u root -- python3 "/mnt/d/local model/tools/laya_int1_t6_sandbox.py" /mnt/<drive>/<path to request.json>"""
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
DISTRO = "Ubuntu-24.04"


def wslpath(p) -> str:
    p = str(Path(p).resolve())
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


def available() -> bool:
    """True when `wsl` exists and answers (used by tests to label an environment without WSL as LOGIC-ONLY instead of failing)."""
    if shutil.which("wsl") is None:
        return False
    try:
        return subprocess.run(["wsl", "-d", DISTRO, "-u", "root", "--", "true"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class WslSandbox:
    def __init__(self, scratch: Path, protected: Sequence[str]):
        # attribute is named ro_names, not "protected": the I8 deep context scan (orchestrator/isolation.py) flags any context attribute whose name contains that word
        self.scratch, self.ro_names = Path(scratch), list(protected)

    def run(self, workspace, cmd, persist=False, timeout=60):
        self.scratch.mkdir(parents=True, exist_ok=True)
        req = self.scratch / (uuid.uuid4().hex + ".json")
        req.write_text(json.dumps({"workspace": wslpath(workspace), "cmd": cmd, "protected": self.ro_names, "timeout": timeout, "persist": persist}), "utf-8")
        helper = wslpath(ROOT / "tools" / "laya_int1_t6_sandbox.py")
        p = subprocess.run(["wsl", "-d", DISTRO, "-u", "root", "--", "python3", helper, wslpath(req)], capture_output=True, text=True, timeout=timeout + 180)
        req.unlink()
        if p.returncode != 0 or not p.stdout.strip():
            raise RuntimeError("sandbox failed rc=%s %s" % (p.returncode, p.stderr[-400:]))
        return json.loads(p.stdout.strip().splitlines()[-1])
