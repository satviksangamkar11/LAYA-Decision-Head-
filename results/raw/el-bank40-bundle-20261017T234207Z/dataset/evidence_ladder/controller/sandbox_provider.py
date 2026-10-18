"""[MXC replacement] The sandbox execution-provider port and the backend factory. Pure standard library.

PORT (unchanged from WslSandbox; callers: AgenticRun._run, replay.RecordingSandbox, the harness tools):
    sandbox = make_sandbox(scratch_dir, protected_names, backend=None)
    result  = sandbox.run(workspace, cmd, persist=False, timeout=60)
    result keys (RESULT_KEYS): exit_code:int, killed:bool, stdout:str, stderr:str, protected_unchanged:bool, pre_tree_sha256:str, post_tree_sha256:str, persisted:bool
    * infrastructure failure RAISES (RuntimeError / OSError / ValueError / subprocess.TimeoutExpired); a backend never returns an invented result;
    * the workspace is only changed when persist=True AND the protected files are unchanged AND the run was not killed;
    * optional attribute `env_label` (journal environment identity; default "wsl-bwrap-systemd" = the previous hard-coded value); backend diagnostics live on the object (`last_diagnostics`), never in the result.

BACKENDS (explicit, never silently switched):
    wsl  the qualified WSL2 + systemd + bubblewrap sandbox (evidence_ladder/controller/wsl_sandbox.py). DEFAULT until the MXC gates in MXC_MIGRATION.md are met.
    mxc  Microsoft MXC (processcontainer) via evidence_ladder/controller/mxc_sandbox.py. Selected with backend="mxc" or LAYA_SANDBOX_BACKEND=mxc.
There is NO "auto" and NO fallback: asking for mxc on a host that cannot run it raises SandboxUnavailable (fail closed); the caller decides whether to switch to wsl, and the backend that ran is
named in the journal environment id, in the receipts and in the diagnostics."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from evidence_ladder.controller.mxc_sandbox import SandboxUnavailable

RESULT_KEYS = ("exit_code", "killed", "stdout", "stderr", "protected_unchanged", "pre_tree_sha256", "post_tree_sha256", "persisted")
RESULT_TYPES = {"exit_code": int, "killed": bool, "stdout": str, "stderr": str, "protected_unchanged": bool, "pre_tree_sha256": str, "post_tree_sha256": str, "persisted": bool}
DEFAULT_BACKEND = "wsl"
BACKENDS = ("wsl", "mxc")
ENV_VAR = "LAYA_SANDBOX_BACKEND"


def selected_backend(backend: Optional[str] = None) -> str:
    b = (backend or os.environ.get(ENV_VAR) or DEFAULT_BACKEND).strip().lower()
    if b not in BACKENDS:
        raise SandboxUnavailable("unknown sandbox backend %r (allowed: %s); there is no auto mode" % (b, ", ".join(BACKENDS)))
    return b


def make_sandbox(scratch, protected: Sequence[str], backend: Optional[str] = None, **kw: Any):
    """-> a sandbox object for the named backend. mxc: raises SandboxUnavailable / UnsupportedPolicy when its capability check or its resource-limit enforcement fails (no fallback)."""
    b = selected_backend(backend)
    if b == "wsl":
        from evidence_ladder.controller.wsl_sandbox import WslSandbox
        return WslSandbox(Path(scratch), protected)
    from evidence_ladder.controller.mxc_sandbox import Limits, MxcSandbox
    kw.setdefault("persistent", True)      # benchmark: a one-shot node start per call is slower than WSL at p95; the persistent bridge keeps NO candidate state (fresh run directory per call, tested)
    kw.setdefault("limits", Limits())      # parity with the WSL cgroup profile (2 GB, 256 processes, 4 cores); the constructor raises if they cannot be enforced
    return MxcSandbox(Path(scratch), protected, **kw)


def backend_available(backend: Optional[str] = None) -> Dict[str, Any]:
    """-> {"backend", "available": bool, "reasons": [...]}; executes no workload (wsl: runs `true`; mxc: one SDK probe)."""
    b = selected_backend(backend)
    if b == "wsl":
        from evidence_ladder.controller.wsl_sandbox import available
        ok = available()
        return {"backend": b, "available": ok, "reasons": [] if ok else ["wsl does not answer"]}
    from evidence_ladder.controller.mxc_sandbox import capabilities
    c = capabilities()
    return {"backend": b, "available": c["available"], "reasons": c["reasons"], "tier": c.get("probe_tier")}


def validate_result(r: Any) -> Dict[str, Any]:
    """Contract check used by tests and the benchmark: every key present with the right type. bool is checked before int (bool is an int subclass)."""
    if not isinstance(r, dict):
        raise ValueError("sandbox result is not a dict")
    for k in RESULT_KEYS:
        if k not in r:
            raise ValueError("sandbox result lacks %r" % k)
        t = RESULT_TYPES[k]
        if (t is int and (isinstance(r[k], bool) or not isinstance(r[k], int))) or (t is not int and not isinstance(r[k], t)):
            raise ValueError("sandbox result %r has type %s, expected %s" % (k, type(r[k]).__name__, t.__name__))
    return r


def main(argv=None) -> int:
    """Direct, model-independent entry point (Claude Code / Codex / Qwen controller / any MCP client can shell out to it):
        python -m evidence_ladder.controller.sandbox_provider --workspace DIR --cmd "python3 test_x.py" [--protected test_x.py ...] [--persist] [--timeout 60] [--backend wsl|mxc] [--scratch DIR]
    Prints ONE JSON object: {"result": {...the 8 contract keys...}, "backend": ..., "env_label": ...} or {"error": {"type", "message"}} with exit code 3 (infrastructure failure, fail closed)."""
    import argparse
    import json
    import tempfile
    ap = argparse.ArgumentParser(prog="sandbox_provider")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--protected", nargs="*", default=[])
    ap.add_argument("--persist", action="store_true")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--backend", choices=BACKENDS)
    ap.add_argument("--scratch")
    a = ap.parse_args(argv)
    scratch = Path(a.scratch) if a.scratch else Path(tempfile.mkdtemp(prefix="laya-sandbox-"))
    try:
        sb = make_sandbox(scratch, a.protected, a.backend)
        try:
            r = validate_result(sb.run(a.workspace, a.cmd, persist=a.persist, timeout=a.timeout))
        finally:
            if hasattr(sb, "close"):
                sb.close()
    except Exception as e:      # noqa: BLE001  every infrastructure failure is reported, none is turned into a result
        print(json.dumps({"error": {"type": type(e).__name__, "message": str(e)[:600]}}))
        return 3
    print(json.dumps({"result": r, "backend": selected_backend(a.backend), "env_label": getattr(sb, "env_label", "wsl-bwrap-systemd")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
