# MXC execution provider: migration and rollback (2026-10-18)

Branch `mxc-execution-provider-2026-10-18` (from `el-arch1-final-closure-2026-10-18`). Not merged to main.

## What exists now
- **One port, two backends.** `evidence_ladder/controller/sandbox_provider.py`: `make_sandbox(scratch, protected, backend=None)` -> object with `run(workspace, cmd, persist=False, timeout=60)` returning the 8 contract keys (`exit_code, killed, stdout, stderr, protected_unchanged, pre_tree_sha256, post_tree_sha256, persisted`).
- `wsl` = the original `WslSandbox` (+ `tools/laya_int1_t6_sandbox.py`, `tools/fable_wsl_runner.py`), **unchanged and still the default**.
- `mxc` = `evidence_ladder/controller/mxc_sandbox.py` -> `tools/mxc_bridge/bridge.mjs` -> Microsoft MXC V1 Node SDK (`@microsoft/mxc-sdk` 1.0.1, MIT), Windows `processcontainer`.
- Selection: `backend=` argument or `LAYA_SANDBOX_BACKEND=wsl|mxc`. **No `auto`, no fallback**: `mxc` on a host that cannot run it raises `SandboxUnavailable`; resource limits that cannot be enforced raise `UnsupportedPolicy`; a command outside the dialect raises `UnsupportedCommand`. Callers treat all of these as tool errors (ERROR evidence, never a pass).
- Direct entry point for any client (Claude Code, Codex, Qwen controller, MCP wrappers): `python -m evidence_ladder.controller.sandbox_provider --workspace DIR --cmd "python3 test_x.py" --protected test_x.py [--persist] [--backend mxc]` prints one JSON object (exit 3 + `{"error": ...}` on infrastructure failure).
- Callers switched to the factory (one line each): `tools/el_arch1_acceptance.py` (default `sandbox_factory`, availability check), `tools/el_arch1_variants.py`, `tools/el_arch1_l2_local.py`, `evidence_ladder/controller/pathb_stored.py`. Tests and the other tools that build `WslSandbox` directly are untouched.
- Journal identity: `AgenticRun.env` hashes the sandbox label. The label is now `getattr(ctx.sandbox, "env_label", "wsl-bwrap-systemd")`, so every WSL/ToySandbox/stub run produces the **same hash as before**; MXC runs hash `mxc-processcontainer`. `RecordingSandbox`/`RecordedSandbox` carry the label and a record adds `sandbox_label` only for non-WSL backends (WSL records are byte-identical).

## Enable MXC on this PC
1. Host: Windows 11 build with Process Isolation (MXC doc: 26100.9278 / 26200.9278 / 26300.9550 / 28000.2804 or later). Checked here: build 26300, `processcontainer` available, tier `base-container`. Windows Home works (it did here).
2. Node >= 24 (here 24.13.1). With Node < 24.21 the bridge uses the captured `run()`; the output cap is then applied after the run (see limits). Node >= 24.21 or >= 26.8 switches automatically to streaming `spawn()` and kills a workload that floods output.
3. `cd tools/mxc_bridge && npm ci --ignore-scripts` (169 MB, 6 packages; no install scripts are run).
4. Python for the sandboxed process: `LAYA_MXC_PYTHON` or the base interpreter of the running Python (`sys.base_prefix\python.exe`; the venv launcher is not used).
5. `set LAYA_SANDBOX_BACKEND=mxc` (or pass `backend="mxc"`).

## Rollback
Unset `LAYA_SANDBOX_BACKEND` (or set it to `wsl`). Nothing else: the WSL implementation, its helper scripts and all `*_wsl` tests are untouched. To remove MXC entirely, delete `tools/mxc_bridge/`, `evidence_ladder/controller/{mxc_sandbox,sandbox_provider,win_job}.py`, `mxc_dialect/` and revert the five one-line caller edits listed above (they only add `make_sandbox` / `env_label`).

## What the MXC backend can and cannot do (verified on this host, see MXC_REPLACEMENT_REPORT.md)
- Commands: `python3|python <args>` run directly (no shell); the POSIX subset of `mxc_dialect/posix_lite.py` (`echo cat ls pwd mkdir touch rm true false`, `; && ||`, `> >>`) runs inside the container. MSYS bash and PowerShell cannot start in the container (exit 0xC0000142), so pipes, `$VAR`, `grep`, `sed`, `git`, other programs are **rejected with `UnsupportedCommand`**, where WSL would run them.
- Isolation observed: host environment not inherited, project/secret files and other user files denied, writes outside the run copy denied, network denied (no network block declared), protected files read-only and hash-verified, timeout kills the whole process tree, links in the input workspace refused, resource limits via a Job Object (2 GB / 256 processes / 4 cores) verified with a memory bomb and a fork bomb.
- Not available here: MXC Linux/macOS backends (NOT_TESTED, wrong OS), WSLC (needs a pulled container image; not approved), Kaggle (bubblewrap is absent and `unshare` is denied there; see `kaggle/mxc_probe_v1.py`).
