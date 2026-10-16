# WSL feasibility check v1 (2026-10-14): STOPPED, question not answered

**Result: 0 of 2 qualified, because nothing reached the test or replay stage. The cause is the host machine, not the tasks: the C: drive filled up
(0.2 GB free) while the first task's dependencies were installed into the WSL disk, and the WSL VM crashed twice ("Bus error", VM restart).**

## What was done
1. Selection rule written first (`SELECTION.md`, plus a method addendum written before any setup ran). Universe: 1,741 instance ids with mixed
   resolved/unresolved runs; excluded by repo: RootSE 39, CodeTraceBench 2, LAYA nebius SWE-agent 862; eligible 838; seeded shuffle (20261014);
   13 rows fetched one at a time from the datasets-server `filter` endpoint (`fetched.jsonl`, reject reasons: apt pre_install 6, conda `packages` 2,
   non-Python test file 1). Tasks: `iterative__datachain-206`, `googleapis__python-storage-1357`. Reserves: `pvlib__pvanalytics-128`,
   `stephenhillier__starlette_exporter-89`. The first fetch pass timed out (endpoint 500 "index loading"); its partial log is kept as
   `fetched-attempt1-timeout.jsonl` and the deterministic rerun produced the same first accept.
2. Prefixes extracted with a streamed parquet read (one row, `tools/wsl_feasibility_prep_v1.py`); both selected runs reach the decision point
   (first test run) at tool call 7 (6 environment steps + 1 think). Note: the datachain prefix contains `pip install -e .`, which needs network in
   the original run; under the no-network rule it would very likely MISMATCH (not measured).
3. Setup of task 1 (network on, logged): clone + checkout OK, Python 3.9.21 (exact version from the task's `environment`) OK, venv OK;
   `uv pip install` of the 216 pinned requirements (torch + NVIDIA CUDA wheels, about 4 GB) crashed the VM twice (`SETUP-*.log`).
4. Task 2 never started (WSL Input/output error).

## Disk state left behind (needs the user)
- WSL disk `C:/Users/Satvik/AppData/Local/wsl/{2aae1072-9bb6-4f3e-9d9c-f4de3b79df9e}/ext4.vhdx`: 17.2 GB. Inside it, this check left
  `/var/lib/fable-runs/feas-v1/` = 9.5 GB (uvcache 6.0 GB, datachain env 3.5 GB). Not deleted (CLAUDE.md: deleting outside the project folder needs a yes).
- Deleting those files inside WSL does NOT give C: space back; the vhdx must also be compacted (e.g. `wsl --shutdown` then `wsl --manage Ubuntu-24.04 --set-sparse true`
  or Optimize-VHD / diskpart compact). Both need the user's yes.

## What a rerun needs
- Free host space first (at least ~15 GB on the drive holding the vhdx), or move the vhdx to D: (D: has 1.2 GB free now).
- Task 1 alone needs ~4 GB of CUDA wheels it never uses (CPU sandbox); a pre-registered CPU-only torch index would change the environment versus the
  recorded run and would need its own rule. The rule could instead exclude tasks whose `requirements` contain torch/CUDA wheels (new versioned rule).
- Tools: `tools/wsl_feasibility_select_v1.py`, `tools/wsl_feasibility_prep_v1.py`, `tools/wsl_feasibility_traj_v1.py`, `tools/wsl_feasibility_run_v1.py`
  (runner untested past setup: the test, replay, editor-emulation and hash code never ran).
