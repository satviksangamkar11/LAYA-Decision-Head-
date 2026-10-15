# WSL feasibility check v2: selection rule (written BEFORE any v2 fetch, draw or run)

User yes (chat, 2026-10-14): rerun the two-task WSL feasibility check as v2. Same question as v1: can a SWE-rebench task give a COMPLETE,
REPRODUCIBLE starting state in the qualified WSL sandbox? v1 (`results/raw/wsl-feasibility-v1-20261014T120445Z/`) stopped because the host C:
drive filled during a torch/CUDA install; its files are not modified. Scope unchanged: no GLM/NVIDIA calls, no candidates, no continuations,
no training, no Kaggle, no bulk downloads.

## Unchanged from v1 (v1 SELECTION.md and its method addendum apply verbatim unless changed below)
- Universe: mixed-outcome instance ids from `data/nebius_swe_rebench/trajectories.parquet`; repo exclusions RootSE / CodeTraceBench / LAYA
  (nebius SWE-agent groups); seeded shuffle `random.Random(20261014)`; walk in order; one task per repo.
- v1 row checks (install_config python version, no pre_install, no env_yml, packages empty or requirements.txt, pip/python-only install,
  pure-Python patch files, FAIL_TO_PASS non-empty). The v1 `pre_install` / non-pip `install` checks are what exclude tasks that need
  Docker-only system packages (apt etc.); they are counted under that name in v2.
- Run choice (seeded choice among resolved runs sorted by trajectory_id), decision point (first edit or test run), test condition
  (FAIL_TO_PASS fail and PASS_TO_PASS pass, sample of 20 if more, on base_commit + test_patch, twice, network off), replays A/B/C from fresh
  copies of the setup snapshot, compare_v2 per step, workspace hash (no `.git/`, plus HEAD and `git status --porcelain`), reproducible iff
  hash(B) == hash(C), socket + DNS probe must fail. QUALIFIED only if setup ok, probe fails, test condition holds in both runs, every prefix
  environment step MATCHes in A, B and C, and hash(B) == hash(C).

## Row reuse
Rows already fetched in v1 are reused: the 4 full rows (`task-*.json`) are re-checked under v2; the 9 v1-rejected rows keep their v1 reject
reason (the v1 checks are unchanged, so the verdict is the same). Any other id is fetched one row at a time from the datasets-server `filter`
endpoint, as in v1. Every visited id and its v2 verdict is logged in `fetched_v2.jsonl`.

## v2 exclusions (applied after the v1 checks, in this order; each one's count is recorded)
- **X1 heavy dependencies:** reject if the task's pinned `requirements` (same line filter as the v1 runner: non-empty, no `#`, `-e `, `@ file:`,
  `git+`), `install` or `pip_packages` name torch / torchvision / torchaudio, tensorflow*, jax / jaxlib, any `nvidia-*` package, or a CUDA
  wheel (`cuda*`, `cupy*`, `*+cu1xx` local version); or if the filtered pinned list has more than 80 lines.
- **X2 network-dependent prefix:** compute the selected run's prefix (up to the decision point, v1 rule). Reject if any `execute_bash` command
  in it matches `(pip3?|uv|poetry|conda|mamba)` ... `(install|sync|add)`, `curl`, `wget`, `git clone|fetch|pull`, or `apt` / `apt-get`
  (regex in `tools/wsl_feasibility_select_v2.py`). Such a prefix cannot be reproduced offline. A task without a decision point is rejected
  as `no_decision_point`.
- Accepted #1 and #2 are the tasks, #3 and #4 the reserves.

## Disk guard (mandatory) and reserves
- Before every stage (setup, each test run, each replay) the runner reads free space on C: (statvfs of `/mnt/c`, which equals
  PowerShell `(Get-PSDrive C).Free`; checked once side by side) and on the WSL root; the orchestrator also reads PowerShell
  `(Get-PSDrive C).Free` before and after each task. All readings go to `DISK.jsonl`.
- ABORT the task cleanly if C: free < 4 GiB before a stage, or, during setup (polled every 5 s), if C: free < 4 GiB or the task's work dir
  (repo + environment + its own uv/pip caches) exceeds 2 GiB. An abort is recorded as `DISK_GUARD_ABORT`, not as a reproducibility verdict.
- A reserve replaces a task only if setup is impossible for a reason unrelated to reproducibility (dependency gone, DISK_GUARD_ABORT);
  the reason is recorded. Reserves are used in order.
- Cleanup after each task: delete only `/var/lib/fable-runs/feas-v2/<instance_id>/` and `/var/lib/fable-runs/feas-v2/runs/*` (paths this
  check created), then `fstrim -av` as root. Nothing else is deleted.

## Bugs in the untested v1 runner code
The v2 runner is `tools/wsl_feasibility_run_v2.py` (copy of v1 plus the disk guard, per-task caches and cleanup). If a test / replay /
editor-emulation / hash stage fails because of a bug in that code, the bug is fixed, recorded in `FIXES.md`, and the stage is rerun; a code
bug is never a reproducibility verdict. The qualification rule above is not changed by any fix.
