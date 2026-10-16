# WSL feasibility check v1: selection rule (written BEFORE any task row was fetched for selection)

Scope (user, chat, 2026-10-14): two-task reproducibility check only. No GLM/NVIDIA calls, no candidates, no continuations, no training, no Kaggle.

Disclosure: before this file was written, ONE datasets-server row (`numpy__numpydoc-101`, the first id printed by a schema check) was fetched only
to learn the row schema. It gets no special treatment below (it is eligible or not by the same rule).

## Universe
1. `data/nebius_swe_rebench/trajectories.parquet`, columns instance_id / repo / resolved only: instance ids with >= 1 run resolved=1 AND >= 1 run resolved=0.
2. Repo exclusions (key = `owner__repo`, lower-cased, substring match = superset, conservative):
   - RootSE: the key occurs in any file name under the RootSE copy `C:/Users/Satvik/AppData/Local/Temp/rootse_apb/rootse/data/**` (dengdan1999/RootSE).
   - CodeTraceBench: the key occurs in any `traj_id` or `artifact_path` of `results/raw/ctb-k1-inventory-20261014T2246Z-full/RUNS.jsonl`.
   - LAYA: the key equals the lower-cased `group` of any row of `data/coding-decision-cases/data/*.parquet` whose `source` contains `nebius/SWE-agent`.
3. Eligible list = sorted unique instance ids left; `random.Random(20261014).shuffle(list)`.

## Draw
Walk the shuffled list. For each id fetch its SWE-rebench row (datasets-server `filter` endpoint, split `test`, else `filtered`; one row, KB). ACCEPT if all hold:
- repo differs from every already accepted task;
- `install_config` present with a `python` version; `pre_install` empty; `env_yml_path` empty; `packages` empty or `requirements.txt`;
  `install` uses only `pip`/`python` commands (no apt, conda, make, docker);
- "pure Python" proxy: every file touched by `patch` and `test_patch` ends in `.py` (or is a non-code text file such as .rst/.md/.txt/.cfg/.toml/.ini);
- FAIL_TO_PASS non-empty.
Every fetched row and its reject reason is logged in `fetched.jsonl`. Accepted #1 and #2 = the tasks; #3 and #4 = reserves. A reserve replaces a task only
if setup is impossible for a reason unrelated to reproducibility (e.g. a dependency no longer downloadable); the reason is recorded.

## Run and decision point (pre-registered)
- Run: the resolved runs of the task sorted by trajectory_id; `random.Random(20261014).choice`.
- Decision point: the first assistant tool call that is (a) a file edit: `str_replace_editor` with command create / str_replace / insert / undo_edit,
  or (b) a test run: `execute_bash` whose command matches `pytest|py\.test|unittest|\btox\b|\bnox\b|runtests|run_tests|manage\.py test|make test|python[0-9.]* \S*test\S*\.py`.
  The prefix is every tool call strictly before it. Tool calls without environment effect (think, task_tracker) are recorded as NO_ENV_EFFECT.
- Test condition: FAIL_TO_PASS and PASS_TO_PASS are run on base_commit + `test_patch` (a separate copy; the agent's starting state is base_commit
  without test_patch, as in the run). PASS_TO_PASS: all if <= 20, else a seeded sample of 20 (`random.Random(20261014).sample`).

## Addendum (method details fixed before any setup or replay ran)
- Workspace layout in the sandbox mirrors the run: the repo is bind-mounted at the recorded `/workspace/<dir>` and the environment at
  `/opt/conda/envs/testbed` (the recorded interpreter path), so no path rewriting is needed. Python = the exact version in the task's `environment`
  field (uv-managed build); packages = the task's `requirements` pins (self/editable lines dropped), then `install_config.install` and `pip_packages`.
- Recorded bash observations: the OpenHands footer lines (`[The command completed with exit code N.]`, `[Current working directory: ...]`,
  `[Python interpreter: ...]`, `[Command finished with exit code N]`) give the recorded exit code and are removed before `compare_v2`.
  Editor `view` calls are emulated (directory: up-to-2-level listing excluding hidden items; file: `cat -n` format) and compared with
  `compare_v2` with exit 0 on both sides.
- Decision-point workspace hash (the reproducibility criterion): sha256 over every file of the repo EXCEPT the `.git/` directory, plus `git rev-parse HEAD`
  and `git status --porcelain` output. (`.git/index` stores stat times refreshed by `git status`, so it is not content.) The full tree hash including `.git`
  is reported for information only.
- Replays: A (step 3), then B and C, each from a fresh copy of the setup snapshot. Reproducible iff hash(B) == hash(C) (A reported too).
- Network-off evidence: a socket + DNS probe runs inside the replay sandbox of each task and must fail.
