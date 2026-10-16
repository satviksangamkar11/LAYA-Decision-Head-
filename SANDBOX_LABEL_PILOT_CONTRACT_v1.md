# Sandbox label pilot: experiment contract v1 (2026-10-14) — NOT approved to run yet

Goal: measure how many genuinely trustworthy "which next action is better" labels a bounded pilot yields. No training, no Kaggle.

## Fixed choices
- **Where:** the qualified WSL sandbox (bubblewrap, no network during runs, no secrets mounted; the approved local exception in CLAUDE.md).
  Network only in a separate, logged setup step (clone + install), never while a candidate or test runs.
- **Tasks:** 8 tasks from `nebius/SWE-rebench` (CC-BY-4.0; per-task repo, base_commit, install config, test command, FAIL_TO_PASS tests),
  fetched as single rows via the datasets-server API (KB, no bulk download). Selection, fixed before any run: pure-Python repos whose install
  config needs no Docker; NOT in RootSE, CodeTraceBench, or LAYA's nebius SWE-agent repo list (858 groups); 1 task per repo; seeded random
  draw from the eligible list.
- **Starting state:** the task's base commit + the recorded prefix of one existing `nebius/SWE-rebench-openhands` run of that task (on disk),
  replayed command by command in the sandbox. ADMITTED only if every replayed observation matches the recorded one under the existing
  fidelity rule (`compare_v2`, formatting-only differences) and the FAIL_TO_PASS tests fail at the state. Decision point: the step where that
  run made its first edit or first test run (pre-registered rule, so states are not trivial orientation steps).
- **Candidates (3 per state, different action types):** (1) the recorded next action; (2) one read/search action and (3) one test-or-edit
  action, proposed by GLM-5.3 via the NVIDIA endpoint (fixed call logic, logged, no resend). A candidate that does not parse or cannot execute is
  recorded as INVALID, never as "worse".
- **Continuation:** after each candidate, the same fixed policy (GLM-5.3, same prompt, temperature, 15-step budget, 20-min wall cap) continues;
  then the task's FAIL_TO_PASS + PASS_TO_PASS tests run. 3 independent continuations per candidate (nondeterminism), each from a freshly restored
  identical workspace (state hash checked before every rollout).

## Labels (before-action selection only; after-action criticism is a separate objective, not produced here)
Per candidate: resolve rate r = passes / 3. For each candidate pair in a state:
- BETTER / WORSE: |r_a - r_b| >= 2/3 (e.g. 3/3 vs 1/3) and no environment failure in any of the 6 rollouts;
- EQUAL: both 3/3 or both 0/3;
- UNKNOWN: anything else, any timeout, setup failure, flaky test (a test whose result differs between reruns of the same final workspace) or
  invalid candidate. UNKNOWN is never turned into a negative.
Recorded per rollout: action validity, execution result, test result, timeout, environment failure (separate fields), state hash, model calls.
A label says "better for resolving this task under this continuation policy", not "optimal action".

## Data hygiene
Unique ids; source, licence, task, repo, base commit, state hash, action text; task- and repo-level split assigned before generation; secret scan
of every stored artifact; observations and outcomes never enter a before-action model input; the fixed LAYA renderer must keep the task and the
proposed action within 2,048 tokens (checked, reported).

## Size, cost (estimates, unmeasured)
8 tasks x 3 candidates x 3 rollouts = 72 rollouts; about 72 x 15 = ~1,100 NVIDIA calls (Rs 0) and ~10-25 h of sandbox wall time.

## Pass / stop
- PASS (scale up): >= 5 of 8 states admitted AND >= 6 BETTER/WORSE pairs AND >= 2 of them involve a non-edit action.
- STOP and report: < 5 states admitted (replay not reproducible enough), or environment failures in > 20% of rollouts, or 0 BETTER/WORSE pairs.
- If the CodeTracer authors supply a verifiable mapping meanwhile, those labels take priority for after-action criticism; this pilot still
  covers before-action selection, which CodeTraceBench does not label.
