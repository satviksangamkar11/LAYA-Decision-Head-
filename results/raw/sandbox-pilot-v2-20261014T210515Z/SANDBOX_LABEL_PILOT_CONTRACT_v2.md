# Sandbox label pilot: experiment contract v2 ("option B: late decision points"), 2026-10-15

Approved: user GO (chat, 2026-10-15), narrowed by the user the same day, before any execution, to a **one-task feasibility test**.
Binding numbers and rules: `configs/thresholds_sandbox_pilot_v2.toml`; fidelity: `configs/fidelity_v3.toml` + `configs/fidelity_v4.toml`
(test-duration masking only, pre-registered before any replay). Code: `tools/sandbox_pilot_v2.py` (+ `tools/sandbox_pilot_prep_v2.py`),
which import the v1 tools unchanged.

**Why v2.** v1 gave no labels because the 15-step continuation started ~65 steps before solved runs ended (decision points at step ~7,
solving runs 37-86 actions, all 12 rollouts 0/3), and the GLM test-or-edit candidate came back as a read action twice.

## Design
- **Decision point:** in one seeded resolved recorded run, the state just before the recorded action at position n_actions - 10
  (10 recorded actions remain). Results concern late-task decisions only.
- **Qualified state** only if (a) the whole prefix replays offline under fidelity v3 (v4 for test-duration tokens), editor edits emulated,
  in three replays; (b) ORACLE: the recorded remaining 10 actions, replayed from the state, make every FAIL_TO_PASS and sampled
  PASS_TO_PASS test pass (test_patch applied only at evaluation, twice); (c) replays B and C give identical decision-point hashes.
  Recorded for information: whether the decision-point state already passes the tests with no further action.
- **Candidates:** R recorded; S read/search and T test-or-edit from GLM-5.3 seeing only what the agent saw (problem statement + prefix with
  replayed observations). Explicit one-line JSON schema with example; up to 4 attempts with the error fed back; an action equivalent to R
  (whitespace-normalised) is recorded and re-asked once. Unparsable / wrong type / cannot execute = INVALID, never "worse".
- **Continuation and evaluation:** identical to v1 (GLM-5.3 full reasoning, streamed, 15 steps, 20-min environment cap, offline, no secrets,
  evaluation twice).
- **Staged rollouts (user):** round 1 = 1 rollout per valid candidate (fresh restore each). If none resolves, STOP and report. Otherwise
  complete to 3 per candidate and compute labels (v1 rules: BETTER/WORSE if |r_a - r_b| >= 2/3 with all 6 rollouts clean; both 3/3 or both
  0/3 = "equal on the measured test outcome", not "equally good"; UNKNOWN otherwise; INVALID).
- **Scope:** 1 task (starlette_exporter-89, then python-storage-1357; at most 2 more walk ids if neither qualifies), <= 9 rollouts.
- **Robustness:** every phase runs detached in WSL, logs to the results folder, resumes from files; nothing completed is resent.
- **Data:** `results/raw/sandbox-pilot-v2-<ts>/`, create-only, secret-scanned, kept apart from RootSE / CAL / TEST / VAL; no training,
  no Kaggle, no commit.

## Verdict rule
GO = qualified late state, >= 2 valid candidates, round 1 has a resolving candidate, and >= 1 clean BETTER/WORSE pair after round 2 with
<= 20% environment-failure rollouts. NO-GO = no qualified state within the cap, > 20% environment failures, or < 2 valid candidates.
HOLD = otherwise.
