# A4 leakage firewall, v1 (column level, written before any data was opened)

Status: v1, 2026-10-01. Frozen by hash in `results/raw/a4-leakage-rules-v1.sha256` before bulk compilation. A change after that is a
new version with a written reason (v1.1 and later), never an edit. Trajectory-internal fields are classified in a later version, after
the read-only inspection of about 20 rows; until then they are **UNKNOWN and not allowed in a state**.

## The rule

A decision state may contain only information observable by the agent at the decision point `t`. Cortex-1 shows what happens
otherwise: its training states carried a `metadata` field holding the answer (`risk_tier`, `cwe`, ...) and its own validation reported
exactly 1.0 [M, from its generator code]. Everything below exists so we cannot repeat that.

## Field classes

`OBSERVABLE_AT_T` may enter the state. `LABEL_SIDE` may be used only to build the label or its evidence tier. `NEVER_INPUT` is not allowed
in any state, candidate or prompt. `UNKNOWN` is treated as NEVER_INPUT until classified.

| Field (SWE-rebench-openhands) | Class | Why |
|---|---|---|
| issue text, repository files the agent has already read, earlier messages, earlier tool outputs, the diff as of `t` | OBSERVABLE_AT_T | the agent had them |
| the action at `t` and everything after it | LABEL_SIDE (action) / NEVER_INPUT (later events) | it is the label, or the future |
| `model_patch` | NEVER_INPUT | the final change |
| `resolved`, `exit_status`, `gen_tests_correct`, `pred_passes_gen_tests` | LABEL_SIDE | episode outcome; evidence tier only |
| `trajectory_id`, `instance_id` | NEVER_INPUT | identifiers; split keys only |
| `repo` | NEVER_INPUT | a repository prior is a shortcut; used for splitting only |
| names of the instance's failing tests (FAIL_TO_PASS style fields, if present) | NEVER_INPUT | they point at the bug |
| `tools` | UNKNOWN | decide after inspection |
| SWE-smith: `patch`, `model`, `resolved`, `traj_id`, `instance_id` | NEVER_INPUT / LABEL_SIDE (`resolved`) | the agent's model name is a shortcut |

## Automated checks (the compiler aborts on any failure; each result is written to `results/raw/`)

1. **Allowlist serialiser.** The state text is built only from event kinds on an explicit allowlist and only from events with index below `t_index`. Unknown kinds raise.
2. **Time order.** Every record stores `t_index`; the label event index is greater than every event used in the state.
3. **Future-string scan.** Distinctive strings of the final patch's added lines and of the label action must not appear in the state unless they also appear in an earlier tool observation from the repository. The rate is reported; a future-sourced hit fails.
4. **Label-vocabulary scan.** No state key or text field named like a label or outcome (`resolved`, `exit_status`, `label`, `risk`, `answer`, ...).
5. **Shortcut baselines**, reported with every benchmark: candidate text only with the state blanked, TF-IDF + logistic regression on candidate text, and the position of the true action (uniform, tested). Head metadata excludes `source` (already enforced in `state/schema.py`). Candidate ids and order are randomised.
6. **Near-duplicate states** across splits (hash and similarity) fail.
7. **Splits** by repository family; instance ids disjoint; a split manifest is hashed (RB3).

## Label evidence tiers (ACTION)

- **STRONG:** the action directly contributes to the demonstrated final change and has supporting evidence (for example an edit to a file in the final patch after the agent read it, or the test that later passes).
- **WEAK:** the action occurred in a trajectory that later resolved. It is not assumed correct.
- **UNKNOWN:** not enough evidence.
- **NEGATIVE:** only with contradicting evidence about that chosen action (it errored on execution, or the agent undid it). A NEGATIVE chosen action gives **no positive label**, and it does not make the other candidates correct.
- Absence of success is not failure: an unresolved trajectory yields no positive and no negative labels.

## What the labels mean, and the claims they allow

Unchosen candidates are **not negatives**: a softmax over candidates pushes them down regardless, so the benchmark metric is
**agreement with the trajectory's action, reported per tier**, never "correctness". Tier weights in the loss are fixed in
`configs/thresholds.toml` before any training run. Baselines (`role=BASELINE`) never enter a target (`usable_as_target`).

## A5 benchmark axes (first class, not an afterthought)

normal order; shuffled order; +3, +6 and +10 distractor candidates; NONE inserted; wording paraphrase of the instruction. The measured
failure curve of `laya-typed` (0.740, 0.663, 0.612, 0.532 on typed-decisions with 0, 3, 6, 10 distractors) is the bar the head must beat on
the coding benchmark.

## Pending

Event schema of the trajectories, which event kinds are observable, token budgets for the state (tail-first, raise on overflow), and the
`tools` field. These are decided after the read-only inspection and recorded as v1.1 before compilation.
