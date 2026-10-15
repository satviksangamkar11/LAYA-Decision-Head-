# S5. Repair and stopping policy (Evidence-Ladder v1, P0 + R3, 2026-10-16)

Schema: `schemas_v1.json#/$defs/stop_record`.

## 1. Repair always starts from a known state

- The repair target is chosen by the controller:
  1. most trusted passes;
  2. then validated-check passes;
  3. then cluster size;
  4. then LAYA (only for a validated decision type);
  5. then DeepConf;
  6. then candidate id.
- The repair prompt uses the fixed template (architecture document, section 6).
- The failure quoted in the prompt is the most concrete one available, in this order:
  1. a trusted record's counterexample;
  2. a failing quoted example;
  3. a critic experiment's observed output;
  4. a Qwen test failure.
- The repair output is a **new candidate**: new hash, `parent` = old id, no evidence inherited (S2 section 3).

## 2. Progress

A round makes progress if at least one of these is true:
- a hard/functional requirement row moved to VERIFIED;
- a previously failing trusted check now passes on the new candidate;
- a new failure is localised: a different failing check, input or line than any seen before for this problem;
- a critic experiment's executed result removed a behaviour cluster.

## 3. Stop rules (first match wins; recorded as `stop_record.cause`)

| cause | condition | next |
|---|---|---|
| `decided` | the answer label meets the approval rule (S4 section 2) | commit |
| `same_failure_twice` | the same failure signature (check, input, observed output) appears on two consecutive candidates in one lineage | stop that lineage; others may continue |
| `unrepaired_regression` | a check that passed on the parent fails on the child, and the next round does not fix it | stop that lineage |
| `no_progress_2` | two consecutive rounds without progress (section 2) | stop the problem |
| `budget` | the per-problem token or wall cap is reached (S6 section 3) | stop the problem |
| `missing_information` | an ambiguous requirement has no trusted check that can resolve it | BLOCK, ask the user |
| `contradictory_requirements` | contradicted hard/functional requirements | BLOCK, report both quotes |
| `tool_failure` | the sandbox or grader errors (status `error`), not the candidate | stop; counted separately from model failures |
| `api_unavailable` | critic quota or payment error, or every key cooling past the wait cap | arm D1 stops calling GLM; the run continues without it and the report says so |

Exhausting the budget is never recorded as `decided` (I5). The best candidate is still reported, labelled from its rows (usually UNVERIFIED or SUPPORTED).

## 4. Failed candidate versus impossible task

- **Candidate failed** (`same_failure_twice`, `unrepaired_regression`, `no_progress_2` on one lineage): sample a new candidate or investigate more, within budget.
- **Task cannot be completed from what is known** (`missing_information`, `contradictory_requirements`): no more generation. Ask the user.

## 5. Operational parameters (tuned on development data, then frozen before the holdout; S6 section 4)

- initial samples k1;
- when to sample more;
- maximum repair lineages per problem;
- the critic trigger;
- the per-problem budget split between sampling and repair.

## 6. Tests (pilot stage 0)

| id | case | expected |
|---|---|---|
| S5-T1 | identical failure on parent and child | `same_failure_twice` |
| S5-T2 | progress on round 1, none on rounds 2 and 3 | stop after round 3, `no_progress_2` |
| S5-T3 | LAYA prefers a candidate with fewer trusted passes | the ranking ignores LAYA across trusted levels (I2) |
| S5-T4 | budget reached during repair | `budget`, never `decided` (I5) |
| S5-T5 | sandbox error on every run | `tool_failure`, not counted as a model failure |
