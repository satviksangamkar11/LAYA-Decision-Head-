# P0 + R3 completion report (Evidence-Ladder v1, 2026-10-16)

Authorised by the user on 2026-10-16 ("Proceed with P0 + R3"). No model inference, no Kaggle runs and no training in P0.
Architecture source of truth: `ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md` (R1, R2, 12c).

## Documents

| file | contract | R3 items inside it |
|---|---|---|
| `schemas_v1.json` | machine-checkable schemas for every record | version binding, independence, critic I/O, stop causes, budgets |
| `S1_requirements.md` | requirement format | explicit vs contract vs derived hypotheses; ambiguity and contradiction |
| `S2_evidence.md` | evidence records, row and answer labels | oracle independence, stale-evidence invalidation, conflicts |
| `S3_registry.md` | checks and scorers | adapter compatibility contract; LAYA restricted to validated decision types |
| `S4_execution_policy.md` | controller, approval by risk, isolation, transactions, critic calls | candidate jail inside the grading kernel (user correction 1); red-team tests; NVIDIA limits and spend stop |
| `S5_repair_policy.md` | repair and stop rules | progress-based stopping; failed candidate vs impossible task |
| `S6_evaluation_protocol.md` | arms, budget, statistics, gates, dataset rules, pilot | A/B/C/D0/D1/E; stage-1 LAYA exploratory only (user correction 2); frozen vs tuned thresholds; adequacy >= 0.80 |

## Validation status

| item | status | evidence |
|---|---|---|
| Six specs + schemas written | DONE | this folder |
| Schemas parse and validate example records | NOT RUN | running code is compute; it runs in pilot stage 0 on Kaggle under the standing rule |
| Contract tests S1-T1..S5-T5, red-team S4-T1..T7 | SPECIFIED, NOT IMPLEMENTED | implementation and the run are pilot stage 0 |
| Adapter compatibility facts | DONE | values copied from `decision_head/fable_features.py` PROTOCOL; capture code hashed 2026-10-16 |
| Dataset D1 | **UNRESOLVED** | S6 section 5: LCB Pro split exists (144 problems), but test coverage of that split is likely missing; nothing else meets all rules |

## Unresolved decisions

| id | decision | needed for |
|---|---|---|
| D1 | stage-1 dataset. Next step: the user accepts the LCB-Pro gate so the 144 ids can be checked against the test-case zip list (metadata only; downloading ~15.3 GB of tests needs a separate yes) | P0 exit |
| D7 | where GLM calls run: (a) staged loop with calls from the PC (recommended), (b) keys as Kaggle Secrets (keys leave the PC) | P0 exit |
| D2 | stage-2 environment | before stage 2 |
| D4 | offline certified check library | stage 3 |
| credits | user confirms remaining NVIDIA trial credits | before the pilot |

## P0 exit gate (all must hold before the pilot is authorised)

1. Six specs + schemas exist (done).
2. D1 resolved against all six rules in S6 section 5, recorded with its source and licence.
3. D7 decided.
4. `configs/thresholds_evidence_ladder_v1.toml` written from S6 sections 1-8 and its sha256 recorded.
5. The user gives GO for the pilot. Its stage 0 runs FC-1..FC-4 and every S1-S5 and red-team test **before** any arm runs. A failure in stage 0 stops the pilot.

## Stage-1 authorisation (after the pilot)

- Pilot stage 0 passes fully.
- The pilot reports headroom, discordant share, per-arm cost and stop causes.
- The main-run size and T are fixed from the pilot and written into the toml.
- The user gives GO.
