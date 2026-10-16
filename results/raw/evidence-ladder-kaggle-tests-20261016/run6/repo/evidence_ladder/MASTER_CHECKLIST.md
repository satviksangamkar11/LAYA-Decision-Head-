# Evidence-Ladder Build Step 2: master checklist (coordinator-owned; statuses DONE / FAILED / BLOCKED / PENDING)

Base: branch `evidence-ladder-p0-2026-10-16`, commit d0466edc33b98eb521d5f7df8d352babcc752e3e (equals origin). Build branch: `evidence-ladder-runtime-step2` (local only; not pushed, not merged).
Rules in force: no push or merge without the user's approval; no model inference, no GLM requests, no NVIDIA keys read, no Kaggle launch; spend Rs 0; compute on Kaggle except the WSL sandbox exception.

## Checkpoint 0: baseline
| item | status | evidence |
|---|---|---|
| Branch, HEAD, tree state verified | DONE | HEAD d0466ed = origin; working tree dirty with unrelated edits (.gitignore, DECISION_HEAD.md, PLAN.md, kaggle/*, pyproject.toml, uv.lock) and 715 untracked paths: preserved, untouched |
| CLAUDE.md rule-removal edit | **DEFECT FOUND, FIXED** | the edit was never committed (the blocked command never ran `git add`); commit ba0aca0's message claimed it. Now committed on the build branch |
| Unrelated changes preserved | DONE | only owned paths are staged |
| 23/23 contract tests recorded | DONE | specs/evidence_ladder_v1/contract_test_results.json (sha256 7f562e29...); negative controls: 1 and 4 tests fail when the library is broken |
| Python/Java sandbox receipts recorded | DONE | python red-team 7/7, reference 40/40, wrong 0/37 pass (fixed_v2/reports); java reference 40/40, hostile candidate blocked (fixed_v1/reports) |
| Combined index: 40 Python, 40 Java, 46 Fable VAL, 335 exposed | DONE | combined-index-v1-20261015T221945Z, validator PASS (structure), 50 sources hashed |
| Existing modules inspected | DONE | contract_lib_v1.py, grade_sandboxed_v2.py, runner_v2.py, grade_sandboxed_java_v1.py, tools/nvidia_keys.py (keys NOT read) |
| Shared interface contract | DONE | evidence_ladder/common/interfaces.py (coordinator-owned) |
| File ownership allocated | DONE | below |
| Agents launched in separate worktrees | DONE | 4 agents, model alias "sonnet", effort medium, isolation worktree; each must self-report its model ID (actual configuration verified from the reports, not assumed) |
| Persistent master checklist | DONE | this file |

Known caveats carried into the build: protected development cases are public in Git history (not a sealed benchmark); D1, D7, NVIDIA credits and the pilot GO are not given; the oracle-adequacy gate (>= 0.80 measured) is unmet for 2 Python tasks (P1-19, P1-26), 2 Java tasks (J1-17, J1-27) on the raw count, and unmeasured for P1-05, P1-10, P1-14, J1-08, J1-10, J1-20; the dataset waiver does not waive it.
Compute policy ANSWERED by the user (2026-10-16): pure-logic unit tests run on **Kaggle only**; only isolated SANDBOX test runs stay in WSL under the 2026-10-13 exception. Consequence: controller, orchestrator, models and contract suites run in a Kaggle CPU kernel (kaggle/evidence_ladder_tests_v1.py, packed by tools/pack_evidence_ladder_tests_v1.py); uploading that code bundle to Kaggle needs the user's explicit yes (CLAUDE.md "Ask first"). Execution/sandbox tests cannot run on Kaggle (bubblewrap) and make no Kaggle security claim.

## File ownership (non-overlapping)
| owner | paths |
|---|---|
| Coordinator | evidence_ladder/common/**, evidence_ladder/MASTER_CHECKLIST.md, evidence_ladder/testing/** (integrated replay), integration commits |
| Agent 1 controller | evidence_ladder/controller/** |
| Agent 2 execution and grading | evidence_ladder/execution/** |
| Agent 3 orchestrator | evidence_ladder/orchestrator/** |
| Agent 4 model interfaces and critic broker | evidence_ladder/models/** |

## Agents (status filled in at Checkpoint 1)
| agent | status | integration-ready |
|---|---|---|
| 1 controller | reported 2026-10-16 (model self-report: claude-sonnet-5-5); 22 tests written, NOT executed; resumed for integration fixes | no |
| 2 execution | reported; 20/20 WSL tests; coordinator re-ran in the main checkout: 19/20 then 20/20 after the receipt-hash re-pin (line-ending defect in the agent's pins, not in the sandbox); resumed to emit per-example public outcomes | WSL path yes; Kaggle untested |
| 3 orchestrator | reported; 21 tests written, NOT executed | no (needs Kaggle run) |
| 4 models | reported; ~29 tests written, NOT executed; resumed to add `completed(packet_id)` | no (needs Kaggle run) |

Integration gaps found at review (all assigned): (a) controller expects per-example public outcomes, execution returned an aggregate -> contract fixed, Agent 2 resumed; (b) controller had no way to read an ingested critic response -> `CriticBroker.completed()` added, Agents 4 and 1 resumed; (c) `.gitignore` ignores `models/`, so evidence_ladder/models needs `git add -f`; (d) LAYA capture-code files are untracked in git: added to the Kaggle bundle; (e) ProblemResult lacked final_artifact/stop_details: added.

## Checkpoint 1 result (verified by the coordinator, not by agent claims)
| suite | where | result |
|---|---|---|
| contract (S1-S5, rollback, schemas) | Kaggle CPU, run 3 | 23/23 |
| controller | Kaggle CPU, run 3 | 25/25 (run 1 had 23/25: two test-fixture bugs, fixed) |
| orchestrator | Kaggle CPU, run 3 | 24/24 (run 1 and 2 had 23/24: a REAL routing bug: one critic response reached only one of two arms' brokers; fixed in pilot.py) |
| models and critic broker | Kaggle CPU, run 3 | 28/28 |
| execution and grading | WSL, coordinator re-run | 21/21 (one defect found and fixed: receipt hashes pinned over CRLF-sensitive bytes) |
Bundle integrity: 71 file hashes verified in every Kaggle run; Python 3.13.15 on Kaggle. Receipts: results/raw/evidence-ladder-kaggle-tests-20261016/run{1,2,3}/.
Models actually used (self-reported by each agent): all four build agents and the replay agent claude-sonnet-5-5 at medium effort; the small-edit agents claude-haiku-5-5.

## Checkpoint 2: integrated deterministic replay: DONE
Kaggle run 5: replay 26/26 (run 4 had 22/26: four real defects, all fixed), plus contract 23, controller 25, orchestrator 24, models 28 = 126/126.
Covered: Python and Java correct and known-wrong solutions (real recorded sandbox outcomes), failed check then repair, stale and conflicting evidence,
GLM malformed / 429 / 500 / timeout / 402, budget exhaustion, irreversible refusal, unsupported LAYA decision, reproducibility (identical hashes on
two runs), protected grading never influencing selection (flipped grader gives identical selections), authorization refusal, per-track separation and
mutation flags. Full record: HANDOFF_2026_10_16_EVIDENCE_LADDER_RUNTIME.md.
## Checkpoint 3: authorization: BLOCKED on the user (D1 waiver, D7, credits, GO)
## Checkpoint 4: pilot execution: BLOCKED (needs Checkpoint 3)
