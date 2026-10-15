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
| Agents launched in separate worktrees | PENDING | |
| Persistent master checklist | DONE | this file |

Known caveats carried into the build: protected development cases are public in Git history (not a sealed benchmark); D1, D7, NVIDIA credits and the pilot GO are not given; the oracle-adequacy gate (>= 0.80 measured) is unmet for 2 Python tasks (P1-19, P1-26), 2 Java tasks (J1-17, J1-27) on the raw count, and unmeasured for P1-05, P1-10, P1-14, J1-08, J1-10, J1-20; the dataset waiver does not waive it.
Open policy question: may pure-logic unit tests (no model, seconds) run in WSL? Until the user answers, tests are executed only for the sandbox exception, plus `py_compile`.

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
| 1 controller | PENDING | |
| 2 execution | PENDING | |
| 3 orchestrator | PENDING | |
| 4 models | PENDING | |

## Checkpoint 2: integrated deterministic replay: PENDING (all items)
## Checkpoint 3: authorization: BLOCKED on the user (D1 waiver, D7, credits, GO)
## Checkpoint 4: pilot execution: BLOCKED (needs Checkpoint 3)
