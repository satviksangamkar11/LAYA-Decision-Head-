# EL-ARCH-1 checklist (single source of truth; update after every step)
Status: DONE / RUNNING / TODO / BLOCKED. Evidence = path + hash. Mocks never count as REAL PASS.

## Rules
- [x] Frozen LAYA-INT-1 checkpoints + receipts read-only
- [x] 3 examples only (ex1_slug, ex2_clamp, ex3_wordcount); no benchmark, training, download, 320 answers
- [ ] Final runs use the real controller, real Qwen (Kaggle T4), real LAYA, WSL, real local GLM (no fakes)
- [ ] Source committed + secret-scanned BEFORE final run; receipts carry commit + content hashes
- [ ] Fault-injection variants of the 3 tasks allowed for O2/O3, safety, budget branches

## Phase A: foundation (DONE, commit 993367c4)
- [x] A1 GLM calls/retries/tokens from real ledgers
- [x] A2 transport error class recorded, no credentials
- [x] A3 wall/token/action/round limits enforced (new versioned config)
- [x] A4 model-derived evidence cannot raise approval/support
- [x] A5 I1-I8 real; missing component fails loudly
- [x] Existing suites stay green; coordinator verified (phaseA 16, controller 25, critic_batch 28, stepE 12, models 30, orchestrator 25 local; laya_integration 24/25 locally: needs T4_scores.json, to be run on Kaggle)

## Phase B: controller (DONE, commit 37422e98; owner: controller/*, models/qwen.py)
- [x] B1 (LOGIC-ONLY for Qwen/LAYA/GLM; sandbox+controller real in tests) real O1/O2/O3 routing
- [x] B2 (LOGIC-ONLY for Qwen/LAYA/GLM; sandbox+controller real in tests) deterministic investigate stage (inspect source, run tests, locate failure)
- [x] B3 (LOGIC-ONLY for Qwen/LAYA/GLM; sandbox+controller real in tests) Qwen generation interface + frozen prompt strategies in controller
- [x] B4 (LOGIC-ONLY for Qwen/LAYA/GLM; sandbox+controller real in tests) LAYA on correctly captured Qwen states
- [x] B5 (LOGIC-ONLY for Qwen/LAYA/GLM; sandbox+controller real in tests) fixed action-risk table + authorization + WSL sandbox
- [x] B6 (LOGIC-ONLY for Qwen/LAYA/GLM; sandbox+controller real in tests) validated model-written checks (L2), behavioural discrimination, repair from observed failures
- [x] B7 (LOGIC-ONLY for Qwen/LAYA/GLM; sandbox+controller real in tests) final verification of actual end state
- [x] B8 (LOGIC-ONLY for Qwen/LAYA/GLM; sandbox+controller real in tests) every step journals inputs, decision, evidence, action, outcome

## Phase C: supporting (DONE, logic-only, commit 0325f59c; owner: orchestrator/*, new models/probe+gate files, next_action.py)
- [x] C1 (LOGIC-ONLY; real probe refit + random-head scores need Kaggle) linear probe + proper random-head control, identical pools, no retraining
- [x] C2 (LOGIC-ONLY; real LAYA on program candidates pending) arm E wired into whole-program stage-1 controller (exploratory transfer only)
- [x] C3 LAYA monotonicity gate (ordering disabled unless supported; never overrides trusted evidence)
- [ ] C4 behaviour clustering beyond public examples (MOVED to Phase B: B6)
- [x] C5 report + budget gates (3 examples -> INSUFFICIENT_DATA, never PASS)
- [x] C6 evaluation oracles inaccessible to runtime; dev cases not called sealed

## Phase D: verify 3 examples (D-1 DONE commit 39b4da05; D-2 RUNNING: rebuild bundles, Kaggle test kernel, Kaggle T4 rounds, then GLM step by coordinator)
- [x] D1 commit reviewed source (scan 0) -> 39b4da05 (D-1 suites verified: 266 local)
- [ ] D2 ex1/ex2/ex3 through the same 8-stage controller
- [ ] D3 real Qwen proposals + real repair where needed (Kaggle T4)
- [ ] D4 real local GLM response enters controller; safe experiment executed
- [ ] D5 actual trusted-evidence conflict
- [ ] D6 fault injection: O2/O3, blocked action, missing info, broken tool, abstain, invalid model evidence, budget exhaustion
- [ ] D7 immutable journal + replay vs committed version

## Final
- [ ] Conformance report: 8 stages, I1-I8, L0-L5, O1-O3, R1/R2, Qwen/adapters/LAYA/probe/random-head, controller/WSL/GLM/repair, budget/safety/reporting, every remaining audit finding
- [ ] Each component: IMPLEMENTED / REAL PASS / LOGIC-ONLY / N/A / BLOCKED + source location + receipt
- [ ] Close only if all applicable blocks pass; else exact blocker + smallest fix
- [ ] Update handoff + memory; push only when asked
