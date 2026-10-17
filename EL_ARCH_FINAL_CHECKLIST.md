# EL-ARCH-1 FINAL ARCHITECTURE CLOSURE: shared checklist (2026-10-17)

Goal (user GO, 2026-10-17): fix and verify every planned coding-pipeline component on ex1_slug / ex2_clamp / ex3_wordcount (+ fault-injection variants). ARCHITECTURE only, not model accuracy.
EVERYTHING LOCAL (user, 2026-10-17): no Kaggle. RTX 3050 + local WSL sandbox + local GLM path only.
Spec: the user's GO text is summarised per row below; spec docs: ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md, PLAN_EL_ARCH_COMPLETION_1.md, FREEZE_LAYA_INT_1.md, EL_ARCH_1_CHECKLIST.md.

Rules for every agent
- Edit ONLY the files you own (below). Need a change in someone else's file -> add a row in "REQUESTS" at the bottom and message the owner; never edit it.
- COORDINATION CORRECTION (user, 2026-10-17): ONLY THE COORDINATOR EDITS THIS FILE. Agents write results to their own file: `EL_ARCH_AGENT1_RESULTS.md`, `EL_ARCH_AGENT2_RESULTS.md`, `EL_ARCH_AGENT3_RESULTS.md` (one section per row id: status DONE / NOT DONE / BLOCKED, test name + file:line, run output summary, receipt path). The coordinator verifies and copies status here. Never mark done without a passing test run you executed.
- Interface candidate-filter <-> controller is FROZEN in `EL_ARCH_INTERFACE_FREEZE_v1.md` (structured accepted candidates, per-option rejection reasons, abstention, stable option ids). Read it; do not deviate; propose changes to the coordinator.
- Two final paths are separate (see the freeze file): PATH A local 0.6B live controller (explicit unavailable-LAYA fallback); PATH B frozen 4B LAYA replay only on exact stored contexts+options. Never pair 0.6B candidates with stored 4B scores.
- Every original architecture requirement maps to an owner + acceptance test in `EL_ARCH_TRACEABILITY.md` (coordinator). No unowned block, no mock-only PASS, no statistical PASS from three examples.
- Every fix needs one positive and one negative test. No vacuous PASS, no mocks of the real model in the acceptance path (mocks only in unit tests).
- Tests: `PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe <suite>` from D:\local model. No GPU jobs (the 3050 is reserved for the final run by the coordinator), no NVIDIA/Kaggle calls, no git add/commit/push, no pip install, never read .secrets.
- Do not redesign, retrain, download or change frozen files (capture/*, kaggle/*, adapters, configs already frozen). New behaviour = new versioned file/template; never edit a frozen prompt template.
- Final message to the coordinator: DONE / NOT DONE / BLOCKED list with evidence.

## File ownership
- AGENT 1 (candidate layer): `evidence_ladder/controller/generation.py`, NEW `evidence_ladder/controller/candidate_filter.py`, NEW `evidence_ladder/controller/tests/phaseF1_tests.py`.
- AGENT 2 (controller/stages): `evidence_ladder/controller/agentic.py`, `controller.py`, `invariants.py`, `risk.py`, `oracle*`, other `controller/*.py` except Agent 1's files; NEW `evidence_ladder/controller/tests/phaseF2_tests.py`.
- AGENT 3 (accounting, driver, variants, report): `tools/el_arch1_*.py`, `evidence_ladder/critic_batch/*`, `evidence_ladder/models/{broker,glm,pc_sender,kaggle_keys}.py`, `evidence_ladder/orchestrator/*`, NEW `evidence_ladder/orchestrator/tests/phaseF3_tests.py`, `ARCHITECTURE_FINAL_REPORT.md` generator.
- COORDINATOR: commit, secret scan, final real local run, final report assembly.

## AGENT 1: Phase 1 candidate layer
- [ ] A1.1 Per-option parsing survives a syntax error in the array (e.g. unescaped quote in ex3 step 1): valid sibling objects are kept, the broken one is rejected with a reason. Evidence:
- [ ] A1.2 One independently selectable next action per option; sequential plans (Read->Edit->Bash of one list) are detected and rejected or split by a documented deterministic rule; new versioned prompt template (never edit qwen_actions_v1). Evidence:
- [ ] A1.3 Reject `old_string == new_string`, unchanged diffs, and an `old_string` that is absent / not unique in the file (function takes file texts). Evidence:
- [ ] A1.4 Validate tool name, required args, path scope (reuse risk.py), risk class, option count; dedupe identical candidates; never invent options to reach k=4. Evidence:
- [ ] A1.5 No valid option -> explicit abstain reason string consumed by the existing journal path (coordinate shape with Agent 2). Evidence:
- [ ] A1.6 Public function `filter_options(...)` documented in the file header for Agent 2 to wire. Evidence:

## AGENT 2: Phase 1 + Phase 2 + Phase 3 controller
- [ ] A2.1 Repeated-failure blocker: key = state hash + normalised action + args + observed failure; same unchanged failure is blocked; a retry after the state or relevant evidence changed is allowed (positive + negative test). Evidence:
- [ ] A2.2 INVESTIGATE gives bounded real source snippets, paths, signatures, failing test output, user constraints (not symbol names only); truncation recorded in the journal; evaluation-only/hidden data provably absent from the proposal context. Evidence:
- [ ] A2.3 Wire Agent 1's `filter_options` into the step before ranking; rejected options journalled with reasons. Evidence:
- [ ] A2.4 REPAIR feeds real failure evidence back into the next proposal prompt; progress/regression/repetition tracked in the SAME controller (no mini-controller). Evidence:
- [ ] A2.5 Stop labels name the exhausted limit (`step_cap`, `round_cap`, `token_cap`, `wall_time`, `model_call_cap`) instead of "budget"; limits are ENFORCED (tokens, steps, rounds, wall time, model calls), with tests that hit each. Evidence:
- [ ] A2.6 Eight stages INTAKE..COMMIT&VERIFY all through the one controller and one journal; O1/O2/O3 routing incl. blocking ambiguity and no-oracle. Evidence:
- [ ] A2.7 I1-I8 positive + negative tests; L0-L5 scope/provenance; model tests and GLM opinions cannot upgrade trusted evidence or authorise. Evidence:
- [ ] A2.8 Missing LAYA scores journalled, no silent learned-score fallback; LAYA ordering at L0 only; monotonicity gate keeps verification-effort ordering disabled unless registered criterion passes. Evidence:
- [ ] A2.9 Evidence conflict path on PATH B only: with stored 4B scores on their exact stored contexts+options, LAYA's top-ranked action loses to an executed trusted check; a context/option mismatch is refused. Evidence:
- [ ] A2.10 Unsafe/irreversible actions cannot bypass authorization (fixed risk table), tested. Evidence:

## AGENT 3: Phase 3 connections + Phase 4 accounting + Phase 5 harness
- [ ] A3.1 GLM accounting reconciled: successful calls, attempted sends, retries, actual tokens, errors == ledger; transport failures classified; retries bounded; credentials never logged. Evidence:
- [ ] A3.2 UTF-8 mojibake root cause found and fixed in future GLM receipts (existing results untouched). Evidence:
- [ ] A3.3 Critic requests stay blinded; experiment types validated; only allowed operations run through WSL; GLM can never authorise. Evidence:
- [ ] A3.4 Provenance hardening: fake / incompatible capture source rejected; 0.6B features never enter the 4B adapter; probe + correctly constructed random-head controls on the same 4B pools, separate from LAYA; replay of stored verified 4B->LAYA scores through the updated controller on PATH B only (exact context sha + option sha match, else refuse; never against 0.6B-generated options). Evidence:
- [ ] A3.5 Fault-injection variants of ex1/ex2/ex3 (malformed JSON, sequential plan, duplicates, empty pool, unsafe action, broken test tool, missing scorer, budget exhaustion) + O2 and O3 variants; runnable locally with the real driver in `--mode local`. Evidence:
- [ ] A3.6 Driver records commit SHA, config hashes, model identity, source hashes, receipt + journal hashes; refuses to start the final run from a dirty tree; secret-scan step on added lines. Evidence:
- [ ] A3.7 Report generator writes ARCHITECTURE_FINAL_REPORT.md rows `Component | Implementation | Real test | Negative test | Receipt | Verdict` (PASS / FAIL / NOT_APPLICABLE / INSUFFICIENT_DATA; no vacuous PASS); statistical gates say INSUFFICIENT_DATA with 3 examples; existing dev tests marked unsealed. Evidence:
- [ ] A3.8 Final-path guard: no mock backend can be used in the acceptance run (receipt says which backend ran). Evidence:

## COORDINATOR
- [ ] C1 All agent suites + existing suites green together on the merged tree
- [ ] C2 Commit + secret scan (0) BEFORE the final run
- [ ] C3 Final real local run (0.6B on 3050, 4B LAYA frozen evidence, local GLM step via the user's terminal) on the committed code
- [ ] C4 Replay from committed code matches receipts
- [ ] C5 ARCHITECTURE_FINAL_REPORT.md; verdict CODING_ARCHITECTURE_INTEGRATION_PASS or the list of blockers

## COORDINATOR-VERIFIED STATUS (I re-ran the suites myself; agents' own claims in EL_ARCH_AGENT*_RESULTS.md)
- AGENT 1: A1.1-A1.6 DONE (phaseF1 15/15, phaseB 36/36; also run on the real 0.6B outputs: ex3 step 1 broken JSON now keeps 3 valid options). Heuristic ceiling: plan rule only recognises Read(P)->Edit(P)->Bash. Rule order: plan check precedes no-op check (a no-op Edit inside a plan is reported as `sequential_plan`).
- AGENT 2: A2.1-A2.10 DONE; T01, T09-T11, T14-T16, T20-T22, T27-T33, T38-T39 DONE (phaseF2 43/43; 12 on the real WSL sandbox, scripted proposer, so real controller + sandbox, NOT a model-accuracy claim). PARTIAL: T26. NOT DONE: T12, T13 real-model halves.
  A2.9 conflict test is on PATH B only: ex1 LAYA top (stored 4B score 1.62) fails its executed dry-run and loses; ex3's filtered pool no longer matches the stored pool -> scorer_unavailable, no pairing.
- AGENT 1 (2nd task): T12/T13/T26 harness `tools/el_arch1_l2_local.py` built, phaseF1b 10/10; real CLI path untested (needs GPU, coordinator runs it in the final run).
- AGENT 3: A3.1-A3.3, A3.5-A3.8 DONE; A3.4 PARTLY (exact-pool admission + replay4b mode done; PATH B on el-arch1-accept-20261017T101325Z is REFUSED by design: current context differs from the stored one, decision (b): that refusal is the negative test, positive PATH B source = laya-integration-20261016T184816Z). Mojibake: no byte-level defect; the earlier "mojibake in saved GLM responses" was the coordinator's own reader (Windows code page); guards added. phaseF3 23/23.
- FOLLOW-UPS SENT (2026-10-18): Agent 3: PATH B positive source, T4_scores.json fallback lookup, promote budget_steps, switch --mode local to prefilter=False. Agent 1: duplicate must be rule `duplicate`, ActionProposerV2 prefilter flag.
- FINAL-RUN READINESS (measured 2026-10-18): WSL answers; 0.6B snapshot present; GPU 1.5/6 GiB used by another process; FREE PHYSICAL 0.7 GB, FREE VIRTUAL 2.6 GB (the 4B streaming load needs about 7 GB commit): the user must close heavy apps before the final run.
- OPEN, ASSIGNED: T12 / T13 / T26 real-model halves (0.6B writes L2 tests; real clustering of 0.6B candidates; multi-cluster discrimination) -> Agent 1 builds a new script + CPU unit tests; the coordinator runs it on the GPU in the final run.

## REQUESTS (cross-agent; append only)
