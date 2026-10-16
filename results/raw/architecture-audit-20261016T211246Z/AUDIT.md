# [LAYA-INT-1 · Aim: integration, not performance] Architecture conformance audit (independent, read-only)

Auditor: read-only. Date: 2026-10-16T21:12Z (2026-10-17 IST). Repo: D:\local model, branch evidence-ladder-runtime-step2, HEAD ff40e4ff.
Scope: built code vs ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md (I1-I8, ladder, router, 8-stage loop, LAYA placement, evaluation, R1/R2/D5), specs/evidence_ladder_v1, PLAN_LAYA_INT_1.md, CLAUDE.md, MASTER_CHECKLIST, handoffs, decision records, receipts.
Nothing was run (no tests, models, Kaggle, NVIDIA). `.secrets/` and `~/.kaggle` were not read. Only file reads and sha256sum. Anything not directly seen is marked UNVERIFIED.

## 0. Bottom line

- The LAYA half is real where it was measured: Kaggle T4 receipts (integration_run2) show a real frozen-Qwen capture bit-identical to the training features on 65/65 records, the 3 adapters + pretrained head reproducing recorded VAL numbers, and runtime re-capture refusing fabricated tensors. All receipt hashes verified OK.
- LAYA-INT-1 is NOT complete. T6 (end-to-end) has no receipt, T5 (controller L0 use) and the pure-logic part of T7 were never executed, and the T6 driver that exists is a parallel mini-controller, not the Evidence-Ladder controller. "Full pipeline incl. GLM critic" (decisions v4/v5) is not wired anywhere.
- Counts: BLOCKER 3, MAJOR 11, MINOR 9, NOTE 5 (28 findings, F01-F28; the table in section 1 has 41 element rows).

## 1. Status table (element | status | evidence | shortcut / risk | severity)

Status words: ENFORCED = in code and exercised on the real component; MOCK-ONLY = code exists, tested only with fakes/fixtures/hand-built inputs; PARTIAL; NOT IMPL; DEVIATES.

| # | Architecture element | Status | Evidence | Shortcut / risk | Sev |
|---|---|---|---|---|---|
| 1 | Loop stage 1 INTAKE (deterministic skeleton R1, R2, one explicit requirement per public example, span-checked quotes) | ENFORCED (stage-1 coding) | controller/intake.py:174-240; contract_lib_v1.py:22 `check_requirement` | "requirement list from the weak model" is a caller-supplied `proposed_requirements`; no Qwen requirements call; phantom-rate not measured; "ambiguous" status that triggers ask-the-user is never set by anything | MINOR (F17) |
| 2 | Stage 2 INVESTIGATE (read, run tests, Qwen proposes, LAYA ranks) | NOT IMPL in controller; ad hoc in T6 driver | no investigate stage in controller.py; tools/laya_int1_t6.py (Read/Bash options) | exists only inside the unfinished T6 driver | MAJOR (F03) |
| 3 | Stage 3 GENERATE (k diverse, >= 2 strategies) | PARTIAL / MOCK-ONLY | controller.py:99-121; qwen_v2b.py; qwen.py:195 "HFQwenBackend NEVER EXECUTED IN THIS BUILD"; controller.py:107 sends `intake.request_text` with template ids "sample_v1"/"repair_v1" | controller prompt has no solution-format block (FC-3 defect still OPEN per checklist); v2/v2b templates never selected by the controller; only fixtures/scripted fakes drove it | MAJOR (F13) |
| 4 | Stage 4 EXECUTE (sandbox, I4 map, clusters) | ENFORCED on real recorded sandbox outcomes; clusters weak | execution/backend.py; controller/evidence.py:20-31; replay fixtures from real WSL runs | behaviour cluster = hash of outputs on the PUBLIC examples only (evidence.py:113): all candidates passing every example share one cluster, "L1 agreement" is vacuous | MAJOR (F14) |
| 5 | Stage 5 DISCRIMINATE (blinded critic, runnable experiment) | MOCK-ONLY | controller.py:249-363; critic_batch 28 tests; checklist "LIVE BATCH NOT DONE" | default trigger needs >= 2 clusters (controller.py:253), rare given F14; never run against real GLM or real Qwen candidates | MAJOR (F13, F14) |
| 6 | Stage 6 RANK lexicographic (I2) | ENFORCED | contract_lib_v1.py:133; controller.py:230-246; invariants.py:254 | scorer/deepconf/cluster only reorder inside equal evidence levels, by design | NOTE |
| 7 | Stage 7 REPAIR (R1.3 progress, stop causes) | ENFORCED logic; generation scripted | controller.py:372-411; contract `stop_cause` | `C.stop_cause(self.entries, 0, 1)` hard-codes budget_used=0, cap=1 so its "budget" cause never fires (budget handled elsewhere); two repair templates (controller.py:38 vs qwen_repair_v1) | MINOR (F23) |
| 8 | Stage 8 COMMIT and VERIFY (re-run trusted checks on real end state) | PARTIAL | controller.py:455-474 | "real end state" is the same sandbox backend re-run; no separate real-commit target in stage 1 | MINOR |
| 9 | Router O1/O2/O3 tiers | NOT IMPL as tiers | controller/router.py:54-84 per-requirement ceilings only; grep "O1/O3/oracle_tier" in evidence_ladder = no hits | architecture section 5 tier table and tier-based loop choice absent | MINOR (F17) |
| 10 | Router re-run after evidence-adding steps (R1.2) | ENFORCED | controller.py:195-199 called at intake, samples, critic, repair, finalize; journaled | none | NOTE |
| 11 | Per-requirement rows + weakest-row label (R1.1) | ENFORCED | contract `row_label`/`answer_label`; decision.py:107-127 | in stage 1 R1 is covered only by a rejecting_only check and R2 by none, so VERIFIED is unreachable; max label SUPPORTED, and "decided" accepts SUPPORTED (controller.py:37,191). Honest, but stage 1 can never say VERIFIED | NOTE |
| 12 | I1 Authority | ENFORCED (post-hoc recompute with same function) | invariants.py:234-251 | `derived_by == "controller"` sub-check is a literal always set by code (contract_lib_v1.py:116): tautology | MINOR (F16) |
| 13 | I2 Monotone safety | ENFORCED | invariants.py:254; next_action.py:27-28 | negative control test exists only in the never-run suite (F02) | NOTE |
| 14 | I3 No self-authorization | ENFORCED for authority; leak for soft support | invariants.py:267; controller.py:324,340-341 | critic experiment "pass" is recorded against a requirement the CRITIC names and `row_label` counts any passing applicable record as SUPPORTED (contract_lib_v1.py:84-86); L0 critic judgement must not support | MINOR (F15) |
| 15 | I4 Unknown is not pass | ENFORCED | evidence.py:20-31; check_i4 | none seen | NOTE |
| 16 | I5 Bounded | PARTIAL | caps in controller; MatchedTokenBudget; check_i5 | wall-time cap not enforced (only recorded); T and k1 PENDING so no run can be configured; T6 "<= 2/3 steps" is operator convention | MINOR |
| 17 | I6 Observe before commit | ENFORCED stage 1; CONVENTION in T6 | invariants.py:308; laya_int1_t6.py:133-147 | action classifier is a regex DENY list (laya_int1_t6.py:33), not the fixed table; no irreversible class | MINOR (F16) |
| 18 | I7 Frozen scorers / journal | ENFORCED | journal.py hash chain; check_i7; registry digest | scorer fingerprint = `weights_hash` from MANIFEST JSON (laya.py:97), not the loaded weight files; weights verified only inside the Kaggle kernel (T3) | MINOR |
| 19 | I8 Sealed evaluation | DEVIATES (documented) | execution/protected.py:1-8; invariants.py:212 (name scan of controller/ only), :333 (ctx field names) | hidden cases are public in Git and at a fixed path; isolation = import hygiene + gate flag | MAJOR (F12) |
| 20 | R2.1 controller per action | PARTIAL (a ranking function, not a pipeline) | controller/next_action.py:14-33 | `"approved": False` literal; no intake->check->commit, no I1/I3-I8, no risk class, no `authorize()`; T6 driver has its own policy | BLOCKER (F03) |
| 21 | R2.2 provenance per record | ENFORCED | evidence.py:46-58 | `receipt_hash` hashes the controller-trimmed outcome, not an execution-side signature | NOTE |
| 22 | R2.3 risk-based approval | PARTIAL | contract_lib_v1.py:114 | only `irreversible` needs `user_confirmed`; "stronger coverage" absent; unused by per-action path | MINOR (F16) |
| 23 | R1.3 progress-based repair | ENFORCED (logic) | controller.py:386-411 | "new failure signature" counts as progress, can reward churn | NOTE |
| 24 | D5 critic blinding (I8 for packets) | ENFORCED (regex + allow-list), MOCK-ONLY transport | glm.py:53-98; critic_batch/validator.py:20-68 | free-text `candidate_code` is only regex-scanned; never exercised with a real Qwen candidate | MINOR |
| 25 | Critic packet contents | ENFORCED | controller.py:256-264 | quotes, code, outputs, `public_expected` (public) only; no LAYA, labels, hidden data | NOTE |
| 26 | LAYA as L0 only (ranking function) | ENFORCED | next_action.py:24-28 | see rows 27-34 | - |
| 27 | LAYA real forward pass T1-T4 | ENFORCED, REAL (Kaggle T4) | integration_run2/receipts T1-T4 (hashes OK) | stored Fable pools, not Qwen-proposed options; tolerances loose; ensemble unmeasured (F18-F20) | NOTE |
| 28 | Probe + random-head controls | NOT IMPL in orchestrator/controller; per-kernel numbers only | arms.py:6; S6 sec.1 line 17; CLAUDE.md "probe mandatory baseline" | never an arm or contrast (probe 0.9434, random head 0.1698 mismatch control, LAYA 0.7925 on VAL) | MAJOR (F07) |
| 29 | NONE option / abstain rule | NOT IMPL; DEVIATES | only comments "no NONE class" (laya_forward.py:10, laya_runtime.py:110) | CLAUDE.md requires NONE + abstain below margin; built code always picks max score or fixed id; no waiver found | MAJOR (F05) |
| 30 | Silent text/TF-IDF/id-order fallback | PARTIAL | laya_runtime.py:475-483 refuses bare text; next_action.py:17-23 `l0_scores=None` gives id order with no journal entry; test codifies it; controller.py:216-220 swallows scorer exceptions | in the next-action library a missing scorer yields a selection indistinguishable from a LAYA tie; arm E can silently equal D1 | MAJOR (F06) |
| 31 | ADAP monotonicity / LAYA keep conditions | NOT IMPL | grep "monoton" = none | future work; no guard in arm E | MINOR (F17) |
| 32 | Replay T7 (re-capture) | ENFORCED, REAL | laya_runtime.py:433-464; T7 receipt | default `sample_n=None` is a random 25% (min 3) with `secrets.randbits` seed (lines 448-450); integration used sample_n=len(batch) | MINOR (F19) |
| 33 | Live T7 (proof by running Qwen) | PARTIAL | laya_runtime.py:344-347 provenance from module constants + caller-supplied device string | the guard cannot tell a fake `capture_fn` from Qwen (unit tests pass `fake_capture` as "Tesla T4"); real check is the Kaggle script's shard-sha test | MAJOR (F10) |
| 34 | Kaggle -> PC trust (T6 import) | CONVENTION | laya_int1_t6.py:173-174; laya_forward.py:297-309 | `origin` is a string in JSON; `features_sha256` unverifiable on PC | MAJOR (F09) |
| 35 | Capture-code pin correction v1->v2 | ENFORCED, DOCUMENTED | CORRECTION_NOTE.md; laya.py:23-52; T2 65/65 bit-identical | correct and honest; files untracked (F11) | NOTE |
| 36 | Matched-token budgets | ENFORCED (logic) | budget.py:141-169 | T/k1 PENDING; `glm_in`/prefill excluded from the cap; "extra Qwen samples for B" rule (budget.py:191) not applied by the runner | MINOR |
| 37 | Frozen thresholds, pending T/k1 | ENFORCED | thresholds toml sha256 matches `.sha256`; T proposal `frozen = false` | pilot cannot start by design (resolve_T raises); recorded accurately | NOTE |
| 38 | Grading gate | ENFORCED gate, UNSEALED data | pilot.py SelectionGate/GatedGrader | see row 19 | MAJOR (F12) |
| 39 | Token/step caps for the end-to-end loop | CONVENTION | PLAN "<= 3 steps" | no code cap | MINOR |
| 40 | G-0 headroom; equal-compute keep/drop rule; false-approval <= B gate | NOT IMPL in report | no headroom/oracle@k in code; report.py has contrasts, rates, no gate verdict | architecture sec. 9 gates are descriptive only | MINOR (F17) |
| 41 | Stage-2 Agentless localise->repair->validate | NOT IMPL | no workflow module | only the 3-task toy driver | MAJOR (F03) |

## 2. Ranked shortcut / gap list

### BLOCKER
- **F01 T6 end-to-end loop never completed.** `results/raw/laya-integration-20261016T184816Z/t6/`: `ex1_slug|ex2_clamp|ex3_wordcount/steps.json` are `[]`; no `step*/receipt.json`; no `END_TO_END_SUMMARY.json`; the only kernel output is `_attempt1_result/t6_step_results.json` (Qwen emitted prose, 0 proposals). `t6_wsl_noop_smoke/run1` is a no-op sandbox smoke. PLAN_LAYA_INT_1 correction 3 is unmet. The checklist line "T6 IN PROGRESS" is accurate.
- **F02 T5 and the pure-logic T7 suite never executed.** `evidence_ladder/models/tests/laya_integration_tests.py` is absent from every Kaggle result (latest `evidence-ladder-kaggle-tests-20261016/run11`: contract 23, controller 25, orchestrator 25, models 30, critic_batch 28, replay 26); `tools/pack_laya_tests_v1.py` exists, no bundle or result. The suite has two vacuous passes: `test_laya_scorer_interface_wiring` returns silently `if s.disabled`, and `test_real_scores_never_override_trusted_evidence` returns silently when no T4_scores.json is supplied. Its fixtures are `fake_capture` (hash-seeded random tensors) and `FakeScorer` (feature mean).
- **F03 The T6 driver is not the Evidence-Ladder controller.** `tools/laya_int1_t6.py` has its own `policy()` (regex DENY, line 33), `trusted_checks()`, execution and harness; it imports only `EvidenceJournal`, `check_i2`, `decide_next_action`. It never runs `EvidenceLadderController`, router, requirement rows, I1/I3-I8, the critic path or `authorize()`. Decisions v4/v5 ("full pipeline ... including the GLM critic") have no code (`grep -i critic tools/laya_int1_t6.py` is empty). "Scores reach the controller and affect only permitted decisions" is shown for a 20-line ranking function, not for the architecture's controller. R2.1 is minimal-only; stage 2 (investigate, Agentless) is absent.

### MAJOR
- **F04 "Trusted beats LAYA" demonstrations use hand-built evidence.** Smoke `S6_controller.json` cases b/c and T6's `trusted_evidence_counterfactual` (laya_int1_t6.py:184-186: `ctr[top_]={"L5":"fail"}; ctr[low_]={"L5":"pass","L3":"pass"}`) inject scripted outcomes onto real LAYA scores. They prove `decide_next_action` obeys inputs, not a real executed conflict. (The driver also computes a real `trusted_evidence_overrode_laya_top`, but no real run exists, F01.)
- **F05 No NONE option, abstain output or margin threshold** in any decision path. Deviates from CLAUDE.md decision-layer rules and "Never pick a random candidate when the decision layer is uncertain" (a fixed-id tie-break is not random but is still a pick). No recorded waiver; the plan only says the trained recipe has no NONE class.
- **F06 Silent LAYA -> id-order fallback in the library.** `decide_next_action(..., l0_scores=None)` writes no "scorer" journal entry and still selects; a test asserts exactly that. In the stage-1 controller `apply_scorer` turns any scorer exception into `scorer_off` plus a journal note, but result and report do not expose it.
- **F07 Probe and random-head controls are in no arm or contrast** (arms.py:6; report.py CONFIRMATORY/EXPLORATORY). Required by S6 sec.1 and CLAUDE.md.
- **F08 Arm E / LAYA is not wired into the stage-1 controller.** `apply_scorer` passes candidate source strings (controller.py:216); `LiveBackend.score` refuses bare text; `SavedScoreBackend` (candidate-sha keyed, pool-dependent logits of a fixed K-pool, laya_forward.py:293-312) is used only in one unit test. No code builds arm E's scorer from the `evidence_ladder_laya_gen_v1` output. Stage-1 LAYA is also out-of-distribution by design (exploratory).
- **F09 Label-based trust at the Kaggle -> PC boundary** (row 34): `origin` string and unverifiable `features_sha256`; SavedScoreBackend filters on that string. T7's principle (prove by re-running Qwen) does not extend to imported results.
- **F10 Live-mode provenance is self-declared** (row 33). The guarded runtime cannot detect a wrong/fake `capture_fn`; protection depends on the calling kernel verifying shard hashes and device.
- **F11 Evidence-producing code is not frozen.** Untracked: `kaggle/evidence_ladder_laya_integration_v1.py`, `kaggle/evidence_ladder_laya_int1_v1/`, `kaggle/evidence_ladder_laya_smoke_v1/`, `kaggle/evidence_ladder_laya_t6_step_v1.py`, `tools/laya_int1_t6.py`, `tools/laya_int1_t6_sandbox.py`, `tools/pack_el_laya_int1_v1.py`, `tools/pack_laya_tests_v1.py`, `configs/laya_capture_code_pins_v2.toml`. Modified vs HEAD: `evidence_ladder/models/laya.py`, `laya_runtime.py` (the run2 bundle manifest hashes equal the current working-tree files, so the bundle is reproducible, but no commit pins it). Run logs (`*.log` in smoke/ and integration_run2/) are 0 bytes. Plan step 8 (freeze) not done.
- **F12 Hidden tests are not sealed.** execution/protected.py:3-5 and MASTER_CHECKLIST: protected cases are public in Git history. I8 holds by import hygiene + gate, not secrecy. D1 dev-only waiver recorded; oracle-adequacy unmet for several tasks. Any number from this data is development evidence only.
- **F13 The real Qwen -> controller -> repair -> critic loop has never run.** HFQwenBackend "NEVER EXECUTED"; controller prompt lacks solution format and its template ids do not name qwen_sample_v2/v2b (FC-3 defect OPEN); T and k1 PENDING so `PilotRunner._preflight` raises; GLM batch critic never called live; Kaggle layered jail (chroot, no namespaces) is weaker than bwrap and not used for candidates. All 157 Kaggle tests use scripted generation, recorded outcomes for 2 tasks (P1-01, J1-02) and 13 hand-written candidates (testing/cases.py).
- **F14 Behaviour clustering is computed on public examples only** (evidence.py:113), so the critic trigger (`len(clusters) < 2`, controller.py:253) rarely fires and "L1 agreement" carries no information.

### MINOR
- **F15** critic "pass" can raise a row to SUPPORTED and the critic chooses the target requirement (controller.py:324,340-341; contract `row_label`).
- **F16** per-action literals and tautologies: `"approved": False` (next_action.py:31); `derived_by` literal makes an I1 sub-check unfailable; DENY-regex policy instead of a fixed table; R2.3 "stronger coverage" absent. Mitigated for T6 by the read-only bind of protected files in the WSL sandbox (UNVERIFIED here; from tools/laya_int1_t6_sandbox.py docstring).
- **F17** unimplemented architecture items: router O1/O2/O3 tiers, weak-model requirement list, L2 `qwen_tests_v1` generation (registered, never produced), ADAP monotonicity check, G-0 headroom gate, equal-compute keep/drop rule, false-approval <= B gate, wall-time cap.
- **F18** deployed LAYA score = mean of 3 adapters' logits (laya_runtime.py:401); the T4 "0.7925" is the mean of per-seed top-1 hits (kernel lines 183-192), so the ensemble's own top-1 is in no receipt (recomputable from T4_scores.json `laya_score` + `ref`; not done here).
- **F19** tolerance max_abs 0.5 vs measured 0.0 (T2): the T7 sensitivity floor is unmeasured (smallest caught perturbation: 5% scale, cos 0.9875); library default replay sampling is a random 25%.
- **F20** T1-T4 use stored Fable pools; Qwen-proposed options enter capture only in T6 (not run). VAL pools were inspected before: fine for integration, not for performance.
- **F21** T6 proposals come from Qwen with thinking closed and "[" prefilled after attempt 1 produced prose (kaggle/evidence_ladder_laya_t6_step_v1.py:393-394): format-forced, not thinking-mode behaviour.
- **F22** T6 tasks are three toy one-edit fixes; an Edit that passes the protected test gets L5 pass and outranks everything, so LAYA only orders no-evidence options and "LAYA changed the choice vs id-order" is near-certain: wiring evidence, not usefulness.
- **F23** `stop_cause(..., 0, 1)` hard-coded budget args; duplicate repair templates.

### NOTE
- **F24** Decision chain v4 (local NVIDIA) -> v5 (local RTX 3050) -> v6 ("Use Kaggle gpu") is consistent; v6 keeps "GLM critic calls local" and "pure-logic tests local or Kaggle" as user overrides of the CLAUDE.md standing compute rule. No local GPU or NVIDIA use is evidenced.
- **F25** PLAN_LAYA_INT_1 "Rules: No NVIDIA calls" is superseded by decisions v4 for the 3 examples; the plan text was not updated.
- **F26** Positive verification: integration_run2 SHA256SUMS all OK; smoke SHA256SUMS OK; run11 TEST_RESULTS.json sha256 47e3d381... matches the checklist; thresholds toml `.sha256` OK; pins v2 toml 1c1e9d2e... and capture/*.py + fable_features.py hashes match the pins; T2 receipt 9e9af31b... and T2b 2b174b70... match CORRECTION_NOTE; FC-2 a6cda193... and stage0 cpu-run7 1338589e... match the checklist. Pre-registration order holds: laya_int1_thresholds mtime 00:05 IST and commit c76a1423 at 00:19:23 IST precede run1 start 00:19:46 IST; pins v2 (00:32) precede run2 (00:59).
- **F27** `tools/fable_capture.py` (v1 pin) was never the training path; the v1->v2 correction is documented with measured agreement (cos >= 0.9999978, max_abs <= 0.0625 versus bit-identical).
- **F28** UNVERIFIED: WSL execution suite 21/21 and Kaggle-jail results (dirs exist, contents not re-read); Kaggle critic kernel isolation; `sarvam-jev/` and `reference/` untouched; whether tools/fable_wsl_runner.py really bind-mounts protected files read-only; history before c76a1423.

## 3. What the current receipts prove / do not prove

PROVE (hash-verified receipts):
1. A real frozen Qwen3-4B-Thinking-2507 (shard sha256 checked in-kernel) on a Tesla T4, fp16, torch 2.11.0+cu128 / transformers 5.16.1 reproduces the stored TRAIN/VAL features bit-for-bit through `capture/capture*.py` (T2: 65/65 identical; T1: shapes, depths 18/24/30, no lm_head call).
2. The scorer path feeds the 3 frozen adapters + pretrained head the same tensors as the training path (T3: max logit diff 0.0 over 53 pools x 3 seeds; weights sha match the manifest; 5,246,982 adapter params each; option-order swap diff 1.9e-6).
3. The head forward pass is finite, permutation-stable (636 trials, max 5.7e-6, 0 ranking violations) and reproduces recorded per-seed VAL top-1 (0.7925/0.8113/0.7736); probe refit reproduces 0.9434; random-head mismatch control 0.1698.
4. Runtime re-capture refuses wrong revision/repo/dtype/depths/recipe/device/code-hash metadata, wrong-dtype tensors, text instead of tensors, bare-text live scoring, and (100% re-capture) fabricated, other-text and 5%-perturbed tensors with valid metadata, including a single fabricated record at each of 6 batch positions.
5. `decide_next_action` obeys its inputs on real scores (smoke S6): scorer is the last key, blocked options are excluded.
6. The capture-code pin mismatch was found and corrected by version, with evidence.

DO NOT PROVE:
1. Any end-to-end loop (T6): no Qwen-proposed options -> capture -> LAYA -> controller -> WSL execution -> feedback receipt exists.
2. That the Evidence-Ladder controller consumes LAYA scores (F03, F08); arm E is unwired.
3. T5 as a suite (never run), a real trusted-vs-LAYA conflict (F04), absence of silent fallback (F06), NONE/abstain behaviour (F05).
4. That live-mode provenance resists a wrong capture function (F10), or that imported Kaggle results are authentic (F09).
5. Any usefulness: LAYA 79% vs probe 94% on VAL, ensemble accuracy unmeasured (F18), no task-level effect, no stage-1 coding result, no critic result, no sealed evaluation (F12).
6. Reproducibility from a commit (F11).

## 4. Minimal fixes for every BLOCKER and MAJOR

- **F01** Run T6 for real: Kaggle propose+capture+score step kernel (script exists; prefill fix already in it), PC `apply` with WSL execution, >= 2 steps on >= 1 task, then summary; create-only receipts; second run for identical hashes. Report any step where real trusted evidence differs from the LAYA top.
- **F02** Add the `laya_integration` suite to the Kaggle test kernel (tools/pack_laya_tests_v1.py), run it, turn the two vacuous `return` paths into failures (or counted skips), ship T4_scores.json in the bundle, record the result path.
- **F03** Route T6 through the real controller: add a per-action entry (e.g. `EvidenceLadderController.decide_action(options, l0_scores, trusted)`) that runs router, registry and I1-I8 checks, use `authorize()` for the commit, replace the DENY regex with a fixed action-class table. If the GLM critic belongs to T6 (decisions v4), call the critic path on the proposed options with blinded packets and store the packet; otherwise write down that the critic is outside T6 and amend the v4/v5 reading.
- **F04** After F01, add a real conflict case: a task where the LAYA top option fails the sandbox dry-run (L5 fail recorded from execution, not injected) and the executed trusted option wins. Keep the synthetic counterfactual labelled "scripted".
- **F05** Implement NONE + margin-based abstain in `decide_next_action` (threshold in a new versioned toml, calibrated on VAL first), or record a user waiver for the integration milestone in CLAUDE.md/PLAN_LAYA_INT_1. Until then every report says DEVIATES.
- **F06** Journal a `scorer: absent` entry with reason when `l0_scores is None`; return `scorer_used` in the decision record and in ProblemResult/report rows; make the T6 summary fail when `scorer_used` is false.
- **F07** Add probe and random-head scorers as sibling slots to arm E (same pools) in the orchestrator and report; keep the random head labelled mismatch control.
- **F08** Either (a) build `SavedScoreBackend` from the gen kernel's per-pool output keyed by (pool id, candidate sha), or (b) call `LiveBackend` per pool with {context_text, options}; add a test that the controller journals `scorer: scored`. Otherwise mark arm E NOT WIRED in the checklist.
- **F09** Stop accepting `origin` as a label: record Kaggle run id and sha256 of `t6_step_results.json` in the receipt, verify `features_sha256` by a sampled replay re-capture in a later Kaggle step, and re-fetch the kernel output from Kaggle for comparison.
- **F10** Have `LiveScorer` take an attestation produced by the capture implementation itself (model `_name_or_path`/revision, computed shard sha256 list, `torch.cuda.get_device_name`) instead of module constants; keep the constants as the expected values it is compared to.
- **F11** Commit the untracked kernels, tools, pins file and the modified laya modules (secret scan first), record the commit hash in a new receipts README, rebuild the bundle from the commit; re-run only if the bundle hash changes. Fix the empty logs or state that logs are not retained.
- **F12** Label every report "DEVELOPMENT EVIDENCE, hidden cases public"; build sealed hidden tests before any claim beyond development; extend `static_no_grader_imports` to orchestrator/ and tools/ drivers.
- **F13** Before any pilot: freeze T and k1 in a new toml, fix the prompt (template v2) and re-run FC-3, run the real Qwen backend once on the 5 public tasks and feed those real generations through the controller, then one live GLM call set. Until then label stage 1 "unrun".
- **F14** Cluster on behaviour over a larger deterministic input set (public inputs plus boundary inputs executed in the sandbox, recorded as L1), or trigger the critic when all candidates pass the public examples, so discrimination is not gated on a signal that is constant by construction.

## 5. Notes
- Line numbers are as read on 2026-10-16/17. Only this file was created; no repo file was modified, nothing run, no secrets read, nothing committed.
