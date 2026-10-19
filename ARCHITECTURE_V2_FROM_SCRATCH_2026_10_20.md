# ARCHITECTURE V2 (from scratch): small model + execution evidence, measured at equal budget (2026-10-20)

Status: DESIGN, not built, not frozen. It replaces the LAYA-head-centred design as the plan of record only if the user accepts it.
Inputs: every recorded experiment (`results/increments.md` rows 0-73, HANDOFF_2026_10_11_STATUS.md section 16, FINDINGS_FROZEN_EL_ARCH1, the 2026-10-18/19 local bank runs),
the code as it stands (`evidence_ladder/`), the LAYA model cards on disk, an outside audit with 8 cracks (pasted by the user 2026-10-20, each checked against the code here),
and outside research (sources at the end). Figures marked (ext) are from outside papers or model cards and were not measured here.

## 0. What the system must prove
Claim to earn: "with this architecture, a 0.6B-1.5B local model solves more tasks, **independently verified**, at the **same compute budget**, without more false successes."
Two claims are kept apart: (a) the architecture (loop, oracles, selection, control) helps an untouched model; (b) changing the model's weights (LoRA) helps.
"Stronger" is never claimed from public-example passes, from a learned score, or from spending more compute.

## 1. Evidence this design stands on
### 1.1 Our own experiments
| Finding | Evidence | Consequence |
|---|---|---|
| A linear probe on raw Qwen states beats every LAYA-head variant (85.8-94.3% vs 54-80%); random-init heads beat the 4 pretrained LAYA heads (83.4% vs 76.9-79.6%) | increments rows 51-61, 63 | The LAYA head on FOREIGN hidden states is a bottleneck; its pretrained context layers destroy a linearly readable signal (87% -> 59%, row 56/58) |
| Imitation of Fable is shortcut-prone: options text alone 62-75%, history rules 55-66%, about 25% of "wrong" alternatives equivalent | rows 8-10, 47-52, HANDOFF 16.1 | Imitation labels are not a correctness target |
| Frozen mean-pooled states do not separate a working fix from one-token-broken copies (all scorers 0%, near uniform); the one probe that "worked" was the original-as-centre shortcut | rows 69-71 | No learned correctness judge from frozen states |
| Learned selectors never changed an outcome: tie with "first valid" (GLM pilots), bank tasks decided by execution | rows 64, 68; bank runs 2026-10-18/19 | Selection is not the bottleneck on the evidence so far (to be measured properly, P1) |
| Repair rescues little: of 15 bank tasks that entered repair, 2 ended SUPPORTED (P1-19, J1-31); 13 stopped on budget | `results/raw/el-bank-local-20261018T2106`, `el-bank-java5hard-wslc-20261018T213058`, `el-bank-java20-wslc-20261018T211547` | Repair as built (shows the previous candidate) is low-yield for a 1.5B model |
| Generation is the bottleneck: 78% of task time; 0.6B thinking loops, 4B thinking caps, format errors; 1.5B coder solved 1 of 5 hard Java | FINDINGS_FROZEN 2-3; bank runs | Effort goes to generation strategy and oracles |
| Sandbox + trusted verifier are cheap (1.6% of time) and qualified (WSLC gates, 24/24 forgery tests) | `results/raw/wslc-gates-20261019`, phaseF2d | They stay as the foundation |
| L2 model-written checks: 0 validated in every run; the deterministic parts (`derive_inputs`, `mutants`, `cluster`) work | `controller/l2.py`, bank runs | Retire quoted-span checks only; reuse the deterministic parts |

### 1.2 What LAYA actually is (model cards on disk + the Laya repo), and why it failed here
LAYA = a ModernBERT-large TEXT encoder (395-421M) + a typed decision head (choice / score / yes-no), trained with proper scoring rules (RLCD), about 20-40 ms per decision.
| Checkpoint | Base zero-shot -> fine-tuned on its own task | Notes |
|---|---|---|
| laya-conductor (routing effort tier) | ~37% -> 87% (ext) | 4,105 real agent turns, 30 min on a T4 |
| laya-stop-completion-judge | AUROC 0.64 -> 0.85 (ext) | 2,627 real turns; **encoder frozen, head only (26.5M params)** |
| laya-typed-decisions | 0.362 -> 0.766 (ext) | 6,000 decisions |
| coding-decision-agent, `code_review` slice (incl. likely_correct) | 0.601 accuracy (ext) | correctness judging is its weakest slice |
| laya-code (code relevance re-ranker) | fine-tuned on 50,926 git-history pairs (ext) | used by laya-codex for retrieval |
Reading: LAYA works as a **fast, calibrated text classifier for a narrow decision when fine-tuned on in-domain labels through its OWN encoder**; zero-shot it is near chance.
We used it in the two ways its own record says will fail: (1) its head bolted onto Qwen hidden states (bypassing the encoder it was trained with); (2) zero-shot "is this code correct" ranking.
LAYA is therefore not useless; it was mis-placed. Its right place (section 3, F) is cheap control decisions with no oracle (stop / continue / how much effort / which files),
trained on our own run logs, judged by compute saved at equal verified solve rate.

### 1.2b Every way we used LAYA, and whether the use itself was wrong
| # | How we used LAYA | Result (ours) | Was the use wrong? |
|---|---|---|---|
| 1 | Released LAYA with its own encoder, zero-shot, on Fable action decisions | 30.1%, chance 29.6%; 108 of 123 contexts over its window (HANDOFF 16.2) | Yes. Decisions average about 8,163 tokens (`capture-cost-model-20261002-205000.json`) against a 512-2,048-token window, and the base model is near chance on any task it was not fine-tuned for (its own cards) |
| 2 | The 31 pretrained head tensors fed Qwen3-4B hidden states through a bridge / adapter: adapter-only, joint, head-only, aligned adapter, sequence input, direct read path, head swap | 54-80%; linear probe 85-94%; random-init heads beat the pretrained ones (rows 51-63) | In design, yes. The head's context layers were trained on ModernBERT token sequences with option marker tokens; given one pooled foreign vector per option they overwrite the signal (87% -> 59%, rows 56-59). This was a fair test of "does LAYA pretraining help on another model's states" (answer: no), not a test of LAYA as designed |
| 3 | Nine-family routing head on Qwen states | head 59.0%, probe 67.1% (row 42) | Same input problem as 2 |
| 4 | Training target: imitate Fable's recorded action | shortcut-prone (options text 62-75%, about 25% equivalent alternatives) | Yes, for any model: the labels do not say which action is correct |
| 5 | Zero-shot text grader on agent traces (CodeTraceBench, RootSE, AgentProcessBench, 2026-10-14) | chance (local-peak 0.29-0.36 vs 0.33; AUROC 0.47-0.54); 0 admissible training labels (`TEXT_GRADER_PLAN_2026_10_14.md` 10-11) | Zero-shot was predictably chance (the stop judge's base was 0.57-0.64 AUROC before its fine-tune); the plan rightly stopped there |
| 6 | Native LAYA V1: coding-decision-agent, own encoder, zero-shot `likely_correct` per candidate, plus an experimental direct choice | per-candidate [0.1098, 0.1292]: the publicly failing program scored higher; direct choice: positional gap 0.444 on identical options; 35-47 s per call (`LOCAL_QWEN_LAYA_EXTENSION_REPORT.md`) | Yes, three ways: `likely_correct` is that checkpoint's weakest slice (code_review 0.601 on its own test set); the Laya repo documents option-order sensitivity, so direct multi-choice was expected to be biased; the run used CPU float32 with 2 threads, while the cards report about 20-40 ms on a GPU, so the latency is a setup artefact, not LAYA's cost |
| 7 | Frozen scorers (incl. LAYA v4 adapters) on the mutation diagnostic | 0% (row 69) | Imitation-trained scorers asked a correctness question they were never trained for |

**Never tried: LAYA the way its successful users use it.** Every working LAYA checkpoint (conductor, stop judge, typed decisions, laya-code) was fine-tuned through its OWN
text encoder, on a short packed state (the stop judge packs the final message tail first), for ONE narrow decision, with 2,600-50,000 real labels. The ledger has no run of that
kind (the 2026-10-04 "fine-tune the conductor" direction has no recorded run). So the verdict is: LAYA is not shown useless; it is shown useless for (a) reading another model's
hidden states and (b) zero-shot code-correctness judging. The untested, plausible use is a fine-tuned control classifier on our own run logs, with the GPU, judged against rules
(section 3, F; step P5).

### 1.3 Outside results that change the design (ext, not measured here)
- **Blind resampling beats self-repair for small code models** (arXiv 2607.26117): on MBPP+ at 1.5B, conditioning on the model's own failed attempt costs 6.1 points (p=0.006);
  retries copy a near-identical program 33-68% of the time vs 2-14% for blind resampling; resampling uses 2.5-5.5x fewer tokens; execution feedback content adds nothing over a placebo.
  This matches our 2-of-15 repair yield, and our `REPAIR_TEMPLATE` shows the previous candidate (`controller.py:43`).
- **S\*** (arXiv 2502.14382): parallel sampling + execution-grounded debugging + selection by **adaptively generated distinguishing inputs, judged by real execution**, lets a 3B model beat GPT-4o-mini (ext).
- **Model-written tests are often invalid**: 34-73% validity depending on model and benchmark (ConVerTest, arXiv 2602.10522, citing Yang et al.); tests written to match the code are circular.
  CodeT's own case study has a wrong consensus passing more generated tests than the correct one (arXiv 2207.10397). Agreement is not truth.
- **Selective generation** (arXiv 2505.13553): fuzz-generated tests + an abstaining selector that controls the false-discovery rate. Abstention is a feature.

## 2. Cracks (8 from the outside audit, each checked in the code, + 5 found here)
| # | Crack | Checked here | Fix in this design |
|---|---|---|---|
| 1 | Evaluation grading could leak into selection | REAL risk, guard exists: `execution/protected.py` raises `GraderIsolationError` while `SelectionGate` is open; `orchestrator/isolation.py` scans imports and the context graph; `registry.py` marks `hidden_tests_v1` grader-only | Two planes (section 3, G): runtime oracle vs evaluation oracle; the gap audit runs offline, after selection, and its results never enter a live run |
| 2 | VERIFIED too broad if it means "a stronger test passed" | REAL: `registry.py` already carries `covers` (requirement ids) and `limitations` per check | VERIFIED = every required requirement covered by an authoritative-in-scope check under the task's declared verification contract; scope and limitations stay in the verdict |
| 3 | Mutation score >= 0.80 is not a truth certificate | Agree | Mutation score is a test-quality diagnostic only; training eligibility needs an independent oracle + requirement coverage + family dedupe (section 3, H) |
| 4 | Selection gap could be mis-measured | Agree: bank pools differ in size and lineage (repairs are not independent samples) | Report Oracle@k (recorded pool), chosen-pass, gap, per task with pool size, lineage and tokens; compare only comparable pools; call it "opportunity in the saved pools", not the model's limit |
| 5 | A critic whose tests change decisions is not "shadow" | Agree | Modes OFF / SHADOW (record only) / ACTIVE (its inputs run, may drive repair); ACTIVE only via an A/B against critic-off; critic expected outputs are never an oracle (already `critic_experiment_v1` limitation) |
| 6 | Replacing all of L2 throws away working code | REAL: `l2.py` has `derive_inputs`, `mutants`, `cluster`, `passes` | Retire `qwen_tests_*` quoted-span checks only; reuse the deterministic functions |
| 7 | An experience store can contaminate evaluation | Agree | Immutable provenance per record; task-family separation before any retrieval or training; evaluation families never enter the store |
| 8 | No cost-controlled definition of "stronger" | Agree, and the controller already has the arms: `single`, `best_of_k`, `repair`, `ladder` (`controller.py:4-7`) | Primary metrics at equal model-token and equal wall-clock budget against the untouched model (section 4) |
| 9 | Repair anchors the small model on its own failure | Our data: 2 of 15 rescued; ext: -6.1 points at 1.5B | Default retry = fresh diverse resample; repair only as a measured arm, rewritten to give the counterexample without the failed code |
| 10 | Public examples are in the prompt, so passing them is weak evidence | Already labelled `public_examples_v1` "examples are few" | Public pass can reach SUPPORTED only; never VERIFIED alone |
| 11 | No sealed evaluation set: the dev bank is public in Git | Stated in `protected.py` and `isolation.py` | Add a sealed external benchmark for milestone claims (section 3, G); limit how often it is looked at |
| 12 | Adaptive overfitting: choosing strategies on the same tasks repeatedly | Agree | Strategy choice on the dev bank only; the sealed set is read once per milestone, with the read logged |
| 13 | LAYA used where its record says it fails | Section 1.2 | LAYA only for control decisions, fine-tuned through its own encoder on our logs, shadow first |

## 3. The architecture
```
              ┌─────────────────────────── RUNTIME PLANE ────────────────────────────┐
 TASK ──► A. TASK CONTRACT ──► B. GENERATOR (any model, diverse fresh samples) ──► FILTER (existing)
             requirements R1..Rn,         ▲                                          │
             verification contract,       │ fresh resample (default) /               ▼
             runtime vs eval split        │ counterexample-only repair (arm)   C. RUNTIME ORACLE (Judge 1, authoritative)
                                          │                                     sandbox + trusted verifier + check registry
                                          │                                     public examples · spec properties · invariants
                                          │                                     discriminating inputs (find disagreement only)
                                          │                                          │
                                F. CONTROL POLICY ◄──────────── D. VERDICT per requirement (VERIFIED / SUPPORTED /
                                rules first; LAYA head             PLAUSIBLE / UNVERIFIED / BLOCKED / ABSTAIN, with scope)
                                (shadow) later                          │            ▲
                                          │                             │            │ inputs/properties only
                                          └── stop / resample / escalate┘     E. DIAGNOSTIC CRITIC (Judge 2: OFF / SHADOW / ACTIVE)
              └──────────────────────────────────────────────────────────────────────┘
                        │ selected final candidate (selection gate closes)
              ┌─────── G. EVALUATION PLANE (isolated, no path back) ───────┐   ┌── H. DATA PLANE ──────────────────────────┐
              │ dev bank grader (public in Git: development evidence)      │   │ provenance, task-family splits,          │
              │ sealed external set: read once per milestone, logged       │   │ eligibility rules, never eval families   │
              └─────────────────────────────────────────────────────────────┘   └── feeds I (LoRA) and F (LAYA) training ──┘
                                                                                 I. MODEL ADAPTATION (optional, separate claim)
```

**A. Task contract.** Each task declares requirement ids, which checks are runtime-visible, and which requirements each check covers (the existing `registry.py` shape).
A task with no independent oracle for a requirement says so up front; such a requirement can never be VERIFIED.

**B. Generator.** Model-agnostic backend (`build_local_backend`, or an API). Default strategy: k fresh samples spread over prompt variants (direct / plan-first / examples-first),
lowest thinking, all tokens counted. Retry = more fresh samples. Repair is an ARM, not the default: the prompt gives the failing input, expected and observed values,
**without** the failed program (anti-anchoring). Which wins is decided by the matched-budget experiment P2, not assumed.

**C. Runtime oracle (Judge 1, the only authority).** The sandbox (MXC WSLC / ProcessContainer / Kaggle jail) and the trusted verifier (`trusted_verify.py`: verdict host-side,
expected values never inside). Check kinds and trust as in `registry.py`. New sources of independent checks, in order of trust:
spec properties and invariants that need no expected output (type, range, idempotence, round-trip, monotonicity, no crash, limits); a reference implementation where one exists;
**discriminating inputs** (`l2.derive_inputs` + S\*-style pairwise distinguishing inputs) that show WHERE candidates disagree. A disagreement is resolved only by an independent check;
otherwise it is recorded UNRESOLVED and blocks VERIFIED. Behaviour clusters (`l2.cluster`) are diagnostic only ("agreement is not correctness" stays).

**D. Verdict.** Per requirement, from the evidence ladder already built (`C.decide`, `evidence.py`): VERIFIED needs every required requirement covered by an authoritative-in-scope
check under the declared contract; public examples alone give SUPPORTED; no independent oracle gives UNVERIFIED; the system may ABSTAIN. Each verdict stores scope and limitations.

**E. Diagnostic critic (Judge 2).** The existing GLM integration. It may only propose inputs, properties and suspected failure modes; its expected outputs are never used as truth.
Modes: OFF, SHADOW (proposals recorded, nothing run), ACTIVE (proposals run through C; results may drive a resample or repair). Triggers for ACTIVE: candidates pass the runtime checks
but disagree on discriminating inputs; the same failure repeats; a requirement has no runtime check. ACTIVE is adopted only if the A/B in P4 shows more independently verified solves per unit budget.

**F. Control policy.** Decisions with no oracle: stop or continue sampling, how many samples (k), which prompt variant next, escalate to the critic, abstain.
Version 1: fixed rules. Version 2: a LAYA checkpoint fine-tuned THROUGH ITS OWN TEXT ENCODER (head-only first, as the stop judge did) on our run logs:
state = task text + round summary (pass counts, failure class, disagreement count, tokens used); label = did the next unit of budget produce a new verified solve.
Starting checkpoints: laya-conductor (effort routing) and laya-stop-completion-judge (stop). Shadow first; adopted only if it saves compute at equal verified solve rate.

**G. Evaluation plane.** Never reachable from runtime code (existing I8 scans + `SelectionGate`). Two evaluators: the dev-bank grader (public in Git: development evidence only)
and a sealed external benchmark for milestone claims (proposal: EvalPlus MBPP+ / HumanEval+, so our 1.5B numbers line up with published 1.5B numbers; needs the user's download yes).
Strategy choice uses the dev bank; the sealed set is read once per milestone and each read is logged.

**H. Data plane.** Every record carries `task_family_id, repository_id, parent_task_id, candidate_hash, oracle_id, evidence_scope, split, budget_used`.
Training eligibility: independent oracle covering the required requirements + mutation score as a quality diagnostic (tasks below 0.80 excluded, tasks with unmeasured scores held out)
+ task-family dedupe + never an evaluation family. Mutated tasks inherit the parent's family id.

**I. Model adaptation (optional, separate claim).** Expert-iteration LoRA on the generator using H-eligible data only, evaluated on the sealed set against the untouched model at equal budget.
Outside evidence says self-generated rejection sampling helps less than distillation and has diminishing returns (RepoST, MedAgentGym; ext). Last in the order.

## 4. Metrics (every result reports all of these, at matched budget)
Independently verified solve rate; solve rate at equal model tokens; at equal wall-clock; false-success rate (SUPPORTED/VERIFIED that the evaluator fails); abstention rate;
tokens and seconds per solve; for selection: Oracle@k of the recorded pool, chosen-pass rate, gap, per task with pool size and lineage. Baselines: `single`, `best_of_k` (blind resampling
at the same token budget), current `ladder`. Small task counts (e.g. 5 hard Java) are reported with intervals, never as a rate on their own.

## 5. Build order (each step has a go / no-go)
| Step | What | Data | Go / no-go |
|---|---|---|---|
| P0 | Contract fixes, no model: verdict semantics (crack 2), critic modes (5), budget accounting per call (8), provenance fields (7), retire quoted-span L2 (6) | code + tests | tests pass; I1-I8 unchanged |
| P1 | Offline gap audit of stored candidates, read-only, after selection, evaluator = dev-bank grader | 28 Py + 8 Java runs | gap small -> stop ranking work; gap large -> study why evidence missed the passing candidate |
| P2 | Resample vs repair vs anti-anchor repair at equal tokens, untouched 1.5B, existing arms | dev bank, local GPU | pick the retry strategy; expected (ext): resample wins at 1.5B |
| P3 | Oracle factory: property checks + discriminating inputs; measure false successes caught | dev bank | false-success rate down without more abstention than agreed |
| P4 | Critic SHADOW then ACTIVE A/B | dev bank | adopt only if verified solves per budget rise |
| P5 | LAYA control head on run logs (head-only, own encoder), shadow | logs from P2-P4 (need hundreds of episodes) | adopt only if compute saved at equal solve rate |
| P6 | Milestone read of the sealed set | sealed benchmark | the headline claim, once |
| P7 | Optional LoRA expert iteration | H-eligible data | separate claim, sealed set, equal budget |
Approvals still needed (current project rules): any training (P5, P7), any download over 1 GB or of an outside benchmark (P6), any Kaggle compute.

## 6. Reuse / retire
Reuse: controller arms and ladder, candidate filter, journal and replay, invariants I1-I8, check registry, `C.decide` / evidence, sandbox port + MXC WSLC + ProcessContainer + Kaggle jail,
trusted verifier, `l2.derive_inputs` / `mutants` / `cluster`, mutation pipeline, critic integration, bank runner and conformance, timeline.
Retire from the runtime path: LAYA head on Qwen states, Fable-imitation adapters, native-LAYA correctness ranking and direct multi-choice, quoted-span model-written checks,
repair-by-default with the previous program in the prompt. Kept on disk as research record.

## 7. Open questions for the user
1. Sealed benchmark: EvalPlus (MBPP+/HumanEval+) or another; download size unmeasured.
2. Acceptable abstention rate (the trade-off with false successes).
3. Whether P5/P7 training is wanted at all, after P1-P4 results.

## Sources (outside)
- Laya repo and model card: https://github.com/NandhaKishorM/laya , https://huggingface.co/convaiinnovations/laya ; local cards in `models/*/README.md`
- Blind resampling vs self-repair in small code models: https://arxiv.org/abs/2607.26117
- S\*: Test Time Scaling for Code Generation: https://arxiv.org/abs/2502.14382
- ConVerTest (validity of LLM-generated tests): https://arxiv.org/abs/2602.10522
- CodeT: https://arxiv.org/abs/2207.10397
- Selective generation for code: https://arxiv.org/abs/2505.13553
- RepoST: https://arxiv.org/abs/2503.07358 ; MedAgentGym: https://arxiv.org/abs/2506.04405
