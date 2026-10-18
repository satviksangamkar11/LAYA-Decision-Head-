# Evidence-Ladder architecture for a weak model (2026-10-16)

Status: **DESIGN ONLY. Nothing built or run.** This document answers the user's request (2026-10-16): take the pasted "Evidence-Guided Weak-Model
Agent" proposal (FormalJudge + CoVer + VeriSpecGen + SpecSyn + T1 + Universal Verifier + ADAP, LAYA optional, no training) as rule-compliant,
check it against research, and either improve it or build a better one.
Inputs: the pasted proposal; `ARCHITECTURE_DESIGN_2026_10_15.md` (A-H research, all combinations); `ARCHITECTURE_A3_FULL_SPEC_2026_10_15.md` (A3);
three verification passes written 2026-10-16 to `results/raw/research-20261016-verify/` (A_formaljudge.md, B_cover_spec.md, C_t1_uv_adap.md).

---

## 0. Verdict in five lines

1. The proposal's diagnosis is right: specs can be wrong, compliance is not completion, and a judge cannot pick an answer that was never generated.
2. Its strongest evidence does not carry over to us. FormalJudge's refinement gain had **Claude-4.5-Opus as the judge** (and as spec writer); our runtime has
   only the weak model. A 4B model writes correct formal specs about **15.8-23% of the time** even with VeriSpecGen (Opus: 86.6%).
3. So **anything the weak model writes (specs, tests, critiques) cannot be an authority**. It can only rank, propose experiments and produce repair
   feedback. Authority must come from things the weak model did not write: tests, compilers, the environment's real state, user-given examples,
   and checks certified offline.
4. The better architecture keeps the proposal's controller, outcome verification and repair loop, but replaces "spec compiler + proof engine" with an
   **evidence ladder** (who may decide what), **sandbox-first execution** (observe instead of predict), an **oracle-tier router**, and a critic that must
   produce **runnable discriminating experiments**. LAYA ranks where it is in-distribution (next-action choice in agent loops) and orders verification
   only if measured informative.
5. "Foolproof" can honestly mean eight enforced invariants (section 3): the system never claims a verification it does not have, never accepts what a
   trusted check rejects, and never lets model output authorise itself. Accuracy gains are empirical and must be measured against an equal-compute baseline.

---

## 1. Fact check of the proposal (what the cited research actually says)

| claim in the proposal | finding | evidence |
|---|---|---|
| FormalJudge, ICML 2026, GitHub htlou/FormalJudge | CONFIRMED: arXiv 2602.11136, ICML 2026 poster; repo MIT | A_formaljudge.md [read] |
| FormalJudge: 3 LLM roles + Dafny/Z3 executor; facts come from LLM extraction | CONFIRMED (paper: 4 phases; solver checks composition, facts are not proven) | A [read] |
| Authors name decomposition and fact extraction as main error sources | CONFIRMED (Section 5) | A [read] |
| Qwen 7B 30.8 -> 49.4, 14B 34.3 -> 74.2 by formal-feedback refinement | Numbers CONFIRMED; models are **Qwen-2.5**; 3 rounds; metric = **safety classification accuracy** on Agent-SafetyBench, not task success | A [read] |
| "used a strong Claude Opus model for spec synthesis" | Worse than stated: **the judge in the refinement was Claude-4.5-Opus**; Opus is also listed as spec-synthesis model | A [read], appendix C.4/C.8 |
| VitaBench / UpwardDeceivers forthcoming | UpwardDeceivers "coming soon" CONFIRMED; VitaBench DIFFERENT (linked to examples) | A [read] |
| FormalJudge on coding | NOT FOUND | A |
| VeriSpecGen 86.6% on VERINA SpecGen with Claude Opus 4.5 | CONFIRMED (arXiv 2604.10392, Lean); **Qwen3-4B-Instruct: 15.8 -> 23.0**; Qwen3-Coder-30B-A3B 29.3 -> 50.3 | B [read] |
| SpecSyn 1,071/1,365 properties | DIFFERENT by version (v2: 1,022/1,421); generators GPT-4-turbo / GPT-5 / DeepSeek-R1; fine-tuned 7-8B baselines near zero (precision 0.55-5.25%) | B [read] |
| CoVer: certifies checks by intervention; 73-predicate corpus; "captures only part of the obligation" | CoVer is from "Not Self-Decidable" (NeurIPS 2026). Check-survives-intervention CONFIRMED in part; **73 predicates NOT FOUND** (paper reports ~22,000 labels, 6 corpora); the "part of the obligation" wording NOT FOUND | B [read] |
| T1: tool-assisted Llama 1B beat Llama 8B on MATH | CONFIRMED with a caveat: the 1B is a **verifier in best-of-64**; the 8B is a single greedy run | C [read] |
| Universal Verifier separates outcome from process | CONFIRMED (arXiv 2604.06240); criteria are "non-overlapping"; verifier is **GPT-5.2**; no small open model tested | C [read] |
| ADAP needs informative score ordering | CONFIRMED as **monotonicity** (higher score not less likely to pass verification); guarantee is cost within a constant factor | C [read abstract] |
| Independent authorization blocks unsafe execution even if a validator is compromised | CONFIRMED (arXiv 2609.17648, self-reported, 60 tasks; signed tokens + policy oracle) | C [read] |
| (added) S* discriminating inputs, LiveCodeBench v4, N=8 | avg pass@1: public tests only 53.8, + generated tests 53.1, + LLM judge 55.6, + adaptive input synthesis 57.5 | C [read] |
| (added) correlated errors across judges | models agree ~60% of the time when both are wrong; shared architecture/provider drives it (Kim et al. 2506.07962) | C [read] |
| (added) small models on Dafny | CodeLlama-7B DafnyBench 28.0%; fine-tuned Qwen2.5-Coder-7B 32.4 -> 56.9 (ATLAS) | B [read] |

Consequences:
- (a) No published result shows a weak model improving itself with self-written formal specs.
- (b) The measured small-model gains come from **real execution** and **selection with a verifier over many samples**.
- (c) An LLM judge adds a little (+1.8 in S*); a grounded discriminating experiment adds a little more (+3.7).
- (d) The biggest lever in that table is simply running the public tests on several samples.

---

## 2. What I keep, change and drop from the proposal

| proposal element | decision | reason |
|---|---|---|
| Runtime controller in ordinary software, permissions, receipts, retry caps, stop conditions | **KEEP** (core) | matches loop-engineering evidence and 2609.17648 |
| Separate pre-action and post-action verification; outcome vs process | **KEEP** (core) | Universal Verifier; prevents premature success |
| Repair from concrete, factual failure messages | **KEEP** (core) | Self-Debugging, S*, Reflexion; self-repair without real feedback is weak |
| UNKNOWN / TIMEOUT / MISSING_EVIDENCE never become PROVED | **KEEP**, made an invariant (I4) | |
| Critic must output a structured challenge with a next check, not a vote | **KEEP and sharpen**: the challenge must be a **runnable** experiment (input, command, state query), else it is dropped | S*; free-form judges add little |
| Ablation ladder; equal-compute control; domain-conditional enablement | **KEEP** (core) | |
| Intent compiler before generation (FormalJudge decomposition) | **CHANGE**: deterministic contract skeleton + weak-model requirement list where **every item must quote a span of the request** (deterministic substring check), otherwise dropped as phantom; ambiguities listed, not resolved by invention | 4B spec accuracy 15.8-23%; traceability idea from VeriSpecGen is cheap when made a string check |
| Spec validation by coverage + traceability + mutation + trusted facts | **CHANGE**: run-time-feasible version only: traceable span, non-vacuity (a check that passes every mutant of a candidate is vacuous), consistency with user-given examples, agreement across independent candidates. A validated model-written check is still only rank-and-feedback evidence (level L2), never approval | no reference solution exists at run time; SpecSyn needs one |
| CoVer certification of checks | **CHANGE**: certification happens **offline**, once, for a reusable check library (level L4), with fixed version and hash; never per task at run time | CoVer's own paper title: LLMs cannot decide what a verifier can check |
| FormalJudge proof engine (Dafny/Z3) | **DEFER**: only for natively formal tasks and only with certified library specs; never with weak-model-written Dafny | no small-model spec-writer evidence; refinement gain needed an Opus judge |
| LAYA as cheap ranker, optional | **CHANGE placement**: LAYA ranks **candidate next actions inside agent loops** (what its adapters were trained on: Fable next-action choice), orders verification effort **only if the monotonicity check passes**, breaks ties among equal-evidence candidates; never approves | ADAP needs monotonic ordering; our adapters are in-distribution only for action choice |
| One semantic critic, small LLM | **KEEP, constrained**: same weak model with a blinded prompt by default (6 GB holds one model); a second small model only if measured better | correlated errors (60% agreement when both wrong) |
| Dafny for maths, LSP for code, primary sources for research... | **KEEP as the oracle-tier router** (section 5) | |

New elements not in the proposal:
- **Evidence ladder** (section 4): one table defines what each kind of evidence may decide. It replaces ad-hoc "trust" judgements.
- **Sandbox-first execution**: every reversible action runs first in a copy (repository copy, dry run, staging), so pre-action checking becomes observation. Prediction is needed only for irreversible actions.
- **Oracle-tier router** at intake: decides the loop, the budget and the strongest possible output flag before any generation.
- **Coverage engine** for the "missing answer" risk: investigate-first, diverse strategies, repair from counterexamples, discriminating inputs.
- **The offline strong-model allowance**, spent once on a certified check library and repair templates. It is not spent per task.

---

## 3. The eight invariants (what "foolproof" can honestly mean)

Each is enforced by controller code and has a unit test that must pass before any run (RC-I1..I8). They do not guarantee correct answers. They
guarantee honest, bounded, non-self-authorising behaviour.

| id | invariant | enforcement |
|---|---|---|
| I1 | **Authority**: an answer is flagged VERIFIED only if level L4 or L5 evidence supports it; the flag always equals the highest level actually reached | flag computed from the evidence record, never from model text |
| I2 | **Monotone safety**: no score (LAYA, probe, critic, confidence) can accept a candidate that L3-L5 evidence rejects, or reorder candidates across evidence levels | lexicographic ranker with assert |
| I3 | **No self-authorisation**: nothing the model writes (spec, test, critique, tool-output text, retrieved page) can change permissions, caps, evidence levels or the check library | permissions and levels live in controller state; model outputs are data only |
| I4 | **Unknown is not pass**: UNKNOWN, TIMEOUT, ERROR, MISSING_EVIDENCE, INVALID_SPEC never map to pass | status mapper with exhaustive test |
| I5 | **Bounded**: caps on rounds, samples, tokens, wall time; exhaustion yields a flagged best effort, never a "decided" answer | counters, terminal states |
| I6 | **Observe before commit**: reversible actions are executed in a sandbox and checked before the real commit; irreversible actions need L4/L5 pre-checks and, where the task allows, user confirmation | action classifier is a fixed table, not a model call |
| I7 | **Frozen scorers, immutable journal**: scorers and the check library are hash-checked at start and end of a run; the journal is append-only | sha256 checks |
| I8 | **Sealed evaluation**: evaluation-only oracles (hidden tests) are unreachable from runtime code | import guard + self-test |

---

## 4. The evidence ladder (who may decide what)

| level | evidence | who produced it | may it approve? | used for |
|---|---|---|---|---|
| L5 | real environment observation: user-supplied tests, pre-existing repo tests, compiler/type checker, actual post-action state, tool results from trusted APIs | the world | **yes** | approval, repair feedback |
| L4 | certified check library: built offline (once), validated by mutation against reference implementations, versioned and hashed; deterministic | offline (human or one-time strong model) | **yes, within its stated scope** | approval, repair feedback |
| L3 | examples and explicit constraints quoted verbatim from the request, checked deterministically | the user | supports, does not prove completeness | rejection, repair feedback, ranking |
| L2 | weak-model-written checks (tests, properties) that passed run-time validation: traceable span, non-vacuous on mutants, consistent with L3/L5, agreed across candidates | the weak model, validated | **no** | ranking, repair feedback |
| L1 | behavioural agreement among independently generated candidates (same outputs on the same inputs) | execution of model output | no | ranking |
| L0 | scores and opinions: LAYA, probe, DeepConf confidence, critic's judgement of concrete outputs | models | no | ordering effort, tie-breaks |

Output flags: VERIFIED (L5/L4) · SUPPORTED (L3/L2, no contradiction) · PLAUSIBLE (L1/L0 only) · UNVERIFIED (nothing) · BLOCKED (a hard constraint fails
or needed information is missing). Every report carries two separate fields: `outcome` (did the end state meet the goal) and `process` (were the constraints
respected).

---

## 5. Oracle-tier router (first decision, deterministic)

| tier | condition (detected by code, not by the model) | loop used | best possible flag |
|---|---|---|---|
| O1 | a trusted executable oracle exists for the goal (tests given, repo tests cover the change, exact expected state known) | generate - execute - repair until L5 passes | VERIFIED |
| O2 | partial oracle: examples in the request, compiler/types, regression suite, certified library checks, invariants | generate - execute - discriminate - repair; L2 checks for ranking | VERIFIED for the covered part, SUPPORTED otherwise |
| O3 | no oracle (subjective, open-ended, missing facts) | best effort + requirement coverage review; ask the user when a blocking ambiguity is found | PLAUSIBLE / BLOCKED |

Domain verifiers sit behind the tiers (coding: compiler, LSP, tests, diff scope; web: real application state, confirmation records; research:
claim-to-source comparison with retrieved primary sources; maths: symbolic computation, SMT on certified encodings; data: schema, row counts,
reproducible recomputation).

---

## 6. Runtime loop (one controller, eight stages)

```
 1 INTAKE      contract skeleton (deterministic): permissions, files/tools in scope, signatures, caps, oracle tier (section 5)
               requirement list (weak model): every item must quote a request span (string check) -> else dropped; ambiguities listed
               O3 with a blocking ambiguity -> ask the user (BLOCKED until answered)
 2 INVESTIGATE gather evidence before proposing: read the relevant code/state, run the existing tests to SEE the failure, LSP diagnostics
               (deterministic tool calls chosen from a fixed recipe per domain; agentic tasks: Qwen proposes next actions, LAYA ranks them)
 3 GENERATE    k diverse candidates: >= 2 prompt strategies x sampling temperature; k set by tier and by measured difficulty signal
 4 EXECUTE     in a sandbox copy: L5 checks, L4 library checks, L3 examples; behaviour clusters (L1); status per check, I4 mapping
 5 DISCRIMINATE if several clusters survive: critic (same Qwen, blinded) proposes a distinguishing input/experiment; it is run; the critic
               judges the concrete outputs against quoted requirements (L0, grounded); model-written properties validated to L2
 6 RANK        lexicographic by evidence level (I2), then LAYA/probe score (only if monotonicity passed), then confidence, then fixed id
 7 REPAIR      if no candidate passes the highest available check: repair the top candidate from its most concrete failure
               (L5 > L4 > L3 > L2) with the fixed template; new candidates go back to 4; at most 3 rounds; no-progress stop
 8 COMMIT and  commit the chosen action for real; re-run the trusted checks on the REAL end state (not the sandbox copy); compare diff scope
   VERIFY      report: answer, flag (section 4), outcome and process fields, evidence receipts, unresolved items
```
Repair template (fixed):
```
Previous candidate: <id>.
Verified failure (level <Lx>): <input> -> expected <e>, observed <o>.
Requirement (quoted): "<span>".
Unchanged constraints: <list>.
Task: propose the smallest corrected change. Do not modify anything outside <scope>.
```

---

## 7. Where the adapter + LAYA head sit, and when they earn the place

| job | stage | why it can work | condition to keep it | fallback |
|---|---|---|---|---|
| rank Qwen's candidate next actions | 2 (agentic tasks) | the adapters were trained on exactly this (Fable next-action choice from Qwen states); measured 79.2% on VAL vs probe 94.3% | improves verified task completion vs Qwen's own first proposal at equal steps | Qwen's own top proposal |
| order verification effort | 4 | ADAP: adaptive verification is efficient only with a monotone score | DEV check: pass rate falls with LAYA rank (monotonicity test, CI) | cheapest-check-first |
| tie-break among equal-evidence candidates | 6 | frozen states carry correctness signal (2606.14530, 2607.05188) | stack with LAYA beats stack with probe and with no learned key | DeepConf, then fixed id |
| decide approval | never | no evidence any learned score should approve | - | - |

Rules: LAYA and the critic are never averaged (different events, not calibrated to each other). The probe on the same states is always run next to
LAYA (CLAUDE.md), and a random-head adapter tests whether the pretrained head matters.

Constraint from the proposal: with no training, the existing Fable adapters are used as they are. They are in-distribution for next-action ranking
and out-of-distribution for ranking whole programs. Retraining the 8 adapter tensors on execution labels is allowed by CLAUDE.md but excluded by the
proposal's "no training". **Decision D3.**

---

## 8. What gets built first (smallest version that tests the core)

The core is everything marked KEEP plus the ladder, the router, sandbox-first execution and repair. Everything else is added only through the ablation
ladder in section 9.

**Stage 1 (function-level coding, tier O1/O2; this is A3 re-based on the ladder):**
- the previous A3 spec carries over (components M0-M13, recheck points RC-00..RC-71), with three changes:
  - M8 ranker = evidence ladder (L5 public tests > L3 statement examples > L2 validated Qwen tests > L1 clusters > L0 LAYA/probe/DeepConf);
  - M7 critic = S*-style distinguishing inputs judged on concrete outputs;
  - M12 adapter retraining becomes optional (D3).
- Data: LiveCodeBench problems released after Qwen's July 2025 release (D1).

**Stage 2 (agentic, tier O1/O2):**
- a fixed Agentless-shaped workflow (localise -> repair -> validate) on tasks with real tests;
- LAYA ranks Qwen's proposed next actions at the localise stage;
- environment: the qualified WSL sandbox (local exception, approved 2026-10-13) or a tool-calling benchmark with executable checks (D2).

**Stage 3 (other domains):**
- only after stages 1-2 show which components earn their place; each domain gets its own verifier interface (section 5).

---

## 9. Evaluation (the architecture must prove itself)

Primary metric: **verified task-completion rate** on independent evaluation oracles (hidden tests / scripted end-state checks) that runtime never sees (I8).
Also: false-approval rate (VERIFIED or SUPPORTED but actually wrong), false-rejection rate, repair success rate, abstention/BLOCKED rate, phantom-requirement
rate (requirements with no source span, before filtering), challenge precision (critic challenges that led to a confirmed failure), tokens / wall time per
solved task.

Ablation ladder (each step on the same frozen task set, each with an **equal-compute twin**: Qwen + the same total tokens/samples spent on plain best-of-k by
public tests):

| step | configuration | question |
|---|---|---|
| B0 | Qwen single attempt | floor |
| B1 | Qwen best-of-k by public tests (equal compute to each later step) | the control every step must beat |
| S1 | + repair from concrete failures | does feedback beat more samples? |
| S2 | + L3 statement examples + L2 validated Qwen tests | do model-written checks help beyond public tests? |
| S3 | + discriminating experiments (critic) | does the critic find real errors (challenge precision) and raise completion? |
| S4 | + LAYA (ordering and tie-break; action ranking in stage 2) vs probe vs random-head | does LAYA beat the probe and no learned key? |
| S5 | + requirement contract (span-checked) | fewer missed constraints without phantom ones? |
| S6 | + certified library checks (L4) / formal checks where applicable | extra real errors found beyond tests? |

Gates (proposals, frozen in a thresholds file before any run):
- **G-0 headroom:** oracle@k minus pass@1 is at least 5 points. If it fails, stop.
- **Per step:** a step stays only if its verified-completion gain over its equal-compute twin has a problem-clustered CI lower bound above 0, or it lowers false approvals without lowering completion.
- **Safety:** false-approval rate is no higher than B1's.
- **Unit tests:** I1-I8 all pass before any run.

A step that helps one domain and hurts another becomes domain-conditional.

Stress set (required): tasks with missing information, misleading tool output, a deliberately wrong statement example, a broken test tool (must give
ERROR, not pass), and planted flawed candidates.

---

## 10. Honest limits

- No architecture makes a weak model correct on arbitrary tasks. When no oracle exists (O3), the best this design can do is say so: PLAUSIBLE or BLOCKED.
- Published gains for the extras are small at our scale. The biggest measured lever is best-of-k with real tests plus repair from real failures.
  Expect the critic, model-written tests and LAYA to add a few points at most, if anything. The ladder measures exactly that.
- Correlated errors: a same-model critic shares the generator's blind spots. Only execution breaks the correlation.
- LAYA: on every test so far the probe matched or beat it; the design gives each LAYA job to whichever wins and reports it.
- Repository-level tasks need working sandboxes; our earlier SWE-rebench pilots did not qualify enough tasks (v2 HOLD). Stage 2's environment choice matters.

---

## 11. Decisions needed

| id | decision | suggestion |
|---|---|---|
| D1 | stage-1 problem set | LiveCodeBench, problems after 2025-07 (licence and size checked before download) |
| D2 | stage-2 environment | qualified WSL sandbox tasks, or a tool-calling benchmark with executable checks |
| D3 | may the 8 adapter tensors be retrained on execution labels (allowed by CLAUDE.md, excluded by the proposal's "no training")? | default no; run it as one extra arm only with your yes |
| D4 | may the one-time offline strong-model use build the certified check library (L4)? | yes, for stage 3; stages 1-2 do not need it |
| D5 | critic = same Qwen (blinded) or a second small model | same Qwen first; a second model only if measured better |
| D6 | GO for P0 (thresholds file + prep script + invariant unit tests, local, no compute) | yes |

---

## 12a. Revision R1 after review (2026-10-16): replaces the matching parts of sections 3-6

**R1.1 Evidence is per requirement, never per answer.**
- Every check result is a record: `{requirement_id, check_id, level, status, assumptions, scope}`.
- A requirement is VERIFIED only when an L4/L5 check whose stated scope covers that requirement passes.
- The answer carries a **coverage table**, one row per requirement: VERIFIED / SUPPORTED / UNVERIFIED / FAILED, plus the assumptions. The answer-level flag is the **weakest row**, not the strongest check.
- Example: "refund >= 0" proved by a solver verifies only that property. "Implements the refund policy" stays UNVERIFIED unless a policy-level oracle (user tests or examples of the policy) passes.
- I1 is restated: VERIFIED for the answer requires every hard and functional requirement to be VERIFIED. Otherwise the flag names what is open.

**R1.2 The router classifies verification opportunities, and it is re-run.**
- The router runs at intake and again after every stage that adds evidence (investigate, retrieval, repair).
- It works per requirement: "which trusted check could establish this requirement now?"
- A requirement can be upgraded when a trusted source appears, for example repo tests found during investigation (L5), a certified library check that matches (L4), or an example quoted in the request (L3).
- It cannot be upgraded by artifacts the model writes. A test Qwen builds stays at L2 however convincing it is. Retrieved documentation is untrusted content until a check is grounded in it.
- The classification itself stays in code (I3); only the inputs change.

**R1.3 Repairs continue on progress, not on a fixed count.**
- Progress means at least one of:
  - a requirement moved to VERIFIED;
  - a failing check now passes;
  - a new failure is localised (a different failing check, input or line);
  - a discriminating experiment removed a candidate cluster.
- Stop rules:
  - the same failure signature twice;
  - a regression (a previously passing check fails) that is not repaired in the next round;
  - two rounds in a row without progress;
  - the hard budget (tokens, wall time) is reached.
- The hard budget stays (I5 bounds every run). It is set on DEV data and reported, and every comparison is at equal total compute, so "more rounds" cannot hide as a gain.

## 12b. Revision R2 after second review (2026-10-16)

- **R2.1 The controller runs per action, not once per task.** Every tool call or edit goes through intake -> check -> commit. Task-level outcome
  verification is still done once at the end.
- **R2.2 Evidence levels are not a total order.** A record counts for a requirement only if its scope covers that requirement. A formal proof of a badly
  specified property can be worth less than one observed behaviour. Every record carries provenance: model statement, retrieved document, tool output,
  test result or certified check. Model statements are never independent evidence.
- **R2.3 Approval depends on risk.** Approval is a function of (requirement coverage, evidence, action risk).
  - Reversible and sandboxed actions: approved on trusted checks.
  - Irreversible actions: need stronger coverage and the user's confirmation, whatever the checks say.
- **R2.4 Six specification deliverables before any code.** P0 now covers:
  - requirement schema;
  - evidence-record schema;
  - verification-tool registry (with scope and assumptions per check);
  - execution and approval policy;
  - progress-based repair policy;
  - evaluation protocol.

## 12c. Decision D5 resolved (user, 2026-10-16, final): critic = `z-ai/glm-5.3-flash` on the NVIDIA API

(Went GLM-5.3-flash -> DeepSeek v4.1 Flash -> back to GLM-5.3-flash, all on the user's instruction.)

- **Availability (logged in `results/raw/phi4-probe-20261016/ledger.jsonl`):**
  - phi-4-mini-instruct: 410, retired on 2026-07-15;
  - granite-34b-code-instruct: 404 on all 14 keys;
  - glm-5.3-flash: 200, answered "OK".
- Every callable model found is 30B-class or larger.
- **Call logic is the existing one:**
  - every `.secrets/nvidia_api_key_<n>.txt` (n >= 2) in rotation;
  - up to 5 requests in flight per key, at most 39 sends per key per minute;
  - a rate-limit response cools that key for 60 s and re-queues the request;
  - every send is logged, and a completed id is never resent;
  - spend Rs 0;
  - "max quality": reasoning effort is never lowered.
- **Packets are blinded:**
  - the critic sees the quoted requirements, the candidates' code and their concrete outputs;
  - never LAYA's pick, never the generator's reasoning, never hidden tests or labels (I8).
- **Consequences, stated in every report:**
  - the critic is an external model stronger than Qwen3-4B, so its contribution is reported as "with an external API critic", not as the weak model alone;
  - the equal-compute arm B gets the critic's generated tokens as extra Qwen samples;
  - challenge precision is measured;
  - the critic stays only if arm D beats arm C.
- **Not deployable locally as-is:** at runtime it sends task text off this machine. Fine for the Kaggle and PC experiments under the NVIDIA rules; a local deployment needs a local critic.

## 12. Checklist

| item | status | evidence |
|---|---|---|
| Proposal read and taken as rule-compliant | DONE | section 2 |
| Proposal's cited claims verified | DONE (3 passes, Haiku 5.5) | section 1; `results/raw/research-20261016-verify/` |
| Keep / change / drop per element, with reasons | DONE | section 2 |
| Invariants, evidence ladder, router, loop, LAYA placement | DONE | sections 3-7 |
| Build order and evaluation ladder with equal-compute control | DONE | sections 8-9 |
| Thresholds file, prep script, invariant tests | NOT DONE | waits for D6 |
| Any compute | NOT DONE | waits for D1, D6 and Kaggle GO |
| D1-D6 | WAITING ON USER | section 11 |

## 13. Revision R3 (2026-10-18): what changed since this document, in one place
Full table with state and "used by the local run" per change: [ARCHITECTURE_AND_RUN_PLAN_2026_10_18.md](ARCHITECTURE_AND_RUN_PLAN_2026_10_18.md). Frozen findings: [FINDINGS_FROZEN_EL_ARCH1_2026_10_18.md](FINDINGS_FROZEN_EL_ARCH1_2026_10_18.md).
- The eight invariants, the ladder L0-L5, the router O1-O3 and the eight stages are unchanged. Two final paths are kept apart: PATH A (live model; LAYA only on the exact accepted pool) and PATH B (stored 4B scores on exact stored contexts).
- New: candidate filter (frozen interface v1), per-step Timeline (hash-neutral), enforced caps with named stop labels, the execution provider port with **MXC** as the replacement sandbox (fail closed, WSL kept for rollback), the Kaggle chroot+seccomp jail (MXC's Linux backend cannot start on Kaggle), the thinking policies recorded in every result, the bank-40 runner, and the local runner for the Qwen2.5-Coder-1.5B model with internal-state capture (diagnostic only).
- Measured so far: toy examples 3/3 VERIFIED on Kaggle 4B with a thinking budget; bank run 32/40 tasks done (30 SUPPORTED, public examples only, hidden grading not done); L2 INSUFFICIENT on all 32 (parser fix in progress).

