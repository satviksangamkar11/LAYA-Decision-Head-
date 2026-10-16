# Architecture design for helping weak (small) models: per-research notes, then all combinations (2026-10-15)

Goal (user): the best possible answers from a small local model (Qwen family), using the adapter (Qwen states -> LAYA) + the pretrained LAYA head
+ loop engineering; no strong model at runtime (offline use allowed); hardware 6 GB GPU / 15 GB RAM + free Kaggle. Method (user): read each research
file, write what we can implement from it, then the next; then re-read everything and design the architecture from all combinations.

Sources: `results/raw/research-20261015-smallmodel/{A..H}.md` (consolidated in `RESEARCH_FINDINGS_SMALL_MODELS_2026_10_15.md`); our own measured
results in `results/raw/` (cited where used). Evidence grades from the files are kept: [abs]/[primary] read from abstract or paper page; [snip]
search snippet; [unv] unverified.

---

## Part 1. What we can implement, one research file at a time

### After reading A (methods for small coding agents)
Implementable:
- A1. **Execution-verified selection over k samples** (Large Language Monkeys, CodeMonkeys, CodeT, R2E-Gym hybrid): sample k candidates, run tests /
  model-written tests, pick. Frozen model, no training. Strongest evidence in A.
- A2. **Model-written reproduction / test scripts** as the oracle when hidden tests are absent (CodeMonkeys, CodeT dual agreement).
- A3. **Agentless 3-stage pipeline** (localise -> repair -> validate): fixed workflow instead of a free agent; fewer decisions for a weak model.
- A4. **Hybrid verifier** = execution-based + execution-free scorer combined (R2E-Gym: each ~42-43% alone, 51% combined [abs]) -> an execution-free
  scorer (our adapter + LAYA could be one) is useful as the SECOND signal next to tests, not alone.
- A5. Training routes (need the "Qwen never trained" rule changed): LoRA SFT on offline trajectories (SWE-smith 14.3% @100 -> 40.2% @6.2k, 32B [abs]),
  RL with verifiable rewards (SWE-Master 4B 27.6 -> 33.4 [snip]).
Warnings: small open models self-repair poorly (4.4% vs 28.9% [snip]); the only small PRM test (SWE-Shepherd) LOWERED resolve rate (51 vs 57 [snip]);
no learned steering component on a small model beat simple baselines [not found]. -> do not make a learned scorer the sole decider.

### After reading B (pass@1 vs pass@k coverage gap)
Implementable / design facts:
- B1. **Pick the task level where the frozen 4B already has coverage.** Untrained 7-32B bases solve 1.8-7% of SWE-bench Verified [primary] -> a selector
  has almost nothing to pick from at repo level. Qwen3-4B thinking: LiveCodeBench v5 54.2 [primary] -> function-level is where selection can work now.
  Repo-level becomes viable only after training (Qwen3-4B SFT/RL 27.6-38.9% [primary]).
- B2. **Verifier ceiling numbers to aim for:** verifiers recover ~50-62% of the pass@1 -> pass@k gap at 32B [primary, calc]; at k=5 (35B model):
  public tests 69%, same-family LLM judge 30%, generated tests 23%, cross-family LLM 17% [primary, one paper]. So: **real tests >> any learned or
  LLM judge**; a learned judge (our LAYA slot) should be expected near the 20-30% band, and only adds value when tests are missing or tied.
- B3. **Measure the gap first.** No source reports oracle-vs-selected for any model <= 8B [not found]. The first experiment must measure pass@1, pass@k
  and selected@k on OUR model; that headroom number decides whether any selector (LAYA or not) is worth building.
- B4. Cost reality at 4B: ~19-54 min per SWE issue with 67-376 env runs [primary]. Repo-level k-sampling is expensive on free Kaggle; function-level
  sampling is cheap.
- B5. Self-consistency recovered only 37% of the gap, patch classifier 50% (Qwen3-32B) [primary, calc] -> majority vote is a strong-enough cheap
  baseline LAYA must beat.

### After reading C (test-time compute)
Implementable (all inference-only, Qwen frozen):
- C1. **S*-style loop** = parallel samples + sequential debugging with real test output + *distinguishing-input synthesis* (generate an input on which
  two surviving candidates differ, run both, ask the model which output is right) [abs: 3B+S* beats GPT-4o-mini]. Most direct small-model code win.
  LAYA slot inside it: rank the survivors when tests cannot separate them.
- C2. **CodeT dual agreement**: candidates x model-written tests matrix; score = (#candidates agreeing) x (#tests passed) [snip, +18.8].
  Cheap, no training; a strong non-learned baseline for the LAYA verifier slot.
- C3. **Repair rounds with execution traces, capped** (Self-Debugging up to +12% [snip]; self-repair without real feedback is weak).
  Cap 2-3 rounds; plateau by ~5 attempts (SWE-World, R2E-Gym).
- C4. **DeepConf**: per-trace confidence from token log-probs; drop low-confidence traces early, confidence-weighted vote [snip, math].
  Free signal from the generator; MANDATORY competitor and a cheap input feature for LAYA.
- C5. **Difficulty-adaptive compute** (Snell): estimate difficulty -> choose k / rounds. Needs a difficulty predictor -> a natural LAYA/adapter job
  (P(solved) after the first sample = difficulty).
- C6. **Thinking-budget sweep** with an inverse-scaling guard: measure accuracy vs thinking tokens on our tasks, stop at the knee.
- C7. **Agentless** as the repo-level baseline structure.
Rejected (compute-matched negatives): debate, mixture-of-agents, self-critique without execution, LATS / MCTS at our scale.

### After reading D (scaffolds, memory, prompt optimisation, steering, constrained decoding)
Implementable:
- D1. **Pre-generation difficulty / pass prediction from the prompt's last token** (2606.14530: Qwen3-4B, AUC 0.91 with length regressed out vs 0.75
  length baseline [abs]). This is EXACTLY our capture protocol (hq = last prefix token). -> LAYA/adapter as a **before-sampling** budget controller:
  P(first attempt passes) decides k. Needs length control + linear probe baseline.
- D2. **Per-step correctness probes in agent trajectories** (2607.05188: AUC up to 0.83; predicts parse / #failing tests; transfers across
  benchmarks [self-reported]) -> LAYA as a per-step "is this trajectory going wrong" monitor = loop controller's stop/restart signal.
- D3. **Observation masking** (half cost, solve rate same [self-reported]) + ACI tool rules (windowed viewer, linter-gated edits). Deterministic,
  zero model calls. Always on.
- D4. **Agentless pipeline** with LAYA at localize-rank and patch re-rank (repeats A3/C7).
- D5. **Execution-verified experience memory** (Reflexion/ReasoningBank with sandbox result as label) + a **retrieval-applies gate**
  (SWE-ContextBench: wrong memories hurt) -> another candidate-scoring job for LAYA (score "does this memory apply").
- D6. **Constrained decoding only at final packaging** (tool name / JSON / diff format) with an abstain option kept; never constrain reasoning
  (Constraint Tax: accuracy 19.7 -> 11.0 at 0.5-1.5B). Logit masking to valid tool names: Qwen2.5-7B 49.5 -> 78.1 [self-reported].
- D7. GEPA reflective prompt evolution (Qwen3-8B beats GRPO [abs]) - prompt-only, but rollout-heavy; later.
Not for pass rate: steering (only style evidence, costs quality), model merging (negative on Qwen2.5-Coder-1.5B).

### After reading E (self-improvement and cheap learning)
Implementable. Training Qwen conflicts with the current rule "Qwen is never trained", so each item is marked by what it trains.
- E1. **Adapter-only outcome training on the loop's own executed attempts (allowed now).** Every sample the loop runs gets a real pass/fail label.
  That stream is free supervision for the adapter (+ optionally the LAYA head), unlike Fable imitation. This is the ReST idea applied to the JUDGE,
  not the generator. It closes the data loop without touching Qwen.
- E2. **ReST-EM / STaR on Qwen (needs rule change):** sample, keep test-passing, LoRA SFT, repeat [abs]. Works only if the base already solves a
  share (STaR chance limit). Warning: SWE-Gym found on-policy self-trajectories did not help at 32B [snip].
- E3. **Short LoRA GRPO with test reward (needs rule change):** AceCoder 7B 80 steps ~48 H100-h [unv]; Tina LoRA ~$9 [abs]; Dr. GRPO loss fix;
  lint-only reward causes short-code hacking [abs] -> tests-only reward. DeepSWE: most gain in the first 200 steps [snip].
- E4. **Offline one-time distillation (allowed: strong model offline once):** s1/LIMO show ~1k curated traces shape behaviour [abs]; 1k
  execution-verified traces -> LoRA on Qwen (needs rule change for the Qwen LoRA itself).
- E5. **Hindsight hints from own failures (HHD, +8 vs +2 claimed [unv])**: reuse the failure buffer; at inference a hint-retrieval variant is
  training-free (overlaps D5 memory).
- E6. **Task sources with executable tests:** SWE-smith bug injection in our WSL sandbox, R2E-Gym/SWE-rebench recipes, rStar-Coder
  (verified problems; >1 GB download needs a yes), Absolute Zero (zero-data self-proposed tasks checked by executor).
- E7. **Training-free self-play** (CoSPlay: code + self-written tests at test time) = another candidate-and-test generator for the loop.
- E8. Execution-gated PRM / PRIME implicit PRM from outcome labels only: the learned-verifier route if LAYA as a verifier underperforms.

### After reading F (capacity, routing, uncertainty)
Implementable:
- F1. **Base model choice is the biggest free lever.** Qwen3.5-9B (official LiveCodeBench v6 65.6, BFCL-V4 66.1, Apache-2.0 [verified card]) likely
  fits 6 GB at Q4_K_M (~5.5 GB, estimate). Qwen3-4B-Thinking (our current backbone) LCB v5 54.2 (different version; not comparable directly).
  RAM 15.3 GB rules out a 30B-A3B at Q4. -> the adapter must be **portable**: one adapter per Qwen model, head shared (user goal).
  Caveat: Qwen3.5-9B at Q4 leaves no VRAM for hidden-state capture at full precision on the laptop; runtime hidden states come from the same quantised
  model -> capture must be done the same way at train and run time (device/dtype parity rule).
- F2. **Hidden-state correctness probes** (repeats D1/D2) are the strongest uncertainty signal found; **multi-sample P(True)** beats token entropy
  for code (ICML 2026 listing [snip]); **top-K token entropy** single-pass 0.78 AUROC (Qwen3-14B [verified abs]) -> cheap feature, too weak alone.
  Do NOT gate on mean token entropy.
- F3. **A model cannot see its own blind spots** (NeurIPS 2026 [snip]) -> the LAYA/probe gate is advisory; real tests stay the authority (matches
  our evidence order).
- F4. **Cascade / escalate-on-disagreement, local only:** two local judges (e.g. LAYA and probe, or two Qwen sizes) agree -> accept; disagree -> run
  more execution or more samples ("LLMs as a Jury" pattern, math evidence). This is our loop-v2 S_vote rule generalised to a compute decision.
- F5. KV cache q8_0 ~ no loss (llama.cpp PR, LLMC); 2-bit KV crashes. Speculative decoding: probably no VRAM room with a 9B target.
- F6. Offline teacher on Kaggle: Qwen3.6-27B (77.2 SWE-V self-reported) for one-time data creation (allowed: strong model offline once).

### After reading G (unconventional / frontier)
Implementable (cheapest first):
- G1. **Evolutionary search with an execution evaluator** (ShinkaEvolve, Apache-2.0; AlphaEvolve pattern): small Qwen as the mutator, tests as fitness.
  This is the "heavy loop" for hard items. LAYA's job inside: cheap pre-filter (score children before paying for execution) and parent choice.
  Must be compared with best-of-N at equal samples (no small-mutator evidence found).
- G2. **Layer repetition at inference** (repeat a middle block of frozen Qwen; no training): a zero-training capacity knob; control = equal FLOPs
  spent on more samples. Speculative.
- G3. **Repository / per-task test-time training** (TTT for ARC: 8B -> 53%, 6x over base [snip]): few LoRA steps per repo on a masked-line loss.
  Touches Qwen weights temporarily (reset after each repo) -> needs the user's ruling.
- G4. **Text-to-LoRA / Doc-to-LoRA hypernetworks**: generate an adapter from a task description in one pass; matches "one adapter per Qwen model"
  in spirit but is a large training job. Later.
- G5. **NVARC pattern** (ARC 2025 winner: ~4B model, heavy reasoning moved offline into synthetic data from stronger open models, $0.20/task):
  exactly our "strong model once offline, small model at runtime" rule. Strongest real-world precedent for our setup.
- G6. **PLUM / CodeDPO**: execution-scored own samples -> preference pairs (PLUM +4.8 HumanEval/MBPP, +11.8 LCB [abs]). Trains Qwen -> rule change.
- G7. **Neurosymbolic tools in the loop** (compiler diagnostics, type checker, static analysis as extra feedback; no training) -> extra
  deterministic feedback channels for the repair rounds and extra features for LAYA.
- G8. Lesson from the TRM critique: credit a mechanism only after ablating voting/augmentation at matched compute -> every loop result needs
  matched-compute controls (we already do this with S_vote etc.).
Not adoptable: Engram, looped/recurrent pretraining (Ouro, Huginn, Coconut), memory layers, diffusion LMs, SAE steering for code.

### After reading H (general agents: tool use, planning, asking, heads, memory)
Implementable:
- H1. **Tool-call verdict from frozen states** (Yeats et al., NeurIPS 2026 workshop [V]: linear probes on 18 tool-calling LLMs detect wrong calls,
  including right-type wrong-value arguments; generalise to new error types). This is LAYA's NATIVE `tool_call.call_verdict` question
  (execute / fix_args / wrong_tool / abstain). Labels are free: schema validation + real execution errors. Strongest fit between LAYA's pretrained
  questions and the literature.
- H2. **Future-action prior from one frozen state** (~3/4 depth, 1M-param MLP, 0.09 ms [U]) -> the adapter's state can also give a cheap
  "next action" proposal prior; our Qwen depths 18/24/30 of 36 already include ~3/4 depth (layer 27-30).
- H3. **Abstain / clarify as an explicit option** (When2Call: Qwen2.5 0.5-7B hallucinate tool calls 20-23%; training cuts it to 1.9% [V]).
  Without training: keep NONE/abstain in every candidate set (already a CLAUDE.md rule) and let LAYA's `needs_review` / `human_review` outputs drive it.
- H4. **ToolRL decomposed reward as a no-training reranker** (format + tool-name Jaccard + param-name overlap + exact values [V]) -> a
  deterministic scorer competitor/feature for the tool-call slot.
- H5. **Plan synthesis from existing trajectories + replanning** (Plan-and-Act) and **training-free periodic summarisation** (ReSum +4.5 [S]) for
  long loops; observation masking (D3) first because it is cheaper.
- H6. Harness fit matters as much as size (Aider Polyglot 45.6 vs 19.1 same model, tailored harness [U]) -> the loop itself is a first-class lever.
- H7. Warning: probes are shown to DETECT, not yet to improve action selection on small models [not found] -> our experiments must measure
  final-answer gain, not AUROC alone.

### After reading loop-A (loop engineering 2026)
- LA1. A loop spec = trigger, goal, **verification step**, **stopping rule**, memory (Macedo 2607.00038). Terminal states named in advance:
  decided / no-progress / exhausted / escalate; **exhausted is never read as a decision** (maps to abstain + report).
- LA2. Ground truth from the environment at each step (Anthropic 2024); a transcript-only evaluator can be fooled -> our stop gate is execution,
  LAYA only proposes.
- LA3. **Freeze the scorer during the loop** and keep it separate from the generator; otherwise the generator optimises the scorer (METR reward
  hacking, Skalse proxy theory). Tests read-only (ImpossibleBench).
- LA4. No-progress stop (same top choice twice / no new distinct candidate) and iteration caps cost nothing.
- LA5. Adaptive stopping on agreement (Adaptive-Consistency: up to 7.9x fewer samples) only where equivalence is checkable; for code,
  equivalence = same outputs on the same inputs (execution), not text.

### After reading loop-B (generate-rerank loops)
- LB1. **No paper shows a loop rescuing a near-chance scorer**; every gain starts from a verifier well above chance. -> LAYA earns a loop role only
  after it beats chance AND the probe on the loop's own target.
- LB2. Execution-based and execution-free verifiers each saturate (~42-43%); the hybrid wins (51%) (R2E-Gym). Execution-free verifiers are biased
  to style; tests have low distinguishability. -> combine, don't choose.
- LB3. **Goodhart under best-of-N with a proxy** (Khalaf et al. 2506.19248): true reward rises then falls with N -> cap N, prefer pessimistic
  selection, tune N on development data.
- LB4. Majority vote over the same head, head-margin gating, head-scored best-of-N on Fable pools would amplify the authorship shortcut. This was
  for the IMITATION target; on an execution target the shortcut does not exist (label = test result, not author).
- LB5. SWE-Replay: branch from earlier states instead of resampling (-17% cost).

### After reading loop-C (our Fable assets)
- LC1. Everything from the imitation phase is reusable as infrastructure (capture protocol, adapters, probe recipe, key rotation, sandbox), but the
  imitation TARGET is closed (authorship-dominated; options-only 68-76%).
- LC2. All measured LAYA numbers so far are on imitation pools; none on an execution target. The architecture below moves LAYA to execution-labelled
  targets, where LB4's shortcut cannot exist.

---

## Part 2. What our own measurements add (re-read of the whole conversation)

| measured (row-level file) | number | consequence for the design |
|---|---|---|
| Head swap, 303 pools OOF (`head-swap-run-20261012T222906Z`) | pretrained head 77.2%, random heads 83.4%, probe 85.8% | the pretrained head's weights add nothing on imitation; adapter does the work |
| Adapter-only vs adapter+head, VAL 46 (`laya-joint-v1`) | 76.1 vs 75.4 | training the head too adds nothing on imitation |
| VAL 53 (`val-eval-v1`) | adapter 79.2%, probe 94.3%, options-only 67.9% | the probe on the same states is the bar to beat |
| Context diagnostic, 303 pools (`context-diag-v1-20261015`) | LAYA ens 92.7% (in-sample), probe OOF 85.8%, options-only 75.6%, Qwen alone 36.6% (letter-A bias 0.64) | (a) Qwen judging its own options is weak (cf. F3); (b) on held-out-like fold 0 LAYA = probe = 84.6%; (c) LAYA-or-probe oracle 287 vs LAYA 281: little complementarity |
| Loop v1/v2 and context-diag loop rules | best rule -1 vs LAYA ens; C_avg +1 net on VAL [0, 0.067], equalled by the no-context control | combining two judges that read the same state gives ~nothing (LB2 says combine DIFFERENT signal types) |
| Outcome head (`outcome-head-v1-20261015`) | all arms ~0.50 AUROC (trained 0.503, unchanged 0.505, TF-IDF 0.490, probe 0.474) | predicting a whole-trajectory outcome from an early step is not learnable here; targets must be close to the outcome (candidate-level P(pass)) |
| Sandbox pilots v1/v2 | NO-GO / HOLD (0 qualified) | repo-level execution labels from SWE-rebench are expensive and fragile on this PC; function-level tests are the cheap label source |
| K1/K2, text-imitation v1 | no evidence beyond controls; gate FAIL | closed |
Common thread: every failure used an **imitation or far-outcome target**; none tested LAYA on **candidate-level correctness with execution labels**,
which is what the literature (D1, D2, F2, H1) says frozen states encode. Also: the adapter, not the head, carried every gain -> design the head's
role so that its pretrained OUTPUTS (action / outcome / needs_review / call_verdict) are used as named loop signals, and test the head against a
random-head adapter every time.

---

## Part 3. All combinations: axes, pruning, surviving architectures, scores

### 3.1 Axes (each architecture = one choice per axis)
| axis | options |
|---|---|
| G base model | G1 Qwen3-4B-Thinking (current, capture exists) · G2 Qwen3.5-9B Q4 (stronger, new adapter) · G3 cascade 4B -> 9B · G4 Qwen3.5-4B |
| T task level | T1 function-level with tests · T2 repo-level (SWE) · T3 tool-calling agent |
| C candidate source | C0 single · C1 k parallel samples · C2 k + repair rounds on real errors · C3 + distinguishing inputs (S*) · C4 evolutionary (Shinka) |
| F runtime feedback | F0 none · F1 parse/lint/type · F2 model-written tests (CodeT) · F3 given public tests · (F4 hidden tests = evaluation only, never runtime) |
| J judge in the LAYA slot | J0 none · J1 majority vote · J2 DeepConf log-prob · J3 linear probe · J4 adapter+LAYA · J5 J4 & J3 combined · J6 J4 + execution features (hybrid) |
| R LAYA role | R1 post-hoc verifier · R2 pre-generation budget (P(solve) from the prompt state) · R3 per-step monitor · R4 tool-call verdict · R5 repair-target picker |
| P loop policy | P0 fixed best-of-k · P1 adaptive stop on agreement · P2 difficulty-budgeted k · P3 retry with feedback (cap 3) · P4 named terminal states incl. abstain |
| Tr training | Tr0 none · Tr1 adapter on own executed outcomes (allowed) · Tr2 adapter + head · Tr3 Qwen LoRA self-training (ReST) · Tr4 one-time offline distillation · Tr5 RL (GRPO) |
| H hygiene | observation masking, ACI tools, constrained final packaging with abstain: always on, no trade-off -> not a free axis |

Raw count: 4 x 3 x 5 x 4 x 7 x 5 x 5 x 6 = **252,000** combinations (R and P are partly multi-select, so the true space is larger).

### 3.2 Pruning rules (each from evidence; a combination failing any rule is dropped)
1. **Must contain adapter + LAYA** (user): J in {J4, J5, J6} or R in {R2, R3, R4, R5}. Removes J0-J3-only designs (they stay as COMPETITORS).
2. **No runtime strong model; fits 6 GB / 15 GB** (user, F): drops 30B-A3B; G3 only as sequential load.
3. **Coverage first** (B1): T2 with a frozen untrained small model has ~2-7% coverage -> drop T2 unless Tr3/Tr4/Tr5.
4. **Feedback dominates judges** (B2, C, LB1, LB2): for code, F0 with any learned judge is dominated by the same design with F2/F3 at small
   extra cost -> drop F0 for T1.
5. **No self-repair without real feedback** (A, C): C2/C3 require F >= F1.
6. **Different signals, not more of the same** (our loop results, LB2): J5 (LAYA + probe on the same states) is dominated by the better of the
   two -> keep J5 only as a diagnostic; J6 (learned + execution) survives.
7. **Rules now in force:** Tr3/Tr5 (training Qwen) need the user's rule change -> kept as a separate "if allowed" tier; Tr4 needs one offline
   strong-model run + GO.
8. **Not recommended by compute-matched evidence:** debate, MoA, self-critique, MCTS at our scale, steering, merging -> never entered the axes.
After pruning: G (4) x T1/T3 x C1-C4 x F1-F3 x J4/J6 x R x P x Tr0-Tr2 collapses into **12 coherent families** (combinations that differ only in
a parameter such as k are one family).

### 3.3 Surviving families, scored
Scores 0-3: Ev = published evidence for the mechanism; Gain = expected final-answer gain; Cost = cheapness on our hardware/Kaggle (3 = cheapest);
Now = allowed under current rules without new approvals beyond the experiment; LAYA = LAYA has a real, testable job (not decoration).
| # | family (axis choices) | Ev | Gain | Cost | Now | LAYA | total |
|---|---|---|---|---|---|---|---|
| A1 | G1 T1 C1 F3/F2 J6 R1 P0 Tr1: best-of-k, tests first, LAYA tie-break among survivors | 3 | 2 | 3 | 3 | 2 | 13 |
| A2 | A1 + R2 P2: LAYA predicts P(solve) from the prompt, sets k (1/4/8/16) | 3 | 2 | 3 | 3 | 3 | 14 |
| **A3** | **A2 + C2 R5 P3 P4: repair rounds on real errors, LAYA picks which failing candidate to repair, named terminal states** | 3 | 3 | 2 | 3 | 3 | **14** |
| A4 | A3 + C3 distinguishing inputs (S*) | 3 | 3 | 2 | 3 | 2 | 13 |
| A5 | A3 + C4 evolutionary heavy mode for items still unsolved, LAYA pre-filters children | 1 | 2 | 1 | 3 | 3 | 10 |
| A6 | A3 on G2 Qwen3.5-9B (new adapter, head shared) | 3 | 3 | 2 | 2 | 3 | 13 |
| A7 | G3 cascade: 4B loop, escalate to 9B when LAYA P(solve) low or tests fail | 2 | 3 | 1 | 2 | 3 | 11 |
| A8 | T3 tool agent: LAYA call_verdict gate (R4) + ToolRL reward + constrained packaging + abstain | 2 | 2 | 2 | 2 | 3 | 11 |
| A9 | T1, F0 only, LAYA verifier alone (A3 in loop-B) | 1 | 1 | 3 | 3 | 3 | 11 (dominated by A1) |
| A10 | A3 + Tr3 ReST self-training of Qwen on verified own outputs | 3 | 3 | 1 | 0 | 2 | 9 (needs rule change) |
| A11 | A3 + Tr4 one-time offline distillation (NVARC pattern) | 3 | 3 | 1 | 1 | 2 | 10 (needs GO) |
| A12 | T2 Agentless + LAYA at localize/rerank, after A10/A11 | 2 | 2 | 0 | 0 | 3 | 7 |
A2 and A3 tie on the total; A3 is chosen because it is a strict superset (A2 is A3 with repair turned off), so one build measures both.

### 3.4 The chosen architecture (A3, "execution-grounded adaptive loop")
```
problem prompt
  -> [1] Qwen prefill -> hq state -> adapter -> LAYA: P(solve) + needs_review         (R2: budget)  competitors: probe, length baseline
  -> [2] k samples (k from P(solve); thinking budget at the measured knee)             (C1)
  -> [3] deterministic filter: parse / lint / type; public tests; model-written tests   (F1-F3, CodeT agreement)
  -> [4] score candidates: tests-passed & agreement FIRST, then LAYA P(correct) from ho   (J6 hybrid, R1)  competitors: probe, DeepConf, vote
  -> [5] any candidate passes all runtime tests?  yes -> decided (LAYA breaks ties)
                                                   no  -> repair round on the real error trace; LAYA picks which candidate to repair (R5)
  -> [6] stop: decided | no-progress (same top twice) | exhausted (cap 3 rounds) -> abstain + report; never a learned pick on exhaustion
  -> [7] every executed candidate's pass/fail -> versioned offline corpus -> adapter retrained on Kaggle (Tr1, grouped CV); live weights never change
always on: observation masking, constrained packaging at the end only, NONE/abstain option, protected tests never writable
```
Why this one: (a) every component has repeated evidence for small models (S*, CodeT, Self-Debugging, Snell, 2606.14530); (b) LAYA gets three
distinct, testable jobs (budget, tie-break, repair target), each with a named competitor, so it keeps a job only where it beats the probe;
(c) labels are execution results, so the authorship shortcut that sank imitation cannot exist; (d) Qwen stays frozen, nothing needs a rule change;
(e) it ports to Qwen3.5-9B by training one new adapter (A6), which is the user's "one adapter per Qwen model" goal.
Honest risk: on every test so far the probe matched or beat LAYA; the likely outcome is that the probe takes some slots. The design keeps LAYA's
slot open to whichever wins, and the pretrained head is tested against a random-head adapter each time.

### 3.5 Cheapest experiment that validates A3 (one Kaggle study, three readouts)
- Data: a public function-level set with tests (candidates: LiveCodeBench subset, MBPP+/HumanEval+). Download size and licence to be checked
  (needs the user's yes if > 1 GB; these are small).
- Run: Qwen3-4B-Thinking frozen, k = 8 per problem, states captured with the existing protocol (hq + ho per candidate), every candidate executed
  in a Kaggle subprocess with timeouts (hidden tests only for evaluation).
- Readout 1 (verifier gap): pass@1, pass@8, selected@8 for vote / DeepConf / probe / adapter+LAYA / random-head adapter / tests-only / hybrid.
- Readout 2 (budget): accuracy vs samples when k is set by P(solve) (LAYA vs probe vs length baseline vs fixed k).
- Readout 3 (repair): one repair round on failures; LAYA-picked vs random vs DeepConf-picked repair target.
- Gates (pre-registered before the run): LAYA keeps a slot only if it beats the probe and DeepConf with a problem-clustered CI above 0; the
  pretrained head counts only if it beats the random-head adapter.

### 3.6 Checklist
| step | status | evidence |
|---|---|---|
| Read A, write notes | DONE | Part 1 A1-A5 |
| Read B, write | DONE | B1-B5 |
| Read C, write | DONE | C1-C7 |
| Read D, write | DONE | D1-D7 |
| Read E, write | DONE | E1-E8 |
| Read F, write | DONE | F1-F6 |
| Read G, write | DONE | G1-G8 |
| Read H, write | DONE | H1-H7 |
| Read loop A/B/C, write | DONE | LA1-LA5, LB1-LB5, LC1-LC2 |
| Re-read our measured results | DONE | Part 2 table |
| Axes and count of combinations | DONE | 3.1 (252,000 raw) |
| Pruning rules with evidence | DONE | 3.2 (8 rules -> 12 families) |
| Score families, choose | DONE | 3.3 -> A3 |
| Validation experiment | DESIGNED, NOT RUN | 3.5 |
| Dataset choice + download | WAITING ON USER | size/licence check first |
| Pre-registration file | NOT DONE | written only after GO |
| Qwen training tiers (A10-A12) | WAITING ON USER | need the "Qwen never trained" rule changed |
