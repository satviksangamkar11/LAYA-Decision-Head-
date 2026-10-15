# B. Generate-rerank-loop methods for action and code selection (web research, 2026-10-15)

Scope: arXiv abstracts, search excerpts, project pages. No models or datasets downloaded.
Tags: [abs] = abstract fetched and checked; [snippet] = seen only in search excerpt, full text not read;
[unverified] = aggregator, blog, or secondary source. Numbers below are the papers' own claims unless stated.

## 1. Methods, gains, scorer needs, failure modes

**Best-of-N with execution or trained verifiers (SWE)**
- CodeMonkeys, arXiv 2501.14723 [abs]: 57.4% SWE-bench Verified at about $2,300. Candidates are picked by voting over model-written tests, then a final selection trajectory. Ensemble selection over existing top submissions: 66.2% [snippet]. Stanford's page lists 57.7%, so the figure needs checking against the PDF. Scorer: model-generated tests, not a trained verifier. Cost: many multi-turn trajectories. Failure: the abstract names no limitation; selection is bounded by test quality.
- SWE-Gym, arXiv 2412.21139 [snippet]: outcome verifier trained on trajectories with unit-test labels; combined with a fine-tuned agent it reaches 32.0% SWE-bench Verified. N and aggregation are not in the excerpt.
- R2E-Gym, arXiv 2504.07164 [snippet]: execution-based and execution-free verifiers each saturate near 42-43%. A hybrid reaches 51% on SWE-bench Verified. Authors' failure notes: test-based verifiers have low distinguishability; execution-free verifiers are biased toward stylistic features. This is the failure most relevant to us.
- SWE-World, arXiv 2602.03419 [snippet]: generative verifier SWR-32B goes 55.0 (K=1) to 68.2 (TTS@8). R2E-Gym-Verifier-14B levels off near TTS@5 (59.4).
- From SWE-ZERO to SWE-HERO, arXiv 2604.01496 [snippet]: Best@32 of 52.9-64.6% with open generative verifiers. Gap to the Pass@K ceiling is 15% at K=16 and widens with K.
- Agentic Rubrics, arXiv 2601.04171 [snippet]: repo-grounded rubric scoring without test execution; Best@16 40.6% (Qwen3-32B) vs 37.1% for the best non-agentic baseline.
- Agentless (arXiv 2407.01489) [unverified, README only]: rerank by regression tests plus generated reproduction tests run on the original codebase.

**Coverage versus selection (the anchor result)**
- Large Language Monkeys, arXiv 2407.21787 [snippet]: coverage grows log-linearly with samples. SWE-bench Lite: 15.9% at 1 sample to 56% at 250 [snippet]. MATH, Llama-3-8B: coverage 79.8% to 95.3%, while majority vote and reward-model selection only move 38.7% to 39.8% [snippet, v1]. A secondary wiki gives different figures; unreconciled.
- Takeaway: more samples only help if the selector can identify the correct one.

**Self-consistency and majority voting over candidates**
- Universal Self-Consistency, arXiv 2311.17311 [snippet]: the LLM picks the most consistent of N responses; matches execution-based voting on code without running anything. Assumes the model can judge consistency.
- Adaptive stopping (sample until confident):
  - Adaptive-Consistency, arXiv 2305.11860: about 3.3x fewer samples on average [snippet].
  - CGES, arXiv 2511.02603 (NeurIPS 2025): 69.4% fewer calls, accuracy within 0.06 pp of self-consistency [snippet].
  - RASC, arXiv 2408.17017 (NAACL 2025): about 70% fewer samples [snippet].
  - MMC, arXiv 2510.17472: anytime-valid stopping rule [snippet].
- Failure: all assume roughly independent votes over a discrete answer space (answer matching). Confidence from a miscalibrated scorer stops early on confident wrong answers. I found no evaluation on action or code pools.

**Permutation-averaged scoring (position bias)**
- Found in the Middle, arXiv 2310.07712 [abs]: shuffle the candidate list, keep instructions fixed, aggregate the rankings. Gains of 7-18% (GPT-3.5) and 8-16% (LLaMA-2 70B) on five listwise datasets. The abstract does not state the number of permutations.
- Judging the Judges, arXiv 2406.07791 [snippet]: position bias is strongly affected by the quality gap between candidates. Near-tied options are where it bites hardest.
- Non-transitivity in LLM-as-a-judge, arXiv 2502.14074 (ICML 2025) [snippet]: round-robin and Swiss tournaments with Bradley-Terry. Position switching beats random assignment for stronger judges; position bias is one cause of non-transitivity, not the only one.
- Balanced permutations for rubric scoring, arXiv 2602.02219 (Feb 2026) [snippet]: aggregating over balanced permutations exposes latent position bias and improves agreement with humans.
- Architectural alternative, arXiv 2604.27599 [snippet]: permutation-invariant listwise reranking (recommendation domain, not code).

**Tree search and value-guided search**
- SWE-Search, arXiv 2410.20285 [abs]: MCTS plus a hybrid LLM value function and a discriminator debate. About 23% relative gain on SWE-bench, averaged over five models, versus agents without MCTS. Limitation [snippet]: the value function sometimes misses critical decision points and underrates good strategies. The discriminator beat the value function for every model except GPT-4o-mini.
- AgentPRM, arXiv 2511.08325 [snippet]: step-level promise and progress rewards; Best-of-N comparisons reported; benchmarks not confirmed.
- WebArbiter, arXiv 2601.21872 [snippet]: says outcome rewards cannot steer inference-time search; LLM-as-judge step rewards are costly and hallucination-prone.
- SWE-Replay, arXiv 2601.22129 [snippet]: branch from earlier trajectories instead of resampling; up to 17.4% lower cost at equal performance.

**Goodhart and flattening**
- Khalaf et al., arXiv 2506.19248 (NeurIPS 2025 spotlight) [abs]: under Best-of-N with a proxy reward, true reward rises, then falls. Claimed unavoidable for a broad class of inference-time methods. Remedy: HedgeTune, which tunes the inference parameter.
- Yu, Wu, Block, arXiv 2604.04648 (ICLR 2026) [snippet]: standard Best-of-N degrades as N grows on GSM8K with some reward models; a pessimistic variant keeps improving.
- Gao, Schulman, Hilton (2023), overoptimization scaling [snippet via secondary]: the same effect appears in Best-of-N.
- Flattening in practice: R2E-Gym single verifiers saturate near 42-43%; SWE-World's R2E-14B verifier plateaus near TTS@5.
- Verification-horizon argument (arXiv 2606.26300 per aggregator) [unverified]: no fixed reward stays effective as the generator improves.

## 2. What scorer quality the gains needed

- Gains in these papers come from one of three sources: execution or tests (CodeMonkeys, R2E-Gym, Agentless); trained outcome verifiers on thousands of trajectories (SWE-Gym, SWE-World); or large LLM judges with logprob-weighted scores (LLM-as-a-Verifier, arXiv 2607.05391: 78.2% SWE-bench Verified, self-reported [snippet]; a blog's 88.0% best-of-5 on Terminal-Bench 2.1 is [unverified]).
- None of the papers I found shows a loop rescuing a scorer near chance. Every positive result starts from a verifier well above chance.

## 3. Our case: 3-5 options, head top-1 about 33% (chance about 30%), options-only text baseline 75%

Premise: the target is agreement with Fable's recorded pick, not correctness. The 75% options-only result (commit 2de1bcc) shows the recorded option is identifiable from text alone. So a loop can raise agreement by exploiting style without making better decisions. Every loop must beat the options-only baseline, not chance.

**A. Plausible without new training (cheap; expected gain small, because they add no information)**
1. Permutation averaging of the head's scores over candidate orders (Found in the Middle). Helps only if the head's top-1 depends on slot position. Diagnostic first: if agreement changes under shuffling, the head reads position. If the head is at chance, averaging keeps it at chance.
2. Balanced or cyclic permutations as a readout diagnostic. Cheap if the head reads precomputed per-option states that do not depend on order.
3. Confidence-gated re-scoring with a budget (adaptive stopping). Only sound if head confidence is calibrated (measure ECE on VAL first). At 33% agreement the gate fires arbitrarily.
4. Style-normalize options before scoring (same rewrite or format applied to all options) so text style cannot separate the recorded option. This attacks the confound directly. It needs new generation, so it needs the user's GO (CLAUDE.md: no further Qwen generation without approval).

**B. Require a better scorer first**
1. Best-of-N or rerank-then-refine: needs a verifier well above chance. All reported gains above rest on that.
2. Execution-verified scoring (sandbox replay, mutation oracle): the only route with signal independent of authorship. It yields execution-verified preference labels (kind 2), not imitation labels. Report separately.
3. Pairwise or Swiss tournaments with Bradley-Terry: needs a judge whose position bias is small relative to its discrimination. Near-tied options will be unstable.
4. Self-consistency over sampled LLM judgments (USC style): works only if the judge is well above chance on this task.
5. MCTS or value search over actions: the value model must be good (SWE-Search's own value was weak). No counterfactual outcomes exist for unexecuted branches; do not fabricate them.

**C. Would amplify the shortcut**
1. Majority vote or self-consistency over votes of the same head on the same pool. Votes are a deterministic function of style; more votes give a more confident style-based pick. Agreement rises for the wrong reason.
2. Best-of-N over extra generated candidates scored by the head. Generated candidates in Fable-like style win on style. This is Goodhart with the head as proxy (Khalaf et al.).
3. Confidence-gated regeneration where the gate is head margin. Margin correlates with style, so the loop keeps stylistically matched options and regenerates the rest.
4. Repeated selection on VAL. Each loop variant tried on VAL is a selection step; use grouped CV for development and VAL once per variant.

**Suggested order**
- Step 0: report options-only (75%) and probe baselines next to every loop result. Chance +10 is only a warning (CLAUDE.md).
- Step 1 (no training): permutation diagnostic, then averaging if the diagnostic shows position dependence. Report paired-bootstrap deltas.
- Step 2 (needs approval): style-normalized options, then re-measure the head and options-only baseline on the same pools.
- Do not run head-vote majority, head-margin gating, or head-scored Best-of-N before the scorer clears chance by a margin.

## 4. Unverified or open
- CodeMonkeys page figure: 57.4% (abstract) vs 57.7% (Stanford page).
- Large Language Monkeys verifier-gap numbers: v1 abstract vs a wiki's different figures.
- SWE-Gym best-of-N details, R2E-Gym 51% setting, LLM-as-a-Verifier 78.2% and 88.0%: all from excerpts or self-reports, full text not read.
- No 2026 paper on "uncertainty-gated regeneration" for code agents was found; searches returned unrelated papers.
- Agentless (2407.01489) and the verification-horizon ID were not read directly.
