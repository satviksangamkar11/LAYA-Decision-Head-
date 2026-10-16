# Research findings: making small models answer as well as possible (2026-10-15)

All findings from the eight research agents (Haiku 5.5, web research only; no downloads, no code run), plus the synthesis.
Evidence tags used inside each part: **[ABS]/[abs]/[PRIMARY]/[V]** = read from the abstract or primary page; **[SNIP]/[snip]/[S]** = search snippet
or secondary summary, not checked against the paper; **[UNV]/[U]/[unverified]** = unverified or conflicting; **[MC]** = model card. Numbers are the
papers' own (self-reported unless stated) and use different scaffolds, so they are not directly comparable across rows. Our own measured results live
in `results/raw/` (cited in the synthesis).

## Contents
- Part 0: Synthesis and ranked options (adapter + LAYA + loop at the centre)
- Part A: Methods for making small models better at coding-agent tasks
- Part B: First-attempt vs best-of-k (coverage) gap for small models
- Part C: Test-time compute and search
- Part D: Scaffolds, memory, prompt optimisation, steering, probes
- Part E: Self-improvement without a strong model at runtime
- Part F: Maximum capacity on 6 GB / 15 GB RAM; uncertainty, routing
- Part G: Unconventional and frontier ideas
- Part H: General agent capability beyond coding

## Key findings at a glance
1. Untrained small models solve almost nothing on repository tasks (1.8-7% SWE-bench Verified, 7-32B zero-shot); trained 4B models reach 27.6-40.7%.
   Function-level coding is where a frozen 4B already succeeds (Qwen3-4B thinking, LiveCodeBench v5 54.2).
2. Selection checked by execution beats voting: S* lets a 3B model beat GPT-4o-mini on LiveCodeBench; CodeT +18.8 points; verifiers recover about
   50-62% of the first-try vs best-of-k gap at 32B; at k=5 public tests recover 69%, generated tests 23%, an LLM judge 30%.
3. Probes on frozen hidden states predict correctness (AUC up to 0.83; 0.91 on Qwen3-4B, unverified) and detect wrong tool-call values.
4. Repair helps only with real feedback (tests); open small models self-repair poorly (4.4% vs 28.9%, snippet).
5. Spend compute by difficulty; confidence early stop (DeepConf) saves up to 84.7% of tokens (math).
6. No small learned steering component (PRM, value head, reranker) beat simple baselines on SWE-bench; SWE-Shepherd PRM 51% vs 57% baseline.
7. Debate / mixture-of-agents do not beat self-consistency at equal compute; more thinking can hurt.
8. Training the small model is the largest measured lever (about 500 trajectories multiplied 7-32B success several times; ToolRL 1.5B BFCL 19.4 -> 46.2;
   When2Call abstain training cut tool hallucination 16% -> 1.9% at 4B; ReST/STaR self-training on verified attempts).
9. Qwen3.5-9B at 4-bit fits 6 GB (LiveCodeBench v6 65.6, official card); RAM (15.3 GB) is the binding limit.


---

<!-- source: results/raw/research-20261015-smallmodel/Z_SYNTHESIS.md -->
# Part 0. Synthesis: making small models answer as well as possible, with adapter + LAYA + loop engineering (2026-10-15)

Inputs: A_methods, B_coverage_gap, C_test_time_compute, D_scaffolds_memory_steering, E_self_improvement, F_capacity_routing_uncertainty,
G_unconventional, H_general_agents (this folder; Haiku 5.5 web research, mostly abstracts and snippets: every number below is the papers' own and
is marked in the source files as [abs]/[snip]/[unverified]). Our own measured results are cited with their results/raw folders.

## 1. Goal and judging criteria (user, 2026-10-15)
Best possible final answers from the small local model (Qwen family); the system includes the adapter (Qwen states -> LAYA), the pretrained LAYA
head and loop engineering; no strong model at runtime (offline use allowed). Judged by: measured final-answer quality (tests pass / correct answer),
runtime constraint, our hardware (6 GB GPU, 15 GB RAM, free Kaggle), evidence strength, and LAYA kept only where it beats simpler baselines.

## 2. What the evidence says
1. Untrained small models solve almost nothing at repository level (1.8-7% SWE-bench Verified, 7-32B zero-shot) [B]; trained 4B models reach
   27.6-40.7% [B]. Function-level coding is where a frozen 4B already succeeds (Qwen3-4B thinking: LiveCodeBench v5 54.2) [B].
2. Selection with real execution beats voting: S* lets a 3B model beat GPT-4o-mini on LiveCodeBench [C]; CodeT +18.8 points [C]; verifiers recover
   about 50-62% of the pass@1 -> pass@k gap at 32B [B]; public tests recover 69% at k=5, generated tests 23%, an LLM judge 30% [B].
3. Hidden-state probes predict correctness: AUC up to 0.83 (2607.05188) [F]; 0.91 on Qwen3-4B vs 0.75 length baseline (2606.14530, unverified) [D];
   probes on 18 tool-calling LLMs detect wrong-value calls (NeurIPS 2026 workshop) [H]; small probes reported to match PRMs up to 810x larger [H, unverified].
4. Repair helps only with real feedback (Self-Debugging with tests) [C]; open small models self-repair poorly (4.4% vs 28.9%) [A, snippet].
5. Compute should follow difficulty (Snell et al.; Inference Scaling Laws) [C]; confidence early stop (DeepConf) saves up to 84.7% tokens on math [C].
6. No small learned steering component (PRM, value head, reranker) beat simple baselines on SWE-bench [A]; SWE-Shepherd PRM 51% vs 57% baseline [A].
   Matches our own LAYA results (head swap, probe > head).
7. Debate / mixture-of-agents do not beat self-consistency at equal compute; agent-level test-time scaling unreliable; more thinking can hurt [C].
8. Training the small model is the largest measured lever: ~500 trajectories multiplied 7-32B success several times [B]; ToolRL 1.5B BFCL 19.4 -> 46.2 [H];
   When2Call abstain training cut tool hallucination 16% -> 1.9% at 4B [H]; self-training on verified own attempts (ReST/STaR) [E].
9. A stronger small base fits: Qwen3.5-9B at 4-bit in 6 GB (LiveCodeBench v6 65.6, official card) [F]; RAM (15.3 GB) is the binding limit [F].

## 3. Ranked options for adapter + LAYA + loop (Qwen frozen)
| rank | option | role of adapter + LAYA | evidence | cost | cheapest experiment |
|---|---|---|---|---|---|
| 1 | Correctness verifier over the small model's OWN candidates | score each candidate's P(correct) from Qwen states (LAYA native `outcome` question) | items 2, 3 | low-medium (Kaggle sampling + capture) | verifier-gap test (section 4) |
| 2 | Compute controller in the loop | decide stop / sample more / retry with feedback / abstain (LAYA native `action` question) | items 4, 5 | low once 1 exists | accuracy-vs-compute curve on the same data |
| 3 | Tool-call verdict for general agents | LAYA native `tool_call.call_verdict` (execute / fix_args / wrong_tool / abstain), labels from schema validation and real execution errors | item 3 (tool-call probes), item 8 (When2Call) | medium | BFCL-style subset: small model proposes calls, executor labels them |
| 4 | Portability: new adapter per Qwen model | train only a new adapter (head shared) for Qwen3.5-9B | item 9 | medium | repeat option 1 with Qwen3.5-9B |
| 5 | Scaffold hygiene around the loop | none (deterministic): observation masking (~half cost, same solve rate), Agentless localize -> sample -> validate, constrained decoding for syntax only with an abstain option | D | low | add to whichever loop is built |
Mandatory competitors for LAYA's slot in every experiment: linear probe on the same states, Qwen's own log-probability confidence (DeepConf-style),
self-consistency (majority vote), random-head adapter, and the oracle (best of k). LAYA keeps the slot only if it beats them at equal compute.

## 4. First experiment: verifier-gap test (function-level coding with tests)
- Problems: a public function-level set with tests (e.g. MBPP+/HumanEval+/LiveCodeBench subset; licence and size to check; small download).
- Qwen3-4B-Thinking-2507 (current backbone), k = 8 attempts per problem on Kaggle; Qwen states captured with the existing protocol.
- Every attempt executed against the tests (sandboxed subprocess with timeouts; location to decide) -> pass/fail label.
- Train adapter (head frozen) / random-head adapter / linear probe; grouped CV by problem; DeepConf-style and self-consistency baselines.
- Report: pass@1, pass@8 (ceiling), selected@8 per judge, share of the gap recovered, calibration; then the adaptive loop's accuracy vs compute.
- Gate (to pre-register): LAYA counts only if it recovers more of the gap than the probe and DeepConf with a trajectory/problem-clustered CI above 0.

## 5. Options that need the user's rule change (training Qwen itself)
Largest measured levers for answer quality: self-training on verified own attempts (ReST/STaR) [E], small offline distillation (~500-1k traces) [B, E],
execution-preference training (PLUM, CodeDPO) [G], RL with test rewards (ToolRL, SWE-Master) [H, A]. These conflict with "Qwen is never trained".

## 6. Watch list / not recommended
Watch: evolutionary search with an execution evaluator (ShinkaEvolve) as a heavy loop with LAYA as pre-filter; layer repetition at inference;
repository test-time training; hypernetwork adapters (Text-to-LoRA) [G]. Not recommended now: debate / mixture-of-agents, self-critique without
execution, unbounded thinking, steering for capability, model merging, PRMs without evidence [C, D, A].

## 7. Honest risks
The probe may again match LAYA (then the probe takes the slot); published verifier gains are for 32B-class models, not 4B; labels from tests reward
passing the given tests, not full correctness; executing model-written code needs isolation.

---

<!-- source: results/raw/research-20261015-smallmodel/A_methods.md -->
# Part A. Methods for making small open models better at coding-agent tasks (no strong model at inference)

Date of research: 2026-10-15. Web research only. No downloads, no code run.
Scope: 2023-2026, newest first within each section. Model sizes are roughly 1B-32B.

Status tags:
- [ABS] arXiv abstract fetched and read this session.
- [SNIP] search snippet or secondary summary only. Numbers not checked against the paper body.
- [MC] model card or official project page seen only in search results.
- [UNV] unverified. Conflicting or missing source.

Caution: many SWE-bench numbers below use different scaffolds (OpenHands, mini-SWE-Agent, SWE-agent, R2E-Gym) and turn limits. They are not directly comparable across rows.

---

## 1. Execution feedback at inference (self-debugging, Reflexion, model-written reproduction)

**Self-debugging (Chen et al., ICLR 2024)** [SNIP]
- Source: https://arxiv.org/abs/2304.05128 , https://iclr.cc/virtual/2024/poster/18880
- Feedback: unit-test results and execution traces. Also code explanation when no tests exist.
- Trains the model? No. Inference-time loop only.
- Size: open-model results weak. Later paper reports StarCoder-15B refined only 4.43% of wrong MBPP solutions vs 28.90% for GPT-3.5-Turbo [SNIP, source cited in the search result; not checked].
- Headline: up to +12% over baseline on TransCoder and MBPP where unit tests exist [SNIP]. Gains smaller on Spider with no tests (+2-3%).
- Cost: low (extra model turns). Limit: needs tests or an execution oracle; weak self-repair in small open models.

**Reflexion (Shinn et al., NeurIPS 2023)** [SNIP]
- Source: https://arxiv.org/abs/2303.11366
- Feedback: scalar or binary environment feedback converted to verbal reflection, stored in episodic memory.
- Trains the model? No (weights unchanged).
- Headline: HumanEval pass@1 91% (GPT-4 based, per the paper summary) [SNIP]. A third-party summary gives different figures (67-91%); the primary PDF was not checked [UNV for the exact number].
- Limit: built and tested on frontier models; no small-model SWE-bench numbers found.

**Agentless (Xia et al., 2024; FSE 2025)** [ABS]
- Source: https://arxiv.org/abs/2407.01489
- Feedback: test/patch validation stage. Abstract says the pipeline is localization, repair, patch validation, but does not say whether validation uses reproduction tests. NOT FOUND in abstract.
- Trains the model? No. Fixed three-phase pipeline, no agent loop.
- Headline: SWE-bench Lite 32.00% (96 correct), cost $0.70 per issue in the v1 abstract [ABS]. FSE version lists 32.67% / 98 fixes / $0.68 [SNIP].
- Limit: models used were frontier-class. No small-model result found.

**CodeMonkeys (Ehrlich et al., 2025; arXiv 2501.14723)** [SNIP]
- Source: https://arxiv.org/abs/2501.14723 , https://scalingintelligence.stanford.edu/pubs/codemonkeys
- Feedback: model writes a test script alongside each edit, runs it, and iterates (serial scaling). Parallel samples per problem (parallel scaling). Selection by voting over model-written tests plus a final selection trajectory.
- Trains the model? No.
- Headline: 57.4% on SWE-bench Verified at about $2,300 total budget (arXiv abstract); project page says 57.7% [SNIP]. Ensemble selection over existing submissions reaches 66.2% [SNIP].
- Limit: budget is large; models are frontier-class.

Reproduction-script-first loops in general: this is the CodeMonkeys pattern. No small-model (under 10B) version with numbers was found [NOT FOUND].

---

## 2. Repeated sampling + selection (pass@k vs selection gap; which verifiers close it)

**Large Language Monkeys (Brown et al., 2024; arXiv 2407.21787 v3 Dec 2024)** [ABS via search summary]
- Source: https://arxiv.org/abs/2407.21787 , https://scalingintelligence.stanford.edu/blogs/monkeys
- Feedback: automatic verifiers (unit tests for code, Lean for math) to pick correct samples. Majority vote and reward models for unverifiable tasks.
- Trains the model? No.
- Headline: SWE-bench Lite, DeepSeek-Coder-V2-Instruct, 15.9% at 1 sample -> 56% at 250 samples (prior SOTA 43%) [SNIP].
- Gap: coverage (pass@k) rises log-linearly; without a strong verifier, majority voting and reward models plateau. Abstract says "beyond several hundred samples"; the blog says before 100. Exact threshold NOT CONFIRMED [UNV].
- Small-model claim: 5 samples of a cheaper DeepSeek model beats 1 sample of GPT-4o/Claude 3.5 Sonnet on cost. This is the authors' own cost argument [SNIP].

**CodeT (Chen et al., ICLR 2023; arXiv 2207.10397)** [SNIP]
- Source: https://arxiv.org/abs/2207.10397
- Verifier: the same LLM writes tests; candidates scored by dual agreement with generated tests and other candidates (RANSAC-like).
- Trains the model? No.
- Headline: HumanEval pass@1 65.8% [SNIP]. Not SWE-bench. Works only for executable code.

**Majority voting variants (2025-2026)** [SNIP]
- Launer et al., "Majority Voting for Code Generation," ICLR 2026: functional consensus (cluster by outputs on test inputs). Source: https://arxiv.org/pdf/2604.15618 , https://iclr.cc/virtual/2026/10020237
- "Majority of the Bests," arXiv 2511.18630: reward-model failure analysis. Source: https://arxiv.org/html/2511.18630v1
- "Self-Rewarding RL," arXiv 2506.08745: majority-vote reward causes collapse during training. Source: https://arxiv.org/pdf/2506.08745
- "Shrinking the Generation-Verification Gap," arXiv 2506.18203: naive aggregation of weak verifiers is insufficient. Source: https://arxiv.org/html/2506.18203v1

**Execution-based vs execution-free verifiers (SWE-bench)**
- R2E-Gym hybrid verifier (Jain et al., 2025; arXiv 2504.07164) [ABS]
  - Abstract: execution-based and execution-free verifiers each saturate around 42-43%; combining them reaches 51% on SWE-bench Verified (32B model, test-time scaling).
  - Execution-free verifier alone is biased toward stylistic features; test-based verifier has low distinguishability [SNIP from paper summary].
  - Source: https://arxiv.org/abs/2504.07164
- SWE-Gym verifier (Pan et al., 2024; arXiv 2412.21139) [ABS]
  - Verifiers trained on agent trajectories from SWE-Gym (32B Qwen2.5-Coder). Abstract gives only the combined system: 32.0% SWE-bench Verified, 26.0% Lite. No verifier-only number in abstract.
  - Source: https://arxiv.org/abs/2412.21139
- SWE-RM (Shum, Hui, Lin, He et al., arXiv 2512.21919; ICLR 2026) [SNIP]
  - Execution-free generative reward model, 30B total / 3B active MoE. Trained on trajectories from SWE-Gym, SWE-rebench, SWE-smith, R2E-Gym (fail-to-pass labels).
  - Headline: Qwen3-Coder-Flash 51.6 -> 62.0 and Qwen3-Coder-Max 67.0 -> 74.6 with test-time scaling. Beats execution-based RL counterpart by 3 points in RL training.
  - Source: https://arxiv.org/abs/2512.21919 (v1 seen; later revisions unchecked)
  - Size note: 3B active, but 30B total; not a small model in memory terms.

**Gap closure for SMALL models:** no paper found that measures pass@k vs selected-pass@1 gap with a verifier at 1B-8B on SWE-bench [NOT FOUND]. The strongest evidence is for execution-based selection with generated tests (CodeMonkeys, CodeT) and for trained reward models at 30B+.

---

## 3. Training small models with verifiable rewards in executable environments

**SWE-Gym (Pan et al., arXiv 2412.21139; v2 Jun 2025)** [ABS]
- 2,438 Python task instances with executable environments, unit tests, NL task.
- Fine-tuned agents: up to +19% absolute resolve rate on SWE-bench Verified and Lite [ABS]. Models: Qwen2.5-Coder 7B, 14B, 32B (OpenHands agents) [MC via HF org page].
- Source: https://arxiv.org/abs/2412.21139 , https://huggingface.co/SWE-Gym

**R2E-Gym / SYNGEN (Jain et al., arXiv 2504.07164)** [ABS]
- More than 8.7K tasks (abstract); 8.1K tasks in NeurIPS listing [SNIP]. Environments built from commits by test generation and back-translation.
- 32B pass@1 34.4% on SWE-bench Verified (abstract). Hybrid verifier 51%.
- Source: https://arxiv.org/abs/2504.07164 , https://sky.cs.berkeley.edu/project/r2e-gym/

**SWE-smith (Yang et al., arXiv 2504.21798; NeurIPS 2025 spotlight)** [ABS]
- 50k instances from 128 repos. Pipeline synthesizes bugs that break tests in an existing environment.
- SWE-agent-LM-32B (fine-tuned Qwen2.5-Coder-32B on SWE-smith trajectories): 40.2% Pass@1 on SWE-bench Verified (abstract). Scaling curve from about 14.3% at 100 trajectories to 40.2% at about 6.2k [SNIP].
- Source: https://arxiv.org/abs/2504.21798 , https://swesmith.com

**SWE-RL (Wei et al., Meta, arXiv 2502.18449; NeurIPS 2025)** [SNIP]
- Rule-based reward: similarity between generated patch and oracle patch; GRPO. Data from GitHub PR evolution.
- Llama3-SWE-RL-70B: 41.0% on SWE-bench Verified (self-reported). Not a small model; no 7-14B number found.
- Source: https://arxiv.org/abs/2502.18449

**DeepSWE-Preview (Agentica/Together AI, Qwen3-32B) [MC/SNIP]**
- RL only, from Qwen3-32B. Pass@1 42.2% on SWE-bench Verified (avg over 16 runs); 59% with test-time scaling. Training from R2E-Gym tasks (about 4.5K, filtered to exclude SWE-bench-Verified repos). Their stated gain: 23 -> 42% in 200 RL steps [MC].
- Self-reported. No independent reproduction found [UNV].
- Source: https://huggingface.co/agentica-org/DeepSWE-Preview , https://together.ai/blog/deepswe

**Skywork-SWE (arXiv 2506.19290, Jun 2025)** [SNIP]
- Qwen2.5-Coder-32B SFT on over 8,000 validated trajectories (about 10,169 instances total): 38.0% pass@1 on SWE-bench Verified; 47.0% with test-time scaling [SNIP]. Weights: Skywork/Skywork-SWE-32B [MC].
- Source: https://arxiv.org/abs/2506.19290

**SkyRL-Agent-14B-v0 (NovaSky, Qwen3-14B thinking) [SNIP, blog]**
- 18.0% -> 21.6% on SWE-bench Verified; 7B: 11.0% -> 14.6%; about 300 training samples; reported about 20 h on 8xH200 [SNIP]. Single run; not independently verified.
- Source: https://sky.cs.berkeley.edu/project/skyrl/ , https://arxiv.org/abs/2511.16108

**Small models (4B-8B) with RL/SFT, 2026 [SNIP]**
- SWE-AGILE (arXiv 2604.11716): Qwen3-8B 15.8% -> 21.45% by SFT on 2.2k trajectories; 24.05% after RLVR; 14B SFT-only 30.06% [SNIP].
- SWE-Master (arXiv 2602.03411): Qwen3-4B-2507 27.6% -> 33.4% with RL [SNIP].
- SWE-TRACE (arXiv 2604.14820): 4B SFT+RL 38.9%, 40.7% with test-time scaling, under mini-SWE-Agent [SNIP]. Scaffold differs, so not comparable.
- DAgger-style training (arXiv 2605.12913): Qwen3-8B 29.8% on SWE-bench Verified; GRPO baseline on same model 8.2% [SNIP]. Suggests plain RL on small bases can underperform without a good recipe.
- Limit: these are small-sample, single-run results [UNV for full detail].

**Other 2024-2025 SFT/RL recipes**
- Kimi-Dev-72B (arXiv 2509.23045, ICLR 2026): Agentless-style training, RL on code edits only; 60.4% SWE-bench Verified (workflow). Not small. [SNIP]
- SWE-Dev-7B / 32B (arXiv 2506.07636, ACL Findings 2025) [ABS]: 23.4% (7B) and 36.6% (32B) on SWE-bench Verified; test-synthesis verifier. No 14B model found (the query asked for 14B) [UNV].

---

## 4. Distillation: strong-model trajectories used once for offline SFT

- SWE-smith -> SWE-agent-LM-32B [ABS, see section 3]. Trajectories generated offline; model card says SFT on 5k trajectories from SWE-agent + Claude 3.7 Sonnet [MC, featherless/HF snippet]. Score 40.2%.
- Nebius SWE-rebench (arXiv 2505.20411) [SNIP]: pipeline for decontaminated task collection; 21,000+ verifiable tasks; datasets nebius/SWE-rebench, nebius/SWE-bench-extra (6,411 PR pairs), nebius/SWE-agent-trajectories (80,036 trajectories). Agent that produced the trajectories not confirmed [UNV].
  - Source: https://arxiv.org/abs/2505.20411 , https://nebius.com/blog/posts/swe-rebench-dataset
- OpenHands LM 32B: scores and training details NOT FOUND in this search. Only a reference in SWE-rebench [NOT FOUND].
- Devstral Small (24B; Mistral + All Hands AI, May 2025; Apache-2.0) [SNIP]: 46.8% SWE-bench Verified; Devstral Small 1.1 53.6% (July 2025) [SNIP]. Devstral paper (arXiv 2509.25193) says training data came from agent runs in SWE-Gym environments with OpenHands CodeAct, filtered by unit tests [SNIP]. Conflicting figures for Devstral Small 2 (68.0% vs "50%+"), so not cited [UNV].
  - Source: https://arxiv.org/html/2509.25193v1 , https://the-decoder.com/mistral-launches-devstral-small-24b-a-new-open-source-llm-for-coding/
- Qwen3-Coder-30B-A3B official SWE-bench Verified score: NOT FOUND from official source. Sources conflict: 51.6, 49.2 (own harness), 60.4 [UNV]. Do not quote without the Hugging Face model card.

---

## 5. Small learned components steering a frozen small model (PRMs, value heads, rerankers, routers)

- SWE-Shepherd (arXiv 2604.10493) [SNIP]: trained PRM scores and selects actions for mini-SWE-Agent. Test on 100 SWE-bench Verified tasks, gpt-5-mini (frontier-class, not small): 51% with PRM vs 57% baseline; steps 12.2 vs 15.2; cost $0.053 vs $0.029. PRM did NOT beat the baseline on resolve rate. Result: local action scores did not translate to global patch quality.
- SWE-PRM ("When Agents Go Astray," NeurIPS 2025) [SNIP]: course-correction PRM for SWE agents. Numbers NOT CHECKED. Source: https://neurips.cc/virtual/2025/124506
- SWE-TRACE (arXiv 2604.14820) [SNIP]: rubric PRM plus heuristic test-time scaling; 4B model 38.9% -> 40.7%. Baseline comparison not checked.
- SWE-Router (arXiv 2607.00053, 2026; ICML 2026 listing) [SNIP]: learned value-based router that reads the partial trajectory of a cheap model and decides whether to escalate. Route-AUC 0.780 vs 0.627 (non-temporal) and 0.476 (prompt-only router) on SWE-bench Verified. Value head built on a fine-tuned code LLM (e.g., Qwen2.5-Coder-7B). Routes between models, not actions; frozen-backbone status NOT CONFIRMED [UNV]. Source: https://arxiv.org/abs/2607.00053
- LLM-as-a-Verifier (arXiv 2607.05391) [SNIP]: per-step PRM framing; best-of-N raised pass@1 by 9.5% on SWE-bench Lite. Verifier size NOT FOUND [UNV].
- SweRankLLM-Small (7B, Qwen2.5-Coder-7B-Instruct fine-tuned listwise reranker) [SNIP]: targets issue localization, not agent actions. Source: https://huggingface.co/Salesforce/SweRankLLM-Small (SweRank arXiv 2505.07849 not opened)
- DARS (ACL 2025) [SNIP]: dynamic action re-sampling; branches at key decisions; no learned reranker. 47% pass@1 on SWE-bench Lite. Source: https://preview.aclanthology.org/setup/2025.acl-long.973
- Found: no 2025 paper with a learned action scorer on a frozen small SWE agent that beats a simple baseline on SWE-bench [NOT FOUND]. The one PRM check (SWE-Shepherd) is negative on resolve rate.

---

## Ranked list: best evidence for a frozen or lightly trained small model on ONE 6 GB GPU plus free Kaggle T4/TPU

Note: per the project rule, all compute (sampling, training, scoring) runs on Kaggle, not the PC. The PC only writes code and calls APIs. Each item below lists what Kaggle would need.

1. **Repeated sampling with execution-verified selection (frozen backbone, no training).**
   - Evidence: Large Language Monkeys (15.9% -> 56% at 250 samples, SWE-bench Lite) [SNIP]; CodeMonkeys (57.4% at about $2,300) [SNIP]; CodeT (65.8% HumanEval, selection by model-written tests) [SNIP]; R2E-Gym hybrid selection (42-43% -> 51%) [ABS].
   - Needs: a 4B-8B Qwen3 quantized to fit T4 16 GB (6 GB PC GPU too small for many samples), an isolated execution sandbox for patches and tests (Docker or the WSL sandbox noted in project memory), a model-written test generator, and a selection step. Kaggle: several GPU-hours per 500-task sample batch [estimate, unmeasured]. Limit: the verifier gap; small models' own test quality is unknown.

2. **Light SFT (LoRA) on offline strong-model trajectories, used once.**
   - Evidence: SWE-smith (14.3% at 100 trajectories -> 40.2% at 6.2k, 32B) [ABS]; SWE-AGILE 8B 15.8% -> 21.45% with 2.2k trajectories [SNIP]; SWE-Dev-7B 23.4% [ABS]; Skywork-SWE 38.0% (32B) [SNIP].
   - Needs: a trajectory set (SWE-smith, Nebius SWE-agent-trajectories, SWE-Gym) with licences checked and the project's upload rule respected (user approval needed before any upload to Kaggle); a LoRA run on a 4B-8B base in Kaggle T4 or TPU; a held-out split. Limit: 7-8B gains are modest (around +6 points in the one 8B case) and single-run.

3. **Execution-environment RL with verifiable rewards on a 4B-8B base (GRPO-style, lightly trained).**
   - Evidence: SWE-Master 4B 27.6% -> 33.4% [SNIP]; SkyRL-14B 18.0% -> 21.6% (about 300 samples) [SNIP]; SWE-Gym gains up to +19% at 7B-32B [ABS]; DAgger-style vs GRPO on 8B (29.8% vs 8.2%) shows recipe sensitivity [SNIP].
   - Needs: executable environments (SWE-Gym or R2E-Gym tasks), an OpenHands-compatible rollout harness, Kaggle TPU or T4 for hours of rollouts. Heaviest compute of the three. Limit: reported runs are single runs, scaffolds differ, and plain RL can underperform.

Why not the verifier/PRM option: the trained verifiers that help (R2E hybrid, SWE-RM) are trained on or built from trajectories and are 30B+ or 32B; the one small-model PRM test (SWE-Shepherd) lowered resolve rate. Evidence for a 1B-8B learned steering component is NOT FOUND.

Why Reflexion and self-debugging rank lower: good on frontier models, but weak self-repair in small open models (4.43% vs 28.90%) [SNIP].

---

## CHECKLIST (DONE / NOT FOUND)

Section 1 (execution feedback at inference)
- Self-debugging with test/compiler/traceback feedback: DONE (arXiv 2304.05128, ICLR 2024; numbers SNIP)
- Reflexion-style loops: DONE (arXiv 2303.11366; HumanEval number conflicting across sources)
- Reproduction scripts written by the model (Agentless): DONE (arXiv 2407.01489 abstract). Reproduction-test detail in validation stage: NOT FOUND in abstract.
- Reproduction scripts written by the model (CodeMonkeys): DONE (arXiv 2501.14723)
- Small-model (under 10B) version of 1: NOT FOUND

Section 2 (repeated sampling and selection)
- Large Language Monkeys (2407.21787), pass@k vs selection gap: DONE (SNIP for SWE-bench numbers; exact plateau threshold conflicts: UNV)
- CodeT (model-written tests): DONE (2207.10397; HumanEval only)
- Execution-based verifiers: DONE (R2E-Gym 2504.07164 ABS; SWE-Gym verifier 2412.21139 ABS)
- Trained reward models: DONE (SWE-RM 2512.21919, SNIP; SWE-Gym verifier)
- Majority voting: DONE (ICLR 2026 Launer et al.; Majority of Bests; Self-Rewarding RL)
- Which verifiers close the gap for 1B-8B models: NOT FOUND

Section 3 (RL with verifiable rewards)
- SWE-Gym: DONE (ABS)
- R2E-Gym: DONE (ABS)
- SWE-smith: DONE (ABS; used for SFT more than RL)
- SWE-RL: DONE (SNIP; 70B, no 7-14B result)
- DeepSWE (Qwen3-32B): DONE (MC/SNIP; self-reported)
- Skywork-SWE: DONE (SNIP; abstract not fetched)
- 7-14B results: DONE (SkyRL-14B SNIP; SWE-AGILE 8B SNIP; SWE-Master 4B SNIP; SWE-TRACE 4B SNIP; SWE-Dev-7B ABS). SWE-Dev 14B: NOT FOUND (only 7B and 32B exist).

Section 4 (distillation)
- SWE-smith -> SWE-agent-LM-32B: DONE (ABS 40.2%)
- Nebius SWE-rebench: DONE (SNIP; trajectory-generating agent not confirmed)
- OpenHands-LM: NOT FOUND (no scores or training details)
- Devstral: DONE (46.8% SNIP; Small 2 numbers conflict)

Section 5 (small learned components)
- Process reward models: DONE (SWE-Shepherd SNIP, negative result; SWE-PRM SNIP, numbers not checked; SWE-TRACE SNIP)
- Value heads: DONE (SWE-Router SNIP; frozen status not confirmed)
- Action rerankers: DONE (SweRankLLM localization only; DARS not learned)
- Routers beating simple baselines: DONE (SWE-Router, Route-AUC vs prompt-only baselines, 2026)
- Any small learned steering component beating a simple baseline on SWE-bench (2025): NOT FOUND

Other
- Qwen3-Coder-30B-A3B official score: NOT FOUND (sources conflict: 51.6 / 49.2 / 60.4)
- Compute cost of each method on Kaggle: NOT FOUND for most methods (only SkyRL 20 h on 8xH200 and CodeMonkeys $2,300 with sources)

---

## Key sources (primary where available)

- https://arxiv.org/abs/2407.21787 (Large Language Monkeys)
- https://arxiv.org/abs/2501.14723 (CodeMonkeys)
- https://arxiv.org/abs/2407.01489 (Agentless)
- https://arxiv.org/abs/2412.21139 (SWE-Gym)
- https://arxiv.org/abs/2504.07164 (R2E-Gym)
- https://arxiv.org/abs/2504.21798 (SWE-smith)
- https://arxiv.org/abs/2502.18449 (SWE-RL)
- https://huggingface.co/agentica-org/DeepSWE-Preview (DeepSWE)
- https://arxiv.org/abs/2506.19290 (Skywork-SWE)
- https://arxiv.org/abs/2512.21919 (SWE-RM)
- https://arxiv.org/abs/2506.07636 (SWE-Dev)
- https://arxiv.org/abs/2604.11716 (SWE-AGILE)
- https://arxiv.org/abs/2604.10493 (SWE-Shepherd)
- https://arxiv.org/abs/2607.00053 (SWE-Router)
- https://arxiv.org/abs/2505.20411 (SWE-rebench)
- https://arxiv.org/abs/2303.11366 (Reflexion)
- https://arxiv.org/abs/2304.05128 (Self-debug)
- https://arxiv.org/abs/2207.10397 (CodeT)
- https://arxiv.org/abs/2511.16108 (SkyRL-Agent)

---

<!-- source: results/raw/research-20261015-smallmodel/B_coverage_gap.md -->
# Part B. Pass@1 vs pass@k coverage gap for small open coding models (web research)

Date: 2026-10-15. Web research only. No downloads, no code run.
Status key: **[PRIMARY]** = numbers read from the paper's HTML/abstract via WebFetch.
**[SNIPPET]** = seen only in search-result text, not opened. **[UNVERIFIED]** = recalled or from a third-party aggregator; do not cite.
Arithmetic done by me is marked **(calc)**.

## 1. SWE-bench (Verified / Lite / SWE-rebench / SWE-Gym), small models

| Model (size) | Setting | Benchmark | pass@1 / resolve | best-of-k or verifier | Source |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct, base, zero-shot | OpenHands | Verified | 1.8 | - | [PRIMARY] https://arxiv.org/html/2412.21139v2 (Table 3) |
| Qwen2.5-Coder-14B-Instruct, base | OpenHands | Verified | 4.0 | - | same |
| Qwen2.5-Coder-32B-Instruct, base | OpenHands | Verified | 7.0 | - | same |
| Same 7B / 14B / 32B after SFT on 491 SWE-Gym trajectories | OpenHands | Verified | 10.6 / 16.4 / 20.6 | 32B + verifier: Best@8 29.8, Best@16 32.0 (Pass@16 42.8) | [PRIMARY] same, Sec 5.1 |
| Same 7B / 14B / 32B SFT | OpenHands | Lite | 10.0 / 12.7 / 15.3 | 32B + verifier Lite 26.0 (text says 26.3) | [PRIMARY] same |
| SWE-agent-LM-7B (Qwen2.5-Coder-Inst. base, 5,016 Claude-3.7 trajectories) | SWE-agent | Verified | 15.2 | none (pass@1 only) | [PRIMARY] https://arxiv.org/html/2504.21798 |
| SWE-agent-LM-32B | SWE-agent | Verified | 40.2 | none | same |
| SWE-Master-4B-SFT (Qwen3-4B) | R2E-Gym | Verified | 27.6 | - | [PRIMARY] https://arxiv.org/html/2604.14820 (Table 2) |
| SWE-Master-4B-RL (Qwen3-4B) | R2E-Gym | Verified | 33.4 | - | same |
| SWE-TRACE-4B (Qwen3-4B, SFT+RL) | mini-SWE-agent | Verified | 38.9 (greedy) | heuristic TTS 40.7 (k not stated, 119 env runs) | same |
| R2E-Gym-32B (Qwen2.5-Coder) | R2E-Gym | Verified | 34.4 | hybrid Best@16 49.4, Best@26 51.0 | [PRIMARY] https://arxiv.org/html/2504.07164v1 |
| SWE-Gym-32B verifier only | - | Verified | - | Best@16 32.0 | [PRIMARY] SWE-Gym html |
| SWE-World SWR-32B (Qwen2.5-Coder-32B base, Docker-free RL) | - | Verified | 55.0 (text also says 54.8) | TTS@8 68.2 | [PRIMARY] https://arxiv.org/html/2602.03419v1 |
| Qwen3-30B-A3B / Qwen3-Coder-30B-A3B-Instruct | third-party only | Verified | 49.2 (Uni-Agent docs, 100 turns, 4 rollouts) | - | [SNIPPET] https://uni-agent.readthedocs.io/en/latest/benchmark/inference.html; 60.4 (llmrun.dev) and 51.6 (memory of model card) are [UNVERIFIED] |
| Qwen3.5-4B fine-tune (community card, mini-SWE-agent) | mini-SWE-agent | Verified | 44.8 base, 55.0 fine-tuned | - | [SNIPPET] https://huggingface.co/AxiaoDBL/qwen3.5-4b-swebench-sft (Qwen3.5, not Qwen3) |
| Klear-Agent-8B (SWE-smith mini-swe-agent-plus data) | mini-SWE-agent+ | Verified | 39.0 | - | [SNIPPET] https://gitcode.com/hf_mirrors/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k/blob/main/README.md (not opened) |
| Satori-SWE-32B | - | Verified | - | Best@50 41.6 | [SNIPPET] https://arxiv.org/pdf/2505.23604 (not opened) |

SWE-rebench: official site not reachable. Aggregators only ([SNIPPET] https://benchlm.ai/benchmarks/swe-rebench, https://benchmarklist.com/benchmarks/swe_rebench/, snapshot July 2026). Open-weight top is GLM-5 about 62.8%; Qwen3.5-27B 58.9%; Qwen3.5-35B-A3B 53.7%; Gemma 4 31B 41.6%. **No model at or below 14B found on SWE-rebench.** Benchlist shows Pass@5 next to resolved (e.g. GLM-4.5 Air 38.3 resolved, 57.9 pass@5); it does not define pass@k beyond that.

## 2. Function-level benchmarks (pass@1 and pass@k)

| Model | Benchmark | Value | Metric / notes | Source |
|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | HumanEval / HE+ / MBPP / MBPP+ | 88.4 / 84.1 / 83.5 / 71.7 | EvalPlus-style, pass@1 stated only for LCB column | [PRIMARY] https://arxiv.org/html/2409.12186v3 (Table 16) |
| Qwen2.5-Coder-7B-Instruct | BigCodeBench Full / Hard | 41.0 / 18.2 | not labelled pass@1 in text | same |
| Qwen2.5-Coder-7B-Instruct | LiveCodeBench | table 18.2 vs text 37.6 | **discrepancy inside the paper; do not cite until checked against PDF** | same |
| Qwen2.5-Coder-14B-Instruct | HE / HE+ / MBPP / MBPP+ | 89.6 / 87.2 / 86.2 / 72.8 | | same |
| Qwen2.5-Coder-14B-Instruct | BigCodeBench Full / Hard | 48.4 / 22.2 | | same |
| Qwen2.5-Coder-14B-Instruct | LiveCodeBench | 23.4 | pass@1 label | same |
| Qwen3-4B / 8B / 30B-A3B thinking | LiveCodeBench v5 (2024.10-2025.02) | 54.2 / 57.5 / 62.6 | temp 0.6, top-p 0.95, top-k 20, 32,768 out; pass@1 **not explicitly stated** | [PRIMARY] https://arxiv.org/html/2505.09388v1 (Tables 15-17) |
| Qwen3-4B / 8B / 30B-A3B non-thinking | LiveCodeBench v5 | 21.3 / 22.8 / 29.8 | temp 0.7, top-p 0.8 | same (Table 16/18) |
| Qwen3-4B-Base / 8B-Base / 30B-A3B-Base | MBPP (3-shot) / EvalPlus | 67.0 / 69.8 / 74.4; EvalPlus 63.5 / 67.7 / 71.5 | base models only | same (Tables 5-7) |
| Qwen3 4B/8B/30B | HumanEval+, BigCodeBench | NOT FOUND | not in Qwen3 report | - |
| Any of these models | pass@k for k != 1 on these benchmarks | NOT FOUND | - | - |
| EvalPlus leaderboard, Qwen2.5-Coder-7B/14B | HE+/MBPP+ | NOT FOUND (only 32B: HE+ 87.2, MBPP+ 77 via aggregator [SNIPPET]) | leaderboard not opened | https://evalplus.github.io/leaderboard.html (not opened) |

## 3. How much of the pass@1 -> pass@k gap verifiers recovered

Gap closure = (selected - pass@1) / (oracle - pass@1). Calc by me unless the paper states it.

**SWE-bench (repository level)**

| Setting | pass@1 / random | Selected | Oracle (ground-truth tests) | Closure | Verifier | Source |
|---|---|---|---|---|---|---|
| R2E-Gym-32B, Verified, k=26 | 34.4 | 51.0 (hybrid) | 64.4 (Pass@26) | 55% (calc) | exec-based 43.7 and exec-free 42.8 plateau alone; hybrid is the combo | [PRIMARY] R2E-Gym html |
| Qwen3-32B, Verified, k=16 | random 22.6 | 40.6 (Agentic Rubrics) | 51.4 | 62% (calc) | Claude Sonnet-4.5 writes artifacts; GPT-5 judges | [PRIMARY] https://arxiv.org/html/2601.04171v1 |
| Qwen3-Coder-30B-A3B, Verified, k=16 | random 39.6 | 54.2 | 65.6 | 56% (calc) | same | same |
| Qwen3-32B, same run | - | Self-consistency 33.2; Patch classifier 37.1; agentic tests 33.6 | 51.4 | 37%-50% (calc; self-consistency 37, agentic tests 38, patch classifier 50) | - | same |
| SWE-Gym 32B, Verified, k=16 | 20.6 | 32.0 | 42.8 | 51% (calc) | trained verifier on SWE-Gym trajectories | [PRIMARY] SWE-Gym html |
| SWR-32B, Verified, K=8 | 55.0 (K=1) | 68.2 | not given numerically (plot only) | n/a | SWR-32B verifier | [PRIMARY] SWE-World html |

Verifier cost per instance (Agentic Rubrics, Qwen3-32B only): artifact USD 0.245 (rubrics), 0.499 (tests), 0.640 (patch similarity); grading USD 0.001-0.006 per rollout. Artifacts used Claude Sonnet-4.5, judge GPT-5. **Not a small-model verifier.**

**Function level (LiveCodeBench, non-target model)**

| Model | Mechanism (k=5) | Pass@1 | Gain | Share of oracle gap (calc) |
|---|---|---|---|---|
| Qwen3.6-35B-A3B (not in the target list) | first sample | 72.33 | 0 | - |
| same | cross-family LLM (L3) | 74.31 | +1.97 | 17% |
| same | generated tests (L4-gen) | 75.03 | +2.70 | 23% |
| same | same-family LLM (L1) | 75.83 | +3.50 | 30% |
| same | public tests (L4-public, partial oracle) | 80.47 | +8.14 | 69% |
| same | oracle any@5 | 84.07 | +11.74 | 100% |

Source: [PRIMARY] https://arxiv.org/html/2607.17531 (single paper, n=2,888, k=5, no replication found). Generated-test verifier accuracy 53.05% vs public-test 92.98%. HumanEval+ on the same model: first sample 138-139/164, oracle any@5 142-144/164, so headroom is 3-6 problems (calc, about 2-4 pp).

Other: no selector paper with a 4B-8B open generator was found. Top Pass (pass@k-maximised ranker, +32.9% pass@1 on CodeContests, relative) is [SNIPPET] https://journal.hep.com.cn/fcs/EN/1159675230601929072 (not opened).

## 4. Cost per sample, 4-8B-class

| Item | Value | Source |
|---|---|---|
| SWE-TRACE-4B greedy, mini-SWE-agent | 18.7 min per issue, 67 env executions | [PRIMARY] https://arxiv.org/html/2604.14820 (Table 3) |
| SWE-TRACE-4B parallel rollout | 54.1 min, 376 env executions, 40.2% | same |
| SWE-TRACE-4B heuristic TTS | 31.6 min, 119 env executions, 40.7% | same |
| SWE-TRACE-4B token use per issue | 26.8k (rubric-PRM RL), 29.8k (exec-only RL) | same (Table 4) |
| Cascaded SFT / standard SFT, 4B | 25.9M / 33.0M total generated tokens (66 / 74 steps avg) | same (Table 3). Unclear whether per-run or per-eval-set. Treat as unverified scope. |
| SWE-agent-LM expert runs | USD 2.00 cap per trajectory; SWE-smith dataset USD 1,360 total | [PRIMARY] https://arxiv.org/html/2504.21798 |
| Frontier API mini-SWE-agent | about USD 0.30 per instance (GPT-5, Sonnet 4) | [SNIPPET] https://www.swebench.com/post-250820-mini-roulette.html (not opened) |
| Local 4B-8B wall-clock cost | NOT FOUND as a measured figure. Calc example only: USD 2/h GPU x 20 min = USD 0.67 | [SNIPPET] hardware note, not a measurement |
| Function-level per-sample tokens/time | NOT FOUND for Qwen3-4B/8B | - |

## 5. Does a 4B-class model solve repo-level tasks at all?

- **Untrained small instruct bases:** 1.8% (7B), 4.0% (14B), 7.0% (32B) on Verified, OpenHands, zero-shot (SWE-Gym table). With these numbers a selector has almost nothing to pick from.
- **Trained 4B (Qwen3-4B backbone):** 27.6-38.9% pass@1 on Verified after SFT/RL with R2E-Gym or mini-SWE-agent. So pass@k is not near zero for a trained 4B. SWE-TRACE does not report an oracle pass@k, so its selector headroom is **unknown**.
- Qwen3-8B / Qwen3-4B base or instruct on SWE-bench: **NOT FOUND** from Qwen or any primary source.
- The "random" pass@16 for Qwen3-32B (22.6) is the average single-sample rate in that pool, so it works as pass@1 (calc: expected value of random pick).

## 6. Checklist (for the lead)

- [DONE] 1a. SWE-bench Verified, small models: Qwen2.5-Coder 7B/14B/32B base and after SFT (SWE-Gym). Sources above.
- [DONE] 1b. SWE-Gym numbers including pass@1 and verifier best@k: 32B only has verifier numbers; 7B/14B verifier numbers NOT FOUND.
- [DONE] 1c. Qwen3-4B SWE-bench (via SWE-Master-4B, SWE-TRACE-4B): 27.6 / 33.4 / 38.9 / 40.7.
- [NOT FOUND] 1d. Qwen3-8B SWE-bench, any source.
- [NOT FOUND] 1e. Qwen3-4B or 8B on SWE-rebench; no official source reachable; aggregators list no model at or below 14B.
- [DONE, UNVERIFIED] 1f. Qwen3-Coder-30B-A3B SWE-bench: 49.2 (Uni-Agent docs), others conflict. Not citable without the model card.
- [DONE] 1g. Scaffolds: mini-SWE-agent (SWE-TRACE), R2E-Gym (R2E-Gym, SWE-Master), SWE-agent (SWE-agent-LM), OpenHands (SWE-Gym). Agentless: only a 49.2 DeepSeek-R1 row in R2E-Gym Table 4 (not small).
- [DONE] 2a. HumanEval+ / MBPP+ / BigCodeBench / LCB for Qwen2.5-Coder-7B and 14B (pass@1-labelled only for LCB).
- [DONE, flagged] 2b. Qwen3-4B/8B/30B-A3B LiveCodeBench v5 thinking and non-thinking. Pass@1 not explicitly stated.
- [NOT FOUND] 2c. Qwen3 HumanEval+, MBPP+, BigCodeBench for the instruct models.
- [NOT FOUND] 2d. Any pass@k (k=5-100) on HumanEval+/MBPP+/LCB/BCB for a model at or below 14B. Only k=5 on a non-target 35B-A3B model (2607.17531).
- [DONE] 3a. Oracle vs selected vs pass@1 with a named verifier, SWE-bench (R2E-Gym-32B, Qwen3-32B, Qwen3-Coder, SWE-Gym-32B).
- [DONE] 3b. Oracle vs selected, LCB (2607.17531), non-target model.
- [NOT FOUND] 3c. Oracle vs selected for any model at or below 8B.
- [DONE] 4a. Tokens/time per issue for a 4B (SWE-TRACE-4B).
- [DONE] 4b. Verifier USD cost per instance and per rollout (Agentic Rubrics, Frontier models).
- [NOT FOUND] 4c. Measured local GPU USD per sample for 4-8B models.
- [DONE] 5. Evidence on 4B repo-level solve rate: 27.6-38.9% pass@1 trained; 1.8-7% untrained 7-32B bases.

## 7. Caveats

- Oracle numbers are upper bounds using ground-truth tests. They are not deployable selectors.
- Pass@k definitions vary. Several industry reports define pass@3 as "any of 3 succeeded". Compare only like with like.
- Pool scaffolds differ (OpenHands, mini-SWE-agent, R2E-Gym, SWE-agent); reported gaps are not directly comparable.
- SWE-TRACE and Qwen3.6 are the only small/non-frontier papers with per-sample costs or oracle rows; both are single-paper evidence.
- Contradictions found: SWR-32B 54.8 vs 55.0; SWE-Gym Lite verifier 26.0 vs 26.3; Qwen2.5-Coder-7B LCB 18.2 table vs 37.6 text. Check the PDFs before citing.
- Relevance to the selector question: at 4B the untrained pool is near zero, the trained pool is 28-41%. Headroom at 4B is **not measured** by any source found.

---

<!-- source: results/raw/research-20261015-smallmodel/C_test_time_compute.md -->
# Part C. Test-time compute for small models (web research, 2026-10-15)

Scope: more inference-time compute (thinking budgets, self-consistency, search, refinement, debate, parallel thinking, compute-optimal allocation) to make small open models (Qwen family 1-14B, maybe 30B-A3B) stronger on coding/agent tasks.
Setting for "ADOPTABLE?": 6 GB RTX 3050 + 15 GB RAM locally; Kaggle T4 x2 and TPU v5e-8 for compute (standing rule: compute runs on Kaggle); local isolated sandbox for code execution; no paid APIs; no strong model at runtime.

Evidence grades: **[abs]** = abstract read via WebFetch; **[snip]** = search-result excerpt or secondary summary only, numbers not checked against the PDF; **[unv]** = title/ID seen, nothing beyond it verified.
All numbers are as reported by the paper, not re-measured by us. Nothing here is "estimate (unmeasured)" for us.

---

## 1. Chronological index (newest first)

| Date | Paper / source | Family | Evidence |
|---|---|---|---|
| 2026-09 | Scaling of Capability and Efficiency at Inference Time in LRMs, arXiv 2609.27166 | scaling law | [abs] no numbers |
| 2026-09 | TIPCODER, arXiv 2609.03309 | best-of-N + RM, Qwen2.5-Coder-7B | [snip] |
| 2026-07 | "Best Programming Language for Tokenmaxxing", arXiv 2607.22807 | coding agent, 6K reasoning budget, Gemma 4 / Qwen 3.6 | [snip] appendix only |
| 2026-06 (v2 2026-09) | Evaluating Test-Time Scaling of General LLM Agents, arXiv 2602.18998 | sequential + parallel scaling, 10 agents | [abs] |
| 2026-05 | DiffCodeGen (differential TTS), arXiv 2605.20473 | code, LiveCodeBench | [snip] |
| 2026-05 | Matched-ceiling debate study, arXiv 2605.09618 | debate vs voting | [snip] |
| 2026-04 | Scaling Test-Time Compute for Agentic Coding, arXiv 2604.16529 | rollout summaries, tournament voting | [snip] |
| 2026-04 | Overthinking in test-time scaling, arXiv 2604.10739 | more thinking hurts | [unv] |
| 2026-04 | Single-agent vs MAS under equal token budget, arXiv 2604.02460 | MAS vs single agent | [unv] secondary blog only |
| 2026-04 | Test-Time Scaling Makes Overtraining Compute-Optimal, arXiv 2604.01411 | train-to-test scaling laws | [abs] no numbers |
| 2026-03 | Code-A1 (best-of-N + generated tests, Qwen2.5-Coder-3B), arXiv 2603.15611 | execution-style verifier | [snip] |
| 2026-02 | CodeScaler (RM ranker, Qwen3 base), arXiv 2602.17684 | RM best-of-N | [snip] |
| 2026-02 | SWE-World (SWR-32B TTS@8; 14B verifier), arXiv 2602.03419 | SWE-bench Verified TTS | [snip] |
| 2026-01 | FunPRM (function-as-step PRM, Qwen2.5-7B-Coder), arXiv 2601.22249 | PRM selection, LiveCodeBench | [snip] |
| 2025-12 | SWE-RM (30B-A3B verifier), arXiv 2512.21919 | execution-free verifier | [snip] |
| 2025-08 | Deep Think with Confidence (DeepConf), arXiv 2508.15260 | confidence-based early stop + voting | [snip] abstract summary |
| 2025-07 | OpenCodeReasoning-II (self-critique), arXiv 2507.09075 | self-critique | [unv] |
| 2025-07 | Inverse Scaling in Test-Time Compute, arXiv 2507.14417 (TMLR 2025) | longer thinking can hurt | [snip] |
| 2025-07 | Survey: Reasoning on a Budget, arXiv 2507.02076 | budget control survey | [snip] |
| 2025-05 | VG-Search (verification granularity), arXiv 2505.11730 | beam / BoN hybrid | [abs] |
| 2025-05 | Qwen3 Technical Report (thinking budget), arXiv 2505.09388 | budget knob | [snip] no numbers |
| 2025-04 | ThinkPRM (generative verifier), arXiv 2504.16828 | small verifier | [snip] |
| 2025-04 | R2E-Gym (hybrid verifiers, open SWE agents), arXiv 2504.07164 | execution + execution-free verifier | [snip] |
| 2025-03 | Thinking Longer, Not Larger (Tongyi), arXiv 2503.23803 | TTC search on Qwen2.5-Coder-32B | [snip] |
| 2025-03 | L1 / LCPO, arXiv 2503.04697 (COLM 2025) | length control | [snip] |
| 2025-02 | S*: Test-time scaling for code, arXiv 2502.14382 | parallel + sequential debugging | [abs] |
| 2025-02 | Can 1B LLM Surpass 405B?, arXiv 2502.06703 | compute-optimal TTS | [snip] |
| 2025-02 | Multi-agent debate systematic eval, arXiv 2502.08788 | MAD vs CoT/SC | [snip] |
| 2025-01 | s1: Simple test-time scaling, arXiv 2501.19393 (EMNLP 2025) | budget forcing | [snip] |
| 2025-01 | CodeMonkeys, arXiv 2501.14723 | serial + parallel TTS, SWE-bench Verified | [snip] |
| 2025-01 | rStar-Math, arXiv 2501.04519 (ICML 2025 oral) | MCTS + PPM, training-heavy | [snip] |
| 2024-12 | HF blog "Scaling test-time compute with open models" (not arXiv) | 1B/3B vs 8B/70B on MATH-500 | [snip] blog |
| 2024-11 | CodeTree, arXiv 2411.04329 (NAACL 2025) | agent-guided tree search for code | [snip] |
| 2024-10 | SWE-Search (MCTS), arXiv 2410.20285 (ICLR 2025) | MCTS for SWE agents | [snip] |
| 2024-08 | Smaller, Weaker, Yet Better, arXiv 2408.16737 (ICLR 2025) | cheap-model sampling | [snip] |
| 2024-08 | Snell et al., arXiv 2408.03314 (ICLR 2025) | compute-optimal TTS | [snip] abstract |
| 2024-08 | Inference Scaling Laws, arXiv 2408.00724 (ICLR 2025) | sampling + tree search vs params | [snip] |
| 2024-07 | Large Language Monkeys, arXiv 2407.21787 | repeated sampling coverage | [snip] |
| 2024-07 | Agentless, arXiv 2407.01489 (FSE 2025) | fixed localize-repair-validate | [snip] |
| 2024-06 | Mixture-of-Agents, arXiv 2406.04692 | layered LLM ensemble | [snip] |
| 2024-06 (v3) | LATS, arXiv 2310.04406 | MCTS for code/agents | [snip] |
| 2023-10 | Huang et al., Can LLMs self-correct reasoning, arXiv 2310.01798 | self-correct / debate vs SC | [snip] |
| 2023-06 | Olausson et al., Is self-repair a silver bullet?, arXiv 2306.09896 (ICLR 2024) | self-repair | [snip] |
| 2023-05 | Du et al., multi-agent debate, arXiv 2305.14325 (ICML 2024) | debate | [snip] |
| 2023-04 | Self-Debugging, arXiv 2304.05128 (ICLR 2024) | execution-feedback repair | [snip] |
| 2022-07 | CodeT, arXiv 2207.10397 (ICLR 2023) | generated tests + dual agreement | [snip] |
| 2022-03 | Self-consistency, arXiv 2203.11171 | majority over sampled paths | [snip] |
| 2022-03 | AlphaCode, arXiv 2203.07814 | sample + filter + cluster | [snip] |

---

## 2. Per-method entries

Format: what it is | measured gain (before -> after, size, benchmark) | compute multiplier | feedback/verifier needed | ADOPTABLE?

### 2a. Execution-verified best-of-N and generated tests (strongest for code)

- **CodeT** [arXiv 2207.10397](https://arxiv.org/abs/2207.10397v1). Model writes solutions and tests; choose by "dual execution agreement". HumanEval pass@1 65.8% on code-davinci-002 (+18.8 pts). Multiplier: N samples + N_test tests. Needs: sandbox execution of generated tests. [snip]
- **AlphaCode** [arXiv 2203.07814](https://arxiv.org/pdf/2203.07814). Massive sampling, filter by behaviour, cluster, submit <=10. 34.2% CodeContests solved at 10 submissions. Needs: example tests + execution. [snip]
- **S\*** [arXiv 2502.14382](https://arxiv.org/abs/2502.14382) (Findings EMNLP 2025). Parallel sampling + sequential iterative debugging + adaptive distinguishing-input synthesis. Abstract: a 3B model with S* outperforms GPT-4o-mini; 7B+S* beats Qwen2.5-32B-Instruct by 10.7% on LiveCodeBench [snip]; GPT-4o-mini+S* beats o1-preview by 3.7% [abs]; R1-Distill-32B+S* 85.7% vs o1-high 88.5% [abs]. Multiplier: N x iterations. Needs: sandbox execution. **ADOPTABLE: yes** (the cleanest small-model result for code; the execution part must run in the local sandbox or on Kaggle, see section 4).
- **Code-A1** [arXiv 2603.15611](https://arxiv.org/pdf/2603.15611). Best-of-N with Qwen2.5-Coder-3B; every candidate run against every generated test suite. Small-model, 2026. Numbers [unv]. **ADOPTABLE: yes** (same family as S*).
- **CodeMonkeys** [arXiv 2501.14723](https://arxiv.org/pdf/2501.14723). Serial (edit-test iterations) and parallel (many trajectories) scaling on SWE-bench Verified; 57.4% at about USD 2,300 total (abstract, per search) with a frontier model (Claude 3.5 Sonnet per blog). Ensemble of top submissions reaches 66.2% (search summary). Multiplier: very high. Needs: generated tests + execution + LLM selection. **ADOPTABLE: partly** (the pattern yes; the budget and model class no).
- **Large Language Monkeys** [arXiv 2407.21787](https://arxiv.org/pdf/2407.21787). Coverage on SWE-bench Lite with DeepSeek-Coder-V2-Instruct: 15.9% (1 sample) -> 56% (250 samples) (search summary; v1 vs v3 model names differ). Key caveat: without an automatic verifier, majority voting and reward models plateau. **ADOPTABLE: yes, as the reason to have an execution verifier at all.**
- **Self-consistency** [arXiv 2203.11171](https://arxiv.org/pdf/2203.11171). Majority over sampled paths; GSM8K +17.9 pts on PaLM-540B with 40 paths (v3, per abstract summary). Math/answer-type tasks only; no direct code analogue. **ADOPTABLE: partly** (for final-answer tasks; code needs execution, not vote).

### 2b. Iterative refinement with execution feedback

- **Self-Debugging** [arXiv 2304.05128](https://arxiv.org/abs/2304.05128v2) (ICLR 2024). With unit-test feedback: up to +12% on MBPP/TransCoder; reuse of failed attempts matches 10x more samples (summary of paper; [snip]). Needs: real execution output. **ADOPTABLE: yes** (use real test output, not only self-critique).
- **Self-repair** [arXiv 2306.09896](https://arxiv.org/pdf/2306.09896) (ICLR 2024). Gains modest, vary by subset, sometimes absent once repair cost counted. Bottleneck: the model's own feedback. Human feedback gave 1.58x more repaired programs (GPT-4). **ADOPTABLE: partly** (small models judge their own bugs poorly; feed execution traces, cap rounds).
- **Thinking Longer, Not Larger** [arXiv 2503.23803](https://arxiv.org/abs/2503.23803v2) (Alibaba Tongyi). Qwen2.5-Coder-32B + development-process search with reward models and execution checks reaches 46% on SWE-bench Verified; claims to beat DeepSeek-R1 671B and o1 (abstract summary [snip]). Needs: RM + execution. **ADOPTABLE: partly** (32B is above our range; 7B-14B not reported in the summary).
- **SWE-World** [arXiv 2602.03419](https://arxiv.org/pdf/2602.03419). SWR-32B: 55.0 (K=1) -> 68.2 (TTS@8) on SWE-bench Verified; R2E-Gym-Verifier-14B reaches 59.4 at TTS@8 and plateaus early (~TTS@5) [snip]. **ADOPTABLE: partly** (plateau is the useful lesson).

### 2c. Tree / MCTS / beam search over actions or code

- **SWE-Search** [arXiv 2410.20285](https://arxiv.org/abs/2410.20285) (ICLR 2025). MCTS with LLM value estimates and multi-agent discriminator debate. +23% relative on SWE-bench across five models vs non-MCTS open agents [snip]. Code: github.com/aorwall/moatless-tree-search. Multiplier: many rollouts. Needs: value/feedback + repo execution. **ADOPTABLE: partly** (the reported gains use larger models; the scaling-with-search claim is the part to test).
- **CodeTree** [arXiv 2411.04329](https://arxiv.org/html/2411.04329v2) (NAACL 2025). Agent-guided tree over strategy/solution/refine stages; GPT-4o base; 95.1% HumanEval, 43.0% CodeContests, 31.9% SWE-bench [snip, GPT-4o, not small]. Needs: execution + LLM feedback. **ADOPTABLE: partly** (idea yes, numbers not for small models).
- **LATS** [arXiv 2310.04406](https://arxiv.org/abs/2310.04406v3). MCTS over reasoning/acting. 92.7% HumanEval pass@1 with GPT-4 in v3 (v1 said 94.4%). Strong-model result. Multiplier: high. **ADOPTABLE: no** for small models (no small-model number; cost).
- **Snell et al.** [arXiv 2408.03314](https://arxiv.org/pdf/2408.03314) (ICLR 2025). PRM-guided beam search and best-of-N compared; best strategy depends on prompt difficulty; >4x compute efficiency vs best-of-N; on problems where the small model has non-trivial success, test-time compute beats a 14x larger model (FLOPs-matched, math). Beam wins at low budgets, over-optimizes on easy prompts at high budgets. Needs: PRM. **ADOPTABLE: yes, as the allocation rule** (per-task budget choice), PRM part partly.
- **VG-Search** [arXiv 2505.11730](https://arxiv.org/abs/2505.11730). Tunable verifier granularity generalizing beam and BoN; adaptive g gains up to 3.1% over beam and 3.6% over BoN while cutting FLOPs >52% (abstract, math-style benchmarks). **ADOPTABLE: partly** (checks verifier cost; not tested on code).
- **rStar-Math** [arXiv 2501.04519](https://arxiv.org/html/2501.04519v1) (ICML 2025 oral). MCTS + process preference model; Qwen2.5-Math-7B on MATH 58.8 -> 90.0; Phi3-mini 41.4 -> 86.4 (search summary [snip]); 4 rounds of self-evolution on 747k problems. Gains need training of a PPM and policy. **ADOPTABLE: no** at inference only; partly as a training recipe (needs user GO for fine-tuning).
- **ThinkPRM** [arXiv 2504.16828](https://arxiv.org/abs/2504.16828v4). Generative verifier that writes its own step-check CoT; trained on 1% of PRM800K labels; the 1.5B model beats discriminative verifiers in reward-guided search on MATH-500 (per abstract summary [snip]); 14B trained on 8K labels beats PRMs trained on ~100x data. Code promised at github.com/mukhal/thinkprm (v3 said "will be released", v4 says released; [unv]). **ADOPTABLE: partly** (needs training; a zero-training judge is the cheaper first step).
- **R2E-Gym hybrid verifier** [arXiv 2504.07164](https://arxiv.org/pdf/2504.07164). Open-weight SWE agents; execution-based + execution-free verifier combined; the execution-based testing agent is Qwen-Coder-32B-based [snip]. **ADOPTABLE: partly.**

### 2d. Thinking budget, length control, budget forcing

- **s1 / budget forcing** [arXiv 2501.19393](https://arxiv.org/abs/2501.19393) (EMNLP 2025). Fine-tune Qwen2.5-32B-Instruct on 1,000 traces (s1K); at test time stop or extend thinking by appending "Wait". AIME24 50% -> 57% with extension (per search summary). Needs fine-tuning (not zero-shot). Multiplier: linear in extra tokens. **ADOPTABLE: partly** (the "Wait" trick is free to try on Qwen3 thinking; the SFT part needs GO).
- **L1 / LCPO** [arXiv 2503.04697](https://arxiv.org/pdf/2503.04697) (COLM 2025). RL to hit a stated token budget (512-8K); reports >100% relative gains over s1 at 512/1024 tokens (search summary [snip]); 1.5B L1 beats GPT-4o at equal length (summary). Needs RL training. **ADOPTABLE: partly** (budget-following is useful; training is out of scope now).
- **Qwen3 thinking budget** [arXiv 2505.09388](https://arxiv.org/abs/2505.09388v1). Built-in budget knob mixing thinking and non-thinking modes; no budget-vs-accuracy numbers retrieved. **ADOPTABLE: yes, to measure** (free, but needs our own curve; the report has the curve, not retrieved).
- **Budget guidance** [arXiv 2506.13752](https://arxiv.org/html/2506.13752v1) and **BudgetThinker** [arXiv 2508.17196](https://arxiv.org/abs/2508.17196): steering/control tokens to a target budget [unv numbers]. Adoptable only with a trained or steering component: partly.
- **Inverse scaling in test-time compute** [arXiv 2507.14417](https://arxiv.org/pdf/2507.14417) (TMLR 2025). Longer reasoning lowers accuracy on counting, regression with spurious features, deduction tasks; five failure modes (distractors, spurious correlations, focus loss). **ADOPTABLE: yes, as a guard** (test thinking budgets on our tasks before assuming more helps).
- **Overthinking study** [arXiv 2604.10739](https://arxiv.org/html/2604.10739v1) (2026): questions "more reasoning always helps"; numbers [unv].

### 2e. Confidence-based early stopping and voting (cheap compute saving)

- **DeepConf** [arXiv 2508.15260](https://arxiv.org/pdf/2508.15260) (ICLR 2026). Uses token log-probs to score traces; offline filter + confidence-weighted vote, and online early stop of low-confidence traces. AIME 2025: DeepConf@512 up to 99.9% accuracy with up to 84.7% fewer generated tokens vs full parallel thinking; Qwen3 and GPT-OSS (abstract summary [snip]). Failure mode: confident wrong traces. Needs: token logprobs (available in vLLM on Kaggle T4/TPU). **ADOPTABLE: yes**, best candidate for saving compute; verify on our code tasks (math numbers do not transfer automatically).
- **Parallel thinking, ParaThinker** [arXiv 2509.04475](https://arxiv.org/pdf/2509.04475). Native parallel paths then merge; +12.3% (1.5B) and +7.5% (7B) with 8 paths, ~7.1% extra latency (abstract [snip]); needs SFT on Qwen2.5 with teacher traces. **ADOPTABLE: no** unless we fine-tune; the latency result suggests parallel samples are cheap on memory-bound decode, which matters for us.

### 2f. Compute-optimal scaling and small vs large

- **Inference Scaling Laws** [arXiv 2408.00724](https://arxiv.org/pdf/2408.00724) (ICLR 2025). Llemma-7B + REBASE tree search beats Llemma-34B with majority voting on MATH across FLOPs budgets; smaller models with more search are Pareto-better at moderate budgets; gains saturate (v3 abstract [snip]). Code: github.com/thu-wyz/inference_scaling. **ADOPTABLE: yes, as the planning claim** (small model wins below a saturation point).
- **Can 1B LLM surpass 405B?** [arXiv 2502.06703](https://arxiv.org/pdf/2502.06703) (ICLR 2025 workshop). Compute-optimal TTS: 0.5B beats GPT-4o, 3B beats 405B, 7B beats o1 on MATH-500/AIME24 (summary [snip]). Math only; depends on policy, PRM and difficulty. **ADOPTABLE: partly.**
- **HF blog** [Scaling test-time compute with open models](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) (Dec 2024, not arXiv): Llama 1B/3B with search beat 8B/70B on MATH-500; open code in search-and-learn [snip]. **ADOPTABLE: partly** (math; code in the repo is reusable).
- **Smaller, Weaker, Yet Better** [arXiv 2408.16737](https://arxiv.org/pdf/2408.16737) (ICLR 2025). Under fixed compute, cheap weak-model samples give better training data than strong-model samples (higher coverage/diversity, more false positives) [snip]. **ADOPTABLE: partly** (a data-generation point; needs training).
- **Train-to-test scaling** [arXiv 2604.01411](https://arxiv.org/abs/2604.01411) (2026): with inference cost counted, optimal training shifts to overtraining; no numbers in abstract [abs]. **ADOPTABLE: no** (we do not pretrain).
- **Scaling of capability/efficiency in LRMs** [arXiv 2609.27166](https://arxiv.org/abs/2609.27166) (2026): capability falls exponentially with problem size; larger models gain with diminishing returns; output length power law unchanged by size [abs]. **ADOPTABLE: partly** (warns that our gains shrink on harder tasks; no numbers).
- **Evaluating TTS of General LLM Agents** [arXiv 2602.18998](https://arxiv.org/html/2602.18998) (2026): on a unified search/coding/reasoning/tool benchmark, neither sequential nor parallel scaling reliably helps; "scaling plateau" and "verification gap" [abs]. **ADOPTABLE: warning** (any plan must be measured on our own harness).

### 2g. Debate, multi-agent, mixture-of-agents

- **Multi-agent debate** [arXiv 2305.14325](https://arxiv.org/abs/2305.14325v1) (ICML 2024). Agents critique each other over rounds; gains reported on factuality/reasoning (secondary: up to +8 factuality, +26 reasoning [unv]). **ADOPTABLE: no** for us (see the negative evidence below).
- **Can LLMs self-correct reasoning** [arXiv 2310.01798](https://arxiv.org/pdf/2310.01798): debate only slightly beats self-consistency at equal responses; with equal responses, debate significantly underperforms majority vote [snip]. Also "Cannot Self-Correct Reasoning Yet".
- **MAD systematic eval** [arXiv 2502.08788](https://arxiv.org/abs/2502.08788v2): five MAD methods, nine benchmarks, four models; none beats CoT in >20% of 36 scenarios; worse vs SC at equal calls (summary [snip]). **ADOPTABLE: no.**
- **Debate vs voting, 2025-2026**: [arXiv 2509.05396](https://arxiv.org/html/2509.05396v2) (debate can degrade vs voting; agents agree reflexively) and [arXiv 2605.09618](https://arxiv.org/pdf/2605.09618) (matched-ceiling; debate barely beats voting) [snip].
- **Mixture-of-Agents** [arXiv 2406.04692](https://arxiv.org/pdf/2406.04692). Layers of LLMs each see previous outputs; AlpacaEval 2.0 win rate 65.8% vs GPT-4 Omni 57.5% (v1 abstract [snip]). Multiplier = layers x agents. Used open models of size 7B-70B, not a coding benchmark. **ADOPTABLE: no** for coding; yes only as a check against single-model self-consistency at equal calls.

### 2h. Other (sampling, coding baselines)

- **Agentless** [arXiv 2407.01489](https://arxiv.org/html/2407.01489v2) (FSE 2025). Fixed localize-repair-validate pipeline; 32.00% SWE-bench Lite (v2) at ~USD 0.70 (search summary; v1 27.33%). Not a search method, but a cheap structure that is easy to run small. **ADOPTABLE: yes, as a baseline** (no LLM-decided actions).
- **Code scaling with RM/PRM selection**: [FunPRM 2601.22249](https://arxiv.org/pdf/2601.22249), [CodeScaler 2602.17684](https://arxiv.org/pdf/2602.17684), [TIPCODER 2609.03309](https://arxiv.org/pdf/2609.03309) (Qwen2.5-Coder-7B best-of-8: HumanEval+ 84.15; with an RM selector 88.21 [snip]). Selectors are RMs, not execution. **ADOPTABLE: partly** (needs a trained RM; not zero-shot).
- **Differential / hybrid TTS for code**: [DiffCodeGen 2605.20473](https://arxiv.org/pdf/2605.20473) (Qwen2.5-Coder-Instruct 7B row 29.4 -> 38.9 on LiveCodeBench; columns unverified [snip]); [OpenCodeReasoning-II 2507.09075](https://arxiv.org/pdf/2507.09075) (self-critique on fine-tuned models [unv]). Partly.
- **Agentic coding under a reasoning budget**: [arXiv 2607.22807](https://arxiv.org/html/2607.22807v1) (Gemma 4 and Qwen 3.6 with a 6K reasoning budget, 40-turn and 600-s caps; appendix only [snip]). Closest setting to ours; results not verified.

---

## 3. Where small models gain and where they do not (summary of the above)

- Gains that repeat across papers for small models: verified selection (execution or tests) over many samples (S*, CodeT, Code-A1, LLM Monkeys); difficulty-aware allocation (Snell, Inference Scaling Laws, VG-Search).
- Gains that depend on a strong judge: self-repair with self-feedback (Olausson), debate and voting without verifiers (2502.08788, 2310.01798), LLM-judge selection on hard tasks.
- Saturation is consistent: sampling coverage grows, selection plateaus (Large Language Monkeys, Snell, R2E-Gym, SWE-World, 2602.18998).
- More thinking can hurt on some tasks (2507.14417); budget control (L1, Qwen3 budget) helps only if the budget is tuned per task.
- Almost all headline numbers are on math (MATH, AIME, GSM8K) or on frontier/32B models. For coding at 1-14B, the verified-selection papers (S*, Code-A1) are the closest evidence.

---

## 4. Top 5 ideas for us, ranked by expected value

1. **Execution-verified best-of-N with generated tests (S*-style, small model).** Strongest measured small-model code gain (S*: 3B beats GPT-4o-mini; Code-A1 3B; CodeT +18.8 pts on HumanEval). Cost: N x generation + tests run. Verifier: local isolated sandbox (needs your OK under the standing rule, since the sandbox is the one named exception) or Kaggle runs. Do first.
2. **Sequential repair with real test output and a fixed round cap** (S* sequential part, Self-Debugging, CodeMonkeys serial scaling). Use execution traces, not self-critique (Olausson). Cap rounds; stop when plateau (SWE-World, R2E-Gym).
3. **DeepConf-style confidence filtering and early stop on Qwen3 logprobs.** Reported large token savings (up to 84.7% on AIME, math only) with no training. Cheap to test on Kaggle vLLM; must be measured on our code tasks.
4. **Per-task compute allocation** (Snell; VG-Search; Inference Scaling Laws): choose beam vs best-of-N and the budget by estimated difficulty; this decides whether a 4B or 7B plus compute beats a bigger model at our budget. Needs a difficulty estimate and a fixed harness.
5. **Thinking-budget sweep with an inverse-scaling check** (Qwen3 budget knob, L1 budget idea, 2507.14417, 2602.18998). Free to run on Kaggle; measure accuracy vs tokens on our tasks; stop at the knee. Do not assume more thinking helps.

Rejected for now: debate and mixture-of-agents (negative compute-matched results), rStar-Math and ThinkPRM training (need fine-tuning GO), LATS and CodeTree numbers (frontier models).

---

## 5. Checklist

DONE (sources read via abstract, search excerpt, or secondary summary; see grades above)
- [x] Thinking budgets / budget forcing: s1, L1, Qwen3 report, BudgetThinker, budget guidance, survey, inverse scaling, overthinking (title only)
- [x] Self-consistency and repeated sampling: original, Large Language Monkeys, DeepConf, Inference Scaling Laws
- [x] Tree / MCTS / beam: SWE-Search, CodeTree, LATS, rStar-Math, Snell beam search, VG-Search, S*
- [x] Iterative refinement with execution feedback: Self-Debugging, Self-repair, S*, CodeMonkeys, Thinking Longer Not Larger, SWE-World
- [x] Debate and multi-agent: Du et al., Huang et al., MAD systematic eval, matched-ceiling and 2025 debate-vs-voting, MoA
- [x] Parallel thinking: ParaThinker; DeepConf early stop
- [x] Compute-optimal and small vs large: Snell, Inference Scaling Laws, 1B vs 405B, HF blog, Smaller Weaker Yet Better, T2 scaling, 2609.27166, 2602.18998
- [x] Verifiers / PRMs for code: ThinkPRM, SWE-RM, R2E-Gym, FunPRM, CodeScaler, TIPCODER, Code-A1
- [x] 2026 coding agent TTS: 2604.16529, SWE-World, Code-A1, DiffCodeGen, 2607.22807 (partial)

NOT FOUND or NOT VERIFIED
- [ ] NOT FOUND: any small-model (7B/14B) tree search on SWE-bench Verified with a published number (closest: SWE-World and SWE-Search, both larger or unclear).
- [ ] NOT FOUND: Qwen3 thinking-budget vs accuracy curves (report abstract has none; full PDF not read).
- [ ] NOT VERIFIED: 2604.10739 (overthinking), 2604.02460 (single vs multi-agent, blog only), 2609.03309 (TIPCODER), 2607.22807 (budget agent study), 2605.20473 (DiffCodeGen columns), 2604.16529 (numbers from secondary pages), 2511.00086 (optimizable graph, lead only), 2512.02008 (lead only).
- [ ] NOT VERIFIED: all snippet-level numbers (marked [snip]) against the PDFs.
- [ ] NOT SEARCHED: speculative thinking / speculative decoding for reasoning; ReST-style search with small verifiers; the 14B-verifier results in full.
- [ ] NOT MEASURED: any cost multiplier on our hardware or on Kaggle. All multipliers above are the papers' own accounting.

Sources (main): arXiv IDs listed above with links; HF blog https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute.

---

<!-- source: results/raw/research-20261015-smallmodel/D_scaffolds_memory_steering.md -->
# Part D. Scaffolds, memory, prompt optimisation, steering, constrained decoding (small open models)

Date: 2026-10-15. Scope: methods that improve a small model (Qwen family, 1-14B) without changing its weights, or with tiny plug-ins. Adoptability is judged for: one 6 GB RTX 3050 + 15 GB RAM locally, free Kaggle T4 x2 / TPU v5e-8, local sandbox that runs tests, no paid APIs, no strong model at runtime, and the LAYA decision head (scores candidate actions from the small model's hidden states).

Method and caveats: web search snippets and abstract-level pages only. No full papers were opened, nothing downloaded, no code run. "(self-reported)" = authors' claim; "(secondary)" = third-party page; "(unverified)" = could not tie to a primary source in this pass. Within each section, newest first.

---

## 1. Scaffolds, tool design, context management

- **SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents** (arXiv 2601.16746, Jan 2026). https://arxiv.org/abs/2601.16746
  - What: goal-conditioned, line-level pruning of tool output (grep/cat) by a 0.6B "skimmer" in middleware.
  - Gains: 23-54% token reduction on SWE-bench Verified with success kept or raised (self-reported). Up to 14.84x on LongCodeQA single-turn.
  - Cost: one extra small-model call per file read. Model card lists MIT (secondary, HF card).
  - ADOPTABLE? partly: a rule-based line filter is free; the learned skimmer is one more model on the 3050. Reuse: its line scorer has the same shape as a candidate scorer.

- **The Complexity Trap: observation masking vs LLM summarization** (arXiv 2508.21433, Aug 2025, rev. Oct 2025). https://arxiv.org/abs/2508.21433
  - What: mask old environment observations in SWE-agent instead of summarizing them with an LLM.
  - Gains: about half the cost of the raw agent; solve rate matched or slightly above summarization. Qwen3-Coder 480B: 53.8% to 54.8% (self-reported, SWE-bench Verified). Hybrid saves a further 7-11% cost.
  - Cost: zero extra model calls.
  - ADOPTABLE? yes: a few lines in our sandbox loop. Caveat: large models, one benchmark, authors' own scaffold.

- **Agentless: Demystifying LLM-based SE Agents** (arXiv 2407.01489, Jul 2024; v2 is the three-phase version). https://arxiv.org/abs/2407.01489 (html v2: https://arxiv.org/html/2407.01489v2)
  - What: fixed pipeline with no agent loop and no model-chosen tool sequence. Hierarchical localization (file, then class/function, then edit line), several diff candidates, self-written reproduction tests to validate and re-rank.
  - Gains: 32.00% SWE-bench Lite at $0.70 (v2); 27.33% at $0.34 (v1) (self-reported). Paper's models are API models; small-model results not seen.
  - ADOPTABLE? yes: the shape suits small models because the model never has to plan the tool sequence. The re-rank step is the slot where LAYA would score candidates.

- **SWE-agent: Agent-Computer Interfaces** (arXiv 2405.15793, May 2024). https://arxiv.org/abs/2405.15793
  - What: interface design rather than model change: windowed file viewer, compact search results, linter-gated edits, explicit error messages.
  - Gains: 12.5% pass@1 SWE-bench with GPT-4 Turbo (paper; secondary gives 12.47%).
  - ADOPTABLE? yes: design rules are free and model-agnostic; apply them to our sandbox tools.

- **mini-SWE-agent, bash-only track** (repo: https://github.com/SWE-agent/mini-SWE-agent; leaderboard: https://www.swebench.com/bash-only.html).
  - What: minimal ReAct loop with bash only; LiteLLM backend; local, Docker or Singularity environments.
  - Gains: secondary, conflicting figures for big models: MiniMax M2.5 75.8% or 80.2%; DeepSeek-V3.2 70.0% (unverified against the live leaderboard). No 1-14B results found.
  - ADOPTABLE? yes as a harness (cheap to run against a local model); the scores themselves are not evidence for small models.

- **CodeAct: Executable Code Actions** (arXiv 2402.01030, 2024). https://arxiv.org/abs/2402.01030
  - What: Python code as the action space; interpreter feedback drives self-debugging.
  - Gains: up to 20% higher success than JSON/text actions over 17 LLMs on API-Bank and the authors' benchmark (self-reported).
  - ADOPTABLE? partly: helpful action space, but executing model code widens the sandbox risk surface.

- **Repository map (Aider)**. https://aider.chat/2024/05/22/swe-bench-lite.html
  - What: tree-sitter definitions and references, ranked by personalized PageRank over a file graph, cut to a token budget (method description from third-party analysis, DeepWiki).
  - Gains: Aider's 26.3% SWE-bench Lite used frontier models; no repo-map ablation found.
  - ADOPTABLE? partly: no model call and cheap to build; must be measured with our small model.

- **Negative data point: small general models on repo-level tasks** (EffGen, arXiv 2602.00887, 2026). https://arxiv.org/abs/2602.00887
  - Qwen2.5-Instruct 7B / 14B / 32B on SWE-Bench Lite: 0.67% / 1.67% / 2.33% raw; 2.67% / 4.33% / 5.67% with its tooling (own harness, unverified).
  - Implication: a scaffold alone may not reach useful levels at 7B; set expectations and measure.

- **Seed-Coder-8B-Instruct** (arXiv 2506.03524, Jun 2025). https://arxiv.org/abs/2506.03524
  - 19.2% SWE-bench Verified under Agentless (self-reported). Not a Qwen model and a trained product, not a plug-in; evidence that the Agentless shape works at about 8B.

- **Context compression, older**: LLMLingua (arXiv 2310.05736, EMNLP 2023; https://arxiv.org/abs/2310.05736): up to 20x token compression with little loss on GSM8K/BBH. LongLLMLingua (arXiv 2310.06839; https://arxiv.org/abs/2310.06839): up to ~17-21% gain at ~4x fewer tokens (versions differ). Not code-specific.
  - ADOPTABLE? partly: prefer code-aware pruning (LongCodeZip below, SWE-Pruner above).
- **LongCodeZip** (arXiv 2510.00446, ASE 2025). https://arxiv.org/abs/2510.00446
  - Function-level chunk ranking by conditional perplexity under a token budget; up to 5.6x compression without task loss on completion, summary, QA (self-reported). Needs a LM for perplexity, which a small Qwen can supply locally. ADOPTABLE? partly; no SWE-bench number seen.

---

## 2. Memory and experience reuse

- **Subtask-level memory for SWE agents** (arXiv 2602.21611, Feb 2026). https://arxiv.org/html/2602.21611v1
  - Insights stored per subtask; mean Pass@1 +4.7 points over vanilla across four backbones, largest +6.8 (Gemini 2.5 Pro) (self-reported). ADOPTABLE? partly: needs a memory store and a retrieval step; gains grow with step count.

- **SWE-ContextBench** (arXiv 2602.08316, Feb 2026). https://arxiv.org/abs/2602.08316
  - Benchmark for context learning in coding. Correctly selected experience improves resolution and cuts cost; unfiltered or wrongly selected experience gives little or negative benefit. ADOPTABLE? yes as a design warning: retrieval precision matters more than volume. Reuse: a learned gate that scores whether a retrieved memory applies is the LAYA idea applied to memory.

- **MemGovern** (Jan 2026; arXiv ID not captured). Experience cards from human engineering material, agentic retrieval, +4.65% SWE-bench Verified resolution (unit unclear). (unverified)

- **SWE-Exp** (listed as arXiv 2507.23361; unverified ID). Experience bank from successful and failed repair attempts; 41.6% Pass@1 on SWE-bench Verified under open-source frameworks (self-reported; model and framework not checked).

- **ReasoningBank: scaling agent self-evolving with reasoning memory** (arXiv 2509.25140, Sep 2025; ICLR 2026 per third party). https://arxiv.org/abs/2509.25140
  - Distills strategies from self-judged successes and failures and retrieves them at test time. SWE-bench Verified 57.4% vs 54.0% no-memory with Gemini-1.5 Pro; -2.8 steps (secondary; numbers not checked in the paper). Code reportedly at github.com/google-research/reasoning-bank (secondary).
  - ADOPTABLE? partly: swap the self-judge for our sandbox test result, which is a stronger label than self-judgement.

- **ACE: Agentic Context Engineering** (arXiv 2510.04618, Oct 2025; ICLR 2026 per secondary). https://arxiv.org/abs/2510.04618
  - Structured, growing playbook with small incremental edits instead of full rewrites (avoids brevity collapse). +10.6% on agent benchmarks (secondary). ADOPTABLE? partly: good pattern for a notes file; gains are on non-SWE tasks.

- **Agent Workflow Memory** (arXiv 2409.07429, Sep 2024; ICML 2025 poster). https://arxiv.org/abs/2409.07429
  - Induces reusable workflows from past runs and injects them. Relative success +51.1% WebArena and +24.6% Mind2Web (relative, not points). Web tasks, not code. ADOPTABLE? partly.

- **Synapse: Trajectory-as-Exemplar Prompting with Memory** (arXiv 2306.07863, ICLR 2024). https://arxiv.org/abs/2306.07863
  - Note: the brief's "2406" should read 2306. Filters screen state to fit more exemplars; embedding retrieval of exemplars. MiniWoB++ 99.2% average; Mind2Web +56% relative step success (self-reported). A different 2026 paper titled Synapse exists (arXiv 2601.02744, not checked). ADOPTABLE? partly: retrieval idea yes, desktop-web tasks no.

- **ExpeL: LLM Agents Are Experiential Learners** (arXiv 2308.10144, AAAI-24). https://arxiv.org/abs/2308.10144
  - Natural-language insights from success/failure comparisons; ADD/UPVOTE/DOWNVOTE/EDIT on the insight list. Numbers not checked. ADOPTABLE? partly.

- **Reflexion** (arXiv 2303.11366, NeurIPS 2023). https://arxiv.org/abs/2303.11366
  - Verbal reflection on failure stored in an episodic buffer; HumanEval 91% pass@1 vs 80% for GPT-4 (abstract). ADOPTABLE? yes in cheap form: our sandbox test output is the feedback signal. Caveat: self-reflection quality on a 7B model is untested.

- **CTIM-Rover, "From Knowledge to Noise"** (arXiv 2505.23422, May 2025). https://arxiv.org/abs/2505.23422 - title signals that episodic memory can add noise; numbers not checked.

---

## 3. Prompt and program optimisation

- **GEPA: Reflective Prompt Evolution Can Outperform RL** (arXiv 2507.19457; v2 Feb 2026; ICLR 2026 oral per secondary). https://arxiv.org/abs/2507.19457
  - Reflective prompt mutation from rollout feedback, keeping a Pareto front of prompts per instance.
  - v1 (Qwen3-8B): up to +19% over GRPO with 24k rollouts, up to 35x fewer rollouts; beats MIPROv2. v2 abstract: average +6%, up to +20% over GRPO. Numbers differ by version; cite the version.
  - Cost: many rollouts plus reflection calls; rollouts can run locally in our sandbox.
  - ADOPTABLE? partly: prompt-only; no SWE-bench result seen. Reuse: Pareto selection resembles ranking candidates.
- **GEPA on small models, code task** (arXiv 2601.08884, Jan 2026; OpenACC pragma generation, not SWE). https://arxiv.org/abs/2601.08884 (secondary summary only). Claims to avoid post-training costs.
- **MIPRO: Optimizing Instructions and Demonstrations for Multi-Stage LM Programs** (arXiv 2406.11695, Jun 2024; EMNLP 2024). https://arxiv.org/abs/2406.11695
  - Joint instruction and demo search; up to ~13% accuracy gain with Llama-3-8B across programs (v2). ADOPTABLE? partly; needs a metric and a rollout budget.
- **TextGrad** (arXiv 2406.07496, Jun 2024). https://arxiv.org/abs/2406.07496 - text critiques as gradients; GPT-4o zero-shot GPQA 51% to 55% (abstract). Nature version cited by repo README (not opened). ADOPTABLE? no for now: the critic would need to be a strong model; untested with a small critic.
- **DSPy** (arXiv 2310.03714, Oct 2023). https://arxiv.org/abs/2310.03714 - compiles prompt pipelines; small models (llama2-13b, T5-770M) gain in the paper's examples (e.g. one math task 9% to 47%, as summarized). ADOPTABLE? partly: a framework dependency; add only if it beats a hand-written prompt.
- **OPRO: LLMs as Optimizers** (arXiv 2309.03409, ICLR 2024). https://arxiv.org/abs/2309.03409 - LLM proposes prompts from scored history; up to 8% GSM8K and 50% BBH over human prompts (PaLM 2). ADOPTABLE? no: superseded by reflective methods; needs a strong optimizer.
- **Vendor claim, unverified**: Arize prompt learning on coding agents, +10-15% on SWE-Bench (frontier models). https://arize.com/docs/ax/cookbooks/prompt-learning/optimizing-coding-agent-prompts-for-execution

---

## 4. Activation steering, probes, value heads, adapters, merging

**Probes on frozen states (closest to LAYA):**
- **Latent Programming Horizons in Coding Agents** (arXiv 2607.05188, Jul 2026; Silva, Tu, Monperrus). https://arxiv.org/abs/2607.05188
  - Residual-stream probes over agent trajectories: correctness AUC up to 0.83 across two models and two benchmarks; predict whether the program parses and how many tests fail; transfer across benchmarks; future-edit signal above chance up to ~25 steps ahead (self-reported).
  - ADOPTABLE? partly: no weight change; needs hidden-state capture (our Qwen state pipeline). Reuse: this is LAYA applied per agent step.
- **Code correctness is linearly decodable before generation** (arXiv 2606.14530, Jun 2026). https://arxiv.org/abs/2606.14530
  - Qwen3-4B-Instruct-2507, 444 LiveCodeBench tasks; logistic probe on the last prompt token predicts first-attempt pass. AUC 0.931 in abstract vs 0.881 in body (versions or setups disagree). With prompt length regressed out: 0.911 vs length baseline 0.754 (abstract).
  - ADOPTABLE? partly: the closest published evidence for a frozen-state head. Our protocol must keep the mandatory linear-probe baseline and a length control.
- **On the robustness of LLMs' internal representation of code correctness** (arXiv 2608.08266, Aug 2026). https://arxiv.org/abs/2608.08266 - only the introduction was seen; no results.
- **Probe-based steering of agentic behaviour** (arXiv 2603.16335, Mar 2026). https://arxiv.org/abs/2603.16335 - SAE-latent probes turned into steering vectors; autonomy steering alpha=2 gives Cohen's d=1.01 over 1,800 rollouts (self-reported). Authors warn that probe R^2 measures correlation, not causal effect.
- **Predicting steerability from early hidden states** (arXiv 2606.11599, Jun 2026). https://arxiv.org/abs/2606.11599 - classifier on early-generation states predicts under-steer, success or over-steer; ~0.7 macro-F1 on unseen concepts (not code).
- **Steering code LLMs for language and library choice** (arXiv 2603.23629, Mar 2026). https://arxiv.org/abs/2603.23629 - difference-in-means layer vectors steer ecosystem choice in three open code LLMs; too-strong steering reduces quality. No correctness gain claimed. ADOPTABLE? no for pass rate.
- **Reward-hacking steering in a coding LLM** (escholarship, no arXiv ID; unverified). Subtracting a "cheating direction" cut cheating 7.8% to 3.8% (prefill) or 1.0% (full), with solve rate 23.3% to 12.6% or 13.5% (secondary). Shows steering's cost.
- **G-ACT** (arXiv 2506.18887, Jun 2025, preprint under review). https://arxiv.org/abs/2506.18887 - clustered steering directions with per-layer probes refined online; tested on LLaMA-3.2 3B; language choice for C++ code; modest probe overhead.
- **Steering for type prediction in code LLMs** (Lucchetti and Guha, arXiv 2404.01903, 2024). https://arxiv.org/abs/2404.01903
- **Representation Engineering** (arXiv 2310.01405, Oct 2023). https://arxiv.org/abs/2310.01405 - population-level directions; background method, safety-focused.

**Verifiers and value heads:**
- **LLM-as-a-Verifier** (arXiv 2607.05391, Jul 2026). https://arxiv.org/abs/2607.05391 - large verifier under best-of-N: +9.5 points SWE-Bench Lite, +18.5 AIME, +21.3 HMMT (self-reported). A ceiling, not a small-model method.
- **Small LLMs as code critics with a value head** (arXiv 2510.23083, Oct 2025). https://arxiv.org/abs/2510.23083 - Phi-4 family; sigmoid head; fine-tuned, not frozen; over 20% gain selecting the best of N code (self-reported). ADOPTABLE? partly: fine-tuning needed.
- **ThinkPRM** (arXiv 2504.16828, Apr 2025). https://arxiv.org/abs/2504.16828 - generative verifier trained on 8K process labels; beats discriminative PRMs with ~100x labels by up to 8 macro-F1 on ProcessBench; LiveCodeBench best-of-N gains (abstract). Training plus per-candidate generation cost.
- **GenPRM** (arXiv 2504.00891, Apr 2025). https://arxiv.org/abs/2504.00891 - step-level CoT verification with code execution.

**LoRA routers, merging:**
- **LoRA-Mixer** (arXiv 2507.00029; ICLR 2026). https://arxiv.org/abs/2507.00029 - routes task LoRA experts into attention projections; plug-and-play over frozen public LoRAs. Gains +7.61 / +4.88 / +3.08 over base on GSM8K / HumanEval / MedQA (self-reported). Code linked in abstract.
- **TT-LoRA MoE** (arXiv 2504.21190, Apr 2025). https://arxiv.org/abs/2504.21190 - frozen experts, router-only training; a single-layer router on the final representation worked best (their ablation).
- **Hard-Routed MoR-LoRA** (arXiv 2606.31413, 2026). https://arxiv.org/abs/2606.31413 - one frozen LoRA per input; LLaMA 3B and 8B; best among learned-composition baselines (self-reported).
- **MIXLORA** (arXiv 2404.15159, 2024). https://arxiv.org/abs/2404.15159 - top-k LoRA experts in FFN; about 9% over PEFT in multi-task (self-reported).
- **Model merging for code**: "Multi-task Code LLMs: Data Mix or Model Merge?" (arXiv 2601.21115, Jan 2026). https://arxiv.org/abs/2601.21115 - on Qwen2.5-Coder-1.5B, Linear/TIES/DARE/DELLA merges underperformed SFT in the visible table (large negative averages; table garbled, unverified). VisCodex (arXiv 2508.09945, Aug 2025) transferred a code task vector to a VLM; MMCode pass@1 4.2 to 6.5 in one appendix setting. Community Qwen2.5-Coder merges have no benchmarks.
- ADOPTABLE? LoRA routing: partly, and only with a user GO for training (one adapter per Qwen model is the current goal). A router that picks an adapter from the state is the LAYA head's job applied to adapters. Merging: no for now (negative evidence).

---

## 5. Constrained decoding and tool-call reliability

- **Don't Adapt Small LMs for Tools; Adapt Tool Schemas** (arXiv 2510.07248, Oct 2025). https://arxiv.org/abs/2510.07248 - logit masking to valid tool names: Qwen2.5-7B single-turn tool selection on RoTBench 49.5% to 78.1% (self-reported).
- **Repair, Not Improvement: decomposing constrained decoding in tool-call abstention** (arXiv 2608.13959, Aug 2026). https://arxiv.org/abs/2608.13959 - a grammar over tool names removes the "no tool" option; JSON prompting costs ~3.9 points and decoding enforcement a further ~1.6 (their claim; function calling excluded).
- **The Constraint Tax** (arXiv 2605.26128, May 2026). https://arxiv.org/abs/2605.26128 - Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B: schema validity 61.5% to 100%, but answer accuracy 19.7% to 11.0%; Qwen2.5-1.5B calendar task 91.5% (prompt-only JSON) vs 48.0% (hard tool schema). Suggested pattern: free reasoning, constrain only the final packaging step.
- **Constrained decoding eliminates structural failures but leaves a scale-dependent semantic gap** (arXiv 2609.23742, Sep 2026). https://arxiv.org/abs/2609.23742 - from search summary only.
- **Tool suppression under JSON-schema + tool calling** (HF papers 2606.25605, Jun 2026). https://huggingface.co/papers/2606.25605 - models stop calling tools while still producing valid schema output; hypothesis: grammar masks make tool-call tokens unreachable; two-pass fix proposed.
- **XGrammar** (arXiv 2411.15100, Nov 2024; v3 May 2025; MLSys 2025). https://arxiv.org/abs/2411.15100 - up to 100x lower per-token CFG latency vs prior engines (self-reported). 2026 comparisons disagree: arXiv 2608.03065 found XGrammar far slower than llguidance on Java grammar masks (309,514 vs 1,352 us per token, Llama 3); arXiv 2605.08737 found identical parse rates and close latency. Benchmark on our own grammars.
- **ToolDec** (arXiv 2310.07075, Oct 2023). https://arxiv.org/abs/2310.07075 - finite-state constraints for tool-call syntax.
- ADOPTABLE? yes for tool-call syntax and final patch packaging; keep an abstain option; do not constrain free reasoning. Reuse: legal-option masking is the same operation LAYA uses over candidate actions.

---

## Top 5 ranked by expected value for us

1. **Fixed Agentless-style pipeline** (localize, sample patches, validate and re-rank with our sandbox tests), with ACI tool rules. LAYA plugs into the localize and re-rank steps. Weight-free, cheap, and the sandbox provides a truth signal.
2. **Observation masking first** (Complexity Trap), then rule-based line pruning of tool output (SWE-Pruner idea) only if context is still the bottleneck. Near-zero cost, measured on SWE-bench Verified.
3. **Probes on frozen hidden states** for candidate scoring and first-attempt pass/fail (2606.14530, 2607.05188). Closest published evidence for LAYA; keep the mandatory linear-probe baseline and length control.
4. **Execution-verified experience memory** (Reflexion and ReasoningBank style, with sandbox test results as labels and a retrieval precision gate per SWE-ContextBench). Medium value: gains reported mostly for large models; noise risk.
5. **Constrained decoding for tool-call syntax and final patch packaging only**, with an abstain option kept (Constraint Tax warning). Cheap; format gains measured; semantic-loss risk.

Lower: GEPA (prompt-only, rollout-heavy); steering (no pass-rate evidence); merging (negative evidence); LoRA routers (needs training, user GO first).

---

## Checklist

DONE
- 1. Scaffolds, ACI, Agentless, mini-SWE-agent, CodeAct, repo map, context compression: DONE (snippets)
- 2. Memory and experience reuse: DONE (snippets)
- 3. Prompt and program optimisation: DONE (snippets)
- 4. Steering, probes, value heads, LoRA routers, merging: DONE (snippets)
- 5. Constrained decoding and tool-call reliability: DONE (snippets)

NOT FOUND
- Bash-only / mini-SWE-agent SWE-bench numbers for models at 1-14B (only frontier and MoE figures; secondary and conflicting)
- Controlled repo-map ablation with a small model
- GEPA, MIPRO or DSPy results on SWE-bench with small models
- Steering shown to raise SWE-bench pass rate (only style and language-choice evidence)
- Frozen-backbone value head for code with SWE-bench evidence (2510.23083 fine-tunes)
- Qwen3-8B / 14B scaffold comparison on SWE-bench Verified
- Merged Qwen-Coder evaluation on recent benchmarks
- Full-text verification: none of the figures above was checked against a full paper; treat all numbers as to-be-measured

---

<!-- source: results/raw/research-20261015-smallmodel/E_self_improvement.md -->
# Part E. Self-improvement and cheap learning for small models (Qwen 1-14B, coding/agent)

Date: 2026-10-15. Method: web search and abstract fetches only. No downloads, no code run.
Labels: **[abs]** = abstract page fetched; **[snip]** = search snippet or third-party summary only; **[unverified]** = number not checked against the paper's tables.
Setting for "ADOPTABLE?": compute on Kaggle T4 x2 / TPU only (CLAUDE.md standing rule); local RTX 3050 for sandbox and writing; no paid APIs; strong model used once offline for data.

## 2026 (newest first)

- **Behavioral Self-Rewarding for SWE agents (BSR)**, arXiv 2610.12598 (Oct 2026) [abs]. Policy compares its own trajectories with the same verifier outcome and adds a bounded self-judgment reward on top of binary RLVR. Models: Qwen3-14B, Qwen3-32B; SWE-bench Verified and Live. Headline gain: "consistent improvements", no numbers in abstract. Data/compute: not stated.
  ADOPTABLE? partly. Idea is cheap, but the judge is the policy, and the models tested are above our 14B ceiling for Kaggle T4 RL.
- **Socratic-SWE**, arXiv 2606.07412 (Jun 2026) [snip]. Traces are distilled into skills; skills drive synthesis of repair tasks, validated by execution. Headline: 50.40% SWE-bench Verified after 3 iterations [unverified]. Model size not retrieved.
  ADOPTABLE? partly. Closed-loop task generation fits us, but the headline comes from an unknown base model and budget.
- **Hindsight Hint Distillation (HHD)**, arXiv 2605.11556 (May 2026) [snip]. Short hints from the model's own failed rollouts scaffold successful rollouts, then the model distills them without hints. Claim: +8% absolute on SWE-bench Verified vs about +2% for RFT and trajectory-synthesis baselines [unverified]. Needs no CoT annotation.
  ADOPTABLE? partly. Cheap, no teacher at runtime, but base size and tables are unchecked.
- **Improving Small LMs for Code Generation with RLVR**, arXiv 2605.30478 (May 2026) [abs]. Qwen3-0.6B and Llama3.2-1B, LoRA, GRPO and GSPO, MBPP. Reward variants: unit-test only, lint only (Ruff), combined. Headline: up to +13 pts pass@1 with combined reward. Warning: lint-only reward drives short code that cuts lint errors without correctness. Data and compute: not stated.
  ADOPTABLE? yes for the reward-design lesson and a small-model LoRA test; the numbers are not yet usable.
- **Code-A1**, arXiv 2603.15611 (Mar 2026) [snip]. Separate code and test models trained with opposing rewards to stop self-generated tests from becoming trivial. Qwen2.5-Coder. Numbers not retrieved.
  ADOPTABLE? partly. Needs two models in RL; the co-evolution idea is worth a later test.
- **EGCA: Execution-Grounded Credit Assignment for GRPO**, arXiv 2603.16158 (Mar 2026) [snip]. Assigns GRPO advantage only to tokens aligned with execution traces versus a reference. Listed under an ACL 2026 link [unverified venue].
  ADOPTABLE? partly. Needs a reference solution per task, which our synthetic tasks would have.
- **CoSPlay**, arXiv 2605.23491 (May 2026) [snip]. Training-free cooperative self-play over code and self-written tests at test time. Numbers not retrieved.
  ADOPTABLE? yes as an inference-time option with no training; not a training route.
- **FunPRM**, arXiv 2601.22249 (Jan 2026) [snip]. Function-as-step process reward model for Best-of-N code selection.
  ADOPTABLE? partly. Needs a PRM trained from execution outcomes; see section 5.
- **Scale-SWE**, arXiv 2602.09892 (Feb 2026) [abs, v2]. Sandboxed multi-agent pipeline builds environments and tests from 6M PRs over 5200 repos, giving 100k verified SWE instances (20k released so far). Qwen3-30B-A3B distilled from 71k DeepSeek-V3.2 trajectories reports 64 resolve rate on SWE-bench Verified [abs and model card; unverified]. Compute: not stated.
  ADOPTABLE? partly. Dataset is large and mostly unreleased; a 30B-A3B student is above our local budget but the pipeline is a template.

## 2025

- **SWE-Bench++**, arXiv 2512.17419 (Dec 2025) [snip]. Generates SWE benchmarks from repos; uses the 5,016 SWE-smith trajectories as the baseline, adds 200/400/800 extra trajectories for scaling. Outcomes not retrieved.
  ADOPTABLE? partly. Useful for a data-scaling curve design.
- **SkyRL-Agent**, arXiv 2511.16108 (Nov 2025) [snip]. SA-SWE-32B from Qwen3-32B, RL only, 39.4% Pass@1 on SWE-bench Verified; 2x8 H100 for training; code public (github.com/NovaSky-AI/SkyRL). Project page: SkyRL-Agent-14B-v0 from Qwen3-14B, 18.0% to 21.6% with about 300 training samples [snip].
  ADOPTABLE? partly. Code is open. Cheapest sized recipe is the 14B sample count, but a full-run budget is not stated.
- **DeepSWE-Preview** (Agentica / Together AI), HF agentica-org/DeepSWE-Preview [snip; date unverified]. Qwen3-32B, RL only, 4,500 R2E-Gym tasks, 64 H100 for 6 days (about 9,200 H100-h, our arithmetic). Pass@1 42.2% without test-time scaling, 59% with it. Most gain in first 200 steps (23% to 42.2%).
  ADOPTABLE? no at 32B; the early-steps curve is useful evidence that short RL gives most of the gain.
- **SWE-RL**, arXiv 2502.18449 (Feb 2025; v2 Dec 2025) [snip]. Meta. GRPO on GitHub PR data with a similarity reward to the oracle patch. Llama3-SWE-RL-70B: 41.0% SWE-bench Verified. Out-of-domain gains; SFT on the same data degraded results on average. github.com/facebookresearch/swe-rl.
  ADOPTABLE? partly. Similarity reward avoids execution, but 70B is out of range; the SFT-degradation finding is a warning for us.
- **DeepCoder-14B-Preview** (Agentica / Together AI), HF agentica-org/DeepCoder-14B-Preview [snip; date unverified]. From R1-Distill-Qwen-14B, GRPO+ (DAPO-inspired, offline difficulty filter, overlong filter, Clip-High), 24K verified problems, 32 H100 for about 2.5 weeks (about 13,400 H100-h, our arithmetic). LiveCodeBench v5 60.6 pass@1 [snip].
  ADOPTABLE? no for compute; yes for recipe details (filtering, context staging).
- **AceCoder**, arXiv 2502.01718 (ACL 2025; v4 May 2025) [snip]. Test-case synthesis (AceCode-87K). RL from Qwen2.5-Coder-7B base: 80 steps, 48 H100-h; +25% HumanEval+ and +6% MBPP+ (v4 claim) [unverified]. v2 reported only 0.7 avg gain from Instruct RL.
  ADOPTABLE? yes, partly: 7B base RL under 50 H100-h is the closest measured cheap-RL data point; needs a Kaggle budget test to confirm.
- **Absolute Zero (AZR)**, arXiv 2505.03335 (May 2025; v3 Oct 2025; NeurIPS 2025) [abs/snip]. Zero external data: model proposes and solves induction, abduction and deduction code tasks, checked by an executor. Qwen2.5-7B-Coder. Authors claim SOTA among code-trained RLVR models. Third-party wiki [unverified]: LiveCodeBench v5 19.9 to 31.7 for the 7B coder.
  ADOPTABLE? partly. Zero-data is attractive; the reported gain must be checked in the PDF tables first.
- **rStar-Coder**, arXiv 2505.21297 (NeurIPS 2025) [abs/snip]. 418K verified competitive problems, 580K long-reasoning solutions, generated tests with mutual verification. Qwen2.5-7B LiveCodeBench 17.4 to 57.3; 14B 23.3 to 62.5 (abstract). Data: HF microsoft/rStar-Coder. This is SFT data, not RL.
  ADOPTABLE? partly. Strong evidence for synthetic verified problems; dataset download is over 1 GB and needs the user's yes.
- **OpenThoughts** (OpenThinker3-7B), arXiv 2506.04178 (ICLR 2026) [snip]. Qwen2.5-7B-Instruct fine-tuned on 1.2M examples with a QwQ-32B teacher. AIME25 53, LCB 51, GPQA 54; +15.3/17.2/20.5 pts over R1-Distill-Qwen-7B. More than 1,000 ablations on sourcing, mixing, filtering, teacher choice. Datasets at openthoughts.ai.
  ADOPTABLE? partly. Gives the filter and teacher ablation list; 1.2M examples is too large for us, the filters are not.
- **Dr. GRPO**, arXiv 2503.20783 [snip]. Removes GRPO length and std normalization; authors report 43.3% AIME24 with a 7B base. Also finds Qwen2.5 bases already reason without templates.
  ADOPTABLE? yes as a free change to the GRPO loss; cheap to apply.
- **DAPO**, arXiv 2503.14476 [unverified; not fetched]. Dynamic sampling, token-level loss, overlong shaping. Described only through secondary summaries.
- **1-shot RLVR**, arXiv 2504.20571 (NeurIPS 2025) [abs/snip]. One math example: Qwen2.5-Math-1.5B MATH500 36.0 to 73.6, six-benchmark average 17.6 to 35.7. Entropy loss is critical. Math only.
  ADOPTABLE? partly. Cheapest possible RL run; coding transfer untested. Test on 2-8 coding tasks first.
- **Tina**, arXiv 2504.15777 (ICLR 2026) [abs/snip]. LoRA RL on a 1.5B base; AIME24 43.33 Pass@1; about $9 post-training and eval (authors' estimate). Math only. Code and logs open.
  ADOPTABLE? yes as the cost template for LoRA RL on a 1.5-4B model; the task is math, not code.
- **PRIME** (implicit PRM), arXiv 2502.01456 [snip]. Online process reward trained from outcome labels only. Qwen2.5-Math-7B: +15.1% average over SFT; 2.5x sample efficiency vs outcome-only RL. github.com/PRIME-RL/PRIME.
  ADOPTABLE? partly. Needs only outcomes, which fits execution labels; math-tested.
- **Execution-grounded PRMs for code**: ORPS "Reasoning Through Execution" (arXiv 2412.15118, ICML 2025) [snip]; CodePRM (ACL Findings 2025) [snip]; PRLCoder, line-level execution-labelled PRM (arXiv 2502.01715, EMNLP 2025) [snip]; ReCode (ACL 2026) [snip]. Execution gates the neural reward to limit hacking. Numbers not retrieved.
  ADOPTABLE? partly. Best path if we want a learned verifier; more compute than plain rejection sampling.
- **SWE-smith**, arXiv 2504.21798 (NeurIPS 2025) [abs/snip]. 50k instances from 128 repos via bug injection and execution-checked tests. SWE-agent-LM-32B: 40.2% SWE-bench Verified; 5,016 Claude 3.7 Sonnet trajectories. Scaling figure values not retrieved; a third-party table gives SWE-agent-LM-7B 15.2% [unverified]. Authors say they did not test RL.
  ADOPTABLE? yes for the bug-injection pipeline (sandbox runs locally, tests are the verifier). The 32B result is out of range; the 7B number is unverified.
- **R2E-Gym**, arXiv 2504.07164 [snip]. SWE-Gen recipe builds about 8.1K executable environments from commits; hybrid verifiers. 32B pass@1 34.4% SWE-bench Verified (one summary); 51% best-of-26 hybrid (secondary) [unverified].
  ADOPTABLE? yes for the environment-building recipe; the sandbox would need the same tooling.
- **Nebius SWE-rebench**, arXiv 2505.20411 (v2 arXiv 2602.23866) [snip]. Automated pipeline: more than 21k interactive Python SWE tasks, execution-verified installs. The Nebius paper is a task-collection pipeline; I did not find a SWE-agent-LM trajectory result attributed to Nebius. Correction to the request wording.
  ADOPTABLE? yes as a task source, subject to the dataset download rule.
- **SWE-Gym**, arXiv 2412.21139 (Dec 2024) [snip]. 2,438 executable Python tasks; 491 trajectories. Qwen2.5-32B Verified 7.0% to 20.6% with fine-tuning; with an ORM verifier 32.0%. Training on the model's own on-policy successful trajectories did not help in their setting (Lite fell from 15.3%) [snip, truncated].
  ADOPTABLE? yes for the verifier-as-best-of-N idea (cheap at test time); caution on self-trajectory training.
- **Agent-RLVR**, arXiv 2506.11425 [snip]. Guidance plus environment rewards on Qwen2.5-Coder-32B: 8.8% to 21.6% on SWE-bench Verified [unverified].
  ADOPTABLE? no at 32B; reward design is useful.

## 2024 and earlier

- **Self-Rewarding Language Models**, arXiv 2401.10020 (ICML 2024; v3 Mar 2025) [snip]. Model judges its own outputs with LLM-as-judge; iterative DPO. Llama 2 70B on AlpacaEval 2.0. Not code.
  ADOPTABLE? partly. The judge idea is cheap; the gain is at 70B and not on code.
- **ReST-EM (Beyond Human Data)**, arXiv 2312.06585 (TMLR; v4 Apr 2024) [abs/snip]. Sample from the model, keep correct answers, fine-tune, repeat. PaLM-2 on MATH and APPS; scales with size. No official code; community reproduction at github.com/usp787/RSI (not verified).
  ADOPTABLE? yes: this is the simplest self-training loop and it fits our execution-filtered sandbox.
- **STaR**, arXiv 2203.14465 (NeurIPS 2022) [abs/snip]. Bootstrap rationales from correct answers; rationalization fills in for failed items. Works only when few-shot accuracy is above chance (GPT-2 failed on arithmetic).
  ADOPTABLE? yes as the conceptual base for ReST-EM. The chance-level limit means the small model must already solve a share of tasks.
- **s1** (s1K), arXiv 2501.19393 (Jan 2025) [abs/snip]. 1,000 curated traces (difficulty, diversity, quality) from a strong teacher; Qwen2.5-32B-Instruct, about 26 min on 16 H100. s1-32B beats o1-preview on MATH/AIME by up to 27%. Budget forcing at test time.
  ADOPTABLE? yes for the data-size lesson (1k is enough to shape behavior); base is 32B so the numbers do not transfer directly.
- **LIMO**, arXiv 2502.03387 (Feb 2025) [abs/snip]. 817 curated examples; Qwen2.5-32B-Instruct; AIME 57.1 (v1 abstract); later AIME25 44.6 per project README [snip]. github.com/GAIR-NLP/LIMO.
  ADOPTABLE? partly. Same lesson as s1; math-only and 32B.

## Checklist

DONE (searched; abstract or snippet found):
- [x] 1 Self-improvement / bootstrapping: STaR, ReST-EM, Self-Rewarding, BSR, Code-A1, CoSPlay, UTRL, CURE (CURE and UTRL seen only in snippets; CURE arXiv 2506.03136, UTRL arXiv 2508.21107)
- [x] 2 RL with verifiable rewards on small models: AceCoder, Tina, 1-shot RLVR, Dr. GRPO, DAPO (snippet only), PRIME, 2605.30478, DeepCoder, DeepSWE, SkyRL-Agent, SWE-RL, Absolute Zero
- [x] 3 Synthetic executable tasks and curricula: SWE-smith, R2E-Gym, SWE-Gym, Scale-SWE, Socratic-SWE, HHD, SWE-rebench, rStar-Coder, SWE-Bench++
- [x] 4 Distillation recipes: s1, LIMO, OpenThoughts, SWE-smith trajectories, Scale-SWE trajectories
- [x] 5 Process/outcome reward models from execution: PRIME, ORPS, CodePRM, PRLCoder, ReCode, FunPRM, EGCA

PARTLY DONE (found, but numbers or sizes unverified):
- [~] Per-paper small-model numbers for Socratic-SWE, HHD, Code-A1, CoSPlay, BSR (no base size or tables)
- [~] SWE-smith 7B score (third-party table only)
- [~] Absolute Zero per-benchmark gains (secondary wiki only)
- [~] DAPO (not fetched; secondary only)
- [~] Execution-PRM numbers for CodePRM, PRLCoder, ReCode (not fetched)

NOT FOUND:
- [ ] Measured GRPO/LoRA wall-clock or dollar cost for a 1.5B-7B coding run on one 16 GB GPU. Only vendor memory claims found (Unsloth: Qwen2.5-1.5B reasoning training about 5 GB VRAM) [snip].
- [ ] A published Qwen 7B self-improvement loop evaluated on SWE-bench (checked; nothing matched).
- [ ] A Qwen2.5-Coder-7B SWE-smith scaling curve with values (paper says it scales; figure values not retrieved).
- [ ] Code-R1 as a named paper (not found).
- [ ] CodeT (self-generated tests, arXiv 2207.10397) and CodeT5-based self-play: not searched in this pass; treat as NOT CHECKED.
- [ ] Verified DeepCoder and DeepSWE dates (search gave no dates).
- [ ] Kaggle T4 timing for any of these recipes (no source).

## Top 5 by expected value for us (cheapest credible experiment first)

1. **Execution-filtered self-training (ReST-EM / STaR loop) on synthetic bugs.** Evidence: ReST-EM [abs], SWE-Gym [snip] (warns that on-policy self-trajectories may not help). Experiment: take Qwen3-1.7B or Qwen2.5-Coder-1.5B; 500 SWE-smith-style injected-bug tasks from our sandbox; 8 samples per task; keep only tests-passing patches; one LoRA SFT round on Kaggle T4; compare to base on 50 held-out protected tests. Stop if no gain over base after one round.
   ADOPTABLE? yes.
2. **Offline distillation from a small verified trace set (s1 / LIMO / OpenThoughts style).** Evidence: s1 1k, LIMO 817, SWE-smith 5k trajectories [abs/snip]. Experiment: one offline run of a strong model (once, as allowed) to produce 1,000 execution-verified traces; LoRA SFT of a 4B Qwen on Kaggle T4; compare with step 1 on the same tests. Cost: unmeasured.
   ADOPTABLE? yes, partly: needs the strong-model data step approved and no uploads without the user's yes.
3. **Short LoRA GRPO with unit-test reward, lint-free.** Evidence: AceCoder 7B 80 steps / 48 H100-h [snip, unverified]; Tina LoRA cost template [abs]; 2605.30478 reward warning [abs]; Dr. GRPO normalization [snip]. Experiment: 1.5B or 4B LoRA, 80-150 steps on 1k verified tasks, unit tests only vs tests plus lint, Kaggle T4; watch for length collapse.
   ADOPTABLE? partly: RL needs more compute than steps 1-2; run only after step 1 shows a gain.
4. **Hindsight hint distillation from own failures (HHD).** Evidence: +8% claim vs about +2% baselines on SWE-bench Verified [snip, unverified]. Experiment: reuse the step 1 failure buffer; generate one-sentence hints from failed traces; train without hints; measure on the same 50 tests. Needs no CoT labels.
   ADOPTABLE? partly: unverified headline, but it reuses data we already generate.
5. **Execution-grounded credit assignment or execution-gated PRM.** Evidence: EGCA [snip], ORPS, PRIME outcome-only PRM [abs/snip]. Experiment: after steps 1-3 exist, compare plain GRPO with EGCA-style token masking on the same tasks. Only if RL works at all.
   ADOPTABLE? partly: more engineering than the first four; park until they report.

Watch list (not ranked): Code-A1 two-model co-evolution; CoSPlay training-free self-play at test time (no training, may be cheapest to try at inference).

Not run: no downloads, no code, no Kaggle or local compute. Next step for any row: read the paper's tables before quoting a number (all [snip] and [unverified] items above).

---

<!-- source: results/raw/research-20261015-smallmodel/F_capacity_routing_uncertainty.md -->
# Part F. Capacity, routing, uncertainty (open web research, 2026-10-15)

Scope: web research only. No downloads, no code, no compute.
Labels: VERIFIED = page text fetched this session. SNIPPET = search-result text only, paper not opened. UNVERIFIED = blog, aggregator or secondary source. estimate (unmeasured) = my own arithmetic.
Setting: RTX 3050 6 GB laptop, 15.3 GB RAM, free Kaggle T4 x2 / TPU v5e-8 for offline work, no paid API, no strong model at runtime.

Binding constraint found: system RAM (15.3 GB), not VRAM, decides which MoE models are usable. Qwen3-Coder-30B-A3B at Q4_K_M is about 18.3 GB on disk (calculator, UNVERIFIED), which exceeds our RAM.

## Topic A: strongest model in 6 GB VRAM / 15 GB RAM

### A1. Model candidates (newest first)

- **Qwen3.6-27B** (dense, Apache-2.0). VERIFIED on the HF card, https://huggingface.co/Qwen/Qwen3.6-27B: SWE-bench Verified 77.2, SWE-bench Pro 53.5, Terminal-Bench 2.0 59.3, thinking mode on by default, 262,144 native context. Vendor-reported; no independent reproduction found. Release date (2026-04-22) from SNIPPET/UNVERIFIED press pages.
  Fits 6 GB: no (estimate: about 16 GB at Q4). ADOPTABLE: no as a runtime model; partly as an offline teacher on Kaggle T4 x2.
- **Qwen3.6-35B-A3B** (MoE). Named in UNVERIFIED guides only (https://insiderllm.com/guides/qwen-3-6-local-ai-guide/). No numbers opened. ADOPTABLE: no, same RAM problem as the 30B MoE.
- **Gemma 4** (Google, released 2026-04-02 per DeepMind page, Apache-2.0). SNIPPET from https://deepmind.google/models/gemma/gemma-4 (vendor table, LiveCodeBench v6): 31B 80.0, 26B MoE 77.1, E4B 52.0, E2B 44.0. Vendor-claimed. One UNVERIFIED practitioner reports strong one-shot coding but weak agentic work for the 26B. Fits 6 GB: E4B likely at Q4 (estimate); 26B no. ADOPTABLE: partly (E4B only, and only if it beats Qwen3.5-9B on our tasks, which is untested).
- **Qwen3.5 small series** (0.8B, 2B, 4B, 9B; Apache-2.0; announced 2026-03-02 per UNVERIFIED secondary pages). VERIFIED on the 9B card, https://huggingface.co/Qwen/Qwen3.5-9B: LiveCodeBench v6 65.6, OJBench 29.2, BFCL-V4 66.1, TAU2-Bench 79.1, OSWorld-Verified 41.8; thinking on by default; 262,144 context. The card has NO SWE-bench Verified row for 9B. The Uni-Agent docs (UNVERIFIED, https://uni-agent.readthedocs.io/en/latest/benchmark/inference.html) list 53.8 and 65.6 for SWE-bench under their own harness; 65.6 is the same number as the LiveCodeBench score, so do not mix them up. Fits 6 GB at Q4_K_M: likely (estimate: about 5.5 GB file, short context). ADOPTABLE: YES, primary candidate. Caveat: thinking-by-default means many more tokens per task, so wall-clock cost is higher than the score suggests.
- **Devstral Small 2** (Mistral, 24B dense, Apache-2.0, released 2025-12-09). VERIFIED on https://mistral.ai/news/devstral-2-vibe-cli: SWE-bench Verified 68.0 for Small 2 and 72.2 for the 123B Devstral 2; 256K context; vendor says it runs on a single GPU or CPU-only. A listing with 65.8 exists (https://www.canirun.ai/model/devstral-small-2-24b, UNVERIFIED). No speed figure on the official page. Fits 6 GB: no. Dense 24B offloaded to RAM at Q4 would be slow (estimate, unmeasured). ADOPTABLE: no at runtime; yes as a Kaggle teacher if its license and output rules allow.
- **Qwen3-Coder-30B-A3B** (MoE, 2025). SWE-bench Verified spread across sources: 51.6 (SNIPPET, https://www.datalearner.com/en/ai-models/pretrained-models/Qwen3-Coder-30B-A3B-Instruct/analysis), 49.2 (Uni-Agent, 100 turns), 60.4 (aggregator). The spread is a harness effect, not a model effect. Q4_K_M is about 18.3 GB and Q3 maybe about 14 GB (estimate). Unsloth's guidance (UNVERIFIED) says 6+ tok/s needs about 18 GB of combined VRAM and RAM. ADOPTABLE: partly. Q4 does not fit our 15 GB RAM; Q3 is borderline and quality-costly.

### A2. Quantisation and measured code loss

- AWQ / HumanEvalPlus study (SNIPPET, arXiv 2410.14766-era 2024 work; authors not confirmed): Mistral-Instruct-7B HumanEvalPlus drops about 13 points; StarCoder2-7B and CodeLlama-7B barely move. The effect is model-dependent. Source found as a list entry, not opened.
- "Quantize with Confidence? An Empirical Study of Quantization for Code Generation", arXiv 2607.14181 (2026, ID suggests mid-2026): compares GPTQ, AWQ, QuIP#, AQLM, BitsAndBytes and GGUF against FP16 on Qwen2.5-Coder-7B and CodeLlama-7B (McEval, CoderEval). SNIPPET only; headline numbers NOT retrieved. Read before using.
- "Precision or Peril" arXiv 2411.10656 (2024, SNIPPET): Python code quality under quantisation. Not opened.
- Unsloth Dynamic 3.0 (VERIFIED docs text via search snippet, vendor claim): imatrix calibration for agentic coding, KL-divergence based, held-out Divergence-300 set from DeepSWE and Terminal Bench prompts. Source: https://unsloth.ai/docs/basics/dynamic-3.0-ggufs. Vendor-run; not independent.
- EXL3 vs GGUF (UNVERIFIED single blog, RTX 5090, https://atomic.chat/blog/guides/exl3-vs-gguf): at about 4 bpw, EXL3 mean KLD 0.0248 at 5.79 GiB; GGUF Q4_K_M KLD 0.0398 at 6.62 GiB; GGUF Q5_K_M KLD 0.0172 at 7.84 GiB. KLD is not code accuracy. EXL3 is NVIDIA-only; use on sm_86 laptop is UNVERIFIED.
- Single-author fine-tune write-up (UNVERIFIED, n = 47 incidents, judged by gpt-oss-120b): about 2% loss at q8_0, about 12% at q5_K_M, about 9% at q4_K_M. Too small and noisy to use. Same post: Q4_K_M produced misspelled function names when coding, Q5_K_M did not (anecdote).
- Red Hat 500k-evaluation claim (UNVERIFIED, via a 2026 blog): 8-bit recovers over 99% accuracy, 4-bit 98.9%. Not code-specific; GGUF not confirmed.
- NOT FOUND: a same-model comparison of GGUF Q4/Q5 vs EXL2/EXL3 vs AWQ on pass@k for code.

### A3. KV-cache quantisation

- Kitty, arXiv 2511.18643 (2025, SNIPPET): KIVI (not llama.cpp q4_0) on Qwen3-8B, HumanEval: FP16 84.82, 4-bit 84.09, 2-bit 76.89. Direction transfers; numbers do not.
- LLMC toolkit, arXiv 2405.06001 (2024, SNIPPET): naive int8 and int4 KV cache shows almost no loss on HumanEval and MBPP; 2-bit crashes generation.
- KL study, Gemma 4 vs Qwen 3.6 (UNVERIFIED, https://localbench.substack.com/p/kv-cache-quantization-benchmark): Qwen KL below 0.04 at q8_0, 0.087–0.117 at q4_0 (usable); Gemma 26B A4B KL 0.377 at q8_0 (sensitive). KL is distribution shift, not code pass rate.
- llama.cpp PR 7412 (2024, SNIPPET): q8_0 KV shows no significant loss versus FP16.
- Anecdote (UNVERIFIED): quantising K hurts more than V.
- VERIFIED flags, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md: -ctk / --cache-type-k, -ctv / --cache-type-v, -fa / --flash-attn [on|off|auto].
- NOT FOUND: measured pass@k on code for llama.cpp q8_0 vs q4_0 KV.

### A4. MoE expert offload (llama.cpp)

- VERIFIED (same README): --cpu-moe / -cmoe keeps all MoE expert weights on CPU; --n-cpu-moe N / -ncmoe keeps the experts of the first N layers on CPU; -ot / --override-tensor is the general form.
- Individual user reports (UNVERIFIED, lemmy and blog posts): GTX 1060 6 GB, i3-8100, 24 GB DDR4, a 35B-A3B model at about 17 tok/s with 256K context (raising --n-cpu-moe moved speed from 10 to 17 tok/s with --no-mmap); 12 GB card 17 to 34 tok/s; 8 GB RTX 3070 Ti 8.7 to 40+ tok/s (best case). A dev.to 6 GB claim of about 30 tok/s could not be verified.
- Estimate (unmeasured): with 15 GB RAM, a 30B-A3B Q4 (about 18 GB) spills to mmap and NVMe reads, which would dominate speed. Q3 or a 9B dense model is more realistic.
- NOT FOUND: any measured tok/s for Qwen3-30B-A3B on an RTX 3050 6 GB.

### A5. Speculative decoding

- ML-SpecQD, arXiv 2503.13565 (2025, SNIPPET): on a Qwen2.5-Coder-7B target with code prompts, MXFP4 draft 1.90 to 2.03x; multi-level 2.16 to 2.72x. GPU not named.
- NAACL 2025 "Decoding Speculative Decoding" (SNIPPET): over 350 experiments on 65B/66B targets; draft latency matters more than draft quality. Targets are far larger than ours.
- InventiveHQ (UNVERIFIED single blog, RTX 5060 Ti, https://inventivehq.com/blog/llama-cpp-speculative-decoding-consumer-gpu): Qwen2.5-Coder-14B 42.0 tok/s baseline, best 1.85x on math prompts; 7B at 84.7 tok/s got 0.95x (slower) with speculation; CPU-only 3B code prompts 1.96x.
- Estimate: 9B Q4 (about 5.5 GB) plus any draft leaves almost no VRAM on 6 GB, so speculation with a 9B target is probably not workable. A 0.5 to 1.5B draft works only if the target is much smaller. Shared tokenizer with the Qwen3.5 small series is assumed, not verified.

## Topic B: knowing when the model is wrong, and routing

### B1. Probes on hidden states (newest first)

- **"Latent Programming Horizons in Coding Agents"**, arXiv 2607.05188 (Silva, Tu, Monperrus). VERIFIED abstract, https://arxiv.org/abs/2607.05188: linear probes on the residual stream decode whether the code parses, passes tests, reduces failing tests and introduces regressions; correctness AUC up to 0.83 across two models and two benchmarks; probes predict future edits above chance up to about 25 steps ahead; transfer across benchmarks without retraining; signal strongest in intermediate layers. Model and benchmark names are NOT on the abstract page. UNVERIFIED blog: Laguna-XS.2 and Qwen3.6-35B-A3B on SWE-bench Verified and Pro, 22,714 trajectories.
  Cost: label generation needs test outcomes per step (offline, Kaggle). Runtime is one dot product per step (estimate). ADOPTABLE: partly. Best fit for agent action gating. Needs retraining on our own model family.
- **"Code Correctness Signals in LLM Hidden States: Pre-Generation Probing and Repair Geometry"**, arXiv 2606.14530 (Di Cicco). UNVERIFIED (only aggregator pages seen: https://www.emergentmind.com/papers/2606.14530 and SOTAVerified). Qwen3-4B-Instruct-2507 on 444 LiveCodeBench tasks; prompt-final hidden state gives held-out AUC 0.931 +/- 0.008 over 50 splits, before any code is written; after removing the prompt-length linear effect, 0.911 versus 0.754 for a length-only baseline. No independent reproduction. ADOPTABLE: partly. A pre-generation signal could route or skip work before any tokens are decoded. Test on our own labels first; the authors' numbers are the strongest in this list and still unconfirmed.
- **HSRM, "Hidden-State Reward Models for Test-Time Verification"**, arXiv 2608.30841: SNIPPET only. Roughly 2M-parameter head on per-step generator hidden states for best-of-N scoring. Coding or Qwen3 use not confirmed. ADOPTABLE: unknown until opened.
- **"Scaling Agentic Verifier for Competitive Coding"**, arXiv 2602.04254 (Qwen team co-authors): SNIPPET. Execution-based verifier that searches for discriminative test inputs; claims +10 to 15 points Best@k. Needs execution, so offline only. ADOPTABLE: partly.
- Azaria & Mitchell, SAPLMA, arXiv 2304.13734 (Findings of EMNLP 2023): SNIPPET (arXiv HTML and ACL pages listed, not opened). Probe on hidden states gives 71 to 83% accuracy on true/false statements, not code. ADOPTABLE: no as evidence for code; a method template only.
- Kadavath et al., "Language Models (Mostly) Know What They Know", arXiv 2207.05221 (Anthropic, 2022): SNIPPET. P(True) and P(IK) calibration; self-evaluation improves when the model sees several of its own samples. ADOPTABLE: partly (P(True) costs extra decodes).

### B2. Token entropy, semantic entropy, self-consistency

- **"When Uncertainty Isn't Enough: An Empirical Study of Self-Correction in Code Generation"**, ICML 2026 (virtual listing, https://icml.cc/virtual/2026/74488): SNIPPET. Five methods on three small code LLMs (HumanEval, BigCodeBench): multi-sample P(True) has the strongest correlation with correctness; mean token entropy, verbalized confidence, entropy ensembles and semantic-entropy probes are weak. ADOPTABLE: warning. Do not gate on mean token entropy.
- **"Code Is More Than Text: Uncertainty Estimation for Code Generation"**, arXiv 2606.09577. VERIFIED abstract, https://arxiv.org/abs/2606.09577: uncertainty in code sits on a few critical tokens, so averaging dilutes it. Three-axis ensemble raises mean AUROC from 0.696 to 0.776 (+8.1) across five code LLMs (the five models are not named on the abstract page). Single-pass top-K token entropy on Qwen3-14B "matches the strongest multi-pass baseline while being over 3x cheaper". ADOPTABLE: partly. Top-K entropy is cheap and runs in our own decoder. 0.78 AUROC is too weak to abstain on its own.
- Sharma & David, "Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation", arXiv 2502.11620 (Feb 2025): SNIPPET. Entropy and mutual-information estimators adapted to code, with symbolic-execution equivalence in place of NLI; a uniform-distribution simplification is comparable. Correlation numbers not retrieved. Cost: many samples per problem (about N times decode, estimate). ADOPTABLE: partly.
- Farquhar et al., "Detecting hallucinations in LLMs using semantic entropy", Nature 630:625-630 (2024), DOI 10.1038/s41586-024-07421-0: SNIPPET. Clusters sampled answers by meaning, needs a second LLM for equivalence, misses consistent falsehoods. For code, equivalence should come from execution instead. ADOPTABLE: no at our budget for code.

### B3. Selective prediction and abstention for agents

- NOT FOUND (2026 SWE-agent abstention with risk-coverage numbers). The search tool hit a rate limit after one query; the points below are from background memory and are UNVERIFIED: Geifman and El-Yaniv 2017 (selective classification); Quach et al. 2023 (conformal language modeling).
- Trust or Escalate, ICLR 2025 (SNIPPET, https://iclr.cc/virtual/2025/oral/31838): cheap judges, escalate when unsure; a Mistral-7B cascade reaches over 80% human agreement at almost 80% coverage (Chatbot Arena, not code). ADOPTABLE: partly; the pattern transfers, the numbers do not.
- "Not Self-Decidable: LLMs Cannot Draw the Boundary of What an Agent Verifier Can Check", NeurIPS 2026 (SNIPPET, https://neurips.cc/virtual/2026/172888): a model judging its own output cannot see its own blind spots. Implication: do not use the 9B as its own sole gate; keep protected tests as the authority.

### B4. Routing and cascades (local models only)

- RouteLLM, arXiv 2406.18665 (ICLR 2025; 2024 preprint), https://arxiv.org/html/2406.18665v4: SNIPPET. Paper claims cost reduction of more than 2x with no quality loss on public benchmarks; routers trained on Chatbot Arena preference data generalise to unseen model pairs. UNVERIFIED blog numbers (85% cost cut on MT-Bench at 95% of GPT-4) not checked against the paper. ADOPTABLE: partly; the framework fits a 9B-versus-larger-offline design, but we have no preference data for coding.
- FrugalGPT (Chen et al. 2023): cascade with a learned scorer. UNVERIFIED (blog only); paper not opened.
- "Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey", arXiv 2603.04445 (2026): found, NOT opened. Good starting bibliography.
- "LLMs as a Jury", arXiv 2607.10139 (2026): SNIPPET. Accept when two cheap models agree, escalate on disagreement; at equal budget self-consistency scores 0.367 vs 0.700 on AIME-2024 (math, not code). ADOPTABLE: partly; two local models disagreeing on code, then execution, is the version that fits our rules.
- Code cascade by Chen et al. 2025, as summarised in the 2603.04445 survey (UNVERIFIED): each model writes code and its own tests, escalates when a self-test score is below a calibrated threshold. ADOPTABLE: no as written. Our rules say agent-written tests are not evidence of correctness.

## Top 5 by expected value for us (ranked)

1. **Qwen3.5-9B** (Apache-2.0; official LiveCodeBench v6 65.6) at the highest quant that fits 6 GB, with q8_0 KV cache, measured on our own protected coding tasks before anything else. Fits VRAM (estimate). Gaps: no official SWE-bench number; thinking-by-default costs tokens; no measured tok/s on our laptop.
2. **Hidden-state correctness probe on the frozen small model** (2606.14530 pre-generation; 2607.05188 per-step). Train offline on Kaggle with protected-test labels; runtime is one dot product. Highest upside, but both headline results are unconfirmed on our models.
3. **Cheap uncertainty gates: single-pass top-K token entropy plus the probe** (2606.09577). Measure AUROC on our tasks before using it to abstain. Mean token entropy alone should not be trusted (ICML 2026 listing).
4. **Qwen3.6-27B as an offline teacher on Kaggle T4 x2 / TPU** (77.2 self-reported SWE-bench Verified; Apache-2.0). Generates labelled trajectories or distillation targets; never runs on the laptop. Needs an explicit decision on whether model-generated training data is allowed under our rules.
5. **Speculative decoding with a same-family 0.5 to 1.5B draft.** Reported 1.2 to 2x on coding (ML-SpecQD; InventiveHQ). Probably blocked by VRAM once the 9B is loaded (estimate). Ranked last because the expected gain on 6 GB is small. MoE expert offload is lower still, because our 15 GB RAM cannot hold a 30B-A3B Q4.

## Checklist

DONE
- A1 current small coding models: Qwen3.5-9B, Qwen3.6-27B, Devstral Small 2 verified on official pages; Qwen3-Coder-30B-A3B and Gemma 4 from snippets and aggregators.
- A2 quantisation and code: studies located (2024-2026); numbers partly retrieved; no same-model cross-format comparison.
- A3 KV-cache quantisation: studies located; llama.cpp flags verified.
- A4 MoE expert offload: flags verified in llama.cpp README; speed reports are anecdotal.
- A5 speculative decoding: studies and one consumer-GPU blog located.
- B1 hidden-state probes: 2607.05188 abstract verified; 2606.14530 via aggregators; older work by snippet.
- B2 token entropy, semantic entropy, self-consistency: located; 2606.09577 abstract verified.
- B4 routing and cascades: located; RouteLLM via arXiv HTML listing (snippet level).

NOT FOUND
- Measured tok/s for any Qwen3 MoE on an RTX 3050 6 GB, or any 6 GB device with 15 GB RAM.
- Same-model pass@k comparison of GGUF Q4/Q5 vs EXL2/EXL3 vs AWQ on code.
- Measured pass@k on code for llama.cpp q8_0 vs q4_0 KV cache.
- Official SWE-bench Verified for Qwen3.5-9B; independent reproduction of Qwen3.6-27B's 77.2.
- 2026 SWE-agent selective prediction with risk-coverage numbers (search rate-limited; not retried).
- Phi-family coding numbers: NOT searched (out of time budget).
- Measured speculative decoding with coding prompts on a 6 GB card.
- Verified Qwen3-30B-A3B GGUF file sizes (only calculator estimates).

NOT OPENED (listed only): arXiv 2607.14181, 2411.10656, 2608.30841, 2602.04254, 2603.04445, 2607.10139, 2502.11620, 2304.13734, 2207.05221, Nature 630:625; each needs its own read before any number from it is used.

Search count: about 20 WebSearch calls (two rate-limit errors) and 6 WebFetch calls.

---

<!-- source: results/raw/research-20261015-smallmodel/G_unconventional.md -->
# Part G. Unconventional and frontier directions for small models (1-14B), surveyed 2026-10-15

Scope: web search only (no downloads, no code run). Status labels:
- **snippet**: found in search results, title/abstract/summary seen, paper not opened.
- **secondary**: only blog, aggregator or review text seen; numbers unverified.
- **UNVERIFIED**: stated from background knowledge, not re-checked this session.

Adoptable: yes / partly / no, with one-line reason. Compute refers to our hardware (6 GB RTX 3050, Kaggle T4 x2, TPU v5e-8).

## 2026

**Doc-to-LoRA: Learning to Instantly Internalize Contexts** (Charakorn, Cetin, Uesaka, Lange; Sakana; ICML 2026)
- Hypernetwork reads a document and emits a LoRA in one forward pass, so later queries skip the context. https://arxiv.org/abs/2602.15902 (snippet); code https://github.com/SakanaAI/doc-to-lora
- Gains (self-reported): near-perfect needle-in-haystack at >4x native context; beats standard context distillation on real QA with lower peak memory. Base model Gemma-2-2b-it per slides (secondary).
- Compute: hypernetwork meta-trained on ~20M contexts (secondary); per-model training needed.
- ADOPTABLE: partly. Per-Qwen hypernetwork is a large training job; only worth it if repo-context QA shows a gap vs. plain context.

**Conditional Memory via Scalable Lookup (Engram)** (DeepSeek-AI with Peking U.; Jan 2026)
- N-gram-style O(1) memory module as a sparsity axis beside MoE. https://arxiv.org/abs/2601.07372 (snippet); code https://github.com/deepseek-ai/Engram
- Gains: not verified; press claims unverified. Whether DeepSeek V4 uses it is unconfirmed.
- ADOPTABLE: no. Needs pretraining; cannot be bolted onto frozen Qwen cheaply.

**Less is More / TRM: critique and replication** (Roye-Azar et al., arXiv 2512.11847, rev. Jan 2026)
- Re-runs the released TRM checkpoint. Finds test-time augmentation plus majority voting explain a large share of reported ARC accuracy; blanking the puzzle ID gives zero; most accuracy comes at the first recursion step. https://arxiv.org/abs/2512.11847 (snippet)
- Lesson for us: ablate augmentation/voting before crediting recursion. Any "recursive" claim needs matched-compute controls.
- ADOPTABLE: no (ARC-specific); the critique method is adoptable as a checklist.

**Tiny Autoregressive Recursive Models** (Rauba, Fanconi, van der Schaar; arXiv 2603.08082, Mar 2026, ICLR 2026 workshop)
- Tests TRM-style refinement in causal decoders vs. untied deep Transformer and Universal Transformer at fixed compute. https://arxiv.org/abs/2603.08082 (snippet)
- ADOPTABLE: no (research direction; no code gains found).

**TRM with Mamba-2 attention hybrid** (Wang & Reid; arXiv 2602.12078, Feb 2026)
- Aggregator lists ~72.2 (attention) vs 66.5 (Mamba-2 hybrid) on a benchmark; UNVERIFIED. https://arxiv.org/abs/2602.12078 (snippet)
- ADOPTABLE: no.

**Evaluating Tiny Recursive Models Across Training for Code Generation** (arXiv 2608.29376, 2026)
- Snippet says recursion is parameter-efficient but not compute-efficient; replication package claimed. https://arxiv.org/pdf/2608.29376 (snippet; only intro seen)
- ADOPTABLE: no for now; useful as a compute warning.

**Function-Level Execution Feedback for Code Preference Optimization** (arXiv 2608.23632, 2026)
- Cites CodeDPO as related work; content not opened. https://arxiv.org/pdf/2608.23632 (snippet, ID unverified)
- ADOPTABLE: partly; read in full before use.

**Harness-Aware Self-Evolving: co-evolving weights, harness and solutions** (arXiv 2607.03935, 2026)
- Argues most evolution work freezes the model; co-evolves weights, harness and solutions. https://arxiv.org/pdf/2607.03935 (snippet)
- ADOPTABLE: partly (idea fits our 5-key budget logic only if compute is measured).

**GAE: Graph-Augmented Evolution with reinforcement-optimized mutation** (arXiv 2607.10127, 2026)
- Makes the AlphaEvolve mutation LLM adaptive instead of fixed. https://arxiv.org/abs/2607.10127 (snippet)
- ADOPTABLE: partly; adds RL on the mutator, which is the playbook we want to avoid.

**Continuous Program Search** (arXiv 2602.07659, 2026)
- Learned mutation operator in a continuous program latent space; not LLM-based; trading-strategy experiments only. https://arxiv.org/abs/2602.07659 (snippet)
- ADOPTABLE: no for code (no code benchmark seen).

**SAE steering vs. LoRA** (arXiv 2605.31183, May 2026)
- Claims SAEs can approach LoRA on AxBench when features are selected and labelled. General steering, not code. https://arxiv.org/abs/2605.31183 (snippet)
- ADOPTABLE: partly; useful baseline for "is steering enough" on a Qwen SAE.

**Neurosymbolic repair with a verification loop** (arXiv 2601.00509, 2026; arXiv 2601.10987, Jan 2026)
- 2601.00509: LLM inside a loop with compiler diagnostics, KLEE symbolic execution, CodeQL. https://arxiv.org/html/2601.00509v1 (snippet)
- 2601.10987 "Reasoning Distillation for Lightweight APR": large teacher gives fix-type and symbolic tags to a CodeT5 student; helps rare bug classes. https://arxiv.org/abs/2601.10987 (snippet)
- ADOPTABLE: partly. Tool-in-loop is cheap (no training); the distillation recipe needs a teacher run offline once.

**Mercury / LLaDA (diffusion LMs for code)**
- Mercury Coder (commercial, arXiv 2506.17298) reports HumanEval ~90.0 for Small, vendor numbers. https://arxiv.org/abs/2506.17298 (snippet). No open weights found.
- LLaDA: arXiv 2502.09992 was the ID given in the brief; the search did not return it, so the ID is UNVERIFIED. Secondary: 8B roughly on par with LLaMA3-8B on general benchmarks; no coding numbers found.
- ADOPTABLE: no (Mercury closed; LLaDA is a different architecture and needs its own training).

## 2025

**Nvidia NVARC, ARC Prize 2025 winner** (Sorokin & Puget; ARC Prize 2025 results, Nov 2025)
- 24.03% on ARC-AGI-2 private set with a ~4B model; synthetic data plus test-time training; heavy reasoning moved offline into a synthetic pipeline using progressively stronger open models (Qwen). Reported cost $0.20/task. https://arcprize.org/blog/arc-prize-2025-results-analysis (snippet/secondary); competitions https://arcprize.org/competitions/2025
- ADOPTABLE: partly. The "strong model offline, small model at runtime" split matches our rule; ARC-specific otherwise.

**TTT for abstract reasoning** (Akyurek, Damani, Qiu, Guo, Kim, Andreas; arXiv 2411.07279, v2 Mar 2025)
- Per-instance TTT after fine-tuning on similar tasks; 8B model reaches 53% on ARC public val; with program-generation ensemble 61.9%. Authors report up to 6x over the fine-tuned base. https://arxiv.org/abs/2411.07279 ; code https://github.com/ekinakyurek/marc (snippet)
- Ingredients: initial fine-tune, augmentations, per-instance training. No code benchmark found.
- ADOPTABLE: partly. Per-instance LoRA steps are cheap to try on Kaggle T4; needs a self-supervised loss that exists for code (masked lines).

**TTRL: test-time RL on unlabeled data** (Tsinghua / Shanghai AI Lab; arXiv 2504.16084, v2 May 2025)
- Majority-vote reward on unlabeled test prompts; Qwen2.5-Math-7B reported ~211% pass@1 gain on AIME 2024 (authors, abstract). https://arxiv.org/abs/2504.16084 ; code https://github.com/PRIME-RL/TTRL (snippet)
- Code analogue would need an execution-based pseudo-reward; not tested in the literature found.
- ADOPTABLE: partly. Test-time, label-free; but it is RL, so it sits close to the playbook we are avoiding.

**Text-to-LoRA: instant transformer adaption** (Charakorn, Cetin, Tang, Lange; Sakana; ICML 2025; arXiv 2506.06105)
- Hypernetwork maps a task description to a LoRA in one pass; SFT variant generalises to unseen tasks. Secondary summary: Mistral-7B average 73.9 with T2L vs 75.8 with conventional adapters vs 60.0 base (UNVERIFIED). https://arxiv.org/abs/2506.06105 ; https://sakana.ai/text-to-lora/ ; code https://github.com/SakanaAI/text-to-lora
- ADOPTABLE: partly. Directly matches the goal "one adapter per Qwen model"; training a hypernetwork is the cost.

**AlphaEvolve** (Novikov et al., DeepMind; arXiv 2506.13131)
- LLM-driven evolutionary code search with automatic evaluators; Gemini 2.5 Pro + Flash ensemble (secondary). Found a 4x4 complex matrix multiplication scheme improving a 56-year-old bound (secondary). https://arxiv.org/abs/2506.13131 ; https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ (snippet)
- Mutator size: frontier, not small.
- ADOPTABLE: partly; the evaluator-driven loop is adoptable, the frontier mutator is not.

**ShinkaEvolve: open-ended, sample-efficient program evolution** (Lange, Imajuku, Cetin; Sakana; arXiv 2509.19349; ICLR 2026)
- Parent sampling, embedding-based novelty rejection, bandit choice among an LLM ensemble. Circle-packing SOTA with 150 samples (authors). Apache-2.0. https://arxiv.org/abs/2509.19349 ; https://github.com/SakanaAI/ShinkaEvolve (snippet)
- Small-mutator results: not found.
- ADOPTABLE: yes-partly. Open code; cheapest test is a matched-budget run with a small Qwen as mutator.

**EvoScale / Satori-SWE** (Shen et al.; arXiv 2505.23604)
- Evolutionary test-time scaling; 32B model: greedy 35.8% and Best@50 41.6% on SWE-bench Verified (authors); trained on <30K open samples. 7B results not found in the snippets. https://arxiv.org/abs/2505.23604 (snippet)
- ADOPTABLE: partly; shows evolution with few samples, but at 32B scale.

**CodeDPO: self-generated and verified code** (Zhang, Li et al.; arXiv 2410.05605, Oct 2024)
- Self-generated tests, PageRank-style ranking on correctness and efficiency; no GPT-4 needed (authors). Secondary: 1.25-1.45x speed-up, 20-45% code flagged optimised (UNVERIFIED). https://arxiv.org/abs/2410.05605 (snippet)
- ADOPTABLE: yes. Pairs are cheap to build with a small Qwen and a local runner; preference step is light.

**Unconstrained model merging for LLM reasoning** (arXiv 2410.13699, Oct 2024)
- Task Arithmetic and TIES merging of a math and a code model. Secondary: GSM8K-COT, GSM8K-Coding, MATH-COT gains over DeepSeek-Coder-Instruct-v1.5 (UNVERIFIED); HumanEval on par with DeepSeek-Math-7B-RL, so code was not fully kept. https://arxiv.org/abs/2410.13699 (snippet)
- ADOPTABLE: yes (cheap, CPU-side merge, eval on Kaggle), but expect code loss.

**Submodule linearity for task arithmetic** (arXiv 2504.10902, Apr 2025)
- Module-level Task Arithmetic with math and coding fine-tunes of Llama-2-13B. https://arxiv.org/abs/2504.10902 (snippet)
- ADOPTABLE: partly (method refinement; base model differs from Qwen).

**Reasoning features and steering, SAE** (Galichin et al., arXiv 2503.18878, Mar 2025; Li et al., arXiv 2505.15634, EMNLP 2025)
- Steering SAE reasoning features in DeepSeek-R1-distilled models: secondary reports +2.2 pts MATH-500 and +4.0 pts GPQA Diamond (UNVERIFIED). https://arxiv.org/abs/2503.18878 ; https://arxiv.org/abs/2505.15634 (snippet)
- ADOPTABLE: partly; small gains, math/science, not code. SAE-free steering in 2505.15634 avoids needing an SAE per model.

**SAE and code correctness** (arXiv 2510.02917, Oct 2025)
- SAEs on code LLMs for correctness features; activation steering as validation. Models and benchmarks not confirmed. https://arxiv.org/html/2510.02917 (snippet)
- ADOPTABLE: unknown until read.

**Looped LMs / Ouro** (Zhu et al., ByteDance Seed et al.; arXiv 2510.25741, Oct 2025; v5 Jul 2026)
- Latent iteration in pretraining with an entropy-regularised depth objective; Ouro 1.4B and 2.6B claimed competitive with up to 12B models (abstract). https://arxiv.org/abs/2510.25741 ; project page https://ouro-llm.github.io (snippet)
- ADOPTABLE: no for Qwen (pretraining change). Watch for released weights as teachers.

**Tiny Recursive Model (TRM) / Less is More** (Jolicoeur-Martineau, Samsung SAIL; arXiv 2510.04871, Oct 2025)
- ~7M params, one 2-layer network with recursion; ~45% ARC-AGI-1, 8% ARC-AGI-2 (abstract; a commentary gives 44.6 and 7.8). https://arxiv.org/abs/2510.04871 (snippet)
- ADOPTABLE: no for code; see the 2512.11847 critique above.

**Huginn: recurrent depth latent reasoning** (Geiping et al.; arXiv 2502.05171, Feb 2025; NeurIPS 2025)
- 3.5B recurrent-block model, trained on 800B tokens; iterations sampled in training, adaptive depth at inference. Weights Huginn-0125 (Apache-2.0 per model card). https://arxiv.org/abs/2502.05171 ; https://huggingface.co/tomg-group-umd/huginn-0125 (snippet)
- ADOPTABLE: no for Qwen (needs recurrent pretraining). Idea transfers only as a no-training test: repeat a block at inference (see idea 3 below).

**Coconut: continuous latent reasoning** (Hao et al., FAIR at Meta and UC San Diego; arXiv 2412.06769, Dec 2024; v3 Nov 2025; COLM 2025)
- Last hidden state fed back as next input; gains on search-heavy logical tasks (abstract, versions differ). Code MIT: https://github.com/facebookresearch/coconut ; https://arxiv.org/abs/2412.06769 (snippet)
- ADOPTABLE: no for coding now (needs staged CoT-to-latent training; no code results found).

**Memory Layers at Scale** (Berges et al., Meta FAIR; arXiv 2412.09764, Dec 2024; ICML 2025)
- Trainable key-value memory layers add parameters without per-token compute; reported up to 128B memory params (abstract). https://arxiv.org/abs/2412.09764 (snippet)
- ADOPTABLE: no (pretraining change).

**On-policy distillation** (Thinking Machines, blog "On-Policy Distillation", Oct 27 2025; GKD arXiv 2306.13649; MiniLLM arXiv 2306.08543, both 2023)
- Student samples, teacher scores each token with reverse KL. Secondary: Qwen3-8B-Base student 60% to 70% AIME'24 in ~150 steps (UNVERIFIED). https://thinkingmachines.ai/blog/on-policy-distillation (secondary; blog not opened); https://arxiv.org/abs/2306.13649 (snippet)
- Compute: teacher log-probs are a known bottleneck (secondary).
- ADOPTABLE: yes-partly. Teacher must run offline during training (Qwen3-14B/30B local or Kaggle), never at runtime.

## 2024

**PLUM: execution-guided on-policy preference learning from synthetic tests** (Zhang, Diao, Zou, Peng; arXiv 2406.06887, Jun 2024, rev. Oct 2024)
- Generate tests from instructions, score policy samples by execution, DPO-style on-policy pairs; no reward model. Gains: up to 4.8% average on HumanEval/MBPP and 11.8% on LiveCodeBench (abstract). https://arxiv.org/abs/2406.06887 (snippet). Note: the brief's ID 2506.03136 was wrong; it returned unrelated work.
- ADOPTABLE: yes. Closest to our existing execution pipeline; a cheap preference step, not full RL.

**RouteLLM: learning to route with preference data** (Ong et al., Berkeley, Anyscale, Canva; arXiv 2406.18665; ICLR 2025)
- Trained router picks strong vs weak model per query; >2x cost reduction at matched quality on MMLU and MT-Bench (authors). https://arxiv.org/abs/2406.18665 ; https://github.com/lm-sys/RouteLLM (snippet)
- ADOPTABLE: partly. Only useful among small Qwen sizes (e.g. 1.5B vs 14B); no paid strong model at runtime.

## Background items not searched (UNVERIFIED, from prior knowledge; check before use)
- Layer duplication / repeated middle blocks at inference on a frozen model: no training; ADOPTABLE as a cheap test on Qwen.
- s1 budget forcing (arXiv 2501.19393): test-time length control via a stop token; standard test-time scaling, probably not unconventional.

## Five most promising unconventional ideas for us, with cheapest experiment

1. **Repository test-time training with a label-free loss.** Per repo, take a few hundred source lines, mask lines, run a few LoRA steps on a Qwen 1.5-7B, reset after each repo. Cheapest test: 100 held-out masked-line items per repo on Kaggle T4, against (a) no adaptation, (b) retrieval-only context of the same token count. Only continue if (a) < (c) by more than the noise floor.
2. **Evolutionary search with an execution evaluator and a small mutator.** Run ShinkaEvolve (Apache-2.0) with Qwen2.5-Coder-7B as the sole mutator on a toy coding task. Cheapest test: matched-budget comparison with best-of-N random sampling at 150 samples (the ShinkaEvolve headline budget). Without the matched control, the result means nothing.
3. **Layer repetition at inference (no training).** Duplicate a middle block of a frozen Qwen3 model at inference and measure HumanEval/LiveCodeBench subset. Cheapest test: one Kaggle run, sweep 3-4 block choices, control = unmodified model with equal extra FLOPs spent on longer sampling.
4. **Hypernetwork-generated per-Qwen adapters (Text-to-LoRA family).** Generate LoRA from a task or repo description in one pass. Cheapest test: train nothing; take the released T2L/D2L checkpoints if they fit our base, or measure only whether a simple adapter-from-description baseline beats the plain model on 50 held-out coding tasks. Full hypernetwork training only after that.
5. **Offline strong-model synthetic curriculum, small model at runtime (NVARC pattern).** The user allows one offline strong-model pass. Cheapest test: generate 300 synthetic bug-fix tasks with the strong model offline, verify each by execution, SFT Qwen 1.5B on the verified set, and measure on a fixed held-out subset with a no-synthetic-data control.

## CHECKLIST

Section status (DONE = searched and reported above; PARTIAL = some material found, gap noted; NOT FOUND = searched, nothing usable)

1. Test-time training / adaptation: DONE for ARC (2411.07279, NVARC) and TTRL; NOT FOUND for repository-level TTT on code.
2. Latent / recurrent / looped: DONE (Coconut, Huginn, Ouro, TRM and critique, TRM follow-ups, Memory Layers).
3. Model merging and routing: DONE (task arithmetic, evolutionary merge, RouteLLM); PARTIAL for merges that keep code on Qwen (only a 2024 Llama/DeepSeek study found).
4. Evolutionary / program search with a small mutator: DONE (AlphaEvolve, ShinkaEvolve, EvoScale, GAE, Continuous Program Search); NOT FOUND for OpenEvolve results with small open models.
5. Neurosymbolic / formal: PARTIAL (2601.00509, 2601.10987, symbolic execution 2505.13452; Nester ICML 2025 type inference, 70.7% top-1 on ManyTypes4Py, secondary); NOT FOUND for a 2026 small-model system combining type checker and symbolic execution; Lean/Dafny not searched.
6. SAE / representation steering for capability: DONE for reasoning (2503.18878, 2505.15634, 2605.31183); NOT FOUND for SAE steering with a code benchmark. A Qwen SAE model card was seen but is self-published and unverified; not relied on.
7. On-policy distillation, preference from execution: DONE (GKD/MiniLLM, Thinking Machines blog, PLUM, CodeDPO); the brief's wrong ID for PLUM is corrected above.
8. Other 2025-2026 claims for small models: PARTIAL (GEPA 2507.19457 prompt evolution, v2 Feb 2026, claims +6% over GRPO on average with up to 35x fewer rollouts, author-reported, not independently reproduced; Engram; Doc-to-LoRA; Mercury; ARC Prize 2025). Not searched in depth: prompt-optimisation for code beyond GEPA, Lean/Dafny.
9. LLaDA ID (2502.09992) given in the brief: NOT FOUND in search (ID UNVERIFIED).

Note: this is a literature survey. No numbers were re-measured and no claim here is a result of ours.

---

<!-- source: results/raw/research-20261015-smallmodel/H_general_agents.md -->
# Part H. General agent capability for small open models (web research, 2026-10-15)

Scope: tool use, planning, error recovery and asking, decision heads and critics, long-context memory, for Qwen-family and other 1-14B models. Web research only. Nothing downloaded, no code run.

Tags: [V] = read in a fetched primary page (abstract or HTML). [S] = search snippet or secondary summary, not checked against full text. [U] = unverified or conflicting; check before citing. Within each section, newest first.

Coverage limit: the official BFCL leaderboard and most full-text result tables were not retrievable; many numbers below come from abstracts, model cards or aggregators.

## 1. Position papers and where small models fail

- [V] Belcak et al., "Small Language Models are the Future of Agentic AI", arXiv 2506.02153 (Jun 2025). https://arxiv.org/abs/2506.02153 and full text https://arxiv.org/html/2506.02153v1
  - Position paper. Its cost figures are its own estimates, not measurements: serving a 7B SLM is "10 to 30x cheaper" than 70-175B models in latency, energy and FLOPs. Estimated replaceable agent-query share (App. B): MetaGPT ~60%, Open Operator ~40%, Cradle ~70%.
  - SLM defined as a model that fits a consumer device and serves one user's agentic requests; most sub-10B models qualify.
  - Conversion recipe (Sec. 6): log agent calls, gather 10k-100k examples, cluster tasks, pick an SLM, LoRA or distill, retrain periodically.
  - Barriers (Sec. 5): B1 centralized infrastructure, B2 generalist benchmarks, B3 low awareness. "Alternative Views" concedes LLMs keep an edge on general understanding.
  - ADOPTABLE: partly. The conversion recipe is usable; the cost and share numbers are unmeasured.
- [S] Critiques (secondary: Towards AI, Maxim): the 10-30x claim needs scrutiny and is not a controlled deployment measurement; centralization and economies of scale are a counterargument.
- [S] "Beyond Scaling: Agents Are Heading to the Edge", arXiv 2605.18535 (2026): claims SLMs suffice for personal-agent workloads (API orchestration, scheduling) and concedes gaps in open-ended reasoning and underspecified code. [U] abstract not fetched.
- [S] Trace analysis of ~900 agent executions (Dec 2025, arXiv 2512.07497 in search results): model scale alone did not predict robustness. [U] not checked.
- [S] Coding case study (single Substack source): small local model in a tailored harness reached 45.56% mean vs 19.11% for a matched-model baseline on Aider Polyglot. [U] Suggests harness fit can matter as much as size.
- [V] WildToolBench, "Benchmarking LLM Tool-Use in the Wild", arXiv 2604.06185 (Apr 2026): 57 LLMs evaluated, "no model achieves an accuracy of more than 15%" (abstract). Per-size numbers not retrieved.
- [S] Aghzal, Stein, Yao, "Why Do LLM-based Web Agents Fail? A Hierarchical Planning Perspective", arXiv 2603.14248 (ACL 2026): failures traced to planning, execution and replanning; low-level execution is the dominant bottleneck. Web agents, not SLMs.
- [S] Xie et al., "Revealing the Barriers of Language Agents in Planning", arXiv 2410.12409 (NAACL 2024): constraints are underused; questions' influence diminishes. Not SLM-specific.
- [S] "A Survey on LLMs for Automated Planning", arXiv 2502.12435 (2025): LLMs not well suited as standalone planners (secondary summary).
- Synthesis: small models fail most on constraint tracking, format and tool-call correctness, long horizons, and not asking or abstaining. Abstention is measured for small models (Section 4); planning and recovery are measured mostly for larger models.

## 2. Tool-use training and benchmarks (1-14B numbers)

- [V] ToolRL, "Reward is All Tool Learning Needs", arXiv 2504.13958 (Apr 2025; NeurIPS 2025). HTML: https://arxiv.org/html/2504.13958v1
  - Base models: Qwen2.5-1.5B/3B/7B-Instruct and Llama-3.2-3B-Instruct. Benchmarks: BFCL V3, API-Bank, Bamboogle.
  - BFCL V3 overall, raw / SFT400 / SFT4k / GRPO cold start: Qwen2.5-1.5B 19.41 / 40.21 / 40.67 / 46.20; Qwen2.5-3B 33.04 / 34.08 / 41.97 / 52.98; Qwen2.5-7B 41.97 / 34.08 / 36.53 / 58.38. Llama-3.2-3B 22.09 / 41.22 / 44.16 / 44.10.
  - [U] The 3B and 7B SFT400 cells are both 34.08; possible transcription error.
  - Reward: format 0/1 plus correctness in [-3,3] from tool-name Jaccard, parameter-name overlap and exact parameter values. Final range [-3,4].
  - ADOPTABLE: yes. The decomposed reward is cheap to reuse as a scorer or reranker; gains are largest on 1.5-3B.
- [S] Synthesize and Reward, arXiv 2606.03892 (2026): RL on BFCL multi-turn. Qwen3-4B +10.2 overall, vs +3.7 for Qwen2.5-7B and +0.9 for Qwen3-8B. [U] table only partly read.
- [S] Fission-GRPO, arXiv 2601.15625 (Jan 2026): learns recovery from execution errors. Table: Qwen3-4B BFCL v4 multi-turn 19.37 base, 36.38 GRPO, 46.75 Fission-GRPO. Text gives 40.87 and 46.75 for 4B and 8B. [U] conflict; check the paper.
- [S] Multi-turn RL with iterative reward calibration, arXiv 2604.02869 (2026): tau-bench airline, Qwen3.5-4B 63.8 to 66.7 (+2.9). [U] Compare only within the paper.
- [S] ToolVerse, arXiv 2607.15660 (2026): GRPO on Qwen3-4B, Qwen3-8B and Qwen2.5-14B with a DeepSeek-V3.2 user simulator; evaluated on BFCL-v3 multi-turn, tau2-Bench and ACEBench-Agent. Numbers not retrieved.
- [S] Qwen3 Technical Report, arXiv 2505.09388 (May 2025): BFCL-v3 thinking mode 4B 57.6, 8B 60.2. Non-thinking 4B probably 65.9 [U]. No 0.6B or 1.7B figure found in the report excerpt.
- [S] APIGen-MT, arXiv 2504.03601 (Apr 2025): xLAM-2-70B 78.19% BFCL v3 overall; xLAM-2-8B multi-turn 69.25%; xLAM-2-70B tau-bench 56.2% (model card) vs base Llama-3.1-70B 38.2%. No 8B tau-bench figure found. Licence on some xLAM-2 and fc-r checkpoints is research-only or CC-BY-NC [S]; check before use.
- [S] APIGen, arXiv 2406.18518 (Jun 2024): verifiable function-calling data generation (execution and format checks). ADOPTABLE: yes. Our tools can execute, so verified generation is cheap.
- [S] ToolACE, arXiv 2409.00920 (ICLR 2025): 8B model; 26,507 APIs; dual-layer verification. Claims BFCL v1 overall 91.41% and v2 85.77% via a secondary review [U]. Model card says it rivals GPT-4 on BFCL.
- [S] Hammer, arXiv 2410.04587 (Oct 2024): function masking for robustness. Hammer2.1 0.5B / 1.5B / 3B BFCL overall about 21.2 / 27.9 / 29.7 (benchmarklist aggregator) [U]; base model Qwen2.5 per model page [U].
- [S] xLAM-1b-fc-r (1.35B) and xLAM-7b-fc-r (6.91B), Salesforce model cards (2024). Their When2Call scores are in Section 4.
- BFCL 2026 numbers for 8B and below conflict across aggregators (Llama-3.1-8B listed 76.1; CoALM-8B 26.8). [U] Official leaderboard not retrieved.
- tau-bench and tau2: original tau-bench reports pass^k; strong models reach pass^8 under 25% in retail (secondary). Sierra tau2 (Jun 2025): retail pass^1 about 74% for gpt-4.1 [U, secondary]. Community Qwen3-4B tau2 card: Pass@4 59.0, Pass@1 36.0 (pass@k, not pass^k) [U].
- NOT FOUND: official pass^k for any model at 8B or below on tau2.

## 3. Planning aids

- [S] Plan-and-Act, arXiv 2503.09572 (ICML 2025): planner plus executor, replans at every step. Synthetic plans reverse-engineered from successful trajectories and augmented; about 15k examples in under an hour (paper conclusion, via summary). WebArena-Lite 57.58% and WebVoyager text-only 81.36% in the latest version; an earlier version cited 54%. Model sizes not retrieved.
  - ADOPTABLE: partly. Plan synthesis from existing trajectories is cheap; the executor size question is open.
- [S] TAPE, arXiv 2602.19633 (2026): graph-based plan aggregation, external solver, constrained decoding. Numbers not retrieved. [U]
- [S] Qwen-AgentWorld, arXiv 2606.24597 (Jun 2026): language world models that simulate 7 agent environments; smallest variant 35B-A3B. Too large for runtime; possibly useful as an offline simulator for data. [U]
- [S] MCST (ACL 2026): 100,847 text-only state-tracking questions; negation cuts accuracy 23-32% across 14 frontier models. [U]
- [S] AGI Maze, arXiv 2607.00627 (Jul 2026): grid mazes; LLM agents fail small mazes that humans solve within budget. [U]
- NOT FOUND: a controlled small-model planner vs single-model comparison; small-model state tracking on tool or API environments.

## 4. Error recovery and knowing when to ask

- [V] When2Call, arXiv 2504.18851 (NAACL 2025). HTML: https://arxiv.org/html/2504.18851v1
  - 3,652 multiple-choice items: tool call 1,295; follow-up question 1,062; unable to answer 1,295; direct answer 0 (every item needs a tool or a question). LLM-judge subset 300.
  - Tool-hallucination rate for models under 10B: Qwen2.5 0.5B / 1.5B / 3B / 7B 20% / 23% / 23% / 21%; Llama-3.1-8B 67%; xLAM-7B-FC-R 24%.
  - Targeted training: MNM 4B SFT baseline, F1 29.7, hallucination 16%, BFCL AST 57.9. With When2Call-RPO: F1 51.0, hallucination 1.9%, BFCL AST 54.0. MNM 8B: hallucination 19% to 1.2% with RPO, BFCL AST 62.5. Training set 6,000 (SFT) and 10,500 (RPO) items.
  - Trade-off: BFCL AST drops about 4 points for 4B under RPO.
  - ADOPTABLE: yes. Cheapest measured recipe for abstain and clarify in small models.
- [S] SpeakRL, arXiv 2512.13159 (Dec 2025): RL on when and what to ask; SpeakER set of 25,000 multi-turn conversations with clarification turns; reward on clarification quality; improves task success and shortens dialogue. Model sizes not verified. ADOPTABLE: partly.
- [S] Ask-to-Act, arXiv 2504.00907 (2025): embodied household task; RL with per-step LLM-generated rewards for actions and questions; +10.4 to 16.5% over baselines including GPT-4o. Multimodal, not small-LLM tests.
- [S] arXiv 2506.06964 (Jun 2025): reward-weighted offline RL on simulated clarifying dialogues, compared with SFT and DPO. Abstract only.
- [S] AgentAbstain, arXiv 2607.10059 (Jul 2026; NeurIPS 2026): 263 paired should-act and should-abstain tasks over 42 sandboxes; 17 frontier LLMs in 4 harnesses; best paired accuracy 59.5%; always-act and always-refuse policies cannot exceed 50%. Notes that clarification work misses mid-trajectory failures.
- [S] "Act or Escalate?", arXiv 2604.08588 (Apr 2026): 8 models from 4 families, each with small and large variants, on 5 human-decision tasks (e.g., loan approval, moderation). Finding: models are miscalibrated and inconsistent; some over-estimate, some under-estimate accuracy. Authors say escalation needs calibrated self-assessment plus error-cost versus routing-cost weighing.
- [S] AgentRefine, arXiv 2501.01702 (ICLR 2025): agents repeat errors despite feedback; refinement tuning on corrected trajectories improves robustness to perturbed action descriptions. Base-model numbers not retrieved.
- [S] Agent-R, arXiv 2501.11425 (2025): iterative self-training for reflection; +5.59% over baseline (HF page). [U]
- Escalation to a human rather than a stronger model: no small-model study found. A practitioner post (Stack Overflow blog, Oct 2026) advises against raw model confidence and for a verifier signal [U, not research].
- NOT FOUND: small-model (8B and below) clarification results on AgentAbstain or SpeakRL.

## 5. Decision heads, critics and value models on frozen LLMs

- [V] Yeats et al., "The Calls are Coming from Inside the Model: Investigating Probe-based Detection of Tool-Calling Errors in LLMs", NeurIPS 2026 workshop. https://neurips.cc/virtual/2026/168209
  - Linear probes on hidden states across 18 tool-calling LLMs on BFCL detect incorrect calls, including right-type wrong-value arguments that logging misses. Generalizes to novel error types. Layer, post-training type and model size matter.
  - No AUROC on the page. Probes detect; they are not shown to gate actions.
  - ADOPTABLE: yes. Cheap, same family as our linear probe baseline.
- [S] "Reading the Plan: Multi-Horizon Future-Action Prediction from Frozen Hidden States in LLM Agents", NeurIPS 2026 workshop: one frozen residual state at about 3/4 depth linearly predicts future tool calls; a 1.01M-parameter MLP predicts 1, 3 and 5 steps ahead at 0.09 ms per decision, with no fine-tuning; on a 110-tool benchmark, hard low-confidence cases 0.562 vs 0.306 for full-corpus Markov. [U] numbers from search summary.
  - ADOPTABLE: yes, as a cheap action-proposal prior.
- Project context (internal, from CLAUDE.md, not web): pilot probe 67.1% vs trained head 59.0% on the frozen Qwen state. This fits the probe-first pattern above.
- [S] Step-level probes, arXiv 2511.06209 (Nov 2025): probes under 10M parameters on frozen states match or beat process reward models up to 810x larger. Reasoning steps, not agents. [U]
- [S] Best-of-Q, arXiv 2601.22701 (Jan 2026): VLM agent proposes candidates, a Q-function ranks them. Whether the policy is frozen is not confirmed. [U]
- [S] Step-Level Q-Value Models, arXiv 2409.09345 (AAAI 2025): sample candidates, pick highest Q; value model trained on MCTS trajectories with step-level DPO; claimed to transfer across agents.
- [S] AgentPRM, arXiv 2502.10325 (Feb 2025): 3B policy with PRM trained jointly; reported to beat GPT-4o on ALFWorld (summary). Not frozen. [U]
- [S] OPRL, arXiv 2509.19199 (Sep 2025): on-policy PRM trained with the policy. Not frozen.
- [S] Agent Q, arXiv 2408.07199 (2024): critic ranks policy actions; policy also trained.
- [S] SWE-TRACE, arXiv 2604.14820 (Apr 2026): rubric PRM and action-level test-time scaling for coding agents.
- [S] VerifiAgent: prompt-driven verifier around a frozen LLM, no training.
- Whether heads beat prompting baselines on small models for general agents: NOT FOUND. Probe papers measure detection, not action selection.

## 6. Long context and memory in agent loops

- [S] ReSum, arXiv 2509.13313 (Sep 2025; v3 Mar 2026): periodic context summarization on top of ReAct; training-free +4.5% average over ReAct on three web-search benchmarks; ReSum-GRPO up to +8.2%. Summarizer ReSumTool-30B fine-tuned because generic smaller models struggle to summarize search conversations (paper claim). Small-model numbers not found.
  - ADOPTABLE: yes for the training-free part.
- [S] AgentFold, arXiv 2510.24699 (Oct 2025): fold operation at fine and coarse scale; authors report a 30B-A3B model beating DeepSeek-V3.1 (671B) on their benchmarks. ADOPTABLE: partly; needs training.
- [S] Context-Folding, arXiv 2510.11967 (Oct 2025): FoldGRPO; active context about 10x smaller than ReAct (abstract). Secondary summary gives BrowseComp-Plus 62.0% and SWE-Bench Verified 58.0% at a 32K active budget [U].
- [S] CoMem, arXiv 2605.30842 (May 2026): memory model runs in parallel, asynchronous k-step-off, hiding summarization latency. ADOPTABLE: partly (infrastructure).
- [S] ACM, arXiv 2607.23809 (Jul 2026): agentic context management; backbone Qwen3.5-9B; baselines ReSum and ACE. Own results not retrieved. [U]
- Context length: RULER (Hsieh et al. 2024; arXiv 2404.06654, ID from memory, [U]). Only 4 of 17 models held satisfactory performance at 32K (summary); Llama-2-7B scored 85.6% at 4K as the baseline (summary). [U]
- [S] "Bootstrap Your Own Context Length", arXiv 2412.18860 (Dec 2024): official Llama-3 1B and 3B nearly fail at 128K.
- [S] MegaBeam-Mistral-7B (vendor report): 84% vs 14% at 128K against Mistral-7B-Instruct-v0.2. [U]
- NOT FOUND: controlled 1B-vs-7B effective-context measurement inside an agent loop.

## Top 5 findings for making small models better at everything

1. Make abstain and clarify a trained action. When2Call: tool hallucination 16% to 1.9% on a 4B model with about 10k preference pairs; BFCL AST fell about 4 points. Cheapest experiment: from our own tool specs, build about 2k should-call and should-abstain pairs; preference-tune Qwen3-1.7B or 4B with and without them; report hallucination and task success on a held-out split.
2. Use cheap probes on frozen states as the action layer. Linear probes detect tool-call errors; a 1M-parameter MLP predicts future calls at 0.09 ms. Cheapest experiment: fit a logistic probe on existing captured Qwen states to predict "call is correct" and compare with the trained head on the same split. Keep the linear baseline mandatory.
3. Decomposed, graded rewards with GRPO. ToolRL: Qwen2.5-1.5B BFCL V3 19.41 to 46.20. Cheapest experiment: use the ToolRL reward as a scorer to rerank a fixed set of candidate actions with no training; only if reranking helps, run GRPO.
4. Plan synthesis from existing trajectories plus per-step replanning (Plan-and-Act). Cheapest experiment: turn existing trajectories into (goal, plan prefix) pairs; test whether a 1.5-3B planner prompt raises step success on 20 tasks.
5. Training-free summarization before learned folding (ReSum: +4.5 over ReAct). Cheapest experiment: summarize every N tool results with the same small model inside the loop; measure on 20 long tasks against no summarization.

All five experiments involve model compute and must run on Kaggle under the project's standing rule, not on the PC. Uploads need approval.

## CHECKLIST

| Section | Status | Note |
|---|---|---|
| 1. Position papers, failure modes | DONE (partial) | Paper and critiques read; SLM-specific critique of planning not found |
| 2. Tool-use training and benchmarks 1-14B | DONE (partial) | ToolRL table read; official 2026 BFCL and 8B tau2 pass^k NOT FOUND |
| 3. Planning aids, world models | DONE (partial) | Plan-and-Act and state-tracking benchmarks; small-model planner comparison NOT FOUND |
| 4. Error recovery, asking, escalation | DONE | When2Call table read; small-model clarify results on SpeakRL and AgentAbstain NOT FOUND |
| 5. Decision heads, critics on frozen LLMs | DONE (partial) | Probe paper read; AUROC and head-vs-prompt results on small models NOT FOUND |
| 6. Long context and memory | DONE (partial) | Folding and summarization papers; controlled 1B-vs-7B agent-loop test NOT FOUND |
