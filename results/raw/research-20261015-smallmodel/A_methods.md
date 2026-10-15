# A. Methods for making small open models better at coding-agent tasks (no strong model at inference)

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
