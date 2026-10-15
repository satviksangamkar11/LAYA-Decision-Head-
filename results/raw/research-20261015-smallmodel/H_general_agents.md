# H: General agent capability for small open models (web research, 2026-10-15)

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
