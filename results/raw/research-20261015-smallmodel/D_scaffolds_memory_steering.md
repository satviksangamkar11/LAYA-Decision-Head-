# D. Scaffolds, memory, prompt optimisation, steering, constrained decoding (small open models)

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
