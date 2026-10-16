# G. Unconventional and frontier directions for small models (1-14B), surveyed 2026-10-15

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
