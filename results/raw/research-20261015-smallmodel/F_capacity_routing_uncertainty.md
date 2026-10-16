# F: capacity, routing, uncertainty (open web research, 2026-10-15)

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
