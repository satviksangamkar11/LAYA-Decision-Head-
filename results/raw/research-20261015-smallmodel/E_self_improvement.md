# E. Self-improvement and cheap learning for small models (Qwen 1-14B, coding/agent)

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
