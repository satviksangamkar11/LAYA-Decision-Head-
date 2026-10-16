# C. Test-time compute for small models (web research, 2026-10-15)

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
