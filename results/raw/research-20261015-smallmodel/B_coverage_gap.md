# B. Pass@1 vs pass@k coverage gap for small open coding models (web research)

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
