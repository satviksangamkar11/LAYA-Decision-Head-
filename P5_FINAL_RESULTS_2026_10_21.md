# P5 final results (P5_PROTOCOL_v2) — 2026-10-21

Run: Kaggle kernel `el-p5-final-v1`, bundle 1dc2e66e, verdict **OK**, 6/6 shards exit 0, 105/105 tasks, 6.48 h on 2 × T4, 4,062 execution calls with no
infrastructure outcome (5 contract per-input kills and 121 LiveCodeBench test timeouts are candidate outcomes, kept). Grading offline on the PC after
every selection was final; LiveCodeBench = official public + private tests; contract cohort = blind v3 (primary), public v1 (secondary), non-blind v2
(disclosed), each selected by name and hash-checked against lock v2. Analysis: `tools/el_p5_analyze.py`, committed before any result (09bb2090), run unchanged →
`results/raw/p5-final-v1/ANALYSIS.json`. Development-grade evidence on a 1.5B model; SUPPORTED is a claim, never counted as correct.

## LiveCodeBench v6, 85 locked easy/medium problems
| Arm | Solved | Rate | Wrong answers | No answer | SUPPORTED claims | False SUPPORTED | Gen. tokens | Wall-equiv. s |
|---|---|---|---|---|---|---|---|---|
| A plain | 10 | 11.8% | 74 | 1 | — | — | 17,192 | — |
| B multi-sample | **27** | **31.8%** | 20 | 38 | 47 | 20 | 327,255 | 14,023 |
| C V1 | 18 | 21.2% | 9 | 58 | 27 | 9 | 110,716 | 5,566 |
| D V1.5 | 18 | 21.2% | 9 | 58 | 27 | 9 | 110,716 | 5,561 |

Paired differences in solve rate (95% bootstrap interval): B−A +20.0 pts [+11.8, +29.4]; C−A = D−A +9.4 [+3.5, +16.5]; **D−C 0.0 [0, 0]**;
C−B = D−B **−10.6 [−18.8, −2.4]**. D and C are identical on every LiveCodeBench task (no contract exists there; coverage 0, as expected).

## Contract cohort, 20 tasks — PRIMARY: blind private set v3 (849 cases)
| Arm | Solved | Rate | Wrong answers | No answer | SUPPORTED claims | False SUPPORTED | Gen. tokens | Median s |
|---|---|---|---|---|---|---|---|---|
| A plain | 9 | 45% | 11 | 0 | — | — | 3,609 | — |
| B multi-sample | **16** | **80%** | 2 | 2 | 18 | 2 | 33,691 | 6.4 |
| C V1 | 13 | 65% | 2 | 5 | 15 | 2 | 29,347 | 33.6 |
| D V1.5 | 13 | 65% | 1 | 6 | 14 | **1** | 29,587 | 23.5 |

Paired: B−A +35 [+15, +55]; C−A = D−A +20 [+5, +40]; D−C 0 [0, 0]; C−B = D−B −15 [−30, 0]; false SUPPORTED D−C −5 pts [−15, 0].
The single D/C difference is **C13**: C answered SUPPORTED with a program that passes 36/40 blind cases; D's authoritative contract check failed it (2 fails) and
D blocked (no answer). D's contract checks ran on 15/20 tasks; C13 is the only task where one failed. Correct answers wrongly blocked by D: **0**.
Wall-equivalent overhead D vs C (median over tasks): 6.8%.

Secondary (public v1) and non-blind (v2) sets: identical B, C, D results; arm A solves 10 instead of 9 (one A answer passes those sets but fails the blind set),
so C−A and D−A become +15 [−5, +35] there. The primary set is the one the criteria use.

## Pre-registered success criteria (judged literally)
| Criterion | Result |
|---|---|
| Contract: D false SUPPORTED lower than C | **holds** (1 vs 2) |
| Contract: correct answers wrongly blocked = 0 | **holds** (0) |
| Contract: D verified solves not lower than C | **holds** (13 = 13) |
| LCB: D verified solves not lower than C | **holds** (18 = 18) |
| LCB: median D-vs-C wall overhead ≤ 10% | **holds** (0.0%) |
| Beats plain Qwen (interval excludes 0) | **holds** for C and D on LCB (+9.4 [+3.5, +16.5]) and on the contract primary set (+20 [+5, +40]) |
| Infrastructure failures unresolved = 0 | **holds** (0; kernel verdict OK, all exit codes 0) |

**Freeze rule: all D-vs-C criteria hold, so V1.5 is frozen as specified.** That is the protocol's decision, and it is recorded as such.

## What the result does and does not show
- **V1.5 vs V1:** the gain is ONE task (C13): one false SUPPORTED removed, no correct answer lost. The interval of the false-SUPPORTED difference, −5 [−15, 0], touches
  zero; the protocol's criterion is a point comparison and is met, but the evidence that contracts reduce false success is a single case. On LiveCodeBench the
  contract layer is inactive (no contracts), so V1.5 = V1 there by construction.
- **Evidence-Ladder vs plain sampling:** C/D beat a single sample (A) with intervals excluding 0 on both cohorts.
- **Negative result, reported as such:** at the same generated-token CAP, plain multi-sampling (B) solves MORE than the ladder (C/D): −10.6 pts on
  LiveCodeBench with an interval excluding 0, −15 pts on the contract cohort (interval reaching 0). The ladder stops early (max_rounds 2): it spends about a third
  of B's tokens and abstains on 58/85 LiveCodeBench tasks. B pays for its solves with more false SUPPORTED claims (LCB 20 vs 9; precision of SUPPORTED 57% vs 67%;
  contract 16/18 = 89% vs D 13/14 = 93%) and about 2.5× the wall-equivalent time. The protocol compares arms at an equal cap, not equal spend; an equal-spend
  comparison was not pre-registered and is not claimed.
- Not measured here: the LAYA head (scorer OFF), Java, larger models.

## Disclosures
Q03 diagnostic-contract effect (Step 2); Kaggle smoke parallel-shard contention (fixed by sequential shards); compute estimate was a projection (actual 6.48 h);
secondary public set publicly accessible since f6ab9d29; non-blind v2 set's author had prior exposure; blind-set author's 853 → 849 count correction.
