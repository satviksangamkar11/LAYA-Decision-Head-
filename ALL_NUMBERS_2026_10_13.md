# All measured numbers, LAYA decision-head project (as of 2026-10-13)

Every figure below was measured in this project; the source section is HANDOFF_2026_10_11_STATUS.md unless stated. "Fable agreement" = the scorer ranks
Fable's recorded action first among 3-8 options (imitation, not task success). "Out of fold" = 5-fold trajectory-disjoint cross-validation on the 303
cleaned TRAIN pools. VAL = 46 cleaned VAL pools (53 before cleaning; development set, inspected many times). Chance is about 29-30%.

## A. Fable agreement (imitation)

| Setup (frozen Qwen3-4B states) | Out of fold (303) | VAL 46 / 53 | Native LAYA items (600) | Source |
|---|---|---|---|---|
| Released LAYA head, no training | 39.9% | 39.1% / - | 71.5% | 16.14, 16.12 |
| Head + trained adapter (adapter only), v3 334 pools | 80.2% (in-dataset) | - / **79.2%** (no context 64.2%) | - | val-eval-v1 |
| Adapter only, cleaned v4 | 77.9% | 76.1% / 75.5% | - | laya-joint-v1 |
| Adapter + head jointly | 78.9% | 75.4% / 76.1% | - | laya-joint-v1 |
| Head only, frozen bridge (best recipe, arm D) | 50.4% | 52.9% | 69.7% | 16.14 |
| Head + linear read path (path only) | 71.1% | 68.1% | 68.7% | 16.15 |
| Same, 31 head tensors with the path off | 41.5% | 45.7% | 69.1% | 16.15 |
| Head swap, adapter only: coding / typed / code / conductor | 77.2 / 79.0 / 76.9 / 79.6% | 78.3 / 73.9 / 72.5 / 75.4% | - | 16.17 |
| Head swap: random heads (3 seeds) | **83.2-83.8%** | 80.4-82.6% | - | 16.17 |
| **Linear probe on raw Qwen states** | **85.8%** | **89.1% / 94.3%** | - | 16.14-16.17 |
| Options-only text detector | - | - / 67.9% | - | val-eval-v1 |
| Read path trained WITHOUT context vs with (112 pools) | 75.6% vs 71.1% | - | - | 16.15 |

Key comparisons: coding head minus random heads -6.2 points (corrected interval -9.0 to -3.7); probe minus coding head +8.6 (+4.4 to +13.2);
typed minus coding +1.8 (not significant).

## B. Execution-verified outcomes

| Experiment | Valid candidates | Pass among valid | Oracle | Informative problems | Learned scorers solved | First valid | Random among valid |
|---|---|---|---|---|---|---|---|
| GLM-5.3 max, 5 VAL problems | 17/25 | 15/17 | 4/5 | 1/5 | 3/5 (all picked the failing fix on the informative one) | 3/5 | 3.5/5 |
| GLM-5.3 low, 5 VAL problems | 5/25 | 3/5 | 2/5 | 1/5 | 2/5 (all picked the passing fix) | 1/5 | 1.5/5 |
| gpt-oss-20b | stopped at API verification (504; empty answer) | | | | | | |
| Combined | 22/50 | 18/22 | 6/10 | **2/10** | one problem each way | | |

## C. Labels and infrastructure

| Item | Number |
|---|---|
| Mutation pilot v1.1 (20 TRAIN states) | 19/20 usable; 152 mutants: 114 fail tests, 36 pass, 2 unknown; 0 syntax crashes |
| Sandbox replay after the language repair | Go 41/41 repaired states; C# 2/3 and Rust 1/1 (formatting rule v2); 18/18 isolation probes |
| Rebuilt scorers | 12/12 runs reproduce their 46 recorded VAL picks exactly |
| Local vs Kaggle | capture parity cosine >= 0.99996, 12/12 top choices; scoring 128/128 picks identical |
| Canonical split check | only 72 of 118 trace-"train" replayable states are canonical TRAIN |

## D. Mutation diagnostic (19 TRAIN states, scored on Kaggle, HANDOFF 16.24)

| Selector | Picks the passing fix (19 states) | 7 states never trained on |
|---|---|---|
| Uniform random | 20.5% | 20.0% |
| Longest / shortest option | 26.3% / 10.5% | 14.3% / 0% |
| LAYA v4 / random head / baseline v1 / probe | 0% / 0% / 1.8% / 0% | 0% / 0% / 4.8% / 0% |
| Same without context | 1.8% / 1.8% / 0% / 0% | 4.8% / 0% / 0% / 0% |

Every imitation-trained scorer's probabilities are within ~0.02 of uniform. (CORRECTED by section E: the features do carry the signal; these scorers were not trained to use it.)

## E. Diff-focused diagnostic (probe trained on pass/fail, leave-one-state-out, Kaggle, HANDOFF 16.25)

| Probe features | All 19 | 7 unseen |
|---|---|---|
| Edit sites +- 8, centred (primary) | 89.5% | 85.7% |
| Edit sites +- 8 uncentred / +- 32 centred / +- 8 no context | 94.7% / 100% / 89.5% | 100% / 100% / 100% |
| Whole-option features (existing), centred | 100% | 100% |
| Lexical (site token ids, no Qwen) / random | 26.3% / 20.5% | 14.3% / 20.0% |

Permutation p = 0.001; verdict PASS, but EXPLAINED BY A SHORTCUT (section F): an untrained "most similar to the others" rule picks the original in 19/19.

## F. Edited-only control (original excluded; 15 states, 36 pass vs 83 fail edited copies; HANDOFF 16.26)

| Selector | Top choice passes | Chance |
|---|---|---|
| Probe, whole-option / edit sites +- 8 | 73.3% / 60.0% | 30.1% |
| Probe, whole-option without context | 80.0% | 30.1% |
| Changed tokens only (no Qwen) / mutation operator only | 66.7% / 60.0% | 30.1% |
| Token consensus (untrained) | 21.4% | 30.1% |
| Operator-matched pools (12 states): probe vs tokens / operator | 58.3% vs 66.7% / 66.7% | 32.4% |

Verdict ONLY WITH ORIGINAL: above chance (p 0.003) but not above the simple baselines; no context needed: edit-type cues, not correctness.

## G. Nebius branching count (natural next-action branches; HANDOFF 16.27)

| stage (EXACT history matches; headline) | branchpoints | tasks | repos |
|---|---|---|---|
| mixed-outcome tasks parsed (19,602 runs, parse 100%, tool calls matched 98.5%) | - | 1,741 | 859 |
| identical history, then >= 2 different actions | 9,980 | 1,738 | 859 |
| ... with >= 2 action TYPES | 5,689 | 1,693 | 833 |
| ... and different resolve rates (single tier) | 3,741 | 1,608 | 793 |
| ... >= 2 runs per branch (repeated) | 1,483 | 1,114 | 594 |
| ... >= 3 runs per branch and gap >= 0.3 (strong) | 256 | 249 | 190 |
| strong, excluding the 361 LAYA-overlap repos | 129 | 126 | 107 |
| verified identical workspace | 0 (no snapshots) | | |

Audit: almost all branches are early orientation steps; no edit / stop among the 15 most common type pairs. Threshold met in count, not in substance.
