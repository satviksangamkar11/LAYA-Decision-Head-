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

Every scorer's probabilities are within ~0.02 of uniform: the frozen, mean-pooled Qwen option features cannot see a one-token bug.
