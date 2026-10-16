# Loop engineering v1, stage 1 (L1 decide-or-abstain): result (2026-10-15)

Kernel `sangeetasangamkar/laya-loop-eng-run-v1` v2 (v1 failed at import: exp4lib reads argv; fixed, no logic change). Pre-registration
`configs/thresholds_loop_eng_v1.toml` (sha256 f5087856...). Reproduction gate PASSED: per seed 0.7826 / 0.8043 / 0.7609 (expected identical).
46 cleaned VAL pools, 32 trajectories. Exploratory (VAL previously examined).

| arm | single pass | L1 t=0.10: coverage / acc decided | t=0.20 | t=0.30 |
|---|---|---|---|---|
| full system (3-seed probability average) | 0.804 [0.667, 0.933] | 0.674 / 0.871 | 0.630 / 0.862 | 0.522 / **1.000** (24/24) |
| no-context control | 0.739 | 0.565 / 0.885 | 0.391 / 0.889 | 0.304 / 0.857 |
| options-only control | 0.761 | 0.717 / 0.848 | 0.543 / 0.920 | 0.500 / 0.957 (22/23) |

Notes: averaging the 3 seeds' probabilities before picking gives 0.804 vs 0.783 mean of per-seed hits (+2.2 points); this is an ensemble, not
part of L1. Escalated pools: full system 0.667 / 0.706 / 0.591 accuracy.

Pre-registered reading: L1 helps only if the full system's gain on decided pools exceeds the same loop's gain on the controls. Gains vs single
pass: full system +6.7 / +5.8 / +19.6 points; options-only +8.8 / +15.9 / +19.6; no-context +14.6 / +15.0 / +11.8. **Not met**: abstention
works as a confidence filter for every arm, the text-only control included, so it is not evidence of better context-aware decisions.
At t = 0.30 the full system is right on all 24 pools it decides vs 22 of 23 for options-only: the same picture within one pool.
