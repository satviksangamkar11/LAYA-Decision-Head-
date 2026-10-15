# Outcome-head v1: result (2026-10-15)

Pre-registration `configs/thresholds_outcome_head_v1.toml` (sha256 e8209a04...). Kernel `sangeetasangamkar/outcome-head-v1`: v1 smoke (pass),
v2 full lost to a Kaggle host-maintenance failure after 12,516 s (no output), v3 full = this result (T4 x2; train 11,462 s, eval 2,211 s).

Checks: encoder hash unchanged, no encoder gradients, type_emb/act_head unchanged, library parity 4.6e-5, cached-head path diff 0.0, own-observation
leak 0/500, passA 27,129 items 0 non-finite. lr 3e-4 by the overfit rule (drops 0.32 / 0.62 / 0.78); w1 0.803; best epoch 4 (val AUROC 0.548).
Train loss 0.702 -> 0.614 over 5 epochs; val AUROC 0.40, 0.42, 0.49, 0.55, 0.55.

Test: 9,792 items, 172 repos (none in LAYA's overlap list), 18,009 within-task matched-depth pairs, resolved share 0.518.

| arm | AUROC [95% CI] | within-task pair AUROC [95% CI] | accuracy |
|---|---|---|---|
| trained head | 0.503 [0.480, 0.526] | 0.498 [0.482, 0.515] | 0.522 |
| unchanged head | 0.505 [0.480, 0.535] | 0.507 [0.492, 0.523] | 0.524 |
| step index | 0.500 | 0.500 | 0.518 |
| action type | 0.511 [0.492, 0.530] | 0.502 [0.490, 0.515] | 0.518 |
| TF-IDF | 0.490 [0.455, 0.525] | 0.504 [0.486, 0.520] | 0.518 |
| frozen-encoder probe | 0.474 [0.443, 0.505] | 0.506 [0.491, 0.523] | 0.513 |

Trained minus unchanged: AUROC -0.002 [-0.033, +0.027], pair -0.009 [-0.028, +0.010]. Trained minus step index: +0.003 [-0.020, +0.026], pair
-0.002 [-0.018, +0.015].

**Gate: FAIL.** Every arm, fitted or not, is at chance. Reading: within tasks that have both resolved and unresolved runs, the state and action at
steps 3 / 10 / 20 carry no detectable information about how the run will end, for any model tried (LAYA trained or unchanged, TF-IDF, probe,
action type). The head did fit the training set (loss fell) without generalising. The label is not learnable from this input, so this is not a
head or training-recipe failure.
