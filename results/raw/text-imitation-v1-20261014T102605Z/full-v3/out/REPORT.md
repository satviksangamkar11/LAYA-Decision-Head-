# text-imitation v1: head-only fine-tune of LAYA Coding_Decision_Agent on its own frozen encoder

Pre-registration `thresholds_text_imitation_v1.toml` sha256 50164361ca8b4dc72f9a90dcfb7a96513156c15203e85997711d4fe5c6dc82ba. Kernel `sangeetasangamkar/text-imitation-v1`, script ti1-2026-10-14a. Agreement with Fable's recorded action (imitation), not verified optimality.

## Gate (pre-registered, verbatim)

> The head passes only if on VAL its top-1 minus the linear probe AND its top-1 minus the options-only baseline both have 95% bootstrap CIs above 0. Otherwise imitation experiments stop; the next step is verified-outcome labels, not encoder fine-tuning.

- head minus probe: -17.4 points (-35.3 - 2.0)
- head minus options-only: -42.4 points (-56.2 - -26.2)
- **verdict: FAIL: imitation experiments stop; next step is verified-outcome labels, not encoder fine-tuning**

## VAL (46 cleaned pools; final models, 3 seeds; fitted baselines on all TRAIN)

pools 46, trajectories 32, chance top-1 29.9%, chance MRR 0.573; by size {"3": {"n": 30, "chance": 0.3333333333333333}, "4": {"n": 11, "chance": 0.25}, "5": {"n": 5, "chance": 0.2}}

| arm | top-1 | 95% CI | MRR | 95% CI | top-1 by size |
|---|---|---|---|---|---|
| head | 32.6% | 20.0 - 46.2 | 0.594 | 0.514 - 0.677 | {"3": 0.467, "4": 0.091, "5": 0.0} |
| unchanged | 30.4% | 18.4 - 43.2 | 0.578 | 0.503 - 0.659 | {"3": 0.433, "4": 0.091, "5": 0.0} |
| probe | 50.0% | 35.4 - 64.6 | 0.700 | 0.605 - 0.796 | {"3": 0.533, "4": 0.455, "5": 0.4} |
| options_only | 75.0% | 64.4 - 85.4 | 0.870 | 0.814 - 0.924 | {"3": 0.75, "4": 0.818, "5": 0.6} |
| action_prior | 42.8% | 39.1 - 45.8 | 0.598 | 0.568 - 0.625 | {"3": 0.467, "4": 0.371, "5": 0.317} |
| random | 29.9% | 28.5 - 31.3 | 0.573 | 0.557 - 0.588 | {"3": 0.333, "4": 0.25, "5": 0.2} |
| head_seed0 | 32.6% | 20.0 - 46.2 | 0.596 | 0.516 - 0.681 | {"3": 0.467, "4": 0.091, "5": 0.0} |
| head_seed1 | 32.6% | 20.0 - 46.2 | 0.592 | 0.514 - 0.675 | {"3": 0.467, "4": 0.091, "5": 0.0} |
| head_seed2 | 32.6% | 20.0 - 46.2 | 0.592 | 0.514 - 0.675 | {"3": 0.467, "4": 0.091, "5": 0.0} |

| paired | top-1 diff | 95% CI |
|---|---|---|
| head_minus_probe | -17.4 | -35.3 - 2.0 |
| head_minus_options_only | -42.4 | -56.2 - -26.2 |
| head_minus_unchanged | +2.2 | 0.0 - 7.1 |
| head_minus_action_prior | -10.1 | -23.3 - 4.4 |

## TRAIN 5-fold CV grouped by trajectory, out-of-fold (335 pools)

pools 335, trajectories 192, chance top-1 28.9%, chance MRR 0.562; by size {"3": {"n": 182, "chance": 0.3333333333333333}, "4": {"n": 110, "chance": 0.25}, "5": {"n": 43, "chance": 0.2}}

| arm | top-1 | 95% CI | MRR | 95% CI | top-1 by size |
|---|---|---|---|---|---|
| head | 33.8% | 28.4 - 39.0 | 0.590 | 0.556 - 0.623 | {"3": 0.412, "4": 0.291, "5": 0.145} |
| unchanged | 34.0% | 28.6 - 39.2 | 0.591 | 0.558 - 0.623 | {"3": 0.404, "4": 0.3, "5": 0.169} |
| probe | 53.7% | 48.8 - 58.9 | 0.729 | 0.700 - 0.760 | {"3": 0.549, "4": 0.518, "5": 0.535} |
| options_only | 72.5% | 67.2 - 77.6 | 0.842 | 0.813 - 0.871 | {"3": 0.753, "4": 0.689, "5": 0.698} |
| action_prior | 36.8% | 35.2 - 38.5 | 0.550 | 0.538 - 0.563 | {"3": 0.399, "4": 0.345, "5": 0.296} |
| random | 28.9% | 28.3 - 29.4 | 0.562 | 0.556 - 0.568 | {"3": 0.333, "4": 0.25, "5": 0.2} |

| paired | top-1 diff | 95% CI |
|---|---|---|
| head_minus_probe | -19.9 | -27.7 - -12.7 |
| head_minus_options_only | -38.7 | -46.1 - -31.9 |
| head_minus_unchanged | -0.1 | -1.2 - 0.9 |
| head_minus_action_prior | -3.0 | -8.8 - 2.5 |

## Overfitting guard (outer folds, seed 0; selection by inner CV only)

| fold | selected epoch | epoch | train top-1 | held-out top-1 | train logloss | held-out logloss |
|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 33.1% | 37.3% | 1.267 | 1.220 |
| 0 | 1 | 1 | 33.2% | 37.3% | 1.266 | 1.220 |
| 0 | 1 | 2 | 32.8% | 37.3% | 1.264 | 1.220 |
| 0 | 1 | 3 | 32.8% | 37.3% | 1.263 | 1.220 |
| 0 | 1 | 4 | 33.0% | 37.3% | 1.262 | 1.220 |
| 0 | 1 | 5 | 34.0% | 35.8% | 1.261 | 1.220 |
| 0 | 1 | 6 | 34.7% | 34.3% | 1.259 | 1.220 |
| 1 | 5 | 0 | 32.9% | 38.1% | 1.249 | 1.290 |
| 1 | 5 | 1 | 32.9% | 37.3% | 1.248 | 1.290 |
| 1 | 5 | 2 | 33.2% | 35.8% | 1.247 | 1.290 |
| 1 | 5 | 3 | 33.6% | 35.8% | 1.246 | 1.290 |
| 1 | 5 | 4 | 34.3% | 35.8% | 1.244 | 1.289 |
| 1 | 5 | 5 | 34.5% | 35.8% | 1.243 | 1.289 |
| 1 | 5 | 6 | 35.8% | 35.8% | 1.241 | 1.289 |
| 2 | 0 | 0 | 35.3% | 28.7% | 1.259 | 1.252 |
| 2 | 0 | 1 | 35.4% | 28.4% | 1.258 | 1.252 |
| 2 | 0 | 2 | 35.8% | 28.4% | 1.257 | 1.252 |
| 2 | 0 | 3 | 36.2% | 28.4% | 1.256 | 1.252 |
| 2 | 0 | 4 | 36.0% | 29.9% | 1.254 | 1.252 |
| 2 | 0 | 5 | 35.8% | 29.9% | 1.253 | 1.252 |
| 2 | 0 | 6 | 35.6% | 31.3% | 1.252 | 1.252 |
| 3 | 6 | 0 | 32.0% | 41.8% | 1.270 | 1.208 |
| 3 | 6 | 1 | 31.7% | 41.8% | 1.268 | 1.210 |
| 3 | 6 | 2 | 32.5% | 41.8% | 1.266 | 1.211 |
| 3 | 6 | 3 | 32.8% | 41.8% | 1.265 | 1.211 |
| 3 | 6 | 4 | 32.3% | 43.3% | 1.263 | 1.212 |
| 3 | 6 | 5 | 33.6% | 43.3% | 1.262 | 1.213 |
| 3 | 6 | 6 | 34.0% | 43.3% | 1.260 | 1.213 |
| 4 | 1 | 0 | 36.5% | 23.9% | 1.243 | 1.316 |
| 4 | 1 | 1 | 36.7% | 23.9% | 1.242 | 1.318 |
| 4 | 1 | 2 | 36.8% | 23.9% | 1.240 | 1.319 |
| 4 | 1 | 3 | 36.9% | 23.9% | 1.239 | 1.321 |
| 4 | 1 | 4 | 37.5% | 22.4% | 1.237 | 1.323 |
| 4 | 1 | 5 | 38.4% | 22.4% | 1.236 | 1.325 |
| 4 | 1 | 6 | 38.8% | 22.4% | 1.233 | 1.329 |

selected epochs [1, 5, 0, 6, 1] -> final epochs 1

## Run record

```
{
 "version": "ti1-2026-10-14a",
 "kernel": "sangeetasangamkar/text-imitation-v1",
 "prereg": "thresholds_text_imitation_v1.toml",
 "prereg_sha256": "50164361ca8b4dc72f9a90dcfb7a96513156c15203e85997711d4fe5c6dc82ba",
 "mode": "full",
 "stages": {
  "cv_seconds": 2788,
  "final_seconds": 197
 },
 "dataset_pools_sha256": "910162125caa1a970af6995fd915c5362d5d0fdd077e624f74bb863fbca85e4e",
 "loaded_dtypes": {
  "encoder": "torch.float32",
  "head": "torch.float32",
  "agent_autocast": "torch.float16"
 },
 "encoder_sha_before": "7bf36bc15bec46dd6dfa551e6742efb5d3db71b34ef9514129287d2590d1969c",
 "survival": {
  "train": {
   "items": 1201,
   "candidate_action_intact": 1073,
   "candidate_action_intact_share": 0.8934221482098251,
   "task_survived": 1201,
   "task_survived_share": 1.0,
   "A_cut": 128,
   "T_cut_task_longer_than_400": 784,
   "fixed_overflow": 0,
   "library_truncated": 0,
   "tokens": {
    "min": 249,
    "max": 1966,
    "mean": 999.0491257285595
   },
   "seqlen_max": 2032,
   "history_units_kept_mean": 3.5611990008326395,
   "history_units_available_mean": 3.701915070774355,
   "items_all_history_kept": 1125,
   "bad_call_json_pools": 0,
   "empty_task_pools": 0
  },
  "val": {
   "items": 159,
   "candidate_action_intact": 150,
   "candidate_action_intact_share": 0.9433962264150944,
   "task_survived": 159,
   "task_survived_share": 1.0,
   "A_cut": 9,
   "T_cut_task_longer_than_400": 102,
   "fixed_overflow": 0,
   "library_truncated": 0,
   "tokens": {
    "min": 358,
    "max": 1415,
    "mean": 836.1635220125786
   },
   "seqlen_max": 1481,
   "history_units_kept_mean": 2.6540880503144653,
   "history_units_available_mean": 2.6540880503144653,
   "items_all_history_kept": 159,
   "bad_call_json_pools": 0,
   "empty_task_pools": 0
  },
  "all": {
   "items": 1360,
   "candidate_action_intact": 1223,
   "candidate_action_intact_share": 0.899264705882353,
   "task_survived": 1360,
   "task_survived_share": 1.0,
   "A_cut": 137,
   "T_cut_task_longer_than_400": 886,
   "fixed_overflow": 0,
   "library_truncated": 0,
   "tokens": {
    "min": 249,
    "max": 1966,
    "mean": 980.0058823529412
   },
   "seqlen_max": 2032,
   "history_units_kept_mean": 3.4551470588235293,
   "history_units_available_mean": 3.5794117647058825,
   "items_all_history_kept": 1284,
   "bad_call_json_pools": 0,
   "empty_task_pools": 0
  }
 },
 "head_sha_orig": "b6553d6b340c43658c2e9cef2899e9de8df6b238e9f9f59844f048c77fd240e6a78ed113521e5b409e31b87a36fae894d4c1513e64ab484ba2c58ee5182083ae",
 "fold_sizes_pools": [
  67,
  67,
  67,
  67,
  67
 ],
 "encoder_sha_after": "7bf36bc15bec46dd6dfa551e6742efb5d3db71b34ef9514129287d2590d1969c",
 "encoder_unchanged": true,
 "type_emb_act_head_unchanged": true,
 "val_C_probe": 0.01,
 "val_C_options_only": 10.0,
 "val_C_action_prior": 1.0,
 "wall_seconds_total": 3163,
 "workers": {
  "processes": 1,
  "gpu": "Tesla T4",
  "torch": "2.11.0+cu128",
  "gpu_mem_max_gb": 3.306114048
 },
 "failures": []
}
```
