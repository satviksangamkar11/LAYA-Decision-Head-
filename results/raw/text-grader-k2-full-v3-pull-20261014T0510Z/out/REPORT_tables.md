# K2 report: frozen unchanged LAYA Coding_Decision_Agent on RootSE (coding) and AgentProcessBench (general tool use)

Report-only. Pre-registration: configs/thresholds_text_grader_k2_v1.toml. Kernel `sangeetasangamkar/text-grader-k2-v1`, script version k2-2026-10-14a.

Caveats (pre-registered): the checkpoint's native agent_trace questions are NOT direct coding-error labels (zero-shot is a repurposing). Fitted arms (names starting `oof_`) are **out-of-fold diagnostic (fitted on the evaluation set's other folds), not a held-out test**. The only unfitted arms are the unchanged checkpoint (`zs_*`) and the fixed rule (`rule_error`). RootSE earlier steps are "not decisive", never verified good; its AUROC is secondary.

## rootse, pre-action, cohort primary_non_OH_72 (items 1365, clusters 72, episodes 71)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.134) | top-3 (chance 0.354) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.282 [0.183, 0.389] | 0.323 [0.245, 0.411] | 0.141 | 0.310 | 0.394 [0.327, 0.464] | - | 0.002 |
| zs_outcome | 0.437 [0.324, 0.549] | 0.488 [0.409, 0.565] | 0.183 | 0.338 | 0.429 [0.365, 0.496] | - | 0.289 |
| zs_needs_review | 0.704 [0.597, 0.806] | 0.647 [0.575, 0.722] | 0.239 | 0.507 | 0.643 [0.575, 0.714] | - | 0.002 |
| zs_risk | 0.606 [0.493, 0.714] | 0.572 [0.489, 0.656] | 0.268 | 0.437 | 0.558 [0.482, 0.632] | - | 0.166 |
| oof_a_step_index | 0.676 [0.563, 0.778] | 0.741 [0.659, 0.814] | 0.366 | 0.690 | 0.603 [0.543, 0.669] | - | - |
| oof_b_action_type | 0.824 [0.729, 0.903] | 0.810 [0.736, 0.874] | 0.494 | 0.783 | 0.862 [0.811, 0.905] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.944 [0.915, 0.969] | 0.718 | 0.859 | 0.870 [0.826, 0.910] | - | - |
| oof_e_probe_cls | 0.901 [0.829, 0.959] | 0.862 [0.792, 0.923] | 0.634 | 0.817 | 0.879 [0.834, 0.919] | - | - |
| oof_e_probe_markers | 0.803 [0.704, 0.889] | 0.747 [0.669, 0.819] | 0.394 | 0.634 | 0.761 [0.698, 0.821] | - | - |

extra: `{"no_tool_call_share_decisive": [0, 71], "no_tool_call_share_earlier": [3, 1294], "runs_without_earlier_step_excluded_from_within_run_metrics": []}`

## rootse, pre-action, cohort secondary_OH_30 (items 629, clusters 30, episodes 27)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.079) | top-3 (chance 0.200) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.519 [0.333, 0.700] | 0.475 [0.357, 0.603] | 0.185 | 0.222 | 0.534 [0.426, 0.644] | - | 0.904 |
| zs_outcome | 0.370 [0.207, 0.556] | 0.443 [0.322, 0.572] | 0.111 | 0.148 | 0.460 [0.360, 0.565] | - | 0.760 |
| zs_needs_review | 0.593 [0.400, 0.778] | 0.507 [0.369, 0.639] | 0.148 | 0.222 | 0.520 [0.386, 0.648] | - | 0.760 |
| zs_risk | 0.630 [0.448, 0.793] | 0.659 [0.535, 0.777] | 0.259 | 0.296 | 0.601 [0.498, 0.703] | - | 0.760 |
| oof_a_step_index | 0.889 [0.760, 1.000] | 0.910 [0.854, 0.957] | 0.370 | 0.815 | 0.698 [0.613, 0.788] | - | - |
| oof_b_action_type | 0.889 [0.759, 1.000] | 0.873 [0.777, 0.945] | 0.401 | 0.779 | 0.862 [0.777, 0.934] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.986 [0.978, 0.994] | 0.704 | 1.000 | 0.890 [0.849, 0.930] | - | - |
| oof_e_probe_cls | 1.000 [1.000, 1.000] | 0.943 [0.911, 0.970] | 0.481 | 0.815 | 0.863 [0.806, 0.912] | - | - |
| oof_e_probe_markers | 0.852 [0.714, 0.964] | 0.752 [0.649, 0.848] | 0.222 | 0.556 | 0.695 [0.606, 0.786] | - | - |

extra: `{"no_tool_call_share_decisive": [18, 30], "no_tool_call_share_earlier": [34, 599], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs027", "rs028", "rs038"]}`

## rootse, pre-action, cohort all_102 (items 1994, clusters 102, episodes 98)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.119) | top-3 (chance 0.312) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.347 [0.258, 0.439] | 0.365 [0.295, 0.435] | 0.153 | 0.286 | 0.433 [0.375, 0.492] | - | 0.012 |
| zs_outcome | 0.418 [0.323, 0.515] | 0.475 [0.407, 0.541] | 0.163 | 0.286 | 0.435 [0.380, 0.491] | - | 0.131 |
| zs_needs_review | 0.673 [0.582, 0.763] | 0.608 [0.541, 0.674] | 0.214 | 0.429 | 0.601 [0.537, 0.662] | - | 0.002 |
| zs_risk | 0.612 [0.515, 0.707] | 0.596 [0.528, 0.664] | 0.265 | 0.398 | 0.575 [0.516, 0.638] | - | 0.060 |
| oof_a_step_index | 0.735 [0.643, 0.818] | 0.788 [0.724, 0.846] | 0.367 | 0.724 | 0.630 [0.580, 0.684] | - | - |
| oof_b_action_type | 0.842 [0.765, 0.910] | 0.828 [0.772, 0.878] | 0.469 | 0.782 | 0.866 [0.826, 0.902] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.956 [0.933, 0.974] | 0.714 | 0.898 | 0.875 [0.842, 0.908] | - | - |
| oof_e_probe_cls | 0.929 [0.874, 0.979] | 0.884 [0.831, 0.930] | 0.592 | 0.816 | 0.872 [0.835, 0.907] | - | - |
| oof_e_probe_markers | 0.816 [0.735, 0.890] | 0.749 [0.688, 0.810] | 0.347 | 0.612 | 0.742 [0.688, 0.796] | - | - |

extra: `{"no_tool_call_share_decisive": [18, 101], "no_tool_call_share_earlier": [37, 1893], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs027", "rs028", "rs038"]}`

## rootse, pre-action, cohort OH_toolcall_only_sensitivity (items 276, clusters 12, episodes 11)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.043) | top-3 (chance 0.130) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.364 [0.091, 0.667] | 0.372 [0.235, 0.539] | 0.091 | 0.091 | 0.460 [0.320, 0.595] | - | 0.874 |
| zs_outcome | 0.182 [0.000, 0.444] | 0.344 [0.186, 0.512] | 0.000 | 0.091 | 0.384 [0.221, 0.559] | - | 0.104 |
| zs_needs_review | 0.545 [0.250, 0.833] | 0.466 [0.257, 0.678] | 0.000 | 0.091 | 0.489 [0.307, 0.664] | - | 0.875 |
| zs_risk | 0.273 [0.000, 0.556] | 0.437 [0.264, 0.623] | 0.091 | 0.091 | 0.474 [0.367, 0.572] | - | 0.474 |
| oof_a_step_index | 1.000 [1.000, 1.000] | 0.942 [0.900, 0.981] | 0.455 | 0.909 | 0.783 [0.722, 0.880] | - | - |
| oof_b_action_type | 0.909 [0.727, 1.000] | 0.843 [0.724, 0.945] | 0.451 | 0.602 | 0.720 [0.571, 0.878] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.998 [0.992, 1.000] | 0.909 | 1.000 | 0.918 [0.871, 0.974] | - | - |
| oof_e_probe_cls | 1.000 [1.000, 1.000] | 0.929 [0.874, 0.972] | 0.364 | 0.818 | 0.824 [0.713, 0.933] | - | - |
| oof_e_probe_markers | 0.909 [0.700, 1.000] | 0.792 [0.653, 0.913] | 0.091 | 0.545 | 0.659 [0.529, 0.803] | - | - |

extra: `{"no_tool_call_share_decisive": [0, 12], "no_tool_call_share_earlier": [0, 264], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs029"]}`

## rootse, post-action, cohort primary_non_OH_72 (items 1365, clusters 72, episodes 71)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.134) | top-3 (chance 0.354) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.338 [0.225, 0.445] | 0.364 [0.278, 0.452] | 0.141 | 0.310 | 0.392 [0.320, 0.466] | - | 0.024 |
| zs_outcome | 0.437 [0.319, 0.557] | 0.513 [0.429, 0.597] | 0.225 | 0.338 | 0.430 [0.363, 0.502] | - | 0.640 |
| zs_needs_review | 0.620 [0.507, 0.722] | 0.591 [0.511, 0.668] | 0.225 | 0.444 | 0.637 [0.563, 0.705] | - | 0.123 |
| zs_risk | 0.535 [0.426, 0.653] | 0.517 [0.436, 0.605] | 0.183 | 0.380 | 0.530 [0.453, 0.606] | - | 0.640 |
| rule_error | 0.423 [0.319, 0.528] | 0.557 [0.507, 0.611] | 0.136 | 0.404 | 0.566 [0.516, 0.620] | - | - |
| oof_a_step_index | 0.676 [0.563, 0.778] | 0.741 [0.659, 0.814] | 0.366 | 0.690 | 0.603 [0.543, 0.669] | - | - |
| oof_b_action_type | 0.824 [0.729, 0.903] | 0.810 [0.736, 0.874] | 0.494 | 0.783 | 0.862 [0.811, 0.905] | - | - |
| oof_c_obs_tfidf | 0.761 [0.657, 0.857] | 0.724 [0.643, 0.797] | 0.270 | 0.669 | 0.755 [0.695, 0.810] | - | - |
| oof_d_text_tfidf | 0.986 [0.957, 1.000] | 0.928 [0.885, 0.962] | 0.690 | 0.859 | 0.880 [0.839, 0.916] | - | - |
| oof_e_probe_cls | 0.915 [0.845, 0.972] | 0.851 [0.779, 0.909] | 0.535 | 0.817 | 0.855 [0.806, 0.899] | - | - |
| oof_e_probe_markers | 0.845 [0.761, 0.928] | 0.777 [0.708, 0.844] | 0.394 | 0.634 | 0.754 [0.692, 0.814] | - | - |

extra: `{"no_tool_call_share_decisive": [0, 71], "no_tool_call_share_earlier": [3, 1294], "runs_without_earlier_step_excluded_from_within_run_metrics": []}`

## rootse, post-action, cohort secondary_OH_30 (items 629, clusters 30, episodes 27)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.079) | top-3 (chance 0.200) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.519 [0.333, 0.720] | 0.583 [0.470, 0.701] | 0.148 | 0.259 | 0.588 [0.482, 0.704] | - | 1.000 |
| zs_outcome | 0.556 [0.360, 0.741] | 0.607 [0.493, 0.719] | 0.148 | 0.296 | 0.616 [0.511, 0.720] | - | 1.000 |
| zs_needs_review | 0.407 [0.214, 0.600] | 0.408 [0.272, 0.541] | 0.074 | 0.148 | 0.389 [0.263, 0.514] | - | 1.000 |
| zs_risk | 0.778 [0.615, 0.923] | 0.697 [0.573, 0.809] | 0.296 | 0.407 | 0.590 [0.468, 0.707] | - | 0.004 |
| rule_error | 0.074 [0.018, 0.146] | 0.434 [0.417, 0.450] | 0.043 | 0.118 | 0.422 [0.407, 0.439] | - | - |
| oof_a_step_index | 0.889 [0.760, 1.000] | 0.910 [0.854, 0.957] | 0.370 | 0.815 | 0.698 [0.613, 0.788] | - | - |
| oof_b_action_type | 0.889 [0.759, 1.000] | 0.873 [0.777, 0.945] | 0.401 | 0.779 | 0.862 [0.777, 0.934] | - | - |
| oof_c_obs_tfidf | 0.815 [0.655, 0.933] | 0.838 [0.721, 0.931] | 0.432 | 0.765 | 0.855 [0.747, 0.939] | - | - |
| oof_d_text_tfidf | 0.963 [0.880, 1.000] | 0.944 [0.861, 0.987] | 0.593 | 1.000 | 0.891 [0.851, 0.929] | - | - |
| oof_e_probe_cls | 0.926 [0.808, 1.000] | 0.868 [0.770, 0.938] | 0.444 | 0.704 | 0.811 [0.723, 0.884] | - | - |
| oof_e_probe_markers | 0.852 [0.704, 0.964] | 0.745 [0.626, 0.847] | 0.259 | 0.556 | 0.746 [0.641, 0.828] | - | - |

extra: `{"no_tool_call_share_decisive": [18, 30], "no_tool_call_share_earlier": [34, 599], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs027", "rs028", "rs038"]}`

## rootse, post-action, cohort all_102 (items 1994, clusters 102, episodes 98)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.119) | top-3 (chance 0.312) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.388 [0.292, 0.480] | 0.424 [0.350, 0.494] | 0.143 | 0.296 | 0.447 [0.386, 0.511] | - | 0.096 |
| zs_outcome | 0.469 [0.374, 0.571] | 0.539 [0.472, 0.607] | 0.204 | 0.327 | 0.483 [0.425, 0.542] | - | 0.557 |
| zs_needs_review | 0.561 [0.464, 0.660] | 0.541 [0.470, 0.614] | 0.184 | 0.362 | 0.555 [0.489, 0.620] | - | 0.460 |
| zs_risk | 0.602 [0.510, 0.700] | 0.567 [0.493, 0.640] | 0.214 | 0.388 | 0.555 [0.493, 0.622] | - | 0.123 |
| rule_error | 0.327 [0.245, 0.410] | 0.523 [0.486, 0.563] | 0.110 | 0.325 | 0.524 [0.488, 0.562] | - | - |
| oof_a_step_index | 0.735 [0.643, 0.818] | 0.788 [0.724, 0.846] | 0.367 | 0.724 | 0.630 [0.580, 0.684] | - | - |
| oof_b_action_type | 0.842 [0.765, 0.910] | 0.828 [0.772, 0.878] | 0.469 | 0.782 | 0.866 [0.826, 0.902] | - | - |
| oof_c_obs_tfidf | 0.776 [0.691, 0.856] | 0.755 [0.689, 0.819] | 0.315 | 0.696 | 0.782 [0.731, 0.830] | - | - |
| oof_d_text_tfidf | 0.980 [0.949, 1.000] | 0.932 [0.896, 0.964] | 0.663 | 0.898 | 0.883 [0.853, 0.912] | - | - |
| oof_e_probe_cls | 0.918 [0.860, 0.969] | 0.856 [0.803, 0.905] | 0.510 | 0.786 | 0.843 [0.801, 0.882] | - | - |
| oof_e_probe_markers | 0.847 [0.773, 0.910] | 0.768 [0.708, 0.825] | 0.357 | 0.612 | 0.752 [0.698, 0.803] | - | - |

extra: `{"no_tool_call_share_decisive": [18, 101], "no_tool_call_share_earlier": [37, 1893], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs027", "rs028", "rs038"]}`

## rootse, post-action, cohort OH_toolcall_only_sensitivity (items 276, clusters 12, episodes 11)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.043) | top-3 (chance 0.130) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.545 [0.250, 0.818] | 0.604 [0.435, 0.761] | 0.091 | 0.182 | 0.612 [0.464, 0.749] | - | 0.904 |
| zs_outcome | 0.273 [0.083, 0.545] | 0.403 [0.258, 0.563] | 0.091 | 0.091 | 0.459 [0.308, 0.614] | - | 0.600 |
| zs_needs_review | 0.636 [0.333, 0.909] | 0.521 [0.329, 0.702] | 0.000 | 0.091 | 0.489 [0.301, 0.679] | - | 0.904 |
| zs_risk | 0.727 [0.417, 1.000] | 0.602 [0.397, 0.775] | 0.000 | 0.182 | 0.509 [0.360, 0.651] | - | 0.600 |
| rule_error | 0.045 [0.000, 0.136] | 0.425 [0.397, 0.451] | 0.006 | 0.033 | 0.424 [0.398, 0.447] | - | - |
| oof_a_step_index | 1.000 [1.000, 1.000] | 0.942 [0.900, 0.981] | 0.455 | 0.909 | 0.783 [0.722, 0.880] | - | - |
| oof_b_action_type | 0.909 [0.727, 1.000] | 0.843 [0.724, 0.945] | 0.451 | 0.602 | 0.720 [0.571, 0.878] | - | - |
| oof_c_obs_tfidf | 0.727 [0.455, 1.000] | 0.749 [0.567, 0.920] | 0.455 | 0.455 | 0.735 [0.526, 0.915] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.994 [0.984, 1.000] | 0.818 | 1.000 | 0.908 [0.858, 0.964] | - | - |
| oof_e_probe_cls | 0.909 [0.700, 1.000] | 0.871 [0.758, 0.953] | 0.364 | 0.636 | 0.784 [0.630, 0.911] | - | - |
| oof_e_probe_markers | 0.727 [0.444, 0.917] | 0.712 [0.483, 0.901] | 0.545 | 0.545 | 0.724 [0.511, 0.889] | - | - |

extra: `{"no_tool_call_share_decisive": [0, 12], "no_tool_call_share_earlier": [0, 264], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs029"]}`

## apb, pre-action, cohort all_tasks (items 7393, clusters 200, episodes 418)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.280) | top-3 (chance 0.730) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.591 [0.531, 0.652] | 0.505 [0.459, 0.552] | 0.270 | 0.742 | 0.464 [0.431, 0.498] | 0.288 [0.249, 0.332] | 0.006 |
| zs_outcome | 0.615 [0.558, 0.667] | 0.559 [0.512, 0.601] | 0.289 | 0.789 | 0.560 [0.521, 0.599] | 0.355 [0.297, 0.417] | 0.002 |
| zs_needs_review | 0.722 [0.676, 0.766] | 0.605 [0.570, 0.641] | 0.330 | 0.833 | 0.469 [0.436, 0.503] | 0.296 [0.258, 0.334] | 0.002 |
| zs_risk | 0.923 [0.894, 0.950] | 0.841 [0.814, 0.868] | 0.653 | 0.900 | 0.580 [0.542, 0.616] | 0.369 [0.311, 0.430] | 0.002 |
| oof_a_step_index | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.741 [0.701, 0.777] | 0.655 [0.566, 0.727] | - |
| oof_b_action_type | 0.572 [0.511, 0.631] | 0.541 [0.489, 0.592] | 0.346 | 0.807 | 0.702 [0.668, 0.732] | 0.536 [0.476, 0.587] | - |
| oof_d_text_tfidf | 0.952 [0.927, 0.973] | 0.926 [0.903, 0.947] | 0.816 | 0.964 | 0.851 [0.802, 0.887] | 0.778 [0.709, 0.831] | - |
| oof_e_probe_cls | 0.892 [0.860, 0.922] | 0.841 [0.812, 0.868] | 0.665 | 0.921 | 0.842 [0.814, 0.866] | 0.745 [0.676, 0.800] | - |
| oof_e_probe_markers | 0.852 [0.814, 0.885] | 0.805 [0.773, 0.835] | 0.624 | 0.928 | 0.801 [0.770, 0.828] | 0.694 [0.622, 0.751] | - |

extra: `{"label_counts_pm1": [4983, 2410], "per_config_episode_win": {"bfcl": {"n": 126, "zs_action": 0.5, "zs_outcome": 0.6666666666666666, "zs_needs_review": 0.7142857142857143, "zs_risk": 0.9126984126984127, "oof_a_step_index": 1.0, "oof_b_action_type": 0.8055555555555556, "oof_d_text_tfidf": 0.8571428571428571, "oof_e_probe_cls": 0.8174603174603174, "oof_e_probe_markers": 0.7619047619047619}, "gaia_dev": {"n": 98, "zs_action": 0.41836734693877553, "zs_outcome": 0.7040816326530612, "zs_needs_review": 0.6326530612244898, "zs_risk": 0.8469387755102041, "oof_a_step_index": 1.0, "oof_b_action_type": 0.3010204081632653, "oof_d_text_tfidf": 1.0, "oof_e_probe_cls": 0.8775510204081632, "oof_e_probe_markers": 0.8571428571428571}, "hotpotqa": {"n": 58, "zs_action": 0.5172413793103449, "zs_outcome": 0.6551724137931034, "zs_needs_review": 0.5, "zs_risk": 0.9310344827586207, "oof_a_step_index": 1.0, "oof_b_action_type": 0.15517241379310345, "oof_d_text_tfidf": 0.9827586206896551, "oof_e_probe_cls": 0.9137931034482759, "oof_e_probe_markers": 0.8793103448275862}, "tau2": {"n": 136, "zs_action": 0.8308823529411765, "zs_outcome": 0.4852941176470588, "zs_needs_review": 0.8897058823529411, "zs_risk": 0.9852941176470589, "oof_a_step_index": 1.0, "oof_b_action_type": 0.7279411764705882, "oof_d_text_tfidf": 0.9926470588235294, "oof_e_probe_cls": 0.9632352941176471, "oof_e_probe_markers": 0.9191176470588235}}}`

## apb, post-action, cohort all_tasks (items 7393, clusters 200, episodes 418)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.280) | top-3 (chance 0.730) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.586 [0.532, 0.642] | 0.531 [0.487, 0.577] | 0.297 | 0.751 | 0.522 [0.485, 0.561] | 0.331 [0.282, 0.386] | 0.008 |
| zs_outcome | 0.751 [0.704, 0.796] | 0.698 [0.658, 0.738] | 0.490 | 0.830 | 0.622 [0.586, 0.659] | 0.454 [0.388, 0.521] | 0.002 |
| zs_needs_review | 0.545 [0.487, 0.603] | 0.467 [0.425, 0.511] | 0.225 | 0.727 | 0.418 [0.386, 0.451] | 0.268 [0.229, 0.307] | 0.115 |
| zs_risk | 0.871 [0.834, 0.906] | 0.786 [0.754, 0.817] | 0.567 | 0.904 | 0.583 [0.545, 0.620] | 0.374 [0.314, 0.438] | 0.002 |
| rule_error | 0.548 [0.512, 0.582] | 0.577 [0.550, 0.606] | 0.398 | 0.779 | 0.582 [0.567, 0.598] | 0.409 [0.362, 0.451] | - |
| oof_a_step_index | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.741 [0.701, 0.777] | 0.655 [0.566, 0.727] | - |
| oof_b_action_type | 0.572 [0.511, 0.631] | 0.541 [0.489, 0.592] | 0.346 | 0.807 | 0.702 [0.668, 0.732] | 0.536 [0.476, 0.587] | - |
| oof_c_obs_tfidf | 0.842 [0.801, 0.878] | 0.802 [0.769, 0.833] | 0.618 | 0.898 | 0.767 [0.733, 0.797] | 0.662 [0.607, 0.708] | - |
| oof_d_text_tfidf | 0.950 [0.924, 0.973] | 0.928 [0.905, 0.948] | 0.821 | 0.947 | 0.868 [0.838, 0.893] | 0.788 [0.727, 0.835] | - |
| oof_e_probe_cls | 0.931 [0.906, 0.953] | 0.889 [0.863, 0.913] | 0.749 | 0.950 | 0.863 [0.839, 0.885] | 0.776 [0.720, 0.822] | - |
| oof_e_probe_markers | 0.852 [0.815, 0.886] | 0.801 [0.768, 0.831] | 0.634 | 0.890 | 0.821 [0.796, 0.844] | 0.715 [0.653, 0.766] | - |

extra: `{"label_counts_pm1": [4983, 2410], "per_config_episode_win": {"bfcl": {"n": 126, "zs_action": 0.5396825396825397, "zs_outcome": 0.8412698412698413, "zs_needs_review": 0.5714285714285714, "zs_risk": 0.9047619047619048, "rule_error": 0.4801587301587302, "oof_a_step_index": 1.0, "oof_b_action_type": 0.8055555555555556, "oof_c_obs_tfidf": 0.9166666666666666, "oof_d_text_tfidf": 0.873015873015873, "oof_e_probe_cls": 0.9126984126984127, "oof_e_probe_markers": 0.8015873015873016}, "gaia_dev": {"n": 98, "zs_action": 0.4897959183673469, "zs_outcome": 0.8367346938775511, "zs_needs_review": 0.3673469387755102, "zs_risk": 0.7244897959183674, "rule_error": 0.5408163265306123, "oof_a_step_index": 1.0, "oof_b_action_type": 0.3010204081632653, "oof_c_obs_tfidf": 0.75, "oof_d_text_tfidf": 0.9897959183673469, "oof_e_probe_cls": 0.8775510204081632, "oof_e_probe_markers": 0.8571428571428571}, "hotpotqa": {"n": 58, "zs_action": 0.603448275862069, "zs_outcome": 0.7413793103448276, "zs_needs_review": 0.39655172413793105, "zs_risk": 0.7586206896551724, "rule_error": 0.49137931034482757, "oof_a_step_index": 1.0, "oof_b_action_type": 0.15517241379310345, "oof_c_obs_tfidf": 0.7931034482758621, "oof_d_text_tfidf": 0.9482758620689655, "oof_e_probe_cls": 0.9655172413793104, "oof_e_probe_markers": 0.7931034482758621}, "tau2": {"n": 136, "zs_action": 0.6911764705882353, "zs_outcome": 0.6102941176470589, "zs_needs_review": 0.7132352941176471, "zs_risk": 0.9926470588235294, "rule_error": 0.6397058823529411, "`
