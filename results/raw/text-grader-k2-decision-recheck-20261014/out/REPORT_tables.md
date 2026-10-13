# K2 report: frozen unchanged LAYA Coding_Decision_Agent on RootSE (coding) and AgentProcessBench (general tool use)

Report-only. Pre-registration: configs/thresholds_text_grader_k2_v1.toml. Kernel `sangeetasangamkar/text-grader-k2-v1`, script version k2-2026-10-14a.

Caveats (pre-registered): the checkpoint's native agent_trace questions are NOT direct coding-error labels (zero-shot is a repurposing). Fitted arms (names starting `oof_`) are **out-of-fold diagnostic (fitted on the evaluation set's other folds), not a held-out test**. The only unfitted arms are the unchanged checkpoint (`zs_*`) and the fixed rule (`rule_error`). RootSE earlier steps are "not decisive", never verified good; its AUROC is secondary.

## rootse, pre-action, cohort primary_non_OH_72 (items 1365, clusters 72, episodes 71)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.134) | top-3 (chance 0.354) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.563 [0.451, 0.686] | 0.479 [0.403, 0.560] | 0.127 | 0.324 | 0.458 [0.389, 0.527] | - | 1.000 |
| zs_outcome | 0.479 [0.361, 0.592] | 0.529 [0.448, 0.609] | 0.155 | 0.324 | 0.502 [0.427, 0.571] | - | 1.000 |
| zs_needs_review | 0.493 [0.377, 0.611] | 0.484 [0.409, 0.559] | 0.127 | 0.296 | 0.506 [0.442, 0.572] | - | 1.000 |
| zs_risk | 0.535 [0.420, 0.653] | 0.510 [0.424, 0.598] | 0.169 | 0.324 | 0.528 [0.455, 0.601] | - | 1.000 |
| oof_a_step_index | 0.676 [0.563, 0.778] | 0.741 [0.659, 0.814] | 0.366 | 0.690 | 0.603 [0.543, 0.669] | - | - |
| oof_b_action_type | 0.824 [0.729, 0.903] | 0.810 [0.736, 0.874] | 0.494 | 0.783 | 0.862 [0.811, 0.905] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.944 [0.915, 0.969] | 0.718 | 0.859 | 0.870 [0.826, 0.910] | - | - |
| oof_e_probe_cls | 0.648 [0.536, 0.761] | 0.598 [0.520, 0.672] | 0.155 | 0.451 | 0.582 [0.525, 0.643] | - | - |
| oof_e_probe_markers | 0.592 [0.479, 0.708] | 0.558 [0.477, 0.643] | 0.225 | 0.423 | 0.571 [0.504, 0.643] | - | - |

extra: `{"no_tool_call_share_decisive": [0, 71], "no_tool_call_share_earlier": [3, 1294], "runs_without_earlier_step_excluded_from_within_run_metrics": []}`

## rootse, pre-action, cohort secondary_OH_30 (items 629, clusters 30, episodes 27)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.079) | top-3 (chance 0.200) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.556 [0.370, 0.750] | 0.460 [0.323, 0.603] | 0.074 | 0.296 | 0.440 [0.327, 0.568] | - | 1.000 |
| zs_outcome | 0.370 [0.192, 0.556] | 0.417 [0.299, 0.547] | 0.037 | 0.148 | 0.432 [0.329, 0.543] | - | 0.816 |
| zs_needs_review | 0.481 [0.286, 0.679] | 0.503 [0.365, 0.641] | 0.074 | 0.296 | 0.504 [0.387, 0.616] | - | 1.000 |
| zs_risk | 0.407 [0.214, 0.600] | 0.491 [0.366, 0.619] | 0.111 | 0.222 | 0.521 [0.407, 0.634] | - | 1.000 |
| oof_a_step_index | 0.889 [0.760, 1.000] | 0.910 [0.854, 0.957] | 0.370 | 0.815 | 0.698 [0.613, 0.788] | - | - |
| oof_b_action_type | 0.889 [0.759, 1.000] | 0.873 [0.777, 0.945] | 0.401 | 0.779 | 0.862 [0.777, 0.934] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.986 [0.978, 0.994] | 0.704 | 1.000 | 0.890 [0.849, 0.930] | - | - |
| oof_e_probe_cls | 0.519 [0.321, 0.692] | 0.485 [0.355, 0.608] | 0.074 | 0.259 | 0.516 [0.419, 0.614] | - | - |
| oof_e_probe_markers | 0.667 [0.481, 0.846] | 0.602 [0.480, 0.714] | 0.074 | 0.259 | 0.586 [0.476, 0.692] | - | - |

extra: `{"no_tool_call_share_decisive": [18, 30], "no_tool_call_share_earlier": [34, 599], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs027", "rs028", "rs038"]}`

## rootse, pre-action, cohort all_102 (items 1994, clusters 102, episodes 98)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.119) | top-3 (chance 0.312) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.561 [0.464, 0.660] | 0.474 [0.406, 0.546] | 0.112 | 0.316 | 0.453 [0.393, 0.515] | - | 0.920 |
| zs_outcome | 0.449 [0.356, 0.550] | 0.498 [0.432, 0.566] | 0.122 | 0.276 | 0.481 [0.424, 0.541] | - | 0.981 |
| zs_needs_review | 0.490 [0.388, 0.590] | 0.489 [0.422, 0.558] | 0.112 | 0.296 | 0.506 [0.447, 0.564] | - | 1.000 |
| zs_risk | 0.500 [0.404, 0.604] | 0.505 [0.434, 0.576] | 0.153 | 0.296 | 0.525 [0.464, 0.584] | - | 1.000 |
| oof_a_step_index | 0.735 [0.643, 0.818] | 0.788 [0.724, 0.846] | 0.367 | 0.724 | 0.630 [0.580, 0.684] | - | - |
| oof_b_action_type | 0.842 [0.765, 0.910] | 0.828 [0.772, 0.878] | 0.469 | 0.782 | 0.866 [0.826, 0.902] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.956 [0.933, 0.974] | 0.714 | 0.898 | 0.875 [0.842, 0.908] | - | - |
| oof_e_probe_cls | 0.612 [0.520, 0.707] | 0.567 [0.498, 0.631] | 0.133 | 0.398 | 0.563 [0.515, 0.609] | - | - |
| oof_e_probe_markers | 0.612 [0.515, 0.704] | 0.570 [0.504, 0.639] | 0.184 | 0.378 | 0.573 [0.513, 0.628] | - | - |

extra: `{"no_tool_call_share_decisive": [18, 101], "no_tool_call_share_earlier": [37, 1893], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs027", "rs028", "rs038"]}`

## rootse, pre-action, cohort OH_toolcall_only_sensitivity (items 276, clusters 12, episodes 11)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.043) | top-3 (chance 0.130) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.455 [0.182, 0.727] | 0.379 [0.211, 0.559] | 0.000 | 0.091 | 0.431 [0.263, 0.595] | - | 1.000 |
| zs_outcome | 0.182 [0.000, 0.417] | 0.282 [0.161, 0.422] | 0.000 | 0.000 | 0.283 [0.168, 0.408] | - | 0.072 |
| zs_needs_review | 0.545 [0.250, 0.818] | 0.558 [0.362, 0.742] | 0.091 | 0.273 | 0.527 [0.349, 0.701] | - | 1.000 |
| zs_risk | 0.273 [0.000, 0.545] | 0.434 [0.272, 0.611] | 0.091 | 0.091 | 0.445 [0.290, 0.605] | - | 0.486 |
| oof_a_step_index | 1.000 [1.000, 1.000] | 0.942 [0.900, 0.981] | 0.455 | 0.909 | 0.783 [0.722, 0.880] | - | - |
| oof_b_action_type | 0.909 [0.727, 1.000] | 0.843 [0.724, 0.945] | 0.451 | 0.602 | 0.720 [0.571, 0.878] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.998 [0.992, 1.000] | 0.909 | 1.000 | 0.918 [0.871, 0.974] | - | - |
| oof_e_probe_cls | 0.455 [0.182, 0.750] | 0.415 [0.228, 0.607] | 0.000 | 0.000 | 0.486 [0.360, 0.610] | - | - |
| oof_e_probe_markers | 0.727 [0.455, 0.919] | 0.622 [0.446, 0.780] | 0.000 | 0.273 | 0.593 [0.438, 0.742] | - | - |

extra: `{"no_tool_call_share_decisive": [0, 12], "no_tool_call_share_earlier": [0, 264], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs029"]}`

## rootse, post-action, cohort primary_non_OH_72 (items 1365, clusters 72, episodes 71)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.134) | top-3 (chance 0.354) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.465 [0.352, 0.580] | 0.432 [0.357, 0.508] | 0.113 | 0.310 | 0.425 [0.359, 0.491] | - | 1.000 |
| zs_outcome | 0.521 [0.403, 0.639] | 0.530 [0.443, 0.613] | 0.169 | 0.408 | 0.515 [0.438, 0.592] | - | 1.000 |
| zs_needs_review | 0.549 [0.431, 0.662] | 0.528 [0.448, 0.603] | 0.155 | 0.338 | 0.517 [0.451, 0.585] | - | 1.000 |
| zs_risk | 0.465 [0.347, 0.583] | 0.500 [0.422, 0.578] | 0.127 | 0.310 | 0.489 [0.415, 0.563] | - | 1.000 |
| rule_error | 0.423 [0.319, 0.528] | 0.557 [0.507, 0.611] | 0.136 | 0.404 | 0.566 [0.516, 0.620] | - | - |
| oof_a_step_index | 0.676 [0.563, 0.778] | 0.741 [0.659, 0.814] | 0.366 | 0.690 | 0.603 [0.543, 0.669] | - | - |
| oof_b_action_type | 0.824 [0.729, 0.903] | 0.810 [0.736, 0.874] | 0.494 | 0.783 | 0.862 [0.811, 0.905] | - | - |
| oof_c_obs_tfidf | 0.761 [0.657, 0.857] | 0.724 [0.643, 0.797] | 0.270 | 0.669 | 0.755 [0.695, 0.810] | - | - |
| oof_d_text_tfidf | 0.986 [0.957, 1.000] | 0.928 [0.885, 0.962] | 0.690 | 0.859 | 0.880 [0.839, 0.916] | - | - |
| oof_e_probe_cls | 0.577 [0.458, 0.690] | 0.504 [0.423, 0.583] | 0.127 | 0.310 | 0.529 [0.464, 0.591] | - | - |
| oof_e_probe_markers | 0.423 [0.310, 0.542] | 0.463 [0.387, 0.546] | 0.127 | 0.352 | 0.513 [0.454, 0.574] | - | - |

extra: `{"no_tool_call_share_decisive": [0, 71], "no_tool_call_share_earlier": [3, 1294], "runs_without_earlier_step_excluded_from_within_run_metrics": []}`

## rootse, post-action, cohort secondary_OH_30 (items 629, clusters 30, episodes 27)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.079) | top-3 (chance 0.200) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.630 [0.444, 0.808] | 0.553 [0.430, 0.675] | 0.111 | 0.222 | 0.529 [0.421, 0.634] | - | 0.432 |
| zs_outcome | 0.667 [0.483, 0.842] | 0.544 [0.433, 0.662] | 0.074 | 0.185 | 0.514 [0.415, 0.613] | - | 0.336 |
| zs_needs_review | 0.333 [0.160, 0.520] | 0.442 [0.326, 0.570] | 0.148 | 0.222 | 0.402 [0.301, 0.512] | - | 0.336 |
| zs_risk | 0.556 [0.375, 0.739] | 0.518 [0.403, 0.624] | 0.037 | 0.185 | 0.517 [0.418, 0.607] | - | 0.627 |
| rule_error | 0.074 [0.018, 0.146] | 0.434 [0.417, 0.450] | 0.043 | 0.118 | 0.422 [0.407, 0.439] | - | - |
| oof_a_step_index | 0.889 [0.760, 1.000] | 0.910 [0.854, 0.957] | 0.370 | 0.815 | 0.698 [0.613, 0.788] | - | - |
| oof_b_action_type | 0.889 [0.759, 1.000] | 0.873 [0.777, 0.945] | 0.401 | 0.779 | 0.862 [0.777, 0.934] | - | - |
| oof_c_obs_tfidf | 0.815 [0.655, 0.933] | 0.838 [0.721, 0.931] | 0.432 | 0.765 | 0.855 [0.747, 0.939] | - | - |
| oof_d_text_tfidf | 0.963 [0.880, 1.000] | 0.944 [0.861, 0.987] | 0.593 | 1.000 | 0.891 [0.851, 0.929] | - | - |
| oof_e_probe_cls | 0.556 [0.370, 0.741] | 0.575 [0.445, 0.705] | 0.111 | 0.333 | 0.553 [0.436, 0.670] | - | - |
| oof_e_probe_markers | 0.333 [0.160, 0.519] | 0.380 [0.276, 0.492] | 0.037 | 0.148 | 0.454 [0.372, 0.537] | - | - |

extra: `{"no_tool_call_share_decisive": [18, 30], "no_tool_call_share_earlier": [34, 599], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs027", "rs028", "rs038"]}`

## rootse, post-action, cohort all_102 (items 1994, clusters 102, episodes 98)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.119) | top-3 (chance 0.312) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.510 [0.410, 0.604] | 0.466 [0.402, 0.529] | 0.112 | 0.286 | 0.456 [0.399, 0.512] | - | 1.000 |
| zs_outcome | 0.561 [0.465, 0.653] | 0.534 [0.468, 0.602] | 0.143 | 0.347 | 0.515 [0.459, 0.575] | - | 0.948 |
| zs_needs_review | 0.490 [0.389, 0.592] | 0.505 [0.440, 0.571] | 0.153 | 0.306 | 0.484 [0.430, 0.540] | - | 1.000 |
| zs_risk | 0.490 [0.390, 0.586] | 0.505 [0.442, 0.569] | 0.102 | 0.276 | 0.497 [0.437, 0.555] | - | 1.000 |
| rule_error | 0.327 [0.245, 0.410] | 0.523 [0.486, 0.563] | 0.110 | 0.325 | 0.524 [0.488, 0.562] | - | - |
| oof_a_step_index | 0.735 [0.643, 0.818] | 0.788 [0.724, 0.846] | 0.367 | 0.724 | 0.630 [0.580, 0.684] | - | - |
| oof_b_action_type | 0.842 [0.765, 0.910] | 0.828 [0.772, 0.878] | 0.469 | 0.782 | 0.866 [0.826, 0.902] | - | - |
| oof_c_obs_tfidf | 0.776 [0.691, 0.856] | 0.755 [0.689, 0.819] | 0.315 | 0.696 | 0.782 [0.731, 0.830] | - | - |
| oof_d_text_tfidf | 0.980 [0.949, 1.000] | 0.932 [0.896, 0.964] | 0.663 | 0.898 | 0.883 [0.853, 0.912] | - | - |
| oof_e_probe_cls | 0.571 [0.474, 0.670] | 0.524 [0.459, 0.587] | 0.122 | 0.316 | 0.536 [0.481, 0.590] | - | - |
| oof_e_probe_markers | 0.398 [0.303, 0.500] | 0.440 [0.376, 0.507] | 0.102 | 0.296 | 0.496 [0.448, 0.544] | - | - |

extra: `{"no_tool_call_share_decisive": [18, 101], "no_tool_call_share_earlier": [37, 1893], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs027", "rs028", "rs038"]}`

## rootse, post-action, cohort OH_toolcall_only_sensitivity (items 276, clusters 12, episodes 11)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.043) | top-3 (chance 0.130) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.818 [0.583, 1.000] | 0.605 [0.460, 0.747] | 0.000 | 0.091 | 0.609 [0.482, 0.736] | - | 0.057 |
| zs_outcome | 0.636 [0.363, 0.909] | 0.496 [0.339, 0.643] | 0.000 | 0.091 | 0.456 [0.294, 0.606] | - | 0.868 |
| zs_needs_review | 0.091 [0.000, 0.300] | 0.305 [0.183, 0.440] | 0.000 | 0.000 | 0.301 [0.185, 0.438] | - | 0.004 |
| zs_risk | 0.545 [0.250, 0.818] | 0.553 [0.431, 0.663] | 0.000 | 0.000 | 0.565 [0.454, 0.663] | - | 0.880 |
| rule_error | 0.045 [0.000, 0.136] | 0.425 [0.397, 0.451] | 0.006 | 0.033 | 0.424 [0.398, 0.447] | - | - |
| oof_a_step_index | 1.000 [1.000, 1.000] | 0.942 [0.900, 0.981] | 0.455 | 0.909 | 0.783 [0.722, 0.880] | - | - |
| oof_b_action_type | 0.909 [0.727, 1.000] | 0.843 [0.724, 0.945] | 0.451 | 0.602 | 0.720 [0.571, 0.878] | - | - |
| oof_c_obs_tfidf | 0.727 [0.455, 1.000] | 0.749 [0.567, 0.920] | 0.455 | 0.455 | 0.735 [0.526, 0.915] | - | - |
| oof_d_text_tfidf | 1.000 [1.000, 1.000] | 0.994 [0.984, 1.000] | 0.818 | 1.000 | 0.908 [0.858, 0.964] | - | - |
| oof_e_probe_cls | 0.636 [0.363, 0.909] | 0.533 [0.357, 0.694] | 0.091 | 0.182 | 0.492 [0.332, 0.646] | - | - |
| oof_e_probe_markers | 0.273 [0.000, 0.556] | 0.383 [0.239, 0.549] | 0.091 | 0.091 | 0.426 [0.311, 0.544] | - | - |

extra: `{"no_tool_call_share_decisive": [0, 12], "no_tool_call_share_earlier": [0, 264], "runs_without_earlier_step_excluded_from_within_run_metrics": ["rs029"]}`

## apb, pre-action, cohort all_tasks (items 7393, clusters 200, episodes 418)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.280) | top-3 (chance 0.730) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.574 [0.525, 0.621] | 0.548 [0.511, 0.584] | 0.325 | 0.754 | 0.512 [0.500, 0.525] | 0.332 [0.288, 0.376] | 0.012 |
| zs_outcome | 0.519 [0.467, 0.568] | 0.505 [0.468, 0.542] | 0.258 | 0.756 | 0.505 [0.489, 0.520] | 0.326 [0.283, 0.368] | 0.924 |
| zs_needs_review | 0.517 [0.470, 0.563] | 0.509 [0.475, 0.545] | 0.294 | 0.715 | 0.487 [0.473, 0.500] | 0.317 [0.274, 0.360] | 0.924 |
| zs_risk | 0.531 [0.485, 0.573] | 0.513 [0.478, 0.544] | 0.294 | 0.758 | 0.497 [0.483, 0.510] | 0.323 [0.281, 0.365] | 0.585 |
| oof_a_step_index | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.741 [0.701, 0.777] | 0.655 [0.566, 0.727] | - |
| oof_b_action_type | 0.572 [0.511, 0.631] | 0.541 [0.489, 0.592] | 0.346 | 0.807 | 0.702 [0.668, 0.732] | 0.536 [0.476, 0.587] | - |
| oof_d_text_tfidf | 0.952 [0.927, 0.973] | 0.926 [0.903, 0.947] | 0.816 | 0.964 | 0.851 [0.802, 0.887] | 0.778 [0.709, 0.831] | - |
| oof_e_probe_cls | 0.459 [0.411, 0.510] | 0.465 [0.427, 0.502] | 0.254 | 0.718 | 0.497 [0.483, 0.510] | 0.332 [0.289, 0.376] | - |
| oof_e_probe_markers | 0.483 [0.437, 0.531] | 0.478 [0.442, 0.514] | 0.254 | 0.718 | 0.492 [0.479, 0.503] | 0.317 [0.274, 0.361] | - |

extra: `{"label_counts_pm1": [4983, 2410], "per_config_episode_win": {"bfcl": {"n": 126, "zs_action": 0.5238095238095238, "zs_outcome": 0.4603174603174603, "zs_needs_review": 0.49206349206349204, "zs_risk": 0.5, "oof_a_step_index": 1.0, "oof_b_action_type": 0.8055555555555556, "oof_d_text_tfidf": 0.8571428571428571, "oof_e_probe_cls": 0.46825396825396826, "oof_e_probe_markers": 0.47619047619047616}, "gaia_dev": {"n": 98, "zs_action": 0.5612244897959183, "zs_outcome": 0.5306122448979592, "zs_needs_review": 0.5102040816326531, "zs_risk": 0.5, "oof_a_step_index": 1.0, "oof_b_action_type": 0.3010204081632653, "oof_d_text_tfidf": 1.0, "oof_e_probe_cls": 0.45918367346938777, "oof_e_probe_markers": 0.5102040816326531}, "hotpotqa": {"n": 58, "zs_action": 0.5862068965517241, "zs_outcome": 0.4827586206896552, "zs_needs_review": 0.5517241379310345, "zs_risk": 0.4482758620689655, "oof_a_step_index": 1.0, "oof_b_action_type": 0.15517241379310345, "oof_d_text_tfidf": 0.9827586206896551, "oof_e_probe_cls": 0.43103448275862066, "oof_e_probe_markers": 0.4482758620689655}, "tau2": {"n": 136, "zs_action": 0.625, "zs_outcome": 0.5808823529411765, "zs_needs_review": 0.5294117647058824, "zs_risk": 0.6176470588235294, "oof_a_step_index": 1.0, "oof_b_action_type": 0.7279411764705882, "oof_d_text_tfidf": 0.9926470588235294, "oof_e_probe_cls": 0.4632352941176471, "oof_e_probe_markers": 0.4852941176470588}}}`

## apb, post-action, cohort all_tasks (items 7393, clusters 200, episodes 418)

| arm | win-rate vs earlier mean [CI] | percentile rank [CI] | top-1 (chance 0.280) | top-3 (chance 0.730) | AUROC (secondary) [CI] | AUPRC (secondary) | Holm adj. p |
|---|---|---|---|---|---|---|---|
| zs_action | 0.522 [0.476, 0.570] | 0.456 [0.420, 0.490] | 0.239 | 0.706 | 0.505 [0.491, 0.518] | 0.328 [0.284, 0.373] | 1.000 |
| zs_outcome | 0.498 [0.446, 0.547] | 0.500 [0.459, 0.540] | 0.273 | 0.761 | 0.508 [0.493, 0.522] | 0.335 [0.289, 0.380] | 1.000 |
| zs_needs_review | 0.522 [0.469, 0.574] | 0.516 [0.478, 0.557] | 0.285 | 0.756 | 0.496 [0.482, 0.510] | 0.324 [0.281, 0.367] | 1.000 |
| zs_risk | 0.507 [0.456, 0.552] | 0.501 [0.460, 0.539] | 0.273 | 0.734 | 0.500 [0.485, 0.515] | 0.324 [0.280, 0.368] | 1.000 |
| rule_error | 0.548 [0.512, 0.582] | 0.577 [0.550, 0.606] | 0.398 | 0.779 | 0.582 [0.567, 0.598] | 0.409 [0.362, 0.451] | - |
| oof_a_step_index | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.741 [0.701, 0.777] | 0.655 [0.566, 0.727] | - |
| oof_b_action_type | 0.572 [0.511, 0.631] | 0.541 [0.489, 0.592] | 0.346 | 0.807 | 0.702 [0.668, 0.732] | 0.536 [0.476, 0.587] | - |
| oof_c_obs_tfidf | 0.842 [0.801, 0.878] | 0.802 [0.769, 0.833] | 0.618 | 0.898 | 0.767 [0.733, 0.797] | 0.662 [0.607, 0.708] | - |
| oof_d_text_tfidf | 0.950 [0.924, 0.973] | 0.928 [0.905, 0.948] | 0.821 | 0.947 | 0.868 [0.838, 0.893] | 0.788 [0.727, 0.835] | - |
| oof_e_probe_cls | 0.536 [0.488, 0.584] | 0.521 [0.485, 0.556] | 0.289 | 0.742 | 0.508 [0.496, 0.521] | 0.329 [0.285, 0.372] | - |
| oof_e_probe_markers | 0.495 [0.451, 0.541] | 0.499 [0.463, 0.533] | 0.270 | 0.756 | 0.493 [0.479, 0.505] | 0.326 [0.281, 0.369] | - |

extra: `{"label_counts_pm1": [4983, 2410], "per_config_episode_win": {"bfcl": {"n": 126, "zs_action": 0.5, "zs_outcome": 0.5, "zs_needs_review": 0.5238095238095238, "zs_risk": 0.373015873015873, "rule_error": 0.4801587301587302, "oof_a_step_index": 1.0, "oof_b_action_type": 0.8055555555555556, "oof_c_obs_tfidf": 0.9166666666666666, "oof_d_text_tfidf": 0.873015873015873, "oof_e_probe_cls": 0.5476190476190477, "oof_e_probe_markers": 0.5317460317460317}, "gaia_dev": {"n": 98, "zs_action": 0.5306122448979592, "zs_outcome": 0.46938775510204084, "zs_needs_review": 0.5102040816326531, "zs_risk": 0.5, "rule_error": 0.5408163265306123, "oof_a_step_index": 1.0, "oof_b_action_type": 0.3010204081632653, "oof_c_obs_tfidf": 0.75, "oof_d_text_tfidf": 0.9897959183673469, "oof_e_probe_cls": 0.5204081632653061, "oof_e_probe_markers": 0.45918367346938777}, "hotpotqa": {"n": 58, "zs_action": 0.5, "zs_outcome": 0.4827586206896552, "zs_needs_review": 0.6206896551724138, "zs_risk": 0.6206896551724138, "rule_error": 0.49137931034482757, "oof_a_step_index": 1.0, "oof_b_action_type": 0.15517241379310345, "oof_c_obs_tfidf": 0.7931034482758621, "oof_d_text_tfidf": 0.9482758620689655, "oof_e_probe_cls": 0.5, "oof_e_probe_markers": 0.5517241379310345}, "tau2": {"n": 136, "zs_action": 0.5441176470588235, "zs_outcome": 0.5220588235294118, "zs_needs_review": 0.4852941176470588, "zs_risk": 0.5882352941176471, "rule_error": 0.6397058823529411, "oof_a_step_index": 1.0, "oof_b_action_type": 0.7279411764705882, "oof_c_ob`
