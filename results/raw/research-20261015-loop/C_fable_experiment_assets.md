# C. Fable imitation experiment assets (read-only inventory, 2026-10-15)

Scope: files read as metadata/JSON only. No models, Kaggle, training or API calls. CAL and TEST contents were NOT opened (only folder names and counts quoted from docs). Paths are relative to `D:/local model/`.

## 1. Frozen dataset v3 (the training pool set)
- Folder: `results/raw/laya-fullaction-set-v3-20261012T055420Z/`
  - `FROZEN_SET.json` (manifest, sha256 of parts), `requests.jsonl.gz` (335 decisions: decision_id, context_text, options, hashes), `private_labels.jsonl.gz` (335 rows; Fable's recorded action is the positive), `PER_OPTION_VERIFICATION.json`, `SOURCES.json`.
- Size: **335 pools, 866 alternatives, 192 trajectories.** Options per pool: 3 = 182, 4 = 110, 5 = 43 (Fable action + 2-4 admitted alternatives).
- Roles (counts only, in FROZEN_SET.json; the per-pool role is NOT stored in the frozen files):
  - `train`: 120 pools / 94 traj / 297 alt
  - `train_extension`: 54 / 35 / 136
  - `heldout_transfer`: 161 / 98 / 433
- Per-pool role can be recovered: the 214 non-train pools are in `results/raw/align-transfer-v1-20261012T055725Z/REPORT-20261012T070534Z.json` (`per_pool.role`); the 120 train pools are exactly the v3 ids with level-1 features in `level12-run-v2-cancelled` (see section 4).
- Cleaned copies (use one, state which): `results/raw/laya-fullaction-clean-v4-20261012T130553Z/train_v4/` (304 pools, 779 alt, 177 traj; roles 112 / 49 / 143), with `REMOVED.json`; rule in `tools/laya_clean_v4.py`.
- Earlier versions (superseded, do not mix): `laya-fullaction-set-v2-20261011T134826Z` (123 pools), `laya-fullaction-set-v1-20261011T082326Z`, `laya-fullaction-stage1-v1-20261011T203622Z` (59 pools).

### Where candidate options come from
- Generator for ALL v3 pools: **NVIDIA `nvidia/nemotron-3-super-120b-a12b`** (not Qwen, not GLM). Driver `tools/nemotron_candidate_pilot.py` (`run_parallel`). Settings (`SETTINGS_V3`): reasoning effort high, reasoning budget 16384, max_tokens 32768, temperature 0.5, top_p 1. Protocol v3t = `protocol_builder_v3t` in `tools/fable_teacher_runner.py` (registry v3). The generator never sees the recorded action.
- Prompts / configs by batch (all under `results/raw/`):
  - `batch_v2` (68 train pools): `hard-batch-v2-20261010T190132Z/`; script `tools/hard_batch_v2.py`, selection `tools/hard_pools.py`; prompt "original".
  - `conventions_40` (16) and `conventions_95` (36): `hard-batch-v3-conv-20261010T225509Z/`, `hard-batch-v3-conv-20261010T231422Z/`; script `tools/hard_batch_v3_conv.py`; prompt "session_conventions_v1" = the base prompt plus the addendum in `CONVENTIONS` (lines 31-35 of that file).
  - `stage1_200` (11 ext + 47 held-out): `hard-batch-v3-conv-20261011T182620Z/`.
  - `remaining_471` (43 ext + 114 held-out): `hard-batch-v3-conv-20261011T204850Z/`.
  - VAL build: `hard-batch-v3-conv-20261012T081844Z/build-20261012T085641Z/strict-20261012T104225Z/`.
- Samples per decision: 5 (v3t) and 3 (toolcond) per `hard_batch_v2.py` / `hard_batch_v3_conv.py`.
- Build chain: `tools/hard_batch_v2_build.py` (screen), `tools/hard_batch_v2_assemble.py` (STRICT_REPORT, NEARDUP_FILTER), `tools/hard_qualify_v1.py` and `tools/hard_qualify_v2_syntax.py` (parsers), freeze `tools/laya_fullaction_freeze_v3.py`.
- Blinded judges (both must accept, per option): GLM-5.3 full reasoning (`z-ai/glm-5.3`, `glm_max64`) plus second judge: Gemma 4 31B (`google/gemma-4-31b-it`, `gemma_stream`) for batch_v2 (68 pools), Muse Glimmer 30B (`meta/muse-glimmer-30b`, `muse_max`) for the rest (267 pools). Scripts: `tools/pair_pilot_v1.py`, `tools/hard_review_v2.py` (review packets), `tools/judge_qual_v1.py` (also lists `moonshotai/kimi-k3`, not used in v3 pools).

### Rejected / screened candidates: SAVED, usable without new generation
- Per-option screen results: `<build>/SCREEN_OPTIONS.jsonl` (e.g. `hard-batch-v3-conv-20261011T182620Z/build-20261011T191653Z/`: 518 rows; fields decision_id, origin, kind, lang, tech, fails, same_tool_as_fable; 488 have no fail). Syntax/parser logs: `SYNTAX*.jsonl`, `PARSER_LEVELS_v*.jsonl`.
- Raw Nemotron samples NOT admitted to pools: `<build>/extra_all.jsonl.gz` (1,020 rows, 198 decisions; origins `nemotron_v3t_gen` 497 and `nemotron_toolcond_gen` 523; fields decision_id, origin, sample_idx, calls, reply). Caveat: `reply` was null in the rows I checked; not verified further.
- Judge verdicts per option (including rejected and disputed): `<batch>/build-*/pair-review-rest-*/` and `pair-pilot-*/` (PAIR_REPORT-*.json, REVIEW_RESULT-*.json). Build dirs: `hard-batch-v2-20261010T190132Z/build-20261010T212540Z/`, `hard-batch-v3-conv-20261010T225509Z/build-20261010T230858Z/`, `...20261010T231422Z/build-20261010T235616Z/`, `...20261011T182620Z/build-20261011T191653Z/`, `...20261011T204850Z/build-20261011T214554Z/`.
- Per-decision generator samples for the other batches are not individually listed here (not found beyond `extra_all`).
- Any new candidate for an existing decision still needs the screen and the pair review before it can enter a pool. The frozen set is create-only.

## 2. VAL pools (development evaluation only; never trained on)
- Raw: `results/raw/laya-fullaction-val-v1-20261012T104239Z/` (53 pools, 36 traj, 134 alt; `FROZEN_SET.json`, `requests.jsonl.gz`, `private_labels.jsonl.gz`, `PER_OPTION_VERIFICATION.json`).
- Cleaned (46 pools, 32 traj, 113 alt; PRIMARY): `results/raw/laya-fullaction-clean-v4-20261012T130553Z/val_v1_clean/` (`requests.jsonl.gz`, `private_labels.jsonl.gz`, `REMOVED.json`). Summary: `CLEAN_REPORT.json` in the parent folder. 14 alternatives removed, 7 pools dropped.
- Per-pool VAL scores: `results/raw/val-eval-v1-20261012T082055Z/REPORT-20261012T111926Z.json` -> `per_pool` (53 entries: head_real, head_no_context, probe_real, probe_no_context, options_only_text).

## 3. CAL / TEST (protected)
- Not in v3. The FROZEN_SET eligibility inventory states VAL 227 / CAL 309 / TEST 713 decisions have no pools.
- Folders seen by name only: `results/raw/cal-outcome-v1-20261012T133201Z/` (outcome task, not imitation). No TEST imitation folder found. Contents not opened.

## 4. Headline results (each figure checked in its file)
| experiment | number | source file (reusable per-pool data) |
|---|---|---|
| Pretrained head, no training, 123-pool set | 36.6% (CI 31.5-41.9) | `results/raw/fullaction-pilot-v1-20261011T182942Z/REPORT-20261011T193329Z.json` (`per_pool.hits.*`, 123 entries); `AUDIT2.json` (`per_pool.probe.*` chosen and scores). Pretrained head on cleaned VAL, head-only: 39.1% (`results/raw/laya-headonly-v1-20261012T161940Z/REPORT-20261012T163328Z.json`). |
| Adapter route, fit on 120 train pools, scored on 160 unseen pools | 63.1% (56.8-69.4); probe 75.6% | `results/raw/align-transfer-v1-20261012T055725Z/REPORT-20261012T070534Z.json` (`per_pool`: decision_id, role, before/after per seed for fresh, aligned_A, aligned_B). |
| Same adapters on original VAL (53), fit on TRAIN only | 79.2% (67.9-89.3); no context 64.2% (52.0-76.3); probe 94.3% real / 81.1% no context; options-only text 67.9% | `results/raw/val-eval-v1-20261012T082055Z/REPORT-20261012T111926Z.json` (`per_pool`). Adapters: `results/raw/laya-finetune-all-v1-20261012T080145Z/adapters/final-s{0,1,2}.pt` (hashes in `results/raw/laya-baseline-v1-20261012T215242Z/MANIFEST.json`). |
| Adapter-only vs adapter + head (joint), cleaned VAL 46 | 76.1% vs 75.4% (diff -0.7, CI -5.9 to +4.3); probe 89.1% | `results/raw/laya-joint-v1-20261012T130647Z/REPORT-20261012T131617Z.json`; checkpoints `models/final-adapter_only-s*.pt`, `models/final-joint-s*.pt`. Per-pool arrays: not found in this report (per-run logs in `runs/`). |
| Head swap, out-of-fold on 303 train pools | pretrained coding head 77.2%; random heads 83.4% (83.8/83.2/83.3); probe 85.8% (CI 81.5-89.9) | `results/raw/head-swap-run-20261012T222906Z/REPORT.json` (`primary`, `linear_probe_out_of_fold`); `runs.jsonl` (per-run `test_pick` arrays). Cleaned VAL (46): coding 78.3% (82.6/78.3/73.9), probe 89.1%. Rebuild check: `results/raw/head-swap-rebuild-run-20261012T225237Z/` (12 runs reproduce their 46 picks). |
| Text-native imitation v1 (frozen LAYA text encoder, head-only, listwise; different pipeline, no Qwen) | head 32.6% (20.0-46.2); probe 50.0% (35.4-64.6); options-only 75.0% (64.4-85.4); gate FAIL | `results/raw/text-imitation-v1-20261014T102605Z/full-v3/out/METRICS.json`, `REPORT.md`, `pool_scores.jsonl.gz` (381 rows: set, pool, traj, label, fold, head, unchanged, probe, options_only, action_prior). |

Caution: the figures use different train sets. The adapter route (63.1 and 79.2) and the joint run use all v3 pools with features, uncleaned; head swap and the joint run's cleaned VAL use train_v4 (303 with features). A loop experiment must name its train set.

## 5. Qwen states / LAYA features: already captured?
- Frozen feature provider: Qwen3-4B-Thinking-2507, revision `768f209d9ea81521153ed38c47d515654e938aea`, `models/qwen3-4b-thinking-2507` (never trained). Representation `fable_capture_v1`: hq = residual at depths 18/24/30 (3 x 2560, fp16); ho = per option (n x 3 x 2560). Limits: prefix 24,000 tokens, option 4,096, 8 options.
- Variants: B = real context; S = same-trajectory other state; X = other trajectory context; A = neutral 11-token prefix (no context).
- Coverage (re-scoring needs no new capture):

| pool set | pools | variants | files |
|---|---|---|---|
| v3 train (120) | 120 | A, B, S, X | `results/raw/level12-run-v2-cancelled-20261011T175252Z/out/level1/{A,B,S,X}.pt` + `meta.json` (copy: `level12-run-v2-20261011T160314Z/`) |
| v3 train_extension + heldout (214) | 214 | B only | `results/raw/v3-context-capture-run-20261012T065106Z/out/level1/B.pt` + `meta.json` (214 ok) |
| VAL original (53) | 53 | A, B | `results/raw/val-capture-run-20261012T111837Z/out/level1/{A,B}.pt` + `meta.json`; local parity copy `val-capture-local-20261012T111437Z/` |

- Not captured: 1 v3 non-train pool (over the prefix limit; the subset is 214 of 215). `level12-bundle-20261012T060047Z/MANIFEST.json` lists over-limit pools (first entry `fable5:py-azure-keyvault-pages:6`); section 15.24 of `HANDOFF_2026_10_11_STATUS.md` names `csharp-azure-arm-lro:7`. The two references disagree; not resolved here.
- Other captures (not v3 pools, or 1 overlap only): `mutation-capture-v2-run-20261013T114559Z/` (19 pools), `glm-pilot-capture-run-20261013T073754Z/` (A, B), `seq-capture-run-20261012T183217Z/` (1,187 requests, not mapped to v3).
- Capture scripts: `kaggle/level12_t4.py` (Level 1, VARIANTS env var; Kaggle GPU, not run here); v3 subset kernel `results/raw/level12-bundle-20261012T060047Z/kernel_ctx/laya-v3-context-capture-run.py` (VARIANTS = "B"); VAL kernel `results/raw/level12-bundle-20261012T104251Z/kernel_val/laya-val-capture-run.py`; Qwen loader and protocol `tools/fable_capture.py`; packaged copies under `*/pkg/capture/capture.py`.
- Probe recipe (logistic regression, C from 1e-3/1e-2/1e-1 by grouped CV): `kaggle/head_exp4_gpu.py` (`probe_scores`).
- Feature loader used by the head: `decision_head/fable_features.py`, `decision_head/fable_train.py`, `decision_head/head.py`.

## 6. Generation tooling (to add candidates the same way)
- NVIDIA key rotation: `tools/nvidia_keys.py` (KeyRotator; KEY_1..5 aliases from `.secrets/nvidia_api_key_2..6.txt`, 429 cools that key). Concurrency per key in `tools/nemotron_candidate_pilot.py` (`PAR_INFLIGHT` default 5; 39 sends/min per key per the user rules).
- Other NVIDIA generators, NOT used for v3 pools: `tools/nvidia_decision_batch.py` (DeepSeek v4.1 Flash, Nemotron Lightning/Super references), `tools/val_outcome_glm_v1.py` (GLM-5.3 outcome pilot, `reasoning_effort="max"`).
- Qwen3-30B proposals: `kaggle/qwen_proposals_v2.py` exists, but no v3 build script calls it (only `tools/fable_pool_selftest.py` references it). Not found in the v3 chain.
- Procedure to add candidates: `tools/nemotron_candidate_pilot.py prepare|run|evaluate <out_dir>`, then `tools/hard_batch_v2_build.py`, `tools/hard_batch_v2_assemble.py` (requires the pair reports), `tools/laya_clean_v4.py`, freeze with `tools/laya_fullaction_freeze_v3.py` (create-only; new filename).

## 7. Not found / open
- Per-pool fold assignment for v3 (only per-trajectory folds in summaries); `tools/align_transfer_v1.py` and `tools/laya_headonly_v1.py` define the grouping.
- Per-pool arrays for `laya-joint-v1` and `head-swap-run` REPORT.json (per-run data is in `runs/` and `runs.jsonl`).
- Outcome-head work (`results/raw/outcome-head-v1-20261015/`, gate FAIL, AUROC about 0.50) is a different target (run outcome, not next-action imitation); not part of this inventory.
