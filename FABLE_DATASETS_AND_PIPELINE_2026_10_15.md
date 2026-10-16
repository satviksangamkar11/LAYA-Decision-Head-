# Fable data: everything we hold, its structure, and the full process (2026-10-15)

Sources: `results/raw/research-20261015-loop/C_fable_experiment_assets.md`, `results/raw/research-20261012/A_fable_sources_inventory.md`,
`HANDOFF_2026_10_11_STATUS.md` (sections 15-16), `AGENT_README.md`, the frozen manifests under `results/raw/`. Counts quoted from those files;
"not verified" marks anything not re-read from data.

## 1. Raw Fable trajectories (what we hold)

| Item | Path | Structure | Size | Notes |
|---|---|---|---|---|
| **DSFFGFG456/fable-5-coding-and-debugging-traces** (the only raw Fable set downloaded; CC-BY-4.0) | `data/frontier/fable5-DSFFGFG456/` (`traces.jsonl`, `data/train-*.parquet` 255 shards, `dataset-manifest.json`) | one row per assistant STEP; cumulative `messages` (system / user / assistant / tool, with tool calls and results); trajectory metadata; tasks named like `bash-logtop`, `py-alert-rules`, `go-switchcase` | 12,490 rows, 2,380 trajectories; 14.4 MB parquet (663 MB JSONL) | Fable-5 attribution is a claim: `model_attested` false and `teacher_model` empty in the sampled rows |
| 18 other public "Fable 5" repos (inventory only, NOT downloaded) | `results/raw/research-20261012/A_fable_sources_inventory.md` | Cursor traces, Claude Code sessions, pi-agent traces, SFT parquet, a 229k-row mirror | not downloaded | many are duplicates (4 groups verified by hash); provenance of every one is an uploader claim |
| Dojigiri samples (Opus 4.8 reference runs), SWE-smith recovery | `data/frontier/dojigiri-samples`, `swesmith-recovery` | per-task rubric/pytest rewards, not per-step | 0.9 GB total with DSFFGFG456 | not Fable; not used for the Fable experiments |

## 2. The imitation set: from trajectories to pools (the process)

| # | Stage | What happens | Output / count | Code |
|---|---|---|---|---|
| 1 | Decision extraction | every tool-using assistant step of a trajectory is a decision (context = the history before it; Fable's recorded action = the positive) | canonical decision manifest | `decision_head/fable_imitation.py` |
| 2 | Whole-trajectory canonical split | TRAIN / VAL / CAL / TEST; no trajectory, task, repo or state hash in two splits (proof in `fable-fidelity-report-20261010/PROOF_REPORT.md`) | eligible decisions (coding, tool action, step >= 1): TRAIN 1,342 in 281 trajectories; VAL 227; CAL 309; TEST 713 | `tools/fable_fidelity_*` |
| 3 | Candidate generation | NVIDIA `nemotron-3-super-120b-a12b`, reasoning high, T 0.5, 5 samples (3 for tool-conditioned prompts) per decision; it never sees the recorded action; prompts "original" and "session_conventions_v1" (matches Fable's style) | 1,074 of 1,342 TRAIN decisions sent (268 first-step decisions not processed) | `tools/nemotron_candidate_pilot.py`, `hard_batch_v2.py`, `hard_batch_v3_conv.py` |
| 4 | Objective screen | parser/syntax levels, same-tool-as-Fable rule, near-duplicate and strict-equivalent filters | `SCREEN_OPTIONS.jsonl`, `SYNTAX*.jsonl`, `PARSER_LEVELS_v*.jsonl` per build | `tools/hard_batch_v2_build.py`, `hard_qualify_v*.py` |
| 5 | Blinded pair review | GLM-5.3 (full reasoning) AND a second judge (Gemma 4 31B for 68 pools, Muse Glimmer 30B for 267) must both accept each option; labels never sent | `pair-review-*`, `REVIEW_RESULT-*.json` | `tools/pair_pilot_v1.py`, `hard_review_v2.py` |
| 6 | Pool assembly | pool = Fable's action + 2-4 admitted alternatives (3-5 options); per-option verification; blind hand audit | yield 31% of generated decisions | `tools/hard_batch_v2_assemble.py` |
| 7 | Freeze v3 (create-only) | 335 pools, 866 alternatives, 192 trajectories; options per pool 3:182, 4:110, 5:43; roles train 120 / train_extension 54 / heldout_transfer 161 | `laya-fullaction-set-v3-20261012T055420Z/` | `tools/laya_fullaction_freeze_v3.py` |
| 8 | Clean v4 | removes strict equivalents (59, 6.8%) and 31 pools | train_v4: 304 pools (303 with features), 779 alternatives, 177 trajectories | `tools/laya_clean_v4.py` |
| 9 | VAL build (same pipeline, evaluation only) | 178 VAL decisions sent, 53 pools (36 trajectories, 134 alternatives); cleaned: 46 pools (32 trajectories, 113 alternatives) | `laya-fullaction-val-v1-*`, `laya-fullaction-clean-v4-*/val_v1_clean/` | same |
| 10 | Feature capture (Kaggle T4) | frozen Qwen3-4B-Thinking-2507: hq = last context token, ho = mean over each option's own tokens, depths 18/24/30, fp16; variants B real, A neutral prefix (no context), S other state of the same trajectory, X other trajectory | TRAIN 120: A,B,S,X; other 214 v3 pools: B only; VAL 53: A,B | `kaggle/level12_t4.py`, `tools/fable_capture.py` |
| 11 | Training / evaluation | adapter (8 tensors) + pretrained or other head, probe, options-only baseline; folds by trajectory | see section 4 | `tools/laya_finetune_all_v1.py`, `kaggle/head_swap_gpu.py`, ... |

Funnel losses among 995 TRAIN decisions (HANDOFF 16.1): 260 before build (114 too few candidates, 103 no candidate using Fable's tool), 23 at the
screen, 377 at review or assembly; 150 were near misses. CAL (309) and TEST (713) have no pools and were never read.

Structure of one pool (`requests.jsonl.gz` + `private_labels.jsonl.gz`): `pool_id` (= `fable5:<task>:<step>`), `trajectory_id`, `context_text`
(system, user, assistant, `[call]`, `[tool result: ...]` blocks), `options` (3-5 candidate actions as JSON tool calls), label index of Fable's
action in `private_labels` (kept separate from the requests).

## 3. Datasets derived from Fable decisions (other uses)

| Set | What | Count | Where |
|---|---|---|---|
| Replay states | decision states rebuilt from traces and replayed in the WSL sandbox (set 1 + Go repair; trace-reconstructed: only files the original session saw exist) | 115 canonical TRAIN states (74 trajectories), 58 VAL (41 trajectories) | `fable-replay-states-go-py-ts-strict-*`, HANDOFF 16.28 |
| Mutation sets | execution-verified edited variants of Fable's actions | 44-state screen, 19-state v1.1 | `mutation-*` |
| VAL oracle | untouched vs Fable's action, real tests | 39 useful, 23 clean fail-to-pass | `val-outcome-v1-*` |
| GLM pilots | generated fixes at VAL states | 10 states | `glm-pilot-*` |
| Branch pairs | Fable recorded branches | 51 pairs, 11 clearly same-state | research D |
| Text-imitation upload | v3 TRAIN 335 + cleaned VAL 46 as text pools | 381 pools | `text-imitation-v1-20261014T102605Z/upload/pools.jsonl.gz` |
| Earlier pool versions (superseded; do not mix) | v1, v1.1, v1.2, v2 (123 pools), stage1 (59) | | `fable-imitation-pools-*`, `laya-fullaction-set-v1/v2` |

Separate and not the v3 set: the 2026-10-04 Qwen capture of 45,500 of 46,826 decisions drawn from nebius, SWE-smith and Fable sources (smoke run:
20/20/20 per source) for the old pool-v2 pipeline (`pool-states-v1-final-*`). Not verified here beyond the docs.

## 4. Results on this data (all agreement with Fable's recorded action; none is task success)

| Experiment | Result | Source |
|---|---|---|
| Pretrained head, no training (123 pools) | 36.6% (chance 29.6%) | fullaction-pilot-v1 |
| Original LAYA with its own encoder, as released | 30.1% (108 of 123 contexts over its window) | HANDOFF 16.2 |
| Adapter + head, 120 pools, 160 unseen | 63.1%; probe 75.6% | align-transfer-v1 |
| Same on VAL 53 | 79.2%; no context 64.2%; probe 94.3%; options-only 67.9% | val-eval-v1 |
| Cleaned VAL 46: adapter only vs adapter+head | 76.1% vs 75.4%; probe 89.1% | laya-joint-v1 |
| Head swap (303 OOF) | pretrained 77.2%, random heads 83.4%, probe 85.8% | head-swap-run |
| Text-native imitation v1 (frozen LAYA text encoder, head-only) | head 32.6%, probe 50.0%, options-only 75.0%, gate FAIL | text-imitation-v1 |
| Loop engineering on cleaned VAL 46 | single pass 80.4% (3-seed average); probe 89.1%; best loop rule 42/46 = probe + 1 pool, reproduced by controls | loop-eng-v1-20261015 |

Reading: the options-only text baseline (68-76%) shows the authorship shortcut (Fable's action vs Nemotron-written alternatives); the probe uses
option states only (it never reads the query state); the pretrained LAYA head adds nothing over a random head.
