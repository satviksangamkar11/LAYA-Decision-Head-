# Kaggle workflow: head training rows (exploratory, no gate)

Nothing is read from or uploaded from this machine. Both notebooks download from the HuggingFace hub inside Kaggle (internet ON).

| Step | Notebook | Device | Output |
|---|---|---|---|
| 1 | `kaggle/laya-training-data-builder.ipynb` | CPU | `/kaggle/working/head_rows/` rows.jsonl, manifest.json, shortcut-check.json |
| 2 | `kaggle/laya-qwen-sampling.ipynb` | T4 | `/kaggle/working/head_states/` shard-*.pt (q, cx at depths 18/24/30, Qwen option probs), manifest.json |

Run step 1, "Save version", attach its output as an input of step 2.

## Sources (ids and revisions verified on HF 2026-10-04)
- `nebius/SWE-rebench-openhands-trajectories` (revision recorded at run time; the old id without `-trajectories` was wrong)
- `SWE-bench/SWE-smith-trajectories`, `data/tool-*.parquet` only (1.03 GB). Changed from `train-*`: the datasets-server exposes only the tool/xml/ticks formats, `tool` has a verified schema (messages, resolved, instance_id, traj_id) and matches `tools/conductor_data.py`. The 8.17 GB figure was storage history
- `DSFFGFG456/fable-5-coding-and-debugging-traces` pinned at `dd283de3f8b7ab5ecb2f4a26a22717f162d71557` (no outcome labels)
- `S4MPL3BI4S/coding-decision-cases` pinned at `2ebe9040038897d8e3037c672d0dccb958a04a1c`, used only to derive contaminated repos inside the notebook
- Qwen: `Qwen/Qwen3-4B-Thinking-2507` at `768f209d9ea81521153ed38c47d515654e938aea`, never trained

## Design
- Each decision: gold action + 2-3 synthetic hard negatives (other_step, wrong_path, wrong_tool, premature_finish; `executed=false`, no outcome) + NONE.
- Real outcome label (`resolved`) only for nebius and swesmith. Fable has none.
- Contaminated repos (seen by Coding_Decision_Agent) are tagged and never in TEST; they may be in TRAIN/CAL (DATASETS_INTERNAL_HEAD.md 2.7). Split by repo hash.
- Shortcut check (candidate-text-only baseline, margin 0.15 fixed before any result) runs at the end of step 1. If `flagged` is true, fix the negatives before capturing states.
- Step 2 is a new T4 capture protocol (`pool_v1_kaggle`), not the registered A5-v3 capture; compare only within this run.

## Not done / unmeasured
- Schemas were read from the HF datasets-server (local pyarrow is not installed). The first code cell downloads one shard per source and asserts the columns, so a mismatch fails before any large download.
- Fable `assistant_step` is an integer index; the gold action is `messages[target_message_index]` (fixed in the builder).
- Nebius message keys beyond role/content/tool_calls were not seen directly (preview truncated); the builder handles them as in `conductor_data.py`.
- Trajectory caps (nebius 4000, swesmith 3000) and `s/decision` on T4 are unmeasured guesses; edit `CFG` after the first shard timing.
- Qwen free-form proposals are not generated (thinking model is slow); Qwen's signal is its option-letter probabilities.
- Head training script does not exist yet; it runs on Kaggle, not the local 3050.

## Known risk, first run (builder v3, reported by the inspector 2026-10-04; figures not re-measured here)
Shortcut check FLAGGED: candidate-text-only baseline picks the gold action 47.1% vs 26.4% chance (margin 20.7 points over the 0.15 pre-set limit). Hard negatives may be separable by text pattern (for example the constant `premature_finish` text). Decision (inspector/user relay): hard negatives are NOT changed yet; assess after Qwen scoring. Until then any head accuracy on these rows may reflect text artifacts, and per-kind accuracy must be reported. Written before Qwen scoring; the 0.15 margin is not edited.
Builder v3 totals as reported: 46,826 rows (train 37,417 / cal 3,614 / test 5,795), zero bad-JSON, zero contaminated test rows.

## 2026-10-04 status
- Final state capture: 45,500 of 46,826 decisions in 91 shards (record: `results/raw/pool-states-v1-final-20261004-185656.json`); the last 1,326 are not captured and will not be re-run. Uploaded as the private dataset `laya-head-states-v1` (user-approved).
- Step 3 (built, NOT pushed or run): `kaggle/hybrid-options-count.ipynb`, Kaggle CPU, counts real wrong options per evidence tier (T1/T2/T3) from the full nebius and SWE-smith `tool` runs; logic in `tools/hybrid_options.py`. Recapture on the T4 waits for the counts and the user's yes (PLAN.md 15).
- Hybrid count v1 ran on Kaggle CPU (scan 634 s, build 36 s [M]); result in `results/raw/hybrid-options-count-v1-20261004.json`. The SWE-smith parser skipped 62% of runs (XML-in-text calls); fixed, v2 notebook regenerated (not pushed by me).
