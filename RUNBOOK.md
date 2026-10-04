# Proposal Pipeline Runbook

**Goal:** Generate proposals from Qwen3-4B-Thinking-2507, assemble into pool with real options, run gates, and recapture states with new option lists for Laya head training.

## Pipeline Steps (Real Measured and Estimates)

### 1. Proposals Full Run
- **Script:** `kaggle/qwen_proposals_v2.py` (with sharding, resume, session guard)
- **Accelerator:** TPU v5e-8 (mode think30, primary) or 2× T4 (mode think_capped, fallback) or nothink (speed fallback)
- **Duration formula:** `tokens = decisions × 4 × cap; time_hours = tokens / (tokens_per_sec × 3600)`
  - Where: decisions = 45,500; cap = thinking cap in tokens (1024 for think30, 2048 for think_capped, 0 for nothink)
  - tokens_per_sec is **unmeasured until smoke test** (estimated 671–795 tok/s for Qwen3-4B on T4/TPU)
- **Preliminary estimates (before smoke):**
  - **think30 on TPU v5e-8 (primary):** (45,500 × 4 × 1024) tokens at ~800 tok/s ≈ 64h (unmeasured, formula-based)
  - **think_capped on 2×T4 (fallback):** (45,500 × 4 × 2048) tokens at ~800 tok/s ≈ 128h (unmeasured, formula-based)
  - **nothink on TPU v5e-8 (speed fallback):** (45,500 × 4 × 0) = 0 thinking, just answer ≈ 30–60m (unmeasured)
- **Input:** laya-head-states-v1 (45,500 decisions), laya-head-rows-v1 (rows.jsonl)
- **Output:** proposals/shard-NNNN.jsonl.gz (2,000 decisions/shard), MANIFEST-w{N}.json
- **Session guard:** Stop cleanly at 8h45m on TPU or 11h on 2×T4
- **Resumable:** By decision_id via dataset mount (RESUME_GLOB in v2)

### 2. Pool Assembly
- **Script:** `kaggle/pool_assembly.py`
- **Accelerator:** CPU (Kaggle kernel)
- **Duration (estimate):** < 1 hour
- **Input:** 
  - rows_pool_v2.jsonl.gz (gold + real options as candidates field)
  - proposals/shard-*.jsonl.gz (Qwen proposals from step 1)
- **Output:** rows_pool_assembled.jsonl.gz, MANIFEST_pool_assembled.json
- **Logic:**
  - Per decision: gold + real_run_* + qwen_proposal (canonical JSON dedupe) + NONE
  - Qwen hit: equals gold (never mislabeled)
  - Join rate: % with real options
  - Provenance and tier per option

### 3. Gates (Validation, Before Recapture)
- **Script:** `kaggle/gates.py`
- **Accelerator:** CPU (Kaggle kernel)
- **Duration (estimate):** < 5 minutes
- **Input:** rows_pool_assembled.jsonl.gz (from step 2)
- **Output:** PASS/FAIL printout, thresholds verified
- **Checks:**
  1. **Leak test:** Gold not in state_text (except prior history) → max_leaks = 0
  2. **Text baseline:** TF-IDF per option type (gold vs real_run vs qwen_proposal)
  3. **Qwen hit rate:** % of proposals matching gold per source → min_rate = 5%
  4. **Split counts:** train/cal/test disjoint (no decision_id overlap)
- **Thresholds file:** `configs/thresholds_pool_v2.toml` (must exist before run, written once)

### 4. State Recapture (New Option Lists)
- **Script:** `kaggle/state_recapture.py` (proven encode_pool + capture functions)
- **Accelerator:** T4 GPU (proven: 1 s/decision) or TPU v5e-8 (untested)
- **Duration:**
  - **On T4 (proven):** 1 s/decision × 45,500 = 12.6 hours (two sessions with resume from dataset)
  - **On TPU (untested):** estimate unknown
- **Input:** rows_pool_assembled.jsonl.gz (candidates field with gold + real + Qwen)
- **Output:** states_pool_v2/shard-NNNNN.pt (protocol pool_v1_kaggle)
- **Hidden state layout (per decision):**
  ```
  q: (3, 2560)           - query (last token) at depths [18, 24, 30]
  cx: (n_candidates, 3, 2560) - candidates
  qwen_probs: (n_candidates,) - Qwen's choice softmax
  order: shuffled candidate order
  gold_pos: gold position in order
  n_tokens: prompt length
  ```
- **Sharding:** 500 decisions per shard
- **Resumable:** By decision_id via dataset mount
- **Session guard:** 11 hours on T4

## Total Time (Measured + Formula-based)

| Step | Accelerator | Time | Status |
|------|------------|------|--------|
| 1. Proposals | TPU (think30) | ~64h | Formula-based (unmeasured, depends on smoke) |
| 2. Pool assembly | CPU | <1h | Estimate |
| 3. Gates | CPU | <5min | Estimate |
| 4. Recapture | T4 | 12.6h | Measured (proven, real data) |
| **Total** | Mixed | ~76-77h | Mixed (1 measured, 3 formula/estimate) |

**Note:** Proposal duration is preliminary and depends on smoke-test measurement of tokens/sec on the target accelerator. Once `think30` smoke completes (recommended: first 50 decisions), recompute using actual throughput and update the formula.

**Parallelization possible:** Steps 2–4 wait for step 1. Steps 2 and 3 can overlap.

## Files

| File | Type | Frozen |
|------|------|--------|
| `kaggle/qwen_proposals_v2.py` | Python (Kaggle) | YES (inspector canonical) |
| `kaggle/pool_assembly.py` | Python (CPU) | Ready (no test fixtures in main) |
| `kaggle/gates.py` | Python (CPU) | Ready (no test fixtures in main) |
| `kaggle/state_recapture.py` | Python (Kaggle T4) | Ready (real encode_pool + capture) |
| `configs/thresholds_pool_v2.toml` | TOML (config) | Must exist before step 3 |
| `RUNBOOK.md` | Markdown (this file) | Ready |

## Resume Strategy

- **Proposals:** Skip shards already in `/kaggle/working/proposals/` (RESUME_GLOB in v2)
- **Pool assembly:** Re-run (idempotent, fast)
- **Gates:** Re-run (idempotent)
- **Recapture:** Skip shards already in `/kaggle/working/states_pool_v2/`

## Dataset Mount for Multi-Session Runs

After step 1 (proposals), create a Kaggle dataset from `/kaggle/working/proposals/` and mount as `rows-pool-v2-partial` for step 2+. Similarly for step 4 (recapture), mount output as `states-pool-v2-partial` for continuation.

## Notes

- **Measured vs formula-based:** Only T4 recapture (12.6h per 45,500 decisions) is measured from real data. Proposal times use the formula `(decisions × 4 × cap) / tokens_per_sec`, where `tokens_per_sec` is estimated at 671–795 but not yet measured. Run a smoke test (first 50 decisions with think30) to measure actual throughput and refine the estimate.
- **TPU recapture:** `state_recapture.py` supports it but is untested; use T4 if TPU smoke fails.
- **Quality gates:** Must pass before committing recapture output to training (runbook phase 15, RB4).
- **Earlier estimates (12.5h think30, 19.4h think_capped):** Were not derived from a formula or measurement; replaced with formula-based estimates on 2026-10-04.
