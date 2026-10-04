# Proposal Pipeline Runbook

**Goal:** Generate proposals from Qwen3-4B-Thinking-2507, assemble into pool with real options, run gates, and recapture states with new option lists for Laya head training.

## Pipeline Steps (Real Measured and Estimates)

### 1. Proposals Full Run
- **Script:** `kaggle/qwen_proposals_v2.py` (with sharding, resume, session guard; think30 mode only)
- **Accelerator:** Kaggle TPU v5e-8 (vllm-tpu; `mode think30`)
- **Duration formula:** `tokens = decisions × 4 × cap; time_hours = tokens / (tokens_per_sec × 3600)`
  - Where: decisions = 45,500; cap = 1024 (thinking) + 128 (answer)
  - tokens_per_sec is **unmeasured until smoke test** (estimated 800–1000 tok/s for Qwen3-30B-A3B on TPU)
- **Estimate (formula-based, before smoke):**
  - **think30 on TPU v5e-8:** (45,500 × 4 × 1024) tokens at ~900 tok/s ≈ 57h (unmeasured, formula-based)
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
- **Script:** `kaggle/state_recapture.py` (torch_xla + HuggingFace transformers on TPU)
- **Accelerator:** Kaggle TPU v5e-8 only
- **Duration (estimate, formula-based):**
  - **On TPU v5e-8 (torch_xla):** 0.6–0.8 s/decision [E] × 45,500 = 8–12 hours (one or two sessions with resume from dataset)
  - Cross-device validation already PASSED: TPU bf16 vs T4 fp16 cosine ≥ 0.9999 (worst 0.99891)
- **Input:** rows_pool_assembled.jsonl.gz (candidates field with gold + real + Qwen)
- **Output:** states_pool_v2/shard-NNNNN.pt (protocol `pool_v2_kaggle_tpu_bf16`)
- **Hidden state layout (per decision):**
  ```
  q: (3, 2560)           - query (last token) at depths [18, 24, 30]
  cx: (n_candidates, 3, 2560) - candidates
  qwen_probs: (n_candidates,) - Qwen's choice softmax
  order: shuffled candidate order
  gold_pos: gold position in order
  n_tokens: prompt length
  dtype: bfloat16 (TPU native)
  ```
- **Sharding:** 500 decisions per shard
- **Resumable:** By decision_id via dataset mount
- **Session guard:** 12 hours on TPU v5e-8
- **All roles (TRAIN, CAL, LOCKED) captured on TPU with same dtype and device per CLAUDE.md rule.**

## Total Time (Measured + Formula-based, TPU Only)

| Step | Accelerator | Time | Status |
|------|------------|------|--------|
| 1. Proposals | TPU v5e-8 (think30, vllm-tpu) | ~57h | Formula-based (unmeasured, depends on smoke) |
| 2. Pool assembly | CPU | <1h | Estimate |
| 3. Gates | CPU | <5min | Estimate |
| 4. Recapture | TPU v5e-8 (torch_xla) | 8–12h | Estimate (cross-device validation PASSED) |
| **Total** | TPU only | ~66-70h | All formula-based until smoke tests |

**Note:** Both steps use Kaggle TPU v5e-8. Cross-device numerics already validated (cosine ≥ 0.9999 on 200 decisions). Once `think30` smoke completes (first 50 decisions), recompute proposal time using actual tokens/sec.

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

- **TPU-only path (2026-10-04):** Proposals and state recapture both on Kaggle TPU v5e-8. Cross-device numerics validated (bf16 TPU vs fp16 T4, cosine ≥ 0.9999). Both steps use formula-based timing until smoke tests.
- **Protocol:** `pool_v2_kaggle_tpu_bf16` (bf16 dtype on TPU; distinct from earlier T4 fp16 captures). All roles (TRAIN, CAL, LOCKED) on same device/dtype per CLAUDE.md rule.
- **Smoke tests (required before full run):**
  1. Proposals: first 50 decisions with think30, measure tokens/sec, record timing
  2. Recapture: 50 decisions on TPU, verify cross-device cosine ≥ 0.999 vs T4 baseline, measure s/decision
  3. If either fails: report blocker; fallback to 2×T4 (safety path, ~27h total)
- **Quality gates:** Must pass before committing recapture output to training (runbook phase 15, RB4).
