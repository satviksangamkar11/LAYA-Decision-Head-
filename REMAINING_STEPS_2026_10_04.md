# Remaining Steps: Proposal Pool and Head Training (2026-10-04)

**Current state:** Pool assembly logic complete (user rule a-g); TPU path approved; cross-device validation passed; 45,500 of 46,826 decisions captured on T4.

## Steps 1-7: Proposal Generation & Pool Assembly (TPU only)

| Step | Owner | Accelerator | Input | Output | Gate | Status |
|---|---|---|---|---|---|---|
| 1. Smoke test proposals (think30) | Push | TPU v5e-8 (vllm-tpu) | 50 real decisions from laya-head-states-v1 | proposals/smoke/, timing | tokens/sec measured | Queued |
| 2. Full proposal run (think30) | Push | TPU v5e-8 (vllm-tpu) | 45,500 decisions (laya-head-states-v1) | proposals/shard-*.jsonl.gz (91 shards) + MANIFEST | parse_ok ≥ 95%, soft_target > 0 | Queued after smoke |
| 3. Pool assembly (user rule a-g) | Done | CPU (local/Kaggle) | proposals/ + rows_pool_v2.jsonl.gz + gold/real candidates | rows_pool_assembled.jsonl.gz + POOL_DROPS.json | exact_hit + near_hit counted, no style leak | ✅ Code ready, tests pass |
| 4. Quality gates (text baseline) | Push | CPU (Kaggle) | rows_pool_assembled.jsonl.gz | GATES_PASS.json (gold not findable, style not separable) | text_only_gold_top1 ≤ chance + 0.15 | Ready (qwen_proposals_v2 needed) |
| 5. TPU capture parity test | Push | TPU v5e-8 (torch_xla) | 200 real decisions from rows_pool_assembled | tpu_parity.json (cosine, agreement, KL, recompiles) | cosine ≥ 0.999 vs T4 baseline | New agent writing (thresholds_tpu_capture_v1.toml first) |
| 6. Full state recapture (TPU) | Push | TPU v5e-8 (torch_xla) | rows_pool_assembled.jsonl.gz | states_pool_v2/shard-*.pt (protocol pool_v2_kaggle_tpu_bf16) | all 3 splits (train/cal/test) same device/dtype | Queued after parity PASS |
| 7. Head training setup | Writer | Kaggle T4 x2 | states_pool_v2/ + soft targets (0.10 middle, from config) | head checkpoint, metrics.json | accuracy > Qwen alone, premature-stop not higher | Queued after recapture |

## Key Unknowns & Approvals

- **qwen_proposals_v2.py** (Inspector implements): 10 required features (raw_text, logprobs, exact/near hit, style fix, tool validation, diversity, manifest, per-source hit rate, context doc, split consistency)
- **tpu_capture_parity.py** (New agent): Stage A (HF + torch_xla) vs Stage B (JAX fallback); pre-register thresholds in configs/thresholds_tpu_capture_v1.toml
- **Thinking cap on 30B:** Measured from smoke, default 1024 tokens for think30 (user decision pending)
- **Soft targets:** Fixed prior (1.0/0.02/0.10/0.25/0.5) vs learnable (measurement pending)

## Files & Versions

**Datasets (Kaggle):**
- `laya-head-rows-v1` - 45,500 decision contexts with gold actions
- `laya-head-states-v1` - 91 shards of Qwen hidden states (captured T4 fp16, pool_v1_kaggle protocol)
- `qwen3-4b-thinking-2507` - model revision 768f209d9e

**Code:**
- `kaggle/qwen_proposals_v2.py` - modes: think30 (TPU), think_capped (T4), nothink (speedup). Selftested ✓
- `kaggle/pool_assembly.py` - user rule a-g implemented, tested ✓
- `kaggle/gates.py` - inverted logic (PASS = baseline cannot find gold), tested ✓
- `kaggle/state_recapture.py` - T4 only (T4_fp16); TPU version pending parity test

**Configs (BEFORE each run):**
- `configs/thresholds_pool_v2.toml` - created, gold findability gate
- `configs/thresholds_tpu_capture_v1.toml` - pending (new agent will write)

## Next 24 Hours

1. ✅ Pool assembly: code + tests done
2. ⏳ qwen_proposals_v2.py: Inspector implements 10 features
3. ⏳ tpu_capture_parity.py: New agent writes + runs smoke
4. 🔜 If parity PASS: run full TPU proposal generation (think30, ~57h)
5. 🔜 If parity PASS: run full TPU state recapture (torch_xla, ~8-12h)
6. 🔜 Head training on 45,500 states (Kaggle T4 x2)

**Blocker:** tpu_capture_parity.py result (torch_xla viability; fallback to 2×T4 if fails)

**Cost:** Rs 0 (Kaggle free quota, one session per account per accelerator, 20 h/week TPU)
