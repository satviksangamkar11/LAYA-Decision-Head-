# TPU study for the LAYA outcome-head run (2026-10-15, read-only)

What actually ran on Kaggle TPU v5e-8 in this project and produced output:
- Qwen proposal generation with **vLLM-TPU** (`vllm-tpu==0.28.0`, JAX `tpu_inference` backend, installed with `uv venv --python 3.12`,
  bf16, tensor parallel 8; `check_tpu.py` asserts 8 TPU devices via `jax.devices()` in a separate process because importing jax in the parent
  takes the libtpu lock before vLLM's engine subprocess). Evidence: `results/raw/tpu-qwen4b-smoke-v11-20261006-111316/` (smoke_result.json:
  60 decisions, 0.34 s per decision warm), `tpu-proposals-v14-v15-verified-20261007/`, `pool-v2-tpu-20261007/`.
- That is decoder-only text GENERATION through vLLM. It is not a PyTorch forward pass.

What has NOT run successfully here:
- **No torch_xla run exists.** `kaggle/tpu_test_capture_torch_xla.ipynb` and `tpu_test_vllm_generate.ipynb` have no saved outputs;
  `kaggle/state_recapture.py` uses `device="cuda"` (it ran on T4). The RUNBOOK's "cross-device validation passed (cosine >= 0.9999)" has no
  TPU result file behind it; the parity results found are T4 vs local GPU.
- No TPU kernel metadata in the repo (every saved kernel-metadata.json has `enable_tpu: false`).

Consequence for LAYA (ModernBERT encoder + custom decision head, laya 0.3.22 PyTorch library with GPU kernels and fp16 autocast):
- vLLM-TPU cannot run it (custom head, pooled scoring, not generation). The only TPU route is torch_xla, with no working precedent here:
  library kernels to bypass, fixed-shape padding to avoid recompiles, bf16 numerics needing a T4 parity gate (CLAUDE.md device rule).
- The proven path for exactly this model is T4 x2 (K2: 18,774 + 6,282 items, 0 failures, prob parity 5e-5 with the library).
