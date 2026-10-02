"""Capture cost model from the measured CAL capture log (no model, no accuracy). Output is create-only: results/raw/capture-cost-model-<time>.json.
    .venv/Scripts/python.exe tools/capture_cost_model.py
Measured: tokens per decision and clean seconds per decision on the RTX 3050. Everything else is an ESTIMATE (unmeasured): the FLOP model, GPU peak speeds
(published specs as recalled, unverified), utilisation (MFU) and prices (pasted by the user from the AIC Cloud page, unverified)."""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
L = [json.loads(l) for l in open(ROOT / "data/v3/capture-log-cal.jsonl", encoding="utf-8")]
R = [(i, r) for i, r in enumerate(L) if "seconds" in r]
tok = lambda r: r["prefix_tokens"] + r["suffix_tokens"]
T = np.array([tok(r) for _, r in R])
# clean timing: ungated, before the concurrent-process period (log index < 460), outliers dropped
C = [(r["seconds"], tok(r)) for i, r in R if 10 <= i < 460 and not r["gates"] and r["seconds"] < 200]
s, t = np.array([c[0] for c in C]), np.array([c[1] for c in C])
H, NQ, NKV, HD, I, LAY = 2560, 32, 8, 128, 9728, 30          # Qwen3-4B-Thinking-2507 config, forward truncated after layer 30
per_layer = H * NQ * HD + 2 * H * NKV * HD + NQ * HD * H + 3 * H * I
flops = lambda n: n * 2 * LAY * per_layer + 0.6 * 4 * n * n * NQ * HD * LAY     # matmul + 0.6 x full attention (block mask)
FL = float(np.mean([flops(x) for x in T]))
FWD = 1.15                                                  # forwards per decision incl. every-50th and every-200th gates (6/50 + 6/200 extra)
GPUS = {"RTX 3090": (36, 71), "RTX 4090": (69, 165), "RTX A6000": (73, 155), "RTX PRO 5000": (116, 260), "RTX 6000 Ada": (120, 364)}   # INR/h, BF16 peak TFLOPS
WORK = {"stage2_capture_cal_train_locked": 4775 * FWD, "stage3_probe_1000": 1000 * FWD, "stage4_lora_10k": 10000 + 3.3 * 10000 * 2 + 2 * 3.3 * 2000 * 2,
        "stage4_lora_30k": 30000 + 3.3 * 30000 * 2 + 2 * 3.3 * 2000 * 2}   # forward-equivalents; LoRA backward with checkpointing about 3.3x a forward
out = dict(created=datetime.now(timezone.utc).isoformat(), source="data/v3/capture-log-cal.jsonl (measured on the RTX 3050)",
           measured=dict(decisions_logged=len(R), tokens_mean=float(T.mean()), tokens_median=float(np.median(T)), tokens_p99=float(np.percentile(T, 99)), tokens_max=int(T.max()),
                         prefix_mean=float(np.mean([r["prefix_tokens"] for _, r in R])), suffix_mean=float(np.mean([r["suffix_tokens"] for _, r in R])),
                         pool_mean=float(np.mean([r["pool"] for _, r in R])), clean_ungated_n=len(C), sec_per_decision_mean=float(s.mean()), sec_per_decision_median=float(np.median(s))),
           estimates_unmeasured=dict(flop_per_decision=FL, effective_tflops_rtx3050=float(np.mean([flops(x) / y for y, x in C]) / 1e12), forwards_per_decision=FWD,
                                     note="FLOP model = 2*params*tokens + 0.6*full attention; 3050 effective TFLOPS is derived from it, not read from a profiler"),
           gpus={})
for g, (price, peak) in GPUS.items():
    d = {"inr_per_hour": price, "bf16_peak_tflops_unverified": peak, "tflops_per_inr": round(peak / price, 2)}
    for mfu in (0.25, 0.35, 0.45):
        sec = FL / (peak * 1e12 * mfu)
        d[f"mfu_{int(mfu * 100)}"] = {"sec_per_forward": round(sec, 3), **{k: dict(hours=round(w * sec / 3600, 2), inr_no_contingency=round(w * sec / 3600 * price)) for k, w in WORK.items()}}
    out["gpus"][g] = d
out["assumptions"] = ["prices pasted by the user from the AIC Cloud page, not checked against the page; GST not included", "MFU 25/35/45% is an assumption; Phase 1 smoke measures the real value",
                      "stage-2 fixed extras (setup, smoke, text embeddings, head fits) are added by the plan, not here", "unchanged streaming capture code adds weight-transfer time per forward; the plan assumes a resident-weights mode"]
p = ROOT / "results/raw" / f"capture-cost-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
with open(p, "x") as f:
    json.dump(out, f, indent=1)
print(p, "\ntokens mean", round(T.mean()), "max", T.max(), "| clean s/decision", round(s.mean(), 2), "| FLOP/decision %.1f T" % (FL / 1e12))
