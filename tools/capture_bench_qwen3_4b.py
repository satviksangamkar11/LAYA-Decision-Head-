"""Capture throughput benchmark for Qwen3-4B-Thinking-2507. Thresholds: configs/thresholds.toml [capture_qwen3_4b] v1."""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, statistics, time, tomllib
from datetime import datetime, timezone
import psutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = r"D:\local model\models\qwen3-4b-thinking-2507"
th = tomllib.load(open("configs/thresholds.toml", "rb"))["capture_qwen3_4b"]
out = "results/raw/capture-qwen3-4b-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
assert not os.path.exists(out)
tok = AutoTokenizer.from_pretrained(P)
text = ""
for f in ["README.md", "PLAN.md", "DECISION_HEAD.md", "CLAUDE.md", "FINAL_GPT_OSS_INTERNAL_LAYA_EXACT_IMPLEMENTATION_RUNBOOK.md"]:
    text += open(f, encoding="utf-8", errors="ignore").read() + "\n"
ids_all = tok(text, return_tensors="pt").input_ids[0]
assert len(ids_all) >= 16384, len(ids_all)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16).eval()
base = model.model
layers = base.layers
sweep = th["sweep_layers"]
state = dict(h2d=0.0, marks={})


def to_dev(a):
    if torch.is_tensor(a): return a.to("cuda")
    if isinstance(a, (tuple, list)): return type(a)(to_dev(v) for v in a)
    if isinstance(a, dict): return {k: to_dev(v) for k, v in a.items()}
    return a


def pre(mod, args, kwargs):
    t = time.perf_counter(); mod.to("cuda"); torch.cuda.synchronize(); state["h2d"] += time.perf_counter() - t
    return to_dev(args), to_dev(kwargs)


def make_post(i):
    def post(mod, args, kwargs, o):
        mod.to("cpu")
        if i in sweep:
            torch.cuda.synchronize(); state["marks"][i] = time.perf_counter()
        return o
    return post


for i, l in enumerate(layers, 1):
    l.register_forward_pre_hook(pre, with_kwargs=True)
    l.register_forward_hook(make_post(i), with_kwargs=True)
base.embed_tokens.register_forward_hook(lambda m, a, o: o.to("cuda"))
base.norm.to("cuda")


def one_pass(x):
    state["h2d"] = 0.0; state["marks"] = {}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        h = base(input_ids=x, use_cache=False).last_hidden_state
        h[:, -1].float().cpu()
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    return dt, state["h2d"], {k: v - t0 for k, v in state["marks"].items()}


def bench(name, nseq, L):
    x = ids_all[: nseq * L].reshape(nseq, L)
    toks = nseq * L
    one_pass(x)  # warm-up
    res = [one_pass(x) for _ in range(3)]
    dts = [r[0] for r in res]
    med = statistics.median(dts)
    depth = {str(k): round(toks / statistics.median([r[2][k] for r in res]), 1) for k in sweep}
    return dict(config=name, tokens_per_pass=toks, pass_seconds=[round(d, 3) for d in dts], tok_s_median=round(toks / med, 1),
                h2d_share=round(statistics.median([r[1] for r in res]) / med, 3), depth_tok_s=depth)


torch.cuda.reset_peak_memory_stats()
A = bench("A: 1x8192", 1, 8192)
B = bench("B: 2x4096", 2, 4096)
peak_vram = torch.cuda.max_memory_allocated() / 2**30
peak_ws = psutil.Process().memory_info().peak_wset / 2**30
v = A["tok_s_median"]
verdict = "GO" if v >= th["go_tok_s"] else "STOP_AND_ASK" if v < th["stop_tok_s"] else "TUNE"
doc = dict(gate="capture_qwen3_4b", thresholds="configs/thresholds.toml [capture_qwen3_4b] v1", A=A, B=B,
           peak_vram_gib=round(peak_vram, 2), peak_working_set_gib=round(peak_ws, 2), verdict=verdict)
with open(out, "x") as fh: json.dump(doc, fh, indent=1)
print(json.dumps(doc, indent=1)); print("->", out)
