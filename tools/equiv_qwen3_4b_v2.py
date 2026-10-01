"""Equivalence check: Qwen3-4B-Thinking-2507, native CPU BF16 forward vs per-layer GPU streaming hooks.

Thresholds: configs/thresholds.toml [equiv_qwen3_4b] v1 (registered before this ran). Output is create-only.
"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")  # AVX-512 torch CPU kernels crash on this CPU
import json, sys, time, tomllib
from datetime import datetime, timezone
import torch
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

P = r"D:\local model\models\qwen3-4b-thinking-2507"
REV = "768f209d9e"
SRC = ["README.md", "PLAN.md", "compiler/state_builder.py", "DECISION_HEAD.md"]
NSEQ, LEN = 4, 256
th = tomllib.load(open("configs/thresholds.toml", "rb"))["equiv_qwen3_4b_v2"]
out_path = "results/raw/equiv-v2-qwen3-4b-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
assert not os.path.exists(out_path)

tok = AutoTokenizer.from_pretrained(P)
ids = []
for f in SRC[:NSEQ]:
    t = tok(open(f, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0]
    assert len(t) >= LEN, f
    ids.append(t[:LEN])
ids = torch.stack(ids)  # [4, 256], identical for every run


t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16).eval()
print("loaded", round(time.time() - t0, 1), "s; cuda", torch.cuda.is_available(), flush=True)


@torch.inference_mode()
def run(x):
    o = model(input_ids=x, output_hidden_states=True, use_cache=False)
    return o.logits.cpu().clone(), [h.cpu().clone() for h in o.hidden_states]


def to_dev(a, dev):
    if torch.is_tensor(a):
        return a.to(dev)
    if isinstance(a, (tuple, list)):
        return type(a)(to_dev(v, dev) for v in a)
    if isinstance(a, dict):
        return {k: to_dev(v, dev) for k, v in a.items()}
    return a


def stream_hooks(mods):
    hs = []
    for m in mods:
        def pre(mod, args, kwargs):
            mod.to("cuda")
            return to_dev(args, "cuda"), to_dev(kwargs, "cuda")

        def post(mod, args, kwargs, out):
            mod.to("cpu")
            return to_dev(out, "cpu")
        hs.append(m.register_forward_pre_hook(pre, with_kwargs=True))
        hs.append(m.register_forward_hook(post, with_kwargs=True))
    return hs



from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(P)
K = th["t1_layers"]
cfg.num_hidden_layers = K
if getattr(cfg, "layer_types", None):
    cfg.layer_types = cfg.layer_types[:K]
small = AutoModelForCausalLM.from_pretrained(P, config=cfg, dtype=torch.bfloat16).eval().to("cuda")
with torch.inference_mode():
    def go(m, x, dev="cuda"):
        o = m(input_ids=x.to(dev), output_hidden_states=True, use_cache=False)
        return [o.logits.cpu()] + [h.cpu() for h in o.hidden_states]
    res1, res2 = go(small, ids), go(small, ids)
small.to("cpu")
torch.cuda.empty_cache()
hk = stream_hooks(list(small.model.layers) + [small.lm_head])
with torch.inference_mode():
    strm = go(small, ids, "cpu")
for h in hk:
    h.remove()
def mad(a, b):
    return max((x.float() - y.float()).abs().max().item() for x, y in zip(a, b))
floor1, d1 = mad(res1, res2), mad(strm, res1)
t1_pass = d1 <= floor1
print("T1 floor(resident vs resident)", floor1, "streamed vs resident", d1, "PASS" if t1_pass else "FAIL", flush=True)
del small

t0 = time.time()
ref_logits, ref_h = run(ids)                      # R: batch of 4
print("R", round(time.time() - t0, 1), "s", flush=True)
t0 = time.time()
solo = [run(ids[i:i + 1]) for i in range(NSEQ)]   # F: each alone
print("F", round(time.time() - t0, 1), "s", flush=True)

hooks = stream_hooks(list(model.model.layers) + [model.lm_head])
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
ours_logits, ours_h = run(ids)                    # O: streamed through GPU
torch.cuda.synchronize()
peak_vram = torch.cuda.max_memory_allocated() / 2**30
print("O", round(time.time() - t0, 1), "s; peak VRAM GiB", round(peak_vram, 2), flush=True)
for h in hooks:
    h.remove()


def metrics(ref_l, ref_hs, x_l, x_hs, i, xi):
    a, b = ref_l[i].float(), x_l[xi].float()
    lp_a, lp_b = Fn.log_softmax(a, -1), Fn.log_softmax(b, -1)
    kl = (lp_a.exp() * (lp_a - lp_b)).sum(-1).mean().item()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    cos = [Fn.cosine_similarity(ra[i].float(), xa[xi].float(), dim=-1).mean().item()
           for ra, xa in zip(ref_hs[1:], x_hs[1:])]
    t2 = a.topk(2, -1).values
    gap = (t2[:, 0] - t2[:, 1])
    dis = a.argmax(-1) != b.argmax(-1)
    big = gap >= 0.5
    diag = dict(n_disagree=int(dis.sum()), gap_at_disagree_max=(gap[dis].max().item() if dis.any() else 0.0),
                gap_at_disagree_mean=(gap[dis].mean().item() if dis.any() else 0.0),
                top1_where_gap_ge_0_5=((~dis[big]).float().mean().item() if big.any() else 1.0), n_gap_ge_0_5=int(big.sum()))
    return dict(kl=kl, top1=top1, diag=diag, final_cos=cos[-1], min_layer_cos=min(cos), max_logit_abs=(a - b).abs().max().item())


rows, ok = [], True
for i in range(NSEQ):
    F = metrics(ref_logits, ref_h, solo[i][0], solo[i][1], i, 0)
    O = metrics(ref_logits, ref_h, ours_logits, ours_h, i, i)
    chk = dict(
        kl=O["kl"] <= th["t2_kl_abs_cap"],
        final_cos=O["final_cos"] >= th["t2_final_cos_min"],
        layer_cos=O["min_layer_cos"] >= th["t2_layer_cos_min"])
    ok &= all(chk.values())
    rows.append(dict(seq=SRC[i], floor=F, ours=O, checks=chk))
    print(SRC[i], "floor", {k: round(v, 6) for k, v in F.items() if k != 'diag'}, "ours", {k: (round(v, 6) if k != 'diag' else v) for k, v in O.items()}, chk, flush=True)

ok &= t1_pass
doc = dict(gate="equiv_qwen3_4b_v2", t1=dict(floor=floor1, streamed_vs_resident=d1, passed=t1_pass), thresholds="configs/thresholds.toml [equiv_qwen3_4b_v2]", revision=REV,
           peak_vram_gib=round(peak_vram, 2), rows=rows, verdict="PASS" if ok else "FAIL")
with open(out_path, "x") as fh:
    json.dump(doc, fh, indent=1)
print("verdict", doc["verdict"], "->", out_path)
