"""Kaggle/T4 variant of tools/capture_textemb_v3.py (a COPY; the original is the 3050 reference). State-free candidate-text embeddings for H0 (A5-v3), spec: configs/a5_v3_spec.toml [capture_v3] text_embeddings.
Same procedure: each unique text is embedded by a SEPARATE forward pass of the frozen Qwen3-4B-Thinking-2507 on 'Next action:' + ' ' + text and NOTHING else (no trajectory token),
depths 18/24/30, forward truncated after layer 30, mean over the candidate span, right padding (causal, so padding cannot reach real tokens), float16 storage, shards of 2048 texts,
the same batching constants (MAX_TOK 6000, MAX_SEQ 256). Differences, all declared: weights resident on the GPU (no layer streaming), fp16 compute, the text list comes from a JSON file
(so no decision rows, trajectories or labels are read here), --require_device, and the registered batch-vs-alone gate (20 random texts, cosine >= 0.9999) runs on EVERY shard (stricter than
the registered first-shard check; never looser). Output shards are dicts text -> fp16 vector, loadable by decision_head.v3_data.load_textemb (it merges all shards by text).
    PYTHONPATH=. python tools/capture_textemb_v3_t4.py --texts texts.json --outdir out/textemb --model DIR [--device cuda] [--dtype fp16] [--require_device T4] [--limit_texts N]"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import argparse, hashlib, json, random, time, tomllib
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture.capture_resident import _no_gqa_in_sdpa
from capture.render import _ids

ap = argparse.ArgumentParser()
ap.add_argument("--texts", required=True, help="JSON list of unique texts, sorted")
ap.add_argument("--outdir", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
ap.add_argument("--require_device", default=None)
ap.add_argument("--limit_texts", type=int, default=None)
ap.add_argument("--gate_texts", type=int, default=20)
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
D = tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))["capture_v3"]["depths"]
GATE = 0.9999        # registered: [capture_v3] text_embeddings, unchanged
NEUTRAL, SHARD, MAX_TOK, MAX_SEQ = "Next action:", 2048, 6000, 256
dev = torch.device(args.device)
if args.require_device:
    assert dev.type == "cuda" and torch.cuda.is_available() and args.require_device in torch.cuda.get_device_name(0), f"wrong device (need {args.require_device})"
texts = json.load(open(args.texts, encoding="utf-8"))
assert texts == sorted(set(texts)), "the text list must be sorted and unique"
if args.limit_texts:
    texts = texts[:args.limit_texts]
DT = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DT).eval()
model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:max(D)])
model = model.to(dev)
if dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] < 8:
    _no_gqa_in_sdpa()      # same attention path as the T4 state capture (capture/capture_resident.py)
base, store, calls = model.model, {}, {"lm": 0}
model.lm_head.register_forward_hook(lambda *a: calls.__setitem__("lm", calls["lm"] + 1))
for i, layer in enumerate(base.layers, 1):
    if i in D:
        layer.register_forward_hook(lambda m, a, k, o, i=i: store.__setitem__(i, o[0] if isinstance(o, tuple) else o), with_kwargs=True)
pre = _ids(tok, NEUTRAL)


@torch.inference_mode()
def embed(batch):
    seqs = [pre + _ids(tok, " " + x) for x in batch]
    L = max(map(len, seqs))
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
    store.clear()
    base(input_ids=ids.to(dev), use_cache=False)
    out = torch.zeros(len(batch), len(D), store[D[0]].shape[-1])
    for dj, d in enumerate(D):
        h = store[d].float()
        for j, s in enumerate(seqs):
            out[j, dj] = h[j, len(pre):len(s)].mean(0).cpu()
    return out


outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
json.dump(dict(device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", dtype=args.dtype, torch=torch.__version__, transformers=__import__("transformers").__version__,
               model=str(args.model), depths=D, texts=len(texts), texts_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest(), gate=GATE, shard=SHARD, max_tok=MAX_TOK, max_seq=MAX_SEQ),
          open(outdir / "textemb-device.json", "w"), indent=1)
t0, done, worst = time.time(), 0, 1.0
for s0 in range(0, len(texts), SHARD):
    f = outdir / f"shard-{s0 // SHARD:04d}.pt"
    if f.exists():
        continue
    chunk = sorted(texts[s0:s0 + SHARD], key=lambda x: len(_ids(tok, x)))
    emb, i = {}, 0
    while i < len(chunk):
        j, toks = i, 0
        while j < len(chunk) and j - i < MAX_SEQ and toks + len(pre) + len(_ids(tok, " " + chunk[j])) <= MAX_TOK:
            toks += len(pre) + len(_ids(tok, " " + chunk[j])); j += 1
        j = max(j, i + 1)
        e = embed(chunk[i:j])
        for k, x in enumerate(chunk[i:j]):
            emb[x] = e[k]
        i = j
    for x in random.Random(s0).sample(chunk, min(args.gate_texts, len(chunk))):      # gate on every shard
        c = F.cosine_similarity(embed([x])[0].double(), emb[x].double(), dim=-1).min().item()
        worst = min(worst, c)
        assert c >= GATE, f"GATE FAILED (batch vs alone): cosine {c:.6f} < {GATE} for {x[:60]!r}"
    assert all(torch.isfinite(v).all() and v.abs().max() < 60000 for v in emb.values())
    torch.save({k: v.half() for k, v in emb.items()}, f)
    done += len(emb)
    print(f"shard {s0 // SHARD} saved, texts so far {done}, worst gate cosine {worst:.6f}, minutes {(time.time() - t0) / 60:.1f}", flush=True)
assert calls["lm"] == 0
print("DONE texts", len(texts), "new", done, "worst_gate_cosine", round(worst, 6), "lm_head_calls", calls["lm"], "minutes", round((time.time() - t0) / 60, 1))
