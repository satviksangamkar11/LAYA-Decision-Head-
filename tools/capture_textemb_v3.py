"""State-free candidate-text embeddings for H0 (A5-v3). Rules: configs/a5_v3_spec.toml [capture_v3] text_embeddings.
Each unique pool text (plus the NONE option) is embedded by a SEPARATE forward pass of the frozen Qwen3-4B-Thinking-2507 on  'Next action:' + ' ' + text  and NOTHING else:
no trajectory token ever enters. Same depths (18/24/30), forward truncated after layer 30, mean over the candidate span, float16 storage. Batched with right padding
(causal attention, so padding cannot reach real tokens). Resumable shards; a gate re-runs 20 random texts alone and requires cosine >= 0.9999 against the batched result.
    PYTHONPATH=. .venv/Scripts/python.exe tools/capture_textemb_v3.py [--limit_texts N] [--outdir DIR]
The locked trajectories are opened only to rebuild candidate pools (logged); no accuracy is computed."""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import argparse, json, random, time, tomllib
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture.pool import NONE_TEXT
from capture.render import _ids
from compiler import a5v2
from compiler.candidates import distractor_pool
from compiler.firewall import load_cal_calibration_only, load_dev, log_locked_access

ap = argparse.ArgumentParser()
ap.add_argument("--limit_texts", type=int, default=None)
ap.add_argument("--outdir", default="data/v3/textemb")
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
D = tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))["capture_v3"]["depths"]
P = r"D:\local model\models\qwen3-4b-thinking-2507"
NEUTRAL, SHARD, MAX_TOK, MAX_SEQ = "Next action:", 2048, 6000, 256

# ---- collect the unique texts of every pool in the three manifests ----
raw_old = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
log_locked_access("capture_textemb_v3: rebuild locked candidate pools to collect candidate TEXTS; no outcomes")
raw_locked = a5v2.load_raw(ROOT / "data/locked_v3/fresh-locked.jsonl")
texts = {NONE_TEXT}
per_role = {}
for role in ("train", "cal", "locked"):
    ids = set(json.load(open(ROOT / f"data/v3/manifest-{role}.json", encoding="utf-8"))["decision_ids"])
    if role == "train":
        rows, raw = load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"), raw_old
    elif role == "cal":
        rows, raw = load_cal_calibration_only(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"), raw_old
    else:
        rows, raw = (json.loads(l) for l in open(ROOT / "data/decisions/locked-v3.jsonl", encoding="utf-8")), raw_locked
    n = 0
    for r in rows:
        if r["decision_id"] not in ids:
            continue
        tid, t = r["decision_id"].rsplit(":", 1)
        for p in distractor_pool(raw[tid]["trajectory"], int(t)):
            texts.add(p[0])
        n += 1
    per_role[role] = n
uniq = sorted(texts)
if args.limit_texts:
    uniq = uniq[:args.limit_texts]
print("decisions per role", per_role, "| unique texts", len(uniq), flush=True)

tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16).eval()
model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:max(D)])
base = model.model
store, calls = {}, {"lm": 0}
model.lm_head.register_forward_hook(lambda *a: calls.__setitem__("lm", calls["lm"] + 1))


def to_dev(a):
    if torch.is_tensor(a): return a.to("cuda")
    if isinstance(a, (tuple, list)): return type(a)(to_dev(v) for v in a)
    if isinstance(a, dict): return {k: to_dev(v) for k, v in a.items()}
    return a


for i, layer in enumerate(base.layers, 1):
    layer.register_forward_pre_hook(lambda m, a, k: (m.to("cuda"), (to_dev(a), to_dev(k)))[1], with_kwargs=True)

    def post(m, a, k, o, i=i):
        m.to("cpu")
        if i in D:
            store[i] = (o[0] if isinstance(o, tuple) else o)
        return o
    layer.register_forward_hook(post, with_kwargs=True)
base.embed_tokens.register_forward_hook(lambda m, a, o: o.to("cuda"))
base.norm.to("cuda")
pre = _ids(tok, NEUTRAL)


@torch.inference_mode()
def embed(batch):
    seqs = [pre + _ids(tok, " " + x) for x in batch]
    L = max(map(len, seqs))
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
    store.clear()
    base(input_ids=ids, use_cache=False)
    out = torch.zeros(len(batch), len(D), store[D[0]].shape[-1])
    for dj, d in enumerate(D):
        h = store[d].float()
        for j, s in enumerate(seqs):
            out[j, dj] = h[j, len(pre):len(s)].mean(0).cpu()
    return out


outdir = ROOT / args.outdir
outdir.mkdir(parents=True, exist_ok=True)
t0, done = time.time(), 0
for s0 in range(0, len(uniq), SHARD):
    f = outdir / f"shard-{s0 // SHARD:04d}.pt"
    if f.exists():
        continue
    chunk = sorted(uniq[s0:s0 + SHARD], key=lambda x: len(_ids(tok, x)))
    emb = {}
    i = 0
    while i < len(chunk):
        j, toks = i, 0
        while j < len(chunk) and j - i < MAX_SEQ and toks + len(pre) + len(_ids(tok, " " + chunk[j])) <= MAX_TOK:
            toks += len(pre) + len(_ids(tok, " " + chunk[j])); j += 1
        j = max(j, i + 1)
        e = embed(chunk[i:j])
        for k, x in enumerate(chunk[i:j]):
            emb[x] = e[k]
        i = j
    if s0 == 0:   # gate: batched padded result vs the same text alone
        sample = random.Random(0).sample(chunk, min(20, len(chunk)))
        for x in sample:
            alone = embed([x])[0]
            c = F.cosine_similarity(alone.double(), emb[x].double(), dim=-1)
            assert c.min().item() >= 0.9999, f"batched padding changed an embedding: cosine {c.min().item():.6f} for {x[:60]!r}"
    assert all(torch.isfinite(v).all() and v.abs().max() < 60000 for v in emb.values())
    torch.save({k: v.half() for k, v in emb.items()}, f)
    done += len(emb)
    print(f"shard {s0 // SHARD} saved, texts so far {done}, minutes {(time.time() - t0) / 60:.1f}", flush=True)
assert calls["lm"] == 0
print("DONE unique texts", len(uniq), "new", done, "lm_head_calls", calls["lm"], "minutes", round((time.time() - t0) / 60, 1))
