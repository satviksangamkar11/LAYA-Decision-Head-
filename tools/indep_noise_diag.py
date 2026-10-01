"""Diagnostic (not a gate): why did candidate independence read 0.999872 on the first real TRAIN decision? Measures, for one decision, the per-candidate cosine between
(a) full-pool packed capture and two half-pool recaptures (the registered gate's comparison), (b) full pool vs the same pool in reversed order (same sequence length),
(c) each of full / half vs a plain separate [prefix + suffix] forward, for the worst candidates. If the plain forward is as far from BOTH packed captures as they are from
each other, the gap is bf16 numerical noise of different sequence shapes, not dependence between candidates. Output create-only; nothing is gated or loosened here.
    PYTHONPATH=. .venv/Scripts/python.exe tools/indep_noise_diag.py DECISION_ID [ROLE]"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, sys, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture.capture import Capturer
from capture.pool import render_pool
from compiler import a5v2, state_builder
from compiler.firewall import load_dev

ROOT = Path(__file__).resolve().parents[1]
did = sys.argv[1]
D = tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))["capture_v3"]["depths"]
P = r"D:\local model\models\qwen3-4b-thinking-2507"
rev = json.load(open(ROOT / "results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
rec = next(r for r in load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl") if r["decision_id"] == did)
raw = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
ev = raw[did.rsplit(":", 1)[0]]["trajectory"]
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16).eval()
model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:max(D)])
C = Capturer(model, D)
st = state_builder.build_state(ev, int(did.rsplit(":", 1)[1]), 32000)
R = render_pool(rec, ev, st, tok, rev)
cos = lambda a, b: F.cosine_similarity(a.double(), b.double(), dim=-1)
n = len(R["suffix_ids"])
cap, _ = C.packed(R)
full = torch.stack([cap[d][1] for d in D], 1)                       # [n, depths, H]
perm = list(range(n))[::-1]
halves = [perm[:n // 2], perm[n // 2:]]
half = torch.zeros_like(full)
for part in halves:
    c2, _ = C.packed(dict(prefix_ids=R["prefix_ids"], query_pos=R["query_pos"], suffix_ids=[R["suffix_ids"][i] for i in part]))
    for j, i in enumerate(part):
        half[i] = torch.stack([c2[d][1][j] for d in D])
c3, _ = C.packed(dict(prefix_ids=R["prefix_ids"], query_pos=R["query_pos"], suffix_ids=[R["suffix_ids"][i] for i in perm]))
rev_full = torch.zeros_like(full)
for j, i in enumerate(perm):
    rev_full[i] = torch.stack([c3[d][1][j] for d in D])
fh, fr = cos(full, half), cos(full, rev_full)                       # [n, depths]
order = fh.min(1).values.argsort()[:8].tolist()
rows = []
for i in order:
    cp = C.plain(R, i)
    plain = torch.stack([cp[d][1][0] for d in D])
    rows.append(dict(candidate=i, tokens=len(R["suffix_ids"][i]), text=R["texts"][i][:70], full_vs_half=[round(x, 6) for x in fh[i].tolist()],
                     full_vs_reversed_full=[round(x, 6) for x in fr[i].tolist()], full_vs_plain=[round(x, 6) for x in cos(full[i], plain).tolist()],
                     half_vs_plain=[round(x, 6) for x in cos(half[i], plain).tolist()]))
out = dict(decision_id=did, n_candidates=n, prefix_tokens=R["n_prefix_tokens"], depths=D, n_below_0_9999_full_vs_half=int((fh < 0.9999).any(1).sum()),
           n_below_0_9999_full_vs_reversed=int((fr < 0.9999).any(1).sum()), min_full_vs_half=[round(x, 6) for x in fh.min(0).values.tolist()],
           min_full_vs_reversed=[round(x, 6) for x in fr.min(0).values.tolist()], median_full_vs_half=[round(x, 6) for x in fh.median(0).values.tolist()], worst_candidates=rows,
           note="plain = separate [prefix+suffix] forward; reversed = same pool, same sequence length, reversed candidate order")
p = ROOT / ("results/raw/indep-noise-diag-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "worst_candidates"}, indent=1))
for r in rows[:6]:
    print(r)
print("->", p)
