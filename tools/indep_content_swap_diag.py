"""Diagnostic (not a gate): exact candidate independence at IDENTICAL sequence shape. For several target candidates of one decision, replace the tokens of every OTHER candidate
by a deterministic shuffle of the other candidates' tokens (same lengths, same total length, same positions), recapture, and compare the target candidate's vector (all depths)
and the query vector with the original full-pool capture. If the block mask works, nothing the other candidates contain can reach the target, so the difference should be zero
(or at most kernel-level noise far below the shape-change noise measured by indep_noise_diag.py). Output create-only; nothing is gated or loosened here.
    PYTHONPATH=. .venv/Scripts/python.exe tools/indep_content_swap_diag.py DECISION_ID"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, random, sys, tomllib
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
ev = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")[did.rsplit(":", 1)[0]]["trajectory"]
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16).eval()
model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:max(D)])
C = Capturer(model, D)
R = render_pool(rec, ev, state_builder.build_state(ev, int(did.rsplit(":", 1)[1]), 32000), tok, rev)
n = len(R["suffix_ids"])
cap, _ = C.packed(R)
full = torch.stack([cap[d][1] for d in D], 1)
q0 = torch.stack([cap[d][0] for d in D])
rows = []
targets = [0, 1, 4, 6, 44, 11, n - 1]            # the worst candidates of the shape-change diagnostic, plus NONE (last)
for i in targets:
    pool = [x for j, x in enumerate(R["suffix_ids"]) if j != i]
    flat = [t for s in pool for t in s]
    random.Random(i).shuffle(flat)
    new, o = [], 0
    for j, s in enumerate(R["suffix_ids"]):
        if j == i:
            new.append(s)
        else:
            new.append(flat[o:o + len(s)])
            o += len(s)
    assert [len(a) for a in new] == [len(b) for b in R["suffix_ids"]] and sum(map(len, new)) == sum(map(len, R["suffix_ids"]))
    changed = sum(a != b for a, b in zip(new, R["suffix_ids"])) - 0
    c2, _ = C.packed(dict(prefix_ids=R["prefix_ids"], query_pos=R["query_pos"], suffix_ids=new))
    v = torch.stack([c2[d][1][i] for d in D])
    q = torch.stack([c2[d][0] for d in D])
    rows.append(dict(candidate=i, tokens=len(R["suffix_ids"][i]), other_candidates_changed=changed, max_abs_diff_candidate=(v - full[i]).abs().max().item(),
                     min_cos_candidate=F.cosine_similarity(v.double(), full[i].double(), dim=-1).min().item(), max_abs_diff_query=(q - q0).abs().max().item()))
    print(rows[-1], flush=True)
out = dict(decision_id=did, n_candidates=n, rows=rows, all_identical=all(r["max_abs_diff_candidate"] == 0.0 and r["max_abs_diff_query"] == 0.0 for r in rows))
p = ROOT / ("results/raw/indep-content-swap-diag-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("all identical:", out["all_identical"], "->", p)
