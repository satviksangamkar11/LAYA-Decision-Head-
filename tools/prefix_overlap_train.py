"""How much of a decision's prompt prefix is shared with the previous decision of the same trajectory (token-identical leading prefix). TRAIN decisions only;
CAL and LOCKED are not touched. Tokenizer only, no model. Output is create-only: results/raw/prefix-overlap-train-<time>.json.
    PYTHONPATH=. .venv/Scripts/python.exe tools/prefix_overlap_train.py"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import collections, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from capture.pool import render_pool
from compiler import a5v2, state_builder
from compiler.firewall import load_dev

ROOT = Path(__file__).resolve().parents[1]
rev = json.load(open(ROOT / "results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
man = json.load(open(ROOT / "data/v3/manifest-train.json", encoding="utf-8"))["decision_ids"]
recs = {r["decision_id"]: r for r in load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl") if r["decision_id"] in set(man)}
raw = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
tok = AutoTokenizer.from_pretrained(r"D:\local model\models\qwen3-4b-thinking-2507")
by = collections.defaultdict(list)
for d in man:
    tid, t = d.rsplit(":", 1)
    by[tid].append(int(t))
rows = []
for tid, ts in by.items():
    ts = sorted(ts)
    if len(ts) < 3:
        continue
    prev = None
    for t in ts[:6]:
        try:
            R = render_pool(recs[f"{tid}:{t}"], raw[tid]["trajectory"], state_builder.build_state(raw[tid]["trajectory"], t, 32000), tok, rev)
        except Exception:
            prev = None
            continue
        ids = list(R["prefix_ids"])
        if prev is not None:
            n = next((k for k, (a, b) in enumerate(zip(prev, ids)) if a != b), min(len(prev), len(ids)))
            rows.append(dict(trajectory=tid, t=t, prefix_tokens=len(ids), common_with_previous=n, previous_prefix_tokens=len(prev)))
        prev = ids
    if len(rows) >= 25:
        break
a = np.array([[r["prefix_tokens"], r["common_with_previous"]] for r in rows])
out = dict(created=datetime.now(timezone.utc).isoformat(), role="train only", pairs=len(rows), prefix_tokens_mean=float(a[:, 0].mean()), common_tokens_mean=float(a[:, 1].mean()),
           share_of_tokens=float(a[:, 1].sum() / a[:, 0].sum()), share_median=float(np.median(a[:, 1] / a[:, 0])), pairs_over_half=int((a[:, 1] / a[:, 0] > 0.5).sum()), rows=rows,
           note="first 25 consecutive pairs from TRAIN trajectories with at least 3 decisions, manifest order; a measurement of potential KV reuse, not a result about decision quality")
p = ROOT / "results/raw" / f"prefix-overlap-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
with open(p, "x") as f:
    json.dump(out, f, indent=1)
print(p, "\npairs", out["pairs"], "share %.1f%%" % (100 * out["share_of_tokens"]), "median %.1f%%" % (100 * out["share_median"]), "over half", out["pairs_over_half"])
