"""Smoke-test cross-check: candidate vectors from the new full-pool capture (capture-record-v2) vs the already validated pilot capture (capture-record-v1) for the same decisions.
Registered in configs/a5_v3_spec.toml [capture_v3] pilot_cross_check: cosine >= 0.9999 at depths 18, 24, 30 for every candidate present in both. Output create-only."""
import json, glob
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
pilot = {}
for f in glob.glob(str(ROOT / "data/pilot/capture/*.pt")):
    d = torch.load(f, map_location="cpu")
    pilot[d["decision_id"]] = d
recs = {}
for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["decision_id"] in pilot:
        recs[r["decision_id"]] = r
res = []
for f in sorted(glob.glob(str(ROOT / "data/v3/capture_smoke/cal/*.pt"))):
    new = torch.load(f, map_location="cpu")
    did = new["decision_id"]
    if did not in pilot:
        continue
    pd, rec = pilot[did], recs[did]
    texts = [c["text"] for c in rec["candidates"]]
    depth_map = {18: pd["depths"].index(18), 24: pd["depths"].index(24), 30: pd["depths"].index(30)}
    worst, matched = 1.0, 0
    for i, x in enumerate(texts):
        if x not in new["texts"]:
            continue
        j = new["texts"].index(x)
        matched += 1
        for dj, d in enumerate(new["depths"]):
            a = pd["options"][depth_map[d], i].double()
            b = new["cands"][j, dj].double()
            worst = min(worst, F.cosine_similarity(a, b, dim=0).item())
    qworst = min(F.cosine_similarity(pd["query"][depth_map[d]].double(), new["query"][dj].double(), dim=0).item() for dj, d in enumerate(new["depths"]))
    res.append(dict(decision_id=did, pilot_candidates=len(texts), matched=matched, worst_candidate_cosine=round(worst, 6), worst_query_cosine=round(qworst, 6)))
out = dict(rows=res, threshold=0.9999, passed=bool(res) and all(r["worst_candidate_cosine"] >= 0.9999 and r["worst_query_cosine"] >= 0.9999 and r["matched"] == r["pilot_candidates"] for r in res))
p = ROOT / ("results/raw/v3-pilot-crosscheck-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)); print("->", p)
