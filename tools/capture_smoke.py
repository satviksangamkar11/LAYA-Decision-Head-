"""Step-2 gates on ONE real decision. Thresholds: configs/thresholds.toml [capture_one_record] v1. Output create-only."""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, random, time, tomllib
from datetime import datetime, timezone
import torch, torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture.capture import Capturer
from capture.render import render
from compiler import state_builder

th = tomllib.load(open("configs/thresholds.toml", "rb"))["capture_one_record"]
P = r"D:\local model\models\qwen3-4b-thinking-2507"
rev = json.load(open("results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
out = "results/raw/capture-one-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
assert not os.path.exists(out)
tok = AutoTokenizer.from_pretrained(P)
raw = {}
for l in open("data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l)
    raw[r["trajectory_id"]] = r
recs = []
for l in open("data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    recs.append(json.loads(l))
    if len(recs) >= 400:
        break
rec = max((r for r in recs if r["cut_stage"] == "none"), key=lambda r: r["t"])
st = state_builder.build_state(raw[rec["provenance"]["trajectory_id"]]["trajectory"], rec["t"], 32000)
R = render(rec, st, tok, rev)
perm = list(range(len(R["order"])))
random.Random(0).shuffle(perm)
R2 = render(rec, st, tok, rev, order=perm)

model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16).eval()
D = th["depths"]
C = Capturer(model, D)
t0 = time.time()
c1, (ids, spans) = C.packed(R)
t1 = time.time() - t0
c2, _ = C.packed(R)
res = dict(decision_id=R["decision_id"], prefix_tokens=R["n_prefix_tokens"], options=len(R["order"]), packed_seconds=round(t1, 1))

Pn = len(R["prefix_ids"])
res["span_ok"] = all(ids[0, a:b].tolist() == R["suffix_ids"][i] and a >= Pn for i, (a, b) in enumerate(spans))
mad = max(max((c1[d][0] - c2[d][0]).abs().max().item(), (c1[d][1] - c2[d][1]).abs().max().item()) for d in D)
res["repeat_max_abs"] = mad
cos_q, cos_o = {d: 1.0 for d in D}, {d: 1.0 for d in D}
for i in range(len(R["order"])):
    p = C.plain(R, i)
    for d in D:
        cos_q[d] = min(cos_q[d], Fn.cosine_similarity(c1[d][0], p[d][0], dim=0).item())
        cos_o[d] = min(cos_o[d], Fn.cosine_similarity(c1[d][1][i], p[d][1][0], dim=0).item())
res["plain_min_cos_query"], res["plain_min_cos_option"] = cos_q, cos_o
c3, _ = C.packed(R2)
pos = {cid: j for j, cid in enumerate(R["order"])}
ind = {d: min(Fn.cosine_similarity(c1[d][1][pos[cid]], c3[d][1][j], dim=0).item() for j, cid in enumerate(R2["order"])) for d in D}
res["indep_min_cos_option"] = ind
res["indep_query_cos"] = {d: Fn.cosine_similarity(c1[d][0], c3[d][0], dim=0).item() for d in D}
res["lm_head_calls"] = C.lm_calls
res["gates"] = dict(repeat=mad == 0.0, plain=min(list(cos_q.values()) + list(cos_o.values())) >= th["plain_cos_min"],
                    span=res["span_ok"], indep=min(ind.values()) >= th["indep_cos_min"], no_lm_head=C.lm_calls == 0)
res["verdict"] = "PASS" if all(res["gates"].values()) else "FAIL"
with open(out, "x") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
print("->", out)
