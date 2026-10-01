"""Benchmark integrity gate (text-level baselines), rule: configs/thresholds.toml [benchmark_integrity_gate] v1.
Runs every non-head baseline on the PILOT train+cal rows (never test/OOD) and reports the strongest on cal covered rows against random + margin.
    PYTHONPATH=. .venv/Scripts/python.exe tools/benchmark_integrity_gate.py
Output results/raw/integrity-gate-A5v1-*.json (create-only)."""
import collections, json, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch, torch.nn.functional as F
from compiler.baselines import TfIdf
from compiler.candidates import distractor_pool, first_call, render
from state.schema import Candidate

ROOT = Path(__file__).resolve().parents[1]
th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["benchmark_integrity_gate"]
man = json.load(open(ROOT / "results/raw/pilot-manifest-v1.json", encoding="utf-8"))
want = set(man["decision_ids"])
recs = {}
for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["decision_id"] in want:
        recs[r["decision_id"]] = r
raw = {}
for l in open(ROOT / "data/raw/swerebench-sample-20261001-sample.jsonl" if False else ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l)
    raw[r["trajectory_id"]] = r
rows = []
for did, r in recs.items():
    tid, t = did.rsplit(":", 1)
    ev = raw[tid]["trajectory"]
    pool = [p[0] for p in distractor_pool(ev, int(t))]
    hist = collections.Counter(render(*c)[0] for e in ev[1:int(t)] if e["role"] == "assistant" and (c := first_call(e)))
    texts = [c["text"] for c in r["candidates"]]
    meta = torch.tensor([Candidate(c["id"], c["text"], action_type=c["action_type"], tool=c["tool"], safety_class=c["safety_class"],
                                   executable=c["id"] != "NONE").head_meta() for c in r["candidates"]])
    real = [i for i, c in enumerate(r["candidates"]) if c["id"] != "NONE"]
    rows.append(dict(did=did, split=r["split"], cov=r["candidate_coverage"], texts=texts, y=r["label_index"], meta=meta, real=real,
                     rank=[pool.index(x) if x in pool else 10 ** 6 for x in texts], freq=[hist.get(x, 0) for x in texts]))
tr = [r for r in rows if r["split"] == "train"]
ca = [r for r in rows if r["split"] == "cal"]
cov = [r for r in ca if r["cov"]]


def acc(rs, pick):
    return sum(int(pick(r) == r["y"]) for r in rs) / len(rs)


pol = {"first": lambda r: 0,
       "most_recent_in_pool": lambda r: min(r["real"], key=lambda i: (r["rank"][i], i)),
       "most_frequent_in_history": lambda r: max(r["real"], key=lambda i: (r["freq"][i], -r["rank"][i], -i)),
       "frequent_then_recent": lambda r: max(r["real"], key=lambda i: (r["freq"][i] > 0, -r["rank"][i], r["freq"][i], -i))}
# metadata-only listwise linear
K = max(len(r["texts"]) for r in rows)
def pack(rs):
    M = torch.zeros(len(rs), K, 12); m = torch.zeros(len(rs), K, dtype=torch.bool)
    for i, r in enumerate(rs):
        n = len(r["texts"]); M[i, :n] = r["meta"]; m[i, :n] = True
    return M, m, torch.tensor([r["y"] for r in rs])
Mt, mt, yt = pack(tr)
lin = torch.nn.Linear(12, 1); opt = torch.optim.Adam(lin.parameters(), lr=0.05)
for _ in range(300):
    s = lin(Mt).squeeze(-1).masked_fill(~mt, float("-inf")); loss = F.cross_entropy(s, yt) + 1e-3 * lin.weight.pow(2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
Mc, mc, yc = pack(ca)
meta_pred = lin(Mc).squeeze(-1).masked_fill(~mc, float("-inf")).argmax(1)
tf_tr = [(r["texts"], r["y"]) for r in tr]; tf_ca = [(r["texts"], r["y"]) for r in ca]
tf = TfIdf(tf_tr); tf.fit(tf_tr, tf_ca)
res = {"cal_covered_n": len(cov), "cal_n": len(ca)}
for name, rs in (("cal_covered", cov), ("cal_all", ca)):
    d = {k: round(acc(rs, p), 4) for k, p in pol.items()}
    idx = {r["did"]: i for i, r in enumerate(ca)}
    d["metadata_only"] = round(sum(int(meta_pred[idx[r["did"]]].item() == r["y"]) for r in rs) / len(rs), 4)
    d["tfidf_candidate_text"] = round(sum(int(int(tf.scores(r["texts"]).argmax()) == r["y"]) for r in rs) / len(rs), 4)
    d["random_expected"] = round(sum(1.0 / len(r["texts"]) for r in rs) / len(rs), 4)
    res[name] = d
strong = max((v, k) for k, v in res["cal_covered"].items() if k != "random_expected")
res["strongest_non_head_on_cal_covered"] = dict(name=strong[1], acc=strong[0])
res["limit"] = round(res["cal_covered"]["random_expected"] + th["max_excess_over_random"], 4)
res["verdict_A5_v1"] = "PASS" if strong[0] <= res["limit"] else "FAIL (diagnostic only)"
p = ROOT / ("results/raw/integrity-gate-A5v1-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1)); print("->", p)
