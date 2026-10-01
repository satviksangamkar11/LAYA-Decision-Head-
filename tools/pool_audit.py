"""A5 candidate-pool shortcut audit (text level, no GPU). Rules: configs/thresholds.toml [a5_v2_rules] v1 and [benchmark_integrity_gate] v1.
Reads the compiled decisions (train + cal splits ONLY), rebuilds each decision's pool/history features from the raw trajectories and reports
(a) within-decision pairwise AUC of 'is the true action' per feature on TRAIN covered rows, (b) an adversarial listwise MLP over all features,
trained on train covered and scored on cal covered. Run on the candidates in the compiled file (A5-v1 sampler) unless --records points elsewhere.
    PYTHONPATH=. .venv/Scripts/python.exe tools/pool_audit.py [--tag A5v1]
Output results/raw/pool-audit-<tag>-*.json (create-only)."""
import argparse, collections, json, re, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch, torch.nn.functional as F
from compiler.candidates import distractor_pool, first_call, render

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="A5v1"); ap.add_argument("--records", default="data/decisions/action-swerebench-sample-v2.jsonl")
args = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
T = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))
th, gate = T["a5_v2_rules"], T["benchmark_integrity_gate"]
raw = {}
for l in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l); raw[r["trajectory_id"]] = r
KINDS = ["view", "edit", "run_tests", "explore", "run_other", "think", "plan", "finish"]
tok = lambda s: set(re.findall(r"[a-z0-9_]+", s.lower()))
rows = []
for l in open(ROOT / args.records, encoding="utf-8"):
    r = json.loads(l)
    if r["split"] not in ("train", "cal") or not r["candidate_coverage"]:
        continue
    tid, t = r["decision_id"].rsplit(":", 1); t = int(t); ev = raw[tid]["trajectory"]
    pool = [p[0] for p in distractor_pool(ev, t)]
    pk = {p[0]: p[1] for p in distractor_pool(ev, t)}
    hist = collections.Counter(render(*c)[0] for e in ev[1:t] if e["role"] == "assistant" and (c := first_call(e)))
    tail = tok(" ".join((e["content"] or "") for e in ev[max(1, t - 4):t]))
    feats, ids = [], []
    for i, c in enumerate(r["candidates"]):
        if c["id"] == "NONE":
            continue
        x = c["text"]
        rank = pool.index(x) if x in pool else 99
        tk = tok(x)
        kind = pk.get(x) or ("edit" if c["action_type"] == "edit" else "finish" if c["action_type"] == "finish" else "view")
        feats.append([min(rank, 40) / 40.0, min(hist.get(x, 0), 6) / 6.0, KINDS.index(kind) / 7.0 if kind in KINDS else 0.5, float(x in pool),
                      min(len(x), 200) / 200.0, (len(tk & tail) / len(tk)) if tk else 0.0, float(c["safety_class"] == "EXEC"), float(c["safety_class"] == "WRITE_LOCAL")])
        ids.append(i)
    if len(ids) >= 2 and r["label_index"] in ids:
        rows.append(dict(split=r["split"], tid=tid, f=torch.tensor(feats), y=ids.index(r["label_index"])))
tr = [r for r in rows if r["split"] == "train"]; ca = [r for r in rows if r["split"] == "cal"]
names = ["pool_rank", "history_frequency", "action_kind", "in_pool_novelty", "text_length", "lexical_overlap_state_tail", "safety_exec", "safety_write"]
auc = {}
for j, n in enumerate(names):
    s = c = 0.0
    for r in tr:
        v = r["f"][:, j]; yv = v[r["y"]]; o = torch.cat([v[:r["y"]], v[r["y"] + 1:]])
        s += ((yv > o).double().sum() + 0.5 * (yv == o).double().sum()).item(); c += len(o)
    auc[n] = round(s / c, 4)
def pad(rs):
    K = max(len(r["f"]) for r in rs); X = torch.zeros(len(rs), K, len(names)); m = torch.zeros(len(rs), K, dtype=torch.bool)
    for i, r in enumerate(rs): X[i, :len(r["f"])] = r["f"]; m[i, :len(r["f"])] = True
    return X, m, torch.tensor([r["y"] for r in rs])
Xt, mt, yt = pad(tr); Xc, mc, yc = pad(ca)
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(len(names), 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)); opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
for _ in range(400):
    s = net(Xt).squeeze(-1).masked_fill(~mt, float("-inf")); loss = F.cross_entropy(s, yt); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pc = net(Xc).squeeze(-1).masked_fill(~mc, float("-inf")).argmax(1)
    ptr = net(Xt).squeeze(-1).masked_fill(~mt, float("-inf")).argmax(1)
rand_c = (1.0 / mc.sum(1).double()).mean().item()
res = dict(tag=args.tag, train_covered_rows=len(tr), cal_covered_rows=len(ca), within_decision_auc=auc, auc_tolerance=th["auc_tolerance"],
           auc_violations=[n for n, a in auc.items() if abs(a - 0.5) > th["auc_tolerance"]],
           adversarial=dict(train_acc=round((ptr == yt).double().mean().item(), 4), cal_covered_acc=round((pc == yc).double().mean().item(), 4),
                            random_expected_cal=round(rand_c, 4), limit=round(rand_c + gate["max_excess_over_random"], 4)))
res["verdict"] = "PASS" if not res["auc_violations"] and res["adversarial"]["cal_covered_acc"] <= res["adversarial"]["limit"] else "FAIL"
p = ROOT / ("results/raw/pool-audit-%s-%s.json" % (args.tag, datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")))
with open(p, "x", encoding="utf-8") as f: json.dump(res, f, indent=1)
print(json.dumps(res, indent=1)); print("->", p)
