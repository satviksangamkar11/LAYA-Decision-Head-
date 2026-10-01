"""A5-v2 sampler + TRAIN-only grouped-CV audit. Spec: configs/a5_v2_spec.toml (registered before this file existed); thresholds: configs/thresholds.toml.
Data: compiler.firewall.load_dev (TRAIN rows only; test/ood/cal cannot be read here). No GPU. Output create-only.
    PYTHONPATH=. .venv/Scripts/python.exe tools/a5_v2_sampler.py
Pipeline: pool entry features -> per-fold per-stratum logistic 'true vs not-true' odds model fitted on the other folds -> Gumbel-top-k distractors with
weight odds^alpha for the held-out fold's decisions (out of fold) -> audit with two DIFFERENT attackers under trajectory-grouped CV, next to the A5-v1 sets."""
import collections, hashlib, json, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from compiler import a5v2
from compiler.a5v2 import DENSE, ND, NH, auc_by_feature, tensors
from compiler.firewall import load_dev

ROOT = Path(__file__).resolve().parents[1]
T = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))
S = a5v2.S
gate, rules = T["benchmark_integrity_gate"], T["a5_v2_rules"]

# ---------- the sampler itself lives ONLY in compiler/a5v2.py; this script just calls it ----------
raw = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
decs, excl = [], collections.Counter()
for r in load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"):
    if not r["candidate_coverage"]:
        excl["uncovered (diagnostic only)"] += 1
        continue
    d, why = a5v2.build_decision(r, raw[r["decision_id"].rsplit(":", 1)[0]]["trajectory"])
    if d is None:
        excl[why] += 1
        continue
    decs.append(d)
res = dict(spec=S, covered_train_decisions=len(decs), exclusions=dict(excl), strata=dict(collections.Counter(d["stratum"] for d in decs)))
v2 = a5v2.oof_sets(decs)
v1 = {d["did"]: a5v2.v1_set(d) for d in decs}
decs_v1 = [d for d in decs if v1[d["did"]]["true_pos"] is not None]


# ---------- audit ----------
def attacker(ds, sets, which, seed_off=0):
    X, m, y = tensors(ds, sets, which)
    tid_f = {t: n % S["cv_folds"] for n, t in enumerate(sorted({d["tid"] for d in ds}))}
    g2 = torch.Generator().manual_seed(S["attacker_seed"])
    order = torch.randperm(len(tid_f), generator=g2).tolist()
    keys = sorted(tid_f)
    fa = {keys[i]: n % S["cv_folds"] for n, i in enumerate(order)}
    fold = torch.tensor([fa[d["tid"]] for d in ds])
    hit = torch.zeros(len(ds))
    hidden = S["attacker_hidden_A"] if which == "A" else S["attacker_hidden_B"]
    for f in range(S["cv_folds"]):
        tr, te = fold != f, fold == f
        torch.manual_seed(S["attacker_seed"] + f)
        net = (torch.nn.Sequential(torch.nn.Linear(X.shape[2], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)) if which == "A" else
               torch.nn.Sequential(torch.nn.Linear(X.shape[2], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(S["attacker_steps"]):
            s = net(X[tr]).squeeze(-1).masked_fill(~m[tr], float("-inf"))
            loss = F.cross_entropy(s, y[tr]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            hit[te] = (net(X[te]).squeeze(-1).masked_fill(~m[te], float("-inf")).argmax(1) == y[te]).float()
    rand = (1.0 / m.sum(1).float())
    out = {"overall": dict(acc=round(hit.mean().item(), 4), random=round(rand.mean().item(), 4), n=len(ds))}
    for st in ("repeat", "first_time"):
        sel = torch.tensor([d["stratum"] == st for d in ds])
        out[st] = dict(acc=round(hit[sel].mean().item(), 4), random=round(rand[sel].mean().item(), 4), n=int(sel.sum()))
    return out


def report(ds, sets, label):
    r = dict(set_size_mean=round(sum(len(sets[d["did"]]["order"]) for d in ds) / len(ds), 2),
             true_position_uniform_check=dict(collections.Counter(sets[d["did"]]["true_pos"] for d in ds).most_common(3)))
    r["attacker_A_8_features"] = attacker(ds, sets, "A")
    r["attacker_B_all_features_plus_hashed_tokens"] = attacker(ds, sets, "B")
    r["auc"] = {"overall": auc_by_feature(ds, sets), "repeat": auc_by_feature(ds, sets, "repeat"), "first_time": auc_by_feature(ds, sets, "first_time")}
    r["auc_violations_overall"] = [n for n, a in r["auc"]["overall"].items() if a is not None and abs(a - 0.5) > rules["auc_tolerance"] and not n.startswith("kind_")]
    ok = True
    for key in ("attacker_A_8_features", "attacker_B_all_features_plus_hashed_tokens"):
        for grp, v in r[key].items():
            ok &= v["acc"] <= v["random"] + gate["max_excess_over_random"]
    r["verdict"] = "PASS" if ok and not r["auc_violations_overall"] else "FAIL"
    r["label"] = label
    return r


res["retention"] = dict(considered_covered=len(decs), usable_v2=len(v2), retention=round(len(v2) / max(1, len(decs)), 4))
res["A5_v1_sets_same_audit"] = report(decs_v1, {d["did"]: v1[d["did"]] for d in decs_v1}, "A5-v1 candidates, same attackers, grouped CV")
res["A5_v2_sets"] = report(decs, v2, "A5-v2 candidates (out-of-fold generation)")
res["reproducibility_example"] = {k: v2[k] for k in list(v2)[:2]}
body = json.dumps(res, indent=1)
p = ROOT / ("results/raw/a5v2-train-cv-audit-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    f.write(body)
Path(str(p)[:-5] + ".sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "  " + p.name + "\n")
for k in ("A5_v1_sets_same_audit", "A5_v2_sets"):
    r = res[k]
    print(k, r["verdict"], "| A:", {g: (v["acc"], v["random"]) for g, v in r["attacker_A_8_features"].items()}, "| B:",
          {g: (v["acc"], v["random"]) for g, v in r["attacker_B_all_features_plus_hashed_tokens"].items()}, "| AUC violations:", r["auc_violations_overall"])
print(res["retention"], res["strata"], "->", p)
