"""Tier-1 closed-form head on next_action_kind, every depth. Rule: configs/thresholds.toml [tier1_next_action_kind] v1. Output create-only.
    PYTHONPATH=. .venv/Scripts/python.exe tools/tier1_next_action_kind.py
"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import hashlib, json, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from decision_head import closed_form, pilot_data

ROOT = Path(__file__).resolve().parents[1]
th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["tier1_next_action_kind"]
DEPTHS = [12, 18, 24, 30, 36]
out = ROOT / ("results/raw/tier1-next-action-kind-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))


def ece(p, y, bins=15):
    conf, pred = p.max(1)
    acc = (pred == y).double()
    e = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.double().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return e


def bal_acc(pred, y, K):
    r = [(pred[y == k] == k).double().mean().item() for k in range(K) if (y == k).any()]
    return sum(r) / len(r)


data = {d: pilot_data.load(d) for d in DEPTHS}
base = data[DEPTHS[0]]
classes = sorted(set(base["kind"]))
K = len(classes)
yk = torch.tensor([classes.index(k) for k in base["kind"]])
tr = torch.tensor([s == "train" for s in base["split"]])
ca = ~tr
traj = [d.rsplit(":", 1)[0] for d in base["decision_id"]]
ytr, yca = yk[tr], yk[ca]
cal_traj = [t for t, c in zip(traj, ca.tolist()) if c]

prior = torch.bincount(ytr, minlength=K).double()
prior = (prior + 1e-6) / (prior + 1e-6).sum()
prior_pred = torch.full_like(yca, prior.argmax().item())
prior_acc = (prior_pred == yca).double()
prior_nll = F.nll_loss(prior.log().expand(len(yca), K), yca).item()

res = dict(classes=classes, n_train=int(tr.sum()), n_cal=int(ca.sum()), prior=dict(acc=prior_acc.mean().item(), nll=prior_nll, majority=classes[prior.argmax()]), depths={})
per_depth_correct = {}
for d in DEPTHS:
    X = data[d]["hq"]
    h = closed_form.fit(X[tr], ytr, K)
    p_tr = closed_form.predict(h, X[tr])
    p_ca = closed_form.predict(h, X[ca])
    pred = p_ca.argmax(1)
    per_depth_correct[d] = (pred == yca).double()
    res["depths"][str(d)] = dict(kind=h["kind"], param=h["param"], T=round(h["T"], 3),
                                 train_oof_acc=round(h["cv_acc"], 4), train_oof_nll=round(h["cv_nll"], 4),
                                 train_fit_acc=round((p_tr.argmax(1) == ytr).double().mean().item(), 4),
                                 cal_acc=round(per_depth_correct[d].mean().item(), 4), cal_nll=round(F.nll_loss(p_ca.clamp_min(1e-12).log(), yca).item(), 4),
                                 cal_ece=round(ece(p_ca, yca), 4), cal_balanced_acc=round(bal_acc(pred, yca, K), 4))
    print(d, res["depths"][str(d)], flush=True)

best = None
for d in DEPTHS:  # ascending = shallower first; a deeper depth must beat by more than the tie margin
    n = res["depths"][str(d)]["cal_nll"]
    if best is None or n < res["depths"][str(best)]["cal_nll"] - th["depth_nll_tie"]:
        best = d
res["selected_depth"] = best

g = torch.Generator().manual_seed(th["bootstrap_seed"])
tj = sorted(set(cal_traj))
idx = {t: [i for i, x in enumerate(cal_traj) if x == t] for t in tj}
diffs = []
head_c, prior_c = per_depth_correct[best], prior_acc
for _ in range(th["bootstrap_resamples"]):
    pick = torch.randint(len(tj), (len(tj),), generator=g).tolist()
    ii = [i for k in pick for i in idx[tj[k]]]
    diffs.append(head_c[ii].mean().item() - prior_c[ii].mean().item())
diffs.sort()
lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
res["sanity"] = dict(delta_acc_vs_prior=round(head_c.mean().item() - prior_c.mean().item(), 4), ci95=[round(lo, 4), round(hi, 4)],
                     clusters=len(tj), passes=lo > 0)
body = json.dumps(res, indent=1)
with open(out, "x", encoding="utf-8") as f:
    f.write(body)
Path(str(out)[:-5] + ".sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "  " + out.name + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "depths"}, indent=1))
print("->", out)
