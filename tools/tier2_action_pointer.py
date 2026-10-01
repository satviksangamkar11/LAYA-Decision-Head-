"""Tier-2 pilot: pointer head on ACTION. Rule: configs/thresholds.toml [tier2_action_pointer] v1 (registered before this ran).
Trains on the 600 pilot-train rows, selects the cell on the 380 pilot-cal rows. The locked benchmark is never loaded. Output create-only.
    PYTHONPATH=. .venv/Scripts/python.exe tools/tier2_action_pointer.py
"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import collections, hashlib, json, statistics, time, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from compiler.baselines import TfIdf
from decision_head import pilot_data
from decision_head.closed_form import GRID
from decision_head.pointer import PointerHead

ROOT = Path(__file__).resolve().parents[1]
th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["tier2_action_pointer"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
out = ROOT / f"results/raw/tier2-action-pointer-{stamp}.json"
NEG = float("-inf")


def nll_of(lg, y):
    return F.cross_entropy(lg, y).item()


def ece_of(lg, y, bins=15):
    p = lg.softmax(1)
    conf, pred = p.max(1)
    acc = (pred == y).double()
    e, edges = 0.0, torch.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += m.double().mean().item() * abs(acc[m].mean().item() - conf[m].double().mean().item())
    return e


def ens_mean(lgs, mask):
    return torch.stack([x.nan_to_num(neginf=0.0) for x in lgs]).mean(0).masked_fill(~mask, NEG)


@torch.no_grad()
def predict(head, D_, sel):
    head.eval()
    return head(D_["hq"][sel], D_["ho"][sel], D_["mask"][sel], D_["meta"][sel]).cpu()


def train_run(D_, tr, va, dim, seed, fixed_epochs=None):
    torch.manual_seed(seed)
    H = D_["hq"].shape[1]
    head = PointerHead(H, dim, 12, th["dropout"]).to(dev)
    head.fit_scalers(D_["hq"][tr], D_["ho"][tr], D_["mask"][tr])
    opt = torch.optim.AdamW(head.parameters(), lr=th["lr"], weight_decay=th["weight_decay"])
    ntr, best, best_ep, best_state = len(tr), 1e9, 0, None
    n_ep = fixed_epochs or th["max_epochs"]
    for ep in range(1, n_ep + 1):
        head.train()
        perm = torch.randperm(ntr, generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(dev)
        for i in range(0, ntr, th["batch"]):
            b = tr[perm[i:i + th["batch"]]]
            lg = head(D_["hq"][b], D_["ho"][b], D_["mask"][b], D_["meta"][b])
            loss = F.cross_entropy(lg, D_["y"][b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), th["clip"])
            opt.step()
        if va is not None and fixed_epochs is None:
            v = nll_of(predict(head, D_, va), D_["y"][va].cpu())
            if v < best:
                best, best_ep = v, ep
                best_state = {k: x.detach().clone() for k, x in head.state_dict().items()}
            if ep >= th["min_epochs"] and ep - best_ep >= th["patience"]:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, best_ep


t0 = time.time()
cells, saved = [], {}
data = {}
for depth in th["depths"]:
    d = pilot_data.load(depth)
    for k in ("hq", "ho", "mask", "meta", "y"):
        d[k] = d[k].to(dev)
    data[depth] = d
base = data[th["depths"][0]]
tr = torch.tensor([i for i, s in enumerate(base["split"]) if s == "train"], device=dev)
ca = torch.tensor([i for i, s in enumerate(base["split"]) if s == "cal"], device=dev)
ycal = base["y"][ca].cpu()
cov_cal = torch.tensor([base["covered"][i] for i in ca.tolist()])
traj = [x.rsplit(":", 1)[0] for x in base["decision_id"]]
cal_traj = [traj[i] for i in ca.tolist()]

for depth in th["depths"]:
    for dim in th["proj_dims"]:
        lgs, eps, states = [], [], []
        for seed in th["seeds"]:
            head, be = train_run(data[depth], tr, ca, dim, seed)
            lgs.append(predict(head, data[depth], ca))
            eps.append(be)
            states.append({k: x.cpu() for k, x in head.state_dict().items()})
        e = ens_mean(lgs, data[depth]["mask"][ca].cpu())
        pred = e.argmax(1)
        cell = dict(depth=depth, D=dim, ens_nll=round(nll_of(e, ycal), 4), ens_acc=round((pred == ycal).double().mean().item(), 4),
                    ens_acc_covered=round((pred == ycal)[cov_cal].double().mean().item(), 4), ens_ece=round(ece_of(e, ycal), 4),
                    seed_best_epochs=eps, single_seed_nll=[round(nll_of(x, ycal), 4) for x in lgs])
        cells.append(cell)
        saved[(depth, dim)] = (states, eps, e)
        print(cell, f"{time.time() - t0:.0f}s", flush=True)

mn = min(c["ens_nll"] for c in cells)
tied = [c for c in cells if c["ens_nll"] <= mn + th["nll_tie"]]
sel = min(tied, key=lambda c: (c["D"], c["depth"]))
depth, dim = sel["depth"], sel["D"]
states, eps, e_cal = saved[(depth, dim)]
fixed = round(statistics.median(eps))

# temperature from out-of-fold TRAIN predictions (folds by trajectory)
trl = tr.tolist()
tj = sorted({traj[i] for i in trl})
g = torch.Generator().manual_seed(0)
order = torch.randperm(len(tj), generator=g).tolist()
fold_of = {tj[o]: n % th["folds"] for n, o in enumerate(order)}
oof = [None] * len(trl)
oof_acc = []
Dd = data[depth]
for f in range(th["folds"]):
    hold = [j for j, i in enumerate(trl) if fold_of[traj[i]] == f]
    keep = [j for j in range(len(trl)) if j not in set(hold)]
    lgs = []
    for seed in th["seeds"]:
        head, _ = train_run(Dd, tr[keep], None, dim, seed, fixed_epochs=fixed)
        lgs.append(predict(head, Dd, tr[hold]))
    e = ens_mean(lgs, Dd["mask"][tr[hold]].cpu())
    for n, j in enumerate(hold):
        oof[j] = e[n]
oof = torch.stack(oof)
yoof = Dd["y"][tr].cpu()
T = GRID[torch.tensor([nll_of(oof / t, yoof) for t in GRID]).argmin()].item()
cal_T = e_cal / T
cal_pred = cal_T.argmax(1)
sel["fixed_epochs_for_oof"] = fixed
sel["temperature"] = round(T, 3)
sel["oof_train_nll_at_T"] = round(nll_of(oof / T, yoof), 4)
sel["oof_train_acc"] = round((oof.argmax(1) == yoof).double().mean().item(), 4)
sel["cal_nll_calibrated"] = round(nll_of(cal_T, ycal), 4)
sel["cal_ece_calibrated"] = round(ece_of(cal_T, ycal), 4)

# baselines on pilot-cal rows
recs = {}
for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    r = json.loads(l)
    recs[r["decision_id"]] = ([c["text"] for c in r["candidates"]], r["label_index"], r["split"])
full_tr = [(t, y) for t, y, s in recs.values() if s == "train"]
full_va = [(t, y) for t, y, s in recs.values() if s == "cal"]
tf_full = TfIdf(full_tr)
tf_full.fit(full_tr, full_va)
pil_tr = [(recs[base["decision_id"][i]][0], recs[base["decision_id"][i]][1]) for i in trl]
pil_va = [(recs[base["decision_id"][i]][0], recs[base["decision_id"][i]][1]) for i in ca.tolist()]
tf_pil = TfIdf(pil_tr)
tf_pil.fit(pil_tr, pil_va)
hits = {"head": (cal_pred == ycal).double(),
        "tfidf_full_train": torch.tensor([float(int(tf_full.scores(t).argmax()) == y) for t, y in pil_va], dtype=torch.double),
        "tfidf_pilot_train_matched": torch.tensor([float(int(tf_pil.scores(t).argmax()) == y) for t, y in pil_va], dtype=torch.double),
        "first": torch.tensor([float(y == 0) for t, y in pil_va], dtype=torch.double),
        "random": torch.tensor([1.0 / len(t) for t, y in pil_va], dtype=torch.double)}
tiers = [base["tier"][i] for i in ca.tolist()]
res_b = {}
for k, v in hits.items():
    res_b[k] = dict(acc_all=round(v.mean().item(), 4), acc_covered=round(v[cov_cal].mean().item(), 4),
                    by_tier={t: round(v[torch.tensor([x == t for x in tiers])].mean().item(), 4) for t in sorted(set(tiers))})
tj_cal = sorted(set(cal_traj))
idx = {t: [i for i, x in enumerate(cal_traj) if x == t] for t in tj_cal}


def boot(a, b, subset=None):
    gg = torch.Generator().manual_seed(th["bootstrap_seed"])
    ds = []
    for _ in range(th["bootstrap_resamples"]):
        pick = torch.randint(len(tj_cal), (len(tj_cal),), generator=gg).tolist()
        ii = [i for k in pick for i in idx[tj_cal[k]]]
        if subset is not None:
            ii = [i for i in ii if subset[i]]
        ds.append(a[ii].mean().item() - b[ii].mean().item())
    ds.sort()
    return [round(ds[int(0.025 * len(ds))], 4), round(ds[int(0.975 * len(ds)) - 1], 4)]


ci = {}
for other in ("tfidf_full_train", "tfidf_pilot_train_matched"):
    ci[f"head_minus_{other}"] = dict(all=dict(delta=round((hits["head"] - hits[other]).mean().item(), 4), ci95=boot(hits["head"], hits[other])),
                                     covered=dict(delta=round((hits["head"] - hits[other])[cov_cal].mean().item(), 4),
                                                  ci95=boot(hits["head"], hits[other], cov_cal.tolist())))
res = dict(config=th, selected=sel, grid=cells, baselines_on_pilot_cal=res_b, bootstrap=ci, n_cal=len(ycal), n_cal_covered=int(cov_cal.sum()),
           note="pilot-cal only; NOT an A5 measurement; A5 stays 0.323 until the locked benchmark is evaluated", seconds=round(time.time() - t0))
body = json.dumps(res, indent=1)
with open(out, "x", encoding="utf-8") as f:
    f.write(body)
Path(str(out)[:-5] + ".sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "  " + out.name + "\n")
mdir = ROOT / "data/pilot/tier2-v1"
mdir.mkdir(parents=True, exist_ok=True)
for s, st in zip(th["seeds"], states):
    torch.save({"state": st, "depth": depth, "D": dim, "seed": s, "temperature": T, "config": "tier2_action_pointer v1"}, mdir / f"head-d{depth}-D{dim}-seed{s}.pt")
print(json.dumps({k: v for k, v in res.items() if k not in ("config", "grid")}, indent=1))
print("->", out)
