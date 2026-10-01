"""Leakage / shortcut controls for the selected Tier-2 cell (depth 24, D 128). DIAGNOSTIC ONLY: nothing here selects or changes the head.
Trains on the 600 pilot-train rows, evaluates on the 380 pilot-cal rows, 3 seeds per control, same hyper-parameters as the registered run.
    A reference            : the registered head again (reproduction)
    B metadata-only        : hq and ho zeroed, so scores depend on the 12 metadata values only
    C state-shuffled eval  : reference models, but each cal row gets another cal row's query vector (options kept)
    D label-shuffled train : the true label replaced by a random valid index in every train row (must fall to chance)
    E query-zeroed         : hq zeroed in train and eval, scores come from option vectors + metadata
Output results/raw/tier2-controls-*.json (create-only).
"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from decision_head import pilot_data
from decision_head.pointer import PointerHead

ROOT = Path(__file__).resolve().parents[1]
th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["tier2_action_pointer"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH, DIM, SEEDS = 24, 128, [0, 1, 2]
d = pilot_data.load(DEPTH)
D0 = {k: d[k].to(dev) for k in ("hq", "ho", "mask", "meta", "y")}
tr = torch.tensor([i for i, s in enumerate(d["split"]) if s == "train"], device=dev)
ca = torch.tensor([i for i, s in enumerate(d["split"]) if s == "cal"], device=dev)
cov = torch.tensor([d["covered"][i] for i in ca.tolist()])


@torch.no_grad()
def pred(head, X, sel, hq=None):
    head.eval()
    return head(X["hq"][sel] if hq is None else hq, X["ho"][sel], X["mask"][sel], X["meta"][sel]).cpu()


def train(X, y_train, seed):
    torch.manual_seed(seed)
    head = PointerHead(X["hq"].shape[1], DIM, 12, th["dropout"]).to(dev)
    head.fit_scalers(X["hq"][tr], X["ho"][tr], X["mask"][tr])
    opt = torch.optim.AdamW(head.parameters(), lr=th["lr"], weight_decay=th["weight_decay"])
    best, best_ep, best_state, n = 1e9, 0, None, len(tr)
    for ep in range(1, th["max_epochs"] + 1):
        head.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(dev)
        for i in range(0, n, th["batch"]):
            b = perm[i:i + th["batch"]]
            lg = head(X["hq"][tr[b]], X["ho"][tr[b]], X["mask"][tr[b]], X["meta"][tr[b]])
            loss = F.cross_entropy(lg, y_train[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), th["clip"]); opt.step()
        v = F.cross_entropy(pred(head, X, ca), X["y"][ca].cpu()).item()
        if v < best:
            best, best_ep, best_state = v, ep, {k: x.clone() for k, x in head.state_dict().items()}
        if ep >= th["min_epochs"] and ep - best_ep >= th["patience"]:
            break
    head.load_state_dict(best_state)
    return head


def ens(lgs, mask):
    return torch.stack([x.nan_to_num(neginf=0.0) for x in lgs]).mean(0).masked_fill(~mask.cpu(), float("-inf"))


def score(e):
    y = D0["y"][ca].cpu()
    p = e.argmax(1)
    return dict(acc=round((p == y).double().mean().item(), 4), acc_covered=round((p == y)[cov].double().mean().item(), 4),
                nll=round(F.cross_entropy(e, y).item(), 4))


res = {}
ytr = D0["y"][tr]
heads = [train(D0, ytr, s) for s in SEEDS]
res["A_reference"] = score(ens([pred(h, D0, ca) for h in heads], D0["mask"][ca]))
g = torch.Generator().manual_seed(0)
shuf = torch.randperm(len(ca), generator=g).to(dev)
res["C_state_shuffled_eval"] = score(ens([pred(h, D0, ca, hq=D0["hq"][ca][shuf]) for h in heads], D0["mask"][ca]))

B = dict(D0, hq=torch.zeros_like(D0["hq"]), ho=torch.zeros_like(D0["ho"]))
res["B_metadata_only"] = score(ens([pred(train(B, ytr, s), B, ca) for s in SEEDS], B["mask"][ca]))
E = dict(D0, hq=torch.zeros_like(D0["hq"]))
res["E_query_zeroed"] = score(ens([pred(train(E, ytr, s), E, ca) for s in SEEDS], E["mask"][ca]))
n_opts = D0["mask"][tr].sum(1)
yfake = (torch.rand(len(tr), generator=torch.Generator().manual_seed(1)).to(dev) * n_opts).long()
res["D_label_shuffled_train"] = score(ens([pred(train(D0, yfake, s), D0, ca) for s in SEEDS], D0["mask"][ca]))
res["random_expected_acc"] = round((1.0 / D0["mask"][ca].sum(1).double()).mean().item(), 4)
out = ROOT / ("results/raw/tier2-controls-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(out, "x", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
print("->", out)
