"""Control F (DIAGNOSTIC ONLY): an option-only linear scorer that never sees the query vector.
score_i = w . standardise(ho_i) + meta_bias(meta_i). Same data, optimiser, early stopping and seeds as the registered Tier-2 run (depth 24).
If it matches the pointer head, the query-option interaction is not what carries the signal. Output create-only."""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from decision_head import pilot_data

ROOT = Path(__file__).resolve().parents[1]
th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["tier2_action_pointer"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = pilot_data.load(24)
X = {k: d[k].to(dev) for k in ("hq", "ho", "mask", "meta", "y")}
tr = torch.tensor([i for i, s in enumerate(d["split"]) if s == "train"], device=dev)
ca = torch.tensor([i for i, s in enumerate(d["split"]) if s == "cal"], device=dev)
cov = torch.tensor([d["covered"][i] for i in ca.tolist()])


class OptionOnly(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.w, self.meta, self.drop = nn.Linear(H, 1, bias=False), nn.Linear(12, 1), nn.Dropout(th["dropout"])
        nn.init.zeros_(self.meta.weight), nn.init.zeros_(self.meta.bias)
        self.register_buffer("mu", torch.zeros(H)), self.register_buffer("sd", torch.ones(H))

    def forward(self, ho, mask, meta):
        s = self.w(self.drop((ho - self.mu) / self.sd)).squeeze(-1) + self.meta(meta).squeeze(-1)
        return s.masked_fill(~mask, float("-inf"))


@torch.no_grad()
def pred(m, sel):
    m.eval()
    return m(X["ho"][sel], X["mask"][sel], X["meta"][sel]).cpu()


def run(seed):
    torch.manual_seed(seed)
    m = OptionOnly(X["ho"].shape[2]).to(dev)
    o = X["ho"][tr][X["mask"][tr]]
    m.mu.copy_(o.mean(0)), m.sd.copy_(o.std(0) + 1e-6)
    opt = torch.optim.AdamW(m.parameters(), lr=th["lr"], weight_decay=th["weight_decay"])
    best, best_ep, st, n = 1e9, 0, None, len(tr)
    for ep in range(1, th["max_epochs"] + 1):
        m.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + ep)).to(dev)
        for i in range(0, n, th["batch"]):
            b = tr[perm[i:i + th["batch"]]]
            loss = F.cross_entropy(m(X["ho"][b], X["mask"][b], X["meta"][b]), X["y"][b])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), th["clip"]); opt.step()
        v = F.cross_entropy(pred(m, ca), X["y"][ca].cpu()).item()
        if v < best:
            best, best_ep, st = v, ep, {k: x.clone() for k, x in m.state_dict().items()}
        if ep >= th["min_epochs"] and ep - best_ep >= th["patience"]:
            break
    m.load_state_dict(st)
    return pred(m, ca), best_ep


outs = [run(s) for s in [0, 1, 2, 3, 4]]
e = torch.stack([o[0].nan_to_num(neginf=0.0) for o in outs]).mean(0).masked_fill(~X["mask"][ca].cpu(), float("-inf"))
y = X["y"][ca].cpu()
p = e.argmax(1)
res = dict(control="F_option_only_linear", seeds=5, best_epochs=[o[1] for o in outs], acc=round((p == y).double().mean().item(), 4),
           acc_covered=round((p == y)[cov].double().mean().item(), 4), nll=round(F.cross_entropy(e, y).item(), 4),
           reference_pointer_head=dict(acc=0.6158, acc_covered=0.6667, nll=1.2536))
out = ROOT / ("results/raw/tier2-control-option-only-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(out, "x", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1)); print("->", out)
