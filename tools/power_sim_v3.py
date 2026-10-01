"""Power simulation for the A5-v3 primary claim (spec: configs/a5_v3_spec.toml [primary_claim], [amendment_2026_10_02_b]). CPU only, no fresh data.
Paired design: each decision has outcome H2-correct vs H0-correct; discordance rate d = a + b with effect delta = a - b; per-cluster effects delta_c ~ N(delta, tau^2).
Cluster sizes are drawn from the empirical TRAIN covered decisions per trajectory capped at 5. Success = lower 95% cluster-bootstrap bound > 0 AND estimate >= 0.05
AND half-width <= 0.04; INCONCLUSIVE = half-width > 0.04. Output create-only."""
import collections, json, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch
from compiler.firewall import load_dev

ROOT = Path(__file__).resolve().parents[1]
spec = tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))["primary_claim"]
cnt = collections.Counter(r["decision_id"].rsplit(":", 1)[0] for r in load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl") if r["candidate_coverage"])
sizes = torch.tensor([min(v, 5) for v in cnt.values()], dtype=torch.float)
g = torch.Generator().manual_seed(0)
GRID, SIMS, B = [100, 200, 400, 600, 800], 200, 400


def sim(N, delta, d, tau):
    ok = inc = 0
    ests, hws = [], []
    for _ in range(SIMS):
        n_c = sizes[torch.randint(len(sizes), (N,), generator=g)]
        dc = (delta + tau * torch.randn(N, generator=g)).clamp(-d, d)
        a, b = (d + dc) / 2, (d - dc) / 2
        # per-cluster sum of paired differences: n_c draws of {+1 w.p. a, -1 w.p. b, 0}
        u = torch.rand(int(n_c.sum()), generator=g)
        owner = torch.repeat_interleave(torch.arange(N), n_c.long())
        pa, pb = a[owner], b[owner]
        diff = torch.where(u < pa, 1.0, torch.where(u < pa + pb, -1.0, 0.0))
        s = torch.zeros(N).index_add_(0, owner, diff)
        idx = torch.randint(N, (B, N), generator=g)
        est = s.sum() / n_c.sum()
        bs = s[idx].sum(1) / n_c[idx].sum(1)
        lo, hi = torch.quantile(bs, 0.025).item(), torch.quantile(bs, 0.975).item()
        hw = (hi - lo) / 2
        ests.append(est.item()); hws.append(hw)
        inc += hw > 0.04
        ok += (lo > 0) and (est.item() >= spec["min_effect"]) and (hw <= 0.04)
    return dict(power=round(ok / SIMS, 3), inconclusive=round(inc / SIMS, 3), mean_halfwidth=round(sum(hws) / SIMS, 4), mean_est=round(sum(ests) / SIMS, 4))


out = dict(mean_cluster_size=round(sizes.mean().item(), 2), n_trajectories_in_train_covered=len(sizes), sims=SIMS, bootstrap=B, grid=GRID, results={})
for d, tau in ((0.5, 0.20), (0.5, 0.10), (0.3, 0.20), (0.3, 0.10)):
    for delta in (0.0, 0.05, 0.08, 0.12):
        out["results"][f"d={d},tau={tau},delta={delta}"] = {str(N): sim(N, delta, d, tau) for N in GRID}
cons = out["results"]["d=0.5,tau=0.2,delta=0.08"]
need = [N for N in GRID if cons[str(N)]["power"] >= 0.80]
out["sizing_rule_result"] = dict(scenario="d=0.5,tau=0.2,delta=0.08", smallest_N_with_power_ge_0_80=need[0] if need else "infeasible up to 800")
p = ROOT / ("results/raw/power-sim-v3-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("mean cluster size", out["mean_cluster_size"], "| trajectories", out["n_trajectories_in_train_covered"])
for k, v in out["results"].items():
    print(k, {N: (x["power"], x["mean_halfwidth"]) for N, x in v.items()})
print(out["sizing_rule_result"], "->", p)
