"""A5-v3 final evaluation: the ONLY module that can open the locked set (and it logs every access). The trainer modules never import this file.
Statistic: per decision delta_i = mean over paired seeds of (H2 correct - H0 correct); trajectory-cluster bootstrap, 2,000 resamples, seed 0.
Spec: configs/a5_v3_spec.toml [primary_claim] and [amendment_2026_10_02_c] final. The locked run is not executed until the user says so after the trainer audit."""
import json
from pathlib import Path

import torch

from compiler import a5v2
from compiler.firewall import log_locked_access
from decision_head import v3_data
from decision_head.v3_model import H0Net, ResidualNet
from decision_head.v3_training import SINGLE_DEPTH, System

ROOT = Path(__file__).resolve().parents[1]
BOOT, BOOT_SEED, MIN_EFFECT, MAX_HALF = 2000, 0, 0.05, 0.04


def load_system(path, emb):
    st = torch.load(path, map_location="cpu", weights_only=False)
    nd = len(v3_data.DEPTHS)
    h0 = H0Net(v3_data.NF, emb, nd, st["proj"], st["cfg"]["dropout"])
    res = ResidualNet(emb, nd, st["proj"], st["treatment"], st["cfg"]["dropout"], v3_data.DEPTHS.index(SINGLE_DEPTH))
    h0.load_state_dict(st["h0"]), res.load_state_dict(st["res"])
    return System(h0.eval(), res.eval(), st["cfg"], st["seed"], st["proj"], st["treatment"], st["record"])


def correctness(systems, decs):
    """({arm: [per-seed list of 0/1 per decision]}) for set 0 of every decision."""
    out = {"H0": [], "H2": []}
    for s in systems:
        h0, h2 = s.scores(decs, 0)
        for arm, lg in (("H0", h0), ("H2", h2)):
            out[arm].append([float(l.argmax().item() == d["sets"][0]["true_pos"]) for l, d in zip(lg, decs)])
    return out


def bootstrap(delta, tids, boot=BOOT, seed=BOOT_SEED):
    """Trajectory-cluster bootstrap of the mean of per-decision deltas. Returns (mean, lo95, hi95)."""
    by = {}
    for x, t in zip(delta, tids):
        by.setdefault(t, []).append(x)
    cl = list(by.values())
    sums, cnt = torch.tensor([sum(c) for c in cl]), torch.tensor([float(len(c)) for c in cl])
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(cl), (boot, len(cl)), generator=g)
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    q = torch.quantile(m, torch.tensor([0.025, 0.975]))
    return (sums.sum() / cnt.sum()).item(), q[0].item(), q[1].item()


def verdict(mean, lo, hi):
    if (hi - lo) / 2 > MAX_HALF:
        return "INCONCLUSIVE (interval half-width above 0.04)"
    return "SUCCESS" if lo > 0 and mean >= MIN_EFFECT else "NOT SUPPORTED"


def compare(systems, decs):
    c = correctness(systems, decs)
    n = len(decs)
    delta = [sum(c["H2"][s][i] - c["H0"][s][i] for s in range(len(systems))) / len(systems) for i in range(n)]
    tids = [d["tid"] for d in decs]
    m, lo, hi = bootstrap(delta, tids)
    return dict(n_decisions=n, n_clusters=len(set(tids)), seeds=len(systems), h0_acc=sum(map(sum, c["H0"])) / (n * len(systems)), h2_acc=sum(map(sum, c["H2"])) / (n * len(systems)),
                delta_mean=m, ci95=[lo, hi], verdict=verdict(m, lo, hi))


def load_locked(manifest_ids, capdir, textemb_dir):
    """The single place the locked trajectories and decisions are opened. Logged. No outcome is computed here."""
    log_locked_access("v3_evaluation.load_locked: final locked evaluation")
    raw = a5v2.load_raw(ROOT / "data/locked_v3/fresh-locked.jsonl")
    want, built = set(manifest_ids), {}
    for l in open(ROOT / "data/decisions/locked-v3.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["decision_id"] in want:
            d, _ = a5v2.build_decision(r, raw[r["decision_id"].rsplit(":", 1)[0]]["trajectory"])
            if d is not None:
                built[d["did"]] = d
    fz = v3_data.frozen_odds()
    sets = {k: [a5v2.generate(d, fz[d["stratum"]])] for k, d in built.items()}
    TE, te_idx = v3_data.load_textemb(textemb_dir)
    return v3_data.assemble(manifest_ids, built, v3_data.load_captures(capdir), TE, te_idx, "locked", sets)
