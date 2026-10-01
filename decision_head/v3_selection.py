"""A5-v3 cell selection and final fitting. Selection uses ONLY trajectory-grouped 5-fold CV inside TRAIN (CAL is not used), one fixed seed (0).
Spec: configs/a5_v3_spec.toml [amendment_2026_10_02_c] cells / selection / final. Nothing here can see CAL outcomes or the locked set."""
import torch
import torch.nn.functional as F

from compiler import a5v2
from decision_head.v3_training import CFG, fit_residual, h0_bundle, nested_subsets

PROJ = [128, 256, 512]
TREATMENTS = ["single", "mean", "learned"]          # simplest first: the tie-break order
CELLS = [(p, t) for p in PROJ for t in TREATMENTS]
TIE = 0.01


def dev_scores(train_decs, cfg=CFG, seed=0, device="cpu", cells=CELLS, log=print):
    """Mean held-out NLL of the H2 system per cell over trajectory-grouped CV inside TRAIN. H0 (which depends only on the projection size) is fitted once per fold and shared."""
    fold_of = a5v2.folds_by_trajectory(train_decs)
    tot = {c: [0.0, 0] for c in cells}
    for f in range(a5v2.S["cv_folds"]):
        tr, te = [d for d in train_decs if fold_of[d["tid"]] != f], [d for d in train_decs if fold_of[d["tid"]] == f]
        if not te:
            continue
        for p in sorted({c[0] for c in cells}):
            bundle = h0_bundle(tr, cfg, seed, p, device)
            for t in [c[1] for c in cells if c[0] == p]:
                sysm = fit_residual(bundle, cfg, t, device)
                _, h2 = sysm.scores(te, 0, device)
                tot[(p, t)][0] += sum(-F.log_softmax(l, 0)[d["sets"][0]["true_pos"]].item() for l, d in zip(h2, te))
                tot[(p, t)][1] += len(te)
                log(f"fold {f} proj {p} {t}: running mean NLL {tot[(p, t)][0] / tot[(p, t)][1]:.4f}")
    return {c: v[0] / v[1] for c, v in tot.items()}


def choose(scores):
    """Lowest mean NLL; cells within 0.01 of it tie, and the tie goes to the smaller projection, then the simpler depth treatment."""
    best = min(scores.values())
    near = [c for c, s in scores.items() if s <= best + TIE]
    return min(near, key=lambda c: (c[0], TREATMENTS.index(c[1])))


def final_fits(train_decs, cell, sizes, seeds, outdir, cfg=CFG, device="cpu"):
    """Selected cell x nested training sizes x paired seeds. One checkpoint per (size, seed); H0 and H2 of a checkpoint share rows, sets and seed."""
    from pathlib import Path
    Path(outdir).mkdir(parents=True, exist_ok=True)
    subs = nested_subsets(train_decs, sizes)
    paths = []
    for n, decs in subs.items():
        for s in seeds:
            sysm = fit_residual(h0_bundle(decs, cfg, s, cell[0], device), cfg, cell[1], device)
            p = Path(outdir) / f"size{n}-seed{s}.pt"
            with open(p, "xb") as f:
                torch.save(sysm.state(), f)
            paths.append(p)
    return paths
