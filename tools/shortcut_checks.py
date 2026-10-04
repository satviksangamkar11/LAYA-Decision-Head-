"""Three shortcut checks for the outcome-labelled head (plan 2026-10-03): repo-disjoint split, state-blind baseline, shuffled-state control.
Labels: done = finish step of a resolved trajectory (else 0); value = trajectory resolved. Output is create-only under results/raw/.
Any head must beat `state_blind` (and its own shuffled-state run) on the SAME folds; run `evaluate(X, y, groups, folds)` with head features to get the comparison.
Metadata only here (no hidden states), so this is the floor a head has to clear, not a head result."""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KINDS = ["view", "explore", "run_other", "edit", "run_tests", "think", "finish", "plan"]


def load(path=ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"):
    rows = [json.loads(l) for l in open(path)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["provenance"]["trajectory_id"]].append(r)
    X, done, value, repo, traj = [], [], [], [], []
    for tid, t in by.items():
        t.sort(key=lambda r: r["t"])
        res = bool(t[0]["label_side"]["resolved"])
        seen = np.zeros(len(KINDS))
        last = [np.zeros(len(KINDS)) for _ in range(3)]
        for i, r in enumerate(t):
            # state-blind features: only what happened BEFORE this decision (no next-action kind, no trajectory length)
            X.append(np.concatenate([[r["t"], np.log1p(r["t"])], seen, *last]))
            k = KINDS.index(r["next_action_kind"])
            done.append(int(res and i == len(t) - 1 and r["next_action_kind"] == "finish"))
            value.append(int(res)); repo.append(r["provenance"]["repo"]); traj.append(tid)
            oh = np.zeros(len(KINDS)); oh[k] = 1
            seen = seen + oh; last = [oh] + last[:2]
    return np.array(X), np.array(done), np.array(value), np.array(repo), np.array(traj)


def folds_by_repo(groups, k=5, seed=0):
    """Repo-disjoint: no repo appears in both train and test of any fold."""
    ug = np.unique(groups); np.random.default_rng(seed).shuffle(ug)
    f = {g: i % k for i, g in enumerate(ug)}
    return np.array([f[g] for g in groups])


def auroc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def fit_predict(Xtr, ytr, Xte, l2=1.0, iters=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]; B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    w = np.zeros(A.shape[1])
    for _ in range(iters):  # plain gradient descent on L2 logistic loss
        p = 1 / (1 + np.exp(-A @ w)); w -= 0.5 * (A.T @ (p - ytr) / len(A) + l2 * w / len(A))
    return B @ w


def evaluate(X, y, groups, folds, shuffle=False, seed=0):
    """Mean out-of-fold AUROC. shuffle=True permutes X rows across the whole set (breaks the X<->y link, keeps marginals): must drop to ~0.5."""
    if shuffle:
        X = X[np.random.default_rng(seed).permutation(len(X))]
    oof = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        oof[te] = fit_predict(X[~te], y[~te], X[te])
    return auroc(y, oof)


if __name__ == "__main__":
    X, done, value, repo, traj = load()
    folds = folds_by_repo(repo)
    for f in np.unique(folds):  # check 1: repo-disjoint by construction
        assert not set(repo[folds == f]) & set(repo[folds != f])
    out = dict(rows=len(X), trajectories=len(set(traj)), repos=len(set(repo)), done_positives=int(done.sum()), value_positive_rate=float(value.mean()),
               auroc_state_blind={n: evaluate(X, y, repo, folds) for n, y in (("done", done), ("value", value))},
               auroc_shuffled={n: evaluate(X, y, repo, folds, shuffle=True) for n, y in (("done", done), ("value", value))},
               note="metadata-only floor; value is per-step so rows of one trajectory share a label (AUROC is over rows, not trajectories)")
    print(json.dumps(out, indent=1))
    p = ROOT / f"results/raw/shortcut-checks-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(p, "x") as fh:
        json.dump(out, fh, indent=1)
