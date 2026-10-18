"""[EL-ARCH-1 F2c / B2] Post-run DIAGNOSTIC analysis of the captured hidden states of the local bank run (Qwen2.5-Coder-1.5B, depths 14/19/23, hidden 1536).
EXPLORATORY, FEW TASKS. Strictly comparison-only: selector=false, can_approve=false; it never approves, never overrides trusted evidence (invariant I2) and writes nothing into any journal.
The numbers are a description of this run, not a result about the model or about LAYA.

    python tools/el_bank_state_probe.py <run dir> [--out report.json] [--depths 14,19,23] [--C 0.01] [--folds 5] [--boot 2000] [--seed 0]

Run dir layout: tasks/<task_id>/journal.jsonl (the controller journal: evidence + behaviour_clusters + candidate entries) and tasks/<task_id>/states/<candidate_id>.pt (evidence_ladder/models/state_capture.py).
Labels come from EXECUTION evidence only: `pass_all` = the candidate passed every public-example / parse record in the journal (trusted L5-style evidence), `majority_cluster` = it sits in the
largest behaviour cluster of its task (when the journal has clusters). Probe: standardised option state (the candidate's own-token mean, per depth, and all depths concatenated), logistic regression
with strong L2 regularisation (C=0.01, class_weight balanced), GROUPED cross-validation BY TASK (a task never appears in train and test). Reported per depth: AUROC (pooled held-out, and its
bootstrap-over-tasks 95% interval), top-1-pick accuracy per informative task vs random (exact expectation) vs candidate-order baseline (the first candidate generated), with 95% bootstrap intervals
over tasks. Informative task = it has at least one passing and one failing candidate (otherwise there is nothing to choose).

The frozen 4B LAYA head cannot consume these states: it expects option states [N, 3 depths, 2560] of Qwen3-4B (probe.check_rows / the capture provenance guard refuse any other shape or
profile; `frozen_laya_guard` below shows the refusal). A 1.5B-specific adapter would need: a new profile (hidden 1536, depths 14/19/23, query cue, option prefix), a recapture of the TRAIN/CAL
pools with that profile on one device class, an adapter + head trained on them (adapter-only first, Qwen frozen), calibration on held-out pools and the pre-registered gates; that is a
separate GO (CLAUDE.md: any further capture or fine-tuning run needs the user's GO)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABEL = "exploratory, few tasks; diagnostic and comparison-only (selector=false, can_approve=false); never overrides trusted evidence (I2)"
MIN_INFORMATIVE_TASKS = 10      # below this the report says INSUFFICIENT_DATA for any claim; numbers are still printed as description


# ------------------------------------------------------------------ labels from execution evidence
def labels_from_journal(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """-> {"order": [candidate ids in generation order], "pass_all": {id: bool}, "cluster": {id: sig}, "majority_cluster": {id: bool}}. Only trusted execution records count."""
    order: List[str] = []
    seen_ev: Dict[str, List[str]] = {}
    for e in entries:
        k, d = e["kind"], e["data"]
        if k == "candidate":
            cid = d.get("id") or d.get("candidate_id")
            if cid and cid not in order:
                order.append(cid)
        elif k == "evidence":
            ev = d["evidence"]
            if ev["check_id"] in ("public_examples_v1", "parse_v1"):
                seen_ev.setdefault(ev["candidate_id"], []).append(ev["status"])
                if ev["candidate_id"] not in order:
                    order.append(ev["candidate_id"])
    pass_all = {c: bool(s) and all(x == "pass" for x in s) for c, s in seen_ev.items()}
    cluster: Dict[str, str] = {}
    for e in entries:
        if e["kind"] == "behaviour_clusters":
            for sig, ids in e["data"]["members"].items():
                for i in ids:
                    cluster[i] = sig
    major = {}
    if cluster:
        sizes: Dict[str, int] = {}
        for sig in cluster.values():
            sizes[sig] = sizes.get(sig, 0) + 1
        top = max(sizes.values())
        major = {c: sizes[s] == top for c, s in cluster.items()}
    return {"order": order, "pass_all": pass_all, "cluster": cluster, "majority_cluster": major}


def frozen_laya_guard(state: Dict[str, Any]):
    """Show that the frozen 4B head's own row check refuses a 1.5B state (hidden 1536, other profile). Raises evidence_ladder.models.probe.ProbeFitError."""
    import torch
    from evidence_ladder.models import probe as P
    ho = state["ho"].float().unsqueeze(0)
    ho = torch.cat([ho, ho], 0) if ho.dim() == 3 else torch.stack([state["ho"].float()] * 2)
    P.check_rows([{"decision_id": state.get("candidate_id"), "ho": ho, "pos": torch.tensor([True, False])}])


# ------------------------------------------------------------------ statistics
def auroc(y: Sequence[int], s: Sequence[float]) -> Optional[float]:
    pos = [v for v, t in zip(s, y) if t]
    neg = [v for v, t in zip(s, y) if not t]
    if not pos or not neg:
        return None
    c = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def _ci(vals: List[float]) -> Optional[List[float]]:
    v = sorted(x for x in vals if x is not None)
    if len(v) < 20:
        return None
    return [round(v[int(0.025 * (len(v) - 1))], 4), round(v[int(0.975 * (len(v) - 1))], 4)]


def analyze(rows: Sequence[Dict[str, Any]], label_key: str = "pass_all", feature: Any = 0, C: float = 0.01, folds: int = 5, boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """rows: [{"task": str, "order": int, "y": {label_key: bool}, "x": {feature: [float]}}]. `feature` is a depth index, or "all" (every depth's vector concatenated, key "all" in x).
    -> metrics dict (JSON-serialisable). Pure function of its input: same rows, same seed, same report."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    tasks = sorted({r["task"] for r in rows})
    X = np.array([r["x"][feature] for r in rows], dtype=np.float64)
    y = np.array([1 if r["y"][label_key] else 0 for r in rows])
    g = np.array([tasks.index(r["task"]) for r in rows])
    n_f = max(2, min(folds, len(tasks)))
    score = np.zeros(len(rows))
    if len(tasks) < 2 or len(set(y.tolist())) < 2:
        return {"status": "INSUFFICIENT_DATA", "reason": "need >= 2 tasks and both classes", "n_tasks": len(tasks), "n_candidates": len(rows)}
    for tr, te in GroupKFold(n_splits=n_f).split(X, y, g):
        if len(set(y[tr].tolist())) < 2:
            score[te] = 0.0
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed).fit(sc.transform(X[tr]), y[tr])
        score[te] = m.decision_function(sc.transform(X[te]))
    per_task = []
    for t in tasks:
        idx = [i for i in range(len(rows)) if rows[i]["task"] == t]
        yt = y[idx]
        if yt.sum() == 0 or yt.sum() == len(idx):
            continue                                                       # uninformative: nothing to choose between
        pick = idx[int(np.argmax(score[idx]))]
        first = min(idx, key=lambda i: rows[i]["order"])
        per_task.append({"task": t, "n": len(idx), "pass_rate": float(yt.mean()), "probe_top1": int(y[pick]), "random_expected": float(yt.mean()), "order_top1": int(y[first])})

    def agg(pt):
        n = len(pt)
        return None if n == 0 else {"probe": sum(p["probe_top1"] for p in pt) / n, "random": sum(p["random_expected"] for p in pt) / n, "candidate_order": sum(p["order_top1"] for p in pt) / n}
    a = agg(per_task)
    ci: Dict[str, Any] = {}
    au_b: List[float] = []
    if per_task:
        d_rand, d_ord, acc = [], [], []
        for _ in range(boot):
            ix = rng.randint(0, len(per_task), len(per_task))
            s = [per_task[i] for i in ix]
            q = agg(s)
            acc.append(q["probe"])
            d_rand.append(q["probe"] - q["random"])
            d_ord.append(q["probe"] - q["candidate_order"])
        ci = {"probe_top1": _ci(acc), "probe_minus_random": _ci(d_rand), "probe_minus_candidate_order": _ci(d_ord)}
    for _ in range(boot):
        tb = rng.randint(0, len(tasks), len(tasks))
        idx = [i for t in tb for i in range(len(rows)) if g[i] == t]
        v = auroc(y[idx].tolist(), score[idx].tolist())
        if v is not None:
            au_b.append(v)
    return {"status": "ok" if len(per_task) >= MIN_INFORMATIVE_TASKS else "INSUFFICIENT_DATA", "n_tasks": len(tasks), "n_informative_tasks": len(per_task), "n_candidates": len(rows), "n_positive": int(y.sum()),
            "folds": n_f, "C": C, "label": label_key, "feature": feature, "auroc": None if auroc(y.tolist(), score.tolist()) is None else round(auroc(y.tolist(), score.tolist()), 4),
            "auroc_ci95": _ci(au_b), "top1": None if a is None else {k: round(v, 4) for k, v in a.items()}, "top1_ci95": ci,
            "selector": False, "can_approve": False, "note": LABEL}


# ------------------------------------------------------------------ loading
def load_rows(run_dir, depths_idx: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch
    root = Path(run_dir) / "tasks"
    rows: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {"tasks_with_states": 0, "states": 0, "skipped_no_label": 0, "profiles": set(), "hidden": set()}
    if not root.is_dir():
        info["profiles"], info["hidden"] = [], []
        return rows, info
    for td in sorted(p for p in root.iterdir() if p.is_dir()):
        sd, jp = td / "states", td / "journal.jsonl"
        if not sd.is_dir() or not jp.exists():
            continue
        lab = labels_from_journal([json.loads(l) for l in jp.read_text("utf-8").splitlines() if l.strip()])
        n0 = len(rows)
        for f in sorted(sd.glob("*.pt")):
            st = torch.load(f, map_location="cpu", weights_only=False)
            cid = st["candidate_id"]
            if cid not in lab["pass_all"]:
                info["skipped_no_label"] += 1
                continue
            ho = st["ho"].float()
            info["profiles"].add(st.get("profile_id"))
            info["hidden"].add(int(ho.shape[1]))
            x = {i: ho[i].tolist() for i in depths_idx}
            x["all"] = [v for i in depths_idx for v in x[i]]
            rows.append({"task": td.name, "order": lab["order"].index(cid) if cid in lab["order"] else 10 ** 6, "x": x, "cand": cid,
                         "y": {"pass_all": lab["pass_all"][cid], "majority_cluster": bool(lab["majority_cluster"].get(cid, False))}, "has_cluster": cid in lab["majority_cluster"]})
            info["states"] += 1
        if len(rows) > n0:
            info["tasks_with_states"] += 1
    info["profiles"], info["hidden"] = sorted(map(str, info["profiles"])), sorted(info["hidden"])
    return rows, info


def report(run_dir, depths: Sequence[int] = (14, 19, 23), C: float = 0.01, folds: int = 5, boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    rows, info = load_rows(run_dir, range(len(depths)))
    out: Dict[str, Any] = {"label": LABEL, "run_dir": str(run_dir), "depths": list(depths), "load": info, "by_depth": {}, "frozen_4b_laya_head": {
        "can_consume_these_states": False, "why": "the frozen head expects Qwen3-4B option states [N, 3, 2560]; probe.check_rows / the capture provenance guard refuse hidden 1536 and any other profile",
        "needs_for_a_1_5b_adapter": "new profile (hidden 1536, depths 14/19/23, cue, option prefix), recapture of the TRAIN/CAL pools on one device class, adapter+head training with Qwen frozen, calibration on held-out pools, pre-registered gates; separate GO"}}
    if not rows:
        out["status"] = "NO_STATES"
        return out
    for key, name in [(i, "depth_%d" % d) for i, d in enumerate(depths)] + [("all", "all_depths")]:
        out["by_depth"][name] = {lk: analyze(rows, lk, key, C, folds, boot, seed) for lk in (("pass_all", "majority_cluster") if any(r["has_cluster"] for r in rows) else ("pass_all",))}
    out["status"] = "ok"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--depths", default="14,19,23")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rep = report(a.run_dir, [int(x) for x in a.depths.split(",")], a.C, a.folds, a.boot, a.seed)
    txt = json.dumps(rep, indent=1, sort_keys=True, default=str)
    if a.out:
        p = Path(a.out)
        if p.exists():
            print("refusing to overwrite %s (create-only)" % p)
            return 2
        p.write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
