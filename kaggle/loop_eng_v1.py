"""Loop engineering v1, stage 1 (L1 decide-or-abstain) on the SAME Fable data and the FROZEN system (pre-registration thresholds_loop_eng_v1.toml).
Frozen Qwen features (pack5.pt) -> 3 frozen adapters (heads.pt baseline_adapters) -> frozen coding head. No generation, no training, no CAL/TEST.
Controls through the same loop: no-context features; options-only TF-IDF conditional logit (fit on the 335 TRAIN pools only).
"""
import collections
import glob
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PREREG_NAME, PREREG_SHA256 = "thresholds_loop_eng_v1.toml", "f50878562103169e436b54fef24a574edfca87b237a6325b441b0f32bcee18c3"
OUT = Path("/kaggle/working/out")
THRESHOLDS = [0.10, 0.20, 0.30]
REPRO = dict(mean=0.7826, each=[0.7826, 0.8043, 0.7609])
C_TEXT, SEED_B, B = [0.01, 0.1, 1.0, 10.0, 100.0], 20261015, 2000


def one(pattern):
    f = glob.glob("/kaggle/input/**/" + pattern, recursive=True)
    assert len(f) == 1, (pattern, f)
    return Path(f[0])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stop(reason, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ABORT.json").write_text(json.dumps(dict(reason=reason, **kw), indent=1, default=str))
    raise SystemExit("STOP: " + reason)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sha(one(PREREG_NAME)) != PREREG_SHA256:
        stop("pre-registration sha256 differs")
    src = one("pack5.pt").parent
    sys.argv = [sys.argv[0], str(src), str(OUT)]   # exp4lib reads its input dir from argv at import (as head_swap_gpu.py)
    sys.path.insert(0, str(src))
    import exp4lib as L
    from head import DecisionHead
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = torch.load(src / "pack5.pt", map_location="cpu", weights_only=False)
    heads = torch.load(one("heads.pt"), map_location="cpu", weights_only=False)
    hcfg, val = dict(pack["head_cfg"]), pack["val"]
    val_nc = [dict(r, hq=r["hq_nc"], ho=r["ho_nc"]) for r in val]
    assert len(val) == 46 and all("ho_nc" in r for r in val)

    # frozen system: coding head + each frozen baseline adapter (exactly as the head-swap zero-training swap)
    def build():
        with torch.random.fork_rng():
            torch.manual_seed(0)
            h = DecisionHead(**hcfg)
        st = {k: v.float() for k, v in heads["heads"]["coding"]["tensors"].items()}
        r = h.load_state_dict(st, strict=False)
        assert len(st) == 31 and not r.unexpected_keys
        with torch.no_grad():
            h.kind_emb.weight.zero_()
        return h.to(dev)
    probs = {"real": [], "nc": []}
    each = []
    for ad in heads["baseline_adapters"]:
        h = build()
        r = h.load_state_dict({k: v.to(dev) for k, v in ad.items() if L.group_of(k) == "adapter" or k.startswith(("mu_", "sd_"))}, strict=False)
        assert not r.unexpected_keys
        sr, sn = L.scores(h, val, dev), L.scores(h, val_nc, dev)
        each.append(float(np.mean([float(int(s.argmax()) == x["ref"]) for s, x in zip(sr, val)])))
        probs["real"].append([torch.softmax(s.double(), 0).numpy() for s in sr])
        probs["nc"].append([torch.softmax(s.double(), 0).numpy() for s in sn])
    rep = dict(prereg_sha256=PREREG_SHA256, device=str(dev), torch=torch.__version__, n_pools=len(val), n_trajectories=len({r["trajectory"] for r in val}))
    seed_mean_hits = float(np.mean([np.mean([float(int(np.argmax(p)) == x["ref"]) for p, x in zip(ps, val)]) for ps in probs["real"]]))
    rep["reproduction"] = dict(each_seed=[round(e, 4) for e in each], three_seed_mean_of_hits=round(seed_mean_hits, 4), expected=REPRO,
                               ok=[round(e, 4) for e in each] == REPRO["each"])
    if not rep["reproduction"]["ok"]:
        (OUT / "REPORT.json").write_text(json.dumps(rep, indent=1))
        stop("reproduction gate failed", reproduction=rep["reproduction"])

    # options-only control: TF-IDF on candidate text, listwise conditional logit fitted on TRAIN pools only
    from sklearn.feature_extraction.text import TfidfVectorizer
    pf = one("pools.jsonl*")
    raw = gzip.open(pf, "rt", encoding="utf-8").read() if str(pf).endswith(".gz") else open(pf, encoding="utf-8").read()
    pools = [json.loads(l) for l in raw.splitlines() if l.strip()]
    tr = [p for p in pools if p["split"] == "train"]
    tv = {p["pool_id"]: p for p in pools if p["split"] == "val"}
    assert set(tv) == {r["decision_id"] for r in val}, "VAL pool ids differ between datasets"
    assert not ({p["trajectory_id"] for p in tr} & {p["trajectory_id"] for p in tv.values()})

    def clogit(Xs, ys, C):
        P, d = len(Xs), Xs[0].shape[1]
        K = max(x.shape[0] for x in Xs)
        X, M = torch.zeros(P, K, d, dtype=torch.float64), torch.zeros(P, K, dtype=torch.bool)
        for i, x in enumerate(Xs):
            X[i, :x.shape[0]], M[i, :x.shape[0]] = x, True
        y, w = torch.tensor(ys), torch.zeros(d, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def cl():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy((X @ w).masked_fill(~M, -1e30), y) + (w @ w) / (2 * C * P)
            loss.backward()
            return loss
        opt.step(cl)
        return w.detach()

    def feats(fit, apply):
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)).fit([o for p in fit for o in p["options"]])
        f = lambda p: torch.from_numpy(v.transform(p["options"]).toarray().astype("float64"))
        return [f(p) for p in fit], [f(p) for p in apply]
    trajs = sorted({p["trajectory_id"] for p in tr})
    fold = {t: i % 3 for i, t in enumerate(random.Random(SEED_B).sample(trajs, len(trajs)))}
    ll = collections.defaultdict(list)
    for g in range(3):
        a = [p for p in tr if fold[p["trajectory_id"]] != g]
        b = [p for p in tr if fold[p["trajectory_id"]] == g]
        Xa, Xb = feats(a, b)
        for C in C_TEXT:
            w = clogit(Xa, [p["label_index"] for p in a], C)
            for x, p in zip(Xb, b):
                s = (x @ w).numpy()
                ll[C].append(float(np.log(np.exp(s - s.max()).sum()) + s.max() - s[p["label_index"]]))
    Cb = min(C_TEXT, key=lambda C: np.mean(ll[C]))
    vids = [r["decision_id"] for r in val]
    Xa, Xv = feats(tr, [tv[i] for i in vids])
    w = clogit(Xa, [p["label_index"] for p in tr], Cb)
    oo = [torch.softmax(x @ w, 0).numpy() for x in Xv]
    rep["options_only_C"] = Cb

    # arms: per pool probability vector and the reference index in THAT vector's order
    arms = {"single_pass": ([np.mean([ps[i] for ps in probs["real"]], 0) for i in range(len(val))], [r["ref"] for r in val]),
            "no_context": ([np.mean([ps[i] for ps in probs["nc"]], 0) for i in range(len(val))], [r["ref"] for r in val]),
            "options_only": (oo, [tv[i]["label_index"] for i in vids])}
    traj = [r["trajectory"] for r in val]
    ut = sorted(set(traj))
    gi = np.array([ut.index(t) for t in traj])
    rng = np.random.default_rng(SEED_B)
    W = np.zeros((B, len(ut)))
    np.add.at(W, (np.repeat(np.arange(B), len(ut)), rng.integers(0, len(ut), (B, len(ut))).ravel()), 1.0)

    def boot_mean(v, m):
        sm, cn = np.bincount(gi, weights=v * m, minlength=len(ut)), np.bincount(gi, weights=m, minlength=len(ut))
        num, den = W @ sm, W @ cn
        b = num[den > 0] / den[den > 0]
        return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)] if len(b) else None
    res = {}
    for a, (P, ref) in arms.items():
        hit = np.array([float(int(np.argmax(p)) == r) for p, r in zip(P, ref)])
        srt = [np.sort(p)[::-1] for p in P]
        margin = np.array([s[0] - s[1] for s in srt])
        ones = np.ones(len(hit))
        out = dict(single_pass_top1=round(float(hit.mean()), 4), single_pass_ci95=boot_mean(hit, ones), margin_mean=round(float(margin.mean()), 4), L1={})
        for t in THRESHOLDS:
            d = (margin >= t).astype(float)
            out["L1"][str(t)] = dict(coverage=round(float(d.mean()), 4), n_decided=int(d.sum()), n_escalated=int(len(d) - d.sum()),
                                     acc_decided=round(float(hit[d == 1].mean()), 4) if d.sum() else None, acc_decided_ci95=boot_mean(hit, d) if d.sum() else None,
                                     acc_escalated=round(float(hit[d == 0].mean()), 4) if (1 - d).sum() else None,
                                     gain_vs_single_pass=round(float(hit[d == 1].mean() - hit.mean()), 4) if d.sum() else None)
        res[a] = out
    rep["arms"] = res
    rep["reading_fixed_beforehand"] = "L1 helps only if, for single_pass, accuracy on decided pools exceeds single-pass accuracy AND the gain is larger than the same loop on options_only and no_context"
    rep["per_pool"] = dict(decision_id=vids, trajectory=traj, **{a: dict(prob=[[round(float(x), 6) for x in p] for p in P], ref=ref) for a, (P, ref) in arms.items()})
    (OUT / "REPORT.json").write_text(json.dumps(rep, indent=1))
    print("LOOP STAGE1 DONE", json.dumps({a: (v["single_pass_top1"], {t: (x["coverage"], x["acc_decided"]) for t, x in v["L1"].items()}) for a, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
