"""Loop engineering v2 (bounded decision-revision experiment A-C) on the SAME frozen Fable data; pre-registration thresholds_loop_eng_v2.toml.
Scorers: 3 frozen adapters + coding head (and their no-context controls), linear probe (exp4lib.probe_scores, fitted on the 303 cleaned TRAIN pools),
options-only TF-IDF conditional logit. Rules combine EXISTING scorers' outputs; every rule picks one existing option for every one of the 46 pools.
CPU only. No generation, no training, no CAL/TEST.
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

PREREG_NAME, PREREG_SHA256 = "thresholds_loop_eng_v2.toml", "55be94ebeba63f82af7357066de185c0930c101e797df0a5cb5e50537eb61975"
OUT = Path("/kaggle/working/out")
GRID, PRIMARY_T = [0.10, 0.20, 0.30], 0.20
REPRO = [0.7826, 0.8043, 0.7609]
C_TEXT, SEED_B, B = [0.01, 0.1, 1.0, 10.0, 100.0], 20261015, 2000


def one(pattern):
    f = glob.glob("/kaggle/input/**/" + pattern, recursive=True)
    assert len(f) == 1, (pattern, f)
    return Path(f[0])


def stop(reason, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ABORT.json").write_text(json.dumps(dict(reason=reason, **kw), indent=1, default=str))
    raise SystemExit("STOP: " + reason)


def softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(one(PREREG_NAME).read_bytes()).hexdigest() != PREREG_SHA256:
        stop("pre-registration sha256 differs")
    src = one("pack5.pt").parent
    sys.argv = [sys.argv[0], str(src), str(OUT)]   # exp4lib reads its input dir from argv at import
    sys.path.insert(0, str(src))
    import exp4lib as L
    from head import DecisionHead
    dev = torch.device("cpu")
    pack = torch.load(src / "pack5.pt", map_location="cpu", weights_only=False)
    heads = torch.load(one("heads.pt"), map_location="cpu", weights_only=False)
    hcfg, tr, val = dict(pack["head_cfg"]), pack["train"], pack["val"]
    val_nc = [dict(r, hq=r["hq_nc"], ho=r["ho_nc"]) for r in val]
    assert len(val) == 46 and len(tr) == 303 and not ({r["trajectory"] for r in tr} & {r["trajectory"] for r in val})
    N, ref, traj = len(val), [r["ref"] for r in val], [r["trajectory"] for r in val]

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
    P = {"seed": [], "seed_nc": []}
    for ad in heads["baseline_adapters"]:
        h = build()
        r = h.load_state_dict({k: v.to(dev) for k, v in ad.items() if L.group_of(k) == "adapter" or k.startswith(("mu_", "sd_"))}, strict=False)
        assert not r.unexpected_keys
        P["seed"].append([softmax(s.double().numpy()) for s in L.scores(h, val, dev)])
        P["seed_nc"].append([softmax(s.double().numpy()) for s in L.scores(h, val_nc, dev)])
    each = [round(float(np.mean([float(int(np.argmax(p)) == r) for p, r in zip(ps, ref)])), 4) for ps in P["seed"]]
    rep = dict(prereg_sha256=PREREG_SHA256, n_pools=N, n_trajectories=len(set(traj)), reproduction=dict(each_seed=each, expected=REPRO, ok=each == REPRO))
    if not rep["reproduction"]["ok"]:
        (OUT / "REPORT.json").write_text(json.dumps(rep, indent=1))
        stop("reproduction gate failed", each=each)

    # probe (per-option logistic regression on ho only), fitted on the 303 cleaned TRAIN pools; recomputed on THESE 46 pools
    probe = [softmax(s) for s in L.probe_scores(tr, val)]
    probe_nc = [softmax(s) for s in L.probe_scores(tr, val_nc)]
    ens = [np.mean([ps[i] for ps in P["seed"]], 0) for i in range(N)]
    ens_nc = [np.mean([ps[i] for ps in P["seed_nc"]], 0) for i in range(N)]
    scorers = {"laya_s0": P["seed"][0], "laya_s1": P["seed"][1], "laya_s2": P["seed"][2], "laya_ens": ens, "laya_nc": ens_nc, "probe": probe, "probe_nc": probe_nc}

    # options-only (text), only combinable if its option order matches pack5 for all 46 pools
    from sklearn.feature_extraction.text import TfidfVectorizer
    pf = one("pools.jsonl*")
    raw = gzip.open(pf, "rt", encoding="utf-8").read() if str(pf).endswith(".gz") else open(pf, encoding="utf-8").read()
    pools = [json.loads(l) for l in raw.splitlines() if l.strip()]
    trt = [p for p in pools if p["split"] == "train"]
    tv = {p["pool_id"]: p for p in pools if p["split"] == "val"}
    vids = [r["decision_id"] for r in val]
    assert set(tv) == set(vids)
    same_order = all(tv[i]["label_index"] == r["ref"] and len(tv[i]["options"]) == len(p) for i, r, p in zip(vids, val, probe))
    rep["options_only_order_verified_same_as_pack5"] = bool(same_order)

    def clogit(Xs, ys, C):
        Pn, d = len(Xs), Xs[0].shape[1]
        K = max(x.shape[0] for x in Xs)
        X, M = torch.zeros(Pn, K, d, dtype=torch.float64), torch.zeros(Pn, K, dtype=torch.bool)
        for i, x in enumerate(Xs):
            X[i, :x.shape[0]], M[i, :x.shape[0]] = x, True
        y, w = torch.tensor(ys), torch.zeros(d, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def cl():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy((X @ w).masked_fill(~M, -1e30), y) + (w @ w) / (2 * C * Pn)
            loss.backward()
            return loss
        opt.step(cl)
        return w.detach()

    def feats(fit, apply):
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)).fit([o for p in fit for o in p["options"]])
        f = lambda p: torch.from_numpy(v.transform(p["options"]).toarray().astype("float64"))
        return [f(p) for p in fit], [f(p) for p in apply]
    trajs = sorted({p["trajectory_id"] for p in trt})
    fold = {t: i % 3 for i, t in enumerate(random.Random(SEED_B).sample(trajs, len(trajs)))}
    ll = collections.defaultdict(list)
    for g in range(3):
        a = [p for p in trt if fold[p["trajectory_id"]] != g]
        b = [p for p in trt if fold[p["trajectory_id"]] == g]
        Xa, Xb = feats(a, b)
        for C in C_TEXT:
            w = clogit(Xa, [p["label_index"] for p in a], C)
            for x, p in zip(Xb, b):
                s = (x @ w).numpy()
                ll[C].append(float(np.log(np.exp(s - s.max()).sum()) + s.max() - s[p["label_index"]]))
    Cb = min(C_TEXT, key=lambda C: np.mean(ll[C]))
    Xa, Xv = feats(trt, [tv[i] for i in vids])
    w = clogit(Xa, [p["label_index"] for p in trt], Cb)
    oo = [softmax((x @ w).numpy()) for x in Xv]
    rep["options_only_C"] = Cb
    oo_hit = np.array([float(int(np.argmax(p)) == tv[i]["label_index"]) for p, i in zip(oo, vids)])
    if same_order:
        scorers["options_only"] = oo

    def hits_of(picks):
        return np.array([float(int(k) == r) for k, r in zip(picks, ref)])
    am = lambda ps: [int(np.argmax(p)) for p in ps]
    stand = {k: hits_of(am(v)) for k, v in scorers.items()}
    rep["standalone"] = {k: dict(correct=int(h.sum()), acc=round(float(h.mean()), 4)) for k, h in stand.items()}
    rep["standalone"]["options_only_text_order"] = dict(correct=int(oo_hit.sum()), acc=round(float(oo_hit.mean()), 4))
    rep["probe_comparability"] = dict(recomputed_on_46=rep["standalone"]["probe"]["acc"], recorded_head_swap_on_46=0.8913, recorded_val_eval_on_53=0.9434)

    # headroom (oracle) over scorer sets
    def oracle(names):
        return int(np.max([stand[n] for n in names], 0).sum())
    rep["oracle_correct_of_46"] = dict(laya_seeds=oracle(["laya_s0", "laya_s1", "laya_s2"]), laya_ens_and_probe=oracle(["laya_ens", "probe"]),
                                       all_context_scorers=oracle(["laya_s0", "laya_s1", "laya_s2", "laya_ens", "probe"]), probe_alone=int(stand["probe"].sum()), laya_ens_alone=int(stand["laya_ens"].sum()))
    rep["both_wrong_laya_ens_and_probe"] = int(((1 - stand["laya_ens"]) * (1 - stand["probe"])).sum())

    # rules: each returns one existing option index per pool
    marg = lambda p: float(np.sort(p)[::-1][0] - np.sort(p)[::-1][1])

    def rule_avg(a, b):
        return [int(np.argmax(0.5 * x + 0.5 * y)) for x, y in zip(a, b)]

    def rule_first(a, b, t):
        return [int(np.argmax(x)) if marg(x) >= t else int(np.argmax(y)) for x, y in zip(a, b)]
    rules = {}
    seeds_arg = [am(P["seed"][k]) for k in range(3)]
    vote = []
    for i in range(N):
        c = collections.Counter(s[i] for s in seeds_arg).most_common()
        vote.append(c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else int(np.argmax(ens[i])))
    rules["S_vote"] = vote
    rules["S_unanimous_else_probe"] = [seeds_arg[0][i] if len({s[i] for s in seeds_arg}) == 1 else int(np.argmax(probe[i])) for i in range(N)]
    variants = {"": (ens, probe), "[nc]": (ens_nc, probe_nc), "[laya_nc+probe]": (ens_nc, probe)}
    if same_order:
        variants["[options_only+probe]"] = (oo, probe)
    for tag, (a, b) in variants.items():
        rules["C_avg" + tag] = rule_avg(a, b)
        for t in GRID:
            rules["C_laya_first_t%.2f%s" % (t, tag)] = rule_first(a, b, t)
            rules["C_probe_first_t%.2f%s" % (t, tag)] = rule_first(b, a, t)
    # bootstrap by trajectory
    ut = sorted(set(traj))
    gi = np.array([ut.index(t) for t in traj])
    rng = np.random.default_rng(SEED_B)
    W = np.zeros((B, len(ut)))
    np.add.at(W, (np.repeat(np.arange(B), len(ut)), rng.integers(0, len(ut), (B, len(ut))).ravel()), 1.0)

    def boot(d):
        sm = np.bincount(gi, weights=d, minlength=len(ut))
        cn = np.bincount(gi, minlength=len(ut)).astype(float)
        b = (W @ sm) / (W @ cn)
        return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]

    def vs(h, base):
        wr, rw = int(((h == 1) & (base == 0)).sum()), int(((h == 0) & (base == 1)).sum())
        return dict(wrong_to_right=wr, right_to_wrong=rw, net_pools=wr - rw, net_gain=round((wr - rw) / N, 4), net_gain_ci95=boot(h - base))
    out_rules = {}
    for name, picks in rules.items():
        h = hits_of(picks)
        out_rules[name] = dict(correct=int(h.sum()), acc=round(float(h.mean()), 4), vs_laya_ens=vs(h, stand["laya_ens"]), vs_probe=vs(h, stand["probe"]))
    rep["rules"], rep["primary_threshold"] = out_rules, PRIMARY_T
    rep["per_pool"] = dict(decision_id=vids, trajectory=traj, ref=ref, picks={k: v for k, v in rules.items()},
                           standalone_picks={k: am(v) for k, v in scorers.items()}, margin_laya_ens=[round(marg(p), 4) for p in ens], margin_probe=[round(marg(p), 4) for p in probe])
    (OUT / "REPORT.json").write_text(json.dumps(rep, indent=1))
    print("LOOP V2 DONE", json.dumps(dict(standalone=rep["standalone"], oracle=rep["oracle_correct_of_46"], both_wrong=rep["both_wrong_laya_ens_and_probe"],
                                          rules={k: (v["correct"], v["vs_probe"]["net_pools"]) for k, v in out_rules.items() if k.endswith("")}), default=str), flush=True)


if __name__ == "__main__":
    main()
