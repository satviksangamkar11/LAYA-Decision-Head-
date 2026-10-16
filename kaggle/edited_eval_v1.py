"""Edited-only mutation control, evaluation (user GO, chat 2026-10-13). Kaggle CPU. Development diagnostic, not a held-out test.

Inputs (mounted): kernel output laya-edited-capture (diff_features.pt), dataset laya-mutation-edited-labels-v1 (labels_edited.json), dataset
laya-mutation-edited-v1 (requests_edited: option texts), dataset laya-mutation-score-v2 + laya-mutation-labels-v1 (the earlier 19 pools: only for the
token-consensus baseline on the ORIGINAL-included pools). Writes /kaggle/working/RESULT.json.
"""
import difflib
import glob
import gzip
import json
import os
import random
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

IN, OUT = "/kaggle/input", "/kaggle/working"
CS = (1e-3, 1e-2, 1e-1)
N_PERM, N_BOOT = 1000, 2000
find = lambda pat: sorted(glob.glob(os.path.join(IN, "**", pat), recursive=True))
PLAN = {
    "written": "2026-10-13, before this evaluation ran",
    "pools": "15 states (= 15 trajectories), every candidate an edited copy (36 PASS, 83 FAIL); the original is not in any pool",
    "co_primary": ["whole_c: existing whole-option mean features, pool-centred over the edited candidates", "w8c: edit sites +- 8 tokens, pool-centred"],
    "probe": "standardise + logistic regression (class_weight balanced) on per-option PASS / FAIL; leave one state out (15 folds); C from %s by inner "
             "grouped 3-fold CV on the training states (highest mean top-choice pass, ties -> smaller C)" % (CS,),
    "metric": "per state: 1 if the highest-scoring candidate PASSES the recorded tests; chance per state = share of passing candidates; reported: mean "
              "over the 15 states, mean chance, paired difference (top choice minus chance) with a %d-run bootstrap over states (95%%), and the "
              "12 operator-matched states separately (also with pools restricted to the operators that occur with both outcomes)" % N_BOOT,
    "baselines": {"chance": "share passing", "no_context": "A features, same probe", "lexical": "bag of the candidate's site token ids, same probe",
                  "operator": "one-hot mutation operator, same probe", "consensus": "unsupervised: the candidate with the highest mean difflib "
                  "similarity of its text to the other candidates (no training)", "consensus_on_original_pools": "same rule on the earlier 19 pools "
                  "that INCLUDE the original: share of states where it picks the original (size of the original-as-centre shortcut)"},
    "permutation": "%d runs per co-primary feature; PASS labels shuffled within each pool keeping the number of passing candidates; C fixed to the real "
                   "run's modal C; p = (1 + #runs with mean top-choice pass >= observed) / (1 + runs); Holm over the 2 co-primary features" % N_PERM,
    "readings (fixed now)": {
        "BEATS BASELINES": "a co-primary feature: bootstrap CI of (top choice - chance) above 0 AND Holm-adjusted permutation p < 0.05 AND at least 15 "
                           "points above every one of lexical, operator and consensus -> evidence of test-outcome discrimination within synthetic mutations",
        "ONLY WITH ORIGINAL": "no co-primary feature beats baselines here while 16.25 passed -> original-versus-mutant shortcut",
        "OPERATOR": "beats baselines on all 15 but not on the 12 operator-matched states (restricted pools) -> likely mutation-operator shortcut",
        "FAILS": "no co-primary feature beats chance -> this representation and probe show no generalising pass / fail signal"},
}


def vecs(rec, key, centre=True):
    x = rec[key].float().reshape(rec[key].shape[0], -1).numpy()
    return x - x.mean(0, keepdims=True) if centre else x


def fit_score(Xtr, ytr, Xte, C):
    A, lab = np.concatenate(Xtr), np.concatenate(ytr)
    sc = StandardScaler().fit(A)
    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(A), lab)
    return [m.decision_function(sc.transform(x)) for x in Xte]


def top_pass(scores, y):
    return float(y[int(np.argmax(scores))])


def run_cv(X, Y, groups, C_fixed=None, restrict=None):
    """Leave one state out. Y: per state 0/1 array. restrict: per state index list for evaluation (None = all). -> per-state top-choice pass, Cs."""
    n, hits, Cs = len(X), [], []
    for k in range(n):
        tr = [i for i in range(n) if i != k]
        C = C_fixed
        if C is None:
            sc = {}
            for c in CS:
                h = []
                for a, b in GroupKFold(3).split(np.zeros(len(tr)), groups=np.array([groups[i] for i in tr])):
                    s = fit_score([X[tr[i]] for i in a], [Y[tr[i]] for i in a], [X[tr[i]] for i in b], c)
                    h += [top_pass(x, Y[tr[i]]) for x, i in zip(s, b)]
                sc[c] = np.mean(h)
            C = max(CS, key=lambda c: (sc[c], -c))
        Cs.append(C)
        s = fit_score([X[i] for i in tr], [Y[i] for i in tr], [X[k]], C)[0]
        idx = restrict[k] if restrict is not None else list(range(len(Y[k])))
        hits.append(top_pass(s[idx], Y[k][idx]))
    return hits, Cs


def boot_ci(d, seed=11):
    rng, n = random.Random(seed), len(d)
    m = sorted(np.mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(N_BOOT))
    return [round(m[int(0.025 * N_BOOT)], 4), round(m[int(0.975 * N_BOOT) - 1], 4)]


def consensus_pick(texts):
    sim = [[difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if i != j else 0.0 for j, b in enumerate(texts)] for i, a in enumerate(texts)]
    m = [sum(r) / (len(texts) - 1) for r in sim]
    best = max(m)
    return [i for i, v in enumerate(m) if v == best]


def load_jsonl(stem):
    p = find(stem + "*")[0]
    return [json.loads(l) for l in (gzip.open if p.endswith(".gz") else open)(p, "rt", encoding="utf-8")]


def main():
    t0 = time.time()
    feats = torch.load(find("diff_features.pt")[0], map_location="cpu", weights_only=False)
    lab = {x["decision_id"]: x for x in json.load(open(find("labels_edited.json")[0]))}
    req = {r["decision_id"]: r for r in load_jsonl("requests_edited.jsonl")}
    B = {r["decision_id"]: r for r in feats["records"]["B"]}
    A = {r["decision_id"]: r for r in feats["records"]["A"]}
    ids = sorted(lab)
    assert set(ids) == set(B) == set(A) == set(req) and len(ids) == 15
    for d in ids:
        assert B[d]["option_sha256"] == req[d]["option_sha256"] and len(lab[d]["passes"]) == B[d]["ho_whole"].shape[0] == len(req[d]["options"])
    Y = [np.array(lab[d]["passes"], dtype=float) for d in ids]
    groups = [d.split(":")[1] for d in ids]
    matched = [i for i, d in enumerate(ids) if lab[d]["operator_matched"]]
    restrict = [[j for j, op in enumerate(lab[d]["operators"]) if op in lab[d]["matched_operators"]] or list(range(len(Y[i]))) for i, d in enumerate(ids)]
    chance = [float(y.mean()) for y in Y]
    sets = {"whole_c": [vecs(B[d], "ho_whole") for d in ids], "w8c": [vecs(B[d], "ho_w8") for d in ids], "w32c": [vecs(B[d], "ho_w32") for d in ids],
            "no_context_whole_c": [vecs(A[d], "ho_whole") for d in ids], "no_context_w8c": [vecs(A[d], "ho_w8") for d in ids]}
    dv = DictVectorizer(sparse=False).fit([{str(t): 1 for t in toks} for d in ids for toks in B[d]["site_token_ids"]])
    sets["lexical"] = [dv.transform([{str(t): toks.count(t) for t in toks} for toks in B[d]["site_token_ids"]]) for d in ids]
    ops = sorted({op for d in ids for op in lab[d]["operators"]})
    sets["operator"] = [np.array([[float(op == o) for o in ops] for op in lab[d]["operators"]]) for d in ids]

    def summ(h, idx=None):
        idx = list(range(len(ids))) if idx is None else idx
        hh, cc = [h[i] for i in idx], [chance[i] for i in idx]
        d = [a - b for a, b in zip(hh, cc)]
        return dict(states=len(idx), top_choice_pass=round(float(np.mean(hh)), 4), chance=round(float(np.mean(cc)), 4),
                    minus_chance=round(float(np.mean(d)), 4), ci95=boot_ci(d))
    res = {}
    for name, X in sets.items():
        h, Cs = run_cv(X, Y, groups)
        hm, _ = run_cv(X, Y, groups, restrict=restrict)
        res[name] = dict(all15=summ(h), operator_matched12=summ(h, matched), operator_matched12_restricted_pools=summ(hm, matched),
                         per_state=dict(zip(ids, h)), C_chosen=Cs)
        print("RESULT", name, json.dumps(res[name]["all15"]), "| matched12", json.dumps(res[name]["operator_matched12"]), flush=True)
    cons = []
    for d, y in zip(ids, Y):
        p = consensus_pick(req[d]["options"])
        cons.append(float(np.mean([y[i] for i in p])))
    res["consensus"] = dict(all15=summ(cons), operator_matched12=summ(cons, matched))
    old_req = {r["decision_id"]: r for r in load_jsonl("requests_level1.jsonl")}
    old_lab = {x["decision_id"]: x for x in json.load(open(find("labels.json")[0]))}
    old = [float(np.mean([float(i == old_lab[d]["fable_index"]) for i in consensus_pick(old_req[d]["options"])])) for d in sorted(old_lab)]
    res["consensus_on_original_pools"] = dict(states=len(old), picks_original=round(float(np.mean(old)), 4),
                                              chance=round(float(np.mean([1 / old_lab[d]["n_options"] for d in old_lab])), 4))
    print("CONSENSUS", json.dumps(res["consensus"]["all15"]), "| on original pools", json.dumps(res["consensus_on_original_pools"]), flush=True)
    perm, rng = {}, random.Random(20261013)
    for name in ("whole_c", "w8c"):
        Cs = res[name]["C_chosen"]
        Cm = max(set(Cs), key=Cs.count)
        shuffles = []
        for _ in range(N_PERM):
            ys = []
            for y in Y:
                z = list(y)
                rng.shuffle(z)
                ys.append(np.array(z))
            shuffles.append(ys)
        null = Parallel(n_jobs=-1)(delayed(lambda ys: float(np.mean(run_cv(sets[name], ys, groups, C_fixed=Cm)[0])))(ys) for ys in shuffles)
        obs = res[name]["all15"]["top_choice_pass"]
        perm[name] = dict(C_fixed=Cm, null_mean=round(float(np.mean(null)), 4), null_95th=round(float(np.percentile(null, 95)), 4), observed=obs,
                          p=round((1 + sum(v >= obs for v in null)) / (1 + N_PERM), 4))
    ps = sorted(perm, key=lambda k: perm[k]["p"])
    for r, k in enumerate(ps):   # Holm over 2
        perm[k]["p_holm"] = round(min(1.0, max(perm[x]["p"] * (2 - i) for i, x in enumerate(ps[:r + 1]))), 4)
    best_base = max(res[b]["all15"]["top_choice_pass"] for b in ("lexical", "operator", "consensus"))
    reading = {}
    for name in ("whole_c", "w8c"):
        a = res[name]["all15"]
        beats = a["ci95"][0] > 0 and perm[name]["p_holm"] < 0.05 and a["top_choice_pass"] - best_base >= 0.15
        m = res[name]["operator_matched12_restricted_pools"]
        reading[name] = dict(ci_above_0=a["ci95"][0] > 0, p_holm_lt_005=perm[name]["p_holm"] < 0.05, margin_over_best_baseline=round(a["top_choice_pass"] - best_base, 4),
                             beats_baselines=bool(beats), matched_restricted_ci_above_0=m["ci95"][0] > 0)
    any_beats = any(v["beats_baselines"] for v in reading.values())
    verdict = ("OPERATOR" if any_beats and not any(v["beats_baselines"] and v["matched_restricted_ci_above_0"] for v in reading.values())
               else "BEATS BASELINES" if any_beats
               else "FAILS" if all(res[n]["all15"]["ci95"][0] <= 0 for n in ("whole_c", "w8c")) else "ONLY WITH ORIGINAL")
    out = dict(plan=PLAN, results=res, permutation=perm, reading=reading, verdict=verdict, seconds=round(time.time() - t0, 1))
    out = json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))   # numpy scalars -> Python (run 1 crashed here)
    json.dump(out, open(os.path.join(OUT, "RESULT.json"), "w"), indent=1)
    print("VERDICT", verdict, json.dumps(out["reading"]), json.dumps(out["permutation"]), flush=True)


if __name__ == "__main__":
    main()
