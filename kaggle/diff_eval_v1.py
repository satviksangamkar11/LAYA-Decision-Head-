"""Diff-focused diagnostic evaluation v1 (user GO, chat 2026-10-13; protocol: LAYA_DECISION_BRAIN_PLAN_2026_10_13.md section 9). Kaggle CPU.

Inputs (mounted): kernel output laya-diff-capture (diff_features.pt), dataset laya-mutation-labels-v1 (labels.json: per state the passing option's
index), dataset laya-mutation-score-v2 (requests: option texts for the length baselines). Writes /kaggle/working/RESULT.json.
"""
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
N_PERM = 1000
find = lambda pat: sorted(glob.glob(os.path.join(IN, "**", pat), recursive=True))
PLAN = {
    "written": "2026-10-13, before this evaluation ran",
    "primary": "B_w8c: real-context span features (edit sites +- 8 tokens), centred by the pool mean; linear probe; leave-one-state-out (19 states = "
               "19 trajectories); C from %s by an inner grouped 3-fold CV on the 18 training states (highest mean top-1, ties -> smaller C)" % (CS,),
    "secondary": ["B_w8u (uncentred)", "B_w32c", "A_w8c (no context)", "B_wholec (existing mean-pooled features, same CV: control a)",
                  "lexical (bag of the option's own site token ids, no Qwen: control d)", "random 1/n, shortest, longest (control c)"],
    "permutation_null": "%d runs; in every pool the passing option is replaced by a uniformly random option (consistently for training and testing); "
                        "same CV but C fixed to the C chosen most often in the real primary run (no inner CV, for runtime); p = (1 + #runs with mean "
                        "top-1 >= observed) / (1 + runs)" % N_PERM,
    "criteria": {"PASS": "primary top-1 >= 0.50 over 19 states AND p < 0.05 AND primary - lexical >= 0.15",
                 "SHORTCUT": "first two hold, lexical margin fails (treated as FAIL)", "FAIL": "top-1 < 0.50 or p >= 0.05"},
    "reported_not_decisive": "the 7 states never in any scorer's training pools; no-context; secondary feature sets",
}


def vecs(rec, key, centre):
    x = rec[key].float().reshape(rec[key].shape[0], -1).numpy()
    return x - x.mean(0, keepdims=True) if centre else x


def run_cv(X, y_idx, states, C_fixed=None):
    """X: list per state of [n_opt, D]; y_idx: passing index per state. -> (per-state hit list, chosen C per fold)."""
    n, hits, chosen = len(X), [], []
    for k in range(n):
        tr = [i for i in range(n) if i != k]
        if C_fixed is None:
            score = {}
            for C in CS:
                h = []
                for a, b in GroupKFold(3).split(np.zeros(len(tr)), groups=np.array([states[i] for i in tr])):
                    h += fit_predict([X[tr[i]] for i in a], [y_idx[tr[i]] for i in a], [X[tr[i]] for i in b], C, [y_idx[tr[i]] for i in b])
                score[C] = np.mean(h)
            C = max(CS, key=lambda c: (score[c], -c))
        else:
            C = C_fixed
        chosen.append(C)
        hits += fit_predict([X[i] for i in tr], [y_idx[i] for i in tr], [X[k]], C, [y_idx[k]])
    return hits, chosen


def fit_predict(Xtr, ytr, Xte, C, yte):
    A = np.concatenate(Xtr)
    lab = np.concatenate([np.eye(len(x))[y] for x, y in zip(Xtr, ytr)])
    sc = StandardScaler().fit(A)
    m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(A), lab)
    return [float(int(np.argmax(m.decision_function(sc.transform(x)))) == y) for x, y in zip(Xte, yte)]


def main():
    t0 = time.time()
    feats = torch.load(find("diff_features.pt")[0], map_location="cpu", weights_only=False)
    lab = {x["decision_id"]: x for x in json.load(open(find("labels.json")[0]))}
    rq = find("requests_level1.jsonl*")[0]
    req = {r["decision_id"]: r for r in (json.loads(l) for l in (gzip.open if rq.endswith(".gz") else open)(rq, "rt", encoding="utf-8"))}
    B = {r["decision_id"]: r for r in feats["records"]["B"]}
    A = {r["decision_id"]: r for r in feats["records"]["A"]}
    ids = sorted(B)
    assert set(ids) == set(lab) == set(A) and len(ids) == 19, "state sets differ"
    for d in ids:
        assert B[d]["option_sha256"] == req[d]["option_sha256"] == A[d]["option_sha256"]
        assert B[d]["ho_w8"].shape[0] == lab[d]["n_options"]
    y = [lab[d]["fable_index"] for d in ids]
    states = [d.split(":")[1] for d in ids]
    unseen = [i for i, d in enumerate(ids) if not lab[d]["in_imitation_pools"]]
    sets = {"B_w8c": [vecs(B[d], "ho_w8", True) for d in ids], "B_w8u": [vecs(B[d], "ho_w8", False) for d in ids],
            "B_w32c": [vecs(B[d], "ho_w32", True) for d in ids], "A_w8c": [vecs(A[d], "ho_w8", True) for d in ids],
            "B_wholec": [vecs(B[d], "ho_whole", True) for d in ids]}
    dv = DictVectorizer(sparse=False).fit([{str(t): 1 for t in toks} for d in ids for toks in B[d]["site_token_ids"]])
    sets["lexical"] = [dv.transform([{str(t): toks.count(t) for t in toks} for toks in B[d]["site_token_ids"]]) for d in ids]
    res, chosen_primary = {}, None
    for name, X in sets.items():
        hits, chosen = run_cv(X, y, states)
        if name == "B_w8c":
            chosen_primary = chosen
        res[name] = dict(top1_all19=round(float(np.mean(hits)), 4), top1_unseen7=round(float(np.mean([hits[i] for i in unseen])), 4), per_state=dict(zip(ids, hits)),
                         C_chosen=chosen)
        print("RESULT", name, res[name]["top1_all19"], res[name]["top1_unseen7"], flush=True)
    lens = [[len(o) for o in req[d]["options"]] for d in ids]
    res["random"] = dict(top1_all19=round(float(np.mean([1 / lab[d]["n_options"] for d in ids])), 4),
                         top1_unseen7=round(float(np.mean([1 / lab[ids[i]]["n_options"] for i in unseen])), 4))
    for nm, f in (("shortest", np.argmin), ("longest", np.argmax)):
        h = [float(int(f(l)) == yy) for l, yy in zip(lens, y)]
        res[nm] = dict(top1_all19=round(float(np.mean(h)), 4), top1_unseen7=round(float(np.mean([h[i] for i in unseen])), 4))
    Cmode = max(set(chosen_primary), key=chosen_primary.count)
    rng = random.Random(20261013)
    perms = [[rng.randrange(lab[d]["n_options"]) for d in ids] for _ in range(N_PERM)]
    null = Parallel(n_jobs=-1)(delayed(lambda yp: float(np.mean(run_cv(sets["B_w8c"], yp, states, C_fixed=Cmode)[0])))(yp) for yp in perms)
    obs = res["B_w8c"]["top1_all19"]
    p = (1 + sum(v >= obs for v in null)) / (1 + N_PERM)
    lex_margin = round(obs - res["lexical"]["top1_all19"], 4)
    passed1, passed2, passed3 = obs >= 0.50, p < 0.05, lex_margin >= 0.15
    verdict = "PASS" if (passed1 and passed2 and passed3) else "SHORTCUT" if (passed1 and passed2) else "FAIL"
    out = dict(plan=PLAN, results=res, permutation=dict(runs=N_PERM, C_fixed=Cmode, null_mean=round(float(np.mean(null)), 4),
                                                         null_95th=round(float(np.percentile(null, 95)), 4), observed=obs, p=round(p, 4)),
               criteria=dict(top1_ge_050=passed1, p_lt_005=passed2, lexical_margin=lex_margin, lexical_margin_ge_015=passed3, verdict=verdict),
               seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(OUT, "RESULT.json"), "w"), indent=1)
    print("VERDICT", verdict, json.dumps(out["criteria"]), json.dumps(out["permutation"]), flush=True)


if __name__ == "__main__":
    main()
