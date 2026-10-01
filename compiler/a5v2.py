"""A5-v2 sampler components (frozen sampler a5v2-odds-1). Mirrors tools/a5_v2_sampler.py (the registered audit) exactly, as importable functions.
Spec: configs/a5_v2_spec.toml. Nothing here reads test/ood/locked data; callers pass the rows they are allowed to use.
Frozen artifact: fit_odds_all() fits ONE odds model per stratum on all old train covered rows; save_frozen() stores it with a hash."""
import collections
import hashlib
import json
import re
import tomllib
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler.candidates import SAFETY, distractor_pool, first_call, render

ROOT = Path(__file__).resolve().parents[1]
S = tomllib.load(open(ROOT / "configs/a5_v2_spec.toml", "rb"))["spec"]
KINDS = ["view", "edit", "run_tests", "explore", "run_other", "think", "plan", "finish"]
DENSE = ["pool_rank_norm", "pool_rank_cap", "history_frequency", *[f"kind_{k}" for k in KINDS], "text_length", "lexical_overlap_tail", "safety_exec",
         "safety_write", "in_last3_actions"]
ND, NH, KMAX = len(DENSE), S["hash_dims"], 14
tok = lambda s: re.findall(r"[a-z0-9_]+", s.lower())


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def load_raw(path):
    raw = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        raw[r["trajectory_id"]] = r
    return raw


def build_decision(r, ev):
    """(decision dict, None) for a usable covered decision, else (None, reason). Same features and the same k rule as the audit."""
    tid, t = r["decision_id"].rsplit(":", 1)
    t = int(t)
    pool = distractor_pool(ev, t)
    texts = [p[0] for p in pool]
    ttext = r["candidates"][r["label_index"]]["text"]
    if ttext not in texts or len(pool) < 3:
        return None, "true action not in pool / pool too small"
    acts = [render(*c)[0] for e in ev[1:t] if e["role"] == "assistant" and (c := first_call(e))]
    hist = collections.Counter(acts)
    last3 = set(acts[-3:])
    tail = set(tok(" ".join((e["content"] or "") for e in ev[max(1, t - 4):t])))
    P = len(pool)
    X = torch.zeros(P, ND + NH)
    for i, (x, kind, _, _) in enumerate(pool):
        tk = tok(x)
        X[i, 0], X[i, 1], X[i, 2] = i / P, min(i, 40) / 40.0, min(hist.get(x, 0), 6) / 6.0
        X[i, 3 + KINDS.index(kind)] = 1.0
        X[i, 11], X[i, 12] = min(len(x), 200) / 200.0, (len(set(tk) & tail) / len(set(tk))) if tk else 0.0
        X[i, 13], X[i, 14], X[i, 15] = float(SAFETY[kind] == "EXEC"), float(SAFETY[kind] == "WRITE_LOCAL"), float(x in last3)
        for w in tk:
            X[i, ND + zlib.crc32(w.encode()) % NH] = 1.0
    ti = texts.index(ttext)
    k = min(2 + int(r["state_hash"][:16], 16) % 12, P - 1)
    v1_texts = [c["text"] for c in r["candidates"] if c["id"] != "NONE"]
    v1 = [texts.index(x) for x in v1_texts if x in texts]
    return dict(did=r["decision_id"], tid=tid, stratum="repeat" if hist.get(ttext, 0) >= 1 else "first_time", X=X, ti=ti, k=k, sh=r["state_hash"],
                pool_hash=hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16], v1=v1), None


def fit_odds(rows):
    X = torch.cat([d["X"] for d in rows])
    y = torch.zeros(len(X))
    o = 0
    for d in rows:
        y[o + d["ti"]] = 1.0
        o += len(d["X"])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    torch.manual_seed(0)
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    Z = (X - mu) / sd
    for _ in range(S["logit_steps"]):
        loss = F.binary_cross_entropy_with_logits(lin(Z).squeeze(-1), y) + S["logit_l2"] * lin.weight.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return dict(w=lin.weight.detach().clone(), b=lin.bias.detach().clone(), mu=mu, sd=sd)


def odds_fn(m):
    return lambda Xq: (((Xq - m["mu"]) / m["sd"]) @ m["w"].T + m["b"]).squeeze(-1)


def generate(d, m):
    """One candidate set for decision d from odds model m (Gumbel-top-k, weight odds^alpha). Deterministic in (state_hash, sampler_version)."""
    odds = odds_fn(m)
    gen = torch.Generator().manual_seed(seed_of(d["sh"], S["sampler_version"]))
    cand = [i for i in range(len(d["X"])) if i != d["ti"]]
    lg = S["alpha"] * odds(d["X"][cand])
    gum = -torch.log(-torch.log(torch.rand(len(cand), generator=gen).clamp(1e-9, 1 - 1e-9)))
    top = (lg + gum).topk(d["k"]).indices.tolist()
    order = [d["ti"]] + [cand[j] for j in top]
    perm = torch.randperm(len(order), generator=gen).tolist()
    order = [order[j] for j in perm]
    return dict(order=order, true_pos=order.index(d["ti"]), seed=seed_of(d["sh"], S["sampler_version"]), pool_hash=d["pool_hash"])


def v1_set(d):
    order = list(d["v1"])
    return dict(order=order, true_pos=order.index(d["ti"]) if d["ti"] in order else None)


def folds_by_trajectory(decs):
    tids = sorted({d["tid"] for d in decs})
    g = torch.Generator().manual_seed(S["cv_seed"])
    return {tids[i]: n % S["cv_folds"] for n, i in enumerate(torch.randperm(len(tids), generator=g).tolist())}


def oof_sets(decs):
    """Out-of-fold sets for old train decisions, exactly as in the registered audit."""
    fold_of = folds_by_trajectory(decs)
    out = {}
    for f in range(S["cv_folds"]):
        for st in ("repeat", "first_time"):
            m = fit_odds([d for d in decs if fold_of[d["tid"]] != f and d["stratum"] == st])
            for d in decs:
                if fold_of[d["tid"]] == f and d["stratum"] == st:
                    out[d["did"]] = generate(d, m)
    return out


def fit_frozen(decs):
    return {st: fit_odds([d for d in decs if d["stratum"] == st]) for st in ("repeat", "first_time")}


def save_frozen(models, path):
    torch.save(models, path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- attackers (different family from the sampler) ----------
def tensors(ds, sets, which):
    n = len(ds)
    feats = ND + NH if which == "B" else 8
    X = torch.zeros(n, KMAX, feats + (1 if which == "B" else 0))
    m = torch.zeros(n, KMAX, dtype=torch.bool)
    y = torch.zeros(n, dtype=torch.long)
    for a, d in enumerate(ds):
        s = sets[d["did"]]
        for j, i in enumerate(s["order"]):
            row = d["X"][i]
            if which == "A":
                kord = float(row[3:11].argmax().item()) / 7.0
                X[a, j] = torch.tensor([float(x) for x in [row[1], row[2], kord, 1.0, row[11], row[12], row[13], row[14]]])
            else:
                X[a, j, :ND + NH] = row
                X[a, j, -1] = len(s["order"]) / KMAX
            m[a, j] = True
        y[a] = s["true_pos"]
    return X, m, y


def fit_attacker(ds, sets, which):
    X, m, y = tensors(ds, sets, which)
    hidden = S["attacker_hidden_A"] if which == "A" else S["attacker_hidden_B"]
    torch.manual_seed(S["attacker_seed"])
    net = (torch.nn.Sequential(torch.nn.Linear(X.shape[2], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)) if which == "A" else
           torch.nn.Sequential(torch.nn.Linear(X.shape[2], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(S["attacker_steps"]):
        s = net(X).squeeze(-1).masked_fill(~m, float("-inf"))
        loss = F.cross_entropy(s, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def score_attacker(net, ds, sets, which):
    X, m, y = tensors(ds, sets, which)
    hit = (net(X).squeeze(-1).masked_fill(~m, float("-inf")).argmax(1) == y).float()
    rand = 1.0 / m.sum(1).float()
    out = {"overall": dict(acc=round(hit.mean().item(), 4), random=round(rand.mean().item(), 4), n=len(ds))}
    for st in ("repeat", "first_time"):
        sel = torch.tensor([d["stratum"] == st for d in ds])
        out[st] = dict(acc=round(hit[sel].mean().item(), 4) if sel.any() else None, random=round(rand[sel].mean().item(), 4) if sel.any() else None, n=int(sel.sum()))
    return out


def auc_by_feature(ds, sets):
    res = {}
    for j, name in enumerate(DENSE):
        s = c = 0.0
        for d in ds:
            st = sets[d["did"]]
            v = torch.stack([d["X"][i][j] for i in st["order"]])
            yv = v[st["true_pos"]]
            o = torch.cat([v[:st["true_pos"]], v[st["true_pos"] + 1:]])
            s += ((yv > o).float().sum() + 0.5 * (yv == o).float().sum()).item()
            c += len(o)
        res[name] = round(s / c, 4) if c else None
    return res
