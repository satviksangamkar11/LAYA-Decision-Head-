"""A5-v2 sampler + TRAIN-only grouped-CV audit. Spec: configs/a5_v2_spec.toml (registered before this file existed); thresholds: configs/thresholds.toml.
Data: compiler.firewall.load_dev (TRAIN rows only; test/ood/cal cannot be read here). No GPU. Output create-only.
    PYTHONPATH=. .venv/Scripts/python.exe tools/a5_v2_sampler.py
Pipeline: pool entry features -> per-fold per-stratum logistic 'true vs not-true' odds model fitted on the other folds -> Gumbel-top-k distractors with
weight odds^alpha for the held-out fold's decisions (out of fold) -> audit with two DIFFERENT attackers under trajectory-grouped CV, next to the A5-v1 sets."""
import collections, hashlib, json, re, tomllib, zlib
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from compiler.candidates import SAFETY, distractor_pool, first_call, render
from compiler.firewall import load_dev

ROOT = Path(__file__).resolve().parents[1]
T = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))
S = tomllib.load(open(ROOT / "configs/a5_v2_spec.toml", "rb"))["spec"]
gate, rules = T["benchmark_integrity_gate"], T["a5_v2_rules"]
KINDS = ["view", "edit", "run_tests", "explore", "run_other", "think", "plan", "finish"]
DENSE = ["pool_rank_norm", "pool_rank_cap", "history_frequency", *[f"kind_{k}" for k in KINDS], "text_length", "lexical_overlap_tail", "safety_exec",
         "safety_write", "in_last3_actions"]
ND, NH = len(DENSE), S["hash_dims"]
A_IDX = [1, 2, None, None, 11, 12, 13, 14]       # attacker A: rank cap, freq, kind ordinal, constant, length, overlap, exec, write
tok = lambda s: re.findall(r"[a-z0-9_]+", s.lower())

raw = {}
for l in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l)
    raw[r["trajectory_id"]] = r


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------- pool entry features for every covered TRAIN decision ----------
decs, excl = [], collections.Counter()
for r in load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"):
    if not r["candidate_coverage"]:
        excl["uncovered (diagnostic only)"] += 1
        continue
    tid, t = r["decision_id"].rsplit(":", 1)
    t = int(t)
    ev = raw[tid]["trajectory"]
    pool = distractor_pool(ev, t)
    texts = [p[0] for p in pool]
    ttext = r["candidates"][r["label_index"]]["text"]
    if ttext not in texts or len(pool) < 3:
        excl["true action not in pool / pool too small"] += 1
        continue
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
    decs.append(dict(did=r["decision_id"], tid=tid, stratum="repeat" if hist.get(ttext, 0) >= 1 else "first_time", X=X, ti=ti, k=k,
                     sh=r["state_hash"], pool_hash=hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16], v1=v1, v1_true=ttext))
for d in decs:
    d["v1_true_pos"] = None
res = dict(spec=S, covered_train_decisions=len(decs), exclusions=dict(excl), strata=dict(collections.Counter(d["stratum"] for d in decs)))

# ---------- folds by trajectory ----------
tids = sorted({d["tid"] for d in decs})
g = torch.Generator().manual_seed(S["cv_seed"])
fold_of = {tids[i]: n % S["cv_folds"] for n, i in enumerate(torch.randperm(len(tids), generator=g).tolist())}


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
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda Xq: lin((Xq - mu) / sd).squeeze(-1).detach()


v2 = {}
for f in range(S["cv_folds"]):
    for st in ("repeat", "first_time"):
        train_rows = [d for d in decs if fold_of[d["tid"]] != f and d["stratum"] == st]
        odds = fit_odds(train_rows)
        for d in decs:
            if fold_of[d["tid"]] != f or d["stratum"] != st:
                continue
            gen = torch.Generator().manual_seed(seed_of(d["sh"], S["sampler_version"]))
            cand = [i for i in range(len(d["X"])) if i != d["ti"]]
            lg = S["alpha"] * odds(d["X"][cand])
            gum = -torch.log(-torch.log(torch.rand(len(cand), generator=gen).clamp(1e-9, 1 - 1e-9)))
            top = (lg + gum).topk(d["k"]).indices.tolist()
            order = [d["ti"]] + [cand[j] for j in top]
            perm = torch.randperm(len(order), generator=gen).tolist()
            order = [order[j] for j in perm]
            v2[d["did"]] = dict(order=order, true_pos=order.index(d["ti"]), seed=seed_of(d["sh"], S["sampler_version"]), pool_hash=d["pool_hash"])
v1 = {}
for d in decs:
    order = list(d["v1"])
    v1[d["did"]] = dict(order=order, true_pos=order.index(d["ti"]) if d["ti"] in order else None)
decs_v1 = [d for d in decs if v1[d["did"]]["true_pos"] is not None]
KMAX = max(len(s["order"]) for s in v2.values())


# ---------- audit ----------
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
                v = [row[1], row[2], kord, 1.0, row[11], row[12], row[13], row[14]]
                X[a, j] = torch.tensor([float(x) for x in v])
            else:
                X[a, j, :ND + NH] = row
                X[a, j, -1] = len(s["order"]) / KMAX
            m[a, j] = True
        y[a] = s["true_pos"]
    return X, m, y


def attacker(ds, sets, which, seed_off=0):
    X, m, y = tensors(ds, sets, which)
    tid_f = {t: n % S["cv_folds"] for n, t in enumerate(sorted({d["tid"] for d in ds}))}
    g2 = torch.Generator().manual_seed(S["attacker_seed"])
    order = torch.randperm(len(tid_f), generator=g2).tolist()
    keys = sorted(tid_f)
    fa = {keys[i]: n % S["cv_folds"] for n, i in enumerate(order)}
    fold = torch.tensor([fa[d["tid"]] for d in ds])
    hit = torch.zeros(len(ds))
    hidden = S["attacker_hidden_A"] if which == "A" else S["attacker_hidden_B"]
    for f in range(S["cv_folds"]):
        tr, te = fold != f, fold == f
        torch.manual_seed(S["attacker_seed"] + f)
        net = (torch.nn.Sequential(torch.nn.Linear(X.shape[2], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)) if which == "A" else
               torch.nn.Sequential(torch.nn.Linear(X.shape[2], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(S["attacker_steps"]):
            s = net(X[tr]).squeeze(-1).masked_fill(~m[tr], float("-inf"))
            loss = F.cross_entropy(s, y[tr]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            hit[te] = (net(X[te]).squeeze(-1).masked_fill(~m[te], float("-inf")).argmax(1) == y[te]).float()
    rand = (1.0 / m.sum(1).float())
    out = {"overall": dict(acc=round(hit.mean().item(), 4), random=round(rand.mean().item(), 4), n=len(ds))}
    for st in ("repeat", "first_time"):
        sel = torch.tensor([d["stratum"] == st for d in ds])
        out[st] = dict(acc=round(hit[sel].mean().item(), 4), random=round(rand[sel].mean().item(), 4), n=int(sel.sum()))
    return out


def auc_by_feature(ds, sets, subset=None):
    res = {}
    for j, name in enumerate(DENSE):
        s = c = 0.0
        for d in ds:
            if subset and d["stratum"] != subset:
                continue
            st = sets[d["did"]]
            v = torch.stack([d["X"][i][j] for i in st["order"]])
            yv = v[st["true_pos"]]
            o = torch.cat([v[:st["true_pos"]], v[st["true_pos"] + 1:]])
            s += ((yv > o).float().sum() + 0.5 * (yv == o).float().sum()).item(); c += len(o)
        res[name] = round(s / c, 4) if c else None
    return res


def report(ds, sets, label):
    r = dict(set_size_mean=round(sum(len(sets[d["did"]]["order"]) for d in ds) / len(ds), 2),
             true_position_uniform_check=dict(collections.Counter(sets[d["did"]]["true_pos"] for d in ds).most_common(3)))
    r["attacker_A_8_features"] = attacker(ds, sets, "A")
    r["attacker_B_all_features_plus_hashed_tokens"] = attacker(ds, sets, "B")
    r["auc"] = {"overall": auc_by_feature(ds, sets), "repeat": auc_by_feature(ds, sets, "repeat"), "first_time": auc_by_feature(ds, sets, "first_time")}
    r["auc_violations_overall"] = [n for n, a in r["auc"]["overall"].items() if a is not None and abs(a - 0.5) > rules["auc_tolerance"] and not n.startswith("kind_")]
    ok = True
    for key in ("attacker_A_8_features", "attacker_B_all_features_plus_hashed_tokens"):
        for grp, v in r[key].items():
            ok &= v["acc"] <= v["random"] + gate["max_excess_over_random"]
    r["verdict"] = "PASS" if ok and not r["auc_violations_overall"] else "FAIL"
    r["label"] = label
    return r


res["retention"] = dict(considered_covered=len(decs), usable_v2=len(v2), retention=round(len(v2) / max(1, len(decs)), 4))
res["A5_v1_sets_same_audit"] = report(decs_v1, {d["did"]: v1[d["did"]] for d in decs_v1}, "A5-v1 candidates, same attackers, grouped CV")
res["A5_v2_sets"] = report(decs, v2, "A5-v2 candidates (out-of-fold generation)")
res["reproducibility_example"] = {k: v2[k] for k in list(v2)[:2]}
body = json.dumps(res, indent=1)
p = ROOT / ("results/raw/a5v2-train-cv-audit-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    f.write(body)
Path(str(p)[:-5] + ".sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "  " + p.name + "\n")
for k in ("A5_v1_sets_same_audit", "A5_v2_sets"):
    r = res[k]
    print(k, r["verdict"], "| A:", {g: (v["acc"], v["random"]) for g, v in r["attacker_A_8_features"].items()}, "| B:",
          {g: (v["acc"], v["random"]) for g, v in r["attacker_B_all_features_plus_hashed_tokens"].items()}, "| AUC violations:", r["auc_violations_overall"])
print(res["retention"], res["strata"], "->", p)
