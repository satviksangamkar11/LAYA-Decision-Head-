"""Context diagnostic v1 (pre-registration thresholds_context_diag_v1.toml): the EXISTING pipeline on the Fable TRAIN pools, with and without context.
Scorers: Qwen alone (answer-letter logits, 2 option orders), LAYA coding head + 3 frozen adapters, probe (out-of-fold), options-only (out-of-fold),
the loop rules of thresholds_loop_eng_v2.toml + one 3-scorer average. No training, no new candidates, no CAL/TEST. Kaggle T4 x2.
MODE smoke: 40 pools (20 with stored no-context features, 20 without); exercises every code path incl. the report; accuracy is NOT read. MODE full: all pools.
"""
import collections
import glob
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODE = "full"           # smoke | full
VERSION = "cd1-2026-10-15a"
PREREG_NAME, PREREG_SHA256 = "thresholds_context_diag_v1.toml", "291c9b5da053b777db7ff14b049faf6c21694a1bb05f758d7c7d9fa8a3262b37"
OUT, WORK = Path("/kaggle/working/out"), Path("/kaggle/working/tmp")
GRID, SEED_B, B = [0.10, 0.20, 0.30], 20261015, 2000
REPRO = [0.7826, 0.8043, 0.7609]
LET = "ABCDEFGH"
T0 = time.time()
SELF = os.path.abspath(globals()["__file__"]) if globals().get("__file__") else None


def log(*a):
    print("[%7.0fs]" % (time.time() - T0), *a, flush=True)


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


def qwen_dir(revision):
    q = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True)
         if os.path.exists(os.path.join(os.path.dirname(c), "model.safetensors.index.json"))]
    if q:
        return q[0]
    from huggingface_hub import snapshot_download
    return snapshot_download("Qwen/Qwen3-4B-Thinking-2507", revision=revision, local_dir="/tmp/qwen")


def no_gqa_patch():
    if torch.cuda.get_device_capability(0)[0] < 8:   # T4: SDPA with enable_gqa falls to the math kernel and OOMs on long prompts (see capture_resident.py)
        import transformers.integrations.sdpa_attention as S
        S.use_gqa_in_sdpa = lambda *a, **k: False


# ------------------------------------------------------------------------------------------------------------------------------------
# Qwen answer-letter scoring (worker process, one per GPU)
# ------------------------------------------------------------------------------------------------------------------------------------
def build_prompt(cond, context, options, order):
    head = ("Task and recent history:\n" + context.strip() + "\n\nOptions:\n") if cond == "ctx" else "Candidate next actions for a coding task.\n\nOptions:\n"
    return head + "".join("[%s] %s\n" % (LET[p], options[ci]) for p, ci in enumerate(order)) + "\nWhich option is the best next action? Answer with one letter.\nAnswer: ["


def letters_worker():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    jobs = json.load(open(WORK / "jobs.json"))
    shard, nshard = int(os.environ["CD_SHARD"]), int(os.environ["CD_NSHARD"])
    qd = json.load(open(WORK / "meta_qwen.json"))["dir"]
    no_gqa_patch()
    tok = AutoTokenizer.from_pretrained(qd)
    model = AutoModelForCausalLM.from_pretrained(qd, dtype=torch.float16).to("cuda").eval()
    let_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LET]
    mine = [j for i, j in enumerate(sorted(jobs, key=lambda j: -len(j["context"]))) if i % nshard == shard]
    res, t = [], time.time()
    for k, j in enumerate(mine):
        n = len(j["options"])
        for cond in ("ctx", "nc"):
            for oname, order in (("orig", list(range(n))), ("rev", list(range(n))[::-1])):
                ids = tok(build_prompt(cond, j["context"], j["options"], order), return_tensors="pt", add_special_tokens=False)["input_ids"]
                if ids.shape[1] > 24000:
                    res.append(dict(id=j["id"], cond=cond, order=oname, probs=None, tokens=int(ids.shape[1])))
                    continue
                with torch.inference_mode():
                    lg = model(input_ids=ids.to("cuda"), use_cache=False, logits_to_keep=1).logits[0, -1].float()
                p = torch.softmax(lg[let_ids[:n]].double(), 0).cpu().numpy()
                by = [0.0] * n
                for pos, ci in enumerate(order):
                    by[ci] = float(p[pos])
                res.append(dict(id=j["id"], cond=cond, order=oname, probs=by, by_position=[float(x) for x in p], tokens=int(ids.shape[1]),
                                finite=bool(np.isfinite(p).all())))
        if (k + 1) % 10 == 0:
            log("worker", shard, "pools", k + 1, "of", len(mine), "seconds", round(time.time() - t))
    json.dump(res, open(WORK / ("letters_%d.json" % shard), "w"))
    log("worker", shard, "done")


def phaseA_worker():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(one("level12_config.json")))
    PROTO, LIM, DEP = cfg["protocol"], cfg["limits"], cfg["protocol"]["depths"]
    sys.path.insert(0, "/kaggle/working/pkg")
    from capture.capture_resident import ResidentCapturer
    jobs = json.load(open(WORK / "jobs.json"))
    qd = json.load(open(WORK / "meta_qwen.json"))["dir"]
    tok = AutoTokenizer.from_pretrained(qd)
    model = AutoModelForCausalLM.from_pretrained(qd, dtype=torch.float16).eval()
    model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:max(DEP)])
    C = ResidentCapturer(model.to("cuda"), DEP)
    pre = tok.encode(cfg["neutral_prefix"] + PROTO["query_cue"], add_special_tokens=False)
    A = {}
    for j in jobs:
        suf = [tok.encode(PROTO["option_prefix"] + o, add_special_tokens=False) for o in j["options"]]
        if len(suf) > LIM["max_options"] or any(len(s) > LIM["max_option_tokens"] or not s for s in suf):
            raise SystemExit("option over limit in no-context capture: " + j["id"])
        cap, _ = C.packed(dict(prefix_ids=pre, suffix_ids=suf, query_pos=len(pre) - 1))
        A[j["id"]] = (torch.stack([cap[d][0] for d in DEP]).half(), torch.stack([cap[d][1] for d in DEP], 1).half())
    torch.save(A, WORK / "A.pt")
    log("phase A worker done", len(A))


# ------------------------------------------------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
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
    hcfg, rows, val = dict(pack["head_cfg"]), pack["train"], pack["val"]
    val_nc = [dict(r, hq=r["hq_nc"], ho=r["ho_nc"]) for r in val]

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
    adapters = []
    for ad in heads["baseline_adapters"]:
        h = build()
        r = h.load_state_dict({k: v.to(dev) for k, v in ad.items() if L.group_of(k) == "adapter" or k.startswith(("mu_", "sd_"))}, strict=False)
        assert not r.unexpected_keys
        adapters.append(h)
    # gate 1: the loaded adapters reproduce the recorded VAL numbers
    each = [round(float(np.mean([float(int(s.argmax()) == x["ref"]) for s, x in zip(L.scores(h, val, dev), val)])), 4) for h in adapters]
    if each != REPRO:
        stop("adapter reproduction gate failed", got=each, expected=REPRO)
    log("adapter reproduction gate passed", each)

    # pools: pack5 TRAIN rows joined with the text pools (ids and labels must agree)
    pf = one("pools.jsonl*")
    raw = gzip.open(pf, "rt", encoding="utf-8").read() if str(pf).endswith(".gz") else open(pf, encoding="utf-8").read()
    tp = {p["pool_id"]: p for p in (json.loads(l) for l in raw.splitlines() if l.strip()) if p["split"] == "train"}
    # the pack is the CLEANED v4 view (strict equivalents removed): apply the order-preserving option map (pool id -> kept v3 option indices, new label index)
    omap = json.load(open(one("v4_option_map.json")))
    for i, m in omap.items():
        if i in tp:
            p0 = tp[i]
            tp[i] = dict(p0, options=[p0["options"][j] for j in m["keep"]], label_index=m["ref"])
    sel = [r for r in rows if r["decision_id"] in tp]
    assert len(sel) == len(rows) == 303, (len(sel), len(rows))
    for r in sel:
        p = tp[r["decision_id"]]
        assert p["label_index"] == r["ref"] and len(p["options"]) == r["ho"].shape[0], r["decision_id"]
    if MODE == "smoke":
        have = [r for r in sel if "hq_nc" in r][:20]
        lack = [r for r in sel if "hq_nc" not in r][:20]
        sel = have + lack
    N = len(sel)
    ids = [r["decision_id"] for r in sel]
    ref, traj, fold = [r["ref"] for r in sel], [r["trajectory"] for r in sel], np.array([r["fold"] for r in sel])
    ks = [len(tp[i]["options"]) for i in ids]
    log("pools", N, "trajectories", len(set(traj)), "with stored no-context features", sum("hq_nc" in r for r in sel))
    meta = dict(version=VERSION, mode=MODE, prereg_sha256=PREREG_SHA256, n_pools=N, n_trajectories=len(set(traj)),
                pools_with_stored_nc=sum("hq_nc" in r for r in sel), adapter_reproduction=each)

    # ---- phase A (own process, frees the GPU on exit): recapture the no-context features (variant A) for ALL pools; parity vs stored
    cfg = json.load(open(one("level12_config.json")))
    PROTO, LIM, DEP = cfg["protocol"], cfg["limits"], cfg["protocol"]["depths"]
    pkg = "/kaggle/working/pkg/capture"
    os.makedirs(pkg, exist_ok=True)
    for n in ("capture.py", "capture_resident.py"):
        shutil.copy(one(n), pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    qd = qwen_dir(PROTO["revision"])
    json.dump(dict(dir=qd), open(WORK / "meta_qwen.json", "w"))
    jobs = [dict(id=r["decision_id"], context=tp[r["decision_id"]]["context_text"], options=tp[r["decision_id"]]["options"]) for r in sel]
    json.dump(jobs, open(WORK / "jobs.json", "w"))
    if not SELF:
        stop("script path unknown")
    t = time.time()
    pa = subprocess.run([sys.executable, SELF], env=dict(os.environ, CD_STAGE="phaseA", CUDA_VISIBLE_DEVICES="0"))
    if pa.returncode != 0:
        stop("phase A (no-context capture) failed", code=pa.returncode)
    A = torch.load(WORK / "A.pt", map_location="cpu")
    meta["phaseA_seconds"] = round(time.time() - t)
    cos = []
    for r in sel:
        if "hq_nc" in r:
            a, b = A[r["decision_id"]], (r["hq_nc"], r["ho_nc"])
            assert a[0].shape == b[0].shape and a[1].shape == b[1].shape, (a[0].shape, b[0].shape, a[1].shape, b[1].shape)
            cos.append(min(torch.nn.functional.cosine_similarity(a[0].double().flatten(), b[0].double().flatten(), 0).item(),
                           torch.nn.functional.cosine_similarity(a[1].double().flatten(), b[1].double().flatten(), 0).item()))
    meta["nc_parity_cosine_min"], meta["nc_parity_pools"] = (min(cos) if cos else None), len(cos)
    log("phase A done; parity", meta["nc_parity_cosine_min"], "on", len(cos), "pools")
    if cos and min(cos) < 0.999:
        json.dump(meta, open(OUT / "META.json", "w"), indent=1, default=str)
        stop("no-context recapture parity failed", cosine_min=min(cos))
    sel_nc = [dict(r, hq=A[r["decision_id"]][0], ho=A[r["decision_id"]][1]) for r in sel]

    # ---- phase B: Qwen letter workers (one per GPU), CPU scoring runs meanwhile
    ng = max(1, torch.cuda.device_count())
    procs = [subprocess.Popen([sys.executable, SELF], env=dict(os.environ, CD_STAGE="letters", CD_SHARD=str(g), CD_NSHARD=str(ng), CUDA_VISIBLE_DEVICES=str(g)))
             for g in range(ng)]

    # ---- CPU: LAYA, probe, options-only
    P = {"ctx": {}, "nc": {}}
    for cond, rs in (("ctx", sel), ("nc", sel_nc)):
        for si, h in enumerate(adapters):
            P[cond]["laya_s%d" % si] = [softmax(s.double().numpy()) for s in L.scores(h, rs, dev)]
        P[cond]["laya_ens"] = [np.mean([P[cond]["laya_s%d" % si][i] for si in range(3)], 0) for i in range(N)]
    log("LAYA scored")
    folds = sorted(set(fold.tolist()))
    P["ctx"]["probe_oof"], P["nc"]["probe_oof"] = [None] * N, [None] * N
    for k in folds:
        te_ix = [i for i in range(N) if fold[i] == k]
        fit = [sel[i] for i in range(N) if fold[i] != k]
        if not te_ix:
            continue
        if len(fit) < 10:
            stop("too few pools for out-of-fold probe", fold=int(k), n_fit=len(fit))
        sc = L.probe_scores(fit, [sel[i] for i in te_ix] + [sel_nc[i] for i in te_ix])   # one call: the same fitted probe on real and no-context rows
        for j, i in enumerate(te_ix):
            P["ctx"]["probe_oof"][i], P["nc"]["probe_oof"][i] = softmax(sc[j]), softmax(sc[len(te_ix) + j])
        log("probe OOF fold", int(k), "done")
    P["ctx"]["probe_in"] = [softmax(s) for s in L.probe_scores(sel, sel)]
    # options-only (text), out-of-fold with the same folds
    from sklearn.feature_extraction.text import TfidfVectorizer
    pool_of = {i: tp[i] for i in ids}

    def clogit(Xs, ys, Cc):
        Pn, d = len(Xs), Xs[0].shape[1]
        K = max(x.shape[0] for x in Xs)
        X, M = torch.zeros(Pn, K, d, dtype=torch.float64), torch.zeros(Pn, K, dtype=torch.bool)
        for i, x in enumerate(Xs):
            X[i, :x.shape[0]], M[i, :x.shape[0]] = x, True
        y, w = torch.tensor(ys), torch.zeros(d, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1, max_iter=300, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def cl():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy((X @ w).masked_fill(~M, -1e30), y) + (w @ w) / (2 * Cc * Pn)
            loss.backward()
            return loss
        opt.step(cl)
        return w.detach()

    def feats(fit_p, apply_p):
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)).fit([o for p in fit_p for o in p["options"]])
        f = lambda p: torch.from_numpy(v.transform(p["options"]).toarray().astype("float64"))
        return [f(p) for p in fit_p], [f(p) for p in apply_p]
    P["ctx"]["options_only_oof"] = [None] * N
    for k in folds:
        te_ix = [i for i in range(N) if fold[i] == k]
        fit_ix = [i for i in range(N) if fold[i] != k]
        if not te_ix:
            continue
        fp, ap = [pool_of[ids[i]] for i in fit_ix], [pool_of[ids[i]] for i in te_ix]
        ut_in = sorted({traj[i] for i in fit_ix})
        inner = {t: j % 3 for j, t in enumerate(random.Random(SEED_B).sample(ut_in, len(ut_in)))}
        ll = collections.defaultdict(list)
        for g in range(3):
            a_ix = [i for i in fit_ix if inner[traj[i]] != g]
            b_ix = [i for i in fit_ix if inner[traj[i]] == g]
            if not a_ix or not b_ix:
                continue
            Xa, Xb = feats([pool_of[ids[i]] for i in a_ix], [pool_of[ids[i]] for i in b_ix])
            for Cc in (0.1, 1.0, 10.0):
                w = clogit(Xa, [ref[i] for i in a_ix], Cc)
                for x, i in zip(Xb, b_ix):
                    s = (x @ w).numpy()
                    ll[Cc].append(float(np.log(np.exp(s - s.max()).sum()) + s.max() - s[ref[i]]))
        Cb = min(ll, key=lambda c: np.mean(ll[c])) if ll else 1.0
        Xa, Xt = feats(fp, ap)
        w = clogit(Xa, [ref[i] for i in fit_ix], Cb)
        for x, i in zip(Xt, te_ix):
            P["ctx"]["options_only_oof"][i] = softmax((x @ w).numpy())
        log("options-only OOF fold", int(k), "done C", Cb)
    P["nc"]["options_only_oof"] = P["ctx"]["options_only_oof"]
    for p in procs:
        p.wait()
    if any(p.returncode != 0 for p in procs):
        json.dump(meta, open(OUT / "META.json", "w"), indent=1, default=str)
        stop("a Qwen letter worker failed", codes=[p.returncode for p in procs])
    lres = [r for g in range(ng) for r in json.load(open(WORK / ("letters_%d.json" % g)))]
    by = collections.defaultdict(dict)
    for r in lres:
        by[(r["id"], r["cond"])][r["order"]] = r
    pos_bias = {c: collections.defaultdict(list) for c in ("ctx", "nc")}
    for cond in ("ctx", "nc"):
        P[cond]["qwen"] = []
        for i, pid in enumerate(ids):
            o = by[(pid, cond)]
            if any(o[x]["probs"] is None for x in ("orig", "rev")):
                stop("Qwen prompt over the token limit", pool=pid, tokens=[o[x]["tokens"] for x in o])
            P[cond]["qwen"].append(np.mean([np.array(o["orig"]["probs"]), np.array(o["rev"]["probs"])], 0))
            for x in ("orig", "rev"):
                for p_, v in enumerate(o[x]["by_position"]):
                    pos_bias[cond][p_].append(v)
        meta["qwen_mean_prob_by_letter_position_" + cond] = {LET[p_]: round(float(np.mean(v)), 4) for p_, v in sorted(pos_bias[cond].items())}
    meta["qwen_nonfinite"] = sum(1 for r in lres if r["probs"] is not None and not r.get("finite", True))
    meta["qwen_prompt_tokens"] = dict(max=max(r["tokens"] for r in lres), mean=round(float(np.mean([r["tokens"] for r in lres])), 1))
    log("Qwen letters collected", len(lres))
    if MODE == "smoke":
        meta["note"] = "smoke: mechanics only, no accuracy computed"
        meta["probs_sum_to_one"] = all(abs(float(np.sum(P[c]["qwen"][i])) - 1.0) < 1e-6 for c in P for i in range(N))
        meta["probs_shapes_ok"] = all(len(P[c][n][i]) == ks[i] for c in P for n in P[c] for i in range(N))
        report(meta, P, ids, ref, traj, fold, ks, L, "SMOKE_REPORT_not_for_reading.json")
        json.dump(meta, open(OUT / "SMOKE_MECHANICS.json", "w"), indent=1, default=str)
        log("SMOKE DONE", json.dumps({k: v for k, v in meta.items() if k not in ("qwen_mean_prob_by_letter_position_ctx", "qwen_mean_prob_by_letter_position_nc")}, default=str)[:1500])
        return
    report(meta, P, ids, ref, traj, fold, ks, L, "REPORT.json")


# ------------------------------------------------------------------------------------------------------------------------------------
def report(meta, P, ids, ref, traj, fold, ks, L, fname):
    N = len(ids)
    ut = sorted(set(traj))
    gi = np.array([ut.index(t) for t in traj])
    rng = np.random.default_rng(SEED_B)
    W = np.zeros((B, len(ut)))
    np.add.at(W, (np.repeat(np.arange(B), len(ut)), rng.integers(0, len(ut), (B, len(ut))).ravel()), 1.0)
    cn = np.bincount(gi, minlength=len(ut)).astype(float)

    def boot(d, mask=None):
        m = np.ones(N) if mask is None else np.asarray(mask, float)
        sm, c = np.bincount(gi, weights=np.asarray(d, float) * m, minlength=len(ut)), np.bincount(gi, weights=m, minlength=len(ut))
        num, den = W @ sm, W @ c
        b = num[den > 0] / den[den > 0]
        return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]

    def boot_diff(h, mask):   # mean(h on ~mask) - mean(h on mask)
        a, b_ = np.bincount(gi, weights=h * (1 - mask), minlength=len(ut)), np.bincount(gi, weights=(1 - mask), minlength=len(ut))
        c, d_ = np.bincount(gi, weights=h * mask, minlength=len(ut)), np.bincount(gi, weights=mask, minlength=len(ut))
        n1, d1, n2, d2 = W @ a, W @ b_, W @ c, W @ d_
        ok = (d1 > 0) & (d2 > 0)
        v = n1[ok] / d1[ok] - n2[ok] / d2[ok]
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    am = lambda ps: [int(np.argmax(p)) for p in ps]
    hits = lambda picks: np.array([float(int(k) == r) for k, r in zip(picks, ref)])
    marg = lambda p: float(np.sort(p)[::-1][0] - np.sort(p)[::-1][1])

    def rule_avg(ps):
        return [int(np.argmax(np.mean([p[i] for p in ps], 0))) for i in range(N)]

    def rule_first(a, b, t):
        return [int(np.argmax(a[i])) if marg(a[i]) >= t else int(np.argmax(b[i])) for i in range(N)]

    def rules_for(cond):
        S = P[cond]
        out = {}
        seeds = [am(S["laya_s%d" % k]) for k in range(3)]
        vote = []
        for i in range(N):
            c = collections.Counter(s[i] for s in seeds).most_common()
            vote.append(c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else int(np.argmax(S["laya_ens"][i])))
        out["S_vote"] = vote
        out["S_unanimous_else_probe"] = [seeds[0][i] if len({s[i] for s in seeds}) == 1 else int(np.argmax(S["probe_oof"][i])) for i in range(N)]
        out["C_avg"] = rule_avg([S["laya_ens"], S["probe_oof"]])
        for t in GRID:
            out["C_laya_first_t%.2f" % t] = rule_first(S["laya_ens"], S["probe_oof"], t)
            out["C_probe_first_t%.2f" % t] = rule_first(S["probe_oof"], S["laya_ens"], t)
        out["C_avg3_laya_probe_qwen"] = rule_avg([S["laya_ens"], S["probe_oof"], S["qwen"]])
        out["control_C_avg_options_only_probe"] = rule_avg([P["ctx"]["options_only_oof"], S["probe_oof"]])
        return out
    picks = {c: {n: am(v) for n, v in P[c].items()} for c in ("ctx", "nc")}
    picks_rules = {c: rules_for(c) for c in ("ctx", "nc")}
    H = {c: {**{n: hits(p) for n, p in picks[c].items()}, **{n: hits(p) for n, p in picks_rules[c].items()}} for c in ("ctx", "nc")}
    rep = dict(meta=meta, chance=round(float(np.mean([1 / k for k in ks])), 4))
    rep["accuracy"] = {c: {n: dict(correct=int(h.sum()), acc=round(float(h.mean()), 4), ci95=boot(h)) for n, h in H[c].items()} for c in H}

    def vs(h, base):
        wr, rw = int(((h == 1) & (base == 0)).sum()), int(((h == 0) & (base == 1)).sum())
        return dict(wrong_to_right=wr, right_to_wrong=rw, net_pools=wr - rw, net_gain=round((wr - rw) / N, 4), ci95=boot(h - base))
    rep["context_gain"] = {}
    for n in H["ctx"]:
        if n in H["nc"] and n not in ("options_only_oof",):
            a, b = H["ctx"][n], H["nc"][n]
            pk = picks["ctx"].get(n, picks_rules["ctx"].get(n)), picks["nc"].get(n, picks_rules["nc"].get(n))
            rep["context_gain"][n] = dict(correct_with=int(a.sum()), correct_without=int(b.sum()), gain_pools=int(a.sum() - b.sum()), gain=round(float((a - b).mean()), 4),
                                         ci95=boot(a - b), wrong_to_right_with_context=int(((a == 1) & (b == 0)).sum()), right_to_wrong_with_context=int(((a == 0) & (b == 1)).sum()),
                                         pick_changed=int(sum(x != y for x, y in zip(*pk))))
    rep["loop_vs_standalone"] = {c: {rn: {sn: vs(H[c][rn], H[c][sn]) for sn in ("qwen", "laya_ens", "probe_oof", "options_only_oof")} for rn in picks_rules[c]} for c in H}
    mask0 = (fold == 0).astype(float)
    h = H["ctx"]["laya_ens"]
    rep["memorisation_indicator_laya_ens_ctx"] = dict(acc_folds_1_to_4=round(float(h[mask0 == 0].mean()), 4), acc_fold_0=round(float(h[mask0 == 1].mean()), 4),
                                                      n_fold0=int(mask0.sum()), diff=round(float(h[mask0 == 0].mean() - h[mask0 == 1].mean()), 4), diff_ci95=boot_diff(h, mask0))
    rep["probe_in_sample_vs_oof_ctx"] = dict(in_sample=rep["accuracy"]["ctx"]["probe_in"]["acc"], out_of_fold=rep["accuracy"]["ctx"]["probe_oof"]["acc"])
    rep["per_pool"] = dict(id=ids, trajectory=traj, fold=fold.tolist(), ref=ref, k=ks,
                           picks={c: {**picks[c], **picks_rules[c]} for c in picks},
                           qwen_prob={c: [[round(float(x), 5) for x in p] for p in P[c]["qwen"]] for c in P},
                           laya_ens_prob={c: [[round(float(x), 5) for x in p] for p in P[c]["laya_ens"]] for c in P})
    (OUT / fname).write_text(json.dumps(rep, indent=1))
    if fname != "REPORT.json":
        log("report code path executed (accuracy not printed in smoke)")
        return
    log("FULL DONE", json.dumps({c: {n: v["acc"] for n, v in rep["accuracy"][c].items() if n in ("qwen", "laya_ens", "probe_oof", "options_only_oof", "C_avg", "C_avg3_laya_probe_qwen")} for c in rep["accuracy"]}))


if __name__ == "__main__":
    if os.environ.get("CD_STAGE") == "phaseA":
        phaseA_worker()
    elif os.environ.get("CD_STAGE") == "letters":
        letters_worker()
    else:
        main()
