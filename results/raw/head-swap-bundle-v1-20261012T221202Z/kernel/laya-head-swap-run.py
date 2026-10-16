"""Head-swap experiment (user GO for preparation, chat 2026-10-13; the Kaggle run is HELD until the user reviews this plan).
Does the pretrained LAYA head add anything a randomly initialised head with a trained adapter cannot? Every head is FROZEN; only the 8 adapter tensors
train. Qwen frozen, no encoder touched, original checkpoints never written, no checkpoint saved, nothing adopted, no CAL / TEST.

  Kaggle: no arguments (pack5.pt from laya-head-exp5-v1, heads.pt from laya-head-swap-v1)    smoke: HR_SMOKE=1 python head_swap_gpu.py <pack5 dir> <heads dir> <out>

Loss, sampling, scaler, scoring, probe and bootstrap come from kaggle/head_exp4_gpu.py (uploaded unchanged as exp4lib.py), whose training loop reproduced
the recorded reference. The adapter-only loop below mirrors decision_head/fable_train.py with warmup_steps = steps - 1 (phase A only), as
tools/laya_joint_v1.py ran it.
"""
import collections
import glob
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(glob.glob("/kaggle/input/**/pack5.pt", recursive=True)[0]).parent
HEADS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(glob.glob("/kaggle/input/**/heads.pt", recursive=True)[0]).parent
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/kaggle/working")
sys.argv = [sys.argv[0], str(SRC), str(OUT)]
sys.path.insert(0, str(SRC))
import exp4lib as L  # noqa: E402
from head import DecisionHead  # noqa: E402

SMOKE = L.SMOKE
SEEDS = (0,) if SMOKE else (0, 1, 2)
RANDOM_SEEDS = (1001, 1002, 1003)
CFG = dict(steps=10 if SMOKE else 300, eval_every=5 if SMOKE else 25, batch_size=8, lr_adapter=5e-4, weight_decay=0.01, grad_clip=1.0, lr_floor=0.1, unknown_weight=0.5)
PRETRAINED = ("coding", "typed", "code", "conductor")
REFERENCE = "coding"
PRIMARY = [("coding minus random (mean of 3)", "coding", "random_mean"), ("typed minus coding", "typed", "coding"), ("code minus coding", "code", "coding"),
           ("conductor minus coding", "conductor", "coding"), ("coding: real context minus no context (same pools)", "coding", "coding_nc"),
           ("probe minus coding", "probe", "coding")]
ALPHA = 0.05 / len(PRIMARY)
GATE = dict(oof_seed0=(0.7789, 0.04), val_3seed=(0.7609, 0.044))

PLAN = {
    "written": "2026-10-13, before any run; Kaggle launch held for the user's review",
    "approved_by": "user, chat 2026-10-13: GO preparation of the head-swap experiment; three random heads; 5-point rule with a multiplicity-adjusted interval; "
                   "no-context for every head; probe on identical folds; hold the Kaggle run until the registered plan is reviewed",
    "questions": {"Q1": "do pretrained heads beat randomly initialised heads after the same adapter training?", "Q2": "does the typed-decisions head beat the coding head?",
                  "Q3": "does removing the coding context reduce the reference head's accuracy?", "Q4": "does the reference system beat the standalone linear probe?"},
    "heads (all frozen; 31 tensors each: 24 context-layer, 6 scorer, 1 type embedding)": {
        "coding": "models/coding-decision-agent (REFERENCE, the head of every earlier result)", "typed": "models/laya-typed-decisions",
        "code": "models/laya-code", "conductor": "models/laya-conductor",
        "random_1001 / _1002 / _1003": "the same DecisionHead architecture built under torch seed 1001 / 1002 / 1003: PyTorch's default initialisation of every "
                                       "layer (the family a fresh head of this architecture starts from); per-tensor norms of every head are reported next to the result"},
    "not loaded from any checkpoint (reconciliation of 36 vs 31)": "act_head.0/2 weight and bias (LAYA's abstain/escalate gate; our head's own gate is frozen and "
                                                                    "unused) and the scalar `temperature`",
    "adapter training (identical for every head)": dict(CFG, seeds=list(SEEDS), init="DecisionHead built under torch seed = run seed, so every head gets the SAME "
                                                        "initial adapter for a given seed", schedule="cosine over all steps to 0.1 x, as the frozen trainer's adapter group",
                                                        checkpoint="lowest recorded-action loss on the development fold, every 25 steps", replay="none (the head is frozen)"),
    "reproduction gate (stop interpreting if it fails)": "coding head, seed 0, out of fold must be within 4 points of 77.89% (tools/laya_joint_v1.py adapter_only, "
                                                         "same 303 pools and folds); the 3-seed cleaned VAL mean within 4.4 points of 76.09%",
    "primary evaluation": "5-fold cross-validation on the 303 cleaned TRAIN pools (test fold k, development fold (k+1) mod 5, three folds fitted; "
                          "trajectory-disjoint); per pool the mean over 3 seeds (seeds are NOT independent examples); paired differences with 2,000 "
                          "bootstrap resamples of whole trajectories",
    "primary comparisons (family of %d; two-sided intervals at %.2f%%)" % (len(PRIMARY), 100 * (1 - ALPHA)): [p[0] for p in PRIMARY],
    "decision rule (development screening)": "a difference counts only if it is at least 5 points AND its corrected interval excludes 0. random_mean = per pool, "
                                             "the mean of the three random heads; the spread of the three is reported",
    "no-context": "every trained model is also scored with the no-context features on the 112 TRAIN pools that have them (test-time removal)",
    "VAL": "after the configuration is fixed: each head, adapter trained on folds 1-4, development fold 0, 3 seeds, scored ONCE on the 46 cleaned VAL pools with "
           "and without context. Development-exposed: a check, not confirmation; nothing is chosen on it",
    "zero-training swap (diagnostic)": "the three Fable-trained adapters of laya-baseline-v1 (trained for the coding head) loaded unchanged in front of every head; "
                                       "scored on cleaned VAL and the 600 native items only (those adapters saw every TRAIN pool). A failure shows "
                                       "incompatibility of that adapter with that head; it does not by itself show why",
    "native 600": "reported for every model, information only (the items are the coding head's own decision types)",
    "readings fixed beforehand": {
        "pretrained heads beat random by the rule": "pretraining contributes under this interface",
        "random within 5 points of coding": "no demonstrated advantage of these pretrained heads under this adapter and recipe; it does NOT show that a "
                                            "better-trained general head could never help",
        "typed beats coding by the rule": "general-decision specialisation transfers to this Fable task; a reason to revisit general-decision training",
        "probe beats coding": "the pretrained-head system adds nothing over a linear read-out on these features (expected from earlier results)",
        "any result": "agreement with Fable only; not coding-task success; one backbone"},
    "leakage safeguards": "folds trajectory-disjoint (asserted); VAL trajectories disjoint from TRAIN (asserted); no CAL / TEST in either dataset; each head's "
                          "31 tensors asserted identical before and after training; the zero-training adapters are never scored on TRAIN pools",
}


def head_state(name, heads):
    if name.startswith("random_"):
        with torch.random.fork_rng():
            torch.manual_seed(int(name.split("_")[1]))
            h = DecisionHead(**HCFG)
        return {k: v.clone() for k, v in h.state_dict().items() if L.group_of(k) == "pretrained"}
    return {k: v.float() for k, v in heads["heads"][name]["tensors"].items()}


def build(name, seed, heads, dev):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        h = DecisionHead(**HCFG)
    st = head_state(name, heads)
    r = h.load_state_dict(st, strict=False)
    assert len(st) == 31 and not r.unexpected_keys and not [k for k in r.missing_keys if L.group_of(k) == "pretrained"], (name, r)
    with torch.no_grad():
        h.kind_emb.weight.zero_()
    return h.to(dev)


def picks(h, rows, dev):
    return [int(s.argmax()) for s in L.scores(h, rows, dev)]


def as_nc(rows):
    return [dict(r, hq=r["hq_nc"], ho=r["ho_nc"]) for r in rows if "ho_nc" in r]


def run(name, seed, fit, devr, tests, nval, heads, dev):
    t0 = time.time()
    h = build(name, seed, heads, dev)
    L.fit_scaler(h, fit)
    named = list(h.named_parameters())
    for n, p in named:
        p.requires_grad = L.group_of(n) == "adapter"
    assert sum(p.requires_grad for _, p in named) == 8
    before = {n: p.detach().clone() for n, p in named if L.group_of(n) == "pretrained"}
    S, f = CFG["steps"], CFG["lr_floor"]
    opt = torch.optim.AdamW([p for _, p in named if p.requires_grad], lr=CFG["lr_adapter"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: f + (1 - f) * 0.5 * (1 + math.cos(math.pi * min(max(s, 0), S) / S)))
    w, g = L.sample_weights(fit), torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    best, nonfinite = None, 0
    for step in range(S):
        idx = torch.multinomial(w, CFG["batch_size"], replacement=True, generator=g).tolist()
        h.train()
        hq, ho, mask, pos, conf, _, prim = L.collate([fit[i] for i in idx], dev)
        loss = L.recorded_action_loss(h(hq, ho, mask, prim)[0], mask, pos, conf, CFG["unknown_weight"])[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for _, p in named if p.requires_grad], CFG["grad_clip"]))
        if math.isfinite(float(loss.detach())) and math.isfinite(gn):
            opt.step()
        else:
            nonfinite += 1
        sched.step()
        if (step + 1) % CFG["eval_every"] == 0 or step + 1 >= S:
            dl = L.row_loss(L.scores(h, devr, dev, bs=16), devr)
            if best is None or dl < best[1]:
                best = (step + 1, dl, {k: v.detach().clone() for k, v in h.state_dict().items()})
    h.load_state_dict(best[2])
    frozen_ok = all(torch.equal(before[n], p.detach()) for n, p in h.named_parameters() if n in before)
    hit = lambda rows: [float(p == r["ref"]) for p, r in zip(picks(h, rows, dev), rows)]
    res = dict(head=name, seed=seed, fit=len(fit), dev=len(devr), best_step=best[0], dev_loss=round(best[1], 4), nonfinite_steps=nonfinite, head_tensors_identical=frozen_ok,
               dev_top1=round(float(np.mean(hit(devr))), 4), fit_top1=round(float(np.mean(hit(fit))), 4), native=L.native_acc(h, nval, dev), seconds=None)
    for k, rs in tests.items():
        res[k + "_pick"] = picks(h, rs, dev)
    res["seconds"] = round(time.time() - t0, 1)
    with open(OUT / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    print("RUN %-12s s%d fit %3d best %3d | dev %.3f fit %.3f native %.3f | head frozen %s | %.0fs" % (
        name, seed, len(fit), best[0], res["dev_top1"], res["fit_top1"], res["native"], frozen_ok, res["seconds"]), flush=True)
    return res


def ci(x, groups, alpha, n=2000, seed=11):
    by = collections.defaultdict(list)
    for v, gg in zip(x, groups):
        by[gg].append(v)
    keys, rng, out = sorted(by), random.Random(seed), []
    for _ in range(n):
        pick = [by[k] for k in rng.choices(keys, k=len(keys))]
        out.append(sum(map(sum, pick)) / sum(map(len, pick)))
    out.sort()
    return [round(out[int(alpha / 2 * n)], 4), round(out[int((1 - alpha / 2) * n) - 1], 4)]


def main():
    global HCFG
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PLAN.json").write_text(json.dumps(PLAN, indent=1), "utf-8")
    print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU", flush=True)
    if not torch.cuda.is_available() and not SMOKE:
        raise SystemExit("STOP: no CUDA device. This experiment runs on the GPU only; nothing was trained.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = torch.load(SRC / "pack5.pt", map_location="cpu", weights_only=False)
    heads = torch.load(HEADS / "heads.pt", map_location="cpu", weights_only=False)
    HCFG = dict(pack["head_cfg"])
    rows, val, nval = pack["train"], pack["val"], pack["native_val"]
    if SMOKE:
        nval = nval[:40]
    val_nc = as_nc(val)
    assert len(val_nc) == len(val)
    fold = lambda ks, rs=None: [r for r in (rows if rs is None else rs) if r["fold"] in ks]
    ft_ = [{r["trajectory"] for r in fold({k})} for k in range(5)]
    assert all(not (ft_[a] & ft_[b]) for a in range(5) for b in range(a)) and not ({r["trajectory"] for r in val} & {r["trajectory"] for r in rows})
    ids, tg, vg = [r["decision_id"] for r in rows], {r["decision_id"]: r["trajectory"] for r in rows}, [r["trajectory"] for r in val]
    ref = {r["decision_id"]: r["ref"] for r in rows}
    names = list(PRETRAINED) + ["random_%d" % s for s in RANDOM_SEEDS]
    folds = (0,) if SMOKE else range(5)
    rep = dict(created_utc=datetime.now(timezone.utc).isoformat(), torch=torch.__version__, cuda_device=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, smoke=SMOKE,
               train_pools=len(rows), train_pools_with_no_context=sum("ho_nc" in r for r in rows), val_pools=len(val), native_val=len(nval), head_sources=heads["sources"])
    save = lambda: (OUT / "REPORT.json").write_text(json.dumps(rep, indent=1), "utf-8")
    norms = {}
    for nm in names:
        st = head_state(nm, heads)
        grp = lambda k: "context_layers" if k.startswith("ctx.") else "scorer" if k.startswith("mlp.") else "type_embedding"
        acc = collections.defaultdict(float)
        for k, v in st.items():
            acc[grp(k)] += float(v.float().norm()) ** 2
        norms[nm] = {k: round(math.sqrt(v), 2) for k, v in acc.items()}
    rep["head_tensor_norms"] = norms
    print("NORMS", json.dumps(norms), flush=True)

    # ---------------- zero-training swap (diagnostic): the Fable-trained adapters of the frozen baseline in front of every head
    zs = {}
    for nm in names:
        out = []
        for i, ad in enumerate(heads["baseline_adapters"]):
            h = build(nm, 0, heads, dev)
            r = h.load_state_dict({k: v.to(dev) for k, v in ad.items() if L.group_of(k) == "adapter" or k.startswith(("mu_", "sd_"))}, strict=False)
            assert not r.unexpected_keys
            vp = picks(h, val, dev)
            out.append(dict(val=round(float(np.mean([float(p == x["ref"]) for p, x in zip(vp, val)])), 4), native=L.native_acc(h, nval, dev)))
        zs[nm] = dict(each_adapter=out, val_mean=round(float(np.mean([o["val"] for o in out])), 4), native_mean=round(float(np.mean([o["native"] for o in out])), 4))
        print("ZEROSWAP %-12s VAL %.3f native %.3f" % (nm, zs[nm]["val_mean"], zs[nm]["native_mean"]), flush=True)
    rep["zero_training_swap"] = zs
    save()

    # ---------------- cross-validation
    cvr = collections.defaultdict(list)
    for nm in names:
        for k in folds:
            d = (k + 1) % 5
            te = fold({k})
            tests = dict(test=te, test_nc=as_nc(te))
            for s in SEEDS:
                r = run(nm, s, fold(set(range(5)) - {k, d}), fold({d}), tests, nval, heads, dev)
                cvr[nm].append(dict(r, test_ids=[x["decision_id"] for x in te], test_nc_ids=[x["decision_id"] for x in tests["test_nc"]]))
        if nm == REFERENCE:
            s0 = {i: float(p == ref[i]) for r in cvr[nm] if r["seed"] == 0 for i, p in zip(r["test_ids"], r["test_pick"])}
            got = float(np.mean(list(s0.values())))
            rep["reproduction_gate_oof_seed0"] = dict(now=round(got, 4), recorded=GATE["oof_seed0"][0], allowed=GATE["oof_seed0"][1],
                                                      passed=bool(abs(got - GATE["oof_seed0"][0]) <= GATE["oof_seed0"][1]))
            print("GATE oof seed 0", json.dumps(rep["reproduction_gate_oof_seed0"]), flush=True)
            save()

    def oof(runs, key="test"):
        acc = collections.defaultdict(list)
        for r in runs:
            for i, p in zip(r[key + "_ids"], r[key + "_pick"]):
                acc[i].append(float(p == ref[i]))
        return {i: float(np.mean(v)) for i, v in acc.items()}

    O = {nm: oof(cvr[nm]) for nm in names}
    Onc = {nm: oof(cvr[nm], "test_nc") for nm in names}
    rnd = ["random_%d" % s for s in RANDOM_SEEDS]
    O["random_mean"] = {i: float(np.mean([O[r][i] for r in rnd])) for i in ids}
    O["coding_nc"] = Onc[REFERENCE]
    if not SMOKE:
        P = {}
        for k in folds:
            te = fold({k})
            P.update({r["decision_id"]: float(int(np.argmax(s)) == r["ref"]) for r, s in zip(te, L.probe_scores(fold(set(range(5)) - {k}), te))})
        O["probe"] = P
    block = lambda d: dict(n=len(d), top1=round(float(np.mean([d[i] for i in ids if i in d])), 4),
                           ci95=ci([d[i] for i in ids if i in d], [tg[i] for i in ids if i in d], 0.05)) if d else dict(n=0)

    def diff(a, b):
        k = [i for i in ids if i in O[a] and i in O[b]]
        dd = [O[a][i] - O[b][i] for i in k]
        gg = [tg[i] for i in k]
        c, m = ci(dd, gg, ALPHA), float(np.mean(dd))
        return dict(n=len(k), delta=round(m, 4), ci_corrected=c, ci95=ci(dd, gg, 0.05), meets_rule=bool(abs(m) >= 0.05 and (c[0] > 0 or c[1] < 0)))

    rep["cross_validation"] = {}
    for nm in names + ["random_mean"]:
        rs = cvr.get(nm, [])
        rep["cross_validation"][nm] = dict(
            out_of_fold=block(O[nm]),
            each_seed=[round(float(np.mean([float(p == ref[i]) for r in rs if r["seed"] == s for i, p in zip(r["test_ids"], r["test_pick"])])), 4) for s in SEEDS] if rs else None,
            no_context_out_of_fold=block(Onc[nm]) if rs else None, same_pools_with_context=block({i: v for i, v in O[nm].items() if i in Onc[REFERENCE]}),
            fit_top1=round(float(np.mean([r["fit_top1"] for r in rs])), 4) if rs else None, native_mean=round(float(np.mean([r["native"] for r in rs])), 4) if rs else None,
            head_tensors_identical_all=all(r["head_tensors_identical"] for r in rs) if rs else None, best_steps=sorted(collections.Counter(r["best_step"] for r in rs).items()))
        c = rep["cross_validation"][nm]
        print("CV %-12s %.3f %s seeds %s | no context %s vs %s | fit %s native %s" % (nm, c["out_of_fold"]["top1"], c["out_of_fold"]["ci95"], c["each_seed"],
              (c["no_context_out_of_fold"] or {}).get("top1"), c["same_pools_with_context"].get("top1"), c["fit_top1"], c["native_mean"]), flush=True)
    rep["random_heads_spread"] = [rep["cross_validation"][r]["out_of_fold"]["top1"] for r in rnd]
    rep["primary"] = {lab: diff(a, b) for lab, a, b in PRIMARY if a in O and b in O}
    for k, v in rep["primary"].items():
        print("PRIMARY %-52s %+.4f corrected %s 95%% %s meets rule %s" % (k, v["delta"], v["ci_corrected"], v["ci95"], v["meets_rule"]), flush=True)
    if "probe" in O:
        rep["linear_probe_out_of_fold"] = block(O["probe"])
    save()

    # ---------------- VAL (configuration fixed): fit folds 1-4, development fold 0, 3 seeds, scored once
    vt = dict(val=val, val_nc=val_nc)
    vr = {nm: [run(nm, s, fold({1, 2, 3, 4}), fold({0}), vt, nval, heads, dev) for s in SEEDS] for nm in names}
    hv = lambda r, k: [float(p == x["ref"]) for p, x in zip(r[k + "_pick"], val)]
    loose = lambda r: [float(p == x["ref"] or p in x["loose"]) for p, x in zip(r["val_pick"], val)]
    pm = lambda rs, f: [float(np.mean([f(r)[i] for r in rs])) for i in range(len(val))]
    base = pm(vr[REFERENCE], lambda r: hv(r, "val"))
    rep["val"] = {}
    for nm, rs in vr.items():
        v, n = pm(rs, lambda r: hv(r, "val")), pm(rs, lambda r: hv(r, "val_nc"))
        rep["val"][nm] = dict(cleaned_val=L.summary(v, vg), each_seed=[round(float(np.mean(hv(r, "val"))), 4) for r in rs], no_context=L.summary(n, vg),
                              minus_coding=None if nm == REFERENCE else L.paired(base, v, vg), with_loose_equivalents=L.summary(pm(rs, loose), vg),
                              native_600=[r["native"] for r in rs], head_tensors_identical=all(r["head_tensors_identical"] for r in rs))
        c = rep["val"][nm]
        print("VAL %-12s %.3f %s seeds %s | no context %.3f | minus coding %s | native %s" % (nm, c["cleaned_val"]["top1"], c["cleaned_val"]["ci95"], c["each_seed"],
              c["no_context"]["top1"], None if c["minus_coding"] is None else (c["minus_coding"]["delta"], c["minus_coding"]["ci95"]), c["native_600"]), flush=True)
    got = rep["val"][REFERENCE]["cleaned_val"]["top1"]
    rep["reproduction_gate_val"] = dict(now=got, recorded=GATE["val_3seed"][0], allowed=GATE["val_3seed"][1], passed=bool(abs(got - GATE["val_3seed"][0]) <= GATE["val_3seed"][1]))
    if not SMOKE:
        pv = [float(int(np.argmax(s)) == r["ref"]) for r, s in zip(val, L.probe_scores(rows, val))]
        rep["linear_probe_val"] = L.summary(pv, vg)
    rep["gates_passed"] = bool(rep.get("reproduction_gate_oof_seed0", {}).get("passed") and rep["reproduction_gate_val"]["passed"])
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print("GATES", rep["gates_passed"], json.dumps(rep["reproduction_gate_val"]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
