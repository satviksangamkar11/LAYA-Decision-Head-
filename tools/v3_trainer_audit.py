"""Step-3 audit of the A5-v3 trainer (decision_head/v3_*.py). Structural gate, not an accuracy run: every check is a hard assertion; the audit fails if any one fails.
Parts: static/firewall checks, sampler set-0 equality with the registered sets, a synthetic world with a planted state signal (positive control) and without (negative control),
and the real smoke tensors (3 CAL captures: shapes/alignment/finiteness/forward only, NO training, NO accuracy; 10 TRAIN captures: full pipeline).
    PYTHONPATH=. .venv/Scripts/python.exe tools/v3_trainer_audit.py
Output create-only: results/raw/v3-trainer-audit-<time>.json (binds the result to the sha256 of the trainer sources)."""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import ast, hashlib, inspect, json, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from compiler import a5v2
from decision_head import v3_data, v3_model, v3_selection, v3_training as T

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ["v3_data", "v3_model", "v3_training", "v3_selection"]
EVAL_LOADED_BY_TRAINER = "decision_head.v3_evaluation" in sys.modules      # must be False: importing the trainer must not import the evaluator
report, ctx = [], {}


class Skip(Exception):
    """A check whose data does not exist yet (recorded as pending, never as passed)."""


def check(name):
    def deco(fn):
        try:
            detail = fn()
            report.append(dict(check=name, passed=True, detail=detail))
            print("PASS", name, "|", detail, flush=True)
        except Skip as e:
            report.append(dict(check=name, passed=None, detail="PENDING: " + str(e)))
            print("PENDING", name, "|", e, flush=True)
        except Exception as e:
            report.append(dict(check=name, passed=False, detail="".join(traceback.format_exception_only(type(e), e)).strip()[:600]))
            print("FAIL", name, "|", report[-1]["detail"], flush=True)
        return fn
    return deco


def expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}")


# ---------------------------------------------------------------- A. static / firewall
@check("A1 trainer modules have no path to the locked set")
def _():
    bad = []
    for m in TRAINER:
        src = (ROOT / f"decision_head/{m}.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr)
                and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                if any(s in n.value.lower() for s in ("locked_v3", "locked-v3", "fresh-locked", "locked_access")):
                    bad.append(f"{m}: string {n.value!r}")
            if isinstance(n, ast.ImportFrom) and n.module == "compiler.firewall" and not {a.name for a in n.names} <= {"load_dev", "load_cal_calibration_only"}:
                bad.append(f"{m}: imports {[a.name for a in n.names]} from the firewall")
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names] + [getattr(n, "module", "") or ""]
                if any("v3_evaluation" in x for x in names):
                    bad.append(f"{m}: imports v3_evaluation")
            if isinstance(n, ast.Name) and n.id in ("load_locked", "log_locked_access"):
                bad.append(f"{m}: uses {n.id}")
    assert not bad, bad
    assert not hasattr(v3_data, "load_locked") and not EVAL_LOADED_BY_TRAINER, "evaluator reachable from the trainer"
    return "no locked path/loader/log call, no evaluator import in " + ", ".join(TRAINER)


@check("A2 H0 is state-free by signature")
def _():
    p = list(inspect.signature(v3_model.H0Net.forward).parameters)
    assert p == ["self", "x", "te", "mask"], p
    return "H0Net.forward(x, te, mask): no query / contextual-candidate argument"


# ---------------------------------------------------------------- B. sampler sets (real TRAIN rows, firewall loader)
built = v3_data._built_train()
decs_all = list(built.values())
multi = v3_data.oof_sets_multi(decs_all)
ctx["built"], ctx["multi"] = built, multi


@check("B1 training set 0 equals the registered out-of-fold set (a5v2.oof_sets); sets are valid")
def _():
    reg = a5v2.oof_sets(decs_all)
    assert set(reg) == set(multi) == set(built)
    diff = [k for k in reg if multi[k][0] != reg[k]]
    assert not diff, diff[:3]
    nd = 0
    for k, ss in multi.items():
        d = built[k]
        assert len(ss) == v3_data.N_TRAIN_SETS
        for s in ss:
            assert s["order"][s["true_pos"]] == d["ti"] and len(set(s["order"])) == len(s["order"]) and s["pool_hash"] == d["pool_hash"]
        nd += any(ss[i]["order"] != ss[0]["order"] for i in (1, 2, 3))
    again = v3_data.oof_sets_multi(decs_all[:200])      # determinism on the same call path (subset: fold models differ, so only check the set-0 seed field)
    assert all(again[k][0]["seed"] == multi[k][0]["seed"] for k in again) and all(again[k][1]["seed"] == multi[k][1]["seed"] for k in again)
    assert all(multi[k][i]["seed"] == a5v2.seed_of(built[k]["sh"], a5v2.S["sampler_version"], "aug", i) for k in list(multi)[:200] for i in (1, 2, 3))
    return f"{len(reg)} decisions: set 0 identical to the registered sets; aug seeds = seed_of(state_hash, version, aug, i); {nd} decisions have at least one differing augmentation set"


# ---------------------------------------------------------------- C. synthetic world
def synth(n_traj, per, signal, seed, role="train", E=16, strength=3.0):
    g = torch.Generator().manual_seed(seed)
    TE = torch.randn(600, 3, E, generator=g).half()
    out = []
    for t in range(n_traj):
        for j in range(per):
            P = int(torch.randint(8, 13, (1,), generator=g))
            idx = torch.randperm(600, generator=g)[:P]
            ti = int(torch.randint(P, (1,), generator=g))
            X = (torch.rand(P, v3_data.NF, generator=g) < 0.1).float()
            X[:, 2] = torch.rand(P, generator=g) * 0.4
            X[ti, 2] += 0.08
            q = torch.randn(3, E, generator=g)
            cx = torch.randn(P, 3, E, generator=g)
            if signal:
                cx[ti] += strength * q
            sets = []
            for _ in range(4):
                k = int(torch.randint(4, 8, (1,), generator=g))
                others = [i for i in torch.randperm(P, generator=g).tolist() if i != ti][:k - 1]
                order = [ti] + others
                order = [order[i] for i in torch.randperm(len(order), generator=g).tolist()]
                sets.append(dict(order=order, true_pos=order.index(ti), seed=0, pool_hash="x"))
            out.append(dict(did=f"s{seed}t{t}:{j}", tid=f"s{seed}t{t}", stratum="repeat", role=role, X=X, ti=ti, texts=[f"txt{int(i)}" for i in idx], q=q.half(), cx=cx.half(),
                            te_idx=idx, TE=TE, sets=sets, pool_hash="x", state_hash="y"))
    return out


CFG = dict(T.CFG, max_epochs=40, min_epochs=5, patience=4, batch=16)
PROJ = 16
train_s, test_s = synth(60, 3, True, 0), synth(40, 3, True, 1)
bundle = T.h0_bundle(train_s, CFG, 0, PROJ)
sysm = T.fit_residual(bundle, CFG, "mean")


def acc(logits, decs):
    return sum(float(l.argmax().item() == d["sets"][0]["true_pos"]) for l, d in zip(logits, decs)) / len(decs)


@check("C1 positive control: planted state signal beats H0 on held-out synthetic decisions")
def _():
    h0, h2 = sysm.scores(test_s)
    a0, a2 = acc(h0, test_s), acc(h2, test_s)
    assert a2 > a0 + 0.05, (a0, a2)
    return f"H0 {a0:.3f} -> H2 {a2:.3f}"


@check("C2 negative control: no planted signal, the registered claim is NOT supported")
def _():
    from decision_head import v3_evaluation as E
    tr, te = synth(60, 3, False, 2), synth(150, 3, False, 3)
    s = T.fit_residual(T.h0_bundle(tr, CFG, 0, PROJ), CFG, "mean")
    h0, h2 = s.scores(te)
    delta = [float(b.argmax().item() == d["sets"][0]["true_pos"]) - float(a.argmax().item() == d["sets"][0]["true_pos"]) for a, b, d in zip(h0, h2, te)]
    m, lo, hi = E.bootstrap(delta, [d["tid"] for d in te])
    assert not (lo > 0 and m >= E.MIN_EFFECT), (m, lo, hi)
    return f"{len(te)} decisions: H2-H0 = {m:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] -> verdict {E.verdict(m, lo, hi)}"


@check("C3 H0 receives no contextual vector; H2 does")
def _():
    g = torch.Generator().manual_seed(5)
    noisy = [dict(d, q=torch.randn(d["q"].shape, generator=g).half(), cx=torch.randn(d["cx"].shape, generator=g).half()) for d in test_s]
    a0, a2 = sysm.scores(test_s)
    b0, b2 = sysm.scores(noisy)
    assert all(torch.equal(x, y) for x, y in zip(a0, b0)), "H0 changed when the contextual vectors were replaced"
    assert any(not torch.allclose(x, y) for x, y in zip(a2, b2)), "H2 ignored the contextual vectors"
    return "replacing every query and contextual candidate vector leaves H0 bit-identical and changes H2"


@check("C4 H2 = frozen H0 score + residual; a fresh residual starts exactly at H0")
def _():
    its = [(d, 0) for d in test_s[:8]]
    b = v3_data.make_batch(its)
    with torch.no_grad():
        h0 = sysm.h0(b["x"], b["te"], b["mask"])
        r = sysm.res(b["q"], b["cx"], b["mask"])
        h2 = v3_model.h2_logits(h0, r, b["mask"])
        fresh = v3_model.ResidualNet(b["q"].shape[-1], 3, PROJ, "learned", 0.1, 1).eval()
        f2 = v3_model.h2_logits(h0, fresh(b["q"], b["cx"], b["mask"]), b["mask"])
    s0, s2 = sysm.scores(test_s[:8])
    assert torch.allclose(torch.stack([x[:4] for x in s0]) if False else s0[0], h0[0, :len(s0[0])]) and torch.allclose(s2[0], h2[0, :len(s2[0])])
    assert torch.equal(f2, h0) or torch.allclose(f2.nan_to_num(-1e9), h0.nan_to_num(-1e9)), "fresh residual is not neutral"
    return "H2 logits == H0 logits + residual; zero-initialised key projection makes H2 start at H0"


@check("C5 candidate order: a permutation only permutes the scores")
def _():
    worst = 0.0
    for d in test_s[:20] + train_s[:5]:
        o = d["sets"][0]["order"]
        perm = torch.randperm(len(o), generator=torch.Generator().manual_seed(len(o))).tolist()
        d2 = dict(d, sets=[dict(order=[o[i] for i in perm], true_pos=perm.index(d["sets"][0]["true_pos"]), seed=0, pool_hash="x")])
        for a, b in zip(sysm.scores([d]), sysm.scores([d2])):
            worst = max(worst, (a[0][perm] - b[0]).abs().max().item())
    assert worst < 1e-5, worst
    return f"max |score(permuted) - permuted(score)| = {worst:.2e} for H0 and H2"


@check("C6 same rows, sets, seed, schedule, scaler policy; OOF H0 scores used for the residual")
def _():
    r = sysm.record
    assert r["h0"]["fit_keys_hash"] == r["residual"]["fit_keys_hash"] and r["h0"]["first_epoch_order_hash"] == r["residual"]["first_epoch_order_hash"]
    assert r["h0"]["seed"] == r["residual"]["seed"] == 0 and r["h0"]["schedule"] == r["residual"]["schedule"]
    st = T.scaler_stats(bundle["a"])
    assert torch.allclose(sysm.h0.mu_f, st["mu_f"]) and torch.allclose(sysm.h0.sd_t, st["sd_t"]) and torch.allclose(sysm.res.mu_c, st["mu_c"]) and torch.allclose(sysm.res.sd_q, st["sd_q"])
    assert r["residual_trained_on_oof_h0_scores"] and len(bundle["folds"]) == a5v2.S["cv_folds"]
    for fo in bundle["folds"]:
        assert not (fo["trained_on"] & fo["scored"])
    its = T.items_of(bundle["a"])[:40]
    final = T.predict(T._h0_fn(sysm.h0), its)
    diff = max((f - bundle["h0s"][(d["did"], k)]).abs().max().item() for f, (d, k) in zip(final, its))
    assert diff > 1e-4, "cross-fitted scores equal the final H0's (not out of fold)"
    covered = {k[0] for k in bundle["h0s"]}
    assert covered >= {d["did"] for d in bundle["a"] + bundle["v"]}
    return f"folds disjoint by trajectory; OOF scores differ from final-H0 scores (max diff {diff:.3f}); identical rows/shuffle/seed/schedule; one scaler fit"


@check("C7 CAL and LOCKED rows cannot enter a gradient step")
def _():
    cal = synth(12, 2, True, 9, role="cal")
    a, v = T.inner_split(cal, 0.1)
    expect_raises(AssertionError, lambda: T.fit_h0(a, v, CFG, 0, PROJ))
    expect_raises(AssertionError, lambda: T.h0_bundle(cal, CFG, 0, PROJ))
    loc = synth(12, 2, True, 9, role="locked")
    a, v = T.inner_split(loc, 0.1)
    expect_raises(AssertionError, lambda: T.fit_h0(a, v, CFG, 0, PROJ))
    return "train_loop refused role=cal and role=locked rows"


@check("C8 training is reproducible for a fixed seed")
def _():
    b2 = T.h0_bundle(train_s, CFG, 0, PROJ)
    s2 = T.fit_residual(b2, CFG, "mean")
    for k, v in sysm.h0.state_dict().items():
        assert torch.equal(v, s2.h0.state_dict()[k]), k
    for k, v in sysm.res.state_dict().items():
        assert torch.equal(v, s2.res.state_dict()[k]), k
    return "second fit with the same seed gives bit-identical H0 and residual weights"


@check("C9 temperature, bootstrap, tie-break and nested subsets")
def _():
    from decision_head import v3_evaluation as E
    g = torch.Generator().manual_seed(0)
    ys = torch.randint(5, (400,), generator=g).tolist()
    lg = [torch.randn(5, generator=g) * 0.0 + F.one_hot(torch.tensor(y), 5).float() * 1.0 * 3 + torch.randn(5, generator=g) * 3 for y in ys]
    t = T.fit_temperature(lg, ys)
    assert t > 1.0 and all(l.argmax() == (l / t).argmax() for l in lg), t
    m, lo, hi = E.bootstrap([1.0] * 50 + [0.0] * 50, [f"t{i // 2}" for i in range(100)])
    assert abs(m - 0.5) < 1e-9 and lo < 0.5 < hi
    assert E.verdict(0.08, 0.02, 0.14).startswith("INCONCLUSIVE") and E.verdict(0.08, 0.05, 0.11) == "SUCCESS" and E.verdict(0.02, -0.01, 0.05) == "NOT SUPPORTED"
    sc = {(128, "single"): 1.00, (128, "mean"): 0.995, (256, "single"): 0.99, (256, "mean"): 0.985, (128, "learned"): 1.2, (256, "learned"): 1.3, (512, "single"): 1.4, (512, "mean"): 1.5, (512, "learned"): 1.6}
    assert v3_selection.choose(sc) == (128, "mean"), v3_selection.choose(sc)         # within 0.01 of the best 0.985: (128,mean) .995, (256,single) .99, (256,mean) -> smallest projection wins; (128,single)=1.00 is outside
    assert v3_selection.choose({(128, "mean"): 0.990, (128, "single"): 0.995, (256, "learned"): 1.2}) == (128, "single")   # same projection, tie -> simpler depth treatment
    subs = T.nested_subsets(train_s, [30, 60, 120])
    assert [len(v) for v in subs.values()] == [30, 60, 120] and subs[30] == subs[60][:30] and subs[60] == subs[120][:60]
    return f"temperature {t:.2f}; bootstrap/verdict rules; tie-break; nested subsets"


@check("C10 NONE is captured but can never be scored")
def _():
    from capture.pool import NONE_TEXT
    d = synth(1, 1, True, 11)[0]
    d["texts"] = d["texts"][:-1] + [NONE_TEXT]
    ok = dict(d)
    d["sets"] = [dict(order=[len(d["texts"]) - 1, d["ti"]] if d["ti"] != len(d["texts"]) - 1 else [d["ti"], 0], true_pos=1, seed=0, pool_hash="x")]
    expect_raises(AssertionError, lambda: v3_data.make_batch([(d, 0)]))
    return "a candidate set containing the NONE option is refused by make_batch (assemble also rejects NONE in the pool and any set index beyond the sampled pool)"


@check("C11 early-stopping split is locked: 10%, trajectory-grouped, deterministic, TRAIN only, set 0")
def _():
    a, v = T.inner_split(train_s, 0.10)
    a2, v2 = T.inner_split(list(reversed(train_s)), 0.10)
    assert {d["did"] for d in v} == {d["did"] for d in v2} and {d["did"] for d in a} == {d["did"] for d in a2}, "split depends on input order"
    ta, tv = {d["tid"] for d in a}, {d["tid"] for d in v}
    assert not (ta & tv) and len(tv) == round(0.10 * len(ta | tv)), (len(ta), len(tv))
    expect_raises(AssertionError, lambda: T.inner_split(train_s, 0.2))
    its = T.items_of(a)
    expect_raises(AssertionError, lambda: T.train_loop(torch.nn.Linear(1, 1), lambda b: None, its, [(v[0], 1)], CFG, 0))
    cal = synth(10, 1, True, 4, role="cal")
    expect_raises(AssertionError, lambda: T.train_loop(torch.nn.Linear(1, 1), lambda b: None, its, [(cal[0], 0)], CFG, 0))
    return f"{len(tv)} validation trajectories of {len(ta | tv)}; order-independent; disjoint; 0.2 refused; non-TRAIN or non-set-0 validation refused"


# ---------------------------------------------------------------- D. real smoke tensors
ids_all = Path(ROOT / "data/v3/smoke_ids.txt").read_text().strip().split(",")
ids_train, ids_cal = ids_all[:10], ids_all[10:]
TE, te_idx = v3_data.load_textemb(ROOT / "data/v3/textemb_smoke_real")


@check("D1 real CAL smoke captures: shapes, hashes, text alignment, finiteness (forward only, no training, no accuracy)")
def _():
    cal, miss = v3_data.load_cal(ids_cal, ROOT / "data/v3/capture_smoke/cal", ROOT / "data/v3/textemb_smoke_real")
    assert len(cal) == 3 and not miss, (len(cal), miss)
    for d in cal:
        assert torch.isfinite(d["q"].float()).all() and torch.isfinite(d["cx"].float()).all() and torch.isfinite(d["TE"][d["te_idx"]].float()).all()
        assert d["role"] == "cal" and len(d["sets"]) == 1
        P = len(d["texts"])
        assert d["cx"].shape == (P, 3, 2560) and d["q"].shape == (3, 2560) and d["te_idx"].shape == (P,)
    stf = [F.cosine_similarity(d["TE"][d["te_idx"]].float().flatten(0, 1), d["cx"].float().flatten(0, 1), dim=-1) for d in cal]
    frac_diff = float(torch.cat(stf).lt(0.9999).float().mean())
    assert frac_diff > 0.99, frac_diff
    h0 = v3_model.H0Net(v3_data.NF, 2560, 3, PROJ, 0.1).eval()
    b = v3_data.make_batch([(d, 0) for d in cal])
    with torch.no_grad():
        s = h0(b["x"], b["te"], b["mask"])
    assert torch.isfinite(s[b["mask"]]).all() and torch.isinf(s[~b["mask"]]).all()
    bad = dict(cal[0].items())
    swapped = {k: v for k, v in v3_data.load_captures(ROOT / "data/v3/capture_smoke/cal").items()}
    c0 = swapped[cal[0]["did"]]
    c0 = dict(c0, texts=[c0["texts"][1], c0["texts"][0]] + c0["texts"][2:])
    swapped[cal[0]["did"]] = c0
    sets = {d["did"]: d["sets"] for d in cal}
    cal_built = {}
    raw = a5v2.load_raw(v3_data.RAW_OLD)
    from compiler.firewall import load_cal_calibration_only
    for r in load_cal_calibration_only(v3_data.DECISIONS):
        if r["decision_id"] in set(ids_cal):
            cal_built[r["decision_id"]] = a5v2.build_decision(r, raw[r["decision_id"].rsplit(":", 1)[0]]["trajectory"])[0]
    expect_raises(AssertionError, lambda: v3_data.assemble(ids_cal, cal_built, swapped, TE, te_idx, "cal", sets))
    return f"3 decisions aligned and finite; state-free embeddings differ from the contextual vectors for {frac_diff:.1%} of candidate-depth pairs; a swapped candidate text is rejected"


@check("D2 real TRAIN smoke captures: full pipeline (cross-fit, H0, residual) with the structural assertions on real tensors")
def _():
    if not any((ROOT / "data/v3/capture_smoke/train").glob("*.pt")):
        raise Skip("no real TRAIN smoke captures yet (the independence gate stopped the smoke capture; waiting on the user's decision)")
    caps = v3_data.load_captures(ROOT / "data/v3/capture_smoke/train")
    tr, miss = v3_data.assemble(ids_train, built, caps, TE, te_idx, "train", multi)
    assert len(tr) >= 8, (len(tr), miss)
    cfg = dict(T.CFG, max_epochs=6, min_epochs=2, patience=2, batch=8)
    s = T.fit_system(tr, cfg, 0, 16, "learned")
    rec = s.record
    assert rec["residual_trained_on_oof_h0_scores"] and rec["h0"]["fit_keys_hash"] == rec["residual"]["fit_keys_hash"]
    g = torch.Generator().manual_seed(1)
    noisy = [dict(d, q=torch.randn(d["q"].shape, generator=g).half(), cx=torch.randn(d["cx"].shape, generator=g).half()) for d in tr]
    a0, a2 = s.scores(tr)
    b0, b2 = s.scores(noisy)
    assert all(torch.equal(x, y) for x, y in zip(a0, b0)) and any(not torch.allclose(x, y) for x, y in zip(a2, b2))
    for d in tr[:4]:
        o = d["sets"][0]["order"]
        perm = list(reversed(range(len(o))))
        d2 = dict(d, sets=[dict(order=[o[i] for i in perm], true_pos=perm.index(d["sets"][0]["true_pos"]), seed=0, pool_hash="x")])
        for x, y in zip(s.scores([d]), s.scores([d2])):
            assert (x[0][perm] - y[0]).abs().max().item() < 1e-4
    return f"{len(tr)} real TRAIN decisions: pipeline ran ({rec['h0']['epochs']} H0 epochs, {rec['residual']['epochs']} residual epochs); blind-H0, state-using H2 and permutation checks hold on real tensors (no accuracy is read)"


# ---------------------------------------------------------------- result
srcs = {m: hashlib.sha256((ROOT / f"decision_head/{m}.py").read_bytes()).hexdigest() for m in TRAINER + ["v3_evaluation"]}
out = dict(created=datetime.now(timezone.utc).isoformat(), passed=all(r["passed"] for r in report if r["passed"] is not None) and not any(r["passed"] is None for r in report),
           complete=not any(r["passed"] is None for r in report), pending=[r["check"] for r in report if r["passed"] is None], n_checks=len(report), checks=report, trainer_sha256=srcs,
           interpretation_notes=["early stopping monitors NLL on an inner trajectory-grouped 10% split of the training rows (not in the spec)",
                                 "the scored candidate sets exclude the NONE option (it is captured; the registered sampler never contains it)",
                                 "H2_secondary (H0 and H2 trained from scratch together) is not implemented in this step"])
p = ROOT / ("results/raw/v3-trainer-audit-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=str)
print("AUDIT", "PASS" if out["passed"] else ("FAIL" if any(r["passed"] is False for r in report) else "INCOMPLETE"), f"{sum(r['passed'] is True for r in report)}/{len(report)} passed, pending {out['pending']}", "->", p)
