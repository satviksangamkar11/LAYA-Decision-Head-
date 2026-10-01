"""A5-v3 trainer: H0 (state-free), cross-fitted H0 scores, and the H2 residual trained on those out-of-fold scores. TRAIN rows only: train_loop asserts it.
Spec: configs/a5_v3_spec.toml [amendment_2026_10_02_c] fixed_training / H2_primary. Interpretation notes (not in the spec, flagged to the user):
 * early stopping monitors NLL on an inner trajectory-grouped 10% validation split of the training rows (set 0 only); CAL is never used for it.
 * the residual and the final H0 share one fit/validation split and one scaler fit, so the two arms differ in nothing but the state branch."""
import copy
import hashlib
import math

import torch
import torch.nn.functional as F

from compiler import a5v2
from decision_head.v3_data import DEPTHS, NF, make_batch
from decision_head.v3_model import EMBED_SD_FLOOR, FEATURE_SD_FLOOR, H0Net, ResidualNet, h2_logits

CFG = dict(lr=3e-4, weight_decay=1e-2, dropout=0.10, clip=1.0, batch=32, max_epochs=100, min_epochs=10, patience=8, val_frac=0.10)
SINGLE_DEPTH = 24


def _sd(var, floor):
    return var.clamp_min(0).sqrt().clamp_min(floor)


def items_of(decs):
    return [(d, k) for d in decs for k in range(len(d["sets"]))]


VAL_FRAC = 0.10        # LOCKED by configs/a5_v3_spec.toml [amendment_2026_10_02_d] early_stopping_lock


def inner_split(decs, frac):
    """Trajectory-grouped fit / validation split that depends only on the trajectory ids (identical for every arm and seed). The fraction is locked at 10%."""
    assert frac == VAL_FRAC and CFG["val_frac"] == VAL_FRAC, "the early-stopping split is locked at 10% of the trajectories"
    tids = sorted({d["tid"] for d in decs}, key=lambda t: hashlib.sha256(("val|" + t).encode()).hexdigest())
    nval = max(1, round(frac * len(tids))) if len(tids) > 1 else 0
    val = set(tids[:nval])
    return [d for d in decs if d["tid"] not in val], [d for d in decs if d["tid"] in val]


def scaler_stats(decs, chunk=64):
    """Mean/sd over the candidates of all sets of the fit decisions (features, state-free embeddings, contextual candidates) and over the decisions' query vectors."""
    its, acc = items_of(decs), {}
    for i in range(0, len(its), chunk):
        b = make_batch(its[i:i + chunk])
        m = b["mask"]
        for key, t in (("f", b["x"][m]), ("t", b["te"][m]), ("c", b["cx"][m])):
            a = acc.setdefault(key, [0, 0.0, 0.0])
            a[0] += len(t)
            a[1] = a[1] + t.double().sum(0)
            a[2] = a[2] + (t.double() ** 2).sum(0)
    q = torch.stack([d["q"].float() for d in decs]).double()
    st = {}
    for key, name, floor in (("f", "f", FEATURE_SD_FLOOR), ("t", "t", EMBED_SD_FLOOR), ("c", "c", EMBED_SD_FLOOR)):
        n, s1, s2 = acc[key]
        mu = s1 / n
        st["mu_" + name], st["sd_" + name] = mu.float(), _sd(s2 / n - mu ** 2, floor).float()
    st["mu_q"], st["sd_q"] = q.mean(0).float(), q.std(0).clamp_min(EMBED_SD_FLOOR).float() if len(q) > 1 else torch.ones_like(q[0]).float()
    return st


@torch.no_grad()
def predict(fn, items, h0s=None, chunk=64, device="cpu"):
    """Per-item logit vectors (unpadded, in set order). fn(batch) -> [B, K] logits."""
    out = []
    for i in range(0, len(items), chunk):
        sub = items[i:i + chunk]
        b = {k: v.to(device) for k, v in make_batch(sub, h0s).items()}
        lg = fn(b).cpu()
        out += [lg[j, :len(d["sets"][k]["order"])] for j, (d, k) in enumerate(sub)]
    return out


def _nll(fn, items, h0s, device):
    lg = predict(fn, items, h0s, device=device)
    return sum(-F.log_softmax(l, 0)[d["sets"][k]["true_pos"]].item() for l, (d, k) in zip(lg, items)) / len(items)


def train_loop(net, fn, fit_items, val_items, cfg, seed, device="cpu", h0s=None):
    """AdamW, listwise cross-entropy only, gradient clip, early stopping on validation NLL. Returns a record whose schedule fields are compared across arms."""
    assert fit_items and val_items, "need fit and validation items"
    assert all(k == 0 for _, k in val_items), "early stopping monitors set 0 only"
    assert all(d["role"] == "train" for d, _ in fit_items + val_items), "only TRAIN rows may enter a gradient step or early stopping (CAL/LOCKED never)"
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, bad, best_state, order_hash, ep = math.inf, 0, None, None, 0
    for ep in range(cfg["max_epochs"]):
        net.train()
        perm = torch.randperm(len(fit_items), generator=gen).tolist()
        if ep == 0:
            order_hash = hashlib.sha256(json_ids(fit_items, perm).encode()).hexdigest()[:16]
        for i in range(0, len(perm), cfg["batch"]):
            b = {k: v.to(device) for k, v in make_batch([fit_items[j] for j in perm[i:i + cfg["batch"]]], h0s).items()}
            loss = F.cross_entropy(fn(b), b["y"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
            opt.step()
        net.eval()
        v = _nll(fn, val_items, h0s, device)
        if v < best - 1e-9:
            best, bad, best_state = v, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if ep + 1 >= cfg["min_epochs"] and bad >= cfg["patience"]:
            break
    net.load_state_dict(best_state)
    net.eval()
    sched = {k: cfg[k] for k in ("lr", "weight_decay", "dropout", "clip", "batch", "max_epochs", "min_epochs", "patience")}
    return dict(epochs=ep + 1, best_val_nll=best, seed=seed, first_epoch_order_hash=order_hash, schedule=sched,
                fit_keys_hash=hashlib.sha256(json_ids(fit_items, range(len(fit_items))).encode()).hexdigest()[:16])


def json_ids(items, order):
    return "|".join(f"{items[j][0]['did']}#{items[j][1]}" for j in order)


def _h0_fn(net):
    def fn(b):
        return net(b["x"], b["te"], b["mask"])           # the ONLY inputs H0 ever sees: features and state-free embeddings
    return fn


def _res_fn(res):
    def fn(b):
        return h2_logits(b["h0"], res(b["q"], b["cx"], b["mask"]), b["mask"])
    return fn


def fit_h0(fit_decs, val_decs, cfg, seed, proj, device="cpu"):
    st = scaler_stats(fit_decs)
    torch.manual_seed(seed)                                  # the initial weights must depend on the seed, not on whatever ran before
    net = H0Net(NF, fit_decs[0]["q"].shape[-1], len(DEPTHS), proj, cfg["dropout"]).to(device)
    net.set_scalers({k: v.to(device) for k, v in st.items() if k.endswith(("_f", "_t"))})
    fn = _h0_fn(net)
    info = train_loop(net, fn, items_of(fit_decs), [(d, 0) for d in val_decs], cfg, seed, device)
    info["scaler_policy"] = hashlib.sha256(repr((FEATURE_SD_FLOOR, EMBED_SD_FLOOR, len(fit_decs))).encode()).hexdigest()[:16]
    return net, info


def crossfit_h0(decs, cfg, seed, proj, device="cpu", nfold=None):
    """Out-of-fold H0 scores for every item of every decision: fold f is scored by an H0 fitted on the other folds only (trajectory-grouped).
    Returns ({(did, k): score vector}, audit records proving the scored trajectories were not in that fold model's training rows)."""
    fold_of = a5v2.folds_by_trajectory(decs)
    h0s, audit = {}, []
    for f in range(nfold or a5v2.S["cv_folds"]):
        tr, te = [d for d in decs if fold_of[d["tid"]] != f], [d for d in decs if fold_of[d["tid"]] == f]
        if not te:
            continue
        a, v = inner_split(tr, CFG["val_frac"])
        net, info = fit_h0(a, v, cfg, seed, proj, device)
        its = items_of(te)
        for (d, k), s in zip(its, predict(_h0_fn(net), its, device=device)):
            h0s[(d["did"], k)] = s
        audit.append(dict(fold=f, trained_on={d["tid"] for d in tr}, scored={d["tid"] for d in te}, info=info))
    return h0s, audit


class System:
    """One trained pair on one training set: the H0 arm (final H0) and the H2 arm (the SAME frozen H0 object + residual)."""

    def __init__(self, h0, res, cfg, seed, proj, treatment, record):
        self.h0, self.res, self.cfg, self.seed, self.proj, self.treatment, self.record = h0, res, cfg, seed, proj, treatment, record

    @torch.no_grad()
    def scores(self, decs, k=0, device="cpu"):
        """(H0 logits per decision, H2 logits per decision) on set k of each decision."""
        its = [(d, k) for d in decs]
        h0 = predict(_h0_fn(self.h0), its, device=device)
        h0s = {(d["did"], k): s for (d, _), s in zip(its, h0)}
        h2 = predict(_res_fn(self.res), its, h0s, device=device)
        return h0, h2

    def state(self):
        return dict(cfg=self.cfg, seed=self.seed, proj=self.proj, treatment=self.treatment, h0=self.h0.state_dict(), res=self.res.state_dict(), record=self.record)


def h0_bundle(train_decs, cfg, seed, proj, device="cpu"):
    """Everything about H0 for one (training set, seed, projection): the final frozen H0 and the cross-fitted scores. Shared by every depth treatment."""
    a, v = inner_split(train_decs, cfg["val_frac"])
    h0, info = fit_h0(a, v, cfg, seed, proj, device)
    h0s, folds = crossfit_h0(train_decs, cfg, seed, proj, device)
    return dict(a=a, v=v, h0=h0, info=info, h0s=h0s, folds=folds, proj=proj, seed=seed)


def fit_residual(bundle, cfg, treatment, device="cpu"):
    a, v, h0s, seed, proj = bundle["a"], bundle["v"], bundle["h0s"], bundle["seed"], bundle["proj"]
    st = scaler_stats(a)
    emb = a[0]["q"].shape[-1]
    torch.manual_seed(seed)
    res = ResidualNet(emb, len(DEPTHS), proj, treatment, cfg["dropout"], DEPTHS.index(SINGLE_DEPTH)).to(device)
    res.set_scalers({k: v_.to(device) for k, v_ in st.items() if k.endswith(("_q", "_c"))})
    ri = train_loop(res, _res_fn(res), items_of(a), [(d, 0) for d in v], cfg, seed, device, h0s=h0s)
    ci = bundle["info"]
    # same rows, same candidate sets, same seed, same optimiser schedule, same shuffle: the two arms differ only in the state branch
    assert ri["fit_keys_hash"] == ci["fit_keys_hash"], "H0 and H2 trained on different rows/sets"
    assert ri["first_epoch_order_hash"] == ci["first_epoch_order_hash"] and ri["seed"] == ci["seed"], "H0 and H2 used different shuffles/seeds"
    assert ri["schedule"] == ci["schedule"], "H0 and H2 used different optimiser schedules"
    for fo in bundle["folds"]:
        assert not (fo["trained_on"] & fo["scored"]), "a cross-fit H0 scored trajectories it was trained on"
    assert {k[0] for k in h0s} >= {d["did"] for d in a + v}, "residual training lacked out-of-fold H0 scores for some rows"
    rec = dict(h0=ci, residual=ri, crossfit=[dict(fold=fo["fold"], n_train_traj=len(fo["trained_on"]), n_scored_traj=len(fo["scored"])) for fo in bundle["folds"]],
               residual_trained_on_oof_h0_scores=True)
    return System(bundle["h0"], res, cfg, seed, proj, treatment, rec)


def fit_system(train_decs, cfg, seed, proj, treatment, device="cpu"):
    return fit_residual(h0_bundle(train_decs, cfg, seed, proj, device), cfg, treatment, device)


def fit_temperature(logit_vecs, ys):
    """One scalar temperature by minimising NLL on calibration logits (CAL's only role). Accuracy is invariant to it."""
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.5, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = sum(-F.log_softmax(l / t.exp(), 0)[y] for l, y in zip(logit_vecs, ys)) / len(ys)
        loss.backward()
        return loss
    opt.step(closure)
    return t.exp().item()


def nested_subsets(decs, sizes, seed="a5v3-lc"):
    """Nested trajectory-grouped training subsets counted in DECISIONS: trajectories in a fixed hash order, the last one truncated to hit each size exactly."""
    by = {}
    for d in decs:
        by.setdefault(d["tid"], []).append(d)
    stream = [d for t in sorted(by, key=lambda t: hashlib.sha256(f"{seed}|{t}".encode()).hexdigest()) for d in by[t]]
    return {n: stream[:n] for n in sizes}
