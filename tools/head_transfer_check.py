"""Exploratory (no gate): does Laya's pretrained decision head (Coding_Decision_Agent) work as an INTERNAL head on frozen Qwen3-4B states?
Qwen is never run or trained here: it reads the registered TRAIN/CAL T4 captures (data/v3/capture_t4, 3 depths x 2560). LOCKED is never loaded.
Arms, same data/seeds/hyper-parameters: B random init | A pretrained ctx+scorer+type_emb, all trainable | F pretrained and FROZEN (only new input projections, layer mix, gate train).
Labels are the A5 next-action labels already in the capture (the only Qwen-captured set; imitation, so this checks MECHANICS, not the outcome-labelled plan).
CAL is only evaluated at the end (no selection on it, no gradient). Output create-only. No threshold, no pass claim.
    PYTHONPATH=. .venv/Scripts/python.exe tools/head_transfer_check.py [--dry_run]
"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import argparse, json, time, math
from datetime import datetime, timezone
from pathlib import Path
import torch, torch.nn.functional as F
from safetensors.torch import load_file
from decision_head import v3_data, v3_training as T
from decision_head.head import DecisionHead, decision_loss

ap = argparse.ArgumentParser(); ap.add_argument("--dry_run", action="store_true"); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
out = ROOT / f"results/raw/head-transfer-check-{stamp}{'-DRYRUN' if a.dry_run else ''}.json"
assert not out.exists()
HP = dict(lr=1e-4, wd=1e-2, batch=16, max_epochs=3 if a.dry_run else 15, patience=3, seeds=[0] if a.dry_run else [0, 1, 2], val_frac=0.10)
CKPT = ROOT / "models/coding-decision-agent/model.safetensors"
t0 = time.time(); log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

# ---- data (TRAIN fit/val split by trajectory; CAL evaluated once) ----
ids = json.load(open(ROOT / "data/v3/manifest-train.json", encoding="utf-8"))["decision_ids"]
built = v3_data._built_train(); sets = v3_data.oof_sets_multi(list(built.values()))
TE, te_idx = v3_data.load_textemb(ROOT / "data/v3/textemb_t4/textemb_t4")
tr, miss = v3_data.assemble(ids, built, v3_data.load_captures(ROOT / "data/v3/capture_t4/train"), TE, te_idx, "train", sets)
cid = json.load(open(ROOT / "data/v3/manifest-cal.json", encoding="utf-8"))["decision_ids"]
cal, miss_c = v3_data.load_cal(cid, ROOT / "data/v3/capture_t4/cal", ROOT / "data/v3/textemb_t4/textemb_t4")
if a.dry_run:
    tr, cal = tr[:80], cal[:40]
fit, val = T.inner_split(tr, HP["val_frac"])
log(f"TRAIN {len(tr)} (fit {len(fit)} / val {len(val)}), CAL {len(cal)}, missing train {len(miss)} cal {len(miss_c)}, device {dev}")
items = lambda decs, ks=None: [(d, k) for d in decs for k in (range(len(d["sets"])) if ks is None else ks)]
fit_i, val_i, cal_i = items(fit), items(val, [0]), items(cal, [0])


def batch(its):
    n = max(len(d["sets"][k]["order"]) for d, k in its)
    B, K, H = len(its), its[0][0]["q"].shape[0], its[0][0]["q"].shape[1]
    hq, ho = torch.zeros(B, K, H), torch.zeros(B, n, K, H)
    mask, y = torch.zeros(B, n, dtype=torch.bool), torch.zeros(B, dtype=torch.long)
    for i, (d, k) in enumerate(its):
        s = d["sets"][k]; o = torch.tensor(s["order"])
        hq[i], ho[i, :len(o)], mask[i, :len(o)], y[i] = d["q"].float(), d["cx"][o].float(), True, s["true_pos"]
    return hq.to(dev), ho.to(dev), mask.to(dev), y.to(dev)


# scaler statistics from the fit split only (exact, streamed)
def stats():
    sq = torch.zeros(3, 2560, dtype=torch.float64); sq2 = sq.clone(); so = sq.clone(); so2 = sq.clone(); nq = no = 0
    for d in fit:
        q = d["q"].double(); sq += q; sq2 += q * q; nq += 1
        o = d["cx"].double(); so += o.sum(0); so2 += (o * o).sum(0); no += o.shape[0]
    f = lambda s, s2, n: ((s / n).float(), ((s2 / n - (s / n) ** 2).clamp_min(0).sqrt()).float().clamp_min(1e-6))
    return f(sq, sq2, nq), f(so, so2, no)
(mq, sdq), (mo, sdo) = stats()


def make(init, seed):
    torch.manual_seed(seed)
    h = DecisionHead(2560, d=1024, layers=2, k_layers=3, scorer="mlp", dropout=0.1)
    with torch.no_grad():
        h.mu_q.copy_(mq); h.sd_q.copy_(sdq); h.mu_o.copy_(mo); h.sd_o.copy_(sdo)
        h.kind_emb.weight.zero_()                         # kind is always ACTION here; neutral in every arm
    rep = None
    if init in ("A", "F"):
        sd = load_file(str(CKPT)); m = {}
        for k, v in sd.items():
            if k.startswith("head.layers."): m["ctx.layers." + k[len("head.layers."):]] = v
            elif k.startswith("scorer."): m["mlp." + k[len("scorer."):]] = v
            elif k == "type_emb.weight": m["prim_emb.weight"] = v
        r = h.load_state_dict({k: v.float() for k, v in m.items()}, strict=False)
        assert not r.unexpected_keys, r.unexpected_keys
        rep = dict(loaded=len(m), unexpected=list(r.unexpected_keys),
                   not_loaded=sorted(k for k in r.missing_keys if k.startswith(("ctx.", "mlp.", "prim_emb"))),
                   new_modules=sorted({k.split(".")[0] for k in r.missing_keys}))
    if init == "F":
        for n_, p in h.named_parameters():
            p.requires_grad = n_.split(".")[0] in ("in_q", "in_o", "mix_q", "mix_o", "gate", "out_norm")
    return h.to(dev), rep


@torch.no_grad()
def evaluate(h, its):
    h.eval(); nll = acc = ch = 0.0
    for i in range(0, len(its), 32):
        sub = its[i:i + 32]; hq, ho, m, y = batch(sub)
        lg, _ = h(hq, ho, m, torch.zeros(len(sub), dtype=torch.long, device=dev))
        nll += F.cross_entropy(lg, y, reduction="sum").item(); acc += (lg.argmax(-1) == y).sum().item(); ch += (1.0 / m.sum(-1)).sum().item()
    n = len(its); return dict(nll=nll / n, top1=acc / n, chance_top1=ch / n, n=n)


def run(init, seed):
    h, rep = make(init, seed)
    opt = torch.optim.AdamW([p for p in h.parameters() if p.requires_grad], lr=HP["lr"], weight_decay=HP["wd"])
    g = torch.Generator().manual_seed(seed); best, bad, state, hist = 1e9, 0, None, []
    for ep in range(HP["max_epochs"]):
        h.train(); perm = torch.randperm(len(fit_i), generator=g).tolist()
        for i in range(0, len(perm), HP["batch"]):
            hq, ho, m, y = batch([fit_i[j] for j in perm[i:i + HP["batch"]]])
            lg, _ = h(hq, ho, m, torch.zeros(len(y), dtype=torch.long, device=dev))
            loss = decision_loss(lg, m, F.one_hot(y, lg.size(1)).float())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0); opt.step()
        v = evaluate(h, val_i); hist.append(dict(epoch=ep, val_nll=v["nll"], val_top1=v["top1"]))
        if v["nll"] < best - 1e-4: best, bad, state = v["nll"], 0, {k: t.detach().clone() for k, t in h.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]: break
    h.load_state_dict(state)
    return dict(init=init, seed=seed, epochs=len(hist), val=evaluate(h, val_i), cal=evaluate(h, cal_i), history=hist, transfer_report=rep)


res = []
for seed in HP["seeds"]:
    for init in ("B", "A", "F"):
        r = run(init, seed); res.append(r)
        log(init, "seed", seed, "epochs", r["epochs"], "val top1 %.3f nll %.3f | CAL top1 %.3f nll %.3f (chance %.3f)" % (r["val"]["top1"], r["val"]["nll"], r["cal"]["top1"], r["cal"]["nll"], r["cal"]["chance_top1"]))
json.dump(dict(label="exploratory (no gate)", dry_run=a.dry_run, hp=HP, device=(torch.cuda.get_device_name(0) if dev == "cuda" else dev),
               checkpoint=str(CKPT), arms={"B": "random init", "A": "pretrained all trainable", "F": "pretrained frozen"},
               n=dict(train=len(tr), fit=len(fit), val=len(val), cal=len(cal)), results=res), open(out, "x"), indent=1)
log("wrote", out)
