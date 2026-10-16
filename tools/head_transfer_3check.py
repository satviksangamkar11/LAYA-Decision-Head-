"""Exploratory (no gate): 3-decision 'does it run' check of Laya's pretrained head as an INTERNAL head on frozen Qwen states (T4 captures). Output create-only."""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, time, torch, torch.nn.functional as F
from datetime import datetime, timezone
from pathlib import Path
from safetensors.torch import load_file
from decision_head import v3_data
from decision_head.head import DecisionHead, decision_loss
R = Path(__file__).resolve().parents[1]; dev = "cuda" if torch.cuda.is_available() else "cpu"
ids = json.load(open(R / "data/v3/manifest-train.json", encoding="utf-8"))["decision_ids"][:3]
built = v3_data._built_train(); sets = v3_data.oof_sets_multi(list(built.values()))
TE, ti = v3_data.load_textemb(R / "data/v3/textemb_t4/textemb_t4")
tr, _ = v3_data.assemble(ids, built, v3_data.load_captures(R / "data/v3/capture_t4/train"), TE, ti, "train", sets)
cid = json.load(open(R / "data/v3/manifest-cal.json", encoding="utf-8"))["decision_ids"][:3]
cal, _ = v3_data.load_cal(cid, R / "data/v3/capture_t4/cal", R / "data/v3/textemb_t4/textemb_t4")
assert len(tr) == 3 and len(cal) == 3, (len(tr), len(cal))
def batch(decs):
    its = [(d, 0) for d in decs]; n = max(len(d["sets"][0]["order"]) for d in decs)
    hq, ho = torch.zeros(3, 3, 2560), torch.zeros(3, n, 3, 2560); m = torch.zeros(3, n, dtype=torch.bool); y = torch.zeros(3, dtype=torch.long)
    for i, (d, k) in enumerate(its):
        s = d["sets"][k]; o = torch.tensor(s["order"]); hq[i], ho[i, :len(o)], m[i, :len(o)], y[i] = d["q"].float(), d["cx"][o].float(), True, s["true_pos"]
    return hq.to(dev), ho.to(dev), m.to(dev), y.to(dev)
torch.manual_seed(0)
h = DecisionHead(2560, d=1024, layers=2, k_layers=3, scorer="mlp", dropout=0.0)
sd = load_file(str(R / "models/coding-decision-agent/model.safetensors")); mp = {}
for k, v in sd.items():
    if k.startswith("head.layers."): mp["ctx.layers." + k[12:]] = v
    elif k.startswith("scorer."): mp["mlp." + k[7:]] = v
    elif k == "type_emb.weight": mp["prim_emb.weight"] = v
r = h.load_state_dict({k: v.float() for k, v in mp.items()}, strict=False); assert not r.unexpected_keys
with torch.no_grad(): h.kind_emb.weight.zero_()
hq, ho, m, y = batch(tr); h.to(dev); pr = torch.zeros(3, dtype=torch.long, device=dev)
opt = torch.optim.AdamW(h.parameters(), lr=1e-4); losses = []
for step in range(5):
    lg, gate = h(hq, ho, m, pr); loss = decision_loss(lg, m, F.one_hot(y, lg.size(1)).float())
    opt.zero_grad(); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)); opt.step(); losses.append(round(loss.item(), 4))
h.eval()
with torch.no_grad():
    cq, co, cm, cy = batch(cal); cl, _ = h(cq, co, cm, pr); p = cl.softmax(-1)
res = dict(label="exploratory (no gate)", n_train=3, n_cal=3, device=torch.cuda.get_device_name(0) if dev == "cuda" else dev, tensors_loaded=len(mp),
           not_loaded=[k for k in r.missing_keys if k.startswith(("ctx.", "mlp.", "prim_emb"))], train_losses_5_steps=losses, grad_norm_last=gn,
           loss_finite=all(map(lambda x: x == x and abs(x) < 1e6, losses)), cal_logits_finite=bool(torch.isfinite(cl[cm]).all()),
           cal_prob_sums=[round(float(s), 4) for s in p.sum(-1)], cal_options=[int(x) for x in cm.sum(-1)], cal_top1_correct=int((cl.argmax(-1) == cy).sum()), gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None)
print(json.dumps(res, indent=1))
o = R / f"results/raw/head-transfer-3check-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"; json.dump(res, open(o, "x"), indent=1); print("wrote", o)
