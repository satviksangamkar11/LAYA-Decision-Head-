"""Structural pre-flight of the T4 Phase-2 data through the registered v3 loader (decision_head/v3_data.py): NO training, NO accuracy, no labels used beyond the loader's own assertions.
    PYTHONPATH=. .venv/Scripts/python.exe tools/preflight_t4_data.py
Checks: every TRAIN and CAL manifest decision loads; captured candidate texts equal the rebuilt pools; every pool text has a T4 text embedding; the NONE option is captured but never scored;
shapes, finiteness; counts per role. Create-only result: results/raw/t4-loader-preflight-<time>.json."""
import os, json, time, hashlib, datetime
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import torch
from decision_head import v3_data
cap, te = "data/v3/capture_t4", "data/v3/textemb_t4/textemb_t4"
out = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), capture_dir=cap, textemb_dir=te)
t0 = time.time()
for role, loader in (("cal", v3_data.load_cal), ("train", v3_data.load_train)):
    ids = json.load(open(f"data/v3/manifest-{role}.json", encoding="utf-8"))["decision_ids"]
    decs, missing = loader(ids, f"{cap}/{role}", te)
    n_sets = [len(d["sets"]) for d in decs]
    q0 = decs[0]["q"]
    out[role] = dict(manifest=len(ids), loaded=len(decs), missing=len(missing), sets_per_decision_min_max=[min(n_sets), max(n_sets)], query_shape=list(q0.shape),
                     all_finite=all(bool(torch.isfinite(d["q"].float()).all()) for d in decs), seconds=round(time.time() - t0, 1))
    print(role, out[role], flush=True)
p = f"results/raw/t4-loader-preflight-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
assert not os.path.exists(p)
json.dump(out, open(p, "w"), indent=1); print("wrote", p)
