"""One GPU worker for the Kaggle head runs. Stages: time (one real cell), select (3-seed cell-selection jobs), screen (pre-LOCKED screen jobs). TRAIN only: this process never loads CAL or LOCKED.
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/head_v3_worker.py --stage select --jobs jobs.json --worker 0 --workers 2 --out out/select-w0.json --capdir CAP --textemb TE
jobs.json is a list; worker i takes jobs[i::workers]. The output file is rewritten after every job (progress survives a crash). --dry_run is a plumbing test only (80 decisions, 2 epochs) and marks every result."""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import torch
from compiler import a5v2
from decision_head import v3_data, v3_screen, v3_selection as SEL, v3_training as T

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["smoke", "time", "select", "screen"])
ap.add_argument("--jobs", required=True); ap.add_argument("--worker", type=int, default=0); ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--out", required=True); ap.add_argument("--capdir", required=True); ap.add_argument("--textemb", required=True)
ap.add_argument("--device", default="cuda"); ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()
if a.stage == "smoke":
    a.dry_run = True           # device and plumbing check on 80 decisions and 2 epochs; its numbers are meaningless and never used
assert not os.path.exists(a.out), f"{a.out} exists (create-only)"
dev = a.device
cuda = dev.startswith("cuda")
if cuda:
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0), "the registered device class is the Tesla T4"
t0 = time.time()
ids = json.load(open("data/v3/manifest-train.json", encoding="utf-8"))["decision_ids"]
built = v3_data._built_train()
exp = json.load(open("EXPECTED.json"))
sets = v3_data.oof_sets_multi(list(built.values()))
got = dict(built_ids_sha256=hashlib.sha256(json.dumps(sorted(built)).encode()).hexdigest(), n_built=len(built), n_manifest=len(ids),
           folds_sha256=hashlib.sha256(json.dumps(sorted(a5v2.folds_by_trajectory(list(built.values())).items())).encode()).hexdigest(),
           sets_sha256=hashlib.sha256(json.dumps({k: sets[k] for k in sorted(sets)}, sort_keys=True).encode()).hexdigest())
assert got == exp, f"the rebuilt TRAIN pool/folds/sets differ from the locally computed ones: {got} vs {exp}"
TE, te_idx = v3_data.load_textemb(a.textemb)
decs, missing = v3_data.assemble(ids, built, v3_data.load_captures(a.capdir), TE, te_idx, "train", sets)
assert not missing and len(decs) == len(ids) == 1986, (len(missing), len(decs))
if a.dry_run:
    decs = decs[:80]
    T.CFG.update(max_epochs=2, min_epochs=1, patience=1)
log = lambda *x: print(f"[w{a.worker} {time.time() - t0:7.0f}s]", *x, flush=True)
log(f"loaded {len(decs)} TRAIN decisions; pool, folds and candidate sets equal the local ones; device {torch.cuda.get_device_name(0) if cuda else dev}")

calls = []
_orig = T.train_loop


def _spy(net, fn, fit_items, val_items, cfg, seed, device="cpu", h0s=None, loss="ce"):
    if cuda:
        torch.cuda.synchronize()
    s = time.perf_counter()
    info = _orig(net, fn, fit_items, val_items, cfg, seed, device, h0s, loss)
    if cuda:
        torch.cuda.synchronize()
    calls.append(dict(kind="residual" if h0s is not None else "h0", seconds=time.perf_counter() - s, epochs=info["epochs"], n_fit=len(fit_items), n_val=len(val_items)))
    return info


T.train_loop = _spy
jobs = json.load(open(a.jobs))[a.worker::a.workers]
res = dict(stage=a.stage, worker=a.worker, workers=a.workers, dry_run=a.dry_run, device=(torch.cuda.get_device_name(0) if cuda else dev), torch=torch.__version__, results=[])


def flush():
    json.dump(res, open(a.out, "w"), indent=1)


for j in jobs:
    calls.clear()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    s = time.perf_counter()
    if a.stage == "smoke":
        cells = [tuple(c) for c in j["cells"]]
        r = dict(job=j, nll={f"{p}:{t}": v for (p, t), v in SEL.dev_scores(decs, T.CFG, j["seed"], dev, cells, log).items()}, out=v3_screen.screen_job(decs, j["fold"], j["seed"], tuple(j["cell"]), dev))
    elif a.stage in ("time", "select"):
        cells = [tuple(c) for c in j["cells"]]
        r = dict(job=j, nll={f"{p}:{t}": v for (p, t), v in SEL.dev_scores(decs, T.CFG, j["seed"], dev, cells, log).items()})
    else:
        r = dict(job=j, out=v3_screen.screen_job(decs, j["fold"], j["seed"], tuple(j["cell"]), dev))
    if cuda:
        torch.cuda.synchronize()
    r.update(seconds=time.perf_counter() - s, peak_vram_gb=(torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None), train_calls=list(calls))
    res["results"].append(r)
    flush()
    log(f"job done {j} in {r['seconds']:.0f}s ({len(calls)} training calls, {sum(c['epochs'] * c['n_fit'] for c in calls)} fit rows processed)")
res["wall_seconds"] = time.time() - t0
flush()
log("DONE")
