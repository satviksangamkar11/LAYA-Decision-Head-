"""A5-v3 pre-LOCKED screen (configs/a5_v3_spec.toml [amendment_2026_10_03_e] e7, e8, e10). TRAIN trajectory-grouped 5-fold CV of H2 minus H0_matched at the selected cell with the paired seeds,
per decision delta_i = mean over seeds of (H2 correct - H0 correct), trajectory-cluster bootstrap (2,000 resamples, seed 0), then PRE_LOCKED_STOP = (point estimate < the registered 0.02).
0.02 is a go/no-go screen only, NOT success. This module has no path to the locked data: it never reads the locked files or calls the evaluator's loader (an audit check greps this file and the
locked access log must not grow during a screen). It imports only `bootstrap` (a pure function) from the evaluator.
Everything a result needs is written by `provenance()`: git commit, trainer/spec hashes, dataset and embedding hashes, model revision, library versions, seeds."""
import hashlib
import json
import os
import platform
import subprocess
import tomllib
from pathlib import Path

import torch

from compiler import a5v2
from decision_head import v3_training as T
from decision_head.v3_evaluation import bootstrap

ROOT = Path(__file__).resolve().parents[1]
TRAINER_FILES = ["v3_data", "v3_model", "v3_training", "v3_selection", "v3_screen"]


def stop_threshold():
    """The single source of the registered number: configs/thresholds.toml [amendment_e_thresholds]."""
    return tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["amendment_e_thresholds"]["pre_locked_screen_point_estimate_min"]


def pre_locked_stop(point_estimate):
    """True means STOP: do not open or use LOCKED. False means the registered LOCKED evaluation may proceed (it is still the user's written yes for the upload)."""
    return bool(point_estimate < stop_threshold())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(capture_verify_json, textemb_verify_json, commit=None, extra=None):
    """Machine-readable provenance. commit: pass explicitly when there is no git (Kaggle); otherwise read from git (HEAD plus a dirty flag over the tracked code and configs)."""
    git = {}
    try:
        git["commit"] = commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        git["dirty"] = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "decision_head", "tools", "configs", "compiler", "capture"], cwd=ROOT).returncode != 0
    except Exception:
        git = {"commit": commit or "unknown (no git)", "dirty": None}
    cv = json.load(open(capture_verify_json))
    ev = json.load(open(textemb_verify_json))
    emb_hash = hashlib.sha256(json.dumps(ev["shard_sha256"], sort_keys=True).encode()).hexdigest()
    rev = json.load(open(ROOT / "results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
    p = dict(git=git,
             trainer_sha256={m: _sha(ROOT / f"decision_head/{m}.py") for m in TRAINER_FILES},
             spec_sha256={f: _sha(ROOT / f"configs/{f}") for f in ("a5_v3_spec.toml", "thresholds.toml")},
             datasets=dict(manifest_train=_sha(ROOT / "data/v3/manifest-train.json"), manifest_cal=_sha(ROOT / "data/v3/manifest-cal.json"),
                           decisions=_sha(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"), capture_all_files_sha256={r: cv["roles"][r]["sha256_of_all_files"] for r in ("train", "cal")},
                           capture_verify_file=str(capture_verify_json)),
             embedding=dict(shard_hashes_sha256=emb_hash, texts=ev["texts_embedded"], device=ev["device_sidecar"], verify_file=str(textemb_verify_json)),
             model_revision=rev,
             libraries=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
                            device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")),
             schedule=dict(T.CFG), selection_seeds=list(T.SELECTION_SEEDS), final_seeds=list(T.FINAL_SEEDS), stop_threshold=stop_threshold())
    if extra:
        p.update(extra)
    return p


def screen_job(train_decs, fold, seed, cell, device="cpu"):
    """One (fold, seed) unit of the screen: fit_system (H0 and H2 on the SAME rows/sets/seed/schedule) on the other folds, score set 0 of the held-out fold.
    Returns {did: [h0_correct, h2_correct, tid]}. Independent of every other job, so jobs can run on different GPUs and merge exactly."""
    assert all(d["role"] == "train" for d in train_decs), "the screen uses TRAIN rows only"
    fold_of = a5v2.folds_by_trajectory(train_decs)
    tr, te = [d for d in train_decs if fold_of[d["tid"]] != fold], [d for d in train_decs if fold_of[d["tid"]] == fold]
    sysm = T.fit_system(tr, T.CFG, seed, cell[0], cell[1], device)
    h0, h2 = sysm.scores(te, 0, device)
    return {d["did"]: [float(a.argmax().item() == d["sets"][0]["true_pos"]), float(b.argmax().item() == d["sets"][0]["true_pos"]), d["tid"]] for a, b, d in zip(h0, h2, te)}


def screen_aggregate(job_results, seeds, cell):
    """job_results: {(fold, seed): screen_job output}. Per decision delta = mean over seeds of (H2 correct - H0 correct), then the trajectory-cluster bootstrap and the registered stop rule."""
    folds = sorted({f for f, _ in job_results})
    assert all((f, s) in job_results for f in folds for s in seeds) and len(job_results) == len(folds) * len(seeds), "every (fold, seed) job is required exactly once"
    rows, tids, a0, a2 = {}, {}, {}, {}
    for (f, s), r in job_results.items():
        for did, (c0, c2, tid) in r.items():
            rows.setdefault(did, []).append(c2 - c0)
            a0.setdefault(did, []).append(c0)
            a2.setdefault(did, []).append(c2)
            tids[did] = tid
    assert all(len(v) == len(seeds) for v in rows.values()), "every decision must be scored once per seed"
    order = sorted(rows)
    delta = {k: sum(rows[k]) / len(seeds) for k in order}
    n = len(order)
    point, lo, hi = bootstrap([delta[k] for k in order], [tids[k] for k in order])
    return dict(delta=[delta[k] for k in order], did=order, tids=[tids[k] for k in order], h0_acc=sum(sum(a0[k]) / len(seeds) for k in order) / n, h2_acc=sum(sum(a2[k]) / len(seeds) for k in order) / n,
                point=point, ci95=[lo, hi], n_decisions=n, n_clusters=len(set(tids.values())), seeds=list(seeds), cell=list(cell), pre_locked_stop=pre_locked_stop(point))


def run_screen(train_decs, cell, seeds=T.SELECTION_SEEDS, device="cpu", log=print):
    """TRAIN-only CV on one process: all (fold, seed) jobs, then the aggregate. Nothing here can see CAL or LOCKED."""
    assert all(d["role"] == "train" for d in train_decs), "the screen uses TRAIN rows only"
    folds = sorted({a5v2.folds_by_trajectory(train_decs)[d["tid"]] for d in train_decs})
    res = {}
    for f in folds:
        for s in seeds:
            res[(f, s)] = screen_job(train_decs, f, s, cell, device)
            log(f"fold {f} seed {s}: H0 {sum(v[0] for v in res[(f, s)].values()) / len(res[(f, s)]):.4f} H2 {sum(v[1] for v in res[(f, s)].values()) / len(res[(f, s)]):.4f}")
    return screen_aggregate(res, seeds, cell)
