"""Writes kaggle/head_v3_t4.ipynb: timing of one real TRAIN cell, the registered 3-seed cell selection and the pre-LOCKED screen (amendment e7/e8) on Kaggle T4 x2, TRAIN rows only.
Run: .venv/Scripts/python.exe tools/make_kaggle_head_notebook.py (overwrites; generated, not hand-edited).
Inputs to attach: the private dataset built by tools/make_kaggle_head_bundle.py, the OUTPUT of the Phase-2 capture notebook (capture_train.tar, capture_manifest.json) and the OUTPUT of the T4 text-embedding notebook (textemb_t4.tar).
No CAL rows or CAL tensors are read, no LOCKED data exists in any input. Spend: free Kaggle quota only."""
import json
from pathlib import Path
MD = r"""# Head v3 on Kaggle T4 x2: timing, 3-seed selection, pre-LOCKED screen (TRAIN only)
Stages: (1) integrity checks of every input, (2) time ONE real TRAIN cell on one T4 and extrapolate, (3) the registered selection (9 cells x seeds {0,1,2}, jobs split over both GPUs),
(4) the pre-LOCKED screen at the selected cell (5 folds x seeds {0,1,2}), PRE_LOCKED_STOP = point estimate < 0.02 (go/no-go only, not success). Create-only results with provenance are written to /kaggle/working.
Device lock as the capture: Tesla T4, torch 2.10.0+cu128 (installed for the worker; Kaggle ignores the environment pin). A guard stops after the timing stage if the projected remaining wall time exceeds 9 h."""
CELLS = [("md", MD),
("code", r'''import subprocess, sys, json, os, glob, shutil, hashlib, time, tarfile
import torch
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv", shell=True, capture_output=True, text=True).stdout)
NG = torch.cuda.device_count(); assert NG >= 1 and all("T4" in torch.cuda.get_device_name(i) for i in range(NG)), "turn on GPU T4 x2"
print("kernel torch", torch.__version__, "python", sys.version.split()[0], "GPUs", NG)
if torch.__version__ != "2.10.0+cu128":
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu128", "--extra-index-url", "https://pypi.org/simple"], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-900:])
    r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"], capture_output=True, text=True); print(r.stdout[-300:], r.stderr[-300:])
wt = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__, torch.version.cuda)"], capture_output=True, text=True).stdout.strip()
print("worker torch", wt); assert wt.split()[0] == "2.10.0+cu128", f"device lock: worker torch must be 2.10.0+cu128, got {wt}"'''),
("code", r'''hits = [h for h in glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True) if "headtrain" in json.load(open(h)).get("purpose", "") or "head timing" in json.load(open(h)).get("purpose", "")]
assert hits, "attach the private head-train dataset"
SRC = os.path.dirname(hits[0]); W = "/kaggle/temp/proj"; shutil.copytree(SRC, W, dirs_exist_ok=True)
man = json.load(open(W + "/MANIFEST.json"))
bad = [f["path"] for f in man["files"] if hashlib.sha256(open(f"{W}/{f['path']}", "rb").read()).hexdigest() != f["sha256"]]; assert not bad, bad
assert man["contains_locked"] is False and man["contains_cal"] is False and man["contains_captured_tensors"] is False
print("bundle OK:", len(man["files"]), "files,", man["train_rows"], "TRAIN rows, commit", man["source_commit"][:8], "dirty at bundle:", man["source_dirty_at_bundle"])
def find(name):
    h = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert len(h) == 1, (name, h); return h[0]
CAPMAN = find("capture_manifest.json")
def dirs_with(pattern, n):
    d = {}
    for f in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True): d.setdefault(os.path.dirname(f), []).append(f)
    return [k for k, v in d.items() if len(v) == n]
# Kaggle may keep a notebook output's tar files as tars or extract them into folders; accept the folder (exactly 1986 TRAIN capture files, exactly 7 text-embedding shards), else extract the tar
CAPD_c = [k for k in dirs_with("*.pt", 1986) if "capture_train" in k or k.rstrip("/").endswith("/train")]
TED_c = dirs_with("shard-*.pt", 7)
if not CAPD_c:
    CAPTAR = find("capture_train.tar"); os.makedirs("/kaggle/temp/cap", exist_ok=True)
    with tarfile.open(CAPTAR) as t: t.extractall("/kaggle/temp/cap")
    CAPD_c = ["/kaggle/temp/cap/train"]
if not TED_c:
    TEMTAR = find("textemb_t4.tar"); os.makedirs("/kaggle/temp/te", exist_ok=True)
    with tarfile.open(TEMTAR) as t: t.extractall("/kaggle/temp/te")
    TED_c = ["/kaggle/temp/te/textemb_t4"]
assert len(CAPD_c) == 1 and len(TED_c) == 1, (CAPD_c, TED_c)
CAPD, TED = CAPD_c[0], TED_c[0]
print("capture dir:", CAPD, "| text-embedding dir:", TED)
# integrity: every TRAIN capture file against the Kaggle capture manifest, the aggregate against the locally verified one, every text-embedding shard against the local verify record
km = json.load(open(CAPMAN)); rec = {e["file"]: e["sha256"] for e in km["files"]["train"]}
hashes = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(CAPD + "/*.pt"))}
assert hashes == rec and len(hashes) == 1986, (len(hashes), len(rec))
cv = json.load(open(W + "/results/raw/phase2-capture-verify-20261003-145647.json"))
agg = hashlib.sha256("".join(f"{k}:{hashes[k]}\n" for k in sorted(hashes)).encode()).hexdigest()
assert agg == cv["roles"]["train"]["sha256_of_all_files"], "capture files differ from the locally verified set"
ev = json.load(open(W + "/results/raw/textemb-t4-verify-20261003-155540.json"))
shh = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(TED + "/shard-*.pt"))}
assert shh == ev["shard_sha256"], "text-embedding shards differ from the locally verified set"
print("inputs verified: 1986 TRAIN capture files (per-file and aggregate sha256), text-embedding shards:", len(shh))'''),
("code", r'''os.makedirs("/kaggle/working/out", exist_ok=True)
gpu = subprocess.Popen("nvidia-smi --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -l 20 -f /kaggle/working/gpu.csv", shell=True)
T0 = time.time(); STAMP = time.strftime("%Y%m%d-%H%M%S")
def run_stage(stage, jobs, tag, workers):
    jp = f"/kaggle/working/out/jobs-{tag}.json"; json.dump(jobs, open(jp, "w"))
    procs = []
    for i in range(workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONPATH=W)
        log = open(f"/kaggle/working/out/{tag}-w{i}.log", "w")
        cmd = [sys.executable, "tools/head_v3_worker.py", "--stage", stage, "--jobs", jp, "--worker", str(i), "--workers", str(workers), "--out", f"/kaggle/working/out/{tag}-w{i}.json", "--capdir", CAPD, "--textemb", TED]
        procs.append((subprocess.Popen(cmd, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT), log)); time.sleep(5)
    t = time.time()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(120)
        done = sum(len(json.load(open(f"/kaggle/working/out/{tag}-w{i}.json"))["results"]) for i in range(workers) if os.path.exists(f"/kaggle/working/out/{tag}-w{i}.json"))
        print(f"[{tag}] {done}/{len(jobs)} jobs, {(time.time() - t) / 60:.1f} min, alive={[p.poll() is None for p, _ in procs]}", flush=True)
        for i, (p, _) in enumerate(procs):
            if p.poll() not in (None, 0):
                print(f"--- {tag} worker {i} exited {p.poll()}; tail:\n" + "".join(open(f"/kaggle/working/out/{tag}-w{i}.log").readlines()[-30:]), flush=True)
                [q.terminate() for q, _ in procs if q.poll() is None]
    rcs = [p.wait() for p, _ in procs]
    for i, rc in enumerate(rcs):
        print(f"{tag} worker {i}: exit {rc}\n" + "".join(open(f"/kaggle/working/out/{tag}-w{i}.log").readlines()[-3:]))
    assert not any(rcs), f"{tag}: a worker failed; see the tail above"
    return [r for i in range(workers) for r in json.load(open(f"/kaggle/working/out/{tag}-w{i}.json"))["results"]]
def finish():
    gpu.terminate()
    with tarfile.open("/kaggle/working/head_out.tar", "w") as t: t.add("/kaggle/working/out", arcname="out")
    print(subprocess.run("ls -la /kaggle/working; du -sh /kaggle/working", shell=True, capture_output=True, text=True).stdout)'''),
("code", r'''# STAGE 0: GPU smoke on 80 decisions and 2 epochs (device placement, all three depth treatments, one screen job): catches device bugs in minutes; its numbers are discarded
try:
    sm = run_stage("smoke", [dict(seed=0, cells=[[128, t] for t in ("single", "mean", "learned")], fold=0, cell=[128, "learned"])], "smoke", 1)[0]
    print("SMOKE OK (numbers meaningless, 80 decisions x 2 epochs):", {k: round(v, 3) for k, v in sm["nll"].items()}, "peak VRAM GB", sm["peak_vram_gb"], "seconds", round(sm["seconds"]))
except BaseException:
    finish(); raise
# STAGE 1: time exactly one real TRAIN cell (proj 256, mean) on one T4: wall time, VRAM, rows/s; GPU utilisation and temperature come from gpu.csv
import statistics
TIMING_CELL = [256, "mean"]
try:
    tr = run_stage("time", [dict(seed=0, cells=[TIMING_CELL])], "time", 1)[0]
    calls = tr["train_calls"]; h0 = [c for c in calls if c["kind"] == "h0"]; rs = [c for c in calls if c["kind"] == "residual"]
    sec = lambda cs: sum(c["seconds"] for c in cs)
    rows = sum(c["epochs"] * c["n_fit"] for c in calls)
    # one job (seed, proj) = per fold: h0 bundle (final fit + 5-fold crossfit) + 3 residual fits; the timing job had 1 residual fit per fold
    job_s = sec(h0) + 3 * sec(rs); sel_wall = 9 * job_s / min(NG, 2) / 3600
    scr_job_s = (sec(h0) + sec(rs)) / 5; scr_wall = 15 * scr_job_s / min(NG, 2) / 3600
    gcsv = [l.strip().split(", ") for l in open("/kaggle/working/gpu.csv")][1:]
    g0 = [r for r in gcsv if r[1] == "0"]
    util = [float(r[3].split()[0]) for r in g0 if r[3].split()[0].replace(".", "").isdigit()]; temp = [float(r[2]) for r in g0]
    timing = dict(cell=TIMING_CELL, seed=0, wall_seconds=tr["seconds"], peak_vram_gb=tr["peak_vram_gb"], h0_calls=len(h0), residual_calls=len(rs), h0_seconds=sec(h0), residual_seconds=sec(rs), fit_rows_processed=rows,
                  rows_per_second=rows / tr["seconds"], gpu0_mean_util=statistics.mean(util) if util else None, gpu0_max_temp_c=max(temp) if temp else None, nll=tr["nll"],
                  projected_selection_wall_hours=sel_wall, projected_screen_wall_hours=scr_wall, workers_assumed=min(NG, 2),
                  versions=dict(python=sys.version.split()[0], torch=wt, device=torch.cuda.get_device_name(0)), source_commit=man["source_commit"], source_dirty_at_bundle=man["source_dirty_at_bundle"])
    json.dump(timing, open(f"/kaggle/working/out/timing-v3-{STAMP}.json", "w"), indent=1)
    print(json.dumps(timing, indent=1))
    assert sel_wall + scr_wall <= 9, f"projected selection + screen wall {sel_wall + scr_wall:.1f} h exceeds the 9 h guard; stopping after the timing stage (ask the user)"
except BaseException:
    finish(); raise'''),
("code", r'''# STAGE 2: registered selection, 9 cells x seeds {0,1,2}; jobs = (seed, projection), each trains all three depth treatments of that projection
sys.path.insert(0, W); os.chdir(W)
from decision_head import v3_selection as SEL, v3_training as T, v3_screen as SCR
try:
    jobs = [dict(seed=s, cells=[[p, t] for t in SEL.TREATMENTS]) for s in T.SELECTION_SEEDS for p in SEL.PROJ]
    res = run_stage("select", jobs, "select", min(NG, 2))
    per = {s: {} for s in T.SELECTION_SEEDS}
    for r in res:
        for k, v in r["nll"].items():
            p, t = k.split(":"); per[r["job"]["seed"]][(int(p), t)] = v
    assert all(sorted(per[s]) == sorted(SEL.CELLS) for s in T.SELECTION_SEEDS), "incomplete selection grid"
    mean = {c: sum(per[s][c] for s in T.SELECTION_SEEDS) / 3 for c in SEL.CELLS}
    CHOSEN = SEL.choose(mean)
    repro = per[0][tuple(TIMING_CELL)] - timing["nll"]["256:mean"]
    sel = dict(per_seed={str(s): {f"{p}:{t}": v for (p, t), v in per[s].items()} for s in per}, mean={f"{p}:{t}": v for (p, t), v in mean.items()}, chosen=list(CHOSEN), tie=SEL.TIE, seeds=list(T.SELECTION_SEEDS),
               final_seeds=list(T.FINAL_SEEDS), repeat_difference_timing_cell_vs_selection_job=repro, job_seconds=[r["seconds"] for r in res], wall_seconds=time.time() - T0,
               provenance=SCR.provenance(W + "/results/raw/phase2-capture-verify-20261003-145647.json", W + "/results/raw/textemb-t4-verify-20261003-155540.json", commit=man["source_commit"],
                                         extra=dict(source_dirty_at_bundle=man["source_dirty_at_bundle"], versions=dict(python=sys.version.split()[0], torch=wt, device=torch.cuda.get_device_name(0)))))
    json.dump(sel, open(f"/kaggle/working/out/selection-v3-{STAMP}.json", "w"), indent=1)
    print("selected cell:", CHOSEN, "mean NLL:", {f"{p}:{t}": round(v, 4) for (p, t), v in mean.items()}, "repeat difference on the timing cell:", repro)
except BaseException:
    finish(); raise'''),
("code", r'''# STAGE 3: the pre-LOCKED screen at the selected cell: 5 folds x seeds {0,1,2}, TRAIN CV only
try:
    jobs = [dict(fold=f, seed=s, cell=list(CHOSEN)) for f in range(5) for s in T.SELECTION_SEEDS]
    res = run_stage("screen", jobs, "screen", min(NG, 2))
    jr = {(r["job"]["fold"], r["job"]["seed"]): r["out"] for r in res}
    scr = SCR.screen_aggregate(jr, T.SELECTION_SEEDS, CHOSEN)
    scr["provenance"] = sel["provenance"]; scr["job_seconds"] = [r["seconds"] for r in res]; scr["wall_seconds"] = time.time() - T0
    scr["note"] = "go/no-go screen only (amendment e7), TRAIN CV, target = agreement with the next action; PRE_LOCKED_STOP true means do not open LOCKED"
    json.dump(scr, open(f"/kaggle/working/out/screen-v3-{STAMP}.json", "w"), indent=1)
    print(json.dumps({k: scr[k] for k in ("cell", "point", "ci95", "h0_acc", "h2_acc", "n_decisions", "n_clusters", "pre_locked_stop")}, indent=1))
    print("PRE_LOCKED_STOP =", scr["pre_locked_stop"])
finally:
    finish()'''),
]


def cell(k, s):
    b = {"metadata": {}, "source": s.splitlines(keepends=True)}
    return {"cell_type": "markdown", **b} if k == "md" else {"cell_type": "code", "execution_count": None, "outputs": [], **b}


nb = {"cells": [cell(k, s) for k, s in CELLS], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).resolve().parents[1] / "kaggle" / "head_v3_t4.ipynb"
p.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n"); print("wrote", p)
