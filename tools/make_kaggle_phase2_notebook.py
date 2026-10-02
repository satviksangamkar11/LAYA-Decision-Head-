"""Writes kaggle/phase2_capture.ipynb: Phase-2 capture of every TRAIN and CAL decision on the Kaggle T4 x2 (fp16, resident weights, one worker per GPU, registered gate cadence 50/200).
Run: .venv/Scripts/python.exe tools/make_kaggle_phase2_notebook.py  (overwrites; generated, not hand-edited). LOCKED is not in this notebook or its bundle (needs its own written approval)."""
import json
from pathlib import Path

MD = r"""# Phase 2: capture all TRAIN and CAL decisions on Kaggle T4 x2
Same capture as the validated Phase-1 smoke test (fp16, resident weights, depths 18/24/30, lm_head never run, registered gate cadence: content-swap and shape-change gates every 50th decision, plain-forward and repeat gates every 200th, first 3 decisions gated).
All roles are captured on one device class and dtype (CLAUDE.md), so CAL is recaptured here too. A failed gate aborts a worker, the notebook then stops with the log tail. Every session asserts the GPU is a Tesla T4.
Run with **Save & Run All**, accelerator `GPU T4 x2`, Internet ON, the private dataset from `tools/make_kaggle_bundle.py --roles train,cal` attached. Output: `capture_train.tar`, `capture_cal.tar`, `capture_manifest.json` (sha256 of every file, per-file hashes), logs. Contains NO LOCKED data."""

CELLS = [
    ("md", MD),
    ("code", r'''import subprocess, sys, json, os, glob, shutil, hashlib, time
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv", shell=True, capture_output=True, text=True).stdout)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, "turn on a GPU accelerator"
assert all("T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "huggingface_hub"], capture_output=True, text=True)
print(r.stdout[-400:], r.stderr[-600:])
import transformers
print("transformers", transformers.__version__)
assert transformers.__version__ == "5.17.0"'''),
    ("code", r'''hits = glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)
assert hits, "attach the private dataset to this notebook"
SRC = os.path.dirname(hits[0]); W = "/kaggle/temp/proj"      # /kaggle/temp is not part of the saved output
shutil.copytree(SRC, W, dirs_exist_ok=True)
man = json.load(open(W + "/MANIFEST.json"))
bad = [f["path"] for f in man["files"] if hashlib.sha256(open(f"{W}/{f['path']}", "rb").read()).hexdigest() != f["sha256"]]
assert not bad, bad
assert man["contains_locked"] is False
ROLES = [r for r in ("train", "cal") if man["decisions_per_role"].get(r)]
assert ROLES == ["train", "cal"], ROLES
print("bundle OK:", len(man["files"]), "files,", man["decisions_per_role"], "decisions, source commit", man["source_commit"][:8], "secret scan", man["secret_scan"])'''),
    ("code", r'''from huggingface_hub import snapshot_download
rev = json.load(open(W + "/results/raw/qwen3-4b-thinking-2507.manifest.json"))
try:
    from kaggle_secrets import UserSecretsClient
    HF = UserSecretsClient().get_secret("HF_TOKEN")   # optional Kaggle Secret; never stored in any file
except Exception:
    HF = None
MODEL = snapshot_download(rev["repo"], revision=rev["revision"], local_dir="/kaggle/temp/qwen3-4b", token=HF)
for f in rev["files"]:
    if f["sha256"]:
        h = hashlib.sha256()
        with open(f"{MODEL}/{f['name']}", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == f["sha256"], f["name"]
print("model OK at revision", rev["revision"][:10])
exec(open(W + "/tools/sdpa_preflight.py").read())'''),
    ("code", r'''N = torch.cuda.device_count()
gpu = subprocess.Popen("nvidia-smi --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -l 30 -f /kaggle/working/gpu.csv", shell=True)
T0 = time.time()

def run_role(role):
    procs = []
    for i in range(N):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONPATH=W, PYTORCH_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, f"{W}/tools/capture_pool_v3_t4.py", role, "--outdir", "out", "--model", MODEL, "--dtype", "fp16", "--resident", "--shard", f"{i}/{N}", "--require_device", "T4"]
        log = open(f"/kaggle/working/{role}-w{i}.out", "w")
        procs.append((subprocess.Popen(cmd, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(60)   # stagger: each worker loads 8 GB of fp16 weights on the CPU first and the notebook has about 29 GB RAM
    total = len(json.load(open(f"{W}/data/v3/manifest-{role}.json"))["decision_ids"])
    t_role = time.time()
    while any(p.poll() is None for p, _ in procs):      # live progress: printed into the notebook log every 2 minutes
        time.sleep(120)
        done = len(glob.glob(f"{W}/out/{role}/*.pt"))
        rate = done / max(time.time() - t_role, 1)
        eta = (total - done) / rate / 60 if rate else float("nan")
        lastsec = []
        for f in sorted(glob.glob(f"{W}/data/v3/capture-log-{role}-smoke-w*of{N}.jsonl")):
            rows = [json.loads(l) for l in open(f) if '"seconds"' in l][-5:]
            lastsec.append(round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 1))
        print(f"[{role}] {done}/{total} files, {(time.time() - t_role) / 60:.1f} min, ~{eta:.0f} min left, alive={[p.poll() is None for p, _ in procs]}, last-5 mean s/decision per worker={lastsec}", flush=True)
        for i, (p, _) in enumerate(procs):
            if p.poll() not in (None, 0):             # fail fast: a failed gate or crash stops the other worker too
                print(f"--- {role} worker {i} exited {p.poll()}; tail:\n" + "".join(open(f"/kaggle/working/{role}-w{i}.out").readlines()[-30:]), flush=True)
                for q, _ in procs:
                    if q.poll() is None:
                        q.terminate()
    rcs = [p.wait() for p, _ in procs]
    for _, l in procs:
        l.close()
    for i, rc in enumerate(rcs):
        print(f"{role} worker {i}: exit {rc}; last lines:\n" + "".join(open(f"/kaggle/working/{role}-w{i}.out").readlines()[-3:]))
    if any(rcs):
        for i, rc in enumerate(rcs):
            if rc:
                print(f"--- {role} worker {i} FAILED, tail:\n" + "".join(open(f"/kaggle/working/{role}-w{i}.out").readlines()[-30:]))
        raise RuntimeError(f"{role}: a worker failed (a failed gate aborts the run); see the tail above")
    print(f"{role}: done in {(time.time() - T0) / 60:.1f} min since start")

def save_outputs():
    """Runs in finally: /kaggle/temp is NOT saved by Kaggle, so whatever was captured (even if a worker failed) is tarred into /kaggle/working before anything else can go wrong."""
    import tarfile
    for role in ROLES:
        if os.path.isdir(f"{W}/out/{role}"):
            with tarfile.open(f"/kaggle/working/capture_{role}.tar", "w") as t:        # fp16 tensors do not compress, so no gzip
                t.add(f"{W}/out/{role}", arcname=role)
    for f in glob.glob(f"{W}/data/v3/capture-log-*-smoke-w*.jsonl") + glob.glob(f"{W}/results/raw/cal-access-log.jsonl"):
        shutil.copy(f, "/kaggle/working/")
    print(subprocess.run("ls -la /kaggle/working | head -30; du -sh /kaggle/working", shell=True, capture_output=True, text=True).stdout)

try:
    for role in ROLES:
        run_role(role)
finally:
    gpu.terminate()
    save_outputs()'''),
    ("code", r'''import statistics
import torch
man_out, summary = {}, {}
for role in ROLES:
    ids = json.load(open(f"{W}/data/v3/manifest-{role}.json", encoding="utf-8"))["decision_ids"]
    logs = [json.loads(l) for f in sorted(glob.glob(f"{W}/data/v3/capture-log-{role}-smoke-w*of{N}.jsonl")) for l in open(f)]
    skipped = {r["decision_id"]: r["skipped"] for r in logs if "skipped" in r}
    done = [r for r in logs if "seconds" in r]
    files = sorted(glob.glob(f"{W}/out/{role}/*.pt"))
    assert len(files) + len(skipped) == len(ids), (role, len(files), len(skipped), len(ids))
    assert len({r["decision_id"] for r in done}) == len(files), "log rows and files differ"
    entries = []
    for f in files:
        d = torch.load(f)
        assert d["role"] == role and d["decision_id"] in set(ids)
        assert torch.isfinite(d["query"]).all() and torch.isfinite(d["cands"]).all()
        entries.append(dict(file=os.path.basename(f), decision_id=d["decision_id"], pool_hash=d["pool_hash"], state_hash=d["state_hash"], n_prefix_tokens=d["n_prefix_tokens"], pool_size=d["pool_size"],
                            sha256=hashlib.sha256(open(f, "rb").read()).hexdigest()))
    ung = [r for r in done if not r["gates"]]
    gated = [r for r in done if r["gates"]]
    summary[role] = dict(manifest=len(ids), files=len(files), skipped=skipped, gated_decisions=len(gated), gates_run=sorted({g for r in gated for g in r["gates"]}),
                         ungated=len(ung), sec_mean=statistics.mean(r["seconds"] for r in ung), sec_median=statistics.median(r["seconds"] for r in ung),
                         tok_per_s=sum(r["prefix_tokens"] + r["suffix_tokens"] for r in ung) / sum(r["seconds"] for r in ung), peak_vram_gb=max(r.get("peak_vram_gb", 0) for r in done))
    man_out[role] = entries
dev = json.load(open(glob.glob(f"{W}/out/train/capture-device-0of{N}.json")[0]))
json.dump(dict(device=dev, wall_minutes=(time.time() - T0) / 60, summary=summary, files=man_out), open("/kaggle/working/capture_manifest.json", "w"), indent=1)
print(json.dumps(dict(device=dev, wall_minutes=round((time.time() - T0) / 60, 1), summary=summary), indent=1))'''),
]


def cell(kind, src):
    base = {"metadata": {}, "source": src.splitlines(keepends=True)}
    return {"cell_type": "markdown", **base} if kind == "md" else {"cell_type": "code", "execution_count": None, "outputs": [], **base}


nb = {"cells": [cell(k, s) for k, s in CELLS], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).resolve().parents[1] / "kaggle" / "phase2_capture.ipynb"
p.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
print("wrote", p)
