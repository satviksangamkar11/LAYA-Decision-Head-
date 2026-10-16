"""Writes the two LOCKED notebooks (kaggle/locked_capture_t4.ipynb, kaggle/locked_textemb_t4.ipynb). NOT RUN, NOT UPLOADED: they exist so the LOCKED step is ready the moment the user approves it in writing
(amendment e10). Both read only the label-stripped bundle from tools/make_kaggle_locked_bundle.py (self-tested by tools/locked_strip_selftest.py). Same device and software lock as TRAIN and CAL (e1):
Tesla T4, fp16, torch 2.10.0+cu128 installed for the WORKER (Kaggle's image drifts), transformers 5.17.0, revision 768f209d9e. They compute no accuracy and read no labels (the bundle has none).
Run: .venv/Scripts/python.exe tools/make_kaggle_locked_notebooks.py   (overwrites; generated, not hand-edited)."""
import json
from pathlib import Path

LOCK = r'''import subprocess, sys, json, os, glob, shutil, hashlib, time, tarfile
import torch
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv", shell=True, capture_output=True, text=True).stdout)
NG = torch.cuda.device_count(); assert NG >= 1 and all("T4" in torch.cuda.get_device_name(i) for i in range(NG)), "turn on GPU T4 x2"
print("kernel torch", torch.__version__, "python", sys.version.split()[0], "GPUs", NG)
# device lock (amendment e1): the WORKER process must run torch 2.10.0+cu128. Kaggle ignores 'Pin to original environment' for committed runs, so install it for the worker.
if torch.__version__ != "2.10.0+cu128":
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu128", "--extra-index-url", "https://pypi.org/simple"], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-900:])
    r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"], capture_output=True, text=True); print(r.stdout[-300:], r.stderr[-300:])
wt = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
print("worker torch", wt); assert wt == "2.10.0+cu128", f"device lock (amendment e1): worker torch must be 2.10.0+cu128, got {wt}"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "huggingface_hub"], capture_output=True, text=True); print(r.stdout[-300:], r.stderr[-400:])
chk = subprocess.run([sys.executable, "-c", "import torch, transformers; from transformers import Qwen3ForCausalLM, AutoTokenizer; print(torch.__version__, transformers.__version__)"], capture_output=True, text=True)
print("worker preflight:", chk.stdout.strip(), chk.stderr[-1500:]); assert chk.returncode == 0 and chk.stdout.split() == ["2.10.0+cu128", "5.17.0"], "worker preflight failed (device lock e1)"'''

BUNDLE = r'''hits = [h for h in glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True) if json.load(open(h)).get("role") == "locked"]
assert len(hits) == 1, f"attach exactly one private LOCKED bundle dataset: {hits}"
SRC = os.path.dirname(hits[0]); W = "/kaggle/temp/proj"; shutil.copytree(SRC, W, dirs_exist_ok=True)
man = json.load(open(W + "/MANIFEST.json"))
bad = [f["path"] for f in man["files"] if hashlib.sha256(open(f"{W}/{f['path']}", "rb").read()).hexdigest() != f["sha256"]]; assert not bad, bad
assert man["contains_labels"] is False and man["contains_train"] is False and man["contains_cal"] is False and man["row_keys"] == ["decision_id", "state_hash"], "not the label-stripped LOCKED bundle"
print("bundle OK:", len(man["files"]), "files,", man["decisions"], "decisions,", man["trajectories"], "trajectories,", man["texts"], "texts, source commit", man["source_commit"][:8], "dirty", man["source_dirty"])'''

MODEL = r'''from huggingface_hub import snapshot_download
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
print("model OK at revision", rev["revision"][:10])'''

CAPTURE_MD = """# LOCKED capture on Kaggle T4 x2 (label-stripped bundle) - NOT to be run before the user's written approval
Same capture as TRAIN and CAL (notebook v7): fp16, resident weights, depths 18/24/30, lm_head never run, registered gate cadence 50/200. The bundle holds decision_id + state_hash and the trajectory events only (no labels, no outcomes);
this notebook computes no accuracy. Device lock e1: Tesla T4, torch 2.10.0+cu128 for the worker, transformers 5.17.0, revision 768f209d9e. Run with Save & Run All, GPU T4 x2, Internet ON, the private LOCKED bundle attached.
Output: capture_locked.tar, capture_manifest_locked.json (sha256 of every file), logs, gpu.csv."""

CAPTURE_RUN = r'''N = torch.cuda.device_count()
gpu = subprocess.Popen("nvidia-smi --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -l 30 -f /kaggle/working/gpu.csv", shell=True)
T0 = time.time(); ROLE = "locked"
print(subprocess.run([sys.executable, W + "/tools/sdpa_preflight.py"], capture_output=True, text=True, cwd=W).stdout[-1500:])

def run_role(role):
    procs = []
    for i in range(N):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONPATH=W, PYTORCH_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, f"{W}/tools/capture_pool_v3_t4.py", role, "--outdir", "out", "--model", MODEL, "--dtype", "fp16", "--resident", "--shard", f"{i}/{N}", "--require_device", "T4"]
        log = open(f"/kaggle/working/{role}-w{i}.out", "w")
        procs.append((subprocess.Popen(cmd, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(60)   # stagger: each worker loads 8 GB of fp16 weights on the CPU first
    total = len(json.load(open(f"{W}/data/v3/manifest-{role}.json"))["decision_ids"])
    t_role = time.time()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(120)
        done = len(glob.glob(f"{W}/out/{role}/*.pt"))
        rate = done / max(time.time() - t_role, 1)
        print(f"[{role}] {done}/{total} files, {(time.time() - t_role) / 60:.1f} min, ~{(total - done) / rate / 60 if rate else float('nan'):.0f} min left, alive={[p.poll() is None for p, _ in procs]}", flush=True)
        for i, (p, _) in enumerate(procs):
            if p.poll() not in (None, 0):
                print(f"--- {role} worker {i} exited {p.poll()}; tail:\n" + "".join(open(f"/kaggle/working/{role}-w{i}.out").readlines()[-30:]), flush=True)
                [q.terminate() for q, _ in procs if q.poll() is None]
    rcs = [p.wait() for p, _ in procs]
    [l.close() for _, l in procs]
    for i, rc in enumerate(rcs):
        print(f"{role} worker {i}: exit {rc}; last lines:\n" + "".join(open(f"/kaggle/working/{role}-w{i}.out").readlines()[-3:]))
    if any(rcs):
        raise RuntimeError(f"{role}: a worker failed (a failed gate aborts the run); see the tails above")

def save_outputs():
    """/kaggle/temp is NOT saved by Kaggle: whatever was captured (even after a failure) is tarred into /kaggle/working first."""
    if os.path.isdir(f"{W}/out/{ROLE}"):
        with tarfile.open(f"/kaggle/working/capture_{ROLE}.tar", "w") as t:
            t.add(f"{W}/out/{ROLE}", arcname=ROLE)
    for f in glob.glob(f"{W}/data/v3/capture-log-*-smoke-w*.jsonl") + glob.glob(f"{W}/results/raw/locked-access-log.jsonl"):
        shutil.copy(f, "/kaggle/working/")
    print(subprocess.run("ls -la /kaggle/working | head -30; du -sh /kaggle/working", shell=True, capture_output=True, text=True).stdout)

try:
    run_role(ROLE)
finally:
    gpu.terminate()
    save_outputs()'''

CAPTURE_SUMMARY = r'''import statistics
import torch
ids = json.load(open(f"{W}/data/v3/manifest-{ROLE}.json", encoding="utf-8"))["decision_ids"]
logs = [json.loads(l) for f in sorted(glob.glob(f"{W}/data/v3/capture-log-{ROLE}-smoke-w*of{N}.jsonl")) for l in open(f)]
skipped = {r["decision_id"]: r["skipped"] for r in logs if "skipped" in r}
done = [r for r in logs if "seconds" in r]
files = sorted(glob.glob(f"{W}/out/{ROLE}/*.pt"))
assert len(files) + len(skipped) == len(ids), (len(files), len(skipped), len(ids))
entries = []
for f in files:
    d = torch.load(f)
    assert d["role"] == ROLE and d["decision_id"] in set(ids) and torch.isfinite(d["query"]).all() and torch.isfinite(d["cands"]).all()
    entries.append(dict(file=os.path.basename(f), decision_id=d["decision_id"], pool_hash=d["pool_hash"], state_hash=d["state_hash"], n_prefix_tokens=d["n_prefix_tokens"], pool_size=d["pool_size"],
                        sha256=hashlib.sha256(open(f, "rb").read()).hexdigest()))
ung = [r for r in done if not r["gates"]]
dev = json.load(open(glob.glob(f"{W}/out/{ROLE}/capture-device-0of{N}.json")[0]))
assert dev["torch"] == "2.10.0+cu128" and dev["transformers"] == "5.17.0" and "T4" in dev["device"] and dev["dtype"] == "fp16", dev
summary = dict(manifest=len(ids), files=len(files), skipped=skipped, gated_decisions=len(done) - len(ung), sec_mean=statistics.mean(r["seconds"] for r in ung), peak_vram_gb=max(r.get("peak_vram_gb", 0) for r in done))
json.dump(dict(device=dev, wall_minutes=(time.time() - T0) / 60, summary=summary, files={ROLE: entries}), open("/kaggle/working/capture_manifest_locked.json", "w"), indent=1)
print(json.dumps(dict(device=dev, wall_minutes=round((time.time() - T0) / 60, 1), summary=summary), indent=1))'''

TEXT_MD = """# LOCKED-pool state-free text embeddings on Kaggle T4 - NOT to be run before the user's written approval
Same procedure and registered gate as the TRAIN+CAL embeddings (notebook v11): 'Next action:' + text, depths 18/24/30, mean over the text span, fp16 storage, shards of 2048, cosine >= 0.9999 batch-vs-alone on EVERY shard.
Texts: the LOCKED-pool texts not already in the TRAIN+CAL list (21,713; the NONE text is in the TRAIN+CAL list). The bundle carries no labels. Device lock e1: Tesla T4, fp16, torch 2.10.0+cu128, transformers 5.17.0, revision 768f209d9e.
Run with Save & Run All, GPU T4 x2 (one GPU used), Internet ON, the private LOCKED bundle attached."""

TEXT_RUN = r'''texts = json.load(open(W + "/data/texts_locked.json")); assert hashlib.sha256(json.dumps(texts).encode()).hexdigest() == man["texts_sha256"] and len(texts) == man["texts"], "text list differs from the bundle manifest"
OUT = "/kaggle/working/textemb_t4_locked"
env = dict(os.environ, PYTHONPATH=W, CUDA_VISIBLE_DEVICES="0", PYTORCH_ALLOC_CONF="expandable_segments:True")
log = open("/kaggle/working/textemb_locked.out", "w")
t0 = time.time()
p = subprocess.Popen([sys.executable, f"{W}/tools/capture_textemb_v3_t4.py", "--texts", f"{W}/data/texts_locked.json", "--outdir", OUT, "--model", MODEL, "--dtype", "fp16", "--require_device", "T4"], cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
try:
    while p.poll() is None:
        time.sleep(60)
        print(f"[textemb locked] {len(glob.glob(OUT + '/shard-*.pt'))} shards, {(time.time() - t0) / 60:.1f} min", flush=True)
    L = open("/kaggle/working/textemb_locked.out").readlines()
    print("exit", p.returncode, "\n" + "".join(L[:25] + [" ...\n"] + L[-25:]) if p.returncode else "\n" + "".join(L[-8:]))
    assert p.returncode == 0, "the embedding run failed (a failed gate aborts it); see the tail above"
finally:
    if p.poll() is None: p.terminate()
    if os.path.isdir(OUT):
        with tarfile.open("/kaggle/working/textemb_t4_locked.tar", "w") as t: t.add(OUT, arcname="textemb_t4_locked")
    print(subprocess.run("ls -la /kaggle/working; du -sh /kaggle/working", shell=True, capture_output=True, text=True).stdout)'''

TEXT_SUMMARY = r'''dev = json.load(open(OUT + "/textemb-device.json"))
assert dev["torch"] == "2.10.0+cu128" and dev["transformers"] == "5.17.0", dev
shards = sorted(glob.glob(OUT + "/shard-*.pt")); n, ents = 0, []
for f in shards:
    d = torch.load(f); n += len(d)
    assert all(v.shape == (3, 2560) and v.dtype == torch.float16 and torch.isfinite(v).all() for v in d.values())
    ents.append(dict(file=os.path.basename(f), texts=len(d), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest()))
assert n == len(texts), (n, len(texts))
last = [l for l in open("/kaggle/working/textemb_locked.out") if l.startswith("DONE")]
json.dump(dict(device=dev, texts=n, shards=ents, done_line=last[-1].strip() if last else None, wall_minutes=(time.time() - t0) / 60), open("/kaggle/working/textemb_manifest_locked.json", "w"), indent=1)
print(json.dumps(dict(device=dev, texts=n, shards=len(shards), done=last[-1].strip() if last else None), indent=1))'''


def cell(k, s):
    b = {"metadata": {}, "source": s.splitlines(keepends=True)}
    return {"cell_type": "markdown", **b} if k == "md" else {"cell_type": "code", "execution_count": None, "outputs": [], **b}


def write(name, cells):
    nb = {"cells": [cell(k, s) for k, s in cells], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    compile("\n".join(s for k, s in cells if k == "code"), name, "exec")
    p = Path(__file__).resolve().parents[1] / "kaggle" / name
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    print("wrote", p)


write("locked_capture_t4.ipynb", [("md", CAPTURE_MD), ("code", LOCK), ("code", BUNDLE), ("code", MODEL), ("code", CAPTURE_RUN), ("code", CAPTURE_SUMMARY)])
write("locked_textemb_t4.ipynb", [("md", TEXT_MD), ("code", LOCK), ("code", BUNDLE), ("code", MODEL), ("code", TEXT_RUN), ("code", TEXT_SUMMARY)])
