"""Writes kaggle/phase1_smoke.ipynb (Phase-1 T4 smoke test). Run: .venv/Scripts/python.exe tools/make_kaggle_notebook.py  (overwrites the notebook; it is generated, not hand-edited)."""
import json
from pathlib import Path

MD = r"""# Phase 1: Kaggle T4 smoke test (20 TRAIN decisions)
Question: is fp16, resident-weights capture of Qwen3-4B-Thinking-2507 on a T4 numerically valid (all hard gates) and how fast is it? **T4 speed is unproven until this runs.**
Run it with **Save & Run All** (Kaggle notebooks docs: an interactive session ends after 20 idle minutes, a committed run may last up to 12 h; the GPU tips page still says 9 h, so plan for 9). Kaggle limits a dataset to 50 top-level files and unpacks uploaded ZIPs with the folder structure intact. T4 x2 = 2 GPUs, 4 CPU cores, about 29 GB RAM. Needs: accelerator `GPU T4 x2`, Internet ON, the private dataset (output of `tools/make_kaggle_bundle.py --limit 20`) attached. Two passes: A = every gate on every decision (validity),
B = registered gate cadence (clean seconds per decision). Nothing here touches CAL or LOCKED. Decision rule (registered in PLAN.md 12.3 / the chat): switch Phase 2 to the T4 only if
mean ungated seconds per decision per GPU is <= 5 AND every gate passed; <= 10 is feasible but no gain over the RTX 3050 (10.2 s); otherwise stay on the 3050."""

CELLS = [
    ("md", MD),
    ("code", r'''import subprocess, sys, json, os, glob, shutil, hashlib, time
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv", shell=True, capture_output=True, text=True).stdout)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, "turn on a GPU accelerator"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "huggingface_hub"], capture_output=True, text=True)
print(r.stdout[-400:], r.stderr[-600:])
import transformers
print("transformers", transformers.__version__)'''),
    ("code", r'''hits = glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)
assert hits, "attach the private dataset to this notebook"
SRC = os.path.dirname(hits[0]); W = "/kaggle/working/proj"
shutil.copytree(SRC, W, dirs_exist_ok=True)
man = json.load(open(W + "/MANIFEST.json"))
bad = [f["path"] for f in man["files"] if hashlib.sha256(open(f"{W}/{f['path']}", "rb").read()).hexdigest() != f["sha256"]]
assert not bad, bad
assert man["contains_cal"] is False and man["contains_locked"] is False
print("bundle OK:", len(man["files"]), "files,", man["decisions"], "TRAIN decisions, source commit", man["source_commit"][:8], "secret scan", man["secret_scan"])'''),
    ("code", r'''from huggingface_hub import snapshot_download
rev = json.load(open(W + "/results/raw/qwen3-4b-thinking-2507.manifest.json"))
try:
    from kaggle_secrets import UserSecretsClient
    HF = UserSecretsClient().get_secret("HF_TOKEN")   # optional Kaggle Secret; never stored in any file
except Exception:
    HF = None
print("hf token from Kaggle Secrets:", bool(HF))
MODEL = snapshot_download(rev["repo"], revision=rev["revision"], local_dir="/kaggle/temp/qwen3-4b", token=HF)
for f in rev["files"]:
    if f["sha256"]:
        h = hashlib.sha256()
        with open(f"{MODEL}/{f['name']}", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == f["sha256"], f["name"]
print("model OK at revision", rev["revision"][:10])'''),
    ("code", r'''exec(open(W + "/tools/sdpa_preflight.py").read())'''),
    ("code", r'''N = torch.cuda.device_count()
gpu = subprocess.Popen("nvidia-smi --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -l 5 -f /kaggle/working/gpu.csv", shell=True)

def run_pass(tag, outdir, extra):
    procs = []
    for i in range(N):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONPATH=W, PYTORCH_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, f"{W}/tools/capture_pool_v3_t4.py", "train", "--limit", "20", "--outdir", outdir, "--model", MODEL, "--dtype", "fp16", "--resident", "--shard", f"{i}/{N}"] + extra
        log = open(f"/kaggle/working/{tag}-w{i}.out", "w")
        procs.append((subprocess.Popen(cmd, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(60)   # stagger: each worker first loads 8 GB of fp16 weights on the CPU and the notebook has about 29 GB RAM
    t0 = time.time()
    rcs = [p.wait() for p, _ in procs]
    for _, l in procs:
        l.close()
    for i, rc in enumerate(rcs):
        if rc:
            print(f"--- {tag} worker {i} FAILED (exit {rc}); last lines:")
            print("".join(open(f"/kaggle/working/{tag}-w{i}.out").readlines()[-25:]))
    print(f"{tag}: exit codes {rcs}, wall {time.time() - t0:.0f} s")
    return rcs

# pass A: every gate on every decision (strictest allowed), plain-forward and repeat every 5th
rcA = run_pass("passA", "out/smoke_gated", ["--gate_every", "1", "--plain_every", "5"])
os.makedirs("/kaggle/working/gatedlogs", exist_ok=True)
for f in glob.glob(f"{W}/data/v3/capture-log-train-smoke-*.jsonl"):
    shutil.move(f, "/kaggle/working/gatedlogs/")
# pass B: registered cadence (gates only on the first 3 decisions), clean timing from the ungated rest
rcB = run_pass("passB", "out/smoke_speed", [])
gpu.terminate()'''),
    ("code", r'''import statistics, csv
def logs(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        rows += [json.loads(l) for l in open(f) if '"seconds"' in l]
    return rows
A = logs("/kaggle/working/gatedlogs/capture-log-train-smoke-*.jsonl")
B = logs(f"{W}/data/v3/capture-log-train-smoke-*.jsonl")
nA = len(glob.glob(f"{W}/out/smoke_gated/train/*.pt")); nB = len(glob.glob(f"{W}/out/smoke_speed/train/*.pt"))
gates_ok = all(rc == 0 for rc in rcA + rcB) and nA == 20 and nB == 20
ung = [r for r in B if not r["gates"]]
sec = [r["seconds"] for r in ung]
tps = [(r["prefix_tokens"] + r["suffix_tokens"]) / r["seconds"] for r in ung]
mean_s = statistics.mean(sec) if sec else float("nan")
temps = []
try:
    temps = [(int(r["temperature.gpu"]), int(r["utilization.gpu [%]"].strip().split()[0])) for r in csv.DictReader(open("/kaggle/working/gpu.csv"), skipinitialspace=True) if r.get("temperature.gpu", "").strip().isdigit()]
except Exception as e:
    print("gpu.csv parse:", e)
summary = dict(gpus=N, all_hard_gates_passed=gates_ok, files_gated_pass=nA, files_speed_pass=nB, exit_codes=rcA + rcB,
               ungated_decisions=len(sec), sec_per_decision_mean=mean_s, sec_per_decision_median=statistics.median(sec) if sec else None,
               tokens_per_sec_mean=statistics.mean(tps) if tps else None, peak_vram_gb=max([r.get("peak_vram_gb", 0) for r in A + B] or [0]),
               max_gpu_temp_c=max([t for t, _ in temps] or [0]), mean_util_pct=statistics.mean([u for _, u in temps] or [0]),
               gpu_hours_for_4775_decisions=4775 * mean_s / 3600 if sec else None)
v = "NO: a gate failed or a pass did not finish" if not gates_ok else ("SWITCH Phase 2 to the T4 (<=5 s per decision, all gates passed)" if mean_s <= 5 else
    ("FEASIBLE but no gain over the 3050 (<=10 s): stay on the 3050" if mean_s <= 10 else "NO: slower than the 3050: stay on the 3050"))
summary["verdict_by_registered_rule"] = v
json.dump(summary, open("/kaggle/working/smoke_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))'''),
    ("code", r'''R = "/kaggle/working/results_smoke"
os.makedirs(R, exist_ok=True)
for src in (f"{W}/out", "/kaggle/working/gatedlogs"):
    shutil.copytree(src, f"{R}/{os.path.basename(src)}", dirs_exist_ok=True)
for f in glob.glob(f"{W}/data/v3/capture-log-train-smoke-*.jsonl") + glob.glob("/kaggle/working/*.out") + ["/kaggle/working/gpu.csv", "/kaggle/working/smoke_summary.json"]:
    shutil.copy(f, R)
shutil.make_archive("/kaggle/working/smoke_results", "gztar", R)
print("download /kaggle/working/smoke_results.tar.gz (about 30 MB)")'''),
]


def cell(kind, src):
    base = {"metadata": {}, "source": src.splitlines(keepends=True)}
    return {"cell_type": "markdown", **base} if kind == "md" else {"cell_type": "code", "execution_count": None, "outputs": [], **base}


nb = {"cells": [cell(k, s) for k, s in CELLS], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).resolve().parents[1] / "kaggle" / "phase1_smoke.ipynb"
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
print("wrote", p)
