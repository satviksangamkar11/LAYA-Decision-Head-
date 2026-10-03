"""Writes kaggle/textemb_t4.ipynb: state-free candidate-text embeddings of the TRAIN+CAL pool texts on one Kaggle T4 (Path A, same device lock as the T4 state capture).
Run: .venv/Scripts/python.exe tools/make_kaggle_textemb_notebook.py (overwrites; generated, not hand-edited). LOCKED-derived texts are not in the bundle."""
import json
from pathlib import Path
MD = r"""# T4 state-free text embeddings (TRAIN + CAL pool texts)
Same procedure as `tools/capture_textemb_v3.py` (3050 reference): 'Next action:' + text, depths 18/24/30, mean over the text span, fp16 storage, shards of 2048, registered gate cosine >= 0.9999 batch-vs-alone (run on EVERY shard here, stricter than registered).
Device lock as the T4 state capture: Tesla T4, fp16, torch 2.10.0+cu128, transformers 5.17.0, revision 768f209d9e. Contains NO LOCKED data. Run with Save & Run All, accelerator GPU T4 x2 (one GPU is used), Internet ON, the private dataset attached."""
CELLS = [("md", MD),
("code", r'''import subprocess, sys, json, os, glob, shutil, hashlib, time, tarfile
import torch
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv", shell=True, capture_output=True, text=True).stdout)
assert all("T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
print("torch", torch.__version__)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "huggingface_hub"], capture_output=True, text=True); print(r.stdout[-300:], r.stderr[-400:])
import transformers; assert transformers.__version__ == "5.17.0"; print("transformers", transformers.__version__)'''),
("code", r'''hits = glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True); assert hits, "attach the private dataset"
SRC = os.path.dirname(hits[0]); W = "/kaggle/temp/proj"; shutil.copytree(SRC, W, dirs_exist_ok=True)
man = json.load(open(W + "/MANIFEST.json"))
bad = [f["path"] for f in man["files"] if hashlib.sha256(open(f"{W}/{f['path']}", "rb").read()).hexdigest() != f["sha256"]]; assert not bad, bad
assert man["contains_locked"] is False and man["contains_trajectories"] is False and man["contains_labels"] is False
texts = json.load(open(W + "/data/texts_trainval.json")); assert hashlib.sha256(json.dumps(texts).encode()).hexdigest() == man["texts_sha256"] and len(texts) == man["texts"]
print("bundle OK:", len(man["files"]), "files,", len(texts), "texts, source commit", man["source_commit"][:8])'''),
("code", r'''from huggingface_hub import snapshot_download
rev = json.load(open(W + "/results/raw/qwen3-4b-thinking-2507.manifest.json"))
MODEL = snapshot_download(rev["repo"], revision=rev["revision"], local_dir="/kaggle/temp/qwen3-4b")
for f in rev["files"]:
    if f["sha256"]:
        h = hashlib.sha256()
        with open(f"{MODEL}/{f['name']}", "rb") as fh:
            for c in iter(lambda: fh.read(1 << 24), b""): h.update(c)
        assert h.hexdigest() == f["sha256"], f["name"]
print("model OK at revision", rev["revision"][:10])'''),
("code", r'''OUT = "/kaggle/working/textemb_t4"
env = dict(os.environ, PYTHONPATH=W, CUDA_VISIBLE_DEVICES="0", PYTORCH_ALLOC_CONF="expandable_segments:True")
log = open("/kaggle/working/textemb.out", "w")
t0 = time.time()
p = subprocess.Popen([sys.executable, f"{W}/tools/capture_textemb_v3_t4.py", "--texts", f"{W}/data/texts_trainval.json", "--outdir", OUT, "--model", MODEL, "--dtype", "fp16", "--require_device", "T4"], cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
try:
    while p.poll() is None:
        time.sleep(60)
        print(f"[textemb] {len(glob.glob(OUT + '/shard-*.pt'))} shards, {(time.time() - t0) / 60:.1f} min", flush=True)
    print("exit", p.returncode, "\n" + "".join(open("/kaggle/working/textemb.out").readlines()[-8:]))
    assert p.returncode == 0, "the embedding run failed (a failed gate aborts it); see the tail above"
finally:
    if p.poll() is None: p.terminate()
    if os.path.isdir(OUT):
        with tarfile.open("/kaggle/working/textemb_t4.tar", "w") as t: t.add(OUT, arcname="textemb_t4")
    print(subprocess.run("ls -la /kaggle/working; du -sh /kaggle/working", shell=True, capture_output=True, text=True).stdout)'''),
("code", r'''import statistics
dev = json.load(open(OUT + "/textemb-device.json"))
shards = sorted(glob.glob(OUT + "/shard-*.pt")); n, ents = 0, []
for f in shards:
    d = torch.load(f); n += len(d)
    assert all(v.shape == (3, 2560) and v.dtype == torch.float16 and torch.isfinite(v).all() for v in d.values())
    ents.append(dict(file=os.path.basename(f), texts=len(d), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest()))
assert n == len(texts), (n, len(texts))
last = [l for l in open("/kaggle/working/textemb.out") if l.startswith("DONE")]
json.dump(dict(device=dev, texts=n, shards=ents, done_line=last[-1].strip() if last else None, wall_minutes=(time.time() - t0) / 60), open("/kaggle/working/textemb_manifest.json", "w"), indent=1)
print(json.dumps(dict(device=dev, texts=n, shards=len(shards), done=last[-1].strip() if last else None), indent=1))''')]
def cell(k, s):
    b = {"metadata": {}, "source": s.splitlines(keepends=True)}
    return {"cell_type": "markdown", **b} if k == "md" else {"cell_type": "code", "execution_count": None, "outputs": [], **b}
nb = {"cells": [cell(k, s) for k, s in CELLS], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).resolve().parents[1] / "kaggle" / "textemb_t4.ipynb"
p.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n"); print("wrote", p)
