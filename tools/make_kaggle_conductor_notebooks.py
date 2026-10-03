"""Writes the two Kaggle notebooks for the Laya-conductor fine-tune (PLAN.md 13.7):
  kaggle/conductor_data_cpu.ipynb      CPU session (no GPU quota): download nebius parquet, select trajectories, build conductor rows.
  kaggle/conductor_finetune_t4.ipynb   GPU T4 x2: conductor weights as base, official training script, eval + shortcut checks.
Run: .venv/Scripts/python.exe tools/make_kaggle_conductor_notebooks.py   (generated, not hand-edited)
The internal head and the Qwen backbone are NOT used here: the conductor reads text (<= 1,024 tokens), not Qwen hidden states."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_SRC = (ROOT / "tools" / "conductor_data.py").read_text(encoding="utf-8")
# the official training script (cell 7 of the shipped notebook; its preprocessing and eval cells are corrupted, so only this cell is reused)
nb = json.loads((ROOT / "models" / "laya-conductor" / "conductor_finetune_colab.ipynb").read_text(encoding="utf-8"))
TRAIN_SRC = "".join(nb["cells"][7]["source"]).split("\n", 1)[1]          # drop the %%writefile line
assert "def main()" in TRAIN_SRC and "proper_reward" in TRAIN_SRC
REV, SHA = "04116e6808345ee4aa9477afa5e405cd48ab8421", "3c8bed1ad91e90151b38fefb3a5625e02307ce491f1eb11659bcdde05e4f1d7e"

def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c

def write(name, cells):
    out = ROOT / "kaggle" / name
    out.write_text(json.dumps({"cells": [cell(*c) for c in cells], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
    print(out)

# ---------------------------------------------------------------- CPU notebook
CPU = [
("markdown", """# Conductor data (CPU session, no GPU quota)
Settings: Accelerator **None**, Internet **ON**. Output: `conductor_rows.jsonl`, `conductor_rows_summary.json`.
Downloads nebius/SWE-rebench-openhands-trajectories (2.08 GB, CC-BY-4.0), picks trajectories (exit_status == submit, per-repo cap, resolved/unresolved balanced, repo-disjoint train/test by repo hash), writes `should_continue` rows (PLAN.md 13.7).
After it finishes: **Save Version** (Save & Run All), then attach this notebook's output to the GPU notebook as an input."""),
("code", """import subprocess, sys, shutil
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow", "huggingface_hub"], check=True)
from huggingface_hub import hf_hub_download
P = hf_hub_download("nebius/SWE-rebench-openhands-trajectories", "trajectories.parquet", repo_type="dataset")
print(P, "free GB on /kaggle/working:", shutil.disk_usage("/kaggle/working").free / 1e9, "| /kaggle/temp:", shutil.disk_usage("/kaggle/temp").free / 1e9)"""),
("code", "open('/kaggle/working/conductor_data.py', 'w', encoding='utf-8').write(" + repr(DATA_SRC) + ")\nexec(open('/kaggle/working/conductor_data.py', encoding='utf-8').read().replace('if __name__', 'if False and __name__'))"),
("code", '''import json, hashlib, random, collections
import pyarrow.parquet as pq

TARGET_TRAJ = 4000     # ponytail: about 12k rows; the official recipe used 4,105 items. Raise once the GPU run time is measured.
PER_REPO_CAP = 12
TEST_FRAC = 0.2        # repo-disjoint test split (shortcut check 1)

f = pq.ParquetFile(P)
print(f.metadata.num_rows, "rows;", f.num_row_groups, "row groups")
meta = []
for rg in range(f.num_row_groups):
    t = f.read_row_group(rg, columns=["trajectory_id", "repo", "resolved", "exit_status"]).to_pylist()
    meta += t
print(len(meta), "scanned;", collections.Counter((m["exit_status"] == "submit", int(m["resolved"])) for m in meta))

def is_test(repo): return int(hashlib.sha256(repo.encode()).hexdigest(), 16) % 100 < TEST_FRAC * 100
elig = [m for m in meta if m["exit_status"] == "submit"]
random.Random(1).shuffle(elig)
per, pick = collections.Counter(), {0: {}, 1: {}}
for m in elig:
    r = int(m["resolved"])
    if per[m["repo"]] >= PER_REPO_CAP or len(pick[r]) >= TARGET_TRAJ // 2: continue
    per[m["repo"]] += 1; pick[r][m["trajectory_id"]] = m["repo"]
sel = {**pick[0], **pick[1]}
print("selected", len(sel), "trajectories, resolved", len(pick[1]), "repos", len(set(sel.values())))

rng = random.Random(2)
rows = []
for rg in range(f.num_row_groups):
    for r in f.read_row_group(rg, columns=["trajectory_id", "repo", "resolved", "trajectory"]).to_pylist():
        if r["trajectory_id"] in sel:
            rows += make_rows(r, rng, "test" if is_test(r["repo"]) else "train")
with open("/kaggle/working/conductor_rows.jsonl", "w", encoding="utf-8") as out:
    for r in rows: out.write(json.dumps(r, ensure_ascii=False) + "\\n")
tr = [r for r in rows if r["split"] == "train"]; te = [r for r in rows if r["split"] == "test"]
share = lambda xs: sum(r["gold"]["should_continue"]["probabilities"]["true"] for r in xs) / max(len(xs), 1)
summary = dict(rows=len(rows), train=len(tr), test=len(te), train_true_share=share(tr), test_true_share=share(te),
               train_repos=len({r["repo"] for r in tr}), test_repos=len({r["repo"] for r in te}),
               repo_overlap=len({r["repo"] for r in tr} & {r["repo"] for r in te}),
               params=dict(TARGET_TRAJ=TARGET_TRAJ, PER_REPO_CAP=PER_REPO_CAP, TEST_FRAC=TEST_FRAC, TASK_CHARS=TASK_CHARS, RESP_CHARS=RESP_CHARS, K_MID=K_MID))
assert summary["repo_overlap"] == 0
json.dump(summary, open("/kaggle/working/conductor_rows_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))'''),
]

# ---------------------------------------------------------------- GPU notebook
GPU = [
("markdown", """# Fine-tune the Laya conductor (GPU T4 x2)
Base = `mvilacad/laya-conductor` (pinned revision, weights sha256 checked). Trained on `should_continue` rows from the CPU notebook, official recipe (DDP, AdamW 2.5e-5 / 1e-4, 4 epochs, proper-scoring RL + CE loss, temperature refit).
Settings: Accelerator **GPU T4 x2**, Internet **ON**, input = the output of `conductor_data_cpu`. The internal head and Qwen are not used. Nothing here is a result until the eval cell prints it."""),
("code", """import subprocess, sys, os, json, glob, hashlib, shutil
import torch
print("devices", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.device_count() >= 1, "turn on a GPU accelerator"
assert all("T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.3.22", "scikit-learn", "huggingface_hub"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-500:])
import laya; print("laya", laya.__version__)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""),
("code", f'''from huggingface_hub import snapshot_download
REV = "{REV}"; SHA = "{SHA}"
try:
    from kaggle_secrets import UserSecretsClient
    HF = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF = None
BASE = snapshot_download("mvilacad/laya-conductor", revision=REV, local_dir="/kaggle/temp/laya-conductor", token=HF, ignore_patterns=["checkpoint_latest/*"])
h = hashlib.sha256()
with open(BASE + "/model.safetensors", "rb") as fh:
    for c in iter(lambda: fh.read(1 << 24), b""): h.update(c)
assert h.hexdigest() == SHA, h.hexdigest()
print("conductor base OK", BASE, os.listdir(BASE))
hits = glob.glob("/kaggle/input/**/conductor_rows.jsonl", recursive=True)
assert hits, "attach the output of the conductor_data_cpu notebook as an input"
ROWS = [json.loads(l) for l in open(hits[0], encoding="utf-8")]
TRAIN = [r for r in ROWS if r["split"] == "train"]; TEST = [r for r in ROWS if r["split"] == "test"]
assert not ({{r["repo"] for r in TRAIN}} & {{r["repo"] for r in TEST}}), "repo overlap between train and test"
print(len(TRAIN), "train rows,", len(TEST), "test rows")'''),
("code", '''# preprocess: rows -> laya training items (official format: ids, markers, qtype, target, label); TRAIN split only
import torch
from transformers import AutoTokenizer
from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence, render_options, QTYPES
_fix_tokenizer_config(BASE)
tok = AutoTokenizer.from_pretrained(BASE + "/tokenizer")
base_cfg = json.load(open(BASE + "/rl_agent_config.json"))
CONT_Q = {"should_continue": {"type": "noul", "instructions": (
    "Did the assistant stop in the middle of the task with work clearly left undone? "
    "TRUE only if the response says it will do more next, lists remaining steps it has not done, "
    "or ends mid-implementation. "
    "FALSE if it reports the task as finished, asks a question, or reaches a natural stopping point.")}}   # identical to the conductor's own question
q = CONT_Q["should_continue"]; qd = {"t": "noul", "ins": q["instructions"], "crit": {}}
items, skipped, trunc = [], 0, 0
for r in TRAIN:
    p = r["gold"]["should_continue"]["probabilities"]
    tgt = [p["false"], p["true"]]
    seq, markers, st = build_sequence(tok, r["state"], qd, base_cfg["max_len"], base_cfg["head_max_len"], return_truncation_stats=True)
    if len(markers) != len(render_options(qd)): skipped += 1; continue
    trunc += st["truncated"]
    items.append({"ids": seq, "markers": markers, "qtype": QTYPES["noul"], "target": tgt, "label": tgt.index(max(tgt))})
print(len(items), "items,", skipped, "skipped,", f"{trunc/len(items):.1%} truncated by the 1,024-token window")
torch.save(items, "/kaggle/working/train_items.pt")'''),
("code", "open('/kaggle/working/train_ddp.py', 'w', encoding='utf-8').write(" + repr(TRAIN_SRC) + ")"),
("code", '''import time
OUT = "/kaggle/working/laya_conductor_ft"; EPOCHS = 4
n = torch.cuda.device_count()
cmd = f"torchrun --standalone --nproc_per_node={n} /kaggle/working/train_ddp.py {BASE} {OUT} {EPOCHS}"
print(cmd); t0 = time.time()
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout: print(line, end="")
assert p.wait() == 0, "training failed"
print("train seconds", time.time() - t0)'''),
("code", '''# eval on the repo-disjoint TEST split: fine-tuned vs original conductor, plus the two shortcut controls (PLAN.md 13.5)
import random, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y); e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins + (lo + 1 / bins >= 1))
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

def score(agent, rows):
    return [agent.predict(r["state"], CONT_Q)["answers"]["should_continue"]["noul"] for r in rows]   # P(TRUE) = P(should continue)

def report(name, p, rows):
    y = [int(r["gold"]["should_continue"]["probabilities"]["true"] > 0.5) for r in rows]
    print(f"{name:34} acc={np.mean([(a >= 0.5) == b for a, b in zip(p, y)]):.3f}  auroc={roc_auc_score(y, p):.3f}  ece={ece(p, y):.3f}  n={len(y)}")

random.Random(3).shuffle(TEST); T = TEST[:2000]
ft = laya.Agent(OUT, device="cuda"); report("fine-tuned conductor", score(ft, T), T)
base = laya.Agent(BASE, device="cuda"); report("original conductor (zero-shot)", score(base, T), T)
# control 1: state-blind baseline (task text only, no last_response)
vec = TfidfVectorizer(max_features=20000).fit([r["state"]["task"] for r in TRAIN])
lr = LogisticRegression(max_iter=1000).fit(vec.transform([r["state"]["task"] for r in TRAIN]), [int(r["gold"]["should_continue"]["probabilities"]["true"] > 0.5) for r in TRAIN])
report("state-blind (task-only TF-IDF + LR)", lr.predict_proba(vec.transform([r["state"]["task"] for r in T]))[:, 1], T)
# control 2: shuffled-state: last_response shuffled across test rows, labels kept
resp = [r["state"]["last_response"] for r in T]; random.Random(4).shuffle(resp)
S = [dict(r, state={"last_response": x, "task": r["state"]["task"]}) for r, x in zip(T, resp)]
report("fine-tuned, last_response shuffled", score(ft, S), T)
print("A head that does not beat BOTH controls has learned nothing about the state.")'''),
]

write("conductor_data_cpu.ipynb", CPU)
write("conductor_finetune_t4.ipynb", GPU)
