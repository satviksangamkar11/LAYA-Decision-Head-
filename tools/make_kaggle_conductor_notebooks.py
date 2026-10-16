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
# Patch 1 (audit 2026-10-04): the official script cuts its 400-item calibration set at random from the TRAIN rows, so rows of the same trajectory sit on both sides and the
# refit temperature is fitted on near-seen data. The notebook writes a repo-disjoint cal_items.pt instead; the script only loads it.
_a = TRAIN_SRC.index("    all_items=torch.load(")
_b = TRAIN_SRC.index("    my_items=train_items[rank::world]")
TRAIN_SRC = (TRAIN_SRC[:_a] + "    train_items=torch.load('/kaggle/working/train_items.pt',weights_only=False)\n"
             "    cal_items=torch.load('/kaggle/working/cal_items.pt',weights_only=False)\n" + TRAIN_SRC[_b:])
assert "cal_items.pt" in TRAIN_SRC and "n_cal" not in TRAIN_SRC
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
("markdown", """# Conductor data, multi-source (CPU session, no GPU quota)
Settings: Accelerator **None**, Internet **ON**. Output: `conductor_rows.jsonl`, `conductor_rows_summary.json`.
Pulls specific parquet files from four datasets, converts each with its own adapter (`tools/conductor_data.py adapt`), applies ONE label rule, and splits by repo across ALL sources together:
nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0), SWE-bench/SWE-smith-trajectories `tool` shards (MIT), nanoswe/nanoswe-trajs-261002 spread shards (MIT), JetBrains-Research/agent-trajectories-swesmith-random-subset (MIT, held out as a different-agent test).
After it finishes: **Save Version** (Save & Run All), then attach this notebook's output to the GPU notebook as an input."""),
("code", """import subprocess, sys, shutil
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow", "huggingface_hub"], check=True)
print("free GB on /kaggle/working:", shutil.disk_usage("/kaggle/working").free / 1e9, "| /kaggle/temp:", shutil.disk_usage("/kaggle/temp").free / 1e9)"""),
("code", "open('/kaggle/working/conductor_data.py', 'w', encoding='utf-8').write(" + repr(DATA_SRC) + ")\nexec(open('/kaggle/working/conductor_data.py', encoding='utf-8').read().replace('if __name__', 'if False and __name__'))"),
("code", """import json, hashlib, random, collections
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HELD_OUT = "jetbrains_mini"   # a different agent and message format: never in train, and its repos are removed from train (source-held-out test)
# name: (hf dataset, files, id column, outcome column, exit_status column or None, trajectories wanted, per-repo cap, message column)
SRC = {
 "nebius":         ("nebius/SWE-rebench-openhands-trajectories", ["trajectories.parquet"], "trajectory_id", "resolved", "exit_status", 2000, 12, "trajectory"),
 "swesmith_tool":  ("SWE-bench/SWE-smith-trajectories", [f"data/tool-0000{i}-of-00008.parquet" for i in (0, 3, 6)], "traj_id", "resolved", None, 1000, 8, "messages"),
 "nanoswe":        ("nanoswe/nanoswe-trajs-261002", [f"train-{i:05d}.parquet" for i in (0, 25, 50, 75, 100, 125, 150)], "traj_hash", "verified", "exit_status", 1000, 8, "messages"),
 "jetbrains_mini": ("JetBrains-Research/agent-trajectories-swesmith-random-subset", ["data/train-00000-of-00001.parquet"], None, "resolved", "exit_status", 500, 8, "messages"),
}
SUBMIT = {"submit", "Submitted"}      # exit_status values meaning the agent chose to finish
TEST_FRAC = 0.2                       # repo-disjoint test split over the non-held-out sources

def is_test(repo): return int(hashlib.sha256(repo.encode()).hexdigest(), 16) % 100 < TEST_FRAC * 100

paths = {n: [hf_hub_download(s[0], f, repo_type="dataset") for f in s[1]] for n, s in SRC.items()}
print({n: len(v) for n, v in paths.items()}, "files downloaded; free GB on /kaggle/working:", shutil.disk_usage("/kaggle/working").free / 1e9)

def keyof(name, idc, rec, path, rg, i):
    return str(rec.get(idc) or f"{name}:{path[-14:]}:{rg}:{i}")

def repo_of(rec):
    return rec.get("repo") or rec["instance_id"].split(".")[0].replace("__", "/", 1)

sel, report = {}, {}
rngp = random.Random(1)
for name, (ds, files, idc, oc, ec, want, cap, mc) in SRC.items():
    ms = []
    for path in paths[name]:
        f = pq.ParquetFile(path)
        have = [c for c in (idc, oc, ec, "instance_id", "repo") if c and c in f.schema_arrow.names]
        for rg in range(f.num_row_groups):
            for i, m in enumerate(f.read_row_group(rg, columns=have).to_pylist()):
                m["_key"] = keyof(name, idc, m, path, rg, i); m["_repo"] = repo_of(m); ms.append(m)
    elig = [m for m in ms if ec is None or m.get(ec) in SUBMIT]
    rngp.shuffle(elig)
    per, pick = collections.Counter(), {0: [], 1: []}
    for m in elig:
        r = int(bool(m[oc]))
        if per[m["_repo"]] >= cap or len(pick[r]) >= want // 2: continue
        per[m["_repo"]] += 1; pick[r].append(m)
    sel[name] = {m["_key"]: m for m in pick[0] + pick[1]}
    report[name] = dict(scanned=len(ms), eligible=len(elig), selected=len(sel[name]), resolved=len(pick[1]), repos=len({m["_repo"] for m in sel[name].values()}))
    print(name, report[name])

held_repos = {m["_repo"] for m in sel[HELD_OUT].values()}
for name in sel:
    if name != HELD_OUT:       # leak guard: any repo that appears in the held-out source leaves the train/test pool
        sel[name] = {k: m for k, m in sel[name].items() if m["_repo"] not in held_repos}
print("held-out repos:", len(held_repos), "| selected after removal:", {n: len(v) for n, v in sel.items()})

rng = random.Random(2)
rows = []
for name, (ds, files, idc, oc, ec, want, cap, mc) in SRC.items():
    for path in paths[name]:
        f = pq.ParquetFile(path)
        have = [c for c in (idc, oc, "instance_id", "repo", mc) if c and c in f.schema_arrow.names]
        for rg in range(f.num_row_groups):
            for i, rec in enumerate(f.read_row_group(rg, columns=have).to_pylist()):
                key = keyof(name, idc, rec, path, rg, i)
                if key not in sel[name]: continue
                rec.setdefault("instance_id", key)
                if name == "nebius":
                    a = {**rec, "source": name}
                else:
                    rec["messages"] = rec[mc]; a = adapt(rec, name)
                a["trajectory_id"] = key
                split = "test_ood" if name == HELD_OUT else ("test" if is_test(a["repo"]) else "train")
                rows += make_rows(a, rng, split)
with open("/kaggle/working/conductor_rows.jsonl", "w", encoding="utf-8") as out:
    for r in rows: out.write(json.dumps(r, ensure_ascii=False) + "\\n")
by = lambda s: [r for r in rows if r["split"] == s]
share = lambda xs: sum(r["gold"]["should_continue"]["probabilities"]["true"] for r in xs) / max(len(xs), 1)
summary = dict(sources=report, rows=len(rows), splits={s: dict(rows=len(by(s)), true_share=share(by(s)), final_rows=sum(r["is_last"] for r in by(s)), repos=len({r["repo"] for r in by(s)})) for s in ("train", "test", "test_ood")},
               rows_per_source_split={f"{n}/{s}": sum(1 for r in rows if r["source"] == n and r["split"] == s) for n in SRC for s in ("train", "test", "test_ood")},
               repo_overlap_train_test=len({r["repo"] for r in by("train")} & {r["repo"] for r in by("test")}),
               repo_overlap_train_ood=len({r["repo"] for r in by("train")} & {r["repo"] for r in by("test_ood")}),
               params=dict(HELD_OUT=HELD_OUT, TEST_FRAC=TEST_FRAC, SRC={n: dict(files=s[1], want=s[5], cap=s[6]) for n, s in SRC.items()}, TASK_CHARS=TASK_CHARS, RESP_CHARS=RESP_CHARS, OBS_CHARS=OBS_CHARS, K_MID=K_MID))
assert summary["repo_overlap_train_test"] == 0 and summary["repo_overlap_train_ood"] == 0
json.dump(summary, open("/kaggle/working/conductor_rows_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))"""),
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
SMOKE = False      # True = timing run: 600 train items, 1 epoch (measure seconds per step BEFORE the full run)
items, skipped, trunc = [], 0, 0
for r in TRAIN:
    p = r["gold"]["should_continue"]["probabilities"]
    tgt = [p["false"], p["true"]]
    seq, markers, st = build_sequence(tok, r["state"], qd, base_cfg["max_len"], base_cfg["head_max_len"], return_truncation_stats=True)
    if len(markers) != len(render_options(qd)): skipped += 1; continue
    trunc += st["truncated"]
    items.append({"ids": seq, "markers": markers, "qtype": QTYPES["noul"], "target": tgt, "label": tgt.index(max(tgt)), "repo": r["repo"]})
print(len(items), "items,", skipped, "skipped,", f"{trunc/len(items):.1%} truncated by the 1,024-token window")
is_cal = lambda repo: int(hashlib.sha256(("cal:" + repo).encode()).hexdigest(), 16) % 10 == 0     # about 10% of train repos, repo-disjoint from the training items
cal_items = [i for i in items if is_cal(i["repo"])]; train_items = [i for i in items if not is_cal(i["repo"])]
assert not ({i["repo"] for i in cal_items} & {i["repo"] for i in train_items})
if SMOKE: train_items, cal_items = train_items[:600], cal_items[:100]
print(len(train_items), "train items,", len(cal_items), "calibration items (repo-disjoint)")
torch.save(train_items, "/kaggle/working/train_items.pt"); torch.save(cal_items, "/kaggle/working/cal_items.pt")'''),
("code", "open('/kaggle/working/train_ddp.py', 'w', encoding='utf-8').write(" + repr(TRAIN_SRC) + ")"),
("code", '''import time
OUT = "/kaggle/working/laya_conductor_ft"; EPOCHS = 1 if SMOKE else 4
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

def report1(name, p, rows):
    y = [int(r["gold"]["should_continue"]["probabilities"]["true"] > 0.5) for r in rows]
    print(f"{name:46} acc={np.mean([(a >= 0.5) == b for a, b in zip(p, y)]):.3f}  auroc={roc_auc_score(y, p):.3f}  ece={ece(p, y):.3f}  n={len(y)}")

def report(name, p, rows):
    # all rows are inflated by "is this a closing summary or a tool call?"; the real question is resolved vs premature finish, so report final steps separately
    report1(name, p, rows)
    idx = [i for i, r in enumerate(rows) if r["is_last"]]
    report1(name + " [FINAL STEPS ONLY]", [p[i] for i in idx], [rows[i] for i in idx])

random.Random(3).shuffle(TEST)
T = [r for r in TEST if r["is_last"]] + [r for r in TEST if not r["is_last"]][:1500]    # ALL final-step test rows (the real question) + a sample of the rest
print("test rows used:", len(T), "| final steps:", sum(r["is_last"] for r in T), "| test repos:", len({r["repo"] for r in T}))
ft = laya.Agent(OUT, device="cuda"); report("fine-tuned conductor", score(ft, T), T)
base = laya.Agent(BASE, device="cuda"); report("original conductor (zero-shot)", score(base, T), T)
# control 1: state-blind baseline (task text only, no last_response)
vec = TfidfVectorizer(max_features=20000).fit([r["state"]["task"] for r in TRAIN])
lr = LogisticRegression(max_iter=1000).fit(vec.transform([r["state"]["task"] for r in TRAIN]), [int(r["gold"]["should_continue"]["probabilities"]["true"] > 0.5) for r in TRAIN])
report("state-blind (task-only TF-IDF + LR)", lr.predict_proba(vec.transform([r["state"]["task"] for r in T]))[:, 1], T)
# control 2: shuffled-state: last_response shuffled across test rows, labels kept
pair = [(r["state"]["last_response"], r["state"]["last_observation"]) for r in T]; random.Random(4).shuffle(pair)
S = [dict(r, state={"last_response": a, "last_observation": o, "task": r["state"]["task"]}) for r, (a, o) in zip(T, pair)]
report("fine-tuned, response+observation shuffled", score(ft, S), T)
# held-out SOURCE (a different agent and message format, its repos removed from train) and per-source scores on the repo-disjoint test split
OOD = [r for r in ROWS if r["split"] == "test_ood"]
if OOD:
    report("fine-tuned on held-out source", score(ft, OOD), OOD)
    report("original conductor on held-out source", score(base, OOD), OOD)
for src in sorted({r["source"] for r in T}):
    sub = [r for r in T if r["source"] == src]
    if sum(r["is_last"] for r in sub) >= 20 and sum(not r["is_last"] for r in sub) >= 1: report(f"fine-tuned, source={src}", score(ft, sub), sub)
print("A head that does not beat BOTH controls has learned nothing about the state.")'''),
]

write("conductor_data_cpu.ipynb", CPU)
write("conductor_finetune_t4.ipynb", GPU)
