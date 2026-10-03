"""Writes kaggle/nebius_scan_cpu.ipynb: CPU-only metadata scan + stratified sample index of
nebius/SWE-rebench-openhands-trajectories (PLAN.md 13). Run it on a Kaggle CPU session (no GPU quota).
Outputs in /kaggle/working: nebius_scan_summary.json, nebius_sample_index.json."""
import json, pathlib

CELL1 = '''\
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "pyarrow"], check=True)
from huggingface_hub import hf_hub_download, snapshot_download
P = hf_hub_download("nebius/SWE-rebench-openhands-trajectories", "trajectories.parquet", repo_type="dataset")
TOK = snapshot_download("Qwen/Qwen3-4B-Thinking-2507", allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "config.json"])
print(P, TOK)
'''

CELL2 = '''\
import json, hashlib, collections, random
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MAX_TOK = 32768      # ponytail: unverified T4 limit; set from the GPU smoke test (PLAN.md 13.5)
TARGET_N = 4000      # ponytail: ~1 week of free GPU [E]; raise when real cost per trajectory is measured
PER_REPO_CAP = 12    # keeps big repos from dominating
TEST_FRAC = 0.2      # repo-disjoint test split (shortcut check 1)

f = pq.ParquetFile(P)
print(f.metadata.num_rows, "rows", f.schema_arrow.names)
tok = AutoTokenizer.from_pretrained(TOK)
CAL_N, CAL_TOK, CAL_CHARS = 150, 0, 0
rows = []
for rg in range(f.num_row_groups):
    t = f.read_row_group(rg, columns=["trajectory_id", "instance_id", "repo", "resolved", "exit_status", "trajectory"]).to_pylist()
    for r in t:
        chars = sum(len(m.get("content") or "") + sum(len(c.get("function", {}).get("arguments") or "") for c in (m.get("tool_calls") or [])) for m in r["trajectory"])
        if len(rows) < CAL_N:
            txt = "".join((m.get("content") or "") + "".join(c.get("function", {}).get("arguments") or "" for c in (m.get("tool_calls") or [])) for m in r["trajectory"])
            CAL_TOK += len(tok(txt).input_ids); CAL_CHARS += len(txt)
        rows.append(dict(id=r["trajectory_id"], inst=r["instance_id"], repo=r["repo"], resolved=int(r["resolved"]),
                         exit=r["exit_status"], turns=len(r["trajectory"]), chars=chars))
print(len(rows), "scanned")

cpt = CAL_TOK / CAL_CHARS   # tokens per char, calibrated on the first CAL_N trajectories
print("tokens per char", cpt)
for r in rows: r["tok"] = int(r["chars"] * cpt)

def q(xs, p): xs = sorted(xs); return xs[int(p * (len(xs) - 1))]
summary = dict(n=len(rows), resolved=sum(r["resolved"] for r in rows), repos=len({r["repo"] for r in rows}),
               exit=collections.Counter(r["exit"] for r in rows).most_common(8),
               tok_q={p: q([r["tok"] for r in rows], p) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
               fit={m: sum(r["tok"] <= m for r in rows) for m in (8192, 16384, 32768, 65536)},
               fit_resolved={m: sum(r["tok"] <= m and r["resolved"] for r in rows) for m in (8192, 16384, 32768, 65536)},
               tokens_per_char=cpt, tag="[E] token counts are chars x calibrated ratio")
json.dump(summary, open("/kaggle/working/nebius_scan_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# stratified sample: repo-disjoint split by hash, cap per repo, balance resolved / unresolved
def is_test(repo): return int(hashlib.sha256(repo.encode()).hexdigest(), 16) % 100 < TEST_FRAC * 100
elig = [r for r in rows if r["tok"] <= MAX_TOK]
random.seed(1); random.shuffle(elig)
per = collections.Counter(); pick = {0: [], 1: []}
for r in elig:
    if per[r["repo"]] >= PER_REPO_CAP or len(pick[r["resolved"]]) >= TARGET_N // 2: continue
    per[r["repo"]] += 1; pick[r["resolved"]].append(r)
sel = pick[0] + pick[1]
for r in sel: r["split"] = "test" if is_test(r["repo"]) else "train"
idx = dict(params=dict(MAX_TOK=MAX_TOK, TARGET_N=TARGET_N, PER_REPO_CAP=PER_REPO_CAP, TEST_FRAC=TEST_FRAC),
           n=len(sel), resolved=len(pick[1]), repos=len({r["repo"] for r in sel}),
           test_repos=len({r["repo"] for r in sel if r["split"] == "test"}), rows=sel)
json.dump(idx, open("/kaggle/working/nebius_sample_index.json", "w"))
print({k: v for k, v in idx.items() if k != "rows"})
'''

def cell(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(True)}
nb = {"cells": [cell(CELL1), cell(CELL2)], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent.parent / "kaggle" / "nebius_scan_cpu.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(out)
