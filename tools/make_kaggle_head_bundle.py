"""Bundle for the Kaggle T4 head runs (timing, 3-seed selection, pre-LOCKED screen): code, configs, TRAIN-only rows and their outcome-stripped trajectories. Create-only.
    PYTHONPATH=. .venv/Scripts/python.exe tools/make_kaggle_head_bundle.py --out kaggle_bundle/headtrain
The registered loader builds the candidate sets from ALL buildable TRAIN decisions (3,907), not only the 1,986 captured ones, so every buildable TRAIN row and its trajectory is included.
No CAL rows, no LOCKED data, no captured tensors (those come from the v7 and v11 notebook outputs). EXPECTED.json holds the hashes of the built ids, the folds and the candidate sets computed
locally by the same loader code; the Kaggle worker asserts them before any training. A secret-scan hit removes the bundle."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.secret_scan import PATTERNS, scan_tree
from compiler import a5v2
from decision_head import v3_data

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
out = ROOT / args.out
assert not out.exists(), f"{out} exists (create-only)"
built = v3_data._built_train()
tids = {d["tid"] for d in built.values()}
sets = v3_data.oof_sets_multi(list(built.values()))
fold_of = a5v2.folds_by_trajectory(list(built.values()))
ids = json.load(open(ROOT / "data/v3/manifest-train.json", encoding="utf-8"))["decision_ids"]
assert set(ids) <= set(built)
expected = dict(built_ids_sha256=hashlib.sha256(json.dumps(sorted(built)).encode()).hexdigest(), n_built=len(built), n_manifest=len(ids),
                folds_sha256=hashlib.sha256(json.dumps(sorted(fold_of.items())).encode()).hexdigest(),
                sets_sha256=hashlib.sha256(json.dumps({k: sets[k] for k in sorted(sets)}, sort_keys=True).encode()).hexdigest())
out.mkdir(parents=True)


def put(rel, src=None, text=None):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if src is not None:
        shutil.copyfile(ROOT / src, p)
    else:
        p.write_text(text, encoding="utf-8", newline="\n")


for d, pat in (("capture", "*.py"), ("compiler", "*.py"), ("state", "*.py"), ("decision_head", "v3_*.py")):
    for f in sorted((ROOT / d).glob(pat)):
        put(f"{d}/{f.name}", src=f"{d}/{f.name}")
for rel in ("tools/head_v3_worker.py", "configs/a5_v2_spec.toml", "configs/a5_v3_spec.toml", "configs/thresholds.toml", "data/a5v2/odds-a5v2-odds-1.pt", "data/v3/manifest-train.json", "data/v3/manifest-cal.json",
            "results/raw/qwen3-4b-thinking-2507.manifest.json", "results/raw/phase2-capture-verify-20261003-145647.json", "results/raw/textemb-t4-verify-20261003-155540.json"):
    put(rel, src=rel)
rows = []
for line in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["split"] == "train" and r["decision_id"] in built:
        rows.append(line if line.endswith("\n") else line + "\n")
assert len(rows) == len(built) and all(json.loads(x)["split"] == "train" for x in rows)
put("data/decisions/action-swerebench-sample-v2.jsonl", text="".join(rows))
raw = []
for line in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["trajectory_id"] in tids:
        raw.append(json.dumps({k: r[k] for k in ("trajectory_id", "instance_id", "repo", "trajectory")}) + "\n")
assert len(raw) == len(tids)
put("data/raw/swerebench-sample-20261001.jsonl", text="".join(raw))
put("EXPECTED.json", text=json.dumps(expected, indent=1))
hits, files, audited = scan_tree(out)
if hits:
    shutil.rmtree(out); print("SECRET SCAN FAILED (values not printed); bundle removed:", hits[:20]); sys.exit(1)
dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "decision_head", "tools", "configs", "compiler", "capture", "state"], cwd=ROOT).returncode != 0
meta = dict(purpose="private Kaggle dataset: head timing, selection and pre-LOCKED screen on T4", role="train rows only", train_rows=len(rows), trajectories=len(tids), manifest_train=len(ids),
            contains_cal=False, contains_locked=False, contains_captured_tensors=False, secret_scan="PASS", audited_false_positives=sorted(audited), patterns=list(PATTERNS), files=files,
            expected=expected, source_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), source_dirty_at_bundle=dirty,
            total_bytes=sum(f["bytes"] for f in files))
(out / "MANIFEST.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(f"bundle {out}: {len(files)} files, {meta['total_bytes'] / 1e6:.1f} MB, {len(rows)} TRAIN rows, {len(tids)} trajectories, secret scan PASS, commit {meta['source_commit'][:8]} dirty={dirty}")
print(json.dumps(expected))
