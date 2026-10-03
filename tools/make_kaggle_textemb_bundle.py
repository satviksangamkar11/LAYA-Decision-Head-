"""Bundle for the T4 state-free text-embedding run (TRAIN + CAL pool texts only; LOCKED-derived texts are NOT included and need their own written approval).
    PYTHONPATH=. .venv/Scripts/python.exe tools/make_kaggle_textemb_bundle.py --out kaggle_bundle/textemb
Contents: code (capture/, compiler/, state/), tools/capture_textemb_v3_t4.py, configs/a5_v3_spec.toml, the Qwen manifest, the unique TRAIN+CAL candidate texts (sorted JSON list, no trajectories, no labels),
MANIFEST.json (sha256 of every file, per-role text counts, secret scan result). Create-only; a secret-scan hit removes the bundle."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.secret_scan import PATTERNS, scan_tree
from capture.pool import NONE_TEXT
from compiler import a5v2
from compiler.candidates import distractor_pool
from compiler.firewall import load_cal_calibration_only, load_dev

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
out = ROOT / args.out
assert not out.exists(), f"{out} exists (create-only)"
raw = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
per_role, union = {}, {NONE_TEXT}
for role, loader in (("train", load_dev), ("cal", load_cal_calibration_only)):
    ids = set(json.load(open(ROOT / f"data/v3/manifest-{role}.json", encoding="utf-8"))["decision_ids"])
    s = set()
    for r in loader(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"):
        if r["decision_id"] in ids:
            tid, t = r["decision_id"].rsplit(":", 1)
            s |= {p[0] for p in distractor_pool(raw[tid]["trajectory"], int(t))}
    per_role[role] = len(s); union |= s
texts = sorted(union)
out.mkdir(parents=True)
for d in ("capture", "compiler", "state"):
    for f in sorted((ROOT / d).glob("*.py")):
        (out / d).mkdir(exist_ok=True); shutil.copyfile(f, out / d / f.name)
for rel in ("tools/capture_textemb_v3_t4.py", "configs/a5_v3_spec.toml", "results/raw/qwen3-4b-thinking-2507.manifest.json"):
    (out / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, out / rel)
(out / "data").mkdir(); (out / "data/texts_trainval.json").write_text(json.dumps(texts), encoding="utf-8", newline="\n")
hits, files, audited = scan_tree(out)
if hits:
    shutil.rmtree(out); print("SECRET SCAN FAILED (values not printed); bundle removed:", hits[:20]); sys.exit(1)
meta = dict(purpose="private Kaggle dataset: state-free text embeddings on T4", texts=len(texts), texts_per_role_without_none=per_role, texts_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
            contains_locked=False, contains_trajectories=False, contains_labels=False, secret_scan="PASS", audited_false_positives=sorted(audited), patterns=list(PATTERNS), files=files,
            source_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
(out / "MANIFEST.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(f"bundle {out}: {len(files)} files, {len(texts)} unique texts (train {per_role['train']}, cal {per_role['cal']}, plus NONE), secret scan PASS")
