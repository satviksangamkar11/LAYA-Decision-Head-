"""LOCKED capture bundle for a PRIVATE Kaggle dataset, with LABELS STRIPPED. Nothing is uploaded by this script; the upload needs the user's explicit written approval (amendment e10).
    PYTHONPATH=. .venv/Scripts/python.exe tools/make_kaggle_locked_bundle.py --out <dir outside the repo or kaggle_bundle/locked>
Why this is label-free (proven by tools/locked_strip_selftest.py, run it on the finished bundle before any upload):
  * decision rows keep ONLY decision_id and state_hash. The capture runner reads rec["state_hash"] (an assertion against the rebuilt state, a function of the events BEFORE the decision) and the
    decision id (its trajectory id and step t). label_index, label_side, next_action_kind, candidates, tier, provenance, resolved flags are dropped;
  * trajectories keep ONLY trajectory_id and trajectory (the events). model_patch, resolved, tests, instance_id, repo, tools, exit_status are dropped. The events after step t remain because the
    candidate pool is built from the whole trajectory (as for TRAIN and CAL); the pool order is a deterministic function of the events, not of a label.
Contents: capture/, compiler/, state/ sources; tools/capture_pool_v3_t4.py, tools/capture_textemb_v3_t4.py, tools/sdpa_preflight.py; the three configs; the Qwen manifest; data/v3/manifest-locked.json (ids);
data/decisions/locked-v3.jsonl and data/locked_v3/fresh-locked.jsonl (stripped, same paths as the unmodified runner expects); data/texts_locked.json (the LOCKED-pool texts NOT already in the TRAIN+CAL text list,
sorted, unique; the NONE text is in the TRAIN+CAL list); MANIFEST.json (sha256 of every file). Secret scan over everything; any hit removes the bundle. Create-only. Every read of the LOCKED files is logged."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.secret_scan import PATTERNS, scan_tree
from capture.pool import NONE_TEXT
from compiler import a5v2
from compiler.candidates import distractor_pool
from compiler.firewall import log_locked_access

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
assert not out.exists(), f"{out} exists (create-only)"
man = json.load(open(ROOT / "data/v3/manifest-locked.json", encoding="utf-8"))
ids = man["decision_ids"]
want = set(ids)
assert len(ids) == len(want) == man["n"] == 2220
log_locked_access("make_kaggle_locked_bundle: read LOCKED decision rows and trajectories to build the label-stripped capture bundle (state_hash and events only); no model outputs, no accuracies")
rows = {}
for line in open(ROOT / "data/decisions/locked-v3.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["decision_id"] in want:
        rows[r["decision_id"]] = dict(decision_id=r["decision_id"], state_hash=r["state_hash"])
assert set(rows) == want, "manifest ids missing from the locked decision rows"
tids = {d.rsplit(":", 1)[0] for d in ids}
traj, texts = {}, set()
for line in open(ROOT / "data/locked_v3/fresh-locked.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["trajectory_id"] in tids:
        traj[r["trajectory_id"]] = dict(trajectory_id=r["trajectory_id"], trajectory=r["trajectory"])
assert set(traj) == tids and len(tids) == man["trajectories"] == 444, (len(traj), len(tids))
for d in ids:
    tid, t = d.rsplit(":", 1)
    texts |= {p[0] for p in distractor_pool(traj[tid]["trajectory"], int(t))}
tv = json.load(open(ROOT / "kaggle_bundle/textemb/data/texts_trainval.json", encoding="utf-8"))
tvm = json.load(open(ROOT / "kaggle_bundle/textemb/MANIFEST.json"))
assert hashlib.sha256(json.dumps(tv).encode()).hexdigest() == tvm["texts_sha256"] and len(tv) == tvm["texts"] == 12602, "the TRAIN+CAL text list changed"
locked_texts = sorted(texts - set(tv) - {NONE_TEXT})
n_overlap = len(texts & set(tv))
out.mkdir(parents=True)


def put(rel, src=None, text=None):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if src is not None:
        shutil.copyfile(ROOT / src, p)
    else:
        p.write_text(text, encoding="utf-8", newline="\n")


for d in ("capture", "compiler", "state"):
    for f in sorted((ROOT / d).glob("*.py")):
        put(f"{d}/{f.name}", src=f"{d}/{f.name}")
for rel in ("tools/capture_pool_v3_t4.py", "tools/capture_textemb_v3_t4.py", "tools/sdpa_preflight.py", "configs/a5_v2_spec.toml", "configs/a5_v3_spec.toml", "configs/thresholds.toml",
            "results/raw/qwen3-4b-thinking-2507.manifest.json", "data/v3/manifest-locked.json"):
    put(rel, src=rel)
put("data/decisions/locked-v3.jsonl", text="".join(json.dumps(rows[d]) + "\n" for d in ids))
put("data/locked_v3/fresh-locked.jsonl", text="".join(json.dumps(traj[t]) + "\n" for t in sorted(traj)))
put("data/texts_locked.json", text=json.dumps(locked_texts))
put("results/raw/.keep", text="")
# Audited 2026-10-03 by shape only (no values read out): public-repository text. aws key = a documented EXAMPLE string; private key = the BEGIN header inside code with no key body;
# openai-style = two hyphenated lowercase word slugs; generic = an upper-case constant name, a lower-case-then-digits dummy and a word placeholder. Keyed by (pattern, sha256 of the matched value).
AUDITED_LOCKED = {("aws key", "1a5d44a2dca19669d72edf4c4f1c27c4c1ca4b4408fbb17f6ce4ad452d78ddb3"): "documented EXAMPLE string",
                  ("generic secret assignment", "4234e71501b4c8a3555c949d7559499b58eb29835afa13ad81d9b2559b0c18a7"): "word placeholder",
                  ("generic secret assignment", "836a65a7efd82a7e9b922c1b73a8e6ed87a501667b0216c1c2906f2264eb6738"): "letters-then-digits dummy",
                  ("generic secret assignment", "b7e10585ad635eb84c260107b331825d044a71980657fd31285590816c7625b2"): "upper-case constant name",
                  ("openai-style key", "65c79bb26a3bd0865c284eadce000de86477216b149a64e090bf5ded9ffec2de"): "hyphenated lowercase word slug",
                  ("openai-style key", "f73f09fd24b2f0bee4da185daf66c6f125042d6581b4ae2a9eb88fe902091153"): "hyphenated lowercase word slug",
                  ("private key", "8bcac7908eb950419537b91e19adc83ce2c9cbfdacf4f81157fdadfec11f7017"): "BEGIN header string inside code, no key body"}
hits, files, audited = scan_tree(out, AUDITED_LOCKED)
if hits:
    shutil.rmtree(out); print("SECRET SCAN FAILED (values not printed); bundle removed:", hits[:20]); sys.exit(1)
meta = dict(purpose="private Kaggle dataset: LOCKED capture bundle, labels stripped (decision_id + state_hash, events only)", role="locked", decisions=len(ids), trajectories=len(tids), contains_locked=True, contains_labels=False,
            contains_cal=False, contains_train=False, row_keys=["decision_id", "state_hash"], trajectory_keys=["trajectory_id", "trajectory"], texts=len(locked_texts), texts_overlap_with_trainval=n_overlap,
            texts_sha256=hashlib.sha256(json.dumps(locked_texts).encode()).hexdigest(), rows_sha256=hashlib.sha256((out / "data/decisions/locked-v3.jsonl").read_bytes()).hexdigest(),
            secret_scan="PASS", audited_false_positives=sorted(audited), audited_locked_notes={f"{k[0]}:{k[1][:12]}": v for k, v in AUDITED_LOCKED.items()}, patterns=list(PATTERNS), files=files, total_bytes=sum(f["bytes"] for f in files),
            source_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            source_dirty=subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "capture", "compiler", "state", "tools", "configs"], cwd=ROOT).returncode != 0,
            upload_status="NOT UPLOADED. Needs the user's explicit written approval (amendment e10).")
(out / "MANIFEST.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(f"bundle {out}: {len(files)} files, {meta['total_bytes'] / 1e6:.1f} MB, {len(ids)} decisions, {len(tids)} trajectories, {len(locked_texts)} LOCKED-only texts ({n_overlap} shared with TRAIN+CAL), secret scan PASS")
