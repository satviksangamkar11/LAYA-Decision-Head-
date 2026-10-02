"""Export a clean, minimal bundle for a private Kaggle dataset (code + pinned-model manifest + TRAIN-only data). Create-only.
    .venv/Scripts/python.exe tools/make_kaggle_bundle.py --limit 20 --out kaggle_bundle/smoke      (Phase-1 smoke: 20 TRAIN decisions)
    .venv/Scripts/python.exe tools/make_kaggle_bundle.py --out kaggle_bundle/train                 (all 1,986 TRAIN decisions)
Contents: capture/, compiler/, state/ sources; tools/capture_pool_v3_t4.py; the two config files; the Qwen manifest (revision + shard sha256; the model itself is downloaded
from Hugging Face by revision on Kaggle, not uploaded); data/v3/manifest-train.json (ids only); the TRAIN decision rows for the chosen ids; their trajectories with outcome fields
removed (only trajectory_id, instance_id, repo, trajectory are kept). CAL, LOCKED, test and OOD data are never included (the role is fixed to train, asserted on the rows).
A repository-wide-style secret scan runs over every file of the bundle; any hit aborts and nothing is left to upload. MANIFEST.json lists every file with its sha256."""
import argparse, hashlib, json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--roles", default="train", help="comma separated: train or train,cal (LOCKED is never bundled here)")
args = ap.parse_args()
ROLES = args.roles.split(",")
assert set(ROLES) <= {"train", "cal"}, "only train and cal can be bundled"
out = ROOT / args.out
assert not out.exists(), f"{out} exists (create-only)"
role_ids = {}
for role in ROLES:
    m = json.load(open(ROOT / f"data/v3/manifest-{role}.json", encoding="utf-8"))
    role_ids[role] = m["decision_ids"][:args.limit] if args.limit else m["decision_ids"]
ids = [d for role in ROLES for d in role_ids[role]]
want, tids = set(ids), {d.rsplit(":", 1)[0] for d in ids}
assert len(want) == len(ids), "a decision id appears in two roles"
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
for rel in ("tools/capture_pool_v3_t4.py", "tools/sdpa_preflight.py", "configs/a5_v2_spec.toml", "configs/a5_v3_spec.toml", "configs/thresholds.toml", "results/raw/qwen3-4b-thinking-2507.manifest.json", *[f"data/v3/manifest-{r}.json" for r in ROLES], "kaggle/phase1_smoke.ipynb", "kaggle/phase2_capture.ipynb"):
    if (ROOT / rel).exists():
        put(rel, src=rel)
(out / "results/raw").mkdir(parents=True, exist_ok=True)

rows = []
for line in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["split"] in ROLES and r["decision_id"] in want:
        assert r["decision_id"] in set(role_ids[r["split"]]), "decision row from the wrong role"
        rows.append(line if line.endswith("\n") else line + "\n")
assert len(rows) == len(want), f"decision rows found {len(rows)} of {len(want)}"
assert {json.loads(x)["split"] for x in rows} <= set(ROLES)
put("data/decisions/action-swerebench-sample-v2.jsonl", text="".join(rows))
raw = []
for line in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["trajectory_id"] in tids:
        raw.append(json.dumps({k: r[k] for k in ("trajectory_id", "instance_id", "repo", "trajectory")}) + "\n")
assert len(raw) == len(tids), f"trajectories found {len(raw)} of {len(tids)}"
put("data/raw/swerebench-sample-20261001.jsonl", text="".join(raw))

PATTERNS = {"huggingface token": r"hf_[A-Za-z0-9]{30,}", "github token": r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}", "aws key": r"AKIA[0-9A-Z]{16}",
            "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "slack token": r"xox[baprs]-[A-Za-z0-9-]{10,}", "openai-style key": r"sk-[A-Za-z0-9_-]{32,}",
            "google key": r"AIza[0-9A-Za-z_-]{35}", "kaggle key": r'"key"\s*:\s*"[0-9a-f]{32}"', "generic secret assignment": r"(?i)(api[_-]?key|secret|passwd|password|token)\s*[:=]\s*['\"][A-Za-z0-9_\-/+=]{16,}['\"]"}
BADNAMES = re.compile(r"(^|/)(\.env.*|id_rsa.*|.*\.pem|.*\.p12|kaggle\.json|credentials.*|\.netrc|.*\.kdbx)$", re.I)
# Audited false positives (2026-10-03): values inside PUBLIC repositories' trajectory text (a placeholder bearer_token, a yt-dlp constant, a tornado test password), keyed by sha256 of the value.
# Any other match still aborts the bundle; the scanner patterns are unchanged.
AUDITED = {"27ad5f3557d72992eedde7fb48d5476b2a801d2c204b65fc13804b6303eff9cb": "elastic/elastic-otel-python", "5cb73eef6081c5418a0d84d16698b55c3410a413f2fcb288f3fb68ea6c1ff810": "yt-dlp/yt-dlp",
           "d6cbb053abf2933889a0ccbf6ac244623a63a2e3397e991dde09266bdaa932d1": "tornadoweb/tornado"}
audited_seen = set()
hits, files = [], []
for p in sorted(x for x in out.rglob("*") if x.is_file()):
    rel = p.relative_to(out).as_posix()
    if BADNAMES.search(rel):
        hits.append((rel, 0, "forbidden file name"))
    for i, l in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for name, pat in PATTERNS.items():
            for m in re.finditer(pat, l):
                if name == "generic secret assignment":
                    v = re.search(r"['\"]([A-Za-z0-9_\-/+=]{16,})['\"]", m.group(0)).group(1)
                    if hashlib.sha256(v.encode()).hexdigest() in AUDITED:
                        audited_seen.add(hashlib.sha256(v.encode()).hexdigest())
                        continue
                hits.append((rel, i, name))
    files.append(dict(path=rel, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
if hits:
    shutil.rmtree(out)
    print("SECRET SCAN FAILED (values not printed); bundle removed:")
    for h in hits[:30]:
        print("  ", *h)
    sys.exit(1)
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
meta = dict(purpose="private Kaggle dataset", role=" + ".join(ROLES) + " only", decisions=len(want), decisions_per_role={r: len(v) for r, v in role_ids.items()}, trajectories=len(tids), source_commit=commit,
            contains_cal="cal" in ROLES, contains_locked=False,
            secret_scan="PASS", audited_false_positives=sorted(audited_seen), patterns=list(PATTERNS), files=files, total_bytes=sum(f["bytes"] for f in files))
(out / "MANIFEST.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(f"bundle {out}: {len(files)} files, {meta['total_bytes'] / 1e6:.1f} MB, {len(want)} decisions ({'+'.join(ROLES)}), {len(tids)} trajectories, secret scan PASS, source commit {commit[:8]}")
