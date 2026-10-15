"""Pack the Evidence-Ladder test bundle for Kaggle (local packing only; NOTHING is uploaded here; an upload needs the user's yes).
Creates results/raw/evidence-ladder-tests-bundle-<UTC>/{dataset/, kernel/}. Create-only. Usage: python tools/pack_evidence_ladder_tests_v1.py [repo_root_with_the_code]
The code tree is copied from the given root (default: this repo); the index/kit worker files always come from this repo."""
import hashlib, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
out = ROOT / "results/raw" / ("evidence-ladder-tests-bundle-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
ds, kn = out / "dataset", out / "kernel"
ds.mkdir(parents=True, exist_ok=False); kn.mkdir()
files = {}

def put(rel, base):
    s = base / rel
    d = ds / rel
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(s, d)
    files[rel.replace("\\", "/")] = hashlib.sha256(d.read_bytes()).hexdigest()

for p in sorted((CODE / "evidence_ladder").rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".md", ".jsonl", ".txt"):
        put(p.relative_to(CODE), CODE)
for rel in ["specs/evidence_ladder_v1/schemas_v1.json", "specs/evidence_ladder_v1/contract/contract_lib_v1.py", "specs/evidence_ladder_v1/contract/test_contracts_v1.py",
            "configs/thresholds_evidence_ladder_v1.toml", "configs/thresholds_evidence_ladder_v1.toml.sha256"]:
    put(rel, ROOT)
IDX = ROOT / "results/raw/combined-index-v1-20261015T221945Z"
for f in ["INDEX.jsonl", "SUMMARY.json", "VALIDATION.json", "SOURCES.json", "README.md"]:
    put(IDX.relative_to(ROOT) / f, ROOT)
for base in [ROOT / "results/raw/codex-pilot1-20261016/fixed_v2/worker", ROOT / "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker"]:
    for p in sorted(base.rglob("*")):
        if p.is_file():
            put(p.relative_to(ROOT), ROOT)
man = {"bundle": "evidence_ladder_tests_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": files,
       "note": "code and public task files only; no protected tests, no keys, no models"}
(ds / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
(ds / "dataset-metadata.json").write_text(json.dumps({"title": "evidence-ladder-tests-bundle-v1", "id": "sangeetasangamkar/evidence-ladder-tests-bundle-v1", "licenses": [{"name": "other"}]}, indent=1))
shutil.copyfile(ROOT / "kaggle/evidence_ladder_tests_v1.py", kn / "evidence-ladder-tests-run-v1.py")
(kn / "kernel-metadata.json").write_text(json.dumps({"id": "sangeetasangamkar/evidence-ladder-tests-run-v1", "title": "evidence-ladder-tests-run-v1", "code_file": "evidence-ladder-tests-run-v1.py",
    "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["sangeetasangamkar/evidence-ladder-tests-bundle-v1"], "kernel_sources": [], "competition_sources": [], "model_sources": []}, indent=1))
print(json.dumps({"out": str(out), "files": len(files), "bytes": sum((ds / f).stat().st_size for f in files)}))
