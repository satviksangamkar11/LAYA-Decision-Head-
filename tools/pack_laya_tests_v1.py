"""[LAYA-INT-1] Pack a Kaggle CPU test kernel that runs ALL existing Evidence-Ladder suites plus the new LAYA integration suite, in its OWN dataset/kernel ids (the coordinator's
test kernel is untouched). Local packing only; stdlib. Create-only. Usage: python tools/pack_laya_tests_v1.py [T4_scores.json]
It runs tools/pack_evidence_ladder_tests_v1.py (unchanged) and patches the generated copy: + suite laya_integration, + the real-score file (optional)."""
import json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
scores = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
r = subprocess.run([sys.executable, str(ROOT / "tools/pack_evidence_ladder_tests_v1.py")], capture_output=True, text=True, cwd=ROOT)
assert r.returncode == 0, r.stderr
out = Path(json.loads(r.stdout.strip().splitlines()[-1])["out"])
ds, kn = out / "dataset", out / "kernel"
man = json.loads((ds / "MANIFEST.json").read_text("utf-8"))
assert "evidence_ladder/models/tests/laya_integration_tests.py" in man["files"], "new suite not in the bundle"
import hashlib
for rel in ("capture/capture.py", "capture/capture_resident.py", "configs/laya_capture_code_pins_v2.toml"):      # v2 capture-code pins: files the real LayaScorer metadata check now hashes
    (ds / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, ds / rel)
    man["files"][rel] = hashlib.sha256((ds / rel).read_bytes()).hexdigest()
(ds / "MANIFEST.json").write_text(json.dumps(man, indent=1), "utf-8")
if scores:
    shutil.copyfile(scores, ds / "laya_t4_scores.json")
    import hashlib
    man["files"]["laya_t4_scores.json"] = hashlib.sha256((ds / "laya_t4_scores.json").read_bytes()).hexdigest()
    (ds / "MANIFEST.json").write_text(json.dumps(man, indent=1), "utf-8")
(ds / "dataset-metadata.json").write_text(json.dumps({"title": "evidence-ladder-laya-tests-bundle-v1", "id": "sangeetasangamkar/evidence-ladder-laya-tests-bundle-v1", "licenses": [{"name": "other"}]}, indent=1))
k = kn / "evidence-ladder-tests-run-v1.py"
src = k.read_text("utf-8")
line = '    "replay": [sys.executable, "evidence_ladder/testing/replay_integration.py"],\n'
assert line in src
src = src.replace(line, line + '    "laya_integration": [sys.executable, "evidence_ladder/models/tests/laya_integration_tests.py"' + (', "laya_t4_scores.json"' if scores else "") + "],\n")
(kn / "evidence-ladder-laya-tests-run-v1.py").write_text(src, "utf-8"); k.unlink()
meta = json.loads((kn / "kernel-metadata.json").read_text("utf-8"))
meta.update({"id": "sangeetasangamkar/evidence-ladder-laya-tests-run-v1", "title": "evidence-ladder-laya-tests-run-v1", "code_file": "evidence-ladder-laya-tests-run-v1.py",
             "dataset_sources": ["sangeetasangamkar/evidence-ladder-laya-tests-bundle-v1"]})
(kn / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), "utf-8")
print(json.dumps({"out": str(out), "dataset": str(ds), "kernel": str(kn), "files": len(man["files"])}))
