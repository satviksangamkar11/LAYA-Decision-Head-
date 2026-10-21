"""[P6 Codeforces validation v3] Pack the code bundle `el_ml_v1` from the multi-language sandbox branch (a worktree path) for a Kaggle CPU kernel.
Code only (bank-40 rules: no protected grader, held-case and key checks). Usage: python tools/pack_el_ml_bundle_v1.py <worktree root> <kernel .py> <kernel slug>"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_el_bank40_v1 as PK      # noqa: E402

WT, KPY, SLUG = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
commit = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
files = dict(PK.code_files(WT))
for rel in PK.SANDBOX_FILES + PK.SPECS + ["tools/el_kaggle_port_qual.py", "tools/el_multilang_qual.py", "tools/lcb_tasks.py", "kaggle/el_bank40_run_v1.py",
                                          "configs/el_arch1_limits_v1.toml", "configs/el_arch1_limits_v2.toml", "configs/backbones_v1.toml", "configs/backbones_v2.toml"]:
    files[rel] = (WT / rel).read_bytes()
PK.B.assert_public_bundle(files, PK.held_markers(ROOT))
files["BUNDLE_MANIFEST.json"] = json.dumps({"bundle": "el_ml_v1", "commit": commit, "branch": "multilang-sandbox-2026-10-21", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                            "files": {k: PK.sha(v) for k, v in files.items()}}, indent=1, sort_keys=True).encode("utf-8")
out = ROOT / "results/raw" / ("el-ml-bundle-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
ds, kd = out / "dataset", out / "kernel"; ds.mkdir(parents=True); kd.mkdir(parents=True)
for rel, data in files.items():
    p = ds / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
(ds / "dataset-metadata.json").write_text(json.dumps({"title": "el-ml-bundle-v1", "id": "%s/el-ml-bundle-v1" % PK.OWNER, "licenses": [{"name": "other"}]}, indent=1))
(kd / KPY.name).write_bytes(KPY.read_bytes())
meta = PK.kernel_meta(SLUG, KPY.name, False, ["%s/el-ml-bundle-v1" % PK.OWNER], ["%s/el-p6-cf-tests-v1" % PK.OWNER])
meta["enable_internet"] = True
(kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(json.dumps({"out": str(out), "files": len(files), "commit": commit}))
