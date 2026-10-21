"""[H2-lite] Pack the Kaggle bundle for one H2-lite GPU run (search round or confirmation). Code (bank-40 rules: no protected grader), the H2-lite runner,
the development tasks' statements + public examples (the worker task file; no grader case), the proposal files, the model manifest and h2_plan.json.
Asserts the public-bundle rule (no held-case marker, no key-shaped string). Usage:
  PYTHONPATH=. python tools/pack_el_h2lite_v1.py --phase search --configs baseline,general,coding --proposals results/raw/h2lite-editor-v1/round1_proposals.json"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_el_bank40_v1 as PK      # noqa: E402

BUNDLE_DS = "%s/el-h2lite-bundle-v1" % PK.OWNER


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True, choices=("search", "confirm")); ap.add_argument("--configs", required=True)
    ap.add_argument("--proposals", required=True)
    a = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    files = dict(PK.code_files(ROOT))
    man = json.loads((ROOT / "configs/el_bank80_rest_v1.json").read_text("utf-8"))
    extra = PK.SANDBOX_FILES + PK.SPECS + ["configs/backbones_v1.toml", "configs/backbones_v2.toml", "configs/el_bank80_rest_v1.json", "configs/el_arch1_limits_v1.toml",
                                           "configs/el_arch1_limits_v2.toml", "configs/critic_concurrency_v1.toml", "H2LITE_PLAN_v1.md",
                                           "tools/el_h2lite_run.py", "tools/el_p5_run.py", "tools/el_p2_generation_compare.py", "tools/el_kaggle_port_qual.py",
                                           "tools/el_bank_local_run.py", "kaggle/el_bank40_run_v1.py", "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json"]
    extra += [s["file"] for s in man["sources"].values()]
    props = [p for p in a.proposals.split(",") if p]
    extra += props
    for rel in extra:
        files[rel] = (ROOT / rel).read_bytes()
    files["h2_plan.json"] = json.dumps({"plan": "H2LITE_PLAN_v1", "phase": a.phase, "configs": a.configs.split(","), "proposals": props,
                                        "model_repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a"}, indent=1).encode("utf-8")
    PK.B.assert_public_bundle(files, PK.held_markers(ROOT))
    for k, v in files.items():
        assert not k.startswith(("private/", "results/raw/p5-eval-only", "configs/p6_contract_tasks_v1")), k
    files["BUNDLE_MANIFEST.json"] = json.dumps({"bundle": "el_h2lite_v1", "commit": commit, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                "files": {k: PK.sha(v) for k, v in files.items()},
                                                "note": "code + development statements/public examples + prompt proposals; no grader, hidden case, key or weights"}, indent=1, sort_keys=True).encode("utf-8")
    out = ROOT / "results/raw" / ("el-h2lite-bundle-%s-" % a.phase + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ds, kd = out / "dataset", out / "kernel"
    ds.mkdir(parents=True); kd.mkdir(parents=True)
    for rel, data in files.items():
        p = ds / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "el-h2lite-bundle-v1", "id": BUNDLE_DS, "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    (kd / "el_h2lite_v1.py").write_bytes((ROOT / "kaggle/el_h2lite_v1.py").read_bytes())
    meta = PK.kernel_meta("el-h2lite-v1", "el_h2lite_v1.py", True, [BUNDLE_DS], [])
    meta["enable_internet"] = True
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "files": len(files), "bytes": sum(len(v) for v in files.values()), "commit": commit}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
