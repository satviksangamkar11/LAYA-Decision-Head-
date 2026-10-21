"""[P6] Pack the Kaggle generation bundle `el_p6_v1` + kernel for the timing smoke or the final run. Code (bank-40 rules: no protected grader), the P6 runner,
the PUBLIC Codeforces statement file (statements + statement examples only), the public contract-task files, the protocol, the model manifest and p6_plan.json.
Never: grading tests, the private blind set, the Codeforces test export, keys. Usage:
  PYTHONPATH=. python tools/pack_el_p6_v1.py smoke            (Codeforces 2033/A, 2039/A, 2044/A + development P1-13, P1-21, P1-31)
  PYTHONPATH=. python tools/pack_el_p6_v1.py final <lock> <final_k1|final_k2> [owner]   (one kernel of the lock's deployment plan; needs a separate GO)"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_el_bank40_v1 as PK      # noqa: E402

SMOKE = [{"name": "smoke_cf", "cohort": "cf", "tasks": ["2033/A", "2039/A", "2044/A"]}, {"name": "smoke_dev", "cohort": "dev", "tasks": ["P1-13", "P1-21", "P1-31"]}]


def main() -> int:
    mode = sys.argv[1]
    if mode == "smoke":
        shards = SMOKE
    else:      # final <lock> <kernel name> [owner]: one kernel of the lock's deployment plan (P6_DEPLOYMENT_v1.md)
        lock = json.loads(Path(sys.argv[2]).read_text("utf-8"))
        kern = next(k for k in lock["deployment"]["kernels"] if k["name"] == sys.argv[3])
        shards, time_limit, mode = kern["shards"], kern["time_limit_s"], kern["name"]
        owner = sys.argv[4] if len(sys.argv) > 4 else PK.OWNER
    if mode == "smoke":
        time_limit, owner = None, PK.OWNER
    bundle_ds = "%s/el-p6-bundle-v1" % owner
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    files = dict(PK.code_files(ROOT))
    man = json.loads((ROOT / "configs/el_bank80_rest_v1.json").read_text("utf-8"))
    extra = PK.SANDBOX_FILES + PK.SPECS + ["configs/backbones_v1.toml", "configs/backbones_v2.toml", "configs/el_bank80_rest_v1.json", "configs/el_arch1_limits_v1.toml",
                                           "configs/el_arch1_limits_v2.toml", "configs/critic_concurrency_v1.toml", "P6_PROTOCOL_v1.md", "P6_OUTPUT_SCHEMA_v1.md", "P6_PROTOCOL_v1.1_AMENDMENT.md", "P6_DEPLOYMENT_v1.md",
                                           "configs/p6_cf_public_v1.jsonl", "tools/el_p6_run.py", "tools/p6_cf_tasks.py", "tools/lcb_tasks.py", "tools/el_p5_run.py",
                                           "tools/el_p2_generation_compare.py", "tools/el_kaggle_port_qual.py", "tools/el_bank_local_run.py", "kaggle/el_bank40_run_v1.py",
                                           "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json"]
    extra += [s["file"] for s in man["sources"].values()]
    extra += [p.relative_to(ROOT).as_posix() for p in (ROOT / "configs/p6_contract_tasks_v1").rglob("*") if p.is_file()]
    for rel in extra:
        files[rel] = (ROOT / rel).read_bytes()
    files["p6_plan.json"] = json.dumps({"protocol": "P6_PROTOCOL_v1", "mode": mode, "time_limit_s": time_limit, "shards": shards, "model_repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                                        "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a"}, indent=1).encode("utf-8")
    PK.B.assert_public_bundle(files, PK.held_markers(ROOT))
    for k in files:
        assert not k.startswith(("private/", "results/raw/p6-cf", "results/raw/p5-eval-only", "data/")), k
        assert not k.endswith((".json.gz",)), k
    files["BUNDLE_MANIFEST.json"] = json.dumps({"bundle": "el_p6_v1", "commit": commit, "mode": mode, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                "files": {k: PK.sha(v) for k, v in files.items()},
                                                "note": "code + public statements/examples + public contract tasks; no grading test, private set, key or weights"}, indent=1, sort_keys=True).encode("utf-8")
    out = ROOT / "results/raw" / ("el-p6-bundle-%s-" % mode + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ds, kd = out / "dataset", out / "kernel"; ds.mkdir(parents=True); kd.mkdir(parents=True)
    for rel, data in files.items():
        p = ds / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "el-p6-bundle-v1", "id": bundle_ds, "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    (kd / "el_p6_run_v1.py").write_bytes((ROOT / "kaggle/el_p6_run_v1.py").read_bytes())
    meta = PK.kernel_meta("el-p6-run-%s-v1" % mode.replace("_", "-"), "el_p6_run_v1.py", True, [bundle_ds], [])
    meta["id"] = "%s/%s" % (owner, meta["title"])
    meta["enable_internet"] = True
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "files": len(files), "bytes": sum(len(v) for v in files.values()), "commit": commit, "shards": [(s["name"], len(s["tasks"])) for s in shards]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
