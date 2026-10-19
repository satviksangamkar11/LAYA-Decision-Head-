"""[P2 Kaggle replicate] Pack the P2 harness for Kaggle (2 x T4): code (as the bank-40 bundle, no protected grader), the harness, the 13 public task records,
the stored round-1 gen_calls.json of the 13 tasks (same relative paths), the model's MANIFEST_pinned.json, and the kernel. No hidden cases, graders, keys or model weights
(the kernel downloads the pinned revision inside Kaggle and checks every sha256 against the manifest; mismatch = fail closed).
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/pack_el_p2_kaggle_v1.py"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_el_bank40_v1 as PK      # noqa: E402  (code_files, sandbox files, specs, held-case markers, public-bundle assert)
import el_bank_local_run as RUN     # noqa: E402  (sha-pinned bank task loader)

OWNER = PK.OWNER
BUNDLE_DS = "%s/el-p2-bundle-v1" % OWNER
L, H, T = "results/raw/el-bank-local-20261018T2106", "results/raw/el-bank-java5hard-wslc-20261018T213058", "results/raw/el-bank-java20-wslc-20261018T211547"
TASKS = [(L, t) for t in ("P1-20", "P1-23", "P1-26", "P1-31", "P1-36", "P1-37", "P1-40")] + [(H, t) for t in ("J1-32", "J1-33", "J1-34", "J1-35")] + [(T, t) for t in ("J1-20", "J1-23")]
SHARDS = [["P1-20", "P1-23", "P1-26", "P1-31", "J1-32", "J1-33", "J1-34"], ["P1-36", "P1-37", "P1-40", "J1-35", "J1-20", "J1-23"]]


def main() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    files = dict(PK.code_files(ROOT))
    for rel in PK.SANDBOX_FILES + PK.SPECS + ["configs/backbones_v1.toml", "configs/backbones_v2.toml", "tools/el_p2_generation_compare.py", "kaggle/el_bank40_run_v1.py",
                                             "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json"]:
        files[rel] = (ROOT / rel).read_bytes()
    recs = RUN.load_task_ids([t for _, t in TASKS])
    for r in recs:
        assert r["public_tests"] and all(set(x) == {"input", "expected"} for x in r["public_tests"]), r["task_id"]
    files["p2_tasks_public.jsonl"] = ("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in recs)).encode("utf-8")
    for run, tid in TASKS:
        rel = "%s/tasks/%s/gen_calls.json" % (run, tid)
        files[rel] = (ROOT / rel).read_bytes()
    files["p2_plan.json"] = json.dumps({"tasks": [list(x) for x in TASKS], "shards": SHARDS, "budget": 6000, "round_n": 2, "profile": "qwen2.5-coder-1.5b-instruct",
                                        "model_repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a"}, indent=1).encode("utf-8")
    PK.B.assert_public_bundle(files, PK.held_markers(ROOT))
    files["BUNDLE_MANIFEST.json"] = json.dumps({"bundle": "el_p2_v1", "tasks": len(recs), "commit": commit, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                "files": {k: PK.sha(v) for k, v in files.items()}, "note": "code + public task records + stored round-1 generations; no hidden cases, graders, keys or model weights"},
                                               indent=1, sort_keys=True).encode("utf-8")
    out = ROOT / "results/raw" / ("el-p2-kaggle-bundle-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ds, kd = out / "dataset", out / "kernel"
    for d in (ds, kd):
        d.mkdir(parents=True, exist_ok=False)
    for rel, data in files.items():
        p = ds / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "el-p2-bundle-v1", "id": BUNDLE_DS, "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    (kd / "el_p2_run_v1.py").write_bytes((ROOT / "kaggle/el_p2_run_v1.py").read_bytes())
    meta = PK.kernel_meta("el-p2-run-v1", "el_p2_run_v1.py", True, [BUNDLE_DS], [])
    meta["enable_internet"] = True          # the pinned model revision is downloaded inside Kaggle and sha256-checked
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    pack = {"out": str(out), "files": len(files), "bytes": sum(len(v) for v in files.values()), "commit": commit, "manifest_sha256": PK.sha(files["BUNDLE_MANIFEST.json"])}
    (out / "PACK.json").write_text(json.dumps(pack, indent=1), encoding="utf-8")
    print(json.dumps(pack, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
