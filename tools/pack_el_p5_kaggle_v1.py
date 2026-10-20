"""[P5 Kaggle smoke] Pack: code (bank-40 bundle rules: no protected grader), P5 runner + qualification + harness, frozen contract tasks + registry, selection, a PUBLIC-ONLY
LiveCodeBench file for the 6 smoke problems (statement, starter code, fn_name, public tests; NO private tests), bank sources for the 3 bank smoke tasks, model manifest,
kernel. Asserts the public-bundle rule (no held-case marker, no key-shaped string). Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/pack_el_p5_kaggle_v1.py"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_el_bank40_v1 as PK      # noqa: E402
import lcb_tasks as L               # noqa: E402

BUNDLE_DS = "%s/el-p5-bundle-v1" % PK.OWNER
SEL = json.loads((ROOT / "configs/p5_selection_v1.json").read_text("utf-8"))


def main() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    files = dict(PK.code_files(ROOT))
    man = json.loads((ROOT / "configs/el_bank80_rest_v1.json").read_text("utf-8"))
    extra = PK.SANDBOX_FILES + PK.SPECS + ["configs/backbones_v1.toml", "configs/backbones_v2.toml", "configs/el_bank80_rest_v1.json", "configs/el_arch1_limits_v1.toml",
                                           "configs/el_arch1_limits_v2.toml", "configs/critic_concurrency_v1.toml", "configs/p5_selection_v1.json", "P5_PROTOCOL_v1.md",
                                           "tools/el_p5_run.py", "tools/el_p5_contract_qual.py", "tools/el_p2_generation_compare.py", "tools/el_kaggle_port_qual.py", "tools/lcb_tasks.py", "tools/el_bank_local_run.py",
                                           "kaggle/el_bank40_run_v1.py", "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json"] + [s["file"] for s in man["sources"].values()]
    extra += [p.relative_to(ROOT).as_posix() for d in ("configs/p5_contract_tasks_v1", "configs/p5_qual_tasks_v1") for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    if "--final" in sys.argv:          # the qualification fixtures (Step 2) are not used by the final run; Q01 shares one trivial case with the contract cohort's hidden set
        extra = [r for r in extra if not r.startswith("configs/p5_qual_tasks_v1/") and r != "tools/el_p5_contract_qual.py"]
        # unused in a python-only final run: the Java pilot grader and support, and the bank pilot task lists (no bank cohort). The python sandbox scripts
        # (grade_sandboxed_v2.py: strict_equal used by judge_each; runner_v2.py: jail self-check) are runtime code and stay.
        extra = [r for r in extra if "codex-pilot1-java-20261016/" not in r and not r.endswith("codex-pilot1-20261016/fixed_v2/worker/tasks.jsonl")]
    for rel in extra:
        files[rel] = (ROOT / rel).read_bytes()
    T = L.load()
    final = "--final" in sys.argv          # FINAL bundle: the locked task set (configs/p5_final_lock_v1.json), public LCB data only
    lock_p = ROOT / "configs/p5_final_lock_v2.json"          # [P5 v2] the v2 lock (same ids and order as v1, plus protocol and grading hashes) is required
    if final and not lock_p.exists():
        sys.exit("refusing: configs/p5_final_lock_v2.json does not exist yet")
    lock = json.loads(lock_p.read_text("utf-8")) if final else None
    lcb_ids = lock["lcb_ids"] if final else SEL["smoke_lcb"]
    rows = []
    for tid in lcb_ids:
        r = json.loads(next(l for l in (ROOT / "data/livecodebench/v6/test6.jsonl").read_text("utf-8").splitlines() if json.loads(l)["question_id"] == tid))
        r["private_test_cases"] = ""          # PUBLIC ONLY: the private tests never leave the PC
        rows.append(r)
        assert T[tid]["public_view"]["public_tests"]
    assert all(r["private_test_cases"] == "" for r in rows)
    files["lcb_public_smoke.jsonl"] = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    if final:      # shards run one after the other; each writes its own run dir, so a lost session loses at most the running shard
        shards = [{"name": "final_lcb_%d" % (k // 17 + 1), "cohort": "lcb", "tasks": lcb_ids[k:k + 17]} for k in range(0, len(lcb_ids), 17)]
        shards.append({"name": "final_contract", "cohort": "contract", "tasks": lock["contract_ids"]})
        files["p5_final_lock_v2.json"] = lock_p.read_bytes()
    else:
        shards = [{"name": "smoke_lcb", "cohort": "lcb", "tasks": SEL["smoke_lcb"]}, {"name": "smoke_bank", "cohort": "bank", "tasks": SEL["smoke_bank"]}]
    if final:      # the final bundle carries both protocol documents (v1 for audit history); the smoke bundle stays as it was
        files["P5_PROTOCOL_v2.md"] = (ROOT / "P5_PROTOCOL_v2.md").read_bytes()
    files["p5_plan.json"] = json.dumps({"protocol": "P5_PROTOCOL_v2" if final else "P5_PROTOCOL_v1", "purpose": "FINAL evaluation (locked set)" if final else "SMOKE only (not the final evaluation)",
                                        "model_repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a", "shards": shards},
                                       indent=1).encode("utf-8")
    PK.B.assert_public_bundle(files, PK.held_markers(ROOT))
    for k, v in files.items():
        assert b"EXPECTED-CANARY" not in v or k == "tools/el_kaggle_port_qual.py", k
        assert not k.startswith("results/raw/p5-eval-only"), k
    files["BUNDLE_MANIFEST.json"] = json.dumps({"bundle": "el_p5_v1", "commit": commit, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                "files": {k: PK.sha(v) for k, v in files.items()},
                                                "note": "code + public task data only; no private/hidden tests, graders, references, keys or model weights"}, indent=1, sort_keys=True).encode("utf-8")
    out = ROOT / "results/raw" / (("el-p5-kaggle-final-bundle-" if final else "el-p5-kaggle-bundle-") + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ds, kd = out / "dataset", out / "kernel"
    ds.mkdir(parents=True); kd.mkdir(parents=True)
    for rel, data in files.items():
        p = ds / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "el-p5-bundle-v1", "id": BUNDLE_DS, "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    (kd / "el_p5_run_v1.py").write_bytes((ROOT / "kaggle/el_p5_run_v1.py").read_bytes())
    meta = PK.kernel_meta("el-p5-run-v1", "el_p5_run_v1.py", True, [BUNDLE_DS], [])
    meta["enable_internet"] = True
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "files": len(files), "bytes": sum(len(v) for v in files.values()), "commit": commit}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
