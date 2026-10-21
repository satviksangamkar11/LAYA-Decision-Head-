"""[P6] Create configs/p6_final_lock_v1.json (create-only) from the frozen N rule result of the timing smoke (P6_PROTOCOL_v1 section 4) plus the two-kernel
deployment split (P6_DEPLOYMENT_v1.md: tasks only; protocol unchanged). Fails closed on any hash or ordering mismatch.
Usage: PYTHONPATH=. python tools/p6_make_lock.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/p6_final_lock_v1.json"
H = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
CF_SPLIT = 32     # kernel 1 (current account, <= 4 h): first 32 Codeforces ids; kernel 2: the other 40 + D01-D20 (P6_DEPLOYMENT_v1.md)

if OUT.exists():
    sys.exit("refusing: %s exists" % OUT)
q = json.loads((ROOT / "results/raw/p6-cf-validation-v1/QUALIFIED_rules_v2.json").read_text("utf-8"))["qualified_in_rule_order"]
ordered = sorted(q, key=lambda i: hashlib.sha256(("P6-2026-10-21:" + i).encode()).hexdigest())
smoke = json.loads((ROOT / "results/raw/p6-smoke-v1/SMOKE_TIMING.json").read_text("utf-8"))
assert len(ordered) == 72 and len(set(ordered)) == 72 and smoke["N"] == 72 and smoke["incomplete_tasks"] == {}
assert not {"2033/A", "2039/A", "2044/A"} & set(ordered), "smoke tasks must be outside the final set"
final = ordered[:smoke["N"]]
contract = ["D%02d" % k for k in range(1, 21)]
man = json.loads((ROOT / "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json").read_text("utf-8"))
lock = {"protocol": "P6_PROTOCOL_v1", "amendments": ["P6_PROTOCOL_v1.1_AMENDMENT.md (logging only)"],
        "protocol_sha256_committed": H("P6_PROTOCOL_v1.md"), "amendment_sha256": H("P6_PROTOCOL_v1.1_AMENDMENT.md"),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip(),
        "runner_sha256": H("tools/el_p6_run.py"),
        "n_rule": {k: smoke[k] for k in ("cf_mean", "dev_mean", "r_cf", "r_ct", "N", "rule_seconds")}, "smoke_receipt": "results/raw/p6-smoke-v1/SMOKE_TIMING.json",
        "cf": {"candidates_ordered": ordered, "N": smoke["N"], "final_ids": final,
               "tests_manifest_sha256": H("results/raw/p6-cf-validation-v1/p6_cf_MANIFEST.json"), "statements_sha256": H("configs/p6_cf_public_v1.jsonl")},
        "contract": {"ids": contract, "frozen_sha256_file": "configs/p6_contract_tasks_v1/FROZEN_SHA256.txt",
                     "frozen_sha256_file_sha256": H("configs/p6_contract_tasks_v1/FROZEN_SHA256.txt"),
                     "grading": {"location": "private/p6-eval-v1/cases_p6.json", "cases_sha256": "d8c57f33fd16ddeb0f0b85a9613a2cefb1ac8d60cc2387110faaeb723e4fd780"}},
        "model": {"repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
                  "manifest_sha256": H("models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json"), "files": len(man["files"])},
        "G": 6000, "arms": ["A", "B", "V2"],
        "deployment": {"doc": "P6_DEPLOYMENT_v1.md", "kernels": [
            {"name": "final_k1", "account": "current", "time_limit_s": 14100, "shards": [{"name": "final_cf_1", "cohort": "cf", "tasks": final[:CF_SPLIT]}]},
            {"name": "final_k2", "account": "second", "time_limit_s": None, "shards": [{"name": "final_cf_2", "cohort": "cf", "tasks": final[CF_SPLIT:]},
                                                                                 {"name": "final_contract", "cohort": "contract", "tasks": contract}]}]}}
assert H("private/p6-eval-v1/cases_p6.json") == lock["contract"]["grading"]["cases_sha256"], "private blind set differs from its frozen hash"
ks = [t for k in lock["deployment"]["kernels"] for s in k["shards"] for t in s["tasks"]]
assert len(ks) == len(set(ks)) == 92 and set(ks) == set(final) | set(contract), "split must cover every task exactly once"
OUT.write_text(json.dumps(lock, indent=1), encoding="utf-8")
print(json.dumps({"N": lock["cf"]["N"], "k1_cf": CF_SPLIT, "k2_cf": len(final) - CF_SPLIT, "k2_contract": 20, "commit": lock["code_commit"]}))
