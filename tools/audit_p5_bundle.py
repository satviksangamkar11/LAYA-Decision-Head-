"""[P5 preflight] Audit a packed P5 Kaggle bundle (dataset dir) for permitted material only. Checks:
manifest hashes; no evaluation-only file (path or name); no line of the reference solutions / brute forces / hidden-case generator (lines >= 30 chars) except shared idioms (also in 3+ bundle files, or in a
file committed before the eval-only material existed; coarse counts reported); no hidden-case input with its expected value within 300 bytes; LiveCodeBench rows = the locked ids with EMPTY private tests; contract tasks carry public tests only; no grader script;
no key-shaped string; plan shards cover exactly the locked ids + 20 contract tasks. Usage: python tools/audit_p5_bundle.py <bundle>/dataset > report.json"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ds = Path(sys.argv[1])
files = {p.relative_to(ds).as_posix(): p.read_bytes() for p in ds.rglob("*") if p.is_file() and p.name != "dataset-metadata.json"}
man = json.loads(files["BUNDLE_MANIFEST.json"])
checks = {}
checks["manifest_hashes"] = all(hashlib.sha256(files[f]).hexdigest() == h for f, h in man["files"].items()) and set(man["files"]) == set(files) - {"BUNDLE_MANIFEST.json"}
EVAL = ROOT / "results/raw/p5-eval-only-v1"
checks["no_eval_only_paths"] = not [f for f in files if "p5-eval-only" in f or Path(f).name in ("hidden_cases.json", "reference_solutions.py", "brute_force.py", "generate_hidden_cases.py")]
lines = {l.strip() for n in ("reference_solutions.py", "brute_force.py", "generate_hidden_cases.py") for l in (EVAL / n).read_text("utf-8").splitlines() if len(l.strip()) >= 30}
hits = sorted({(f, l[:60]) for f, b in files.items() for l in lines if l.encode() in b})
import subprocess
EVAL_COMMIT_TIME = int(subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%ct", "f6ab9d29"], capture_output=True, text=True).stdout.strip())
lines_elsewhere = set()          # lines that also occur in code committed BEFORE the eval-only material existed = shared idioms, not copied answers
for f in {f for f, _ in hits}:
    t = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%ct", "--diff-filter=A", "--", f], capture_output=True, text=True).stdout.split()
    if t and int(t[-1]) < EVAL_COMMIT_TIME:
        lines_elsewhere |= {l for (ff, l) in hits if ff == f}
generic = {l.strip()[:60] for l in lines if sum(l.encode() in b for b in files.values()) >= 3}       # idiom present in 3+ bundle files
suspect = [(f, l) for f, l in hits if l not in lines_elsewhere and l not in generic]
checks["no_reference_or_generator_lines"] = not suspect
hc = json.loads((EVAL / "hidden_cases.json").read_text("utf-8"))
pub = {json.dumps(t["input"], sort_keys=True) for l in (ROOT / "configs/p5_contract_tasks_v1/tasks.jsonl").read_text("utf-8").splitlines() for t in json.loads(l)["public_tests"]}
pairs = [(json.dumps(c["input"], sort_keys=True), json.dumps(c["expected"])) for cs in hc.values() for c in cs if json.dumps(c["input"], sort_keys=True) not in pub]
blob = b"\n".join(files.values())
def near(b, x, y, w=300):
    i = b.find(x)
    while i >= 0:
        if y in b[max(0, i - w): i + len(x) + w]:
            return True
        i = b.find(x, i + 1)
    return False


coarse_input_hits = sum(1 for p in pairs if len(p[0]) > 12 and p[0].encode() in blob)
pair_hits = [(f, p) for p in pairs for f, b in files.items() if p[0].encode() in b and near(b, p[0].encode(), ('"expected": ' + p[1]).encode())]
checks["no_hidden_input_expected_pairs"] = not pair_hits
lock = json.loads((ROOT / ("configs/p5_final_lock_v2.json" if "p5_final_lock_v2.json" in files else "configs/p5_final_lock_v1.json")).read_text("utf-8"))
checks["bundled_lock_matches_repo"] = ("p5_final_lock_v2.json" not in files) or json.loads(files["p5_final_lock_v2.json"]) == lock
PRIV = [ROOT / "private/p5-eval-v2/cases_v2.json", ROOT / "private/p5-eval-v3/cases_v3.json"]        # private grading sets: no case (input + expected) may be in the bundle
priv_pairs = [(json.dumps(c["input"], sort_keys=True), json.dumps(c["expected"])) for p in PRIV if p.exists() for cs in json.loads(p.read_text("utf-8")).values() for c in cs]
priv_hits = [(f, x) for x in priv_pairs if x[0] not in pub for f, b in files.items() if x[0].encode() in b and near(b, x[0].encode(), ('"expected": ' + x[1]).encode())]
checks["no_private_set_pairs"] = not priv_hits
checks["no_private_folder_files"] = not [f for f in files if f.startswith("private/") or "p5-eval-v" in f]
rows = [json.loads(l) for l in files["lcb_public_smoke.jsonl"].decode().splitlines()]
checks["lcb_rows_are_locked_ids"] = [r["question_id"] for r in rows] == lock["lcb_ids"]
checks["lcb_private_tests_empty"] = all(r["private_test_cases"] == "" for r in rows)
ct = [json.loads(l) for l in files["configs/p5_contract_tasks_v1/tasks.jsonl"].decode().splitlines()]
checks["contract_tasks_public_only"] = len(ct) == 20 and all(set(t) == {"difficulty", "language", "prompt", "public_tests", "task_id"} for t in ct)
checks["no_grader_script"] = not [f for f in files if f.endswith(("el_p5_grade.py", "protected.py")) or "grader" in Path(f).name.lower()]
KEY = re.compile(rb"nvapi-[A-Za-z0-9_\-]{20,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|BEGIN (?:RSA|OPENSSH) PRIVATE")
checks["no_key_strings"] = not [f for f, b in files.items() if KEY.search(b)]
plan = json.loads(files["p5_plan.json"])
lt = [t for s in plan["shards"] if s["cohort"] == "lcb" for t in s["tasks"]]
if "p5_final_lock_v2.json" in files:      # [P5 v2 provenance] the final bundle names v2 and carries the exact protocol bytes the lock pins
    checks["plan_names_protocol_v2"] = plan.get("protocol") == "P5_PROTOCOL_v2"
    checks["bundled_protocol_v2_matches_lock"] = "P5_PROTOCOL_v2.md" in files and hashlib.sha256(files["P5_PROTOCOL_v2.md"]).hexdigest() == lock["protocol_sha256"]
    checks["runner_stamps_protocol_v2"] = b'"protocol": "P5_PROTOCOL_v2"' in files["tools/el_p5_run.py"] and b'"protocol": "P5_PROTOCOL_v1"' not in files["tools/el_p5_run.py"]
checks["plan_covers_lock"] = lt == lock["lcb_ids"] and [t for s in plan["shards"] if s["cohort"] == "contract" for t in s["tasks"]] == lock["contract_ids"]
print(json.dumps({"bundle": str(ds), "commit": man.get("commit"), "files": len(files), "bytes": sum(map(len, files.values())), "all_ok": all(checks.values()),
                  "checks": checks, "reference_line_hits_coarse": len(hits), "reference_line_suspect": suspect[:10],
                  "hidden_inputs_present_coarse": coarse_input_hits, "hidden_input_expected_pairs": pair_hits[:5],
                  "private_set_pairs_scanned": len(priv_pairs), "private_set_pair_hits": priv_hits[:5], "shards": [(s["name"], s["cohort"], len(s["tasks"])) for s in plan["shards"]],
                  "hidden_pairs_scanned": len(pairs), "reference_lines_scanned": len(lines)}, indent=1))
sys.exit(0 if all(checks.values()) else 1)
