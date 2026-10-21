"""[P6] Merge + completeness check of the final run (P6_DEPLOYMENT_v1.md "After the run"), before any grading. Reads only run outputs and the lock.
Every lock task must appear exactly once across the run dirs, with RESULTS rows A, B, V2, _task; an ANSWERS row per arm; a CANDIDATE_USE row and >= 1
CANDIDATES row; Codeforces tasks an IDENTITY row (identical). Each run dir's RUN_CONFIG tool_sha256 must equal the lock's runner_sha256.
Writes <out> (create-only) with the per-task status and the run dir of every task. Usage: python tools/p6_check_complete.py <lock> <out json> <run dir>..."""
import json
import sys
from collections import defaultdict
from pathlib import Path

lock_p, out_p, dirs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(d) for d in sys.argv[3:]]
if out_p.exists():
    sys.exit("refusing: %s exists" % out_p)
lock = json.loads(lock_p.read_text("utf-8"))
rows = lambda p: [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()] if p.exists() else []
want = {t: "cf" for t in lock["cf"]["final_ids"]} | {t: "contract" for t in lock["contract"]["ids"]}
where, problems = defaultdict(list), []
per = {}
for d in dirs:
    cfg = json.loads((d / "RUN_CONFIG.json").read_text("utf-8"))
    if cfg["tool_sha256"] != lock["runner_sha256"]:
        problems.append("%s: runner sha differs from the lock" % d.name)
    if not (d / "DONE.json").exists():
        problems.append("%s: no DONE.json" % d.name)
    res, ans, use, cand, idn = rows(d / "RESULTS.jsonl"), rows(d / "ANSWERS.jsonl"), rows(d / "CANDIDATE_USE.jsonl"), rows(d / "CANDIDATES.jsonl"), rows(d / "IDENTITY.jsonl")
    for t in cfg["tasks"]:
        where[t].append(d.name)
        arms = sorted(r["arm"] for r in res if r["task"] == t)
        a_arms = sorted(r["arm"] for r in ans if r["task"] == t)
        i = [r for r in idn if r["task"] == t]
        ok = {"results_arms": arms == ["A", "B", "V2", "_task"], "answer_arms": a_arms == ["A", "B", "V2"],
              "candidate_use": sum(r["task"] == t for r in use) == 1, "candidates": sum(r["task"] == t for r in cand) >= 1,
              "identity": (len(i) == 1 and i[0]["identical"]) if want.get(t) == "cf" else not i}
        per[t] = {"run": d.name, **ok, "complete": all(ok.values())}
extra = sorted(set(where) - set(want))
missing = sorted(set(want) - set(where))
dup = sorted(t for t, w in where.items() if len(w) > 1)
incomplete = sorted(t for t, p in per.items() if not p["complete"])
rep = {"lock": str(lock_p), "tasks_expected": len(want), "tasks_found": len(where), "missing": missing, "extra": extra, "duplicated": dup,
       "incomplete": incomplete, "run_problems": problems, "complete": not (missing or extra or dup or incomplete or problems), "per_task": per}
out_p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in rep.items() if k != "per_task"}, indent=1))
sys.exit(0 if rep["complete"] else 1)
