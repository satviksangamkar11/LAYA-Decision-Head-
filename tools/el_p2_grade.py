"""[P2] Offline evaluation of a finished P2 run (after every arm stopped): each unique candidate in <p2 dir>/CANDIDATES.jsonl graded on the development grader with the
reference engine (engine.run, WSL bubblewrap). Run INSIDE WSL. Writes <p2 dir>/GRADED.jsonl (create-only): counts and failure class only, no case inputs or expected values.
Development evidence only (the grader cases are public in Git history). Usage (inside WSL): python3 tools/el_p2_grade.py <p2 dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_ladder.execution import config as K, engine            # noqa: E402
from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate      # noqa: E402

d = Path(sys.argv[1])
out = d / "GRADED.jsonl"
if out.exists():
    sys.exit("refusing: %s exists" % out)
lang = {r["task"]: r["language"] for r in map(json.loads, (d / "RESULTS.jsonl").read_text(encoding="utf-8").splitlines())}
gate = SelectionGate(); gate.close()
grader, cfg = ProtectedGrader(gate), K.SandboxConfig()
seen = set()
with open(out, "a", encoding="utf-8") as f:
    for c in map(json.loads, (d / "CANDIDATES.jsonl").read_text(encoding="utf-8").splitlines()):
        key = (c["task"], c["id"])
        if key in seen:
            continue
        seen.add(key)
        cases = grader._cases(lang[c["task"]], c["task"])
        for _ in range(2):
            v = engine.judge(lang[c["task"]], engine.run(lang[c["task"]], c["source"].encode("utf-8"), [x["input"] for x in cases], cfg), cases, cfg)
            if v.failure_class != "sandbox_error":
                break
        row = {"task": c["task"], "id": c["id"], "dev": "INFRA_ERROR" if v.failure_class == "sandbox_error" else ("PASS" if v.status == "pass" else "FAIL"),
               "dev_passed": v.passed, "dev_total": v.total, "dev_failure": v.failure_class or None}
        f.write(json.dumps(row) + "\n"); f.flush()
        print(row["task"], row["id"][:8], row["dev"], row["dev_passed"], "/", row["dev_total"], flush=True)
