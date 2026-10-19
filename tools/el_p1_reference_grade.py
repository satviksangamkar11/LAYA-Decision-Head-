"""[P1] Second grading column: every stored candidate of the given runs graded on the REFERENCE engine (engine.run: WSL bubblewrap, one run per candidate, the qualified
path the development grader was built on). Run INSIDE WSL. Writes <p1 dir>/REFERENCE.jsonl (create-only file), counts and failure class only.
Usage (inside WSL): python3 tools/el_p1_reference_grade.py <p1 dir> <run dir> [...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_ladder.execution import config as K, engine            # noqa: E402
from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate      # noqa: E402

out = Path(sys.argv[1]) / "REFERENCE.jsonl"
if out.exists():
    sys.exit("refusing: %s exists" % out)
gate = SelectionGate(); gate.close()
grader, cfg = ProtectedGrader(gate), K.SandboxConfig()
with open(out, "a", encoding="utf-8") as f:
    for run in sys.argv[2:]:
        for tdir in sorted((Path(run) / "tasks").iterdir()):
            if not (tdir / "DONE").exists() or not (tdir / "journal.jsonl").exists():
                continue
            task = json.loads((tdir / "result.json").read_text(encoding="utf-8"))
            cases = grader._cases(task["language"], task["task"])
            for line in (tdir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["kind"] != "candidate":
                    continue
                row = {"task": task["task"], "candidate": r["data"]["id"]}
                for _ in range(2):
                    v = engine.judge(task["language"], engine.run(task["language"], r["data"]["source"].encode("utf-8"), [c["input"] for c in cases], cfg), cases, cfg)
                    if v.failure_class != "sandbox_error":
                        break
                row.update(ref="INFRA_ERROR" if v.failure_class == "sandbox_error" else ("PASS" if v.status == "pass" else "FAIL"), ref_passed=v.passed, ref_total=v.total,
                           ref_failure=v.failure_class or None)
                f.write(json.dumps(row, sort_keys=True) + "\n"); f.flush()
                print(row["task"], row["candidate"][:8], row["ref"], row["ref_passed"], "/", row["ref_total"], row["ref_failure"], flush=True)
