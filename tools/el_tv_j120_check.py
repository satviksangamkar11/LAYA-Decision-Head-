"""[v3 regression] The P2 J1-20 candidates that the development grader passes but trusted verifier v2 rejected (NoSuchMethodError on a specific solve return type):
run their PUBLIC examples through the real WSLC backend with the trusted verifier. Prints counts only. Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_tv_j120_check.py <p2 dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_ladder.common.interfaces import Artifact, TaskSpec          # noqa: E402
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend    # noqa: E402
from evidence_ladder.execution.trusted_verify import VERSION              # noqa: E402
import el_bank_local_run as RUN                                            # noqa: E402
import el_p2_generation_compare as P2                                      # noqa: E402

d = Path(sys.argv[1])
graded = {(g["task"], g["id"]): g["dev"] for g in map(json.loads, (d / "GRADED.jsonl").read_text(encoding="utf-8").splitlines())}
rec = RUN.load_task_ids(["J1-20"])[0]
task = TaskSpec("J1-20", "java1:J1-20", "java", "java", rec["prompt"], rec["public_tests"], rec.get("difficulty", ""))
exe = ProviderExecutionBackend(backend="wslc", trusted=True)
exe.start(languages=("java",))
seen, rows = set(), []
for c in map(json.loads, (d / "CANDIDATES.jsonl").read_text(encoding="utf-8").splitlines()):
    if c["task"] != "J1-20" or c["id"] in seen or graded.get(("J1-20", c["id"])) != "PASS":
        continue
    seen.add(c["id"])
    pub = P2.public_result(exe.run_public(Artifact.make("J1-20", "java", c["source"], None, "check"), task), len(task.public_tests))
    rows.append({"id": c["id"][:12], "dev": "PASS", "public_pass_v2_run": c["public_pass"], "public_pass_now": pub["pass_all"], "public_passed_now": pub["passed"]})
    print(json.dumps(rows[-1]), flush=True)
print(json.dumps({"verifier": VERSION, "checked": len(rows), "public_pass_now": sum(r["public_pass_now"] for r in rows)}))
