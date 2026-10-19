"""[P2 prep] Parity check of the trusted verifier (v2 time accounting) against the reference grading of P1 (REFERENCE.jsonl), on chosen candidates.
Grades each named candidate on the development cases with the trusted verifier in the given backend; prints counts / failure class / accounting only.
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_tv_parity_check.py <p1 dir> <backend> <task dir>:<candidate prefix> [...]"""
import json
import sys
from pathlib import Path

from evidence_ladder.execution import config as K, engine
from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend

p1, backend, items = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
ref = {(r["task"], r["candidate"]): r for r in map(json.loads, (p1 / "REFERENCE.jsonl").read_text(encoding="utf-8").splitlines())}
gate = SelectionGate(); gate.close()
grader, cfg = ProtectedGrader(gate), K.SandboxConfig()
b = None
out = []
for it in items:
    tdir, prefix = it.rsplit(":", 1)
    tdir = Path(tdir)
    task = json.loads((tdir / "result.json").read_text(encoding="utf-8"))
    src = next(json.loads(l)["data"] for l in (tdir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(l)["kind"] == "candidate" and json.loads(l)["data"]["id"].startswith(prefix))
    if b is None:
        b = ProviderExecutionBackend(backend=backend, config=cfg, trusted=True)
        b.start(languages=(task["language"],))
    cases = grader._cases(task["language"], task["task"])
    res = b._engine_run(task["language"], src["source"].encode("utf-8"), [c["input"] for c in cases])
    v, _ = engine.judge_each(task["language"], res, cases, cfg)
    r = ref[(task["task"], src["id"])]
    tv = "PASS" if v.status == "pass" else ("INFRA" if v.failure_class == "sandbox_error" else "FAIL")
    row = {"task": task["task"], "candidate": src["id"][:12], "trusted_v2": tv, "passed": v.passed, "total": v.total, "failure": v.failure_class or None,
           "reference": r["ref"], "reference_passed": r["ref_passed"], "agree": tv == r["ref"] and v.passed == r["ref_passed"],
           "accounting": b.trusted_verifier.accounting}
    out.append(row)
    print(json.dumps(row), flush=True)
print(json.dumps({"checked": len(out), "agree": sum(r["agree"] for r in out)}))
