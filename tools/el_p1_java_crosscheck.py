"""[P1] Cross-check of suspicious offline-grading rows on the reference engine (engine.run, WSL bubblewrap, the qualified Java path), run inside WSL.
Prints counts and the failure class only (no case inputs or expected values). Usage (inside WSL): python3 tools/el_p1_java_crosscheck.py <task dir> <candidate id prefix> [...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_ladder.execution import config as K, engine            # noqa: E402
from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate      # noqa: E402

tdir, prefixes = Path(sys.argv[1]), sys.argv[2:]
cands = {}
for line in (tdir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r["kind"] == "candidate":
        cands[r["data"]["id"]] = r["data"]["source"]
task = json.loads((tdir / "result.json").read_text(encoding="utf-8"))
gate = SelectionGate(); gate.close()
cases = ProtectedGrader(gate)._cases(task["language"], task["task"])
cfg = K.SandboxConfig()
for p in prefixes:
    cid = next(k for k in cands if k.startswith(p))
    res = engine.run(task["language"], cands[cid].encode("utf-8"), [c["input"] for c in cases], cfg)
    v = engine.judge(task["language"], res, cases, cfg)
    print(task["task"], p, v.status, v.passed, "/", v.total, v.failure_class, "first_fail", v.first_fail, (v.detail or "")[:120], "|", (v.stderr_tail or "")[-300:].replace("\n", " "))
