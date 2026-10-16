"""Records REAL sandbox outcomes (SandboxExecutionBackend) and REAL protected-grader results (ProtectedGrader, gate opened then closed as the orchestrator does)
for the fixed case table into fixtures/*.json. Run in WSL as root (approved sandbox-test exception):
  python3 evidence_ladder/testing/record_execution_fixtures.py [--out DIR]
Stored: public outcomes and experiment outcomes (duration_s / stderr_tail stripped: nondeterministic and never read downstream), backend_info receipt,
HiddenGrade fields only. No hidden inputs or expected values are stored."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evidence_ladder.common.interfaces import Artifact, GraderIsolationError  # noqa: E402
from evidence_ladder.execution import config as K  # noqa: E402
from evidence_ladder.execution.artifacts import ArtifactStore  # noqa: E402
from evidence_ladder.execution.backend import SandboxExecutionBackend  # noqa: E402
from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate  # noqa: E402
from evidence_ladder.execution.taskload import load_tasks  # noqa: E402
from evidence_ladder.testing.cases import CASES, EXPERIMENTS, PY_REF  # noqa: E402

STRIP = ("duration_s", "stderr_tail")


def clean(o):
    return {k: v for k, v in o.items() if k not in STRIP}


def tests_sha(task):
    return hashlib.sha256(json.dumps([{"input": t["input"], "expected": t["expected"]} for t in task.public_tests], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(__file__).parent / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    oracle_line = next(l for l in (K.PY_ROOT / "grader" / "oracles.py").read_text(encoding="utf-8").splitlines() if l.startswith("def p01("))
    assert oracle_line.split("return", 1)[1].split("#")[0].strip() == PY_REF.split("return", 1)[1].strip(), "PY_REF drifted from oracle p01"
    cfg = K.SandboxConfig(py_timeout=4.0, java_run_timeout=6.0)
    store = ArtifactStore(tempfile.mkdtemp(prefix="el-rec-"))
    be = SandboxExecutionBackend(store, cfg)
    tasks = {"python": {t.task_id: t for t in load_tasks("python")}, "java": {t.task_id: t for t in load_tasks("java")}}
    arts, entries = {}, {}
    for name, (tid, lang, src, note) in sorted(CASES.items()):
        a = Artifact.make(tid, lang, src, origin="recorded:" + name)
        arts[name] = a
        t = tasks[lang][tid]
        e = {"case": name, "note": note, "task_id": tid, "language": lang, "artifact_id": a.artifact_id, "source": src,
             "public_tests_sha256": tests_sha(t), "public": [clean(o) for o in be.run_public(a, t)], "experiments": []}
        for kind, payload in EXPERIMENTS[lang]:
            e["experiments"].append({"kind": kind, "payload": payload, "outcome": clean(be.run_experiment(a, {"kind": kind, "payload": payload}))})
        entries[name] = e
        print("recorded %-20s public=%s" % (name, [o["status"] for o in e["public"]]), flush=True)
    receipt = be.backend_info()
    gate = SelectionGate()
    gr = ProtectedGrader(gate, store, cfg)
    try:
        gr.grade(arts["py_ref"])
        raise SystemExit("grader answered while the gate was open")
    except GraderIsolationError:
        print("grader refused while gate open: ok")
    gate.close()
    grades = {}
    for name, a in sorted(arts.items()):
        h = gr.grade(a)
        grades[name] = {k: h[k] for k in ("artifact_id", "task_id", "passed", "passed_tests", "total_tests", "failure", "sha256_before_run")}
        print("graded %-20s passed=%s %s/%s %s" % (name, h["passed"], h["passed_tests"], h["total_tests"], h["failure"]), flush=True)
    (out / "execution_outcomes.json").write_text(json.dumps({"_note": "REAL sandbox outcomes recorded in WSL (bubblewrap jail); duration_s and stderr_tail stripped.",
                                                             "backend_info": receipt, "cases": entries}, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "hidden_grades.json").write_text(json.dumps({"_note": "REAL ProtectedGrader results (HiddenGrade fields only). Development oracle, public in git history, not a sealed benchmark.",
                                                        "grades": grades}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY cases=%d public_outcomes=%d experiments=%d grades=%d passed_protected=%s" % (
        len(entries), sum(len(e["public"]) for e in entries.values()), sum(len(e["experiments"]) for e in entries.values()), len(grades),
        sorted(n for n, g in grades.items() if g["passed"])))


if __name__ == "__main__":
    main()
