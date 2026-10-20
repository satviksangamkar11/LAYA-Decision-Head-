"""[P5] OFFLINE grading of a finished P5 run's answers (after every arm's selection is final). The ONLY P5 component that reads evaluation material.
contract cohort (P5_PROTOCOL_v2): the grading set is NAMED explicitly (primary = blind private/p5-eval-v3/cases_v3.json; secondary = public
results/raw/p5-eval-only-v1/hidden_cases.json; nonblind = private/p5-eval-v2/cases_v2.json) and its sha256 must equal configs/p5_final_lock_v2.json, else the
grader refuses (no default, no fallback); judged by the trusted verifier on MXC (engine.judge_each);
lcb cohort: LiveCodeBench public + private tests (official comparison rules, lcb_runner) on MXC; bank cohort (smoke): the development grader (protected.py).
Writes <run>/GRADED.jsonl (create-only): per task and arm PASS / FAIL / NO_ANSWER, counts only. Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_p5_grade.py <run dir>
[out name] [primary|secondary|nonblind  (contract cohort only, required)]. E.g. ... <run> GRADED_primary.jsonl primary"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from evidence_ladder.common.interfaces import Artifact          # noqa: E402
from evidence_ladder.execution import config as K, engine       # noqa: E402

run = Path(sys.argv[1]).resolve()        # ABSOLUTE: a relative scratch path makes the MXC container start in C:\Windows\System32 (known; it broke the first smoke grading)
out = run / (sys.argv[2] if len(sys.argv) > 2 else "GRADED.jsonl")      # create-only; a re-grade after a grader fix uses a new name
if out.exists():
    sys.exit("refusing: %s exists" % out)
cfgr = json.loads((run / "RUN_CONFIG.json").read_text("utf-8"))
if not (run / "DONE.json").exists():
    sys.exit("refusing: the run is not finished (no DONE.json)")
answers = [json.loads(l) for l in (run / "ANSWERS.jsonl").read_text("utf-8").splitlines()]
cohort = cfgr["cohort"]
grade = None
if cohort == "lcb":
    import lcb_tasks as L
    from evidence_ladder.controller.sandbox_provider import make_sandbox
    from evidence_ladder.execution.lcb_runner import LcbRunner
    T = L.load()
    runner = LcbRunner(make_sandbox(run / "_grade_sandbox", [], backend="mxc").run, run / "_grade_lcb")

    def grade(tid, src):
        pv, go = T[tid]["public_view"], T[tid]["grader_only"]
        g = runner.grade(src, list(pv["public_tests"]) + list(go["private_tests"]), pv["fn_name"], stop_at_first_fail=True)
        return g["pass"], g["passed"], g["total"], (g["first_fail"] or {}).get("status")
else:
    from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
    cfg = K.SandboxConfig()
    exe = ProviderExecutionBackend(backend="mxc", scratch=run / "_grade_engine", config=cfg, trusted=True)
    exe.start(languages=("python",))
    if cohort == "contract":      # [P5 v2] fail closed: the grading set is named explicitly and its bytes must match configs/p5_final_lock_v2.json; no default, no fallback
        import hashlib
        SETS = {"primary": ("private/p5-eval-v3/cases_v3.json", lambda g: g["primary"]["cases_sha256"]),
                "secondary": ("results/raw/p5-eval-only-v1/hidden_cases.json", lambda g: g["secondary"]["files"]["hidden_cases.json"]),
                "nonblind": ("private/p5-eval-v2/cases_v2.json", lambda g: g["disclosed_non_blind_validation"]["files"]["cases_v2.json"])}
        which = sys.argv[3] if len(sys.argv) > 3 else None
        if which not in SETS:
            sys.exit("refusing: the contract cohort needs an explicit grading set: primary | secondary | nonblind")
        rel, locked = SETS[which]
        want = locked(json.loads((ROOT / "configs/p5_final_lock_v2.json").read_text("utf-8"))["grading"])
        raw = (ROOT / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != want:
            sys.exit("refusing: %s does not match its locked sha256 %s" % (rel, want))
        cases_by = json.loads(raw.decode("utf-8"))
        missing = sorted({a["task"] for a in answers} - set(cases_by))
        if missing:
            sys.exit("refusing: grading set %s has no cases for %s" % (which, missing))
    else:
        from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate
        gate = SelectionGate(); gate.close(); pg = ProtectedGrader(gate)
        cases_by = None

    def grade(tid, src):
        cases = cases_by[tid] if cases_by is not None else pg._cases("python", tid)
        res = exe._engine_run("python", src.encode("utf-8"), [c["input"] for c in cases])
        v, _ = engine.judge_each("python", res, cases, cfg)
        if v.failure_class == "sandbox_error":
            raise RuntimeError("sandbox error while grading %s" % tid)
        return v.status == "pass", v.passed, v.total, v.failure_class or None

with open(out, "a", encoding="utf-8") as f:
    for a in answers:
        if not a["source"]:
            row = {"task": a["task"], "arm": a["arm"], "grade": "NO_ANSWER"}
        else:
            ok, p, t, fc = grade(a["task"], a["source"])
            row = {"task": a["task"], "arm": a["arm"], "grade": "PASS" if ok else "FAIL", "passed": p, "total": t, "first_fail": fc}
        f.write(json.dumps(row) + "\n"); f.flush()
        print(json.dumps(row), flush=True)
