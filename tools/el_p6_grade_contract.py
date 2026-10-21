"""[P6] OFFLINE grading of a finished P6 contract-cohort run (after every arm's selection is final). The ONLY P6 component that reads the private blind cases.
Cases come from the P6 lock `contract.grading.location`; their sha256 must equal `contract.grading.cases_sha256` and the answers' tasks must be the lock's
`contract.ids`, else the grader refuses (fail closed, no fallback). Judged by the MXC trusted verifier. Writes <run>/GRADED_contract_private.jsonl (create-only):
{"task","arm","grade","passed","total","first_fail"}, counts only.
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_p6_grade_contract.py <run dir> <lock json>"""
import contextlib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(run, lock_path, cases_root=ROOT):
    run = Path(run).resolve()        # ABSOLUTE: a relative scratch path makes the MXC container start in C:\Windows\System32
    out = run / "GRADED_contract_private.jsonl"
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    if not (run / "DONE.json").exists():
        sys.exit("refusing: the run is not finished (no DONE.json)")
    cohort = json.loads((run / "RUN_CONFIG.json").read_text("utf-8"))["cohort"]
    if cohort != "contract":
        sys.exit("refusing: cohort is %r, not contract" % cohort)
    lock = json.loads(Path(lock_path).read_text("utf-8"))["contract"]
    g = lock["grading"]
    raw = (Path(cases_root) / g["location"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != g["cases_sha256"]:
        sys.exit("refusing: %s does not match its locked sha256 %s" % (g["location"], g["cases_sha256"]))
    cases_by = json.loads(raw.decode("utf-8"))
    answers = [json.loads(l) for l in (run / "ANSWERS.jsonl").read_text("utf-8").splitlines() if l.strip()]
    ids = set(lock["ids"])
    bad = sorted({a["task"] for a in answers} - ids)
    if bad:
        sys.exit("refusing: tasks not in the lock's contract ids: %s" % bad)
    miss = sorted(ids - set(cases_by))
    if miss:
        sys.exit("refusing: grading set has no cases for %s" % miss)
    sys.path.insert(0, str(ROOT))
    from evidence_ladder.execution import config as K, engine
    from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
    cfg = K.SandboxConfig()
    exe = ProviderExecutionBackend(backend="mxc", scratch=run / "_grade_engine", config=cfg, trusted=True)
    exe.start(languages=("python",))
    with open(out, "x", encoding="utf-8") as f, contextlib.closing(exe):
        for a in answers:
            if not a["source"]:
                row = {"task": a["task"], "arm": a["arm"], "grade": "NO_ANSWER"}
            else:
                cases = cases_by[a["task"]]
                res = exe._engine_run("python", a["source"].encode("utf-8"), [c["input"] for c in cases])
                v, _ = engine.judge_each("python", res, cases, cfg)
                if v.failure_class == "sandbox_error":
                    raise RuntimeError("sandbox error while grading %s" % a["task"])
                row = {"task": a["task"], "arm": a["arm"], "grade": "PASS" if v.status == "pass" else "FAIL", "passed": v.passed, "total": v.total,
                       "first_fail": v.failure_class or None}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
