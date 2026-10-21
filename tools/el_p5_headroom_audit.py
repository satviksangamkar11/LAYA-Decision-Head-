"""[P5 bounded headroom audit, post-hoc, NOT a benchmark] For every task B did not solve (graded FAIL or NO_ANSWER in the saved P5 final run), hidden-grade
EVERY saved B candidate (P2CANDIDATES, public-passing or not) and classify the task:
  selection_headroom   B answered wrong (or abstained) although a saved candidate passes hidden grading AND passes the public examples
  check_anomaly        a saved candidate passes hidden grading but was recorded as failing the public examples (extraction / public-check error to inspect)
  generation_failure   no saved candidate passes hidden grading
Also counts generated samples that were never saved as candidates (early stop; their source was not logged, so they cannot be graded).
Hidden grades only describe saved candidates; nothing feeds back into any runtime. Reuses the P5 grading set-up of tools/el_p5_diagnostics.py.
Writes <out>/AUDIT.json (create-only; task ids and counts only, no hidden cases). Usage: PYTHONPATH=. python tools/el_p5_headroom_audit.py <new out dir>"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
OUT = Path(sys.argv[1]).resolve()
if OUT.exists():
    sys.exit("refusing: %s exists" % OUT)
import lcb_tasks as L                                                             # noqa: E402
from evidence_ladder.controller.sandbox_provider import make_sandbox             # noqa: E402
from evidence_ladder.execution import config as K, engine                        # noqa: E402
from evidence_ladder.execution.lcb_runner import LcbRunner                       # noqa: E402
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend   # noqa: E402

OUT.mkdir(parents=True)
FIN = ROOT / "results/raw/p5-final-v1"
LOCK = json.loads((ROOT / "configs/p5_final_lock_v2.json").read_text("utf-8"))
rows = lambda p: [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()] if p.exists() else []
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()

cfg = K.SandboxConfig()
exe = ProviderExecutionBackend(backend="mxc", scratch=OUT / "_engine", config=cfg, trusted=True)
exe.start(languages=("python",))
runner = LcbRunner(make_sandbox(OUT / "_sandbox", [], backend="mxc").run, OUT / "_lcb")
T = L.load()
raw_v3 = (ROOT / "private/p5-eval-v3/cases_v3.json").read_bytes()
assert hashlib.sha256(raw_v3).hexdigest() == LOCK["grading"]["primary"]["cases_sha256"], "blind set differs from its lock"
CASES_V3 = json.loads(raw_v3.decode("utf-8"))
_cache = {}


def hidden_ok(cohort, tid, src):
    k = (cohort, tid, sha(src))
    if k not in _cache:
        if cohort == "lcb":
            pv, go = T[tid]["public_view"], T[tid]["grader_only"]
            _cache[k] = runner.grade(src, list(pv["public_tests"]) + list(go["private_tests"]), pv["fn_name"], stop_at_first_fail=True)["pass"]
        else:
            res = exe._engine_run("python", src.encode("utf-8"), [c["input"] for c in CASES_V3[tid]])
            _cache[k] = engine.judge_each("python", res, CASES_V3[tid], cfg)[0].status == "pass"
    return _cache[k]


def audit(cohort, shards, ids):
    grade, cand, gen_seeds = {}, defaultdict(list), defaultdict(dict)
    for s in shards:
        d = FIN / ("p5_" + s)
        for r in rows(d / "GRADED.jsonl"):
            if r["arm"] == "B":
                grade[r["task"]] = r["grade"]
        for r in rows(d / "P2CANDIDATES.jsonl"):
            cand[r["task"]].append(r)
        for r in rows(d / "CALLS.jsonl"):
            if r["arm"] == "round1":
                gen_seeds[r["task"]][r["seed"]] = r.get("extract_status")
        for r in rows(d / "P2CALLS.jsonl"):
            gen_seeds[r["task"]][r["seed"]] = r.get("extract_status")
    per, cls = [], defaultdict(list)
    for t in ids:
        saved = {c["seed"] for c in cand[t]}
        unsaved = {s: st for s, st in gen_seeds[t].items() if s not in saved}
        rec = {"task": t, "B_grade": grade.get(t), "saved": len(cand[t]), "unsaved_generated": len(unsaved),
               "unsaved_extract_ok": sum(1 for st in unsaved.values() if st == "ok")}
        if grade.get(t) != "PASS":
            ok = [c for c in cand[t] if c.get("source") and hidden_ok(cohort, t, c["source"])]
            rec["hidden_correct_saved"] = len(ok)
            rec["hidden_correct_public_pass"] = sum(1 for c in ok if c["public_pass"])
            rec["class"] = ("selection_headroom" if rec["hidden_correct_public_pass"] else "check_anomaly" if ok else "generation_failure")
            cls[(rec["B_grade"], rec["class"])].append(t)
        per.append(rec)
    return {"B_grades": {g: sum(1 for p in per if p["B_grade"] == g) for g in sorted({str(p["B_grade"]) for p in per})},
            "not_solved_by_class": {"%s/%s" % k: v for k, v in sorted(cls.items())},
            "unsaved_generated_total": sum(p["unsaved_generated"] for p in per),
            "unsaved_extract_ok_total": sum(p["unsaved_extract_ok"] for p in per),
            "unsaved_on_tasks_B_answered_wrong": sum(p["unsaved_generated"] for p in per if p["B_grade"] == "FAIL"),
            "per_task": per}


rep = {"note": "bounded post-hoc audit of B's saved P5 candidates on tasks B did not solve; hidden grades describe saved candidates only; not a benchmark",
       "lcb": audit("lcb", ["final_lcb_%d" % k for k in range(1, 6)], LOCK["lcb_ids"]),
       "contract": audit("contract", ["final_contract"], LOCK["contract_ids"])}
(OUT / "AUDIT.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_task"} for k, v in rep.items() if k != "note"}, indent=1))
