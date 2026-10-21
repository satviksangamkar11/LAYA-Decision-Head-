"""Self-contained test of the P6 analysis and contract grader on synthetic fixtures in a temp folder (never touches private/).
Usage: PYTHONPATH=. .venv/Scripts/python.exe -B tools/test_p6_grading.py     (the last check runs 1 trivial program in the real MXC sandbox)"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import el_p6_analyze as AN          # noqa: E402
import el_p6_grade_contract as GC   # noqa: E402


def wr(p, rows):
    Path(p).write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


def mk(d, cohort, spec, graded, identity=None):
    """spec: {(task, arm): (label, grade, logical, own, verdicts)}; writes RESULTS/ANSWERS/graded file."""
    d.mkdir(parents=True)
    res, ans, gra = [], [], []
    for (t, a), (label, grade, lg, own, vd) in spec.items():
        res.append({"cohort": cohort, "task": t, "arm": a, "label": label, "answer_sha256": None, "public_pass": None, "tokens_generated_logical": lg, "tokens_generated_own": own,
                    "tokens_prompt_logical": 10, "calls": 1, "candidates": 1, "rounds": 1, "seconds": 2.0, "gate_verdicts": vd, "rejected_by_contract": [], "stop": "accepted"})
        ans.append({"cohort": cohort, "task": t, "arm": a, "source": None if grade == "NO_ANSWER" else "pass"})
        gra.append({"task": t, "arm": a, "grade": grade})
    wr(d / "RESULTS.jsonl", res); wr(d / "ANSWERS.jsonl", ans); wr(d / graded, gra)
    if identity:
        wr(d / "IDENTITY.jsonl", identity)


def cf_spec(n, a_pass, b_pass):
    s = {}
    for i in range(n):
        t = "p%d" % i
        s[(t, "A")] = ("ANSWERED", "PASS" if i in a_pass else "FAIL", 100, 0, [])
        s[(t, "B")] = ("SUPPORTED", "PASS" if i in b_pass else "FAIL", 300, 200, [])
        s[(t, "V2")] = ("SUPPORTED", "PASS" if i in b_pass else "FAIL", 300, 0, ["unavailable"])
    return s


def test_analysis(tmp):
    # cf, hand-made: 4 tasks, A solves {0}, B solves {0,1,2} -> diff +2/4, extra 2, token ratio 3, no CI claim at n=4 beyond bounds
    ids = ["p0", "p1", "p2", "p3"]
    idn = [{"task": t, "identical": True, "same_answer": True, "same_requests": True, "same_tokens": True, "verdicts": ["unavailable"], "all_unavailable": True} for t in ids]
    mk(tmp / "cf1", "cf", cf_spec(4, {0}, {0, 1, 2}), "GRADED_cf.jsonl", idn)
    lock = {"protocol": "P6_PROTOCOL_v1", "cf": {"final_ids": ids}, "contract": {"ids": ["D%02d" % i for i in range(1, 6)]}}
    r = AN.analyze(lock, [str(tmp / "cf1")], [])["cf"]
    assert r["arms"]["A"]["solved"] == 1 and r["arms"]["B"]["solved"] == 3 and r["extra_correct_B_over_A"] == 2
    assert r["arms"]["A"]["wrong_answers_unclaimed"] == 3 and r["arms"]["B"]["false_acceptances"] == 1
    assert r["B_minus_A"]["diff_point"] == 0.5 and r["token_ratio_B_over_A"]["logical"] == 3.0 and r["own_tokens_per_extra_correct"] == 400.0
    assert r["identity"]["pass"] and AN.analyze(lock, [str(tmp / "cf1")], [])["infrastructure"]["complete"]
    # 10 tasks, B beats A on every task -> bootstrap CI is [1, 1]: excludes 0
    ids10 = ["p%d" % i for i in range(10)]
    mk(tmp / "cf2", "cf", cf_spec(10, set(), set(range(10))), "GRADED_cf.jsonl", [dict(idn[0], task=t) for t in ids10])
    rep = AN.analyze({"cf": {"final_ids": ids10}}, [str(tmp / "cf2")], [])
    assert rep["cf"]["B_minus_A"]["ci95"] == [1.0, 1.0] and rep["criteria"]["cf_B_beats_A"] is True
    # identity failure is reported, never silently passed; a missing grade is infrastructure
    bad = [dict(x) for x in idn]; bad[2]["identical"] = False
    spec = cf_spec(4, {0}, {0, 1, 2})
    mk(tmp / "cf3", "cf", spec, "GRADED_cf.jsonl", bad)
    gl = [json.loads(l) for l in (tmp / "cf3/GRADED_cf.jsonl").read_text().splitlines()]
    wr(tmp / "cf3/GRADED_cf.jsonl", [g for g in gl if (g["task"], g["arm"]) != ("p3", "B")])
    rep = AN.analyze(lock, [str(tmp / "cf3")], [])
    assert rep["cf"]["identity"]["failures"] == ["p2"] and rep["criteria"]["cf_identity_check_pass"] is False
    assert rep["infrastructure"]["missing_grades"] == ["cf/p3/B"] and rep["cf"]["B_minus_A"]["n_tasks"] == 3
    # contract: B -> V2 transitions
    L, S = "SUPPORTED", "BLOCKED"
    B = {"D01": (L, "PASS"), "D02": (L, "FAIL"), "D03": (L, "FAIL"), "D04": (S, "NO_ANSWER"), "D05": (L, "PASS")}
    V = {"D01": (L, "PASS"), "D02": (S, "NO_ANSWER"), "D03": (L, "PASS"), "D04": (S, "NO_ANSWER"), "D05": (L, "FAIL")}
    spec = {}
    for t in B:
        spec[(t, "A")] = ("ANSWERED", "FAIL", 50, 0, [])
        spec[(t, "B")] = (*B[t], 100, 50, [])
        spec[(t, "V2")] = (*V[t], 100, 50, ["accept", "reject"])
    mk(tmp / "ct", "contract", spec, "GRADED_contract_private.jsonl")
    rep = AN.analyze(lock, [], [str(tmp / "ct")])
    c = rep["contract"]
    assert c["transitions_B_to_V2"] == {"unchanged": 2, "wrong_answer_prevented": 1, "wrong_answer_repaired": 1, "correct_lost": 1, "correct_gained": 1, "other_changed": 0}
    assert c["arms"]["B"]["false_acceptances"] == 2 and c["arms"]["V2"]["false_acceptances"] == 1 and c["arms"]["V2"]["solved"] == 2
    assert c["V2_gate_verdict_mix"] == {"accept": 5, "reject": 5} and c["V2_minus_B"]["diff_point"] == 0.0
    assert rep["criteria"]["contract_V2_false_acceptances_le_B"] and not rep["criteria"]["contract_correct_lost_is_0"] and rep["criteria"]["contract_V2_solves_ge_B"]
    assert not rep["criteria"]["contract_gate_ok"] and rep["infrastructure"]["complete"]
    print("analysis OK")


def test_grader(tmp):
    cases = {"D01": [{"input": 1, "expected": 2}, {"input": 5, "expected": 6}]}
    raw = json.dumps(cases).encode()
    (tmp / "fx").mkdir(); (tmp / "fx/cases.json").write_bytes(raw)
    good = hashlib.sha256(raw).hexdigest()

    def lock(sha, ids):
        p = tmp / ("lock_%s.json" % sha[:6]); p.write_text(json.dumps({"contract": {"ids": ids, "grading": {"location": "fx/cases.json", "cases_sha256": sha}}}))
        return p

    def run_dir(name, answers):
        d = tmp / name; d.mkdir()
        (d / "DONE.json").write_text("{}"); (d / "RUN_CONFIG.json").write_text(json.dumps({"cohort": "contract"}))
        wr(d / "ANSWERS.jsonl", answers)
        return d
    ans = [{"cohort": "contract", "task": "D01", "arm": "A", "source": "def solve(data):\n    return data + 1\n"},
           {"cohort": "contract", "task": "D01", "arm": "B", "source": "def solve(data):\n    return data\n"},
           {"cohort": "contract", "task": "D01", "arm": "V2", "source": None}]

    def refuses(fn):
        try:
            fn()
        except SystemExit as e:
            return str(e)
        raise AssertionError("did not refuse")
    msg = refuses(lambda: GC.main(run_dir("r1", ans), lock("0" * 64, ["D01"]), tmp))
    assert "locked sha256" in msg
    msg = refuses(lambda: GC.main(run_dir("r2", ans + [{"cohort": "contract", "task": "D99", "arm": "A", "source": "x"}]), lock(good, ["D01"]), tmp))
    assert "D99" in msg
    assert not list(tmp.glob("r[12]/GRADED*")), "refusal must not write grades"
    d = run_dir("r3", ans)
    GC.main(d, lock(good, ["D01"]), tmp)
    got = {(g["arm"]): g["grade"] for g in map(json.loads, (d / "GRADED_contract_private.jsonl").read_text().splitlines())}
    assert got == {"A": "PASS", "B": "FAIL", "V2": "NO_ANSWER"}, got
    assert "exists" in refuses(lambda: GC.main(d, lock(good, ["D01"]), tmp))     # create-only
    print("grader OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="p6test-", ignore_cleanup_errors=True) as t:
        test_analysis(Path(t)); test_grader(Path(t))
    print("ALL PASS")
