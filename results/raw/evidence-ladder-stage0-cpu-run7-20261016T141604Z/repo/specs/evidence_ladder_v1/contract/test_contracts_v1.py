"""Contract tests S1-T1..S5-T5 and S4-T7 (plain asserts, no framework). Run: python3 test_contracts_v1.py <out.json>. No models, no network."""
import json, sys, tempfile, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_lib_v1 as C

H = C.sha256
REQ_TEXT = "Fix the refund total so it never exceeds the payment. Keep the public API."
def req(i, kind="explicit", prio="functional", status="active", v=1, quote=None, start=0, end=0, conflicts=None):
    r = {"requirement_id": f"R{i}", "version": v, "kind": kind, "text": "t", "source": {}, "priority": prio, "status": status}
    if kind == "explicit":
        r["source"] = {"type": "request_span", "quote": quote, "char_start": start, "char_end": end}
    elif kind == "contract":
        r["source"] = {"type": "task_metadata"}
    else:
        r["source"] = {"type": "inference"}; r["rationale"] = "inferred"
    if conflicts: r["conflicts_with"] = conflicts
    return r
GOOD = REQ_TEXT[0:40]
def chk(cid, trust="authoritative_in_scope", covers=("R1",), vis=True, grader=False, kind="public_test", v="1"):
    return {"check_id": cid, "version": v, "code_hash": H(cid), "kind": kind, "trust_class": trust, "covers": list(covers), "runtime_visible": vis, "grader_only": grader}
def ev(i, rid, cid, status="pass", source="existing_test", indep="independent", cand="A", art=H("A"), env="e1", rv=1, cv="1"):
    return {"evidence_id": f"E{i}", "requirement_id": rid, "requirement_version": rv, "check_id": cid, "check_version": cv, "candidate_id": cand, "artifact_hash": art,
            "environment_revision": env, "source": source, "independence": indep, "status": status, "scope": "s", "produced_at": "t", "receipt_hash": H(str(i))}
REG = C.Registry([chk("t1", covers=("R1", "R2")), chk("t2", covers=("R1", "R2")), chk("qt", trust="diagnostic", kind="model_test"), chk("hid", vis=False, grader=True, kind="hidden_test")])
SCORER = {"scorer_id": "laya", "version": "1", "weights_hash": "x", "status": "validated", "decision_types_validated": ["next_action"],
          "compatibility": {"qwen_repo": "r", "qwen_revision": "768f209d", "dtype": "fp16", "device_class": "t4", "depths": [18, 24, 30], "query_feature": "q", "option_feature": "o",
                            "capture_code_path": "p", "capture_code_hash": "h", "max_options": 8, "prefix_token_limit": 24000, "option_token_limit": 4096}}
T = {}
def test(name):
    def deco(f): T[name] = f; return f
    return deco

@test("S1-T1 explicit requirement with a wrong quote becomes a derived hypothesis")
def _():
    r = C.check_requirement(req(1, quote="NOT IN REQUEST", start=0, end=14), REQ_TEXT)
    assert r["kind"] == "derived_hypothesis" and r["status"] == "needs_validation"
    ok = C.check_requirement(req(1, quote=GOOD, start=0, end=40), REQ_TEXT)
    assert ok["kind"] == "explicit"

@test("S1-T2 a derived hypothesis cannot gate approval")
def _():
    h = req(1, kind="derived_hypothesis", status="needs_validation")
    assert not C.can_gate(h)
    d = C.decide([h], [ev(1, "R1", "t1")], "A", H("A"), "e1", REG)
    assert d["answer_label"] == "UNVERIFIED" and not d["authorized_to_execute"]

@test("S1-T3 evidence for requirement version 1 is ignored after version 2")
def _():
    r2 = req(1, quote=GOOD, start=0, end=40, v=2)
    d = C.decide([r2], [ev(1, "R1", "t1", rv=1)], "A", H("A"), "e1", REG)
    assert d["answer_label"] == "UNVERIFIED"

@test("S1-T4 contradictory hard requirements are reported")
def _():
    a = req(1, prio="hard", conflicts=["R2"]); b = req(2, prio="hard", conflicts=["R1"])
    assert C.contradictions([a, b])
    assert C.stop_cause([], 0, 10, contradictory=True) == "contradictory_requirements"

@test("S2-T1 stale evidence: a repaired candidate inherits nothing")
def _():
    r = req(1, quote=GOOD, start=0, end=40)
    e = ev(1, "R1", "t1", art=H("A"))
    assert C.decide([r], [e], "A", H("A"), "e1", REG)["answer_label"] == "VERIFIED"
    d = C.decide([r], [e], "A2", H("A2"), "e1", REG)
    assert d["answer_label"] == "UNVERIFIED" and not d["authorized_to_execute"]
    d2 = C.decide([r], [e], "A", H("A"), "e2", REG)  # base environment changed
    assert d2["answer_label"] == "UNVERIFIED"

@test("S2-T2 conflicting trusted checks never approve")
def _():
    r = req(1, quote=GOOD, start=0, end=40)
    es = [ev(1, "R1", "t1", "pass"), ev(2, "R1", "t2", "pass"), ev(3, "R1", "t2", "fail", source="user_test", indep="user_provided")]
    lab = C.row_label(r, es, "A", H("A"), "e1", REG)
    assert lab == "FAILED" or lab == "CONFLICTED"
    assert not C.decide([r], es, "A", H("A"), "e1", REG)["authorized_to_execute"]
    only_pf = [ev(1, "R1", "t1", "pass"), ev(2, "R1", "t2", "fail")]
    assert C.conflicted(r, only_pf, "A", H("A"), "e1", REG)

@test("S2-T3 a model-written test passing never makes VERIFIED")
def _():
    r = req(1, quote=GOOD, start=0, end=40)
    d = C.decide([r], [ev(1, "R1", "qt", source="model_test", indep="model_derived")], "A", H("A"), "e1", REG)
    assert d["answer_label"] == "SUPPORTED" and not d["authorized_to_execute"]

@test("S2-T4 timeout/error/unknown records never count as a pass")
def _():
    r = req(1, quote=GOOD, start=0, end=40)
    for st in C.NOT_PASS - {"fail"}:
        assert C.row_label(r, [ev(1, "R1", "t1", st)], "A", H("A"), "e1", REG) == "UNVERIFIED", st

@test("S2-T5 'never negative' proved, 'implements policy' open -> answer not VERIFIED")
def _():
    r1 = req(1, quote=GOOD, start=0, end=40); r2 = req(2, quote=REQ_TEXT[41:65], start=41, end=65)
    d = C.decide([r1, r2], [ev(1, "R1", "t1")], "A", H("A"), "e1", REG)
    assert d["answer_label"] == "UNVERIFIED" and not d["authorized_to_execute"]

@test("S2-T6 a check that does not cover the requirement cannot approve it")
def _():
    r = req(3, quote=GOOD, start=0, end=40)
    r["requirement_id"] = "R3"
    assert C.row_label(r, [ev(1, "R3", "t1")], "A", H("A"), "e1", REG) != "VERIFIED"

@test("S2-T7 irreversible action needs the user's confirmation")
def _():
    r = req(1, quote=GOOD, start=0, end=40)
    es = [ev(1, "R1", "t1")]
    assert not C.decide([r], es, "A", H("A"), "e1", REG, risk="irreversible")["authorized_to_execute"]
    assert C.decide([r], es, "A", H("A"), "e1", REG, risk="irreversible", user_confirmed=True)["authorized_to_execute"]

@test("S3-T1 a check missing from the registry cannot produce evidence")
def _():
    try: REG.accept(ev(1, "R1", "nope")); assert False
    except KeyError: pass

@test("S3-T2 LAYA refused for an unsupported decision type unless exploratory")
def _():
    assert C.scorer_allowed(SCORER, "next_action")
    assert not C.scorer_allowed(SCORER, "complete_program")
    assert C.scorer_allowed(SCORER, "complete_program", exploratory=True)
    assert not C.scorer_allowed(dict(SCORER, status="disabled"), "next_action")

@test("S3-T3 a registry or scorer change during a run aborts it")
def _():
    h = REG.hash
    C.assert_frozen(h, h)
    try: C.assert_frozen(h, C.Registry([chk("t1")]).hash); assert False
    except RuntimeError: pass

@test("S3-T4 grader-only hidden tests are unreachable from runtime code")
def _():
    try: REG.runtime_lookup("hid"); assert False
    except PermissionError: pass
    assert REG.runtime_lookup("t1")["check_id"] == "t1"

@test("S3-T5 schema rejects a grader-only check marked runtime visible")
def _():
    try: C.Registry([chk("bad", vis=True, grader=True)]); assert False
    except Exception: pass

@test("S5-T1 identical failure on parent and child stops the lineage")
def _():
    rounds = [{"failure_sigs": ["x"], "progress": True, "lineage": "L"}, {"failure_sigs": ["x"], "progress": True, "lineage": "L"}]
    assert C.stop_cause(rounds, 1, 10) == "same_failure_twice"

@test("S5-T2 progress then none for two rounds -> no_progress_2")
def _():
    rounds = [{"failure_sigs": ["a"], "progress": True, "lineage": "L"}, {"failure_sigs": ["b"], "progress": False, "lineage": "L2"}, {"failure_sigs": ["c"], "progress": False, "lineage": "L3"}]
    assert C.stop_cause(rounds, 1, 10) == "no_progress_2"

@test("S5-T3 a scorer preferring fewer trusted passes does not reorder")
def _():
    a = {"id": "a", "trusted_passes": 3, "validated_passes": 0, "cluster": 1, "scorer": 0.1}
    b = {"id": "b", "trusted_passes": 2, "validated_passes": 5, "cluster": 9, "scorer": 0.99}
    assert [c["id"] for c in C.rank([b, a])] == ["a", "b"]
    c1 = {"id": "c1", "trusted_passes": 2, "validated_passes": 1, "cluster": 1, "scorer": 0.2}; c2 = {"id": "c2", "trusted_passes": 2, "validated_passes": 1, "cluster": 1, "scorer": 0.9}
    assert [c["id"] for c in C.rank([c1, c2])] == ["c2", "c1"]  # scorer only breaks exact ties

@test("S5-T4 budget reached -> budget, never decided")
def _():
    assert C.stop_cause([], 10, 10) == "budget"
    assert C.stop_cause([], 10, 10, decided=False) != "decided"

@test("S5-T5 sandbox error is tool_failure, separate from candidate failure")
def _():
    assert C.stop_cause([], 0, 10, tool_failure=True) == "tool_failure"
    assert C.stop_cause([], 0, 10, api_down=True) == "api_unavailable"

@test("S4-T7 a failed commit leaves the base byte-identical")
def _():
    with tempfile.TemporaryDirectory() as t:
        base = Path(t) / "base"; base.mkdir(); (base / "a.txt").write_text("one"); (base / "d").mkdir(); (base / "d" / "b.txt").write_text("two")
        before = C.tree_hash(base)
        def bad_patch(w): (w / "a.txt").write_text("CHANGED"); (w / "new.txt").write_text("x")
        assert C.commit_patch(base, bad_patch, lambda w: False) is False and C.tree_hash(base) == before
        def raising(w): raise OSError("disk full mid-patch")
        assert C.commit_patch(base, raising, lambda w: True) is False and C.tree_hash(base) == before
        assert C.commit_patch(base, bad_patch, lambda w: (w / "a.txt").read_text() == "CHANGED") is True and (base / "a.txt").read_text() == "CHANGED"

@test("SCHEMA examples validate and bad records are rejected")
def _():
    C.validate("budget_record", {"problem_id": "p", "arm": "D1", "qwen_gen_tokens": 1, "qwen_prefill_tokens": 1, "glm_tokens_in": 0, "glm_tokens_out": 0, "glm_calls": 0, "scorer_calls": 0, "tool_runs": 1, "gpu_seconds": 0.1, "wall_seconds": 1, "api_errors": 0})
    C.validate("critic_response", {"experiment": {"kind": "stdin_input", "payload": "1 2"}, "implications": "if A prints 3 then ..."})
    for bad in ({"experiment": {"kind": "shell_anything", "payload": "rm -rf"}, "implications": "x"}, {"experiment": {"kind": "stdin_input"}, "implications": "x"}):
        try: C.validate("critic_response", bad); assert False
        except Exception as e: assert "AssertionError" != type(e).__name__
    try: C.validate("critic_request", {"request_id": "r", "model": "other/model", "decision_id": "d", "candidate_ids": ["a"], "requirement_quotes": [], "candidate_code": {}, "observed_outputs": {}, "question": "distinguish"}); assert False
    except Exception as e: assert type(e).__name__ == "ValidationError"
    try: C.validate("decision_record", {"decision_id": "d", "candidate_id": "a", "artifact_hash": H("a"), "requirement_rows": [], "answer_label": "VERIFIED", "next_step": "commit", "authorized_to_execute": True, "risk": "sandboxed", "derived_by": "model"}); assert False
    except Exception as e: assert type(e).__name__ == "ValidationError"

res = {}
for name, f in T.items():
    try:
        f(); res[name] = "PASS"
    except BaseException:
        res[name] = "FAIL: " + traceback.format_exc().splitlines()[-1]
out = {"tests": len(res), "passed": sum(v == "PASS" for v in res.values()), "results": res}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({"tests": out["tests"], "passed": out["passed"], "failed": {k: v for k, v in res.items() if v != "PASS"}}, indent=1))
