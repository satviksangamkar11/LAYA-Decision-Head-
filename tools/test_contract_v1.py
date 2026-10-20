"""Focused checks for evidence_ladder/contracts/contract_v1.py (no sandbox, no model). Run: PYTHONPATH=. .venv/Scripts/python.exe tools/test_contract_v1.py"""
import json
import sys

from evidence_ladder.contracts import contract_v1 as CV

BASE = {"contract_version": CV.VERSION, "task_id": "T", "provenance": {"source": "author", "written_before_candidates": True},
        "input": {"type": "object", "fields": {"n": {"type": "int", "min": 1, "max": 10}}, "exact": True}, "output": {"type": "int", "min": 0}, "edges": [{"n": 10}]}


def doc(**kw):
    d = json.loads(json.dumps(BASE)); d.update(kw); return d


def raises(f):
    try:
        f()
    except CV.ContractError:
        return True
    return False


def test_schema_closed_language():
    assert raises(lambda: CV.check_schema({"type": "float"}))
    assert raises(lambda: CV.check_schema({"type": "int", "minimum": 1}))
    assert CV.matches({"type": "list", "items": {"type": "str", "pattern": "[a-z]*"}, "max_len": 2}, ["ab", ""])
    assert not CV.matches({"type": "int"}, True)                                  # bool is not an int


def test_provenance_and_trust():
    assert CV.load(doc()).authoritative
    assert not CV.load(doc(provenance={"source": "model:qwen", "written_before_candidates": True})).authoritative        # model-proposed: diagnostic, always
    assert not CV.load(doc(provenance={"source": "author", "written_before_candidates": False})).authoritative          # written after candidates: diagnostic
    assert raises(lambda: CV.load(doc(provenance={"source": "guess"})))
    assert raises(lambda: CV.load(doc(edges=[{"n": 11}])))                                                             # an edge outside the schema


def test_validity_and_hard_checks():
    c = CV.load(doc())
    inputs = [{"n": 3}, {"n": 0}, {"n": 5}, {"n": 7}]
    ev = CV.evaluate(c, inputs, [(True, 4), (False, "boom"), (True, -1), (False, "ValueError")])
    checks = sorted((h["input"], h["check"]) for h in ev["hard"])
    assert checks == [(2, "property:output_schema"), (3, "crash_on_valid_input")], checks          # input 1 is invalid (n=0): its crash is NOT evidence
    assert ev["mode"] == "authoritative" and ev["checked_valid_inputs"] == 3


def test_sequential_crash_attribution():
    c = CV.load(doc())
    ev = CV.evaluate(c, [{"n": 2}, {"n": 4}], [None, None], whole_run_crashed=True)
    assert [h["input"] for h in ev["hard"]] == [0] and len(ev["uncertain"]) == 1          # only the first missing input is attributed
    ev = CV.evaluate(c, [{"n": 0}, {"n": 4}], [None, None], whole_run_crashed=True)
    assert ev["hard"] == []                                                                 # the first missing input is invalid: no hard evidence


def test_self_check_is_necessary_not_sufficient():
    m = CV.load(doc(provenance={"source": "model:x", "written_before_candidates": True}, output={"type": "int", "min": 1}))
    assert CV.self_check(m, [{"input": {"n": 2}, "expected": 5}]) == []        # a wrong "always positive" rule can pass the public examples...
    assert not m.authoritative                                                  # ...and still never becomes authoritative
    assert CV.self_check(CV.load(doc()), [{"input": {"n": 2}, "expected": -3}]) == ["public example 0 violates output_schema"]


def test_parser_all_or_nothing():
    d = CV.derive_from_text("J", 'Input {"n": integer 1..10000, "k": integer 1..1000}. People ...')
    assert d and d["input"]["fields"]["n"] == {"type": "int", "min": 1, "max": 10000} and d["provenance"]["source"] == "derived:" + CV.PARSER_ID
    assert CV.derive_from_text("J", 'Input {"n": integer 1..10, "grid": rectangular matrix}. ...') is None          # one unknown field -> no contract at all
    assert CV.derive_from_text("J", "Given an array nums, return ...") is None


if __name__ == "__main__":
    tests = {k: v for k, v in globals().items() if k.startswith("test_")}
    failed = {}
    for k, f in tests.items():
        try:
            f()
        except Exception as e:      # noqa: BLE001
            failed[k] = "%s: %s" % (type(e).__name__, e)
    print(json.dumps({"tests": len(tests), "passed": len(tests) - len(failed), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)
