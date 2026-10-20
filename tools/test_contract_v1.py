"""Focused checks for evidence_ladder/contracts/contract_v1.py (no sandbox, no model). Run: PYTHONPATH=. .venv/Scripts/python.exe tools/test_contract_v1.py"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from evidence_ladder.contracts import contract_v1 as CV
from evidence_ladder.contracts import registry as RG

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


def _reg(tmp, d, authority="authoritative"):
    p = tmp / "c.json"; p.write_text(json.dumps(d), encoding="utf-8")
    return {"registry_version": RG.REGISTRY_VERSION, "entries": {"T": {"path": "c.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "authority": authority}}}


def test_provenance_and_trust():
    tmp = Path(tempfile.mkdtemp())
    assert not CV.load(doc()).authoritative                                                        # a document alone never grants authority (self-claims ignored)
    assert RG.trusted_contract("T", _reg(tmp, doc()), tmp).authoritative                           # only the host registry does
    assert not RG.trusted_contract("T", _reg(tmp, doc(provenance={"source": "model:qwen", "written_before_candidates": True})), tmp).authoritative   # model: never
    assert not RG.trusted_contract("T", _reg(tmp, doc(), "diagnostic"), tmp).authoritative
    reg = _reg(tmp, doc()); (tmp / "c.json").write_text(json.dumps(doc(edges=[])), encoding="utf-8")
    assert raises(lambda: RG.trusted_contract("T", reg, tmp))                                      # tampered after pinning: fail closed
    assert RG.trusted_contract("X", reg, tmp) is None                                              # no entry, no parser: no contract, no verdict
    d = RG.trusted_contract("J", None, tmp, 'Input {"n": integer 1..9}. Return ...', parsers=[CV.PARSER_ID])
    assert d is not None and d.authoritative
    assert raises(lambda: RG.trusted_contract("J", None, tmp, "x", parsers=["llm_guess"]))         # only allowlisted parsers
    assert raises(lambda: CV.load(doc(provenance={"source": "guess"})))
    assert raises(lambda: CV.load(doc(edges=[{"n": 11}])))                                                             # an edge outside the schema


def test_validity_and_hard_checks():
    c = CV.load(doc()); c.authority = "authoritative"
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
