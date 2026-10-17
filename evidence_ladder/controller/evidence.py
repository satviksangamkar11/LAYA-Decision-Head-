"""Turns execution outcomes into schema-valid evidence records. The status mapper is the only place that decides pass/fail (I4): anything that is not exactly
"pass" is never a pass; unrecognised statuses become invalid_spec. Only deterministic outcome fields reach the journal (no durations, no stderr)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from evidence_ladder.common.contract_import import C
from .journal import canon

CAND_FAIL = {"compile_error", "runtime_error", "wrong_answer", "missing_output", "invalid_output"}
NOT_PASS = {"fail", "unknown", "timeout", "error", "missing_evidence", "invalid_spec"}


def _s(x: Any, n: int = 2000) -> str:
    s = x if isinstance(x, str) else json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)
    return s[:n]


def map_status(outcome: Dict[str, Any]) -> str:
    s = outcome.get("status")
    if s == "pass":
        return "pass"
    if s == "fail":
        return "fail"
    if s == "error" and outcome.get("failure_class") in CAND_FAIL:
        return "fail"                      # the candidate crashed or produced nothing: a real rejection, not an infrastructure problem
    if s in NOT_PASS:
        return s
    return "invalid_spec"


def is_tool_failure(outcome: Dict[str, Any]) -> bool:
    return outcome.get("failure_class") == "sandbox_error"


def trim(o: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (_s(o[k]) if k in ("detail", "observed") else o[k]) for k in ("check_id", "status", "detail", "counterexample", "observed", "failure_class", "example_index") if k in o}


MODEL_SOURCES = {"model_test", "critic_experiment", "candidate_agreement", "scorer"}


class ModelEvidenceError(Exception):
    """[EL-ARCH-1 A4] model-derived evidence named a requirement the controller does not hold."""


def is_model_derived(source: str, independence: str, check_trust: str) -> bool:
    return independence == "model_derived" or source in MODEL_SOURCES or check_trust == "diagnostic"


def cap_model_status(status: str, rid: str, check_id: str, roles: Optional[Dict[str, str]]) -> str:
    """[EL-ARCH-1 A4] Model-derived evidence (critic experiments, model-written checks, behaviour clusters; LAYA/probe scores never enter as evidence at all) may REJECT or stay unknown,
    but a `pass` is kept only where the router's fixed table (router.SOFT_BY_ROLE) lists that check as soft support for the controller's own role of that requirement; anywhere
    else it is recorded as `unknown`. The requirement id always comes from the controller's list (Factory.make raises on any other id), never from the model."""
    if status != "pass":
        return status
    from .router import SOFT_BY_ROLE
    return "pass" if check_id in SOFT_BY_ROLE.get((roles or {}).get(rid, "other"), []) else "unknown"


class Factory:
    def __init__(self, registry, reqs: List[Dict[str, Any]], env: str, roles: Optional[Dict[str, str]] = None):
        self.registry, self.env, self.n = registry, env, 0
        self.ver = {r["requirement_id"]: r["version"] for r in reqs}
        self.roles = roles

    def make(self, art, rid: str, check_id: str, source: str, independence: str, status: str, scope: str, outcome: Dict[str, Any],
             counterexample: Optional[Dict[str, str]] = None, assumptions: Optional[List[str]] = None) -> Dict[str, Any]:
        if rid not in self.ver:
            raise ModelEvidenceError("requirement %r is not in the controller's requirement list (check %s)" % (rid, check_id))
        if is_model_derived(source, independence, self.registry.checks[check_id]["trust_class"]):
            status = cap_model_status(status, rid, check_id, self.roles)
        self.n += 1
        ev = {"evidence_id": "E%04d" % self.n, "requirement_id": rid, "requirement_version": self.ver[rid], "check_id": check_id,
              "check_version": self.registry.checks[check_id]["version"], "candidate_id": art.artifact_id, "artifact_hash": art.artifact_id,
              "environment_revision": self.env, "source": source, "independence": independence, "status": status, "scope": scope,
              "produced_at": "t%d" % self.n,           # logical clock: wall time would break the reproducibility receipt
              "receipt_hash": C.sha256(canon({"o": trim(outcome), "a": art.artifact_id, "c": check_id, "r": rid, "s": status}))}
        if assumptions:
            ev["assumptions"] = list(assumptions)
        if counterexample:
            ev["counterexample"] = {k: _s(v) for k, v in counterexample.items() if k in ("input", "expected", "observed")}
        return ev


def public_records(task, art, outcomes: List[Dict[str, Any]], intake, fac: Factory) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], Dict[str, Any]]:
    """-> ([(evidence, trimmed_outcome)], summary). Execution contract (see report): one outcome per public example with check_id "public_examples_v1"
    (position = example index unless it carries "example_index"), optionally "parse_v1" and "limits_v1" outcomes."""
    ex = [o for o in outcomes if o.get("check_id") == "public_examples_v1"]
    by_idx = {}
    for pos, o in enumerate(ex):
        by_idx[o["example_index"] if isinstance(o.get("example_index"), int) else pos] = o
    recs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    examples, sigs, tool_fail = [], [], any(is_tool_failure(o) for o in outcomes)
    for i, rid in enumerate(intake.example_ids):
        o = by_idx.get(i)
        t = task.public_tests[i]
        if o is None:
            o = {"check_id": "public_examples_v1", "status": "missing_evidence", "detail": "no outcome reported for this example"}
        st = map_status(o)
        cx = None
        if st == "fail":
            cx = dict(o.get("counterexample") or {})
            cx.setdefault("input", _s(t.get("input", ""))); cx.setdefault("expected", _s(t.get("expected", ""))); cx.setdefault("observed", _s(o.get("observed", o.get("failure_class", ""))))
        recs.append((fac.make(art, rid, "public_examples_v1", "user_test", "user_provided", st, "the statement's own example %d" % (i + 1), o, cx), trim(o)))
        examples.append({"i": i, "status": st, "observed": _s(o.get("observed", ""), 200) if "observed" in o else ""})
        if st != "pass":
            sigs.append(C.sha256(canon([o.get("check_id"), _s(t.get("input", ""), 500), _s(o.get("observed", o.get("failure_class", o.get("status", ""))), 500)]))[:16])
    statuses = [e["status"] for e in examples]

    parse = next((o for o in outcomes if o.get("check_id") == "parse_v1"), None)
    if parse is None and statuses and all(s == "pass" for s in statuses):
        # the examples ran, so the program parsed; recorded as implied, with the assumption written down
        recs.append((fac.make(art, intake.contract_id, "parse_v1", "environment_observation", "independent", "pass", "implied by a successful public-example run", {"check_id": "parse_v1", "status": "pass"},
                              assumptions=["parse outcome inferred from the example run"]), {"check_id": "parse_v1", "status": "pass", "implied": True}))
        pst = "pass"
    elif parse is not None:
        pst = map_status(parse)
        recs.append((fac.make(art, intake.contract_id, "parse_v1", "environment_observation", "independent", pst, "syntax", parse), trim(parse)))
        if pst != "pass":
            sigs.append(C.sha256(canon(["parse_v1", _s(parse.get("detail", ""), 200)]))[:16])
    else:
        pst = None
    lim = next((o for o in outcomes if o.get("check_id") == "limits_v1"), None)
    lst = None
    if lim is not None:
        lst = map_status(lim)
        recs.append((fac.make(art, intake.contract_id, "limits_v1", "environment_observation", "independent", lst, "limits on public examples", lim), trim(lim)))
        if lst != "pass":
            sigs.append(C.sha256(canon(["limits_v1", _s(lim.get("detail", ""), 200)]))[:16])

    # the "all legal inputs" requirement: no check covers it, so these records can only ever yield SUPPORTED (never VERIFIED)
    if statuses:
        allst = "pass" if all(s == "pass" for s in statuses) else ("fail" if "fail" in statuses else "unknown")
        src, ind = ("request_example", "user_provided") if allst == "pass" else ("environment_observation", "independent")
        recs.append((fac.make(art, intake.all_inputs_id, "public_examples_v1", src, ind, allst, "public examples only; not a proof for all legal inputs",
                              {"check_id": "public_examples_v1", "status": allst}), {"check_id": "public_examples_v1", "status": allst, "scope": "all_inputs_row"}))
    beh = C.sha256(canon([[e["status"], e["observed"]] for e in examples]))
    summary = {"examples": examples, "parse": pst, "limits": lst, "tool_failure": tool_fail, "sigs": sorted(set(sigs)), "behaviour": beh,
               "pass_count": sum(1 for s in statuses if s == "pass"), "all_pass": bool(statuses) and all(s == "pass" for s in statuses)}
    return recs, summary
