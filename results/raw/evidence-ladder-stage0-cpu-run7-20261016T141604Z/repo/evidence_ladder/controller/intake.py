"""Requirement intake (S1). Deterministic. Request text = task prompt + rendered public examples.
Requirement ids: R1 contract, R2 "correct on all legal inputs" (explicit, hard), R3.. one explicit requirement per public example (quoted with offsets),
then model-proposed ones (derived hypotheses unless their quote matches the request exactly). Derived hypotheses are tracked and never gate (C.can_gate)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evidence_ladder.common.contract_import import C

ALL_INPUTS_TEXT = "The program is correct on all legal inputs of the statement."


def _s(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class Intake:
    request_text: str
    reqs: List[Dict[str, Any]]
    roles: Dict[str, str]                 # requirement_id -> contract | all_inputs | example | derived | other
    contract_id: str
    all_inputs_id: str
    example_ids: List[str]
    stop: Optional[str] = None            # None | missing_information | contradictory_requirements
    stop_details: str = ""
    notes: List[str] = field(default_factory=list)

    def by_id(self, rid: str) -> Dict[str, Any]:
        return next(r for r in self.reqs if r["requirement_id"] == rid)


def render_request(task) -> tuple:
    """-> (request_text, [(input, expected, quote, start, end), ...]). The quote of an example is exactly request_text[start:end]."""
    text = task.prompt
    spans = []
    for i, t in enumerate(task.public_tests):
        inp, exp = _s(t.get("input", "")), _s(t.get("expected", ""))
        text += "\n\nExample %d\n" % (i + 1)
        block = "Input:\n%s\nOutput:\n%s" % (inp, exp)
        start = len(text)
        text += block
        spans.append((inp, exp, block, start, len(text)))
    return text, spans


def bump(req: Dict[str, Any], **changes) -> Dict[str, Any]:
    """Any change to text, kind or priority creates version + 1; evidence for the old version no longer applies (S1 section 3)."""
    r = dict(req)
    r.update(changes)
    r["version"] = req["version"] + 1
    C.validate("requirement", r)
    return r


def build(task, proposed: Optional[List[Dict[str, Any]]] = None) -> Intake:
    text, spans = render_request(task)
    reqs: List[Dict[str, Any]] = []
    roles: Dict[str, str] = {}
    notes: List[str] = []

    def add(r, role):
        r = C.check_requirement(r, text)         # an explicit requirement with a wrong quote is demoted to a derived hypothesis, never dropped
        reqs.append(r)
        roles[r["requirement_id"]] = "derived" if r["kind"] == "derived_hypothesis" else role
        return r

    add({"requirement_id": "R1", "version": 1, "kind": "contract",
         "text": "Entry point `solve` in language %s, within the stated limits." % task.language,
         "source": {"type": "task_metadata"}, "priority": "hard", "status": "active"}, "contract")
    add({"requirement_id": "R2", "version": 1, "kind": "explicit", "text": ALL_INPUTS_TEXT,
         "source": {"type": "request_span", "quote": task.prompt, "char_start": 0, "char_end": len(task.prompt)},
         "priority": "hard", "status": "active"}, "all_inputs")
    example_ids = []
    for i, (inp, exp, quote, a, b) in enumerate(spans):
        rid = "R%d" % (3 + i)
        example_ids.append(rid)
        add({"requirement_id": rid, "version": 1, "kind": "explicit", "text": "Public example %d: output exactly the quoted output for the quoted input." % (i + 1),
             "source": {"type": "request_span", "quote": quote, "char_start": a, "char_end": b}, "priority": "functional", "status": "active"}, "example")

    # two quoted examples with the same input but different outputs contradict each other (S1-T4)
    seen: Dict[str, str] = {}
    for rid, (inp, exp, _q, _a, _b) in zip(example_ids, spans):
        if inp in seen and seen[inp][1] != exp:
            other = seen[inp][0]
            for x, y in ((rid, other), (other, rid)):
                r = next(q for q in reqs if q["requirement_id"] == x)
                r["conflicts_with"] = sorted(set(r.get("conflicts_with", [])) | {y}); r["status"] = "contradicted"
        else:
            seen.setdefault(inp, (rid, exp))

    nxt = 3 + len(spans)
    for p in (proposed or []):
        r = dict(p)
        r["requirement_id"] = r.get("requirement_id") or "R%d" % nxt
        nxt = max(nxt, int(r["requirement_id"][1:])) + 1
        r.setdefault("version", 1); r.setdefault("priority", "quality"); r.setdefault("status", "active")
        if r.get("kind", "derived_hypothesis") == "derived_hypothesis":
            r["kind"] = "derived_hypothesis"; r.setdefault("source", {"type": "inference"}); r.setdefault("rationale", "model-inferred")
            r["status"] = r["status"] if r["status"] in ("validated", "ambiguous", "contradicted", "rejected") else "needs_validation"
        add(r, "derived" if r["kind"] == "derived_hypothesis" else "other")
    for r in reqs:        # check_requirement may have demoted an 'explicit' proposal: the role follows the final kind
        if r["kind"] == "derived_hypothesis":
            roles[r["requirement_id"]] = "derived"

    # a derived hypothesis keeps `validated` only if something outside the models validated it: stage 1 has no such source, so it never silently becomes gating
    for r in reqs:
        if r["kind"] == "derived_hypothesis" and r["status"] == "validated":
            r["status"] = "needs_validation"; notes.append("%s: validated status ignored (no trusted validation source in stage 1)" % r["requirement_id"])

    stop, details = None, ""
    contra = C.contradictions(reqs)
    if contra:
        ids = sorted({x for a, b in contra for x in (a, b)})
        stop = "contradictory_requirements"
        details = " | ".join("%s: %r" % (i, next(r for r in reqs if r["requirement_id"] == i)["source"].get("quote", "")[:200]) for i in ids)
    else:
        amb = [r for r in reqs if r["status"] == "ambiguous" and C.can_gate(r) and r["priority"] in ("hard", "functional")]
        if amb:
            stop = "missing_information"
            details = "; ".join("%s: %s" % (r["requirement_id"], r.get("ambiguity_notes", "")[:200]) for r in amb)
    return Intake(text, reqs, roles, "R1", "R2", example_ids, stop, details, notes)
