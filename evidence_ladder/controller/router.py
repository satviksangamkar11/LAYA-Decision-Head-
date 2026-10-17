"""Verification router (R1.2): per requirement, which trusted check could establish it NOW? Re-run after every step that adds evidence; the table is journaled.
The ceiling depends only on the registry (trusted checks that cover the requirement) and the requirement's status. Model-written tests, critic experiments and
retrieved text can never raise it: they only appear in `soft_evidence` (ranking/SUPPORTED)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from evidence_ladder.common.contract_import import C

TRUSTED = ("authoritative", "authoritative_in_scope")
# which diagnostic / rejecting checks may add soft (non-approving) support, by requirement role: a fixed table, not a model call (I3)
SOFT_BY_ROLE = {
    "contract": ["parse_v1", "limits_v1"],
    "example": ["public_examples_v1"],
    "all_inputs": ["public_examples_v1", "qwen_tests_v1", "critic_experiment_v1", "behaviour_cluster_v1"],
    "derived": ["qwen_tests_v1", "critic_experiment_v1", "behaviour_cluster_v1"],
    "other": [],
}


def route(reqs: List[Dict[str, Any]], roles: Dict[str, str], registry, evidence: List[Dict[str, Any]], candidate_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if registry is None:
        from .components import MissingComponentError
        raise MissingComponentError("check registry", "the router derives every ceiling from it")
    table = []
    for r in reqs:
        rid = r["requirement_id"]
        gates = C.can_gate(r) and r["priority"] in ("hard", "functional")
        trusted = sorted(c["check_id"] for c in registry.checks.values()
                         if c["runtime_visible"] and not c["grader_only"] and c["trust_class"] in TRUSTED and rid in c["covers"])
        soft = sorted(cid for cid in SOFT_BY_ROLE.get(roles.get(rid, "other"), []) if cid in registry.checks and registry.checks[cid]["runtime_visible"])
        if not C.can_gate(r):
            ceiling, route = "RANKING_ONLY", "none"
        elif trusted and gates:
            ceiling, route = "VERIFIED", "trusted_check"
        elif gates and soft:
            ceiling, route = "SUPPORTED", "soft_only"
        else:
            ceiling, route = "UNVERIFIED", "none"
        ev = [e for e in evidence if e["requirement_id"] == rid and (candidate_id is None or e["candidate_id"] == candidate_id)]
        n_trusted = sum(1 for e in ev if C.authoritative(e, registry))
        table.append({"requirement_id": rid, "kind": r["kind"], "role": roles.get(rid, "other"), "gates": gates, "route": route, "ceiling": ceiling,
                      "trusted_checks": trusted, "soft_checks": soft, "trusted_evidence": n_trusted, "soft_evidence": len(ev) - n_trusted})
    return table


def answer_ceiling(table: List[Dict[str, Any]]) -> str:
    """Best answer label reachable with the current registry: the weakest gating row decides (R1.1)."""
    rows = [t["ceiling"] for t in table if t["gates"]]
    if rows and all(c == "VERIFIED" for c in rows):
        return "VERIFIED"
    if rows and all(c in ("VERIFIED", "SUPPORTED") for c in rows):
        return "SUPPORTED"
    return "UNVERIFIED"
