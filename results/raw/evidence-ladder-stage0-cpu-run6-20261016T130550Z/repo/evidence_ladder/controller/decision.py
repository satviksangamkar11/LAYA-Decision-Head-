"""Decision records for the controller: a thin wrapper over C.decide (never a copy of it).
Extension: a `fail` from an independent, execution-derived record of a rejecting_only check (parse_v1, limits_v1) cannot approve anything, but it MUST be able to
reject. C.decide only counts authoritative checks as FAILED, so this wrapper downgrades such rows to FAILED and the answer to BLOCKED. It only ever downgrades."""
from __future__ import annotations

from typing import Any, Dict, List

from evidence_ladder.common.contract_import import C


def _rejected_rows(reqs, evs, cand_id, art_hash, env, registry) -> set:
    out = set()
    for r in reqs:
        if not (C.can_gate(r) and r["priority"] in ("hard", "functional")):
            continue
        for e in evs:
            if (C.applies(e, r, cand_id, art_hash, env, registry) and e["status"] == "fail" and e["independence"] != "model_derived"
                    and e["source"] in C.AUTHORITATIVE_SOURCES and registry.checks[e["check_id"]]["trust_class"] == "rejecting_only"):
                out.add(r["requirement_id"])
    return out


def decide_stage1(reqs: List[Dict[str, Any]], evs: List[Dict[str, Any]], cand_id: str, art_hash: str, env: str, registry,
                  risk: str = "sandboxed", user_confirmed: bool = False, decision_type: str = "complete_program") -> Dict[str, Any]:
    """Label from evidence only (I1). Evidence of other candidates, other artifact hashes, other versions or other environments is ignored by C.applies (S2 section 3)."""
    mine = [e for e in evs if e["candidate_id"] == cand_id]
    d = C.decide(reqs, mine, cand_id, art_hash, env, registry, risk, user_confirmed, decision_type)
    rej = _rejected_rows(reqs, mine, cand_id, art_hash, env, registry)
    changed = False
    by_id = {r["requirement_id"]: r for r in reqs}
    for row in d["requirement_rows"]:
        if row["requirement_id"] in rej and row["label"] not in ("FAILED", "CONFLICTED"):
            row["label"] = "FAILED"
            changed = True
        # C.row_label reports FAILED whenever any trusted record fails; trusted records that disagree are CONFLICTED (S2 section 4), a stronger statement of the same block
        if row["label"] == "FAILED" and C.conflicted(by_id[row["requirement_id"]], mine, cand_id, art_hash, env, registry):
            row["label"] = "CONFLICTED"
            changed = True
    if changed:
        d["answer_label"] = C.answer_label([r["label"] for r in d["requirement_rows"]])
        d["next_step"], d["authorized_to_execute"] = "block", False
        C.validate("decision_record", d)
    return d


def authorize(reqs, evs, cand_id, art_hash, env, registry, risk: str = "sandboxed", user_confirmed: bool = False) -> bool:
    """Risk-aware authorization (S4 section 2): every hard/functional row VERIFIED, and irreversible actions additionally need the user's confirmation."""
    return bool(decide_stage1(reqs, evs, cand_id, art_hash, env, registry, risk, user_confirmed, "commit")["authorized_to_execute"])


def with_next(d: Dict[str, Any], next_step: str) -> Dict[str, Any]:
    """Change only the routing hint. Never touches the label or the authorization."""
    out = dict(d)
    out["next_step"] = next_step
    C.validate("decision_record", out)
    return out


def blocked_without_candidate(reason_rows: List[Dict[str, Any]], decision_id: str, cand_id: str = "none", art_hash: str = "") -> Dict[str, Any]:
    """Decision for a problem stopped at intake (missing information / contradiction), or for an artifact whose hash no longer matches: the answer is BLOCKED."""
    d = {"decision_id": decision_id, "decision_type": "complete_program", "candidate_id": cand_id, "artifact_hash": art_hash or C.sha256(""),
         "requirement_rows": [{"requirement_id": r["requirement_id"], "label": "UNVERIFIED", "evidence_ids": []} for r in reason_rows
                              if C.can_gate(r) and r["priority"] in ("hard", "functional")],
         "answer_label": "BLOCKED", "next_step": "block", "authorized_to_execute": False, "risk": "sandboxed", "derived_by": "controller"}
    C.validate("decision_record", d)
    return d
