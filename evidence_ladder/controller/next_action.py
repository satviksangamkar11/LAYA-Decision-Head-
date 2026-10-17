"""[LAYA-INT-1] Per-action decision (architecture R2.1): rank candidate NEXT ACTIONS with trusted evidence first and the LAYA score as L0 last.
Pure standard library + the contract core. A learned score never approves, never reorders across trusted evidence levels (I2) and never rescues a candidate a trusted check rejected.

trusted[i] = {name: value}: value "pass" counts +1 trusted pass; "fail" REJECTS the option (it cannot be selected); "fail_nonblocking" counts -1 (real execution showed the action
does not reach the goal, but it stays allowed). The scorer enters ONLY as `scorer` in C.rank, after trusted_passes / validated_passes / cluster.

No silent fallback (F06): scores are required unless the caller states `scorer_unavailable_reason`; then a "scorer_unavailable" journal entry is written and the order is the id order.
Abstain (F05, waiver configs/laya_none_abstain_waiver_v1.toml): when no option is allowed the decision abstains (selected None, next_step gather_evidence) and an "abstain" entry is written;
no NONE option is ever added to the LAYA pool.
Journal entries: "scorer" | "scorer_unavailable", "scorer_comparison" (optional, probe / random head, never used for the order), "ranking" (items + order, checkable by invariants.check_i2), "abstain" (optional), "action_decision"."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from evidence_ladder.common.contract_import import C

BLOCKING, NONBLOCKING_FAIL, PASS = "fail", "fail_nonblocking", "pass"


def decide_next_action(journal, decision_id: str, option_texts: Sequence[str], l0_scores: Optional[Sequence[float]], trusted: Optional[Sequence[Dict[str, Optional[str]]]] = None,
                       *, scorer_id: Optional[str] = None, scorer_provenance: Optional[Dict[str, Any]] = None, scorer_unavailable_reason: Optional[str] = None,
                       comparison_scores: Optional[Dict[str, Sequence[float]]] = None) -> Dict[str, Any]:
    n = len(option_texts)
    if l0_scores is None and not scorer_unavailable_reason:
        raise ValueError("no LAYA scores and no scorer_unavailable_reason: a silent id-order fallback is not allowed")
    if l0_scores is not None and len(l0_scores) != n:
        raise ValueError("wrong number of scores")
    trusted = list(trusted) if trusted is not None else [{} for _ in range(n)]
    ids = ["%s#%d" % (decision_id, i) for i in range(n)]
    blocked = [any(v == BLOCKING for v in (trusted[i] or {}).values()) for i in range(n)]
    items = [{"id": ids[i], "trusted_passes": sum(1 for v in (trusted[i] or {}).values() if v == PASS) - sum(1 for v in (trusted[i] or {}).values() if v == NONBLOCKING_FAIL),
              "validated_passes": 0, "cluster": 0, "scorer": float(l0_scores[i]) if l0_scores is not None else 0.0, "deepconf": 0.0} for i in range(n)]
    if l0_scores is not None:
        journal.append("scorer", {"state": "scored", "scorer_id": scorer_id, "decision_id": decision_id, "level": "L0", "ids": ids, "scores": [float(x) for x in l0_scores],
                                  "provenance": scorer_provenance or {}})
    else:
        journal.append("scorer_unavailable", {"decision_id": decision_id, "reason": str(scorer_unavailable_reason)[:300], "fallback": "id_order"})
    if comparison_scores:      # [EL-ARCH-1 C1] probe / random-head scores on the SAME options, journaled with their role and NEVER passed to C.rank (comparison only, not selectors)
        from evidence_ladder.models.laya_forward import ROLES
        for k, v in comparison_scores.items():
            if k not in ("probe", "random_head") or len(v) != n:
                raise ValueError("comparison scores must be 'probe' or 'random_head' with one score per option")
        journal.append("scorer_comparison", {"decision_id": decision_id, "ids": ids, "selector": False,
                                             "scores": {k: {"role": ROLES[k], "score": [float(x) for x in v]} for k, v in sorted(comparison_scores.items())}})
    allowed_items = [it for it, b in zip(items, blocked) if not b]
    order = [x["id"] for x in C.rank(allowed_items)] + [ids[i] for i in range(n) if blocked[i]]
    journal.append("ranking", {"step": decision_id, "items": allowed_items, "order": [o for o in order if not blocked[ids.index(o)]]})
    sel = next((o for o in order if not blocked[ids.index(o)]), None)
    out = {"decision_id": decision_id, "selected_id": sel, "selected_index": ids.index(sel) if sel else None, "order": order, "blocked": [ids[i] for i in range(n) if blocked[i]],
           "approved": False, "level": "L0_ordering_only", "derived_by": "controller", "abstain": sel is None, "scorer_available": l0_scores is not None}
    if sel is None:
        out["abstain_reason"] = "no allowed option: every option was rejected by a trusted check" if n else "no options proposed"
        out["next_step"] = "gather_evidence"
        journal.append("abstain", {"decision_id": decision_id, "reason": out["abstain_reason"], "n_options": n})
    journal.append("action_decision", out)
    return out
