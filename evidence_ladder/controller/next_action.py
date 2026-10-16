"""[LAYA-INT-1] Per-action decision (architecture R2.1, minimal): rank candidate NEXT ACTIONS with trusted evidence first and the LAYA score as L0 last.
Pure standard library + the contract core. A learned score never approves, never reorders across trusted evidence levels (I2) and never rescues a candidate a trusted check failed.

trusted[i] = {"L5": "pass"|"fail"|None, "L3": "pass"|"fail"|None, "policy": "pass"|"fail"|None}: a trusted FAIL (any level, or a hard-policy fail) excludes the option from selection.
trusted_passes = number of trusted "pass" entries. The scorer enters ONLY as `scorer` in C.rank, after trusted_passes / validated_passes / cluster.
Writes journal entries: "scorer" (who scored, exploratory flag, scores), "ranking" (the items and the order, checkable by invariants.check_i2) and "action_decision"."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from evidence_ladder.common.contract_import import C


def decide_next_action(journal, decision_id: str, option_texts: Sequence[str], l0_scores: Optional[Sequence[float]], trusted: Optional[Sequence[Dict[str, Optional[str]]]] = None,
                       *, scorer_id: Optional[str] = None, scorer_provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    n = len(option_texts)
    if l0_scores is not None and len(l0_scores) != n:
        raise ValueError("wrong number of scores")
    trusted = list(trusted) if trusted is not None else [{} for _ in range(n)]
    ids = ["%s#%d" % (decision_id, i) for i in range(n)]
    blocked = [any(v == "fail" for v in (trusted[i] or {}).values()) for i in range(n)]
    items = [{"id": ids[i], "trusted_passes": sum(1 for v in (trusted[i] or {}).values() if v == "pass"), "validated_passes": 0, "cluster": 0,
              "scorer": float(l0_scores[i]) if l0_scores is not None else 0.0, "deepconf": 0.0} for i in range(n)]
    if l0_scores is not None:
        journal.append("scorer", {"state": "scored", "scorer_id": scorer_id, "decision_id": decision_id, "level": "L0", "ids": ids, "scores": [float(x) for x in l0_scores],
                                  "provenance": scorer_provenance or {}})
    order = [x["id"] for x in C.rank([it for it, b in zip(items, blocked) if not b])] + [ids[i] for i in range(n) if blocked[i]]
    journal.append("ranking", {"step": decision_id, "items": [it for it, b in zip(items, blocked) if not b], "order": [o for o in order if not blocked[ids.index(o)]]})
    sel = next((o for o in order if not blocked[ids.index(o)]), None)
    out = {"decision_id": decision_id, "selected_id": sel, "selected_index": ids.index(sel) if sel else None, "order": order, "blocked": [ids[i] for i in range(n) if blocked[i]],
           "approved": False, "level": "L0_ordering_only", "derived_by": "controller"}
    journal.append("action_decision", out)
    return out
