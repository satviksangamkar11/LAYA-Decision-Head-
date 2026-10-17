"""[EL-ARCH-1 B7] Final verification of the ACTUAL end state (architecture stage 8). After the last action the trusted checks are re-run on the real workspace (AgenticRun.finish -> _observe("finalize"),
never on a dry-run scratch copy), the diff against the intake snapshot is compared with the declared scope, and the final label is set through decision.authorize / decide_stage1 on the end-state
evidence. The report carries two separate field groups: `outcome` (did the end state meet the goal) and `process` (were the constraints respected). Pure standard library."""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .decision import authorize


def diff_scope(baseline: Dict[str, str], now: Dict[str, str], scope: Optional[Sequence[str]], protected: Sequence[str]) -> Dict[str, Any]:
    """added / removed / modified files since intake; violations = any change outside the scope patterns (scope None = no scope declared) and ANY change to a protected file."""
    added = sorted(set(now) - set(baseline))
    removed = sorted(set(baseline) - set(now))
    modified = sorted(p for p in set(now) & set(baseline) if now[p] != baseline[p])
    prot = {Path(p).as_posix() for p in protected}
    viol = []
    for p in added + removed + modified:
        if p in prot or Path(p).name in prot:
            viol.append(p + " (protected)")
        elif scope is not None and not any(fnmatch.fnmatch(p, pat) for pat in scope):
            viol.append(p + " (outside scope)")
    return {"added": added, "removed": removed, "modified": modified, "violations": viol, "ok": not viol}


def final_report(run, fin: Dict[str, Any], decision: Dict[str, Any], before: Optional[str] = None) -> Dict[str, Any]:
    from . import tiers as TIERS
    sid = fin["state_id"]
    ds = diff_scope(run.baseline, TIERS.snapshot(run.task.workspace), run.task.scope, run.task.protected)
    auth = authorize(run.reqs, run.journal.evidence(sid), sid, sid, run.env, run.registry, "sandboxed", False)      # the label's authorization, recomputed from the end-state evidence
    tier = (run.tier_last or {}).get("tier")
    outcome = {"end_state_id": sid, "trusted_checks_rerun_on_real_end_state": True, "tests_pass": bool(fin["tests_pass"]), "answer_label": decision["answer_label"],
               "label_ceiling": TIERS.CEILING.get(tier), "authorized_by_authorize": bool(auth), "state_unchanged_since_last_action": (before is None or before == sid)}
    process = {"protected_unchanged": bool(fin["protected_unchanged"]), "diff": {k: ds[k] for k in ("added", "modified", "removed")}, "scope_declared": run.task.scope,
               "diff_scope_ok": ds["ok"], "diff_violations": ds["violations"], "actions_executed": run.actions, "actions_denied_by_risk_table": run.denied, "tier": tier,
               "constraints_respected": bool(fin["protected_unchanged"]) and ds["ok"]}
    return {"outcome": outcome, "process": process, "coverage": coverage_table(run, sid, decision)}


def coverage_table(run, sid: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    """[A2 / T15] one row per requirement: its label, and every evidence record behind it with check, status, scope, source, independence, trust class and whether it counts as authoritative.
    `weakest` is the row that sets the answer flag (the flag is the weakest row; one open requirement keeps it below VERIFIED)."""
    from evidence_ladder.common.contract_import import C
    evs = run.journal.evidence(sid)
    by = {r["requirement_id"]: r for r in run.reqs}
    rows = []
    for row in decision["requirement_rows"]:
        recs = [e for e in evs if e["requirement_id"] == row["requirement_id"]]
        rows.append({"requirement_id": row["requirement_id"], "kind": by[row["requirement_id"]]["kind"], "priority": by[row["requirement_id"]]["priority"], "label": row["label"],
                     "evidence": [{"evidence_id": e["evidence_id"], "check_id": e["check_id"], "status": e["status"], "scope": e["scope"], "source": e["source"], "independence": e["independence"],
                                   "trust_class": run.registry.checks[e["check_id"]]["trust_class"], "authoritative": bool(C.authoritative(e, run.registry))} for e in recs]})
    weakest = next((r for r in rows if C.answer_label([r["label"]]) == decision["answer_label"]), None)
    return {"rows": rows, "answer_label": decision["answer_label"], "weakest": None if weakest is None else {"requirement_id": weakest["requirement_id"], "label": weakest["label"]}}
