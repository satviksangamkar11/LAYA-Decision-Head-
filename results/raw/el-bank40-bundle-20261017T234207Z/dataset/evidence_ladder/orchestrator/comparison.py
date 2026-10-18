"""[EL-ARCH-1 C1/C2/C3] Comparison scorers in the report: LAYA, the linear probe (mandatory baseline) and the random-head floor control, scored on the SAME candidate pools (recorded by
program_scorer.ProgramPoolScorer) and judged against trusted outcomes of the candidates. None of them is a selector here and nothing in this module produces a keep/drop for LAYA:
stage 1 is an out-of-distribution transfer test (arms.E_LABEL). The monotonicity gate is run per scorer as a diagnostic; ordering by LAYA stays disabled unless it is SUPPORTED.
Outcomes come from the protected grader after the selection gate closed (PilotRunner.grade_comparison_candidates); this module only reads them."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from evidence_ladder.controller import monotonicity as MONO
from evidence_ladder.models import laya_forward as LF
from evidence_ladder.orchestrator.arms import E_LABEL

LABEL = ("COMPARISON ONLY: scorers are ranked on identical pools, none is a selector, and stage 1 is out-of-distribution for the next-action head; "
         "no keep/drop for LAYA is produced from this data (" + E_LABEL + ")")


def build_pools(events: Sequence[Mapping[str, Any]], outcomes: Mapping[str, Mapping[str, bool]]) -> Dict[str, List[dict]]:
    """events: ProgramPoolScorer sink rows. outcomes: {task_id: {code_sha256: passed}}. -> {scorer_id: [pool]} for pools with a score and an outcome for EVERY distinct candidate.
    Several events of one task (the controller re-scores as its live set shrinks) are different pools and get distinct ids."""
    out: Dict[str, List[dict]] = {}
    for n, ev in enumerate(events):
        oc = outcomes.get(ev["task_id"], {})
        ids = ev["distinct_code_sha256"]
        if len(ids) < 2 or any(i not in oc for i in ids):
            continue
        for sid, s in ev["scorers"].items():
            if s.get("status") == "ok" and len(s["scores"]) == len(ids):
                out.setdefault(sid, []).append({"id": "%s#%d" % (ev["task_id"], n), "scores": list(s["scores"]), "passed": [bool(oc[i]) for i in ids], "role": s["role"]})
    return out


def top1_tie_aware(scores: Sequence[float], passed: Sequence[bool]) -> float:
    top = max(scores)
    tied = [i for i, v in enumerate(scores) if v == top]
    return sum(1.0 for i in tied if passed[i]) / len(tied)


def scorer_summary(pools: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    informative = [p for p in pools if any(p["passed"]) and not all(p["passed"])]
    return {"role": pools[0]["role"] if pools else None, "n_pools": len(pools), "n_informative_pools": len(informative),
            "top1_pass_rate_all_pools": (sum(top1_tie_aware(p["scores"], p["passed"]) for p in pools) / len(pools)) if pools else None,
            "top1_pass_rate_informative": (sum(top1_tie_aware(p["scores"], p["passed"]) for p in informative) / len(informative)) if informative else None,
            "pool_mean_pass_rate_informative": (sum(sum(p["passed"]) / len(p["passed"]) for p in informative) / len(informative)) if informative else None}


def comparison_report(events: Sequence[Mapping[str, Any]], outcomes: Mapping[str, Mapping[str, bool]], *, gates_cfg: Optional[dict] = None, seed: int = 0) -> Dict[str, Any]:
    pools = build_pools(events, outcomes)
    rep: Dict[str, Any] = {"label": LABEL, "selectors": [], "keep_drop": None, "n_events": len(events), "scorers": {}}
    for sid in sorted(pools):
        rep["scorers"][sid] = dict(scorer_summary(pools[sid]), monotonicity=MONO.monotonicity_gate(pools[sid], config=gates_cfg, seed=seed))
    if not events:
        rep["note"] = "no pool was scored (arm E not run, or the scorer refused every pool)"
    elif LF.ROLES["probe"] not in {s["role"] for s in rep["scorers"].values()}:
        rep["warning"] = "no probe scores for these pools: the mandatory baseline is missing, so no statement about LAYA versus the probe can be made"
    return rep
