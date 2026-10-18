"""[EL-ARCH-1 C5] Evaluation gates of architecture section 9: G-0 headroom, per-step equal-compute keep/drop, false-approval cap vs B, safety, I1-I8-before-run.
Minimum sample sizes and parameters: configs/el_arch1_gates_v1.toml (versioned, written before any run). Statistical gates return INSUFFICIENT_DATA, never PASS, when n is below the minimum.
Arm E (stage-1 LAYA) has no gate here: step_gate("E") refuses. Pure standard library; no model, no network."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from evidence_ladder.orchestrator import stats
from evidence_ladder.orchestrator.arms import laya_keep_drop
from evidence_ladder.orchestrator.base import REPO, ConfigError

GATES_PATH = REPO / "configs" / "el_arch1_gates_v1.toml"
INSUFFICIENT, PASS, FAIL, KEEP, DROP = "INSUFFICIENT_DATA", "PASS", "FAIL", "KEEP", "DROP"
STEP_ARMS = ("C", "D0", "D1")      # each is compared with its equal-compute twin, arm B (plain best-of-k at matched tokens)
TWIN = "B"
INVARIANTS = tuple("I%d" % i for i in range(1, 9))


def load_gates(path=GATES_PATH) -> dict:
    import tomllib
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    need = {"min_n": ("g0_problems", "step_pairs", "safety_approved", "monotonicity_informative_pools", "monotonicity_pools_per_rank"), "g0": ("headroom_points",),
            "per_step": ("ci_level", "bootstrap_resamples", "compute_match_tolerance"), "monotonicity": ("auc_lower_bound", "rank_tolerance")}
    for sec, keys in need.items():
        miss = [k for k in keys if k not in cfg.get(sec, {})]
        if miss:
            raise ConfigError("gates file lacks [%s] %s" % (sec, miss))
    return cfg


def total_tokens(rec: Mapping[str, Any]) -> int:
    """Everything spent on a task, INCLUDING inputs: qwen_gen + qwen_prefill + glm_in + glm_out (the cap T counts only generated tokens)."""
    return int(rec["qwen_gen_tokens"]) + int(rec["qwen_prefill_tokens"]) + int(rec["glm_tokens_in"]) + int(rec["glm_tokens_out"])


def g0_headroom(problems: Sequence[Sequence[bool]], cfg: Optional[dict] = None) -> Dict[str, Any]:
    """problems: per problem, the hidden-oracle outcomes of its k samples (first = the single attempt). pass@1 = first sample, oracle@k = any sample."""
    cfg = cfg or load_gates()
    n, need = len(problems), cfg["min_n"]["g0_problems"]
    out: Dict[str, Any] = {"gate": "G-0 headroom", "n_problems": n, "min_n": need, "required_points": cfg["g0"]["headroom_points"]}
    if n == 0 or n < need:
        out.update(status=INSUFFICIENT, reason="%d problems with per-sample grades, minimum %d" % (n, need))
        return out
    p1 = [int(bool(s[0])) for s in problems]
    ok = [int(any(s)) for s in problems]
    pts = 100.0 * (sum(ok) - sum(p1)) / n
    out.update(pass_at_1=sum(p1) / n, oracle_at_k=sum(ok) / n, headroom_points=pts, status=PASS if pts >= cfg["g0"]["headroom_points"] else FAIL,
               reason="headroom %.1f points vs required %.1f; FAIL means stop" % (pts, cfg["g0"]["headroom_points"]))
    return out


def matched_compute(rows: Sequence[Mapping[str, Any]], step_arm: str, twin_arm: str = TWIN, cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Equal-compute check on the tasks both arms ran: total tokens INCLUDING inputs must agree within the tolerance on the median task."""
    cfg = cfg or load_gates()
    a = {r["task_id"]: total_tokens(r["budget_record"]) for r in rows if r["arm"] == step_arm}
    b = {r["task_id"]: total_tokens(r["budget_record"]) for r in rows if r["arm"] == twin_arm}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"matched": False, "reason": "no task has budget records for both %s and %s" % (step_arm, twin_arm)}
    med = statistics.median(abs(a[t] - b[t]) / max(b[t], 1) for t in ids)
    return {"matched": med <= cfg["per_step"]["compute_match_tolerance"], "median_relative_gap": med, "tolerance": cfg["per_step"]["compute_match_tolerance"], "n_tasks": len(ids),
            "step_tokens_total": sum(a[t] for t in ids), "twin_tokens_total": sum(b[t] for t in ids)}


def false_approval(res, rows, arm) -> Dict[str, Any]:
    lab = {(r["task_id"], r["arm"]): r["answer_label"] for r in rows}
    appr = [t for t, r in res.items() if arm in r and lab.get((t, arm)) in ("VERIFIED", "SUPPORTED")]
    fa = sum(not res[t][arm]["passed"] for t in appr)
    return {"arm": arm, "approved": len(appr), "false_approvals": fa, "rate": fa / len(appr) if appr else None}


def safety_gate(res, rows, arm: str, ref_arm: str = TWIN, cfg: Optional[dict] = None) -> Dict[str, Any]:
    """False-approval rate of `arm` is no higher than the reference arm's (B1). Both arms need the minimum number of approved answers."""
    cfg = cfg or load_gates()
    need = cfg["min_n"]["safety_approved"]
    a, b = false_approval(res, rows, arm), false_approval(res, rows, ref_arm)
    out = {"gate": "safety: false-approval rate of %s no higher than %s" % (arm, ref_arm), "arm": a, "reference": b, "min_approved": need}
    if a["approved"] < need or b["approved"] < need:
        out.update(status=INSUFFICIENT, reason="approved answers %d (%s) and %d (%s), minimum %d each" % (a["approved"], arm, b["approved"], ref_arm, need))
    else:
        out.update(status=PASS if a["rate"] <= b["rate"] else FAIL, reason="false-approval rate %.3f vs %.3f" % (a["rate"], b["rate"]))
    return out


def step_gate(res, rows, step_arm: str, twin_arm: str = TWIN, cfg: Optional[dict] = None, seed: int = 0) -> Dict[str, Any]:
    """Keep a step only if its verified-completion gain over its equal-compute twin has a task-clustered CI lower bound above 0, or it lowers false approvals without lowering completion."""
    if step_arm == "E":
        laya_keep_drop()                      # raises StageOneLayaDecisionRefused
    if step_arm not in STEP_ARMS:
        raise ValueError("no keep/drop gate for arm %r" % step_arm)
    cfg = cfg or load_gates()
    ids = sorted(t for t, r in res.items() if step_arm in r and twin_arm in r)
    out: Dict[str, Any] = {"gate": "per-step keep/drop: %s vs equal-compute twin %s" % (step_arm, twin_arm), "n_pairs": len(ids), "min_pairs": cfg["min_n"]["step_pairs"]}
    if len(ids) < cfg["min_n"]["step_pairs"]:
        out.update(status=INSUFFICIENT, reason="%d paired problems, minimum %d: no keep/drop verdict" % (len(ids), cfg["min_n"]["step_pairs"]))
        return out
    mc = matched_compute(rows, step_arm, twin_arm, cfg)
    out["compute_match"] = mc
    if not mc["matched"]:
        why = mc.get("reason") or "median gap %.3f > %.2f" % (mc["median_relative_gap"], mc["tolerance"])
        out.update(status=INSUFFICIENT, reason="the twin is not equal-compute (%s): the comparison is not the one the gate defines" % why)
        return out
    x = [int(res[t][step_arm]["passed"]) for t in ids]
    y = [int(res[t][twin_arm]["passed"]) for t in ids]
    ci = stats.bootstrap_diff_ci(x, y, cfg["per_step"]["bootstrap_resamples"], seed, level=cfg["per_step"]["ci_level"])
    fs, ft = false_approval(res, rows, step_arm), false_approval(res, rows, twin_arm)
    need = cfg["min_n"]["safety_approved"]
    out.update(completion_gain=ci, false_approval_step=fs, false_approval_twin=ft)
    if ci["lo"] > 0:
        out.update(status=KEEP, reason="completion gain CI lower bound %.3f > 0" % ci["lo"])
    elif fs["approved"] >= need and ft["approved"] >= need and fs["rate"] < ft["rate"] and ci["estimate"] >= 0:
        out.update(status=KEEP, reason="lower false-approval rate (%.3f vs %.3f) without lowering completion" % (fs["rate"], ft["rate"]))
    else:
        out.update(status=DROP, reason="completion gain CI lower bound %.3f is not above 0 and no qualifying false-approval reduction" % ci["lo"])
    return out


def invariants_gate(unit_results: Optional[Mapping[str, bool]]) -> Dict[str, Any]:
    """I1-I8 unit tests all pass BEFORE any run. Not statistical: a missing or false entry is FAIL, never INSUFFICIENT_DATA."""
    r = dict(unit_results or {})
    bad = [i for i in INVARIANTS if r.get(i) is not True]
    return {"gate": "I1-I8 unit tests before the run", "status": FAIL if bad else PASS, "not_passing": bad, "reason": ("missing or failing: %s" % bad) if bad else "all eight pass"}


def overall(statuses: Sequence[str]) -> str:
    """FAIL dominates, then INSUFFICIENT_DATA; PASS only when nothing is missing."""
    if FAIL in statuses:
        return FAIL
    if INSUFFICIENT in statuses:
        return INSUFFICIENT
    return PASS


def section_gates(res, rows, *, headroom_samples=None, invariant_results=None, cfg: Optional[dict] = None, seed: int = 0) -> Dict[str, Any]:
    cfg = cfg or load_gates()
    g: Dict[str, Any] = {"config": "configs/el_arch1_gates_v1.toml", "g0": g0_headroom(headroom_samples or [], cfg), "invariants": invariants_gate(invariant_results),
                         "steps": {a: step_gate(res, rows, a, TWIN, cfg, seed) for a in STEP_ARMS}, "safety": {a: safety_gate(res, rows, a, TWIN, cfg) for a in STEP_ARMS},
                         "arm_E": "no keep/drop gate: exploratory transfer test"}
    flat = [g["g0"]["status"], g["invariants"]["status"]] + [s["status"] for s in g["steps"].values()] + [s["status"] for s in g["safety"].values()]
    g["overall"] = overall([PASS if s in (KEEP, DROP) else s for s in flat])   # a DROP is a verdict about the step, not a missing gate
    return g
