"""Per-track paired report. Three separate tracks; no function here can combine them."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from evidence_ladder.orchestrator import gates as gates_mod
from evidence_ladder.orchestrator import stats
from evidence_ladder.orchestrator.arms import ARMS, CONFIRMATORY, E_LABEL, EXPLORATORY
from evidence_ladder.orchestrator.base import DEV_LABEL, SYNTHETIC_LABEL, PoolingError
from evidence_ladder.orchestrator.budget import MATCHING_LABEL
from evidence_ladder.orchestrator.index import FABLE_RUN_STATUS, WAIVER_NOTE
from evidence_ladder.orchestrator.isolation import NOT_SEALED_STATEMENT

TRACKS = ("python", "java", "fable")
HOLDOUT_NOTE = "PRESERVED FOR HOLDOUT; not confirmatory on the development banks"
POOL_MSG = "tracks are never pooled: Python, Java and Fable use different scorers and different claims"


class EmptyTrackError(Exception):
    pass


def pool_tracks(*_a, **_k):
    raise PoolingError(POOL_MSG)


def track_accuracy(track_results: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    """Per-arm solved counts within ONE track. Refuses when nothing was graded."""
    out = {}
    for arm in ARMS:
        vals = [bool(r[arm]["passed"]) for r in track_results.values() if arm in r]
        if vals:
            out[arm] = {"n_graded": len(vals), "solved": sum(vals), "rate": sum(vals) / len(vals)}
    if not out:
        raise EmptyTrackError("zero graded tasks: no number is produced")
    return out


def _vecs(res, a, b, skip=()):
    ids = sorted(t for t, r in res.items() if a in r and b in r and t not in skip)
    return ids, [int(res[t][a]["passed"]) for t in ids], [int(res[t][b]["passed"]) for t in ids]


def _contrast(res, a, b, skip, seed, boot=True):
    ids, x, y = _vecs(res, a, b)
    c = {"name": f"{a}_vs_{b}", "challenger": a, "baseline": b}
    if not ids:
        c["refused"] = "no task graded under both arms"
        return c
    tab = stats.paired_table(x, y)
    c.update(n_pairs=len(ids), table=tab, mcnemar_p=stats.mcnemar_exact(tab["gained"], tab["lost"]), warnings=stats.sample_size_warnings(len(ids)))
    if boot:
        c["bootstrap_95"] = stats.bootstrap_diff_ci(x, y, 2000, seed)
    ids2, x2, y2 = _vecs(res, a, b, skip)
    if ids2:
        t2 = stats.paired_table(x2, y2)
        c["excluding_flagged_tasks"] = {"n_pairs": len(ids2), "table": t2, "mcnemar_p": stats.mcnemar_exact(t2["gained"], t2["lost"])}
    else:
        c["excluding_flagged_tasks"] = {"refused": "no unflagged paired task"}
    return c


def _rates(res, rows):
    lab = {(r["task_id"], r["arm"]): r["answer_label"] for r in rows}
    out = {}
    for arm in ARMS:
        appr = [t for t, r in res.items() if arm in r and lab.get((t, arm)) in ("VERIFIED", "SUPPORTED")]
        blk = [t for t, r in res.items() if arm in r and lab.get((t, arm)) == "BLOCKED"]
        fa = sum(not res[t][arm]["passed"] for t in appr)
        fb = sum(bool(res[t][arm]["passed"]) for t in blk)
        out[arm] = {"approved": len(appr), "false_approvals": fa, "false_approval_rate": fa / len(appr) if appr else None,
                    "blocked": len(blk), "false_blocks": fb, "false_block_rate": fb / len(blk) if blk else None}
    return out


def _cost(res, rows):
    out = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        gen = sum(r["budget_record"]["qwen_gen_tokens"] + r["budget_record"]["glm_tokens_out"] for r in rs)
        solved = sum(bool(r[arm]["passed"]) for r in res.values() if arm in r)
        out[arm] = {"tasks": len(rs), "generated_tokens_total": gen, "generated_tokens_per_task": gen / len(rs),
                    "generated_tokens_per_solved": gen / solved if solved else None, "glm_calls": sum(r["budget_record"]["glm_calls"] for r in rs),
                    "api_errors": sum(r["budget_record"]["api_errors"] for r in rs), "tool_runs": sum(r["budget_record"]["tool_runs"] for r in rs),
                    "gpu_seconds": sum(r["budget_record"]["gpu_seconds"] for r in rs), "wall_seconds": sum(r["budget_record"]["wall_seconds"] for r in rs)}
    return out


def _section(track, res, meta, rows, synthetic, seed, headroom_samples=None, invariant_results=None):
    for t in res:
        if t not in meta or meta[t].track != track:
            raise PoolingError(f"task {t} is not in track {track}: " + POOL_MSG)
    flags = {t: s.mutation_status for t, s in meta.items() if s.mutation_status.get("flagged")}
    sec: Dict[str, Any] = {"track": track, "n_tasks": len(meta), "n_tasks_with_any_grade": len(res),
                           "mutation_flags": flags, "mutation_flag_counts": dict(Counter(s["state"] for s in flags.values())),
                           "waiver_note": WAIVER_NOTE,
                           "coverage": {a: {"graded": sum(a in r for r in res.values()), "missing": sorted(set(meta) - {t for t, r in res.items() if a in r})}
                                        for a in ARMS}}
    try:
        acc = track_accuracy(res)
    except EmptyTrackError as e:
        sec["numbers"] = "REFUSED: " + str(e)
        return sec
    sec["per_arm"] = {a: ({k: v for k, v in d.items() if k != "rate"} if synthetic else d) for a, d in acc.items()}
    skip = set(flags)
    con = {f"{a}_vs_{b}": _contrast(res, a, b, skip, seed) for a, b in CONFIRMATORY}
    holm = stats.holm({k: c["mcnemar_p"] for k, c in con.items() if "mcnemar_p" in c})
    for k, c in con.items():
        c["status"] = HOLDOUT_NOTE
        c["confirmatory"] = False
        if k in holm:
            c["holm_adjusted_p"] = holm[k]
    exp = {f"{a}_vs_{b}": _contrast(res, a, b, skip, seed) for a, b in EXPLORATORY}
    exp["E_vs_D1"].update(label=E_LABEL, verdict=None, note="stage-1 LAYA is an exploratory transfer test; no keep/drop decision from this data")
    sec.update(contrasts=con, exploratory=exp, safety=_rates(res, rows), cost=_cost(res, rows),
               stop_causes={a: dict(Counter(r["stop_cause"] for r in rows if r["arm"] == a)) for a in ARMS if any(r["arm"] == a for r in rows)})
    sec["gates"] = gates_mod.section_gates(res, rows, headroom_samples=headroom_samples, invariant_results=invariant_results, seed=seed)   # [EL-ARCH-1 C5] INSUFFICIENT_DATA below the minimum n
    return sec


def make_report(results: Dict[str, Dict[str, Dict[str, dict]]], tasks: Dict[str, Dict[str, Any]], selection_rows: List[dict], *,
                T: Optional[int] = None, synthetic: bool = False, seed: int = 0, headroom_samples: Optional[Dict[str, list]] = None,
                invariant_results: Optional[Dict[str, bool]] = None, comparison: Optional[dict] = None) -> dict:
    """results: {track: {task_id: {arm: grade dict with 'passed'}}}; tasks: {track: {task_id: TaskSpec}}; selection_rows: final per task-arm log rows."""
    if set(results) - set(TRACKS):
        raise PoolingError("unknown track key; " + POOL_MSG)
    rep: Dict[str, Any] = {"label": DEV_LABEL, "matching": MATCHING_LABEL, "T": T, "pooled": "NOT PRODUCED: " + POOL_MSG, "tracks": {},
                           "evaluation_oracle_status": NOT_SEALED_STATEMENT,
                           "comparison_scorers": comparison if comparison is not None else {"status": "NOT_RUN", "note": "arm E / comparison scorers did not run"}}
    if synthetic:
        rep["banner"] = SYNTHETIC_LABEL
    for track in TRACKS:
        if track == "fable":
            rep["tracks"][track] = {"track": track, "numbers": "NOT PRODUCED: " + FABLE_RUN_STATUS}
            if results.get("fable"):
                raise NotImplementedError("Fable results cannot be reported: " + FABLE_RUN_STATUS)
            continue
        rows = [r for r in selection_rows if r.get("track") == track]
        rep["tracks"][track] = _section(track, results.get(track, {}), tasks.get(track, {}), rows, synthetic, seed, (headroom_samples or {}).get(track), invariant_results)
    return rep


def render_markdown(rep: dict) -> str:
    L = [f"# {rep['label']}", ""]
    if rep.get("banner"):
        L += [f"**{rep['banner']}**", ""]
    L += [f"Budget: {rep['matching']}. T = {rep['T']}.", rep["pooled"], "", rep["evaluation_oracle_status"], ""]
    cs = rep.get("comparison_scorers", {})
    if cs.get("scorers"):
        L += ["## Comparison scorers (not selectors; no keep/drop)", cs["label"], "", "| scorer | role | pools | informative | top-1 pass (informative) | pool mean pass | monotonicity |", "|---|---|---|---|---|---|---|"]
        for sid, v in cs["scorers"].items():
            f = lambda x: "-" if x is None else "%.3f" % x
            L.append(f"| {sid} | {v['role']} | {v['n_pools']} | {v['n_informative_pools']} | {f(v['top1_pass_rate_informative'])} | {f(v['pool_mean_pass_rate_informative'])} | {v['monotonicity']['status']} |")
        L.append("")
    for track, s in rep["tracks"].items():
        L += [f"## Track: {track}", ""]
        if "per_arm" not in s:
            L += [str(s.get("numbers", "")), ""]
            continue
        L += [f"tasks {s['n_tasks']}; flagged for oracle adequacy: {s['mutation_flag_counts'] or 'none'}", s["waiver_note"], "",
              "| contrast | pairs | gained | lost | both | neither | exact McNemar p | Holm p | status |", "|---|---|---|---|---|---|---|---|---|"]
        for name, c in {**s["contrasts"], **s["exploratory"]}.items():
            t = c.get("table")
            if t:
                L.append(f"| {name} | {c['n_pairs']} | {t['gained']} | {t['lost']} | {t['both']} | {t['neither']} | {c['mcnemar_p']:.4f} | "
                         f"{c.get('holm_adjusted_p', '-')} | {c.get('status') or c.get('label') or 'exploratory'} |")
            else:
                L.append(f"| {name} | - | - | - | - | - | - | - | {c.get('refused')} |")
        L.append("")
        g = s["gates"]
        L += [f"Gates (configs/el_arch1_gates_v1.toml): overall {g['overall']}; G-0 {g['g0']['status']}; I1-I8 {g['invariants']['status']}; "
              + "; ".join(f"{a}: {v['status']}" for a, v in g["steps"].items()), ""]
    return "\n".join(L) + "\n"
