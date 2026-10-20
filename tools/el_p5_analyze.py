"""[P5 final] Analysis of the graded final run, exactly as pre-registered in P5_PROTOCOL_v2 (written and committed BEFORE any final result existed).
Inputs: the final output dir (one run dir per shard: p5_final_lcb_1..5, p5_final_contract) with RESULTS.jsonl, ANSWERS.jsonl, CALLS.jsonl and the grades
(LCB: GRADED.jsonl; contract: GRADED_primary.jsonl, GRADED_secondary.jsonl, GRADED_nonblind.jsonl), plus configs/p5_final_lock_v2.json.
Measures per cohort and arm: solved (PASS), wrong answer (answered, FAIL), false SUPPORTED (an arm's SUPPORTED/VERIFIED claim that grading FAILS; arms B, C, D),
abstention (no answer), correct answers wrongly blocked (D vs C: C correct, D no answer), median seconds, tokens. Paired 95% intervals: bootstrap over tasks,
10,000 resamples, random.Random(1), percentile. Success criteria evaluated literally. Infrastructure: missing tasks / shard failures listed separately.
Usage: python tools/el_p5_analyze.py <final output dir> > <dir>/ANALYSIS.json"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = Path(sys.argv[1])
lock = json.loads((ROOT / "configs/p5_final_lock_v2.json").read_text("utf-8"))
ARMS = ("A", "B", "C", "D")
CLAIM = {"SUPPORTED", "VERIFIED"}


def rows(p):
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()] if p.exists() else []


def load(shards, grade_name):
    res, grd, infra = {}, {}, []
    for s in shards:
        d = D / ("p5_" + s)
        if not (d / "DONE.json").exists():
            infra.append({"shard": s, "issue": "no DONE.json (shard incomplete)"})
        for r in rows(d / "RESULTS.jsonl"):
            if r["arm"] in ARMS:
                res[(r["task"], r["arm"])] = r
        for g in rows(d / grade_name):
            grd[(g["task"], g["arm"])] = g
    return res, grd, infra


def boot(tasks, f, n=10000):
    r = random.Random(1)
    vals = sorted(f([tasks[r.randrange(len(tasks))] for _ in tasks]) for _ in range(n))
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n) - 1], 4)]


def cohort(tasks, res, grd):
    ok = lambda t, a: grd.get((t, a), {}).get("grade") == "PASS"
    ans = lambda t, a: grd.get((t, a), {}).get("grade") in ("PASS", "FAIL")
    claim = lambda t, a: res.get((t, a), {}).get("label") in CLAIM
    missing = [(t, a) for t in tasks for a in ARMS if (t, a) not in res or (t, a) not in grd]
    out = {"n_tasks": len(tasks), "missing_task_arm": missing, "arms": {}}
    for a in ARMS:
        secs = [res[(t, a)].get("seconds") for t in tasks if (t, a) in res and res[(t, a)].get("seconds") is not None]
        toks = [res[(t, a)].get("tokens_generated") or 0 for t in tasks if (t, a) in res]
        out["arms"][a] = {"solved": sum(ok(t, a) for t in tasks), "wrong_answer": sum(ans(t, a) and not ok(t, a) for t in tasks),
                          "no_answer": sum(not ans(t, a) for t in tasks),
                          "false_supported": None if a == "A" else sum(claim(t, a) and not ok(t, a) for t in tasks),
                          "supported_claims": None if a == "A" else sum(claim(t, a) for t in tasks),
                          "median_seconds": statistics.median(secs) if secs else None, "generated_tokens_total": sum(toks),
                          "generated_tokens_median": statistics.median(toks) if toks else None}
        out["arms"][a]["solve_rate"] = round(out["arms"][a]["solved"] / len(tasks), 4) if tasks else None
    diff = lambda x, y: (lambda ts: (sum(ok(t, x) for t in ts) - sum(ok(t, y) for t in ts)) / len(ts))
    out["paired_solve_rate_diff_ci95"] = {"%s-%s" % (x, y): {"point": round(diff(x, y)(tasks), 4), "ci95": boot(tasks, diff(x, y))}
                                          for x, y in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "C"), ("C", "B"), ("D", "B"))}
    fs = lambda ts: (sum(claim(t, "D") and not ok(t, "D") for t in ts) - sum(claim(t, "C") and not ok(t, "C") for t in ts)) / len(ts)
    out["paired_false_supported_diff_D_minus_C_ci95"] = {"point": round(fs(tasks), 4), "ci95": boot(tasks, fs)}
    out["wrongly_blocked_D_vs_C"] = [t for t in tasks if ok(t, "C") and not ans(t, "D")]
    out["D_differs_from_C"] = [t for t in tasks if grd.get((t, "C"), {}).get("grade") != grd.get((t, "D"), {}).get("grade")]
    return out


def wall_equiv_overhead(shards):
    import subprocess
    dirs = [str(D / ("p5_" + s)) for s in shards if (D / ("p5_" + s) / "CALLS.jsonl").exists()]      # a missing shard is reported under infrastructure
    if not dirs:
        return {"median_D_vs_C_wall_equivalent_overhead": None, "tasks": 0, "note": "no shard output"}
    p = subprocess.run([sys.executable, str(ROOT / "tools/el_p5_cost_accounting.py")] + dirs, capture_output=True, text=True)
    rep = json.loads(p.stdout)
    eq = {}
    for v in rep.values():
        for r in v["rows"]:
            eq[(r["task"], r["arm"])] = r["wall_equivalent_s"]
    rel = sorted((eq[(t, "D")] - eq[(t, "C")]) / eq[(t, "C")] for (t, a) in eq if a == "C" and eq.get((t, "D")) is not None and eq[(t, "C")])
    tot = {a: round(sum(v["totals_by_arm"].get(a, {}).get("wall_equivalent_s", 0) for v in rep.values()), 1) for a in ARMS}
    gen = {a: round(sum(v["totals_by_arm"].get(a, {}).get("reported_gen_tokens", 0) for v in rep.values())) for a in ARMS}
    return {"median_D_vs_C_wall_equivalent_overhead": round(rel[len(rel) // 2], 4) if rel else None, "tasks": len(rel),
            "wall_equivalent_s_total": tot, "logical_generated_tokens_total": gen}


def main():
    lcb_shards = ["final_lcb_%d" % k for k in range(1, 6)]
    rep = {"protocol": lock["protocol"], "protocol_sha256": lock["protocol_sha256"], "lock": "configs/p5_final_lock_v2.json"}
    summ = json.loads((D / "RUN_SUMMARY.json").read_text("utf-8")) if (D / "RUN_SUMMARY.json").exists() else {}
    rep["kernel"] = {k: summ.get(k) for k in ("verdict", "bundle", "gpus", "shard_exit_codes", "total_seconds", "error")}
    res, grd, infra = load(lcb_shards, "GRADED.jsonl")
    rep["lcb"] = cohort(lock["lcb_ids"], res, grd)
    rep["lcb"]["cost"] = wall_equiv_overhead(lcb_shards)
    rep["infrastructure"] = list(infra)
    rep["contract"] = {}
    for which in ("primary", "secondary", "nonblind"):
        res, grd, infra = load(["final_contract"], "GRADED_%s.jsonl" % which)
        rep["contract"][which] = cohort(lock["contract_ids"], res, grd)
        if which == "primary":
            rep["infrastructure"] += infra
    rep["contract"]["cost"] = wall_equiv_overhead(["final_contract"])
    P, L = rep["contract"]["primary"], rep["lcb"]
    excl0 = lambda ci: ci[0] > 0 or ci[1] < 0
    rep["success_criteria"] = {
        "contract_D_false_supported_lower_than_C": P["arms"]["D"]["false_supported"] < P["arms"]["C"]["false_supported"],
        "contract_correct_wrongly_blocked_zero": len(P["wrongly_blocked_D_vs_C"]) == 0,
        "contract_D_solves_not_lower_than_C": P["arms"]["D"]["solved"] >= P["arms"]["C"]["solved"],
        "lcb_D_solves_not_lower_than_C": L["arms"]["D"]["solved"] >= L["arms"]["C"]["solved"],
        "lcb_median_D_wall_overhead_le_10pct": (L["cost"]["median_D_vs_C_wall_equivalent_overhead"] or 0) <= 0.10,
        "beats_plain_lcb": {k: excl0(L["paired_solve_rate_diff_ci95"][k]["ci95"]) and L["paired_solve_rate_diff_ci95"][k]["point"] > 0 for k in ("C-A", "D-A")},
        "beats_plain_contract_primary": {k: excl0(P["paired_solve_rate_diff_ci95"][k]["ci95"]) and P["paired_solve_rate_diff_ci95"][k]["point"] > 0 for k in ("C-A", "D-A")},
        "infrastructure_unresolved": len(rep["infrastructure"]) + len(L["missing_task_arm"]) + len(P["missing_task_arm"]),
    }
    sc = rep["success_criteria"]
    rep["freeze_v15_all_D_vs_C_criteria_hold"] = all(sc[k] for k in ("contract_D_false_supported_lower_than_C", "contract_correct_wrongly_blocked_zero",
                                                                    "contract_D_solves_not_lower_than_C", "lcb_D_solves_not_lower_than_C",
                                                                    "lcb_median_D_wall_overhead_le_10pct")) and sc["infrastructure_unresolved"] == 0
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
