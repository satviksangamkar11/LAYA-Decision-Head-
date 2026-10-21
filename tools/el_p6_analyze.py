"""[P6] FROZEN analysis (pre-registered measures). Reads finished run dirs (RESULTS.jsonl, IDENTITY.jsonl, GRADED_cf.jsonl / GRADED_contract_private.jsonl) and the
P6 lock, prints ONE JSON report. Nothing here changes after results are seen.
Conventions: "accepted" = the public label (SUPPORTED / ANSWERED); "solved" = graded PASS. Logical tokens count every generation an arm used (shared ones too), own tokens
only the cache misses the arm itself caused; seconds are wall time and are NOT tokens. Rows or grades that are missing are infrastructure/implementation failures, listed
separately and never counted as candidate failures; per-arm counts and paired statistics use only tasks complete for the arms involved.
Bootstrap: resample tasks with replacement, 10000 times, random.Random(1), percentile CI (sorted means [int(.025*N)] and [int(.975*N)-1]).
wrong_answer_repaired is a subset of correct_gained; other_changed holds every change that fits no named transition.
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_p6_analyze.py --lock configs/p6_final_lock_v1.json [--cf RUNDIR ...] [--contract RUNDIR ...]"""
import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ARMS = ("A", "B", "V2")


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()] if p.exists() else []


def collect(dirs, ids, grade_file, infra, cohort):
    """-> (res[(task,arm)], gr[(task,arm)]); duplicates / unknown tasks / missing rows and grades go to infra."""
    res, gr = {}, {}
    for d in dirs:
        for r in jl(Path(d) / "RESULTS.jsonl"):
            if r["arm"] == "_task":
                continue
            k = (r["task"], r["arm"])
            if r["task"] not in ids:
                infra["unexpected_tasks"].append("%s/%s" % (cohort, r["task"]))
            elif k in res:
                infra["duplicate_rows"].append("%s/%s/%s" % (cohort, *k))
            else:
                res[k] = r
        for g in jl(Path(d) / grade_file):
            k = (g["task"], g["arm"])
            if g["task"] in ids and k not in gr:
                gr[k] = g
    for t in sorted(ids):
        for a in ARMS:
            if (t, a) not in res:
                infra["missing_rows"].append("%s/%s/%s" % (cohort, t, a))
            elif (t, a) not in gr:
                infra["missing_grades"].append("%s/%s/%s" % (cohort, t, a))
    return res, gr


def gv(gr, t, a):
    g = gr.get((t, a))
    return g["grade"] if g else None


def arm_stats(res, gr, ids, a):
    done = [t for t in sorted(ids) if (t, a) in res and (t, a) in gr]
    rows = [(res[(t, a)], gv(gr, t, a)) for t in done]
    o = {"tasks_complete": len(done), "solved": sum(g == "PASS" for _, g in rows), "wrong_answer": sum(g == "FAIL" for _, g in rows),
         "no_answer": sum(g == "NO_ANSWER" for _, g in rows),
         "tokens_generated_logical": sum(r["tokens_generated_logical"] for r, _ in rows), "tokens_generated_own": sum(r["tokens_generated_own"] for r, _ in rows),
         "seconds_median": statistics.median([r["seconds"] for r, _ in rows]) if rows else None, "calls": sum(r["calls"] for r, _ in rows)}
    if a == "A":
        o["wrong_answers_unclaimed"] = sum(r["label"] == "ANSWERED" and g == "FAIL" for r, g in rows)      # A makes no claim: not a false acceptance
    else:
        o["false_acceptances"] = sum(r["label"] == "SUPPORTED" and g == "FAIL" for r, g in rows)          # public label SUPPORTED, graded FAIL
        o["supported"] = sum(r["label"] == "SUPPORTED" for r, _ in rows)
        o["blocked"] = sum(r["label"] == "BLOCKED" for r, _ in rows)
    return o


def paired(res, gr, ids, x, y):
    """solve-rate difference y - x over tasks complete for both arms, bootstrap CI."""
    ts = [t for t in sorted(ids) if all((t, a) in res and (t, a) in gr for a in (x, y))]
    d = [(gv(gr, t, y) == "PASS") - (gv(gr, t, x) == "PASS") for t in ts]
    if not d:
        return {"n_tasks": 0}
    rng, n, N = random.Random(1), len(d), 10000
    m = sorted(sum(rng.choices(d, k=n)) / n for _ in range(N))
    lo, hi = m[int(0.025 * N)], m[int(0.975 * N) - 1]
    return {"n_tasks": n, "solved_x": sum(gv(gr, t, x) == "PASS" for t in ts), "solved_y": sum(gv(gr, t, y) == "PASS" for t in ts), "diff_point": sum(d) / n,
            "ci95": [lo, hi], "ci_excludes_0": lo > 0 or hi < 0, "tasks": ts}


def analyze(lock, cf_dirs, contract_dirs):
    infra = {"missing_rows": [], "missing_grades": [], "duplicate_rows": [], "unexpected_tasks": [], "identity_missing": []}
    rep = {"protocol": lock.get("protocol"), "note": "development results: public acceptance (label) and graded correctness are separate; tokens are logical, seconds are wall time"}
    crit = {}
    if cf_dirs:
        ids = set(lock["cf"]["final_ids"])
        res, gr = collect(cf_dirs, ids, "GRADED_cf.jsonl", infra, "cf")
        idn = {}
        for d in cf_dirs:
            for r in jl(Path(d) / "IDENTITY.jsonl"):
                idn.setdefault(r["task"], r)
        infra["identity_missing"] = sorted(ids - set(idn))
        bad = sorted(t for t, r in idn.items() if t in ids and not (r["identical"] and r["all_unavailable"]))
        a, b = arm_stats(res, gr, ids, "A"), arm_stats(res, gr, ids, "B")
        p = paired(res, gr, ids, "A", "B")
        tr = {"logical": b["tokens_generated_logical"] / a["tokens_generated_logical"] if a["tokens_generated_logical"] else None,
              "A_logical": a["tokens_generated_logical"], "B_logical": b["tokens_generated_logical"], "B_own": b["tokens_generated_own"]}
        extra = b["solved"] - a["solved"]
        rep["cf"] = {"tasks_in_lock": len(ids), "arms": {"A": a, "B": b}, "B_minus_A": {k: v for k, v in p.items() if k != "tasks"}, "extra_correct_B_over_A": extra,
                     "token_ratio_B_over_A": tr, "own_tokens_per_extra_correct": b["tokens_generated_own"] / extra if extra > 0 else None,
                     "identity": {"tasks_checked": len(idn), "failures": bad, "pass": not bad and not infra["identity_missing"],
                                  "note": "any failure is an IMPLEMENTATION failure (V2 must equal B exactly, every gate verdict unavailable)"}}
        crit["cf_B_beats_A"] = bool(p.get("ci_excludes_0") and p.get("diff_point", 0) > 0)
        crit["cf_identity_check_pass"] = rep["cf"]["identity"]["pass"]
    if contract_dirs:
        ids = set(lock["contract"]["ids"])
        res, gr = collect(contract_dirs, ids, "GRADED_contract_private.jsonl", infra, "contract")
        st = {a: arm_stats(res, gr, ids, a) for a in ARMS}
        tr = Counter()
        both = [t for t in sorted(ids) if all((t, a) in res and (t, a) in gr for a in ("B", "V2"))]
        for t in both:
            bg, vg, bl = gv(gr, t, "B"), gv(gr, t, "V2"), res[(t, "B")]["label"]
            tr["unchanged"] += bg == vg
            tr["wrong_answer_prevented"] += bg == "FAIL" and bl == "SUPPORTED" and vg == "NO_ANSWER"
            tr["wrong_answer_repaired"] += bg == "FAIL" and vg == "PASS"
            tr["correct_lost"] += bg == "PASS" and vg != "PASS"
            tr["correct_gained"] += bg != "PASS" and vg == "PASS"
        tr["other_changed"] = sum(1 for t in both if gv(gr, t, "B") != gv(gr, t, "V2")) - (tr["wrong_answer_prevented"] + tr["correct_lost"] + tr["correct_gained"])
        mix = Counter(json.dumps(v, sort_keys=True) if not isinstance(v, str) else v for t in sorted(ids) if (t, "V2") in res for v in res[(t, "V2")]["gate_verdicts"])
        vb, ba = paired(res, gr, ids, "B", "V2"), paired(res, gr, ids, "A", "B")
        rep["contract"] = {"tasks_in_lock": len(ids), "arms": st, "V2_minus_B": {k: v for k, v in vb.items() if k != "tasks"}, "B_minus_A": {k: v for k, v in ba.items() if k != "tasks"},
                           "transitions_B_to_V2": dict(tr), "tasks_in_transitions": len(both), "V2_gate_verdict_mix": dict(mix)}
        crit["contract_V2_false_acceptances_le_B"] = st["V2"]["false_acceptances"] <= st["B"]["false_acceptances"]
        crit["contract_correct_lost_is_0"] = tr["correct_lost"] == 0
        crit["contract_V2_solves_ge_B"] = st["V2"]["solved"] >= st["B"]["solved"]
        crit["contract_gate_ok"] = all(crit[k] for k in crit if k.startswith("contract_"))
    rep["criteria"] = crit
    rep["infrastructure"] = {**infra, "complete": not any(infra.values()),
                             "note": "implementation/infrastructure failures, never candidate failures; criteria above use only complete tasks"}
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock", required=True)
    ap.add_argument("--cf", nargs="*", default=[])
    ap.add_argument("--contract", nargs="*", default=[])
    a = ap.parse_args()
    if not (a.cf or a.contract):
        sys.exit("give at least one of --cf / --contract run dirs")
    print(json.dumps(analyze(json.loads(Path(a.lock).read_text("utf-8")), a.cf, a.contract), indent=1))
