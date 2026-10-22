"""[P7 stage 1] CPU-only diagnostics over the SAVED P6 logs (no generation, no execution) + the pre-registration power table.
Reads the P6 run dirs (CANDIDATES.jsonl, CANDIDATE_USE.jsonl, EXEC.jsonl) and the P7 candidate grades; writes create-only:
  per_task.jsonl  one row per task: B's samples in order, tokens, duplicates, public passes, strict-correct positions
  STAGE1.json     pooled success curve, marginal value of more samples on unsolved tasks, duplicates, static crash flags, power table
Strict correct: Codeforces = graded PASS on the generated tests AND passes the statement examples (P7_HEADROOM definition);
contract = graded PASS on the private blind set. Public pass = every run_public status "pass" in EXEC.jsonl.
Crash flags are STATIC (ast only, nothing executed): no stderr was logged, so exception types are not known here.
Usage: uv run python tools/el_p7_stage1.py --out results/raw/p7-stage1-v1"""
import argparse
import ast
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P6 = ROOT / "results/raw/p6-final-v1"
G7 = ROOT / "results/raw/p7-cand-grade-v1"
COHORTS = {
    "cf": ([P6 / "final_k1/p6_final_cf_1", P6 / "final_k2/p6_final_cf_2"], G7 / "cf_out/GRADED_cf.jsonl"),
    "contract": ([P6 / "final_k2/p6_final_contract"], G7 / "contract/GRADED_contract_private.jsonl"),
}
Z_A, Z_B = 1.959964, 0.841621          # two-sided alpha 0.05, power 0.80


def jl(p):
    return [json.loads(l) for l in Path(p).read_text("utf-8").splitlines() if l.strip()]


def static_flags(code):
    """ast-only flags for one program; never executes it."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"syntax_error": True}
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    recursive = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name
                    for name, f in funcs.items() for c in ast.walk(f))
    src = code
    return {"syntax_error": False, "recursive": recursive, "setrecursionlimit": "setrecursionlimit" in src,
            "reads_sys_stdin": "sys.stdin" in src, "uses_input": "input(" in src,
            "top_level_call": any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) for s in tree.body)
            or any(isinstance(s, ast.If) for s in tree.body)}


def cohort(name, dirs, grade_file):
    grades = {(g["task"], g["arm"]): g for g in jl(grade_file)}
    cand, use, pub = {}, {}, {}
    for d in dirs:
        for c in jl(d / "CANDIDATES.jsonl"):
            k = (c["task"], c["seed"])
            assert k not in cand or cand[k]["code_sha256"] == c["code_sha256"], k
            cand[k] = c
        for u in jl(d / "CANDIDATE_USE.jsonl"):
            use[u["task"]] = u["B"]
        for e in jl(d / "EXEC.jsonl"):
            if e["method"] == "run_public":
                pub[(e["task"], e["artifact"])] = all(s[2] == "pass" for s in e["statuses"])
    rows, missing = [], Counter()
    for task in sorted(use):
        seq = []
        for b in use[task]:
            c = cand.get((task, b["seed"]))
            if c is None:
                missing["candidate_row"] += 1
                continue
            sha = c["code_sha256"]
            g = grades.get((task, sha))
            if sha and g is None:
                missing["grade"] += 1
            p = pub.get((task, sha), b["public_pass"])
            seq.append({"seed": b["seed"], "tokens": c["tokens_generated"], "sha": sha, "extract": c["extract_status"], "public_pass": bool(p),
                        "grade": g and g["grade"], "first_fail": g and g["first_fail"], "passed": g and g["passed"], "total": g and g["total"],
                        "strict": bool(g and g["grade"] == "PASS" and (p or name == "contract"))})
        cum, strict_pos, pub_pos = 0, [], []
        for i, s in enumerate(seq):
            cum += s["tokens"]
            s["cum_tokens"] = cum
            if s["strict"]:
                strict_pos.append(i + 1)
            if s["public_pass"]:
                pub_pos.append(i + 1)
        shas = [s["sha"] for s in seq if s["sha"]]
        rows.append({"cohort": name, "task": task, "samples": len(seq), "tokens": cum, "distinct": len(set(shas)),
                     "extract_fail": sum(s["extract"] != "ok" for s in seq), "public_pass_pos": pub_pos, "strict_pos": strict_pos,
                     "first_strict_tokens": next((s["cum_tokens"] for s in seq if s["strict"]), None),
                     "best_tests_before_first_fail": max([s["passed"] or 0 for s in seq if s["grade"] == "FAIL"], default=None),
                     "first_fail": dict(Counter(s["first_fail"] for s in seq if s["grade"] == "FAIL")), "seq": seq})
    return rows, dict(missing)


def marginal(rows):
    """Unsolved tasks: 0 strict successes in S samples. Upper 95% one-sided bound on a common per-sample rate, and the implied upper bound on
    extra tasks solved if each unsolved task got as many samples again. Heterogeneous rates make the true value smaller for the same S, never larger."""
    uns = [r for r in rows if not r["strict_pos"]]
    S = sum(r["samples"] for r in uns)
    p95 = 1 - 0.05 ** (1 / S) if S else None
    extra = sum(1 - (1 - p95) ** r["samples"] for r in uns) if S else None
    return {"unsolved_tasks": len(uns), "samples_on_unsolved": S, "per_sample_rate_upper95": p95, "extra_solves_if_budget_doubled_upper95": extra}


def power_table():
    """Paired task-level design. d_i = treatment_i - baseline_i in {-1,0,1}; g = P(d=+1), l = P(d=-1), delta = g - l.
    Tasks needed for 80% power, two-sided 0.05 (normal approximation; the paired bootstrap of el_p6_analyze is close at these n)."""
    out = []
    for delta in (0.05, 0.10, 0.15, 0.20):
        for l in (0.0, 0.03, 0.05, 0.10):
            g = delta + l
            var = g + l - delta ** 2
            out.append({"delta": delta, "loss_rate": l, "gain_rate": round(g, 3), "tasks_needed": math.ceil((Z_A + Z_B) ** 2 * var / delta ** 2)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = ROOT / ap.parse_args().out
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    report, all_rows = {"note": "development diagnostic over saved P6 logs; CPU only; nothing generated or executed"}, []
    for name, (dirs, gf) in COHORTS.items():
        rows, missing = cohort(name, dirs, gf)
        all_rows += rows
        solved = [r for r in rows if r["strict_pos"]]
        crash = [s for r in rows for s in r["seq"] if s["first_fail"] == "runtime_error"]
        uniq = {s["sha"]: s for s in crash}
        code = {c["code_sha256"]: c["code"] for d in dirs for c in jl(d / "CANDIDATES.jsonl")}
        flags = Counter()
        for sha, s in uniq.items():
            f = static_flags(code[sha])
            flags.update(k for k, v in f.items() if v)
            flags["public_pass_then_crash"] += s["public_pass"]
            flags["crash_on_first_test"] += (s["passed"] == 0)
        n = sum(r["samples"] for r in rows)
        report[name] = {
            "tasks": len(rows), "missing": missing, "samples": n, "tokens": sum(r["tokens"] for r in rows),
            "solved_strict": len(solved),
            "first_strict_sample_pos": sorted(r["strict_pos"][0] for r in solved),
            "first_strict_tokens": sorted(r["first_strict_tokens"] for r in solved),
            "tasks_with_public_pass": sum(bool(r["public_pass_pos"]) for r in rows),
            "duplicate_rate": 1 - sum(r["distinct"] for r in rows) / n if n else None,
            "extract_fail_rate": sum(r["extract_fail"] for r in rows) / n if n else None,
            "first_fail_all_samples": dict(sum((Counter(r["first_fail"]) for r in rows), Counter())),
            "marginal_value_of_more_samples": marginal(rows),
            "near_miss_unsolved": sorted(((r["best_tests_before_first_fail"] or 0, r["task"]) for r in rows if not r["strict_pos"]), reverse=True)[:10],
            "runtime_error_distinct_programs": len(uniq), "runtime_error_static_flags": dict(flags),
        }
    report["power_table"] = power_table()
    out.mkdir(parents=True)
    (out / "per_task.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in all_rows), encoding="utf-8", newline="\n")
    (out / "STAGE1.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "power_table"}, indent=1, sort_keys=True))
    print(json.dumps(report["power_table"]))


if __name__ == "__main__":
    main()
