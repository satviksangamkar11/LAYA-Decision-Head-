"""[P5 Step 1(c)] Cost accounting from a finished P5 run's records (no execution). Per task and arm:
  logical_gen_tokens   : tokens of every generation the arm USED (shared round-1 and reused calls included; the protocol's budget measure),
  actual_gen_tokens    : tokens of generations the arm itself caused (cache misses only),
  inference_s          : model seconds of those cache misses (CALLS.jsonl actual_seconds),
  arm_wall_s           : the arm's wall seconds (RESULTS.jsonl; arm A has none: its only sample is the shared round 1),
  non_inference_s      : arm_wall_s - inference_s (execution + controller overhead),
  logical_inference_s  : recorded model seconds of EVERY generation the arm requested (C and D re-request round 1 through the cache, B does not: compare C with D only), charged at the seconds of the call that first produced it,
  wall_equivalent_s    : logical_inference_s + non_inference_s: a wall time comparable across arms despite the shared cache (for the protocol's D-vs-C wall criterion).
Shared round 1 is reported once per task (round1_*). Usage: python tools/el_p5_cost_accounting.py <run dir> [<run dir> ...] > out.json"""
import json
import sys
from collections import defaultdict
from pathlib import Path

rep = {}
for d in map(Path, sys.argv[1:]):
    calls = [json.loads(l) for l in (d / "CALLS.jsonl").read_text("utf-8").splitlines()]
    res = [json.loads(l) for l in (d / "RESULTS.jsonl").read_text("utf-8").splitlines()]
    first = {}
    for c in calls:
        if not c["reused"]:
            first.setdefault((c["task"], c["purpose"], c["seed"], c["prompt_sha256"]), c["actual_seconds"])
    acc = defaultdict(lambda: {"calls": 0, "reused_calls": 0, "logical_gen_tokens": 0, "actual_gen_tokens": 0, "inference_s": 0.0, "logical_inference_s": 0.0})
    for c in calls:
        a = acc[(c["task"], c["arm"])]
        a["logical_inference_s"] = round(a["logical_inference_s"] + first.get((c["task"], c["purpose"], c["seed"], c["prompt_sha256"]), 0.0), 2)
        a["calls"] += 1; a["reused_calls"] += c["reused"]; a["logical_gen_tokens"] += c["tokens_generated"]
        if not c["reused"]:
            a["actual_gen_tokens"] += c["tokens_generated"]; a["inference_s"] = round(a["inference_s"] + c["actual_seconds"], 2)
    rows = []
    for r in res:
        if r["arm"] in ("_task",):
            continue
        a = acc[(r["task"], r["arm"])]; r1 = acc[(r["task"], "round1")]
        row = {"task": r["task"], "arm": r["arm"], "label": r["label"], "reported_gen_tokens": r.get("tokens_generated"),
               "arm_calls": a["calls"], "arm_reused_calls": a["reused_calls"], "arm_actual_gen_tokens": a["actual_gen_tokens"], "arm_inference_s": a["inference_s"],
               "arm_wall_s": r.get("seconds"), "arm_non_inference_s": round(r["seconds"] - a["inference_s"], 1) if r.get("seconds") is not None else None,
               "arm_logical_inference_s": a["logical_inference_s"],
               "wall_equivalent_s": round(a["logical_inference_s"] + r["seconds"] - a["inference_s"], 1) if r.get("seconds") is not None else None,
               "round1_gen_tokens": r1["actual_gen_tokens"], "round1_inference_s": r1["inference_s"]}
        rows.append(row)
    tot = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for k in ("reported_gen_tokens", "arm_actual_gen_tokens", "arm_inference_s", "arm_wall_s", "arm_non_inference_s", "arm_logical_inference_s", "wall_equivalent_s"):
            tot[row["arm"]][k] += row[k] or 0
    eq = {(row["task"], row["arm"]): row["wall_equivalent_s"] for row in rows}
    rel = sorted((eq[(t, "D")] - eq[(t, "C")]) / eq[(t, "C")] for (t, a) in eq if a == "C" and eq.get((t, "D")) and eq[(t, "C")])
    rep[str(d)] = {"rows": rows, "d_vs_c_wall_equivalent_overhead": {"per_task": [round(x, 3) for x in rel], "median": round(rel[len(rel) // 2], 3) if rel else None}, "totals_by_arm": {a: {k: round(v, 1) for k, v in t.items()} for a, t in tot.items()},
                   "round1_total": {"gen_tokens": sum(v["actual_gen_tokens"] for (t, a), v in acc.items() if a == "round1"),
                                    "inference_s": round(sum(v["inference_s"] for (t, a), v in acc.items() if a == "round1"), 1)}}
print(json.dumps(rep, indent=1))
