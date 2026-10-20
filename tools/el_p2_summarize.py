"""[P2] Per-task / per-arm summary of one or more graded P2 directories (RESULTS.jsonl + CANDIDATES.jsonl + GRADED.jsonl + CALLS.jsonl).
Usage: python tools/el_p2_summarize.py <out json> <p2 dir> [<p2 dir> ...] [--only-tasks a,b] [--skip-tasks a,b]"""
import json
import sys
from pathlib import Path

ARMS = ("fresh", "failure", "counterexample_guided")


def load(d, only=None, skip=()):
    d = Path(d)
    rd = lambda n: [json.loads(l) for l in (d / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    R, C, K = rd("RESULTS.jsonl"), rd("CANDIDATES.jsonl"), rd("CALLS.jsonl")
    G = {(g["task"], g["id"]): g for g in rd("GRADED.jsonl")}
    keep = lambda t: (only is None or t in only) and t not in skip
    return [r for r in R if keep(r["task"])], [c for c in C if keep(c["task"])], [k for k in K if keep(k["task"])], G


def main():
    args = sys.argv[1:]
    out, dirs, only, skip = Path(args[0]), [], None, set()
    i = 1
    while i < len(args):
        if args[i] == "--only-tasks": only = set(args[i + 1].split(",")); i += 2
        elif args[i] == "--skip-tasks": skip = set(args[i + 1].split(",")); i += 2
        else: dirs.append(args[i]); i += 1
    R, C, K, G = [], [], [], {}
    for d in dirs:
        r, c, k, g = load(d, only, skip)
        R += r; C += c; K += k; G.update(g)
    tasks = list(dict.fromkeys(r["task"] for r in R))
    per_task, per_arm = [], {a: dict(tasks=0, available=0, public_solved=0, selected_correct=0, false_supported=0, tokens_generated=0, tokens_prompt=0,
                                      model_seconds=0.0, sandbox_seconds=0.0, calls=0, candidates=0, correct_candidates=0) for a in ARMS}
    for t in tasks:
        r1 = [c for c in C if c["task"] == t and c["arm"] == "fresh" and c["round"] == 1]
        row = {"task": t, "round1_candidates": len(r1), "round1_correct": sum(G[(t, c["id"])]["dev"] == "PASS" for c in r1)}
        for a in ARMS:
            r = next(x for x in R if x["task"] == t and x["arm"] == a)
            cs = {c["id"]: c for c in C if c["task"] == t and c["arm"] == a}
            ok = sum(G[(t, i)]["dev"] == "PASS" for i in cs)
            sel = G[(t, r["selected"])]["dev"] if r["selected"] else None
            s = per_arm[a]
            s["tasks"] += 1; s["available"] += ok > 0; s["public_solved"] += r["label"] == "SUPPORTED"; s["selected_correct"] += sel == "PASS"; s["false_supported"] += sel == "FAIL"
            s["tokens_generated"] += r["arm_tokens_generated"]; s["tokens_prompt"] += r["arm_tokens_prompt"]; s["calls"] += r["calls"]; s["candidates"] += len(cs); s["correct_candidates"] += ok
            s["model_seconds"] += r["model_seconds"]; s["sandbox_seconds"] += r["sandbox_seconds"]
            row[a] = {"label": r["label"], "selected_dev": sel, "candidates": len(cs), "correct_candidates": ok, "tokens_generated": r["arm_tokens_generated"], "calls": r["calls"]}
        per_task.append(row)
    for s in per_arm.values():
        s["model_seconds"], s["sandbox_seconds"] = round(s["model_seconds"], 1), round(s["sandbox_seconds"], 1)
    doc = {"dirs": dirs, "tasks": tasks, "per_arm": per_arm, "per_task": per_task, "label": "development evidence; grader cases public in Git; 13 tasks: descriptive, no significance claim"}
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for a, s in per_arm.items():
        print(a, json.dumps(s))
    for row in per_task:
        print(row["task"], "r1 ok %d/%d" % (row["round1_correct"], row["round1_candidates"]), " | ".join("%s %s ok%d/%d sel=%s" % (a[:5], row[a]["label"][:4], row[a]["correct_candidates"], row[a]["candidates"], row[a]["selected_dev"]) for a in ARMS))


if __name__ == "__main__":
    main()
