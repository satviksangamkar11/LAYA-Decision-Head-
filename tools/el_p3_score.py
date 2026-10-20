"""[P3] SCORING step, after the decisions were frozen and committed: join DECISIONS/POOLS with the existing offline development grades
(P1: REFERENCE.jsonl; P2 local and Kaggle: GRADED.jsonl). Usage: python tools/el_p3_score.py <p3 dir>"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = Path(sys.argv[1])
G = {}
for f in [ROOT / "results/raw/p1-offline-grading-20261019T192425Z/REFERENCE.jsonl"]:
    for r in map(json.loads, f.read_text("utf-8").splitlines()):
        G[r["candidate"]] = r["ref"]
for f in [ROOT / "results/raw/p2-generation-compare-v2-20261019T215005Z/GRADED.jsonl"] + [Path(x) for x in glob.glob(str(ROOT / "results/raw/p2-kaggle-out-*/p2_shard*/GRADED.jsonl"))]:
    for r in map(json.loads, f.read_text("utf-8").splitlines()):
        G[r["id"]] = r["dev"]
D = [json.loads(l) for l in (d / "DECISIONS.jsonl").read_text("utf-8").splitlines()]
P = [json.loads(l) for l in (d / "POOLS.jsonl").read_text("utf-8").splitlines()]
missing = [x["candidate"] for x in D if x["candidate"] not in G]
assert not missing, "ungraded candidates: %s" % missing[:3]


def group(pool):
    return pool.split(":")[0]


cand = {}
for x in D:
    g, hf = G[x["candidate"]], bool(x["hard_failures"])
    s = cand.setdefault(group(x["pool"]), dict(public_passing=0, grader_correct=0, grader_wrong=0, true_catch=0, incorrect_downgrade=0, missed_wrong=0))
    s["public_passing"] += 1; s["grader_correct" if g == "PASS" else "grader_wrong"] += 1
    if hf and g != "PASS": s["true_catch"] += 1
    if hf and g == "PASS": s["incorrect_downgrade"] += 1
    if not hf and g != "PASS": s["missed_wrong"] += 1
ans, rows = {}, []
for p in P:
    a = ans.setdefault(group(p["pool"]), dict(pools_answered=0, before_correct=0, before_false_supported=0, after_correct=0, after_false_supported=0, after_blocked=0, answer_changed=0,
                                             shadow_flag=0, shadow_flag_on_wrong_answer=0, shadow_flag_on_correct_answer=0))
    o, n = p["original_answer"], p["p3_answer"]
    if o is None:
        continue
    a["pools_answered"] += 1
    a["before_correct" if G[o] == "PASS" else "before_false_supported"] += 1
    if n is None: a["after_blocked"] += 1
    else: a["after_correct" if G[n] == "PASS" else "after_false_supported"] += 1
    a["answer_changed"] += o != n
    if p["shadow_disagreement"]:
        a["shadow_flag"] += 1; a["shadow_flag_on_wrong_answer" if G[o] != "PASS" else "shadow_flag_on_correct_answer"] += 1
    if o != n or p["shadow_disagreement"] or G[o] != "PASS":
        rows.append({"pool": p["pool"], "before": G[o], "after": (G[n] if n else "BLOCKED"), "changed": o != n, "shadow": p["shadow_disagreement"],
                     "hard_checks": sorted({h["check"] for x in D if x["pool"] == p["pool"] and x["candidate"] in p["hard_failed"] for h in x["hard_failures"]})})
doc = {"label": "development evidence only; decisions frozen before scoring (commit e6ff22a0)", "per_candidate": cand, "per_answer": ans, "rows": rows}
(d / "SCORE.json").write_text(json.dumps(doc, indent=1))
print(json.dumps(doc, indent=1))
