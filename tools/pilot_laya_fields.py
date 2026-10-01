"""Attach per-row Laya annotations to the frozen pilot manifest. ANNOTATIONS ONLY: they never change sampling, labels or the manifest.
Run with laya-audit/.venv from the project root:
    PYTHONPATH=. laya-audit/.venv/Scripts/python.exe tools/pilot_laya_fields.py
Output data/pilot/laya-fields-v1.jsonl (create-only) + results/raw/laya-fields-v1.sha256.
Per row: which Laya checkpoints are in-lane for the decision kind (from the lane registry), the laya-typed BASELINE distribution over the
candidates, its argmax/confidence/margin, whether its state was cut, and the pilot strata (tier, coverage, next_action_kind) copied from the
decision record. `typed_correct` is an analysis column against the trajectory label; laya-typed is never a training target for ACTION.
"""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
from compiler.state_builder import StateOverflow, build_state
from state.schema import BASELINES, LANES, Candidate
from teachers.baselines import LayaTypedBaseline

ROOT = Path(__file__).resolve().parents[1]
man = json.load(open(ROOT / "results/raw/pilot-manifest-v1.json", encoding="utf-8"))
want = set(man["decision_ids"])
recs = {}
for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["decision_id"] in want:
        recs[d["decision_id"]] = d
assert len(recs) == len(want) == man["n"]
raw = {}
for l in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l)
    raw[r["trajectory_id"]] = r
typed = LayaTypedBaseline()
out_path = ROOT / "data/pilot/laya-fields-v1.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with open(out_path, "x", encoding="utf-8") as f:
    for n, did in enumerate(man["decision_ids"]):
        d = recs[did]
        tid, t = did.rsplit(":", 1)
        for budget in (2400, 4000, 8000, 32000):  # same rule as the A5 typed baseline
            try:
                state = build_state(raw[tid]["trajectory"], int(t), budget)["text"]
                break
            except StateOverflow:
                continue
        cands = [Candidate(c["id"], c["text"], action_type=c["action_type"], tool=c["tool"], safety_class=c["safety_class"],
                           executable=c["id"] != "NONE") for c in d["candidates"]]
        o = typed.decide(did, "ACTION", state, cands)
        assert not o.problems(), o.problems()
        pr = sorted(zip(o.candidate_ids, o.probabilities), key=lambda p: -p[1])
        label_id = d["candidates"][d["label_index"]]["id"]
        row = {"decision_id": did, "split": d["split"], "tier": d["tier"], "candidate_coverage": d["candidate_coverage"],
               "next_action_kind": d["next_action_kind"],
               "in_lane_for_ACTION": {k: ("ACTION" in v) for k, v in LANES.items()},
               "laya_typed": {"role": "BASELINE", "distribution": dict(zip(o.candidate_ids, o.probabilities)), "argmax": pr[0][0],
                              "confidence": pr[0][1], "margin": pr[0][1] - pr[1][1], "state_cut": o.meta["state_cut"],
                              "state_budget_chars": budget, "n_options": o.meta["n_options"], "typed_correct": pr[0][0] == label_id}}
        f.write(json.dumps(row) + "\n")
        if n % 100 == 0:
            print(f"{n}/{man['n']} {time.time() - t0:.0f}s", flush=True)
body = out_path.read_bytes()
(ROOT / "results/raw/laya-fields-v1.sha256").write_text(hashlib.sha256(body).hexdigest() + "  data/pilot/laya-fields-v1.jsonl\n")
rows = [json.loads(l) for l in body.decode().splitlines()]
acc = sum(r["laya_typed"]["typed_correct"] for r in rows) / len(rows)
cov = [r for r in rows if r["candidate_coverage"]]
print("rows", len(rows), "typed acc all %.3f covered %.3f (n=%d)" % (acc, sum(r["laya_typed"]["typed_correct"] for r in cov) / len(cov), len(cov)),
      "cut", sum(r["laya_typed"]["state_cut"] for r in rows))
