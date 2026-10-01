"""Run the laya-typed BASELINE on the locked A5 benchmark, every axis. Run with laya-audit/.venv from the project root:
    laya-audit/.venv/Scripts/python.exe -m compiler.typed_baseline BENCHMARK.jsonl SAMPLE.jsonl NAME

The state is rebuilt by the project's own builder with a small budget (2,400 characters, tail-first, cut markers) so the checkpoint sees the issue and
the latest events inside its 640-token window instead of the head of a long state; any remaining explicit cut is counted. The same candidate texts
and NONE option as every other arm are passed. Output results/raw/a5-typed-baseline-NAME.json (create-only): aggregates in the same shape as the
other baselines plus per-decision predictions for paired comparison. It is a baseline: role BASELINE, never a target.
"""
import collections
import json
import sys
import time
from pathlib import Path

from compiler.baselines import AXES, evaluate
from compiler.state_builder import StateOverflow, build_state
from state.schema import Candidate
from teachers.baselines import LayaTypedBaseline

ROOT = Path(__file__).resolve().parents[1]
NONE_TEXT = "None of these options, or the evidence is insufficient"


def cands(texts):
    return [Candidate("NONE", t, action_type="none", safety_class="NONE", executable=False) if t == NONE_TEXT else Candidate(f"c{i}", t)
            for i, t in enumerate(texts)]


def main(bench_path, sample_path, name):
    bench = [json.loads(l) for l in open(bench_path, encoding="utf-8")]
    rows = {}
    for l in open(sample_path, encoding="utf-8"):
        r = json.loads(l)
        rows[r["trajectory_id"]] = r
    typed = LayaTypedBaseline()
    lookup, info, t0 = {}, collections.Counter(), time.time()
    for n, r in enumerate(bench):
        tid, t = r["decision_id"].rsplit(":", 1)
        evs = rows[tid]["trajectory"]
        for budget in (2400, 4000, 8000, 32000):   # smallest budget whose protected core fits; the wrapper cuts explicitly if still too long
            try:
                state = build_state(evs, int(t), budget)["text"]
                break
            except StateOverflow:
                continue
        info[f"state_budget_{budget}"] += 1
        for ax in AXES:
            x = {"texts": r["texts"], "label_index": r["label_index"]} if ax == "normal" else r["variants"][ax]
            if x is None:
                continue
            c = cands(x["texts"])
            out = typed.decide(r["decision_id"] + ":" + ax, "ACTION", state, c)
            assert not out.problems(), out.problems()
            lookup[(r["decision_id"], ax)] = out.candidate_ids.index(max(zip(out.candidate_ids, out.probabilities), key=lambda p: p[1])[0])
            info["state_cut_by_wrapper"] += bool(out.meta["state_cut"])
            info["predictions"] += 1
        if n % 100 == 0:
            print(f"{n}/{len(bench)} decisions, {time.time() - t0:.0f}s", flush=True)
    res, per = {}, {}
    for ax in AXES:
        res[ax], per[ax] = evaluate(bench, None, ax, lookup=lookup)
    out = {"benchmark": str(bench_path), "model_revision": typed.revision, "calibration": "shipped", "state_budget_chars": 2400, "info": dict(info),
           "decision_ids": [r["decision_id"] for r in bench], "per_decision": {"laya_typed_baseline": per}, "results": {"laya_typed_baseline": res}}
    with open(ROOT / "results" / "raw" / f"a5-typed-baseline-{name}.json", "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("axes:", {ax: res[ax]["all"] for ax in AXES})
    print("normal by key:", {k: v["accuracy"] for k, v in res["normal"].items()})
    print("info:", dict(info))


if __name__ == "__main__":
    main(*sys.argv[1:4])
