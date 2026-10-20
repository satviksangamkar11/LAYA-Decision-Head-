"""[P5 preflight] Apply the frozen N rule (P5_PROTOCOL_v1, cohort 1) and LOCK the final task set before any final candidate exists.
Rule: the largest N <= 91 whose estimated GPU time is <= 10 h at the smoke-measured rate, minimum 40; tasks = the first N of final_lcb_order (seed 20261020).
Rates: mean per-task wall seconds of each smoke run (all four arms, RESULTS.jsonl `_task` rows); the SLOWER measured rate is used. Writes configs/p5_final_lock_v1.json
(create-only). Usage: python tools/lock_p5_final_v1.py"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "configs/p5_final_lock_v1.json"
if out.exists():
    sys.exit("refusing: %s exists (locked)" % out)
sha = lambda b: hashlib.sha256(b).hexdigest()
SMOKES = {"local_rtx3050_mxc": "results/raw/p5-smoke-lcb-20261020T112829Z", "kaggle_t4_jail_parallel_shards": "results/raw/p5-kaggle-smoke-v2/p5_smoke_lcb"}
CSMOKES = {"local_rtx3050_mxc": "results/raw/p5-smoke-bank-20261020T112829Z", "kaggle_t4_jail_parallel_shards": "results/raw/p5-kaggle-smoke-v2/p5_smoke_bank"}


def rate(p):
    w = [r["wall_seconds"] for r in map(json.loads, (ROOT / p / "RESULTS.jsonl").read_text("utf-8").splitlines()) if r["arm"] == "_task"]
    return sum(w) / len(w), len(w)


rates = {k: rate(p) for k, p in SMOKES.items()}
crates = {k: rate(p) for k, p in CSMOKES.items()}
lcb_rate = max(r for r, _ in rates.values())
con_rate = max(r for r, _ in crates.values())
sel = json.loads((ROOT / "configs/p5_selection_v1.json").read_text("utf-8"))
order = sel["final_lcb_order"]
cap_s = 10 * 3600
n = max([k for k in range(40, min(91, len(order)) + 1) if k * lcb_rate <= cap_s] or [0])
if n < 40:
    sys.exit("N rule gives < 40: stop and report")
ids = order[:n]
contract_ids = [json.loads(l)["task_id"] for l in (ROOT / "configs/p5_contract_tasks_v1/tasks.jsonl").read_text("utf-8").splitlines()]
smoke = set(sel["smoke_lcb"]) | {"abc388_a", "abc389_a", "3747", "3753"}
assert not smoke & set(ids), "final set overlaps smoke / adapter-validation problems"
lock = {"protocol": "P5_PROTOCOL_v1", "protocol_sha256": sha((ROOT / "P5_PROTOCOL_v1.md").read_bytes()),
        "rule": "largest N <= 91 with N x rate <= 10 h at the smoke-measured rate (slower of the two measured smoke runs), minimum 40; first N of final_lcb_order",
        "smoke_rates_s_per_lcb_task": {k: {"mean_s": round(r, 1), "tasks": m} for k, (r, m) in rates.items()},
        "smoke_rates_s_per_contract_style_task": {k: {"mean_s": round(r, 1), "tasks": m} for k, (r, m) in crates.items()},
        "rate_used_s": round(lcb_rate, 1), "available": len(order), "N": n, "lcb_ids": ids, "lcb_ids_sha256": sha(json.dumps(ids).encode()),
        "lcb_file_sha256": sha((ROOT / "data/livecodebench/v6/test6.jsonl").read_bytes()),
        "contract_ids": contract_ids, "contract_frozen": (ROOT / "configs/p5_contract_tasks_v1/FROZEN_SHA256.txt").read_text("utf-8").splitlines(),
        "selection_sha256": sha((ROOT / "configs/p5_selection_v1.json").read_bytes()),
        "estimate_unmeasured": {"lcb_hours": round(n * lcb_rate / 3600, 2), "contract_hours": round(len(contract_ids) * con_rate / 3600, 2),
                                "total_hours": round((n * lcb_rate + len(contract_ids) * con_rate) / 3600, 2),
                                "note": "linear extrapolation of smoke means; contract rate from the 3 bank smoke tasks (not the contract cohort itself)"},
        "model": {"repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a"}, "G": 6000, "arms": ["A", "B", "C", "D"]}
out.write_text(json.dumps(lock, indent=1), encoding="utf-8")
print(json.dumps({k: lock[k] for k in ("smoke_rates_s_per_lcb_task", "rate_used_s", "available", "N", "lcb_ids_sha256", "estimate_unmeasured")}, indent=1))
