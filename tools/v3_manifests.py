"""Build the seeded, capped, covered-only decision manifests for the A5-v3 capture. Rules: configs/a5_v3_spec.toml [v3_manifests] and [amendment_2026_10_02_c].
TRAIN comes through the firewall (train rows only), CAL through the logged calibration-only loader, LOCKED from the compiled locked decision records (the access is logged by
the compile step and again here). No accuracy or model output is computed. Output create-only: data/v3/manifest-<role>.json and results/raw/v3-manifests.json."""
import collections, hashlib, json, tomllib
from datetime import datetime, timezone
from pathlib import Path
from compiler import a5v2
from compiler.firewall import load_cal_calibration_only, load_dev, log_locked_access

ROOT = Path(__file__).resolve().parents[1]
M = tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))["v3_manifests"]
raw_old = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
log_locked_access("v3_manifests: read the locked fresh trajectories to select covered decisions by id order; no outcomes")
raw_locked = a5v2.load_raw(ROOT / "data/locked_v3/fresh-locked.jsonl")
key = lambda did: hashlib.sha256(f"{M['seed']}|{did}".encode()).hexdigest()


def select(rows, raw, cap, target=None):
    rows = [r for r in rows if r["candidate_coverage"]]
    rows.sort(key=lambda r: key(r["decision_id"]))
    per, out, strata, skipped = collections.Counter(), [], collections.Counter(), collections.Counter()
    for r in rows:
        tid = r["decision_id"].rsplit(":", 1)[0]
        if per[tid] >= cap:
            continue
        d, why = a5v2.build_decision(r, raw[tid]["trajectory"])
        if d is None:
            skipped[why] += 1
            continue
        per[tid] += 1
        out.append(r["decision_id"])
        strata[d["stratum"]] += 1
        if target and len(out) >= target:
            break
    return out, dict(strata), dict(skipped), per


res, now = {}, datetime.now(timezone.utc).isoformat()
(ROOT / "data/v3").mkdir(parents=True, exist_ok=True)
jobs = {
    "train": (list(load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl")), raw_old, M["caps"]["train"], M["targets"]["train"]),
    "cal": (list(load_cal_calibration_only(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl")), raw_old, M["caps"]["cal"], M["targets"]["cal"]),
    "locked": ([json.loads(l) for l in open(ROOT / "data/decisions/locked-v3.jsonl", encoding="utf-8")], raw_locked, M["caps"]["locked"], None),
}
for role, (rows, raw, cap, target) in jobs.items():
    ids, strata, skipped, per = select(rows, raw, cap, target)
    dist = collections.Counter(per.values())
    body = json.dumps(dict(role=role, seed=M["seed"], cap=cap, target=target, n=len(ids), trajectories=len(per), strata=strata, skipped=skipped,
                           decisions_per_trajectory_distribution=dict(sorted(dist.items())), decision_ids=ids), indent=1)
    p = ROOT / f"data/v3/manifest-{role}.json"
    with open(p, "x", encoding="utf-8") as f:
        f.write(body)
    res[role] = dict(n=len(ids), trajectories=len(per), strata=strata, skipped=skipped, per_trajectory=dict(sorted(dist.items())),
                     clusters_with_5=sum(1 for v in per.values() if v >= 5), sha256=hashlib.sha256(body.encode()).hexdigest())
    print(role, {k: v for k, v in res[role].items() if k != "sha256"}, flush=True)
with open(ROOT / "results/raw/v3-manifests.json", "x", encoding="utf-8") as f:
    json.dump(dict(created=now, spec="configs/a5_v3_spec.toml [v3_manifests]", roles=res), f, indent=1)
