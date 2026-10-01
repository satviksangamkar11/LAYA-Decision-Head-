"""Build the pilot capture manifest (configs/pilot_manifest_v1.toml). Create-only. Hashes the result."""
import collections, hashlib, json, random, sys, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg = tomllib.loads((ROOT / "configs" / "pilot_manifest_v1.toml").read_text(encoding="utf-8"))["pilot"]
rows = [json.loads(l) for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8")]
bench = {json.loads(l)["decision_id"] for l in open(ROOT / "data/benchmark/a5-v1.jsonl", encoding="utf-8")}
pool = [d for d in rows if d["split"] in ("train", "cal")]
assert not any(d["split"] in cfg["excluded"] for d in pool) and not any(d["decision_id"] in bench for d in pool), "locked data leaked into pool"
tr_t = {d["provenance"]["trajectory_id"] for d in pool if d["split"] == "train"}
assert not tr_t & {d["provenance"]["trajectory_id"] for d in pool if d["split"] == "cal"}
rng = random.Random(cfg["seed"])
order = list(pool)
rng.shuffle(order)
chosen, per_traj, per_repo, quota = [], collections.Counter(), collections.Counter(), dict(cfg["target"])
taken = set()
shortages = []


def can(d):
    sp, tj, rp = d["split"], d["provenance"]["trajectory_id"], d["provenance"]["repo"]
    return (d["decision_id"] not in taken and quota[sp] > 0 and per_traj[tj] < cfg["max_per_trajectory"][sp] and per_repo[rp] < cfg["max_per_repo"])


def take(d):
    chosen.append(d); taken.add(d["decision_id"]); quota[d["split"]] -= 1
    per_traj[d["provenance"]["trajectory_id"]] += 1; per_repo[d["provenance"]["repo"]] += 1


def fill(name, pred, n):
    got = sum(1 for d in chosen if pred(d))
    for d in order:
        if got >= n: break
        if pred(d) and can(d):
            take(d); got += 1
    if got < n: shortages.append(dict(requirement=name, wanted=n, got=got))


fill("tier NEGATIVE", lambda d: d["tier"] == "NEGATIVE", cfg["floor_tier_NEGATIVE"])
kinds = collections.Counter(d["next_action_kind"] for d in pool)
for k, _ in sorted(kinds.items(), key=lambda kv: kv[1]):
    fill(f"kind {k}", lambda d, k=k: d["next_action_kind"] == k, cfg["floor_per_next_action_kind"])
fill("tier STRONG", lambda d: d["tier"] == "STRONG", cfg["floor_tier_STRONG"])
fill("covered", lambda d: d["candidate_coverage"], cfg["floor_covered"])
for d in order:                                   # remainder: random over the pool = proportional to pool composition
    if can(d): take(d)
for sp, left in quota.items():
    if left: shortages.append(dict(requirement=f"split {sp} target", wanted=cfg["target"][sp], got=cfg["target"][sp] - left))


def cnt(f, sel): return dict(collections.Counter(f(d) for d in sel))


man = dict(config=cfg, config_sha256=hashlib.sha256((ROOT / "configs/pilot_manifest_v1.toml").read_bytes()).hexdigest(),
           n=len(chosen), by_split=cnt(lambda d: d["split"], chosen), by_tier=cnt(lambda d: d["tier"], chosen),
           by_coverage=cnt(lambda d: d["candidate_coverage"], chosen), by_kind=cnt(lambda d: d["next_action_kind"], chosen),
           tier_x_split=cnt(lambda d: (d["split"], d["tier"]), chosen) and {f"{k[0]}/{k[1]}": v for k, v in cnt(lambda d: (d["split"], d["tier"]), chosen).items()},
           repos=len(per_repo), trajectories=len(per_traj), max_per_traj=max(per_traj.values()), max_per_repo=max(per_repo.values()),
           shortages=shortages, cal_is_pilot_only=True, decision_ids=sorted(d["decision_id"] for d in chosen))
body = json.dumps(man, indent=1, sort_keys=True)
out = ROOT / "results/raw/pilot-manifest-v1.json"
with open(out, "x", encoding="utf-8") as f: f.write(body)
with open(ROOT / "results/raw/pilot-manifest-v1.sha256", "x") as f: f.write(hashlib.sha256(body.encode()).hexdigest() + "  pilot-manifest-v1.json\n")
print(json.dumps({k: v for k, v in man.items() if k != "decision_ids"}, indent=1))
