"""Fetch the FRESH A5-v3 pool: repository-disjoint from every existing split, split deterministically by repository hash, locked part closed and hashed.
Spec: configs/a5_v3_spec.toml ([amendment_2026_10_02_b] fresh_data_split). Provenance and hashing only: this script never scores or summarises anything about the
trajectories' contents beyond repository, instance and trajectory ids. Outputs are create-only.
    PYTHONPATH=. .venv/Scripts/python.exe tools/fetch_fresh_v3.py
"""
import hashlib, json, os, stat, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = "nebius/SWE-rebench-openhands-trajectories"
CHUNKS, PER, SHIFT, TARGET = 44, 25, 1013, 560           # staggered offsets so no old chunk is re-read; stop once TARGET fresh-repository rows are kept
old_repos, old_inst, old_traj = set(), set(), set()
for l in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l)
    old_repos.add(r["repo"]); old_inst.add(r["instance_id"]); old_traj.add(r["trajectory_id"])


def get(url, tries=6):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            err = e
            time.sleep(5 * (i + 1))
    raise err


total = get(f"https://datasets-server.huggingface.co/size?dataset={DS}")["size"]["dataset"]["num_rows"]
kept, seen, dropped = [], set(), {"old_repo": 0, "old_instance_or_trajectory": 0, "duplicate_in_fresh": 0}
for c in range(CHUNKS):
    if len(kept) >= TARGET:
        break
    off = min(c * (total - PER) // (CHUNKS - 1) + SHIFT, total - PER)
    d = get(f"https://datasets-server.huggingface.co/rows?dataset={DS}&config=default&split=train&offset={off}&length={PER}")
    for r in d["rows"]:
        row = dict(r["row"], _offset=r["row_idx"])
        if row["repo"] in old_repos:
            dropped["old_repo"] += 1; continue
        if row["instance_id"] in old_inst or row["trajectory_id"] in old_traj:
            dropped["old_instance_or_trajectory"] += 1; continue
        if row["trajectory_id"] in seen:
            dropped["duplicate_in_fresh"] += 1; continue
        seen.add(row["trajectory_id"]); kept.append(row)
    print(f"chunk {c}: offset {off}, kept {len(kept)}, dropped {sum(dropped.values())}", flush=True)
bucket = lambda repo: int(hashlib.sha256(repo.encode()).hexdigest()[:8], 16) % 5
fdev = [r for r in kept if bucket(r["repo"]) == 0]
locked = [r for r in kept if bucket(r["repo"]) != 0]
assert not ({r["repo"] for r in fdev} & {r["repo"] for r in locked}), "F-dev and locked must be repository-disjoint"
assert not ({r["repo"] for r in kept} & old_repos), "fresh pool must be repository-disjoint from the existing splits"
(ROOT / "data/locked_v3").mkdir(parents=True, exist_ok=True)
paths = {"fdev": ROOT / "data/raw/swerebench-fresh-fdev.jsonl", "locked": ROOT / "data/locked_v3/fresh-locked.jsonl"}
for name, rows in (("fdev", fdev), ("locked", locked)):
    with open(paths[name], "x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
lk_hash = hashlib.sha256(paths["locked"].read_bytes()).hexdigest()
os.chmod(paths["locked"], stat.S_IREAD)
prov = dict(dataset=DS, total_rows=total, chunks=CHUNKS, per_chunk=PER, offset_shift=SHIFT, target=TARGET, dropped=dropped,
            kept=len(kept), fdev_rows=len(fdev), locked_rows=len(locked), fdev_repos=len({r["repo"] for r in fdev}), locked_repos=len({r["repo"] for r in locked}),
            fdev_trajectory_ids_sha256=hashlib.sha256("\n".join(sorted(r["trajectory_id"] for r in fdev)).encode()).hexdigest(),
            locked_trajectory_ids_sha256=hashlib.sha256("\n".join(sorted(r["trajectory_id"] for r in locked)).encode()).hexdigest(),
            locked_file_sha256=lk_hash, split_rule="sha256(repo)[:8] as int mod 5 == 0 -> F-dev; else locked",
            existing_repos_excluded=len(old_repos), disjoint_from_existing_splits=True)
with open(ROOT / "results/raw/fresh-v3-provenance.json", "x", encoding="utf-8") as f:
    json.dump(prov, f, indent=1)
(ROOT / "results/raw/fresh-v3-locked.sha256").write_text(lk_hash + "  data/locked_v3/fresh-locked.jsonl\n")
print(json.dumps(prov, indent=1))
