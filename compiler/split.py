"""Repository-disjoint splits (RB3). The split depends on the repository string only, never on the trajectory, label or tier.

OOD: whole owner families (the part before '/'), about 10%. Of the remaining repositories: train about 70%, calibration 15%, test 15%.
Calibration data is never used for gradient updates; the test split is never used for selection.
"""
import hashlib
import json

VERSION = "split-v1"


def _h(s):
    return int(hashlib.sha256(f"{VERSION}:{s}".encode()).hexdigest()[:8], 16) % 100


def split_of(repo):
    if _h(repo.split("/")[0]) < 10:
        return "ood"
    r = _h(repo)
    return "train" if r < 70 else "cal" if r < 85 else "test"


def manifest(rows):
    """repo -> split for every repository in rows, plus a hash, and the lineage checks that must hold."""
    m = {r["repo"]: split_of(r["repo"]) for r in rows}
    inst, traj = {}, {}
    for r in rows:
        s = m[r["repo"]]
        for key, d in ((r["instance_id"], inst), (r["trajectory_id"], traj)):
            if d.setdefault(key, s) != s:
                raise ValueError(f"{key} appears in two splits")
    owners = {}
    for repo, s in m.items():
        if s == "ood":
            continue
    ood_owners = {repo.split("/")[0] for repo, s in m.items() if s == "ood"}
    if any(repo.split("/")[0] in ood_owners and s != "ood" for repo, s in m.items()):
        raise ValueError("an OOD owner family also appears in another split")
    body = json.dumps(dict(sorted(m.items())), sort_keys=True)
    return {"version": VERSION, "repos": dict(sorted(m.items())), "sha256": hashlib.sha256(body.encode()).hexdigest()}
