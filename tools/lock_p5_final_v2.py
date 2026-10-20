"""[P5 v2] Final lock v2 (create-only): everything in configs/p5_final_lock_v1.json (task ids and order UNCHANGED) + the P5_PROTOCOL_v2 hash + the grading-set
hashes: PRIMARY = the blind private set (private/p5-eval-v3, content outside git and Kaggle), SECONDARY = the public set (results/raw/p5-eval-only-v1),
DISCLOSED non-blind validation set = private/p5-eval-v2. Protocol hashes are taken over the COMMITTED bytes (git show HEAD:<file>), so they do not depend on
line-ending conversion at checkout. Refuses unless the blind set's CHECK_v3.json says all_ok and every line of its SHA256SUMS_v3.txt matches.
Usage: python tools/lock_p5_final_v2.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "configs/p5_final_lock_v2.json"
if out.exists():
    sys.exit("refusing: %s exists (locked)" % out)
sha = lambda b: hashlib.sha256(b).hexdigest()
committed = lambda rel: sha(subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:" + rel], capture_output=True, check=True).stdout)


def sums(folder, name):
    rows = {}
    for ln in (folder / name).read_text("utf-8").splitlines():
        if ln.strip():
            h, f = ln.split(None, 1)
            f = f.strip().lstrip("*")
            if sha((folder / f).read_bytes()) != h:
                sys.exit("refusing: %s/%s differs from %s" % (folder, f, name))
            rows[f] = h
    return rows


v3, v2, pub = ROOT / "private/p5-eval-v3", ROOT / "private/p5-eval-v2", ROOT / "results/raw/p5-eval-only-v1"
chk = json.loads((v3 / "CHECK_v3.json").read_text("utf-8"))
if not chk.get("all_ok"):
    sys.exit("refusing: blind set CHECK_v3.json is not all_ok")
lock = json.loads((ROOT / "configs/p5_final_lock_v1.json").read_text("utf-8"))
lock.update({
    "lock_version": "p5_final_lock_v2",
    "supersedes": "configs/p5_final_lock_v1.json (kept; task ids and order unchanged)",
    "protocol": "P5_PROTOCOL_v2", "protocol_sha256": committed("P5_PROTOCOL_v2.md"), "protocol_v1_sha256": lock.pop("protocol_sha256"),
    "protocol_sha256_committed": committed("P5_PROTOCOL_v2.md"), "protocol_v1_sha256_committed": committed("P5_PROTOCOL_v1.md"),
    "lock_v1_sha256": sha((ROOT / "configs/p5_final_lock_v1.json").read_bytes()),
    "grading": {
        "primary": {"set": "blind private v3", "location": "private/p5-eval-v3 (outside git and Kaggle)", "cases_sha256": sha((v3 / "cases_v3.json").read_bytes()),
                    "check_sha256": sha((v3 / "CHECK_v3.json").read_bytes()), "sha256sums_file_sha256": sha((v3 / "SHA256SUMS_v3.txt").read_bytes()),
                    "files": sums(v3, "SHA256SUMS_v3.txt"), "total_cases": sum(len(v) for v in json.loads((v3 / "cases_v3.json").read_text("utf-8")).values())},
        "secondary": {"set": "public v1 (publicly accessible since commit f6ab9d29)", "location": "results/raw/p5-eval-only-v1", "files": sums(pub, "SHA256SUMS.txt")},
        "disclosed_non_blind_validation": {"set": "private v2 (author had prior exposure; see its PROVENANCE.md)", "location": "private/p5-eval-v2",
                                           "files": sums(v2, "SHA256SUMS_v2.txt")},
    },
})
out.write_text(json.dumps(lock, indent=1), encoding="utf-8", newline="\n")
print(json.dumps({k: lock[k] for k in ("lock_version", "protocol_sha256_committed", "N", "lcb_ids_sha256")} | {"primary": {k: v for k, v in lock["grading"]["primary"].items() if k != "files"}}, indent=1))
