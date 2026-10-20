"""Verify every pinned sha256 of the frozen contract / evaluation material against the bytes under <root> (default: this checkout):
configs/p5_contract_tasks_v1/FROZEN_SHA256.txt (root-relative paths), results/raw/p5-eval-only-v1/SHA256SUMS.txt (paths relative to its folder),
both P5 contract registries, and the properties_module pin of every registered P5 contract and every bank_v1 contract.
Usage: python tools/verify_p5_pins.py [root]. Exit 0 only if every pin matches."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
pins = []
for f, rel_to in (("configs/p5_contract_tasks_v1/FROZEN_SHA256.txt", ""), ("results/raw/p5-eval-only-v1/SHA256SUMS.txt", "results/raw/p5-eval-only-v1/")):
    for ln in (root / f).read_text("utf-8").splitlines():
        if ln.strip():
            h, p = ln.split(None, 1)
            pins.append((f, rel_to + p.strip().lstrip("*"), h))
docs = []
for reg in ("configs/p5_contract_tasks_v1/REGISTRY_v1.json", "configs/p5_qual_tasks_v1/REGISTRY_qual_v1.json"):
    for tid, e in json.loads((root / reg).read_text("utf-8"))["entries"].items():
        pins.append((reg, e["path"], e["sha256"]))
        docs.append(e["path"])
docs += sorted(p.relative_to(root).as_posix() for p in (root / "configs/contracts/bank_v1").glob("*.json"))
for d in docs:
    pm = json.loads((root / d).read_text("utf-8")).get("properties_module")
    if pm:
        pins.append((d, pm["path"], pm["sha256"]))
bad = [(src, p) for src, p, h in pins if not (root / p).exists() or hashlib.sha256((root / p).read_bytes()).hexdigest() != h]
print(json.dumps({"root": str(root), "pins": len(pins), "files": len({p for _, p, _ in pins}), "mismatch": bad}, indent=1))
sys.exit(1 if bad else 0)
