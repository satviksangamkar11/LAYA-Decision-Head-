"""[P4] Express the P3 author specs (tools/p3_specs_v1.py) as task_contract_v1 documents for the 28 bank tasks.
Provenance is honest: source "author", written AFTER candidates existed (P3), so written_before_candidates = false -> these contracts are NOT authoritative;
they serve the P4 regression (the contract layer must reproduce P3's decisions) and as format examples. Usage: python tools/make_bank_contracts_v1.py"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import p3_specs_v1 as S      # noqa: E402

out = ROOT / "configs/contracts/bank_v1"
out.mkdir(parents=True, exist_ok=True)
sha = hashlib.sha256((ROOT / "tools/p3_specs_v1.py").read_bytes()).hexdigest()
for tid, sp in sorted(S.SPECS.items()):
    doc = {"contract_version": "task_contract_v1", "task_id": tid,
           "provenance": {"source": "author", "author": "coordinator (P3 specs)", "created_utc": "2026-10-20", "written_before_candidates": False,
                          "basis": "task text + public examples only; no grader case read"},
           "input": {"type": "any"}, "output": None, "edges": sp.edges,
           "properties_module": {"path": "tools/p3_specs_v1.py", "sha256": sha, "spec_key": tid}}
    p = out / ("%s.json" % tid)
    if p.exists():
        raise SystemExit("refusing to overwrite %s" % p)
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
print(len(S.SPECS), "contracts written to", out)
