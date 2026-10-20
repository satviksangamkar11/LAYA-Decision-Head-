"""[P4] Regression: the contract layer (evidence_ladder/contracts/contract_v1.py) must reproduce P3's frozen hard-failure decisions on the same candidates,
plus the pre-agreed P4 rule (a whole-run crash in a sequential runner belongs to the first input without a result). Run INSIDE WSL (reference engine).
Writes <out>/REGRESSION.json. Usage (inside WSL): python3 tools/el_p4_contract_regression.py <p3 dir> <new out dir>"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from evidence_ladder.contracts import contract_v1 as CV          # noqa: E402
from evidence_ladder.controller import l2                         # noqa: E402
from evidence_ladder.execution import config as K                 # noqa: E402
import el_p3_offline_checks as P3                                  # noqa: E402

p3, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    sys.exit("refusing: %s exists" % out)
out.mkdir(parents=True)
frozen = {(d["pool"], d["candidate"]): bool(d["hard_failures"]) for d in map(json.loads, (p3 / "DECISIONS.jsonl").read_text().splitlines())}
tasks, cfg = P3.tasks_by_id(), K.SandboxConfig()
contracts = {p.stem: CV.load(json.loads(p.read_text()), ROOT) for p in (ROOT / "configs/contracts/bank_v1").glob("*.json")}
self_check = {t: CV.self_check(c, tasks[t]["public_tests"]) for t, c in contracts.items()}
rows, same, changed = [], 0, []
for pl in P3.pools():
    pp = [c for c in pl["candidates"] if c["public_pass"]]
    if not pp:
        continue
    tid = pl["task"]; c = contracts[tid]; pub = tasks[tid]["public_tests"]
    inputs = list(c.edges) + l2.derive_inputs(pub, n=6)
    for cand in pp:
        res, items, term = P3.run_inputs(tasks[tid]["language"], cand["source"], inputs, cfg)
        results = [items[k] if k < len(items) else None for k in range(len(inputs))]
        ev = CV.evaluate(c, inputs, results, whole_run_crashed=(term == "crash"))
        now, before = bool(ev["hard"]), frozen[(pl["pool"], cand["id"])]
        same += now == before
        if now != before:
            changed.append({"pool": pl["pool"], "candidate": cand["id"][:12], "p3": before, "p4": now, "checks": sorted({h["check"] for h in ev["hard"]})})
        rows.append({"pool": pl["pool"], "candidate": cand["id"], "hard": ev["hard"], "mode": ev["mode"]})
doc = {"candidates": len(rows), "same_as_p3": same, "changed": changed, "contracts": len(contracts),
       "self_check_problems": {t: v for t, v in self_check.items() if v}, "modes": sorted({r["mode"] for r in rows})}
(out / "REGRESSION.json").write_text(json.dumps(doc, indent=1))
(out / "ROWS.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(json.dumps(doc, indent=1))
