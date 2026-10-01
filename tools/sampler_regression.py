"""Step-1 regression: the audit script's candidate sets must equal the canonical compiler/a5v2.py sets, decision by decision.
Runs (a) the sampler section of the OLD tools/a5_v2_sampler.py exactly as committed in git HEAD, (b) the sampler section of the current working copy,
(c) compiler.a5v2.oof_sets on the same TRAIN covered decisions, and compares features, k, pool hash, seed, order and true position. Nothing is written except the
create-only result. TRAIN rows only (load_dev).
    PYTHONPATH=. .venv/Scripts/python.exe tools/sampler_regression.py [GIT_REV]"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from compiler import a5v2
from compiler.firewall import load_dev

ROOT = Path(__file__).resolve().parents[1]
REV = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
CUT = "# ---------- audit ----------"


def run_sampler_section(src):
    head = src.split(CUT)[0]
    ns = {"__name__": "sampler_section", "__file__": str(ROOT / "tools/a5_v2_sampler.py")}
    exec(compile(head, "a5_v2_sampler(section)", "exec"), ns)
    return ns["decs"], ns["v2"]


old_src = subprocess.run(["git", "show", f"{REV}:tools/a5_v2_sampler.py"], capture_output=True, text=True, cwd=ROOT, check=True).stdout
new_src = (ROOT / "tools/a5_v2_sampler.py").read_text(encoding="utf-8")
old_decs, old_v2 = run_sampler_section(old_src)
new_decs, new_v2 = run_sampler_section(new_src)

raw = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
decs = []
for r in load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"):
    if not r["candidate_coverage"]:
        continue
    d, _ = a5v2.build_decision(r, raw[r["decision_id"].rsplit(":", 1)[0]]["trajectory"])
    if d is not None:
        decs.append(d)
can_v2 = a5v2.oof_sets(decs)


def compare(name, A_decs, A_v2, B_decs, B_v2):
    a, b = {d["did"]: d for d in A_decs}, {d["did"]: d for d in B_decs}
    bad = []
    if list(a) != list(b):
        bad.append("decision id lists differ")
    for k in a:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if not (torch.equal(x["X"], y["X"]) and x["ti"] == y["ti"] and x["k"] == y["k"] and x["pool_hash"] == y["pool_hash"] and x["stratum"] == y["stratum"] and x["sh"] == y["sh"]):
            bad.append(f"{k}: decision features differ")
        sa, sb = A_v2.get(k), B_v2.get(k)
        if sa != sb:
            bad.append(f"{k}: set differs")
    if set(A_v2) != set(B_v2):
        bad.append("set key lists differ")
    return dict(compared=name, decisions=len(a), mismatches=len(bad), first_mismatches=bad[:5], identical=not bad)


rows = [compare(f"old({REV}) vs canonical", old_decs, old_v2, decs, can_v2),
        compare("working copy vs canonical", new_decs, new_v2, decs, can_v2),
        compare(f"old({REV}) vs working copy", old_decs, old_v2, new_decs, new_v2)]
frozen_hash = __import__("hashlib").sha256((ROOT / "data/a5v2/odds-a5v2-odds-1.pt").read_bytes()).hexdigest()
out = dict(rev=REV, rows=rows, frozen_artifact_sha256=frozen_hash, passed=all(r["identical"] for r in rows) and len(decs) > 0,
           fields_compared="features X, true index, k, pool hash, stratum, state hash, ordered set, true position, seed")
p = ROOT / ("results/raw/sampler-regression-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)); print("->", p)
