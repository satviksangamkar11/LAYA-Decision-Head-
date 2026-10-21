"""[P6 v1.1] Mechanical proof that the logging-only amendment changes no decision. CPU only; scripted model; real MXC sandbox; no grading data.
For each scenario the FROZEN v1 runner (tools/el_p6_run.py at commit 027c91ea) and the amended runner run on the same scripted outputs:
  1 decisions unchanged: CALLS (requests, seeds, max tokens, reuse, tokens), RESULTS, ANSWERS and IDENTITY are identical (timing fields excluded)
  2 CANDIDATES.jsonl holds exactly one row per NEW generation (= every non-reused call), with the same request identity and the extracted code
  3 early stop is visible: when B accepts the seed-1 sample, seeds 2-4 are in CANDIDATES but in no arm's evaluated list (not_evaluated)
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/test_p6_logging_v11.py"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FROZEN = "027c91ea"
TMP = Path(tempfile.mkdtemp(prefix="p6v11-")).resolve()
REF = ROOT / "tools" / "_p6_run_v1_frozen_tmp.py"
fence = lambda code: "<think>\n\n</think>\n\n```python\n%s```\n" % code
CF_GOOD = "t = int(input())\nfor _ in range(t):\n    l, r = map(int, input().split())\n    print(1 if l == r == 1 else r - l)\n"
CF_BAD = "t = int(input())\nfor _ in range(t):\n    input()\n    print(0)\n"
D13_HARD = ("def solve(data):\n    t = {(): [], (3, 1, 3, 2, 1): [3, 1, 2], (7, 7, 7): [7], (-1, 0, -1): [-1, 0]}\n    return t.get(tuple(data['nums']), [])\n")
D13_GOOD = "def solve(data):\n    out = []\n    for v in data['nums']:\n        if v not in out:\n            out.append(v)\n    return out\n"
SCEN = {"cf_later_round": ("cf", "2063/A", [fence(CF_BAD)] * 6 + [fence(CF_GOOD)]),
        "cf_seed1_accepted": ("cf", "2063/A", [fence(CF_GOOD)] * 4),
        "cf_seed1_no_code": ("cf", "2063/A", ["no code here"] + [fence(CF_GOOD)] * 3),
        "contract_gate": ("contract", "D13", [fence(D13_HARD)] + [fence(D13_GOOD)] * 3)}
TIMING = {"actual_seconds", "seconds", "wall_seconds"}


def run(script, name, cohort, task, texts):
    sp = TMP / (name + ".json"); sp.write_text(json.dumps(texts))
    out = TMP / name
    p = subprocess.run([PY, "-B", str(script), str(out), "--cohort", cohort, "--tasks", task, "--exec", "mxc", "--script", str(sp)], cwd=str(ROOT),
                       env={**os.environ, "PYTHONPATH": str(ROOT), "ATEN_CPU_CAPABILITY": "avx2"}, capture_output=True, text=True, timeout=1200)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-1500:])
    return lambda f: [json.loads(l) for l in (out / f).read_text("utf-8").splitlines() if l.strip()] if (out / f).exists() else []


strip = lambda rows: [{k: v for k, v in r.items() if k not in TIMING} for r in rows]
REF.write_bytes(subprocess.run(["git", "show", "%s:tools/el_p6_run.py" % FROZEN], cwd=str(ROOT), capture_output=True, check=True).stdout)
checks = {}
try:
    for name, (cohort, task, texts) in SCEN.items():
        old, new = run(REF, name + "_v1", cohort, task, texts), run(ROOT / "tools/el_p6_run.py", name + "_v11", cohort, task, texts)
        checks[name + ":1_decisions_unchanged"] = all(strip(old(f)) == strip(new(f)) for f in ("CALLS.jsonl", "RESULTS.jsonl", "ANSWERS.jsonl", "IDENTITY.jsonl"))
        fresh = [(c["seed"], c["prompt_sha256"], c["max_new_tokens"], c["purpose"]) for c in new("CALLS.jsonl") if not c["reused"]]
        cand = new("CANDIDATES.jsonl")
        checks[name + ":2_one_row_per_generation"] = (sorted(fresh) == sorted((c["seed"], c["prompt_sha256"], c["max_new_tokens"], c["purpose"]) for c in cand)
                                                      and len(cand) == len(fresh) and all((c["code"] is None) == (c["extract_status"] != "ok") for c in cand))
        if name == "cf_seed1_accepted":
            use = new("CANDIDATE_USE.jsonl")[0]
            evaluated = {1} | {x["seed"] for arm in ("B", "V2") for x in use[arm]}
            checks[name + ":3_unevaluated_logged"] = (evaluated == {1} and {c["seed"] for c in cand} == {1, 2, 3, 4} and use["generated_seeds"] == [1, 2, 3, 4])
finally:
    REF.unlink(missing_ok=True)
print(json.dumps({"frozen_runner_commit": FROZEN, "checks": checks, "all_ok": all(checks.values())}, indent=1))
sys.exit(0 if checks and all(checks.values()) else 1)
