"""Kaggle CPU kernel: run the Evidence-Ladder pure-logic test suites (no models, no GPU, no internet, no secrets). Standing rule: computation runs on Kaggle.
The sandbox/execution tests are NOT run here (they need bubblewrap and run in WSL under the approved sandbox-test exception); this kernel makes NO security claim.
Input: a Kaggle dataset containing MANIFEST.json (bundle id evidence_ladder_tests_v1) plus the code tree. Output: /kaggle/working/TEST_RESULTS.json."""
import hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

OUT = Path("/kaggle/working")
t0 = time.time()
cands = sorted(Path("/kaggle/input").glob("**/MANIFEST.json"))
man_path = next((p for p in cands if json.loads(p.read_text(encoding="utf-8")).get("bundle") == "evidence_ladder_tests_v1"), None)
res = {"bundle_found": man_path is not None, "python": sys.version, "platform": platform.platform(), "suites": {}}
if man_path is None:
    (OUT / "TEST_RESULTS.json").write_text(json.dumps(res, indent=1)); sys.exit("bundle not found")
man = json.loads(man_path.read_text(encoding="utf-8"))
src = man_path.parent
bad = [f for f, h in man["files"].items() if not (src / f).is_file() or hashlib.sha256((src / f).read_bytes()).hexdigest() != h]
res["manifest_files"] = len(man["files"]); res["hash_mismatches"] = bad
if bad:
    (OUT / "TEST_RESULTS.json").write_text(json.dumps(res, indent=1)); sys.exit("bundle hash mismatch: %r" % bad[:3])
repo = OUT / "repo"
if repo.exists():
    shutil.rmtree(repo)
shutil.copytree(src, repo)
(repo / "MANIFEST.json").unlink()  # drop only the bundle's top-level manifest; nested MANIFEST.json files (e.g. the LAYA baseline manifest) must stay
SUITES = {
    "contract": [sys.executable, "specs/evidence_ladder_v1/contract/test_contracts_v1.py", str(OUT / "contract_test_results.json")],
    "controller": [sys.executable, "evidence_ladder/controller/tests/run_tests.py"],
    "orchestrator": [sys.executable, "evidence_ladder/orchestrator/tests/run_tests.py"],
    "models": [sys.executable, "evidence_ladder/models/tests/run_tests.py"],
    "replay": [sys.executable, "evidence_ladder/testing/replay_integration.py"],
}
env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
for name, cmd in SUITES.items():
    if not (repo / cmd[1]).is_file():
        res["suites"][name] = {"status": "MISSING"}; continue
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=600, env=env)
    last = None
    for line in reversed(p.stdout.strip().splitlines() or [""]):
        try:
            last = json.loads(line); break
        except ValueError:
            continue
    if last is None:  # multi-line pretty JSON: parse the whole stdout
        try: last = json.loads(p.stdout)
        except ValueError: last = None
    res["suites"][name] = {"exit_code": p.returncode, "summary": last, "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:]}
res["all_exit_zero"] = all(v.get("exit_code") == 0 for v in res["suites"].values())
res["seconds"] = round(time.time() - t0, 1)
(OUT / "TEST_RESULTS.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: (v.get("exit_code"), (v.get("summary") or {}).get("passed"), (v.get("summary") or {}).get("tests")) for k, v in res["suites"].items()}), "all_exit_zero:", res["all_exit_zero"])
