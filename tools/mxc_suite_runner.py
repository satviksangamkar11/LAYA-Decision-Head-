"""[MXC replacement] Run the CPU regression suites (and optionally the 75 fault-injection variants) against ONE sandbox backend and record the results create-only.

    PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe tools/mxc_suite_runner.py results/raw/mxc-baseline-wsl-<UTC> --backend wsl [--variants]

--backend only sets LAYA_SANDBOX_BACKEND for the child processes (see evidence_ladder/controller/sandbox_provider.py); the suites themselves are unchanged. No GPU suites are run.
Output dir: SUITES.json (per suite: exit code, seconds, parsed {"tests","passed","failed","skipped"} when printed, stdout/stderr tails), VARIANTS_SUMMARY.json when --variants."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SUITES = [
    "evidence_ladder/controller/tests/phaseA_tests.py", "evidence_ladder/controller/tests/phaseB_tests.py", "evidence_ladder/controller/tests/phaseD1_tests.py",
    "evidence_ladder/controller/tests/phaseF1_tests.py", "evidence_ladder/controller/tests/phaseF1b_tests.py", "evidence_ladder/controller/tests/phaseF2_tests.py",
    "evidence_ladder/controller/tests/run_tests.py", "evidence_ladder/critic_batch/tests/run_tests.py", "evidence_ladder/critic_batch/tests/stepE_tests.py",
    "evidence_ladder/execution/tests/run_tests.py", "evidence_ladder/models/tests/run_tests.py", "evidence_ladder/models/tests/backbone_tests.py",
    "evidence_ladder/models/tests/lowthink_tests.py", "evidence_ladder/orchestrator/tests/phaseC_tests.py", "evidence_ladder/orchestrator/tests/phaseF3_tests.py",
    "evidence_ladder/orchestrator/tests/run_tests.py",
]
EXTRA = ["evidence_ladder/sandbox_provider_tests_placeholder"]   # replaced by the provider suites once they exist


def parse(out: str):
    i = out.find('{\n "tests"')
    if i < 0:
        i = out.find('{"tests"')
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(out[i:])[0]
    except ValueError:
        return None


def child_env(backend: str):
    env = dict(os.environ, PYTHONPATH=str(ROOT), ATEN_CPU_CAPABILITY="avx2", LAYA_SANDBOX_BACKEND=backend, PYTHONIOENCODING="utf-8")
    return env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--backend", required=True, choices=["wsl", "mxc"])
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--suite", action="append", help="run only these suite paths (repeatable)")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        ap.error("output dir exists (create-only): %s" % out)
    out.mkdir(parents=True)
    res = {"backend": a.backend, "started_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "suites": {}}
    suites = a.suite or [s for s in SUITES + [e for e in EXTRA if (ROOT / e).exists()]]
    for s in suites:
        if not (ROOT / s).exists():
            res["suites"][s] = {"missing": True}
            continue
        t0 = time.time()
        try:
            p = subprocess.run([str(PY), s], cwd=str(ROOT), env=child_env(a.backend), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=a.timeout)
            res["suites"][s] = {"exit_code": p.returncode, "seconds": round(time.time() - t0, 1), "summary": parse(p.stdout), "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-800:]}
        except subprocess.TimeoutExpired:
            res["suites"][s] = {"exit_code": None, "timeout": a.timeout, "seconds": round(time.time() - t0, 1)}
        print(s, res["suites"][s].get("exit_code"), (res["suites"][s].get("summary") or {}).get("passed"), "/", (res["suites"][s].get("summary") or {}).get("tests"), flush=True)
    if a.variants:
        t0 = time.time()
        vdir = out / "variants"
        p = subprocess.run([str(PY), "tools/el_arch1_acceptance.py", str(vdir), "--variants-only"], cwd=str(ROOT), env=child_env(a.backend), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=a.timeout * 3)
        res["variants"] = {"exit_code": p.returncode, "seconds": round(time.time() - t0, 1), "stdout_tail": p.stdout[-2500:], "stderr_tail": p.stderr[-1200:]}
        sm = vdir / "VARIANTS_SUMMARY.json"
        if sm.exists():
            res["variants"]["summary_file"] = str(sm)
            try:
                res["variants"]["summary"] = json.loads(sm.read_text("utf-8"))
            except ValueError:
                pass
        print("variants", p.returncode, flush=True)
    res["finished_utc"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(out / "SUITES.json", "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
