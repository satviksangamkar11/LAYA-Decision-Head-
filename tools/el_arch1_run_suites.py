"""[EL-ARCH-1 F3] Runs every test file listed in tools/el_arch1_final_report.TEST_FILES as its own process (PYTHONPATH=., ATEN_CPU_CAPABILITY=avx2) and writes ONE create-only SUITE_RESULTS_<UTC>.json
(file, exit code, tests, passed, failed {name: error tail}, parsed) that the report generator resolves test references against. No GPU job is started by these files; the WSL ones use the sandbox.
  PYTHONPATH=. .venv/Scripts/python.exe tools/el_arch1_run_suites.py <out dir> [--only <substring>]"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import el_arch1_final_report as FR  # noqa: E402


def run_one(rel: str, timeout: int = 1800) -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT), ATEN_CPU_CAPABILITY="avx2", PYTHONUTF8="1")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(ROOT / rel)], cwd=str(ROOT), env=env, capture_output=True, timeout=timeout)
        out = p.stdout.decode("utf-8", "replace"); rc = p.returncode
    except subprocess.TimeoutExpired:
        return {"file": rel, "exit": None, "parsed": False, "error": "timeout", "failed": {}, "seconds": timeout}
    doc = None
    for i in range(len(out)):                                 # the suites print one (indented) JSON object; take the last that parses
        if out[i] == "{":
            try: doc = json.loads(out[i:]); break
            except ValueError: continue
    r = {"file": rel, "exit": rc, "parsed": isinstance(doc, dict) and "tests" in doc, "seconds": round(time.time() - t0, 1)}
    if r["parsed"]: r.update(tests=doc["tests"], passed=doc["passed"], failed=doc.get("failed", {}), skipped=doc.get("skipped"))
    else: r.update(failed={}, stdout_tail=out[-400:], stderr_tail=p.stderr.decode("utf-8", "replace")[-400:])
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--only")
    a = ap.parse_args()
    res = [run_one(f) for f in FR.TEST_FILES if (ROOT / f).is_file() and (not a.only or a.only in f)]
    doc = {"utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "suites": res, "all_exit_zero": all(r["exit"] == 0 for r in res)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / ("SUITE_RESULTS_%s.json" % doc["utc"]), "x", encoding="utf-8") as f: json.dump(doc, f, indent=1)
    print(json.dumps({r["file"].split("/")[-1]: (r["exit"], r.get("passed"), r.get("tests")) for r in res}, indent=0)); return 0 if doc["all_exit_zero"] else 1


if __name__ == "__main__":
    sys.exit(main())
