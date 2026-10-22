"""[P6 Codeforces, grading v1] Kaggle CPU kernel. No GPU, no model. Grades the cf ANSWERS.jsonl of a FINISHED run against the validated tests.
1. code bundle `el_ml_v1` hash check; jail rootfs; jail self-check; python port qualification (fail closed, no fallback).
2. plan file `P6_GRADE_PLAN.json` (found under /kaggle/input): {"answers_path": "<ANSWERS.jsonl, absolute or relative to /kaggle/input>", "final_ids": [...],
   "tests_manifest_sha256": "<sha256 of the v1 export MANIFEST.json>"}. Refuses on a task id outside final_ids or a tests-manifest hash mismatch.
3. tests from the kernel source el-p6-cf-tests-v1 (p6_cf/MANIFEST.json hash-checked, then tests_<id with / -> _>.json.gz).
4. every answer: LcbRunner stdin, Python, default 6 s per test, ALL tests (no early stop), timeouts counted; NO_ANSWER when source is null.
Writes GRADED_cf.jsonl ({"task","arm","grade","passed","total","first_fail","timeouts"}) and RUN_SUMMARY.json. Test inputs/outputs are never printed or saved."""
import glob, gzip, hashlib, json, os, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working")
RS = {"kernel": "el_p6_cf_grade_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def finish(verdict, **kw):
    RS.update(verdict=verdict, total_seconds=round(time.time() - T0, 1), **kw)
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(RS, indent=1, default=str))
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)


def main():
    man_p = next(p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_ml_v1")
    bundle, man = os.path.dirname(man_p), json.load(open(man_p))
    if [f for f, h in man["files"].items() if hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]:
        finish("FAILED_BUNDLE_HASH")
    plan = json.load(open(next(iter(glob.glob("/kaggle/input/**/P6_GRADE_PLAN.json", recursive=True)))))
    ap = plan["answers_path"]
    ap = ap if ap.startswith("/") else os.path.join("/kaggle/input", ap)
    if not os.path.isfile(ap):      # path fix 2026-10-22 (mount layout differs): exactly one file of that name under /kaggle/input, hash-checked below
        hits = glob.glob("/kaggle/input/**/" + os.path.basename(ap), recursive=True)
        if len(hits) != 1:
            finish("FAILED_ANSWERS_NOT_FOUND", hits=hits[:5])
        ap = hits[0]
    if plan.get("answers_sha256") and hashlib.sha256(open(ap, "rb").read()).hexdigest() != plan["answers_sha256"]:
        finish("REFUSED_ANSWERS_HASH_MISMATCH")
    RS["answers_file"] = ap
    answers = [json.loads(l) for l in open(ap, encoding="utf-8").read().splitlines() if l.strip()]
    bad = sorted({a["task"] for a in answers} - set(plan["final_ids"]))
    if bad:
        finish("REFUSED_TASK_NOT_IN_PLAN", tasks=bad)
    mp = Path(next(iter(glob.glob("/kaggle/input/**/p6_cf/MANIFEST.json", recursive=True))))
    v1 = mp.parent
    if hashlib.sha256(mp.read_bytes()).hexdigest() != plan["tests_manifest_sha256"]:
        finish("REFUSED_TESTS_MANIFEST_MISMATCH")
    vman = json.loads(mp.read_text())
    if [f for f, m in vman.items() if hashlib.sha256((v1 / f).read_bytes()).hexdigest() != m["sha256"]]:
        finish("FAILED_V1_TEST_HASH")
    RS.update(answers=len(answers), bundle_commit=man.get("commit"))
    work = OUT / "bundle"; shutil.copytree(bundle, work, dirs_exist_ok=True)
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "kaggle"))
    java = shutil.which("java"); JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0")
    env = dict(os.environ, PYTHONPATH=str(work))
    import el_bank40_run_v1 as BK
    from evidence_ladder.execution import kaggle_jail as KJ, config as K
    RS["rootfs"] = KJ.build_rootfs(JAVA_HOME)
    if BK.selfcheck(KJ, K, str(work)).get("status") != "PASS":
        finish("JAIL_UNAVAILABLE")
    q = subprocess.run([sys.executable, str(work / "tools/el_kaggle_port_qual.py"), str(OUT / "qual_python")], cwd=str(work), env=env, capture_output=True, text=True, timeout=1800)
    RS["python_port_qualified"] = q.returncode == 0
    if q.returncode != 0:
        finish("PORT_NOT_QUALIFIED", detail=(q.stdout + q.stderr)[-800:])
    from evidence_ladder.execution.kaggle_port import KaggleJailPort
    from evidence_ladder.execution.lcb_runner import LcbRunner
    runner = LcbRunner(KaggleJailPort().run, Path(tempfile.mkdtemp(prefix="grade-")))
    runner.capture_limit = 1 << 20      # as the validation kernel: long outputs are compared, not cut
    cache, out = {}, OUT / "GRADED_cf.jsonl"
    if out.exists():
        finish("REFUSED_OUTPUT_EXISTS")
    with open(out, "w", encoding="utf-8") as f:
        for a in answers:
            row = {"task": a["task"], "arm": a["arm"]}
            if not a.get("source"):
                row.update(grade="NO_ANSWER", passed=0, total=0, first_fail=None, timeouts=0)
            else:
                if a["task"] not in cache:
                    with gzip.open(v1 / ("tests_%s.json.gz" % a["task"].replace("/", "_")), "rt", encoding="utf-8") as g:
                        cache[a["task"]] = json.load(g)
                tests, passed, ff, to, ie = cache[a["task"]], 0, None, 0, 0
                if plan.get("time_limit_s") and time.time() - T0 > plan["time_limit_s"]:      # P7 option; P6 plans have none
                    shutil.rmtree(work, ignore_errors=True); finish("TIME_LIMIT", graded_rows=sum(1 for _ in open(out)))
                for t in tests:         # ALL tests (P6); P7 candidate plans may set stop_at_first_fail (pass/fail only, 'passed' is then a lower bound)
                    if ff is not None and plan.get("stop_at_first_fail"):
                        break
                    try:
                        o = runner.run_case(a["source"], {"input": t["input"], "output": t["output"], "testtype": "stdin"}, None)["status"]
                    except Exception as e:       # noqa: BLE001   hardening 2026-10-22: a grader/sandbox exception is an INFRASTRUCTURE error, never a candidate runtime_error
                        o = "infra_error"
                        RS.setdefault("infra_errors", []).append({"task": a["task"], "arm": a["arm"], "error": repr(e)[:200]})
                    passed += o == "pass"; to += o == "timeout"; ie += o == "infra_error"
                    if o != "pass" and ff is None:
                        ff = o
                row.update(grade="INFRA_ERROR" if ie else "PASS" if tests and passed == len(tests) else "FAIL", passed=passed, total=len(tests), first_fail=ff, timeouts=to, infra_errors=ie)
            f.write(json.dumps(row) + "\n"); f.flush()
            log(row["task"], row["arm"], row["grade"], row["passed"], "/", row["total"])
    shutil.rmtree(work, ignore_errors=True)
    finish("INFRA_ERRORS" if RS.get("infra_errors") else "OK", graded=len(answers))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:      # noqa: BLE001
        RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-3000:]; traceback.print_exc()
        finish("FAILED")
