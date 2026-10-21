"""[P6 Codeforces, validation v3 = rules v2] Kaggle CPU kernel. No GPU, no model.
1. code bundle `el_ml_v1` (multi-language sandbox branch bca45ad9) hash check; jail rootfs built WITH the C++ toolchain; jail self-check; python port qualification;
   C++ qualification in the REAL Kaggle jail (tools/el_multilang_qual.py --backend kaggle-jail --languages cpp). Any failure stops the run (no fallback).
2. accepted submissions (open-r1/codeforces-submissions @ 07e7bf31…, 36 shards, on Kaggle only) for the 77 candidates in qualified languages (Python, C++).
3. tests from the v1 kernel output (kernel source el-p6-cf-tests-v1), hash-checked against its MANIFEST.
4. per candidate (P6_CF_QUALIFICATION_RULES_v2 rule 3): references in submission-id order across qualified languages, first of up to 3, every test, LcbRunner stdin,
   6 s per test; per attempt: compile error, pass / wrong_answer / timeout / runtime_error counts.
Writes VALIDATION_v3.json, SOURCES_v3.json, QUAL_cpp_kaggle.json, RUN_SUMMARY.json."""
import concurrent.futures as CF
import glob, gzip, hashlib, json, os, shutil, subprocess, sys, tempfile, time, traceback, urllib.request
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working")
REV_SUB = "07e7bf31f284b0e90e02dfc7ab675d5821e1c068"
URL = "https://huggingface.co/datasets/open-r1/codeforces-submissions/resolve/%s/data/train-%%05d-of-00036.parquet" % REV_SUB
RS = {"kernel": "el_p6_cf_validate_v3", "rules": "P6_CF_QUALIFICATION_RULES_v2", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def finish(verdict, **kw):
    RS.update(verdict=verdict, total_seconds=round(time.time() - T0, 1), **kw)
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(RS, indent=1, default=str))
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)


def lang_of(name):
    n = (name or "").lower()
    return "python" if "py" in n else ("cpp" if "c++" in n else None)


def sid(r):
    s = str(r["submission_id"])
    return (0, int(s)) if s.isdigit() else (1, s)


def main():
    import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    man_p = next(p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_ml_v1")
    bundle, man = os.path.dirname(man_p), json.load(open(man_p))
    if [f for f, h in man["files"].items() if hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]:
        finish("FAILED_BUNDLE_HASH")
    v1 = Path(next(p for p in glob.glob("/kaggle/input/**/p6_cf/MANIFEST.json", recursive=True))).parent
    vman = json.loads((v1 / "MANIFEST.json").read_text())
    if [f for f, m in vman.items() if hashlib.sha256((v1 / f).read_bytes()).hexdigest() != m["sha256"]]:
        finish("FAILED_V1_TEST_HASH")
    want = sorted(json.loads((v1 / "META.json").read_text()))
    RS.update(candidates=len(want), bundle_commit=man.get("commit"))
    work = OUT / "bundle"; shutil.copytree(bundle, work, dirs_exist_ok=True)
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "kaggle"))
    java = shutil.which("java"); JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0")
    env = dict(os.environ, PYTHONPATH=str(work))
    import el_bank40_run_v1 as BK
    from evidence_ladder.execution import kaggle_jail as KJ, config as K
    RS["rootfs"] = KJ.build_rootfs(JAVA_HOME, toolchains=("cpp",))
    if BK.selfcheck(KJ, K, str(work)).get("status") != "PASS":
        finish("JAIL_UNAVAILABLE")
    q = subprocess.run([sys.executable, str(work / "tools/el_kaggle_port_qual.py"), str(OUT / "qual_python")], cwd=str(work), env=env, capture_output=True, text=True, timeout=1800)
    RS["python_port_qualified"] = q.returncode == 0
    if q.returncode != 0:
        finish("PORT_NOT_QUALIFIED", detail=(q.stdout + q.stderr)[-800:])
    q = subprocess.run([sys.executable, str(work / "tools/el_multilang_qual.py"), str(OUT / "qual_cpp"), "--backend", "kaggle-jail", "--languages", "cpp"], cwd=str(work), env=env,
                       capture_output=True, text=True, timeout=3600)
    qf = OUT / "qual_cpp" / "QUAL_kaggle-jail.json"
    if qf.exists():
        shutil.copy(qf, OUT / "QUAL_cpp_kaggle.json")
    RS["cpp_qualification_exit"] = q.returncode
    if q.returncode != 0:
        finish("CPP_NOT_QUALIFIED", detail=(q.stdout + q.stderr)[-1500:])
    log("C++ qualified in the real Kaggle jail")
    raw = Path(tempfile.mkdtemp(prefix="sub-", dir="/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else None))
    sources, refs = {}, {p: [] for p in want}
    wanted = pa.array(want)

    def one(k):
        dest = raw / ("s%02d.parquet" % k)
        h, n = hashlib.sha256(), 0
        with urllib.request.urlopen(URL % k, timeout=900) as r, open(dest, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                h.update(b); n += len(b); f.write(b)
        t = pq.read_table(dest, columns=["submission_id", "source", "programmingLanguage", "problem_id", "verdict"])
        t = t.filter(pc.and_(pc.is_in(t.column("problem_id"), value_set=wanted), pc.equal(t.column("verdict"), "OK")))
        rows = [r for r in t.to_pylist() if lang_of(r["programmingLanguage"])]
        dest.unlink()
        return k, {"bytes": n, "sha256": h.hexdigest()}, rows
    with CF.ThreadPoolExecutor(max_workers=6) as ex:
        for k, info, rows in ex.map(one, range(36)):
            sources["data/train-%05d-of-00036.parquet" % k] = info
            for r in rows:
                refs[r["problem_id"]].append(r)
    for p in refs:
        refs[p].sort(key=sid)
    log("references found for", sum(1 for v in refs.values() if v), "of", len(want))
    from evidence_ladder.execution.kaggle_port import KaggleJailPort
    from evidence_ladder.execution.lcb_runner import LcbRunner
    runner = LcbRunner(KaggleJailPort().run, Path(tempfile.mkdtemp(prefix="val-")))
    runner.capture_limit = 1 << 20          # the Kaggle jail port's capture (as tools/el_multilang_qual.py uses for kaggle-jail); long Codeforces outputs must be compared, not cut
    val = {}
    for p in want:
        with gzip.open(v1 / ("tests_%s.json.gz" % p.replace("/", "_")), "rt", encoding="utf-8") as f:
            tests = json.load(f)
        rec = {"tests": len(tests), "references_available": len(refs[p]), "attempts": []}
        for s in refs[p][:3]:
            lang = lang_of(s["programmingLanguage"])
            st = {"pass": 0, "wrong_answer": 0, "timeout": 0, "runtime_error": 0, "compile_error": 0, "other": 0}
            t1 = time.time()
            for t in tests:
                try:
                    o = runner.run_case(s["source"], {"input": t["input"], "output": t["output"], "testtype": "stdin"}, None, language=lang)
                    k = o["status"] if o["status"] in st else ("compile_error" if "compile" in str(o.get("detail", "")).lower() else "other")
                except Exception as e:      # noqa: BLE001
                    k = "compile_error" if "compil" in str(e).lower() else "other"
                st[k] += 1
                if k == "compile_error":
                    break
            rec["attempts"].append({"submission_id": s["submission_id"], "language": s["programmingLanguage"], "runner_language": lang, **st,
                                    "seconds": round(time.time() - t1, 1)})
            if st["pass"] == len(tests) and tests:
                break
        rec["validated"] = any(a["pass"] == rec["tests"] and rec["tests"] > 0 for a in rec["attempts"])
        val[p] = rec
        log(p, "tests", rec["tests"], "refs", rec["references_available"], "validated", rec["validated"],
            [(a["runner_language"], a["pass"], a["wrong_answer"], a["timeout"], a["runtime_error"], a["compile_error"], a["other"]) for a in rec["attempts"]])
    (OUT / "VALIDATION_v3.json").write_text(json.dumps(val, indent=1))
    (OUT / "SOURCES_v3.json").write_text(json.dumps(sources, indent=1))
    shutil.rmtree(raw, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
    finish("OK", validated=sum(v["validated"] for v in val.values()), with_reference=sum(1 for v in val.values() if v["references_available"]))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:      # noqa: BLE001
        RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-3000:]; traceback.print_exc()
        finish("FAILED")
