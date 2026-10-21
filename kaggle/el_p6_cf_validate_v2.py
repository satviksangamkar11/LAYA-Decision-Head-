"""[P6 option A, validation v2] Kaggle CPU kernel. Same frozen rule as P6_CF_QUALIFICATION_RULES_v1 rule 3, wider reference source: v1 used the 74 MB
`selected_accepted` subset, which holds Python solutions for only 2 of the 77 candidates. v2 scans the full open-r1/codeforces-submissions `default` config
(pinned 07e7bf31…, 36 shards, 5.55 GB, downloaded and filtered on Kaggle only) for ACCEPTED (verdict OK) Python/PyPy submissions to the candidates, then validates
every candidate exactly as v1: the first of up to 3 in submission-id order, every test, trusted jail, LcbRunner stdin rules, 6 s per test.
Tests come from the v1 kernel's output (kernel source `el-p6-cf-tests-v1`, hash-checked against its MANIFEST); nothing is re-exported.
Writes /kaggle/working/VALIDATION_v2.json, SOURCES_v2.json, RUN_SUMMARY.json. No GPU, no model."""
import concurrent.futures as CF
import glob, gzip, hashlib, json, os, shutil, subprocess, sys, tempfile, time, traceback, urllib.request
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working")
REV_SUB = "07e7bf31f284b0e90e02dfc7ab675d5821e1c068"
URL = "https://huggingface.co/datasets/open-r1/codeforces-submissions/resolve/%s/data/train-%%05d-of-00036.parquet" % REV_SUB
RS = {"kernel": "el_p6_cf_validate_v2", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rev_submissions": REV_SUB}


def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def finish(verdict, **kw):
    RS.update(verdict=verdict, total_seconds=round(time.time() - T0, 1), **kw)
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(RS, indent=1, default=str))
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)


def download(url, dest):
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=900) as r, open(dest, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            h.update(b); n += len(b); f.write(b)
    return {"url": url, "bytes": n, "sha256": h.hexdigest()}


def main():
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    man_p = next(p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_p5_v1")
    bundle, man = os.path.dirname(man_p), json.load(open(man_p))
    if [f for f, h in man["files"].items() if hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]:
        finish("FAILED_BUNDLE_HASH")
    v1 = Path(next(p for p in glob.glob("/kaggle/input/**/p6_cf/MANIFEST.json", recursive=True))).parent
    vman = json.loads((v1 / "MANIFEST.json").read_text())
    badt = [f for f, m in vman.items() if hashlib.sha256((v1 / f).read_bytes()).hexdigest() != m["sha256"]]
    if badt:
        finish("FAILED_V1_TEST_HASH", bad=badt[:5])
    want = sorted(json.loads((v1 / "META.json").read_text()))
    RS.update(candidates=len(want), v1_manifest_files=len(vman))
    work = OUT / "bundle"; shutil.copytree(bundle, work, dirs_exist_ok=True)
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "kaggle"))
    java = shutil.which("java"); JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0")
    import el_bank40_run_v1 as BK
    from evidence_ladder.execution import kaggle_jail as KJ, config as K
    KJ.build_rootfs(JAVA_HOME)
    if BK.selfcheck(KJ, K, str(work)).get("status") != "PASS":
        finish("JAIL_UNAVAILABLE")
    q = subprocess.run([sys.executable, str(work / "tools/el_kaggle_port_qual.py"), str(OUT / "qualification")], cwd=str(work), env=dict(os.environ, PYTHONPATH=str(work)),
                       capture_output=True, text=True, timeout=1800)
    if q.returncode != 0:
        finish("PORT_NOT_QUALIFIED")
    raw = Path(tempfile.mkdtemp(prefix="sub-", dir="/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else None))
    sources, refs = {}, {p: [] for p in want}
    wanted = pa.array(want)

    def one(k):
        dest = raw / ("s%02d.parquet" % k)
        info = download(URL % k, dest)
        t = pq.read_table(dest, columns=["submission_id", "source", "programmingLanguage", "problem_id", "verdict"])
        t = t.filter(pc.and_(pc.is_in(t.column("problem_id"), value_set=wanted), pc.equal(t.column("verdict"), "OK")))
        rows = [r for r in t.to_pylist() if "py" in (r["programmingLanguage"] or "").lower()]
        dest.unlink()
        return k, info, rows
    with CF.ThreadPoolExecutor(max_workers=6) as ex:
        for k, info, rows in ex.map(one, range(36)):
            sources["data/train-%05d-of-00036.parquet" % k] = info
            for r in rows:
                refs[r["problem_id"]].append(r)
            log("shard", k, "MB", round(info["bytes"] / 1e6, 1), "python accepted kept", len(rows))
    for p in refs:
        refs[p].sort(key=lambda r: r["submission_id"])
    log("python accepted solutions found for", sum(1 for v in refs.values() if v), "of", len(want))
    from evidence_ladder.execution.kaggle_port import KaggleJailPort
    from evidence_ladder.execution.lcb_runner import LcbRunner
    runner = LcbRunner(KaggleJailPort().run, Path(tempfile.mkdtemp(prefix="val-")))
    val = {}
    for p in want:
        with gzip.open(v1 / ("tests_%s.json.gz" % p.replace("/", "_")), "rt", encoding="utf-8") as f:
            tests = json.load(f)
        rec = {"tests": len(tests), "python_accepted_available": len(refs[p]), "attempts": []}
        for s in refs[p][:3]:
            st = {"pass": 0, "wrong_answer": 0, "timeout": 0, "runtime_error": 0}
            t1 = time.time()
            for t in tests:
                st[runner.run_case(s["source"], {"input": t["input"], "output": t["output"], "testtype": "stdin"}, None)["status"]] += 1
            rec["attempts"].append({"submission_id": s["submission_id"], "language": s["programmingLanguage"], **st, "seconds": round(time.time() - t1, 1)})
            if st["pass"] == len(tests) and tests:
                break
        rec["validated"] = any(a["pass"] == rec["tests"] and rec["tests"] > 0 for a in rec["attempts"])
        val[p] = rec
        log(p, "tests", rec["tests"], "refs", rec["python_accepted_available"], "validated", rec["validated"],
            [(a["pass"], a["wrong_answer"], a["timeout"], a["runtime_error"]) for a in rec["attempts"]])
    (OUT / "VALIDATION_v2.json").write_text(json.dumps(val, indent=1))
    (OUT / "SOURCES_v2.json").write_text(json.dumps(sources, indent=1))
    shutil.rmtree(raw, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
    finish("OK", validated=sum(v["validated"] for v in val.values()), with_reference=sum(1 for v in val.values() if v["python_accepted_available"]),
           downloaded_bytes=sum(s["bytes"] for s in sources.values()))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:      # noqa: BLE001
        RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-3000:]; traceback.print_exc()
        finish("FAILED")
