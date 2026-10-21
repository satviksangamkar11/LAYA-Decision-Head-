"""[P6 option A] Kaggle CPU kernel: selective retrieval + validation of the Codeforces candidate tests. No GPU, no model, no generation.
1. bundle hash check (el_p5_v1 code bundle), Kaggle jail rootfs + self-check, jail-port qualification (fail closed)
2. pinned downloads (open-r1/codeforces @ fbe3f6e9…, open-r1/codeforces-submissions @ 07e7bf31…): the verifiable test-split metadata parquet, the 39 contest
   test files `generated_tests/test_cases_<contest>.parquet` (parallel), the accepted-submissions parquet; every file's size and sha256 recorded
3. keep only the 77 candidates' rows (a raw contest file is deleted right after filtering)
4. validation in the trusted jail (LcbRunner, stdin, official comparison rules, 6 s per test): accepted PYTHON human solutions, deterministic order (by
   submission id), up to 3 per problem until one passes every test; per solution: passed / failed / timeouts / runtime errors
5. export /kaggle/working/p6_cf/: tests_<problem>.json.gz (inputs + outputs), META.json (statement fields + examples + limits + provenance), VALIDATION.json,
   SOURCES.json (sizes + hashes), MANIFEST.json (sha256 of every exported file). Grading data: private Kaggle output only (never git, GLM or a GPU bundle)."""
import concurrent.futures as CF
import glob, gzip, hashlib, json, os, shutil, subprocess, sys, tempfile, time, traceback, urllib.request
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working"); EXP = OUT / "p6_cf"
REV_CF = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
REV_SUB = "07e7bf31f284b0e90e02dfc7ab675d5821e1c068"
CF_URL = "https://huggingface.co/datasets/open-r1/codeforces/resolve/%s/%%s" % REV_CF
SUB_URL = "https://huggingface.co/datasets/open-r1/codeforces-submissions/resolve/%s/%%s" % REV_SUB
RS = {"kernel": "el_p6_cf_tests_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rev_codeforces": REV_CF, "rev_submissions": REV_SUB}


def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def finish(verdict, **kw):
    RS.update(verdict=verdict, total_seconds=round(time.time() - T0, 1), **kw)
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(RS, indent=1, default=str))
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)


def download(url, dest):
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            h.update(b); n += len(b); f.write(b)
    return {"url": url, "bytes": n, "sha256": h.hexdigest()}


def main():
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    man_p = next(p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_p5_v1")
    bundle, man = os.path.dirname(man_p), json.load(open(man_p))
    bad = [f for f, h in man["files"].items() if hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]
    if bad:
        finish("FAILED_BUNDLE_HASH", bad=bad[:5])
    work = OUT / "bundle"; shutil.copytree(bundle, work, dirs_exist_ok=True)
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "kaggle"))
    java = shutil.which("java"); JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0")
    cands = list(CANDS_INLINE)
    if len(cands) != 77:
        finish("FAILED_CANDIDATES", n=len(cands))
    want = set(cands); contests = sorted({c.split("/")[0] for c in cands})
    RS.update(candidates=len(want), contests=len(contests), bundle_commit=man.get("commit"))
    import el_bank40_run_v1 as BK
    from evidence_ladder.execution import kaggle_jail as KJ, config as K
    KJ.build_rootfs(JAVA_HOME)
    jail = BK.selfcheck(KJ, K, str(work))
    if jail.get("status") != "PASS":
        finish("JAIL_UNAVAILABLE", jail=jail)
    q = subprocess.run([sys.executable, str(work / "tools/el_kaggle_port_qual.py"), str(OUT / "qualification")], cwd=str(work), env=dict(os.environ, PYTHONPATH=str(work)),
                       capture_output=True, text=True, timeout=1800)
    if q.returncode != 0:
        finish("PORT_NOT_QUALIFIED", detail=q.stdout[-500:] + q.stderr[-500:])
    log("jail + port qualified")
    EXP.mkdir(parents=True, exist_ok=True); raw = Path(tempfile.mkdtemp(prefix="raw-", dir="/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else None))
    sources = {}
    # ---- metadata (statement fields, examples, limits) of the candidates
    sources["verifiable/test"] = download(CF_URL % "verifiable/test-00000-of-00001.parquet", raw / "meta.parquet")
    mt = pq.read_table(raw / "meta.parquet").to_pylist()
    meta = {r["id"]: {k: r.get(k) for k in ("id", "contest_id", "index", "contest_name", "contest_start", "title", "description", "input_format", "output_format",
                                              "note", "examples", "time_limit", "memory_limit", "rating", "tags", "testset_size", "official_tests",
                                              "official_tests_complete", "input_mode", "generated_checker", "interaction_format")} for r in mt if r["id"] in want}
    for m in meta.values():
        m["url"] = "https://codeforces.com/contest/%s/problem/%s" % (m["contest_id"], m["index"])
    (EXP / "META.json").write_text(json.dumps(meta, default=str))
    log("metadata:", len(meta), "of", len(want))
    # ---- contest test files: parallel download, filter, delete
    tests = {p: [] for p in want}

    def one(c):
        dest = raw / ("tc_%s.parquet" % c)
        info = download(CF_URL % ("generated_tests/test_cases_%s.parquet" % c), dest)
        t = pq.read_table(dest, columns=["problem_id", "input", "output", "test_i"])
        t = t.filter(pc.is_in(t.column("problem_id"), value_set=pa.array(sorted(want))))
        rows = t.to_pylist()
        dest.unlink()
        return c, info, rows
    with CF.ThreadPoolExecutor(max_workers=6) as ex:
        for c, info, rows in ex.map(one, contests):
            sources["generated_tests/test_cases_%s.parquet" % c] = info
            for r in rows:
                tests[r["problem_id"]].append({"i": r["test_i"], "input": r["input"], "output": r["output"], "kind": "generated"})
            log("contest", c, "MB", round(info["bytes"] / 1e6, 1), "kept rows", len(rows))
    for p, m in meta.items():          # complete official tests are added (deduplicated by input)
        if m.get("official_tests_complete"):
            have = {t["input"] for t in tests[p]}
            tests[p] += [{"i": "o%d" % k, "input": o["input"], "output": o["output"], "kind": "official"} for k, o in enumerate(m["official_tests"] or []) if o["input"] not in have]
    for p in tests:
        tests[p].sort(key=lambda t: str(t["i"]))
    # ---- accepted human solutions
    sources["selected_accepted"] = download(SUB_URL % "selected_accepted/train-00000-of-00001.parquet", raw / "acc.parquet")
    acc = pq.read_table(raw / "acc.parquet", columns=["submission_id", "source", "programmingLanguage", "problem_id", "verdict"]).to_pylist()
    py = {}
    for r in acc:
        if r["problem_id"] in want and "py" in (r["programmingLanguage"] or "").lower():
            py.setdefault(r["problem_id"], []).append(r)
    for p in py:
        py[p].sort(key=lambda r: r["submission_id"])
    log("python accepted solutions for", len(py), "of", len(want), "candidates")
    # ---- validation in the trusted jail
    from evidence_ladder.execution.kaggle_port import KaggleJailPort
    from evidence_ladder.execution.lcb_runner import LcbRunner
    runner = LcbRunner(KaggleJailPort().run, Path(tempfile.mkdtemp(prefix="val-")))
    val = {}
    for p in sorted(want):
        rec = {"tests": len(tests[p]), "generated": sum(t["kind"] == "generated" for t in tests[p]), "official": sum(t["kind"] == "official" for t in tests[p]),
               "python_solutions_available": len(py.get(p, [])), "attempts": []}
        for s in py.get(p, [])[:3]:
            st = {"pass": 0, "wrong_answer": 0, "timeout": 0, "runtime_error": 0}
            t1 = time.time()
            for t in tests[p]:
                st[runner.run_case(s["source"], {"input": t["input"], "output": t["output"], "testtype": "stdin"}, None)["status"]] += 1
            rec["attempts"].append({"submission_id": s["submission_id"], "language": s["programmingLanguage"], **st, "seconds": round(time.time() - t1, 1)})
            if st["pass"] == len(tests[p]) and len(tests[p]) > 0:
                break
        rec["validated"] = any(a["pass"] == rec["tests"] and rec["tests"] > 0 for a in rec["attempts"])
        val[p] = rec
        log(p, "tests", rec["tests"], "validated", rec["validated"], [(a["pass"], a["timeout"], a["wrong_answer"], a["runtime_error"]) for a in rec["attempts"]])
        with gzip.open(EXP / ("tests_%s.json.gz" % p.replace("/", "_")), "wt", encoding="utf-8") as f:
            json.dump(tests[p], f)
    (EXP / "VALIDATION.json").write_text(json.dumps(val, indent=1))
    (EXP / "SOURCES.json").write_text(json.dumps(sources, indent=1))
    mani = {f.name: {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(EXP.iterdir())}
    (EXP / "MANIFEST.json").write_text(json.dumps(mani, indent=1))
    shutil.rmtree(raw, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
    finish("OK", export_bytes=sum(m["bytes"] for m in mani.values()), validated=sum(v["validated"] for v in val.values()),
           downloaded_bytes=sum(s["bytes"] for s in sources.values()))


CANDS_INLINE = []      # replaced at pack time

if __name__ == "__main__":
    try:
        main()
    except BaseException as e:      # noqa: BLE001
        RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-3000:]; traceback.print_exc()
        finish("FAILED")
