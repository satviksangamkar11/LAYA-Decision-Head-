"""[P5 Step 1(a)] Jail flakiness: the EXACT P1-21 candidate that timed out in arm D of the Kaggle smoke (answer 449e72bb..., re.fullmatch), run through the SAME
execution path as the smoke (KaggleJailPort + ProviderExecutionBackend(trusted=True), one backend object calibrated once, as in a shard), 20 times isolated and
20 times under controlled CPU load (os.cpu_count() busy-loop processes). Two call shapes per trial: public example 4 alone, and the full public batch exactly as the
controller calls it (run_public). Records every verdict, per-input seconds, the verifier's time accounting (launch overhead, candidate seconds, outcome) and the load.
Measurement only: no limit, timeout or retry is changed. CPU only, no model, no internet. Output: /kaggle/working/FLAKE.json"""
import glob, hashlib, json, multiprocessing, os, shutil, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
CAND = ('def solve(data):\n    import re\n    pattern = data["pattern"]\n    text = data["text"]\n    \n'
        '    # Use regular expression to match the pattern with the text\n    if re.fullmatch(pattern, text):\n        return True\n    else:\n        return False')
CAND_SHA = "449e72bbf49ccc7dd78c542281235426d12212e1ccaa39ac9b0f2e832685a0f7"
N = 20


def burn():
    while True:
        pass


def main():
    R = {"kernel": "el_p5_jail_flake_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "nproc": os.cpu_count()}
    assert hashlib.sha256(CAND.encode()).hexdigest() == CAND_SHA, "candidate text differs from the smoke answer"
    man_p = next(p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_p5_v1")
    work = OUT / "bundle"; shutil.copytree(os.path.dirname(man_p), work, dirs_exist_ok=True)
    R["bundle_commit"] = json.load(open(man_p)).get("commit")
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "tools")); sys.path.insert(0, str(work / "kaggle"))
    java = shutil.which("java")
    JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0")
    os.chdir(work)
    from evidence_ladder.execution import kaggle_jail as KJ
    KJ.build_rootfs(JAVA_HOME)
    from evidence_ladder.common.interfaces import Artifact, TaskSpec
    from evidence_ladder.execution.kaggle_port import KaggleJailPort
    from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
    import el_bank_local_run as RUN
    rec = RUN.load_task_ids(["P1-21"])[0]
    pub = rec["public_tests"]
    R["public_examples"] = len(pub); R["example4_input"] = pub[3]["input"]
    task = TaskSpec("P1-21", "bank:P1-21", "python", "python", rec["prompt"], pub, rec.get("difficulty", ""))
    art = Artifact.make("P1-21", "python", CAND)
    exe = ProviderExecutionBackend(KaggleJailPort(), backend="kaggle-jail", scratch=OUT / "_engine", trusted=True)
    exe.start(languages=("python",))
    tv = exe.trusted_verifier
    task4 = TaskSpec("P1-21", "bank:P1-21", "python", "python", rec["prompt"], [pub[3]], rec.get("difficulty", ""))
    rows = []

    def trial(cond, k):
        for shape, t in (("example4_alone", task4), ("full_public_batch", task)):
            tv.drain_observations()
            t0 = time.time()
            outs = exe.run_public(art, t)
            rows.append({"condition": cond, "trial": k, "shape": shape, "wall_s": round(time.time() - t0, 3), "loadavg": os.getloadavg(),
                         "statuses": [(o.get("example_index"), o["status"], o.get("failure_class", "")) for o in outs if o["check_id"] != "parse_v1"],
                         "accounting": dict(tv.accounting), "per_input": tv.drain_observations()})

    for k in range(N):
        trial("isolated", k)
    R["launch_overhead_s"] = tv.launch_overhead.get("python")
    procs = [multiprocessing.Process(target=burn, daemon=True) for _ in range(os.cpu_count() or 4)]
    for p in procs: p.start()
    time.sleep(5)
    try:
        for k in range(N):
            trial("loaded", k)
    finally:
        for p in procs: p.terminate()
    R["load_processes"] = len(procs)
    summ = {}
    for r in rows:
        key = "%s/%s" % (r["condition"], r["shape"])
        s = summ.setdefault(key, {"n": 0, "all_pass": 0, "timeouts": 0, "outcomes": {}, "max_input_s": 0.0})
        s["n"] += 1
        s["all_pass"] += all(st == "pass" for _, st, _ in r["statuses"])
        s["timeouts"] += any(st == "timeout" for _, st, _ in r["statuses"])
        o = r["accounting"].get("outcome"); s["outcomes"][o] = s["outcomes"].get(o, 0) + 1
        s["max_input_s"] = max([s["max_input_s"]] + [x["seconds"] for x in r["per_input"]])
    R.update(summary=summ, rows=rows, ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (OUT / "FLAKE.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps(summ, indent=1), flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:      # noqa: BLE001
        traceback.print_exc()
        (OUT / "FLAKE_ERROR.txt").write_text(traceback.format_exc())
