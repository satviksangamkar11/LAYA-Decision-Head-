"""[P5 Step 1(a), follow-up A2] Jail flakiness with a SECOND CONCURRENT JAIL USER (the smoke ran the LCB and bank shards in parallel, both launching jails).
Same candidate (P1-21 answer 449e72bb...), same path (KaggleJailPort + ProviderExecutionBackend(trusted=True), one backend calibrated once) and same two call shapes
as v1, 20 trials per condition:
  isolated      : nothing else running (baseline on this machine);
  peer_trivial  : a second process runs trivial stdin candidates through its own KaggleJailPort + LcbRunner in a tight loop (jail launch contention);
  peer_heavy    : the second process runs a busy-loop stdin candidate that the LCB runner kills at its 6 s limit, over and over (kill / cleanup contention,
                  what a wrong LCB candidate does).
Measurement only: no limit, timeout or retry is changed. CPU only, no model, no internet. Output: /kaggle/working/FLAKE2.json"""
import glob, hashlib, json, multiprocessing, os, shutil, sys, tempfile, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
CAND = ('def solve(data):\n    import re\n    pattern = data["pattern"]\n    text = data["text"]\n    \n'
        '    # Use regular expression to match the pattern with the text\n    if re.fullmatch(pattern, text):\n        return True\n    else:\n        return False')
CAND_SHA = "449e72bbf49ccc7dd78c542281235426d12212e1ccaa39ac9b0f2e832685a0f7"
N = 20
PEER = {"peer_trivial": "print(input())\n", "peer_heavy": "while True:\n    pass\n"}


def peer(kind, stop, stats_path):
    from evidence_ladder.execution.kaggle_port import KaggleJailPort
    from evidence_ladder.execution.lcb_runner import LcbRunner
    lr = LcbRunner(KaggleJailPort().run, Path(tempfile.mkdtemp(prefix="peer-")))
    st = {}
    while not os.path.exists(stop):
        o = lr.run_case(PEER[kind], {"input": "1", "output": "1", "testtype": "stdin"}, None)
        st[o["status"]] = st.get(o["status"], 0) + 1
        Path(stats_path).write_text(json.dumps(st))


def main():
    R = {"kernel": "el_p5_jail_flake_v2", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "nproc": os.cpu_count()}
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
    task = TaskSpec("P1-21", "bank:P1-21", "python", "python", rec["prompt"], pub, rec.get("difficulty", ""))
    task4 = TaskSpec("P1-21", "bank:P1-21", "python", "python", rec["prompt"], [pub[3]], rec.get("difficulty", ""))
    art = Artifact.make("P1-21", "python", CAND)
    exe = ProviderExecutionBackend(KaggleJailPort(), backend="kaggle-jail", scratch=OUT / "_engine", trusted=True)
    exe.start(languages=("python",))
    tv = exe.trusted_verifier
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
    R["peers"] = {}
    for kind in ("peer_trivial", "peer_heavy"):
        stop, stats = str(OUT / ("stop_" + kind)), str(OUT / ("stats_" + kind + ".json"))
        p = multiprocessing.Process(target=peer, args=(kind, stop, stats), daemon=True)
        p.start()
        time.sleep(8)
        try:
            for k in range(N):
                trial(kind, k)
        finally:
            Path(stop).write_text("stop")
            p.join(30)
            if p.is_alive():
                p.terminate()
        R["peers"][kind] = json.loads(Path(stats).read_text()) if os.path.exists(stats) else None
    summ = {}
    for r in rows:
        key = "%s/%s" % (r["condition"], r["shape"])
        s = summ.setdefault(key, {"n": 0, "ex4_pass": 0, "timeouts": 0, "outcomes": {}, "max_input_s": 0.0, "max_candidate_s": 0.0})
        s["n"] += 1
        s["ex4_pass"] += any(i in (3, 0) and st == "pass" for i, st, _ in r["statuses"][-1:])
        s["timeouts"] += any(st == "timeout" for _, st, _ in r["statuses"])
        o = r["accounting"].get("outcome"); s["outcomes"][o] = s["outcomes"].get(o, 0) + 1
        s["max_input_s"] = max([s["max_input_s"]] + [x["seconds"] for x in r["per_input"]])
        s["max_candidate_s"] = max(s["max_candidate_s"], r["accounting"].get("candidate_s") or 0)
    R.update(summary=summ, rows=rows, ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (OUT / "FLAKE2.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({"summary": summ, "peers": R["peers"]}, indent=1), flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:      # noqa: BLE001
        traceback.print_exc()
        (OUT / "FLAKE2_ERROR.txt").write_text(traceback.format_exc())
