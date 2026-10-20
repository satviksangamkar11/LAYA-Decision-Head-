"""[P5 Kaggle smoke] Runs tools/el_p5_run.py on Kaggle (1 GPU: P100, or 2 x T4 in parallel): (1) bundle hash check, (2) Kaggle jail rootfs + bank-40 self-check (fail closed), (3) pinned model download,
every file sha256-checked (fail closed), (4) QUALIFICATION of the jail port for the trusted verifier (tools/el_kaggle_port_qual.py; fail closed: no generation if any
check fails), (5) one process per GPU on its own task shard, all four arms on the SAME sandbox (Kaggle jail port + trusted verifier). No grader, no private test,
no reference, no key: answers are graded offline on the PC afterwards with the same grader as every other run."""
import glob, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working")
RS = {"kernel": "el_p5_run_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def finish(verdict, **kw):
    RS.update(verdict=verdict, total_seconds=round(time.time() - T0, 1), ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw)
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(RS, indent=1, default=str), encoding="utf-8")
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)


def main():
    man_p = next((p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_p5_v1"), None)
    if not man_p: finish("FAILED_NO_BUNDLE")
    bundle, man = os.path.dirname(man_p), json.load(open(man_p))
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(bundle, f)) or hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad[:5])
    RS["bundle"] = {"files": len(man["files"]), "commit": man.get("commit")}
    work = OUT / "bundle"; shutil.copytree(bundle, work, dirs_exist_ok=True)
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "kaggle"))
    plan = json.load(open(work / "p5_plan.json"))
    java = shutil.which("java")
    JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", EVIDENCE_LADDER_ALLOW_INFERENCE="1")
    try:
        import el_bank40_run_v1 as BK
        from evidence_ladder.execution import kaggle_jail as KJ, engine, config as K
        KJ.build_rootfs(JAVA_HOME)
        jail = BK.selfcheck(KJ, K, str(work))
        if jail.get("status") != "PASS": raise RuntimeError(json.dumps(jail, default=str)[:600])
        RS["jail"] = jail
    except BaseException as e:      # noqa: BLE001
        finish("JAIL_UNAVAILABLE", error=repr(e)[:600])
    q = subprocess.run([sys.executable, str(work / "tools/el_kaggle_port_qual.py"), str(OUT / "qualification")], cwd=str(work), env=dict(os.environ, PYTHONPATH=str(work)),
                       capture_output=True, text=True, timeout=1800)
    log("qualification:", q.stdout[-800:], q.stderr[-800:])
    RS["qualification"] = json.loads((OUT / "qualification/QUAL.json").read_text()) if (OUT / "qualification/QUAL.json").exists() else {"error": q.stderr[-500:]}
    if q.returncode != 0: finish("PORT_NOT_QUALIFIED", note="no generation was run: the Kaggle jail port failed a qualification check")
    md = OUT / "qwen25coder15b"
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(plan["model_repo"], revision=plan["model_revision"], local_dir=str(md))
    except BaseException as e:      # noqa: BLE001
        finish("FAILED_MODEL_DOWNLOAD", error=repr(e)[:600], note="internet must be ON for this kernel")
    mf = json.load(open(work / "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json"))
    shabad = []
    for f, m in mf["files"].items():
        if f.endswith((".safetensors", ".json")) and f != "MANIFEST_pinned.json":
            h = hashlib.sha256()
            with open(md / f, "rb") as fh:
                for ch in iter(lambda: fh.read(1 << 24), b""): h.update(ch)
            if h.hexdigest() != m["sha256"]: shabad.append(f)
    if shabad: finish("FAILED_MODEL_SHA", files=shabad)
    RS["model"] = {"repo": plan["model_repo"], "revision": plan["model_revision"], "sha256_ok": True}
    ng = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
    RS["gpus"] = ng
    if ng < 1: finish("FAILED_NO_GPU")
    gc = subprocess.run([sys.executable, "-c", "import torch; x = torch.ones(64, 64, device='cuda', dtype=torch.float16); "
                         "print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), (x @ x).sum().item())"], capture_output=True, text=True)
    RS["gpu_check"] = (gc.stdout or gc.stderr)[-400:]
    if gc.returncode != 0: finish("GPU_UNSUPPORTED", note="torch cannot run fp16 on this GPU (e.g. P100 sm_60 dropped by the image's torch build)")
    procs = []
    sequential = ng < 2                     # one GPU: shards run one after the other (never two model processes on one GPU)
    for i, sh in enumerate(plan["shards"]):
        if sequential and procs:
            procs[-1][1].wait()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % ng), PYTHONPATH=str(work))
        cmd = [sys.executable, str(work / "tools/el_p5_run.py"), str(OUT / ("p5_%s" % sh["name"])), "--cohort", sh["cohort"], "--tasks", ",".join(sh["tasks"]), "--exec", "kaggle",
               "--model-dir", str(md), "--model-manifest", str(work / "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json"), "--lcb-file", str(work / "lcb_public_smoke.jsonl")]
        lf = open(OUT / ("%s.log" % sh["name"]), "w", encoding="utf-8")
        procs.append((sh["name"], subprocess.Popen(cmd, cwd=str(work), env=env, stdout=lf, stderr=subprocess.STDOUT)))
        log("shard", sh["name"], "started:", sh["tasks"])
    pos = {n: 0 for n, _ in procs}
    while True:
        alive = any(p.poll() is None for _, p in procs)
        for n, _ in procs:
            try:
                with open(OUT / ("%s.log" % n), "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos[n]); chunk = f.read(); pos[n] = f.tell()
                for ln in chunk.splitlines():
                    if ln.strip() and "Loading weights" not in ln: log("[%s]" % n, ln[:300])
            except OSError:
                pass
        if not alive: break
        time.sleep(20)
    RS["shard_exit_codes"] = {n: p.wait() for n, p in procs}
    shutil.rmtree(md, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
    finish("OK" if all(v == 0 for v in RS["shard_exit_codes"].values()) else "PARTIAL")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:      # noqa: BLE001
        RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-3000:]; traceback.print_exc()
        finish("FAILED")
