"""[P2 Kaggle replicate] Runs tools/el_p2_generation_compare.py on Kaggle with 2 x T4: one process per GPU (CUDA_VISIBLE_DEVICES), each on its own task shard,
candidates executed in the Kaggle jail (built and self-checked first; fail closed, no unsandboxed fallback). The pinned model revision is downloaded inside Kaggle and every
file is sha256-checked against MANIFEST_pinned.json (mismatch = fail closed). No grader, no hidden cases, no keys: grading happens offline on the PC afterwards."""
import glob, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working")
RS = {"kernel": "el_p2_run_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def finish(verdict, **kw):
    RS.update(verdict=verdict, total_seconds=round(time.time() - T0, 1), ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw)
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(RS, indent=1, default=str), encoding="utf-8")
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)


def main():
    man_p = next((p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_p2_v1"), None)
    if not man_p: finish("FAILED_NO_BUNDLE")
    bundle, man = os.path.dirname(man_p), json.load(open(man_p))
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(bundle, f)) or hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad[:5])
    RS["bundle"] = {"files": len(man["files"]), "commit": man.get("commit")}
    work = OUT / "bundle"; shutil.copytree(bundle, work, dirs_exist_ok=True)       # writable copy (the harness resolves paths relative to it)
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "kaggle"))
    plan = json.load(open(work / "p2_plan.json"))
    # ---- the jail (as the bank-40 kernel: rootfs + self-check; fail closed)
    java = shutil.which("java")
    JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", EVIDENCE_LADDER_ALLOW_INFERENCE="1")
    try:
        import el_bank40_run_v1 as BK
        from evidence_ladder.execution import kaggle_jail as KJ, engine, config as K
        KJ.build_rootfs(JAVA_HOME)
        jail = BK.selfcheck(KJ, K, str(work))
        engine.py_mod(); engine.java_mod()
        jail["bwrap_swapped"] = bool(engine.py_mod().bwrap_cmd is KJ.bwrap_cmd)
        if jail.get("status") != "PASS" or not jail["bwrap_swapped"]: raise RuntimeError(json.dumps(jail, default=str)[:600])
        RS["jail"] = jail
    except BaseException as e:      # noqa: BLE001
        finish("JAIL_UNAVAILABLE", error=repr(e)[:600])
    # ---- model: pinned revision, every file sha256-checked
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
    runs = dict(plan["tasks"] and [(t, r) for r, t in plan["tasks"]])
    shards = plan["shards"] if ng >= 2 else [sum(plan["shards"], [])]
    procs = []
    for i, sh in enumerate(shards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONPATH=str(work))
        cmd = [sys.executable, str(work / "tools/el_p2_generation_compare.py"), str(OUT / ("p2_shard%d" % i)), "--tasks"] + ["%s:%s" % (runs[t], t) for t in sh] + \
              ["--budget", str(plan["budget"]), "--round-n", str(plan["round_n"]), "--profile", plan["profile"], "--task-records", str(work / "p2_tasks_public.jsonl"),
               "--exec", "kaggle", "--model-dir", str(md), "--model-manifest", str(work / "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json")]
        lf = open(OUT / ("shard%d.log" % i), "w", encoding="utf-8")
        procs.append((i, subprocess.Popen(cmd, cwd=str(work), env=env, stdout=lf, stderr=subprocess.STDOUT)))
        log("shard", i, "started:", sh)
    pos = {i: 0 for i, _ in procs}
    while True:
        alive = any(p.poll() is None for _, p in procs)
        for i, _ in procs:
            try:
                with open(OUT / ("shard%d.log" % i), "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos[i]); chunk = f.read(); pos[i] = f.tell()
                for ln in chunk.splitlines():
                    if ln.strip(): log("[s%d]" % i, ln[:300])
            except OSError:
                pass
        if not alive: break
        time.sleep(20)
    rcs = {i: p.wait() for i, p in procs}
    RS["shard_exit_codes"] = rcs
    shutil.rmtree(md, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)       # outputs only (no weights, no bundle copy)
    finish("OK" if all(v == 0 for v in rcs.values()) else "PARTIAL")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:      # noqa: BLE001
        RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-3000:]; traceback.print_exc()
        finish("FAILED")
