"""[H2-lite] Kaggle GPU kernel: (1) bundle hash check, (2) Kaggle jail rootfs + self-check (fail closed), (3) jail-port qualification (fail closed),
(4) pinned model download, every file sha256-checked, (5) GPU fp16 check, (6) tools/el_h2lite_run.py for the phase in h2_plan.json (one process, one GPU,
configurations one after another, one jail user). No grader, no hidden case, no key. The development grade is computed offline on the PC afterwards."""
import glob, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

T0 = time.time()
OUT = Path("/kaggle/working")
RS = {"kernel": "el_h2lite_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def finish(verdict, **kw):
    RS.update(verdict=verdict, total_seconds=round(time.time() - T0, 1), ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw)
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(RS, indent=1, default=str), encoding="utf-8")
    for d in ("qwen25coder15b", "bundle"):
        shutil.rmtree(OUT / d, ignore_errors=True)
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)


def main():
    man_p = next((p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_h2lite_v1"), None)
    if not man_p: finish("FAILED_NO_BUNDLE")
    bundle, man = os.path.dirname(man_p), json.load(open(man_p))
    bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(bundle, f)) or hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad[:5])
    RS["bundle"] = {"files": len(man["files"]), "commit": man.get("commit")}
    work = OUT / "bundle"; shutil.copytree(bundle, work, dirs_exist_ok=True)
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "kaggle"))
    plan = json.load(open(work / "h2_plan.json")); RS["plan"] = plan
    java = shutil.which("java"); JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", EVIDENCE_LADDER_ALLOW_INFERENCE="1")
    try:
        import el_bank40_run_v1 as BK
        from evidence_ladder.execution import kaggle_jail as KJ, config as K
        KJ.build_rootfs(JAVA_HOME)
        jail = BK.selfcheck(KJ, K, str(work))
        if jail.get("status") != "PASS": raise RuntimeError(json.dumps(jail, default=str)[:600])
        RS["jail"] = jail.get("status")
    except BaseException as e:      # noqa: BLE001
        finish("JAIL_UNAVAILABLE", error=repr(e)[:600])
    q = subprocess.run([sys.executable, str(work / "tools/el_kaggle_port_qual.py"), str(OUT / "qualification")], cwd=str(work), env=dict(os.environ, PYTHONPATH=str(work)),
                       capture_output=True, text=True, timeout=1800)
    RS["qualification_ok"] = q.returncode == 0
    if q.returncode != 0: finish("PORT_NOT_QUALIFIED", detail=(q.stdout + q.stderr)[-800:])
    md = OUT / "qwen25coder15b"
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(plan["model_repo"], revision=plan["model_revision"], local_dir=str(md))
    except BaseException as e:      # noqa: BLE001
        finish("FAILED_MODEL_DOWNLOAD", error=repr(e)[:600])
    mf = json.load(open(work / "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json"))
    shabad = []
    for f, m in mf["files"].items():
        if f.endswith((".safetensors", ".json")) and f != "MANIFEST_pinned.json":
            h = hashlib.sha256()
            with open(md / f, "rb") as fh:
                for ch in iter(lambda: fh.read(1 << 24), b""): h.update(ch)
            if h.hexdigest() != m["sha256"]: shabad.append(f)
    if shabad: finish("FAILED_MODEL_SHA", files=shabad)
    RS["model_sha256_ok"] = True
    gc = subprocess.run([sys.executable, "-c", "import torch; x = torch.ones(8, 8, device='cuda', dtype=torch.float16); print(torch.cuda.get_device_name(0), float((x @ x).sum()))"],
                        capture_output=True, text=True)
    RS["gpu_check"] = (gc.stdout or gc.stderr)[-300:]
    if gc.returncode != 0: finish("GPU_UNAVAILABLE")
    cmd = [sys.executable, str(work / "tools/el_h2lite_run.py"), str(OUT / ("h2_" + plan["phase"])), "--phase", plan["phase"], "--configs", ",".join(plan["configs"]),
           "--proposals", ",".join(str(work / p) for p in plan["proposals"]), "--exec", "kaggle", "--model-dir", str(md),
           "--model-manifest", str(work / "models/qwen2.5-coder-1.5b-instruct/MANIFEST_pinned.json")]
    with open(OUT / "run.log", "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, cwd=str(work), env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH=str(work)), stdout=lf, stderr=subprocess.STDOUT)
        pos = 0
        while p.poll() is None:
            time.sleep(30)
            with open(OUT / "run.log", encoding="utf-8", errors="replace") as f:
                f.seek(pos); chunk = f.read(); pos = f.tell()
            for ln in chunk.splitlines():
                if ln.strip() and "Loading weights" not in ln: log(ln[:300])
    RS["exit_code"] = p.returncode
    finish("OK" if p.returncode == 0 else "FAILED_RUN")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:      # noqa: BLE001
        RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-3000:]; traceback.print_exc()
        finish("FAILED")
