"""[P5 Step 2] Contract-route qualification ON KAGGLE: bundle hash check, Kaggle jail rootfs, jail-port qualification (fail closed), then
tools/el_p5_contract_qual.py --exec kaggle (scripted model, synthetic tasks Q01-Q04 only; no generation, no GPU, no internet). Output: /kaggle/working/qual/QUAL.json"""
import glob, hashlib, json, os, shutil, subprocess, sys, traceback
from pathlib import Path

OUT = Path("/kaggle/working")


def main():
    man_p = next(p for p in glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True) if json.load(open(p)).get("bundle") == "el_p5_v1")
    bundle, man = os.path.dirname(man_p), json.load(open(man_p))
    bad = [f for f, h in man["files"].items() if hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]
    if bad:
        print("BUNDLE_HASH_FAIL", bad[:5]); return
    print("bundle commit", man.get("commit"), flush=True)
    work = OUT / "bundle"; shutil.copytree(bundle, work, dirs_exist_ok=True)
    java = shutil.which("java")
    JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    env = dict(os.environ, EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(work))
    sys.path.insert(0, str(work))
    from evidence_ladder.execution import kaggle_jail as KJ
    KJ.build_rootfs(JAVA_HOME)
    q = subprocess.run([sys.executable, str(work / "tools/el_kaggle_port_qual.py"), str(OUT / "port_qualification")], cwd=str(work), env=env, capture_output=True, text=True, timeout=1800)
    print("port qualification:", q.stdout[-600:], q.stderr[-600:], flush=True)
    if q.returncode != 0:
        print("PORT_NOT_QUALIFIED: contract qualification not run"); return
    r = subprocess.run([sys.executable, str(work / "tools/el_p5_contract_qual.py"), str(OUT / "qual"), "--exec", "kaggle"], cwd=str(work), env=env, capture_output=True, text=True, timeout=3600)
    print(r.stdout[-6000:], r.stderr[-3000:], "exit", r.returncode, flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:      # noqa: BLE001
        traceback.print_exc()
        (OUT / "ERROR.txt").write_text(traceback.format_exc())
