"""Sandboxed Java grader (Linux/WSL). Both javac and java run inside bubblewrap as an unprivileged user: new user/pid/net/ipc/uts namespaces, ro /usr and ro JDK,
rw only the per-task work directory (it holds just Solution.java + Json.java + Runner.java), empty tmpfs /tmp, no host mounts, no network, prlimit on processes/file size/CPU,
wall timeout with whole-group kill. Protected cases are read only by this trusted parent and reach the candidate one input per line on stdin.
Usage: python3 grade_sandboxed_java_v1.py --solutions DIR --arm A --out report.json [--kit KIT_DIR] [--tasks J1-01,...] [--extra-source FILE ...] [--timeout 10]"""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path

JDK = "/opt/toolchains/jdk-26.0.1"


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    return a == b


def jail(workdir, cmd, rw=True):
    pre = ["runuser", "-u", "nobody", "--"] if os.getuid() == 0 else []
    return pre + ["bwrap", "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-net", "--unshare-uts", "--unshare-cgroup-try", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                  "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin", "--ro-bind", JDK, JDK,
                  "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind" if rw else "--ro-bind", str(workdir), "/work", "--chdir", "/work",
                  "--clearenv", "--setenv", "PATH", "/usr/bin:" + JDK + "/bin", "--setenv", "JAVA_HOME", JDK, "--setenv", "LC_ALL", "C.UTF-8"] + cmd


def run(cmd, stdin, timeout):
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", start_new_session=True)
    try:
        out, err = p.communicate(stdin, timeout=timeout)
        return p.returncode, out, err, False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return -9, out or "", err or "", True


def grade_one(kit, task_id, source, cases, extra, timeout):
    if not source.is_file():
        return {"status": "MISSING", "passed": 0, "total": len(cases)}
    raw = source.read_bytes(); sha = hashlib.sha256(raw).hexdigest()  # before execution
    w = Path(tempfile.mkdtemp(prefix="jjail-")); os.chmod(w, 0o777)
    try:
        (w / "Solution.java").write_bytes(raw)
        for f in ("Json.java", "Runner.java"):
            shutil.copyfile(kit / "worker" / "support" / f, w / f)
        for e in extra:
            shutil.copyfile(e, w / Path(e).name)
        for f in w.iterdir():
            os.chmod(f, 0o644)
        rc, out, err, to = run(jail(w, ["/usr/bin/prlimit", "--nproc=256", "--fsize=4194304", "--cpu=40", JDK + "/bin/javac", "-proc:none", "-encoding", "UTF-8", "-d", "/work",
                                        "/work/Json.java", "/work/Runner.java", "/work/Solution.java"] + ["/work/" + Path(e).name for e in extra]), "", 60)
        if to or rc != 0:
            return {"status": "COMPILE_TIMEOUT" if to else "COMPILE_ERROR", "passed": 0, "total": len(cases), "sha256_before_run": sha, "error": err[-600:]}
        ins = "\n".join(json.dumps(c["input"], ensure_ascii=False, separators=(",", ":")) for c in cases) + "\n"
        rc, out, err, to = run(jail(w, ["/usr/bin/prlimit", "--nproc=256", "--fsize=1048576", "--nofile=256", "--cpu=30", JDK + "/bin/java", "-Xmx128m", "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-Dstdin.encoding=UTF-8", "-cp", "/work", "Runner"]), ins, timeout)
        if to:
            return {"status": "RUNTIME_TIMEOUT", "passed": 0, "total": len(cases), "sha256_before_run": sha, "stderr_tail": err[-1500:]}
        lines = out.splitlines()
        if rc != 0 or len(lines) != len(cases):
            return {"status": "RUNTIME_ERROR", "passed": 0, "total": len(cases), "sha256_before_run": sha, "stderr_tail": err[-1500:], "outputs_provided": len(lines)}
        passed = 0
        for line, c in zip(lines, cases):
            try:
                passed += same(json.loads(line), c["expected"])
            except ValueError:
                pass
        return {"status": "PASS" if passed == len(cases) else "FAIL", "passed": passed, "total": len(cases), "sha256_before_run": sha, "stderr_tail": err[-1500:]}
    finally:
        shutil.rmtree(w, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", type=Path, required=True)
    ap.add_argument("--arm", default="unknown")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kit", type=Path, default=Path(__file__).resolve().parents[2] / "kit" / "Pilot1_Java")
    ap.add_argument("--cases", type=Path, default=None)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--extra-source", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args()
    cases = json.loads((a.cases or a.kit / "grader" / "protected_cases.json").read_text(encoding="utf-8"))
    want = set(a.tasks.split(",")) if a.tasks else None
    res, t0 = [], time.monotonic()
    for t in cases:
        if want and t["id"] not in want:
            continue
        r = grade_one(a.kit, t["id"], a.solutions / (t["id"] + ".java"), t["cases"], a.extra_source, a.timeout)
        r["task_id"] = t["id"]; res.append(r)
    out = {"arm": a.arm, "status": "SANDBOXED_BWRAP_JAVA", "tasks": len(res), "solved": sum(r["status"] == "PASS" for r in res), "duration_seconds": round(time.monotonic() - t0, 1), "results": res}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"arm": a.arm, "solved": out["solved"], "tasks": out["tasks"], "seconds": out["duration_seconds"]}))


if __name__ == "__main__":
    main()
