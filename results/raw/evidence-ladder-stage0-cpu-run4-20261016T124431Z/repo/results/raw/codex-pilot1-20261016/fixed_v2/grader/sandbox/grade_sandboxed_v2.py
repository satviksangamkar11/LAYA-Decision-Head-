"""v2 sandboxed grader (Linux/WSL). Candidate code runs inside bubblewrap: new user/pid/net/ipc/uts namespaces, unprivileged uid, read-only /usr, empty tmpfs /tmp,
NO host paths mounted (the protected cases are read only by this trusted parent and reach the candidate one input at a time on stdin), no network, resource limits,
wall timeout with whole-process-group kill. Strict, type-sensitive comparison. Candidate hash is taken BEFORE execution.
Usage: python3 grade_sandboxed_v2.py --solutions DIR --arm A --out report.json [--cases protected_cases.json] [--timeout 10]"""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner_v2.py"


def strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(strict_equal(a[k], b[k]) for k in a)
    return a == b


def bwrap_cmd(cand, runner):
    pre = ["runuser", "-u", "nobody", "--"] if os.getuid() == 0 else []  # unprivileged real uid, so RLIMIT_NPROC is enforced
    return pre + ["bwrap", "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-net", "--unshare-uts", "--unshare-cgroup-try",
            "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
            "--ro-bind", str(runner), "/jail/runner.py", "--ro-bind", str(cand), "/jail/cand.py",
            "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "PYTHONHASHSEED", "0",
            "/usr/bin/python3", "-I", "-S", "/jail/runner.py", "/jail/cand.py"]


def grade_one(file, data, timeout):
    fail, answers, stderr = None, [], ""
    if not file.exists():
        return "MISSING_SUBMISSION", 0, None, ""
    raw_bytes = file.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()  # before execution
    if len(raw_bytes) > 131072:
        return "OVERSIZED_SUBMISSION", 0, sha, ""
    stdin = "".join(json.dumps(c["input"], ensure_ascii=False) + "\n" for c in data)
    jd = Path(tempfile.mkdtemp(prefix="jail-")); os.chmod(jd, 0o755)
    cand = jd / "cand.py"; cand.write_bytes(raw_bytes); os.chmod(cand, 0o644)
    runner = jd / "runner.py"; shutil.copyfile(RUNNER, runner); os.chmod(runner, 0o644)
    p = subprocess.Popen(bwrap_cmd(cand, runner), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", start_new_session=True)
    try:
        out, err = p.communicate(stdin, timeout=timeout)
        stderr = err[-2000:]
        if p.returncode != 0:
            fail = "PROCESS_ERROR"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        stderr = (err or "")[-2000:]
        fail = "TIMEOUT"
    shutil.rmtree(jd, ignore_errors=True)
    ok = 0
    if fail is None:
        for line in out.splitlines():
            try:
                v = json.loads(line)
            except ValueError:
                fail = "INVALID_OUTPUT"; break
            if not isinstance(v, dict):
                fail = "INVALID_OUTPUT"; break
            answers.append(v)
        if fail is None and len(answers) != len(data):
            fail = "MISSING_OUTPUT"
    if fail is None:
        for exp, got in zip(data, answers):
            if got.get("ok") is True and strict_equal(got.get("value"), exp["expected"]):
                ok += 1
        if ok != len(data):
            fail = "WRONG_ANSWER_OR_RUNTIME_ERROR"
    return fail, ok, sha, stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", type=Path, required=True)
    ap.add_argument("--arm", default="unknown")
    ap.add_argument("--out", type=Path, default=Path("result.json"))
    ap.add_argument("--cases", type=Path, default=HERE.parent / "protected_cases.json")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default all)")
    a = ap.parse_args()
    cases = json.loads(a.cases.read_text(encoding="utf-8"))
    want = set(a.tasks.split(",")) if a.tasks else None
    results, t0 = [], time.monotonic()
    for t in cases:
        if want and t["task_id"] not in want:
            continue
        fail, ok, sha, err = grade_one(a.solutions / (t["task_id"] + ".py"), t["cases"], a.timeout)
        results.append({"task_id": t["task_id"], "passed": fail is None, "passed_tests": ok, "total_tests": len(t["cases"]), "failure": fail, "sha256_before_run": sha, "stderr_tail": err})
    out = {"arm": a.arm, "status": "SANDBOXED_BWRAP", "tasks": len(results), "solved": sum(r["passed"] for r in results),
           "solve_rate": sum(r["passed"] for r in results) / max(1, len(results)), "duration_seconds": round(time.monotonic() - t0, 2), "results": results,
           "isolation": "bubblewrap: user/pid/net/ipc/uts namespaces, uid 65534, ro /usr, tmpfs /tmp, no host mounts, rlimits, group kill"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"arm": a.arm, "solved": out["solved"], "tasks": out["tasks"]}))


if __name__ == "__main__":
    main()
