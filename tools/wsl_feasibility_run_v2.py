"""WSL feasibility check v2 (user-approved 2026-10-14; rule results/raw/wsl-feasibility-v2-*/SELECTION_v2.md). Copy of v1 plus the disk guard,
per-task caches and cleanup.: can a SWE-rebench task give a complete, reproducible starting state in the qualified sandbox?
Runs as root in WSL Ubuntu-24.04. Isolation = the qualified runner's systemd scope + bubblewrap recipe (tools/fable_wsl_runner.py), with two mount changes
fixed in SELECTION.md: the repo at its recorded /workspace/<dir>, the environment at /opt/conda (read-only in replays and tests).
Network only in the setup step (`--share-net`); replays and tests run with every namespace unshared.

  python3 wsl_feasibility_run_v2.py <prefix_json> <out_dir>   (prefix_json from tools/wsl_feasibility_select_v2.py)
"""
import hashlib, json, os, random, re, shlex, shutil, subprocess, sys, threading, time, uuid
from pathlib import Path

sys.path.insert(0, "/mnt/d/local model/tools")
from fable_wsl_runner import ENV, TC, compare_v2, sha  # noqa: E402

BASE = Path("/var/lib/fable-runs/feas-v2")
GiB = 1 << 30
C_MIN, TASK_MAX = 4 * GiB, 2 * GiB  # disk guard (SELECTION_v2.md)
SEED = 20261014
UV = TC + "/uv-0.11.25/uv"
FOOT = re.compile(r"^\[(The command completed with exit code (-?\d+)\.|Current working directory: [^\n]*|Python interpreter: [^\n]*|Command finished with exit code (-?\d+))\]\s*$", re.M)
WRAP = 'cd "$(cat /state/cwd)" 2>/dev/null || cd /workspace\nexec 2>&1\n{ %s\n}\n__rc=$?\npwd > /state/cwd\nexit $__rc\n'
SECRET = re.compile(r"(nvapi-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.)")


def argv(mounts, cmd, net=False, timeout=300, mem="2G", chdir="/workspace"):
    a = ["systemd-run", "--scope", "--quiet", "--collect", "-p", "MemoryMax=" + mem, "-p", "MemorySwapMax=0", "-p", "TasksMax=256", "-p", "CPUQuota=400%",
         "--uid=replay", "--gid=replay", "--", "timeout", "-s", "KILL", str(timeout),
         "bwrap", "--unshare-all", *(["--share-net"] if net else []), "--die-with-parent", "--new-session", "--uid", "999", "--gid", "999",
         "--ro-bind", "/usr", "/usr", "--tmpfs", "/usr/lib/wsl", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
         "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc", "--ro-bind", TC, TC]
    if net:
        a += ["--ro-bind", "/mnt/wsl/resolv.conf", "/mnt/wsl/resolv.conf"]  # /etc/resolv.conf is a symlink there
    for kind, src, dst in mounts:
        a += [kind, str(src), dst]
    a += ["--proc", "/proc", "--dev", "/dev", "--chdir", chdir, "--clearenv", "--remount-ro", "/usr/lib/wsl", "--remount-ro", "/"]
    env = dict(ENV, PATH="/opt/conda/envs/testbed/bin:" + ENV["PATH"], UV_PYTHON_INSTALL_DIR="/opt/conda/py", UV_CACHE_DIR="/opt/conda/.uvcache", UV_LINK_MODE="hardlink",
               UV_CONCURRENT_INSTALLS="1", UV_CONCURRENT_DOWNLOADS="2",  # attempt 1 crashed the WSL VM (host commit limit) during a parallel torch install
               PIP_NO_CACHE_DIR="1")
    for k, v in env.items():
        a += ["--setenv", k, v]
    return a + ["/bin/bash", "-c", cmd]


def run(mounts, cmd, **kw):
    t0 = time.monotonic()
    p = subprocess.run(argv(mounts, cmd, **kw), capture_output=True)
    return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace"), round(time.monotonic() - t0, 1)


def c_free():
    s = os.statvfs("/mnt/c")  # the Windows C: drive that holds the WSL vhdx
    return s.f_bavail * s.f_frsize


def disk_ok(stage, out_dir, iid):
    r = os.statvfs("/")
    row = dict(t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), instance_id=iid, stage=stage, c_free_gib=round(c_free() / GiB, 3),
               wsl_root_free_gib=round(r.f_bavail * r.f_frsize / GiB, 3), source="statvfs /mnt/c (runner)")
    row["ok"] = row["c_free_gib"] * GiB >= C_MIN
    with open(out_dir / "DISK.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row["ok"]


def du(p):
    return int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0] or 0)


def run_guarded(mounts, cmd, tdir, out_dir, iid, **kw):
    """run() with the disk guard polled every 5 s: kill if C: free < 4 GiB or the task dir > 2 GiB"""
    t0 = time.monotonic()
    p = subprocess.Popen(argv(mounts, cmd, **kw), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf, peak, why = [], 0, None
    threading.Thread(target=lambda: buf.append(p.stdout.read()), daemon=True).start()
    while p.poll() is None:
        time.sleep(5)
        size, cf = du(tdir), c_free()
        peak = max(peak, size)
        if cf < C_MIN or size > TASK_MAX:
            why = "C: free %.2f GiB, task dir %.2f GiB" % (cf / GiB, size / GiB)
            p.kill()  # pid is `timeout` (systemd-run --scope execs); bwrap has --die-with-parent
            p.wait()
            break
    time.sleep(1)
    disk_ok("setup-end", out_dir, iid)
    out = (buf[0] if buf else b"").decode("utf-8", "replace")
    return p.returncode, out, round(time.monotonic() - t0, 1), dict(abort=why, peak=peak)


def abort(res, out_dir, t0, where):
    res["qualified"], res["reason"] = False, "DISK_GUARD_ABORT " + where
    return finish(res, out_dir, t0, [])


def cleanup(iid, out_dir):
    """delete only what this check created for the task, then trim the vhdx"""
    for p in [BASE / iid, *(BASE / "runs").glob("*")]:
        shutil.rmtree(p, ignore_errors=True)
    t = subprocess.run(["fstrim", "-av"], capture_output=True, text=True)
    with open(out_dir / "CLEANUP.log", "a") as f:
        f.write("# %s cleanup %s: removed %s and runs/*; fstrim exit %d\n%s%s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), iid, BASE / iid,
                                                                                t.returncode, t.stdout, t.stderr))
    disk_ok("after-cleanup", out_dir, iid)


def new_run(snapshot, d):
    """fresh copy of the setup snapshot: <run>/ws/<dir>, <run>/tmp/home, <run>/state/cwd"""
    r = BASE / "runs" / uuid.uuid4().hex
    shutil.copytree(snapshot, r / "ws" / d, symlinks=True)
    (r / "tmp" / "home").mkdir(parents=True)
    (r / "state").mkdir()
    (r / "state" / "cwd").write_text("/workspace\n")
    return r


def mounts_of(r, conda, d):
    return [("--bind", r / "ws" / d, "/workspace/" + d), ("--ro-bind", conda, "/opt/conda"), ("--bind", r / "tmp", "/tmp"), ("--bind", r / "state", "/state")]


def chown(r):
    subprocess.run(["chown", "-R", "replay:replay", str(r)], check=True)


def ws_hash(repo):
    files = {}
    for f in sorted(repo.rglob("*")):
        rel = f.relative_to(repo)
        if rel.parts[0] == ".git" or f.is_dir():
            continue
        files[str(rel)] = ("link:" + str(f.readlink())) if f.is_symlink() else sha(f.read_bytes())
    git = lambda *c: subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), *c], capture_output=True, text=True).stdout
    full = sha(json.dumps({str(f.relative_to(repo)): sha(f.read_bytes()) for f in sorted(repo.rglob("*")) if f.is_file() and not f.is_symlink()}, sort_keys=True))
    return sha(json.dumps(dict(files=files, head=git("rev-parse", "HEAD"), status=git("status", "--porcelain")), sort_keys=True)), full, len(files)


# ------------------------------------------------------------------ str_replace_editor `view` emulation (OpenHands 0.54 output format)
def host_path(p, r, conda, d):
    p = p.rstrip("/") or "/"
    if p == "/workspace" or p.startswith("/workspace/"):
        return r / "ws" / p[len("/workspace/"):] if p != "/workspace" else r / "ws"
    if p.startswith("/opt/conda/"):
        return conda / p[len("/opt/conda/"):]
    return None


def view(args, r, conda, d):
    p = args.get("path") or ""
    h = host_path(p, r, conda, d)
    if h is None:
        return None
    if h.is_dir():
        p = p.rstrip("/")
        out = [p + "/"]
        for c in h.iterdir():
            if c.name.startswith("."):
                continue
            out.append(p + "/" + c.name + ("/" if c.is_dir() else ""))
            if c.is_dir():
                out += [p + "/" + c.name + "/" + g.name + ("/" if g.is_dir() else "") for g in c.iterdir() if not g.name.startswith(".")]
        hidden = sum(c.name.startswith(".") for c in h.iterdir())
        txt = "Here's the files and directories up to 2 levels deep in %s, excluding hidden items:\n%s" % (p, "\n".join([out[0]] + sorted(out[1:])))
        if hidden:
            txt += "\n\n%d hidden files/directories in this directory are excluded. You can use 'ls -la %s' to see them." % (hidden, p)
        return txt
    if h.is_file():
        lines = h.read_text("utf-8", "replace").expandtabs().split("\n")
        start = 1
        vr = args.get("view_range")
        if vr:
            s, e = int(vr[0]), int(vr[1])
            lines = lines[s - 1:] if e == -1 else lines[s - 1:e]
            start = s
        return "Here's the result of running `cat -n` on %s:\n%s\n" % (p, "\n".join("%6d\t%s" % (start + i, l) for i, l in enumerate(lines)))
    return "ERROR:\nInvalid `path` parameter: %s. The path %s does not exist. Please provide a valid path." % (p, p)


def strip_footer(rec):
    m = [x for x in FOOT.finditer(rec or "")]
    codes = [int(x.group(2) or x.group(3)) for x in m if (x.group(2) or x.group(3))]
    return FOOT.sub("", rec or "").rstrip("\n"), (codes[0] if codes else None)


def masked_diff(a, b, n=6):
    import difflib
    return "\n".join(list(difflib.unified_diff(a.splitlines(), b.splitlines(), "recorded", "replayed", n=0, lineterm=""))[:n * 4])[:1500]


def replay(task, snapshot, conda, label):
    d = task["workspace_dir"].rstrip("/").split("/")[-1]
    r = new_run(snapshot, d)
    chown(r)
    log = []
    for st in task["prefix"]:
        name, a = st["name"], st["arguments"]
        rec, rec_exit = strip_footer(st["recorded"]) if name == "execute_bash" else (st["recorded"] or "", 0)
        row = dict(replay=label, step=st["index"], tool=name, command=(a.get("command") or "")[:300], path=a.get("path"))
        if name in ("think", "task_tracker", "finish"):
            log.append(dict(row, verdict="NO_ENV_EFFECT"))
            continue
        if name == "execute_bash":
            if a.get("is_input") in (True, "true"):
                log.append(dict(row, verdict="MISMATCH", reason="is_input not replayable"))
                break
            chown(r)
            code, out, secs = run(mounts_of(r, conda, d), WRAP % a.get("command", ""), timeout=300)
            c = compare_v2(code, out, rec_exit, rec)
            row.update(ours_exit=code, recorded_exit=rec_exit, seconds=secs)
        elif name == "str_replace_editor" and a.get("command") == "view":
            out = view(a, r, conda, d)
            if out is None:
                log.append(dict(row, verdict="MISMATCH", reason="path outside workspace"))
                break
            c = compare_v2(0, out, 0, rec)
        else:
            log.append(dict(row, verdict="MISMATCH", reason="unexpected tool in prefix"))
            break
        row.update(exact=c["exact"], signature=c["signature"], verdict="MATCH" if c["match"] else "MISMATCH")
        if not c["match"]:
            row["diff"] = masked_diff(rec, out)
        log.append(row)
        if not c["match"]:
            break
    h, full, nfiles = ws_hash(r / "ws" / d)
    shutil.rmtree(r, ignore_errors=True)
    return log, dict(replay=label, workspace_sha256=h, full_tree_sha256_info=full, files=nfiles)


def test_ids_status(out, ids):
    st = {}
    for i in ids:
        if re.search(r"^PASSED %s\s*$" % re.escape(i), out, re.M):
            st[i] = "PASSED"
        elif re.search(r"^(FAILED|ERROR) %s(\s|$)" % re.escape(i), out, re.M):
            st[i] = "FAILED"
        else:
            st[i] = "NOT_REPORTED"
    return st


def main():
    task = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(sys.argv[2])
    T = task["task"]
    iid, ic = T["instance_id"], T["install_config"]
    d = task["workspace_dir"].rstrip("/").split("/")[-1]
    tdir = BASE / iid
    if tdir.exists():
        sys.exit("work dir exists: " + str(tdir))
    t_start = time.monotonic()
    res = dict(instance_id=iid, repo=T["repo"], base_commit=T["base_commit"], trajectory_id=task["trajectory_id"], workspace_dir=task["workspace_dir"],
               prefix_steps=len(task["prefix"]), decision=dict(kind=task["decision"]["kind"], index=task["decision"]["index"]) if task["decision"] else None)
    slog = open(out_dir / ("SETUP-%s.log" % iid), "w")
    # ---------------- 1. setup (network ON, logged)
    pyver = re.search(r"^\s*-\s*python=(\d+\.\d+\.\d+)", T.get("environment") or "", re.M)
    pyver = pyver.group(1) if pyver else ic["python"]
    reqs = [l for l in (T.get("requirements") or "").splitlines() if l.strip() and not re.search(r"@ file:|^-e |git\+|^#", l)]
    (tdir / "conda").mkdir(parents=True)
    (tdir / "ws").mkdir()
    (tdir / "tmp" / "home").mkdir(parents=True)
    (tdir / "tmp" / "reqs.txt").write_text("\n".join(reqs) + "\n")
    chown(tdir)
    script = "\n".join([
        "set -ex", "cd /workspace",
        "git clone --quiet https://github.com/%s.git %s" % (T["repo"], d), "cd %s" % d,
        "git -c advice.detachedHead=false checkout --quiet %s" % T["base_commit"],
        "%s python install %s" % (UV, pyver),
        "%s venv --seed --python %s /opt/conda/envs/testbed" % (UV, pyver),
        "%s pip install --python /opt/conda/envs/testbed/bin/python -r /tmp/reqs.txt" % UV,
        ic["install"],
        *(["pip install " + " ".join(shlex.quote(x) for x in ic["pip_packages"])] if ic.get("pip_packages") else []),
        "python --version", "pip freeze > /tmp/freeze.txt", "git status --porcelain | head -20"])
    slog.write("# setup (network ON) %s python %s, %d pinned requirements\n%s\n\n" % (iid, pyver, len(reqs), script))
    if not disk_ok("setup", out_dir, iid):
        return abort(res, out_dir, t_start, "before setup")
    code, out, secs, guard = run_guarded([("--bind", tdir / "ws", "/workspace"), ("--bind", tdir / "conda", "/opt/conda"), ("--bind", tdir / "tmp", "/tmp")],
                                         script, tdir, out_dir, iid, net=True, timeout=3600, mem="3G")
    shutil.rmtree(tdir / "conda" / ".uvcache", ignore_errors=True)  # environment files stay (hardlinks); only the download cache goes
    res["setup_task_dir_bytes_peak"] = guard["peak"]
    if guard["abort"]:
        slog.write(out[-60000:] + "\n# DISK_GUARD_ABORT %s\n" % guard["abort"])
        return abort(res, out_dir, t_start, "during setup: " + guard["abort"])
    slog.write(out[-60000:] + "\n# exit %s, %.1f s\n" % (code, secs))
    slog.flush()
    freeze = (tdir / "tmp" / "freeze.txt").read_text() if (tdir / "tmp" / "freeze.txt").exists() else ""
    res["setup"] = dict(ok=code == 0, exit=code, seconds=secs, python=pyver, pinned_requirements=len(reqs), pip_freeze_sha256=sha(freeze),
                        pip_freeze_lines=len(freeze.splitlines()), python_version_out=(re.findall(r"^Python \d\S*", out, re.M) or [None])[-1])
    (out_dir / ("pip-freeze-%s.txt" % iid)).write_text(freeze)
    if code != 0:
        res["qualified"], res["reason"] = False, "setup failed (exit %s)" % code
        return finish(res, out_dir, t_start, [])
    snapshot, conda = tdir / "ws" / d, tdir / "conda"
    # network-off probe inside the replay sandbox
    r = new_run(snapshot, d)
    chown(r)
    pc, po, _ = run(mounts_of(r, conda, d), "python -c \"import socket; socket.create_connection(('1.1.1.1', 80), timeout=5)\" ; a=$?; getent hosts github.com; b=$?; "
                    "test $a -ne 0 && test $b -ne 0", timeout=60)
    shutil.rmtree(r, ignore_errors=True)
    res["network_off_probe"] = dict(socket_and_dns_failed=pc == 0, tail=po[-300:])
    # ---------------- 2. initial test condition (network OFF), twice, on base + test_patch
    f2p, p2p_all = T["FAIL_TO_PASS"], T["PASS_TO_PASS"]
    p2p = p2p_all if len(p2p_all) <= 20 else random.Random(SEED).sample(p2p_all, 20)
    tests = []
    for k in range(2):
        if not disk_ok("test-%d" % (k + 1), out_dir, iid):
            return abort(res, out_dir, t_start, "before test run %d" % (k + 1))
        r = new_run(snapshot, d)
        ap = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(r / "ws" / d), "apply", "-"], input=T["test_patch"].encode(), capture_output=True)
        chown(r)
        cmd = "cd /workspace/%s && %s %s" % (d, ic["test_cmd"], " ".join(shlex.quote(x) for x in f2p + p2p))
        code, out, secs = run(mounts_of(r, conda, d), cmd, timeout=1800, mem="4G")
        shutil.rmtree(r, ignore_errors=True)
        st = test_ids_status(out, f2p + p2p)
        tests.append(dict(run=k + 1, test_patch_applied=ap.returncode == 0, exit=code, seconds=secs,
                          f2p_failed=sum(st[i] != "PASSED" for i in f2p), f2p_total=len(f2p), f2p_not_reported=sum(st[i] == "NOT_REPORTED" for i in f2p),
                          p2p_passed=sum(st[i] == "PASSED" for i in p2p), p2p_total=len(p2p), p2p_universe=len(p2p_all), status=st, tail=out[-1500:]))
        slog.write("\n# test condition run %d (network OFF) exit %s %.1f s\n%s\n" % (k + 1, code, secs, out[-8000:]))
    res["tests"] = [{k: v for k, v in t.items() if k != "tail"} for t in tests]
    res["tests_ok"] = all(t["test_patch_applied"] and t["f2p_failed"] == t["f2p_total"] and t["f2p_not_reported"] == 0 and t["p2p_passed"] == t["p2p_total"]
                          for t in tests) and tests[0]["status"] == tests[1]["status"]
    # ---------------- 3./4. prefix replay A, then B and C from fresh copies (network OFF)
    rows, hashes = [], []
    for label in ("A", "B", "C"):
        if not disk_ok("replay-" + label, out_dir, iid):
            return abort(res, out_dir, t_start, "before replay " + label)
        log, h = replay(task, snapshot, conda, label)
        rows += log
        hashes.append(h)
    slog.close()
    n_exec = [r_ for r_ in rows if r_["replay"] == "A" and r_["verdict"] != "NO_ENV_EFFECT"]
    res["replay"] = {lab: dict(matched=sum(x["verdict"] == "MATCH" for x in rows if x["replay"] == lab),
                               mismatched=sum(x["verdict"] == "MISMATCH" for x in rows if x["replay"] == lab),
                               no_env_effect=sum(x["verdict"] == "NO_ENV_EFFECT" for x in rows if x["replay"] == lab)) for lab in "ABC"}
    res["prefix_env_steps"] = len([s for s in task["prefix"] if s["name"] not in ("think", "task_tracker", "finish")])
    res["hashes"] = hashes
    res["hash_equal_BC"] = hashes[1]["workspace_sha256"] == hashes[2]["workspace_sha256"]
    res["hash_equal_ABC"] = len({x["workspace_sha256"] for x in hashes}) == 1
    all_match = all(res["replay"][lab]["mismatched"] == 0 and res["replay"][lab]["matched"] == res["prefix_env_steps"] for lab in "ABC")
    reasons = [m for ok, m in ((res["setup"]["ok"], "setup failed"), (res["network_off_probe"]["socket_and_dns_failed"], "network probe did not fail"),
                               (res["tests_ok"], "initial FAIL_TO_PASS/PASS_TO_PASS condition not met in both runs"),
                               (all_match, "prefix replay mismatch"), (res["hash_equal_BC"], "decision-point workspace hashes differ")) if not ok]
    res["qualified"], res["reason"] = not reasons, "; ".join(reasons) or None
    return finish(res, out_dir, t_start, rows)


def finish(res, out_dir, t0, rows):
    res["wall_seconds"] = round(time.monotonic() - t0, 1)
    iid = res["instance_id"]
    with open(out_dir / "REPLAY.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(dict(instance_id=iid, **r)) + "\n")
    (out_dir / ("result-%s.json" % iid)).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("tests",)}, indent=1))


if __name__ == "__main__":
    _iid = json.loads(Path(sys.argv[1]).read_text())["instance_id"]
    try:
        main()
    finally:
        if "--keep" not in sys.argv:
            cleanup(_iid, Path(sys.argv[2]))
