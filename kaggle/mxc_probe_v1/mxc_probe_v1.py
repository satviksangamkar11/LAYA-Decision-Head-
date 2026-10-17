"""[MXC replacement] Kaggle kernel PROBE v1: can an MXC-style Linux sandbox (bubblewrap + user namespaces, the backend MXC uses on Linux) run inside a Kaggle kernel, and what does the kernel offer if it cannot?
GPU off, internet off, no dataset, no secrets. Prints only check names and PASS / FAIL / NOT_AVAILABLE; the detail strings go into the create-only receipt /kaggle/working/MXC_PROBE_RECEIPT.json
(environment variable VALUES are never read or written; only a fixed list of file/binary facts).

Statuses:  PASS = the property was observed.  FAIL = the property was tested and does NOT hold.  NOT_AVAILABLE = the check could not be run because a prerequisite is missing (the dependency is named).
Nothing is installed (no apt, no pip, no download). Everything runs as the kernel user (normally root in a seccomp-filtered container, see the 2026-10-16 FC-1 probe)."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
RECEIPT = OUT_DIR / "MXC_PROBE_RECEIPT.json"
CHECKS = []
CLONE_NEWUSER, CLONE_NEWNET, CLONE_NEWPID, CLONE_NEWNS = 0x10000000, 0x40000000, 0x20000000, 0x00020000


def rec(name, status, detail=""):
    CHECKS.append({"check": name, "status": status, "detail": str(detail)[:600]})
    print("%-44s %s" % (name, status), flush=True)


def sh(argv, timeout=20, **kw):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)
        return p.returncode, p.stdout[-400:], p.stderr[-400:]
    except subprocess.TimeoutExpired:
        return "timeout", "", ""
    except OSError as e:
        return "oserror", "", str(e)[:200]


def read(p):
    try:
        return Path(p).read_text().strip()[:120]
    except OSError as e:
        return "ERR " + type(e).__name__


def raw_unshare(flags):
    """unshare(2) in a forked child so the probe process keeps its namespaces. -> (ok, errno)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.unshare(flags)
        os.write(w, (b"0" if rc == 0 else b"e%d" % ctypes.get_errno()))
        os._exit(0)
    os.close(w)
    data = os.read(r, 32).decode()
    os.close(r)
    os.waitpid(pid, 0)
    return data == "0", (None if data == "0" else data)


def main():
    t0 = time.time()
    facts = {"python": sys.version.split()[0], "platform": sys.platform, "uname": list(os.uname()), "uid": os.getuid(), "gid": os.getgid(),
             "status": {k: v for k, v in (l.split(":\t", 1) for l in Path("/proc/self/status").read_text().splitlines() if l.split(":")[0] in ("CapEff", "CapBnd", "NoNewPrivs", "Seccomp", "Seccomp_filters"))},
             "userns_clone": read("/proc/sys/kernel/unprivileged_userns_clone"), "max_user_namespaces": read("/proc/sys/user/max_user_namespaces"),
             "apparmor_restrict_userns": read("/proc/sys/kernel/apparmor_restrict_unprivileged_userns"), "cgroup": read("/proc/self/cgroup")}
    which = {n: shutil.which(n) for n in ("bwrap", "unshare", "nsenter", "setpriv", "runuser", "chroot", "prlimit", "timeout", "node", "npm", "python3", "slirp4netns", "firejail", "nsjail", "apt-get")}
    facts["which"] = which

    # ---- 1. kernel namespace support (raw syscalls, no bwrap needed)
    for name, flag in (("user", CLONE_NEWUSER), ("net", CLONE_NEWNET), ("pid", CLONE_NEWPID), ("mount", CLONE_NEWNS)):
        ok, err = raw_unshare(flag)
        rec("unshare_syscall_" + name, "PASS" if ok else "FAIL", "unshare(CLONE_NEW%s) %s" % (name.upper(), "succeeded" if ok else "errno " + str(err)))
    if which["unshare"]:
        rc, o, e = sh([which["unshare"], "-Ur", "true"])
        rec("unshare_cli_user_map_root", "PASS" if rc == 0 else "FAIL", "rc=%s %s" % (rc, e.strip()[:160]))
    else:
        rec("unshare_cli_user_map_root", "NOT_AVAILABLE", "unshare binary missing")

    # ---- 2. bubblewrap (the binary MXC's Linux backend shells out to; MXC does NOT bundle it)
    bw = which["bwrap"]
    rec("bwrap_binary_present", "PASS" if bw else "FAIL", bw or "bwrap not on PATH (MXC requires bwrap >= 0.5.0 on PATH; nothing can be installed: no network, no apt)")
    if bw:
        rc, o, e = sh([bw, "--version"])
        rec("bwrap_version", "PASS" if rc == 0 else "FAIL", "%s %s" % (o.strip(), e.strip()[:120]))
    base = [bw, "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"] if bw else None
    work = Path(tempfile.mkdtemp(prefix="mxcprobe-"))
    (work / "ws").mkdir()
    (work / "ws" / "prot.txt").write_text("protected\n")
    (work / "secret_outside.txt").write_text("outside\n")
    bwrap_runs = False
    if base:
        rc, o, e = sh(base + ["--bind", str(work / "ws"), "/work", "--chdir", "/work", "/bin/sh", "-c", "echo sandbox_ok"])
        bwrap_runs = rc == 0 and "sandbox_ok" in o
        rec("bwrap_unshare_all_trivial_command", "PASS" if bwrap_runs else "FAIL", "rc=%s out=%s err=%s" % (rc, o.strip()[:60], e.strip()[:200]))
    else:
        rec("bwrap_unshare_all_trivial_command", "NOT_AVAILABLE", "needs bwrap")
    dep = "needs a working bwrap --unshare-all" if not bwrap_runs else ""
    if bwrap_runs:
        py = which["python3"] or "python3"
        rc, o, e = sh(base + ["--bind", str(work / "ws"), "/work", "--chdir", "/work", py, "-c", "import socket,sys\ntry:\n socket.create_connection(('1.1.1.1',53),timeout=3);print('CONNECTED')\nexcept Exception as e:\n print('BLOCKED',type(e).__name__)"])
        rec("bwrap_network_blocked", "PASS" if "BLOCKED" in o else "FAIL", o.strip()[:120] or e[:120])
        rc, o, e = sh(base + ["--bind", str(work / "ws"), "/work", "--ro-bind", str(work / "ws" / "prot.txt"), "/work/prot.txt", "--chdir", "/work", "/bin/sh", "-c", "echo x > prot.txt 2>/dev/null && echo WROTE || echo DENIED; echo y > free.txt && echo free_ok"])
        rec("bwrap_protected_file_readonly", "PASS" if "DENIED" in o and "free_ok" in o and (work / "ws" / "prot.txt").read_text() == "protected\n" else "FAIL", o.strip()[:120])
        t = time.time()
        rc, o, e = sh(["timeout", "-s", "KILL", "2"] + base + ["--bind", str(work / "ws"), "/work", "/bin/sh", "-c", "sleep 30"], timeout=20)
        rec("bwrap_timeout_kills_tree", "PASS" if rc in (137, -9) and time.time() - t < 10 else "FAIL", "rc=%s after %.1fs" % (rc, time.time() - t))
        (work / "ws" / "link").symlink_to("/etc/hostname")
        rc, o, e = sh(base + ["--bind", str(work / "ws"), "/work", "--chdir", "/work", "/bin/sh", "-c", "cat ../../etc/shadow 2>&1 | head -c 80; echo; cat ../../tmp/../" + str(work).lstrip("/") + "/secret_outside.txt 2>&1 | head -c 80"])
        rec("bwrap_path_traversal_blocked", "PASS" if "outside" not in o and "root:" not in o else "FAIL", o.strip()[:160])
    else:
        for n in ("bwrap_network_blocked", "bwrap_protected_file_readonly", "bwrap_timeout_kills_tree", "bwrap_path_traversal_blocked"):
            rec(n, "NOT_AVAILABLE", dep)

    # ---- 3. node (the MXC Node SDK needs >= 24; the bridge cannot run without it)
    if which["node"]:
        rc, o, e = sh([which["node"], "--version"])
        rec("node_available", "PASS" if rc == 0 else "FAIL", o.strip())
        try:
            major = int(o.strip().lstrip("v").split(".")[0])
        except ValueError:
            major = 0
        rec("node_version_ge_24", "PASS" if major >= 24 else "FAIL", o.strip())
    else:
        rec("node_available", "NOT_AVAILABLE", "node not on PATH (an MXC Node SDK would have to be vendored together with a node binary in a dataset)")
        rec("node_version_ge_24", "NOT_AVAILABLE", "no node")

    # ---- 4. what the kernel offers WITHOUT namespaces (the already-built fallback jail: chroot + uid drop + seccomp + rlimits; evidence_ladder/execution/kaggle_jail.py)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        res = []
        try:
            d = tempfile.mkdtemp(prefix="mxcchroot-")
            os.chroot(d)
            os.chdir("/")
            res.append("chroot_ok")
        except OSError as e:
            res.append("chroot_" + type(e).__name__)
        try:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
            res.append("setuid_nobody_ok" if os.getuid() == 65534 else "setuid_noop")
        except OSError as e:
            res.append("setuid_" + type(e).__name__)
        os.write(w, ",".join(res).encode())
        os._exit(0)
    os.close(w)
    got = os.read(r, 200).decode()
    os.close(r)
    os.waitpid(pid, 0)
    rec("fallback_chroot_available", "PASS" if "chroot_ok" in got else "FAIL", got)
    rec("fallback_uid_drop_available", "PASS" if "setuid_nobody_ok" in got else "FAIL", got)
    try:
        s = socket.create_connection(("1.1.1.1", 53), timeout=3)
        s.close()
        rec("kernel_internet_off", "FAIL", "outbound TCP connected (internet is ON for this kernel)")
    except OSError as e:
        rec("kernel_internet_off", "PASS", "connect failed: %s" % type(e).__name__)
    t = time.time()
    p = subprocess.Popen([sys.executable, "-c", "import time,subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);time.sleep(60)"], start_new_session=True)
    time.sleep(1)
    os.killpg(p.pid, signal.SIGKILL)
    p.wait(timeout=10)
    rec("process_group_kill", "PASS", "killpg reaped the tree in %.1fs (no isolation, only cleanup)" % (time.time() - t))

    # ---- 5. provider-level guards that do not depend on the kernel (the Linux provider reuses them)
    d = Path(tempfile.mkdtemp(prefix="mxcguard-"))
    (d / "real.txt").write_text("a")
    (d / "evil").symlink_to("/etc/hostname")
    found = []
    for dp, dns, fns in os.walk(d, followlinks=False):
        for n in dns + fns:
            if os.path.islink(os.path.join(dp, n)):
                found.append(n)
    rec("guard_symlink_detected_before_copy", "PASS" if found == ["evil"] else "FAIL", found)

    # ---- verdict
    names = {c["check"]: c["status"] for c in CHECKS}
    usable = names.get("bwrap_unshare_all_trivial_command") == "PASS" and names.get("node_version_ge_24") == "PASS"
    verdict = {"mxc_linux_bubblewrap_usable": usable,
               "reason": "bwrap runs a trivial --unshare-all command AND node >= 24 exists" if usable else
               "unusable: " + ", ".join("%s=%s" % (k, v) for k, v in names.items() if k in ("bwrap_binary_present", "bwrap_unshare_all_trivial_command", "unshare_syscall_user", "node_available") and v != "PASS"),
               "if_unusable": "the pipeline must FAIL CLOSED (no unsandboxed fallback). Available substitute: the repo's chroot + uid-drop + seccomp jail (evidence_ladder/execution/kaggle_jail.py), documented as weaker (no pid/net/ipc/uts namespaces)."}
    out = {"probe": "mxc_probe_v1", "seconds": round(time.time() - t0, 1), "facts": facts, "checks": CHECKS, "verdict": verdict}
    with open(RECEIPT, "x", encoding="utf-8") as f:       # create-only
        json.dump(out, f, indent=1, ensure_ascii=False)
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)
    print("VERDICT mxc_linux_bubblewrap_usable =", usable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
