"""Stage 0 FC-1 probe (no model code, no candidate code, no secrets, no internet). Records which isolation primitives exist in a Kaggle kernel."""
import ctypes, json, os, platform, resource, shutil, socket, subprocess, sys
R = {"python": sys.version, "platform": platform.platform(), "uname": list(os.uname()), "uid": os.getuid(), "gid": os.getgid()}
def sh(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout[-1500:], "err": p.stderr[-800:]}
    except Exception as e:
        return {"exc": repr(e)}
R["id"] = sh("id")
R["status"] = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in open("/proc/self/status") if l.split(":")[0] in ("CapEff", "CapBnd", "NoNewPrivs", "Seccomp", "Seccomp_filters")}
R["which"] = {b: shutil.which(b) for b in ("bwrap", "unshare", "nsenter", "runuser", "su", "setpriv", "chroot", "prlimit", "firejail", "nsjail", "apt-get", "sudo", "nvidia-smi", "java", "javac", "unshare", "mount", "python3")}
for f in ("/proc/sys/kernel/unprivileged_userns_clone", "/proc/sys/user/max_user_namespaces", "/proc/sys/kernel/apparmor_restrict_unprivileged_userns", "/proc/version", "/proc/self/cgroup"):
    try: R[f] = open(f).read()[:300]
    except OSError as e: R[f] = "ERR " + type(e).__name__
R["dmesg_gvisor"] = sh("dmesg | head -5")
R["mounts"] = open("/proc/self/mounts").read()[:3000]
R["unshare_user"] = sh("unshare -U -r id")
R["unshare_user_net"] = sh("unshare -U -r -n sh -c 'id; ip link 2>/dev/null || cat /proc/net/dev'")
R["unshare_pid"] = sh("unshare -p -f --mount-proc sh -c 'ps aux | head -3'")
R["unshare_all"] = sh("unshare -U -r -m -n -p -i -u -f --mount-proc sh -c 'id; mount -t tmpfs none /mnt && echo tmpfs_ok'")
R["unshare_user_nonroot"] = sh("setpriv --reuid=65534 --regid=65534 --clear-groups unshare -U -n id")
R["unshare_root_net"] = sh("unshare -n sh -c 'cat /proc/net/dev'")
R["unshare_root_pid_mount"] = sh("unshare -m -p -f --mount-proc sh -c 'echo $$; ls /proc | head -5; mount -t tmpfs none /mnt && echo tmpfs_ok'")
R["chroot_test"] = sh("mkdir -p /tmp/cr/bin && cp /bin/true /tmp/cr/bin/ 2>&1; chroot /tmp/cr /bin/true; echo rc=$?")
R["setuid_nobody"] = sh([sys.executable, "-c", "import os;os.setgid(65534);os.setuid(65534);print(os.getuid())"])
R["runuser_nobody"] = sh("runuser -u nobody -- id")
R["setpriv_nobody"] = sh("setpriv --reuid=65534 --regid=65534 --clear-groups id")
R["rlimit_nproc_existing"] = resource.getrlimit(resource.RLIMIT_NPROC)
R["rlimits"] = {n: resource.getrlimit(getattr(resource, n)) for n in ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_NPROC")}
# seccomp via prctl: can we install a filter (needs NO_NEW_PRIVS); only check that prctl works
libc = ctypes.CDLL(None, use_errno=True)
R["prctl_no_new_privs"] = libc.prctl(38, 1, 0, 0, 0), ctypes.get_errno()
R["unshare_syscall_in_child"] = None
pid = os.fork()
if pid == 0:
    r = libc.unshare(0x10000000 | 0x40000000)  # CLONE_NEWUSER | CLONE_NEWNET
    os._exit(0 if r == 0 else (ctypes.get_errno() or 255))
_, st = os.waitpid(pid, 0); R["unshare_syscall_in_child"] = {"exit": os.waitstatus_to_exitcode(st)}
try: socket.create_connection(("1.1.1.1", 80), timeout=3); R["internet"] = True
except OSError as e: R["internet"] = "blocked: " + type(e).__name__
R["listdir_input"] = sh("ls -la /kaggle /kaggle/input /kaggle/working | head -40")
R["apt_cache"] = sh("ls /var/cache/apt/archives | head; apt-cache policy bubblewrap 2>&1 | head -4")
R["find_bwrap"] = sh("find / -xdev \( -name 'bwrap' -o -name 'nsjail' -o -name 'bubblewrap*' \) 2>/dev/null | head")
R["cpu"] = {"nproc": os.cpu_count(), "mem": open("/proc/meminfo").read().splitlines()[:2]}
R["nvidia"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | head -3")
json.dump(R, open("/kaggle/working/PROBE.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ("uid", "status", "which", "unshare_user", "unshare_all", "setuid_nobody", "internet")}, default=str)[:3000])
