"""Kaggle jail (FC-1 result): bubblewrap and every namespace are unavailable in a Kaggle kernel (seccomp denies unshare; no bwrap; no apt). What IS available to root:
chroot (CAP_SYS_CHROOT), setuid/setgid to nobody, rlimits (bind a non-root uid), prctl NO_NEW_PRIVS, a user-installed seccomp-bpf filter, mknod.
Layers built here, in order, by a root launcher: (1) chroot into a prebuilt root-owned rootfs that holds ONLY the interpreter/JDK + stdlib + the candidate and runner
(no /proc, no host paths, no /kaggle/input, no secrets), (2) drop all groups, gid/uid 65534, all capabilities lost with the uid change, (3) NO_NEW_PRIVS,
(4) seccomp-bpf: socket() for any family but AF_UNIX returns EPERM, ptrace denied, (5) rlimits, (6) minimal environment.
NOT provided (unlike bwrap): pid/net/ipc/uts namespaces, a private tmpfs (the rootfs /tmp is wiped by the launcher before each run), hiding other processes except through the chroot's missing /proc.
Network is therefore blocked by seccomp (+ the kernel having internet off), not by a network namespace. Use: engine.install_kaggle_jail() (flag EL_JAIL=kaggle).
Launcher: python kaggle_jail.py <rootfs> <mode> ...  (mode py: <cand> <runner>; mode java: <workdir> <rw|ro> <JDK> -- cmd...)"""
from __future__ import annotations

import ctypes
import os
import resource
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOTFS = Path(os.environ.get("EL_KJAIL_ROOT", "/tmp/kjail-root"))
NOBODY = 65534
SELF = Path(__file__).resolve()


# ------------------------------------------------------------------ rootfs (built once, trusted, as root)
def _ldd(path: str) -> set:
    out = subprocess.run(["ldd", path], capture_output=True, text=True).stdout
    libs = set()
    for ln in out.splitlines():
        for tok in ln.split():
            if tok.startswith("/") and os.path.exists(tok):
                libs.add(tok)
    return libs


def _copy(src: str, root: Path) -> None:
    dst = root / src.lstrip("/")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.isdir(src) and not os.path.islink(src):
        shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True, ignore=shutil.ignore_patterns("site-packages", "dist-packages", "__pycache__", "test", "tests", "idlelib", "turtledemo", "ensurepip"))
    else:
        if os.path.islink(src):
            if not dst.is_symlink() and not dst.exists():
                os.symlink(os.readlink(src), dst)
            _copy(os.path.realpath(src), root)
        else:
            shutil.copy2(src, dst)


def build_rootfs(java_home: str = "") -> dict:
    """-> receipt of what the rootfs contains. Idempotent per kernel (rebuilt from scratch)."""
    if ROOTFS.exists():
        shutil.rmtree(ROOTFS)
    ROOTFS.mkdir(parents=True)
    py = os.path.realpath("/usr/bin/python3")
    probe = subprocess.run(["/usr/bin/python3", "-I", "-S", "-c", "import sys,json;print(json.dumps([p for p in sys.path if p.startswith('/usr')]))"], capture_output=True, text=True).stdout
    import json
    pydirs = json.loads(probe)
    libs = _ldd(py)
    for d in pydirs:
        if os.path.isdir(d):
            _copy(d, ROOTFS)
            for so in Path(d).rglob("*.so*"):
                libs |= _ldd(str(so))
    _copy("/usr/bin/python3", ROOTFS)
    _copy(py, ROOTFS)
    tools = ["/usr/bin/prlimit"]
    for t in tools:
        _copy(t, ROOTFS)
        libs |= _ldd(t)
    if java_home:
        _copy(java_home, ROOTFS)
        for f in Path(java_home).rglob("*"):
            if f.is_file() and (f.suffix == ".so" or f.parent.name == "bin"):
                libs |= _ldd(str(f))
    for lib in sorted(libs | {"/lib64/ld-linux-x86-64.so.2"}):
        if os.path.exists(lib):
            _copy(lib, ROOTFS)
    for d in ("tmp", "jail", "work", "dev", "etc"):
        (ROOTFS / d).mkdir(exist_ok=True)
    os.mknod(ROOTFS / "dev/null", 0o666 | 0o020000, os.makedev(1, 3))
    os.mknod(ROOTFS / "dev/urandom", 0o444 | 0o020000, os.makedev(1, 9))
    (ROOTFS / "etc/passwd").write_text("nobody:x:65534:65534::/nonexistent:/usr/sbin/nologin\n")
    for p in ROOTFS.rglob("*"):         # everything root-owned and not group/other-writable
        try:
            os.lchown(p, 0, 0)
        except OSError:
            pass
    os.chmod(ROOTFS, 0o755)
    for d in ("tmp", "jail", "work"):
        os.chmod(ROOTFS / d, 0o755)
    size = sum(f.stat().st_size for f in ROOTFS.rglob("*") if f.is_file() and not f.is_symlink())
    return {"rootfs": str(ROOTFS), "bytes": size, "python": py, "java_home": java_home or None, "libs": len(libs)}


# ------------------------------------------------------------------ command builders used by engine (same call shapes as the bwrap builders)
def bwrap_cmd(cand, runner):
    return [sys.executable, "-I", str(SELF), str(ROOTFS), "py", str(cand), str(runner)]


def java_jail(workdir, cmd, rw=True, jdk=""):
    jdk = jdk or os.environ.get("EL_KJAIL_JDK", "")
    return [sys.executable, "-I", str(SELF), str(ROOTFS), "java", str(workdir), "rw" if rw else "ro", jdk, "--"] + list(cmd)


# ------------------------------------------------------------------ launcher (runs as root, forks the jailed child)
def _seccomp_filter() -> bytes:
    """x86_64 only. socket(family != AF_UNIX) -> EPERM; ptrace -> EPERM; everything else allowed. Limitation: x32-ABI syscall numbers are not filtered (receipt records this)."""
    BPF_LD, BPF_W, BPF_ABS, BPF_JMP, BPF_JEQ, BPF_K, BPF_RET = 0x00, 0x00, 0x20, 0x05, 0x10, 0x00, 0x06
    ALLOW, EPERM, KILL = 0x7FFF0000, 0x00050000 | 1, 0x80000000
    def ins(code, jt, jf, k): return struct.pack("HBBI", code, jt, jf, k)
    LD = BPF_LD | BPF_W | BPF_ABS
    JEQ = BPF_JMP | BPF_JEQ | BPF_K
    RET = BPF_RET | BPF_K
    # explicit layout (indexes):
    prog = [ins(LD, 0, 0, 4),            # 0 load arch
            ins(JEQ, 1, 0, 0xC000003E),  # 1 x86_64? skip next if yes
            ins(RET, 0, 0, KILL),        # 2
            ins(LD, 0, 0, 0),            # 3 load nr
            ins(JEQ, 0, 1, 101),         # 4 ptrace? else skip 1
            ins(RET, 0, 0, EPERM),       # 5
            ins(JEQ, 0, 4, 41),          # 6 socket? else jump to 11 (allow)
            ins(LD, 0, 0, 16),           # 7 load args[0] (family)
            ins(JEQ, 2, 0, 1),           # 8 AF_UNIX -> jump to 11 (allow)
            ins(RET, 0, 0, EPERM),       # 9
            ins(RET, 0, 0, ALLOW),       # 10 (unused landing)
            ins(RET, 0, 0, ALLOW)]       # 11
    return b"".join(prog)


def _install_seccomp() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:                    # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "no_new_privs")
    f = _seccomp_filter()
    buf = ctypes.create_string_buffer(f, len(f))
    class Prog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    prog = Prog(len(f) // 8, ctypes.cast(buf, ctypes.c_void_p))
    if libc.prctl(22, 2, ctypes.byref(prog)) != 0:          # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
        raise OSError(ctypes.get_errno(), "seccomp")


def _wipe(d: Path) -> None:
    for p in d.iterdir():
        shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()


def _launch() -> int:
    root, mode = Path(sys.argv[1]), sys.argv[2]
    try:
        for d in ("tmp", "jail", "work"):
            _wipe(root / d)
        os.chmod(root / "tmp", 0o1777)
        if mode == "py":
            cand, runner = sys.argv[3], sys.argv[4]
            shutil.copyfile(cand, root / "jail/cand.py"); shutil.copyfile(runner, root / "jail/runner.py")
            for n in ("cand.py", "runner.py"):
                os.chmod(root / "jail" / n, 0o444)
            argv = ["/usr/bin/python3", "-I", "-S", "/jail/runner.py", "/jail/cand.py"]
            env = {"PATH": "/usr/bin", "PYTHONHASHSEED": "0"}
            workdir = None
        else:
            workdir, rw, jdk = Path(sys.argv[3]), sys.argv[4] == "rw", sys.argv[5]
            for f in workdir.iterdir():
                if f.is_file() and not f.is_symlink():
                    shutil.copyfile(f, root / "work" / f.name)
            os.chmod(root / "work", 0o777 if rw else 0o755)
            argv = sys.argv[7:]
            env = {"PATH": "/usr/bin:" + jdk + "/bin", "JAVA_HOME": jdk, "LC_ALL": "C.UTF-8"}
    except Exception as e:
        sys.stderr.write("bwrap: kjail setup failed: %r\n" % e); return 125
    pid = os.fork()
    if pid == 0:
        try:
            os.chroot(root); os.chdir("/tmp" if mode == "py" else "/work")
            os.setgroups([]); os.setgid(NOBODY); os.setuid(NOBODY)
            _install_seccomp()
            os.execve(argv[0] if argv[0].startswith("/") else "/usr/bin/" + argv[0], argv, env)
        except BaseException as e:
            sys.stderr.write("bwrap: kjail child setup failed: %r\n" % e)
            os._exit(125)
    _, st = os.waitpid(pid, 0)
    if mode == "java" and sys.argv[4] == "rw":              # copy back regular files only (never follow symlinks); the host dir is the trusted parent's own temp dir
        for f in (root / "work").iterdir():
            if f.is_file() and not f.is_symlink() and f.stat().st_size < (64 << 20):
                shutil.copyfile(f, workdir / f.name)
    rc = os.waitstatus_to_exitcode(st)
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(_launch())
