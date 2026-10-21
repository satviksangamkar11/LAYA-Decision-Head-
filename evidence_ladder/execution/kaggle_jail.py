"""Kaggle jail (FC-1 result): bubblewrap and every namespace are unavailable in a Kaggle kernel (seccomp denies unshare; no bwrap; no apt). What IS available to root:
chroot (CAP_SYS_CHROOT), setuid/setgid to nobody, rlimits (bind a non-root uid), prctl NO_NEW_PRIVS, a user-installed seccomp-bpf filter, mknod.
Layers built here, in order, by a root launcher: (1) chroot into a prebuilt root-owned rootfs that holds ONLY the interpreter/JDK + stdlib + the candidate and runner
(no /proc, no host paths, no /kaggle/input, no secrets), (2) drop all groups, gid/uid 65534, all capabilities lost with the uid change, (3) NO_NEW_PRIVS,
(4) seccomp-bpf: wrong arch or x32 syscall numbers KILL, socket() of ANY family (incl. AF_UNIX) EPERM, ptrace denied, (5) rlimits, (6) minimal environment.
NOT provided (unlike bwrap): pid/net/ipc/uts namespaces. /tmp is a per-run private directory (symlink to /runs/<random>, removed after the run; launchers are serialized by flock; leftover uid-65534 processes are reaped).
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


# ------------------------------------------------------------------ [MULTILANG 2026-10-21] optional toolchains for the trusted verifier (trusted_native.py). EXPLICIT list only: build_rootfs(toolchains=("c","cpp","javascript","go","rust")).
# STATUS: staging logic exercised on a WSL Ubuntu-24.04 root proxy only (results/raw/multilang-sandbox-*); NOT run on Kaggle (no kernel was started). What a Kaggle kernel must provide on the host:
#   c / cpp      gcc / g++ with cc1 / cc1plus, binutils `as` and `ld`, libc dev files (crt1.o crti.o crtn.o libc.so libc_nonshared.a libm.so ...), /usr/include (+ c++ headers for cpp)
#   javascript   a `node` executable (any recent; copied with the libraries ldd reports)
#   go           a GOROOT (bin/go, pkg/tool/<os_arch>/*, src/) found from `go` on PATH (or EL_KJAIL_GOROOT); builds run with CGO_ENABLED=0
#   rust         a sysroot (`rustc --print sysroot`: bin/rustc, lib/*.so, lib/rustlib/<target>/lib) and a working `cc` (so "c" must be staged too)
# Anything missing makes build_rootfs RAISE for that language (never a silent partial rootfs). The run-time environment of these tools is written to <rootfs>/.toolchains.json and read by the launcher.
def _real(name: str):
    p = shutil.which(name)
    return os.path.realpath(p) if p else None


def _tree(src: str, root: Path, skip=()) -> int:
    """copy a directory tree (symlinks kept, idempotent: existing entries are left alone) minus the given relative sub-paths; -> bytes copied"""
    base, total = root / src.lstrip("/"), 0
    for dp, dns, fns in os.walk(src, followlinks=False):
        rel = os.path.relpath(dp, src).replace("\\", "/")
        pre = "" if rel == "." else rel + "/"
        dns[:] = [d for d in dns if pre + d not in skip]
        (base / rel).mkdir(parents=True, exist_ok=True)
        for n in dns + [f for f in fns if pre + f not in skip]:
            p, q = os.path.join(dp, n), base / rel / n
            if q.is_symlink() or (q.exists() and not os.path.isdir(p)):
                continue
            if os.path.islink(p):
                os.symlink(os.readlink(p), q)
            elif os.path.isfile(p):
                shutil.copy2(p, q)
                total += q.stat().st_size
    return total


def _alias(root: Path, directory: str, real: str, names) -> None:
    """copy the real executable and make every name in `names` a relative symlink to it (host alternatives/symlink chains are not reproduced)"""
    _copy(real, root)
    d = root / directory.lstrip("/")
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        link = d / n
        if link.is_symlink() or link.exists():
            if link.resolve() == (root / real.lstrip("/")).resolve():
                continue
            link.unlink()
        os.symlink(os.path.relpath(root / real.lstrip("/"), d), link)


def _stage_toolchains(names, root: Path, libs: set) -> dict:
    wanted = list(dict.fromkeys(names))
    bad = [n for n in wanted if n not in ("c", "cpp", "javascript", "go", "rust")]
    if bad:
        raise ValueError("unknown toolchain(s) %r (allowed: c cpp javascript go rust)" % bad)
    if "rust" in wanted and "c" not in wanted:
        wanted.insert(0, "c")                                  # rustc links through `cc`
    path, ld, env, rec = [], [], {"HOME": "/tmp"}, {}
    for n in wanted:
        before = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink())
        if n in ("c", "cpp"):
            gcc, gpp = _real("gcc"), _real("g++")
            if not gcc or (n == "cpp" and not gpp):
                raise RuntimeError("toolchain %s: gcc%s not found on the host" % (n, " / g++" if n == "cpp" else ""))
            sub = lambda *a: subprocess.run(list(a), capture_output=True, text=True).stdout.strip()
            multiarch = sub(gcc, "-print-multiarch") or "x86_64-linux-gnu"
            cc1 = sub(gcc, "-print-prog-name=cc1")
            if not os.path.isabs(cc1) or not os.path.exists(cc1):
                raise RuntimeError("toolchain c: cc1 not found (gcc -print-prog-name=cc1 -> %r)" % cc1)
            gdir = os.path.dirname(cc1)
            _alias(root, "/usr/bin", gcc, ["gcc", "cc", os.path.basename(gcc)])
            libs |= _ldd(gcc)
            for tool in ("as", "ld", "ld.bfd"):
                real = _real(tool)
                if real:
                    _alias(root, "/usr/bin", real, [tool])
                    libs |= _ldd(real)
            install = next((l.split(":", 1)[1].strip() for l in sub(gcc, "-print-search-dirs").splitlines() if l.startswith("install:")), gdir)      # headers / crtbegin / libgcc live here, cc1 may be elsewhere (libexec)
            for d in dict.fromkeys([gdir, os.path.normpath(install)]):
                _tree(d, root, ("plugin", "lto1", "lto-wrapper", "install-tools"))
                for f in Path(d).iterdir():
                    if f.is_file() and not f.is_symlink() and os.access(f, os.X_OK):
                        libs |= _ldd(str(f))
            for inc in ("/usr/include",):
                _tree(inc, root)
            mdir = "/usr/lib/" + multiarch
            for pat in ("crt*.o", "Scrt1.o", "gcrt1.o", "libc.so", "libc.a", "libc_nonshared.a", "libm.so", "libm.a", "libm-*.a", "libmvec.a", "libmvec.so", "libmvec_nonshared.a",
                        "libpthread.a", "libdl.a", "librt.a", "libutil.a", "libgcc_s.so*", "libstdc++.so*"):
                for f in Path(mdir).glob(pat):
                    _copy(str(f), root)
            for so in ("libc.so.6", "libm.so.6", "libmvec.so.1", "libgcc_s.so.1", "libdl.so.2", "libpthread.so.0", "librt.so.1", "libutil.so.1"):
                p = sub(gcc, "-print-file-name=" + so)
                if os.path.isabs(p) and os.path.exists(p):
                    libs.add(os.path.realpath(p)); libs.add(p)
            if n == "cpp":
                _alias(root, "/usr/bin", gpp, ["g++", "c++", os.path.basename(gpp)])
                libs |= _ldd(gpp)
                p = sub(gcc, "-print-file-name=libstdc++.so.6")
                if os.path.isabs(p) and os.path.exists(p):
                    libs.add(os.path.realpath(p)); libs.add(p)
        elif n == "javascript":
            node = _real("node")
            if not node:
                raise RuntimeError("toolchain javascript: node not found on the host")
            shutil.copy2(node, root / "usr/bin/node")
            libs |= _ldd(node)
        elif n == "go":
            goroot = os.environ.get("EL_KJAIL_GOROOT") or (os.path.dirname(os.path.dirname(_real("go") or "")) if _real("go") else "")
            if not goroot or not os.path.isfile(goroot + "/bin/go"):
                raise RuntimeError("toolchain go: no GOROOT (go not on PATH and EL_KJAIL_GOROOT unset)")
            _tree(goroot, root, ("test", "doc", "misc", "api", "lib/wasm", "src/cmd/vendor/golang.org/x/tools"))
            for f in (Path(goroot) / "bin").iterdir():
                libs |= _ldd(str(f))
            path.append(goroot + "/bin")
            env.update(GOROOT=goroot, GOCACHE="/tmp/gocache", GOPATH="/tmp/gopath", GOTOOLCHAIN="local", GOTELEMETRY="off", CGO_ENABLED="0", GOFLAGS="-mod=mod", GOPROXY="off", GOSUMDB="off")
        elif n == "rust":
            rustc = _real("rustc")
            if not rustc:
                raise RuntimeError("toolchain rust: rustc not found on the host")
            sysroot = subprocess.run([rustc, "--print", "sysroot"], capture_output=True, text=True).stdout.strip()
            if sysroot.rstrip("/") in ("", "/usr", "/usr/local"):
                raise RuntimeError("toolchain rust: the sysroot is the system prefix %r (a distribution rustc); copying it would stage /usr/lib wholesale. Point PATH at a standalone toolchain (rustup / official tarball)" % sysroot)
            if not sysroot or not os.path.isfile(sysroot + "/bin/rustc"):
                raise RuntimeError("toolchain rust: sysroot %r has no bin/rustc" % sysroot)
            (root / sysroot.lstrip("/") / "bin").mkdir(parents=True, exist_ok=True)
            shutil.copy2(sysroot + "/bin/rustc", root / sysroot.lstrip("/") / "bin/rustc")
            libs |= _ldd(sysroot + "/bin/rustc")
            _tree(sysroot + "/lib", root, ("rustlib/src", "rustlib/etc"))
            for so in Path(sysroot + "/lib").glob("*.so*"):
                libs |= _ldd(str(so))
            path.append(sysroot + "/bin")
            ld.append(sysroot + "/lib")
        rec[n] = {"bytes_added": sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink()) - before}
    import json
    (root / ".toolchains.json").write_text(json.dumps({"path": path, "ld": ld, "env": env, "staged": wanted}))
    return rec


def build_rootfs(java_home: str = "", toolchains=()) -> dict:
    """-> receipt of what the rootfs contains. Idempotent per kernel (rebuilt from scratch). toolchains: explicit list of languages to stage for the trusted verifier ([MULTILANG]); () = unchanged rootfs."""
    if ROOTFS.exists():
        shutil.rmtree(ROOTFS)
    ROOTFS.mkdir(parents=True)
    for top in ("lib", "lib64", "bin", "sbin"):          # merged-/usr hosts: mirror the top-level symlinks so copied libraries and symlink targets resolve inside the chroot
        if os.path.islink("/" + top):
            os.symlink(os.readlink("/" + top), ROOTFS / top)
            (ROOTFS / os.path.normpath(os.path.join("/", os.readlink("/" + top))).lstrip("/")).mkdir(parents=True, exist_ok=True)
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
    tc_receipt = _stage_toolchains(toolchains, ROOTFS, libs) if toolchains else None
    # symlinks copied from the host that point outside their copied tree (Debian: $JAVA_HOME/lib/libjli.so -> /usr/lib/x86_64-linux-gnu/jli/libjli.so): copy the real target too
    for _ in range(3):
        for link in [p for p in ROOTFS.rglob("*") if p.is_symlink()]:
            tgt = os.path.normpath(os.path.join("/", str(link.relative_to(ROOTFS)).rsplit("/", 1)[0], os.readlink(link))) if not os.readlink(link).startswith("/") else os.readlink(link)
            if os.path.exists(tgt) and not (ROOTFS / tgt.lstrip("/")).exists():
                _copy(tgt, ROOTFS)
                if os.path.isfile(tgt):
                    libs |= _ldd(tgt)
    for lib in sorted(libs | {"/lib64/ld-linux-x86-64.so.2"}):
        if os.path.exists(lib):
            _copy(lib, ROOTFS)
    for d in ("jail", "work", "runs", "dev", "etc"):
        (ROOTFS / d).mkdir(exist_ok=True)
    os.mknod(ROOTFS / "dev/null", 0o666 | 0o020000, os.makedev(1, 3))
    os.mknod(ROOTFS / "dev/urandom", 0o444 | 0o020000, os.makedev(1, 9))
    os.chmod(ROOTFS / "dev/null", 0o666); os.chmod(ROOTFS / "dev/urandom", 0o444)      # mknod mode is masked by the umask
    for p in ROOTFS.rglob("*"):         # everything root-owned and not group/other-writable
        try:
            os.lchown(p, 0, 0)
        except OSError:
            pass
    os.chmod(ROOTFS, 0o755)
    for d in ("jail", "work"):
        os.chmod(ROOTFS / d, 0o755)
    os.chmod(ROOTFS / "runs", 0o711)
    size = sum(f.stat().st_size for f in ROOTFS.rglob("*") if f.is_file() and not f.is_symlink())
    rec = {"rootfs": str(ROOTFS), "bytes": size, "python": py, "java_home": java_home or None, "libs": len(libs)}
    if tc_receipt is not None:
        rec["toolchains"] = tc_receipt
    return rec


# ------------------------------------------------------------------ command builders used by engine (same call shapes as the bwrap builders)
def bwrap_cmd(cand, runner):
    return [sys.executable, "-I", str(SELF), str(ROOTFS), "py", str(cand), str(runner)]


def java_jail(workdir, cmd, rw=True, jdk=""):
    jdk = jdk or os.environ.get("EL_KJAIL_JDK", "")
    return [sys.executable, "-I", str(SELF), str(ROOTFS), "java", str(workdir), "rw" if rw else "ro", jdk, "--"] + list(cmd)


# ------------------------------------------------------------------ launcher (runs as root, forks the jailed child)
def _seccomp_filter() -> bytes:
    """x86_64 only. Wrong arch -> KILL; any syscall number with the x32 bit (0x40000000) -> KILL; socket() of ANY family (incl. AF_UNIX, path and abstract) -> EPERM;
    ptrace -> EPERM; everything else allowed (socketpair stays allowed: an anonymous pair reaches nothing outside the process tree)."""
    ALLOW, EPERM, KILL = 0x7FFF0000, 0x00050000 | 1, 0x80000000
    LD, JEQ, JSET, RET = 0x20, 0x15, 0x45, 0x06
    def ins(code, jt, jf, k): return struct.pack("HBBI", code, jt, jf, k)
    prog = [ins(LD, 0, 0, 4),            # 0 load seccomp_data.arch
            ins(JEQ, 1, 0, 0xC000003E),  # 1 AUDIT_ARCH_X86_64? skip the KILL
            ins(RET, 0, 0, KILL),        # 2
            ins(LD, 0, 0, 0),            # 3 load nr
            ins(JSET, 0, 1, 0x40000000), # 4 x32 bit set? fall into KILL
            ins(RET, 0, 0, KILL),        # 5
            ins(JEQ, 0, 1, 101),         # 6 ptrace
            ins(RET, 0, 0, EPERM),       # 7
            ins(JEQ, 0, 1, 41),          # 8 socket (every family)
            ins(RET, 0, 0, EPERM),       # 9
            ins(RET, 0, 0, ALLOW)]       # 10
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


def _reap() -> None:
    """Kill every leftover process of the jail uid (daemonised candidates escape the parent's process-group kill). A child that setuid()s to nobody and kill(-1)s reaches only uid 65534."""
    for _ in range(2):
        pid = os.fork()
        if pid == 0:
            try:
                os.setgroups([]); os.setgid(NOBODY); os.setuid(NOBODY); os.kill(-1, 9)
            finally:
                os._exit(0)
        os.waitpid(pid, 0)


def _clean(root: Path) -> None:
    _reap()
    for d in ("jail", "work", "runs"):
        (root / d).mkdir(exist_ok=True)
        _wipe(root / d)
    t = root / "tmp"
    if t.is_symlink() or t.exists():
        t.unlink() if t.is_symlink() else shutil.rmtree(t)


def _launch() -> int:
    import fcntl, uuid
    root, mode = Path(sys.argv[1]), sys.argv[2]
    lock = open(root / ".lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)      # one jailed run at a time: nothing is shared between concurrent runs
    try:
        _clean(root)
        run = root / "runs" / uuid.uuid4().hex                              # per-run private /tmp: fresh nobody-owned 0700 dir, removed after the run; /runs is 0711 (traverse, no listing)
        run.mkdir(); os.chown(run, NOBODY, NOBODY); os.chmod(run, 0o700); os.chmod(root / "runs", 0o711)
        os.symlink("/runs/" + run.name, root / "tmp")
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
                    if f.stat().st_mode & 0o111:                                     # [MULTILANG] keep the execute bit of host-staged programs (copyfile drops it); files without it are copied exactly as before
                        os.chmod(root / "work" / f.name, 0o755)
            os.chmod(root / "work", 0o777 if rw else 0o755)
            argv = sys.argv[7:]
            env = {"PATH": "/usr/bin:" + jdk + "/bin", "JAVA_HOME": jdk, "LC_ALL": "C.UTF-8",
                   "LD_LIBRARY_PATH": jdk + "/lib:" + jdk + "/lib/server"}      # no /proc in the chroot, so ld.so cannot resolve $ORIGIN (RUNPATH of java/javac)
            tcf = root / ".toolchains.json"
            if tcf.is_file():                                                       # [MULTILANG] toolchains staged by build_rootfs(toolchains=...): extra PATH entries and environment
                import json
                tc = json.loads(tcf.read_text())
                env["PATH"] += "".join(":" + p for p in tc.get("path", []))
                env["LD_LIBRARY_PATH"] += "".join(":" + p for p in tc.get("ld", []))      # no /proc in the chroot: $ORIGIN RUNPATHs (rustc_driver) cannot resolve
                env.update(tc.get("env", {}))
    except Exception as e:
        sys.stderr.write("bwrap: kjail setup failed: %r\n" % e); return 125
    pid = os.fork()
    if pid == 0:
        step = "chroot"
        try:
            exe = argv[0] if argv[0].startswith("/") else "/usr/bin/" + argv[0]
            if not argv[0].startswith("/") and not os.path.exists(str(root) + exe):      # [MULTILANG] a tool staged outside /usr/bin (go, rustc): resolve through the jail PATH; names that exist in /usr/bin resolve exactly as before
                for d in env["PATH"].split(":"):
                    if os.path.exists(str(root) + d + "/" + argv[0]):
                        exe = d + "/" + argv[0]
                        break
            if mode == "py":
                resource.setrlimit(resource.RLIMIT_CPU, (30, 30))      # CPU-time backstop; the wall-clock group kill by the parent is the primary limit
            os.chroot(root); step = "chdir"; os.chdir("/tmp" if mode == "py" else "/work")
            step = "setgroups/setgid/setuid"; os.setgroups([]); os.setgid(NOBODY); os.setuid(NOBODY)
            step = "seccomp"; _install_seccomp()
            step = "execve " + exe; os.execve(exe, argv, env)
        except BaseException as e:
            # report the failing step and the first missing path component of the executable and its ELF interpreter (no secrets involved)
            miss = []
            for pth in (exe, "/lib64/ld-linux-x86-64.so.2", "/lib/x86_64-linux-gnu/libc.so.6", "/tmp", "/jail/runner.py", "/jail/cand.py"):
                try:
                    os.stat(pth)
                except OSError as e2:
                    miss.append("%s:%s" % (pth, e2.strerror))
            sys.stderr.write("bwrap: kjail child setup failed at step '%s': %r; missing in chroot: %s\n" % (step, e, miss))
            os._exit(125)
    _, st = os.waitpid(pid, 0)
    if mode == "java" and sys.argv[4] == "rw":              # copy back regular files only (never follow symlinks); the host dir is the trusted parent's own temp dir
        for f in (root / "work").iterdir():
            if f.is_file() and not f.is_symlink() and f.stat().st_size < (64 << 20):
                shutil.copyfile(f, workdir / f.name)
    _clean(root)                                             # kill leftovers, delete the private /tmp
    rc = os.waitstatus_to_exitcode(st)
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(_launch())
