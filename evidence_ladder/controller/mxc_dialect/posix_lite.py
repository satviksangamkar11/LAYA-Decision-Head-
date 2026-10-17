"""[MXC replacement] posix-lite: a tiny, deterministic POSIX-style command runner for the Windows ProcessContainer, which cannot start MSYS bash or PowerShell (both die with 0xC0000142 inside the
container, verified). It runs INSIDE the MXC container under the sandboxed Python, so every file access it makes is subject to the MXC filesystem policy; it adds no privilege.

    python posix_lite.py -c "<command>"      (the provider passes the original model/harness command string unchanged)

Supported subset (anything else is rejected BEFORE execution by parse(), and the provider turns that into UnsupportedCommand = a tool error, never a pass):
    commands   echo [-n]  cat FILE...  ls [-a] [PATH...]  pwd  true  false  mkdir [-p] DIR...  touch FILE...  rm [-f] FILE...  python3|python ARGS...
    operators  ;   &&   ||   and output redirection  > FILE  >> FILE  (also > /dev/null, 2> /dev/null)
    quoting    single and double quotes, backslash escapes (shlex posix rules); NO $variables, backticks, $( ), globs, pipes, background &, input redirection, subshells, here-docs, newlines.
An unknown program behaves like bash: "bash: line 1: <prog>: command not found", exit 127. Output is written as UTF-8 with LF line endings (like the WSL sandbox). This is NOT bash: parity is
limited to the subset above and is measured by tests/phaseMXC_tests.py against the WSL sandbox."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys

PY_NAMES = ("python3", "python")
PUNCT = set(";&|<>()")


class Unsupported(ValueError):
    pass


def parse(cmd):
    """-> list of (op_before, argv, redirects) with op_before in (None, ';', '&&', '||'); redirects = [(fd, '>'|'>>', path)]. Raises Unsupported for anything outside the subset."""
    if not isinstance(cmd, str) or not cmd.strip() or len(cmd) > 4000 or "\x00" in cmd:
        raise Unsupported("empty, oversized or binary command")
    if "\n" in cmd or "\r" in cmd:
        raise Unsupported("newlines are not supported")
    if "$" in cmd or "`" in cmd:
        raise Unsupported("variables / command substitution are not supported")
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError as e:
        raise Unsupported("unparsable command: %s" % e)
    out, argv, redirs, op, i = [], [], [], None, 0
    while i < len(toks):
        t = toks[i]
        if t in (";", "&&", "||"):
            if not argv:
                raise Unsupported("empty command before %r" % t)
            out.append((op, argv, redirs))
            argv, redirs, op = [], [], t
        elif t in (">", ">>"):
            if i + 1 >= len(toks) or set(toks[i + 1]) <= PUNCT:
                raise Unsupported("redirection without a target")
            redirs.append((1, t, toks[i + 1]))
            i += 1
        elif set(t) <= PUNCT:
            raise Unsupported("shell operator %r is not supported" % t)
        else:
            argv.append(t)
        i += 1
    if not argv:
        raise Unsupported("trailing operator")
    out.append((op, argv, redirs))
    # "2> file" arrives as ['2', '>', 'file']: a lone trailing "2"/"1" before a redirection is the fd
    fixed = []
    for op, argv, redirs in out:
        if redirs and argv and argv[-1] in ("1", "2") and len(redirs) == 1 and _fd_adjacent(cmd, argv[-1], redirs[0][1]):
            fd = int(argv[-1])
            argv = argv[:-1]
            redirs = [(fd, redirs[0][1], redirs[0][2])]
            if not argv:
                raise Unsupported("redirection without a command")
        fixed.append((op, argv, redirs))
    return fixed


def _fd_adjacent(cmd, digit, arrow):
    return (digit + arrow) in cmd


def _open_target(path, mode):
    if path == "/dev/null":
        path = os.devnull
    return open(path, "ab" if mode == ">>" else "wb")


def _w(stream, text):
    stream.write(text.encode("utf-8"))


def _run_builtin(argv, out, err):
    name, a = argv[0], argv[1:]
    if name == "echo":
        nl = True
        if a and a[0] == "-n":
            nl, a = False, a[1:]
        _w(out, " ".join(a) + ("\n" if nl else ""))
        return 0
    if name == "true":
        return 0
    if name == "false":
        return 1
    if name == "pwd":
        _w(out, os.getcwd().replace("\\", "/") + "\n")
        return 0
    if name == "cat":
        rc = 0
        for f in a:
            try:
                with open(f, "rb") as fh:
                    out.write(fh.read())
            except OSError as e:
                _w(err, "cat: %s: %s\n" % (f, _why(e)))
                rc = 1
        return rc
    if name == "ls":
        show_all = False
        paths = []
        for x in a:
            if x.startswith("-") and len(x) > 1:
                if set(x[1:]) - set("a1"):
                    raise Unsupported("ls option %s" % x)
                show_all = show_all or "a" in x
            else:
                paths.append(x)
        rc = 0
        multi = len(paths) > 1
        for i, p in enumerate(paths or ["."]):
            if not os.path.lexists(p):
                _w(err, "ls: cannot access '%s': No such file or directory\n" % p)
                rc = 2
            elif os.path.isdir(p):
                if multi:
                    _w(out, ("\n" if i else "") + p + ":\n")
                names = sorted(os.listdir(p))
                if show_all:
                    names = [".", ".."] + names
                for n in names:
                    if show_all or not n.startswith("."):
                        _w(out, n + "\n")
            else:
                _w(out, p + "\n")
        return rc
    if name == "mkdir":
        parents = False
        rc = 0
        for x in a:
            if x == "-p":
                parents = True
            elif x.startswith("-"):
                raise Unsupported("mkdir option %s" % x)
            else:
                try:
                    os.makedirs(x, exist_ok=parents)
                except OSError as e:
                    _w(err, "mkdir: cannot create directory '%s': %s\n" % (x, _why(e)))
                    rc = 1
        return rc
    if name == "touch":
        rc = 0
        for x in a:
            try:
                with open(x, "ab"):
                    pass
            except OSError as e:
                _w(err, "touch: cannot touch '%s': %s\n" % (x, _why(e)))
                rc = 1
        return rc
    if name == "rm":
        force = False
        rc = 0
        for x in a:
            if x == "-f":
                force = True
            elif x.startswith("-"):
                raise Unsupported("rm option %s" % x)
            else:
                try:
                    os.remove(x)
                except OSError as e:
                    if not (force and isinstance(e, FileNotFoundError)):
                        _w(err, "rm: cannot remove '%s': %s\n" % (x, _why(e)))
                        rc = 1
        return rc
    return None


def _why(e):
    if isinstance(e, FileNotFoundError):
        return "No such file or directory"
    if isinstance(e, PermissionError):
        return "Permission denied"
    if isinstance(e, IsADirectoryError):
        return "Is a directory"
    if isinstance(e, FileExistsError):
        return "File exists"
    return type(e).__name__


def run(cmd):
    plan = parse(cmd)
    so, se = sys.stdout.buffer, sys.stderr.buffer
    rc = 0
    for op, argv, redirs in plan:
        if (op == "&&" and rc != 0) or (op == "||" and rc == 0):
            continue
        out, err, opened = so, se, []
        try:
            for fd, mode, path in redirs:
                try:
                    f = _open_target(path, mode)
                except OSError as e:
                    _w(se, "bash: %s: %s\n" % (path, _why(e)))
                    rc = 1
                    raise
                opened.append(f)
                if fd == 2:
                    err = f
                else:
                    out = f
            name = argv[0]
            if name in PY_NAMES:
                so.flush()
                se.flush()
                rc = subprocess.run([sys.executable] + argv[1:], stdout=out if out is not so else None, stderr=err if err is not se else None).returncode
            else:
                r = _run_builtin(argv, out, err)
                if r is None:
                    _w(se, "/bin/bash: line 1: %s: command not found\n" % name)
                    rc = 127
                else:
                    rc = r
        except OSError:
            rc = 1
        finally:
            for f in opened:
                f.close()
            so.flush()
            se.flush()
    return rc


def main(argv):
    if len(argv) != 3 or argv[1] != "-c":
        sys.stderr.write("usage: posix_lite.py -c <command>\n")
        return 2
    try:
        return run(argv[2])
    except Unsupported as e:
        sys.stderr.write("posix_lite: unsupported: %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
