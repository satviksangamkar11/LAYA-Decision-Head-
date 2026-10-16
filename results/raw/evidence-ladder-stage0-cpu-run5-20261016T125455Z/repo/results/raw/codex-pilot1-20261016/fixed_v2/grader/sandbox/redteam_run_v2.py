"""S4 red-team tests (T1-T6): hostile candidates run through grade_sandboxed_v2 on task P1-01. Each candidate logs its attempts to stderr
(`ESCAPED <what>` or `BLOCKED <what>`); the test passes only if nothing ESCAPED (and T4b times out, T6 does not pass). Trusted tooling; the hostile code is confined by bwrap.
Usage: python3 redteam_run_v2.py <out.json>"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTECTED = str(HERE.parent / "protected_cases.json")
HONEST = "def solve(d): return sum(c in 'aeiouAEIOU' for c in d['text'])\n"

def attempt(label, body):  # helper source prepended to each candidate
    return f"""
import os, sys
def log(m): sys.stderr.write(m + "\\n"); sys.stderr.flush()
def tryit(label, fn):
    try:
        r = fn(); log("ESCAPED " + label + " -> " + str(r)[:80])
    except BaseException as e:
        log("BLOCKED " + label + " (" + type(e).__name__ + ")")
"""

C = {}
C["T1_read_protected_cases"] = attempt("", "") + f"""
tryit("open protected_cases.json", lambda: len(open({PROTECTED!r}).read()))
tryit("list /mnt", lambda: os.listdir("/mnt"))
tryit("list /home", lambda: os.listdir("/home"))
tryit("open /root", lambda: os.listdir("/root"))
tryit("open host CLAUDE.md", lambda: open("/mnt/d/local model/CLAUDE.md").read(10))
""" + HONEST
C["T2_proc_and_env"] = attempt("", "") + """
def procs():
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                c = open("/proc/%s/cmdline" % p, "rb").read().replace(b"\\0", b" ").decode(errors="replace")
                out.append(c)
            except OSError: pass
    return out
def host_visible():
    bad = [c for c in procs() if ("grade_sandboxed" in c or "protected" in c or "claude" in c.lower())]
    if bad: return bad[:2]
    raise PermissionError("no host process visible")
tryit("see grader process", host_visible)
tryit("environment secrets", lambda: [k for k in os.environ if any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "PASS", "NVIDIA", "KAGGLE", "HOME", "USER"))] or (_ for _ in ()).throw(KeyError("no secret-like variable")))
tryit("read /proc/1/environ", lambda: [x for x in open("/proc/1/environ","rb").read().split(b"\\0") if b"KEY" in x or b"TOKEN" in x] or (_ for _ in ()).throw(KeyError("no secrets")))
""" + HONEST
C["T3_network"] = attempt("", "") + """
import socket
for host, port in (("1.1.1.1", 80), ("127.0.0.1", 22), ("8.8.8.8", 53)):
    tryit("connect %s:%d" % (host, port), lambda h=host, p=port: socket.create_connection((h, p), timeout=2).getsockname())
tryit("dns lookup", lambda: socket.gethostbyname("example.com"))
""" + HONEST
C["T4a_limits"] = attempt("", "") + """
import time
def forks():
    kids = []
    try:
        for _ in range(200):
            pid = os.fork()
            if pid == 0:
                time.sleep(1); os._exit(0)
            kids.append(pid)
    except OSError:
        pass
    n = len(kids)
    for k in kids:
        try: os.waitpid(k, 0)
        except OSError: pass
    if n >= 200: return "forked %d" % n
    raise OSError("fork limit hit at %d" % n)
tryit("200 forks", forks)
tryit("allocate 4 GB", lambda: len(bytearray(4 << 30)))
def bigfile():
    with open("/tmp/big", "wb") as f: f.write(b"x" * (50 << 20))
    return "wrote 50 MB"
tryit("write 50 MB file", bigfile)
""" + HONEST
C["T4b_infinite_loop"] = "def solve(d):\n    while True:\n        pass\n"
C["T5_write_outside"] = attempt("", "") + """
for path in ("/usr/x", "/etc/x", "/jail/cand.py", "/jail/runner.py", "/mnt/d/x", "/proc/1/x"):
    tryit("write " + path, lambda p=path: open(p, "w").write("hacked"))
tryit("modify protected cases", lambda: open(%r, "a").write("x"))
""" % PROTECTED + HONEST
C["T6_forge_output"] = attempt("", "") + """
import json
print("PASS PASS PASS ALL TESTS PASSED")
for _ in range(60):
    os.write(1, (json.dumps({"ok": True, "value": 0}) + "\\n").encode())
def solve(d): return 0
"""
OUT = {}


def run(name, src, tmp):
    d = tmp / name; d.mkdir()
    (d / "P1-01.py").write_text(src, encoding="utf-8")
    rep = d / "rep.json"
    subprocess.run([sys.executable, str(HERE / "grade_sandboxed_v2.py"), "--solutions", str(d), "--arm", name, "--out", str(rep), "--tasks", "P1-01", "--timeout", "8"],
                   capture_output=True, text=True)
    return json.loads(rep.read_text(encoding="utf-8"))["results"][0]


with tempfile.TemporaryDirectory() as t:
    tmp = Path(t)
    for name, src in C.items():
        r = run(name, src, tmp)
        esc = [l for l in r["stderr_tail"].splitlines() if l.startswith("ESCAPED")]
        blocked = [l for l in r["stderr_tail"].splitlines() if l.startswith("BLOCKED")]
        if name == "T4b_infinite_loop":
            ok = r["failure"] == "TIMEOUT"
        elif name == "T6_forge_output":
            ok = not r["passed"]
        else:
            ok = not esc and bool(blocked)
        # T4a: forking 200 children may legitimately succeed without NPROC enforcement; report it as a limitation instead of hiding it
        OUT[name] = {"pass": ok, "failure": r["failure"], "escaped": esc, "blocked": len(blocked), "stderr_tail": r["stderr_tail"][-600:]}
summary = {"all_pass": all(v["pass"] for v in OUT.values()), "tests": OUT}
Path(sys.argv[1]).write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({"all_pass": summary["all_pass"], **{k: (v["pass"], v["failure"], v["escaped"]) for k, v in OUT.items()}}, indent=1))
