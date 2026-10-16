"""Execution-layer tests. Run inside WSL as root (the jail drops to `nobody`):  python3 evidence_ladder/execution/tests/run_tests.py
Prints one JSON line {"tests","passed","failed":{name: reason},"notes":{...}}; exit 1 on any failure. Plain asserts; no pytest."""
from __future__ import annotations

import ast
import dataclasses
import json
import os
import stat
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ["EL_SECRET_CANARY"] = "canary-do-not-leak-7f3a"      # parent-env secret stand-in: must never be visible inside a jail

from evidence_ladder.common.interfaces import Artifact, ControllerContext, GraderIsolationError, HiddenGrade, TaskSpec  # noqa: E402
from evidence_ladder.execution import config as K  # noqa: E402
from evidence_ladder.execution.artifacts import ArtifactIntegrityError, ArtifactStore  # noqa: E402
from evidence_ladder.execution.backend import SandboxExecutionBackend  # noqa: E402
from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate  # noqa: E402  (tests may import it; production modules may not)
from evidence_ladder.execution.receipts import isolation_receipt, replay_existing_receipts  # noqa: E402
from evidence_ladder.execution.taskload import load_tasks  # noqa: E402

CFG = K.SandboxConfig(py_timeout=4.0, java_run_timeout=6.0)
TMP = Path(tempfile.mkdtemp(prefix="el-tests-"))
STORE = ArtifactStore(TMP / "store")
BE = SandboxExecutionBackend(STORE, CFG)
PY_TASKS = {t.task_id: t for t in load_tasks("python")}
JAVA_TASKS = {t.task_id: t for t in load_tasks("java")}
NOTES: dict = {}
TESTS: list = []


def test(f):
    TESTS.append(f)
    return f


# ------------------------------------------------------------------ fixtures
def py_ref_src(tid: str) -> str:
    return (K.PY_ROOT / "grader" / "oracles.py").read_text(encoding="utf-8") + f"\n\ndef solve(d):\n    return p{int(tid[3:]):02d}(d)\n"


def java_ref_src(tid: str) -> str:
    t = (K.JAVA_ROOT / "kit" / "Pilot1_Java" / "grader" / "ReferenceSolutions.java").read_text(encoding="utf-8").replace("\r\n", "\n")
    head, _, body = t.partition("\npublic final class ReferenceSolutions")
    assert head.strip().splitlines() == ["import java.util.*;", "import java.time.*;"], head
    return ("import java.util.*;\nimport java.time.*;\npublic final class Solution {\n public static Object solve(Map<String,Object> d){ return ReferenceSolutions.solve("
            + str(int(tid[3:])) + ", d); }\n}\nfinal class ReferenceSolutions" + body)


def wrong_py():
    src = (K.PY_ROOT / "grader" / "wrong_solutions_v2.py").read_text(encoding="utf-8").split("\nclass TO(")[0]
    ns: dict = {}
    exec(src, ns)           # trusted: only defines the table W of small hand-written wrong programs
    return ns["W"]


def art(task_id, lang, src, origin="test"):
    return Artifact.make(task_id, lang, src, origin=origin)


def pubs(outs):
    """The per-example public outcomes (contract: one per public example, in order, each with example_index)."""
    ex = [o for o in outs if o["check_id"] == "public_examples_v1"]
    assert [o["example_index"] for o in ex] == list(range(len(ex))), ex
    assert outs[0]["check_id"] == "parse_v1" and all(o["check_id"] == "public_examples_v1" for o in outs[1:]), [o["check_id"] for o in outs]
    return ex


def pub(outs, cid):
    """parse_v1 outcome, or for public_examples_v1 the first non-pass example outcome (else the first one): the aggregate view the older assertions use."""
    if cid != "public_examples_v1":
        return next(o for o in outs if o["check_id"] == cid)
    ex = pubs(outs)
    assert all(isinstance(o["observed"], str) for o in ex), ex
    return next((o for o in ex if o["status"] != "pass"), ex[0])


def run_pub(a, lang=None):
    t = (PY_TASKS if a.language == "python" else JAVA_TASKS)[a.task_id]
    return BE.run_public(a, t)


def jlist(*lines):
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ reference / wrong / classification
@test
def py_reference_passes_public():
    for tid in PY_TASKS:
        outs = run_pub(art(tid, "python", py_ref_src(tid), "reference"))
        assert pub(outs, "parse_v1")["status"] == "pass", (tid, outs)
        p = pub(outs, "public_examples_v1")
        assert p["status"] == "pass" and p["failure_class"] == "" and "counterexample" not in p, (tid, p)


@test
def py_wrong_fails_with_counterexample():
    W = wrong_py()
    failing, passing = [], []
    for tid, lst in W.items():
        for name, src in lst:
            p = pub(run_pub(art(tid, "python", src, "known_wrong:" + name)), "public_examples_v1")
            (failing if p["status"] == "fail" else passing).append((tid, name, p))
    assert failing, "no known-wrong solution was caught by public examples"
    for tid, name, p in failing:
        ce = p["counterexample"]
        assert set(ce) == {"input", "expected", "observed"} and p["failure_class"] in ("wrong_answer", "runtime_error"), (tid, name, p)
        assert ce["observed"] != ce["expected"]
    assert all(p["status"] == "pass" for _, _, p in passing)       # never anything but exact pass/fail
    NOTES["py_wrong_total"] = len(failing) + len(passing)
    NOTES["py_wrong_caught_by_public_examples"] = len(failing)
    NOTES["py_wrong_passing_public_but_caught_by_protected"] = [f"{t}:{n}" for t, n, _ in passing]


@test
def py_known_wrong_exact_counterexample():
    a = art("P1-01", "python", "def solve(d): return sum(c in 'aeiouyAEIOUY' for c in d['text'])", "known_wrong:counts_y_too")
    p = pub(run_pub(a), "public_examples_v1")
    assert p["status"] == "fail" or p["status"] == "pass"     # whichever; counterexample must be the FIRST failing public case if it fails
    bad = art("P1-01", "python", "def solve(d): return 99", "known_wrong:const")
    p = pub(run_pub(bad), "public_examples_v1")
    first = PY_TASKS["P1-01"].public_tests[0]
    assert p["status"] == "fail" and p["failure_class"] == "wrong_answer"
    assert json.loads(p["counterexample"]["input"]) == first["input"] and json.loads(p["counterexample"]["expected"]) == first["expected"] and p["counterexample"]["observed"] == "99"


@test
def py_compile_error_runtime_error_timeout_missing():
    o = run_pub(art("P1-01", "python", "def solve(d) return 1\n"))
    assert pub(o, "parse_v1")["status"] == "fail" and pub(o, "parse_v1")["failure_class"] == "compile_error"
    assert pub(o, "public_examples_v1")["status"] == "missing_evidence" and pub(o, "public_examples_v1")["failure_class"] == "compile_error"
    p = pub(run_pub(art("P1-01", "python", "def solve(d): raise ValueError('x')\n")), "public_examples_v1")
    assert p["status"] == "fail" and p["failure_class"] == "runtime_error" and p["counterexample"]["observed"] == "<error: ValueError>", p
    t0 = time.monotonic()
    p = pub(run_pub(art("P1-01", "python", "def solve(d):\n    while True: pass\n")), "public_examples_v1")
    assert p["status"] == "timeout" and p["failure_class"] == "timeout" and p["counterexample"]["observed"] == "<timeout>", p
    assert time.monotonic() - t0 < 20
    p = pub(run_pub(art("P1-01", "python", "x = 1\n")), "public_examples_v1")            # no solve()
    assert p["status"] == "fail" and p["failure_class"] == "runtime_error", p
    p = pub(run_pub(art("P1-01", "python", "import os\ndef solve(d):\n    os._exit(0)\n")), "public_examples_v1")   # dies silently
    assert p["status"] == "fail" and p["failure_class"] in ("missing_output", "runtime_error"), p
    p = pub(run_pub(art("P1-01", "python", "def solve(d): return 'x' * (10 << 20)\n")), "public_examples_v1")           # output flood
    assert p["status"] != "pass" and p["failure_class"] in ("invalid_output", "runtime_error", "wrong_answer"), p
    p = pub(run_pub(art("P1-01", "python", "#" * 200000 + "\ndef solve(d): return 0\n")), "public_examples_v1")           # oversized
    assert p["status"] != "pass", p


@test
def strict_type_sensitivity():
    t = TaskSpec("X-1", "py1:X-1", "python", "python", "p", [{"input": {"a": 1}, "expected": 1}])
    for src, want in (("def solve(d): return 1", "pass"), ("def solve(d): return True", "fail"), ("def solve(d): return 1.0", "fail"), ("def solve(d): return [1]", "fail")):
        p = pub(BE.run_public(art("X-1", "python", src), t), "public_examples_v1")
        assert p["status"] == want, (src, p)
    tj = TaskSpec("X-1", "java1:X-1", "java", "java", "p", [{"input": {"a": 1}, "expected": True}])
    p = pub(BE.run_public(art("X-1", "java", "import java.util.*;\npublic final class Solution { public static Object solve(Map<String,Object> d){ return Integer.valueOf(1); } }\n"), tj), "public_examples_v1")
    assert p["status"] == "fail" and p["failure_class"] == "wrong_answer", p                  # 1 is not true


@test
def task_language_mismatch_is_invalid_spec():
    o = BE.run_public(art("P1-01", "java", "class Solution {}"), PY_TASKS["P1-01"])
    assert o[0]["status"] == "invalid_spec"


@test
def per_example_outcomes():
    t = PY_TASKS["P1-01"]
    n = len(t.public_tests)
    ex = pubs(BE.run_public(art("P1-01", "python", py_ref_src("P1-01"), "reference"), t))
    assert len(ex) == n and all(o["status"] == "pass" and "counterexample" not in o for o in ex)
    assert [json.loads(o["observed"]) for o in ex] == [c["expected"] for c in t.public_tests], ex
    wrong = art("P1-01", "python", "def solve(d): return sum(c in 'aeiouyAEIOUY' for c in d['text'])", "known_wrong:counts_y_too")
    ex = pubs(BE.run_public(wrong, t))
    bad = [i for i, c in enumerate(t.public_tests) if "y" in c["input"]["text"].lower()]
    assert bad and [o["status"] for o in ex] == ["fail" if i in bad else "pass" for i in range(n)], [o["status"] for o in ex]
    for i in bad:
        assert ex[i]["failure_class"] == "wrong_answer" and set(ex[i]["counterexample"]) == {"input", "expected", "observed"}
        assert json.loads(ex[i]["counterexample"]["input"]) == t.public_tests[i]["input"]
    assert all("counterexample" not in ex[i] for i in range(n) if i not in bad)
    # java: only 1900 fails
    jt = JAVA_TASKS["J1-02"]
    jw = "import java.util.*;\npublic final class Solution { public static Object solve(Map<String,Object> d){ int y=((Number)d.get(\"year\")).intValue(); return y%4==0; } }\n"
    ex = pubs(BE.run_public(art("J1-02", "java", jw, "known_wrong"), jt))
    assert [o["status"] for o in ex] == ["pass", "fail", "pass", "pass"] and ex[1]["counterexample"]["observed"] == "true", ex
    # compile error / timeout / runtime-load failure / sandbox integrity: every example non-pass with the right class
    for src, cls, st in (("def solve(d) return 1\n", "compile_error", "missing_evidence"), ("def solve(d):\n    while True: pass\n", "timeout", "timeout"),
                         ("x = 1\n", "runtime_error", "fail")):
        outs = BE.run_public(art("P1-01", "python", src), t)
        ex = pubs(outs)
        assert len(ex) == n and all(o["status"] == st and o["failure_class"] == cls for o in ex), (src, ex)
    outs = BE.run_public(art("J1-01", "java", "public final class Solution { broken \n"), JAVA_TASKS["J1-01"])
    assert all(o["status"] == "missing_evidence" and o["failure_class"] == "compile_error" for o in pubs(outs))


# ------------------------------------------------------------------ java
@test
def java_reference_passes_public_and_wrong_fails():
    for tid in ("J1-01", "J1-02", "J1-09", "J1-12", "J1-24"):
        outs = run_pub(art(tid, "java", java_ref_src(tid), "reference"))
        assert pub(outs, "parse_v1")["status"] == "pass", (tid, outs)
        assert pub(outs, "public_examples_v1")["status"] == "pass", (tid, outs)
    wrong = "import java.util.*;\npublic final class Solution { public static Object solve(Map<String,Object> d){ int y=((Number)d.get(\"year\")).intValue(); return y%4==0; } }\n"
    p = pub(run_pub(art("J1-02", "java", wrong, "known_wrong:no_century_rule")), "public_examples_v1")
    assert p["status"] == "fail" and p["failure_class"] == "wrong_answer", p
    ce = p["counterexample"]
    assert json.loads(ce["input"]) == {"year": 1900} and ce["expected"] == "false" and ce["observed"] == "true", ce


@test
def java_compile_runtime_timeout_missing_classes():
    o = run_pub(art("J1-01", "java", "public final class Solution { broken \n"))
    assert pub(o, "parse_v1")["status"] == "fail" and pub(o, "parse_v1")["failure_class"] == "compile_error" and "stderr_tail" in pub(o, "parse_v1")
    assert pub(o, "public_examples_v1")["status"] == "missing_evidence"
    pre = "import java.util.*;\npublic final class Solution { public static Object solve(Map<String,Object> d){ "
    p = pub(run_pub(art("J1-01", "java", pre + "throw new IllegalStateException(\"boom\"); } }\n")), "public_examples_v1")
    assert p["status"] == "fail" and p["failure_class"] == "runtime_error", p
    p = pub(run_pub(art("J1-01", "java", pre + "while(true){} } }\n")), "public_examples_v1")
    assert p["status"] == "timeout" and p["failure_class"] == "timeout", p
    p = pub(run_pub(art("J1-01", "java", pre + "System.exit(0); return null; } }\n")), "public_examples_v1")
    assert p["status"] == "fail" and p["failure_class"] == "missing_output", p
    p = pub(run_pub(art("J1-01", "java", pre + "long[] a = new long[400_000_000]; return a.length; } }\n")), "public_examples_v1")   # -Xmx128m
    assert p["status"] == "fail" and p["failure_class"] == "runtime_error", p
    p = pub(run_pub(art("J1-01", "java", pre + "StringBuilder s=new StringBuilder(); for(int i=0;i<1<<20;i++)s.append(\"0123456789\"); System.out.println(s); System.out.println(s); return true; } }\n")), "public_examples_v1")
    assert p["status"] != "pass" and p["failure_class"] == "invalid_output", p


@test
def java_utf8_pinned_and_non_ascii_task():
    probe = ("import java.util.*;\npublic final class Solution { public static Object solve(Map<String,Object> d){ return System.getProperty(\"file.encoding\")+\"|\"+"
             "System.getProperty(\"stdout.encoding\")+\"|\"+java.nio.charset.Charset.defaultCharset()+\"|\"+System.getenv(\"LC_ALL\"); } }\n")
    o = BE.run_experiment(art("J1-09", "java", probe), {"kind": "function_call", "payload": json.dumps({"text": "x", "shift": 1})})
    assert o["status"] == "pass" and o["observed"] == "UTF-8|UTF-8|UTF-8|C.UTF-8", o
    t = JAVA_TASKS["J1-09"]
    assert any(ord(ch) > 127 for c in t.public_tests for ch in json.dumps(c, ensure_ascii=False)), "J1-09 should contain non-ASCII public input"
    p = pub(run_pub(art("J1-09", "java", java_ref_src("J1-09"), "reference")), "public_examples_v1")
    assert p["status"] == "pass", p


# ------------------------------------------------------------------ hostile candidates
HOST = str(REPO / "CLAUDE.md")
PY_PRE = ("import os, sys\ndef log(m):\n    sys.stderr.write(m + '\\n'); sys.stderr.flush()\n"
          "def t(label, fn):\n    try:\n        r = fn(); log('ESCAPED ' + label + ' -> ' + str(r)[:60])\n    except BaseException as e:\n        log('BLOCKED ' + label + ' ' + type(e).__name__)\n")
PY_HOSTILE = {
    "read_host_files": PY_PRE + f"t('host file', lambda: open({HOST!r}).read(5))\nt('etc passwd', lambda: open('/etc/passwd').read(5))\nt('list /mnt', lambda: os.listdir('/mnt'))\n"
                       "t('list /home', lambda: os.listdir('/home'))\nt('list /root', lambda: os.listdir('/root'))\ndef solve(d): return -1\n",
    "env_canary": PY_PRE + "def leak():\n    for p in ('/proc/1/environ', '/proc/self/environ'):\n        try:\n            if b'canary-do-not-leak' in open(p, 'rb').read(): return p\n        except OSError: pass\n"
                  "    raise KeyError('no canary')\nt('canary', leak)\nt('env var', lambda: os.environ['EL_SECRET_CANARY'])\ndef solve(d): return -1\n",
    "network": PY_PRE + "import socket\nt('connect 1.1.1.1', lambda: socket.create_connection(('1.1.1.1', 80), timeout=2))\nt('connect localhost', lambda: socket.create_connection(('127.0.0.1', 22), timeout=2))\n"
              "t('dns', lambda: socket.gethostbyname('example.com'))\ndef solve(d): return -1\n",
    "write_outside": PY_PRE + "for p in ('/usr/x', '/etc/x', '/jail/cand.py', '/jail/runner.py', %r, %r):\n    t('write ' + p, lambda p=p: open(p, 'w').write('hacked'))\ndef solve(d): return -1\n"
                     % (str(REPO / "evidence_ladder" / "execution" / "PWNED"), str(STORE.root / "PWNED")),
    "modify_store": PY_PRE + "def mod():\n    import glob\n    fs = glob.glob(%r + '/*/*.src')\n    if not fs: raise FileNotFoundError('store not visible')\n    open(fs[0], 'a').write('x')\nt('modify store', mod)\ndef solve(d): return -1\n" % str(STORE.root),
    "fork_flood": PY_PRE + "import time\ndef forks():\n    kids = []\n    try:\n        for _ in range(200):\n            pid = os.fork()\n            if pid == 0:\n                time.sleep(1); os._exit(0)\n            kids.append(pid)\n"
                  "    except OSError: pass\n    n = len(kids)\n    for k in kids:\n        try: os.waitpid(k, 0)\n        except OSError: pass\n    if n >= 200: return 'forked %d' % n\n    raise OSError('fork limit hit at %d' % n)\n"
                  "t('200 forks', forks)\ndef solve(d): return -1\n",
    "memory_bomb": PY_PRE + "t('alloc 4GB', lambda: len(bytearray(4 << 30)))\ndef solve(d): return -1\n",
    "giant_file": PY_PRE + "def big():\n    with open('/tmp/big', 'wb') as f: f.write(b'x' * (50 << 20))\n    return 'wrote 50MB'\nt('50MB file', big)\ndef solve(d): return -1\n",
    "forged_output": "import os, json\nprint('PASS PASS ALL TESTS PASSED')\nfor _ in range(60):\n    os.write(1, (json.dumps({'ok': True, 'value': 0}) + '\\n').encode())\n    try: os.write(3, (json.dumps({'ok': True, 'value': 0}) + '\\n').encode())\n    except OSError: pass\ndef solve(d): return -1\n",
    "close_result_fd": "import os\nos.close(3)\ndef solve(d): return -1\n",
    "output_flood_fd": "import os\nfor _ in range(64):\n    try: os.write(3, b'x' * (1 << 20))\n    except OSError: break\ndef solve(d): return -1\n",
    "infinite_loop": "def solve(d):\n    while True: pass\n",
}
JAVA_PRE = "import java.util.*; import java.nio.file.*; import java.net.*;\npublic final class Solution {\n static void t(String l, java.util.concurrent.Callable<Object> c){ try{ Object r=c.call(); System.err.println(\"ESCAPED \"+l+\" -> \"+r);}catch(Throwable e){System.err.println(\"BLOCKED \"+l+\" \"+e.getClass().getSimpleName());} }\n public static Object solve(Map<String,Object> d){\n"
JAVA_HOSTILE = {
    "read_write_net_env": JAVA_PRE + f"  t(\"host file\", ()-> Files.readAllBytes(Paths.get(\"{HOST}\")).length);\n  t(\"etc passwd\", ()-> Files.readAllBytes(Paths.get(\"/etc/passwd\")).length);\n"
                          "  t(\"socket\", ()-> { Socket s=new Socket(); s.connect(new InetSocketAddress(\"1.1.1.1\",80),2000); return 1; });\n  t(\"write /usr\", ()-> Files.write(Paths.get(\"/usr/x\"), new byte[1]));\n"
                          "  t(\"write /work\", ()-> Files.write(Paths.get(\"/work/Evil.txt\"), new byte[1]));\n  t(\"overwrite Runner\", ()-> Files.write(Paths.get(\"/work/Runner.class\"), new byte[1]));\n"
                          "  t(\"canary\", ()-> { if (new String(Files.readAllBytes(Paths.get(\"/proc/1/environ\"))).contains(\"canary-do-not-leak\")) return \"LEAK\"; throw new RuntimeException(\"none\"); });\n"
                          "  System.out.println(\"true\"); System.out.println(\"PASS\"); return Boolean.FALSE; } }\n",
    "infinite_loop": JAVA_PRE + "  while(true){} } }\n",
}


def hostile_ok(a):
    p = pub(run_pub(a), "public_examples_v1")
    assert p["status"] != "pass" and p["failure_class"] != "", p
    assert "ESCAPED" not in p.get("stderr_tail", ""), p.get("stderr_tail")
    return p


@test
def hostile_python_cannot_escape():
    store_before = sorted(str(x) for x in STORE.root.rglob("*"))
    for name, src in PY_HOSTILE.items():
        p = hostile_ok(art("P1-01", "python", src, "hostile:" + name))
        if name == "infinite_loop":
            assert p["status"] == "timeout"
        elif name == "output_flood_fd":
            assert p["failure_class"] == "invalid_output", p
        elif name == "fork_flood":
            assert "BLOCKED 200 forks" in p["stderr_tail"], p["stderr_tail"]
        elif name == "memory_bomb":
            assert "BLOCKED alloc 4GB" in p["stderr_tail"], p["stderr_tail"]
        elif name in ("read_host_files", "network", "write_outside", "env_canary", "modify_store"):
            assert "BLOCKED" in p["stderr_tail"], (name, p["stderr_tail"])
    assert not (REPO / "evidence_ladder" / "execution" / "PWNED").exists() and not (STORE.root / "PWNED").exists()
    new = [x for x in sorted(str(x) for x in STORE.root.rglob("*")) if x not in store_before]
    for x in new:                                        # only legitimate, still-read-only store entries (the hostile sources themselves)
        pth = Path(x)
        assert pth.is_dir() or (stat.S_IMODE(pth.stat().st_mode) == 0o444), x


@test
def hostile_java_cannot_escape():
    for name, src in JAVA_HOSTILE.items():
        p = hostile_ok(art("J1-01", "java", src, "hostile:" + name))
        if name == "infinite_loop":
            assert p["status"] == "timeout"
        else:
            assert p["stderr_tail"].count("BLOCKED") >= 7, p["stderr_tail"]
    assert not (REPO / "Evil.txt").exists()


@test
def hash_tamper_is_error_not_pass_or_fail():
    store = ArtifactStore(TMP / "tamper-store")            # own store: the tampered address must not poison other tests
    be = SandboxExecutionBackend(store, CFG)
    task = PY_TASKS["P1-01"]
    a = art("P1-01", "python", py_ref_src("P1-01"), "reference")
    assert pub(be.run_public(a, task), "public_examples_v1")["status"] == "pass"
    f = store.path_of(a.artifact_id)
    assert stat.S_IMODE(f.stat().st_mode) == 0o444
    os.chmod(f, 0o644)
    f.write_text("def solve(d): return 0\n")              # swap the stored bytes under the same address
    os.chmod(f, 0o444)
    for o in be.run_public(a, task):
        assert o["status"] == "error" and o["failure_class"] == "sandbox_error", o
    e = be.run_experiment(a, {"kind": "function_call", "payload": "{\"text\": \"a\"}"})
    assert e["status"] == "error" and e["failure_class"] == "sandbox_error", e
    g = SelectionGate(); g.close()
    try:
        ProtectedGrader(g, store, CFG).grade(a)
        raise AssertionError("grader accepted a tampered artifact")
    except ArtifactIntegrityError:
        pass
    forged = Artifact("0" * 64, "P1-01", "python", "def solve(d): return 1\n")        # id does not match source
    assert all(o["status"] == "error" for o in be.run_public(forged, task))
    b = art("P1-02", "python", py_ref_src("P1-02"), "reference")                      # a different artifact is unaffected
    assert pub(be.run_public(b, PY_TASKS["P1-02"]), "public_examples_v1")["status"] == "pass"


@test
def candidates_are_isolated_from_each_other():
    tmpdir = Path(tempfile.gettempdir())
    leftovers = lambda: {p.name for p in tmpdir.iterdir() if p.name.startswith(("jail-", "jjail-"))}      # noqa: E731
    before = leftovers()
    a = art("P1-01", "python", PY_PRE + "open('/tmp/marker', 'w').write('A')\nos.environ['EL_X'] = 'A'\nt('mod store', lambda: open(%r, 'a').write('x'))\ndef solve(d): return -1\n" % str(STORE.root / "zz"), "hostile:A")
    b = art("P1-01", "python", PY_PRE + "t('see marker', lambda: open('/tmp/marker').read())\nt('see env', lambda: os.environ['EL_X'])\ndef solve(d): return sum(c in 'aeiouAEIOU' for c in d['text'])\n", "sample:B")
    pa = pub(run_pub(a), "public_examples_v1")
    sha_a = STORE.path_of(a.artifact_id).read_bytes()
    pb = pub(run_pub(b), "public_examples_v1")
    assert pb["status"] == "pass" and "ESCAPED" not in pb["stderr_tail"] and pb["stderr_tail"].count("BLOCKED") == 2, pb
    assert STORE.path_of(a.artifact_id).read_bytes() == sha_a and pa["status"] != "pass"
    assert leftovers() == before, leftovers() - before


# ------------------------------------------------------------------ experiments
@test
def experiments():
    ref = art("P1-01", "python", py_ref_src("P1-01"), "reference")
    for kind in ("function_call", "stdin_input"):
        o = BE.run_experiment(ref, {"kind": kind, "payload": json.dumps({"text": "hello world"})})
        assert o["status"] == "pass" and o["observed"] == 3 and o["check_id"] == "critic_experiment_v1", o
    for kind in ("command", "state_query", "bogus"):
        o = BE.run_experiment(ref, {"kind": kind, "payload": "ls"})
        assert o["status"] == "invalid_spec", o
    for payload in ("{not json", "[1]", "x" * 20001, json.dumps({"text": "a" * 20000})):
        o = BE.run_experiment(ref, {"kind": "function_call", "payload": payload})
        assert o["status"] == "invalid_spec", (payload[:20], o)
    assert BE.run_experiment(ref, {"kind": "function_call", "payload": 5})["status"] == "invalid_spec"
    o = BE.run_experiment(ref, {"kind": "function_call", "payload": "{}"})            # solve raises KeyError
    assert o["status"] == "fail" and o["failure_class"] == "runtime_error" and o["observed"] == "<error: KeyError>", o
    big = art("P1-01", "python", "def solve(d): return 'y' * 100000\n")
    o = BE.run_experiment(big, {"kind": "function_call", "payload": "{}"})
    assert o["status"] == "pass" and len(o["observed"]) < 2100 and o["observed"].endswith("[truncated]"), len(o["observed"])
    o = BE.run_experiment(art("P1-01", "python", "def solve(d):\n    while True: pass\n"), {"kind": "function_call", "payload": "{}"})
    assert o["status"] == "timeout", o
    jref = art("J1-01", "java", java_ref_src("J1-01"), "reference")
    o = BE.run_experiment(jref, {"kind": "function_call", "payload": json.dumps({"text": "abba"})})
    assert o["status"] == "pass" and o["observed"] is True, o


# ------------------------------------------------------------------ isolation of the hidden grader
@test
def static_no_protected_import():
    root = REPO / "evidence_ladder" / "execution"
    checked = []
    for f in sorted(root.glob("*.py")):
        if f.name == "protected.py":
            continue
        text = f.read_text(encoding="utf-8")
        assert "protected_cases" not in text, f.name
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.Import):
                assert not any("protected" in a.name for a in n.names), f.name
            if isinstance(n, ast.ImportFrom):
                assert "protected" not in (n.module or "") and not any("protected" in a.name for a in n.names), f.name
        checked.append(f.name)
    assert {"backend.py", "engine.py", "artifacts.py", "receipts.py", "config.py", "__init__.py", "taskload.py"} <= set(checked), checked
    assert "protected_cases" in (root / "protected.py").read_text(encoding="utf-8")
    assert "grader" not in {f.name for f in dataclasses.fields(ControllerContext)}
    code = "import sys; import evidence_ladder.execution.backend, evidence_ladder.execution.receipts; sys.exit(1 if 'evidence_ladder.execution.protected' in sys.modules else 0)"
    assert os.system(f"cd {str(REPO)!r} && python3 -c \"{code}\"") == 0           # importing the backend must not load protected.py


@test
def gate_open_raises_closed_grades_and_shape():
    gate = SelectionGate()
    gr = ProtectedGrader(gate, STORE, CFG)
    a = art("P1-01", "python", py_ref_src("P1-01"), "reference")
    try:
        gr.grade(a)
        raise AssertionError("graded while the gate was open")
    except GraderIsolationError:
        pass
    gate.close()
    h = gr.grade(a)
    assert set(h) == set(HiddenGrade.__annotations__) and h["passed"] is True and h["passed_tests"] == h["total_tests"] == 47 and h["failure"] is None, h
    assert h["sha256_before_run"] == a.artifact_id and h["artifact_id"] == a.artifact_id
    for bad in ("def solve(d): return 1", "def solve(d) return", "def solve(d):\n    while True: pass\n"):
        r = gr.grade(art("P1-01", "python", bad))
        assert r["passed"] is False and r["failure"] in ("WRONG_ANSWER", "COMPILE_ERROR", "TIMEOUT"), r
        assert set(r) == set(HiddenGrade.__annotations__)           # counts/boolean only
    assert gr.grade(art("P9-99", "python", "def solve(d): return 1"))["failure"] == "NO_PROTECTED_CASES"
    gate2 = SelectionGate()
    assert gate2.is_open and gate.is_open is False


@test
def protected_python_40_references_and_all_wrong():
    gate = SelectionGate(); gate.close()
    gr = ProtectedGrader(gate, STORE, K.SandboxConfig(py_timeout=10.0))
    ok = [t for t in PY_TASKS if gr.grade(art(t, "python", py_ref_src(t), "reference"))["passed"]]
    assert len(ok) == 40, sorted(set(PY_TASKS) - set(ok))
    NOTES["python_references_protected_passed"] = len(ok)
    escaped = []
    n = 0
    for tid, lst in wrong_py().items():
        for name, src in lst:
            n += 1
            if gr.grade(art(tid, "python", src, "known_wrong:" + name))["passed"]:
                escaped.append(f"{tid}:{name}")
    assert not escaped, escaped
    NOTES["python_known_wrong_protected_caught"] = f"{n}/{n}"


@test
def protected_java_40_references():
    gate = SelectionGate(); gate.close()
    gr = ProtectedGrader(gate, STORE, K.SandboxConfig(java_run_timeout=10.0))
    ok = []
    for t in JAVA_TASKS:
        h = gr.grade(art(t, "java", java_ref_src(t), "reference"))
        if h["passed"]:
            ok.append(t)
        else:
            NOTES.setdefault("java_ref_failures", []).append((t, h["failure"], h["passed_tests"], h["total_tests"]))
    assert len(ok) == 40, NOTES.get("java_ref_failures")
    NOTES["java_references_protected_passed"] = len(ok)
    wrong = "import java.util.*;\npublic final class Solution { public static Object solve(Map<String,Object> d){ int y=((Number)d.get(\"year\")).intValue(); return y%4==0; } }\n"
    h = gr.grade(art("J1-02", "java", wrong, "known_wrong"))
    assert h["passed"] is False and h["failure"] == "WRONG_ANSWER", h


# ------------------------------------------------------------------ receipts
@test
def existing_receipts_intact():
    r = replay_existing_receipts()
    bad = {k: v for k, v in r["files"].items() if not v["match"]}
    assert r["intact"] and not bad, (bad, r["summaries"])
    NOTES["replayed_receipt_files"] = len(r["files"])
    NOTES["java_redteam_escaped_markers_in_old_receipt"] = r["summaries"]["java_redteam"]["escaped_markers_in_its_own_log"]


@test
def isolation_receipt_contents():
    r = isolation_receipt(CFG)
    j = r["python_jail"]
    assert j["uid"] == 65534 and j["write_usr"].startswith("blocked") and j["write_etc"].startswith("blocked") and j["network"].startswith("blocked"), j
    assert j["write_tmp"] == "ALLOWED" and set(j["env_keys"]) <= {"PATH", "PYTHONHASHSEED", "LC_CTYPE", "PWD"}, j      # only the empty tmpfs is writable
    assert all(j["namespaces_differ_from_host"][n] for n in ("user", "pid", "net", "ipc", "uts")), j["namespaces_differ_from_host"]
    assert j["rlimits"]["NPROC"] == [32, 32] and j["rlimits"]["AS"] == [1 << 30, 1 << 30], j["rlimits"]
    assert r["java_toolchain"]["javac"] and "error" not in str(r["java_toolchain"]).lower(), r["java_toolchain"]
    NOTES["isolation_receipt"] = {"uid": j["uid"], "python": j["python"], "java": r["java_toolchain"]}
    info = BE.backend_info()
    assert info["python_jail"]["uid"] == 65534


def main() -> int:
    failed = {}
    t0 = time.monotonic()
    for f in TESTS:
        t = time.monotonic()
        try:
            f()
        except BaseException:
            failed[f.__name__] = traceback.format_exc()[-1500:]
        NOTES.setdefault("seconds", {})[f.__name__] = round(time.monotonic() - t, 1)
    out = {"tests": len(TESTS), "passed": len(TESTS) - len(failed), "failed": failed, "notes": NOTES, "total_seconds": round(time.monotonic() - t0, 1)}
    print(json.dumps(out))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
