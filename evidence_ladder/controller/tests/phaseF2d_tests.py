"""[EL-ARCH-1 F2d / Agent 2] PHASE 2: trusted verifier (evidence_ladder/execution/trusted_verify.py, engine version trusted_verify_v1) + the WSLC audit side (conformance --expect-engine mxc-wslc, offline replay
check, security-gate checker, final-report generator, model-evidence / I4 negative tests). Run from the repo root:
    PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe evidence_ladder/controller/tests/phaseF2d_tests.py
The python malicious-candidate tests execute REAL candidate code in a real subprocess through a local port (cwd = a private workspace, minimal environment). That port is NOT a security sandbox: what the tests prove
is VERDICT ADMISSION: whatever the candidate does to files, stdout, its exit status, the runner files or the network, a fabricated pass is not accepted. Containment itself (no network, no host files) is the backend's
and is tested on the real MXC sandbox when it is available (skipped with the reason otherwise). Resource exhaustion (memory / fork) is NOT run for real on this PC: those cases are scripted through the sandbox port
(killed / exit codes the real backend reports) and assert that the verdict is never a pass. Java uses a scripted port (no JDK on this host).
Names: *_logic = real code, fakes where stated; *_mxc = real MXC sandbox."""
import contextlib
import hashlib
import inspect
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phaseB_tests as B                                                                  # noqa: E402
import run_tests as RT                                                                     # noqa: E402
from evidence_ladder.execution import config as K, engine, trusted_verify as TV            # noqa: E402

TESTS, SKIPPED = {}, {}
CFG = K.SandboxConfig()


def test(f):
    TESTS[f.__name__] = f
    return f


# ====================================================================== local port (NOT a sandbox) + helpers
class LocalPort:
    """sandbox.run(workspace, cmd, persist=False, timeout=60) that really starts the process (python only). Records every call and, at call time, a listing of the workspace files with their bytes."""

    def __init__(self, expected_probe=None):
        self.calls, self.staged, self.probe = [], [], expected_probe

    def __call__(self, workspace, cmd, persist=False, timeout=60):
        ws = Path(workspace)
        self.calls.append({"cmd": cmd, "persist": persist, "timeout": timeout})
        self.staged.append({p.name: p.read_bytes() for p in ws.iterdir() if p.is_file()})
        argv = shlex.split(cmd, posix=True)
        if argv[0] == "python3":
            argv[0] = sys.executable
        env = {k: v for k, v in os.environ.items() if k in ("SystemRoot", "PATH", "TEMP", "TMP", "PYTHONHASHSEED")}
        p = subprocess.Popen(argv, cwd=str(ws), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        killed = False
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            killed = True
        return {"exit_code": -9 if killed else p.returncode, "killed": killed, "stdout": out.decode("utf-8", "replace"), "stderr": err.decode("utf-8", "replace"), "protected_unchanged": True,
                "pre_tree_sha256": "a" * 64, "post_tree_sha256": "b" * 64, "persisted": False}


def verifier(port=None, **kw):
    port = port or LocalPort()
    return TV.TrustedVerifier(port, CFG, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), **{"launch_overhead": 0.0, **kw}), port


def verdict(code, cases, tv=None, **kw):
    """-> (judge_each verdict, per-case list, RunResult, port): the full host-side path: the candidate sees inputs only, the comparison with `expected` happens in engine.judge_each"""
    tv, port = (tv, tv._run) if tv else verifier(**kw)
    res = tv.run("python", code.encode("utf-8"), [c["input"] for c in cases])
    v, each = engine.judge_each("python", res, cases, CFG)
    return v, each, res, port


CASES = [{"input": {"x": i}, "expected": i * 2} for i in (1, 2, 3)]
HONEST = "def solve(d):\n    return d['x'] * 2\n"


def never_pass(v, each):
    assert v.status != "pass" and all(s != "pass" for s, _, _ in each), (v, each)


def need_py_mod():
    if not K.PY_GRADE_SCRIPT.exists():
        raise B.Skip("the receipted python grading module (strict_equal) is not on this machine: %s" % K.PY_GRADE_SCRIPT)


# ====================================================================== v2 time accounting (2026-10-20): launch overhead is not the candidate's time; harness budget is infrastructure
class SlowLaunchPort(LocalPort):
    """LocalPort that adds a fixed delay per launch (simulates WSLC / JVM start-up)."""
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def __call__(self, workspace, cmd, persist=False, timeout=60):
        time.sleep(self.delay)
        return super().__call__(workspace, cmd, persist, timeout)


@test
def tv_v2_launch_overhead_is_not_charged_to_the_candidate_logic():
    need_py_mod()
    cfg = K.SandboxConfig(py_timeout=1.0)                     # legacy batch limit 1 s; 20 inputs x 0.15 s launch = 3 s of pure overhead (v1 called this a candidate timeout)
    port = SlowLaunchPort(0.15)
    tv = TV.TrustedVerifier(port, cfg, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), per_input_timeout=1.0)      # overhead calibrated on this port
    cases = [{"input": {"x": i}, "expected": i * 2} for i in range(20)]
    res = tv.run("python", HONEST.encode("utf-8"), [c["input"] for c in cases])
    v, _ = engine.judge_each("python", res, cases, cfg)
    assert v.status == "pass", (v, tv.accounting)
    assert tv.accounting["launch_overhead_s"] >= 0.15 and tv.accounting["candidate_s"] < cfg.py_timeout and tv.accounting["outcome"] == "ran"
    assert any(c["cmd"] == "python3 -c 0" for c in port.calls)                    # calibration ran on the same port


@test
def tv_v2_four_outcomes_are_kept_apart_logic():
    need_py_mod()
    cases = [{"input": {"x": i}, "expected": i * 2} for i in range(6)]
    slow = "import time\ndef solve(d):\n    time.sleep(0.4)\n    return d['x'] * 2\n"
    # candidate compute total above the legacy batch limit -> candidate timeout
    tv = TV.TrustedVerifier(LocalPort(), K.SandboxConfig(py_timeout=1.0), scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), per_input_timeout=5.0, launch_overhead=0.0)
    v, _ = engine.judge_each("python", tv.run("python", slow.encode(), [c["input"] for c in cases]), cases, CFG)
    assert v.status == "timeout" and v.failure_class == "timeout" and tv.accounting["outcome"] == "candidate_total_timeout", tv.accounting
    # one input above the per-input kill -> candidate timeout
    tv = TV.TrustedVerifier(LocalPort(), K.SandboxConfig(py_timeout=30.0), scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), per_input_timeout=1.0, launch_overhead=0.0)
    v, _ = engine.judge_each("python", tv.run("python", b"import time\ndef solve(d):\n    time.sleep(20)\n", [c["input"] for c in cases]), cases, CFG)
    assert v.status == "timeout" and tv.accounting["outcome"] == "per_input_timeout", tv.accounting
    # harness budget exhausted -> sandbox_error (infrastructure), never a candidate verdict
    tv = TV.TrustedVerifier(SlowLaunchPort(0.3), K.SandboxConfig(py_timeout=30.0), scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), per_input_timeout=5.0, total_timeout=0.5, launch_overhead=0.0)
    v, _ = engine.judge_each("python", tv.run("python", HONEST.encode(), [c["input"] for c in cases]), cases, CFG)
    assert v.status == "error" and v.failure_class == "sandbox_error" and tv.accounting["outcome"] == "harness_budget", (v, tv.accounting)
    # sandbox port failure -> raises
    def dead(*a, **k):
        raise OSError("port down")
    tv = TV.TrustedVerifier(dead, CFG, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), launch_overhead=0.0)
    try:
        tv.run("python", HONEST.encode(), [1])
        raise AssertionError("no raise")
    except TV.TrustedVerifyInfraError:
        pass


# ====================================================================== PHASE 2: trusted verifier, python
@test
def tv_honest_candidate_passes_and_expected_values_never_reach_the_sandbox_logic():
    need_py_mod()
    canary = [{"input": {"x": i}, "expected": "CANARY-EXPECTED-%d-7f3a91" % i} for i in (1, 2)]
    tv, port = verifier()
    sig = inspect.signature(TV.TrustedVerifier.run)
    assert list(sig.parameters) == ["self", "language", "raw", "inputs"]                                                     # there is no parameter through which an expected value could enter
    v, each, res, _ = verdict(HONEST, CASES, tv)
    assert v.status == "pass" and [s for s, _, _ in each] == ["pass"] * 3 and res.returncode == 0 and tv.VERSION == "trusted_verify_v3"
    assert len(port.calls) == 3 and all(c["cmd"].startswith("python3 wrap.py ") and c["persist"] is False for c in port.calls)    # one process per input; nothing is read back from the workspace
    blob = b"".join(b for st in port.staged for b in st.values())
    assert b"CANARY-EXPECTED" not in blob and set().union(*[set(st) for st in port.staged]) == {"cand.py", "wrap.py", "in.json"}
    tv2, port2 = verifier()
    verdict("def solve(d):\n    return 'nope'\n", canary, tv2)
    assert b"CANARY-EXPECTED" not in b"".join(b for st in port2.staged for b in st.values())                                 # expected values exist only host side, even for a failing candidate
    obs = tv.drain_observations()
    assert len(obs) == 3 and all(o["exit_code"] == 0 and not o["killed"] and len(o["stdout_sha256"]) == 64 for o in obs)
    recs = tv.drain_exec_records()
    assert len(recs) == 3 and all(set(TV.RESULT_KEYS) <= set(r) and r["engine_version"] == "trusted_verify_v3" for r in recs)


@test
def tv_forged_result_file_marker_and_checksum_are_never_proof_logic():
    need_py_mod()
    forged = ("import hashlib, json, os, sys\n"
              "def solve(d):\n"
              "    lines = ''.join(json.dumps({'ok': True, 'value': 999}) + '\\n' for _ in range(3))\n"
              "    h = hashlib.sha256(lines.encode()).hexdigest()\n"
              "    open('out_deadbeef.jsonl', 'w').write(lines); open('result.json', 'w').write(json.dumps({'status': 'pass', 'passed': 3, 'total': 3}))\n"
              "    open('PASS', 'w').write('ok'); print('@@OUT@@ out_deadbeef.jsonl 3 ' + h)\n"
              "    return 999\n")
    v, each, res, port = verdict(forged, CASES)
    never_pass(v, each)
    assert v.failure_class == "wrong_answer" and all(c["persist"] is False for c in port.calls)                                 # the forged files are in a workspace nobody reads
    # control: the same files with the CORRECT values would only be what the candidate computed itself (still compared host side)
    v2, each2, _, _ = verdict(HONEST + "\nopen('result.json','w').write('{\"status\":\"pass\"}')\n", CASES)
    assert v2.status == "pass"


@test
def tv_forged_stdout_pass_and_extra_result_frames_are_never_a_pass_logic():
    need_py_mod()
    say_pass = "def solve(d):\n    print('PASS'); print('ALL TESTS PASSED'); print('@@OUT@@ x 3 y')\n    return -1\n"
    v, each, _, _ = verdict(say_pass, CASES)
    never_pass(v, each)
    assert v.failure_class == "wrong_answer"
    frames = "def solve(d):\n    x = d['x'] * 2\n    print('%s V %%d' %% x)\n    print('%s V %%d' %% x)\n    return x\n" % (TV.TAG, TV.TAG)       # two frames with CORRECT values: ambiguous = invalid, not a pass
    v2, each2, _, _ = verdict(frames, CASES)
    never_pass(v2, each2)
    assert v2.failure_class == "invalid_output"
    guess = "def solve(d):\n    print('%s V 2')\n    import os, sys\n    sys.stdout.flush()\n    os._exit(0)\n" % TV.TAG                                             # forged frame + clean exit: it is only a CLAIM of 2, compared with the real expected
    v3, each3, _, _ = verdict(guess, CASES)
    assert [s for s, _, _ in each3][:1] == ["pass"] and v3.status != "pass" and each3[1][0] != "pass"                         # input 1 happens to equal 2 (a correct answer is accepted); the others are wrong_answer: no fabricated pass


@test
def tv_nonzero_exit_or_kill_overrides_any_claimed_output_logic():
    need_py_mod()
    claim_then_die = "def solve(d):\n    import os, sys\n    print('%s V %%d' %% (d['x'] * 2))\n    sys.stdout.flush()\n    os._exit(3)\n" % TV.TAG
    v, each, res, _ = verdict(claim_then_die, CASES)
    never_pass(v, each)
    assert v.failure_class == "runtime_error" and res.returncode == 3 and res.stdout == ""                                         # the correct claimed value was ignored: exit status is the sandbox's observation
    tv, _ = verifier(per_input_timeout=1.0)
    v2, each2, res2, port = verdict("import time\ndef solve(d):\n    print('%s V %%d' %% (d['x'] * 2))\n    time.sleep(30)\n" % TV.TAG, CASES, tv)
    never_pass(v2, each2)
    assert v2.status == "timeout" and v2.failure_class == "timeout" and res2.timed_out and len(port.calls) == 1                      # killed at the first input, later inputs never run


@test
def tv_modified_runner_files_and_leftover_state_cannot_affect_other_inputs_logic():
    need_py_mod()
    vandal = ("import os\n"
              "def solve(d):\n"
              "    seen = os.path.exists('marker')\n"
              "    open('marker', 'w').write('x')\n"
              "    open('wrap.py', 'w').write('import sys; sys.exit(0)')\n"
              "    for n in ('in.json',):\n"
              "        try: os.remove(n)\n"
              "        except OSError: pass\n"
              "    return 'stale' if seen else d['x'] * 2\n")
    v, each, _, port = verdict(vandal, CASES)
    assert v.status == "pass"                                                                                                 # every input ran in a freshly staged workspace: no leftover marker, host-written wrap.py each time
    assert all(st["wrap.py"].decode() == TV.PY_WRAPPER for st in port.staged)
    # tampering with wrap.py cannot turn a failure into a pass either: a candidate that only rewrites wrap.py to exit 0 emits no frame
    killer = "import os\nopen('wrap.py', 'w').write('print(1)')\ndef solve(d):\n    return -5\n"
    v2, each2, _, _ = verdict(killer, CASES)
    never_pass(v2, each2)


@test
def tv_candidate_cannot_find_expected_values_anywhere_it_can_read_logic():
    need_py_mod()
    cases = [{"input": {"x": i}, "expected": "SECRET-EXPECTED-%d-c0ffee9" % i} for i in (1, 2, 3)]
    scanner = ("import os\n"
               "def solve(d):\n"
               "    roots = {os.getcwd(), os.path.dirname(os.getcwd()), os.path.dirname(os.path.dirname(os.getcwd())), os.environ.get('TEMP', '.')}\n"
               "    for r in roots:\n"
               "        for dp, dn, fn in os.walk(r):\n"
               "            for f in fn:\n"
               "                try:\n"
               "                    t = open(os.path.join(dp, f), 'rb').read(200000)\n"
               "                except OSError:\n"
               "                    continue\n"
               "                i = t.find(b'SECRET-EXPECTED-%d-' % d['x'])\n"
               "                if i >= 0:\n"
               "                    return t[i:i + 28].decode()\n"
               "    return 'not found'\n")
    tv, _ = verifier(per_input_timeout=20.0)
    v, each, res, _ = verdict(scanner, cases, tv)
    never_pass(v, each)
    assert all(o == "not found" for o in [json.loads(l)["value"] for l in res.stdout.splitlines()])                            # nothing readable from its workspace, its parents or TEMP holds an expected value


@test
def tv_ordinary_candidate_failures_stay_candidate_outcomes_and_infra_failures_raise_logic():
    need_py_mod()
    v, each, _, _ = verdict("def solve(d)\n    return 1\n", CASES)
    assert v.failure_class == "compile_error" and all(s == "fail" for s, _, _ in each)                                         # SyntaxError at load: a candidate failure
    v, each, _, _ = verdict("def solve(d):\n    return d['x'] * 3\n", CASES)
    assert v.failure_class == "wrong_answer"
    v, each, _, _ = verdict("def solve(d):\n    raise ValueError('boom')\n", CASES)
    assert v.failure_class == "runtime_error" and each[0][2].startswith("<error: ValueError")
    v, each, _, _ = verdict("import os\ndef solve(d):\n    os._exit(0)\n", CASES)
    never_pass(v, each)                                                                                                       # exits cleanly without ever emitting a frame: missing output
    big = "def solve(d):\n    print('A' * 5000000)\n    return 1\n"
    v, each, res, _ = verdict(big, CASES)
    never_pass(v, each)
    assert res.overflow and v.failure_class == "invalid_output"
    # infrastructure: the port raises -> raised, never a verdict; a port that returns a result without the contract fields is also infrastructure
    def dead(*a, **k):
        raise OSError("sandbox gone")
    tv = TV.TrustedVerifier(dead, CFG, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), launch_overhead=0.0)
    try:
        tv.run("python", HONEST.encode(), [{"x": 1}])
        assert False
    except TV.TrustedVerifyInfraError as e:
        assert e.infra is True
    tv2 = TV.TrustedVerifier(lambda *a, **k: {"exit_code": 0}, CFG, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), launch_overhead=0.0)
    try:
        tv2.run("python", HONEST.encode(), [{"x": 1}])
        assert False
    except TV.TrustedVerifyInfraError:
        pass


@test
def tv_resource_exhaustion_reported_by_the_backend_is_never_a_pass_scripted_logic():
    need_py_mod()
    for result in ({"exit_code": 137, "killed": False, "stdout": "", "stderr": "Killed (memory limit)"},             # memory / OOM kill
                   {"exit_code": 1, "killed": False, "stdout": "", "stderr": "MemoryError"},
                   {"exit_code": -1, "killed": False, "stdout": "", "stderr": "fork: Resource temporarily unavailable"},       # fork exhaustion
                   {"exit_code": -9, "killed": True, "stdout": "%s V 2\n" % TV.TAG, "stderr": "wall-clock"}):                     # killed after printing a frame
        full = dict({"protected_unchanged": True, "pre_tree_sha256": "a" * 64, "post_tree_sha256": "b" * 64, "persisted": False}, **result)
        tv = TV.TrustedVerifier(lambda ws, cmd, persist=False, timeout=60, _r=full: dict(_r), CFG, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), launch_overhead=0.0)
        res = tv.run("python", HONEST.encode(), [c["input"] for c in CASES])
        v, each = engine.judge_each("python", res, CASES, CFG)
        never_pass(v, each)
        assert tv.drain_observations()[0]["exit_code"] == result["exit_code"]                                                 # the genuine exit status is recorded


@test
def tv_mxc_real_sandbox_blocks_network_and_host_files_and_admission_still_holds_mxc():
    need_py_mod()
    try:
        from evidence_ladder.controller.sandbox_provider import backend_available, make_sandbox
        av = backend_available("mxc")
    except Exception as e:      # noqa: BLE001
        raise B.Skip("MXC not importable: %s" % e)
    if not av.get("available"):
        raise B.Skip("MXC unavailable here: %s" % av.get("reasons"))
    sb = make_sandbox(Path(tempfile.mkdtemp(prefix="f2d-mxc-")), [], backend="mxc")
    tv = TV.TrustedVerifier(sb.run, CFG, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), per_input_timeout=30.0)
    probe = ("import socket\n"
             "def solve(d):\n"
             "    out = {}\n"
             "    try:\n"
             "        s = socket.create_connection(('127.0.0.1', 9), timeout=2); s.close(); out['net'] = 'open'\n"
             "    except Exception as e:\n"
             "        out['net'] = 'blocked'\n"
             "    try:\n"
             "        open(r'%s', 'rb').read(10); out['host_file'] = 'read'\n"
             "    except Exception:\n"
             "        out['host_file'] = 'blocked'\n"
             "    return out\n" % str(ROOT / "CLAUDE.md").replace("\\", "\\\\"))
    res = tv.run("python", probe.encode(), [{"x": 1}])
    got = json.loads(res.stdout.splitlines()[0])["value"]
    assert got == {"net": "blocked", "host_file": "blocked"}, got
    v, each, res2, _ = verdict(HONEST, CASES, tv)
    assert v.status == "pass"


@test
def tv_java_one_process_per_input_with_host_compiled_runner_scripted_logic():
    """no JDK here: a scripted port stands in for javac / java; what is asserted is the supervisor's behaviour (staging, status handling, admission)"""
    cases = [{"input": {"x": i}, "expected": i * 2} for i in (1, 2, 3)]
    support = Path(tempfile.mkdtemp(prefix="f2d-support-"))
    for n in ("Json.java", "Runner.java"):
        (support / n).write_text("// host support " + n, "utf-8")
    log = []

    def port(mode):
        def run(ws, cmd, persist=False, timeout=60):
            w = Path(ws)
            files = {p.name: p.read_bytes() for p in w.iterdir() if p.is_file()}
            log.append((cmd, persist, files))
            base = {"protected_unchanged": True, "pre_tree_sha256": "a" * 64, "post_tree_sha256": "b" * 64, "persisted": persist}
            if cmd.startswith("javac") and "-cp ." in cmd:
                assert "Solution.java" not in files and files.get("Solution.class") == b"SOL"          # the link step sees host sources + the candidate's CLASS only, never its source
                (w / "Json.class").write_bytes(b"HOST-JSON"); (w / "Runner.class").write_bytes(b"HOST-RUNNER")
                return dict(base, exit_code=0, killed=False, stdout="", stderr="")
            if cmd.startswith("javac"):
                if b"SYNTAXERR" in files["Solution.java"]:
                    return dict(base, exit_code=1, killed=False, stdout="", stderr="Solution.java:1: error")
                (w / "Json.class").write_bytes(b"CANDIDATE-JSON"); (w / "Runner.class").write_bytes(b"CANDIDATE-RUNNER"); (w / "Solution.class").write_bytes(b"SOL")
                return dict(base, exit_code=0, killed=False, stdout="", stderr="")
            x = json.loads(files["in.json"])["x"]
            if mode == "honest":
                return dict(base, exit_code=0, killed=False, stdout="%d\n" % (x * 2), stderr="")
            if mode == "forged_lines":
                return dict(base, exit_code=0, killed=False, stdout="PASS\n%d\n" % (x * 2), stderr="")
            if mode == "claim_then_crash":
                return dict(base, exit_code=134, killed=False, stdout="%d\n" % (x * 2), stderr="Aborted")
            if mode == "timeout":
                return dict(base, exit_code=-9, killed=True, stdout="", stderr="")
            raise AssertionError(mode)
        return run
    def jv(mode, src=b"class Solution {}"):
        tv = TV.TrustedVerifier(port(mode), CFG, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), java_support=support, launch_overhead=0.0)
        res = tv.run("java", src, [c["input"] for c in cases])
        v, each = engine.judge_each("java", res, cases, CFG)
        return tv, res, v, each
    if not K.JAVA_GRADE_SCRIPT.exists():
        raise B.Skip("the receipted java grading module (same) is not on this machine")
    tv, res, v, each = jv("honest")
    assert v.status == "pass" and [s for s, _, _ in each] == ["pass"] * 3
    runs = [(c, p, f) for c, p, f in log if "-cp . Runner" in c]
    assert len(runs) == 3 and all(p is False and "< in.json" in c for c, p, _ in runs)                                         # one java process per input, stdin staged as a file, nothing persisted back
    assert all(f["Runner.class"] == b"HOST-RUNNER" and f["Json.class"] == b"HOST-JSON" and f["Solution.class"] == b"SOL" for _, _, f in runs)    # the candidate's compile step cannot replace the host-compiled Runner / Json
    del log[:]
    _, res, v, each = jv("honest", b"SYNTAXERR")
    assert v.failure_class == "compile_error" and all(s == "fail" for s, _, _ in each)                                         # an ordinary candidate outcome
    _, res, v, each = jv("forged_lines")
    never_pass(v, each)                                                                                                       # a printed PASS line makes two lines: invalid, never a pass
    _, res, v, each = jv("claim_then_crash")
    never_pass(v, each)
    assert v.failure_class == "runtime_error"
    _, res, v, each = jv("timeout")
    never_pass(v, each)
    assert v.status == "timeout"


# ====================================================================== WSLC audit side: conformance --expect-engine mxc-wslc, replay check, security gates, final report, I4 / model evidence
SAVED = ROOT / "results/raw/el-bank40-pass1-out-20261018T1752"
DIGEST = "sha256:" + "d" * 64
SRC = "e" * 64
TOOLCHAIN = {"javac": "javac 21.0.4", "java": "openjdk 21.0.4", "python": "Python 3.12.3"}


def _conf():
    import el_bank_conformance as CF
    return CF


def _wslc_fixture(n=4, tasks=("J1-08", "J1-12")):
    """a WSLC-shaped run built from REAL saved task directories (copied under new ids) + the new provenance fields the WSLC runner / backend write (fixture augmentation, stated)."""
    if not SAVED.exists():
        raise B.Skip("saved bank run not on this machine: %s" % SAVED)
    run = Path(tempfile.mkdtemp(prefix="f2d-wslc-"))
    summary = json.loads((SAVED / "RUN_SUMMARY.json").read_text("utf-8"))
    summary.update(engine="mxc-wslc", engine_status="OK", source_sha256=SRC, profile={"id": "qwen2.5-coder-1.5b-instruct", "sha256": "f" * 64}, l2_version=2)
    (run / "RUN_SUMMARY.json").write_text(json.dumps(summary), "utf-8")
    shutil.copy(SAVED / "TIMELINE.jsonl", run / "TIMELINE.jsonl")
    ids = []
    for i in range(n):
        tid = "J2-%02d" % (i + 1)
        ids.append(tid)
        shutil.copytree(SAVED / "tasks" / tasks[i % len(tasks)], run / "tasks" / tid)
        td = run / "tasks" / tid
        r = json.loads((td / "result.json").read_text("utf-8"))
        r.update(engine="mxc-wslc", l2_version=2, profile_sha256="f" * 64)
        r["l2"] = {"verdict": "VALIDATED", "parsed": 3, "reason": ""}                                                  # the L2 outcome is not what is audited here
        (td / "result.json").write_text(json.dumps(r), "utf-8")
        rec = {"exit_code": 0, "killed": False, "stdout": "x", "stderr": "", "protected_unchanged": True, "pre_tree_sha256": "a" * 64, "post_tree_sha256": "b" * 64, "persisted": False,
               "backend": "mxc-wslc", "image_digest": DIGEST, "toolchain": dict(TOOLCHAIN), "source_sha256": SRC, "language": "java"}
        (td / "exec_results.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n", "utf-8")
    (run / "expected.json").write_text(json.dumps(ids), "utf-8")
    return run, ids


def _wslc_audit(run, ids=None, **kw):
    kw.setdefault("expect", "mxc-wslc")
    kw.setdefault("tasks_path", str(run / "expected.json"))
    kw.setdefault("expect_count", len(json.loads((run / "expected.json").read_text("utf-8"))))
    return _conf().audit_run(run, **kw)


def _edit_exec(run, tid, fn):
    p = run / "tasks" / tid / "exec_results.jsonl"
    recs = [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]
    for r in recs:
        fn(r)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), "utf-8")


def _edit_json(p, fn):
    d = json.loads(p.read_text("utf-8"))
    fn(d)
    p.write_text(json.dumps(d), "utf-8")


@test
def w_wslc_conformance_positive_with_receipts_and_candidate_results_separate_logic():
    run, ids = _wslc_fixture()
    rep = _wslc_audit(run)
    assert rep["verdict"] == "PIPELINE_COMPLETE" and rep["counts"]["executed"] == 4 and rep["counts"]["broken"] == 0, rep["broken"]
    c = rep["wslc"]["criteria"]
    assert c["zero_not_tested"] and c["zero_not_tested_no_jdk"] and c["zero_silent_fallback"] and c["all_receipts_complete"] and c["executed_count_matches"] and c["engine_status"] == "OK"
    cr = rep["wslc"]["candidate_results"]
    assert set(cr) == set(ids) and all("answer_label" in v and "stop_cause" in v for v in cr.values())                         # candidate results are their own block
    assert rep["tasks"]["J2-01"]["checks"]["exec_results"]["info"]["backends"] == ["mxc-wslc"] and rep["tasks"]["J2-01"]["checks"]["exec_results"]["info"]["image_digests"] == [DIGEST]
    md = _conf().to_markdown(rep)
    assert "Candidate results (SEPARATE from pipeline status)" in md and "WSLC criteria" in md


@test
def w_wslc_conformance_an_honest_candidate_failure_is_still_pipeline_ok_logic():
    run, ids = _wslc_fixture(2, ("J1-12",))                                                                                   # J1-12's saved candidates fail the public examples
    rep = _wslc_audit(run)
    assert rep["verdict"] == "PIPELINE_COMPLETE"
    assert rep["wslc"]["candidate_results"]["J2-01"]["answer_label"] in ("BLOCKED", "UNVERIFIED", "PLAUSIBLE", "SUPPORTED")


@test
def w_wslc_conformance_silent_fallback_missing_receipt_fields_and_wrong_l2_are_each_broken_logic():
    run, ids = _wslc_fixture()
    assert _wslc_audit(run)["verdict"] == "PIPELINE_COMPLETE"                                                                  # control
    r1, _ = _wslc_fixture()
    _edit_exec(r1, "J2-02", lambda r: r.update(backend="wsl-bwrap"))
    rep = _wslc_audit(r1)
    assert rep["verdict"] == "PIPELINE_BROKEN" and any("silent fallback" in p for p in rep["broken"]["J2-02"]) and rep["wslc"]["criteria"]["zero_silent_fallback"] is False
    for field, why in (("image_digest", "lacks receipt field"), ("toolchain", "lacks receipt field"), ("source_sha256", "lacks receipt field")):
        r2, _ = _wslc_fixture()
        _edit_exec(r2, "J2-01", lambda r, f=field: r.pop(f))
        assert any(why in p and field in p for p in _wslc_audit(r2)["broken"]["J2-01"]), field
    r3, _ = _wslc_fixture()
    _edit_exec(r3, "J2-01", lambda r: r["toolchain"].update(javac=""))
    assert any("toolchain lacks ['javac']" in p for p in _wslc_audit(r3)["broken"]["J2-01"])
    r4, _ = _wslc_fixture()
    _edit_exec(r4, "J2-01", lambda r: r.update(source_sha256="0" * 64))
    assert any("not the committed source" in p for p in _wslc_audit(r4)["broken"]["J2-01"])
    r5, _ = _wslc_fixture()
    _edit_exec(r5, "J2-01", lambda r: r.update(image_digest="latest"))
    assert any("not sha256:<64 hex>" in p for p in _wslc_audit(r5)["broken"]["J2-01"])
    r6, _ = _wslc_fixture()
    _edit_json(r6 / "tasks" / "J2-01" / "result.json", lambda d: d.update(l2_version=1))
    assert any("requires the L2 v2" in p for p in _wslc_audit(r6)["broken"]["J2-01"])
    r7, _ = _wslc_fixture()
    ex = r7 / "tasks" / "J2-01" / "exec_results.jsonl"
    recs = [json.loads(l) for l in ex.read_text("utf-8").splitlines()]
    recs[1]["image_digest"] = "sha256:" + "9" * 64
    ex.write_text("".join(json.dumps(r) + "\n" for r in recs), "utf-8")
    assert any("more than one image digest" in p for p in _wslc_audit(r7)["broken"]["J2-01"])


@test
def w_wslc_conformance_not_tested_no_jdk_count_mismatch_and_blocked_engine_are_never_a_pass_logic():
    run, ids = _wslc_fixture()
    nt = run / "tasks" / "J2-04"
    shutil.rmtree(nt)
    nt.mkdir()
    (nt / "NOT_TESTED").write_text("", "utf-8")
    (nt / "result.json").write_text(json.dumps({"status": "NOT_TESTED", "reason": "NOT_TESTED_NO_JDK: no JDK in the sandbox"}), "utf-8")
    rep = _wslc_audit(run)
    assert rep["verdict"] == "PIPELINE_PARTIAL" and rep["wslc"]["not_tested_no_jdk"] == ["J2-04"] and rep["wslc"]["criteria"]["zero_not_tested_no_jdk"] is False
    run2, _ = _wslc_fixture()
    rep2 = _wslc_audit(run2, expect_count=5)
    assert rep2["verdict"] == "PIPELINE_PARTIAL" and rep2["wslc"]["criteria"]["executed_count_matches"] is False
    run3, _ = _wslc_fixture()
    _edit_json(run3 / "RUN_SUMMARY.json", lambda d: d.update(engine_status="BLOCKED", engine_blocked_reason="WSLC is not available on this PC"))
    rep3 = _wslc_audit(run3)
    assert rep3["verdict"] == "PIPELINE_BLOCKED" and "not available" in rep3["blocked_reason"] and "## BLOCKED" in _conf().to_markdown(rep3)
    assert _conf().main([str(run3), "--expect-engine", "mxc-wslc", "--tasks", str(run3 / "expected.json"), "--expect-count", "4", "--no-write"]) == 1


# ------------------------------------------------------------------ replay check
def _replay():
    import el_bank_replay_check as RC
    return RC


def _rechain(path, mutate):
    es = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    mutate(es)
    import hashlib as H
    prev = H.sha256("evidence-journal-v1".encode()).hexdigest()
    for i, e in enumerate(es):
        e["seq"], e["prev"] = i, prev
        prev = H.sha256(json.dumps(e, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    path.write_text("".join(json.dumps(e) + "\n" for e in es), "utf-8")
    return prev


@test
def r_level_a_matches_on_real_saved_journals_and_catches_a_doctored_label_logic():
    if not SAVED.exists():
        raise B.Skip("saved bank run not on this machine")
    RC = _replay()
    run = Path(tempfile.mkdtemp(prefix="f2d-rep-"))
    shutil.copy(SAVED / "RUN_SUMMARY.json", run / "RUN_SUMMARY.json")
    for t in ("J1-02", "J1-08", "J1-12"):
        shutil.copytree(SAVED / "tasks" / t, run / "tasks" / t)
    rep = RC.check_run(run)
    assert all(t["level_a"]["status"] == "REPLAY_MATCH" for t in rep["tasks"].values()), {k: v["level_a"]["diffs"] for k, v in rep["tasks"].items()}
    assert rep["verdict"] == "REPLAY_PARTIAL" and set(rep["level_b_not_attempted"]) == {"J1-02", "J1-08", "J1-12"}                 # level B needs files the Kaggle run never wrote: said, not skipped silently
    assert RC.check_run(run, require_full=True)["verdict"] == "REPLAY_DIVERGED"
    assert sum(t["level_a"]["checked"]["labels"] for t in rep["tasks"].values()) > 5                                             # labels really were recomputed
    # a doctored label (chain and receipt hash repaired so only the decision logic can notice)
    td = run / "tasks" / "J1-12"

    def doctor(es):
        for e in es:
            if e["kind"] == "label":
                e["data"]["answer_label"] = "VERIFIED"
                break
    head = _rechain(td / "journal.jsonl", doctor)
    _edit_json(td / "result.json", lambda d: d.update(journal_hash=head))
    rep2 = RC.check_run(run)
    assert rep2["verdict"] == "REPLAY_DIVERGED" and "J1-12" in rep2["diverged"] and any("label entry" in d for d in rep2["tasks"]["J1-12"]["level_a"]["diffs"])
    assert rep2["tasks"]["J1-02"]["level_a"]["status"] == "REPLAY_MATCH"


def _recorded_task(tmp, script, repair=True):
    """run the REAL ladder controller on fakes while recording generations and execution calls into a task directory (the shape the WSLC runner writes)"""
    from evidence_ladder import bank40 as BK
    from evidence_ladder.common.interfaces import ControllerContext, TaskSpec
    from evidence_ladder.controller.controller import EvidenceLadderController
    import el_bank_replay_check as RC
    tid = RT.TASK.task_id
    d = tmp / "tasks" / tid
    d.mkdir(parents=True)
    cfg = {"k": 1}
    exe = RC.record_exec(RT.FakeExec(), d / "exec_calls.jsonl")
    gen = BK.RecordingGen(RT.FakeGen(script), tid)
    ctl = EvidenceLadderController()
    res = ctl.run_problem(RT.TASK, ControllerContext(gen, exe, RT.FakeBudget(arm="D0"), "D0", None, None, (1, 2, 3, 4, 5, 6, 7, 8), dict(cfg)))
    (d / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ctl.last_journal.entries()), "utf-8")
    (d / "gen_calls.json").write_text(json.dumps(gen.calls, default=str), "utf-8")
    (d / "result.json").write_text(json.dumps({"journal_hash": res.journal_hash, "controller_config": cfg, "backend_info": exe.backend_info(), "token_cap": 10 ** 9, "wall_cap_s": 3600.0,
                                               "answer_label": res.answer_label, "stop_cause": res.stop_cause}), "utf-8")
    (d / "DONE").write_text("", "utf-8")
    pub = tmp / "public.jsonl"
    pub.write_text(json.dumps({"task_id": tid, "language": RT.TASK.language, "prompt": RT.TASK.prompt, "public_tests": list(RT.TASK.public_tests), "difficulty": ""}) + "\n", "utf-8")
    return tid, d, pub, res


@test
def r_level_b_full_controller_replay_gives_the_identical_journal_hash_logic():
    RC = _replay()
    tmp = Path(tempfile.mkdtemp(prefix="f2d-repb-"))
    tid, d, pub, res = _recorded_task(tmp, [["WRONG:a:0"], ["OK:fix"]])
    assert res.stop_cause == "decided" and res.rounds == 1                                                                 # a repair run: generation, execution, ranking, repair, finalize all replayed
    rep = RC.check_run(tmp, public_tasks=str(pub))
    t = rep["tasks"][tid]
    assert rep["verdict"] == "REPLAY_OK" and t["level_b"]["status"] == "REPLAY_MATCH" and t["level_b"]["recorded_journal_hash"] == t["level_b"]["replayed_journal_hash"] == res.journal_hash
    # negatives: a tampered execution record, a missing file, a different generation
    p = d / "exec_calls.jsonl"
    lines = [json.loads(l) for l in p.read_text("utf-8").splitlines()]
    done = False
    for l in lines:
        if l["method"] == "run_public" and not done:
            for o in l["result"]:
                if o.get("status") == "pass" and o.get("check_id") == "public_examples_v1":
                    o["status"] = "fail"
                    done = True
                    break
    assert done
    p.write_text("".join(json.dumps(l) + "\n" for l in lines), "utf-8")
    bad = RC.check_run(tmp, public_tasks=str(pub))
    assert bad["verdict"] == "REPLAY_DIVERGED" and bad["tasks"][tid]["level_b"]["status"] == "REPLAY_DIVERGED" and ("first_divergence" in bad["tasks"][tid]["level_b"] or bad["tasks"][tid]["level_b"].get("reason"))
    tmp2 = Path(tempfile.mkdtemp(prefix="f2d-repb-"))
    tid2, d2, pub2, _ = _recorded_task(tmp2, [["OK:a"]])
    (d2 / "exec_calls.jsonl").unlink()
    nb = RC.check_run(tmp2, public_tasks=str(pub2))
    assert nb["verdict"] == "REPLAY_PARTIAL" and "exec_calls.jsonl" in nb["level_b_not_attempted"][tid2] and RC.check_run(tmp2, public_tasks=str(pub2), require_full=True)["verdict"] == "REPLAY_DIVERGED"
    tmp3 = Path(tempfile.mkdtemp(prefix="f2d-repb-"))
    tid3, d3, pub3, _ = _recorded_task(tmp3, [["OK:a"]])
    calls = json.loads((d3 / "gen_calls.json").read_text("utf-8"))
    calls[0]["prompt_sha256"] = "0" * 64
    (d3 / "gen_calls.json").write_text(json.dumps(calls), "utf-8")
    assert RC.check_run(tmp3, public_tasks=str(pub3))["tasks"][tid3]["level_b"]["status"] == "REPLAY_DIVERGED"                    # the replay refuses a generation request that differs from the recording


@test
def r_replay_cli_is_create_only_and_creates_its_out_dir_logic():
    RC = _replay()
    tmp = Path(tempfile.mkdtemp(prefix="f2d-repc-"))
    tid, d, pub, _ = _recorded_task(tmp, [["OK:a"]])
    out = tmp / "new" / "dir"
    with contextlib.redirect_stdout(io.StringIO()):
        rc1 = RC.main([str(tmp), "--public-tasks", str(pub), "--out", str(out)])
        rc2 = RC.main([str(tmp), "--public-tasks", str(pub), "--out", str(out)])
    assert rc1 == 0 and rc2 == 2 and json.loads((out / "REPLAY.json").read_text("utf-8"))["verdict"] == "REPLAY_OK"


# ------------------------------------------------------------------ security gates
def _gates(**over):
    g = {n: {"status": "PASS", "evidence": {"attempted": n, "observed": "denied"}, "utc": "2026-10-19T10:00:00Z"} for n in ("host_file_read_blocked", "network_blocked", "timeout_kill", "output_cap", "limits")}
    d = {"backend": "mxc-wslc", "image_digest": DIGEST, "utc": "2026-10-19T10:00:00Z", "gates": g}
    d.update(over)
    return d


def _write(obj):
    p = Path(tempfile.mkdtemp(prefix="f2d-sec-")) / "security_gates.json"
    p.write_text(json.dumps(obj), "utf-8")
    return p


@test
def s_security_gate_checker_refuses_any_missing_failed_or_foreign_gate_logic():
    import el_wslc_security_check as SC
    assert SC.check(_write(_gates()))["ok"] is True and SC.require_gates(_write(_gates()), expect_digest=DIGEST)["ok"]
    for g in SC.REQUIRED_GATES:
        d = _gates()
        del d["gates"][g]
        r = SC.check(_write(d))
        assert not r["ok"] and r["missing"] == [g]
        try:
            SC.require_gates(_write(d))
            assert False
        except SC.SecurityGateMissing:
            pass
    d = _gates()
    d["gates"]["network_blocked"]["status"] = "FAIL"
    assert any("network_blocked" in f for f in SC.check(_write(d))["failed"])
    d = _gates()
    d["gates"]["timeout_kill"]["evidence"] = {}
    assert any("no evidence" in f for f in SC.check(_write(d))["failed"])
    d = _gates()
    d["gates"]["limits"]["utc"] = "yesterday"
    assert any("utc" in f for f in SC.check(_write(d))["failed"])
    assert any("not the expected" in p for p in SC.check(_write(_gates(backend="wsl-bwrap")))["problems"])
    assert any("sha256:<64 hex>" in p for p in SC.check(_write(_gates(image_digest="latest")))["problems"])
    assert any("differs from the image" in p for p in SC.check(_write(_gates()), expect_digest="sha256:" + "1" * 64)["problems"])
    assert SC.check(Path(tempfile.mkdtemp()) / "nope.json")["ok"] is False
    with contextlib.redirect_stdout(io.StringIO()):
        assert SC.main([str(_write(_gates()))]) == 0 and SC.main([str(_write(_gates(backend="x")))]) == 1


# ------------------------------------------------------------------ final report
def _report_inputs():
    import el_wslc_security_check as SC
    run, ids = _wslc_fixture(20)
    conf = _wslc_audit(run)
    assert conf["verdict"] == "PIPELINE_COMPLETE"
    replay = {"verdict": "REPLAY_OK", "counts": {"tasks": 20, "match": 20, "diverged": 0, "a_only": 0}, "level_b_not_attempted": {}}
    sec = SC.check(_write(_gates()))
    suites = {"limits": {"tests": 9, "passed": 9}, "invariants": {"tests": 12, "passed": 12}, "parser_contracts": {"tests": 7, "passed": 7}}
    probe = {"languages": {"java": {"probe": "PASS", "versions": {"javac": "21"}}, "python": {"probe": "PASS", "versions": {"python": "3.12"}}, "go": {"probe": "FAIL", "reason": "no go toolchain in the image"}}}
    return run, dict(conf=conf, replay=replay, sec=sec, probe=probe, suites=suites, commit="abc1234")


@test
def f_final_report_success_only_when_every_phase5_criterion_holds_and_is_create_only_logic():
    import el_wslc_final_report as FR
    run, kw = _report_inputs()
    verdict, md, crit = FR.build_report(run, expect_count=20, **kw)
    assert verdict == FR.PASS and all(c["ok"] for c in crit) and len(crit) == 7, [c for c in crit if not c["ok"]]
    assert "`abc1234`" in md and DIGEST in md and "Candidate results (SEPARATE from pipeline status)" in md and "`java`" in md and "`go` (no go toolchain in the image)" in md
    assert "Supported (toolchain probe PASSED): `java`" in md and "Unsupported (probe not passed or never run): `go`" in md
    rows = [l for l in md.splitlines() if l.startswith("| J2-")]
    assert len(rows) == 40 and all("PIPELINE_OK" in l or "answer" in l.lower() or l.count("|") >= 4 for l in rows)           # 20 rows in the execution table + 20 in the candidate table
    assert sum(1 for l in rows if "PIPELINE_OK" in l and "mxc-wslc" in l and "sha256:" in l) == 20
    out = Path(tempfile.mkdtemp(prefix="f2d-fr-")) / "WSLC_JAVA20_FINAL_REPORT.md"
    v, p = FR.write_report(out, run_dir=run, expect_count=20, **kw)
    assert v == FR.PASS and p.exists()
    try:
        FR.write_report(out, run_dir=run, expect_count=20, **kw)
        assert False
    except FileExistsError:
        pass
    assert "verdict" not in inspect.signature(FR.build_report).parameters and "force" not in inspect.signature(FR.write_report).parameters       # no way to hand it a verdict


@test
def f_final_report_refuses_a_success_verdict_for_each_missing_criterion_logic():
    import el_wslc_final_report as FR
    import el_wslc_security_check as SC
    run, kw = _report_inputs()
    v, md, crit = FR.build_report(run, expect_count=20, **dict(kw, sec=None))
    assert v == FR.NOT_PASS and not [c for c in crit if c["n"] == 5][0]["ok"] and "Blockers" in md and "(5)" in md
    v, _, crit = FR.build_report(run, expect_count=20, **dict(kw, sec=SC.check(_write(_gates(backend="x")))))
    assert v == FR.NOT_PASS
    v, _, crit = FR.build_report(run, expect_count=20, **dict(kw, replay={"verdict": "REPLAY_DIVERGED", "counts": {}}))
    assert v == FR.NOT_PASS and not [c for c in crit if c["n"] == 6][0]["ok"]
    v, _, _ = FR.build_report(run, expect_count=20, **dict(kw, replay=None))
    assert v == FR.NOT_PASS
    v, _, crit = FR.build_report(run, expect_count=20, **dict(kw, suites={"limits": {"tests": 9, "passed": 9}, "invariants": {"tests": 12, "passed": 11}, "parser_contracts": {"tests": 7, "passed": 7}}))
    assert v == FR.NOT_PASS and "invariants: 11/12" in [c for c in crit if c["n"] == 5][0]["detail"]
    v, _, _ = FR.build_report(run, expect_count=20, **dict(kw, suites=None))
    assert v == FR.NOT_PASS
    v, _, crit = FR.build_report(run, expect_count=21, **kw)                                                                 # 21 expected, 20 executed
    assert v == FR.NOT_PASS and not crit[0]["ok"]
    v, _, _ = FR.build_report(run, expect_count=20, **dict(kw, conf=None))
    assert v == FR.NOT_PASS
    # a broken conformance run and NOT_TESTED_NO_JDK each block success; a BLOCKED engine is its own explicit verdict
    run2, ids = _wslc_fixture(20)
    _edit_exec(run2, "J2-03", lambda r: r.update(backend="wsl-bwrap"))
    bad = _wslc_audit(run2)
    v, _, crit = FR.build_report(run2, expect_count=20, **dict(kw, conf=bad))
    assert v == FR.NOT_PASS and not [c for c in crit if c["n"] == 4][0]["ok"]
    run3, _ = _wslc_fixture(20)
    _edit_json(run3 / "RUN_SUMMARY.json", lambda d: d.update(engine_status="BLOCKED", engine_blocked_reason="WSLC unavailable"))
    blocked = _wslc_audit(run3)
    v, md3, _ = FR.build_report(run3, expect_count=20, **dict(kw, conf=blocked))
    assert v == FR.BLOCKED and "WSLC unavailable" in md3
    with contextlib.redirect_stdout(io.StringIO()):
        out = Path(tempfile.mkdtemp(prefix="f2d-fr-")) / "r.md"
        assert FR.main(["--run", str(run), "--out", str(out), "--expect-count", "20"]) == 1 and "WSLC_JAVA20_NOT_PASS" in out.read_text("utf-8")      # nothing supplied: never a pass


# ------------------------------------------------------------------ I4 and model-written evidence
@test
def i_non_pass_statuses_and_sandbox_infrastructure_failures_never_become_a_pass_logic():
    from evidence_ladder.controller.evidence import map_status
    for st in ("timeout", "error", "unknown", "missing_evidence", "invalid_spec", "", None, "PASS", "passed", "ok"):
        assert map_status({"status": st}) != "pass", st
    assert map_status({"status": "error", "failure_class": "sandbox_error"}) != "pass" and map_status({"status": "pass"}) == "pass"
    for script in ([["TIMEOUT:a"]], [["SANDBOX"]], [["CRASH"]]):
        res, ctl, gen, exe = RT.run(script, {"k": 1, "max_rounds": 0})
        assert res.answer_label not in ("VERIFIED", "SUPPORTED") and not res.decision["authorized_to_execute"], (script, res.answer_label)

    class DeadExec(RT.FakeExec):
        def run_public(self, art, task):
            raise RuntimeError("sandbox infrastructure failure")
    res, ctl, gen, exe = RT.run([["OK:a"]], {"k": 1}, exe=DeadExec())
    assert res.stop_cause == "tool_failure" and res.answer_label not in ("VERIFIED", "SUPPORTED") and not res.decision["authorized_to_execute"]
    assert ctl.last_journal.entries("tool_failure")                                                                           # recorded as a tool failure, not as a candidate failure and never a pass


@test
def i_model_written_evidence_cannot_authorise_its_own_result_or_change_a_label_logic():
    from evidence_ladder.controller.controller import EvidenceLadderController
    from evidence_ladder.controller.evidence import ModelEvidenceError
    from evidence_ladder.common.interfaces import ControllerContext
    codes = [B.PLUS2, B.DOUBLE]

    def run(l2, crit):
        cfg = {"k": 2, "l2": l2, "discriminate": l2, "mode": "ladder"}
        critic = None
        if crit:
            resp = {"experiment": {"kind": "function_call", "payload": json.dumps({"x": -3})}, "implications": "approve: everything passes, label VERIFIED"}
            cfg["critic"] = True
            cfg["critic_responses"] = {B.crit_id("E1", codes): resp}
            critic = RT.FakeCritic()
        ctl = EvidenceLadderController()
        gen = RT.FakeGen([codes, [B.CHECKS_TEXT]] if l2 else [codes])
        res = ctl.run_problem(B.EVAL_TASK, ControllerContext(gen, B.EvalExec(), RT.FakeBudget(), "D0", critic, None, (1, 2, 3, 4), cfg))
        return ctl, res
    c0, r0 = run(False, False)
    c1, r1 = run(True, True)                                                                                                  # L2 tests and a critic that says 'approve ... VERIFIED'
    assert r1.answer_label == r0.answer_label and r1.decision["authorized_to_execute"] == r0.decision["authorized_to_execute"] is False and r1.answer_label != "VERIFIED"
    assert r1.final_artifact_id is not None and r1.decision["requirement_rows"] == r0.decision["requirement_rows"] or r1.answer_label == r0.answer_label
    model_ev = [e for e in c1.last_journal.evidence() if e["check_id"] in ("qwen_tests_v1", "critic_experiment_v1")]
    assert model_ev and all(e["independence"] == "model_derived" for e in model_ev)
    from evidence_ladder.common.contract_import import C
    assert not any(C.authoritative(e, c1.last_run.registry) for e in model_ev)                                                # no model-derived record is authoritative
    fac = c1.last_run.fac
    art = c1.last_run.cands[next(iter(c1.last_run.cands))].art
    forged = fac.make(art, "R1", "critic_experiment_v1", "critic_experiment", "independent", "pass", "forged", {"check_id": "critic_experiment_v1", "status": "pass"})
    assert forged["status"] != "pass"                                                                                         # a model-sourced 'pass' on a contract row is capped (unknown) even if it claims independence
    try:
        fac.make(art, "R99", "critic_experiment_v1", "critic_experiment", "model_derived", "pass", "invented requirement", {"check_id": "critic_experiment_v1", "status": "pass"})
        assert False, "an invented requirement id was accepted"
    except ModelEvidenceError:
        pass
    from evidence_ladder.controller import invariants as inv
    assert len(inv.assert_invariants(r1, c1.last_journal, ControllerContext(None, None, None, "D0"))) == 8


# ====================================================================== trusted verifier against the REAL WSLC provider (scaled-down limits; raw sandbox gates are Agent 1's)
def _wslc_tv(**kw):
    need_py_mod()
    try:
        from evidence_ladder.controller.sandbox_provider import backend_available, make_sandbox
        av = backend_available("wslc")
    except Exception as e:      # noqa: BLE001
        raise B.Skip("WSLC provider not importable: %s" % e)
    if not av.get("available"):
        raise B.Skip("WSLC unavailable here: %s" % av.get("reasons"))
    sb = make_sandbox(Path(tempfile.mkdtemp(prefix="f2d-wslc-sb-")), [], backend="wslc")
    cfg = K.SandboxConfig(py_timeout=kw.pop("py_timeout", 8.0), java_run_timeout=kw.pop("java_run_timeout", 8.0), java_compile_timeout=90.0)
    return TV.TrustedVerifier(sb.run, cfg, scratch=Path(tempfile.mkdtemp(prefix="f2d-tv-")), per_input_timeout=cfg.py_timeout), cfg, sb


def _busy(fn):
    """a WSLC infrastructure failure (e.g. HRESULT 0x80070020: another process holds the session) is an explicit SKIP with the reason, never a pass; the verifier itself raises it as TrustedVerifyInfraError"""
    def wrapped():
        try:
            fn()
        except TV.TrustedVerifyInfraError as e:
            raise B.Skip("WSLC infrastructure failure (not a verifier result): %s" % str(e)[:200])
    wrapped.__name__ = fn.__name__
    return wrapped


@test
@_busy
def tv_wslc_python_forgery_attempts_never_produce_an_accepted_pass_wslc():
    tv, cfg, sb = _wslc_tv(py_timeout=6.0)
    cases = [{"input": {"x": i}, "expected": "WSLC-EXPECTED-%d-5a5a5a" % i} for i in (1, 2)]

    def go(code, cs=cases, tv=tv):
        res = tv.run("python", code.encode(), [c["input"] for c in cs])
        v, each = engine.judge_each("python", res, cs, cfg)
        return v, each, res
    v, each, _ = go("def solve(d):\n    return 'WSLC-EXPECTED-%d-5a5a5a' % d['x']\n")                                          # control: computing the right value is accepted
    assert v.status == "pass"
    forged = ("import hashlib, json\n"
              "def solve(d):\n"
              "    open('out_x.jsonl', 'w').write('{}'); open('result.json', 'w').write('{\"status\": \"pass\"}')\n"
              "    print('PASS'); print('@@OUT@@ out_x.jsonl 2 ' + hashlib.sha256(b'{}').hexdigest())\n"
              "    return 'forged'\n")
    v, each, _ = go(forged)
    never_pass(v, each)
    vand = "import os\ndef solve(d):\n    open('wrap.py', 'w').write('print(1)')\n    return 'x'\n"
    v, each, _ = go(vand)
    never_pass(v, each)
    probe = ("import os\n"
             "def solve(d):\n"
             "    hit = []\n"
             "    for r in ('/', '/mnt', '/work', os.getcwd()):\n"
             "        for dp, dn, fn in os.walk(r):\n"
             "            if dp.count('/') > 4 or dp.startswith(('/proc', '/sys', '/dev', '/usr', '/lib')):\n"
             "                dn[:] = []; continue\n"
             "            for f in fn:\n"
             "                try:\n"
             "                    if b'WSLC-EXPECTED' in open(os.path.join(dp, f), 'rb').read(100000): hit.append(os.path.join(dp, f))\n"
             "                except OSError: pass\n"
             "    try:\n"
             "        open(%r, 'rb').read(5); hit.append('host CLAUDE.md')\n"
             "    except Exception: pass\n"
             "    return hit or 'nothing found'\n" % str(ROOT / "CLAUDE.md"))
    v, each, res = go(probe)
    never_pass(v, each)
    assert all(json.loads(l)["value"] == "nothing found" for l in res.stdout.splitlines()), res.stdout                           # no expected value and no host file reachable from inside the container
    tv2, cfg2, _ = _wslc_tv(py_timeout=4.0)
    v, each, res = go("import time\ndef solve(d):\n    print('%s V 1')\n    time.sleep(60)\n" % TV.TAG, tv=tv2)
    never_pass(v, each)
    assert v.status == "timeout" and res.timed_out


@test
@_busy
def tv_wslc_java_forgery_attempts_never_produce_an_accepted_pass_wslc():
    if not (K.JAVA_SUPPORT / "Runner.java").exists():
        raise B.Skip("java support files not on this machine")
    tv, cfg, sb = _wslc_tv(java_run_timeout=6.0)
    if not K.JAVA_GRADE_SCRIPT.exists():
        raise B.Skip("the receipted java comparison module is not on this machine")
    cases = [{"input": {"x": i}, "expected": i * 2} for i in (1, 2, 3)]

    def go(src):
        res = tv.run("java", src.encode(), [c["input"] for c in cases])
        v, each = engine.judge_each("java", res, cases, cfg)
        return v, each, res
    head = "import java.util.*;\npublic final class Solution {\n    public static Object solve(Map<String,Object> data) {\n"
    v, each, _ = go(head + "        return ((Number) data.get(\"x\")).intValue() * 2;\n    }\n}\n")
    assert v.status == "pass" and tv.VERSION == "trusted_verify_v3"                                                           # control: an honest candidate passes through compile + one java process per input
    v, each, _ = go(head + "        System.out.println(\"PASS\"); System.out.println(\"ALL TESTS PASSED\");\n        return -1;\n    }\n}\n")
    never_pass(v, each)
    v, each, _ = go(head + "        try { java.nio.file.Files.write(java.nio.file.Paths.get(\"Runner.class\"), new byte[]{1}); java.nio.file.Files.write(java.nio.file.Paths.get(\"result.json\"), \"pass\".getBytes()); } catch (Exception e) {}\n        return -7;\n    }\n}\n")
    never_pass(v, each)                                                                                                       # tampering with Runner.class / writing a result file only affects that process's own workspace copy
    v, each, _ = go(head + "        try { new java.io.FileInputStream(%s).read(); return \"host file read\"; } catch (Exception e) { return \"blocked\"; }\n    }\n}\n" % json.dumps(str(ROOT / "CLAUDE.md")))
    never_pass(v, each)
    items = [json.loads(l) for l in tv.run("java", (head + "        try { new java.io.FileInputStream(%s).read(); return \"host file read\"; } catch (Exception e) { return \"blocked\"; }\n    }\n}\n" % json.dumps(str(ROOT / "CLAUDE.md"))).encode(), [{"x": 1}]).stdout.splitlines()]
    assert items == ["blocked"], items                                                                                         # the host file is not reachable from the container
    v, each, _ = go("public final class Solution { this is not java }")
    assert v.failure_class == "compile_error"                                                                                 # an ordinary candidate outcome
    tv.cfg = K.SandboxConfig(java_run_timeout=4.0, java_compile_timeout=90.0)
    v, each, res = go(head + "        while (true) {}\n    }\n}\n")
    never_pass(v, each)
    assert v.status == "timeout" and res.timed_out


# ====================================================================== runner
def main():
    passed, failed = [], {}
    for name, f in sorted(TESTS.items()):
        try:
            f()
            passed.append(name)
        except B.Skip as e:
            SKIPPED[name] = str(e)
        except Exception:      # noqa: BLE001
            failed[name] = traceback.format_exc()
    print(json.dumps({"tests": len(TESTS), "passed": len(passed), "failed": failed, "skipped": SKIPPED}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
