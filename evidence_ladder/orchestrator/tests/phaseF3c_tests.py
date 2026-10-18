"""[EL-BANK-LOCAL] CPU tests of the local bank run: the provider-port engine (MXC) with a fake provider that really runs the portable runner, the WSL comparison adapter with fake transports, fail-closed
behaviour, Java -> NOT_TESTED_NO_JDK, the resilient runner (resume, a crashing task, engine failures, STOP file), output readable by the driver compile; plus REAL checks (skipped when unavailable):
the real MXC sandbox on the public examples of P1-01 (lookup solution passes, wrong / broken / looping / hostile candidates do not) and the real WSL adapter.
  PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe evidence_ladder/orchestrator/tests/phaseF3c_tests.py   -> JSON {"tests","passed","failed","skipped"}"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
from evidence_ladder import bank40 as B                                  # noqa: E402
from evidence_ladder.common.interfaces import Artifact, TaskSpec         # noqa: E402
from evidence_ladder.execution import provider_engine as PE, wsl_engine as WE   # noqa: E402
import el_bank_local_run as LR                                           # noqa: E402
import el_bank40_driver as DR                                            # noqa: E402

TESTS, MEASURED = [], {}


class Skip(Exception):
    pass


def test(f):
    TESTS.append(f); return f


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % exc.__name__)


RESULT0 = {"exit_code": 0, "killed": False, "stdout": "", "stderr": "", "protected_unchanged": True, "pre_tree_sha256": "a", "post_tree_sha256": "a", "persisted": True}


class LocalPortSandbox:
    """TEST double of the provider port: runs `python3 <args>` in the workspace with the host Python (NO isolation: only for contract tests). persist=True keeps the files in place."""
    env_label = "fake-provider-label"

    def __init__(self, tamper=False, kill=False, raise_exc=None):
        self.tamper, self.kill, self.raise_exc, self.cmds = tamper, kill, raise_exc, []

    def run(self, workspace, cmd, persist=False, timeout=60):
        self.cmds.append(cmd)
        if self.raise_exc: raise self.raise_exc
        if self.kill: return dict(RESULT0, exit_code=-1, killed=True, persisted=False)
        argv = cmd.split()
        assert argv[0] == "python3", cmd                                  # the MXC dialect: one direct python command, no shell
        p = subprocess.run([sys.executable] + argv[1:], cwd=workspace, capture_output=True, text=True, timeout=timeout)
        if self.tamper:
            f = next(Path(workspace).glob("out_*.jsonl")); f.write_text(json.dumps({"ok": True, "value": "FORGED"}) + "\n", "utf-8")
        return dict(RESULT0, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr[-6000:])


def _task(tests=None):
    tests = tests or [{"input": {"x": 1}, "expected": 2}, {"input": {"x": 2}, "expected": 4}]
    return TaskSpec("P1-99", "py1:P1-99", "python", "python", "double x", tests, "easy")


def _art(src):
    return Artifact.make("P1-99", "python", src)


def _be(**kw):
    return PE.ProviderExecutionBackend(LocalPortSandbox(**kw.pop("sb", {})), scratch=Path(tempfile.mkdtemp()), **kw)


GOOD = "def solve(d):\n    return d['x'] * 2\n"


# ------------------------------------------------------------------ provider-port engine (MXC path), contract over a fake provider
@test
def t_provider_engine_contract_shapes_and_verdicts():
    be = _be()
    ok = be.run_public(_art(GOOD), _task())
    assert [o["check_id"] for o in ok] == ["parse_v1", "public_examples_v1", "public_examples_v1"] and all(o["status"] == "pass" for o in ok) and [o["example_index"] for o in ok[1:]] == [0, 1]
    bad = be.run_public(_art("def solve(d):\n    return 0\n"), _task())
    assert bad[0]["status"] == "pass" and [o["status"] for o in bad[1:]] == ["fail", "fail"] and bad[1]["failure_class"] == "wrong_answer" and bad[1]["counterexample"]["expected"] == "2"
    syn = be.run_public(_art("def solve(d:\n  pass"), _task())
    assert syn[0]["status"] == "fail" and syn[0]["failure_class"] == "compile_error" and all(o["status"] == "missing_evidence" for o in syn[1:])
    rse = be.run_public(_art("def solve(d):\n    raise ValueError('x')\n"), _task())
    assert [o["failure_class"] for o in rse[1:]] == ["runtime_error", "runtime_error"]
    crash = be.run_public(_art("import os\nos._exit(5)\ndef solve(d): return 1\n"), _task())
    assert all(o["status"] == "fail" and o["failure_class"] == "runtime_error" for o in crash[1:])               # a hard crash of the candidate is a candidate failure, not an engine failure
    forged = be.run_public(_art("def solve(d):\n    print('{\"ok\": true, \"value\": 2}'); return 0\n"), _task())
    assert forged[1]["status"] == "fail"                                                                         # printing a fake result line changes nothing
    ex = be.run_experiment(_art(GOOD), {"kind": "function_call", "payload": json.dumps({"x": 21})})
    assert ex["status"] == "pass" and ex["observed"] == 42 and ex["check_id"] == "critic_experiment_v1"
    assert be.run_experiment(_art(GOOD), {"kind": "shell", "payload": "ls"})["status"] == "invalid_spec"
    assert be.run_experiment(_art("def solve(d):\n    raise KeyError('k')\n"), {"kind": "function_call", "payload": "{}"})["status"] == "fail"


@test
def t_provider_engine_timeout_overflow_and_tamper_are_handled_not_passed():
    t = be = _be(sb={"kill": True}).run_public(_art(GOOD), _task())
    assert [o["status"] for o in t[1:]] == ["timeout", "timeout"]                                                 # a killed run has no persisted result: every example times out
    big = _be()
    big.cfg = type(big.cfg)(max_output_bytes=40)
    o = big.run_public(_art("def solve(d):\n    return 'x' * 100\n"), _task())
    assert all(x["status"] != "pass" for x in o[1:])                                                              # output cap exceeded -> never a pass
    raises(PE.ProviderEngineError, _be(sb={"tamper": True}).run_public, _art(GOOD), _task())                      # result file differs from the runner's marker -> engine error, not a verdict


@test
def t_provider_engine_fails_closed_on_provider_errors_and_unavailable_mxc():
    from evidence_ladder.controller.mxc_sandbox import SandboxUnavailable
    for exc in (RuntimeError("bridge died"), OSError("pipe"), SandboxUnavailable("no mxc"), subprocess.TimeoutExpired("x", 1)):
        e = raises(PE.ProviderEngineError, _be(sb={"raise_exc": exc}).run_public, _art(GOOD), _task())
        assert e.infra is True
    be = PE.ProviderExecutionBackend(None, backend="mxc", scratch=Path(tempfile.mkdtemp()))
    import evidence_ladder.controller.sandbox_provider as SP
    real = SP.make_sandbox
    SP.make_sandbox = lambda *a, **k: (_ for _ in ()).throw(SandboxUnavailable("mxc is not available on this host"))
    try:
        e = raises(PE.ProviderEngineError, be.start)
        assert "mxc" in str(e) and "unavailable" in str(e)                                                        # no fallback to WSL, no result
        raises(PE.ProviderEngineError, be.run_public, _art(GOOD), _task())
    finally:
        SP.make_sandbox = real


@test
def t_provider_engine_runs_a_single_python_command_and_records_the_backend_label():
    sb = LocalPortSandbox()
    be = PE.ProviderExecutionBackend(sb, scratch=Path(tempfile.mkdtemp()))
    be.run_public(_art(GOOD), _task())
    assert sb.cmds and all(c.startswith("python3 runner.py ") and not any(ch in c for ch in ";&|<>") for c in sb.cmds)
    info = be.backend_info()
    assert info["engine"] == "fake-provider-label" and info["backend"] == "mxc" and info["calls"] == 1 and info["java"]["status"] == "NOT_TESTED_NO_JDK" and info["seconds_per_call"]["n"] == 1
    assert "wsl" not in json.dumps(info).lower() or info["engine"] != "wsl-bwrap"


@test
def t_java_is_not_runnable_on_mxc_and_never_moved_elsewhere():
    cap = PE.java_capability("mxc", which=lambda n: None, environ={})
    assert cap["runnable"] is False and cap["status"] == "NOT_TESTED_NO_JDK" and "no JDK" in cap["reason"]
    cap2 = PE.java_capability("mxc", which=lambda n: "C:/jdk/bin/javac", environ={})
    assert cap2["runnable"] is False and "python3" in cap2["reason"]                                              # even with a host JDK the dialect cannot start it
    assert PE.java_capability("wsl")["runnable"] is True
    e = raises(PE.JavaNotRunnable, _be().run_public, Artifact.make("J1-01", "java", "class Solution {}"), TaskSpec("J1-01", "java1:J1-01", "java", "java", "p", [{"input": {}, "expected": 1}]))
    assert e.infra is True and "NOT_TESTED" in str(e) or "JDK" in str(e)


# ------------------------------------------------------------------ WSL comparison adapter, fake transports
class FakeBackend:
    def run_public(self, a, t): return [{"check_id": "parse_v1", "status": "pass"}] + [{"check_id": "public_examples_v1", "example_index": i, "status": "pass"} for i in range(len(t.public_tests))]
    def run_experiment(self, a, e): return {"check_id": "critic_experiment_v1", "status": "pass", "observed": 1}
    def backend_info(self): return {"b": 1}


class InProcessTransport:
    def __init__(self, hello_ok=True, fail=None):
        self.backend, self.hello_ok, self.fail, self.reqs = FakeBackend(), hello_ok, fail, []

    def call(self, req):
        self.reqs.append(req)
        if self.fail: raise self.fail
        return WE.handle(self.backend, req, lambda: {"ok": self.hello_ok, "engine": WE.ENGINE_LABEL, "os": "Linux", "bwrap_path": "/usr/bin/bwrap" if self.hello_ok else None, "jdk": "/opt/toolchains/jdk-26.0.1"})


@test
def t_wsl_adapter_contract_json_forwarding_and_label():
    tr = InProcessTransport()
    be = WE.WslExecutionBackend(transport=tr)
    assert be.start()["engine"] == "wsl-bwrap"
    out = be.run_public(_art(GOOD), _task())
    assert out[0]["check_id"] == "parse_v1" and len(out) == 3
    assert be.run_experiment(_art(GOOD), {"kind": "function_call", "payload": "{}"})["observed"] == 1
    r = tr.reqs[1]
    assert r["op"] == "run_public" and r["artifact"]["artifact_id"] == _art(GOOD).artifact_id and r["task"]["public_tests"] == _task().public_tests and "hidden" not in json.dumps(r)
    json.dumps(tr.reqs)                                                                                           # every request is plain JSON
    assert be.backend_info()["engine"] == "wsl-bwrap" and be.backend_info()["calls"] == 3


@test
def t_wsl_adapter_fails_closed():
    raises(WE.WslEngineError, WE.WslExecutionBackend(transport=InProcessTransport(hello_ok=False)).start)         # no bwrap / jail probe -> handshake refused
    raises(WE.WslEngineError, WE.WslExecutionBackend(transport=InProcessTransport(fail=WE.WslEngineError("pipe closed"))).run_public, _art(GOOD), _task())
    class Bad:
        def call(self, req): return {"ok": False, "error": "boom"}
    raises(WE.WslEngineError, WE.WslExecutionBackend(transport=Bad()).run_public, _art(GOOD), _task())
    class Junk:
        def call(self, req): return "not a dict"
    raises(WE.WslEngineError, WE.WslExecutionBackend(transport=Junk()).run_public, _art(GOOD), _task())
    def no_wsl(*a, **k): raise FileNotFoundError("wsl.exe")
    tr = WE.WslProcessTransport(REPO, popen=no_wsl)
    raises(WE.WslEngineError, tr.call, {"op": "hello"})                                                           # missing wsl.exe: error, no unsandboxed fallback
    assert WE.handle(FakeBackend(), {"op": "nope"})["ok"] is False
    assert WE.wsl_path(Path("D:/local model")) == "/mnt/d/local model"


@test
def t_wsl_process_transport_restarts_once_and_times_out():
    class P:
        def __init__(self, lines, alive=True):
            import io
            self.stdin = io.BytesIO(); self.stdout = io.BytesIO(b"".join(lines)); self._alive = alive
        def poll(self): return None if self._alive else 1
        def kill(self): self._alive = False
    made = []
    def popen(cmd, **k):
        p = P([] if not made else [json.dumps({"ok": True, "result": 1, "id": 2}).encode() + b"\n"])
        made.append(p); return p
    tr = WE.WslProcessTransport(REPO, popen=popen)
    resp = tr.call({"op": "info"})
    assert resp["ok"] and tr.restarts == 1 and len(made) == 2                                                     # first server died (EOF) -> restarted once -> answered
    class Dead:
        def __init__(self): import io; self.stdin = io.BytesIO(); self.stdout = io.BytesIO(b"");
        def poll(self): return None
        def kill(self): pass
    tr2 = WE.WslProcessTransport(REPO, popen=lambda c, **k: Dead())
    raises(WE.WslEngineError, tr2.call, {"op": "info"})                                                           # still dead after the restart -> raises


# ------------------------------------------------------------------ the runner
class FakeGen:
    gpu_name = "FAKE-3050"
    snapshot_verified = True

    def __init__(self, crash_ids=()):
        self.language, self.crash_ids = "python", set(crash_ids)

    def model_info(self): return {"backend": "fake", "profile_id": "fake", "profile_sha256": "f" * 64, "snapshot_verified": True}

    def generate(self, req):
        from evidence_ladder.models import qwen as Q
        if req.task_id in self.crash_ids: raise RuntimeError("generation exploded for " + req.task_id)
        return [Q.make_generation("```python\ndef solve(data):\n    return %d\n```" % (s % 3), tokens_prompt=10, tokens_generated=20 + s, seed=s, prompt_template_id=req.prompt_template_id,
                                  prompt_sha256=B.sha(req.prompt), language=self.language, wall_seconds=0.01) for s in list(req.seeds)[:req.n]]


class FakeExe:
    env_label = "mxc-processcontainer"

    def __init__(self, engine_fail_ids=(), start_error=None):
        self.engine_fail_ids, self.start_error, self.closed = set(engine_fail_ids), start_error, False

    def start(self):
        if self.start_error: raise self.start_error
        return {"engine": self.env_label, "backend": "mxc"}

    def _k(self, a):
        import re
        m = re.search(r"return (\d)", a.source); return int(m.group(1)) if m else 9

    def run_public(self, a, t):
        if a.task_id in self.engine_fail_ids: raise PE.ProviderEngineError("bridge died")
        ok = self._k(a) in (0, 1)
        return [{"check_id": "parse_v1", "status": "pass"}] + [{"check_id": "public_examples_v1", "example_index": i, "status": "pass" if ok else "fail", "failure_class": "" if ok else "wrong_answer",
                                                                 "observed": self._k(a)} for i in range(len(t.public_tests))]

    def run_experiment(self, a, e):
        if a.task_id in self.engine_fail_ids: raise PE.ProviderEngineError("bridge died")
        return {"check_id": "critic_experiment_v1", "status": "pass", "observed": self._k(a)}

    def backend_info(self): return {"engine": self.env_label, "backend": "mxc", "calls": 0}
    def close(self): self.closed = True


def _tasks(*ids):
    return [{"task_id": i, "language": "java" if i.startswith("J") else "python", "difficulty": "medium", "prompt": "p " + i, "public_tests": [{"input": {"x": 1}, "expected": 1}, {"input": {"x": 2}, "expected": 2}]} for i in ids]


def _run(out, tasks, gen=None, exe=None, **kw):
    gen = gen or FakeGen(); exe = exe or FakeExe()
    kw.setdefault("critic_client", LR.GlmInlineCritic(key_loader=lambda: {}))      # NEVER the real keys in tests: no key -> an explicit api_unavailable record
    kw.setdefault("resources_fn", lambda: {"free_ram_gb": 16.0, "free_vram_gb": 5.0})
    kw.setdefault("finalize", False)
    return LR.run(Path(out), tasks, lambda o: gen, lambda o: exe, **kw), gen, exe


@test
def t_runner_layout_policy_label_and_driver_compile():
    tmp = Path(tempfile.mkdtemp())
    try:
        rs, gen, exe = _run(tmp / "o", _tasks("P1-13", "P1-14"))
        assert rs["verdict"] == "OK" and rs["done"] == 2 and rs["thinking_policy"]["policy"] == "model_variant" and rs["thinking_policy"]["profile_id"] == LR.PROFILE and rs["thinking_policy"]["ceiling"] == 4096
        assert rs["scorer"] == "none" and rs["jail"]["engine"] == "mxc-processcontainer" and rs["sandbox"] == "mxc" and rs["critic"]["state"] == "inline" and rs["engine"] == "mxc" and rs["l2_version"] == 2 and rs["profile"]["sha256"] == "f" * 64 and rs["commit"] and rs["config_sha256"] and rs["internal_states"] == "disabled"
        for tid in ("P1-13", "P1-14"):
            d = tmp / "o" / "tasks" / tid
            for f in ("result.json", "journal.jsonl", "gen_calls.json", "critic_requests.json", "controller_timeline.json", "DONE"):
                assert (d / f).is_file(), f
            r = json.loads((d / "result.json").read_text("utf-8"))
            assert r["status"] == "OK" and r["thinking_policy"] == rs["thinking_policy"] and r["engine"] == "mxc-processcontainer" and r["wall_cap_s"] == 3600.0 and r["l2_version"] == 2 and r["profile_sha256"] == "f" * 64 and (tmp / "o" / "tasks" / tid / "exec_results.jsonl").is_file() is False and "hidden grading NOT done" in r["label"]
        assert (tmp / "o" / "TIMELINE.jsonl").is_file() and (tmp / "o" / "TIMELINE_SUMMARY.json").is_file() and (tmp / "o" / "RUN_SUMMARY.json").is_file() and exe.closed
        comp = DR.compile_results(DR.read_run(tmp / "o"))                                                          # the Kaggle driver reads the local layout unchanged
        assert comp["tasks_finished"] == 2 and comp["thinking_policy"]["policy"] == "model_variant" and comp["sandbox"] in ("mxc", "mxc-processcontainer")
        assert "model_variant" in DR.render_md(comp) and "NOT_TESTED" in DR.render_md(comp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_resume_skips_finished_tasks_and_create_only_otherwise():
    tmp = Path(tempfile.mkdtemp())
    try:
        out = tmp / "o"
        rs, _, _ = _run(out, _tasks("P1-13"))
        raises(FileExistsError, _run, out, _tasks("P1-13", "P1-14"))                                              # not empty and no --resume: refused
        calls = []
        class Counting(FakeGen):
            def generate(self, req): calls.append(req.task_id); return super().generate(req)
        rs2, _, _ = _run(out, _tasks("P1-13", "P1-14"), gen=Counting(), resume=True)
        assert rs2["verdict"] == "OK" and rs2["resumed_tasks"] == ["P1-13"] and set(calls) == {"P1-14"} and rs2["done"] == 2        # the finished task was not regenerated
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_one_crashing_task_is_recorded_and_the_run_continues():
    tmp = Path(tempfile.mkdtemp())
    try:
        rs, _, _ = _run(tmp / "o", _tasks("P1-13", "P1-14", "P1-15"), gen=FakeGen(crash_ids={"P1-14"}))
        assert rs["verdict"] == "PARTIAL" and rs["done"] == 2 and rs["errors"] == ["P1-14"]
        r = json.loads((tmp / "o/tasks/P1-14/result.json").read_text("utf-8"))
        assert r["status"] == "ERROR" and "exploded" in r["error"] and not (tmp / "o/tasks/P1-14/DONE").exists() and (tmp / "o/tasks/P1-15/DONE").is_file()
        rs2, _, _ = _run(tmp / "o", _tasks("P1-13", "P1-14", "P1-15"), resume=True)                                 # the crashed task is retried on resume
        assert rs2["verdict"] == "OK" and rs2["done"] == 3 and rs2["errors"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_engine_failure_invalidates_the_task_and_three_in_a_row_stop_the_run():
    tmp = Path(tempfile.mkdtemp())
    try:
        rs, _, _ = _run(tmp / "a", _tasks("P1-13", "P1-14", "P1-15"), exe=FakeExe(engine_fail_ids={"P1-14"}))
        assert rs["errors"] == ["P1-14"] and rs["done"] == 2 and rs["verdict"] == "PARTIAL"                         # the engine died during P1-14 although the controller carried on: no DONE
        r = json.loads((tmp / "a/tasks/P1-14/result.json").read_text("utf-8"))
        assert r["status"] == "ERROR" and r["error_class"] == "EngineFailure"
        rs2, _, _ = _run(tmp / "b", _tasks("P1-13", "P1-14", "P1-15", "P1-16", "P1-17"), exe=FakeExe(engine_fail_ids={"P1-13", "P1-14", "P1-15"}))
        assert rs2["verdict"] == "ENGINE_UNAVAILABLE" and rs2["done"] == 0 and "P1-16" in rs2["pending_not_started"] and not (tmp / "b/tasks/P1-16").exists()      # stopped, nothing burned on a dead engine
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_fails_closed_when_the_engine_cannot_start_and_runs_no_task():
    tmp = Path(tempfile.mkdtemp())
    try:
        rs, gen, _ = _run(tmp / "o", _tasks("P1-13"), exe=FakeExe(start_error=PE.ProviderEngineError("sandbox backend 'mxc' unavailable")))
        assert rs["verdict"] == "ENGINE_UNAVAILABLE" and rs["done"] == 0 and "unavailable" in rs["error"] and not list((tmp / "o/tasks").glob("*"))
        assert (tmp / "o/RUN_SUMMARY.json").is_file()                                                              # a receipt exists even when nothing ran
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_java_is_not_tested_on_mxc_not_executed_and_listed():
    tmp = Path(tempfile.mkdtemp())
    try:
        class Spy(FakeGen):
            seen = []
            def generate(self, req): Spy.seen.append(req.task_id); return super().generate(req)
        rs, _, exe = _run(tmp / "o", _tasks("P1-13", "J1-20", "J1-21"), gen=Spy())
        assert rs["verdict"] == "OK" and rs["done"] == 1 and rs["not_tested"] == {"J1-20": "NOT_TESTED_NO_JDK", "J1-21": "NOT_TESTED_NO_JDK"} and set(Spy.seen) == {"P1-13"}
        r = json.loads((tmp / "o/tasks/J1-20/result.json").read_text("utf-8"))
        assert r["status"] == "NOT_TESTED" and r["stop_cause"] == "NOT_TESTED_NO_JDK" and "JDK" in r["reason"] and (tmp / "o/tasks/J1-20/NOT_TESTED").is_file() and not (tmp / "o/tasks/J1-20/DONE").exists()
        assert rs["errors"] == [] and rs["java_capability"]["runnable"] is False
        comp = DR.compile_results(DR.read_run(tmp / "o"))
        assert comp["not_tested"] == {"J1-20": "NOT_TESTED_NO_JDK", "J1-21": "NOT_TESTED_NO_JDK"} and comp["unfinished_or_errored"] == [] and comp["tasks_finished"] == 1
        rs_w, _, _ = _run(tmp / "w", _tasks("J1-20"), sandbox="wsl")                                                # the explicit WSL comparison runs Java (nothing hidden, labelled)
        assert rs_w["not_tested"] == {} and rs_w["sandbox"] == "wsl"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_stop_file_and_summary_survive_a_stop():
    tmp = Path(tempfile.mkdtemp())
    try:
        out = tmp / "o"
        class StopAfterFirst(FakeGen):
            def generate(self, req):
                (out / "STOP").write_text("x"); return super().generate(req)
        rs, _, _ = _run(out, _tasks("P1-13", "P1-14", "P1-15"), gen=StopAfterFirst())
        assert rs["verdict"] == "STOPPED" and rs["done"] == 1 and "P1-14" in rs["pending_not_started"] and json.loads((out / "RUN_SUMMARY.json").read_text("utf-8"))["verdict"] == "STOPPED"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_manifest_is_public_only_and_complete():
    tasks = LR.load_tasks()
    assert len(tasks) == 48 and sum(1 for t in tasks if t["language"] == "python") == 28 and sum(1 for t in tasks if t["language"] == "java") == 20
    assert all(set(t) == {"task_id", "language", "difficulty", "prompt", "public_tests"} for t in tasks)
    man = json.loads((REPO / "configs/el_bank80_rest_v1.json").read_text("utf-8"))
    assert not set(man["done_in_kaggle_run"]) & {t["task_id"] for t in tasks}                                      # none of the 32 Kaggle-finished tasks is run again
    assert len(LR.load_tasks(limit=3)) == 3
    p = LR.policy_for()
    assert p == {"policy": "model_variant", "profile_id": "qwen2.5-coder-1.5b-instruct", "model_glob": "local:qwen2.5-coder-1.5b-instruct", "ceiling": 4096}
    raises(B.RequestRefused, LR.policy_for, LR.PROFILE, 10)


@test
def t_think_shim_makes_a_non_thinking_model_extractable_and_is_a_noop_otherwise():
    from evidence_ladder.common.interfaces import GenerationRequest
    from evidence_ladder.models import qwen as Q
    NL = chr(10)
    plain = "Here:" + NL + "```python" + NL + "def solve(d):" + NL + "    return 1" + NL + "```"
    assert Q.make_generation(plain, tokens_prompt=1, tokens_generated=1, seed=0, prompt_template_id="t", prompt_sha256="h").extract_status == "no_code"      # the frozen extractor rejects it
    class G:
        language = "python"

        def generate(self, r):
            return [Q.make_generation(plain, tokens_prompt=1, tokens_generated=1, seed=0, prompt_template_id="t", prompt_sha256="h"),
                    Q.make_generation("<think>" + NL + "x" + NL + "</think>" + NL + plain, tokens_prompt=1, tokens_generated=1, seed=1, prompt_template_id="t", prompt_sha256="h")]
    sh = LR.ThinkShimGen(G())
    a, b = sh.generate(GenerationRequest("T", "p", 2, (0, 1), 10))
    assert a.extract_status == "ok" and "def solve" in a.code and a.text.endswith(plain) and sh.count == 1                      # the model's words are untouched, only an empty think block is prefixed
    assert b.extract_status == "ok" and b.text.count("</think>") == 1                                                            # a generation that already has one is left alone


@test
def t_inline_critic_is_called_in_the_same_run_and_applied_with_replayed_generations():
    tmp = Path(tempfile.mkdtemp())
    try:
        seen = {"calls": 0}

        def transport(packet, key_label):
            seen["calls"] += 1
            body = {"choices": [{"message": {"role": "assistant", "content": json.dumps({"experiment": {"kind": "function_call", "payload": "{}", "expected_by_candidate": {}}, "implications": "probe"}),
                                             "reasoning_content": "r"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            return 200, body
        gen = FakeGen()
        n_gen = []

        class Counting(FakeGen):
            def generate(self, req): n_gen.append(1); return super().generate(req)
        rs, _, _ = _run(tmp / "o", _tasks("P1-13"), gen=Counting(), critic_client=LR.GlmInlineCritic(transport=transport))
        r = json.loads((tmp / "o/tasks/P1-13/result.json").read_text("utf-8"))
        assert seen["calls"] >= 1 and r["critic_called"] is True and r["critic"]["state"] == "applied" and r["critic"]["applied_in_controller"] is True and r["critic"]["replay_exhausted"] is True
        assert r["stop_details"] != "critic_pending" and r["pending_critic_packets"] == [] and (tmp / "o/tasks/P1-13/DONE").is_file()         # the task completed inside this run
        assert (tmp / "o/tasks/P1-13/journal_before_critic.jsonl").is_file()
        kinds = [json.loads(l)["kind"] for l in (tmp / "o/tasks/P1-13/journal.jsonl").read_text("utf-8").splitlines()]
        assert "critic" in kinds
        before = len(n_gen)
        assert before == r["generate_calls"]                                                                         # the second controller pass regenerated NOTHING (replay)
        assert r["answer_label"] in ("SUPPORTED", "PLAUSIBLE", "UNVERIFIED", "BLOCKED", "VERIFIED") and "hidden grading NOT done" in r["label"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_inline_critic_without_keys_or_with_a_failing_api_is_recorded_and_the_run_continues():
    tmp = Path(tempfile.mkdtemp())
    try:
        rs, _, _ = _run(tmp / "a", _tasks("P1-13", "P1-14"))                                                          # default test critic: no key loader result -> api_unavailable, zero calls
        for tid in ("P1-13", "P1-14"):
            r = json.loads((tmp / "a/tasks" / tid / "result.json").read_text("utf-8"))
            assert r["status"] == "OK" and r["critic"]["called"] is True and r["critic"]["state"] == "api_unavailable" and "zero calls" in r["critic"]["detail"] and r["critic"]["applied_in_controller"] is False
            assert (tmp / "a/tasks" / tid / "DONE").is_file() and r["pending_critic_packets"]                              # honest: still pending, label not blocked, nothing faked
        assert rs["verdict"] == "OK"

        def boom(packet, key_label):
            raise RuntimeError("Authorization: " + "Bear" + "er " + "nv" + "api-" + "ABCDEFGHIJKLMNOP" + " leaked?")
        rs2, _, _ = _run(tmp / "b", _tasks("P1-13"), critic_client=LR.GlmInlineCritic(transport=boom))
        r2 = json.loads((tmp / "b/tasks/P1-13/result.json").read_text("utf-8"))
        assert r2["status"] == "OK" and r2["critic"]["state"] in ("api_unavailable", "critic_failed") and (tmp / "b/tasks/P1-13/DONE").is_file()
        blob = "".join(p.read_text("utf-8", "replace") for p in (tmp / "b").rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".txt"))
        assert ("nv" + "api-" + "ABCDEFGHIJKLMNOP") not in blob                                                                    # a key-shaped string from an exception never reaches a file
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_preflight_refuses_low_resources_unverified_model_and_runs_nothing():
    tmp = Path(tempfile.mkdtemp())
    try:
        rs, gen, _ = _run(tmp / "a", _tasks("P1-13"), resources_fn=lambda: {"free_ram_gb": 1.0, "free_vram_gb": 5.0})
        assert rs["verdict"] == "PREFLIGHT_REFUSED" and "free RAM" in rs["error"] and rs["done"] == 0 and not list((tmp / "a/tasks").glob("*"))
        rs, _, _ = _run(tmp / "b", _tasks("P1-13"), resources_fn=lambda: {"free_ram_gb": 16.0, "free_vram_gb": 1.0})
        assert rs["verdict"] == "PREFLIGHT_REFUSED" and "VRAM" in rs["error"]
        rs, _, _ = _run(tmp / "c", _tasks("P1-13"), resources_fn=lambda: {"free_ram_gb": 16.0, "free_vram_gb": None})
        assert rs["verdict"] == "PREFLIGHT_REFUSED"                                                                    # unknown free VRAM is a refusal, not a guess
        rs, _, _ = _run(tmp / "e", _tasks("P1-13"), resources_fn=lambda: {"free_ram_gb": 16.0, "free_vram_gb": 5.0, "free_commit_gb": 2.0}, min_free_commit_gb=4.5)
        assert rs["verdict"] == "PREFLIGHT_REFUSED" and "commit" in rs["error"] and not list((tmp / "e/tasks").glob("*"))
        class Unverified(FakeGen):
            def model_info(self): return {"profile_id": "x", "snapshot_verified": False}
        rs, _, _ = _run(tmp / "d", _tasks("P1-13"), gen=Unverified())
        assert rs["verdict"] == "PREFLIGHT_REFUSED" and "snapshot" in rs["error"] and not list((tmp / "d/tasks").glob("*"))
        assert (tmp / "d/RUN_SUMMARY.json").is_file()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_state_hook_is_optional_never_breaks_a_task_and_is_called_per_candidate():
    tmp = Path(tempfile.mkdtemp())
    try:
        calls = []

        class SC:
            enabled = True
            def capture(self, tid, cid, ctx, text): calls.append((tid, cid, "def solve" in text)); return None
        rs, _, _ = _run(tmp / "a", _tasks("P1-13"), state_hook=lambda d: SC())
        assert rs["internal_states"] == "hook" and calls and all(c[0] == "P1-13" and c[2] for c in calls)
        def bad_hook(d): raise RuntimeError("hook exploded")
        rs2, _, _ = _run(tmp / "b", _tasks("P1-13"), state_hook=bad_hook)
        r = json.loads((tmp / "b/tasks/P1-13/result.json").read_text("utf-8"))
        assert r["status"] == "OK" and "exploded" in r["state_capture_error"] and (tmp / "b/tasks/P1-13/DONE").is_file()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_REAL_mxc_selftest_exercises_every_stage_rich_repair_and_cheap_cases():
    from evidence_ladder.controller.sandbox_provider import backend_available
    if not backend_available("mxc")["available"]: raise Skip("MXC unavailable")
    tmp = Path(tempfile.mkdtemp())
    try:
        rc = LR.selftest(tmp / "st")
        rs = json.loads((tmp / "st/RUN_SUMMARY.json").read_text("utf-8"))
        miss = LR.selftest_checks(tmp / "st", rs, [LR.RICH_TASK, LR.REPAIR_TASK, LR.CHEAP_TASK])
        assert rc == 0 and miss == [], (rc, miss)                                                                 # every stage exercised, conformance PIPELINE_COMPLETE for all three tasks
        assert rs["pipeline_verdict"] == "PIPELINE_COMPLETE"
        sc = rs["stage_coverage"]
        assert sc["executed"] == 3 and sc["l2_validated"] >= 1 and sc["clusters_ge2"] >= 1 and sc["critic_applied"] >= 1 and sc["repair"] >= 1 and "STAGE COVERAGE" in sc["line"]
        assert rs["jail"]["engine"] == "mxc-processcontainer" and rs["l2_version"] == 2 and rs["done"] == 3
        r = json.loads((tmp / "st/tasks/P1-02/result.json").read_text("utf-8"))
        assert r["l2"]["verdict"] == "VALIDATED" and r["clusters"]["extended_clusters"] >= 2 and r["critic"]["state"] == "applied" and all(x["independence"] == "model_derived" for x in r["critic"]["critic_evidence"])
        # negative: remove a stage and the selftest reports it
        (tmp / "st/tasks/P1-02/controller_timeline.json").unlink()
        assert any("controller_timeline.json" in m for m in LR.selftest_checks(tmp / "st", rs, [LR.RICH_TASK]))
        j = (tmp / "st/tasks/P1-03/journal.jsonl")
        j.write_text("".join(l for l in j.read_text("utf-8").splitlines(True) if '"origin": "repair:' not in l), "utf-8")
        assert any("no repair candidate" in m for m in LR.selftest_checks(tmp / "st", rs, [LR.REPAIR_TASK]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_REAL_mxc_works_with_relative_paths_everywhere():
    from evidence_ladder.controller.sandbox_provider import backend_available
    if not backend_available("mxc")["available"]: raise Skip("MXC unavailable")
    base = Path(tempfile.mkdtemp())
    old = os.getcwd()
    try:
        os.chdir(base)
        spec = _p101()
        be = PE.ProviderExecutionBackend(backend="mxc", scratch=Path("rel_scratch"))                                 # RELATIVE scratch: used to make the container look for runner.py in C:\Windows\system32
        assert be.scratch.is_absolute()
        be.start()
        out = be.run_public(Artifact.make("P1-01", "python", _lookup(spec)), spec)
        assert all(o["status"] == "pass" for o in out), [(o["check_id"], o["status"], o.get("stderr_tail", "")[:120]) for o in out][:3]
        # the runner entry resolves a relative out dir as well
        rs, _, _ = _run(Path("rel_out"), _tasks("P1-13"))
        assert Path(rs["out_dir_abs"]).is_absolute() and Path(rs["out_dir_abs"]) == (base / "rel_out").resolve() and (base / "rel_out/tasks/P1-13/DONE").is_file()
        be.close()
    finally:
        os.chdir(old)
        shutil.rmtree(base, ignore_errors=True)


@test
def t_critic_timeouts_are_configurable_recorded_and_reach_the_transport():
    import evidence_ladder.critic_batch.transport as TR
    seen = {}
    real = TR.make_transport
    ok_body = {"choices": [{"message": {"role": "assistant", "content": json.dumps({"experiment": {"kind": "function_call", "payload": "{}", "expected_by_candidate": {}}, "implications": "probe"}),
                                       "reasoning_content": "r"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def fake_make(keys, **kw):
        seen.update(kw, labels=list(keys))
        return lambda packet, label: (200, ok_body)
    TR.make_transport = fake_make
    tmp = Path(tempfile.mkdtemp())
    try:
        c = LR.GlmInlineCritic(key_loader=lambda: {"key_1": "k1", "key_2": "k2"})
        assert c.settings()["read_timeout_s"] == 420.0 and c.settings()["total_timeout_s"] == 1800.0 and c.settings()["reasoning_effort"] == "max" and c.settings()["max_retries"] == 1
        rs, _, _ = _run(tmp / "o", _tasks("P1-13"), critic_client=LR.GlmInlineCritic(key_loader=lambda: {"key_1": "k1", "key_2": "k2"}, read_timeout_s=555.0, total_timeout_s=900.0))
        assert seen["read_timeout_s"] == 555.0 and seen["total_timeout_s"] == 900.0 and seen["labels"] == ["key_1", "key_2"]
        assert rs["critic"]["settings"]["read_timeout_s"] == 555.0 and rs["critic"]["settings"]["total_timeout_s"] == 900.0
        r = json.loads((tmp / "o/tasks/P1-13/result.json").read_text("utf-8"))
        assert r["critic"]["state"] == "applied" and r["critic"]["settings"]["read_timeout_s"] == 555.0
        assert "k1" not in json.dumps(r) and "k2" not in json.dumps(rs, default=str).replace("key_2", "")          # key values never recorded
    finally:
        TR.make_transport = real
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_critic_first_call_times_out_then_the_bounded_retry_succeeds():
    tmp = Path(tempfile.mkdtemp())
    try:
        n = {"calls": 0}

        def flaky(packet, label):
            n["calls"] += 1
            if n["calls"] == 1:
                e = TimeoutError("timed out waiting for response headers"); e.el_phase = "connect"; raise e
            body = {"choices": [{"message": {"role": "assistant", "content": json.dumps({"experiment": {"kind": "function_call", "payload": "{}", "expected_by_candidate": {}}, "implications": "probe"}),
                                             "reasoning_content": "r"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            return 200, body
        rs, _, _ = _run(tmp / "o", _tasks("P1-13"), critic_client=LR.GlmInlineCritic(transport=flaky))
        r = json.loads((tmp / "o/tasks/P1-13/result.json").read_text("utf-8"))
        assert n["calls"] == 2 and r["critic"]["state"] == "applied" and r["critic"]["applied_in_controller"] is True and (tmp / "o/tasks/P1-13/DONE").is_file()
        n["calls"] = 0
        def always(packet, label):
            n["calls"] += 1
            raise TimeoutError("timed out")
        rs2, _, _ = _run(tmp / "p", _tasks("P1-13"), critic_client=LR.GlmInlineCritic(transport=always))
        r2 = json.loads((tmp / "p/tasks/P1-13/result.json").read_text("utf-8"))
        assert n["calls"] == 2 and r2["critic"]["state"] == "api_unavailable" and r2["status"] == "OK" and (tmp / "p/tasks/P1-13/DONE").is_file()      # bounded (1 + 1 retry), then explicit state, task still finished
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_retry_critic_reruns_only_the_critic_step_with_saved_generations_and_archives_the_old_task():
    tmp = Path(tempfile.mkdtemp())
    try:
        out = tmp / "o"
        rs, _, _ = _run(out, _tasks("P1-13", "P1-14"))                                                              # first run: no key -> api_unavailable on both
        old = json.loads((out / "tasks/P1-13/result.json").read_text("utf-8"))
        assert old["critic"]["state"] == "api_unavailable" and old["pending_critic_packets"] and LR.needs_critic_retry(out / "tasks/P1-13")
        calls = []

        class NoRegeneration(FakeGen):
            def generate(self, req): calls.append(req.task_id); return super().generate(req)

        def ok(packet, label):
            body = {"choices": [{"message": {"role": "assistant", "content": json.dumps({"experiment": {"kind": "function_call", "payload": "{}", "expected_by_candidate": {}}, "implications": "probe"}),
                                             "reasoning_content": "r"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            return 200, body
        rs2, _, _ = _run(out, _tasks("P1-13", "P1-14"), gen=NoRegeneration(), resume=True, retry_critic=True, finalize=True, critic_client=LR.GlmInlineCritic(transport=ok))
        assert calls == [], calls                                                                                       # nothing was regenerated: the saved generations were replayed
        assert [e["applied"] for e in rs2["critic_retry"]] == [True, True]
        for tid in ("P1-13", "P1-14"):
            new = json.loads((out / "tasks" / tid / "result.json").read_text("utf-8"))
            assert new["critic"]["state"] == "applied" and new["critic"]["applied_in_controller"] and new["retry"]["critic_retry"] is True and (out / "tasks" / tid / "DONE").is_file()
            assert new["retry"]["previous"]["critic"]["state"] == "api_unavailable" and new["retry"]["critic_settings"]["read_timeout_s"] == 420.0 and new["retry"]["started_utc"]
            assert new["pending_critic_packets"] == []
            arch = out / "tasks_archive" / (tid + ".v1")
            assert (arch / "result.json").is_file() and json.loads((arch / "result.json").read_text("utf-8"))["critic"]["state"] == "api_unavailable" and (arch / "journal.jsonl").is_file()     # old evidence kept
            assert new["journal_hash"] != old["journal_hash"] or tid != "P1-13"                                        # the critic entry is in the new journal
        assert (out / "retry_critic_log.jsonl").is_file() and len((out / "retry_critic_log.jsonl").read_text("utf-8").splitlines()) == 2
        assert not (out / "_retry").exists() and (out / "BANK_RESULTS.json").is_file() and (out / "CONFORMANCE.json").is_file()      # reports rebuilt at the end
        assert not LR.needs_critic_retry(out / "tasks/P1-13")                                                           # nothing left to retry
        # label: the controller decides through trusted evidence only; both labels are recorded in the log
        assert all("label_before" in e and "label_after" in e for e in rs2["critic_retry"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_retry_critic_failure_leaves_the_finished_task_untouched_and_keeps_the_attempt():
    tmp = Path(tempfile.mkdtemp())
    try:
        out = tmp / "o"
        _run(out, _tasks("P1-13"))
        before = (out / "tasks/P1-13/result.json").read_bytes()

        def dead(packet, label): raise TimeoutError("timed out")
        rs, _, _ = _run(out, _tasks("P1-13"), resume=True, retry_critic=True, critic_client=LR.GlmInlineCritic(transport=dead))
        assert rs["critic_retry"][0]["applied"] is False and rs["critic_retry"][0].get("failed_attempt_kept_at")
        assert (out / "tasks/P1-13/result.json").read_bytes() == before and (out / "tasks/P1-13/DONE").is_file() and not (out / "tasks_archive").exists()      # the original is untouched
        assert (out / rs["critic_retry"][0]["failed_attempt_kept_at"]).is_dir()
        raises(ValueError, _run, tmp / "x", _tasks("P1-13"), retry_critic=True)                                         # --retry-critic needs --resume
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ REAL sandboxes (skipped when unavailable)
def _p101():
    t = next(json.loads(l) for l in (REPO / "results/raw/codex-pilot1-20261016/fixed_v2/worker/tasks.jsonl").read_text("utf-8").splitlines() if '"P1-01"' in l[:40])
    return TaskSpec("P1-01", "py1:P1-01", "python", "python", t["prompt"], t["public_tests"], "easy")


def _lookup(spec):
    table = {json.dumps(c["input"], sort_keys=True): c["expected"] for c in spec.public_tests}
    return "import json\nT = %r\ndef solve(d):\n    return T[json.dumps(d, sort_keys=True)]\n" % table


@test
def t_REAL_mxc_public_examples_lookup_passes_wrong_and_hostile_fail():
    from evidence_ladder.controller.sandbox_provider import backend_available
    av = backend_available("mxc")
    if not av["available"]: raise Skip("MXC unavailable: %s" % av["reasons"])
    spec = _p101()
    be = PE.ProviderExecutionBackend(backend="mxc", scratch=Path(tempfile.mkdtemp()))
    info = be.start()
    assert info["engine"] == "mxc-processcontainer" and info["backend"] == "mxc"
    cases = {"lookup": (_lookup(spec), True), "wrong": ("def solve(d):\n    return None\n", False), "syntax": ("def solve(d:\n  pass", False), "raises": ("def solve(d):\n    raise ValueError('x')\n", False),
             "hard_crash": ("import os\nos._exit(5)\ndef solve(d): return 1\n", False), "network": ("import socket\ndef solve(d):\n    socket.create_connection(('1.1.1.1', 53), 2)\n    return 1\n", False),
             "forged_print": ("def solve(d):\n    print('{\"ok\": true, \"value\": 1}'); return 1\n", False),
             "host_file": ("def solve(d):\n    return open(r'%s').read()[:20]\n" % str(REPO / "CLAUDE.md"), False)}
    secs = {}
    for name, (src, should_pass) in cases.items():
        t0 = time.time()
        out = be.run_public(Artifact.make("P1-01", "python", src), spec)
        secs[name] = round(time.time() - t0, 3)
        assert (all(o["status"] == "pass" for o in out)) is should_pass, (name, [(o["check_id"], o["status"], o.get("failure_class")) for o in out][:3])
    loop = be.run_public(Artifact.make("P1-01", "python", "def solve(d):\n    while True: pass\n"), spec)
    assert all(o["status"] == "timeout" for o in loop[1:])
    ex = be.run_experiment(Artifact.make("P1-01", "python", _lookup(spec)), {"kind": "function_call", "payload": json.dumps(spec.public_tests[0]["input"])})
    assert ex["status"] == "pass"
    MEASURED["mxc_seconds_per_candidate_run"] = secs
    be.close()


@test
def t_REAL_wsl_adapter_handshake_and_public_examples():
    from evidence_ladder.controller.wsl_sandbox import available
    if not available(): raise Skip("WSL does not answer")
    spec = _p101()
    be = WE.WslExecutionBackend(repo=REPO)
    try:
        h = be.start()
        assert h["engine"] == "wsl-bwrap" and h["ok"] and h["bwrap_version"] and h["java_exists"]
        ok = be.run_public(Artifact.make("P1-01", "python", _lookup(spec)), spec)
        bad = be.run_public(Artifact.make("P1-01", "python", "def solve(d):\n    return None\n"), spec)
        assert all(o["status"] == "pass" for o in ok) and bad[1]["status"] == "fail"
        assert be.backend_info()["engine"] == "wsl-bwrap"
    finally:
        be.close()


if __name__ == "__main__":
    failed, skipped = {}, {}
    for t in TESTS:
        try:
            t()
        except Skip as s:
            skipped[t.__name__] = str(s)
        except Exception:
            failed[t.__name__] = traceback.format_exc()[-1800:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed) - len(skipped), "failed": failed, "skipped": skipped, "measured": MEASURED}, indent=1))
    sys.exit(1 if failed else 0)
