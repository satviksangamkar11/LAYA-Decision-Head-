"""[EL-WSLC] CPU tests of the capability-based language dispatch (evidence_ladder/execution/toolchains.py + provider_engine.py) and of the runner's WSLC mode: the toolchain registry, the real-probe rule
(RUNNABLE only after the probe passed INSIDE the backend), Java compile + run through the port with the input staged as a file, fail-closed behaviour, manifest checks, provenance fields, and the Java
PARITY check against the qualified WSL Java engine (skipped when WSL is not available). A real WSLC test runs only if the 'wslc' provider is registered and answers.
  PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe evidence_ladder/orchestrator/tests/phaseF3d_tests.py   -> JSON {"tests","passed","failed","skipped"}"""
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
from evidence_ladder import bank40 as B                                  # noqa: E402
from evidence_ladder.common.interfaces import Artifact, TaskSpec         # noqa: E402
from evidence_ladder.controller.mxc_sandbox import UnsupportedCommand, SandboxUnavailable   # noqa: E402
from evidence_ladder.execution import provider_engine as PE, toolchains as TC, wsl_engine as WE   # noqa: E402
import el_bank_local_run as LR                                           # noqa: E402

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


R0 = {"exit_code": 0, "killed": False, "stdout": "", "stderr": "", "protected_unchanged": True, "pre_tree_sha256": "a", "post_tree_sha256": "a", "persisted": True}


class ScriptedSandbox:
    """Provider-port double for the PROBE tests: answers by a function of the command."""
    env_label = "fake-wslc-label"
    image_digest = "sha256:" + "ab" * 32

    def __init__(self, fn):
        self.fn, self.cmds = fn, []

    def run(self, workspace, cmd, persist=False, timeout=60):
        self.cmds.append(cmd)
        return self.fn(workspace, cmd)


def _ok_probe(ws, cmd):
    if cmd.startswith("python3 --version"): return dict(R0, stdout="Python 3.12.3\n")
    if "javac -version" in cmd or "javac -version" in cmd: return dict(R0, stderr="javac 21.0.4\nopenjdk version \"21.0.4\"\n")
    raise AssertionError("unexpected command " + cmd)


def _be(sb, backend="wslc"):
    return PE.ProviderExecutionBackend(sb, backend=backend, scratch=Path(tempfile.mkdtemp()))


# ------------------------------------------------------------------ registry and probe
@test
def t_registry_declares_seven_languages_and_only_tested_runners():
    assert set(TC.REGISTRY) == {"python", "java", "c", "cpp", "javascript", "go", "rust"}
    assert {k for k, v in TC.REGISTRY.items() if v.runner} == {"python", "java"}                                   # c / cpp / js / go / rust are registered, never claimed
    for lang, tc in TC.REGISTRY.items():
        assert tc.extension.startswith(".") and tc.probe and tc.run and tc.tool_label
    assert TC.not_tested_status("java") == "NOT_TESTED_NO_JDK" and TC.not_tested_status("c") == "NOT_TESTED_NO_GCC" and TC.not_tested_status("zig") == "NOT_TESTED_NO_TOOLCHAIN"


@test
def t_language_is_runnable_only_after_the_probe_passes_inside_the_backend():
    be = _be(ScriptedSandbox(_ok_probe))
    caps = be.probe_toolchains(["python", "java"])
    assert caps["python"]["runnable"] and caps["python"]["version"].startswith("Python 3.12") and caps["java"]["runnable"] and caps["java"]["version"].startswith("javac 21")
    assert be.toolchain_versions() == {"python": caps["python"]["version"], "java": caps["java"]["version"]}
    # registered but without an implemented runner: never runnable, even though no probe was needed to say so
    c = be.probe_toolchains(["c", "cpp", "javascript", "go", "rust", "zig"])
    assert all(not v["runnable"] for v in c.values()) and c["c"]["status"] == "NOT_TESTED_NO_RUNNER" and c["zig"]["status"] == "NOT_TESTED_NO_TOOLCHAIN"
    assert not any("gcc" in x or "rustc" in x for x in be.sandbox.cmds)                                           # nothing is probed that cannot be run
    # the probe fails inside the backend -> NOT_TESTED with the exact reason
    def no_java(ws, cmd):
        if "javac" in cmd: return dict(R0, exit_code=127, stderr="sh: javac: not found\n")
        return _ok_probe(ws, cmd)
    be2 = _be(ScriptedSandbox(no_java))
    j = be2.probe_toolchains(["python", "java"])["java"]
    assert j["runnable"] is False and j["status"] == "NOT_TESTED_NO_JDK" and "not found" in j["reason"] and be2.toolchains["python"]["runnable"]
    # the backend cannot even start a shell command (the MXC process container)
    def mxc_like(ws, cmd):
        if cmd.startswith("sh "): raise UnsupportedCommand("the dialect runs python3 <args> only")
        return _ok_probe(ws, cmd)
    j = _be(ScriptedSandbox(mxc_like), "mxc").probe_toolchains(["java"])["java"]
    assert j["runnable"] is False and j["status"] == "NOT_TESTED_NO_JDK" and "UnsupportedCommand" in j["reason"]


@test
def t_unprobed_or_unrunnable_languages_raise_and_never_execute():
    sb = ScriptedSandbox(_ok_probe)
    be = _be(sb)
    art = Artifact.make("J1-01", "java", "class Solution {}")
    spec = TaskSpec("J1-01", "java1:J1-01", "java", "java", "p", [{"input": {}, "expected": 1}])
    raises(PE.JavaNotRunnable, be.run_public, art, spec)                                                          # not probed -> not runnable
    be.probe_toolchains(["java"]); sb.cmds.clear()
    be.toolchains["java"] = {"runnable": False, "status": "NOT_TESTED_NO_JDK", "reason": "x"}
    raises(PE.JavaNotRunnable, be.run_public, art, spec)
    assert sb.cmds == []                                                                                          # nothing reached the sandbox
    e = raises(PE.JavaNotRunnable, be.run_public, Artifact.make("C1", "c", "int main(){}"), TaskSpec("C1", "c1:C1", "c", "c", "p", [{"input": {}, "expected": 1}]))
    assert e.infra is True


@test
def t_probe_infrastructure_failure_raises_instead_of_hiding_a_language():
    def boom(ws, cmd): raise RuntimeError("bridge died")
    raises(PE.ProviderEngineError, _be(ScriptedSandbox(boom)).probe_toolchains, ["python"])
    class Gone:
        def run(self, *a, **k): raise SandboxUnavailable("no wslc")
    raises(PE.ProviderEngineError, _be(Gone()).probe_toolchains, ["java"])


# ------------------------------------------------------------------ Java through the port, against REAL javac / java inside WSL (test double of the port; NOT the product sandbox)
def wsl_ok() -> bool:
    try:
        from evidence_ladder.controller.wsl_sandbox import available
        return bool(available())
    except Exception:      # noqa: BLE001
        return False


class WslShellPortSandbox:
    """TEST DOUBLE of the provider port: runs the command in the workspace inside WSL (real sh, timeout, javac, java). It stands in for a container ONLY here; it is not isolated and is never used by the runner."""
    env_label = "test-double-wsl-shell"
    image_digest = None

    def run(self, workspace, cmd, persist=False, timeout=60):
        wp = WE.wsl_path(Path(workspace))
        (Path(workspace) / "cmd.sh").write_text(cmd + "\n", encoding="utf-8", newline="\n")
        p = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "-e", "bash", "-c", 'cd "%s" && export PATH=/opt/toolchains/jdk-26.0.1/bin:/opt/toolchains/bin:$PATH && bash cmd.sh' % wp],
                           capture_output=True, timeout=timeout)
        return dict(R0, exit_code=p.returncode, stdout=p.stdout.decode("utf-8", "replace")[-6000:], stderr=p.stderr.decode("utf-8", "replace")[-6000:])


JREF01 = ("import java.util.*;\npublic class Solution {\n  public static Object solve(Map<String,Object> d) {\n    String t = (String) d.get(\"text\"); StringBuilder b = new StringBuilder();\n"
          "    for (char c : t.toCharArray()) if ((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')) b.append(Character.toLowerCase(c));\n"
          "    String s = b.toString(); return new StringBuilder(s).reverse().toString().equals(s);\n  }\n}\n")
JCASES = {"reference": (JREF01, True),
          "always_true": ("import java.util.*;\npublic class Solution { public static Object solve(Map<String,Object> d) { return true; } }\n", False),
          "compile_error": ("public class Solution { public static Object solve(java.util.Map<String,Object> d) { return 1 +; } }\n", False),
          "throws": ("import java.util.*;\npublic class Solution { public static Object solve(Map<String,Object> d) { throw new IllegalStateException(\"boom\"); } }\n", False),
          "infinite_loop": ("import java.util.*;\npublic class Solution { public static Object solve(Map<String,Object> d) { while (true) { } } }\n", False),
          "wrong_type": ("import java.util.*;\npublic class Solution { public static Object solve(Map<String,Object> d) { return \"yes\"; } }\n", False)}


def _j01():
    for l in (REPO / "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker/tasks.jsonl").read_text("utf-8").splitlines():
        if l.strip() and json.loads(l)["id"] == "J1-01":
            d = json.loads(l)
            return TaskSpec("J1-01", "java1:J1-01", "java", "java", d["prompt"], d["public_tests"], "easy")


def _started_java_backend():
    be = _be(WslShellPortSandbox(), "wslc")
    caps = be.probe_toolchains(["python", "java"])
    assert caps["java"]["runnable"], caps
    return be


def _summary(outs):
    return [(o["check_id"], o.get("example_index"), o["status"], o.get("failure_class"), o.get("observed")) for o in outs]


@test
def t_java_through_the_port_reference_passes_wrong_and_broken_candidates_do_not():
    if not wsl_ok(): raise Skip("WSL unavailable")
    be, spec = _started_java_backend(), _j01()
    secs = {}
    for name, (src, should_pass) in JCASES.items():
        t0 = time.time()
        out = be.run_public(Artifact.make("J1-01", "java", src), spec)
        secs[name] = round(time.time() - t0, 2)
        assert (all(o["status"] == "pass" for o in out)) is should_pass, (name, _summary(out)[:3])
    ce = be.run_public(Artifact.make("J1-01", "java", JCASES["compile_error"][0]), spec)
    assert ce[0]["failure_class"] == "compile_error" and all(o["status"] == "missing_evidence" for o in ce[1:])
    th = be.run_public(Artifact.make("J1-01", "java", JCASES["throws"][0]), spec)
    assert th[1]["failure_class"] == "runtime_error"
    lp = be.run_public(Artifact.make("J1-01", "java", JCASES["infinite_loop"][0]), spec)
    assert all(o["status"] == "timeout" for o in lp[1:])
    ex = be.run_experiment(Artifact.make("J1-01", "java", JREF01), {"kind": "function_call", "payload": json.dumps({"text": "abba"})})
    assert ex["status"] == "pass" and ex["observed"] is True
    assert be.exec_log and all(set(("exit_code", "killed", "stdout", "stderr", "protected_unchanged", "pre_tree_sha256", "post_tree_sha256", "persisted", "backend", "image_digest", "toolchain", "source_sha256", "language")) <= set(x) for x in be.exec_log)
    assert {x["language"] for x in be.exec_log} == {"java"} and be.exec_log[0]["toolchain"]["java"]
    MEASURED["java_seconds_per_candidate_run_via_wsl_shell_double"] = secs


@test
def t_java_engine_fails_closed_when_the_shell_script_cannot_run():
    spec = _j01()
    be = _be(ScriptedSandbox(_ok_probe))
    be.probe_toolchains(["java"])
    be.sandbox.fn = lambda ws, cmd: dict(R0, exit_code=127, stderr="sh: not found")                                 # no compile.rc left behind: sh / timeout / javac missing
    raises(PE.ProviderEngineError, be.run_public, Artifact.make("J1-01", "java", JREF01), spec)
    be.sandbox.fn = lambda ws, cmd: (_ for _ in ()).throw(RuntimeError("container died"))
    raises(PE.ProviderEngineError, be.run_public, Artifact.make("J1-01", "java", JREF01), spec)
    be.sandbox.fn = lambda ws, cmd: dict(R0, exit_code=-1, killed=True, persisted=False)
    out = be.run_public(Artifact.make("J1-01", "java", JREF01), spec)
    assert all(o["status"] == "timeout" for o in out[1:])                                                           # wall-clock kill = a timeout verdict, not a pass


@test
def t_java_input_is_staged_as_a_file_and_redirected_not_piped():
    seen = {}

    def fn(ws, cmd):
        if "javac -version" in cmd or cmd.startswith("python3"): return _ok_probe(ws, cmd)
        seen["cmd"] = cmd
        seen["files"] = sorted(p.name for p in Path(ws).iterdir())
        seen["in"] = (Path(ws) / "in.json").read_text("utf-8")
        return dict(R0, exit_code=1, stderr="stop")
    be = _be(ScriptedSandbox(fn))
    be.probe_toolchains(["java"])
    try:
        be.run_public(Artifact.make("J1-01", "java", JREF01), _j01())
    except PE.ProviderEngineError:
        pass
    assert seen["cmd"].startswith("sh -c ") and "< in.json" in seen["cmd"] and "javac -proc:none -encoding UTF-8" in seen["cmd"] and "-Xmx128m" in seen["cmd"] and "timeout" in seen["cmd"]
    assert {"Solution.java", "Json.java", "Runner.java", "in.json"} <= set(seen["files"])
    lines = seen["in"].splitlines()
    assert len(lines) == len(_j01().public_tests) and json.loads(lines[0]) == _j01().public_tests[0]["input"]        # one JSON input per line, exactly the public inputs


# ------------------------------------------------------------------ PARITY with the qualified WSL Java engine
@test
def t_PARITY_java_verdicts_equal_the_qualified_wsl_engine():
    if not wsl_ok(): raise Skip("WSL unavailable")
    spec = _j01()
    new = _started_java_backend()
    ref = WE.WslExecutionBackend(repo=REPO)
    try:
        ref.start()
        for name, (src, _) in JCASES.items():
            art = Artifact.make("J1-01", "java", src)
            a, b = _summary(ref.run_public(art, spec)), _summary(new.run_public(art, spec))
            assert a == b, (name, a, b)                                                                              # same verdict, same status / failure class / observed per example
        art = Artifact.make("J1-01", "java", JREF01)
        for payload in ({"text": "abba"}, {"text": "ab"}, {"text": ""}):
            ea = ref.run_experiment(art, {"kind": "function_call", "payload": json.dumps(payload)})
            eb = new.run_experiment(art, {"kind": "function_call", "payload": json.dumps(payload)})
            assert (ea["status"], ea.get("observed"), ea.get("failure_class")) == (eb["status"], eb.get("observed"), eb.get("failure_class")), (payload, ea, eb)
        MEASURED["parity_candidates"] = list(JCASES)
    finally:
        ref.close()


# ------------------------------------------------------------------ manifest and runner
@test
def t_java20_manifest_is_exact_public_and_disjoint_from_the_kaggle_java_set():
    p = REPO / "configs/el_bank_java20_wslc_v1.json"
    man = json.loads(p.read_text("utf-8"))
    ids = [t["task"] for t in man["tasks"]]
    assert len(ids) == 20 == man["total"] and all(t["language"] == "java" for t in man["tasks"]) and ids == ["J1-17", "J1-20"] + ["J1-%d" % i for i in range(23, 41)]
    kag = json.loads((REPO / "configs/el_bank40_v1.json").read_text("utf-8"))
    assert not set(ids) & set(kag["java"]["tasks"])
    rest = json.loads((REPO / "configs/el_bank80_rest_v1.json").read_text("utf-8"))
    assert ids == [t["task"] for t in rest["tasks"] if t["language"] == "java"] and man["sources"] == rest["sources"] and not (REPO / "configs/el_wslc_java20_v1.json").exists()
    tasks = LR.load_tasks(p)
    assert len(tasks) == 20 and all(set(t) == {"task_id", "language", "difficulty", "prompt", "public_tests"} for t in tasks)       # only public prompts / tests are visible to the runtime
    LR.public_guard(tasks)
    bad = dict(man, total=21)
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "m.json").write_text(json.dumps(bad), "utf-8")
        raises(AssertionError, LR.load_tasks, tmp / "m.json")                                                        # a manifest with the wrong count is refused
        bad2 = dict(man, source_sha256="0" * 64); (tmp / "m2.json").write_text(json.dumps(bad2), "utf-8")
        raises(AssertionError, LR.load_tasks, tmp / "m2.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class FakeGen:
    gpu_name = "FAKE"
    language = "java"

    def model_info(self): return {"backend": "fake", "profile_id": "fake", "profile_sha256": "f" * 64, "snapshot_verified": True}

    def generate(self, req):
        from evidence_ladder.models import qwen as Q
        return [Q.make_generation("```java\n" + JREF01 + "```", tokens_prompt=5, tokens_generated=9, seed=s, prompt_template_id=req.prompt_template_id, prompt_sha256=B.sha(req.prompt), language="java", wall_seconds=0.01)
                for s in list(req.seeds)[:req.n]]


class ProbingExe:
    """Execution backend double with the probe API: java runnable or not, per test."""
    env_label = "mxc-wslc"
    image_digest = "sha256:" + "cd" * 32

    def __init__(self, java_ok: bool, start_error=None):
        self.java_ok, self.start_error, self.source_sha256, self.toolchains, self.started = java_ok, start_error, None, {}, False

    def probe_toolchains(self, langs): return {}

    def start(self, languages=("python", "java")):
        if self.start_error: raise self.start_error
        self.started = True
        self.toolchains = {"python": {"runnable": True, "status": "RUNNABLE", "reason": "", "version": "Python 3.12.3"},
                           "java": ({"runnable": True, "status": "RUNNABLE", "reason": "", "version": "javac 21.0.4"} if self.java_ok else
                                    {"runnable": False, "status": "NOT_TESTED_NO_JDK", "reason": "probe 'javac' exited 127: not found"})}
        return {"engine": self.env_label, "backend": "wslc", "toolchains": {k: self.toolchains[k] for k in languages if k in self.toolchains}, "image_digest": self.image_digest}

    def run_public(self, a, t):
        ok = "isPal" in a.source or "reverse" in a.source
        return [{"check_id": "parse_v1", "status": "pass"}] + [{"check_id": "public_examples_v1", "example_index": i, "status": "pass" if ok else "fail", "failure_class": "" if ok else "wrong_answer", "observed": "x"}
                                                              for i in range(len(t.public_tests))]

    def run_experiment(self, a, e): return {"check_id": "critic_experiment_v1", "status": "pass", "observed": True}
    def backend_info(self): return {"engine": self.env_label, "backend": "wslc"}
    def drain_exec_log(self): return [dict(R0, backend="wslc", image_digest=self.image_digest, toolchain={"java": "javac 21.0.4"}, source_sha256=self.source_sha256, language="java")]
    def close(self): pass


def _jtasks(*ids):
    return [{"task_id": i, "language": "java", "difficulty": "medium", "prompt": "Return palindrome for " + i, "public_tests": [{"input": {"text": "aba"}, "expected": True}, {"input": {"text": "ab"}, "expected": False}]} for i in ids]


def _run(out, tasks, exe, **kw):
    kw.setdefault("critic_client", LR.GlmInlineCritic(key_loader=lambda: {}))
    kw.setdefault("resources_fn", lambda: {"free_ram_gb": 16.0, "free_vram_gb": 5.0, "free_commit_gb": 9.0})
    kw.setdefault("finalize", False)
    return LR.run(Path(out), tasks, lambda o: FakeGen(), lambda o: exe, **kw)


@test
def t_runner_wslc_runs_java_after_a_real_probe_and_records_the_backend_identity():
    tmp = Path(tempfile.mkdtemp())
    try:
        exe = ProbingExe(java_ok=True)
        rs = _run(tmp / "o", _jtasks("J1-17", "J1-20"), exe, sandbox="wslc")
        assert rs["verdict"] == "OK" and rs["done"] == 2 and rs["not_tested"] == {} and rs["engine_status"] == "OK"                    # Java runs: zero NOT_TESTED_NO_JDK
        assert rs["toolchains"]["java"]["version"] == "javac 21.0.4" and rs["image_digest"] == exe.image_digest and rs["engine"] == "mxc-wslc" and rs["jail"]["engine"] == "mxc-wslc"
        assert len(rs["source_sha256"]) == 64 and rs["code_commit"] == rs["commit"] and rs["scorer"] == "none" and rs["l2_version"] == 2 and rs["profile"]["sha256"] == "f" * 64
        assert exe.source_sha256 == rs["source_sha256"] and exe.started
        r = json.loads((tmp / "o/tasks/J1-17/result.json").read_text("utf-8"))
        assert r["engine"] == "mxc-wslc" and r["toolchain"]["version"] == "javac 21.0.4" and len(r["source_sha256"]) == 64 and r["controller_config"]["l2_version"] == 2 and len(r["task_prompt_sha256"]) == 64
        ex = [json.loads(l) for l in (tmp / "o/tasks/J1-17/exec_results.jsonl").read_text("utf-8").splitlines()]
        assert ex and {"exit_code", "killed", "stdout", "stderr", "protected_unchanged", "pre_tree_sha256", "post_tree_sha256", "persisted", "backend", "image_digest", "toolchain", "source_sha256"} <= set(ex[0])
        calls = [json.loads(l) for l in (tmp / "o/tasks/J1-17/exec_calls.jsonl").read_text("utf-8").splitlines()]
        assert calls and {"method", "artifact_id", "experiment", "result"} == set(calls[0]) and calls[0]["method"] in ("run_public", "run_experiment")
        assert isinstance(calls[0]["result"], list) or calls[0]["method"] == "run_experiment"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_marks_a_language_not_tested_with_the_probe_reason_and_never_falls_back():
    tmp = Path(tempfile.mkdtemp())
    try:
        rs = _run(tmp / "o", _jtasks("J1-17", "J1-20"), ProbingExe(java_ok=False), sandbox="wslc")
        assert rs["done"] == 0 and rs["not_tested"] == {"J1-17": "NOT_TESTED_NO_JDK", "J1-20": "NOT_TESTED_NO_JDK"} and rs["toolchains"]["java"]["runnable"] is False
        r = json.loads((tmp / "o/tasks/J1-17/result.json").read_text("utf-8"))
        assert r["status"] == "NOT_TESTED" and "exited 127" in r["reason"] and (tmp / "o/tasks/J1-17/NOT_TESTED").is_file() and not (tmp / "o/tasks/J1-17/DONE").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_wslc_unavailable_blocks_the_run_with_a_reason_and_runs_nothing():
    tmp = Path(tempfile.mkdtemp())
    import evidence_ladder.controller.sandbox_provider as SP
    real = SP.make_sandbox
    SP.make_sandbox = lambda *a, **k: (_ for _ in ()).throw(SandboxUnavailable("MXC WSLC unavailable (test)"))      # deterministic: independent of whether the real WSLC happens to be free
    try:
        rs = LR.run(tmp / "o", _jtasks("J1-17"), lambda o: FakeGen(), LR.exe_factory_for("wslc"), sandbox="wslc", critic_client=LR.GlmInlineCritic(key_loader=lambda: {}), finalize=False,
                    resources_fn=lambda: {"free_ram_gb": 16.0, "free_vram_gb": 5.0, "free_commit_gb": 9.0})
        assert rs["verdict"] == "ENGINE_UNAVAILABLE" and rs["engine_status"].startswith("BLOCKED") and "wslc" in rs["engine_status"].lower() and rs["done"] == 0
        assert not list((tmp / "o/tasks").glob("*")) and (tmp / "o/RUN_SUMMARY.json").is_file()                          # no ProcessContainer / WSL / host fallback
        rs2 = _run(tmp / "p", _jtasks("J1-17"), ProbingExe(java_ok=True, start_error=PE.ProviderEngineError("wslc image missing")), sandbox="wslc")
        assert rs2["verdict"] == "ENGINE_UNAVAILABLE" and "image missing" in rs2["engine_status"] and rs2["done"] == 0
    finally:
        SP.make_sandbox = real
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_runner_loads_the_model_then_starts_the_engine_and_probes_before_any_task():
    tmp = Path(tempfile.mkdtemp())
    try:
        order = []

        class Exe(ProbingExe):
            def start(self, languages=("python", "java")):
                order.append("probe"); return super().start(languages)

        def gen_factory(o): order.append("model"); return FakeGen()
        LR.run(tmp / "o", _jtasks("J1-17"), gen_factory, lambda o: Exe(True), sandbox="wslc", critic_client=LR.GlmInlineCritic(key_loader=lambda: {}), finalize=False,
               resources_fn=lambda: {"free_ram_gb": 16.0, "free_vram_gb": 5.0, "free_commit_gb": 9.0})
        assert order == ["model", "probe"]      # the model is on the GPU before the sandbox VM starts (RAM)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_REAL_mxc_python_with_the_trusted_verifier_flag_matches_the_legacy_runner():
    from evidence_ladder.controller.sandbox_provider import backend_available
    if not backend_available("mxc")["available"]: raise Skip("MXC unavailable")
    t = next(json.loads(l) for l in (REPO / "results/raw/codex-pilot1-20261016/fixed_v2/worker/tasks.jsonl").read_text("utf-8").splitlines() if '"P1-01"' in l[:40])
    spec = TaskSpec("P1-01", "py1:P1-01", "python", "python", t["prompt"], t["public_tests"], "easy")
    table = {json.dumps(c["input"], sort_keys=True): c["expected"] for c in t["public_tests"]}
    NL = chr(10)
    good = "import json" + NL + "T = %r" % table + NL + "def solve(d):" + NL + "    return T[json.dumps(d, sort_keys=True)]" + NL
    legacy = PE.ProviderExecutionBackend(backend="mxc", scratch=Path(tempfile.mkdtemp()))
    trusted = PE.ProviderExecutionBackend(backend="mxc", scratch=Path(tempfile.mkdtemp()), trusted=True)
    assert legacy.backend_info()["engine_version"] == "legacy_runner" and trusted.backend_info()["engine_version"] == "trusted_verify_v1"
    for src in (good, "def solve(d):" + NL + "    return None" + NL, "def solve(d:" + NL + "  pass", "def solve(d):" + NL + "    raise ValueError('x')" + NL):
        art = Artifact.make("P1-01", "python", src)
        assert _summary(legacy.run_public(art, spec)) == _summary(trusted.run_public(art, spec)), src[:30]
    assert trusted.exec_log and all(x["backend"] == "mxc" and "language" in x for x in trusted.exec_log)
    legacy.close(); trusted.close()


@test
def t_REAL_wslc_probe_and_java_run_if_the_provider_is_registered():
    from evidence_ladder.controller import sandbox_provider as SP
    if "wslc" not in getattr(SP, "BACKENDS", ()): raise Skip("the 'wslc' provider is not registered yet (Agent 1)")
    av = SP.backend_available("wslc")
    if not av["available"]: raise Skip("wslc unavailable: %s" % av["reasons"])
    be = PE.ProviderExecutionBackend(backend="wslc", scratch=Path(tempfile.mkdtemp()))
    h = be.start(["python", "java"])
    assert h["toolchains"]["java"]["runnable"] and h["toolchains"]["python"]["runnable"], h["toolchains"]
    out = be.run_public(Artifact.make("J1-01", "java", JREF01), _j01())
    assert all(o["status"] == "pass" for o in out)
    bad = be.run_public(Artifact.make("J1-01", "java", JCASES["always_true"][0]), _j01())
    assert any(o["status"] == "fail" for o in bad[1:])
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
