"""[EL-BANK40] CPU tests (no model, no network, no Kaggle) of the 40-task run: thinking-policy refusal, packer public-only asserts, shard split, resume, time guard, timeline shape, replay, results
compiler, push order, and the fail-closed paths of the kernel itself (run in a subprocess against a fake /kaggle tree: a missing jail and a missing policy both abort with a receipt and run nothing).
  PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe evidence_ladder/orchestrator/tests/phaseF3b_tests.py   -> JSON {"tests","passed","failed"}"""
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
from evidence_ladder import bank40 as B   # noqa: E402
import pack_el_bank40_v1 as PK            # noqa: E402
import el_bank40_driver as DR             # noqa: E402

TESTS = []
BUDGET = {"policy": "budget", "think_budget": 512, "answer_tokens": 1500}


def test(f):
    TESTS.append(f); return f


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % exc.__name__)


# ------------------------------------------------------------------ thinking policy
@test
def t_policy_is_required_explicit_and_validated():
    for bad in ({}, {"policy": "full_thinking"}, {"policy": "budget"}, {"policy": "budget", "think_budget": None, "answer_tokens": 100}, {"policy": "budget", "think_budget": 10, "answer_tokens": 100},
                {"policy": "prompt_brief", "brief_instruction": "a\nb", "ceiling": 4096}, {"policy": "prompt_brief", "brief_instruction": "x", "ceiling": 99}, {"policy": "model_variant", "ceiling": 4096}, None):
        raises(B.RequestRefused, B.check_policy, bad)
    assert B.check_policy(BUDGET) == BUDGET
    assert B.check_policy({"policy": "prompt_brief", "brief_instruction": "Think briefly.", "ceiling": 8192})["ceiling"] == 8192
    assert B.check_policy({"policy": "model_variant", "profile_id": "p", "model_glob": "g", "ceiling": 4096})["profile_id"] == "p"


@test
def t_no_thinking_number_is_hard_coded_in_the_kernel_or_driver():
    src = (REPO / "kaggle/el_bank40_run_v1.py").read_text("utf-8")
    import re as _re
    assert not _re.search(r"think_budget\s*=\s*\d", src) and "REFUSED_NO_THINKING_POLICY" in src and "check_policy" in src
    assert "= 512" not in src and "=512" not in src
    assert not hasattr(B, "THINK_BUDGET")


@test
def t_brief_prompt_wrapper_appends_one_line_and_counts_ceiling_hits_as_failures():
    from evidence_ladder.common.interfaces import GenerationRequest
    from evidence_ladder.models import qwen as Q
    seen = []

    class Inner:
        def generate(self, req):
            seen.append(req)
            return [Q.make_generation("x", tokens_prompt=5, tokens_generated=req.max_new_tokens, seed=0, prompt_template_id="t", prompt_sha256="h")]
    tl = B.Timeline(Path(tempfile.mkdtemp()) / "t.jsonl")
    g = B.BriefPromptGen(Inner(), "Reason briefly.", 2048, tl, "T")
    g.generate(GenerationRequest("T", "prompt", 1, (0,), 99999, "sample", True, "tpl"))
    assert seen[0].prompt.endswith("\n\nReason briefly.") and seen[0].max_new_tokens == 2048 and g.ceiling_hits == 1
    assert any(e["event"] == "ceiling_hit" for e in B.read_events(tl.path))


# ------------------------------------------------------------------ packer
@test
def t_packer_bundle_is_public_only_and_asserts_catch_violations():
    files = PK.build_files(BUDGET, REPO, "testcommit")
    rep = B.assert_public_bundle(files, PK.held_markers(REPO))
    assert rep["violations"] == 0 and rep["markers_checked"] >= 8
    names = set(files)
    assert "evidence_ladder/execution/protected.py" not in names and not any("protected_cases" in n or "oracles" in n for n in names)
    assert not any("/tests/" in "/" + n for n in names) and "run_config.json" in names and "bank40_tasks_public.jsonl" in names
    tasks = [json.loads(l) for l in files["bank40_tasks_public.jsonl"].decode().splitlines()]
    assert len(tasks) == 40 and {t["language"] for t in tasks} == {"python", "java"} and all(set(t) == {"task_id", "language", "difficulty", "prompt", "public_tests"} for t in tasks)
    # negative controls
    marker = PK.held_markers(REPO)[0]
    raises(AssertionError, B.assert_public_bundle, dict(files, **{"evidence_ladder/x.py": b"# " + marker}), [marker])
    raises(AssertionError, B.assert_public_bundle, dict(files, **{"grader/protected_cases.json": b"{}"}))
    raises(AssertionError, B.assert_public_bundle, dict(files, **{"evidence_ladder/execution/protected.py": b"x = 1"}))
    raises(AssertionError, B.assert_public_bundle, dict(files, **{"cfg.json": ('{"k": "' + "nv" + "api-" + "ABCDEFGHIJKLMNOP" + '"}').encode()}))      # assembled at runtime: key-shaped for the packer guard, no key-shaped literal in the source
    bad_task = json.dumps({"task_id": "P1-01", "language": "python", "difficulty": "easy", "prompt": "p", "public_tests": [{"input": {}, "expected": 1}], "hidden_tests": []}) + "\n"
    raises(AssertionError, B.assert_public_bundle, dict(files, **{"bank40_tasks_public.jsonl": bad_task.encode()}))


@test
def t_packer_refuses_a_missing_or_bad_policy_and_pins_the_task_sources():
    import argparse
    ns = argparse.Namespace(thinking_policy="budget", think_budget=None, answer_tokens=1500, brief=None, ceiling=None, profile_id=None, model_glob=None)
    raises(B.RequestRefused, PK.policy_from_args, ns)
    ns.think_budget = 512
    assert PK.policy_from_args(ns) == BUDGET
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "configs").mkdir()
        cfg = json.loads((REPO / "configs/el_bank40_v1.json").read_text("utf-8")); cfg["sources"]["python"]["sha256"] = "0" * 64
        (tmp / "configs/el_bank40_v1.json").write_text(json.dumps(cfg), "utf-8")
        for lang in ("python", "java"):
            sp = tmp / cfg["sources"][lang]["file"]; sp.parent.mkdir(parents=True, exist_ok=True); sp.write_bytes((REPO / cfg["sources"][lang]["file"]).read_bytes())
        raises(AssertionError, PK.load_bank, tmp)                     # a task source that differs from its pinned sha256 is refused
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_packer_cli_writes_bundle_two_kernels_and_imports_clean():
    out = Path(tempfile.mkdtemp())
    try:
        assert PK.main(["--thinking-policy", "budget", "--think-budget", "512", "--answer-tokens", "1500", "--out-root", str(out)]) == 0
        pack = next(out.glob("el-bank40-bundle-*"))
        pj = json.loads((pack / "PACK.json").read_text("utf-8"))
        assert pj["import_smoke"] is True and pj["thinking_policy"] == BUDGET
        k1 = json.loads((pack / "kernel_pass1/kernel-metadata.json").read_text("utf-8")); k2 = json.loads((pack / "kernel_pass2/kernel-metadata.json").read_text("utf-8"))
        assert k1["enable_gpu"] is True and k1["enable_internet"] is False and k2["enable_gpu"] is False and k2["enable_internet"] is False
        assert 'RUN_MODE = "pass1"' in (pack / "kernel_pass1/el_bank40_run_pass1.py").read_text("utf-8") and 'RUN_MODE = "pass2"' in (pack / "kernel_pass2/el_bank40_run_pass2.py").read_text("utf-8")
        txt = DR.plan(pack)
        assert "PUSH ORDER" in txt and "kernels push" in txt and "JAIL_UNAVAILABLE" in txt and (pack / "PUSH_ORDER.md").is_file()
        raises(FileExistsError, DR.plan, pack)                        # create-only
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ------------------------------------------------------------------ shards, resume, time guard
@test
def t_shard_split_is_deterministic_disjoint_and_balanced():
    tasks = PK.load_bank(REPO)
    a, b = B.shard_tasks(tasks, 2), B.shard_tasks(list(reversed(tasks)), 2)
    assert a == b                                                     # input order does not matter
    assert set(a[0]).isdisjoint(a[1]) and set(a[0]) | set(a[1]) == {t["task_id"] for t in tasks} and len(a[0]) == len(a[1]) == 20
    lang = {t["task_id"]: t["language"] for t in tasks}
    assert sorted(sum(1 for x in s if lang[x] == "python") for s in a) == [10, 10]
    for n in (1, 3, 4):
        s = B.shard_tasks(tasks, n); flat = [x for sh in s for x in sh]
        assert len(flat) == len(set(flat)) == 40
    raises(ValueError, B.shard_tasks, tasks, 0)


@test
def t_resume_skips_finished_tasks_and_retries_unfinished_ones():
    tmp = Path(tempfile.mkdtemp())
    try:
        for tid, done in (("P1-01", True), ("P1-02", False), ("J1-01", True)):
            d = tmp / "prev" / "tasks" / tid; d.mkdir(parents=True); (d / "result.json").write_text("{}", "utf-8")
            if done: B.write_atomic(d / "DONE", "x")
        fin = B.finished_task_ids(tmp / "prev")
        assert sorted(fin) == ["J1-01", "P1-01"]
        todo, skipped = B.plan_work(["P1-01", "P1-02", "P1-03", "J1-01"], fin)
        assert todo == ["P1-02", "P1-03"] and skipped == ["P1-01", "J1-01"]                      # an unfinished task (no DONE) is retried, a finished one never repeated
        assert not list((tmp / "prev").glob("**/.tmp-*"))                                         # atomic writes leave no temp files
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_time_guard_stops_new_tasks_near_the_session_limit():
    now = [1000.0]
    g = B.TimeGuard(1000.0, 11.0, clock=lambda: now[0])
    assert g.should_start()
    now[0] = 1000.0 + 10.99 * 3600; assert g.should_start()
    now[0] = 1000.0 + 11.0 * 3600; assert not g.should_start()
    raises(ValueError, B.TimeGuard, 0.0, 12.5)
    raises(ValueError, B.TimeGuard, 0.0, 0.0)


# ------------------------------------------------------------------ timeline
@test
def t_timeline_file_shape_summary_percentiles_and_wrappers():
    from evidence_ladder.common.interfaces import GenerationRequest
    from evidence_ladder.models import qwen as Q
    tmp = Path(tempfile.mkdtemp())
    try:
        clock = iter(range(1000, 100000, 1)).__next__
        tl = B.Timeline(tmp / "w0.jsonl", "w0", clock=lambda: float(clock()), mono=lambda: float(clock()))
        class G:
            def generate(self, r): return [Q.make_generation("t", tokens_prompt=3, tokens_generated=7, seed=0, prompt_template_id="p", prompt_sha256="h")]
        class E:
            def run_public(self, a, t): return []
            def run_experiment(self, a, e): return {}
        g, e = B.TimedGen(G(), tl, "T1", 0), B.TimedExec(E(), tl, "T1")
        g.generate(GenerationRequest("T1", "p", 1, (0,), 10)); e.run_public(object(), None); e.run_experiment(object(), {})
        with tl.span("task", task="T1"): pass
        lines = [json.loads(l) for l in (tmp / "w0.jsonl").read_text("utf-8").splitlines()]
        assert all({"ts_utc", "ts", "mono", "worker", "event"} <= set(x) for x in lines)
        assert all(x["ts_utc"].endswith("Z") and len(x["ts_utc"]) == 24 for x in lines)
        names = [x["event"] for x in lines]
        assert names.count("generate_start") == names.count("generate_end") == 1 and "sandbox_public_end" in names and "sandbox_experiment_end" in names and "generate_tokens" in names
        s = B.summarize(B.read_events(tmp / "w0.jsonl"))
        assert s["stages"]["generate"]["n"] == 1 and s["stages"]["sandbox_public"]["n"] == 1 and s["model_seconds"] > 0 and s["sandbox_seconds"] > 0 and 0 < s["model_vs_sandbox"]["model_share"] < 1
        assert B.pct([1, 2, 3, 4, 100], 50) == 3 and B.pct([1, 2, 3, 4, 100], 95) > 4 and B.pct([], 50) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ record / replay
@test
def t_replay_serves_recorded_generations_in_order_and_refuses_any_difference():
    from evidence_ladder.common.interfaces import GenerationRequest
    from evidence_ladder.models import qwen as Q

    class Inner:
        n = 0
        def generate(self, r):
            Inner.n += 1
            return [Q.make_generation("text%d" % Inner.n, tokens_prompt=1, tokens_generated=2, seed=s, prompt_template_id=r.prompt_template_id, prompt_sha256=B.sha(r.prompt)) for s in r.seeds[:r.n]]
    rec = B.RecordingGen(Inner(), "T")
    r1, r2 = GenerationRequest("T", "prompt one", 2, (1, 2), 100, "sample", True, "tpl"), GenerationRequest("T", "prompt two", 2, (1, 2), 100, "repair", True, "tpl")
    o1, o2 = rec.generate(r1), rec.generate(r2)
    rp = B.SequenceReplayGen(json.loads(json.dumps(rec.calls)))
    assert [g.text for g in rp.generate(r1)] == [g.text for g in o1] and [g.text for g in rp.generate(r2)] == [g.text for g in o2] and rp.exhausted()
    raises(B.ReplayMismatch, rp.generate, r1)                                                                   # nothing left: never invents a generation
    rp = B.SequenceReplayGen(rec.calls)
    raises(B.ReplayMismatch, rp.generate, GenerationRequest("T", "prompt CHANGED", 2, (1, 2), 100, "sample", True, "tpl"))
    rp = B.SequenceReplayGen(rec.calls)
    raises(B.ReplayMismatch, rp.generate, GenerationRequest("T", "prompt one", 2, (1, 3), 100, "sample", True, "tpl"))      # different seeds


# ------------------------------------------------------------------ results compiler
def _fake_run(tmp: Path, name: str, tasks: dict, policy=BUDGET, extra_summary=None):
    d = tmp / name; (d / "tasks").mkdir(parents=True); (d / "timeline").mkdir()
    ev = []
    for tid, (lang, label, done) in tasks.items():
        td = d / "tasks" / tid; td.mkdir()
        res = {"task": tid, "pass": "pass1", "thinking_policy": policy, "scorer": "none", "language": lang, "difficulty": "easy", "status": "OK" if done else "ERROR", "stop_cause": "decided", "stop_details": "", "answer_label": label,
               "rounds": 1, "candidates": 4, "public": {"examples": 3, "best": 3, "candidates_passing_all": 2, "candidates_with_examples": 4}, "l2": {"verdict": "INSUFFICIENT"}, "clusters": {"public_clusters": 2, "extended_clusters": 2},
               "discrimination": {"triggered": True, "critic": "pending"}, "tokens_generated": 1000, "tokens_prompt": 500, "generate_calls": 3, "ceiling_hits": 0, "invariants": {"failed": None},
               "start_utc": "2026-10-18T10:00:00.000Z", "end_utc": "2026-10-18T10:05:00.000Z", "seconds": 300.0, "worker": 0}
        (td / "result.json").write_text(json.dumps(res), "utf-8")
        if done: (td / "DONE").write_text("x", "utf-8")
        for i, (nm, t0, sec) in enumerate((("generate", 100.0, 20.0), ("sandbox_public", 125.0, 2.0))):
            ev.append({"ts_utc": "2026-10-18T10:01:%02d.000Z" % (i * 10), "ts": t0, "mono": t0, "worker": "w0", "event": nm + "_start", "task": tid})
            ev.append({"ts_utc": "2026-10-18T10:02:%02d.000Z" % (i * 10), "ts": t0 + sec, "mono": t0 + sec, "worker": "w0", "event": nm + "_end", "task": tid, "seconds": sec, "ok": True})
    (d / "TIMELINE.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev), "utf-8")
    (d / "timeline" / "gpu_util.jsonl").write_text(json.dumps({"ts_utc": "x", "ts": 100.0, "gpu": 0, "util_pct": 90, "mem_mib": 1000}) + "\n" + json.dumps({"ts_utc": "x", "ts": 130.0, "gpu": 1, "util_pct": 2, "mem_mib": 10}) + "\n", "utf-8")
    summ = {"verdict": "OK", "thinking_policy": policy, "total_seconds": 600.0, "jail": {"status": "PASS", "java_home": "/usr/lib/jvm/x", "bwrap_swapped": True}, "gpus": 2, "critic_packets": {"exported": 3}, **(extra_summary or {})}
    (d / "RUN_SUMMARY.json").write_text(json.dumps(summ), "utf-8")
    return d


@test
def t_results_compiler_rows_percentiles_gpu_and_labels():
    tmp = Path(tempfile.mkdtemp())
    try:
        p1 = _fake_run(tmp, "p1", {"P1-01": ("python", "VERIFIED", True), "J1-01": ("java", "SUPPORTED", True), "P1-02": ("python", "BLOCKED", False)})
        c = DR.compile_results(DR.read_run(p1))
        ids = [r["task"] for r in c["rows"]]
        assert ids == ["J1-01", "P1-01"] and c["unfinished_or_errored"] == ["P1-02"] and c["tasks_finished"] == 2                  # a task without DONE gets NO label row
        r = c["rows"][1]
        assert r["final_label"] == "VERIFIED" and r["public_examples"]["examples"] == 3 and r["clusters"] == {"public": 2, "extended": 2} and r["critic_used"] and r["tokens"]["generated"] == 1000
        assert [s["stage"] for s in r["steps"]] == ["generate", "sandbox_public"] and r["steps"][0]["start_utc"].endswith("Z") and r["steps"][0]["seconds"] == 20.0
        assert c["stage_summary"]["stages"]["generate"]["p50_s"] == 20.0 and c["model_vs_sandbox"]["model_seconds"] > c["model_vs_sandbox"]["sandbox_seconds"]
        assert c["gpu"]["per_gpu"]["0"]["mean_util_pct"] == 90.0 and c["gpu"]["per_gpu"]["1"]["idle_samples_pct"] == 100.0
        md = DR.render_md(c)
        assert "development results (D1 waiver), public examples only; hidden grading NOT done" in md and '"think_budget": 512' in md and "scorer: none" in md and "p95 s" in md
        # pass 2 overrides the pass-1 row of the same task
        p2 = _fake_run(tmp, "p2", {"P1-01": ("python", "SUPPORTED", True)}, extra_summary={"critic_responses": 3})
        c2 = DR.compile_results(DR.read_run(p1), DR.read_run(p2))
        assert {r["task"]: r["final_label"] for r in c2["rows"]}["P1-01"] == "SUPPORTED" and c2["kernel_wall_seconds"]["total"] == 1200.0 and c2["critic"]["pass2_responses"] == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_driver_refuses_results_without_a_policy_or_with_a_mixed_policy():
    tmp = Path(tempfile.mkdtemp())
    try:
        d = _fake_run(tmp, "nopol", {"P1-01": ("python", "VERIFIED", True)}, policy=None)
        raises(B.RequestRefused, DR.read_run, d)
        d2 = _fake_run(tmp, "mixed", {"P1-01": ("python", "VERIFIED", True)})
        r = json.loads((d2 / "tasks/P1-01/result.json").read_text("utf-8")); r["thinking_policy"] = {"policy": "budget", "think_budget": 1024, "answer_tokens": 1500}
        (d2 / "tasks/P1-01/result.json").write_text(json.dumps(r), "utf-8")
        raises(DR.IngestError, DR.read_run, d2)
        raises(DR.IngestError, DR.read_run, tmp / "empty")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_driver_builds_pass1_and_critic_datasets_create_only():
    tmp = Path(tempfile.mkdtemp())
    try:
        p1 = _fake_run(tmp, "p1", {"P1-01": ("python", "VERIFIED", True), "P1-02": ("python", "BLOCKED", False)})
        (p1 / "critic_packets").mkdir(); (p1 / "critic_packets" / "packets.jsonl").write_text("{}\n", "utf-8")
        rec = DR.pass1_datasets(p1, tmp / "out")
        assert rec["done"] == 1 and rec["unfinished"] == ["P1-02"] and (tmp / "out/pass1_out/tasks/P1-01/DONE").is_file() and not (tmp / "out/pass1_out/tasks/P1-02").exists()
        raises(DR.IngestError, DR.pass1_datasets, p1, tmp / "out")
        co = tmp / "critic" / "critic_out"; co.mkdir(parents=True)
        (co / "responses.jsonl").write_text(json.dumps({"packet_id": "a" * 64, "response": {"experiment": {}}}) + "\n", "utf-8"); (co / "MANIFEST.json").write_text(json.dumps({"states": {"ok": 1}}), "utf-8")
        assert DR.critic_dataset(tmp / "critic", tmp / "o2")["responses"] == 1
        (co / "responses.jsonl").write_text(json.dumps({"packet_id": "", "response": {}}) + "\n", "utf-8")
        raises(DR.IngestError, DR.critic_dataset, tmp / "critic", tmp / "o3")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_driver_merges_resumed_kernel_versions_and_refuses_mixed_policies():
    tmp = Path(tempfile.mkdtemp())
    try:
        v1 = _fake_run(tmp, "v1", {"P1-01": ("python", "VERIFIED", True), "P1-02": ("python", "BLOCKED", False)})
        v2 = _fake_run(tmp, "v2", {"P1-01": ("python", "VERIFIED", True), "P1-02": ("python", "SUPPORTED", True), "J1-01": ("java", "PLAUSIBLE", True)})
        m = DR.merge_runs([DR.read_run(v1), DR.read_run(v2)])
        c = DR.compile_results(m)
        assert c["tasks_finished"] == 3 and {r["task"]: r["final_label"] for r in c["rows"]} == {"J1-01": "PLAUSIBLE", "P1-01": "VERIFIED", "P1-02": "SUPPORTED"} and c["unfinished_or_errored"] == []
        assert c["kernel_wall_seconds"]["pass1"] == 1200.0
        v3 = _fake_run(tmp, "v3", {"P1-01": ("python", "VERIFIED", True)}, policy={"policy": "budget", "think_budget": 256, "answer_tokens": 1500})
        raises(DR.IngestError, DR.merge_runs, [DR.read_run(v1), DR.read_run(v3)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ the worker end to end on CPU (fake model + fake executor; real controller, real critic broker, real files)
def _load_kernel(mode: str, out: Path, bundle: Path):
    import importlib.util
    src = (REPO / "kaggle/el_bank40_run_v1.py").read_text("utf-8").replace('RUN_MODE = "pass1"', 'RUN_MODE = "%s"' % mode, 1).replace('Path("/kaggle/working")', 'Path(%r)' % str(out))
    kp = out.parent / ("kernel_%s.py" % mode)
    kp.write_text(src, "utf-8")
    spec = importlib.util.spec_from_file_location("el_bank40_kernel_" + mode, kp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.find_bundle = lambda: (str(bundle), {})
    return m


def _install_fakes():
    """-> restore(). A fake 4B (seed -> constant solution), a fake CUDA, a fake executor (candidates 0 and 1 pass the public examples, behaviour differs -> 2 clusters -> the critic is triggered)."""
    import re as _r
    import types
    from evidence_ladder.models import qwen as Q
    from evidence_ladder.execution import backend as EB
    saved = (Q.HFQwenBackend, EB.SandboxExecutionBackend, sys.modules.get("torch"))

    class FakeQwen:
        def __init__(self, **kw):
            self.kw, self.language, self.snapshot_verified = kw, "python", False

        def _load(self):
            return None

        def model_info(self):
            return {"backend": "fake"}

        def generate(self, req):
            out = []
            for s in list(req.seeds)[:req.n]:
                k = s % 3
                if self.language == "python":
                    txt = "<think>\nfake\n</think>\n```python\ndef solve(data):\n    return %d\n```" % k
                else:
                    txt = "<think>\nfake\n</think>\n```java\nclass Solution { }\n```"
                out.append(Q.make_generation(txt, tokens_prompt=10, tokens_generated=20 + s, seed=s, prompt_template_id=req.prompt_template_id, prompt_sha256=B.sha(req.prompt),
                                             language=self.language, wall_seconds=0.01))
            return out

    class FakeExec:
        def __init__(self, store=None, cfg=None):
            pass

        def _k(self, art):
            m = _r.search(r"return (\d)", art.source)
            return int(m.group(1)) if m else 9

        def run_public(self, art, task):
            ok = self._k(art) in (0, 1)
            return [{"check_id": "parse_v1", "status": "pass"}] + [{"check_id": "public_examples_v1", "example_index": i, "status": "pass" if ok else "fail",
                                                                     "failure_class": "" if ok else "wrong_answer", "observed": self._k(art)} for i in range(len(task.public_tests))]

        def run_experiment(self, art, exp):
            return {"check_id": "critic_experiment_v1", "status": "pass", "observed": self._k(art)}

        def backend_info(self):
            return {"backend": "fake"}

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_name=lambda i: "FAKE-GPU")
    Q.HFQwenBackend, EB.SandboxExecutionBackend, sys.modules["torch"] = FakeQwen, FakeExec, torch

    def restore():
        Q.HFQwenBackend, EB.SandboxExecutionBackend = saved[0], saved[1]
        if saved[2] is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved[2]
    return restore


@test
def t_worker_pass1_then_pass2_replay_end_to_end():
    import time as _t
    tmp = Path(tempfile.mkdtemp())
    restore = _install_fakes()
    try:
        files = PK.build_files(BUDGET, REPO, "t")
        bundle = tmp / "bundle"
        bundle.mkdir()
        (bundle / "bank40_tasks_public.jsonl").write_bytes(files["bank40_tasks_public.jsonl"])
        out1 = tmp / "w1" / "out"
        (out1 / "timeline").mkdir(parents=True)
        k1 = _load_kernel("pass1", out1, bundle)
        rf = tmp / "req1.json"
        rf.write_text(json.dumps({"thinking_policy": BUDGET, "todo": ["P1-01", "J1-01"], "model_dir": "fake", "model_sha_verified": True, "t0": _t.time(), "guard_h": 11.0}), "utf-8")
        assert k1.worker(0, str(rf)) == 0
        for tid in ("P1-01", "J1-01"):
            d = out1 / "tasks" / tid
            r = json.loads((d / "result.json").read_text("utf-8"))
            assert r["status"] == "OK", r.get("error")
            assert (d / "DONE").is_file() and (d / "journal.jsonl").is_file() and (d / "gen_calls.json").is_file() and (d / "critic_requests.json").is_file()
            assert r["thinking_policy"] == BUDGET and r["scorer"] == "none" and "hidden grading NOT done" in r["label"] and r["start_utc"].endswith("Z") and r["end_utc"].endswith("Z")
            assert r["generate_calls"] >= 1 and r["tokens_generated"] > 0 and r["language"] in ("python", "java")
            ct = json.loads((d / "controller_timeline.json").read_text("utf-8"))
            assert ct["rows"] and ct["summary"]["stages"], "the controller's own timeline was not written"
        r1 = json.loads((out1 / "tasks/P1-01/result.json").read_text("utf-8"))
        assert r1["discrimination"]["triggered"] and r1["pending_critic_packets"], r1["stop_cause"]          # 2 clusters -> the blinded critic packet waits for the critic kernel
        names = {e["event"] for e in B.read_events(out1 / "timeline" / "worker0.jsonl")}
        assert {"gpu", "model_load_start", "model_load_end", "task_start", "task_end", "generate_start", "generate_end", "sandbox_public_end", "sandbox_experiment_end"} <= names
        assert sorted(B.finished_task_ids(out1)) == ["J1-01", "P1-01"]
        # pass 2: replays the recorded generations (constructing a model would raise), applies a critic response
        from evidence_ladder.critic_batch import validator as V
        reqs = json.loads((out1 / "tasks/P1-01/critic_requests.json").read_text("utf-8"))
        pid = r1["pending_critic_packets"][0]
        V.validate_packet({"packet_id": pid, "request": reqs[pid]})          # blinded: the validator accepts what pass 1 exported
        resp = {"experiment": {"kind": "function_call", "payload": json.dumps({"x": 1}), "expected_by_candidate": {}}, "implications": "probe the two behaviours"}
        out2 = tmp / "w2" / "out"
        (out2 / "timeline").mkdir(parents=True)
        k2 = _load_kernel("pass2", out2, bundle)
        import evidence_ladder.models.qwen as Q

        def no_model(**kw):
            raise AssertionError("pass 2 must not construct a model")
        Q.HFQwenBackend = no_model
        rf2 = tmp / "req2.json"
        rf2.write_text(json.dumps({"thinking_policy": BUDGET, "todo": ["P1-01"], "pass1_dir": str(out1), "critic_responses": {pid: resp}, "t0": _t.time(), "guard_h": 11.0}), "utf-8")
        assert k2.worker(0, str(rf2)) == 0
        r2 = json.loads((out2 / "tasks/P1-01/result.json").read_text("utf-8"))
        assert r2["status"] == "OK", r2.get("error")
        assert r2["pass"] == "pass2" and r2["replay_exhausted"] is True and r2["gpu"] is None
        assert r2["stop_details"] != "critic_pending" and r2["pending_critic_packets"] == [] and (out2 / "tasks/P1-01/DONE").is_file()
        # the compiler reads both
        for o, s in ((out1, {"verdict": "OK", "thinking_policy": BUDGET, "total_seconds": 10.0}), (out2, {"verdict": "OK", "thinking_policy": BUDGET, "total_seconds": 5.0, "critic_responses": 1})):
            (o / "RUN_SUMMARY.json").write_text(json.dumps(s), "utf-8")
            (o / "TIMELINE.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in B.read_events(*sorted((o / "timeline").glob("worker*.jsonl")))), "utf-8")
        comp = DR.compile_results(DR.read_run(out1), DR.read_run(out2))
        rows = {r["task"]: r for r in comp["rows"]}
        assert set(rows) == {"P1-01", "J1-01"} and rows["P1-01"]["pass"] == "pass2" and rows["J1-01"]["pass"] == "pass1" and rows["P1-01"]["steps"]
        assert "BANK40 RESULTS" in DR.render_md(comp)
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_worker_time_guard_records_pending_and_runs_no_new_task():
    import time as _t
    tmp = Path(tempfile.mkdtemp())
    restore = _install_fakes()
    try:
        bundle = tmp / "bundle"
        bundle.mkdir()
        (bundle / "bank40_tasks_public.jsonl").write_bytes(PK.build_files(BUDGET, REPO, "t")["bank40_tasks_public.jsonl"])
        out = tmp / "o"
        (out / "timeline").mkdir(parents=True)
        k = _load_kernel("pass1", out, bundle)
        rf = tmp / "r.json"
        rf.write_text(json.dumps({"thinking_policy": BUDGET, "todo": ["P1-01", "P1-02"], "model_dir": "fake", "t0": _t.time() - 11.5 * 3600, "guard_h": 11.0}), "utf-8")
        assert k.worker(0, str(rf)) == 0
        assert json.loads((out / "timeline/worker0.pending.json").read_text("utf-8")) == ["P1-01", "P1-02"]
        assert not (out / "tasks").exists() or not list((out / "tasks").glob("*"))
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ the kernel itself, fail closed (subprocess against a fake /kaggle tree)
def _run_kernel(mode: str, files: dict):
    tmp = Path(tempfile.mkdtemp())
    inp, out = tmp / "input", tmp / "working"
    bundle = inp / "bundle"; out.mkdir(); bundle.mkdir(parents=True)
    for rel, data in files.items():
        p = bundle / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    src = (REPO / "kaggle/el_bank40_run_v1.py").read_text("utf-8").replace('RUN_MODE = "pass1"', 'RUN_MODE = "%s"' % mode, 1)
    src = src.replace('Path("/kaggle/working")', 'Path(%r)' % str(out)).replace('"/kaggle/input/**/', '%r + "/**/' % str(inp)).replace('"/kaggle/input/*"', repr(str(inp) + "/*"))
    kp = tmp / "kernel.py"; kp.write_text(src, "utf-8")
    r = subprocess.run([sys.executable, str(kp)], capture_output=True, text=True, cwd=str(tmp), timeout=300, env={**__import__("os").environ, "PYTHONPATH": ""})
    return tmp, out, r


def _with_manifest(files: dict) -> dict:
    files = {k: v for k, v in files.items() if k != "BUNDLE_MANIFEST.json"}
    man = json.loads(PK.build_files(BUDGET, REPO, "t")["BUNDLE_MANIFEST.json"])
    man["files"] = {k: B.sha(v) for k, v in files.items()}
    return dict(files, **{"BUNDLE_MANIFEST.json": json.dumps(man).encode()})


@test
def t_kernel_aborts_with_a_receipt_when_the_jail_cannot_start_and_runs_nothing():
    files = _with_manifest(PK.build_files(BUDGET, REPO, "t"))
    tmp, out, r = _run_kernel("pass1", files)
    try:
        s = json.loads((out / "RUN_SUMMARY.json").read_text("utf-8"))
        assert s["verdict"] == "JAIL_UNAVAILABLE" and s["jail"]["status"] == "FAILED" and "unsandboxed" in s["note"], (s.get("verdict"), r.stderr[-500:])
        assert s["thinking_policy"] == BUDGET and s["scorer"] == "none" and "hidden grading NOT done" in s["label"]
        assert not list((out / "tasks").glob("*")) if (out / "tasks").exists() else True              # nothing ran, nothing fell back
        assert not (out / "critic_packets").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_kernel_refuses_to_start_without_a_thinking_policy():
    f = PK.build_files(BUDGET, REPO, "t")
    f["run_config.json"] = json.dumps({"seeds": [1]}).encode()
    tmp, out, r = _run_kernel("pass1", _with_manifest(f))
    try:
        s = json.loads((out / "RUN_SUMMARY.json").read_text("utf-8"))
        assert s["verdict"] == "REFUSED_NO_THINKING_POLICY" and "jail" not in s and "thinking_policy" not in s, s
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    f["run_config.json"] = json.dumps({"thinking_policy": {"policy": "budget", "think_budget": None, "answer_tokens": 1500}}).encode()
    tmp, out, r = _run_kernel("pass1", _with_manifest(f))
    try:
        assert json.loads((out / "RUN_SUMMARY.json").read_text("utf-8"))["verdict"] == "REFUSED_NO_THINKING_POLICY"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_kernel_refuses_a_bundle_with_a_tampered_file_or_a_hidden_case_file():
    files = PK.build_files(BUDGET, REPO, "t")
    tampered = dict(files); tampered["evidence_ladder/bank40.py"] = files["evidence_ladder/bank40.py"] + b"\n# tampered\n"
    tmp, out, r = _run_kernel("pass1", tampered)
    try:
        assert json.loads((out / "RUN_SUMMARY.json").read_text("utf-8"))["verdict"] == "FAILED_BUNDLE_HASH"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    leaked = _with_manifest(dict(files, **{"results/raw/x/grader/protected_cases.json": b"{}"}))
    tmp, out, r = _run_kernel("pass1", leaked)
    try:
        assert json.loads((out / "RUN_SUMMARY.json").read_text("utf-8"))["verdict"] == "REFUSED_BUNDLE_NOT_PUBLIC"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_pass2_kernel_also_fails_closed_without_the_jail():
    tmp, out, r = _run_kernel("pass2", _with_manifest(PK.build_files(BUDGET, REPO, "t")))
    try:
        assert json.loads((out / "RUN_SUMMARY.json").read_text("utf-8"))["verdict"] == "JAIL_UNAVAILABLE"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    failed = {}
    for t in TESTS:
        try:
            t()
        except Exception:
            failed[t.__name__] = traceback.format_exc()[-1800:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)
