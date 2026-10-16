"""Integration replay: REAL controller + REAL StagedCriticBroker/GLM adapter + REAL budgets + REAL PilotRunner, with scripted generations (FixtureGenerationBackend)
and REAL recorded sandbox/grader outcomes (fixtures/, recorded in WSL by record_execution_fixtures.py). No model, no network, no sandbox here.
Run from the repo root:  python3 evidence_ladder/testing/replay_integration.py
Prints {"tests","passed","failed":{...},"open_defects":{...},"fixed_defects":[...]} and exits 1 if anything unexpected failed.
Tests marked @expected_defect keep the DESIRED assertion; while the defect is open they are listed under open_defects (not failed); when fixed they pass."""
import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evidence_ladder.common.contract_import import C  # noqa: E402
from evidence_ladder.common.interfaces import (Artifact, ControllerContext, GraderIsolationError, NotAuthorized, UnsupportedDecisionType,  # noqa: E402
                                               sha256_text)
from evidence_ladder.controller import intake as intake_mod  # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController  # noqa: E402
from evidence_ladder.controller.decision import authorize, decide_stage1  # noqa: E402
from evidence_ladder.controller.evidence import Factory  # noqa: E402
from evidence_ladder.models import glm, laya, qwen  # noqa: E402
from evidence_ladder.models.broker import CriticStopped, StagedCriticBroker  # noqa: E402
from evidence_ladder.orchestrator import authorization, index, pairing, pilot, report  # noqa: E402
from evidence_ladder.orchestrator.arms import ARMS  # noqa: E402
from evidence_ladder.orchestrator.base import THRESHOLDS, ConfigError, PoolingError  # noqa: E402
from evidence_ladder.orchestrator.budget import MatchedTokenBudget  # noqa: E402
from evidence_ladder.orchestrator.dryrun import FakeClock  # noqa: E402
from evidence_ladder.testing.cases import CASES  # noqa: E402
from evidence_ladder.testing.recorded import (FIXDIR, RecordedExecutionBackend, RecordedGrader, UnrecordedError, load_grades,  # noqa: E402
                                              load_outcomes)

TESTS, DEFECTS = [], {}
OUT = load_outcomes()
MASTER, K1 = 20261016, 4


def test(f):
    TESTS.append(f)
    return f


def expected_defect(where, fix):
    def deco(f):
        DEFECTS[f.__name__] = {"where": where, "fix": fix}
        TESTS.append(f)
        return f
    return deco


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % exc.__name__)


_TASKS = None


def tasks():
    global _TASKS
    if _TASKS is None:
        _TASKS = index.build_tasks(index.load_index())
    return _TASKS


def task_of(tid):
    return next(t for t in tasks()["python" if tid.startswith("P") else "java"] if t.task_id == tid)


def py1():
    return task_of("P1-01")


def jv1():
    return task_of("J1-02")


def art(name):
    tid, lang, src, _ = CASES[name]
    return Artifact.make(tid, lang, src, origin="recorded:" + name)


def rec_ce(name, example_index):
    return next(o for o in OUT["cases"][name]["public"] if o.get("example_index") == example_index)["counterexample"]


def labels(res):
    return {r["requirement_id"]: r["label"] for r in res.decision["requirement_rows"]}


def labels_of(d):
    return {x["requirement_id"]: x["label"] for x in d["requirement_rows"]}


# ------------------------------------------------------------------ scripted generation + controller harness
def rec(task, seed, purpose, name, tg=40, tp=120):
    tid, lang, src, _ = CASES[name]
    assert tid == task.task_id
    text = "plan\n</think>\n\n```%s\n%s\n```\n" % (lang, src)
    g = qwen.make_generation(text, tokens_prompt=tp, tokens_generated=tg, seed=seed, prompt_template_id="qwen_sample_v1", prompt_sha256="0" * 64, language=lang)
    assert g.extract_status == "ok" and g.code == src, name
    return {"task_id": tid, "seed": seed, "purpose": purpose, "generation": qwen.generation_to_dict(g)}


class SpyGen:
    def __init__(self, inner):
        self.inner, self.requests = inner, []

    def generate(self, request):
        self.requests.append(request)
        return self.inner.generate(request)

    def model_info(self):
        return self.inner.model_info()


class Run:
    pass


def run_ctl(task, samples, repairs=(), cfg=None, arm="D0", critic=None, scorer=None, execution=None, budget=None, T=10 ** 6, ctl=None):
    k = len(samples)
    seeds = tuple(101 + i for i in range(k))
    recs = [rec(task, s, "sample", n) for s, n in zip(seeds, samples)] + [rec(task, 10000 + k + i, "repair", n) for i, n in enumerate(repairs)]
    r = Run()
    r.gen = SpyGen(qwen.FixtureGenerationBackend(recs))
    r.exe = execution or RecordedExecutionBackend()
    r.ctl = ctl or EvidenceLadderController()
    ctx = ControllerContext(r.gen, r.exe, budget or MatchedTokenBudget(task.task_id, arm, T), arm, critic, scorer, seeds, dict({"mode": "ladder", "k": k}, **(cfg or {})))
    r.ctx, r.res = ctx, r.ctl.run_problem(task, ctx)
    return r


def facts(r):
    res = r.res
    return {"journal_hash": res.journal_hash, "decision": res.decision, "stop": res.stop_cause, "details": res.stop_details, "final": res.final_artifact_id,
            "label": res.answer_label, "budget": res.budget_record, "rounds": res.rounds}


def stop_entry(r):
    return r.ctl.last_journal.entries("stop")[-1]["data"]


class Conflicting:
    """First call per artifact: its own recorded outcome; later calls for the good artifact: the recorded outcome of a known-wrong artifact (a re-check that disagrees)."""
    def __init__(self, inner, good, bad):
        self.inner, self.good, self.bad, self.n = inner, good, bad, {}

    def run_public(self, a, task):
        self.n[a.artifact_id] = self.n.get(a.artifact_id, 0) + 1
        if a.artifact_id == self.good.artifact_id and self.n[a.artifact_id] >= 2:
            return self.inner.run_public(self.bad, task)
        return self.inner.run_public(a, task)

    def run_experiment(self, a, e):
        return self.inner.run_experiment(a, e)

    def backend_info(self):
        return self.inner.backend_info()


# scenario runners (re-run by the reproducibility test)
def S_a():
    return run_ctl(py1(), ["py_ref"])


def S_b():
    return run_ctl(py1(), ["py_wrong_y"], ["py_wrong_y_v2"])


def S_c_ref():
    return run_ctl(jv1(), ["java_ref"])


def S_c_wrong():
    return run_ctl(jv1(), ["java_wrong"], ["java_wrong_v2"])


def S_d():
    return run_ctl(py1(), ["py_wrong_y"], ["py_ref"])


def S_e2():
    return run_ctl(py1(), ["py_ref"], execution=Conflicting(RecordedExecutionBackend(), art("py_ref"), art("py_wrong_y")))


def S_g1():
    return run_ctl(py1(), ["py_ref", "py_lower_first"], T=50)


def S_g2():
    return run_ctl(py1(), ["py_wrong_y"], ["py_ref"], T=60)


SCEN = [S_a, S_b, S_c_ref, S_c_wrong, S_d, S_e2, S_g1, S_g2]


# ------------------------------------------------------------------ a. python reference
@test
def a_python_reference_supported_never_verified():
    g = RecordedGrader()
    r = S_a()
    res = r.res
    assert res.stop_cause == "decided" and res.stop_details == "" and res.rounds == 0
    assert res.answer_label == "SUPPORTED" and res.answer_label != "VERIFIED" and not res.decision["authorized_to_execute"]
    assert labels(res) == {"R1": "SUPPORTED", "R2": "SUPPORTED", "R3": "VERIFIED", "R4": "VERIFIED", "R5": "VERIFIED", "R6": "VERIFIED"}, labels(res)
    assert res.final_artifact_id == art("py_ref").artifact_id and res.final_artifact.source == CASES["py_ref"][2]
    assert len(r.exe.public_calls) == 2 and set(r.exe.public_calls) == {("P1-01", art("py_ref").artifact_id)}      # evaluation + final re-check
    assert g.calls == [] and "grader" not in vars(r.ctx)                                                             # the context has no grader at all
    assert r.ctl.last_journal.verify_chain() and res.journal_hash == r.ctl.last_journal.journal_hash()
    C.validate("decision_record", res.decision)


# ------------------------------------------------------------------ b. python known wrong only
@test
def b_python_wrong_only_is_not_decided():
    r = S_b()
    res = r.res
    assert res.stop_cause == "same_failure_twice" and res.rounds == 1, (res.stop_cause, res.rounds)
    assert res.answer_label == "BLOCKED" and not res.decision["authorized_to_execute"] and res.stop_cause != "decided"
    assert res.final_artifact_id in (art("py_wrong_y").artifact_id, art("py_wrong_y_v2").artifact_id)
    ce = rec_ce("py_wrong_y", 2)
    fails = [e for e in r.ctl.last_journal.evidence(res.final_artifact_id) if e["status"] == "fail" and e.get("counterexample")]
    assert fails and fails[0]["requirement_id"] == "R5" and fails[0]["counterexample"] == ce, (fails, ce)
    assert labels(res)["R5"] in ("FAILED", "CONFLICTED")
    assert any(e["data"]["cause"] == "same_failure_twice" for e in r.ctl.last_journal.entries("lineage_stopped"))


# ------------------------------------------------------------------ c. java
@test
def c_java_reference_supported_and_known_wrong_blocked():
    r = S_c_ref()
    res = r.res
    assert res.stop_cause == "decided" and res.answer_label == "SUPPORTED" and not res.decision["authorized_to_execute"]
    assert labels(res) == {"R1": "SUPPORTED", "R2": "SUPPORTED", "R3": "VERIFIED", "R4": "VERIFIED", "R5": "VERIFIED", "R6": "VERIFIED"}, labels(res)
    w = S_c_wrong()
    res = w.res
    assert res.stop_cause == "same_failure_twice" and res.answer_label == "BLOCKED" and res.stop_cause != "decided"
    ce = rec_ce("java_wrong", 1)
    assert json.loads(ce["input"]) == {"year": 1900} and ce["expected"] == "false" and ce["observed"] == "true"
    fails = [e for e in w.ctl.last_journal.evidence(res.final_artifact_id) if e["status"] == "fail" and e.get("counterexample")]
    assert fails and fails[0]["requirement_id"] == "R4" and fails[0]["counterexample"] == ce, fails


# ------------------------------------------------------------------ d. wrong then repaired
@test
def d_failed_check_then_repair_uses_real_counterexample_and_fresh_evidence():
    r = S_d()
    res, j = r.res, r.ctl.last_journal
    wrong, ref = art("py_wrong_y"), art("py_ref")
    assert res.stop_cause == "decided" and res.rounds == 1 and res.final_artifact_id == ref.artifact_id
    assert res.final_artifact.parent_id == wrong.artifact_id and res.final_artifact.origin == "repair:" + wrong.artifact_id
    rq = r.gen.requests[1]
    ce = rec_ce("py_wrong_y", 2)
    assert rq.purpose == "repair" and rq.n == 1
    assert "Previous candidate: " + wrong.artifact_id in rq.prompt
    assert "%s -> expected %s, observed %s" % (ce["input"], ce["expected"], ce["observed"]) in rq.prompt, rq.prompt
    pe, ce_ev = j.evidence(wrong.artifact_id), j.evidence(ref.artifact_id)
    assert pe and ce_ev and {e["evidence_id"] for e in pe}.isdisjoint({e["evidence_id"] for e in ce_ev})
    assert all(e["candidate_id"] == ref.artifact_id and e["artifact_hash"] == ref.artifact_id for e in ce_ev)
    own = {e["evidence_id"] for e in ce_ev}
    assert all(set(row["evidence_ids"]) <= own for row in res.decision["requirement_rows"])
    assert ("P1-01", ref.artifact_id) in r.exe.public_calls                                                           # the child was executed itself
    first = [e for e in j.entries("label") if e["data"]["candidate_id"] == wrong.artifact_id][0]["data"]
    assert first["answer_label"] == "BLOCKED" and res.answer_label == "SUPPORTED"


# ------------------------------------------------------------------ e. stale and conflicting evidence
@test
def e1_stale_evidence_is_never_reused():
    r = S_d()
    run_, j = r.ctl.last_run, r.ctl.last_journal
    wrong, ref = art("py_wrong_y"), art("py_ref")
    d = decide_stage1(run_.reqs, j.evidence(wrong.artifact_id), ref.artifact_id, ref.artifact_id, run_.env, run_.registry)       # parent's records, child's id
    assert d["answer_label"] == "UNVERIFIED" and not d["authorized_to_execute"] and all(x["evidence_ids"] == [] for x in d["requirement_rows"])
    forged = [dict(e, candidate_id=ref.artifact_id) for e in j.evidence(wrong.artifact_id)]                                      # relabelled but artifact_hash is the parent's
    d = decide_stage1(run_.reqs, forged, ref.artifact_id, ref.artifact_id, run_.env, run_.registry)
    assert d["answer_label"] == "UNVERIFIED" and set(labels_of(d).values()) == {"UNVERIFIED"}
    own = j.evidence(ref.artifact_id)
    assert decide_stage1(run_.reqs, own, ref.artifact_id, ref.artifact_id, run_.env, run_.registry)["answer_label"] == "SUPPORTED"
    d = decide_stage1(run_.reqs, own, ref.artifact_id, ref.artifact_id, "env-other", run_.registry)                                # other environment revision
    assert d["answer_label"] == "UNVERIFIED" and not d["authorized_to_execute"]
    bumped = [intake_mod.bump(q, text=q["text"] + " (edited)") if q["requirement_id"] == "R3" else q for q in run_.reqs]       # requirement changed: old evidence stale
    d = decide_stage1(bumped, own, ref.artifact_id, ref.artifact_id, run_.env, run_.registry)
    assert labels_of(d)["R3"] == "UNVERIFIED" and d["answer_label"] == "UNVERIFIED"


@test
def e2_conflicting_evidence_replay_blocks():
    r = S_e2()
    res = r.res
    assert res.answer_label == "BLOCKED" and res.stop_cause == "tool_failure" and res.stop_details.startswith("finalize:"), (res.stop_cause, res.stop_details)
    assert "CONFLICTED" in labels(res).values() and not res.decision["authorized_to_execute"] and res.stop_cause != "decided"
    run_ = r.ctl.last_run
    a = art("py_ref")
    fac = Factory(run_.registry, run_.reqs, run_.env)
    fac.n = 500
    ev = r.ctl.last_journal.evidence(a.artifact_id)
    ev.append(fac.make(a, "R3", "public_examples_v1", "existing_test", "independent", "fail", "second trusted check", {"status": "fail"}))
    d = decide_stage1(run_.reqs, ev, a.artifact_id, a.artifact_id, run_.env, run_.registry)
    assert labels_of(d)["R3"] == "CONFLICTED" and d["answer_label"] == "BLOCKED" and not d["authorized_to_execute"]


# ------------------------------------------------------------------ f. GLM critic through the real broker
GLM_FX = json.loads((REPO / "evidence_ladder" / "models" / "fixtures" / "glm_responses.json").read_text(encoding="utf-8"))
OK_RAW = {"choices": [{"message": {"content": json.dumps({"experiment": {"kind": "function_call", "payload": '{"text": "hello world"}'},
                                                          "implications": "synthetic: distinguishes the candidates"})}}], "usage": {"prompt_tokens": 50, "completion_tokens": 20}}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_broker():
    d = Path(tempfile.mkdtemp(prefix="el-replay-ledger-"))
    clock = Clock()
    return StagedCriticBroker(d / "ledger.jsonl", config=glm.GlmConfig(reasoning_params={"x_test_only": "max"}), clock=clock), clock, d / "ledger.jsonl"


def ledger_kinds(path):
    return [json.loads(l)["event"] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


D1_CFG = {"critic": True, "critic_experiment": True}


def critic_pass(b, samples=("py_ref",)):
    return run_ctl(py1(), list(samples), cfg=D1_CFG, arm="D1", critic=b)


@test
def f1_glm_fixture_states_through_the_real_broker():
    seen = set()
    for c in GLM_FX["cases"]:
        b, clock, _ = make_broker()
        pid = b.build_packet(GLM_FX["request"])["packet_id"]
        r = b.ingest(pid, c["http_status"], c["raw_response"])
        assert r["state"] == c["expect_state"], (c["name"], r["state"])
        if "expect_reason" in c:
            assert r["reason"] == c["expect_reason"], (c["name"], r["reason"])
        if "expect_backoff_s" in c:
            assert r["backoff_s"] == c["expect_backoff_s"], (c["name"], r["backoff_s"])
        if "expect_stop" in c:
            assert r["stop_all_critic_calls"] is c["expect_stop"] and b.stop_all_critic_calls is c["expect_stop"], c["name"]
        if c["expect_state"] == "ok":
            assert b.completed(pid) == r["response"] and "experiment" in r["response"]
        else:
            assert b.completed(pid) is None and r["response"] is None
        if c["expect_state"] in ("ok", "malformed", "unavailable"):
            assert b.ingest(pid, 200, c["raw_response"])["state"] == "duplicate"                                            # idempotent by packet id
        seen.add(c["name"])
    assert {"malformed_json", "rate_limited_retry_after", "server_error_500", "timeout", "payment_required_402"} <= seen


@test
def f2_backoff_schedule_and_retry_exhaustion():
    b, clock, _ = make_broker()
    pid = b.build_packet(GLM_FX["request"])["packet_id"]
    assert len(b.export_pending()) == 1
    raw = {"body": {"error": "internal"}}
    for i, want in enumerate([2.0, 4.0, 8.0]):
        r = b.ingest(pid, 500, raw)
        assert r["state"] == "retry" and r["backoff_s"] == want, (i, r)
        assert b.export_pending() == []                                                                                  # backoff is computed, never slept: not due yet
        clock.t += want
        ep = b.export_pending()
        assert len(ep) == 1 and ep[0]["packet_id"] == pid and ep[0]["attempt"] == i + 2
    r = b.ingest(pid, 500, raw)
    assert r["state"] == "unavailable" and r["reason"].startswith("retries_exhausted") and b.completed(pid) is None
    assert b.counters["api_errors"] == 4
    b2, clock2, _ = make_broker()                                                                                       # 429 honours Retry-After
    p2 = b2.build_packet(GLM_FX["request"])["packet_id"]
    b2.export_pending()
    r = b2.ingest(p2, 429, {"headers": {"Retry-After": "17"}, "body": {}})
    assert r["state"] == "retry" and r["backoff_s"] == 17.0 and r["retry_after_s"] == 17.0
    clock2.t += 16.9
    assert b2.export_pending() == []
    clock2.t += 0.2
    assert len(b2.export_pending()) == 1


@test
def f3_payment_error_stops_all_critic_calls_and_survives_replay():
    b, clock, path = make_broker()
    pid = b.build_packet(GLM_FX["request"])["packet_id"]
    r = b.ingest(pid, 402, {"body": {"error": "Payment Required"}})
    assert r["state"] == "unavailable" and r["stop_all_critic_calls"] is True and b.stop_all_critic_calls
    other = dict(GLM_FX["request"], request_id="other", decision_id="other-d")
    raises(CriticStopped, b.build_packet, other)
    raises(CriticStopped, b.export_pending)
    assert b.completed(pid) is None
    b2 = StagedCriticBroker(path, config=glm.GlmConfig(reasoning_params={"x_test_only": "max"}), clock=clock)           # new process reads the same ledger
    assert b2.stop_all_critic_calls is True
    raises(CriticStopped, b2.build_packet, other)
    assert "sk-" not in path.read_text(encoding="utf-8") and "nvapi-" not in path.read_text(encoding="utf-8")


@test
def f4_controller_waits_for_the_broker_and_never_retries_itself():
    b, clock, path = make_broker()
    r1 = critic_pass(b)
    assert r1.res.stop_cause == "api_unavailable" and r1.res.stop_details == "critic_pending" and r1.res.decision["next_step"] == "discriminate"
    pid = next(iter(b.packets))
    assert b.ingest(pid, 429, {"headers": {"Retry-After": "17"}, "body": {}})["state"] == "retry"
    r2 = critic_pass(b)                                                                                                  # immediately: the packet is not due
    assert r2.res.stop_cause == "api_unavailable" and r2.res.stop_details == "critic_pending"
    kinds = ledger_kinds(path)
    assert kinds.count("packet_built") == 1 and kinds.count("ingest") == 1 and b.counters["sends"] == 0                   # the controller sent / ingested / re-queued nothing
    clock.t += 17
    assert len(b.export_pending()) == 1
    assert b.ingest(pid, 200, OK_RAW)["state"] == "ok"
    r3 = critic_pass(b)
    assert r3.res.stop_cause == "decided" and stop_entry(r3)["wasted_critic"] == 0
    assert [e["data"]["state"] for e in r3.ctl.last_journal.entries("critic")] == ["applied"]
    assert ("P1-01", art("py_ref").artifact_id, "function_call", '{"text": "hello world"}') in r3.exe.experiment_calls


def two_pass(status, raw):
    b, clock, path = make_broker()
    r1 = critic_pass(b)
    assert r1.res.stop_cause == "api_unavailable"
    pid = next(iter(b.packets))
    b.ingest(pid, status, raw)
    return critic_pass(b)


@expected_defect("evidence_ladder/controller/controller.py:269-272 + models/broker.py:133-137",
                 "completed() returns None for malformed/unavailable packets, so the controller cannot tell them from pending and returns api_unavailable forever. "
                 "Add a non-mutating StagedCriticBroker.terminal_state(pid) and, in critic_stage after done(pid) is None, map 'malformed' -> {'__malformed__':True} "
                 "and 'unavailable' -> {'state':'unavailable'} (both branches already exist at controller.py:276-285)")
def f5_malformed_critic_output_continues_without_critic():
    r = two_pass(200, GLM_FX["cases"][2]["raw_response"])                                                                # malformed_json
    assert r.res.stop_cause == "decided" and stop_entry(r)["wasted_critic"] == 1
    assert [e["data"]["state"] for e in r.ctl.last_journal.entries("critic")] == ["malformed"]


@expected_defect("evidence_ladder/controller/controller.py:266 (build_packet) + models/broker.py:101-103",
                 "after a 402 the broker sets stop_all_critic_calls and build_packet raises CriticStopped on the controller's second pass, which propagates out of run_problem. "
                 "Wrap controller.py:266 in try/except CriticStopped (compare by class name to avoid an import): note critic state 'unavailable', critic_done=True, wasted_critic+=1, return None")
def f6_payment_error_second_pass_continues_without_critic():
    r = two_pass(402, {"body": {"error": "Payment Required"}})
    assert r.res.stop_cause == "decided" and stop_entry(r)["wasted_critic"] == 1


# ------------------------------------------------------------------ g. budgets
@test
def g1_budget_exhausted_after_samples_is_never_decided():
    r = S_g1()
    res = r.res
    assert res.stop_cause == "budget" and res.stop_details == "token_cap" and res.stop_cause != "decided"
    assert res.budget_record["qwen_gen_tokens"] == 80 and res.rounds == 0 and res.final_artifact is not None
    assert not res.decision["authorized_to_execute"] and stop_entry(r)["budget_hit"] is True
    assert not r.ctl.last_journal.entries("finalize")                                                                     # no finalize without a decision


@test
def g2_budget_exhausted_during_repair_stops_even_with_a_good_child():
    r = S_g2()
    res = r.res
    assert res.stop_cause == "budget" and res.rounds == 1 and res.final_artifact.source == CASES["py_ref"][2]
    assert res.stop_cause != "decided" and res.budget_record["qwen_gen_tokens"] == 80


@test
def g3_models_token_budget_has_the_same_semantics():
    b = qwen.TokenBudget("P1-01", "D0", 50)
    r = run_ctl(py1(), ["py_ref", "py_lower_first"], budget=b)
    assert r.res.stop_cause == "budget" and r.res.budget_record["qwen_gen_tokens"] == 80 and b.exhausted()
    C.validate("budget_record", r.res.budget_record)
    m = MatchedTokenBudget("P1-01", "D0", 50)
    raises(ValueError, m.charge, "glm_out", 1)                                                                            # arm D0 has no GLM tokens


# ------------------------------------------------------------------ h. irreversible actions
@test
def h_irreversible_needs_confirmation():
    r = S_a()
    run_ = r.ctl.last_run
    a = art("py_ref")
    evs = r.ctl.last_journal.evidence(a.artifact_id)
    ex = [q for q in run_.reqs if run_.intake.roles[q["requirement_id"]] == "example"]
    args = (ex, evs, a.artifact_id, a.artifact_id, run_.env, run_.registry)
    assert authorize(*args, "sandboxed")                                                                                  # all rows verified by the recorded public run
    assert not authorize(*args, "irreversible", False)
    assert authorize(*args, "irreversible", True)
    assert not authorize(run_.reqs, evs, a.artifact_id, a.artifact_id, run_.env, run_.registry, "irreversible", True)     # with R2 stage 1 is only SUPPORTED
    for confirmed in (False, True):
        e = run_ctl(py1(), ["py_ref"], cfg={"risk": "irreversible", "user_confirmed": confirmed})
        assert e.res.decision["risk"] == "irreversible" and not e.res.decision["authorized_to_execute"] and e.res.answer_label == "SUPPORTED"


# ------------------------------------------------------------------ i. LAYA
EXPECTED_PATHS = [e["path"] for e in laya.EXPECTED_CAPTURE_CODE]


@test
def i1_unsupported_decision_type_is_rejected_unless_exploratory():
    s = laya.LayaScorer()
    pool = ["a", "b"]
    if s.disabled:
        raises(laya.ScorerDisabled, s.score, "next_action", pool)
    else:
        raises(UnsupportedDecisionType, s.score, "complete_program", pool)
        raises(UnsupportedDecisionType, s.score, "tool_call", pool, False)
        raises(NotAuthorized, s.score, "complete_program", pool, True)                                                   # exploratory is allowed, but no forward pass is authorized
        raises(NotAuthorized, s.score, "next_action", pool)
        raises(NotImplementedError, laya.LayaScorer(authorized=True).score, "next_action", pool)
    assert s.scorer_def["decision_types_validated"] == ["next_action"]


@test
def i2_capture_code_hash_is_wired_through_the_real_scorer_metadata():
    s = laya.LayaScorer()
    rep = s.meta.capture_report
    assert set(rep) == set(EXPECTED_PATHS)
    for e in laya.EXPECTED_CAPTURE_CODE:
        h = laya.file_sha256(REPO / e["path"])
        assert h is not None, "capture file missing from the bundle: " + e["path"]
        assert rep[e["path"]]["sha256"] == h and rep[e["path"]]["match"] == (h == e["sha256"])
    assert s.scorer_def["compatibility"]["capture_code_hash"] == ";".join(str(rep[p]["sha256"]) for p in EXPECTED_PATHS)
    manifest_ok = (REPO / laya.MANIFEST_REL).exists()
    assert s.disabled == (not (all(v["match"] for v in rep.values()) and manifest_ok)), (s.disabled, rep)
    root = Path(tempfile.mkdtemp(prefix="el-replay-capture-"))
    for p in EXPECTED_PATHS:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_bytes((REPO / p).read_bytes())
    same = laya.build_meta("laya_adapter_v1", "v1", "laya", repo_root=root, manifest_path=REPO / laya.MANIFEST_REL)
    assert same.scorer_def["status"] == s.scorer_def["status"]                                                           # an exact copy behaves like the original
    (root / EXPECTED_PATHS[0]).write_bytes((REPO / EXPECTED_PATHS[0]).read_bytes() + b"\n# tampered\n")
    bad = laya.build_meta("laya_adapter_v1", "v1", "laya", repo_root=root, manifest_path=REPO / laya.MANIFEST_REL)
    assert bad.scorer_def["status"] == "disabled" and "capture code hash mismatch" in bad.disabled_reason and not bad.capture_report[EXPECTED_PATHS[0]]["match"]
    raises(laya.ScorerDisabled, laya.LayaScorer(repo_root=root).score, "next_action", ["a"])


@test
def i3_controller_skips_laya_for_complete_program_unless_exploratory():
    r = run_ctl(py1(), ["py_ref", "py_timeout"], arm="E", cfg={"scorer": True}, scorer=laya.LayaScorer())
    assert r.res.stop_cause == "decided" and r.res.final_artifact_id == art("py_ref").artifact_id
    assert [e["data"]["state"] for e in r.ctl.last_journal.entries("scorer")] == ["skipped"]


@expected_defect("evidence_ladder/controller/controller.py:217",
                 "with exploratory=True the real LayaScorer raises NotAuthorized (no forward pass in this build) or ScorerDisabled, and apply_scorer only catches UnsupportedDecisionType, "
                 "so arm E crashes the pilot on any task with two live candidates. Catch Exception there: journal {'state':'refused'}, set scorer_off, continue")
def i4_exploratory_laya_without_forward_pass_must_not_crash_the_run():
    r = run_ctl(py1(), ["py_ref", "py_timeout"], arm="E", cfg={"scorer": True, "exploratory": True}, scorer=laya.LayaScorer())
    assert r.res.stop_cause == "decided" and r.res.final_artifact_id == art("py_ref").artifact_id
    assert all(e["data"]["state"] != "scored" for e in r.ctl.last_journal.entries("scorer"))


# ------------------------------------------------------------------ pilot harness (real PilotRunner)
PILOT_PLAN = {"P1-01": (["py_wrong_y", "py_ref", "py_lower_first", "py_syntax"], []),
              "J1-02": (["java_wrong", "java_wrong_v2", "java_compile_error", "java_wrong_v3"], ["java_ref", "java_ref", "java_ref"])}
CRITIC_PLAN = {"P1-01": (["py_ref", "py_timeout", "py_syntax", "py_const99"], [])}


def pilot_records(plan):
    recs = []
    for tid, (samples, repairs) in plan.items():
        t = task_of(tid)
        seeds = pairing.arm_seeds(MASTER, tid, "B", K1)
        assert len(seeds) == K1 and len(set(seeds)) == K1
        recs += [rec(t, s, "sample", n) for s, n in zip(seeds, samples)]
        recs += [rec(t, 10000 + K1 + i, "repair", n) for i, n in enumerate(repairs)]
    return recs


CRITIC_LOC = "kaggle_secrets_critic_kernel"   # the runner's configured critic location; the approval must name the same one


def valid_manifest(root):
    """Fabricated approvals, written ONLY inside the test's own temp directory. Never a real approval."""
    assert any(p.startswith("el-replay") for p in Path(root).resolve().parts)
    return authorization.write_authorization_manifest({**{k: True for k in authorization.APPROVALS if k != "d7_critic_location"},
                                                       "d7_critic_location": CRITIC_LOC}, out_dir=Path(root) / "auth")


def make_pilot(root, flip=False, arms=ARMS, plan=PILOT_PLAN, auth="valid", transport=None, T=10 ** 6, k1=K1, dry_run=False, sub=None):
    root = Path(root)
    gate = pilot.SelectionGate()
    grader = RecordedGrader(gate=gate, flip=flip)
    gen = SpyGen(qwen.FixtureGenerationBackend(pilot_records(plan)))
    ts = {"python": [py1()], "java": [jv1()]} if sub is None else sub
    ts = {k: v for k, v in ts.items() if any(t.task_id in plan for t in v)}
    mf = valid_manifest(root) if auth == "valid" else auth
    runner = pilot.PilotRunner(tasks=ts, controller=EvidenceLadderController(), generation=gen, execution=RecordedExecutionBackend(), grader=grader, gate=gate,
                               out_dir=root / "out", T=T, k1=k1, master_seed=MASTER, n_total=K1, arms=arms,
                               critic_factory=lambda t, a: StagedCriticBroker(root / "ledgers" / ("%s-%s.jsonl" % (t.task_id, a)),
                                                                              config=glm.GlmConfig(reasoning_params={"x_test_only": "max"}), clock=Clock()),
                               scorer=None, transport=transport, authorization_manifest=mf, dry_run=dry_run, clock=FakeClock(), critic_location=CRITIC_LOC)
    return runner, grader, gen


def tmp(prefix="el-replay-"):
    return Path(tempfile.mkdtemp(prefix=prefix))


def file_hashes(out):
    out = Path(out)
    return {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}


def pilot_full(flip=False):
    runner, grader, gen = make_pilot(tmp(), flip=flip)
    out = runner.run()
    return runner, grader, out


# ------------------------------------------------------------------ j. reproducibility
@test
def j_whole_scenario_set_is_reproducible():
    for f in SCEN:
        a, b = facts(f()), facts(f())
        assert a == b, f.__name__
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert facts(S_b())["journal_hash"] != facts(S_d())["journal_hash"]                                                   # the receipt does depend on the content
    r1, _, o1 = pilot_full()
    r2, _, o2 = pilot_full()
    h1, h2 = file_hashes(r1.out), file_hashes(r2.out)
    assert h1 == h2 and {"report.json", "report.md", "selection/selection_log.jsonl", "selection/frozen_artifacts.json", "grading/grades.jsonl"} <= set(h1), sorted(h1)
    assert o1["selection"] == o2["selection"] and o1["report"] == o2["report"]
    for t in r1.records.task_ids():
        for a in ARMS:
            x, y = r1.records.get(t, a), r2.records.get(t, a)
            assert x == y and x["journal_hash"] == y["journal_hash"] and x["final_artifact_id"] == y["final_artifact_id"]


# ------------------------------------------------------------------ k. grading never influences selection
@test
def k_protected_grading_never_influences_selection():
    ra, ga, _ = make_pilot(tmp(), flip=False)
    rb, gb, _ = make_pilot(tmp(), flip=True)
    ra.run_selection()
    rb.run_selection()
    assert ga.calls == [] and gb.calls == [] and ga.refused == 0                                                          # nothing graded while selecting
    some = next(iter(ra._artifacts.values()))
    raises(GraderIsolationError, ga.grade, some)                                                                          # gate still open
    raises(GraderIsolationError, gb.grade, some)
    raises(GraderIsolationError, ra.grade_all)                                                                            # nothing frozen yet
    assert ga.calls == [] and gb.calls == [] and ga.refused == 1
    ha, hb = ra.freeze_and_close(), rb.freeze_and_close()
    assert ha == hb and ra.gate.is_closed() and ga.calls == []                                                            # closing the gate does not grade either
    assert (ra.out / "selection" / "selection_log.jsonl").read_bytes() == (rb.out / "selection" / "selection_log.jsonl").read_bytes()
    assert (ra.out / "selection" / "frozen_artifacts.json").read_bytes() == (rb.out / "selection" / "frozen_artifacts.json").read_bytes()
    for t in ra.records.task_ids():
        for a in ARMS:
            x, y = ra.records.get(t, a), rb.records.get(t, a)
            assert x == y and x["journal_hash"] == y["journal_hash"] and x["final_artifact_id"] == y["final_artifact_id"]
    gg = pilot.GatedGrader(ga, ra.gate, set())
    raises(GraderIsolationError, gg.grade, some)                                                                          # not a frozen final artifact
    ra.grade_all()
    rb.grade_all()
    assert ga.calls and gb.calls
    A = {(g["task_id"], g["arm"]): g for g in ra._grades}
    B = {(g["task_id"], g["arm"]): g for g in rb._grades}
    assert set(A) == set(B) and len(A) == 12
    for key, g in A.items():
        assert g["artifact_id"] == B[key]["artifact_id"] and g["graded"] and B[key]["passed"] == (not g["passed"]), key
    assert A[("P1-01", "A")]["passed"] is False and A[("J1-02", "A")]["passed"] is False and A[("J1-02", "B")]["passed"] is False
    assert all(A[("J1-02", arm)]["passed"] is True for arm in ("C", "D0", "D1", "E"))                                     # repaired to the reference
    assert ra.records.get("J1-02", "C")["answer_label"] == "SUPPORTED" and ra.records.get("J1-02", "B")["answer_label"] == "BLOCKED"
    raises(GraderIsolationError, ra._one, ra.tasks["python"][0], "A", 3, None, pilot._Tally())                            # no selection after grading


# ------------------------------------------------------------------ l. authorization
@test
def l_pilot_refuses_without_a_complete_valid_manifest():
    root = tmp("el-replay-auth-sandbox-")
    inc = authorization.write_authorization_manifest({"d1_waiver": True, "d7_critic_location": CRITIC_LOC, "nvidia_credits_confirmed": True}, out_dir=root / "inc")
    good = valid_manifest(root)
    tam = root / "tampered.json"
    m = json.loads(good.read_text(encoding="utf-8"))
    m["approvals"]["pilot_go"] = False
    tam.write_text(json.dumps(m), encoding="utf-8")
    for i, mf in enumerate((None, root / "missing.json", inc, tam)):
        runner, grader, gen = make_pilot(root / ("run%d" % i), auth=mf)
        raises(NotAuthorized, runner.run_selection)
        assert gen.requests == [] and grader.calls == [] and not (runner.out / "selection").exists(), i                   # nothing started
    raises(ValueError, authorization.write_authorization_manifest, {"bogus": True}, root / "x")
    assert authorization.require_authorization(good, critic_location=CRITIC_LOC)["approvals"] == {
        **{k: True for k in authorization.APPROVALS if k != "d7_critic_location"}, "d7_critic_location": CRITIC_LOC}
    assert str(good).startswith(str(root)) and "el-replay" in str(good)
    runner, grader, gen = make_pilot(root / "ok")
    info = runner.run_selection()
    assert info["rows"] == 12 and gen.requests
    raises(ConfigError, make_pilot(root / "noT", T=None)[0].run_selection)
    raises(ConfigError, make_pilot(root / "nok1", k1=None)[0].run_selection)
    raises(ConfigError, make_pilot(root / "dry", dry_run=True)[0].run_selection)                                          # dry runs need a scripted fake backend


# ------------------------------------------------------------------ m. per-track separation and mutation flags
@test
def m_tracks_stay_separate_and_mutation_flags_survive():
    runner, _, out = pilot_full()
    results = {}
    for g in runner._grades:
        results.setdefault(g["track"], {}).setdefault(g["task_id"], {})[g["arm"]] = g
    meta = {tr: {t.task_id: t for t in ts} for tr, ts in tasks().items()}
    rows = [runner.records.get(t, a) for t in runner.records.task_ids() for a in ARMS]
    rep = report.make_report(results, meta, rows, T=runner.T, seed=MASTER)
    py, jv = rep["tracks"]["python"], rep["tracks"]["java"]
    assert py["track"] == "python" and jv["track"] == "java" and py["n_tasks"] == 40 and jv["n_tasks"] == 40
    st = lambda sec: {t: f["state"] for t, f in sec["mutation_flags"].items()}   # noqa: E731
    assert st(py) == {"P1-19": "below_threshold", "P1-26": "below_threshold", "P1-05": "unmeasured", "P1-10": "unmeasured", "P1-14": "unmeasured"}, st(py)
    assert st(jv) == {"J1-17": "below_threshold", "J1-27": "below_threshold", "J1-08": "unmeasured", "J1-10": "unmeasured", "J1-20": "unmeasured"}, st(jv)
    assert py["mutation_flag_counts"] == {"below_threshold": 2, "unmeasured": 3} and jv["mutation_flag_counts"] == {"below_threshold": 2, "unmeasured": 3}
    assert all(f["flagged"] and f["waiver_applies"] is False for f in list(py["mutation_flags"].values()) + list(jv["mutation_flags"].values()))
    assert "P1-01" not in py["mutation_flags"] and "J1-02" not in jv["mutation_flags"]
    assert set(py["per_arm"]) == set(ARMS) and all(v["n_graded"] == 1 for v in py["per_arm"].values()) and all(v["n_graded"] == 1 for v in jv["per_arm"].values())
    assert py["per_arm"]["A"]["solved"] == 0 and jv["per_arm"]["B"]["solved"] == 0 and jv["per_arm"]["C"]["solved"] == 1      # numbers stay inside their own track
    assert "excluding_flagged_tasks" in py["contrasts"]["C_vs_B"] and rep["pooled"].startswith("NOT PRODUCED") and rep["tracks"]["fable"]["numbers"].startswith("NOT PRODUCED")
    raises(PoolingError, report.pool_tracks, py, jv)
    raises(PoolingError, report.make_report, {"java": {"P1-01": {"A": {"passed": True}}}}, meta, [], seed=1)              # a Python task inside the Java track
    raises(NotImplementedError, report.make_report, {"fable": {"X": {"A": {"passed": True}}}}, meta, [], seed=1)
    md = report.render_markdown(rep)
    assert "## Track: python" in md and "## Track: java" in md and "never pooled" in md
    assert out["report"]["tracks"]["python"]["n_tasks"] == 1                                                              # the runner's own report covers only what it ran


# ------------------------------------------------------------------ n. critic two-pass through the real PilotRunner
@expected_defect("evidence_ladder/orchestrator/pilot.py:84-100 (_Critic) + controller/controller.py:269,291",
                 "(1) _Critic does not forward completed(pid), so pass 2 never sees the ingested response (controller.py:269 uses getattr(ctx.critic,'completed',None)); add "
                 "`def completed(self, pid): return self._i.completed(pid)`. (2) the controller's own export_pending() call (controller.py:291, journal count) marks the packet "
                 "'exported', so pilot.py:159/186 export_pending() returns [] and nothing is ever transported; make _Critic.export_pending() call "
                 "self._i.export_pending(include_exported=True) (and pilot.py:159/186 use the proxy), or have the controller use pending_summary()")
def n_pilot_critic_two_pass_with_the_real_broker():
    root = tmp()
    runner, grader, gen = make_pilot(root, arms=("D1",), plan=CRITIC_PLAN, transport=lambda packets: [(p["packet_id"], 200, OK_RAW) for p in packets])
    info = runner.run_selection()
    assert info["pending_packets"] >= 1 and info["transport_used"] is True, info
    row = runner.records.get("P1-01", "D1")
    assert row["pass"] == 2 and row["stop_cause"] == "decided" and row["final_artifact_id"] == art("py_ref").artifact_id, row
    assert row["critic_pending"] is False and row["api_calls"] >= 1


# ------------------------------------------------------------------ o. replay doubles never invent
@test
def o_recorded_doubles_refuse_unknown_inputs_and_fixtures_are_clean():
    be = RecordedExecutionBackend()
    unknown = Artifact.make("P1-01", "python", "def solve(d): return 7")
    raises(UnrecordedError, be.run_public, unknown, py1())
    raises(UnrecordedError, be.run_experiment, art("py_ref"), {"kind": "function_call", "payload": '{"text": "never recorded"}'})
    other = task_of("P1-02")
    raises(UnrecordedError, be.run_public, art("py_ref"), other)                                                          # public tests differ from the recorded ones
    forged = Artifact("0" * 64, "P1-01", "python", CASES["py_ref"][2])
    raises(UnrecordedError, be.run_public, forged, py1())
    raises(UnrecordedError, RecordedGrader().grade, unknown)
    outs = be.run_public(art("py_ref"), py1())
    outs[0]["status"] = "fail"
    assert be.run_public(art("py_ref"), py1())[0]["status"] == "pass"                                                     # callers get copies
    gr = load_grades()["grades"]
    assert len(gr) == len(CASES) and all(set(g) == {"artifact_id", "task_id", "passed", "passed_tests", "total_tests", "failure", "sha256_before_run"} for g in gr.values())
    assert sum(1 for g in gr.values() if g["passed"]) == 2 and gr["py_lower_first"]["passed"] is False and gr["py_wrong_y"]["passed_tests"] < gr["py_wrong_y"]["total_tests"]
    for p in sorted(FIXDIR.glob("*.json")):
        text = p.read_text(encoding="utf-8")
        assert "nvapi-" not in text and "Bearer " not in text and ".secrets" not in text and "api_key" not in text.lower(), p.name
    ob = load_outcomes()
    assert ob["backend_info"]["python_jail"]["uid"] == 65534 and set(ob["cases"]) == set(CASES)
    for name, e in ob["cases"].items():
        assert sha256_text(e["source"]) == e["artifact_id"] == art(name).artifact_id and all("duration_s" not in o for o in e["public"]), name
    assert all(o["status"] == "pass" for o in ob["cases"]["py_ref"]["public"]) and ob["cases"]["py_timeout"]["public"][1]["status"] == "timeout"
    assert ob["cases"]["py_syntax"]["public"][0]["failure_class"] == "compile_error" and ob["cases"]["java_compile_error"]["public"][0]["failure_class"] == "compile_error"


def main():
    failed, open_def, fixed = {}, {}, []
    for f in TESTS:
        try:
            f()
            if f.__name__ in DEFECTS:
                fixed.append(f.__name__)
        except BaseException:
            tb = traceback.format_exc()[-1800:]
            if f.__name__ in DEFECTS:
                open_def[f.__name__] = dict(DEFECTS[f.__name__], error=tb)
            else:
                failed[f.__name__] = tb
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed) - len(open_def), "failed": failed, "open_defects": open_def, "fixed_defects": fixed}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
