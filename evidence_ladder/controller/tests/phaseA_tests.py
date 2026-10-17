"""[EL-ARCH-1 Phase A] Plain-assert tests for A1 (GLM accounting from the real ledgers), A2 (transport error class, no credentials), A3 (enforced limits), A4 (model-derived evidence capped),
A5 (I1-I8 real, MissingComponentError). No network, no keys. Run from the repo root: python evidence_ladder/controller/tests/phaseA_tests.py   (prints {"tests","passed","failed"}; exit 1 on failure).
The live step-E ledgers under results/raw/laya-integration-* are used READ-ONLY (copied before anything touches them)."""
import io
import json
import shutil
import socket
import ssl
import sys
import tempfile
import traceback
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_tests as RT                                                                  # noqa: E402  the controller test fakes (FakeGen, FakeExec, run, ...)
from evidence_ladder.common.contract_import import C                                     # noqa: E402
from evidence_ladder.common.interfaces import ControllerContext                          # noqa: E402
from evidence_ladder.controller import invariants as inv                                 # noqa: E402
from evidence_ladder.controller import router as router_mod                              # noqa: E402
from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask, LimitReached   # noqa: E402
from evidence_ladder.controller.components import MissingComponentError                  # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController               # noqa: E402
from evidence_ladder.controller.decision import decide_stage1                            # noqa: E402
from evidence_ladder.controller.evidence import Factory, ModelEvidenceError              # noqa: E402
from evidence_ladder.controller.journal import EvidenceJournal, JournalError              # noqa: E402
from evidence_ladder.controller.limits import Limiter, Limits, LimitsConfigError, load_limits   # noqa: E402
from evidence_ladder.critic_batch import accounting as ACC                               # noqa: E402
from evidence_ladder.critic_batch import transport as TR                                 # noqa: E402
from evidence_ladder.models import experiments as EXP, glm                               # noqa: E402
from evidence_ladder.models._util import describe_error, redact_message                  # noqa: E402
from evidence_ladder.models.broker import StagedCriticBroker                             # noqa: E402
from evidence_ladder.models.pc_sender import KeyScheduler, PcSender                      # noqa: E402

LIVE = ROOT / "results/raw/laya-integration-20261016T184816Z/stepE_20261017T070508Z/critic_batch"
TESTSRC = 'from slug import slugify\n\nassert slugify("Hello, World!") == "hello-world"\nprint("all tests passed")\n'
SLUG = 'import re\n\n\ndef slugify(text):\n    text = text.strip().lower()\n    return re.sub(r"[^a-z0-9]+", "-", text)\n'
CMD = "python3 test_slug.py"
CFG = glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=EXP.SUPPORTED_KINDS)
FAKEKEY = "nvapi" + "-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"   # split so no key-shaped literal sits in the source
READ = json.dumps({"name": "Read", "arguments": {"file_path": "slug.py"}})


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeSandbox:
    def run(self, workspace, cmd, persist=False, timeout=60):
        ok = cmd != CMD
        return dict(exit_code=0 if ok else 1, killed=False, stdout="'a'\n" if ok else "", stderr="" if ok else "AssertionError", protected_unchanged=True)


def make_run(limits=None, clock=None, config=None, scorer=None, steps=1, qwen_meta=None, arm="E", critic=None):
    w = Path(tempfile.mkdtemp(prefix="phaseA-")) / "ws"
    w.mkdir(parents=True)
    (w / "slug.py").write_text(SLUG, "utf-8")
    (w / "test_slug.py").write_text(TESTSRC, "utf-8")
    task = AgenticTask("t", "Fix slug.py so that `%s` passes. Do not modify test_slug.py." % CMD, w, CMD, ["test_slug.py"], "sys", max_steps=10)
    r = AgenticRun(task, AgenticContext(sandbox=FakeSandbox(), scorer=scorer, arm=arm, config=dict(config or {}), limits=limits, clock=clock, critic=critic), lambda m: "ctx")
    r.begin()
    for _ in range(steps):
        r.step([READ], qwen_meta or {})
    return r


def big(**kw):
    d = dict(max_wall_seconds=1e9, max_qwen_tokens=10 ** 9, max_glm_tokens=10 ** 9, max_actions=100, max_rounds=100)
    d.update(kw)
    return Limits(**d)


def critic_request(r):
    return r.critic_request()


# ====================================================================== A1
def test_a1_accounting_from_the_live_step_e_ledgers_readonly():
    a = ACC.account(LIVE / "broker_ledger.jsonl", LIVE / "progress.jsonl")
    t = a["totals"]
    assert (t["calls"], t["ok"], t["retries"], t["failed_attempts"], t["packets"]) == (6, 3, 3, 3, 3), t
    assert t["estimated"] is True and t["tokens_in"] > 0 and t["tokens_out"] > 0                  # provider sent no usage: estimated from text length, flagged
    for pid, p in a["per_packet"].items():
        assert (p["calls"], p["ok"], p["retries"], p["failed_attempts"]) == (2, 1, 1, 1) and p["estimated"] and len(p["attempts"]) == 2
        assert p["attempts"][0]["tokens_out"] == 0 and p["attempts"][0]["error"]["category"] == "timeout"       # the failed attempt consumed no estimated tokens and keeps its error class
    assert sum(p["tokens_out"] for p in a["per_packet"].values()) == t["tokens_out"]
    ev = [json.loads(l) for l in (LIVE / "progress.jsonl").read_text("utf-8").splitlines()]
    assert sum(1 for e in ev if e["event"] == "send") == t["calls"]                               # independent count straight from the file


def test_a1_result_json_budget_equals_ledger_counts():
    d = Path(tempfile.mkdtemp(prefix="phaseA-ledger-"))
    shutil.copytree(LIVE, d / "cb")
    acc = ACC.account(d / "cb/broker_ledger.jsonl", d / "cb/progress.jsonl")

    class Stub:                                  # a critic whose ledgers are the real live ones
        ledger_path = d / "cb/broker_ledger.jsonl"
        cfg = CFG

    r = make_run()
    r.ctx.critic = Stub()
    r.critic_pids = list(acc["per_packet"])
    res = r.finish()
    b = res.budget_record
    prog = [json.loads(l) for l in (d / "cb/progress.jsonl").read_text("utf-8").splitlines()]
    assert b["glm_calls"] == sum(1 for e in prog if e["event"] == "send") == 6
    assert b["api_errors"] == 3 and b["glm_tokens_in"] == acc["totals"]["tokens_in"] > 0 and b["glm_tokens_out"] == acc["totals"]["tokens_out"] > 0
    assert r.journal.entries("glm_accounting")[-1]["data"]["totals"]["retries"] == 3
    assert b["wall_seconds"] >= 0.0
    assert (LIVE / "progress.jsonl").read_bytes() == (d / "cb/progress.jsonl").read_bytes()       # the fixture was not modified


def test_a1_provider_usage_is_used_when_present_and_not_flagged():
    d = Path(tempfile.mkdtemp())
    b = StagedCriticBroker(d / "l.jsonl", config=CFG, max_retries=2)
    req = make_run().critic_request()
    pid = b.build_packet(req)["packet_id"]
    b.export_pending()
    b.log_send(pid, "key_1")
    b.ingest(pid, 503, {"error": "overloaded"})
    b.packets[pid]["not_before"] = 0.0
    b.export_pending()
    b.log_send(pid, "key_1")
    raw = {"choices": [{"message": {"content": json.dumps({"experiment": {"kind": "command", "payload": CMD}, "implications": "i"})}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 5}}
    b.ingest(pid, 200, raw)
    t = ACC.account(d / "l.jsonl")["totals"]
    assert (t["calls"], t["ok"], t["retries"], t["failed_attempts"], t["tokens_in"], t["tokens_out"], t["estimated"]) == (2, 1, 1, 1, 7, 5, False), t


# ====================================================================== A2
def test_a2_describe_error_classes():
    cases = ((ConnectionResetError(104, "reset"), "connect", "connect", 104), (ConnectionResetError(104, "reset"), "stream", "read", 104), (ssl.SSLError(1, "tlsv1 alert"), "connect", "tls", 1),
             (socket.timeout("timed out"), "stream", "timeout", None), (urllib.error.URLError(ssl.SSLError("bad cert")), "connect", "tls", None),
             (urllib.error.URLError(ConnectionResetError(104, "x")), "connect", "connect", 104))
    for exc, phase, cat, en in cases:
        d = describe_error(exc, phase)
        assert d["category"] == cat and d["errno"] == en and d["error_class"].startswith(type(exc).__name__), (exc, d)
    assert describe_error(ValueError("x"))["category"] == "other"
    m = redact_message("GET https://x.y/v1?api_key=SECRET&b=1 failed: Authorization: Bearer abc.def-123 " + FAKEKEY + " " + "z" * 40, secrets=["SECRET"])
    for bad in ("SECRET", "abc.def", FAKEKEY, "z" * 40, "b=1"):
        assert bad not in m, (bad, m)


class Resp:
    status = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)


def send_with(opener_or_exc, max_retries=0):
    """Real StagedCriticBroker + PcSender + make_transport with an injected opener; returns (broker, ledger text)."""
    d = Path(tempfile.mkdtemp(prefix="phaseA-send-"))
    b = StagedCriticBroker(d / "l.jsonl", config=CFG, max_retries=max_retries)
    b.build_packet(make_run().critic_request())

    def opener(req, timeout):
        if isinstance(opener_or_exc, BaseException):
            raise opener_or_exc
        return opener_or_exc(req, timeout)
    PcSender(b, KeyScheduler(["key_1"]), sleep=lambda s: None).send_pending(TR.make_transport({"key_1": FAKEKEY}, opener=opener))
    ing = [json.loads(l) for l in (d / "l.jsonl").read_text("utf-8").splitlines() if '"event":"ingest"' in l]
    return ing[-1], (d / "l.jsonl").read_text("utf-8")


def test_a2_injected_transport_errors_are_classified_and_leak_no_key():
    hdr = "Authorization: Bearer " + FAKEKEY
    cases = ((ConnectionResetError(104, "reset by peer " + hdr), "connect", "ConnectionResetError"), (ssl.SSLError(1, "handshake failed " + FAKEKEY), "tls", "SSLError"),
             (socket.timeout("read timed out"), "timeout", "TimeoutError"), (urllib.error.URLError("dns failure https://api.x/v1?key=" + FAKEKEY), "connect", "URLError"))
    for exc, cat, cls in cases:
        ing, text = send_with(exc)
        e = ing["error"]
        assert ing["reason"].endswith("transport_failure") and ing["http_status"] is None, ing["reason"]       # the old reason is kept ...
        assert e["error_class"].startswith(cls) and e["category"] == cat, e                                       # ... and the class is added
        assert FAKEKEY not in text and "Bearer " + FAKEKEY not in text and "?key=" not in text
    cut = lambda req, timeout: Resp([b'data: {"choices":[{"delta":{"content":"{"}}]}\n'])                         # stream ends without finish_reason / [DONE]
    ing, text = send_with(cut)
    assert ing["error"]["category"] == "truncated" and ing["error"]["error_class"] == "TruncatedStream"

    def http5xx(req, timeout):
        raise urllib.error.HTTPError("https://x", 503, "unavailable", {}, io.BytesIO(b'{"error": "overloaded, token ' + FAKEKEY.encode() + b'"}'))
    ing, text = send_with(http5xx)
    e = ing["error"]
    assert ing["http_status"] == 503 and e["category"] == "http" and e["http_status_class"] == "5xx" and e["error_class"] == "HTTP503" and FAKEKEY not in text and FAKEKEY not in json.dumps(e)


# ====================================================================== A3
def test_a3_limits_file_is_explicit_and_loud():
    for prof in ("agentic", "ladder"):
        L = load_limits(prof)
        assert L.max_wall_seconds > 0 and L.max_qwen_tokens > 0 and L.max_glm_tokens > 0 and L.max_actions > 0 and L.max_rounds > 0
    bad = Path(tempfile.mkdtemp()) / "x.toml"
    bad.write_text("[profile.agentic]\nmax_actions = 3\n", "utf-8")
    for path, prof in ((bad, "agentic"), (bad, "nope"), (bad.parent / "missing.toml", "agentic")):
        try:
            load_limits(prof, path)
            assert False
        except LimitsConfigError:
            pass
    c = Clock()
    lm = Limiter(Limits(10.0, 100, 100, 2, 2), c)
    lm.start()
    assert lm.exceeded(100, 100, 1, 1) is None and lm.exceeded(101)["resource"] == "max_qwen_tokens" and Limiter(Limits(10.0, 100, 100, 2, 2), c).exceeded(0, 101)["resource"] == "max_glm_tokens"
    assert lm.exceeded(0, 0, 2, 0)["details"] == "limit:max_actions used=2 cap=2"
    c.t = 10.5
    assert lm.exceeded()["resource"] == "max_wall_seconds"


def test_a3_agentic_wall_time_actions_rounds_tokens_stop_as_budget_with_the_resource_named():
    c = Clock()
    r = make_run(limits=big(max_wall_seconds=100.0), clock=c, steps=1)
    c.t = 101.0
    try:
        r.step([READ])
        assert False
    except LimitReached as e:
        assert "max_wall_seconds" in str(e)
    res = r.finish()
    assert res.stop_cause == "budget" and res.stop_details.startswith("limit:max_wall_seconds") and res.budget_record["wall_seconds"] == 101.0 and not res.decision["authorized_to_execute"]
    assert r.journal.entries("stop")[-1]["data"]["details"] == "limit:max_wall_seconds"           # no wall seconds in the journal (reproducible hash)
    r = make_run(limits=big(max_actions=1), steps=1)
    try:
        r.step([READ])
        assert False
    except LimitReached:
        pass
    res = r.finish()
    assert res.stop_cause == "budget" and res.stop_details == "limit:max_actions used=1 cap=1" and len(r.steps) == 1
    r = make_run(limits=big(max_rounds=2), steps=2)
    try:
        r.step([READ])
        assert False
    except LimitReached:
        pass
    assert r.finish().stop_details == "limit:max_rounds used=2 cap=2"
    r = make_run(limits=big(max_qwen_tokens=1000), steps=1, qwen_meta={"new_tokens": 900, "prompt_tokens": 200})
    try:
        r.step([READ], {})
        assert False
    except LimitReached:
        pass
    assert r.finish().stop_details == "limit:max_qwen_tokens used=1100 cap=1000"
    r = make_run(limits=big(max_actions=1), steps=1)
    r.step_no = 0                                                                                    # a decided run never becomes a budget stop
    assert r.limit_check()["resource"] == "max_actions"


def test_a3_glm_token_limit_blocks_the_critic_stage_and_i5_catches_overshoot():
    class Stub:
        ledger_path = None
    r = make_run(limits=big(max_glm_tokens=10))
    r.ctx.critic = Stub()
    r.budget.set(glm_tokens_in=8, glm_tokens_out=8)
    assert r.critic_stage() == "unavailable" and r.journal.entries("critic")[-1]["data"]["reason"] == "limit:max_glm_tokens"
    res = r.finish()
    assert res.stop_cause == "budget" and res.stop_details == "limit:max_glm_tokens used=16 cap=10"
    # I5 itself: a result above a cap that did not stop as budget is a violation
    r2 = make_run(limits=big())
    res2 = r2.finish()
    res2.budget_record = dict(res2.budget_record, qwen_gen_tokens=5)
    res2.stop_cause = "no_progress_2"
    assert "max_qwen_tokens" in inv.check_i5(res2, r2.journal, Limits(1e9, 1, 1, 100, 100))
    res2.stop_cause = "decided"                                                                       # token overshoot inside the deciding step is physically unavoidable: allowed
    assert "max_qwen_tokens" not in inv.check_i5(res2, r2.journal, Limits(1e9, 1, 1, 100, 100))


def test_a3_ladder_controller_enforces_actions_rounds_wall_and_tokens():
    # actions: one attempt allowed although cfg says k=3
    res, ctl, gen, _ = RT.run([["WRONG:a:1", "WRONG:b:2", "WRONG:c:3"], ["OK:z"]], {"k": 3}, ctl=EvidenceLadderController(limits=big(max_actions=1)))
    assert res.stop_cause == "budget" and res.stop_details == "limit:max_actions used=1 cap=1" and gen.calls[0].n == 1, (res.stop_cause, res.stop_details)
    assert any(e["data"].get("resource") == "max_actions" for e in ctl.last_journal.entries("limit"))
    # rounds
    res, ctl, gen, _ = RT.run([["WRONG:a:1"], ["WRONG:b:2"], ["WRONG:c:3"]], {"k": 1}, ctl=EvidenceLadderController(limits=big(max_rounds=1)))
    assert res.stop_cause == "budget" and res.stop_details == "limit:max_rounds used=1 cap=1" and res.rounds == 1, (res.stop_cause, res.stop_details)
    # wall time with an injected clock
    c = Clock()
    seen = {}

    def hook(art):
        c.t += 50.0
    res, ctl, gen, _ = RT.run([["WRONG:a:1"], ["WRONG:b:2"], ["WRONG:c:3"]], {"k": 1}, exe=RT.FakeExec(hook=hook), ctl=EvidenceLadderController(limits=big(max_wall_seconds=60.0), clock=c))
    assert res.stop_cause == "budget" and res.stop_details.startswith("limit:max_wall_seconds"), (res.stop_cause, res.stop_details)
    # Qwen tokens: the fake generator charges 30 per attempt (10 prompt + 20 generated)
    res, ctl, gen, _ = RT.run([["WRONG:a:1"], ["WRONG:b:2"], ["WRONG:c:3"]], {"k": 1}, ctl=EvidenceLadderController(limits=big(max_qwen_tokens=40)))
    assert res.stop_cause == "budget" and res.stop_details.startswith("limit:max_qwen_tokens"), (res.stop_cause, res.stop_details)
    # a decided problem is never turned into a budget stop
    res, ctl, gen, _ = RT.run([["OK:a"]], {"k": 1}, ctl=EvidenceLadderController(limits=big(max_actions=1, max_rounds=1)))
    assert res.stop_cause == "decided"
    EvidenceLadderController()                                                                      # the default profile loads from the committed file


# ====================================================================== A4
def test_a4_unknown_or_invented_requirement_is_refused_before_it_exists():
    res, ctl, _, _ = RT.run([["OK:a"]], {"k": 1})
    run_ = ctl.last_run
    fac = Factory(run_.registry, run_.reqs, run_.env, run_.intake.roles)
    art = RT.Artifact.make("T1", "python", "OK:zzz")
    for rid in ("R99", "hidden_1", "", "R1; DROP"):
        try:
            fac.make(art, rid, "critic_experiment_v1", "critic_experiment", "model_derived", "pass", "invented", {"status": "pass"})
            assert False, rid
        except ModelEvidenceError:
            pass


def test_a4_critic_named_requirement_is_ignored_and_example_row_pass_is_capped():
    codes = ["OK:a", "OK:b"]
    exe = RT.FakeExec(exp_table={"OK:a": {"status": "pass", "observed": "2"}, "OK:b": {"status": "pass", "observed": "2"}})
    resp = {"experiment": {"kind": "stdin_input", "payload": "1"}, "implications": "x", "requirement_id": "R3"}                   # the critic names an EXAMPLE row (R3)
    cfg = {"k": 2, "critic": True, "critic_experiment": True, "critic_responses": {RT.crit_rid(codes): resp}}
    try:
        C.validate("critic_response", resp)
    except Exception:
        resp.pop("requirement_id")
    res, ctl, _, _ = RT.run([codes], cfg, critic=RT.FakeCritic(), exe=exe)
    ev = [e for e in ctl.last_journal.evidence() if e["check_id"] == "critic_experiment_v1"]
    assert ev and all(e["requirement_id"] == "R2" for e in ev), [e["requirement_id"] for e in ev]                                  # the controller's all-inputs row, never the critic's pick
    assert all(e["independence"] == "model_derived" for e in ev)
    # a critic pass that WOULD sit on a contract / example row is recorded as unknown
    run_ = ctl.last_run
    fac = Factory(run_.registry, run_.reqs, run_.env, run_.intake.roles)
    art = RT.Artifact.make("T1", "python", "OK:q")
    for rid, want in (("R1", "unknown"), ("R3", "unknown"), ("R2", "pass")):
        assert fac.make(art, rid, "critic_experiment_v1", "critic_experiment", "model_derived", "pass", "s", {"status": "pass"})["status"] == want, rid
    assert fac.make(art, "R1", "critic_experiment_v1", "critic_experiment", "model_derived", "fail", "s", {"status": "fail"})["status"] == "fail"      # it can still reject


def test_a4_critic_pass_cannot_lift_unverified_or_blocked_and_forged_records_fail_i3():
    res, ctl, _, _ = RT.run([["WRONG:a:1"]], {"k": 1, "max_rounds": 0, "mode": "single"})
    run_ = ctl.last_run
    cid = list(run_.cands)[0]
    c = run_.cands[cid]
    j, fac = ctl.last_journal, Factory(run_.registry, run_.reqs, run_.env, run_.intake.roles)
    fac.n = 100                                                                                                  # fresh evidence ids
    before = decide_stage1(run_.reqs, j.evidence(cid), cid, cid, run_.env, run_.registry)
    assert before["answer_label"] == "BLOCKED"
    for rid in ("R1", "R2", "R3", "R4"):                                                                       # critic pass everywhere, through the Factory
        j.add_evidence(fac.make(c.art, rid, "critic_experiment_v1", "critic_experiment", "model_derived", "pass", "s", {"status": "pass"}), {"status": "pass"})
    after = decide_stage1(run_.reqs, j.evidence(cid), cid, cid, run_.env, run_.registry)
    assert after["answer_label"] == "BLOCKED" and {r["requirement_id"]: r["label"] for r in after["requirement_rows"]}["R3"] in ("FAILED", "CONFLICTED")
    # UNVERIFIED stays UNVERIFIED: a lone example row with only a capped critic pass
    reg = run_.registry
    j2 = EvidenceJournal(reg)
    e = fac.make(c.art, "R3", "critic_experiment_v1", "critic_experiment", "model_derived", "pass", "s", {"status": "pass"})
    j2.add_evidence(e, {"status": "pass"})
    d = decide_stage1([r for r in run_.reqs if r["requirement_id"] == "R3"], j2.evidence(cid), cid, cid, run_.env, reg)
    assert d["answer_label"] == "UNVERIFIED" and d["requirement_rows"][0]["label"] == "UNVERIFIED"
    # a FORGED record that bypasses the Factory (status pass on a row that has no soft route) is caught by I3
    j3 = EvidenceJournal(reg)
    j3.append("requirements", {"reqs": run_.reqs})
    j3.append("routing", {"step": "x", "candidate_id": None, "ceiling": "UNVERIFIED", "table": router_mod.route(run_.reqs, run_.intake.roles, reg, [], None)})
    forged = dict(e, status="pass", evidence_id="E7777")
    j3.add_evidence(forged, {"status": "pass"})
    assert "outside the soft-support route" in inv.check_i3(j3)
    j3b = EvidenceJournal(reg)
    j3b.append("requirements", {"reqs": run_.reqs})
    j3b.add_evidence(dict(e, requirement_id="R9", evidence_id="E7778"), {"status": "unknown"})
    assert "outside the controller's list" in inv.check_i3(j3b)


# ====================================================================== A5
def expect_missing(fn, name):
    try:
        fn()
    except MissingComponentError as e:
        assert name in e.component, (name, e.component)
        return e
    raise AssertionError("no MissingComponentError for " + name)


def test_a5_agentic_missing_components_raise_or_journal_and_stop():
    w = make_run().task.workspace
    mk = lambda ctx, **t: AgenticRun(AgenticTask("t", "u `%s`" % CMD, w, t.get("cmd", CMD), t.get("protected", ["test_slug.py"]), "s"), ctx, lambda m: "")
    expect_missing(lambda: mk(AgenticContext(sandbox=None)), "sandbox")
    expect_missing(lambda: mk(AgenticContext(sandbox=FakeSandbox()), protected=[]), "protected")
    expect_missing(lambda: mk(AgenticContext(sandbox=FakeSandbox()), cmd=""), "protected")
    r = mk(AgenticContext(sandbox=FakeSandbox(), arm="E", config={"require": ["scorer"]}))
    expect_missing(r.begin, "scorer")
    assert r.journal.entries("component_missing")[0]["data"]["component"] == "scorer"          # the journal says why the run stopped
    r = mk(AgenticContext(sandbox=FakeSandbox(), arm="E", config={"require": ["critic"]}))
    expect_missing(r.begin, "critic")
    r2 = make_run(config={"require": ["critic"]}, critic=object())                                   # present at begin, gone before the critic stage
    r2.ctx.critic = None
    expect_missing(r2.critic_stage, "critic")
    r3 = make_run(arm="E")                                                                         # not required: waived, but stated in the journal (never silent)
    assert [e["data"]["waived"] for e in r3.journal.entries("component_missing")] == [True]


def test_a5_ladder_controller_missing_components_raise():
    cfgs = (({"k": 1, "critic": True, "require": ["critic"]}, "critic"), ({"k": 1, "scorer": True, "exploratory": True, "require": ["scorer"]}, "scorer"))
    for cfg, name in cfgs:
        gen, exe = RT.FakeGen([["OK:a"]]), RT.FakeExec()
        ctx = ControllerContext(gen, exe, RT.FakeBudget(arm="E"), "E", None, None, (1,), dict(cfg, mode="ladder"))
        expect_missing(lambda: EvidenceLadderController().run_problem(RT.TASK, ctx), name)
        assert not gen.calls                                                                       # it stopped BEFORE any model call
    gen, ctl = RT.FakeGen([["OK:a"]]), EvidenceLadderController()                                  # not required: the run continues, and the journal says what was missing
    ctl.run_problem(RT.TASK, ControllerContext(gen, RT.FakeExec(), RT.FakeBudget(arm="E"), "E", None, None, (1,), {"mode": "ladder", "k": 1, "critic": True, "scorer": True}))
    assert sorted(e["data"]["component"] for e in ctl.last_journal.entries("component_missing")) == ["critic", "scorer"] and all(e["data"]["waived"] for e in ctl.last_journal.entries("component_missing"))
    for ctx in (ControllerContext(None, RT.FakeExec(), RT.FakeBudget(), "D0"), ControllerContext(RT.FakeGen([]), None, RT.FakeBudget(), "D0"), ControllerContext(RT.FakeGen([]), RT.FakeExec(), None, "D0")):
        expect_missing(lambda: EvidenceLadderController().run_problem(RT.TASK, ctx), "")


def test_a5_registry_router_journal_and_invariants_need_their_parts():
    expect_missing(lambda: router_mod.route([], {}, None, []), "registry")
    expect_missing(lambda: EvidenceJournal(None).add_evidence({"evidence_id": "E1"}), "registry")
    try:
        EvidenceJournal(None).add_evidence({"evidence_id": "E1"})
    except JournalError:
        pass                                                                                      # still a JournalError for old callers
    res, ctl, _, _ = RT.run([["OK:a"]], {"k": 1})
    bare = EvidenceJournal(ctl.last_run.registry)                                                  # a journal that never routed: the router was not used
    expect_missing(lambda: inv.assert_invariants(res, bare, ControllerContext(None, None, None, "D0")), "router")


def test_a5_i1_checks_the_journaled_decision_not_a_literal():
    res, ctl, _, _ = RT.run([["OK:a"]], {"k": 1})
    j = ctl.last_journal
    assert inv.check_i1(res, j) == ""
    tampered = dict(res.decision, requirement_rows=[dict(r, label="VERIFIED") for r in res.decision["requirement_rows"]])
    res.decision = tampered
    assert inv.check_i1(res, j) != ""                                                              # label unchanged, rows rewritten: caught (a `derived_by` literal would have passed)
    res2, ctl2, _, _ = RT.run([["OK:a"]], {"k": 1})
    res2.decision = dict(res2.decision, derived_by="controller", next_step="commit")
    assert "journaled" in inv.check_i1(res2, ctl2.last_journal)


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = {}
    for name, fn in tests:
        try:
            fn()
        except Exception:      # noqa: BLE001
            failed[name] = traceback.format_exc()[-700:]
    print(json.dumps({"tests": len(tests), "passed": len(tests) - len(failed), "failed": failed}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
