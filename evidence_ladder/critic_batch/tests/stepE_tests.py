"""[LAYA-INT-1 step E] Plain-assert tests: shared experiment allow-list (prompt, parser, executor), explicit rejection of unsupported/unsafe experiments, controller critic_stage with a mock
broker and a fake sandbox, I1-I8, I8 blinding of the packet. No network, no keys. Run: python evidence_ladder/critic_batch/tests/stepE_tests.py   (prints {"tests", "passed", "failed"}; exit 1 on failure)"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask  # noqa: E402
from evidence_ladder.critic_batch import validator as V  # noqa: E402
from evidence_ladder.models import experiments as EXP, glm  # noqa: E402
from evidence_ladder.models.broker import StagedCriticBroker  # noqa: E402

TESTSRC = 'from slug import slugify\n\nassert slugify("Hello, World!") == "hello-world"\nprint("all tests passed")\n'
SLUG = 'import re\n\n\ndef slugify(text):\n    text = text.strip().lower()\n    return re.sub(r"[^a-z0-9]+", "-", text)\n\n\ndef _private():\n    pass\n'
CMD = "python3 test_slug.py"
CFG = glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=EXP.SUPPORTED_KINDS)


def ws():
    d = Path(tempfile.mkdtemp(prefix="stepE-")) / "ws"
    d.mkdir(parents=True)
    (d / "slug.py").write_text(SLUG, "utf-8")
    (d / "test_slug.py").write_text(TESTSRC, "utf-8")
    return d


def rejects(exp, code, w=None, protected=("test_slug.py",)):
    try:
        EXP.plan(exp, w or ws(), protected, CMD)
    except EXP.Rejected as e:
        assert e.code == code, (exp, e.code, e.detail)
        return e
    raise AssertionError("not rejected: %r" % (exp,))


# ------------------------------------------------------------------ shared allow-list
def test_prompt_parser_executor_share_one_allowlist():
    req = {"request_id": "r", "model": glm.GLM_MODEL, "decision_id": "d", "candidate_ids": ["c"], "requirement_quotes": ["q"], "candidate_code": {"c": "x"},
           "observed_outputs": {"c": [{"input": "i", "observed": "o", "status": "fail"}]}, "question": "challenge_requirement"}
    user = glm.build_messages(req, CFG.allowed_kinds)[1]["content"]
    for k in EXP.SUPPORTED_KINDS:
        assert "- %s:" % k in user
    for k in set(EXP.SCHEMA_KINDS) - set(EXP.SUPPORTED_KINDS):
        assert k not in user, k                                               # the prompt never offers a kind the executor cannot run
    old = glm.build_messages(req)[1]["content"]
    assert "stdin_input" in old and "command" not in old                       # default (Stage-1) prompt unchanged
    for k in EXP.SUPPORTED_KINDS:
        raw = {"choices": [{"message": {"content": json.dumps({"experiment": {"kind": k, "payload": "x"}, "implications": "i"})}}]}
        assert glm.parse_critic_output(raw, CFG.allowed_kinds).status == "ok"
    for k in set(EXP.SCHEMA_KINDS) - set(EXP.SUPPORTED_KINDS):
        raw = {"choices": [{"message": {"content": json.dumps({"experiment": {"kind": k, "payload": "x"}, "implications": "i"})}}]}
        p = glm.parse_critic_output(raw, CFG.allowed_kinds)
        assert p.status == "malformed" and p.reason == "unsupported_kind", k
    assert glm.parse_critic_output({"choices": [{"message": {"content": json.dumps({"experiment": {"kind": "command", "payload": "x"}, "implications": "i"})}}]}).reason == "unsupported_kind"


# ------------------------------------------------------------------ supported kinds
def test_supported_function_call_and_commands_plan():
    w = ws()
    p = EXP.plan({"kind": "function_call", "payload": 'slugify("  A--B  ")'}, w, ["test_slug.py"], CMD)
    assert p["kind"] == "function_call" and p["cmd"].startswith("python3 -c ") and "slugify" in p["cmd"]
    p = EXP.plan({"kind": "function_call", "payload": "slugify(text='x y', )"}, w, ["test_slug.py"], CMD)
    assert "text" in p["cmd"]
    assert EXP.plan({"kind": "command", "payload": CMD}, w, ["test_slug.py"], CMD)["cmd"] == CMD
    assert EXP.plan({"kind": "command", "payload": "python3 slug.py a b"}, w, ["test_slug.py"], CMD)["cmd"] == "python3 slug.py a b"


def test_judge_statuses():
    run = dict(exit_code=0, stdout="'a-b'\n", stderr="", killed=False)
    assert EXP.judge("a-b", run)[0] == "pass" and EXP.judge("x", run)[0] == "fail" and EXP.judge(None, run)[0] == "unknown"
    assert EXP.judge("a-b", dict(run, killed=True))[0] == "unknown" and EXP.judge("a-b", dict(run, exit_code=1))[0] == "fail"


# ------------------------------------------------------------------ rejected
def test_unsupported_kinds_rejected():
    for k in ("stdin_input", "state_query", "rm_rf", ""):
        rejects({"kind": k, "payload": "x"}, "unsupported_kind")
    rejects("nope", "malformed_experiment")
    rejects({"kind": "command", "payload": 5}, "malformed_experiment")


def test_unsafe_function_calls_rejected():
    for pl, code in (("__import__('os').system('ls')", "unsafe_payload"), ("slugify(open('x'))", "unsafe_payload"), ("slugify(*['a'])", "unsafe_payload"), ("slugify(**{'text': 'a'})", "unsafe_payload"),
                     ("slugify('a'); import os", "unsafe_payload"), ("slugify('a')\nimport os", "unsafe_payload"), ("os.system('x')", "unsafe_payload"), ("_private()", "unknown_function"),
                     ("x = 1", "unsafe_payload"), ("lambda: 1", "unsafe_payload"), ("slugify(1 if 1 else 2)", "unsafe_payload"), ("a" * 2001, "unsafe_payload")):
        rejects({"kind": "function_call", "payload": pl}, code)
    w = ws()
    (w / "helper.py").write_text("def secret_fn():\n    return 1\n", "utf-8")
    EXP.plan({"kind": "function_call", "payload": "secret_fn()"}, w, ["test_slug.py"], CMD)       # a normal non-test module function is fine
    rejects({"kind": "function_call", "payload": "slugify('a')"}, "unknown_function", w=w, protected=("test_slug.py", "slug.py"))      # functions of protected files are not callable


def test_unsafe_commands_rejected():
    cases = (("python3 slug.py; ls", "shell_metacharacter"), ("python3 slug.py && ls", "shell_metacharacter"), ("python3 slug.py | cat", "shell_metacharacter"), ("python3 slug.py > x", "shell_metacharacter"),
             ("python3 slug.py $(id)", "shell_metacharacter"), ("python3 slug.py `id`", "shell_metacharacter"), ("python3 slug.py\nls", "shell_metacharacter"), ("echo 'a'", "shell_metacharacter"),
             ("cat test_slug.py", "unsupported_command"), ("curl http://x.y", "unsupported_command"), ("rm -rf .", "unsupported_command"), ("python3 -c x", "unsupported_command"),
             ("python3 test_slug.py x", "protected_path"), ("python3 hidden_tests.py", "unsafe_path_or_network"), ("python3 ../x.py", "unsupported_command"), ("python3 /etc/x.py", "unsupported_command"),
             ("python3 slug.py http://a.b", "unsafe_path_or_network"), ("python3 slug.py a/b", "unsafe_payload"), ("python3 nofile.py", "unknown_file"), ("", "unsafe_payload"))
    for pl, code in cases:
        rejects({"kind": "command", "payload": pl}, code)


# ------------------------------------------------------------------ controller
class FakeSandbox:
    def __init__(self, test_exit=1):
        self.calls, self.test_exit = [], test_exit

    def run(self, workspace, cmd, persist=False, timeout=60):
        self.calls.append(cmd)
        if cmd == CMD:
            return dict(exit_code=self.test_exit, killed=False, stdout="all tests passed\n" if not self.test_exit else "", stderr="" if not self.test_exit else "AssertionError", protected_unchanged=True)
        return dict(exit_code=0, killed=False, stdout="'a-b'\n", stderr="", protected_unchanged=True)


def make_run(test_exit=1):
    w = ws()
    sb = FakeSandbox(test_exit)
    task = AgenticTask(task_id="t", user_text="Fix slug.py so that `%s` passes. Do not modify test_slug.py." % CMD, workspace=w, test_cmd=CMD, protected=["test_slug.py"], system_text="sys", max_steps=1)
    r = AgenticRun(task, AgenticContext(sandbox=sb, arm="E", config={}), lambda m: "ctx")
    r.begin()
    r.step([json.dumps({"name": "Read", "arguments": {"file_path": "slug.py"}})], {})
    return r, sb


def broker_with(r, exp_or_raw, status=200):
    b = StagedCriticBroker(Path(tempfile.mkdtemp(prefix="stepE-b-")) / "ledger.jsonl", config=CFG, max_retries=0)
    pid = b.build_packet(r.critic_request())["packet_id"]
    b.export_pending()
    raw = {"choices": [{"message": {"content": exp_or_raw if isinstance(exp_or_raw, str) else json.dumps(exp_or_raw)}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    b.ingest(pid, status, raw)
    r.ctx.critic = b
    return b, pid


def finish_ok(r):
    res = r.finish()
    assert len(r.invariants_checked) == 8, r.invariants_checked          # I1-I8 all checked and passing (finish raises InvariantViolation otherwise)
    return res


def crit_entries(r):
    return [e["data"] for e in r.journal.entries("critic")]


def test_controller_executes_supported_experiment_and_label_unchanged():
    r, sb = make_run()
    before = r.steps[-1]["label"]
    sid = r.last_state
    broker_with(r, {"experiment": {"kind": "function_call", "payload": 'slugify("  A--B  ")', "expected_by_candidate": {sid: "a-b"}}, "implications": "i"})
    assert r.critic_stage() == "applied"
    assert r.critic_result["executed"] and r.critic_result["experiment_status"] == "pass" and any("slugify" in c for c in sb.calls)
    ev = [e for e in r.journal.evidence(sid) if e["check_id"] == "critic_experiment_v1"]
    assert len(ev) == 1 and ev[0]["independence"] == "model_derived"
    assert finish_ok(r).answer_label == before                               # the critic never moves a label
    assert crit_entries(r)[-1]["state"] == "applied" and crit_entries(r)[-1]["label_unchanged_by_critic"] is True


def test_controller_runs_exact_test_command():
    r, sb = make_run()
    broker_with(r, {"experiment": {"kind": "command", "payload": CMD}, "implications": "i"})
    n = len(sb.calls)
    assert r.critic_stage() == "applied" and sb.calls[n] == CMD and r.critic_result["experiment_status"] == "unknown"
    finish_ok(r)


def test_controller_rejects_unsafe_experiment_explicitly_and_never_runs_it():
    r, sb = make_run()
    broker_with(r, {"experiment": {"kind": "command", "payload": "python3 slug.py; cat ../.secrets/x"}, "implications": "i"})
    n = len(sb.calls)
    assert r.critic_stage() == "rejected" and len(sb.calls) == n
    e = crit_entries(r)[-1]
    assert e["state"] == "rejected" and e["executed"] is False and e["reason_code"] == "shell_metacharacter" and e["reason"]
    assert not [x for x in r.journal.evidence(r.last_state) if x["check_id"] == "critic_experiment_v1"]
    finish_ok(r)


def test_controller_rejects_unsupported_kind_explicitly():
    r, sb = make_run()
    broker_with(r, {"experiment": {"kind": "stdin_input", "payload": "abc"}, "implications": "i"})      # parser already marks it malformed/unsupported_kind
    n = len(sb.calls)
    assert r.critic_stage() == "rejected" and len(sb.calls) == n
    e = crit_entries(r)[-1]
    assert e["state"] == "rejected" and e["executed"] is False and "unsupported_kind" in e["reason"]
    finish_ok(r)


def test_controller_handles_malformed_and_unavailable():
    r, sb = make_run()
    broker_with(r, "this is not json at all")
    assert r.critic_stage() == "rejected" and "not_json" in crit_entries(r)[-1]["reason"]
    finish_ok(r)
    r, sb = make_run()
    broker_with(r, "", status=200)
    assert r.critic_stage() == "rejected" and "empty_content" in crit_entries(r)[-1]["reason"]
    r, sb = make_run()
    broker_with(r, "x", status=402)
    assert r.critic_stage() in ("unavailable", "rejected") and crit_entries(r)[-1]["state"] != "applied"
    r, sb = make_run()
    r.ctx.critic = StagedCriticBroker(Path(tempfile.mkdtemp()) / "l.jsonl", config=CFG)
    assert r.critic_stage() == "pending" and crit_entries(r)[-1]["state"] == "pending"


def test_i8_packet_is_blinded():
    r, _ = make_run()
    req = r.critic_request()
    text = json.dumps(req)
    V.validate_packet({"packet_id": StagedCriticBroker(Path(tempfile.mkdtemp()) / "l.jsonl", config=CFG).build_packet(req)["packet_id"], "request": req})
    assert "assert slugify" not in text and "all tests passed" not in text.split("observed_outputs")[0]      # no protected test source in the candidate code
    for w in ("hidden", "laya", "think", "reasoning"):
        assert w not in text.lower(), w
    assert set(req) <= glm.TOP_KEYS


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = {}
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failed[name] = "%s: %s | %s" % (type(e).__name__, e, traceback.format_exc().strip().splitlines()[-3:][0].strip())
    print(json.dumps({"tests": len(tests), "passed": len(tests) - len(failed), "failed": failed}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
