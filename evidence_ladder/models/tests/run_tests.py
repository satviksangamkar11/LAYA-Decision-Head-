"""Plain-assert tests for evidence_ladder/models. Run: python3 evidence_ladder/models/tests/run_tests.py
Prints {"tests": n, "passed": n, "failed": {name: message}}; exit code 1 on any failure. Needs jsonschema (used by C.validate). No network, no inference."""
import ast
import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from evidence_ladder.common.contract_import import C  # noqa: E402
from evidence_ladder.common.interfaces import (BudgetExhausted, Generation, GenerationRequest, NotAuthorized, UnsupportedDecisionType)  # noqa: E402
from evidence_ladder.models import broker as B, glm, laya, qwen  # noqa: E402
from evidence_ladder.models.pc_sender import KeyScheduler, PcSender  # noqa: E402
from evidence_ladder.models.kaggle_keys import split_keys  # noqa: E402

GLM_FX = json.loads((MODELS / "fixtures" / "glm_responses.json").read_text(encoding="utf-8"))
QWEN_FX = json.loads((MODELS / "fixtures" / "qwen_generations.json").read_text(encoding="utf-8"))
REQ = GLM_FX["request"]
CASES = {c["name"]: c for c in GLM_FX["cases"]}
GOOD = CASES["valid_experiment"]["raw_response"]
CFG = glm.GlmConfig(reasoning_params={"TEST_ONLY_reasoning_param": "x"})


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, d):
        self.t += d


def new_broker(tmp, clock=None, **kw):
    clock = clock or FakeClock()
    kw.setdefault("config", CFG)
    return B.StagedCriticBroker(Path(tmp) / "ledger.jsonl", clock=clock.now, **kw), clock


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % exc.__name__)


# ================================================================= qwen
def test_extract_fixture_cases():
    for r in QWEN_FX["records"]:
        g = r["generation"]
        code, st = qwen.extract_code(g["text"])
        assert (code, st) == (g["code"], g["extract_status"]), (r["seed"], code, st)


def test_extract_edge_cases():
    assert qwen.extract_code("no think\n```python\nx=1\n```") == ("", "no_code")                       # thinking never closed
    t = "a\n</think>\n```python\nx=1\n```\ntext\n```python\ny=2\n```\n"
    assert qwen.extract_code(t) == ("y=2", "ok")                                                       # last block wins
    assert qwen.extract_code("</think>\n```text\nfoo\n```") == ("", "no_code")                          # wrong language tag
    assert qwen.extract_code("</think>\n```\nx=1\n```") == ("x=1", "ok")                                # untagged accepted
    assert qwen.extract_code("</think>\n```python\nx=1\n```\n```python\ny=") == ("", "no_code")         # last fence unclosed
    assert qwen.extract_code("</think>\r\n```python\r\nx=1\r\n```\r\n") == ("x=1\r".replace("\r", ""), "ok")
    assert qwen.extract_code("x</think>y</think>\n```python\nz=3\n```") == ("z=3", "ok")                # last </think>
    assert qwen.extract_code("</think>\n```java\nclass A{}\n```", "java") == ("class A{}", "ok")
    assert qwen.extract_code("</think>\n```python\n   \n```") == ("", "no_code")                       # empty block


def test_template_hashes_stable():
    assert set(qwen.TEMPLATES) == {"qwen_sample_v1", "qwen_repair_v1", "qwen_tests_v1"}
    hs = set()
    for k, v in qwen.TEMPLATES.items():
        assert qwen.template_hash(k) == hashlib.sha256(v.encode("utf-8")).hexdigest()
        hs.add(qwen.template_hash(k))
    assert len(hs) == 3
    kw = dict(task_prompt="T", examples=qwen.format_examples([{"input": "1", "expected": "2"}]), language="python")
    p1, h1 = qwen.render("qwen_sample_v1", **kw)
    p2, h2 = qwen.render("qwen_sample_v1", **kw)
    assert (p1, h1) == (p2, h2) and h1 == hashlib.sha256(p1.encode()).hexdigest()
    raises(ValueError, qwen.render, "qwen_sample_v1", task_prompt="T")
    pr, _ = qwen.render("qwen_repair_v1", candidate_id="c1", level="L3", input="1 2", expected="3", observed="4", requirement="sum", constraints="none",
                        scope="solve()", task_prompt="T", source="def f(): pass", language="python")
    assert pr.startswith("Previous candidate: c1.\nVerified failure (level L3): 1 2 -> expected 3, observed 4.\nRequirement (quoted): \"sum\".")
    assert "smallest corrected change" in pr


def test_fixture_backend_determinism_and_roundtrip():
    be = qwen.FixtureGenerationBackend(QWEN_FX["records"])
    req = GenerationRequest("SYN-01", "p", 3, [1, 2, 3], 100)
    a, b = be.generate(req), be.generate(req)
    assert a == b and [g.seed for g in a] == [1, 2, 3]
    raises(qwen.FixtureMissing, be.generate, GenerationRequest("SYN-01", "p", 1, [99], 100))
    raises(qwen.FixtureMissing, be.generate, GenerationRequest("SYN-01", "p", 1, [1], 100, purpose="repair"))
    raises(ValueError, be.generate, GenerationRequest("SYN-01", "p", 3, [1], 100))
    for g in a:
        s = qwen.generation_to_json(g)
        assert qwen.generation_to_json(qwen.generation_from_json(s)) == s and qwen.generation_from_json(s) == g
    raises(ValueError, qwen.generation_from_dict, dict(qwen.generation_to_dict(a[0]), bogus=1))
    assert be.model_info()["synthetic"] is True
    raises(ValueError, qwen.FixtureGenerationBackend, QWEN_FX["records"] + [QWEN_FX["records"][0]])


def test_hf_backend_gated():
    raises(NotAuthorized, qwen.HFQwenBackend)                                        # default: not authorized
    raises(NotAuthorized, qwen.HFQwenBackend, authorized="yes")                      # only literal True counts
    raises(NotAuthorized, qwen.HFQwenBackend, authorized=True, env={"EVIDENCE_LADDER_ALLOW_INFERENCE": "1"}, path_exists=lambda p: False)   # off Kaggle
    raises(NotAuthorized, qwen.HFQwenBackend, authorized=True, env={}, path_exists=lambda p: True)                                         # no flag
    raises(NotAuthorized, qwen.HFQwenBackend, authorized=True, env={"EVIDENCE_LADDER_ALLOW_INFERENCE": "0"}, path_exists=lambda p: True)
    ok = qwen.HFQwenBackend(authorized=True, env={"EVIDENCE_LADDER_ALLOW_INFERENCE": "1"}, path_exists=lambda p: p == "/kaggle")   # gate only; nothing loaded
    assert ok._model is None and ok.model_info()["revision"] == qwen.QWEN_REVISION
    assert qwen.SAMPLING == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def test_budget_charging():
    b = qwen.TokenBudget("P1", "A", 100)
    be = qwen.FixtureGenerationBackend(QWEN_FX["records"])
    gens = qwen.generate_and_charge(be, GenerationRequest("SYN-01", "p", 1, [1], 10), b)
    assert b.used["qwen_gen"] == 40 and b.used["qwen_prefill"] == 120
    C.validate("budget_record", b.record())
    b.charge("glm_out", 20)
    assert b.generated() == 60 and not b.exhausted()
    e = raises(BudgetExhausted, qwen.charge_generations, b, [Generation("", "", "no_code", 1, 50, 0, "m", "r", "t", "0" * 64)])
    assert b.used["qwen_gen"] == 90 + 0 or b.generated() > 100                      # tokens recorded before raising
    assert b.exhausted()
    raises(BudgetExhausted, qwen.generate_and_charge, be, GenerationRequest("SYN-01", "p", 1, [2], 10), b)
    raises(ValueError, b.charge, "bogus", 1)
    raises(ValueError, b.charge, "qwen_gen", -1)


# ================================================================= glm
def _bad(**over):
    r = json.loads(json.dumps(REQ))
    r.update(over)
    return r


def test_blinding_rejections():
    glm.check_blinding(REQ)
    bad = [
        _bad(laya_choice="cand_a"), _bad(probe_scores={"cand_a": 0.9}), _bad(scorer_pick="cand_b"), _bad(qwen_reasoning="step by step"),
        _bad(thinking_text="<think>..."), _bad(hidden_tests=[{"in": "1", "out": "2"}]), _bad(labels={"cand_a": "correct"}), _bad(random_extra="x"),
        _bad(candidate_code={"cand_a": "x = 1  # </think>", "cand_b": "y"}),
        _bad(requirement_quotes=["LAYA preferred cand_a"]),
        _bad(requirement_quotes=["see hidden tests for details"]),
        _bad(observed_outputs={"cand_a": [{"input": "1", "expected": "3"}]}),                       # expected output of an unspecified case
        _bad(observed_outputs={"cand_a": [{"input": "1", "observed": "3", "laya_rank": 1}]}),
        _bad(observed_outputs={"cand_a": [{"input": "1", "observed": "3", "mystery": 1}]}),         # not on allow-list
        _bad(observed_outputs={"zzz": []}),                                                          # not a candidate id
        _bad(candidate_code={"cand_a": "x"}),                                                        # keys != candidate_ids
    ]
    for r in bad:
        raises(glm.BlindingError, glm.check_blinding, r)
        raises(glm.BlindingError, glm.validate_request, r)


def test_request_response_schemas():
    out = glm.validate_request(REQ)
    assert glm.schema_error("critic_request", out) is None
    assert glm.schema_error("critic_request", _bad(model="other/model")) is not None
    r = json.loads(json.dumps(REQ))
    del r["question"]
    assert glm.schema_error("critic_request", r) is not None
    raises(ValueError, glm.validate_request, _bad(question="nonsense"))
    good = json.loads(GOOD["choices"][0]["message"]["content"])
    assert glm.schema_error("critic_response", good) is None
    assert glm.schema_error("critic_response", {"experiment": {"kind": "weird", "payload": "x"}, "implications": "i"}) is not None
    assert glm.schema_error("critic_response", {"experiment": {"kind": "stdin_input", "payload": "x" * 20001}, "implications": "i"}) is not None


def test_reasoning_param_pending():
    e = raises(glm.ConfigError, glm.build_payload, REQ, glm.GlmConfig())
    assert "FC-2" in str(e)
    raises(glm.ConfigError, glm.build_payload, REQ, glm.GlmConfig(reasoning_params={}))
    raises(glm.ConfigError, glm.build_payload, REQ, glm.GlmConfig(reasoning_params={"stream": False}))
    p = glm.build_payload(REQ, CFG)
    assert p["model"] == "z-ai/glm-5.3-flash" and p["stream"] is True and p["max_tokens"] >= 16384 and p["TEST_ONLY_reasoning_param"] == "x"
    assert [m["role"] for m in p["messages"]] == ["system", "user"] and "cand_a" in p["messages"][1]["content"]
    assert glm.build_payload(REQ, CFG) == p                                                          # deterministic
    for forbidden in ("laya", "scorer", "hidden"):
        assert forbidden not in p["messages"][1]["content"].lower()


def test_response_parser_fixtures():
    for name, c in CASES.items():
        if c["http_status"] != 200:
            continue
        pr = glm.parse_critic_output(c["raw_response"])
        assert pr.status == c["expect_state"], (name, pr)
        if pr.status == "malformed":
            assert pr.reason == c["expect_reason"], (name, pr.reason)
        else:
            assert glm.schema_error("critic_response", pr.response) is None
    pr = glm.parse_critic_output(CASES["valid_experiment"]["raw_response"])
    assert pr.reasoning.startswith("SYNTHETIC reasoning") and (pr.tokens_in, pr.tokens_out) == (410, 900)
    pr = glm.parse_critic_output(CASES["null_content_reasoning_only"]["raw_response"])
    assert pr.content is None and "experiment" in pr.reasoning and pr.response is None          # reasoning is never mined for the answer
    chunks = CASES["stream_chunks_valid"]["raw_response"]
    sse = "".join("data: %s\n\n" % json.dumps(c) for c in chunks) + "data: [DONE]\n\n"
    p2 = glm.parse_critic_output(sse)
    assert p2.status == "ok" and p2.response == glm.parse_critic_output(chunks).response and p2.tokens_out == 9
    assert glm.parse_critic_output("<html>502</html>").reason == "unreadable_body"
    assert glm.parse_critic_output(12345).reason == "unreadable_body"
    wrapped = glm.parse_critic_output({"headers": {}, "body": GOOD})
    assert wrapped.status == "ok"


# ================================================================= broker
def test_broker_every_fixture_classification():
    for name, c in CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            b, clk = new_broker(tmp)
            pid = b.build_packet(REQ)["packet_id"]
            b.export_pending()
            r = b.ingest(pid, c["http_status"], c["raw_response"])
            assert r["state"] == c["expect_state"], (name, r)
            if "expect_stop" in c:
                assert b.stop_all_critic_calls is c["expect_stop"], name
            if "expect_backoff_s" in c:
                assert r["backoff_s"] == c["expect_backoff_s"], (name, r)
            if c["expect_state"] == "ok":
                assert glm.schema_error("critic_response", r["response"]) is None


def test_broker_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        b, _ = new_broker(tmp)
        p1 = b.build_packet(REQ)
        p2 = b.build_packet(json.loads(json.dumps(REQ)))
        assert p1["packet_id"] == p2["packet_id"] == B.packet_id_of(glm.validate_request(REQ)) and p1["new"] and not p2["new"]
        assert len(b.export_pending()) == 1 and b.export_pending() == []                            # exported once, never re-listed
        r1 = b.ingest(p1["packet_id"], 200, GOOD)
        r2 = b.ingest(p1["packet_id"], 200, GOOD)
        r3 = b.ingest(p1["packet_id"], 500, {"body": "late"})
        assert r1["state"] == "ok" and r2["state"] == "duplicate" and r3["state"] == "duplicate" and r2["response"] == r1["response"]
        assert b.counters["ingested"] == 1 and b.counters["duplicates"] == 2 and b.counters["tokens_out"] == 900
        assert b.export_pending() == [] and b.export_pending(include_exported=True) == []           # completed ids never resent
        raises(KeyError, b.ingest, "0" * 64, 200, GOOD)


def test_broker_budget_charge_once():
    with tempfile.TemporaryDirectory() as tmp:
        bud = qwen.TokenBudget("P", "D1", 10000)
        b, _ = new_broker(tmp, budget=bud)
        pid = b.build_packet(REQ)["packet_id"]
        b.ingest(pid, 200, GOOD)
        b.ingest(pid, 200, GOOD)
        assert bud.used["glm_in"] == 410 and bud.used["glm_out"] == 900


def test_broker_bounded_retries_and_backoff_schedule():
    with tempfile.TemporaryDirectory() as tmp:
        b, clk = new_broker(tmp)
        pid = b.build_packet(REQ)["packet_id"]
        seen = []
        for _ in range(3):
            b.export_pending(include_exported=True)
            r = b.ingest(pid, 500, {"body": "x"})
            assert r["state"] == "retry"
            seen.append(r["backoff_s"])
        assert seen == [2.0, 4.0, 8.0]
        r = b.ingest(pid, 503, {"body": "x"})
        assert r["state"] == "unavailable" and r["reason"].startswith("retries_exhausted")
        assert b.ingest(pid, 200, GOOD)["state"] == "duplicate" and b.counters["ingested"] == 4
        assert b.stop_all_critic_calls is False                                                      # exhaustion alone is not a spend stop
    with tempfile.TemporaryDirectory() as tmp:
        b, _ = new_broker(tmp, max_retries=1)
        pid = b.build_packet(REQ)["packet_id"]
        assert b.ingest(pid, None, {"error": "timeout"})["state"] == "retry"
        assert b.ingest(pid, None, {"error": "timeout"})["state"] == "unavailable"


def test_broker_retry_after_honoured():
    with tempfile.TemporaryDirectory() as tmp:
        b, clk = new_broker(tmp)
        pid = b.build_packet(REQ)["packet_id"]
        assert len(b.export_pending()) == 1
        r = b.ingest(pid, 429, CASES["rate_limited_retry_after"]["raw_response"])
        assert r["state"] == "retry" and r["backoff_s"] == 17.0 and r["not_before"] == clk.t + 17.0
        clk.t += 16.0
        assert b.export_pending() == [] and b.pending_summary()["waiting_until"] == [1017.0]
        clk.t += 1.0
        ex = b.export_pending()
        assert len(ex) == 1 and ex[0]["attempt"] == 2
        r = b.ingest(pid, 429, CASES["rate_limited_no_header"]["raw_response"])
        assert r["backoff_s"] == 60.0
        pid2 = b.build_packet(dict(REQ, request_id="other"))["packet_id"]
        assert b.ingest(pid2, 429, {"headers": {"retry-after": "999999"}})["backoff_s"] == 3600.0   # capped


def test_broker_payment_stops_everything():
    for name in ("payment_required_402", "forbidden_payment_text_403", "quota_exceeded_text"):
        with tempfile.TemporaryDirectory() as tmp:
            b, _ = new_broker(tmp)
            pid = b.build_packet(REQ)["packet_id"]
            other = b.build_packet(dict(REQ, request_id="second"))["packet_id"]
            c = CASES[name]
            r = b.ingest(pid, c["http_status"], c["raw_response"])
            assert r["state"] == "unavailable" and r["stop_all_critic_calls"] is True and b.stop_all_critic_calls
            raises(B.CriticStopped, b.build_packet, dict(REQ, request_id="third"))
            raises(B.CriticStopped, b.export_pending)
            b2 = B.StagedCriticBroker(Path(tmp) / "ledger.jsonl", config=CFG)                        # replay keeps the stop
            assert b2.stop_all_critic_calls
            raises(B.CriticStopped, b2.export_pending)
            assert other in b2.packets


def test_broker_malformed_state():
    with tempfile.TemporaryDirectory() as tmp:
        b, _ = new_broker(tmp)
        pid = b.build_packet(REQ)["packet_id"]
        r = b.ingest(pid, 200, CASES["malformed_json"]["raw_response"])
        assert r["state"] == "malformed" and r["response"] is None and r["reason"] == "not_json"
        assert b.export_pending(include_exported=True) == [] and b.counters["malformed"] == 1
        assert b.ingest(pid, 200, GOOD)["state"] == "duplicate"                                      # never retried with the same request id


def test_broker_ledger_append_only_and_keyless():
    with tempfile.TemporaryDirectory() as tmp:
        b, _ = new_broker(tmp)
        path = Path(tmp) / "ledger.jsonl"
        pid = b.build_packet(REQ)["packet_id"]
        b.export_pending()
        before = path.read_text(encoding="utf-8")
        b.log_send(pid, "k2")
        b.ingest(pid, 500, {"body": {"error": "oops key nv" + "api-ABCDEF123456 Authorization: Bearer abc.def-123"}})
        b.ingest(pid, 200, GOOD)
        after = path.read_text(encoding="utf-8")
        assert after.startswith(before) and len(after) > len(before)                                 # append-only
        lines = [json.loads(x) for x in after.splitlines()]
        assert [x["seq"] for x in lines] == list(range(1, len(lines) + 1))
        assert "nv" + "api-ABCDEF" not in after and "abc.def-123" not in after and "[REDACTED]" in after
        cfg = lines[0]
        assert cfg["event"] == "config" and cfg["limits"] == {"max_in_flight_per_key": 10, "max_sends_per_key_per_minute": 39, "rate_limit_cooldown_s": 60}
        b2 = B.StagedCriticBroker(path, config=CFG)                                                  # replay restores state without writing
        assert b2.packets[pid]["state"] == b.packets[pid]["state"] and path.read_text(encoding="utf-8") == after
        assert b2.counters == b.counters


def test_broker_completed_and_replay():
    with tempfile.TemporaryDirectory() as tmp:
        b, _ = new_broker(tmp)
        assert b.ledger_path == Path(tmp) / "ledger.jsonl"
        ids = {n: b.build_packet(dict(REQ, request_id=n))["packet_id"] for n in ("ok", "bad", "dead", "wait", "fresh")}
        assert b.build_packet(dict(REQ, request_id="ok"))["packet_id"] == ids["ok"]                  # controller re-builds, same id
        assert all(b.completed(i) is None for i in ids.values()) and b.completed("0" * 64) is None   # nothing ingested / unknown
        r_ok = b.ingest(ids["ok"], 200, GOOD)
        b.ingest(ids["bad"], 200, CASES["malformed_json"]["raw_response"])
        b.ingest(ids["dead"], 402, CASES["payment_required_402"]["raw_response"])                   # also sets stop_all
        assert b.stop_all_critic_calls
        b2 = B.StagedCriticBroker(Path(tmp) / "ledger.jsonl", config=CFG)
        b3 = B.StagedCriticBroker(Path(tmp) / "ledger.jsonl", config=CFG)
        for x in (b, b2, b3):
            assert x.completed(ids["ok"]) == r_ok["response"] and glm.schema_error("critic_response", x.completed(ids["ok"])) is None
            for n in ("bad", "dead", "wait", "fresh"):
                assert x.completed(ids[n]) is None, n
        assert b2.ingest(ids["ok"], 200, {"other": 1})["state"] == "duplicate"                       # duplicate does not change the answer
        assert b2.completed(ids["ok"]) == r_ok["response"]
    with tempfile.TemporaryDirectory() as tmp:                                                       # retry state is not completed
        b, _ = new_broker(tmp)
        pid = b.build_packet(REQ)["packet_id"]
        b.ingest(pid, 500, {"body": "x"})
        assert b.completed(pid) is None
        b.ingest(pid, 200, GOOD)
        assert b.completed(pid) is not None and B.StagedCriticBroker(b.ledger_path, config=CFG).completed(pid) == b.completed(pid)


# ================================================================= pc sender
def test_scheduler_limits_fake_clock():
    s = KeyScheduler(["k1"])
    got = [s.acquire(0.0) for _ in range(11)]
    assert got == ["k1"] * 10 + [None]                                                              # 10 in flight
    s.release("k1", 0.0)
    assert s.acquire(0.0) == "k1"
    s2 = KeyScheduler(["k1"])
    for _ in range(39):
        assert s2.acquire(0.0) == "k1"
        s2.release("k1", 0.0)
    assert s2.acquire(0.0) is None and s2.acquire(59.9) is None                                      # 39 per minute
    assert s2.next_ready_time(10.0) == 60.0
    assert s2.acquire(60.0) == "k1"
    s3 = KeyScheduler(["a", "b"])
    s3.acquire(0.0)
    s3.release("a", 1.0, rate_limited=True)
    assert s3.free_slots("a", 30.0) == 0 and s3.free_slots("a", 61.0) > 0                            # cools 60 s
    s3.release("b", 2.0, rate_limited=True, retry_after=120)
    assert s3.free_slots("b", 100.0) == 0 and s3.free_slots("b", 122.0) > 0


def _loaded_sender(tmp, n_packets, keys, executor=None):
    clk = FakeClock()
    b = B.StagedCriticBroker(Path(tmp) / "ledger.jsonl", config=CFG, clock=clk.now)
    for i in range(n_packets):
        b.build_packet(dict(REQ, request_id="r%d" % i))
    return b, clk, PcSender(b, KeyScheduler(keys), clock=clk.now, sleep=clk.sleep, executor=executor)


def _max_window(times):
    times = sorted(times)
    return max((sum(1 for u in times if t <= u < t + 60) for t in times), default=0)


def test_sender_respects_per_minute_limit():
    with tempfile.TemporaryDirectory() as tmp:
        b, clk, sender = _loaded_sender(tmp, 100, ["k1", "k2"])
        sent = []

        def transport(p, k):
            sent.append((clk.t, k))
            return 200, GOOD
        res = sender.send_pending(transport)
        assert res["sent"] == 100 and res["states"] == {"ok": 100} and not res["stopped"]
        for k in ("k1", "k2"):
            ts = [t for t, kk in sent if kk == k]
            assert ts and _max_window(ts) <= 39, (k, _max_window(ts))
        assert len({json.dumps(p) for p in b.packets}) == 100 and clk.t > 1000.0                     # had to wait for the window


def test_sender_in_flight_wave_limit():
    class Rec:
        def __init__(self):
            self.waves = []

        def map(self, fn, items):
            items = list(items)
            self.waves.append([k for _, k in items])
            return [fn(i) for i in items]
    with tempfile.TemporaryDirectory() as tmp:
        ex = Rec()
        b, clk, sender = _loaded_sender(tmp, 25, ["k1", "k2"], executor=ex)
        res = sender.send_pending(lambda p, k: (200, GOOD))
        assert res["sent"] == 25 and ex.waves and len(ex.waves) >= 2                                  # 2 keys x 10 per wave: 25 needs 2+ waves
        for w in ex.waves:
            assert all(w.count(k) <= 10 for k in set(w)), w


def test_sender_rate_limit_cools_key_and_requeues():
    with tempfile.TemporaryDirectory() as tmp:
        b, clk, sender = _loaded_sender(tmp, 20, ["k1", "k2"])
        sent, hit = [], []

        def transport(p, k):
            sent.append((clk.t, k))
            if not hit and k == "k1":
                hit.append(clk.t)
                return 429, {"headers": {"Retry-After": "17"}, "body": {"error": "Too Many Requests"}}
            return 200, GOOD
        res = sender.send_pending(transport)
        assert res["states"].get("ok") == 20 and res["states"].get("retry") == 1 and res["sent"] == 21
        t0 = hit[0]
        assert not [t for t, k in sent if k == "k1" and t0 < t < t0 + 60]                           # k1 silent for 60 s
        assert len({p for p, v in b.packets.items() if v["state"] == "ok"}) == 20


def test_sender_stops_on_payment_and_survives_transport_errors():
    with tempfile.TemporaryDirectory() as tmp:
        b, clk, sender = _loaded_sender(tmp, 30, ["k1"])
        n = []

        def transport(p, k):
            n.append(1)
            return 402, {"body": {"error": "Payment Required"}}
        res = sender.send_pending(transport)
        assert res["stopped"] and len(n) <= 10 and b.stop_all_critic_calls                            # one wave of up to 10 in flight
    with tempfile.TemporaryDirectory() as tmp:
        b, clk, sender = _loaded_sender(tmp, 1, ["k1"])
        calls = []

        def flaky(p, k):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("t")
            if len(calls) == 2:
                raise RuntimeError("boom")
            return 200, GOOD
        res = sender.send_pending(flaky)
        assert res["states"] == {"retry": 2, "ok": 1} and len(calls) == 3


# ================================================================= laya
def _fake_repo(tmp):
    root = Path(tmp)
    (root / "tools").mkdir()
    (root / "decision_head").mkdir()
    (root / "tools" / "fable_capture.py").write_bytes(b"cap")
    (root / "decision_head" / "fable_features.py").write_bytes(b"feat")
    man = {"pretrained_head": {"sha256": "h"}, "adapters": {"final-s0": {"sha256": "a"}, "final-s1": {"sha256": "b"}, "final-s2": {"sha256": "c"}},
           "linear_probe": {"code": "x.py (sha256 " + "7" * 64 + ")"}}
    (root / "m.json").write_text(json.dumps(man), encoding="utf-8")
    exp = tuple({"path": p, "sha256": hashlib.sha256(d).hexdigest()} for p, d in (("tools/fable_capture.py", b"cap"), ("decision_head/fable_features.py", b"feat")))
    return root, root / "m.json", exp


def test_laya_decision_type_gate_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        root, man, exp = _fake_repo(tmp)
        s = laya.LayaScorer(repo_root=root, manifest_path=man, expected_capture=exp)
        d = s.scorer_def
        C.validate("scorer_def", d)
        assert d["status"] == "validated" and d["decision_types_validated"] == ["next_action"] and not s.disabled
        c = d["compatibility"]
        assert c["qwen_revision"] == "768f209d9ea81521153ed38c47d515654e938aea" and c["dtype"] == "fp16" and c["depths"] == [18, 24, 30]
        assert (c["max_options"], c["prefix_token_limit"], c["option_token_limit"]) == (8, 24000, 4096)
        assert c["capture_code_hash"] == ";".join(e["sha256"] for e in exp)
        raises(UnsupportedDecisionType, s.score, "complete_program", [])                             # S3-T2
        raises(UnsupportedDecisionType, s.score, "tool_call", [], False)
        raises((NotAuthorized, NotImplementedError), s.score, "complete_program", [], True)           # exploratory: allowed, but no forward pass exists
        raises(NotAuthorized, s.score, "next_action", [1, 2])
        raises(NotImplementedError, laya.LayaScorer(repo_root=root, manifest_path=man, expected_capture=exp, authorized=True).score, "next_action", [])


def test_laya_disabled_on_capture_hash_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        root, man, exp = _fake_repo(tmp)
        (root / "tools" / "fable_capture.py").write_bytes(b"cap CHANGED")                            # S3-T5
        s = laya.LayaScorer(repo_root=root, manifest_path=man, expected_capture=exp)
        C.validate("scorer_def", s.scorer_def)
        assert s.disabled and s.scorer_def["status"] == "disabled" and "fable_capture.py" in s.meta.disabled_reason
        assert s.meta.capture_report["tools/fable_capture.py"]["match"] is False
        raises(laya.ScorerDisabled, s.score, "next_action", [])
        raises(laya.ScorerDisabled, s.score, "complete_program", [], True)                           # exploratory does not override disabled
        assert not s.allowed("next_action", True)
    with tempfile.TemporaryDirectory() as tmp:
        s = laya.LayaScorer(repo_root=tmp, manifest_path=Path(tmp) / "nope.json")                    # nothing present: disabled, never a crash
        assert s.disabled
        C.validate("scorer_def", s.scorer_def)


def test_probe_and_random_head_interfaces():
    with tempfile.TemporaryDirectory() as tmp:
        root, man, exp = _fake_repo(tmp)
        for cls, sid in ((laya.ProbeScorer, "probe_v1"), (laya.RandomHeadScorer, "random_head_v1"), (laya.LayaScorer, "laya_adapter_v1")):
            s = cls(repo_root=root, manifest_path=man, expected_capture=exp)
            assert s.scorer_id == sid and callable(s.score)
            C.validate("scorer_def", s.scorer_def)
            raises(NotAuthorized, s.score, "next_action", [1])
            raises(UnsupportedDecisionType, s.score, "complete_program", [1])
        p = laya.ProbeScorer(repo_root=root, manifest_path=man, expected_capture=exp)
        assert p.scorer_def["weights_hash"].startswith("recipe:" + "7" * 64)
        assert laya.RandomHeadScorer(repo_root=root, manifest_path=man, expected_capture=exp).scorer_def["status"] == "transfer_test"
        # the three share one compatibility block (same Qwen features)
        assert p.scorer_def["compatibility"] == laya.LayaScorer(repo_root=root, manifest_path=man, expected_capture=exp).scorer_def["compatibility"]


# ================================================================= kaggle keys
def test_kaggle_keys_split():
    a, b, c = "a" * 3 + "1", "b" * 3 + "2", "c" * 3 + "3"
    text = "# note\n" + a + "\n\n  " + b + " , " + a + "\n" + c + "\n"
    assert split_keys(text) == {"key_1": a, "key_2": b, "key_3": c}                                  # comma + newline, blanks and # skipped, dup dropped
    raises(ValueError, split_keys, "# only\n\n")
    raises(ValueError, split_keys, "")
    bad = "a" * 3 + " 1"
    e = raises(ValueError, split_keys, "ok" + "1" * 3 + "\n" + bad)
    assert "line 2" in str(e) and bad not in str(e) and "aaa" not in str(e)                          # message never carries key text


# ================================================================= static
def test_static_no_keys_no_network():
    banned = [".sec" + "rets", "nvidia" + "_keys"]
    net = {"requests", "urllib", "urllib3", "socket", "http", "aiohttp", "httpx", "ssl", "ftplib", "smtplib"}
    files = [p for p in MODELS.rglob("*.py")]
    assert len(files) >= 7
    for p in files:
        src = p.read_text(encoding="utf-8")
        for b in banned:
            assert b not in src, "%s mentions %s" % (p.name, b)
        for node in ast.walk(ast.parse(src)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for m in mods:
                assert m.split(".")[0] not in net, "%s imports %s" % (p.name, m)


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
