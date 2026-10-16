"""Plain-assert tests for evidence_ladder/critic_batch (Step 4). Run on Kaggle (standing rule): python3 evidence_ladder/critic_batch/tests/run_tests.py
Synthetic packets, mocked transport, no network. Prints {"tests": n, "passed": n, "failed": {...}}; exit 1 on any failure."""
import ast
import copy
import io
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
from email.message import Message
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from evidence_ladder.critic_batch import batch as BT, transport as TR, validator as V  # noqa: E402
from evidence_ladder.critic_batch.config import ConcurrencyConfigError, load_concurrency, scheduler_from_config  # noqa: E402
from evidence_ladder.models import broker as B, glm  # noqa: E402
from evidence_ladder.models._util import canonical_json  # noqa: E402
from evidence_ladder.models.broker import packet_id_of  # noqa: E402
from evidence_ladder.models.pc_sender import KeyScheduler  # noqa: E402

FX = json.loads((ROOT / "evidence_ladder" / "models" / "fixtures" / "glm_responses.json").read_text(encoding="utf-8"))
REQ = FX["request"]
CASES = {c["name"]: c for c in FX["cases"]}
GOOD = CASES["valid_experiment"]["raw_response"]
CC = load_concurrency()
FAKEKEY = "FAKEKEY" + "z" * 24          # not credential-shaped on purpose


class Crash(BaseException):
    """Simulated process death: not an Exception, so nothing in the sender swallows it."""


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, d):
        self.t += d


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % exc.__name__)


def mk_packets(tmp, n, name="pk"):
    reqs = {}
    for i in range(n):
        r = dict(REQ, request_id="r%d" % i)
        reqs[packet_id_of(r)] = r
    V.write_packets(Path(tmp) / name, reqs)
    return Path(tmp) / name, sorted(reqs)


def raw_packet_dir(tmp, requests, name="rawpk"):
    """Write a packet dir WITHOUT validation (what a hostile or buggy exporter could produce)."""
    d = Path(tmp) / name
    d.mkdir(parents=True)
    data = ("\n".join(canonical_json({"packet_id": packet_id_of(r), "request": r}) for r in requests) + "\n").encode("utf-8")
    (d / "packets.jsonl").write_bytes(data)
    (d / "PACKETS_MANIFEST.json").write_text(json.dumps({"bundle": V.PACKET_BUNDLE, "count": len(requests), "packets_sha256": V._sha(data)}))
    return d


def mk_runner(tmp, transport, n=6, keys=("key_1", "key_2"), packets=None, out="out", **kw):
    pk = packets or mk_packets(tmp, n)[0]
    clk = kw.pop("clk", None) or FakeClock()
    return BT.BatchRunner(pk, Path(tmp) / out, transport, list(keys), CC, clock=clk.now, sleep=clk.sleep, poll_s=0.01, **kw), clk


def events(out, kind=None):
    evs = [json.loads(x) for x in (Path(out) / "progress.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [e for e in evs if kind is None or e["event"] == kind]


def bundle_records(out):
    return [json.loads(x) for x in (Path(out) / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


# ================================================================= CP1: validator
FORBIDDEN_CASES = {
    "hidden_tests": lambda r: r.update(requirement_quotes=["The hidden tests check overflow."]),
    "protected_cases": lambda r: r.update(requirement_quotes=["protected case 4 uses a large n"]),
    "reference_solution": lambda r: r["candidate_code"].update(cand_a="# reference solution\nprint(1)"),
    "private_labels": lambda r: r.update(requirement_quotes=["private label: cand_b is right"]),
    "laya": lambda r: r.update(requirement_quotes=["LAYA picked cand_b"]),
    "qwen_private_reasoning": lambda r: r.update(requirement_quotes=["<think>I think cand_a</think>"]),
    "secret": [lambda r: r.update(requirement_quotes=["nvapi" + "-" + "A" * 30]),
               lambda r: r.update(requirement_quotes=["hf" + "_" + "B" * 30]),
               lambda r: r.update(requirement_quotes=["KGAT" + "_" + "C" * 30]),
               lambda r: r.update(requirement_quotes=["sk" + "-" + "D" * 30]),
               lambda r: r.update(requirement_quotes=["Authorization: Bearer " + "E" * 30])],
}
FORBIDDEN_KEYS = {"hidden_tests": "hidden_tests", "protected_cases": "protected_cases", "reference_solution": "reference_solution",
                  "private_labels": "private_labels", "laya": "laya_choice"}


def test_validator_accepts_fixture():
    assert V.validate_packet({"packet_id": packet_id_of(REQ), "request": REQ}) == json.loads(canonical_json(REQ))


def test_validator_rejects_every_forbidden_content_class():
    for cls, muts in FORBIDDEN_CASES.items():
        for m in (muts if isinstance(muts, list) else [muts]):
            r = copy.deepcopy(REQ)
            m(r)
            e = raises(V.PacketRejected, V.validate_packet, {"packet_id": packet_id_of(r), "request": r})
            assert e.cls == cls, (cls, e.cls)
            raises(V.PacketRejected, V.write_packets, tempfile.mkdtemp(), {packet_id_of(r): r})      # the PC export side refuses too


def test_validator_rejects_forbidden_field_names():
    for cls, key in FORBIDDEN_KEYS.items():
        r = dict(copy.deepcopy(REQ), **{key: ["x"]})
        e = raises(V.PacketRejected, V.validate_packet, {"packet_id": packet_id_of(r), "request": r})
        assert e.cls == cls, (key, e.cls)
    for key in ("gold_label", "oracle", "extra_notes", "scores"):
        r = dict(copy.deepcopy(REQ), **{key: "x"})
        assert raises(V.PacketRejected, V.validate_packet, {"packet_id": packet_id_of(r), "request": r}).cls in ("blinding", "laya")
    r = copy.deepcopy(REQ)
    r["observed_outputs"]["cand_a"][0]["verdict_from_grader"] = "pass"          # unknown key inside observed_outputs
    assert raises(V.PacketRejected, V.validate_packet, {"packet_id": packet_id_of(r), "request": r}).cls == "blinding"


def test_validator_rejects_malformed_and_unexpected_lines():
    pid = packet_id_of(REQ)
    assert raises(V.PacketRejected, V.validate_packet, {"packet_id": pid, "request": REQ, "label": "A"}).cls == "unexpected_fields"
    assert raises(V.PacketRejected, V.validate_packet, {"request": REQ}).cls == "unexpected_fields"
    assert raises(V.PacketRejected, V.validate_packet, ["x"]).cls == "unexpected_fields"
    assert raises(V.PacketRejected, V.validate_packet, {"packet_id": 5, "request": REQ}).cls == "malformed"
    assert raises(V.PacketRejected, V.validate_packet, {"packet_id": "0" * 64, "request": REQ}).cls == "id_mismatch"
    bad = dict(REQ, question="free_text")
    assert raises(V.PacketRejected, V.validate_packet, {"packet_id": packet_id_of(bad), "request": bad}).cls == "schema"


def test_packets_roundtrip_manifest_and_tamper():
    with tempfile.TemporaryDirectory() as tmp:
        d, ids = mk_packets(tmp, 5)
        acc, rej = V.read_packets(d)
        assert [p for p, _ in acc] == ids and rej == []
        man = json.loads((d / "PACKETS_MANIFEST.json").read_text())
        assert man["count"] == 5 and man["bundle"] == V.PACKET_BUNDLE
        data = (d / "packets.jsonl").read_bytes()
        (d / "packets.jsonl").write_bytes(data.replace(b"r1", b"r9"))
        raises(V.ManifestError, V.read_packets, d)
        raises(V.ManifestError, V.read_packets, Path(tmp) / "nowhere")


def test_read_packets_rejects_and_records_bad_lines_keeps_good():
    with tempfile.TemporaryDirectory() as tmp:
        bad_hidden = dict(REQ, request_id="h1", requirement_quotes=["see hidden tests"])
        bad_think = dict(REQ, request_id="h2", requirement_quotes=["<think>x</think>"])
        good = dict(REQ, request_id="g1")
        d = raw_packet_dir(tmp, [good, bad_hidden, bad_think, good])                   # also a duplicate of `good`
        acc, rej = V.read_packets(d)
        assert [p for p, _ in acc] == [packet_id_of(good)]
        assert sorted(x["class"] for x in rej) == ["duplicate_packet", "hidden_tests", "qwen_private_reasoning"], rej
        assert all(x["packet_id"] for x in rej)


def test_static_kernel_metadata_and_dataset_dir():
    ds = "someone/evidence-ladder-critic-packets-v1"
    ok = {"dataset_sources": [ds], "kernel_sources": [], "competition_sources": [], "model_sources": [], "enable_gpu": False, "enable_internet": True}
    V.check_kernel_metadata(ok, ds)
    for k, v in [("dataset_sources", [ds, "someone/evidence-ladder-tests-bundle-v1"]), ("dataset_sources", []), ("kernel_sources", ["a/b"]),
                 ("model_sources", ["m"]), ("enable_gpu", True), ("enable_internet", False)]:
        raises(ValueError, V.check_kernel_metadata, dict(ok, **{k: v}), ds)
    raises(ValueError, V.check_kernel_metadata, dict(ok, dataset_sources=["someone/evidence-ladder-tests-bundle-v1"]), "someone/evidence-ladder-tests-bundle-v1")
    meta = json.loads((ROOT / "kaggle" / "evidence_ladder_critic_v2" / "kernel-metadata.json").read_text()) if (ROOT / "kaggle" / "evidence_ladder_critic_v2").is_dir() else None
    if meta:                                                                    # present only where the kernel directory travels with the code
        V.check_kernel_metadata(meta, "sangeetasangamkar/evidence-ladder-critic-packets-v1")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for rel in ["packets.jsonl", "PACKETS_MANIFEST.json", "CODE_MANIFEST.json", "dataset-metadata.json", "code/evidence_ladder/a.py", "code/configs/x.toml"]:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text("x")
        assert V.check_dataset_dir(d) == []
        for rel in ["hidden_cases.json", "code/tests/protected_cases.json", "code/data.bin", "extra.txt", "code/worker/ref.py", "code/.secrets/k.py"]:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text("x")
        assert sorted(V.check_dataset_dir(d)) == sorted(["hidden_cases.json", "code/tests/protected_cases.json", "code/data.bin", "extra.txt", "code/worker/ref.py", "code/.secrets/k.py"])


# ================================================================= concurrency reconciliation
def test_concurrency_config_reads_file_and_agrees_with_broker():
    cc = load_concurrency()
    assert cc["limits"] == B.LIMITS and cc["limits"]["max_in_flight_per_key"] == 10
    r = cc["ramp"]
    assert r["start_level"] == 2 and r["floor_level"] == 1 and r["start_level"] * 14 < 140
    s = scheduler_from_config(["a", "b"], cc)
    assert s.level == {"a": 2, "b": 2} and s.max_in_flight == 10


def test_concurrency_config_refuses_missing_or_disagreeing():
    raises(ConcurrencyConfigError, load_concurrency, Path("/nonexistent/critic_concurrency_v1.toml"))
    text = (ROOT / "configs" / "critic_concurrency_v1.toml").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        for old, new in [("max_in_flight_per_key = 10", "max_in_flight_per_key = 5"), ("max_sends_per_key_per_minute = 39", "max_sends_per_key_per_minute = 40"),
                         ("rate_limit_cooldown_s = 60", "rate_limit_cooldown_s = 30"), ("start_level = 2", "start_level = 11"), ("[ramp]", "[nothing]")]:
            assert old in text
            p = Path(tmp) / "c.toml"
            p.write_text(text.replace(old, new, 1), encoding="utf-8")
            raises(ConcurrencyConfigError, load_concurrency, p)
        d2 = Path(tmp) / "alongside"
        d2.mkdir()
        (d2 / "critic_concurrency_v1.toml").write_text(text, encoding="utf-8")
        (d2 / "thresholds_evidence_ladder_v1.toml").write_text("tampered = true\n", encoding="utf-8")        # frozen file present but different hash
        raises(ConcurrencyConfigError, load_concurrency, d2 / "critic_concurrency_v1.toml")
    # the runner has no other source of limits: it needs a loaded config dict
    raises(TypeError, BT.BatchRunner, "a", "b", None, ["k"])


def test_ramp_raises_slowly_and_drops_on_429():
    s = scheduler_from_config(["a"], CC)
    t = 0.0
    assert [s.acquire(t) for _ in range(3)] == ["a", "a", None]                    # starts at 2, not 10
    s.release("a", t)
    s.release("a", t)

    def cycle(rl=False):
        nonlocal t
        t += 61.0
        assert s.acquire(t) == "a"
        s.release("a", t, rate_limited=rl)
    for _ in range(5):
        cycle()
    assert s.level["a"] == 3                                                        # +1 after 5 clean completions
    for _ in range(35):
        cycle()
    assert s.level["a"] == 10                                                       # capped at the user rule
    for _ in range(10):
        cycle()
    assert s.level["a"] == 10
    t += 61.0
    s.acquire(t)
    s.release("a", t, rate_limited=True)
    assert s.level["a"] == 5 and s.free_slots("a", t + 30) == 0 and s.free_slots("a", t + 61) > 0     # halves, cools 60 s
    for _ in range(5):
        cycle()
    assert s.level["a"] == 6                                                        # one 429 in the last 20 (0.05 < 0.10): may rise again
    t += 61.0
    s.acquire(t)
    s.release("a", t, rate_limited=True)
    assert s.level["a"] == 3
    for _ in range(5):
        cycle()
    assert s.level["a"] == 3                                                        # two 429s in the window (0.10 is not below 0.10): held
    for _ in range(20):
        cycle()
    assert s.level["a"] > 3                                                         # once they age out it rises again


def test_ramp_default_scheduler_unchanged_without_ramp():
    s = KeyScheduler(["k"])
    assert [s.acquire(0.0) for _ in range(11)] == ["k"] * 10 + [None]


# ================================================================= CP2/CP4: the batch loop
def test_first_burst_is_start_level_not_cap():
    with tempfile.TemporaryDirectory() as tmp:
        st = {"in": 0, "peak": 0}
        lock = threading.Lock()
        gate = threading.Event()

        def transport(p, k):
            with lock:
                st["in"] += 1
                st["peak"] = max(st["peak"], st["in"])
            gate.wait(10)
            with lock:
                st["in"] -= 1
            return 200, GOOD
        runner, _ = mk_runner(tmp, transport, n=30, keys=["key_%d" % i for i in range(1, 15)])      # 14 keys
        th = threading.Thread(target=runner.run)
        th.start()
        time.sleep(0.6)
        assert st["in"] == 28, st                                                   # 14 keys x start level 2, not 140
        gate.set()
        th.join(30)
        assert not th.is_alive() and st["peak"] <= 28 + 14 * 2


def test_streaming_no_wave_barrier_and_out_of_order_completion():
    with tempfile.TemporaryDirectory() as tmp:
        st = {"n": 0, "in": 0, "peak": 0, "others_first": None}
        lock = threading.Lock()
        box = {}

        def transport(p, k):
            with lock:
                st["n"] += 1
                a = st["n"]
                st["in"] += 1
                st["peak"] = max(st["peak"], st["in"])
            try:
                if a == 1:                                                          # the first call is the slowest
                    st["slow"] = p["packet_id"]
                    end = time.time() + 15
                    while time.time() < end and sum(1 for e in list(box["r"].ledger.events) if e["event"] == "result") < 5:
                        time.sleep(0.01)
                    st["others_first"] = sum(1 for e in list(box["r"].ledger.events) if e["event"] == "result") >= 5
                else:
                    time.sleep(0.01 * a)
                return 200, GOOD
            finally:
                with lock:
                    st["in"] -= 1
        runner, _ = mk_runner(tmp, transport, n=6)
        box["r"] = runner
        res = runner.run()
        assert st["others_first"] is True, "5 other answers were not persisted while the slowest call was still in flight (wave barrier?)"
        assert st["peak"] <= 4 and res["summary"]["states"] == {"ok": 6}
        sends = [e["packet_id"] for e in events(Path(tmp) / "out", "send")]
        results = [e["packet_id"] for e in events(Path(tmp) / "out", "result")]
        assert st["slow"] == results[-1] and len(results) == 6 and len(sends) == 6    # the slowest call is answered last
        assert all(e["seq"] for e in events(Path(tmp) / "out"))


def test_429_cooldown_requeue_and_ramp_drop():
    with tempfile.TemporaryDirectory() as tmp:
        st = {"hit": None, "k1_sends": [], "n1": 0}
        lock = threading.Lock()
        clk = FakeClock()

        def transport(p, k):
            with lock:
                if k == "key_1":
                    st["k1_sends"].append(clk.now())
                    st["n1"] += 1
                    if st["n1"] == 1:
                        st["hit"] = clk.now()
                        return 429, {"headers": {"Retry-After": "17"}, "body": {"error": "Too Many Requests"}}
            return 200, GOOD
        runner, _ = mk_runner(tmp, transport, n=10, clk=clk)
        res = runner.run()
        assert res["summary"]["states"].get("ok") == 10 and res["summary"]["states"].get("retry") == 1
        evs = events(Path(tmp) / "out")
        hit = next(e for e in evs if e["event"] == "result" and e["http_status"] == 429)
        late = [e for e in evs if e["event"] == "send" and e["key_label"] == "key_1" and e["seq"] > hit["seq"]]
        assert all(e["ts"] >= hit["ts"] + 60 for e in late), "key_1 sent during its cooldown"
        ing = [e for e in events(Path(tmp) / "out", "ingested") if e.get("state") == "retry"]
        assert len(ing) == 1 and ing[0]["key_label"] == "key_1" and ing[0]["level"] == 1          # level 2 halved to 1
        assert all(r["state"] == "ok" for r in bundle_records(Path(tmp) / "out"))                # the 429 packet was re-queued and finished


def test_402_stops_everything_and_restart_sends_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        st = {"n": 0}
        lock = threading.Lock()

        def transport(p, k):
            with lock:
                st["n"] += 1
                a = st["n"]
            if a == 3:
                return 402, {"body": {"error": "Payment Required"}}
            time.sleep(0.02)
            return 200, GOOD
        runner, _ = mk_runner(tmp, transport, n=12)
        res = runner.run()
        assert res["summary"]["stopped"] is True and res["manifest"]["stop_all_critic_calls"] is True
        sent = st["n"]
        assert sent < 12 and res["manifest"]["states"].get("not_completed", 0) >= 1
        assert "unavailable" in res["manifest"]["states"]
        n2 = []
        runner2, _ = mk_runner(tmp, lambda p, k: n2.append(1) or (200, GOOD), n=12)
        res2 = runner2.run()
        assert n2 == [] and res2["manifest"]["stop_all_critic_calls"] is True


def test_timeout_connection_loss_retry_then_ok():
    with tempfile.TemporaryDirectory() as tmp:
        seen = {}
        lock = threading.Lock()

        def transport(p, k):
            with lock:
                seen[p["packet_id"]] = seen.get(p["packet_id"], 0) + 1
                first = len(seen) <= 2 and seen[p["packet_id"]] == 1
                which = len(seen)
            if first and which == 1:
                raise TimeoutError("read timed out")
            if first and which == 2:
                raise ConnectionResetError("peer reset")
            return 200, GOOD
        runner, _ = mk_runner(tmp, transport, n=4)
        res = runner.run()
        assert res["summary"]["states"].get("ok") == 4 and res["summary"]["states"].get("retry") == 2
        errs = [e for e in events(Path(tmp) / "out", "result") if e["http_status"] is None]
        assert sorted(e["raw"]["error"] for e in errs) == ["timeout", "transport_exception"]
        assert [r["attempts_failed"] for r in bundle_records(Path(tmp) / "out")].count(1) == 2


def test_exhausted_server_errors_become_unavailable_and_malformed_is_recorded_not_retried():
    with tempfile.TemporaryDirectory() as tmp:
        d, ids = mk_packets(tmp, 3)
        calls = {}

        def transport(p, k):
            pid = p["packet_id"]
            calls[pid] = calls.get(pid, 0) + 1
            if pid == ids[0]:
                return 500, CASES["server_error_500"]["raw_response"]
            if pid == ids[1]:
                return 200, CASES["malformed_json"]["raw_response"]
            return 200, GOOD
        runner, _ = mk_runner(tmp, transport, packets=d)
        res = runner.run()
        recs = {r["packet_id"]: r for r in bundle_records(Path(tmp) / "out")}
        assert recs[ids[0]]["state"] == "unavailable" and recs[ids[0]]["reason"].startswith("retries_exhausted") and calls[ids[0]] == 4     # 1 + 3 retries
        assert recs[ids[1]]["state"] == "malformed" and recs[ids[1]]["reason"] == "not_json" and calls[ids[1]] == 1 and recs[ids[1]]["response"] is None
        assert recs[ids[2]]["state"] == "ok" and recs[ids[2]]["response"]["experiment"]["kind"] == "stdin_input"
        assert recs[ids[2]]["usage"] == {"prompt_tokens": 410, "completion_tokens": 900} and recs[ids[2]]["model"] == glm.GLM_MODEL and recs[ids[2]]["attempt"] == 1
        assert res["manifest"]["states"] == {"malformed": 1, "ok": 1, "unavailable": 1}


def test_crash_restart_marks_uncertain_resends_once_then_gives_up():
    with tempfile.TemporaryDirectory() as tmp:
        d, ids = mk_packets(tmp, 6)
        out = Path(tmp) / "out"
        sent1, lock = [], threading.Lock()

        def t1(p, k):
            with lock:
                sent1.append(p["packet_id"])
                a = len(sent1)
            if a == 3:
                raise Crash()
            time.sleep(0.02)
            return 200, GOOD
        r1, _ = mk_runner(tmp, t1, packets=d)
        try:
            r1.run()
            raise AssertionError("the crash must propagate")
        except Crash:
            pass
        done1 = {e["packet_id"] for e in events(out, "ingested")}
        uncertain1 = set(sent1) - done1
        assert uncertain1                                                           # at least the crashed call
        sent2 = []
        crash_all = {"on": True}

        def t2(p, k):
            sent2.append(p["packet_id"])
            if crash_all["on"]:
                raise Crash()
            return 200, GOOD
        r2, _ = mk_runner(tmp, t2, packets=d)
        try:
            r2.run()
            raise AssertionError("the second crash must propagate")
        except Crash:
            pass
        assert not (set(sent2) & done1), "a packet completed before the crash was resent"
        assert uncertain1 <= set(sent2), "uncertain packets must be resent once"
        assert {e["packet_id"] for e in events(out, "uncertain")} >= uncertain1
        # third start: packets uncertain for the SECOND time are not sent again
        sent3 = []
        r3, _ = mk_runner(tmp, lambda p, k: sent3.append(p["packet_id"]) or (200, GOOD), packets=d)
        res3 = r3.run()
        twice = {p for p in ids if sum(1 for e in events(out, "uncertain") if e["packet_id"] == p) >= 2}
        assert twice and not (set(sent3) & twice) and not (set(sent3) & done1)
        recs = {r["packet_id"]: r for r in bundle_records(out)}
        assert all(recs[p]["state"] == "unavailable" for p in twice)
        assert all(recs[p]["state"] == "ok" for p in ids if p not in twice), {p: recs[p]["state"] for p in ids}
        assert len(recs) == 6 and res3["recovered"].get("uncertain_exhausted", 0) == len(twice)


def test_progress_ledger_fsync_torn_line_and_no_credentials():
    with tempfile.TemporaryDirectory() as tmp:
        real, n = os.fsync, {"c": 0}

        def counting(fd):
            n["c"] += 1
            return real(fd)
        os.fsync = counting
        try:
            def transport(p, k):
                return 200, dict(GOOD, leaked_header="Bearer " + "Q" * 30)
            runner, _ = mk_runner(tmp, transport, n=3)
            runner.run()
        finally:
            os.fsync = real
        evs = events(Path(tmp) / "out")
        assert n["c"] >= len(evs)                                                   # one fsync per progress line
        text = (Path(tmp) / "out" / "progress.jsonl").read_text(encoding="utf-8") + (Path(tmp) / "out" / "broker_ledger.jsonl").read_text(encoding="utf-8")
        assert "Q" * 30 not in text and "Bearer Q" not in text
        p = Path(tmp) / "out" / "progress.jsonl"
        good_lines = len(p.read_text(encoding="utf-8").splitlines())
        with open(p, "a", encoding="utf-8") as f:
            f.write('{"event": "send", "packet_id": "torn')                         # crash mid-write
        led = BT.ProgressLedger(p)
        assert len(led.events) == good_lines and len(p.read_text(encoding="utf-8").splitlines()) == good_lines
        led.append({"event": "x"})
        assert json.loads(p.read_text(encoding="utf-8").splitlines()[-1])["seq"] == good_lines + 1


def test_leaking_packets_are_rejected_recorded_and_never_sent():
    with tempfile.TemporaryDirectory() as tmp:
        good = dict(REQ, request_id="g")
        leaks = [dict(REQ, request_id="l%d" % i) for i in range(3)]
        leaks[0]["requirement_quotes"] = ["protected case 2 expects 42"]
        leaks[1]["candidate_code"] = {"cand_a": "# <think>use cand_a</think>", "cand_b": "x"}
        leaks[2]["reference_solution"] = "print(3)"
        d = raw_packet_dir(tmp, [good] + leaks)
        sent = []
        runner, _ = mk_runner(tmp, lambda p, k: sent.append(p["packet_id"]) or (200, GOOD), packets=d)
        res = runner.run()
        assert sent == [packet_id_of(good)]
        rej = json.loads((Path(tmp) / "out" / "rejected.json").read_text())
        assert sorted(x["class"] for x in rej) == ["protected_cases", "qwen_private_reasoning", "reference_solution"]
        assert res["manifest"]["rejected"] == rej and res["manifest"]["states"] == {"ok": 1}


# ================================================================= CP3: bundle, PC import, full replay
def pc_flow(tmp, n, transport, clk=None):
    clk = clk or FakeClock()
    pc = B.StagedCriticBroker(Path(tmp) / "pc_ledger.jsonl", config=glm.GlmConfig(reasoning_params=BT.REASONING), clock=clk.now)
    for i in range(n):
        pc.build_packet(dict(REQ, request_id="p%d" % i))
    man = BT.export_for_kernel(pc, Path(tmp) / "pk")
    assert man["count"] == n
    runner, _ = mk_runner(tmp, transport, packets=Path(tmp) / "pk", clk=clk)
    res = runner.run()
    return pc, Path(tmp) / "out", res


def test_full_replay_request_to_kernel_to_bundle_to_pc_ingestion():
    with tempfile.TemporaryDirectory() as tmp:
        lock, seen = threading.Lock(), {}

        def transport(p, k):
            with lock:
                seen[p["packet_id"]] = seen.get(p["packet_id"], 0) + 1
                order = list(seen).index(p["packet_id"])
                n = seen[p["packet_id"]]
            if order == 0 and n == 1:
                return 429, CASES["rate_limited_retry_after"]["raw_response"]
            if order == 1:
                return 200, CASES["malformed_json"]["raw_response"]
            if order == 2:
                return 500, CASES["server_error_500"]["raw_response"]
            return 200, GOOD
        pc, out, res = pc_flow(tmp, 6, transport)
        assert res["manifest"]["states"] == {"malformed": 1, "ok": 4, "unavailable": 1}
        recs = bundle_records(out)
        assert [r["packet_id"] for r in recs] == sorted(r["packet_id"] for r in recs)                     # sorted
        man = json.loads((out / "MANIFEST.json").read_text())
        assert man["responses_sha256"] == V._sha((out / "responses.jsonl").read_bytes()) and man["bundle"] == V.RESPONSE_BUNDLE
        imp = BT.import_bundle(out, pc)
        assert len(imp["ingested"]) == 6 and imp["rejected"] == [] and imp["mismatched_states"] == []
        for r in recs:
            assert pc.packets[r["packet_id"]]["state"] == r["state"]
            if r["state"] == "ok":
                assert pc.completed(r["packet_id"]) == r["response"]
            else:
                assert pc.completed(r["packet_id"]) is None
        assert pc.terminal_state(next(r["packet_id"] for r in recs if r["state"] == "malformed")) == "malformed"
        again = BT.import_bundle(out, pc)                                                               # duplicates
        assert again["ingested"] == [] and sorted(x["class"] for x in again["rejected"]) == ["duplicate_already_terminal"] * 6
        assert pc.counters["duplicates"] == 6
        # determinism: rewriting the bundle from the same state gives the same hash
        assert runner_hash(out) == man["responses_sha256"]


def runner_hash(out):
    return V._sha((Path(out) / "responses.jsonl").read_bytes())


def _rehash(out):
    m = json.loads((out / "MANIFEST.json").read_text())
    m["responses_sha256"] = V._sha((out / "responses.jsonl").read_bytes())
    (out / "MANIFEST.json").write_text(json.dumps(m, sort_keys=True))


def test_importer_rejects_duplicates_mismatches_unknown_and_tampered_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        pc, out, _ = pc_flow(tmp, 4, lambda p, k: (200, GOOD))
        lines = (out / "responses.jsonl").read_text(encoding="utf-8").splitlines()
        recs = [json.loads(x) for x in lines]
        # hash tamper without re-hashing the manifest
        (out / "responses.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        raises(V.ManifestError, BT.import_bundle, out, pc)
        # crafted bundle: duplicate record, packet-id/hash mismatch, unknown packet, not completed
        dup = lines[0]
        mism = dict(recs[1], request_sha256="0" * 64)
        unknown = dict(recs[2], packet_id="f" * 64, request_sha256="f" * 64)
        notdone = dict(recs[3], state="not_completed")
        (out / "responses.jsonl").write_text("\n".join([lines[0], dup, canonical_json(mism), canonical_json(unknown), canonical_json(notdone)]) + "\n", encoding="utf-8")
        _rehash(out)
        imp = BT.import_bundle(out, pc)
        assert [x["packet_id"] for x in imp["ingested"]] == [recs[0]["packet_id"]]
        assert sorted(x["class"] for x in imp["rejected"]) == ["duplicate_in_bundle", "id_hash_mismatch", "not_completed", "unknown_packet"]
        assert pc.packets[recs[1]["packet_id"]]["state"] != "ok" and pc.packets[recs[3]["packet_id"]]["state"] != "ok"
        assert pc.max_retries == 3                                                                    # restored after the import


def test_importer_402_in_bundle_stops_pc_broker():
    with tempfile.TemporaryDirectory() as tmp:
        pc, out, res = pc_flow(tmp, 5, lambda p, k: (402, CASES["payment_required_402"]["raw_response"]))
        assert res["manifest"]["stop_all_critic_calls"] is True
        BT.import_bundle(out, pc)
        assert pc.stop_all_critic_calls is True


# ================================================================= transport (mocked opener)
class FakeResp:
    def __init__(self, lines, status=200):
        self.status, self._lines = status, lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._lines)


def sse(*events_, done=True):
    out = [("data: " + json.dumps(e) + "\n").encode() for e in events_]
    return out + ([b"data: [DONE]\n"] if done else [])


GOOD_SSE = sse({"choices": [{"delta": {"reasoning_content": "think "}}]}, {"choices": [{"delta": {"content": GOOD["choices"][0]["message"]["content"]}, "finish_reason": "stop"}]},
               {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 9}})
PACKET = {"packet_id": "p", "endpoint": TR.ENDPOINT, "payload": {"model": glm.GLM_MODEL, "messages": [], "stream": True}, "attempt": 1}


def test_transport_streams_parses_and_hides_the_key():
    seen = {}

    def opener(req, timeout):
        seen["auth"], seen["timeout"], seen["body"] = req.get_header("Authorization"), timeout, json.loads(req.data)
        return FakeResp(GOOD_SSE)
    t = TR.make_transport({"key_1": FAKEKEY}, opener=opener, read_timeout_s=33.0)
    status, raw = t(PACKET, "key_1")
    assert status == 200 and seen["auth"] == "Bearer " + FAKEKEY and seen["timeout"] == 33.0 and seen["body"]["stream"] is True
    assert FAKEKEY not in json.dumps(raw)
    p = glm.parse_critic_output(raw)
    assert p.status == "ok" and p.tokens_in == 7 and p.tokens_out == 9 and "think" in p.reasoning


def test_transport_http_errors_keep_retry_after_and_never_leak_the_key():
    def mk(code, ra=None):
        h = Message()
        if ra:
            h["Retry-After"] = ra
        h["X-Other"] = "1"

        def opener(req, timeout):
            raise urllib.error.HTTPError(TR.ENDPOINT, code, "x", h, io.BytesIO(b"slow down"))
        return TR.make_transport({"key_1": FAKEKEY}, opener=opener)
    s, raw = mk(429, "17")(PACKET, "key_1")
    assert s == 429 and B.retry_after_s(raw) == 17.0 and "X-Other" not in raw["headers"] and FAKEKEY not in json.dumps(raw)
    s, raw = mk(402)(PACKET, "key_1")
    assert s == 402 and B.StagedCriticBroker._classify(None, 402, raw)[0] == "stop"


def test_transport_failures_raise_and_truncation_total_timeout_stream_error():
    def boom(exc):
        def opener(req, timeout):
            raise exc
        return TR.make_transport({"key_1": FAKEKEY}, opener=opener)
    for exc in (TimeoutError("t"), ConnectionResetError("r"), urllib.error.URLError("dns")):
        raises(type(exc), boom(exc), PACKET, "key_1")
    cut = TR.make_transport({"key_1": FAKEKEY}, opener=lambda r, timeout: FakeResp(sse({"choices": [{"delta": {"content": "{"}}]}, done=False)))
    assert cut(PACKET, "key_1") == (None, {"error": "truncated_stream"})
    err = TR.make_transport({"key_1": FAKEKEY}, opener=lambda r, timeout: FakeResp(sse({"error": {"message": "overloaded"}})))
    assert err(PACKET, "key_1")[0] == 502
    tick = {"t": 0.0}

    def clock():
        tick["t"] += 100.0
        return tick["t"]
    slow = TR.make_transport({"key_1": FAKEKEY}, opener=lambda r, timeout: FakeResp(GOOD_SSE), total_timeout_s=250.0, clock=clock)
    raises(TimeoutError, slow, PACKET, "key_1")


def test_transport_through_the_sender_maps_exceptions_to_retries():
    with tempfile.TemporaryDirectory() as tmp:
        calls = {"n": 0}

        def opener(req, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionResetError("x")
            return FakeResp(GOOD_SSE)
        runner, _ = mk_runner(tmp, TR.make_transport({"key_1": FAKEKEY, "key_2": FAKEKEY + "2"}, opener=opener), n=3)
        res = runner.run()
        assert res["summary"]["states"] == {"ok": 3, "retry": 1}
        assert FAKEKEY not in (Path(tmp) / "out" / "progress.jsonl").read_text(encoding="utf-8")


# ================================================================= static
def test_static_no_key_literals_and_network_only_in_transport():
    net = {"requests", "urllib", "urllib3", "socket", "http", "aiohttp", "httpx", "ssl", "ftplib", "smtplib"}
    banned = [".sec" + "rets", "nvidia" + "_keys", "KAGGLE_" + "KEY"]
    pkg = ROOT / "evidence_ladder" / "critic_batch"
    for p in pkg.glob("*.py"):
        src = p.read_text(encoding="utf-8")
        for b in banned:
            assert b not in src, (p.name, b)
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        if p.name != "transport.py":
            assert not (mods & net), (p.name, mods & net)
    assert "print(" not in (pkg / "transport.py").read_text(encoding="utf-8") and "logging" not in (pkg / "transport.py").read_text(encoding="utf-8")


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = {}
    for name, fn in tests:
        try:
            fn()
        except BaseException as e:
            failed[name] = "%s: %s | %s" % (type(e).__name__, e, traceback.format_exc().strip().splitlines()[-3:][0].strip())
    print(json.dumps({"tests": len(tests), "passed": len(tests) - len(failed), "failed": failed}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
