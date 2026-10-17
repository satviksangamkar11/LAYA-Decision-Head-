"""[EL-ARCH-1 T12 / T13 / T26] CPU-only tests of the plumbing of tools/el_arch1_l2_local.py with LABELLED STUBS (scripted generation text, in-process execution): they prove the harness wiring and the
ladder's L2 / clustering / discrimination behaviour, NOT the 0.6B model. The real-model half is the script itself, run by the coordinator. A test that needs WSL is named *_wsl and is skipped if WSL
does not answer. Run from the repo root: python evidence_ladder/controller/tests/phaseF1b_tests.py   (prints {"tests","passed","failed","skipped"})."""
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_tests as RT                                                                  # noqa: E402  controller fakes (FakeGen, FakeCritic)
import el_arch1_l2_local as H                                                           # noqa: E402
from evidence_ladder.models import qwen as Q                                            # noqa: E402

TESTS, SKIPPED = {}, {}


class Skip(Exception):
    pass


def test(f):
    TESTS[f.__name__] = f
    return f


class LocalExec(H.WslFunctionExec):
    """LOGIC-ONLY: runs the candidate in-process instead of the WSL sandbox (same result protocol)."""
    def __init__(self):
        super().__init__(Path(tempfile.gettempdir()), sandbox=object())

    def _execute(self, source, inputs):
        ns = {}
        try:
            exec(source, ns)
        except BaseException as e:      # noqa: BLE001
            return {"load_error": repr(e)}
        res = []
        for i in inputs:
            try:
                res.append({"ok": True, "value": ns["solve"](json.loads(json.dumps(i)))})
            except BaseException as e:      # noqa: BLE001
                res.append({"ok": False, "err": repr(e)})
        self.calls += 1
        return {"results": res}


class StubCritic(RT.FakeCritic):
    pass


# Two clamp candidates that agree on every public example and on the L2 check and differ only at x == 0 (a derived boundary input); both contain comparisons and arithmetic so mutants exist.
LO_FIRST = "def solve(data):\n    x, lo, hi = data['x'], data['lo'], data['hi']\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x + 0\n"
HI_FIRST = "def solve(data):\n    x, lo, hi = data['x'], data['lo'], data['hi']\n    if x == 0:\n        return 1\n    if x > hi:\n        return hi\n    if x < lo:\n        return lo\n    return x * 1\n"
QUOTE = "limit x to the range [lo, hi]"
GOOD_CHECK = {"input": {"x": 3, "lo": 0, "hi": 5}, "expected": 3, "quote": QUOTE}          # hits the `return x` path: the arithmetic mutants fail it
VACUOUS_CHECK = {"input": {"x": -100, "lo": 0, "hi": 5}, "expected": 0, "quote": QUOTE}    # hits `return lo`: no mutant changes it
CKEY = ("ex2_clamp", "clamp")


def checks_text(*checks):
    return "t</think>" + json.dumps(list(checks))


def harness(script, out, critic=None, exe=None, final=False, **kw):
    gen = RT.FakeGen(script)
    return H.run_harness(out, gen, exe or LocalExec(), critic, final=final, examples=[CKEY], **kw), gen


def tmp_out():
    return Path(tempfile.mkdtemp()) / "run"


# ------------------------------------------------------------------ task derivation
@test
def f1b_tasks_are_derived_from_the_driver_source():
    for name, key in H.EXAMPLES:
        t = H.derive_task(name, key)
        assert len(t.public_tests) == 3 and t.language == "python" and "solve(data)" in t.prompt and all(isinstance(x["input"], dict) for x in t.public_tests)
        assert "test_" not in t.prompt.split("Current implementation")[0]                  # the protected test file is not named to the model
    c = H.derive_task("ex2_clamp", "clamp")
    assert c.public_tests[0] == {"input": {"x": 5, "lo": 0, "hi": 10}, "expected": 5} and QUOTE in c.prompt
    try:
        H.derive_task("bad", "slug", {"slug": {"files": {"slug.py": "def f(x):\n    return x\n", "test_slug.py": "print(1)\n"}, "user": "u"}})
        assert False
    except ValueError:
        pass


# ------------------------------------------------------------------ T12 / T13 / T26 positives
@test
def f1b_valid_l2_check_two_clusters_and_discrimination_triggered():
    out = tmp_out()
    rec, gen = harness([[LO_FIRST, HI_FIRST], [LO_FIRST, HI_FIRST], [checks_text(GOOD_CHECK)]], out, critic=StubCritic())
    r = rec["tasks"]["ex2_clamp"]
    assert r["l2"]["verdict"] == "VALIDATED" and r["l2"]["validated"] == 1, r["l2"]                                           # T12 positive
    assert r["clusters"]["public_clusters"] == 1 and r["clusters"]["extended_clusters"] == 2, r["clusters"]                    # T13 positive: equal on the examples, split by derived inputs
    assert r["discrimination"]["verdict"] == "TRIGGERED" and r["discrimination"]["critic"] in ("pending", "applied"), r["discrimination"]      # T26 positive
    assert r["stop_cause"] == "api_unavailable" and r["stop_details"] == "critic_pending" and r["pending_critic_packets"]      # the blinded packet waits for the local GLM step
    assert (out / "RECEIPT.json").is_file() and (out / "ex2_clamp" / "journal.jsonl").is_file() and rec["label"] == "LOGIC_ONLY_STUB" and rec["stub"] is True
    assert r["journal_chain_ok"] and r["invariants"]["failed"] is None and len(r["invariants"]["checked"]) == 8


@test
def f1b_nothing_is_approved_by_l2_clusters_or_critic():
    rec, _ = harness([[LO_FIRST, HI_FIRST], [LO_FIRST, HI_FIRST], [checks_text(GOOD_CHECK)]], tmp_out(), critic=StubCritic())
    r = rec["tasks"]["ex2_clamp"]
    assert r["never_approved_by_l2_clusters_or_critic"] and r["answer_label"] != "VERIFIED" and r["authorized_to_execute"] is False
    assert r["l2_evidence"] and all(e["independence"] == "model_derived" for e in r["l2_evidence"])          # L2 evidence is model-derived soft evidence only
    assert rec["summary"]["ex2_clamp"]["approved_by_model_evidence"] is False


# ------------------------------------------------------------------ negatives
@test
def f1b_vacuous_and_untraceable_checks_are_rejected_with_reasons():
    bad_quote = dict(GOOD_CHECK, quote="no such words in the request")
    rec, _ = harness([[LO_FIRST, HI_FIRST], [LO_FIRST, HI_FIRST], [checks_text(VACUOUS_CHECK, bad_quote)]], tmp_out(), critic=StubCritic())
    l2 = rec["tasks"]["ex2_clamp"]["l2"]
    assert l2["verdict"] == "INSUFFICIENT" and l2["parsed"] == 2
    reasons = " | ".join(d["reason"] for d in l2["dropped"])
    assert "vacuous" in reasons and "verbatim span" in reasons, reasons


@test
def f1b_no_valid_l2_text_and_one_cluster_are_valid_receipt_results():
    rec, _ = harness([[LO_FIRST], [LO_FIRST], ["t</think>Here are some tests"]], tmp_out(), critic=StubCritic())
    r = rec["tasks"]["ex2_clamp"]
    assert r["l2"]["verdict"] == "INSUFFICIENT" and r["l2"]["reason"] == "no valid test parsed from the model"
    assert r["discrimination"]["verdict"] == "NOT_TRIGGERED" and r["stop_cause"] == "decided" and not rec["errors"]


@test
def f1b_a_wrong_candidate_set_with_equal_behaviour_does_not_trigger_discrimination():
    twin = LO_FIRST.replace("x + 0", "x + 0 ")
    rec, _ = harness([[LO_FIRST, twin], [LO_FIRST, twin], [checks_text(GOOD_CHECK)]], tmp_out(), critic=StubCritic())
    r = rec["tasks"]["ex2_clamp"]
    assert r["discrimination"]["verdict"] == "NOT_TRIGGERED" and (r["clusters"]["extended_clusters"] in (None, 1))


# ------------------------------------------------------------------ guards
@test
def f1b_stub_backend_refused_in_final_mode_and_existing_dir_refused():
    out = tmp_out()
    for gen in (RT.FakeGen([]), Q.FixtureGenerationBackend([])):
        try:
            H.run_harness(out, gen, LocalExec(), None, final=True, examples=[CKEY])
            assert False, "stub accepted"
        except H.StubBackendRefused:
            pass
    assert not out.exists()                                                                       # refused before anything was written
    try:
        H.assert_real_backend(object())
        assert False
    except H.StubBackendRefused:
        pass

    class Liar(RT.FakeGen):                                                                       # claims to be the real backend but is not an HFQwenBackend
        def model_info(self):
            return {"backend": "hf", "local": True, "profile_id": H.BACKBONE, "snapshot_verified": True}
    try:
        H.assert_real_backend(Liar([]))
        assert False
    except H.StubBackendRefused:
        pass
    # an unverified real backend class is refused too (snapshot not verified)
    real = Q.HFQwenBackend(authorized=True, local=True, env={"EVIDENCE_LADDER_ALLOW_INFERENCE": "1"})
    try:
        H.assert_real_backend(real)
        assert False
    except H.StubBackendRefused:
        pass
    done = tmp_out()
    harness([[LO_FIRST], [LO_FIRST], ["t</think>[]"]], done)
    try:
        harness([[LO_FIRST]], done)
        assert False, "existing out dir accepted"
    except FileExistsError:
        pass


@test
def f1b_receipt_is_create_only_and_names_the_backend():
    out = tmp_out()
    rec, _ = harness([[LO_FIRST], [LO_FIRST], ["t</think>[]"]], out)
    assert rec["backend"]["label"] == "LOGIC_ONLY_STUB" and rec["script_sha256"] and rec["critic"]["kind"] is None
    try:
        H.write_new(out / "RECEIPT.json", "x")
        assert False
    except FileExistsError:
        pass


@test
def f1b_a_crash_in_one_task_is_recorded_not_hidden():
    class Boom(RT.FakeGen):
        def generate(self, req):
            raise RuntimeError("backend exploded")
    rec = H.run_harness(tmp_out(), Boom([]), LocalExec(), None, final=False, examples=[CKEY])
    assert "ex2_clamp" in rec["errors"] and "backend exploded" in rec["errors"]["ex2_clamp"]["error"] and rec["tasks"] == {}


@test
def f1b_sandbox_runner_protocol_wsl():
    from evidence_ladder.controller.wsl_sandbox import available
    if not available():
        raise Skip("WSL does not answer")
    with tempfile.TemporaryDirectory() as d:
        exe = H.WslFunctionExec(Path(d))
        t = H.derive_task("ex2_clamp", "clamp")
        art = H.Artifact.make("ex2_clamp", "python", LO_FIRST)
        out = exe.run_public(art, t)
        assert out[0]["status"] == "pass" and all(o["status"] == "pass" for o in out[1:]), out
        e = exe.run_experiment(art, {"kind": "function_call", "payload": json.dumps({"x": 9, "lo": 0, "hi": 5})})
        assert e["status"] == "pass" and e["observed"] == 5
        broken = exe.run_public(H.Artifact.make("ex2_clamp", "python", "def solve(:\n"), t)
        assert broken[0]["status"] == "fail" and broken[0]["failure_class"] == "compile_error"
        crash = exe.run_experiment(H.Artifact.make("ex2_clamp", "python", "def solve(data):\n    return data['nope']\n"), {"kind": "function_call", "payload": "{}"})
        assert crash["status"] == "error" and crash["failure_class"] == "runtime_error"


def main():
    failed = {}
    for name, f in TESTS.items():
        try:
            f()
        except Skip as s:
            SKIPPED[name] = str(s)
        except (Exception, SystemExit):      # noqa: BLE001
            failed[name] = traceback.format_exc()[-1100:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed) - len(SKIPPED), "failed": failed, "skipped": SKIPPED}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
