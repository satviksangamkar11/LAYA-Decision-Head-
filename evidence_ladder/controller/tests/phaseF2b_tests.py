"""[EL-ARCH-1 F2b / Agent 2] Tests for the OPTIONAL timeline side channel (controller/timeline.py): real timestamps for every journal entry, stage boundary, model call and sandbox call,
kept OUTSIDE the journal. Every behaviour has a positive and a negative test; the central negative is the hash-identity test: the same task, with and without a timeline, gives the same
journal hash, the same chain and the same result, and a failing timeline / failing clock changes no decision. Run from the repo root:
    PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe evidence_ladder/controller/tests/phaseF2b_tests.py
Tests named *_logic use the in-process ToySandbox / fakes (the controller, journal and timeline code are real); *_wsl use the real WSL sandbox."""
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phaseB_tests as B                                                                  # noqa: E402
import run_tests as RT                                                                     # noqa: E402
from evidence_ladder.common.interfaces import ControllerContext                            # noqa: E402
from evidence_ladder.controller import investigate as INV, replay as REP                   # noqa: E402
from evidence_ladder.controller.agentic import AgenticContext, AgenticRun                  # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController                 # noqa: E402
from evidence_ladder.controller.timeline import Timeline                                   # noqa: E402
from evidence_ladder.controller.wsl_sandbox import WslSandbox                              # noqa: E402

TESTS, SKIPPED = {}, {}
opt = B.opt


def test(f):
    TESTS[f.__name__] = f
    return f


class Clk:
    """deterministic clocks: every call advances 0.5 s"""
    def __init__(self):
        self.t = 100.0

    def mono(self):
        self.t += 0.5
        return self.t

    def utc(self):
        return "2026-10-18T00:00:%06.3fZ" % (self.t % 60)


def det():
    c = Clk()
    return Timeline(utc=c.utc, mono=c.mono)


class BadTimeline:
    """every method raises: the controllers must not change any decision because of it"""
    def __getattr__(self, name):
        def boom(*a, **k):
            raise RuntimeError("timeline broken: " + name)
        return boom


# ------------------------------------------------------------------ the Timeline class
@test
def tl_rows_and_summary_have_the_documented_shapes_logic():
    tl = det()
    tl.enter("a", step=1)
    tl.event("model_call", 2.0, model="m", tokens_prompt=3, tokens_generated=4)
    with tl.span("s", x=1):
        pass
    tl.sandbox_call(0.25, "protected_tests_v1", "pass")
    tl.enter("b")
    rows = tl.rows()
    json.dumps(rows)                                                                          # JSON-serialisable
    kinds = [r["row"] for r in rows]
    assert kinds == ["event", "span", "event", "stage", "stage"], kinds
    st = [r for r in rows if r["row"] == "stage"]
    assert [r["name"] for r in st] == ["a", "b"] and all(r["duration_s"] > 0 and r["start_utc_iso"] and r["end_monotonic_s"] > r["start_monotonic_s"] for r in st)
    mc = next(r for r in rows if r["row"] == "event" and r["kind"] == "model_call")
    assert mc["seconds"] == 2.0 and mc["tokens_prompt"] == 3 and mc["tokens_generated"] == 4 and mc["model"] == "m" and mc["stage"] == "a" and mc["step"] == 1
    sb = next(r for r in rows if r["row"] == "event" and r["kind"] == "sandbox")
    assert sb["check_id"] == "protected_tests_v1" and sb["status"] == "pass" and sb["seconds"] == 0.25
    sp = next(r for r in rows if r["row"] == "span")
    assert sp["name"] == "s" and sp["x"] == 1 and sp["parent_stage"] == "a" and sp["duration_s"] > 0
    s = tl.summary()
    assert s["stages"]["a"]["count"] == 1 and s["events"]["model_call"]["total_s"] == 2.0 and s["events"]["sandbox"]["count"] == 1 and s["spans"]["s"]["count"] == 1 and s["clock_errors"] == 0


@test
def tl_percentiles_only_where_meaningful_logic():
    tl = det()
    for sec in (1.0, 2.0, 3.0, 4.0, 10.0):
        tl.event("sandbox", sec)
    ev = tl.summary()["events"]["sandbox"]
    assert ev["count"] == 5 and ev["total_s"] == 20.0 and ev["p50"] == 3.0 and ev["p95"] == 10.0
    tl2 = det()
    tl2.event("sandbox", 1.5)
    one = tl2.summary()["events"]["sandbox"]
    assert one["count"] == 1 and one["p50"] == 1.5 and one["p95"] is None                  # a p95 of one sample is not reported


@test
def tl_failing_clock_blanks_the_row_and_never_raises_logic():
    def bad():
        raise OSError("no clock")
    tl = Timeline(utc=bad, mono=bad)
    tl.enter("x")
    tl.event("sandbox", 0.1)
    tl.on_entry(0, "run_start")
    rows = tl.rows()
    assert rows and all(r.get("clock_error") for r in rows if r["row"] in ("stage", "entry", "event"))
    assert next(r for r in rows if r["row"] == "stage")["duration_s"] is None and tl.summary()["clock_errors"] >= 3
    ok = det()                                                                                  # negative control: a working clock has no errors
    ok.enter("x")
    assert tl_no_errors(ok)


def tl_no_errors(tl):
    return tl.summary()["clock_errors"] == 0 and not any(r.get("clock_error") for r in tl.rows())


@test
def tl_default_clocks_are_real_utc_and_monotonic_logic():
    tl = Timeline()
    tl.on_entry(0, "k")
    r = tl.rows()[0]
    from datetime import datetime
    assert datetime.fromisoformat(r["utc_iso"]).tzinfo is not None and isinstance(r["monotonic_s"], float)


# ------------------------------------------------------------------ agentic mode
def _agentic(tl, scorer=None, steps=2):
    ws, T = B.make_ws("slug")
    r = AgenticRun(B.atask("slug", ws, T, max_steps=3), AgenticContext(sandbox=B.ToySandbox(), scorer=scorer, arm="E", config={}, timeline=tl), lambda m: "ctx")
    r.begin()
    INV.investigate(r)
    r.step([opt(B.SLUG_BAD), opt(B.READ_TEST)], {"new_tokens": 9, "prompt_tokens": 20, "kaggle_seconds": 1.25})
    r.step([opt(B.SLUG_FIX)], {"new_tokens": 5, "prompt_tokens": 20, "kaggle_seconds": 0.75})
    res = r.finish()
    return r, res


def _sig(r, res):
    return {"hash": res.journal_hash, "chain": r.journal.chain_hashes(), "label": res.answer_label, "stop": res.stop_cause, "details": res.stop_details, "kinds": [e["kind"] for e in r.journal.entries()],
            "rounds": res.rounds, "decision": res.decision}


@test
def tl_agentic_journal_hash_is_identical_with_and_without_the_timeline_logic():
    r0, res0 = _agentic(None, scorer=_Rec())
    tl = det()
    r1, res1 = _agentic(tl, scorer=_Rec())
    assert _sig(r0, res0) == _sig(r1, res1)                                                     # byte-identical journal, chain, label, stop, decision
    rows = tl.rows()
    assert sum(1 for x in rows if x["row"] == "entry") == len(r1.journal) and [x["seq"] for x in rows if x["row"] == "entry"] == list(range(len(r1.journal)))      # EVERY journal entry has a row
    assert "timeline" not in json.dumps(r1.journal.entries()) and "utc_iso" not in json.dumps(r1.journal.entries())            # nothing of it inside an entry
    # negative: two timeline runs with different clocks still give the same journal hash
    tl2 = Timeline(utc=lambda: "2000-01-01T00:00:00+00:00", mono=lambda: 5.0)
    r2, res2 = _agentic(tl2, scorer=_Rec())
    assert res2.journal_hash == res0.journal_hash


class _Rec:
    scorer_id = "f2b_scorer"
    scorer_def = {"weights_hash": "f2b"}

    def score(self, t, pool, exploratory=False):
        return [float(len(pool["options"]) - i) for i in range(len(pool["options"]))]


@test
def tl_agentic_stage_map_model_calls_and_sandbox_calls_are_recorded_logic():
    tl = det()
    r, res = _agentic(tl, scorer=_Rec())
    rows = tl.rows()
    stages = [x["name"] for x in rows if x["row"] == "stage"]
    first = {n: stages.index(n) for n in ("intake", "investigate", "filter", "check", "rank", "commit", "observe", "repair", "commit_verify")}
    assert first["intake"] < first["investigate"] < first["filter"] < first["check"] < first["rank"] < first["commit"] < first["observe"] < first["repair"] < first["commit_verify"], stages
    mcs = [x for x in rows if x["row"] == "event" and x["kind"] == "model_call"]
    q = [x for x in mcs if x["model"] == "qwen"]
    assert [(x["seconds"], x["tokens_prompt"], x["tokens_generated"]) for x in q] == [(1.25, 20, 9), (0.75, 20, 5)]               # generation tokens + seconds as the backend reported them
    assert [x["model"] for x in mcs].count("laya_scorer") == 2 and all(x["seconds"] and x["seconds"] > 0 for x in mcs if x["model"] == "laya_scorer")
    sbs = [x for x in rows if x["row"] == "event" and x["kind"] == "sandbox"]
    assert {x["check_id"] for x in sbs} >= {"protected_tests_v1", "action_dry_run_v1"} and all(x["status"] in ("pass", "fail") and x["seconds"] > 0 for x in sbs)
    assert any(x["check_id"] == "action_dry_run_v1" and x["status"] == "fail" for x in sbs) and any(x["check_id"] == "action_dry_run_v1" and x["status"] == "pass" for x in sbs)
    s = tl.summary()
    assert s["stages"]["rank"]["count"] == 2 and s["events"]["sandbox"]["count"] == len(sbs) and s["entries"] == len(r.journal)


@test
def tl_agentic_failing_timeline_and_failing_clock_change_no_decision_logic():
    r0, res0 = _agentic(None, scorer=_Rec())
    r1, res1 = _agentic(BadTimeline(), scorer=_Rec())                                           # every Timeline method raises
    assert _sig(r0, res0) == _sig(r1, res1)
    def bad():
        raise OSError("clock gone")
    tl = Timeline(utc=bad, mono=bad)
    r2, res2 = _agentic(tl, scorer=_Rec())                                                      # every clock call raises
    assert _sig(r0, res0) == _sig(r2, res2) and tl.summary()["clock_errors"] > 0
    # negative control: the sandbox call itself is never retried or swallowed by the timeline wrapper
    class CountSb(B.ToySandbox):
        pass
    ws, T = B.make_ws("slug")
    sb = CountSb()
    r = AgenticRun(B.atask("slug", ws, T, max_steps=3), AgenticContext(sandbox=sb, arm="E", config={}, timeline=det()), lambda m: "ctx")
    r.begin()
    n0 = len(sb.calls)
    r.step([opt(B.READ_TEST)], {})
    assert len(sb.calls) - n0 == 1


@test
def tl_agentic_replay_is_identical_with_the_timeline_on_and_off_logic():
    ws, T = B.make_ws("slug")
    tl = det()
    rr = REP.RecordedRun(B.atask("slug", ws, T, max_steps=3), AgenticContext(sandbox=B.ToySandbox(), arm="E", config={}, timeline=tl), lambda m: "ctx")
    rr.begin()
    rr.step([opt(B.SLUG_FIX)], {"new_tokens": 3, "prompt_tokens": 4, "kaggle_seconds": 0.5})
    res = rr.finish()
    assert tl.rows() and any(x["row"] == "event" and x["kind"] == "model_call" for x in tl.rows())          # RecordedRun passed the timeline through
    rec = rr.record()
    out = REP.replay(rec)                                                                      # the replay has NO timeline
    assert out["identical"] and out["recorded_journal_hash"] == res.journal_hash
    assert "utc_iso" not in json.dumps(rec["chain"]) and "timeline" not in rec


@test
def tl_agentic_stage_boundaries_on_the_real_sandbox_wsl():
    B.need_wsl()
    ws, T = B.make_ws("clamp")
    tl = Timeline()                                                                              # REAL clocks
    sb = WslSandbox(Path(tempfile.mkdtemp(prefix="f2b-sb-")), T["protected"])
    r = AgenticRun(B.atask("clamp", ws, T, max_steps=2), AgenticContext(sandbox=sb, arm="E", config={}, timeline=tl), lambda m: "ctx")
    r.begin()
    INV.investigate(r)
    r.step([opt({"name": "Edit", "arguments": {"file_path": "clamp.py", "old_string": "return min(lo, max(x, hi))", "new_string": "return max(lo, min(x, hi))"}})], {})
    res = r.finish()
    assert res.answer_label == "VERIFIED"
    sbs = [x for x in tl.rows() if x["row"] == "event" and x["kind"] == "sandbox"]
    assert len(sbs) >= 4 and all(x["seconds"] > 0 for x in sbs) and {x["status"] for x in sbs} == {"pass", "fail"}      # real WSL runs took real time; baseline fails, edit passes
    s = tl.summary()
    assert s["wall_s"] > 0 and s["events"]["sandbox"]["total_s"] > 0 and s["entries"] == len(r.journal)


# ------------------------------------------------------------------ ladder mode
def _ladder(timeline, repair=False):
    if repair:
        ctl = EvidenceLadderController(timeline=timeline)
        gen, exe = RT.FakeGen([["WRONG:a:0"], ["OK:fix"]]), RT.FakeExec()
        res = ctl.run_problem(RT.TASK, ControllerContext(gen, exe, RT.FakeBudget(arm="D0"), "D0", None, None, (1, 2, 3, 4, 5, 6, 7, 8), {"k": 1}))
        return ctl, res
    codes = [B.PLUS2, B.DOUBLE]
    cfg = {"k": 2, "l2": True, "discriminate": True, "critic": True, "mode": "ladder"}
    resp = {"experiment": {"kind": "function_call", "payload": json.dumps({"x": -3})}, "implications": "outputs differ"}
    cfg["critic_responses"] = {B.crit_id("E1", codes): resp}
    ctl = EvidenceLadderController(timeline=timeline)
    gen = RT.FakeGen([codes, [B.CHECKS_TEXT]])
    ctx = ControllerContext(gen, B.EvalExec(), RT.FakeBudget(), "D0", RT.FakeCritic(), None, (1, 2, 3, 4), cfg)
    return ctl, ctl.run_problem(B.EVAL_TASK, ctx)


@test
def tl_ladder_journal_hash_is_identical_with_and_without_the_timeline_logic():
    for repair in (False, True):
        c0, r0 = _ladder(None, repair)
        tl = det()
        c1, r1 = _ladder(tl, repair)
        assert r0.journal_hash == r1.journal_hash and c0.last_journal.chain_hashes() == c1.last_journal.chain_hashes() and r0.answer_label == r1.answer_label and r0.stop_cause == r1.stop_cause
        assert sum(1 for x in tl.rows() if x["row"] == "entry") == len(c1.last_journal) and "utc_iso" not in json.dumps(c1.last_journal.entries())
    ctl_bad, rb = _ladder(BadTimeline(), False)                                                  # negative: a broken timeline changes nothing here either
    assert rb.journal_hash == _ladder(None, False)[1].journal_hash


@test
def tl_ladder_stage_map_covers_intake_generate_execute_l2_cluster_discriminate_rank_commit_logic():
    tl = det()
    ctl, res = _ladder(tl, False)
    stages = [x["name"] for x in tl.rows() if x["row"] == "stage"]
    for n in ("intake", "generate", "execute", "l2_generate", "l2_validate", "cluster", "discriminate", "rank", "commit_verify"):
        assert n in stages, (n, stages)
    assert stages.index("intake") < stages.index("generate") < stages.index("execute") < stages.index("l2_generate") < stages.index("l2_validate") < stages.index("cluster") < stages.index("discriminate") < stages.index("rank") < stages.index("commit_verify")
    rows = tl.rows()
    mcs = [x for x in rows if x["row"] == "event" and x["kind"] == "model_call"]
    assert len(mcs) >= 3 and all("tokens_generated" in x and "seconds" in x for x in mcs) and {x["purpose"] for x in mcs} >= {"sample", "tests"}      # k candidates + the L2 test generation
    sbs = [x for x in rows if x["row"] == "event" and x["kind"] == "sandbox"]
    assert {x["check_id"] for x in sbs} >= {"public_examples_v1", "experiment"} and all(x["status"] for x in sbs)


@test
def tl_ladder_repair_rounds_are_a_stage_with_the_round_number_logic():
    tl = det()
    ctl, res = _ladder(tl, True)
    rep = [x for x in tl.rows() if x["row"] == "stage" and x["name"] == "repair"]
    assert res.rounds == 1 and len(rep) == 1 and rep[0]["round"] == 1 and rep[0]["duration_s"] > 0
    assert any(x["row"] == "event" and x["kind"] == "model_call" and x.get("purpose") == "repair" for x in tl.rows())
    tl0 = det()                                                                                  # negative: a first-try success has no repair stage
    _c, r0 = (lambda c: (c, c.run_problem(RT.TASK, ControllerContext(RT.FakeGen([["OK:a"]]), RT.FakeExec(), RT.FakeBudget(arm="D0"), "D0", None, None, (1, 2, 3, 4), {"k": 1}))))(EvidenceLadderController(timeline=tl0))
    assert not [x for x in tl0.rows() if x["row"] == "stage" and x["name"] == "repair"] and r0.rounds == 0


# ------------------------------------------------------------------ runner
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
