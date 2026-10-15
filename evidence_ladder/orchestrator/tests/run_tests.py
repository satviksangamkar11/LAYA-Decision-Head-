"""Orchestrator tests: plain asserts, deterministic, no model, no network.  Run from the repo root:
    python3 evidence_ladder/orchestrator/tests/run_tests.py
Prints JSON {"tests","passed","failed":{name: error}} and exits 1 on any failure. Fabricated approvals exist ONLY inside temp sandbox directories here."""
import dataclasses
import inspect
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from evidence_ladder.common import interfaces as I   # noqa: E402
from evidence_ladder.orchestrator import arms, authorization, budget, dryrun, index, pairing, pilot, report, stats   # noqa: E402
from evidence_ladder.orchestrator.base import THRESHOLDS, ConfigError, PoolingError   # noqa: E402

TESTS = []


def test(f):
    TESTS.append(f)
    return f


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


_IDX = index.load_index()
_TASKS = index.build_tasks(_IDX)


# ---------------------------------------------------------------- index
@test
def index_verified_counts():
    assert index.counts(_IDX.rows) == {"python": 40, "java": 40, "fable_val": 46, "fable_exposed": 335}
    assert len(_IDX.rows) == 461


@test
def index_tampered_line_rejected():
    with tempfile.TemporaryDirectory() as t:
        for n in ("INDEX.jsonl", "SUMMARY.json", "VALIDATION.json"):
            shutil.copy(Path(_IDX.directory) / n, t)
        index.load_index(t)   # untouched copy loads
        p = Path(t) / "INDEX.jsonl"
        raw = p.read_bytes()
        assert b'"easy"' in raw
        p.write_bytes(raw.replace(b'"easy"', b'"hard"', 1))
        raises(index.IndexVerificationError, index.load_index, t)
        p.write_bytes(raw)
        s = json.loads((Path(t) / "SUMMARY.json").read_text())
        s["index_sha256"] = "0" * 64
        (Path(t) / "SUMMARY.json").write_text(json.dumps(s))
        raises(index.IndexVerificationError, index.load_index, t)   # SUMMARY and VALIDATION disagree


@test
def taskspec_has_no_protected_data():
    fields = {f.name for f in dataclasses.fields(I.TaskSpec)}
    assert fields == {"task_id", "index_item_id", "track", "language", "prompt", "public_tests", "difficulty", "mutation_status"}
    assert len(_TASKS["python"]) == 40 and len(_TASKS["java"]) == 40
    for tr, ts in _TASKS.items():
        for t in ts:
            assert t.track == tr and t.language == tr
            assert len(t.public_tests) == 4 and all(set(c) == {"input", "expected"} for c in t.public_tests)
            assert "wrong_solutions" not in t.mutation_status and "protected" not in json.dumps(t.mutation_status).lower()
    bad = dict(_IDX.rows[0])
    bad["source"] = {"files": ["results/raw/x/grader/protected_cases.json"]}
    raises(index.IndexVerificationError, index._worker_rows, bad)   # only worker files are ever read


@test
def mutation_flags_present():
    by = {t.task_id: t.mutation_status for ts in _TASKS.values() for t in ts}
    for tid in ("P1-19", "P1-26", "J1-17", "J1-27"):
        assert by[tid]["state"] == "below_threshold" and by[tid]["flagged"], tid
        assert by[tid]["kill_rate"] < 0.80
    for tid in ("P1-05", "P1-10", "P1-14", "J1-08", "J1-10", "J1-20"):
        assert by[tid]["state"] == "unmeasured" and by[tid]["flagged"], tid
    assert all(not s["waiver_applies"] and "NOT waive" in s["waiver_note"] for s in by.values())
    assert sum(s["flagged"] for s in by.values()) >= 10


@test
def fable_track_separate_and_not_runnable():
    ft = index.build_fable_track(_IDX)
    assert len(ft.val_items) == 46 and len(ft.exposed_items) == 335
    assert all(x["eval_role"] == "primary_eval_recorded_action_agreement" and "flag" not in x for x in ft.val_items)
    assert all(x["flag"] == "exposed_not_for_laya_evaluation" for x in ft.exposed_items)
    assert ft.run_supported is False and "NOT IMPLEMENTED" in ft.run_status
    raises(NotImplementedError, ft.run)
    raises(NotImplementedError, dryrun.make_runner, tempfile.gettempdir(), tasks={"fable": []})


# ---------------------------------------------------------------- arms
@test
def arm_configs():
    assert arms.arm_config("A") == {"mode": "single"}
    assert arms.arm_config("B", k=5) == {"mode": "best_of_k", "k": 5}
    assert arms.arm_config("C") == {"mode": "repair"}
    assert arms.arm_config("D0") == {"mode": "ladder"}
    assert arms.arm_config("D1") == {"mode": "ladder", "critic": True}
    assert arms.arm_config("E") == {"mode": "ladder", "critic": True, "scorer": True, "exploratory": True}
    raises(ConfigError, arms.arm_config, "B")
    raises(ConfigError, arms.arm_config, "Z")


@test
def e_label_and_no_laya_decision():
    assert arms.E_LABEL == "exploratory_transfer_no_keep_drop" and arms.arm_label("E") == arms.E_LABEL and arms.arm_label("D1") == ""
    raises(arms.StageOneLayaDecisionRefused, arms.laya_keep_drop, {"anything": 1})
    ctx = pairing.fresh_context(_TASKS["python"][0], "E", generation=None, execution=None, budget=None, master_seed=1, k1=2)
    assert ctx.config["arm_label"] == arms.E_LABEL and ctx.config["exploratory"] is True


# ---------------------------------------------------------------- budget
def check_budget_schema(rec):
    req = json.loads((REPO / "specs/evidence_ladder_v1/schemas_v1.json").read_text())["$defs"]["budget_record"]["required"]
    assert set(rec) == set(req), set(rec) ^ set(req)
    for k, v in rec.items():
        if k in ("problem_id", "arm"):
            assert isinstance(v, str)
        elif k in ("gpu_seconds", "wall_seconds"):
            assert isinstance(v, float) and v >= 0
        else:
            assert isinstance(v, int) and not isinstance(v, bool) and v >= 0
    assert rec["arm"] in arms.ARMS


@test
def budget_T_pending_is_config_error():
    e = raises(ConfigError, budget.MatchedTokenBudget, "p", "A")
    assert "PENDING" in str(e) and "token_cap_T" in str(e)
    raises(ConfigError, budget.MatchedTokenBudget, "p", "A", 0)
    raises(ConfigError, budget.MatchedTokenBudget, "p", "A", True)


@test
def budget_separate_counters_and_schema():
    b = budget.MatchedTokenBudget("P", "D1", 1000)
    b.charge("qwen_gen", 100)
    b.charge("qwen_prefill", 5000)
    b.charge("glm_in", 300)
    b.charge("glm_out", 50)
    b.note_call("glm_calls")
    b.note_call("tool_runs", 3)
    b.note_time(gpu_seconds=1.5, wall_seconds=2.0)
    r = b.record()
    check_budget_schema(r)
    assert (r["qwen_gen_tokens"], r["qwen_prefill_tokens"], r["glm_tokens_in"], r["glm_tokens_out"]) == (100, 5000, 300, 50)
    assert (r["glm_calls"], r["tool_runs"], r["gpu_seconds"], r["wall_seconds"]) == (1, 3, 1.5, 2.0)
    assert b.generated() == 150 and b.remaining() == 850 and not b.exhausted()   # prefill and glm_in do not count toward T
    assert "not an equal-cost" in b.sidecar()["matching"]
    raises(ValueError, b.charge, "bogus", 1)
    raises(ValueError, b.charge, "qwen_gen", -1)


@test
def budget_ceiling_counts_qwen_plus_glm_out():
    b = budget.MatchedTokenBudget("P", "D1", 100)
    b.charge("qwen_gen", 60)
    b.charge("glm_out", 40)
    assert b.exhausted() and b.generated() == 100   # exactly T: no room left, but not crossed
    raises(I.BudgetExhausted, b.charge, "qwen_gen", 1)
    assert b.record()["qwen_gen_tokens"] == 61       # spent tokens are still recorded
    b2 = budget.MatchedTokenBudget("P", "D1", 100)
    b2.charge("qwen_prefill", 10 ** 6)
    b2.charge("glm_in", 10 ** 6)
    assert not b2.exhausted()
    raises(I.BudgetExhausted, b2.charge, "glm_out", 101)


@test
def budget_arm_b_gets_glm_tokens_back():
    T = 1000
    d1 = budget.MatchedTokenBudget("P", "D1", T)
    d1.charge("qwen_gen", 700)
    d1.charge("glm_out", 300)
    assert d1.exhausted()
    assert budget.extra_qwen_samples_for_b(300, 100) == 3 and budget.extra_qwen_samples_for_b(299, 100) == 2
    b = budget.MatchedTokenBudget("P", "B", T)
    for _ in range(10):
        b.charge("qwen_gen", 100)       # B spends the whole T on Qwen samples: D1's 300 GLM tokens come back as 3 extra samples
    assert b.generated() == T and b.exhausted()
    raises(I.BudgetExhausted, b.charge, "qwen_gen", 1)
    raises(ValueError, budget.MatchedTokenBudget("P", "B", T).charge, "glm_out", 1)   # B has no GLM


# ---------------------------------------------------------------- pairing
@test
def seeds_paired_and_stable():
    for tid in ("P1-01", "J1-02"):
        base = pairing.arm_seeds(7, tid, "B", 4, 6)
        for arm in ("C", "D0", "D1", "E"):
            s = pairing.arm_seeds(7, tid, arm, 4, 6)
            assert s[:4] == base[:4] and len(s) == 6
        assert pairing.arm_seeds(7, tid, "A", 4) == base[:1]
        assert pairing.arm_seeds(7, tid, "B", 4, 6) == base   # stable across calls (sha256-derived, no builtin hash())
        assert all(0 <= x < 2 ** 32 for x in base)
    assert pairing.arm_seeds(7, "P1-01", "B", 4) != pairing.arm_seeds(7, "P1-02", "B", 4)
    assert pairing.arm_seeds(7, "P1-01", "B", 4) != pairing.arm_seeds(8, "P1-01", "B", 4)
    raises(ConfigError, pairing.arm_seeds, 7, "P1-01", "B", None)
    assert "hash(" not in inspect.getsource(pairing).replace("sha(", "").replace("starting_state_hash(", "").replace("_hash(", "")


@test
def identical_starting_state_fresh_context():
    t = _TASKS["python"][0]
    ctxs = {a: pairing.fresh_context(t, a, generation=object(), execution=object(), budget=object(), master_seed=3, k1=3, n_total=5) for a in arms.ARMS}
    assert len({id(c) for c in ctxs.values()}) == 6                      # a new context object per arm
    assert len({tuple(c.seeds[:3]) for a, c in ctxs.items() if a != "A"}) == 1
    assert ctxs["A"].seeds == ctxs["B"].seeds[:1]
    h = {pairing.starting_state_hash(t, 3, 3) for _ in arms.ARMS}
    assert len(h) == 1
    assert pairing.starting_state_hash(t, 3, 3) != pairing.starting_state_hash(_TASKS["python"][1], 3, 3)
    assert ctxs["B"].config["k"] == 5 and ctxs["D0"].config["mode"] == "ladder"
    pr = pairing.PairedRecords()
    pr.add("t1", "A", 1)
    pr.add("t1", "B", 2)
    pr.add("t2", "A", 3)
    assert pr.paired("A", "B") == ["t1"] and pr.get("t1", "B") == 2
    raises(ValueError, pr.add, "t1", "A", 9)


# ---------------------------------------------------------------- authorization (fabricated approvals live only in temp sandboxes)
def _sandbox_toml(t):
    for suffix in ("", ".sha256"):
        shutil.copy(str(THRESHOLDS) + suffix, Path(t) / ("thr.toml" + suffix))
    return Path(t) / "thr.toml"


FULL = {k: True for k in authorization.APPROVALS}


@test
def authorization_refuses_by_default_and_partial():
    with tempfile.TemporaryDirectory() as t:
        thr = _sandbox_toml(t)
        raises(I.NotAuthorized, authorization.require_authorization, None, thr)
        raises(I.NotAuthorized, authorization.require_authorization, Path(t) / "missing.json", thr)
        p = authorization.write_authorization_manifest(None, t, thr)           # default state: nothing approved
        assert json.loads(p.read_text())["approvals"] == {k: False for k in authorization.APPROVALS}
        raises(I.NotAuthorized, authorization.require_authorization, p, thr)
        for drop in authorization.APPROVALS:                                   # any three of four still refuse
            p = authorization.write_authorization_manifest({k: True for k in authorization.APPROVALS if k != drop}, t, thr)
            e = raises(I.NotAuthorized, authorization.require_authorization, p, thr)
            assert drop in str(e)
        raises(ValueError, authorization.write_authorization_manifest, {"bogus": True}, t, thr)
        raises(ValueError, authorization.write_authorization_manifest, {"pilot_go": "yes"}, t, thr)


@test
def authorization_hash_mismatch_refused_and_valid_accepted():
    with tempfile.TemporaryDirectory() as t:
        thr = _sandbox_toml(t)
        p = authorization.write_authorization_manifest(FULL, t, thr)
        m = authorization.require_authorization(p, thr)
        assert all(m["approvals"].values()) and len(m["manifest_sha256"]) == 64
        good = p.read_text()
        d = json.loads(good)
        d["approvals"]["pilot_go"] = False
        p.write_text(json.dumps(d))
        raises(I.NotAuthorized, authorization.require_authorization, p, thr)   # approvals edited, manifest hash stale
        d = json.loads(authorization.write_authorization_manifest({**FULL, "pilot_go": False}, t, thr).read_text())
        d["approvals"]["pilot_go"] = True
        p.write_text(json.dumps(d))
        raises(I.NotAuthorized, authorization.require_authorization, p, thr)   # flipped a False to True without recomputing the hash
        p = authorization.write_authorization_manifest(FULL, t, thr)
        authorization.require_authorization(p, thr)
        thr.write_text(thr.read_text() + "\n# edited after approval\n")
        raises(I.NotAuthorized, authorization.require_authorization, p, thr)   # thresholds toml no longer matches its .sha256
        raises(I.NotAuthorized, authorization.write_authorization_manifest, FULL, t, thr)
    with tempfile.TemporaryDirectory() as t:
        a = Path(t) / "a"
        b = Path(t) / "b"
        a.mkdir()
        b.mkdir()
        ta, tb = _sandbox_toml(a), _sandbox_toml(b)
        tb.write_text(tb.read_text() + "\n# other version\n")
        import hashlib
        Path(str(tb) + ".sha256").write_text(hashlib.sha256(tb.read_bytes()).hexdigest() + "\n")
        p = authorization.write_authorization_manifest(FULL, t, ta)
        raises(I.NotAuthorized, authorization.require_authorization, p, tb)    # approvals were given for a different thresholds file


# ---------------------------------------------------------------- stats
@test
def stats_hand_computed():
    assert stats.paired_table([1, 1, 0, 0, 1], [1, 0, 1, 0, 0]) == {"gained": 2, "lost": 1, "both": 1, "neither": 1}
    assert stats.mcnemar_exact(5, 1) == 14 / 64                  # n=6: 2*(1+6)/64
    assert stats.mcnemar_exact(1, 5) == 14 / 64
    assert stats.mcnemar_exact(0, 5) == 2 / 32
    assert stats.mcnemar_exact(3, 3) == 1.0 and stats.mcnemar_exact(0, 0) == 1.0
    h = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert abs(h["a"] - 0.03) < 1e-12 and abs(h["c"] - 0.06) < 1e-12 and abs(h["b"] - 0.06) < 1e-12   # 3*.01, 2*.03, max(.06, .04)
    assert stats.holm({"a": 0.5, "b": 0.9}) == {"a": 1.0, "b": 1.0}   # 2*.5 = 1.0, then monotone
    assert stats.holm({"a": 0.4, "b": 0.6})["a"] == 0.8
    raises(ValueError, stats.paired_table, [1], [1, 0])


@test
def bootstrap_reproducible_and_degenerate():
    x, y = [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]
    r = stats.bootstrap_diff_ci(x, y, 2000, 5)
    assert (r["estimate"], r["lo"], r["hi"], r["resamples"]) == (1.0, 1.0, 1.0, 2000)
    x, y = [1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0]
    a, b = stats.bootstrap_diff_ci(x, y, 2000, 11), stats.bootstrap_diff_ci(x, y, 2000, 11)
    assert a == b and a["lo"] <= a["estimate"] <= a["hi"] and a["estimate"] == 0.25   # diffs 1,0,0,1,0,1,-1,0
    c = stats.bootstrap_diff_ci(x, y, 2000, 11, clusters=[0, 0, 1, 1, 2, 2, 3, 3])
    assert c == stats.bootstrap_diff_ci(x, y, 2000, 11, clusters=[0, 0, 1, 1, 2, 2, 3, 3])
    assert len(stats.sample_size_warnings(40)) == 1 and len(stats.sample_size_warnings(10)) == 2 and stats.sample_size_warnings(500) == []


# ---------------------------------------------------------------- report
def _spec(tid, track, state="measured_ok"):
    return I.TaskSpec(tid, "x:" + tid, track, track, "p", (), "", {"state": state, "flagged": state != "measured_ok"})


def _row(tid, track, arm, label="VERIFIED"):
    br = {"problem_id": tid, "arm": arm, "qwen_gen_tokens": 10, "qwen_prefill_tokens": 1, "glm_tokens_in": 0, "glm_tokens_out": 0, "glm_calls": 0,
          "scorer_calls": 0, "tool_runs": 1, "gpu_seconds": 0.0, "wall_seconds": 0.0, "api_errors": 0}
    return {"task_id": tid, "track": track, "arm": arm, "answer_label": label, "stop_cause": "decided", "budget_record": br}


def _res(vals):
    return {t: {a: {"passed": bool(v)} for a, v in arms_.items()} for t, arms_ in vals.items()}


def _mini():
    ids = ["P1", "P2", "P3", "P4"]
    B, C = [0, 0, 1, 1], [1, 1, 1, 0]
    res = {t: {a: {"passed": bool(v)} for a, v in zip(arms.ARMS, [0, B[i], C[i], C[i], C[i], C[i]])} for i, t in enumerate(ids)}
    meta = {t: _spec(t, "python", "below_threshold" if t == "P2" else "measured_ok") for t in ids}
    rows = [_row(t, "python", a, "VERIFIED") for t in ids for a in arms.ARMS]
    return res, meta, rows


@test
def report_contrasts_preserved_for_holdout():
    res, meta, rows = _mini()
    rep = report.make_report({"python": res}, {"python": meta}, rows, T=100)
    s = rep["tracks"]["python"]
    c = s["contrasts"]["C_vs_B"]
    assert c["table"] == {"gained": 2, "lost": 1, "both": 1, "neither": 0} and c["mcnemar_p"] == stats.mcnemar_exact(2, 1)
    assert set(s["contrasts"]) == {"C_vs_B", "D0_vs_C", "D1_vs_D0"}
    for k, v in s["contrasts"].items():
        assert "PRESERVED FOR HOLDOUT" in v["status"] and v["confirmatory"] is False and "holm_adjusted_p" in v and v["warnings"]
    assert s["contrasts"]["D0_vs_C"]["table"]["both"] == 3 and s["contrasts"]["D0_vs_C"]["mcnemar_p"] == 1.0
    assert s["contrasts"]["C_vs_B"]["excluding_flagged_tasks"]["n_pairs"] == 3
    assert s["mutation_flags"]["P2"]["state"] == "below_threshold" and s["mutation_flag_counts"] == {"below_threshold": 1}
    assert s["exploratory"]["E_vs_D1"]["label"] == arms.E_LABEL and s["exploratory"]["E_vs_D1"]["verdict"] is None
    assert s["per_arm"]["C"]["rate"] == 0.75 and s["safety"]["A"]["false_approval_rate"] == 1.0
    assert rep["label"] == "DEVELOPMENT EVIDENCE (not a held-out claim)" and rep["pooled"].startswith("NOT PRODUCED")
    assert s["coverage"]["A"]["graded"] == 4 and s["coverage"]["A"]["missing"] == []
    assert "NOT waive" in s["waiver_note"]
    md = report.render_markdown(rep)
    assert "DEVELOPMENT EVIDENCE" in md and "C_vs_B" in md


@test
def report_refuses_pooling_and_empty_tracks():
    raises(PoolingError, report.pool_tracks, {"python": 1}, {"java": 2})
    res, meta, rows = _mini()
    java_meta = {"J1": _spec("J1", "java")}
    raises(PoolingError, report.make_report, {"python": {"J1": {"A": {"passed": True}}}}, {"python": {**meta, **java_meta}}, rows)   # java task under python
    raises(PoolingError, report.make_report, {"python": {"J9": {"A": {"passed": True}}}}, {"python": meta}, rows)                     # unknown task
    raises(PoolingError, report.make_report, {"mixed": {}}, {}, [])
    raises(report.EmptyTrackError, report.track_accuracy, {})
    rep = report.make_report({"python": res}, {"python": meta, "java": java_meta}, rows)
    j = rep["tracks"]["java"]
    assert j["numbers"].startswith("REFUSED") and "per_arm" not in j and "contrasts" not in j
    assert rep["tracks"]["fable"]["numbers"].startswith("NOT PRODUCED")
    raises(NotImplementedError, report.make_report, {"fable": {"x": {"A": {"passed": True}}}}, {}, [])
    assert "overall" not in rep and "pooled_accuracy" not in json.dumps(rep)


# ---------------------------------------------------------------- pilot
class SpyController:
    def __init__(self, inner):
        self.inner, self.calls, self.ctxs = inner, [], []

    def run_problem(self, task, ctx):
        self.ctxs.append(ctx)
        r = self.inner.run_problem(task, ctx)
        self.calls.append((task.task_id, ctx.arm, r.stop_cause, tuple(ctx.seeds)))
        return r


class OrderGate(pilot.SelectionGate):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def close_selection(self, h=None):
        self.log.append("close")
        super().close_selection(h)


class OrderGrader(dryrun.FakeGrader):
    def __init__(self, log):
        self.log = log

    def grade(self, a):
        self.log.append("grade")
        return super().grade(a)


@test
def two_phase_critic_flow():
    with tempfile.TemporaryDirectory() as t:
        spy = SpyController(dryrun.FakeController())
        r = dryrun.make_runner(t, controller=spy)
        r.run()
        calls = [c for c in spy.calls if c[1] in ("D1", "E")]
        pend = [c for c in calls if c[2] == "api_unavailable"]
        assert pend, "pass 1 must end with api_unavailable for some critic runs"
        for tid, arm, _, seeds in pend:   # each pending run is replayed once with identical seeds, and the replay ends differently
            rep = [c for c in calls if c[0] == tid and c[1] == arm]
            assert len(rep) == 2, ("replay count", tid, arm, rep)
            assert rep[0][3] == rep[1][3], ("seeds differ between passes", tid, arm, rep)
            assert rep[1][2] != "api_unavailable", ("pass 2 still pending", tid, arm, rep)
        rows = [json.loads(l) for l in (Path(t) / "selection/selection_log_all_passes.jsonl").read_text().splitlines()]
        assert any(x["pass"] == 1 and x["critic_pending"] and x["stop_cause"] == "api_unavailable" for x in rows)
        assert (Path(t) / "critic_packets/pass1_pending.jsonl").is_file()
        final = [json.loads(l) for l in (Path(t) / "selection/selection_log.jsonl").read_text().splitlines()]
        assert len(final) == 7 * 6 and all(x["stop_cause"] != "api_unavailable" and x["label"].startswith("DEVELOPMENT EVIDENCE") for x in final)
        d1 = [x for x in rows if x["arm"] == "D1" and x["pass"] == 2][0]
        for k in ("budget_record", "wall_seconds", "gpu_seconds", "tool_calls", "api_calls", "stop_cause", "journal_hash", "answer_label", "evidence_ids"):
            assert k in d1
        check_budget_schema(d1["budget_record"])
        assert d1["budget_record"]["glm_tokens_out"] == 40 and d1["api_calls"] == 1     # pass-2 replay carries the ingested critic call
    with tempfile.TemporaryDirectory() as t:     # no transport: packets are exported and the run stays pending (nothing invented)
        r = dryrun.make_runner(t, transport=None)
        r.run()
        final = [json.loads(l) for l in (Path(t) / "selection/selection_log.jsonl").read_text().splitlines()]
        assert any(x["critic_pending"] and x["stop_cause"] == "api_unavailable" for x in final)


@test
def grading_only_after_gate_close():
    log = []
    with tempfile.TemporaryDirectory() as t:
        r = dryrun.make_runner(t, gate=OrderGate(log), grader=OrderGrader(log))
        r.run()
        assert log.count("close") == 1 and log[0] == "close" and log.count("grade") > 0
        assert (Path(t) / "grading/grades.jsonl").is_file()
    with tempfile.TemporaryDirectory() as t:
        r = dryrun.make_runner(t)
        raises(I.GraderIsolationError, r.grade_all)             # nothing frozen yet
        r.run_selection()
        raises(I.GraderIsolationError, r.grade_all)             # selected but gate not closed
        art = I.Artifact.make("SYN-P1", "python", "GOOD:SYN-P1:1")
        gg = pilot.GatedGrader(dryrun.FakeGrader(), r.gate, {("SYN-P1", art.artifact_id)})
        raises(I.GraderIsolationError, gg.grade, art)           # gate open
        r.freeze_and_close()
        gg.grade(art)                                           # gate closed and artifact frozen
        raises(I.GraderIsolationError, pilot.GatedGrader(dryrun.FakeGrader(), r.gate, set()).grade, art)   # not a frozen artifact
        raises(I.GraderIsolationError, r.gate.close_selection)  # closing twice
        r.grade_all()
        raises(I.GraderIsolationError, r._one, r.tasks["python"][0], "A", 1, None, pilot._Tally())   # selection after grading is forbidden


def _reachable(root, targets):
    seen, stack, hit = set(), [root], []
    ids = {id(x) for x in targets}
    while stack:
        o = stack.pop()
        if id(o) in seen or isinstance(o, (str, bytes, int, float, bool, type(None))):
            continue
        seen.add(id(o))
        if id(o) in ids:
            hit.append(o)
        if isinstance(o, dict):
            stack.extend(list(o.keys()) + list(o.values()))
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
        elif hasattr(o, "__dict__"):
            stack.extend(vars(o).values())
        for s in getattr(type(o), "__slots__", ()) if not hasattr(o, "__dict__") else ():
            stack.append(getattr(o, s, None))
        if inspect.isfunction(o) or inspect.ismethod(o):
            stack.extend(c.cell_contents for c in (getattr(o, "__closure__", None) or ()))
    return hit


@test
def grade_results_never_reach_selection():
    assert "grader" not in {f.name for f in dataclasses.fields(I.ControllerContext)}
    src = inspect.getsource(pairing.fresh_context).lower()
    assert "grad" not in src, "fresh_context must not mention grading"
    with tempfile.TemporaryDirectory() as t:
        spy = SpyController(dryrun.FakeController())
        r = dryrun.make_runner(t, controller=spy)
        r.run()
        assert r._grades and spy.ctxs
        targets = [r._grades, r._grader, r.gate] + r._grades
        for ctx in spy.ctxs:
            assert not _reachable(ctx, targets), "a controller context can reach protected grading state"
        # sanity: the walker does find things when they are there
        assert _reachable({"x": [r._grades]}, [r._grades])


@test
def pilot_refuses_without_T_k1_authorization():
    with tempfile.TemporaryDirectory() as t:
        e = raises(ConfigError, dryrun.make_runner(t, T=None).run_selection)
        assert "PENDING" in str(e)
        raises(ConfigError, dryrun.make_runner(t, k1=None).run_selection)
        raises(ConfigError, dryrun.make_runner(t, dry_run=True, generation=type("G", (), {"model_info": lambda s: {"fake": False}})()).run_selection)
        assert not (Path(t) / "selection").exists()
        r = dryrun.make_runner(t, dry_run=False, authorization_manifest=None)
        raises(I.NotAuthorized, r.run_selection)                # default: nothing authorized, no generation attempted
        assert not r._rows_all
        thr = _sandbox_toml(t)
        p = authorization.write_authorization_manifest({**FULL, "nvidia_credits_confirmed": False}, t, thr)
        raises(I.NotAuthorized, dryrun.make_runner(t, dry_run=False, authorization_manifest=p, thresholds_path=thr).run_selection)
        p = authorization.write_authorization_manifest(FULL, t, thr)   # fabricated approvals, test sandbox only
        r = dryrun.make_runner(t, dry_run=False, authorization_manifest=p, thresholds_path=thr)
        r.run_selection()
        assert r._rows_all


@test
def dryrun_reproducible_and_claims_no_accuracy():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ha, hb = dryrun.run_dryrun(a), dryrun.run_dryrun(b)
        assert ha == hb and "report.json" in ha and "grading/grades.jsonl" in ha and "DRYRUN_MANIFEST.json" in ha
        rep = json.loads((Path(a) / "report.json").read_text())
        assert rep["banner"].startswith("SYNTHETIC") and rep["tracks"]["python"]["contrasts"]
        assert all("rate" not in v for v in rep["tracks"]["python"]["per_arm"].values())
        m = json.loads((Path(a) / "DRYRUN_MANIFEST.json").read_text())
        assert m["claims_accuracy"] is False
        assert str(a) not in (Path(a) / "report.json").read_text() + (Path(a) / "selection/frozen_artifacts.json").read_text()


def main():
    failed = {}
    for f in TESTS:
        try:
            f()
        except Exception:
            failed[f.__name__] = traceback.format_exc(limit=4)
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
