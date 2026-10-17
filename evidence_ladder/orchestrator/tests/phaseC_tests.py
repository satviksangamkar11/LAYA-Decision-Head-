"""[EL-ARCH-1 Phase C] Plain-assert tests for C1 (probe refit, proper random head, role labels), C2 (arm E wiring), C3 (monotonicity gate), C5 (gates, budget), C6 (oracle isolation).
Run from the repo root: python evidence_ladder/orchestrator/tests/phaseC_tests.py   (prints {"tests","passed","failed"}; exit 1 on failure).
EVERYTHING that stands in for a model (fake LAYA backend, synthetic pools, synthetic tensors, scripted controller) is LOGIC-ONLY: it proves wiring and rules, never a measured result.
No network, no keys, no Kaggle. Needs torch + sklearn (CPU, ATEN_CPU_CAPABILITY=avx2)."""
import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evidence_ladder" / "controller" / "tests"))

import run_tests as RT                                                       # noqa: E402  controller test fakes
from evidence_ladder.common.interfaces import Artifact, ControllerContext, GraderIsolationError, UnsupportedDecisionType, sha256_text   # noqa: E402
from evidence_ladder.controller import invariants as inv                     # noqa: E402
from evidence_ladder.controller import monotonicity as MONO                  # noqa: E402
from evidence_ladder.controller.journal import EvidenceJournal               # noqa: E402
from evidence_ladder.controller.next_action import decide_next_action        # noqa: E402
from evidence_ladder.models import laya_forward as LF                        # noqa: E402
from evidence_ladder.models import laya_runtime as RTM                       # noqa: E402
from evidence_ladder.models import probe as PR                               # noqa: E402
from evidence_ladder.models import random_head as RH                         # noqa: E402
from evidence_ladder.models.laya import LayaScorer, ProbeScorer, RandomHeadScorer   # noqa: E402
from evidence_ladder.orchestrator import arms, budget, comparison, dryrun, gates, isolation, report, stats   # noqa: E402
from evidence_ladder.orchestrator.base import ConfigError                    # noqa: E402
from evidence_ladder.orchestrator.program_scorer import ProgramPoolScorer, ScorerRefusedPool   # noqa: E402

TESTS = []


def test(f):
    TESTS.append(f)
    return f


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected " + exc.__name__)


def system_block():
    b = sorted((ROOT / "results/raw").glob("el-laya-int1-bundle-*"))[-1]
    return (b / "gate/system_block.txt").read_text("utf-8")


def synth_rows(n_pools=12, n_opt=3, seed=0):
    """LOGIC-ONLY synthetic TRAIN pools: positive option = last one, with a separable shift."""
    import torch
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n_pools):
        ho = torch.randn(n_opt, 3, LF.HIDDEN, generator=g).half()
        ho[-1] += 2.0
        rows.append({"decision_id": "p%d" % i, "ho": ho, "pos": torch.tensor([False] * (n_opt - 1) + [True])})
    return rows


# =============================================================== C1
@test
def c1_probe_refit_is_deterministic_and_checks_its_inputs():
    import torch
    rows = synth_rows()
    a, b = PR.refit_probe(rows, seed=0), PR.refit_probe(rows, seed=0)
    assert a["fingerprint"] == b["fingerprint"] and a["order_sha256"] == b["order_sha256"] and a["role"] == PR.ROLE
    assert a["n_pools"] == 12 and a["n_options"] == 36 and a["C"] == PR.PROBE_C
    ho = rows[0]["ho"]
    sc = PR.probe_scores(a, ho)
    assert sc["role"] == PR.ROLE and max(range(3), key=lambda i: sc["score"][i]) == 2          # the planted shift is found
    raises(PR.ProbeFitError, PR.refit_probe, rows, expect_pools=PR.TRAIN_POOLS)                # exactly the 334 TRAIN pools are required for the real refit
    raises(PR.ProbeFitError, PR.refit_probe, [])
    bad = [dict(rows[0], ho=rows[0]["ho"][:, :2])]
    raises(PR.ProbeFitError, PR.refit_probe, bad)                                              # wrong depth count
    raises(PR.ProbeFitError, PR.refit_probe, [dict(rows[0], pos=torch.zeros(3, dtype=torch.bool))])   # no positive option
    assert PR.order_sha(rows) != PR.order_sha(list(reversed(rows)))                            # the row order is part of the fit's identity and is recorded


@test
def c1_random_head_is_untrained_fixed_seeded_and_loads_nothing():
    import torch
    import safetensors.torch as st
    real_load, real_tload = st.load_file, torch.load

    def boom(*a, **k):
        raise AssertionError("a checkpoint was loaded")
    st.load_file, torch.load = boom, boom
    try:
        h1, h1b, h2 = RH.build_random_head(11), RH.build_random_head(11), RH.build_random_head(12)
    finally:
        st.load_file, torch.load = real_load, real_tload
    assert RH.state_sha256(h1) == RH.state_sha256(h1b) != RH.state_sha256(h2)                  # fixed seeds: reproducible, seeds differ
    assert h1.cfg["hidden"] == LF.HIDDEN and h1.cfg["k_layers"] == 3 and h1.cfg["layers"] == 2 and h1.cfg["scorer"] == "mlp"   # same architecture as the shipped head
    assert RH.ROLE != LF.ROLES["laya"] and RH.OLD_CONTROL_LABEL.endswith("MISMATCH_CONTROL")
    pre = ROOT / "models/coding-decision-agent/model.safetensors"
    if pre.exists():       # no pretrained tensor leaks in
        w = real_load(str(pre))["scorer.0.weight"].float() if "scorer.0.weight" in real_load(str(pre)) else None
        if w is not None:
            assert not torch.equal(h1.mlp[0].weight.detach().float(), w)
    assert float(h1.kind_emb.weight.abs().sum()) == 0.0 and float(h1.mu_q.abs().sum()) == 0.0 and float(h1.sd_q.min()) == 1.0   # identity scaler, nothing fitted


@test
def c1_random_head_scores_have_role_seeds_and_the_laya_shape():
    import torch
    hs = {"random-s%d" % s: RH.build_random_head(s) for s in RH.RANDOM_SEEDS}
    hq = torch.randn(3, LF.HIDDEN).half()
    ho = torch.randn(4, 3, LF.HIDDEN).half()
    r = RH.random_head_scores(hs, hq, ho)
    assert len(r["score"]) == 4 and r["role"] == RH.ROLE and r["seeds"] == sorted(RH.RANDOM_SEEDS) and set(r["per_seed_logits"]) == set(hs)
    assert all(abs(r["score"][i] - sum(v[i] for v in r["per_seed_logits"].values()) / 3) < 1e-9 for i in range(4))


@test
def c1_feature_scorer_labels_every_output_with_its_role_and_takes_a_seeded_random_head_set():
    import torch
    heads = {"final-s0": RH.build_random_head(1), "final-s1": RH.build_random_head(2), "final-s2": RH.build_random_head(3)}     # LOGIC-ONLY stand-ins for the frozen heads
    fit = PR.refit_probe(synth_rows())
    fs = RTM.FeatureScorer(heads, probe=fit["probe"], random_head={"random-s11": RH.build_random_head(11)})
    out = fs._score(torch.randn(3, LF.HIDDEN).half(), synth_rows(1)[0]["ho"])
    assert {k: v["role"] for k, v in out.items()} == LF.ROLES and len(out["random_head"]["per_seed_logits"]) == 1
    old = RTM.FeatureScorer(heads, random_head=RH.build_random_head(1))._score(torch.randn(3, LF.HIDDEN).half(), synth_rows(1)[0]["ho"])
    assert old["random_head"]["role"] == LF.ROLES["random_head"] and "per_seed_logits" not in old["random_head"]    # the old single-head path still works


# =============================================================== C2
class FakeBackend:
    """LOGIC-ONLY stand-in for LiveBackend / VerifiedSavedScoreBackend: refuses anything but {context_text, options}; score = option length (so scores differ per candidate)."""
    def __init__(self, f=lambda o: float(len(o))):
        self.f, self.requests = f, []

    def score(self, decision_type, pool):
        req = pool[0] if len(pool) == 1 and isinstance(pool[0], dict) else None
        if req is None or "context_text" not in req:
            raise RTM.ProvenanceError("bare candidates refused")
        self.requests.append(req)
        return [self.f(o) for o in req["options"]]


def laya(f=lambda o: float(len(o))):
    return LayaScorer(authorized=True, backend=FakeBackend(f), expected_capture=())


def comparison_set():
    return {"probe": ProbeScorer(authorized=True, backend=FakeBackend(lambda o: -float(len(o))), expected_capture=()),
            "random_head": RandomHeadScorer(authorized=True, backend=FakeBackend(lambda o: float(sum(map(ord, o)) % 7)), expected_capture=())}


@test
def c2_controller_scores_rendered_programs_and_comparison_scorers_never_select():
    sink = []
    inner = laya(lambda o: 5.0 if "TIMEOUT" in o else 0.0)
    ps = ProgramPoolScorer(inner, RT.TASK, system_block=system_block(), comparison=comparison_set(), sink=sink)
    res, ctl, _, _ = RT.run([["OK:a", "TIMEOUT:b"]], {"k": 2, "scorer": True, "exploratory": True}, scorer=ps)
    sc = ctl.last_journal.entries("scorer")
    assert [e["data"]["state"] for e in sc] == ["scored"] and sc[0]["data"]["exploratory"] is True                  # F08: the controller journals scorer: scored
    assert res.final_artifact.source == "OK:a"                                                                      # I2: trusted evidence still wins over the LAYA score
    req = inner._backend.requests[0]
    assert req["context_text"].startswith(system_block()) and "Next action" not in req["context_text"]           # cue is appended by the capture, not the renderer
    assert len(req["options"]) == 2 and all(json.loads(o)["name"] == "Write" and json.loads(o)["arguments"]["file_path"] == "solution.py" for o in req["options"])
    ev = sink[0]
    assert ev["label"] == LF.EXPLORATORY_LABEL and ps.label == LF.EXPLORATORY_LABEL
    roles = {k: v["role"] for k, v in ev["scorers"].items()}
    assert roles == {"laya_adapter_v1": "scorer_under_test", "probe_v1": "mandatory_baseline_linear_probe", "random_head_v1": "untrained_random_init_floor_control"}
    rk = ctl.last_journal.entries("ranking")[-1]["data"]
    laya_scores = {i["id"]: i["scorer"] for i in rk["items"]}
    assert set(laya_scores.values()) <= {0.0, 5.0}                                                                  # only the LAYA score entered the ranking items; probe / random scores did not
    assert ps.last_render["n_distinct"] == 2 and ps.last_render["label"] == LF.EXPLORATORY_LABEL


@test
def c2_rendering_refusals_are_explicit_and_not_exploratory_is_refused():
    ps = ProgramPoolScorer(laya(), RT.TASK, system_block=system_block())
    raises(ScorerRefusedPool, ps.score, "complete_program", ["same", "same"], True)             # fewer than 2 distinct programs: not scored, no NONE option added
    raises(UnsupportedDecisionType, ps.score, "complete_program", ["a", "b"], False)            # stage-1 LAYA runs only as a flagged exploratory transfer test
    raises(ScorerRefusedPool, ProgramPoolScorer, LayaScorer(authorized=True, backend=LF.SavedScoreBackend([], "laya"), expected_capture=()), RT.TASK, system_block=system_block())
    raises(ValueError, ProgramPoolScorer(laya(), RT.TASK, system_block="not the training system block").score, "complete_program", ["a", "b"], True)   # hash-pinned
    bare = RTM.LiveBackend(live=None)
    raises(RTM.ProvenanceError, bare.score, "complete_program", ["a", "b"])                     # the live backend still refuses bare strings: this wrapper is the only way in
    res, ctl, _, _ = RT.run([["OK:a", "OK:a2"]], {"k": 2, "scorer": True, "exploratory": True}, scorer=ProgramPoolScorer(LayaScorer(authorized=False, expected_capture=()), RT.TASK, system_block=system_block()))
    un = ctl.last_journal.entries("scorer_unavailable")
    assert un and "NotAuthorized" in un[0]["data"]["reason"] and res.stop_cause == "decided"   # unauthorized scorer: explicit journal entry, run continues


@test
def c2_duplicate_sources_share_a_score_and_comparison_failures_are_recorded_not_invented():
    sink = []
    comp = {"probe": ProbeScorer(authorized=False, expected_capture=())}
    ps = ProgramPoolScorer(laya(lambda o: float(len(o))), RT.TASK, system_block=system_block(), comparison=comp, sink=sink)
    out = ps.score("complete_program", ["aa", "bbbb", "aa"], True)
    assert out[0] == out[2] and out[1] != out[0] and len(sink[0]["distinct_code_sha256"]) == 2
    p = sink[0]["scorers"]["probe_v1"]
    assert p["status"] == "unavailable" and "NotAuthorized" in p["reason"] and "scores" not in p


@test
def c2_stage_one_has_no_laya_keep_drop_anywhere():
    raises(arms.StageOneLayaDecisionRefused, arms.laya_keep_drop)
    raises(arms.StageOneLayaDecisionRefused, gates.step_gate, {}, [], "E")
    assert arms.arm_label("E") == arms.E_LABEL and arms.E_SELECTOR_KINDS == ("laya",) and "laya" not in arms.E_COMPARISON_KINDS
    rep = comparison.comparison_report([], {})
    assert rep["keep_drop"] is None and rep["selectors"] == [] and "no pool was scored" in rep["note"]


@test
def c2_next_action_journals_comparison_scores_without_using_them():
    j = EvidenceJournal()
    r = decide_next_action(j, "d", ["x", "y", "z"], [1.0, 2.0, 3.0], None, scorer_id="laya_adapter_v1",
                           comparison_scores={"probe": [9.0, 0.0, 0.0], "random_head": [0.0, 9.0, 0.0]})
    assert r["selected_index"] == 2 and inv.check_i2(j) == ""                                  # the order comes from the LAYA score alone
    e = j.entries("scorer_comparison")[0]["data"]
    assert e["selector"] is False and e["scores"]["probe"]["role"] == LF.ROLES["probe"] and e["scores"]["random_head"]["role"] == LF.ROLES["random_head"]
    raises(ValueError, decide_next_action, EvidenceJournal(), "d", ["x", "y"], [1.0, 2.0], None, comparison_scores={"laya": [1.0, 2.0]})
    raises(ValueError, decide_next_action, EvidenceJournal(), "d", ["x", "y"], [1.0, 2.0], None, comparison_scores={"probe": [1.0]})
    j2 = EvidenceJournal()
    decide_next_action(j2, "d", ["x", "y", "z"], [1.0, 2.0, 3.0], None, scorer_id="laya_adapter_v1")
    assert not j2.entries("scorer_comparison")                                                # no comparison given: the journal is exactly as before


@test
def c2_pilot_wiring_end_to_end_with_scripted_controller_logic_only():
    class Tol(dryrun.FakeController):
        def run_problem(self, task, ctx):
            inner = ctx.scorer

            class P:
                scorer_id = getattr(inner, "scorer_id", "x")

                def score(self, dt, pool, exploratory=False):
                    try:
                        return inner.score(dt, pool, exploratory)
                    except ScorerRefusedPool:
                        return [0.0] * len(pool)
            if ctx.scorer is not None:
                import dataclasses
                ctx = dataclasses.replace(ctx, scorer=P())
            return super().run_problem(task, ctx)
    with tempfile.TemporaryDirectory() as d:
        r = dryrun.make_runner(d, controller=Tol(), scorer=laya(), system_block=system_block(), comparison_scorers=comparison_set())
        info = r.run()
        assert r.scorer_events and all(e["label"] == LF.EXPLORATORY_LABEL for e in r.scorer_events)
        cs = info["report"]["comparison_scorers"]
        assert cs["keep_drop"] is None and cs["selectors"] == [] and set(cs["scorers"]) <= {"laya_adapter_v1", "probe_v1", "random_head_v1"} and cs["scorers"]
        assert (Path(d) / "selection" / "scorer_events.jsonl").exists() and (Path(d) / "grading" / "comparison_candidates.jsonl").exists()
        md = (Path(d) / "report.md").read_text("utf-8")
        assert "Comparison scorers (not selectors; no keep/drop)" in md and "NOT sealed" in md
    r2 = dryrun.make_runner(tempfile.mkdtemp(), scorer=dryrun.FakeScorer())     # without a system block the scorer is passed through unchanged (old behaviour)
    assert r2._scorer_for(None).__class__ is dryrun.FakeScorer
    r3 = dryrun.make_runner(tempfile.mkdtemp())
    raises(GraderIsolationError, r3.grade_comparison_candidates)                # comparison outcomes are graded only after the gate closed and grading ran


# =============================================================== C3
def pools(n, rng, kind):
    out = []
    for i in range(n):
        sc = [rng.random() for _ in range(4)]
        order = sorted(range(4), key=lambda j: -sc[j])
        p = {"mono": [0.9, 0.6, 0.3, 0.1], "anti": [0.1, 0.3, 0.6, 0.9], "flat": [0.5] * 4}[kind]
        passed = [False] * 4
        for rank, j in enumerate(order):
            passed[j] = rng.random() < p[rank]
        out.append({"id": "q%d" % i, "scores": sc, "passed": passed})
    return out


@test
def c3_three_examples_give_insufficient_data_and_ordering_stays_disabled():
    g = MONO.monotonicity_gate([{"id": "a", "scores": [1, 2], "passed": [True, False]}, {"id": "b", "scores": [2, 1], "passed": [True, False]}, {"id": "c", "scores": [1, 2], "passed": [False, True]}])
    assert g["status"] == MONO.INSUFFICIENT and g["laya_ordering_enabled"] is False and g["can_override_trusted_evidence"] is False
    v = MONO.verification_order(g, [{"id": "x", "trusted_passes": 0, "cost": 1, "scorer": 9.0}, {"id": "y", "trusted_passes": 0, "cost": 5, "scorer": 0.0}])
    assert v["order"] == ["x", "y"] and v["laya_used"] is False and v["fallback"] == "cheapest_check_first"      # cheapest-check-first, the score is ignored
    v2 = MONO.verification_order(g, [{"id": "x", "trusted_passes": 0, "cost": 5, "scorer": 9.0}, {"id": "y", "trusted_passes": 0, "cost": 1, "scorer": 0.0}])
    assert v2["order"] == ["y", "x"]
    assert MONO.verification_order(None, [])["gate_status"] == "NOT_RUN"


@test
def c3_synthetic_larger_n_supported_not_supported_and_ordering_follows_the_gate():
    rng = random.Random(7)
    ok = MONO.monotonicity_gate(pools(200, rng, "mono"), seed=1)                     # LOGIC-ONLY synthetic pools
    assert ok["status"] == MONO.SUPPORTED and ok["laya_ordering_enabled"] is True and ok["auc_ci"][0] > 0.5 and ok["can_override_trusted_evidence"] is False
    bad = MONO.monotonicity_gate(pools(200, rng, "anti"), seed=1)
    assert bad["status"] == MONO.NOT_SUPPORTED and bad["laya_ordering_enabled"] is False
    flat = MONO.monotonicity_gate(pools(200, rng, "flat"), seed=1)
    assert flat["status"] == MONO.NOT_SUPPORTED
    assert MONO.monotonicity_gate(pools(200, random.Random(7), "mono"), seed=1)["auc"] == ok["auc"]    # deterministic
    items = [{"id": "hi", "trusted_passes": 0, "cost": 9, "scorer": 9.0}, {"id": "lo", "trusted_passes": 0, "cost": 1, "scorer": 0.0}]
    assert MONO.verification_order(ok, items)["order"] == ["hi", "lo"] and MONO.verification_order(bad, items)["order"] == ["lo", "hi"]
    few = MONO.monotonicity_gate(pools(20, rng, "mono"))
    assert few["status"] == MONO.INSUFFICIENT


@test
def c3_laya_never_overrides_trusted_evidence_in_any_gate_state():
    items = [{"id": "trusted", "trusted_passes": 2, "cost": 9, "scorer": -5.0}, {"id": "scored", "trusted_passes": 0, "cost": 1, "scorer": 99.0}]
    for gate in (None, {"status": MONO.INSUFFICIENT}, {"status": MONO.NOT_SUPPORTED}, {"status": MONO.SUPPORTED, "laya_ordering_enabled": True},
                 {"status": MONO.SUPPORTED, "laya_ordering_enabled": False}):          # a SUPPORTED status alone, without the enabled flag, does not switch it on
        assert MONO.verification_order(gate, items)["order"][0] == "trusted"
    assert MONO.verification_order({"status": MONO.SUPPORTED, "laya_ordering_enabled": False}, items[::-1])["laya_used"] is False
    # the per-action decision (I2): the same property in next_action.decide_next_action via C.rank
    j = EvidenceJournal()
    r = decide_next_action(j, "d", ["a", "b"], [-9.0, 99.0], [{"L5": "pass"}, {}], scorer_id="laya_adapter_v1")
    assert r["selected_index"] == 0 and inv.check_i2(j) == ""


# =============================================================== C5
def graded(n_tasks, f_step, f_b, label="SUPPORTED"):
    res, rows = {}, []
    for i in range(n_tasks):
        t = "t%d" % i
        res[t] = {"B": {"passed": f_b(i)}, "D0": {"passed": f_step(i)}, "C": {"passed": f_step(i)}, "D1": {"passed": f_step(i)}}
        for a in ("B", "C", "D0", "D1"):
            rows.append({"task_id": t, "arm": a, "answer_label": label, "budget_record": {"qwen_gen_tokens": 1000, "qwen_prefill_tokens": 500, "glm_tokens_in": 0, "glm_tokens_out": 0}})
    return res, rows


@test
def c5_gates_file_is_versioned_and_loud():
    cfg = gates.load_gates()
    assert cfg["min_n"]["step_pairs"] >= 30 and cfg["g0"]["headroom_points"] == 5.0 and "reason" in cfg
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.toml"
        p.write_text("[min_n]\nstep_pairs = 3\n", encoding="utf-8")
        raises(ConfigError, gates.load_gates, p)
    assert (ROOT / "configs/thresholds_evidence_ladder_v1.toml.sha256").exists()


@test
def c5_three_problems_every_statistical_gate_is_insufficient_never_pass():
    res, rows = graded(3, lambda i: True, lambda i: False)
    g = gates.section_gates(res, rows, headroom_samples=[[False, True]] * 3, invariant_results={k: True for k in gates.INVARIANTS})
    assert g["g0"]["status"] == gates.INSUFFICIENT and all(s["status"] == gates.INSUFFICIENT for s in g["steps"].values())
    assert all(s["status"] == gates.INSUFFICIENT for s in g["safety"].values()) and g["invariants"]["status"] == gates.PASS
    assert g["overall"] == gates.INSUFFICIENT and "PASS" not in {g["g0"]["status"]} | {s["status"] for s in g["steps"].values()}
    assert gates.g0_headroom([])["status"] == gates.INSUFFICIENT
    assert gates.invariants_gate(None)["status"] == gates.FAIL and gates.invariants_gate({"I1": True})["not_passing"][0] == "I2"       # not statistical: missing = FAIL
    assert comparison.comparison_report([], {})["scorers"] == {}


@test
def c5_synthetic_larger_n_gates_give_verdicts():
    cfg = gates.load_gates()
    n = 60
    keep = gates.step_gate(*graded(n, lambda i: i % 2 == 0 or i % 3 == 0, lambda i: i % 2 == 0 and i % 3 != 0), "D0")    # LOGIC-ONLY synthetic outcomes
    assert keep["status"] == gates.KEEP and keep["completion_gain"]["lo"] > 0 and keep["compute_match"]["matched"]
    drop = gates.step_gate(*graded(n, lambda i: i % 2 == 0, lambda i: i % 2 == 0), "D0")
    assert drop["status"] == gates.DROP
    res, rows = graded(n, lambda i: True, lambda i: False)
    for r in rows:
        if r["arm"] == "D0":
            r["budget_record"]["qwen_gen_tokens"] = 5000      # twin spent far less: not equal-compute
    nm = gates.step_gate(res, rows, "D0")
    assert nm["status"] == gates.INSUFFICIENT and "not equal-compute" in nm["reason"]
    h = gates.g0_headroom([[False, True]] * 20 + [[True, True]] * 20 + [[False, False]] * 20)
    assert h["status"] == gates.PASS and abs(h["headroom_points"] - 100 * 20 / 60) < 1e-9
    assert gates.g0_headroom([[True, True]] * 40)["status"] == gates.FAIL
    # safety: false-approval rate no higher than B's
    res, rows = graded(40, lambda i: i % 10 != 0, lambda i: i % 4 != 0)
    assert gates.safety_gate(res, rows, "D0")["status"] == gates.PASS
    res, rows = graded(40, lambda i: i % 2 == 0, lambda i: i % 10 != 0)
    assert gates.safety_gate(res, rows, "D0")["status"] == gates.FAIL
    assert gates.overall([gates.PASS, gates.INSUFFICIENT]) == gates.INSUFFICIENT and gates.overall([gates.INSUFFICIENT, gates.FAIL]) == gates.FAIL


@test
def c5_false_approval_reduction_keeps_a_step_that_adds_no_completion():
    n = 80
    res, rows = {}, []
    for i in range(n):
        t = "t%d" % i
        same = i % 2 == 0
        res[t] = {"B": {"passed": same}, "D1": {"passed": same}}
        rows.append({"task_id": t, "arm": "B", "answer_label": "SUPPORTED", "budget_record": {"qwen_gen_tokens": 100, "qwen_prefill_tokens": 0, "glm_tokens_in": 0, "glm_tokens_out": 0}})
        rows.append({"task_id": t, "arm": "D1", "answer_label": "SUPPORTED" if same else "BLOCKED", "budget_record": {"qwen_gen_tokens": 100, "qwen_prefill_tokens": 0, "glm_tokens_in": 0, "glm_tokens_out": 0}})
    out = gates.step_gate(res, rows, "D1")
    assert out["status"] == gates.KEEP and "false-approval" in out["reason"]


@test
def c5_matched_token_accounting_counts_inputs_and_applies_the_extra_b_rule():
    b = budget.MatchedTokenBudget("p", "D1", T=10_000)
    b.charge("qwen_gen", 100); b.charge("qwen_prefill", 400); b.charge("glm_in", 300); b.charge("glm_out", 50)
    assert b.generated() == 150 and b.total_incl_inputs() == 850 and b.sidecar()["total_tokens_incl_inputs"] == 850 and gates.total_tokens(b.record()) == 850
    plan = budget.extra_b_samples_plan([{"problem_id": "t", "glm_tokens_out": 400, "glm_tokens_in": 900}, {"problem_id": "z", "glm_tokens_out": 0, "glm_tokens_in": 0}],
                                       [{"problem_id": "t", "qwen_gen_tokens": 200}, {"problem_id": "z", "qwen_gen_tokens": 200}], 4)
    assert plan == {"t": {"extra_samples": 8, "glm_out_tokens": 400, "mean_qwen_sample_tokens": 50.0, "glm_in_tokens_not_matched": 900}}


@test
def c5_runner_gives_b_the_extra_samples_and_records_it():
    class Heavy(dryrun.FakeController):
        def run_problem(self, task, ctx):
            if ctx.arm == "D1":
                ctx.budget.charge("glm_out", 360)
            return super().run_problem(task, ctx)
    with tempfile.TemporaryDirectory() as d:
        r = dryrun.make_runner(d, controller=Heavy())
        r.run()
        plan = json.loads((Path(d) / "selection" / "b_rebalance.json").read_text("utf-8"))["tasks"]
        assert plan and all(v["extra_applied"] > 0 and v["b_k_after"] == v["b_k_before"] + v["extra_applied"] for v in plan.values())
        allp = [json.loads(l) for l in (Path(d) / "selection" / "selection_log_all_passes.jsonl").read_text("utf-8").splitlines()]
        assert any(x["arm"] == "B" and x["pass"] == 3 for x in allp)
    with tempfile.TemporaryDirectory() as d:        # opt-out keeps the old behaviour
        dryrun.make_runner(d, controller=Heavy(), match_b_to_critic=False).run()
        assert not (Path(d) / "selection" / "b_rebalance.json").exists()


@test
def c5_report_carries_gates_and_never_a_pass_for_three_problems():
    with tempfile.TemporaryDirectory() as d:
        info = dryrun.make_runner(d).run()
    rep = info["report"]
    py = rep["tracks"]["python"]
    assert py["gates"]["overall"] == gates.FAIL and py["gates"]["g0"]["status"] == gates.INSUFFICIENT and py["gates"]["invariants"]["status"] == gates.FAIL   # no unit results supplied: FAIL (never PASS), and FAIL dominates
    assert all(v["status"] == gates.INSUFFICIENT for v in py["gates"]["steps"].values())
    md = report.render_markdown(rep)
    assert "Gates (configs/el_arch1_gates_v1.toml): overall FAIL; G-0 INSUFFICIENT_DATA; I1-I8 FAIL" in md


# =============================================================== C6
@test
def c6_no_runtime_module_imports_or_names_the_hidden_cases():
    assert isolation.static_oracle_scan() == []
    assert inv.static_no_grader_imports() == []                                  # the existing controller-only scan stays green next to the wider one
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "pkg"
        (pkg / "runtime").mkdir(parents=True)
        (pkg / "runtime" / "bad.py").write_text("from evidence_ladder.execution.protected import ProtectedGrader\nP = 'x/grader/" + "protected" + "_cases.json'\n", encoding="utf-8")
        (pkg / "runtime" / "good.py").write_text("import json\n", encoding="utf-8")
        (pkg / "tests").mkdir()
        (pkg / "tests" / "t.py").write_text("from evidence_ladder.execution.protected import ProtectedGrader\n", encoding="utf-8")
        found = isolation.static_oracle_scan(pkg)
        assert len(found) == 3 and all(f.startswith("runtime/bad.py") for f in found), found   # tests are exempt, runtime is not


@test
def c6_deep_check_finds_a_grader_hidden_inside_a_backend_that_the_name_scan_misses():
    from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate

    class Backend:
        def __init__(self, helper):
            self.helper = helper                 # innocent attribute name

        def run_public(self, a, t):
            return []
    gate = SelectionGate()
    evil = Backend(ProtectedGrader(gate, python_cases=Path("nowhere.json"), java_cases=Path("nowhere.json")))
    res, ctl, _, _ = RT.run([["OK:a"]], {"k": 1})
    ctx = ControllerContext(RT.FakeGen([]), evil, RT.FakeBudget(arm="D0"), "D0", None, None, (1,), {})
    assert inv.check_i8(ctl.last_journal, ctx) == ""                             # the field-name scan is blind to it
    assert "ProtectedGrader" in isolation.check_i8_deep(ctx)
    ctx2 = ControllerContext(RT.FakeGen([]), RT.FakeExec(), RT.FakeBudget(arm="D0"), "D0", None, None, (1,), {"path": "results/x/grader/" + "protected" + "_cases.json"})
    assert "names" in isolation.check_i8_deep(ctx2)
    ctx3 = ControllerContext(RT.FakeGen([]), RT.FakeExec(), RT.FakeBudget(arm="D0"), "D0", None, None, (1,), {"k": 2})
    assert isolation.check_i8_deep(ctx3) == ""


@test
def c6_grader_reads_nothing_before_the_gate_closes():
    from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate
    gate = SelectionGate()
    missing = Path(tempfile.gettempdir()) / "definitely_missing_cases_dir" / "cases.json"
    g = ProtectedGrader(gate, python_cases=missing, java_cases=missing)          # construction touches no file
    src = "def solve(d): return d"
    raises(GraderIsolationError, g.grade, Artifact.make("x", "python", src))     # open gate: refused before any case file is opened (a missing file would raise FileNotFoundError)


@test
def c6_reports_state_that_the_development_cases_are_not_sealed():
    s = isolation.NOT_SEALED_STATEMENT
    assert "NOT sealed" in s and "public" in s.lower() and "Git history" in s
    with tempfile.TemporaryDirectory() as d:
        rep = dryrun.make_runner(d).run()["report"]
    assert rep["evaluation_oracle_status"] == s and rep["label"].startswith("DEVELOPMENT EVIDENCE")
    md = report.render_markdown(rep)
    assert md.count("sealed") == s.count("sealed") and "NOT sealed" in md        # the word appears only inside the honest statement
    ok_phrases = ("not sealed", "not_sealed", "not a sealed", "never sealed", "unsealed", "no sealed", "sealed (encrypted", "sealed hidden", "sealed tests", "is not sealed")
    src = [p for p in (ROOT / "evidence_ladder").rglob("*.py") if "tests" not in p.parts and p.name != "isolation.py"]
    off = [(p.name, line.strip()[:120]) for p in src for line in p.read_text("utf-8").splitlines() if "sealed" in line.lower() and not any(k in line.lower() for k in ok_phrases)]
    assert off == [], off      # no runtime / report code speaks of a sealed benchmark except to deny or to name the future step


def main():
    failed = {}
    for f in TESTS:
        try:
            f()
        except Exception:
            failed[f.__name__] = traceback.format_exc(limit=5)
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
