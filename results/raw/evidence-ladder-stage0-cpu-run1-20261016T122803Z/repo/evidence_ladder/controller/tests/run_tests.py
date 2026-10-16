"""Controller tests. Run from the repo root: python3 evidence_ladder/controller/tests/run_tests.py
Plain asserts, deterministic, scripted fakes only (no model, no network). Prints {"tests":N,"passed":N,"failed":{}} and exits 1 on failure. jsonschema is needed (C.validate)."""
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from evidence_ladder.common.contract_import import C                                    # noqa: E402
from evidence_ladder.common.interfaces import (Artifact, BudgetExhausted, ControllerContext, Generation, TaskSpec,   # noqa: E402
                                               UnsupportedDecisionType, sha256_text)
from evidence_ladder.controller import intake as intake_mod, invariants as inv, router as router_mod   # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController                             # noqa: E402
from evidence_ladder.controller.decision import authorize, decide_stage1                              # noqa: E402
from evidence_ladder.controller.evidence import Factory                                               # noqa: E402
from evidence_ladder.controller.journal import EvidenceJournal                                        # noqa: E402
from evidence_ladder.controller.registry import build_registry                                        # noqa: E402

H = sha256_text
TESTS = {}


def test(f):
    TESTS[f.__name__] = f
    return f


# ------------------------------------------------------------------ scripted fakes
class FakeGen:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate(self, req):
        self.calls.append(req)
        codes = self.script.pop(0) if self.script else []
        return [Generation(text=c, code=c, extract_status="ok" if c else "no_code", tokens_prompt=10, tokens_generated=20, seed=s, model="fake",
                           model_revision="r", prompt_template_id=req.prompt_template_id, prompt_sha256=H(req.prompt)) for c, s in zip(codes, req.seeds)]

    def model_info(self):
        return {"repo": "fake"}


class FakeExec:
    """Candidate code is a behaviour tag: OK*, WRONG:<obs>:*, PARTIAL, TIMEOUT*, SANDBOX, CRASH."""
    def __init__(self, exp_table=None, flip=None, hook=None):
        self.exp_table, self.flip, self.hook, self.calls, self.exp_calls = exp_table or {}, flip or {}, hook, {}, []

    def _ex(self, code, i, t):
        base = {"check_id": "public_examples_v1", "example_index": i}
        if code.startswith("OK"):
            return dict(base, status="pass", observed=t["expected"])
        if code.startswith("WRONG"):
            return dict(base, status="fail", observed=code.split(":")[1], failure_class="wrong_answer")
        if code == "PARTIAL":
            return dict(base, status="pass", observed=t["expected"]) if i == 0 else dict(base, status="fail", observed="p", failure_class="wrong_answer")
        if code.startswith("TIMEOUT"):
            return dict(base, status="timeout", failure_class="timeout")
        if code == "SANDBOX":
            return dict(base, status="error", failure_class="sandbox_error")
        return dict(base, status="error", failure_class="runtime_error")

    def run_public(self, art, task):
        if self.hook:
            self.hook(art)
        n = self.calls[art.source] = self.calls.get(art.source, 0) + 1
        code = "WRONG:flaky:0" if self.flip.get(art.source) == n else art.source
        return [{"check_id": "parse_v1", "status": "pass"}] + [self._ex(code, i, t) for i, t in enumerate(task.public_tests)]

    def run_experiment(self, art, experiment):
        self.exp_calls.append(art.artifact_id)
        out = dict({"check_id": "critic_experiment_v1", "status": "pass", "observed": "x"}, **self.exp_table.get(art.source, {}))
        return {k: v for k, v in out.items() if v is not None}

    def backend_info(self):
        return {"user": "nobody", "toolchain": "fake-1"}


class FakeBudget:
    def __init__(self, cap=10 ** 9, arm="D0"):
        self.cap, self.gen, self.pre, self.arm = cap, 0, 0, arm

    def charge(self, kind, tokens):
        if kind == "qwen_gen":
            self.gen += tokens
            if self.gen > self.cap:
                raise BudgetExhausted()
        elif kind == "qwen_prefill":
            self.pre += tokens

    def exhausted(self):
        return self.gen >= self.cap

    def record(self):
        return {"problem_id": "p", "arm": self.arm, "qwen_gen_tokens": self.gen, "qwen_prefill_tokens": self.pre, "glm_tokens_in": 0, "glm_tokens_out": 0,
                "glm_calls": 0, "scorer_calls": 0, "tool_runs": 0, "gpu_seconds": 0.0, "wall_seconds": 0.0, "api_errors": 0}


class FakeCritic:
    def __init__(self, done=None):
        self.built, self.pending, self.done = [], [], dict(done or {})

    def build_packet(self, req):
        self.built.append(req)
        self.pending.append("pk-" + req["request_id"])          # the broker's packet id differs from the controller's request id
        return {"packet_id": "pk-" + req["request_id"], "request": req, "created": True, "new": True}

    def completed(self, packet_id):
        return self.done.get(packet_id)

    def export_pending(self):
        return list(self.pending)

    def ingest(self, packet_id, http_status, raw_response):
        return {"state": "ok", "response": raw_response}


class FakeScorer:
    scorer_id = "fake_laya"

    def __init__(self):
        self.calls = []

    def score(self, decision_type, pool, exploratory=False):
        self.calls.append((decision_type, exploratory))
        if decision_type != "next_action" and not exploratory:
            raise UnsupportedDecisionType(decision_type)
        return [1.0 if "TIMEOUT" in p else 0.0 for p in pool]


TASK = TaskSpec("T1", "py1:T1", "python", "python", "Double the number. Define solve.",
                [{"input": "1", "expected": "2"}, {"input": "2", "expected": "4"}])


def run(script, cfg=None, arm="D0", exe=None, critic=None, scorer=None, budget=None, seeds=(1, 2, 3, 4, 5, 6, 7, 8), task=TASK, ctl=None):
    gen, exe = FakeGen(script), exe or FakeExec()
    ctl = ctl or EvidenceLadderController()
    ctx = ControllerContext(gen, exe, budget or FakeBudget(arm=arm), arm, critic, scorer, seeds, dict(cfg or {}))
    res = ctl.run_problem(task, ctx)
    return res, ctl, gen, exe


def labels(res):
    return {r["requirement_id"]: r["label"] for r in res.decision["requirement_rows"]}


def stop_entry(ctl):
    return ctl.last_journal.entries("stop")[-1]["data"]


# ------------------------------------------------------------------ ladder behaviour
@test
def correct_first_attempt():
    res, ctl, gen, _ = run([["OK:a", "WRONG:a:1"]], {"k": 2})
    assert res.stop_cause == "decided" and res.rounds == 0 and len(gen.calls) == 1
    assert res.final_artifact.source == "OK:a" and res.final_artifact_id == H("OK:a")
    assert res.answer_label == "SUPPORTED" and not res.decision["authorized_to_execute"]       # I1: stage 1 can never be VERIFIED
    assert labels(res) == {"R1": "SUPPORTED", "R2": "SUPPORTED", "R3": "VERIFIED", "R4": "VERIFIED"}
    assert res.decision["derived_by"] == "controller" and res.evidence_ids and res.stop_details == ""
    C.validate("decision_record", res.decision)
    assert ctl.last_journal.verify_chain() and res.journal_hash == ctl.last_journal.journal_hash()


@test
def wrong_then_successful_repair():
    res, ctl, gen, _ = run([["WRONG:a:0"], ["OK:fix"]], {"k": 1})
    assert res.stop_cause == "decided" and res.rounds == 1 and res.final_artifact.source == "OK:fix"
    assert res.final_artifact.parent_id == H("WRONG:a:0")
    rq = gen.calls[1]
    assert rq.purpose == "repair" and rq.n == 1
    assert "Previous candidate: " + H("WRONG:a:0") in rq.prompt and "1 -> expected 2, observed a" in rq.prompt and "Verified failure (level L3)" in rq.prompt
    first = [e for e in ctl.last_journal.entries("label") if e["data"]["candidate_id"] == H("WRONG:a:0")][0]["data"]
    assert first["answer_label"] == "BLOCKED"


@test
def stale_evidence_after_repair():
    res, ctl, _, _ = run([["WRONG:a:0"], ["OK:fix"]], {"k": 1})
    j, parent, child = ctl.last_journal, H("WRONG:a:0"), H("OK:fix")
    pe, ce = j.evidence(parent), j.evidence(child)
    assert pe and ce and {e["evidence_id"] for e in pe}.isdisjoint({e["evidence_id"] for e in ce})
    assert all(e["artifact_hash"] == child and e["candidate_id"] == child for e in ce)
    run_ = ctl.last_run
    d = decide_stage1(run_.reqs, pe, child, child, run_.env, run_.registry)          # S2-T1: the parent's records say nothing about the child
    assert d["answer_label"] == "UNVERIFIED" and not d["authorized_to_execute"]
    assert all(r["evidence_ids"] == [] for r in d["requirement_rows"])


@test
def conflicting_trusted_evidence_blocks():
    # unit: two independent trusted records disagree on one artifact -> CONFLICTED, never approved
    res, ctl, _, _ = run([["OK:a"]], {"k": 1})
    run_ = ctl.last_run
    art = run_.cands[H("OK:a")].art
    fac = Factory(run_.registry, run_.reqs, run_.env)
    fac.n = 500
    ev = ctl.last_journal.evidence(art.artifact_id)
    ev.append(fac.make(art, "R3", "public_examples_v1", "existing_test", "independent", "fail", "second trusted check", {"status": "fail"}))
    d = decide_stage1(run_.reqs, ev, art.artifact_id, art.artifact_id, run_.env, run_.registry)
    assert labels_of(d)["R3"] == "CONFLICTED" and d["answer_label"] == "BLOCKED" and not d["authorized_to_execute"]
    # end to end: the sandbox contradicts itself between the run and the final re-check -> block
    res, ctl, _, _ = run([["OK:a"]], {"k": 1}, exe=FakeExec(flip={"OK:a": 2}))
    assert res.answer_label == "BLOCKED" and res.stop_cause == "tool_failure" and res.stop_details.startswith("finalize:")
    assert "CONFLICTED" in labels(res).values()


def labels_of(d):
    return {r["requirement_id"]: r["label"] for r in d["requirement_rows"]}


@test
def same_failure_twice_stops():
    res, ctl, _, _ = run([["WRONG:a:0"], ["WRONG:a:1"]], {"k": 1})
    assert res.stop_cause == "same_failure_twice" and res.rounds == 1 and res.answer_label == "BLOCKED"
    assert any(e["data"]["cause"] == "same_failure_twice" for e in ctl.last_journal.entries("lineage_stopped"))


@test
def no_progress_two_rounds():
    # failures alternate a, b, a, b: round 1 localises a new failure (progress); rounds 2 and 3 repeat seen failures (no progress)
    res, ctl, _, _ = run([["WRONG:a:0"], ["WRONG:b:1"], ["WRONG:a:2"], ["WRONG:b:3"]], {"k": 1, "max_rounds": 5})
    assert res.stop_cause == "no_progress_2" and res.rounds == 3, (res.stop_cause, res.rounds)


@test
def budget_exhaustion_is_never_decided():
    res, ctl, _, _ = run([["OK:a", "OK:b"]], {"k": 2}, budget=FakeBudget(cap=25))
    assert res.stop_cause == "budget" and res.answer_label == "SUPPORTED" and res.final_artifact is not None
    res, ctl, gen, _ = run([["OK:a"]], {"k": 1}, budget=FakeBudget(cap=0))
    assert res.stop_cause == "budget" and not gen.calls and res.final_artifact_id is None and res.answer_label == "UNVERIFIED"
    # budget reached during repair
    res, ctl, _, _ = run([["WRONG:a:0"], ["WRONG:b:1"]], {"k": 1}, budget=FakeBudget(cap=25))
    assert res.stop_cause == "budget" and res.rounds == 1


@test
def tool_failure_not_counted_as_model_failure():
    res, ctl, _, _ = run([["SANDBOX"]], {"k": 1})
    assert res.stop_cause == "tool_failure" and res.answer_label not in ("VERIFIED", "SUPPORTED")


@test
def model_tests_never_verify():
    res, ctl, _, _ = run([["OK:a"]], {"k": 1})
    run_ = ctl.last_run
    art = Artifact.make("T1", "python", "OK:zzz")
    j = EvidenceJournal(run_.registry)
    fac = Factory(run_.registry, run_.reqs, run_.env)
    evs = []
    for rid in ("R3", "R4", "R2", "R1"):
        evs.append(fac.make(art, rid, "qwen_tests_v1", "model_test", "model_derived", "pass", "model-written test", {"status": "pass"}))
        # forged: claims a trusted source but the check is diagnostic and the independence is model_derived
        evs.append(fac.make(art, rid, "qwen_tests_v1", "user_test", "model_derived", "pass", "forged", {"status": "pass"}))
    for e in evs:
        j.add_evidence(e, {"status": "pass"})
    d = decide_stage1(run_.reqs, evs, art.artifact_id, art.artifact_id, run_.env, run_.registry)
    assert d["answer_label"] == "SUPPORTED" and "VERIFIED" not in labels_of(d).values()
    assert inv.check_i3(j) == ""
    # S3-T1 / S3-T4
    bad = dict(evs[0], check_id="not_registered", evidence_id="E9999")
    try:
        j.add_evidence(bad, {})
        assert False
    except KeyError:
        pass
    try:
        run_.registry.runtime_lookup("hidden_tests_v1")
        assert False
    except PermissionError:
        pass


@test
def derived_hypothesis_cannot_gate():
    prop = [{"kind": "derived_hypothesis", "text": "empty input is handled", "rationale": "inferred", "priority": "hard", "status": "validated"},
            {"kind": "explicit", "text": "model claims a quote", "source": {"type": "request_span", "quote": "NOPE", "char_start": 0, "char_end": 4}, "priority": "hard"}]
    res, ctl, _, _ = run([["OK:a"]], {"k": 1, "proposed_requirements": prop})
    reqs = {r["requirement_id"]: r for r in ctl.last_run.reqs}
    assert reqs["R5"]["kind"] == "derived_hypothesis" and reqs["R5"]["status"] == "needs_validation" and not C.can_gate(reqs["R5"])
    assert reqs["R6"]["kind"] == "derived_hypothesis" and reqs["R6"]["status"] == "needs_validation"          # S1-T1
    assert "R5" not in labels(res) and "R6" not in labels(res)
    assert res.stop_cause == "decided" and res.answer_label == "SUPPORTED"


@test
def intake_contradiction_and_ambiguity_stop_before_generation():
    t = TaskSpec("T2", "py1:T2", "python", "python", "Double the number.", [{"input": "1", "expected": "2"}, {"input": "1", "expected": "3"}])
    res, ctl, gen, _ = run([["OK:a"]], {"k": 1}, task=t)
    assert res.stop_cause == "contradictory_requirements" and not gen.calls and res.answer_label == "BLOCKED" and "R3" in res.stop_details and "R4" in res.stop_details
    amb = [{"kind": "explicit", "text": "which rounding?", "source": {"type": "request_span", "quote": "Double", "char_start": 0, "char_end": 6},
            "priority": "functional", "status": "ambiguous", "ambiguity_notes": "round up or down"}]
    res, ctl, gen, _ = run([["OK:a"]], {"k": 1, "proposed_requirements": amb})
    assert res.stop_cause == "missing_information" and not gen.calls and res.answer_label == "BLOCKED"


@test
def router_ceilings_ignore_model_evidence():
    it = intake_mod.build(TASK, [{"kind": "derived_hypothesis", "text": "x", "rationale": "r"}])
    reg = build_registry(it.example_ids)
    t0 = router_mod.route(it.reqs, it.roles, reg, [])
    art = Artifact.make("T1", "python", "OK:q")
    fac = Factory(reg, it.reqs, "e")
    ev = [fac.make(art, "R3", "qwen_tests_v1", "model_test", "model_derived", "pass", "s", {"status": "pass"})]
    t1 = router_mod.route(it.reqs, it.roles, reg, ev, art.artifact_id)
    assert [r["ceiling"] for r in t0] == [r["ceiling"] for r in t1] == ["SUPPORTED", "SUPPORTED", "VERIFIED", "VERIFIED", "RANKING_ONLY"]
    assert t1[2]["soft_evidence"] == 1 and t1[2]["trusted_evidence"] == 0 and router_mod.answer_ceiling(t1) == "SUPPORTED"


# ------------------------------------------------------------------ critic
def crit_rid(codes):
    return "crit-" + C.sha256("T1|" + "|".join(sorted(H(c) for c in codes)))[:16]


GOOD_RESP = {"experiment": {"kind": "stdin_input", "payload": "1"}, "implications": "outputs differ"}


@test
def critic_experiment_demotes_and_records_breakage():
    codes = ["OK:a", "OK:b", "OK:c"]
    exe = FakeExec(exp_table={"OK:a": {"status": "pass", "observed": "2"}, "OK:b": {"status": "error", "failure_class": "runtime_error", "observed": None},
                              "OK:c": {"status": "pass", "observed": "3"}})
    cfg = {"k": 3, "critic": True, "critic_experiment": True, "critic_responses": {crit_rid(codes): GOOD_RESP}}
    res, ctl, _, _ = run([codes], cfg, critic=FakeCritic(), exe=exe)
    assert res.stop_cause == "decided" and res.final_artifact.source == "OK:a"
    j = ctl.last_journal
    crit = [e["data"] for e in j.entries("critic")][-1]
    assert crit["state"] == "applied" and crit["demoted"] == [H("OK:b")]
    c_ev = j.evidence(H("OK:c"))
    assert any(e["check_id"] == "critic_experiment_v1" and e["independence"] == "model_derived" and e["source"] == "critic_experiment" for e in c_ev)
    sep = [e for e in c_ev if e["source"] == "environment_observation" and e["status"] == "fail" and e["check_id"] == "public_examples_v1"]
    assert len(sep) == 1 and sep[0]["requirement_id"] == "R3"                      # separate execution-derived record
    assert ctl.last_run.cands[H("OK:c")].decision["answer_label"] == "BLOCKED"
    assert all(e["status"] != "fail" for e in j.evidence(H("OK:b")))                # the critic alone never records a candidate failure


@test
def critic_malformed_is_logged_and_run_continues():
    cfg = {"k": 1, "critic": True, "critic_experiment": True, "critic_responses": {crit_rid(["OK:a"]): "garbage"}}
    res, ctl, _, _ = run([["OK:a"]], cfg, critic=FakeCritic())
    assert res.stop_cause == "decided" and stop_entry(ctl)["wasted_critic"] == 1
    assert [e["data"]["state"] for e in ctl.last_journal.entries("critic")] == ["malformed"]
    cfg["critic_responses"] = {crit_rid(["OK:a"]): {"experiment": {"kind": "nonsense"}, "implications": "x"}}
    res, ctl, _, _ = run([["OK:a"]], cfg, critic=FakeCritic())
    assert res.stop_cause == "decided" and stop_entry(ctl)["wasted_critic"] == 1


@test
def critic_pending_gives_api_unavailable_then_second_pass_decides():
    critic = FakeCritic()
    cfg = {"k": 1, "critic": True, "critic_experiment": True}
    res, ctl, _, _ = run([["OK:a"]], cfg, critic=critic, exe=FakeExec(exp_table={"OK:a": {"observed": "2"}}))
    assert res.stop_cause == "api_unavailable" and res.stop_details == "critic_pending" and res.decision["next_step"] == "discriminate"
    assert res.answer_label == "SUPPORTED" and len(critic.export_pending()) == 1
    rid = critic.built[0]["request_id"]
    assert rid == crit_rid(["OK:a"]) and critic.built[0]["model"] == "z-ai/glm-5.3-flash"
    C.validate("critic_request", critic.built[0])
    res2, _, _, _ = run([["OK:a"]], cfg, critic=FakeCritic(done={"pk-" + rid: GOOD_RESP}), exe=FakeExec(exp_table={"OK:a": {"observed": "2"}}))       # second pass: the broker returns the ingested response
    assert res2.stop_cause == "decided" and res2.stop_details == ""
    cfg["critic_responses"] = {rid: GOOD_RESP}                                                    # test-only fallback convention
    res3, _, _, _ = run([["OK:a"]], cfg, critic=FakeCritic(), exe=FakeExec(exp_table={"OK:a": {"observed": "2"}}))
    assert res3.stop_cause == "decided"


# ------------------------------------------------------------------ scorer
@test
def scorer_is_a_late_tiebreak_and_gated_by_decision_type():
    codes = ["OK:a", "TIMEOUT:b"]
    sc = FakeScorer()
    res, ctl, _, _ = run([codes], {"k": 2, "scorer": True}, scorer=sc)
    assert sc.calls == [] and res.final_artifact.source == "OK:a"                       # complete_program is not validated and the run is not exploratory
    assert [e["data"]["state"] for e in ctl.last_journal.entries("scorer")] == ["skipped"]
    sc = FakeScorer()
    res, ctl, _, _ = run([codes], {"k": 2, "scorer": True, "exploratory": True}, scorer=sc)
    assert ("complete_program", True) in sc.calls and res.final_artifact.source == "OK:a"      # the scorer prefers TIMEOUT:b; I2: trusted levels win
    rk = ctl.last_journal.entries("ranking")[-1]["data"]
    assert rk["order"][0] == H("OK:a")
    sc = FakeScorer()
    res, ctl, _, _ = run([codes], {"k": 2, "scorer": True, "scorer_validated_types": ["complete_program"]}, scorer=sc)
    assert [e["data"]["state"] for e in ctl.last_journal.entries("scorer")] == ["refused"] and res.stop_cause == "decided"


# ------------------------------------------------------------------ authorization and finalize
@test
def irreversible_needs_confirmation():
    reqs = [C.check_requirement({"requirement_id": "R1", "version": 1, "kind": "explicit", "text": "t",
                                 "source": {"type": "request_span", "quote": "abcdef", "char_start": 0, "char_end": 6}, "priority": "functional", "status": "active"}, "abcdef")]
    chk = {"check_id": "t1", "version": "1", "code_hash": H("t1"), "kind": "public_test", "trust_class": "authoritative", "covers": ["R1"], "runtime_visible": True, "grader_only": False}
    reg = C.Registry([chk])
    ev = [{"evidence_id": "E1", "requirement_id": "R1", "requirement_version": 1, "check_id": "t1", "check_version": "1", "candidate_id": "A", "artifact_hash": H("A"),
           "environment_revision": "e", "source": "existing_test", "independence": "independent", "status": "pass", "scope": "s", "produced_at": "t", "receipt_hash": H("r")}]
    assert authorize(reqs, ev, "A", H("A"), "e", reg, "sandboxed") and authorize(reqs, ev, "A", H("A"), "e", reg, "reversible_real")
    assert not authorize(reqs, ev, "A", H("A"), "e", reg, "irreversible", False)
    assert authorize(reqs, ev, "A", H("A"), "e", reg, "irreversible", True)
    res, _, _, _ = run([["OK:a"]], {"k": 1, "risk": "irreversible"})
    assert not res.decision["authorized_to_execute"] and res.decision["risk"] == "irreversible"


@test
def finalize_blocks_on_hash_mismatch():
    res, ctl, _, exe = run([["OK:a"]], {"k": 1})
    jh, n_entries = res.journal_hash, len(ctl.last_journal)
    f0 = ctl.finalize(res.final_artifact)
    assert f0["ok"] is True and f0["finalize_hash"] == ctl.finalize_hash and f0["finalize_hash"] != jh
    assert ctl.last_journal.journal_hash() == jh and len(ctl.last_journal) == n_entries          # the returned journal is untouched
    tampered = Artifact(res.final_artifact.artifact_id, "T1", "python", "OK:evil")
    n = exe.calls.get("OK:evil", 0)
    f = ctl.finalize(tampered)
    assert f["ok"] is False and f["reason"] == "hash_mismatch" and f["decision"]["answer_label"] == "BLOCKED" and exe.calls.get("OK:evil", 0) == n


# ------------------------------------------------------------------ determinism and registry freeze
@test
def journal_hash_is_reproducible():
    cfg = {"k": 2, "max_rounds": 2}
    a, _, _, _ = run([["WRONG:a:0", "PARTIAL"], ["OK:fix"]], cfg)
    b, _, _, _ = run([["WRONG:a:0", "PARTIAL"], ["OK:fix"]], cfg)
    c, _, _, _ = run([["WRONG:a:0", "PARTIAL"], ["OK:fiy"]], cfg)
    d, _, _, _ = run([["WRONG:a:0", "PARTIAL"], ["OK:fix"]], cfg, seeds=(9, 8, 7, 6))
    assert a.journal_hash == b.journal_hash and a.journal_hash != c.journal_hash and a.journal_hash != d.journal_hash


@test
def registry_change_during_run_aborts():
    ctl = EvidenceLadderController()
    exe = FakeExec(hook=lambda art: ctl._registry.checks["parse_v1"].__setitem__("trust_class", "authoritative"))
    try:
        run([["OK:a"]], {"k": 1}, exe=exe, ctl=ctl)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "I7" in str(e)


# ------------------------------------------------------------------ modes
@test
def mode_single():
    res, ctl, gen, _ = run([["WRONG:a:0"]], {"mode": "single", "k": 5}, arm="A")
    assert len(gen.calls) == 1 and gen.calls[0].n == 1 and res.rounds == 0 and res.stop_cause == "budget" and res.stop_details == "attempt_cap:single"
    assert res.answer_label == "BLOCKED"
    res, _, _, _ = run([["OK:a"]], {"mode": "single"}, arm="A")
    assert res.stop_cause == "decided" and res.answer_label == "SUPPORTED"


@test
def mode_best_of_k():
    res, ctl, gen, _ = run([["WRONG:a:0", "PARTIAL", "OK:z"]], {"mode": "best_of_k", "k": 3}, arm="B")
    assert len(gen.calls) == 1 and gen.calls[0].n == 3 and res.final_artifact.source == "OK:z" and res.stop_cause == "decided"
    res, ctl, gen, _ = run([["WRONG:a:0", "PARTIAL", "WRONG:b:1"]], {"mode": "best_of_k", "k": 3}, arm="B")
    assert len(gen.calls) == 1 and res.final_artifact.source == "PARTIAL" and res.stop_cause == "budget" and res.rounds == 0


@test
def mode_repair_has_no_critic_or_scorer():
    critic, sc = FakeCritic(), FakeScorer()
    cfg = {"mode": "repair", "k": 1, "critic": True, "critic_experiment": True, "scorer": True, "exploratory": True}
    res, ctl, gen, _ = run([["WRONG:a:0"], ["OK:fix"]], cfg, arm="C", critic=critic, scorer=sc)
    assert res.stop_cause == "decided" and res.rounds == 1 and critic.built == [] and sc.calls == []
    assert not ctl.last_journal.entries("critic") and not ctl.last_journal.entries("scorer")


# ------------------------------------------------------------------ invariants
@test
def invariant_negative_controls():
    j = EvidenceJournal(build_registry([]))
    j.append("evidence", {"evidence": {"evidence_id": "E1", "status": "pass", "candidate_id": "A"}, "outcome": {"status": "fail"}})
    assert "pass" in inv.check_i4(j)
    j2 = EvidenceJournal(build_registry([]))
    items = [{"id": "a", "trusted_passes": 0, "validated_passes": 0, "cluster": 0, "scorer": 9.0, "deepconf": 0.0},
             {"id": "b", "trusted_passes": 2, "validated_passes": 0, "cluster": 0, "scorer": 0.0, "deepconf": 0.0}]
    j2.append("ranking", {"step": "x", "items": items, "order": ["a", "b"]})
    assert inv.check_i2(j2) != ""
    j3 = EvidenceJournal(build_registry([]))
    j3.append("finalize", {"candidate_id": "A", "ok": True})
    assert inv.check_i6(j3) != ""
    j4 = EvidenceJournal(build_registry([]))
    j4.append("registry", {"phase": "start", "hash": "a", "scorer": None})
    j4.append("registry", {"phase": "end", "hash": "b", "scorer": None})
    assert inv.check_i7(j4) != ""
    res, ctl, _, _ = run([["OK:a"]], {"k": 1})
    assert inv.assert_invariants(res, ctl.last_journal, ControllerContext(None, None, None, "D0")) == ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]
    res.answer_label = "VERIFIED"
    assert inv.check_i1(res, ctl.last_journal) != ""


@test
def static_no_grader_imports():
    assert inv.static_no_grader_imports() == [], inv.static_no_grader_imports()
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "bad.py").write_text("from somewhere import Grader\nimport protected_tests\n", encoding="utf-8")
        assert len(inv.static_no_grader_imports(Path(d))) == 2


def main():
    failed = {}
    for name, f in TESTS.items():
        try:
            f()
        except Exception:      # noqa: BLE001
            failed[name] = traceback.format_exc()[-600:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
