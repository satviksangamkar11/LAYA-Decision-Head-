"""[LAYA-INT-1 · Aim: integration, not performance] Pure-logic suite: provenance refusal logic (T7, fake capture function), LAYA scorer interface wiring, controller use of L0 scores (T5).
No Qwen, no GPU, no network. Needs torch (CPU) for tensors. Run: python evidence_ladder/models/tests/laya_integration_tests.py [path to T4_scores.json]
Prints {"tests": n, "passed": n, "failed": {...}}; exit 1 on failure. The real-score tests (T5 with the scores of the GPU kernel) run when a T4_scores.json is found (argv[1] or env LAYA_T4_SCORES)."""
import hashlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from evidence_ladder.common.contract_import import C  # noqa: E402
from evidence_ladder.common.interfaces import NotAuthorized, UnsupportedDecisionType  # noqa: E402
from evidence_ladder.controller import invariants as INV  # noqa: E402
from evidence_ladder.controller.journal import EvidenceJournal  # noqa: E402
from evidence_ladder.controller.next_action import decide_next_action  # noqa: E402
from evidence_ladder.models import laya, laya_forward as LF, laya_runtime as RT  # noqa: E402

DEV = "Tesla T4"


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % (exc,))


def fake_capture(context_text, options):
    """Deterministic stand-in for the frozen Qwen: features are a function of the TEXT only (seeded by its hash). Same text -> same tensors; other text -> other tensors."""
    seed = int(hashlib.sha256((context_text + "\x00" + "\x01".join(options)).encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(3, 2560, generator=g).half(), torch.randn(len(options), 3, 2560, generator=g).half(), {"status": "ok", "mode": "fake"})


class FakeScorer:
    def _score(self, hq, ho):
        return {"laya": {"per_seed_logits": {}, "score": [float(ho[i].float().mean()) for i in range(ho.shape[0])]}}


def record(i, device=DEV, **over):
    ctx, opts = "context %d" % i, ["act %d a" % i, "act %d b" % i, "act %d c" % i]
    hq, ho, _ = fake_capture(ctx, opts)
    r = {"decision_id": "d%d" % i, "context_text": ctx, "options": opts, "hq": hq, "ho": ho, "provenance": RT.make_provenance(device=device, context_text=ctx, options=opts)}
    r.update(over)
    return r


def score(batch, **kw):
    return RT.score_stored(batch, capture_fn=fake_capture, scorer=FakeScorer(), sample_n=kw.pop("sample_n", len(batch)), seed=kw.pop("seed", 0), **kw)


# ------------------------------------------------------------------ T7 provenance (logic)
def test_genuine_batch_is_accepted_and_scored():
    out = score([record(i) for i in range(4)])
    assert out["origin"] == "stored_features_verified_by_recapture" and out["sampled"] == 4 and len(out["scores"]) == 4 and all(c["passed"] for c in out["comparisons"])


def test_metadata_mismatches_are_refused():
    for k, v in (("model_revision", "0" * 40), ("model_repo", "Qwen/Qwen3-4B"), ("dtype", "bfloat16"), ("depths", [12, 18, 24]), ("recipe_id", "other_v9"), ("hidden", 2048),
                 ("device", "NVIDIA GeForce RTX 3050"), ("capture_code_sha256", {"capture/capture.py": "0" * 64, "capture/capture_resident.py": "1" * 64}), ("query_cue", "\nAnswer:")):
        b = [record(0), record(1)]
        b[0] = dict(b[0], provenance=dict(b[0]["provenance"], **{k: v}))
        e = raises(RT.ProvenanceError, score, b)
        assert "provenance" in str(e) or "captured on" in str(e) or "depths" in str(e) or "capture code" in str(e), (k, str(e))
    raises(RT.ProvenanceError, score, [dict(record(0), provenance=None)])
    raises(RT.ProvenanceError, score, [{k: v for k, v in record(0).items() if k != "provenance"}])


def test_tensor_level_refusals():
    for bad in (dict(hq=record(0)["hq"].float()), dict(ho=record(0)["ho"].float()), dict(hq=torch.zeros(3, 100).half()), dict(ho=record(0)["ho"][:1]),
                dict(hq="not a tensor"), dict(hq=torch.full((3, 2560), float("nan")).half())):
        raises(RT.ProvenanceError, score, [dict(record(0), **bad)])


def test_fabricated_tensors_with_perfect_metadata_are_caught_by_recapture_not_metadata():
    r = record(0)
    RT.check_metadata(r["provenance"])                                           # the metadata IS valid
    for name, hq, ho in (("random", torch.randn(3, 2560).half(), torch.randn(3, 3, 2560).half()),
                         ("other text", fake_capture("context 9", r["options"])[0], fake_capture("context 9", r["options"])[1]),
                         ("perturbed 5%", r["hq"], (r["ho"].float() + 0.05 * float(r["ho"].float().std()) * torch.randn(r["ho"].shape)).half())):
        bad = dict(r, hq=hq, ho=ho)
        e = raises(RT.ProvenanceError, score, [bad, record(1)])
        assert "fresh capture" in str(e), (name, str(e))                             # refused by the re-capture comparison
    e = raises(RT.ProvenanceError, score, [record(1), dict(r, ho=torch.randn(3, 3, 2560).half()), record(2)])
    assert "whole batch refused" in str(e)


def test_text_and_embedding_inputs_have_no_path():
    live = RT.LiveScorer(fake_capture, FakeScorer(), DEV)
    assert [p for p in inspect.signature(live.score_texts).parameters] == ["context_text", "options"]       # no tensor argument exists
    raises(RT.ProvenanceError, live.score_texts, "ctx", "a single string")
    raises(RT.ProvenanceError, live.score_texts, "ctx", ["only one option"])
    be = RT.LiveBackend(live)
    raises(RT.ProvenanceError, be.score, "next_action", ["candidate A", "candidate B"])                    # bare text candidates
    raises(RT.ProvenanceError, be.score, "next_action", torch.zeros(2, 5))                                 # an embedding matrix
    out = be.score("next_action", {"context_text": "c", "options": ["x", "y"]})
    assert len(out) == 2 and be.last["origin"] == "live_capture_same_process" and be.last["provenance"]["model_revision"] == RT.QWEN_REVISION


def test_capture_exclusion_refuses_the_batch():
    def excl(ctx, opts): return None, None, {"status": "excluded_over_limit"}
    raises(RT.ProvenanceError, RT.score_stored, [record(0)], capture_fn=excl, scorer=FakeScorer(), sample_n=1, seed=0)


def test_default_sampling_is_a_random_subset_and_limits_are_recorded():
    batch = [record(i) for i in range(8)]
    out = RT.score_stored(batch, capture_fn=fake_capture, scorer=FakeScorer(), seed=1)
    assert out["sampled"] == 3 and len(out["verified_ids"]) == 3 and out["batch"] == 8       # >= 25% and at least 3
    out2 = RT.score_stored(batch, capture_fn=fake_capture, scorer=FakeScorer(), seed=1)
    assert out2["verified_ids"] == out["verified_ids"]                                         # seeded -> reproducible receipt


def test_single_fabricated_record_anywhere_is_caught_at_100_percent():
    n = 6
    for pos in range(n):
        batch = [record(i) for i in range(n)]
        batch[pos] = dict(batch[pos], hq=torch.randn(3, 2560).half(), ho=torch.randn(3, 3, 2560).half())     # perfect metadata, fabricated tensors
        out = raises(RT.ProvenanceError, score, batch)                                                          # score() uses sample_n = len(batch)
        assert "fresh capture" in str(out) and ("d%d" % pos) in str(out), (pos, str(out))


def test_capture_pins_v2_are_versioned_and_consistent():
    assert laya.EXPECTED_CAPTURE_CODE_V1[0]["path"] == "tools/fable_capture.py"                                 # history kept
    v2 = laya.load_capture_pins("v2")
    assert [e["path"] for e in v2] == ["capture/capture.py", "capture/capture_resident.py", "decision_head/fable_features.py"] and tuple(v2) == tuple(laya.EXPECTED_CAPTURE_CODE)
    assert {e["path"]: e["sha256"] for e in v2}["capture/capture.py"] == RT.CAPTURE_CODE_SHA256["capture/capture.py"]
    assert {e["path"]: e["sha256"] for e in v2}["capture/capture_resident.py"] == RT.CAPTURE_CODE_SHA256["capture/capture_resident.py"]
    raises(ValueError, laya.load_capture_pins, "v9")


# ------------------------------------------------------------------ scorer interface (laya.py)
def test_laya_scorer_interface_wiring():
    s = laya.LayaScorer()                                                                      # real repo manifest and capture-code hashes (v2 pins)
    assert not s.disabled, "the real LayaScorer is disabled in this bundle: " + s.meta.disabled_reason      # never a silent pass
    live = RT.LiveScorer(fake_capture, FakeScorer(), DEV)
    a = laya.LayaScorer(authorized=True, backend=RT.LiveBackend(live))
    pool = {"context_text": "c", "options": ["x", "y", "z"]}
    out = a.score("next_action", pool)
    assert len(out) == 3 and all(isinstance(v, float) for v in out)
    raises(UnsupportedDecisionType, a.score, "complete_program", pool)                          # L0 for next_action only
    raises(NotAuthorized, laya.LayaScorer(backend=RT.LiveBackend(live)).score, "next_action", pool)   # not authorized -> refuses even with a backend
    raises(NotImplementedError, laya.LayaScorer(authorized=True).score, "next_action", pool)
    raises(RT.ProvenanceError, a.score, "next_action", ["plain", "text"])


def test_saved_score_backend_refuses_unproven_records():
    ok = {"origin": "stored_features_verified_by_recapture", "candidate_source_sha256": LF.sha256_text("p1"), "scores": {"laya": {"status": "ok", "score": 1.5}}}
    bad = {"origin": "unknown", "candidate_source_sha256": LF.sha256_text("p2"), "scores": {"laya": {"status": "ok", "score": 9.0}}}
    be = LF.SavedScoreBackend([ok, bad], "laya")
    assert be.score("next_action", ["p1"]) == [1.5] and be.refused == 1
    raises(KeyError, be.score, "next_action", ["p2"])


# ------------------------------------------------------------------ T5 controller
def _decide(scores, trusted=None, j=None, name="d"):
    j = j or EvidenceJournal()
    return j, decide_next_action(j, name, ["opt%d" % i for i in range(len(scores))], scores, trusted, scorer_id="laya_adapter_v1", scorer_provenance={"origin": "test"})


def test_l0_orders_equal_evidence_candidates_and_never_approves():
    j, r = _decide([0.1, 2.0, -1.0, 0.7])
    assert r["selected_index"] == 1 and [int(x.split("#")[1]) for x in r["order"]] == [1, 3, 0, 2] and r["approved"] is False and r["level"] == "L0_ordering_only"
    assert INV.check_i2(j) == "" and j.entries("scorer")[0]["data"]["level"] == "L0"


def test_top_laya_failing_trusted_evidence_is_never_selected():
    scores = [0.1, 9.0, 0.2]
    for lvl in ("L5", "L3", "policy"):
        tr = [{}, {lvl: "fail"}, {}]
        j, r = _decide(scores, tr)
        assert r["selected_index"] != 1 and r["blocked"] == ["d#1"] and 1 not in [int(x.split("#")[1]) for x in r["order"][:2]] and INV.check_i2(j) == ""
    j, r = _decide([1.0, 2.0], [{"L5": "fail"}, {"L3": "fail"}])
    assert r["selected_id"] is None                                                             # nothing trusted-acceptable: no action, not the best score


def test_trusted_evidence_outranks_any_laya_score():
    j, r = _decide([5.0, 4.0, -50.0], [{}, {"L3": "pass"}, {"L5": "pass", "L3": "pass"}])
    assert r["selected_index"] == 2 and [int(x.split("#")[1]) for x in r["order"]] == [2, 1, 0] and INV.check_i2(j) == ""


def test_scorer_unavailable_is_explicit_never_silent():
    a = EvidenceJournal()
    raises(ValueError, decide_next_action, a, "d", ["x", "y", "z"], None, None)                          # no scores and no stated reason: refused
    r = decide_next_action(a, "d", ["x", "y", "z"], None, None, scorer_unavailable_reason="ProvenanceError: no verified score")
    assert r["selected_index"] == 0 and r["approved"] is False and r["scorer_available"] is False
    e = a.entries("scorer_unavailable")
    assert len(e) == 1 and "no verified score" in e[0]["data"]["reason"] and e[0]["data"]["fallback"] == "id_order" and not a.entries("scorer")


def test_journal_is_reproducible():
    h1 = _decide([0.3, 0.9, 0.1], [{}, {}, {"L5": "fail"}])[0].journal_hash()
    h2 = _decide([0.3, 0.9, 0.1], [{}, {}, {"L5": "fail"}])[0].journal_hash()
    assert h1 == h2


def test_abstain_when_no_option_is_allowed_and_no_none_option_is_added():
    j, r = _decide([1.0, 2.0], [{"policy": "fail"}, {"L5": "fail"}])
    assert r["abstain"] is True and r["selected_id"] is None and r["next_step"] == "gather_evidence" and r["blocked"] == ["d#0", "d#1"]
    e = j.entries("abstain")
    assert len(e) == 1 and e[0]["data"]["n_options"] == 2                                                # the pool still has exactly the 2 proposed options (no NONE added)
    j2 = EvidenceJournal()
    r2 = decide_next_action(j2, "d", [], [], [])
    assert r2["abstain"] is True and j2.entries("abstain")
    j3, r3 = _decide([1.0, 2.0], [{"L5": "fail_nonblocking"}, {}])                                        # a non-blocking fail demotes but does not reject
    assert r3["abstain"] is False and r3["selected_index"] == 1 and r3["order"] == ["d#1", "d#0"]


def test_invariant_negative_control_detects_a_reordering_across_evidence():
    j, _ = _decide([1.0, 2.0], [{"L5": "pass"}, {}])
    assert INV.check_i2(j) == ""
    bad = EvidenceJournal()
    items = [{"id": "a", "trusted_passes": 0, "validated_passes": 0, "cluster": 0, "scorer": 9.0, "deepconf": 0.0},
             {"id": "b", "trusted_passes": 2, "validated_passes": 0, "cluster": 0, "scorer": 0.0, "deepconf": 0.0}]
    bad.append("ranking", {"step": "x", "items": items, "order": ["a", "b"]})                  # the learned score put 'a' above 'b' despite trusted evidence
    assert INV.check_i2(bad) != ""


STORED_T4_SCORES_SHA256 = "d4f98029920bc880b91199007fc5ab5098b611af520b93ba8b2478fd353f0164"


def test_real_scores_never_override_trusted_evidence():
    p = os.environ.get("LAYA_T4_SCORES") or (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else None)
    if not (p and Path(p).exists()):      # [EL-ARCH-1 A3] fall back to the stored copy of the real Kaggle T4 scores (read-only), pinned by sha256; same assertions below
        import hashlib
        stored = Path(__file__).resolve().parents[3] / "results/raw/laya-integration-20261016T184816Z/integration_run2/T4_scores.json"
        if stored.exists():
            assert hashlib.sha256(stored.read_bytes()).hexdigest() == STORED_T4_SCORES_SHA256, "stored T4_scores.json does not match its pinned sha256"
            p = str(stored)
    assert p and Path(p).exists(), "T4_scores.json (real LAYA scores of the Kaggle T4 run) must ship with this suite; a missing file is a FAILURE, not a skip"
    pools = json.load(open(p, encoding="utf-8"))["pools"]
    assert len(pools) >= 3
    for pool in pools:
        sc = pool["laya_score"]; n = len(sc); top = max(range(n), key=sc.__getitem__)
        j, r0 = _decide(sc, None, name=pool["decision_id"])
        assert r0["selected_index"] == top and INV.check_i2(j) == ""
        tr = [{} for _ in range(n)]; tr[top] = {"L5": "fail"}
        j, r1 = _decide(sc, tr, name=pool["decision_id"])
        assert r1["selected_index"] != top and r1["approved"] is False and INV.check_i2(j) == ""
        low = min(range(n), key=sc.__getitem__); tr = [{} for _ in range(n)]; tr[low] = {"L3": "pass"}
        assert _decide(sc, tr, name=pool["decision_id"])[1]["selected_index"] == low


# ------------------------------------------------------------------ per-action controller (agentic.py) with a fake sandbox
import json as _json
import shutil
import tempfile
from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask

_TEST = 'from slug import slugify\nassert slugify("Hello, World!") == "hello-world"\nprint("all tests passed")\n'
_SLUG = 'import re\n\n\ndef slugify(text):\n    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower())\n'
_USER = "Fix slug.py so that `python3 test_slug.py` passes. Do not modify test_slug.py."


class FakeSandbox:
    """Deterministic stand-in for the WSL sandbox: the test passes iff slug.py strips hyphens; protected_unchanged compares test_slug.py."""
    def __init__(self): self.calls = 0

    def run(self, workspace, cmd, persist=False, timeout=60):
        self.calls += 1
        ws = Path(workspace)
        fixed = '.strip("-")' in (ws / "slug.py").read_text()
        is_test = cmd.startswith("python3 test_slug.py")
        ok = is_test and fixed
        return {"exit_code": 0 if (ok or not is_test) else 1, "killed": False, "stdout": "all tests passed\n" if ok else "", "stderr": "" if (ok or not is_test) else "AssertionError\n",
                "protected_unchanged": (ws / "test_slug.py").read_text() == _TEST, "pre_tree_sha256": "x", "post_tree_sha256": "y"}


FIX = _json.dumps({"name": "Edit", "arguments": {"file_path": "slug.py", "old_string": 'return re.sub(r"[^a-z0-9]+", "-", text.strip().lower())', "new_string": 'return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")'}})
WRONG = _json.dumps({"name": "Edit", "arguments": {"file_path": "slug.py", "old_string": ".strip().lower()", "new_string": ".lower()"}})
PROT = _json.dumps({"name": "Edit", "arguments": {"file_path": "test_slug.py", "old_string": "hello-world", "new_string": "x"}})
RMRF = _json.dumps({"name": "Bash", "arguments": {"command": "rm -rf /"}})
READ = _json.dumps({"name": "Read", "arguments": {"file_path": "slug.py"}})


class ScorePref:
    """Scorer-interface stand-in whose order is fixed by the caller (the test decides which option the scorer likes best)."""
    scorer_id = "pref_scorer"

    def __init__(self, scores): self.scores = scores

    def score(self, decision_type, pool, exploratory=False):
        assert decision_type == "next_action" and set(pool) == {"context_text", "options"}
        return list(self.scores[:len(pool["options"])])


def _run(scorer, tmp, name="t"):
    ws = Path(tmp) / name / "ws"
    ws.mkdir(parents=True)
    (ws / "slug.py").write_text(_SLUG); (ws / "test_slug.py").write_text(_TEST)
    task = AgenticTask(name, _USER, ws, "python3 test_slug.py", ["test_slug.py"], "SYSTEM", 2)
    r = AgenticRun(task, AgenticContext(sandbox=FakeSandbox(), scorer=scorer, arm="E"), lambda msgs: "\n\n".join(str(m.get("content")) for m in msgs))
    r.begin()
    return r


def test_agentic_real_executed_evidence_outranks_the_laya_top_choice():
    """SUPERSEDES the 5-option design (laya_order [0,2,3,1,4], allowed_set [0,1,4], policy-failed PROT/RMRF, fail_nonblocking WRONG): the frozen candidate filter now removes the
    protected-file edit (protected_path) and, because Read(slug.py) -> Edit(slug.py) -> Bash is a sequential plan, the WRONG edit and `rm -rf /` (sequential_plan) BEFORE ranking, by design.
    The intent is unchanged and no trusted-evidence-beats-LAYA assertion is loosened: executed trusted evidence outranks LAYA's top choice, on (a) the same five options (what the filter
    leaves), (b) an admissible pool that keeps a failing edit so the dry-run loss is still exercised, (c) the stored real 4B scores on the exact stored ex1 context (PATH B)."""
    from evidence_ladder.controller import candidate_filter as CF
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(ScorePref([9.0, 1.0, 5.0, 4.0, 0.0]), tmp)                 # the scorer likes READ (0) best and the fixing edit (4) least
        s = r.step([READ, WRONG, PROT, RMRF, FIX])
        f = r.journal.entries("candidate_filter")[0]["data"]
        assert {x["rule"] for x in f["rejected"]} == {"protected_path", "sequential_plan"} and f["counts"] == {"in": 5, "accepted": 2, "rejected": 3}        # PROT, WRONG, RMRF removed before ranking
        assert [a["tool"] for a in f["accepted"]] == ["Read", "Edit"] and [x["arguments"].get("file_path") for x in f["accepted"]] == ["slug.py", "slug.py"] and s["options"][1].count(".strip(\\\"-\\\")") == 1
        assert s["trusted"][1]["L5_dry_run"] == "pass"                                                                                           # dry-run went through the sandbox port, not injected
        assert s["laya_order"] == [0, 1] and s["allowed_set"] == [0, 1]
        assert s["chosen_index"] == 1 and s["trusted_evidence_overrode_laya"] is True and s["laya_top_was_chosen"] is False                      # LAYA's top (the Read) loses to the executed pass
        assert s["tests_pass"] and s["label"] == "VERIFIED" and s["authorized"]
        res = r.finish()
        assert res.answer_label == "VERIFIED" and res.stop_cause == "decided" and r.invariants_checked == ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]
    with tempfile.TemporaryDirectory() as tmp:                                                          # (b) a pool where LAYA's top is a real failing edit: dry-run FAIL, chosen = the passing edit
        r = _run(ScorePref([9.0, 5.0, 0.0]), tmp, name="b")
        s = r.step([READ, WRONG, FIX])
        assert s["laya_order"] == [0, 1, 2] and s["allowed_set"] == [0, 1, 2]
        assert s["trusted"][2]["L5_dry_run"] == "pass" and s["trusted"][1]["L5_dry_run"] == "fail_nonblocking"
        assert s["chosen_index"] == 2 and s["trusted_evidence_overrode_laya"] is True and s["laya_top_was_chosen"] is False
        assert s["tests_pass"] and s["label"] == "VERIFIED" and s["authorized"]
        res = r.finish()
        assert res.answer_label == "VERIFIED" and res.stop_cause == "decided" and r.invariants_checked == ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]
    from evidence_ladder.controller import pathb_stored as PB                                           # (c) PATH B: stored verified 4B scores, exact stored context, real sandbox
    try:
        live, backends, sysb = PB.load_stored_4b()
        from evidence_ladder.controller.wsl_sandbox import available
        if not available():
            return
    except PB.StoredCaptureMissing:
        return
    r, s = PB.run_exact_pool("ex1_slug:1", live, backends, sysb)
    top = s["laya_order"][0]
    assert s["checks"][top]["dry_run"]["status"] == "fail" and s["chosen_index"] != top and s["trusted_evidence_overrode_laya"] is True and s["laya_top_was_chosen"] is False


def test_agentic_laya_orders_what_trusted_evidence_leaves_and_may_change_the_id_order_choice():
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(ScorePref([1.0, 3.0, 2.0]), tmp)
        s = r.step([READ, _json.dumps({"name": "Bash", "arguments": {"command": "ls"}}), _json.dumps({"name": "Read", "arguments": {"file_path": "test_slug.py"}})])
        assert s["id_order_choice"] == 0 and s["chosen_index"] == 1 and s["laya_changed_the_choice_vs_id_order"] is True and s["trusted_evidence_overrode_laya"] is False


def test_agentic_missing_scorer_is_journaled_not_silent_and_step_limit_stops():
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(None, tmp)
        s = r.step([READ, WRONG])
        assert s["scores"] is None and "AttributeError" in s["scorer_unavailable_reason"] and r.journal.entries("scorer_unavailable")
        r.step([READ, WRONG])
        res = r.finish()
        assert res.stop_cause == "budget" and res.answer_label == "BLOCKED" and not res.decision["authorized_to_execute"]


def test_agentic_abstains_when_everything_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(ScorePref([2.0, 1.0]), tmp)
        s = r.step([PROT, RMRF])
        assert s["chosen_index"] is None and s["decision"]["abstain"] is True and r.journal.entries("abstain") and s["tests_pass"] is False


def test_agentic_replay_gives_identical_journal_hashes():
    hs = []
    for i in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            r = _run(ScorePref([9.0, 1.0, 5.0, 4.0, 0.0]), tmp, name="same")
            r.step([READ, WRONG, PROT, RMRF, FIX])
            hs.append(r.finish().journal_hash)
    assert hs[0] == hs[1]


# ------------------------------------------------------------------ imported Kaggle scores (verified by an independent re-capture run)
def _kres(scores=(1.0, 2.0), **over):
    opts = ["opt a", "opt b"]
    d = {"status": "ok", "origin": "live_capture_same_process", "provenance": RT.make_provenance(device=DEV, context_text="ctx", options=opts), "context_sha256": LF.sha256_text("ctx"),
         "options": opts, "option_sha256": [LF.sha256_text(o) for o in opts], "raw_output_sha256": "r", "features_sha256": "f", "scores": {"laya": {"score": list(scores)}}}
    d.update(over)
    return d


def test_verified_saved_scores_need_two_agreeing_runs_and_feed_the_laya_scorer():
    be = RT.VerifiedSavedScoreBackend("laya")
    assert be.add(_kres(), _kres(), live_run="live@v1", verify_run="verify@v2", verify_mode="replay") is True
    pool = {"context_text": "ctx", "options": ["opt a", "opt b"]}
    assert be.score("next_action", pool) == [1.0, 2.0] and be.last["origin"] == "saved_scores_verified_by_independent_recapture"
    sc = laya.LayaScorer(authorized=True, backend=be)
    assert not sc.disabled and sc.score("next_action", pool) == [1.0, 2.0]
    raises(RT.ProvenanceError, be.score, "next_action", {"context_text": "other", "options": ["opt a", "opt b"]})       # never verified -> refused
    raises(RT.ProvenanceError, be.score, "next_action", ["bare", "text"])
    for kw, bad in (({"live_run": "x", "verify_run": "x"}, _kres()), ({"verify_mode": "propose"}, _kres()), ({}, _kres(scores=(9.0, 0.0))), ({}, _kres(features_sha256="g")),
                    ({}, _kres(origin="kaggle_said_so")), ({}, _kres(provenance=dict(_kres()["provenance"], model_revision="0" * 40)))):
        be2 = RT.VerifiedSavedScoreBackend("laya")
        args = dict(live_run="live@v1", verify_run="verify@v2", verify_mode="replay")
        args.update(kw)
        assert be2.add(_kres(), bad, **args) is False and be2.refused, kw
        raises(RT.ProvenanceError, be2.score, "next_action", pool)


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
