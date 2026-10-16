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


# ------------------------------------------------------------------ scorer interface (laya.py)
def test_laya_scorer_interface_wiring():
    s = laya.LayaScorer()                                                                      # real repo manifest and capture-code hashes
    if s.disabled:
        return                                                                                 # bundle without the baseline files: covered by models/tests/run_tests.py
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


def test_no_scorer_means_deterministic_id_order_and_journal_is_reproducible():
    a = EvidenceJournal(); r = decide_next_action(a, "d", ["x", "y", "z"], None, None)
    assert r["selected_index"] == 0 and r["approved"] is False and not a.entries("scorer")
    h1 = _decide([0.3, 0.9, 0.1], [{}, {}, {"L5": "fail"}])[0].journal_hash()
    h2 = _decide([0.3, 0.9, 0.1], [{}, {}, {"L5": "fail"}])[0].journal_hash()
    assert h1 == h2


def test_invariant_negative_control_detects_a_reordering_across_evidence():
    j, _ = _decide([1.0, 2.0], [{"L5": "pass"}, {}])
    assert INV.check_i2(j) == ""
    bad = EvidenceJournal()
    items = [{"id": "a", "trusted_passes": 0, "validated_passes": 0, "cluster": 0, "scorer": 9.0, "deepconf": 0.0},
             {"id": "b", "trusted_passes": 2, "validated_passes": 0, "cluster": 0, "scorer": 0.0, "deepconf": 0.0}]
    bad.append("ranking", {"step": "x", "items": items, "order": ["a", "b"]})                  # the learned score put 'a' above 'b' despite trusted evidence
    assert INV.check_i2(bad) != ""


def test_real_scores_never_override_trusted_evidence():
    p = os.environ.get("LAYA_T4_SCORES") or (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else None)
    if not p or not Path(p).exists():
        return                                                                                  # real-score receipt not shipped in this bundle
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
