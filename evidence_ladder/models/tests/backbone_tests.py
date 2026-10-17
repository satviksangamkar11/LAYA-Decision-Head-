"""[EL-ARCH-1 local run] Backbone-profile suite: pure logic, no model, no GPU needed (CPU torch for tensors). Run: ATEN_CPU_CAPABILITY=avx2 python evidence_ladder/models/tests/backbone_tests.py
Covers: profile loading + cross-check with the former hard-coded 4B constants, a DUMMY THIRD profile (capture shapes / provenance logic without a real model), the profile-keyed provenance guard
(wrong model/revision/dtype/depths/device refused; head_compatible=[] refused by every LAYA path with a clear message; fabricated tensors still fail re-capture, also with an extra registered device),
the parity-gated device registration, and the profile-aware generation backend gate / greedy seed reuse."""
import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from evidence_ladder.common.interfaces import GenerationRequest, NotAuthorized  # noqa: E402
from evidence_ladder.models import backbone as BB, laya, laya_forward as LF, laya_runtime as RT, profile_capture as PC, qwen as Q  # noqa: E402

P4, P06 = BB.get("qwen3-4b-thinking-2507"), BB.get("qwen3-0.6b")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % (exc,))


def prov(profile=P4, device="Tesla T4", **over):
    p = RT.make_provenance(device=device, context_text="ctx", options=["a", "b"], profile=profile)
    p.update(over)
    return p


# ------------------------------------------------------------------ profile loading
def test_4b_profile_equals_the_former_hardcoded_constants():
    assert LF.DEPTHS == (18, 24, 30) and LF.HIDDEN == 2560 and LF.QUERY_CUE == "\n\nNext action:"
    assert RT.QWEN_REPO == "Qwen/Qwen3-4B-Thinking-2507" and RT.QWEN_REVISION == "768f209d9ea81521153ed38c47d515654e938aea" and RT.DEVICE_CLASS == "Tesla T4"
    assert Q.QWEN_REPO == RT.QWEN_REPO and Q.QWEN_REVISION == RT.QWEN_REVISION and Q.SAMPLING == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert P4.head_compatible and P4.generation["do_sample"] is True and P4.enable_thinking is None


def test_toml_4b_entry_equals_the_embedded_one_and_a_difference_raises():
    BB.load_all()      # does not raise
    d = Path(tempfile.mkdtemp()) / "b.toml"
    d.write_text((ROOT / BB.CONFIG_REL).read_text("utf-8").replace("hidden_size = 2560", "hidden_size = 2561"), "utf-8")
    raises(BB.ProfileError, BB.load_all, d)


def test_06b_profile_values_and_proportional_depths():
    assert (P06.hidden_size, P06.num_layers, P06.depths) == (1024, 28, (14, 19, 23))
    assert tuple(round(d / P4.num_layers * P06.num_layers) for d in P4.depths) == P06.depths
    assert P06.head_compatible == () and P06.generation == {"do_sample": False} and P06.enable_thinking is True
    cfg = json.loads((ROOT / "models/qwen3-0.6b/config.json").read_text("utf-8"))
    assert cfg["hidden_size"] == P06.hidden_size and cfg["num_hidden_layers"] == P06.num_layers
    assert json.loads((ROOT / "models/qwen3-0.6b/MANIFEST_pinned.json").read_text("utf-8"))["revision"] == P06.revision


def test_profile_hash_changes_with_content_and_validation():
    assert P4.sha256 != P06.sha256 and len(P4.sha256) == 64
    raises(BB.ProfileError, BB.from_dict, "x", {})
    d = dict(BB._BUILTIN_4B, depths=[18, 99])
    raises(BB.ProfileError, BB.from_dict, "x", d)
    raises(BB.ProfileError, BB.get, "no-such-model")


# ------------------------------------------------------------------ a dummy third profile (no real model)
THIRD = BB.from_dict("dummy-third", dict(BB._BUILTIN_4B, repo="x/y", revision="r" * 40, hidden_size=8, num_layers=6, depths=[2, 4, 5], device_classes=["Fake GPU"], head_compatible=["laya_dummy_set"]))


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 50 + 1 for c in text]


class FakeCapturer:
    """Option vector at depth d = f(own tokens, d) only (like the packed block mask guarantees); query vector = f(prefix, d)."""
    lm_calls = 0

    def __init__(self, profile): self.p = profile

    def packed(self, rec):
        H = self.p.hidden_size
        q_seed = sum(rec["prefix_ids"]) + 1
        cap = {}
        for d in self.p.depths:
            q = torch.full((H,), float(q_seed % 7 + d))
            ops = torch.stack([torch.full((H,), float(sum(s) % 11 + d)) for s in rec["suffix_ids"]])
            cap[d] = (q, ops)
        return cap, None


class FakeModel:
    class model:
        layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])


def test_dummy_third_profile_capture_shapes_follow_the_profile():
    cap = PC.make_capture_fn(THIRD, FakeModel(), FakeTok(), FakeCapturer(THIRD))
    hq, ho, info = cap("some context", ["opt one", "opt two", "opt three"])
    assert tuple(hq.shape) == (3, 8) and tuple(ho.shape) == (3, 3, 8) and hq.dtype == torch.float16 and info["status"] == "ok" and info["depth_keys"] == [2, 4, 5] and info["profile_id"] == "dummy-third"
    assert len(FakeModel.model.layers) == 6          # the per-call truncation is restored
    hq2, ho2, _ = cap("some context", ["opt three", "opt one", "opt two"])      # option order independence: rows follow the options
    assert torch.equal(ho2[0], ho[2]) and torch.equal(ho2[1], ho[0]) and torch.equal(hq2, hq)


def test_dummy_third_profile_limits_and_provenance_logic():
    cap = PC.make_capture_fn(THIRD, FakeModel(), FakeTok(), FakeCapturer(THIRD))
    assert cap("c", [])[2]["status"] == "excluded_over_limit"
    assert cap("c", ["x"] * 9)[2]["status"] == "excluded_over_limit"
    hq, ho, _ = cap("c", ["a", "b"])
    RT.check_tensors(hq, ho, 2, THIRD)                                       # shape rule taken from the profile
    e = raises(RT.ProvenanceError, RT.check_tensors, hq, ho, 2)              # ... and the 4B default refuses [3,8]
    assert "not [3,2560]" in str(e)
    p = RT.make_provenance(device="Fake GPU", context_text="c", options=["a", "b"], profile=THIRD)
    assert p["profile_id"] == "dummy-third" and p["profile_sha256"] == THIRD.sha256 and p["hidden"] == 8 and p["depths"] == [2, 4, 5]


# ------------------------------------------------------------------ the profile-keyed guard
def test_4b_provenance_accepted_and_legacy_record_without_profile_accepted():
    RT.check_metadata(prov())
    legacy = prov(); legacy.pop("profile_id"); legacy.pop("profile_sha256")
    RT.check_metadata(legacy)          # records made before profiles existed mean the 4B profile


def test_guard_refuses_wrong_model_revision_dtype_depths_device_recipe_hash():
    for over in ({"model_repo": "Qwen/Qwen3-0.6B"}, {"model_revision": "0" * 40}, {"dtype": "bfloat16"}, {"depths": [14, 19, 23]}, {"recipe_id": "other"}, {"hidden": 1024},
                 {"device": "NVIDIA GeForce RTX 3050 Laptop GPU"}, {"device": "Tesla P100"}, {"capture_code_sha256": {"capture/capture.py": "0" * 64}}, {"profile_sha256": "0" * 64}, {"profile_id": "no-such"}):
        raises(RT.ProvenanceError, RT.check_metadata, prov(**over))


def test_head_incompatible_profile_states_are_refused_with_a_clear_message():
    e = raises(RT.ProvenanceError, RT.check_metadata, prov(P06, device="NVIDIA GeForce RTX 3050 Laptop GPU"))
    assert "DIAGNOSTIC" in str(e) and "head_compatible" in str(e) and "qwen3-0.6b" in str(e)
    hq, ho = torch.zeros(3, 1024).half(), torch.zeros(2, 3, 1024).half()
    e = raises(RT.ProvenanceError, RT.check_tensors, hq, ho, 2, P06)
    assert "DIAGNOSTIC" in str(e)
    raises(RT.ProvenanceError, RT.check_tensors, hq, ho, 2)                  # default (4B) profile: wrong shape
    raises(RT.ProvenanceError, RT.LiveScorer, lambda c, o: None, None, "NVIDIA GeForce RTX 3050", P06)


def test_layascorer_refuses_06b_states_with_the_reason_on_record():
    be = RT.VerifiedSavedScoreBackend("laya")
    rec = {"status": "ok", "origin": "live_capture_same_process", "provenance": prov(P06, device="NVIDIA GeForce RTX 3050 Laptop GPU"), "context_sha256": LF.sha256_text("ctx"), "option_sha256": [LF.sha256_text("a"), LF.sha256_text("b")],
           "raw_output_sha256": "x", "features_sha256": "y", "options": ["a", "b"], "scores": {"laya": {"score": [0.1, 0.2]}}}
    assert be.add(rec, rec, live_run="r1", verify_run="r2", verify_mode="replay") is False
    sc = laya.LayaScorer(authorized=True, backend=be)
    e = raises(RT.ProvenanceError, sc.score, "next_action", [{"context_text": "ctx", "options": ["a", "b"]}])
    assert "DIAGNOSTIC" in str(e), str(e)


def _fake_capture(context_text, options):
    seed = int(hashlib.sha256((context_text + "\x00" + "\x01".join(options)).encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(3, 2560, generator=g).half(), torch.randn(len(options), 3, 2560, generator=g).half(), {"status": "ok"}


def _registration(verdict="PASS", device_prefix="NVIDIA GeForce RTX 3050", tamper=False):
    d = Path(tempfile.mkdtemp())
    res = d / "PARITY_RESULT.json"
    res.write_text(json.dumps({"verdict": verdict, "device_prefix": device_prefix}), "utf-8")
    reg = d / "reg.toml"
    reg.write_text('[registration]\nprofile_id = "qwen3-4b-thinking-2507"\ndevice_prefix = "%s"\nparity_result = "%s"\nparity_result_sha256 = "%s"\n'
                   % (device_prefix, res.as_posix(), "0" * 64 if tamper else hashlib.sha256(res.read_bytes()).hexdigest()), "utf-8")
    return reg


def test_device_registration_needs_a_passed_hash_matched_parity_result_and_is_per_profile():
    BB.clear_registered_devices()
    dev = "NVIDIA GeForce RTX 3050 Laptop GPU"
    raises(RT.ProvenanceError, RT.check_metadata, prov(device=dev))                               # unregistered: refused
    raises(BB.ProfileError, BB.register_device, P4, dev, _registration(verdict="FAIL"))
    raises(BB.ProfileError, BB.register_device, P4, dev, _registration(tamper=True))
    raises(BB.ProfileError, BB.register_device, P4, "NVIDIA A100", _registration())
    raises(BB.ProfileError, BB.register_device, P06, dev, _registration())                         # the registration names the 4B profile
    raises(RT.ProvenanceError, RT.check_metadata, prov(device=dev))
    BB.register_device(P4, dev, _registration())
    try:
        RT.check_metadata(prov(device=dev))                                                        # registered for this process
        raises(RT.ProvenanceError, RT.check_metadata, prov(device="NVIDIA A100"))                  # every other device is still refused
        raises(RT.ProvenanceError, RT.check_metadata, prov(device="Tesla P100"))
        # T7 cases re-run with the extended list: a fabricated tensor with perfect metadata on the registered device fails the re-capture comparison
        ctx, opts = "context 1", ["a", "b", "c"]
        hq, ho, _ = _fake_capture(ctx, opts)
        good = {"decision_id": "d1", "context_text": ctx, "options": opts, "hq": hq, "ho": ho, "provenance": RT.make_provenance(device=dev, context_text=ctx, options=opts)}
        class FS:
            def _score(self, hq, ho): return {"laya": {"score": [0.0] * ho.shape[0]}}
        out = RT.score_stored([good], capture_fn=_fake_capture, scorer=FS(), sample_n=1, seed=1)
        assert out["origin"] == "stored_features_verified_by_recapture"
        fake = dict(good, hq=torch.randn(3, 2560).half(), ho=torch.randn(3, 3, 2560).half())
        e = raises(RT.ProvenanceError, RT.score_stored, [fake], capture_fn=_fake_capture, scorer=FS(), sample_n=1, seed=1)
        assert "differ from a fresh capture" in str(e)
        wrong_rev = dict(good, provenance=dict(good["provenance"], model_revision="0" * 40))
        raises(RT.ProvenanceError, RT.score_stored, [wrong_rev], capture_fn=_fake_capture, scorer=FS(), sample_n=1, seed=1)
    finally:
        BB.clear_registered_devices()
    raises(RT.ProvenanceError, RT.check_metadata, prov(device=dev))                                # cleared again


# ------------------------------------------------------------------ profile-aware generation backend
def test_hf_backend_local_gate_and_model_info_carry_the_profile():
    env = {"EVIDENCE_LADDER_ALLOW_INFERENCE": "1"}
    raises(NotAuthorized, Q.HFQwenBackend, authorized=True, env=env, path_exists=lambda p: False, profile=P06)            # not local, no /kaggle: refused as before
    raises(NotAuthorized, Q.HFQwenBackend, authorized=True, env={}, path_exists=lambda p: False, profile=P06, local=True)  # local still needs the inference switch
    raises(NotAuthorized, Q.HFQwenBackend, authorized=False, env=env, path_exists=lambda p: False, profile=P06, local=True)
    b = Q.HFQwenBackend(authorized=True, env=env, path_exists=lambda p: False, profile=P06, local=True)
    mi = b.model_info()
    assert mi["profile_id"] == "qwen3-0.6b" and mi["profile_sha256"] == P06.sha256 and mi["repo"] == "Qwen/Qwen3-0.6B" and mi["generation"] == {"do_sample": False} and mi["enable_thinking"] is True and mi["local"] is True
    d = Q.HFQwenBackend(authorized=True, env=env, path_exists=lambda p: True)       # default profile, Kaggle path unchanged
    assert d.model_info()["repo"] == Q.QWEN_REPO and d.model_info()["profile_id"] == "qwen3-4b-thinking-2507" and d.model_info()["local"] is False


def test_greedy_profile_repeats_seed_0_instead_of_regenerating():
    b = Q.HFQwenBackend(authorized=True, env={"EVIDENCE_LADDER_ALLOW_INFERENCE": "1"}, path_exists=lambda p: False, profile=P06, local=True)
    calls = []

    def one(torch, model, tok, request, seed, prompt_sha, lang):
        calls.append(seed)
        return Q.make_generation("</think>[]", tokens_prompt=10, tokens_generated=5, seed=seed, prompt_template_id="t", prompt_sha256=prompt_sha, language=lang, wall_seconds=1.5, model=P06.repo, model_revision=P06.revision)
    b._load = lambda: (object(), object()); b._generate_one = one
    gens = b.generate(GenerationRequest("t:1", "prompt", 2, (0, 1), 10, "sample", True, "qwen_actions_v1"))
    assert calls == [0] and [g.seed for g in gens] == [0, 1] and gens[0].text == gens[1].text and gens[1].tokens_generated == 0 and gens[0].tokens_generated == 5
    assert gens[0].model == "Qwen/Qwen3-0.6B" and gens[0].model_revision == P06.revision
    calls.clear()                                      # run_jobs calls generate once per (job, seed): the cache must hold across calls
    for sd in (0, 1):
        g = b.generate(GenerationRequest("t:2", "prompt2", 1, (sd,), 10, "sample", True, "qwen_actions_v1"))[0]
    assert calls == [0] and g.seed == 1 and g.tokens_generated == 0
    # a sampling profile (4B) still generates once per seed
    b4 = Q.HFQwenBackend(authorized=True, env={"EVIDENCE_LADDER_ALLOW_INFERENCE": "1"}, path_exists=lambda p: True)
    calls.clear(); b4._load = lambda: (object(), object()); b4._generate_one = one
    b4.generate(GenerationRequest("t:1", "prompt", 2, (0, 1), 10, "sample", True, "qwen_actions_v1"))
    assert calls == [0, 1]


def test_replayed_scorer_error_keeps_the_recorded_exception_type_name():
    from evidence_ladder.controller.replay import RecordedScorer
    rs = RecordedScorer("laya_adapter_v1", {}, [{"error": "ProvenanceError: no verified score", "error_type": "ProvenanceError"}])
    e = raises(RuntimeError, rs.score, "next_action", [])
    assert type(e).__name__ == "ProvenanceError" and str(e) == "no verified score"


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
