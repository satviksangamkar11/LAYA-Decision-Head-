"""Runtime path of the LAYA scorer: it scores ONLY features that the frozen Qwen3-4B-Thinking-2507 itself produced, and it proves that by running the model, never by trusting metadata.

Live mode   LiveScorer.score_texts(context_text, options): captures from the frozen Qwen in the SAME process (capture_fn) and scores. There is no entry that takes feature tensors.
Replay mode score_stored(records, ...): stored features (fixture/offline) are accepted only after the runtime RE-CAPTURES a random subset from the recorded TEXT with the same recipe
            and the comparison passes the pre-registered tolerance (configs/laya_head_gate_v1.toml [a_feature_recapture]); any mismatch refuses the WHOLE batch.
Metadata (check_metadata) is a cheap first filter only (wrong revision / dtype / depths / recipe / device / code hash -> refuse early); a record with perfect metadata but fabricated
tensors is caught by the re-capture comparison. Text, TF-IDF or embedding inputs have no code path and are refused.
Output is L0: scores per option of ONE pool (N >= 2, no NONE class). Nothing here approves anything. torch is imported lazily."""
from __future__ import annotations

import math
import random
import secrets
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from evidence_ladder.models import laya_forward as LF

QWEN_REPO, QWEN_REVISION = "Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea"
RECIPE_ID = "fable_capture_v1"
CAPTURE_CODE_SHA256 = {"capture/capture.py": "d7e817d68fb6e0696ababd8add51ca929b92a2207d39885b11f0025db52710b8",          # byte-identical to the pkg/ copies of the TRAIN and VAL capture runs
                       "capture/capture_resident.py": "8fb5e492480f6130366fc3bb235e991e5c3e777a74f32fc609d63cb15831106a"}
DEVICE_CLASS = "Tesla T4"
DEFAULT_TOL = {"min_cosine": 0.9999, "max_abs_diff": 0.5}      # configs/laya_head_gate_v1.toml [a_feature_recapture]
PROVENANCE_FIELDS = ("model_repo", "model_revision", "recipe_id", "depths", "dtype", "hidden", "query_cue", "option_prefix", "device", "capture_code_sha256", "context_sha256", "option_sha256")


class ProvenanceError(ValueError):
    """Features are not (provably) frozen-Qwen3-4B-Thinking-2507 states of the training recipe. The scorer refuses; there is no fallback."""


def make_provenance(*, device: str, context_text: str, options: Sequence[str]) -> Dict[str, Any]:
    return {"model_repo": QWEN_REPO, "model_revision": QWEN_REVISION, "recipe_id": RECIPE_ID, "depths": list(LF.DEPTHS), "dtype": "float16", "hidden": LF.HIDDEN,
            "query_cue": LF.QUERY_CUE, "option_prefix": " ", "device": device, "capture_code_sha256": dict(CAPTURE_CODE_SHA256),
            "context_sha256": LF.sha256_text(context_text), "option_sha256": [LF.sha256_text(o) for o in options]}


def check_metadata(prov: Any) -> None:
    """Cheap first filter. NEVER sufficient: metadata cannot prove where tensors came from."""
    if not isinstance(prov, Mapping):
        raise ProvenanceError("no provenance record: features of unknown origin are refused")
    miss = [k for k in PROVENANCE_FIELDS if k not in prov]
    if miss:
        raise ProvenanceError("provenance lacks fields %s" % miss)
    want = {"model_repo": QWEN_REPO, "model_revision": QWEN_REVISION, "recipe_id": RECIPE_ID, "dtype": "float16", "hidden": LF.HIDDEN, "query_cue": LF.QUERY_CUE, "option_prefix": " "}
    for k, v in want.items():
        if prov[k] != v:
            raise ProvenanceError("provenance %s = %r, expected %r (not the frozen Qwen3-4B-Thinking-2507 training recipe)" % (k, prov[k], v))
    if list(prov["depths"]) != list(LF.DEPTHS):
        raise ProvenanceError("depths %r, expected %r" % (prov["depths"], list(LF.DEPTHS)))
    if not str(prov["device"]).startswith(DEVICE_CLASS):
        raise ProvenanceError("captured on %r; the adapters were trained on %s features" % (prov["device"], DEVICE_CLASS))
    if dict(prov["capture_code_sha256"]) != CAPTURE_CODE_SHA256:
        raise ProvenanceError("capture code hashes differ from the code that produced the training features")


def check_tensors(hq, ho, n_options: Optional[int] = None) -> None:
    import torch
    if not (torch.is_tensor(hq) and torch.is_tensor(ho)):
        raise ProvenanceError("features must be tensors captured from the frozen Qwen (text, embeddings and TF-IDF are refused)")
    if hq.dtype != torch.float16 or ho.dtype != torch.float16:
        raise ProvenanceError("features must be float16 (training dtype), got %s / %s" % (hq.dtype, ho.dtype))
    if tuple(hq.shape) != (len(LF.DEPTHS), LF.HIDDEN) or ho.dim() != 3 or tuple(ho.shape[1:]) != (len(LF.DEPTHS), LF.HIDDEN):
        raise ProvenanceError("feature shapes %s / %s are not [3,2560] / [N,3,2560]" % (tuple(hq.shape), tuple(ho.shape)))
    if ho.shape[0] < 2 or (n_options is not None and ho.shape[0] != n_options):
        raise ProvenanceError("option count %d (expected %s, minimum 2)" % (ho.shape[0], n_options))
    if not (torch.isfinite(hq).all() and torch.isfinite(ho).all()):
        raise ProvenanceError("non-finite features")


def compare_features(a: Tuple[Any, Any], b: Tuple[Any, Any], tol: Mapping[str, float] = DEFAULT_TOL) -> Dict[str, Any]:
    """(hq, ho) vs (hq, ho): worst cosine (float64, per vector) and worst abs diff, against the pre-registered tolerance."""
    import torch
    import torch.nn.functional as F
    cos = min(F.cosine_similarity(a[0].double(), b[0].double(), dim=-1).min().item(), F.cosine_similarity(a[1].double(), b[1].double(), dim=-1).min().item())
    mad = max((a[0].float() - b[0].float()).abs().max().item(), (a[1].float() - b[1].float()).abs().max().item())
    return {"cos_min": cos, "max_abs_diff": mad, "passed": bool(cos >= tol["min_cosine"] and mad <= tol["max_abs_diff"])}


class FeatureScorer:
    """The real forward pass. INTERNAL: reached only through LiveScorer (own capture) or score_stored (after re-capture verification)."""

    def __init__(self, heads: Mapping[str, Any], probe=None, random_head=None):
        self.heads, self.probe, self.random_head = dict(heads), probe, random_head

    def _score(self, hq, ho) -> Dict[str, Any]:
        out = {}
        per = {n: LF.head_logits(h, hq, ho) for n, h in self.heads.items()}
        out["laya"] = {"per_seed_logits": per, "score": [sum(per[n][i] for n in per) / len(per) for i in range(ho.shape[0])]}
        if self.probe is not None:
            out["probe"] = {"score": LF.probe_decision(self.probe, ho)}
        if self.random_head is not None:
            if isinstance(self.random_head, Mapping):    # [EL-ARCH-1 C1] the proper control: several seeded untrained heads (evidence_ladder.models.random_head)
                from evidence_ladder.models import random_head as RH
                out["random_head"] = RH.random_head_scores(self.random_head, hq, ho)
            else:                                        # the old single-head control (mismatch control when built on a trained adapter)
                out["random_head"] = {"score": LF.head_logits(self.random_head, hq, ho)}
        for k, v in out.items():
            v.setdefault("role", LF.ROLES[k])
        return out


CaptureFn = Callable[[str, Sequence[str]], Tuple[Any, Any, Dict[str, Any]]]   # (context_text, options) -> (hq, ho, info) from the frozen Qwen; hq None on exclusion


class LiveScorer:
    """Live mode: text in, scores out. The capture happens inside this call, in this process, on the frozen Qwen; no tensor argument exists."""

    def __init__(self, capture_fn: CaptureFn, scorer: FeatureScorer, device_name: str):
        self._capture, self._scorer, self.device = capture_fn, scorer, device_name

    def score_texts(self, context_text: str, options: Sequence[str]) -> Dict[str, Any]:
        if not isinstance(context_text, str) or not isinstance(options, (list, tuple)) or not all(isinstance(o, str) for o in options):
            raise ProvenanceError("live scoring takes the context text and option texts")
        if len(options) < 2:
            raise ProvenanceError("a pool needs at least 2 options (no NONE class is added)")
        t = time.time()
        hq, ho, info = self._capture(context_text, list(options))
        if hq is None:
            raise ProvenanceError("capture excluded this decision: %s" % info.get("status"))
        check_tensors(hq, ho, len(options))
        res = self._scorer._score(hq, ho)
        return {"scores": res, "provenance": make_provenance(device=self.device, context_text=context_text, options=options), "capture": info,
                "seconds": round(time.time() - t, 3), "origin": "live_capture_same_process"}


def score_stored(records: Sequence[Mapping[str, Any]], *, capture_fn: CaptureFn, scorer: FeatureScorer, tol: Mapping[str, float] = DEFAULT_TOL,
                 sample_n: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Replay mode. records: {decision_id, context_text, options, provenance, hq, ho}. Metadata filter on every record, then a RANDOM subset (>= 25% of the batch, at least
    min(3, len)) is re-captured from the recorded text and compared with the stored tensors; one mismatch refuses the WHOLE batch. Limitation, recorded: records outside the sample
    are covered statistically, not individually; sample_n = len(records) verifies all. -> {scores: {decision_id: ...}, verified_ids, comparisons, seed}."""
    if not records:
        raise ProvenanceError("empty batch")
    for r in records:
        for k in ("context_text", "options", "hq", "ho", "provenance"):
            if k not in r:
                raise ProvenanceError("record %r lacks %s" % (r.get("decision_id"), k))
        check_metadata(r["provenance"])
        if r["provenance"]["context_sha256"] != LF.sha256_text(r["context_text"]) or list(r["provenance"]["option_sha256"]) != [LF.sha256_text(o) for o in r["options"]]:
            raise ProvenanceError("record %r: recorded text does not match the provenance hashes" % r.get("decision_id"))
        check_tensors(r["hq"], r["ho"], len(r["options"]))
    n = max(min(3, len(records)), math.ceil(0.25 * len(records))) if sample_n is None else min(int(sample_n), len(records))   # integration tests pass sample_n=len(batch): 100%
    seed = secrets.randbits(32) if seed is None else int(seed)
    chosen = random.Random(seed).sample(range(len(records)), n)
    comps = []
    for i in sorted(chosen):
        r = records[i]
        hq, ho, info = capture_fn(r["context_text"], list(r["options"]))
        if hq is None:
            raise ProvenanceError("re-capture of %r excluded (%s): batch refused" % (r.get("decision_id"), info.get("status")))
        c = dict(compare_features((hq, ho), (r["hq"], r["ho"]), tol), decision_id=r.get("decision_id"))
        comps.append(c)
        if not c["passed"]:
            raise ProvenanceError("stored features of %r differ from a fresh capture of the same text (cos_min %.6f, max_abs %.4f): whole batch refused"
                                  % (r.get("decision_id"), c["cos_min"], c["max_abs_diff"]))
    out = {r.get("decision_id", i): scorer._score(r["hq"], r["ho"]) for i, r in enumerate(records)}
    return {"scores": out, "verified_ids": [records[i].get("decision_id") for i in sorted(chosen)], "comparisons": comps, "seed": seed, "sampled": n, "batch": len(records),
            "origin": "stored_features_verified_by_recapture"}


class LiveBackend:
    """Backend for evidence_ladder.models.laya scorers (decision type next_action): pool = {context_text, options} or a one-element list of it. The options are scored by a capture
    made inside this call. A list of plain strings (no context) is refused: there is no text-only fallback."""

    def __init__(self, live: LiveScorer, kind: str = "laya"):
        assert kind in ("laya", "probe", "random_head"), kind
        self.live, self.kind, self.last = live, kind, None

    def score(self, decision_type: str, pool: Any) -> List[float]:
        req = pool[0] if isinstance(pool, (list, tuple)) and len(pool) == 1 and isinstance(pool[0], Mapping) else pool
        if not isinstance(req, Mapping) or "context_text" not in req or "options" not in req:
            raise ProvenanceError("the scorer needs {context_text, options} to capture from the frozen Qwen; bare candidates/text are refused (no text-only fallback)")
        self.last = self.live.score_texts(req["context_text"], req["options"])
        sc = self.last["scores"].get(self.kind)
        if sc is None:
            raise ProvenanceError("scorer kind %r not available" % self.kind)
        return [float(x) for x in sc["score"]]


class VerifiedSavedScoreBackend:
    """PC-side backend for scores computed on the Kaggle T4. Imported results are NOT trusted by label: a pool is loaded only if TWO independent Kaggle runs agree on it,
    the live run (Qwen proposes, captures, scores) and a verification run (separate kernel version/process: re-captures the SAME saved proposals from the same context
    text with the training recipe and re-scores). Required equal in both: context_sha256, option_sha256, raw_output_sha256, features_sha256 and every score vector
    (laya per seed, probe, random head); both must carry origin live_capture_same_process, valid metadata provenance, and the verification run must be in replay mode.
    Anything else is refused (ProvenanceError) and the controller records scorer_unavailable. Pool = {context_text, options}; keyed by the text hashes."""

    def __init__(self, kind: str = "laya"):
        assert kind in ("laya", "probe", "random_head"), kind
        self.kind, self._pools, self.last, self.refused = kind, {}, None, []

    def add(self, live: Mapping[str, Any], verify: Mapping[str, Any], *, live_run: str, verify_run: str, verify_mode: str) -> bool:
        why = None
        try:
            if live_run == verify_run:
                raise ProvenanceError("live and verification results come from the same run")
            if verify_mode != "replay":
                raise ProvenanceError("the verification run is not a replay of the saved proposals")
            for r in (live, verify):
                if r.get("status") != "ok" or r.get("origin") != "live_capture_same_process":
                    raise ProvenanceError("a result lacks live-capture origin")
                check_metadata(r.get("provenance"))
            for k in ("context_sha256", "option_sha256", "raw_output_sha256", "features_sha256", "options"):
                if live.get(k) != verify.get(k):
                    raise ProvenanceError("live and verification runs disagree on %s" % k)
            if live["scores"] != verify["scores"]:
                raise ProvenanceError("live and verification runs disagree on the scores")
            if [LF.sha256_text(o) for o in live["options"]] != list(live["option_sha256"]):
                raise ProvenanceError("option texts do not match their hashes")
        except ProvenanceError as e:
            why = str(e)
        if why:
            self.refused.append({"context_sha256": live.get("context_sha256"), "reason": why}); return False
        self._pools[(live["context_sha256"], tuple(live["option_sha256"]))] = {"live": live, "verify": verify, "live_run": live_run, "verify_run": verify_run}
        return True

    def score(self, decision_type: str, pool: Any) -> List[float]:
        req = pool[0] if isinstance(pool, (list, tuple)) and len(pool) == 1 and isinstance(pool[0], Mapping) else pool
        if not isinstance(req, Mapping) or "context_text" not in req or "options" not in req:
            raise ProvenanceError("the scorer needs {context_text, options}; bare candidates/text are refused (no text-only fallback)")
        key = (LF.sha256_text(req["context_text"]), tuple(LF.sha256_text(o) for o in req["options"]))
        rec = self._pools.get(key)
        if rec is None:
            raise ProvenanceError("no verified score for this context/option pool (unverified or never scored): refused")
        sc = rec["live"]["scores"].get(self.kind)
        if sc is None:
            raise ProvenanceError("scorer kind %r not available" % self.kind)
        self.last = {"provenance": rec["live"]["provenance"], "origin": "saved_scores_verified_by_independent_recapture", "live_run": rec["live_run"], "verify_run": rec["verify_run"],
                     "features_sha256": rec["live"]["features_sha256"]}
        return [float(x) for x in sc["score"]]
