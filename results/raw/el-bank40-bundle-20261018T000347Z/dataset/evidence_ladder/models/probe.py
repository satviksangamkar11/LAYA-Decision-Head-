"""[EL-ARCH-1 C1] The mandatory baseline: the linear probe on the SAME frozen-Qwen option states LAYA scores. No weights file exists (MANIFEST linear_probe.frozen_as): the probe is a pure
function of the stored TRAIN features and is refitted at scoring time. refit_probe() is that function: same rows in the same order -> the same fit, fingerprinted so a refit can be
compared with an earlier one. Recipe = laya_forward.fit_probe (kaggle/head_exp4_gpu.py probe_scores with C pinned): standardised raw option states (3 depths x 2560 flattened),
logistic regression, class_weight balanced, max_iter 2000. No adapter, head or encoder is touched. Heavy imports are lazy (torch/sklearn arrive with the rows)."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence

from evidence_ladder.models import laya_forward as LF

ROLE = "mandatory_baseline_linear_probe"
TRAIN_POOLS = 334            # MANIFEST: the 334 v3 TRAIN pools (v2 + v3 captures)
PROBE_C = 0.01               # configs/laya_int1_thresholds_v1.toml [d_probe] probe_C (pinned; the original C was chosen by grouped CV in (1e-3, 1e-2, 1e-1))
SEED = 0


class ProbeFitError(ValueError):
    pass


def check_rows(rows: Sequence[Mapping[str, Any]], depths: int = len(LF.DEPTHS), hidden: int = LF.HIDDEN) -> None:
    """Every row: ho [N, depths, hidden], pos bool [N] with at least one positive; N >= 2. Same pools, same depths as the head sees."""
    if not rows:
        raise ProbeFitError("no TRAIN rows")
    for r in rows:
        ho, pos = r["ho"], r["pos"]
        if ho.dim() != 3 or tuple(ho.shape[1:]) != (depths, hidden) or ho.shape[0] < 2 or len(pos) != ho.shape[0]:
            raise ProbeFitError("row %r: ho %s / pos %s do not match [N>=2, %d, %d]" % (r.get("decision_id"), tuple(ho.shape), tuple(pos.shape), depths, hidden))
        if not bool(pos.any()):
            raise ProbeFitError("row %r has no positive option" % r.get("decision_id"))


def order_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(r.get("decision_id")) for r in rows).encode("utf-8")).hexdigest()


def fingerprint(probe) -> str:
    """sha256 of the fitted scaler and logistic model (float64 bytes): equal fingerprints mean an identical probe."""
    sc, m = probe
    h = hashlib.sha256()
    for a in (sc.mean_, sc.scale_, m.coef_, m.intercept_):
        h.update(a.astype("float64").tobytes())
    return h.hexdigest()


def refit_probe(rows: Sequence[Mapping[str, Any]], *, C: float = PROBE_C, seed: int = SEED, expect_pools: Optional[int] = None,
                depths: int = len(LF.DEPTHS), hidden: int = LF.HIDDEN) -> Dict[str, Any]:
    """Pure refit. rows = the stored TRAIN pools in their stored order ({decision_id, ho tensor [N,3,2560], pos bool tensor [N]}); expect_pools=TRAIN_POOLS for the real refit.
    -> {probe (scaler, model), fingerprint, order_sha256, n_pools, n_options, C, seed, role, recipe}. The seed is passed to the solver; lbfgs uses no randomness, so the fit depends on the rows and their order only."""
    check_rows(rows, depths, hidden)
    if expect_pools is not None and len(rows) != expect_pools:
        raise ProbeFitError("%d TRAIN pools, expected %d (the probe is defined on exactly those pools)" % (len(rows), expect_pools))
    probe = LF.fit_probe(rows, C=C, random_state=seed)
    return {"probe": probe, "fingerprint": fingerprint(probe), "order_sha256": order_sha(rows), "n_pools": len(rows), "n_options": int(sum(len(r["pos"]) for r in rows)), "C": C, "seed": seed,
            "role": ROLE, "recipe": "standardised raw option states, logistic regression class_weight balanced, max_iter 2000 (laya_forward.fit_probe)"}


def probe_scores(fit: Mapping[str, Any], ho) -> Dict[str, Any]:
    """Scores of ONE candidate pool through the same interface shape LAYA's scorer returns ({'score': [...]}), labelled with its role."""
    return {"score": LF.probe_decision(fit["probe"], ho), "role": ROLE, "fingerprint": fit["fingerprint"]}
