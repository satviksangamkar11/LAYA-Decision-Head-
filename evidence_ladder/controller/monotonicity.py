"""[EL-ARCH-1 C3] LAYA monotonicity gate (architecture section 7, ADAP): verification effort may be ordered by the LAYA score only if, on development pools with TRUSTED outcomes,
the pass rate falls with LAYA rank. Status is SUPPORTED / NOT_SUPPORTED / INSUFFICIENT_DATA; ordering by LAYA is DISABLED unless SUPPORTED (default: disabled).
A gate state never lets a learned score act above trusted evidence: verification_order always sorts by trusted passes first (invariant I2) and uses the score only below it.
Parameters come from configs/el_arch1_gates_v1.toml [monotonicity] and [min_n]. Pure standard library."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SUPPORTED, NOT_SUPPORTED, INSUFFICIENT = "SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_DATA"
GATES_PATH = Path(__file__).resolve().parents[2] / "configs" / "el_arch1_gates_v1.toml"


def load_gate_config(path=GATES_PATH) -> dict:
    import tomllib
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _pair_counts(scores: Sequence[float], passed: Sequence[bool]):
    """(concordant + 0.5 ties, pairs) over (passing, failing) candidate pairs of ONE pool: a pair is concordant when the passing candidate has the higher score."""
    c = n = 0.0
    for i, pi in enumerate(passed):
        if not pi:
            continue
        for j, pj in enumerate(passed):
            if pj:
                continue
            n += 1
            c += 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
    return c, n


def position_pass_rates(pools: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Pass rate by LAYA rank position (1 = best). A group tied on score spreads each member's weight evenly over the positions the group occupies."""
    num: Dict[int, float] = {}
    den: Dict[int, float] = {}
    npools: Dict[int, int] = {}
    for p in pools:
        order = sorted(range(len(p["scores"])), key=lambda i: -float(p["scores"][i]))
        pos = 0
        while pos < len(order):
            grp = [order[pos]]
            while pos + len(grp) < len(order) and float(p["scores"][order[pos + len(grp)]]) == float(p["scores"][grp[0]]):
                grp.append(order[pos + len(grp)])
            for off in range(len(grp)):
                r = pos + off + 1
                num[r] = num.get(r, 0.0) + sum(1.0 for i in grp if p["passed"][i]) / len(grp)
                den[r] = den.get(r, 0.0) + 1.0
            pos += len(grp)
        for r in range(1, len(order) + 1):
            npools[r] = npools.get(r, 0) + 1
    return [{"rank": r, "pools": npools[r], "pass_rate": num[r] / den[r]} for r in sorted(den)]


def _auc(pools, idx) -> Optional[float]:
    c = n = 0.0
    for i in idx:
        a, b = _pair_counts(pools[i]["scores"], pools[i]["passed"])
        c += a
        n += b
    return c / n if n else None


def monotonicity_gate(pools: Sequence[Mapping[str, Any]], *, config: Optional[dict] = None, seed: int = 0) -> Dict[str, Any]:
    """pools: [{"id", "scores": [LAYA score per candidate], "passed": [trusted outcome per candidate, bool]}]. Never raises on small data: it returns INSUFFICIENT_DATA."""
    cfg = config or load_gate_config()
    mn, mc = cfg["min_n"], cfg["monotonicity"]
    for p in pools:
        if len(p["scores"]) != len(p["passed"]):
            raise ValueError("pool %r: scores and outcomes differ in length" % p.get("id"))
    informative = [p for p in pools if len(p["scores"]) >= 2 and any(p["passed"]) and not all(p["passed"])]
    out: Dict[str, Any] = {"n_pools": len(pools), "n_informative_pools": len(informative), "min_informative_pools": mn["monotonicity_informative_pools"],
                           "role": "development check of a LAYA keep condition; not evidence of task success",
                           "laya_ordering_enabled": False, "can_override_trusted_evidence": False}
    if len(informative) < mn["monotonicity_informative_pools"]:
        out.update(status=INSUFFICIENT, reason="%d informative pools (a pool with at least one passing and one failing candidate), minimum %d: no verdict"
                   % (len(informative), mn["monotonicity_informative_pools"]))
        return out
    auc = _auc(informative, range(len(informative)))
    rng = random.Random(seed)
    boots = sorted(a for a in (_auc(informative, [rng.randrange(len(informative)) for _ in informative]) for _ in range(mc["bootstrap_resamples"])) if a is not None)
    lo_i, hi_i = int(math.floor((1 - mc["ci_level"]) / 2 * len(boots))), int(math.ceil((1 + mc["ci_level"]) / 2 * len(boots))) - 1
    ci = (boots[lo_i], boots[min(hi_i, len(boots) - 1)])
    rates = [r for r in position_pass_rates(informative) if r["pools"] >= mn["monotonicity_pools_per_rank"]]
    shape_ok = all(b["pass_rate"] <= a["pass_rate"] + mc["rank_tolerance"] for a, b in zip(rates, rates[1:]))
    out.update(auc=auc, auc_ci=list(ci), auc_lower_bound_required=mc["auc_lower_bound"], position_pass_rates=rates, shape_tolerance=mc["rank_tolerance"], seed=seed)
    if len(rates) < 2:
        out.update(status=INSUFFICIENT, reason="fewer than 2 rank positions reach %d pools: the monotone shape cannot be checked" % mn["monotonicity_pools_per_rank"])
    elif ci[0] > mc["auc_lower_bound"] and shape_ok:
        out.update(status=SUPPORTED, reason="CI lower bound above %.2f and pass rate non-increasing with rank (tolerance %.2f)" % (mc["auc_lower_bound"], mc["rank_tolerance"]),
                   laya_ordering_enabled=True)
    else:
        out.update(status=NOT_SUPPORTED, reason="CI lower bound %.3f vs required >%.2f; monotone shape %s" % (ci[0], mc["auc_lower_bound"], "holds" if shape_ok else "violated"))
    return out


def verification_order(gate: Optional[Mapping[str, Any]], items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Order candidates for verification effort. items: {id, trusted_passes, cost, scorer}. Key 1 is ALWAYS trusted passes (I2): no gate state lets the score reorder across it.
    The LAYA score is a key only when gate["status"] == SUPPORTED; otherwise the fallback is cheapest-check-first, then id."""
    enabled = bool(gate and gate.get("status") == SUPPORTED and gate.get("laya_ordering_enabled") is True)
    if enabled:
        key = lambda c: (-c["trusted_passes"], -float(c.get("scorer", 0.0)), c.get("cost", 0), c["id"])
    else:
        key = lambda c: (-c["trusted_passes"], c.get("cost", 0), c["id"])
    return {"order": [c["id"] for c in sorted(items, key=key)], "laya_used": enabled,
            "fallback": None if enabled else "cheapest_check_first", "gate_status": (gate or {}).get("status", "NOT_RUN")}
