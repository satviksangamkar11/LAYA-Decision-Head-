"""Concurrency config loader: the batch sender's ONLY source of limits is configs/critic_concurrency_v1.toml (decision recorded in that file).
Refuses (ConcurrencyConfigError) if the file is missing, malformed, disagrees with broker.LIMITS, or names a different hash for the frozen thresholds file."""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from evidence_ladder.models.broker import LIMITS
from evidence_ladder.models.pc_sender import KeyScheduler

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "configs" / "critic_concurrency_v1.toml"
RAMP_KEYS = ("start_level", "step", "raise_after_completions", "recent_window", "max_recent_429_rate", "floor_level")


class ConcurrencyConfigError(Exception):
    pass


def load_concurrency(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_file():
        raise ConcurrencyConfigError("concurrency config missing: %s" % p.name)
    raw = p.read_bytes()
    try:
        d = tomllib.loads(raw.decode("utf-8"))
        lim, ramp, meta = d["limits"], d["ramp"], d["meta"]
    except (ValueError, KeyError) as e:
        raise ConcurrencyConfigError("concurrency config malformed: %s" % type(e).__name__)
    for k, v in LIMITS.items():
        if lim.get(k) != v:
            raise ConcurrencyConfigError("limit %s: file has %r, broker.LIMITS has %r" % (k, lim.get(k), v))
    if set(lim) != set(LIMITS):
        raise ConcurrencyConfigError("unexpected limit keys: %s" % sorted(set(lim) ^ set(LIMITS)))
    miss = [k for k in RAMP_KEYS if k not in ramp]
    if miss:
        raise ConcurrencyConfigError("ramp keys missing: %s" % miss)
    cap = lim["max_in_flight_per_key"]
    if not (1 <= ramp["floor_level"] <= ramp["start_level"] <= cap and ramp["step"] >= 1 and ramp["recent_window"] >= 1 and 0 < ramp["max_recent_429_rate"] <= 1):
        raise ConcurrencyConfigError("ramp values out of range")
    frozen = p.parent / "thresholds_evidence_ladder_v1.toml"
    if frozen.is_file() and hashlib.sha256(frozen.read_bytes()).hexdigest() != meta.get("frozen_thresholds_sha256"):
        raise ConcurrencyConfigError("frozen thresholds file hash differs from the one recorded in the concurrency config")
    return {"limits": dict(lim), "ramp": {k: ramp[k] for k in RAMP_KEYS}, "sha256": hashlib.sha256(raw).hexdigest(), "version": meta.get("version")}


def scheduler_from_config(key_labels: Sequence[str], cfg: Dict[str, Any]) -> KeyScheduler:
    L = cfg["limits"]
    return KeyScheduler(key_labels, max_in_flight=L["max_in_flight_per_key"], per_minute=L["max_sends_per_key_per_minute"],
                        cooldown_s=L["rate_limit_cooldown_s"], ramp=cfg["ramp"])
