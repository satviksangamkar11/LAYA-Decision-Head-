"""[EL-ARCH-1 A3] Per-problem limits (wall time, Qwen tokens, GLM tokens, actions, rounds) read from configs/el_arch1_limits_v1.toml and ENFORCED by the controllers
with an injected clock. exceeded() -> None or {"resource", "used", "cap", "details"}; the controllers stop with stop_cause "budget" and put `details` in stop_details.
A missing/incomplete limits file raises LimitsConfigError (no silent unlimited run)."""
from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "el_arch1_limits_v1.toml"
KEYS = ("max_wall_seconds", "max_qwen_tokens", "max_glm_tokens", "max_actions", "max_rounds")


class LimitsConfigError(Exception):
    pass


@dataclass(frozen=True)
class Limits:
    max_wall_seconds: float
    max_qwen_tokens: int
    max_glm_tokens: int
    max_actions: int
    max_rounds: int

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in KEYS}


def load_limits(profile: str, path=DEFAULT_PATH) -> Limits:
    try:
        d = tomllib.loads(Path(path).read_text(encoding="utf-8"))["profile"][profile]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        raise LimitsConfigError("cannot read limits profile %r from %s: %s" % (profile, path, type(e).__name__))
    missing = [k for k in KEYS if k not in d or isinstance(d[k], bool) or not isinstance(d[k], (int, float)) or d[k] <= 0]
    if missing:
        raise LimitsConfigError("limits profile %r: missing or non-positive %s" % (profile, missing))
    return Limits(float(d["max_wall_seconds"]), int(d["max_qwen_tokens"]), int(d["max_glm_tokens"]), int(d["max_actions"]), int(d["max_rounds"]))


class Limiter:
    """Wall clock starts at start(). `used` values come from the caller (budget record, action and round counters), so the limiter itself never sees a model."""

    def __init__(self, limits: Limits, clock: Callable[[], float] = time.monotonic):
        self.limits, self.clock, self._t0 = limits, clock, None
        self.hit: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        if self._t0 is None:
            self._t0 = self.clock()

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else max(0.0, self.clock() - self._t0)

    def exceeded(self, qwen_tokens: int = 0, glm_tokens: int = 0, actions: int = 0, rounds: int = 0, *, at_cap_counts: bool = False) -> Optional[Dict[str, Any]]:
        """Counters (actions, rounds) are 'exhausted' when used >= cap (nothing more may START); wall time and tokens when used > cap. at_cap_counts is for the post-hoc invariant (used > cap)."""
        L = self.limits
        checks = (("max_wall_seconds", self.elapsed(), L.max_wall_seconds, False), ("max_qwen_tokens", qwen_tokens, L.max_qwen_tokens, False),
                  ("max_glm_tokens", glm_tokens, L.max_glm_tokens, False), ("max_actions", actions, L.max_actions, True), ("max_rounds", rounds, L.max_rounds, True))
        for name, used, cap, counter in checks:
            if (used >= cap and counter and not at_cap_counts) or used > cap:
                r = {"resource": name, "used": used, "cap": cap, "details": "limit:%s used=%s cap=%s" % (name, round(used, 3) if isinstance(used, float) else used, cap)}
                self.hit = self.hit or r
                return r
        return None
