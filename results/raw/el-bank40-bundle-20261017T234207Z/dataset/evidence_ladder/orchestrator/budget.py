"""Matched-token budget. T caps Qwen generated + GLM output tokens per task and arm. This is a matched-token design, NOT an equal-cost design."""
from __future__ import annotations

from typing import Any, Dict

from evidence_ladder.common.interfaces import BudgetExhausted
from evidence_ladder.orchestrator.base import ConfigError, load_toml

MATCHING_LABEL = "matched-token experiment, not an equal-cost experiment"
_KINDS = ("qwen_gen", "qwen_prefill", "glm_in", "glm_out")
_CALLS = ("glm_calls", "scorer_calls", "tool_runs", "api_errors")


def resolve_T(explicit=None, toml_path=None):
    """T must be an explicit positive int (from a future versioned toml). The frozen toml holds a PENDING string; that is a ConfigError."""
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit <= 0:
            raise ConfigError(f"token cap T must be a positive int, got {explicit!r}")
        return explicit
    pending = load_toml(toml_path) if toml_path else load_toml()
    v = pending.get("pending", {}).get("values", {}).get("token_cap_T")
    raise ConfigError(f"token cap T is PENDING (thresholds toml [pending.values].token_cap_T = {v!r}); set it from pilot stage 0b in a new versioned toml")


class MatchedTokenBudget:
    def __init__(self, problem_id: str, arm: str, T: int = None):
        if T is None:
            resolve_T(None)   # raises ConfigError naming the pending value
        self.T = resolve_T(T)
        self.problem_id, self.arm = problem_id, arm
        self.c = {k: 0 for k in _KINDS}
        self.n = {k: 0 for k in _CALLS}
        self.gpu_seconds = 0.0
        self.wall_seconds = 0.0

    def generated(self) -> int:
        return self.c["qwen_gen"] + self.c["glm_out"]

    def total_incl_inputs(self) -> int:
        """[EL-ARCH-1 C5] Accounting only (the cap T stays on generated tokens): prefill and GLM input tokens are counted too, so equal-compute twins can be checked."""
        return sum(self.c.values())

    def remaining(self) -> int:
        return max(0, self.T - self.generated())

    def charge(self, kind: str, tokens: int) -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown token kind {kind!r}")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("tokens must be a non-negative int")
        if kind in ("glm_in", "glm_out") and self.arm in ("A", "B", "C", "D0"):
            raise ValueError(f"arm {self.arm} has no GLM; its tokens are spent only on Qwen")
        self.c[kind] += tokens   # tokens already spent are always recorded, then the ceiling is checked
        if self.generated() > self.T:
            raise BudgetExhausted(f"generated tokens {self.generated()} crossed T={self.T} (qwen_gen + glm_out)")

    def exhausted(self) -> bool:
        return self.generated() >= self.T   # no room left; charge() raises only when T is exceeded

    def note_call(self, name: str, n: int = 1) -> None:
        self.n[name] += n    # KeyError on an unknown counter is intended

    def note_time(self, gpu_seconds: float = 0.0, wall_seconds: float = 0.0) -> None:
        self.gpu_seconds += gpu_seconds
        self.wall_seconds += wall_seconds

    def record(self) -> Dict[str, Any]:
        """budget_record per schemas_v1.json."""
        return {"problem_id": self.problem_id, "arm": self.arm, "qwen_gen_tokens": self.c["qwen_gen"], "qwen_prefill_tokens": self.c["qwen_prefill"],
                "glm_tokens_in": self.c["glm_in"], "glm_tokens_out": self.c["glm_out"], "glm_calls": self.n["glm_calls"],
                "scorer_calls": self.n["scorer_calls"], "tool_runs": self.n["tool_runs"], "gpu_seconds": float(self.gpu_seconds),
                "wall_seconds": float(self.wall_seconds), "api_errors": self.n["api_errors"]}

    def sidecar(self) -> Dict[str, Any]:
        return {"T": self.T, "generated_tokens": self.generated(), "remaining": self.remaining(), "exhausted": self.exhausted(), "total_tokens_incl_inputs": self.total_incl_inputs(),
                "matching": MATCHING_LABEL, "gpu_seconds_basis": "sum of generation wall seconds reported by the generation backend (approximation)"}


def extra_qwen_samples_for_b(glm_out_tokens: int, mean_qwen_sample_tokens: float) -> int:
    """Accounting rule for B: the tokens D1 spends on GLM output are given back to B as additional Qwen samples (matched-token, same T)."""
    if mean_qwen_sample_tokens <= 0:
        raise ValueError("mean sample tokens must be positive")
    return int(glm_out_tokens // mean_qwen_sample_tokens)


def extra_b_samples_plan(d1_records, b_records, b_k: int) -> dict:
    """[EL-ARCH-1 C5] The rule, applied per task by the runner: arm B gets back as extra Qwen samples what D1 spent on GLM output tokens.
    d1_records / b_records: budget_records ({problem_id, qwen_gen_tokens, glm_tokens_out, ...}). Mean sample size = B's own generated tokens / b_k.
    -> {task_id: {extra_samples, glm_out_tokens, mean_qwen_sample_tokens, glm_in_tokens_not_matched}}. GLM input tokens are reported, not converted (T counts generated tokens only)."""
    b = {r["problem_id"]: r for r in b_records}
    out = {}
    for r in d1_records:
        t = r["problem_id"]
        if t not in b or r["glm_tokens_out"] <= 0 or b[t]["qwen_gen_tokens"] <= 0:
            continue
        mean = b[t]["qwen_gen_tokens"] / b_k
        out[t] = {"extra_samples": extra_qwen_samples_for_b(r["glm_tokens_out"], mean), "glm_out_tokens": r["glm_tokens_out"], "mean_qwen_sample_tokens": mean,
                  "glm_in_tokens_not_matched": r["glm_tokens_in"]}
    return out
