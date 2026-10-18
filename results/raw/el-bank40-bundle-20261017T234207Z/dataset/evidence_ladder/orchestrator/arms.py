"""Arm -> controller config. Agent 1's controller implements the modes; this file only names them."""
from __future__ import annotations

from evidence_ladder.orchestrator.base import ConfigError

ARMS = ("A", "B", "C", "D0", "D1", "E")
E_LABEL = "exploratory_transfer_no_keep_drop"
# (challenger, baseline) in the order the toml names them: C_vs_B means C against baseline B.
CONFIRMATORY = (("C", "B"), ("D0", "C"), ("D1", "D0"))
EXPLORATORY = (("B", "A"), ("E", "D1"))


# [EL-ARCH-1 C2] Arm E: LAYA is the only scorer the controller sees (L0). The probe (mandatory baseline) and the random-head floor control score the SAME rendered pools and go to the report only.
E_SELECTOR_KINDS = ("laya",)
E_COMPARISON_KINDS = ("probe", "random_head")


class StageOneLayaDecisionRefused(Exception):
    pass


def arm_config(arm: str, k: int = None) -> dict:
    if arm == "A":
        return {"mode": "single"}
    if arm == "B":
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ConfigError("arm B needs k (best-of-k); k is a tuned operational parameter, PENDING in the thresholds toml")
        return {"mode": "best_of_k", "k": k}
    if arm == "C":
        return {"mode": "repair"}
    if arm == "D0":
        return {"mode": "ladder"}
    if arm == "D1":
        return {"mode": "ladder", "critic": True}
    if arm == "E":
        return {"mode": "ladder", "critic": True, "scorer": True, "exploratory": True}
    raise ConfigError(f"unknown arm {arm!r}")


def arm_label(arm: str) -> str:
    return E_LABEL if arm == "E" else ""


def needs_critic(arm: str) -> bool:
    return arm in ("D1", "E")


def laya_keep_drop(*_a, **_k):
    """Stage-1 LAYA is an exploratory transfer test. No keep/drop decision may come from this data."""
    raise StageOneLayaDecisionRefused(E_LABEL)
