"""Common random numbers and identical starting state per arm. Paired records are keyed by task_id."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from evidence_ladder.common.interfaces import ControllerContext, TaskSpec
from evidence_ladder.orchestrator.arms import arm_config, arm_label
from evidence_ladder.orchestrator.base import ConfigError, sha


def seed_at(master_seed: int, task_id: str, i: int, stream: str = "shared") -> int:
    return int(sha(f"{master_seed}|{task_id}|{stream}|{i}")[:8], 16)


def arm_seeds(master_seed: int, task_id: str, arm: str, k1: int, n_total: Optional[int] = None) -> List[int]:
    """Arm A: the first shared seed. Arms B..E: the SAME first k1 seeds, then arm-specific extra seeds up to n_total."""
    if not isinstance(k1, int) or k1 < 1:
        raise ConfigError("k1 is PENDING (tuned operational parameter); pass a positive int")
    shared = [seed_at(master_seed, task_id, i) for i in range(k1)]
    if arm == "A":
        return shared[:1]
    extra = [seed_at(master_seed, task_id, j, f"extra:{arm}") for j in range(max(0, (n_total or k1) - k1))]
    return shared + extra


def starting_state_hash(task: TaskSpec, master_seed: int, k1: int) -> str:
    """Identical for every arm of one task: the inputs from which a fresh context is built."""
    return sha({"task": task.task_id, "prompt": task.prompt, "public": list(task.public_tests),
                "shared_seeds": arm_seeds(master_seed, task.task_id, "B", k1)})


def fresh_context(task: TaskSpec, arm: str, *, generation, execution, budget, critic=None, scorer=None,
                  master_seed: int, k1: int, n_total: Optional[int] = None) -> ControllerContext:
    """A new context object per task and arm, built only from the same inputs. It holds no reference to anything evaluated after selection."""
    cfg = dict(arm_config(arm, k=(n_total or k1) if arm == "B" else None))
    cfg.update(k1=k1, n_total=n_total or k1, arm_label=arm_label(arm), task_id=task.task_id)
    return ControllerContext(generation=generation, execution=execution, budget=budget, arm=arm, critic=critic, scorer=scorer,
                             seeds=tuple(arm_seeds(master_seed, task.task_id, arm, k1, n_total)), config=cfg)


class PairedRecords:
    def __init__(self):
        self._d: Dict[str, Dict[str, Any]] = {}

    def add(self, task_id: str, arm: str, rec: Any) -> None:
        if arm in self._d.setdefault(task_id, {}):
            raise ValueError(f"duplicate record for {task_id}/{arm}")
        self._d[task_id][arm] = rec

    def get(self, task_id: str, arm: str):
        return self._d.get(task_id, {}).get(arm)

    def task_ids(self) -> List[str]:
        return sorted(self._d)

    def paired(self, arm_x: str, arm_y: str) -> List[str]:
        """task ids that have both arms."""
        return [t for t in self.task_ids() if arm_x in self._d[t] and arm_y in self._d[t]]
