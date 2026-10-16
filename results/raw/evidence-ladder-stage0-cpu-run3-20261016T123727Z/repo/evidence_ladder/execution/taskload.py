"""Loads TaskSpec objects from the PUBLIC worker task files only (prompt + public tests). Never reads hidden data."""
from __future__ import annotations

import json
from typing import List

from evidence_ladder.common.interfaces import TaskSpec

from . import config as K


def load_tasks(language: str) -> List[TaskSpec]:
    path = K.PY_TASKS if language == "python" else K.JAVA_TASKS
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        tid = d.get("task_id") or d["id"]
        out.append(TaskSpec(task_id=tid, index_item_id=("py1:" if language == "python" else "java1:") + tid, track=language, language=language,
                            prompt=d["prompt"], public_tests=d["public_tests"], difficulty=d.get("difficulty", "")))
    return out
