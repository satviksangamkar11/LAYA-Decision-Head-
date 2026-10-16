"""Replay doubles built from the REAL recordings in fixtures/. RecordedExecutionBackend implements ExecutionBackend, RecordedGrader implements Grader.
An unknown source/experiment raises UnrecordedError: nothing is ever invented. Pure stdlib (Kaggle-safe)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidence_ladder.common.interfaces import Artifact, GraderIsolationError, HiddenGrade, TaskSpec, sha256_text

FIXDIR = Path(__file__).resolve().parent / "fixtures"


class UnrecordedError(KeyError):
    pass


def tests_sha(public_tests) -> str:
    return hashlib.sha256(json.dumps([{"input": t["input"], "expected": t["expected"]} for t in public_tests], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_outcomes(path=None) -> Dict[str, Any]:
    return json.loads(Path(path or FIXDIR / "execution_outcomes.json").read_text(encoding="utf-8"))


def load_grades(path=None) -> Dict[str, Any]:
    return json.loads(Path(path or FIXDIR / "hidden_grades.json").read_text(encoding="utf-8"))


class RecordedExecutionBackend:
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or load_outcomes()
        self._info = data["backend_info"]
        self._pub: Dict[tuple, Any] = {}
        self._exp: Dict[tuple, Any] = {}
        for e in data["cases"].values():
            k = (e["task_id"], e["artifact_id"])
            self._pub[k] = (e["public_tests_sha256"], e["public"])
            for x in e["experiments"]:
                self._exp[k + (x["kind"], x["payload"])] = x["outcome"]
        self.public_calls: List[tuple] = []
        self.experiment_calls: List[tuple] = []

    def run_public(self, artifact: Artifact, task: TaskSpec) -> List[Dict[str, Any]]:
        if sha256_text(artifact.source) != artifact.artifact_id:
            raise UnrecordedError("artifact id does not match its source (integrity)")
        k = (artifact.task_id, artifact.artifact_id)
        if k not in self._pub:
            raise UnrecordedError("no recorded public outcome for %s %s" % k)
        want, outs = self._pub[k]
        if tests_sha(task.public_tests) != want:
            raise UnrecordedError("public tests of %s differ from the recorded ones" % task.task_id)
        self.public_calls.append(k)
        return copy.deepcopy(outs)

    def run_experiment(self, artifact: Artifact, experiment: Dict[str, Any]) -> Dict[str, Any]:
        k = (artifact.task_id, artifact.artifact_id, experiment.get("kind"), experiment.get("payload"))
        if k not in self._exp:
            raise UnrecordedError("no recorded experiment %r" % (k,))
        self.experiment_calls.append(k)
        return copy.deepcopy(self._exp[k])

    def backend_info(self) -> Dict[str, Any]:
        return copy.deepcopy(self._info)


class RecordedGrader:
    """grade() serves only the recorded HiddenGrade. gate: object with is_closed() (pilot gate) or an is_open property; while open -> GraderIsolationError (not logged as served).
    flip=True returns the opposite verdict (used to prove selection never depends on grading)."""
    def __init__(self, data: Optional[Dict[str, Any]] = None, gate=None, flip: bool = False):
        data = data or load_grades()
        self._g = {(g["task_id"], g["artifact_id"]): g for g in data["grades"].values()}
        self._gate, self.flip = gate, flip
        self.calls: List[tuple] = []
        self.refused = 0

    def _open(self) -> bool:
        g = self._gate
        if g is None:
            return False
        if hasattr(g, "is_closed"):
            return not g.is_closed()
        return bool(g.is_open)

    def grade(self, artifact: Artifact) -> HiddenGrade:
        if self._open():
            self.refused += 1
            raise GraderIsolationError("protected grading requested while selection is still running")
        k = (artifact.task_id, artifact.artifact_id)
        if k not in self._g:
            raise UnrecordedError("no recorded protected grade for %s %s" % k)
        self.calls.append(k)
        g = dict(self._g[k])
        if self.flip:
            g["passed"] = not g["passed"]
            g["passed_tests"] = g["total_tests"] - g["passed_tests"]
            g["failure"] = None if g["passed"] else "WRONG_ANSWER"
        return HiddenGrade(**g)
