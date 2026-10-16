"""Hidden-case grader. THE ONLY module allowed to import or name the protected-case files; the controller / repair / selection path must never import it.

HONEST STATUS OF THE DATA: these "protected" development cases were committed to Git history, so they are PUBLIC and this is NOT a sealed benchmark.
They serve as a development oracle only; sealed (encrypted, grading-kernel-only) hidden tests are a separate, later step (S4 section 3.2).

Isolation rules enforced here: grade() raises GraderIsolationError while the SelectionGate is open (selection still running); results leave only as HiddenGrade
(counts and a boolean, never per-case inputs or expected values); candidates see one input per line on stdin and nothing else; sandbox failures are reported
as failure "SANDBOX_ERROR" (not a candidate verdict); an artifact integrity mismatch raises ArtifactIntegrityError."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidence_ladder.common.interfaces import Artifact, GraderIsolationError, HiddenGrade

from . import config as K
from . import engine
from .artifacts import ArtifactStore

PY_PROTECTED = K.PY_ROOT / "grader" / "protected_cases.json"
JAVA_PROTECTED = K.JAVA_ROOT / "fixed_v1" / "grader" / "protected_cases.json"   # the "fixed_v1" set (differs from the kit's original; sha256 36b40e19...)

_FAIL_NAMES = {"compile_error": "COMPILE_ERROR", "runtime_error": "RUNTIME_ERROR", "timeout": "TIMEOUT", "sandbox_error": "SANDBOX_ERROR",
               "wrong_answer": "WRONG_ANSWER", "missing_output": "MISSING_OUTPUT", "invalid_output": "INVALID_OUTPUT"}


class SelectionGate:
    """Open while candidate selection runs; the orchestrator closes it once, irreversibly, before final grading."""
    def __init__(self) -> None:
        self._open = True

    @property
    def is_open(self) -> bool:
        return self._open

    def close(self) -> None:
        self._open = False


class ProtectedGrader:
    def __init__(self, gate: SelectionGate, store: Optional[ArtifactStore] = None, config: Optional[K.SandboxConfig] = None,
                 python_cases: Path = PY_PROTECTED, java_cases: Path = JAVA_PROTECTED):
        self._gate, self._store, self._cfg = gate, store or ArtifactStore(), config or K.SandboxConfig()
        self._paths = {"python": Path(python_cases), "java": Path(java_cases)}
        self._cache: Dict[str, Dict[str, List[dict]]] = {}

    def _cases(self, language: str, task_id: str) -> Optional[List[dict]]:
        if language not in self._cache:
            data = json.loads(self._paths[language].read_text(encoding="utf-8"))
            self._cache[language] = {(t.get("task_id") or t["id"]): t["cases"] for t in data}
        return self._cache[language].get(task_id)

    def grade(self, artifact: Artifact) -> HiddenGrade:
        if self._gate.is_open:
            raise GraderIsolationError("protected grading requested while selection is still running (gate open)")
        raw = self._store.verified_bytes(artifact)          # raises ArtifactIntegrityError on mismatch; hash taken before execution
        base = {"artifact_id": artifact.artifact_id, "task_id": artifact.task_id, "sha256_before_run": artifact.artifact_id}
        cases = self._cases(artifact.language, artifact.task_id)
        if cases is None:
            return HiddenGrade(passed=False, passed_tests=0, total_tests=0, failure="NO_PROTECTED_CASES", **base)
        v = None
        for _ in range(2):                                   # one retry if the jail itself failed to start
            res = engine.run(artifact.language, raw, [c["input"] for c in cases], self._cfg)
            v = engine.judge(artifact.language, res, cases, self._cfg)
            if v.failure_class != "sandbox_error":
                break
        name = None if v.status == "pass" else ("OVERSIZED_SUBMISSION" if v.detail.startswith("source ") else _FAIL_NAMES.get(v.failure_class, "FAIL"))
        return HiddenGrade(passed=v.status == "pass", passed_tests=v.passed, total_tests=v.total, failure=name, **base)
