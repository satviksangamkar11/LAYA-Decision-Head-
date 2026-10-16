"""SandboxExecutionBackend: the ExecutionBackend of the Evidence-Ladder runtime for python and java, built on the existing receipted bubblewrap sandboxes.
Sees ONLY TaskSpec.public_tests and model-proposed experiments; it has no code path to hidden tests (a static test enforces that no module but the hidden grader
imports or names the hidden-case files). Status policy (invariant I4): only an exact match on every public case is "pass"; sandbox/tool/integrity failures are
status "error" with failure_class "sandbox_error" and are never candidate failures."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from evidence_ladder.common.interfaces import Artifact, CheckOutcome, TaskSpec

from . import config as K
from . import engine
from .artifacts import ArtifactIntegrityError, ArtifactStore

PUBLIC_CHECK = "public_examples_v1"
PARSE_CHECK = "parse_v1"
EXPERIMENT_CHECK = "critic_experiment_v1"
SUPPORTED_EXPERIMENTS = ("stdin_input", "function_call")
MAX_PAYLOAD_CHARS = 20000


def _js(x: Any, cap: int) -> str:
    s = json.dumps(x, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= cap else s[:cap] + "...[truncated]"


class SandboxExecutionBackend:
    def __init__(self, store: Optional[ArtifactStore] = None, config: Optional[K.SandboxConfig] = None):
        self.store = store or ArtifactStore()
        self.cfg = config or K.SandboxConfig()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _err(check_id: str, msg: str) -> CheckOutcome:
        return {"check_id": check_id, "status": "error", "detail": msg, "failure_class": "sandbox_error"}

    def _bytes(self, artifact: Artifact) -> bytes:
        return self.store.verified_bytes(artifact)

    # ------------------------------------------------------------- public checks
    def run_public(self, artifact: Artifact, task: TaskSpec) -> List[CheckOutcome]:
        """Returns [parse_v1, then ONE public_examples_v1 outcome per public example, in task.public_tests order, each with example_index]."""
        if artifact.language != task.language or artifact.task_id != task.task_id:
            return [{"check_id": PUBLIC_CHECK, "status": "invalid_spec", "detail": "artifact/task language or task_id mismatch", "failure_class": ""}]
        cases = list(task.public_tests)
        try:
            raw = self._bytes(artifact)
        except ArtifactIntegrityError as e:
            msg = "integrity: " + str(e)
            return [self._err(PARSE_CHECK, msg)] + [{**self._err(PUBLIC_CHECK, msg), "example_index": i} for i in range(len(cases))]
        if not cases:
            return [{"check_id": PUBLIC_CHECK, "status": "missing_evidence", "detail": "task has no public tests", "failure_class": ""}]
        res = engine.run(artifact.language, raw, [c["input"] for c in cases], self.cfg)
        v, each = engine.judge_each(artifact.language, res, cases, self.cfg)
        cap = self.cfg.max_observed_chars
        if v.failure_class == "compile_error":
            parse = {"check_id": PARSE_CHECK, "status": "fail", "detail": v.detail, "failure_class": "compile_error", "stderr_tail": v.stderr_tail}
        elif v.failure_class == "sandbox_error":
            parse = self._err(PARSE_CHECK, v.detail)
        elif v.detail == "compile timeout":
            parse = {"check_id": PARSE_CHECK, "status": "timeout", "detail": v.detail, "failure_class": "timeout"}
        elif res.phase_status == "oversized":
            parse = {"check_id": PARSE_CHECK, "status": "fail", "detail": v.detail, "failure_class": "invalid_output"}
        else:
            parse = {"check_id": PARSE_CHECK, "status": "pass", "detail": "source compiled/loaded in the jail", "failure_class": ""}
        parse["duration_s"] = v.duration
        outs: List[CheckOutcome] = [parse]
        for i, (st, cls, obs) in enumerate(each):
            o: CheckOutcome = {"check_id": PUBLIC_CHECK, "example_index": i, "duration_s": v.duration, "stderr_tail": v.stderr_tail}
            if parse["status"] != "pass":
                # nothing ran per example: compile/parse failure is not an example verdict
                o.update(status="error" if cls == "sandbox_error" else ("timeout" if parse["status"] == "timeout" else "missing_evidence"),
                         failure_class=cls if cls == "sandbox_error" else parse["failure_class"], detail="not run: " + parse["detail"], observed="")
            else:
                o.update(status=st, failure_class=cls, observed=obs, detail=f"example {i}: {'matched' if st == 'pass' else (cls or v.detail)}")
                if st != "pass" and obs:
                    c = cases[i]
                    o["counterexample"] = {"input": _js(c["input"], cap), "expected": _js(c["expected"], cap), "observed": obs}
            outs.append(o)
        return outs

    # ------------------------------------------------------------- model-proposed experiments
    def run_experiment(self, artifact: Artifact, experiment: Dict[str, Any]) -> CheckOutcome:
        def bad(msg: str) -> CheckOutcome:
            return {"check_id": EXPERIMENT_CHECK, "status": "invalid_spec", "detail": msg, "failure_class": ""}
        if not isinstance(experiment, dict):
            return bad("experiment is not an object")
        kind, payload = experiment.get("kind"), experiment.get("payload")
        if kind not in SUPPORTED_EXPERIMENTS:
            return bad(f"experiment kind {kind!r} is not supported in stage 1 (allowed: {', '.join(SUPPORTED_EXPERIMENTS)})")
        if not isinstance(payload, str) or len(payload) > MAX_PAYLOAD_CHARS:
            return bad(f"payload must be a string of at most {MAX_PAYLOAD_CHARS} characters")
        try:
            obj = json.loads(payload)
        except ValueError as e:
            return bad("payload is not valid JSON: " + str(e)[:100])
        if not isinstance(obj, dict):
            return bad("payload must be a JSON object (solve takes an object)")
        try:
            raw = self._bytes(artifact)
        except ArtifactIntegrityError as e:
            return self._err(EXPERIMENT_CHECK, "integrity: " + str(e))
        res = engine.run(artifact.language, raw, [obj], self.cfg)
        status, cls, observed, detail = engine.observe(artifact.language, res, self.cfg)
        out: CheckOutcome = {"check_id": EXPERIMENT_CHECK, "status": status, "detail": detail, "failure_class": cls, "duration_s": res.duration,
                             "stderr_tail": res.stderr_tail}
        if status in ("pass", "fail"):
            cap = self.cfg.max_observed_chars
            s = json.dumps(observed, ensure_ascii=False)
            out["observed"] = observed if len(s) <= cap else s[:cap] + "...[truncated]"
        return out

    # ------------------------------------------------------------- receipt
    def backend_info(self) -> Dict[str, Any]:
        from .receipts import isolation_receipt
        return isolation_receipt(self.cfg)
