"""[EL-ARCH-1 C2] Arm E, wired into the whole-program stage-1 controller: EXPLORATORY TRANSFER ONLY, OUT-OF-DISTRIBUTION.
The controller's complete_program path calls  scorer.score("complete_program", [candidate source strings], exploratory)  and the live LAYA backends refuse bare strings (there is no text-only
fallback). ProgramPoolScorer sits between them, PER TASK: it renders the candidate programs as a proper next-action decision (laya_forward.decision_request: training system block +
task statement + public examples + required format; each DISTINCT program as a Write tool-call option; no NONE option; fewer than 2 distinct programs -> refused), hands {context_text, options}
to the inner scorer (LayaScorer with LiveBackend or VerifiedSavedScoreBackend), and maps the scores back to the controller's candidates. Duplicate sources get the same score.
Comparison scorers (probe, random head) score the SAME rendered pool; their numbers are recorded for the report and NEVER returned to the controller (not selectors).
The arm label stays EXPLORATORY. Nothing here approves anything, and no keep/drop for LAYA comes out of stage 1 (arms.laya_keep_drop refuses)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from evidence_ladder.common.interfaces import UnsupportedDecisionType
from evidence_ladder.models import laya_forward as LF

OOD_LABEL = LF.EXPLORATORY_LABEL


class ScorerRefusedPool(Exception):
    """The pool cannot be rendered as a decision (fewer than 2 distinct programs, or a backend that would need bare text). The controller records scorer_unavailable."""


class ProgramPoolScorer:
    def __init__(self, inner, task, *, system_block: str, qwen=None, comparison: Optional[Mapping[str, Any]] = None, sink: Optional[List[dict]] = None):
        if qwen is None:
            from evidence_ladder.models import qwen as qwen   # noqa: PLW0127 only format_examples / SOLUTION_FORMATS are used
        self.inner, self.task, self.system_block, self.qwen = inner, task, system_block, qwen
        self.comparison = dict(comparison or {})
        self.sink = sink if sink is not None else []
        self.last_render: Optional[Dict[str, Any]] = None
        if isinstance(getattr(inner, "_backend", None), LF.SavedScoreBackend):
            raise ScorerRefusedPool("candidate-keyed SavedScoreBackend scores are pool-dependent and carry no pool/context proof; use VerifiedSavedScoreBackend or LiveBackend")

    @property
    def scorer_id(self) -> str:
        return self.inner.scorer_id

    @property
    def label(self) -> str:
        return OOD_LABEL

    def render(self, codes: Sequence[str]) -> Dict[str, Any]:
        t = self.task
        task = {"language": t.language, "prompt": t.prompt, "public_tests": list(t.public_tests)}
        req = LF.decision_request(self.system_block, task, list(codes), self.qwen)
        if len(req["options"]) < 2:
            raise ScorerRefusedPool("%d distinct program(s): a pool needs at least 2 options (no NONE option is added)" % len(req["options"]))
        return req

    def score(self, decision_type: str, pool: Sequence[Any], exploratory: bool = False) -> List[float]:
        if exploratory is not True:
            raise UnsupportedDecisionType("whole-program scoring is out-of-distribution for the next-action head and runs only as a flagged exploratory transfer test")
        codes = [getattr(p, "source", p) for p in pool]      # the controller passes source strings; a candidate object contributes its source
        if not all(isinstance(c, str) for c in codes):
            raise ScorerRefusedPool("candidates must be program source text")
        req = self.render(codes)
        backend_pool = [{"context_text": req["context_text"], "options": req["options"]}]
        unique = self.inner.score(decision_type, backend_pool, exploratory)
        if len(unique) != len(req["options"]):
            raise ScorerRefusedPool("scorer returned %d scores for %d options" % (len(unique), len(req["options"])))
        self.last_render = {"label": OOD_LABEL, "decision_type": decision_type, "context_sha256": LF.sha256_text(req["context_text"]), "option_sha256": [LF.sha256_text(o) for o in req["options"]],
                            "n_codes": len(codes), "n_distinct": len(req["options"])}
        event = {"task_id": self.task.task_id, "label": OOD_LABEL, "code_sha256": [LF.sha256_text(c) for c in codes], "distinct_code_sha256": self._distinct_sha(codes),
                 "scorers": {self.inner.scorer_id: {"role": LF.ROLES.get(getattr(self.inner, "KIND", "laya"), "scorer_under_test"), "status": "ok", "scores": [float(x) for x in unique]}}}
        for kind, sc in sorted(self.comparison.items()):
            try:
                event["scorers"][sc.scorer_id] = {"role": LF.ROLES[kind], "status": "ok", "scores": [float(x) for x in sc.score(decision_type, backend_pool, exploratory)]}
            except Exception as ex:      # noqa: BLE001 a comparison scorer that cannot score is recorded as unavailable, never turned into a number
                event["scorers"][sc.scorer_id] = {"role": LF.ROLES[kind], "status": "unavailable", "reason": "%s: %s" % (type(ex).__name__, str(ex)[:200])}
        self.sink.append(event)
        return [float(unique[i]) for i in req["index_of_code"]]

    @staticmethod
    def _distinct_sha(codes: Sequence[str]) -> List[str]:
        seen: List[str] = []
        for c in codes:
            h = LF.sha256_text(c)
            if h not in seen:
                seen.append(h)
        return seen
