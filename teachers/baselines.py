"""Baseline adapters: models that run under the same benchmark interface as the head but are NEVER training targets (plan A6).

laya-typed-decisions is a baseline, not a teacher: on typed-decisions choice questions its accuracy fell 0.740 -> 0.663 -> 0.612 -> 0.532 as
0, 3, 6 and 10 distractor options were added (measured here). Records carry role="BASELINE", so state.schema.usable_as_target is False.
The same candidate texts and NONE option the head sees are passed as criteria; the instruction text is fixed per decision kind below
and recorded in meta. The state is cut explicitly to STATE_TOKENS and the cut is recorded. Run with laya-audit/.venv from the project root.
"""
import hashlib
import json
import time
from dataclasses import replace

from state.schema import BASELINES
from teachers.laya_teachers import _LayaTeacher

INSTRUCTIONS = {"ACTION": "Which option is the best next action for this coding task?",
                "RECOVERY": "Which option is the best next step after this failure?",
                "CONTINUE": "Should the agent keep working on this task?",
                "COMPLETION": "Is the task fully done?"}


class LayaTypedBaseline(_LayaTeacher):
    name, folder = "laya-typed", "laya-typed-decisions"
    STATE_TOKENS = 640   # the checkpoint's state room is about 768 tokens minus the question head; cut here, not silently inside laya

    def decide(self, decision_id, decision_type, state_text="", candidates=()):
        """candidates: [Candidate] including NONE. Returns a DISTRIBUTION record over all of them, role BASELINE."""
        if decision_type not in BASELINES[self.name]:
            return replace(self._na(decision_id, decision_type), role="BASELINE")
        t0 = time.perf_counter()
        ids = [c.candidate_id for c in candidates]
        tok = self.agent.tok
        tid = tok(state_text, add_special_tokens=False)["input_ids"]
        cut = len(tid) > self.STATE_TOKENS
        state = tok.decode(tid[: self.STATE_TOKENS]) if cut else state_text
        q = {"q": {"type": "choice", "instructions": INSTRUCTIONS[decision_type], "criteria": {c.candidate_id: c.text for c in candidates}}}
        a = self.agent.predict(state, q)["answers"]["q"]
        out = self._answer(decision_id, decision_type, ids, [a["probabilities"][i] for i in ids], t0, a)
        return replace(out, role="BASELINE", meta={"n_options": len(ids), "state_tokens": self.STATE_TOKENS, "state_cut": cut,
                                                   "instruction": INSTRUCTIONS[decision_type]})
