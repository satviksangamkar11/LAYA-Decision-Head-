"""Lane-scoped teacher wrappers for the stop-completion judge and the conductor (plan track A3).

Each wrapper loads its checkpoint offline, pinned by the sha256 in results/raw/<model>.manifest.json, answers only the decision
kinds in state.schema.LANES, and returns a TeacherOutput. Outside its lane it returns applicable=False. Probabilities are over the
semantic options the teacher actually scored; NONE is not among them (the teacher has no opinion on it), so candidate_ids is a
subset of the decision's options. Temperatures are the shipped ones (calibration_revision "shipped"); the refit on our data is a
later step. Needs the `laya` package, which lives in laya-audit/.venv: run with that interpreter from the project root.
"""
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")             # a stray TensorFlow import can deadlock model construction
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # a load must never download
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from state.schema import LANES, TeacherOutput

ROOT = Path(__file__).resolve().parents[1]

CONTINUE_Q = {"should_continue": {"type": "noul", "instructions": (   # the question the conductor was trained with (its notebook)
    "Did the assistant stop in the middle of the task with work clearly left undone? "
    "TRUE only if the response says it will do more next, lists remaining steps it has not done, "
    "or ends mid-implementation. "
    "FALSE if it reports the task as finished, asks a question, or reaches a natural stopping point.")}}
EFFORT_Q = {"role": {"type": "choice", "instructions": "How much reasoning effort does this coding request need?",
    "criteria": {"smol": "trivial mechanical edit: rename, typo, format, commit message, one-liner, add import",
                 "default": "normal coding: implement a well-defined feature, fix a bug with known cause, write a test",
                 "slow": "hard: debug an unknown cause, architecture or design, large refactor, security review"}}}


class _LayaTeacher:
    name = ""      # key in state.schema.LANES
    folder = ""    # models/<folder>

    def __init__(self, device="cuda"):
        import laya
        man = json.load(open(ROOT / "results" / "raw" / f"{self.folder}.manifest.json", encoding="utf-8"))
        sha = next(f["sha256"] for f in man["files"] if f["name"] == "model.safetensors")
        self.revision = man["revision"]
        self.agent = laya.load(str(ROOT / "models" / self.folder), device=device, expected_sha256={"model.safetensors": sha})

    def _na(self, decision_id, decision_type):
        return TeacherOutput(decision_id, decision_type, self.name, self.revision, applicable=False)

    def _answer(self, decision_id, decision_type, ids, probs, t0, raw):
        top = max(range(len(ids)), key=probs.__getitem__)
        s = sum(probs)
        probs = [p / s for p in probs]  # the shipped rounding leaves sums off by ~1e-4
        return TeacherOutput(decision_id, decision_type, self.name, self.revision, candidate_ids=list(ids), probabilities=probs,
                             selected_candidate=ids[top], confidence=probs[top], calibration_revision="shipped",
                             latency_ms=1000 * (time.perf_counter() - t0),
                             raw_ref=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16])


class JudgeTeacher(_LayaTeacher):
    """COMPLETION: did the assistant's final message actually deliver the request? Binary: complete / incomplete."""
    name, folder = "laya-stop-completion-judge", "laya-stop-completion-judge"

    def __init__(self, device="cuda"):
        super().__init__(device)
        spec = importlib.util.spec_from_file_location("judge_state_pack", ROOT / "models" / self.folder / "state_pack.py")
        self._pack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._pack)  # the shipped helper: pure string slicing, read before use; keeps the trained packing

    def decide(self, decision_id, decision_type, user_request="", final_message=""):
        if decision_type not in LANES[self.name]:
            return self._na(decision_id, decision_type)
        t0 = time.perf_counter()
        a = self.agent.predict(self._pack.pack_state(user_request, final_message),
                               {"completion": self._pack.COMPLETION_QUESTION})["answers"]["completion"]
        p = a["probabilities"]
        return self._answer(decision_id, decision_type, ["complete", "incomplete"], [p["complete"], p["incomplete"]], t0, a)


class ConductorTeacher(_LayaTeacher):
    """CONTINUE: did the agent stop with work left (continue / stop). ESCALATE: reasoning effort tier (smol / default / slow)."""
    name, folder = "laya-conductor", "laya-conductor"

    def decide(self, decision_id, decision_type, last_response="", task="", request="", context_tokens=None):
        if decision_type not in LANES[self.name]:
            return self._na(decision_id, decision_type)
        t0 = time.perf_counter()
        if decision_type == "CONTINUE":
            a = self.agent.predict({"last_response": last_response, "task": task}, CONTINUE_Q)["answers"]["should_continue"]
            p_cont = a["noul"]   # P(stopped mid-task) = P(the agent should continue)
            return self._answer(decision_id, decision_type, ["continue", "stop"], [p_cont, 1 - p_cont], t0, a)
        state = {"request": request} if context_tokens is None else {"request": request, "context_tokens": context_tokens}
        a = self.agent.predict(state, EFFORT_Q)["answers"]["role"]
        ids = ["smol", "default", "slow"]
        return self._answer(decision_id, decision_type, ids, [a["probabilities"][i] for i in ids], t0, a)


class CodeRelevanceTeacher(_LayaTeacher):
    """CODE_RELEVANCE: an independent P(relevant to this change) for each code chunk (a noul question per chunk).

    The scores are NOT a distribution: no softmax, no normalisation across chunks, and each chunk's score does not depend on the
    others (checked in selfcheck). The model card says P rarely exceeds 0.5, so use the scores for ranking, not as a 0.5 threshold.
    Inputs are cut explicitly, never silently: the state to the card's production setting (128 tokens) and the task to 128 tokens;
    what was cut is recorded in meta. Question and state formats are the ones on the model card.
    """
    name, folder = "laya-code", "laya-code"
    STATE_TOKENS = 128
    TASK_TOKENS = 128

    def _cut(self, text, n):
        ids = self.agent.tok(text, add_special_tokens=False)["input_ids"]
        return (text, False) if len(ids) <= n else (self.agent.tok.decode(ids[:n]), True)

    def decide(self, decision_id, decision_type, task="", chunks=()):
        """chunks: [{"id", "path", "start", "end", "code"}]. Returns one INDEPENDENT_SCORES record."""
        if decision_type not in LANES[self.name]:
            return self._na(decision_id, decision_type)
        t0 = time.perf_counter()
        task, task_cut = self._cut(task, self.TASK_TOKENS)
        states, cut_items = [], []
        for c in chunks:
            s, cut = self._cut(f"file: {c['path']} (lines {c['start']}-{c['end']})\n{c['code']}", self.STATE_TOKENS)
            states.append(s)
            if cut:
                cut_items.append(c["id"])
        question = {"relevant": {"type": "noul", "instructions": f'Is this source code relevant to the software change: "{task}"?'}}
        raw = [r["answers"]["relevant"] for r in self.agent.predict_batch(states, question)]
        return TeacherOutput(decision_id, decision_type, self.name, self.revision, form="INDEPENDENT_SCORES",
                             item_ids=[c["id"] for c in chunks], scores=[a["noul"] for a in raw], calibration_revision="shipped",
                             latency_ms=1000 * (time.perf_counter() - t0),
                             raw_ref=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16],
                             meta={"state_tokens": self.STATE_TOKENS, "truncated_items": cut_items, "task_truncated": task_cut})
