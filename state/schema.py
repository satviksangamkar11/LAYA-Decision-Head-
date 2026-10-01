"""Decision-layer contracts (runbook 5.1 and 5.2): candidate, state, hash. Plain dataclasses, no dependencies.

Rules enforced here, not left to callers:
- every decision carries a NONE option and has 2..MAX_CANDIDATES options (CLAUDE.md decision-layer rules);
- an outcome may only be set on a candidate that was really executed (never fabricate a counterfactual outcome);
- head metadata never includes the candidate's `source`: it would leak the label.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field

ACTION_TYPES = ("tool_call", "edit", "finish", "ask", "none")
SAFETY_CLASSES = ("READ_ONLY", "WRITE_LOCAL", "EXEC", "NETWORK", "DESTRUCTIVE", "NONE")
SOURCES = ("MODEL", "TRAJECTORY", "TEACHER", "RETRIEVAL", "PATCH", "COUNTERFACTUAL")
MAX_CANDIDATES = 16
NONE_ID = "NONE"
META_DIM = len(ACTION_TYPES) + len(SAFETY_CLASSES) + 1  # one-hot action type, one-hot safety class, executable


@dataclass
class Candidate:
    candidate_id: str
    text: str
    action_type: str = "tool_call"
    tool: str = ""
    target: dict = field(default_factory=dict)
    arguments: dict = field(default_factory=dict)
    expected_effect: str = ""
    preconditions: list = field(default_factory=list)
    source: str = "MODEL"
    executable: bool = True
    safety_class: str = "READ_ONLY"
    teacher_scores: dict = field(default_factory=dict)  # teacher name -> {"distribution":..., "revision":..., "lane":...}
    executed: bool = False
    outcome_if_executed: dict | None = None  # only ever filled from a real execution

    def problems(self):
        p = []
        if not self.candidate_id or not self.text:
            p.append("candidate_id and text are required")
        if self.action_type not in ACTION_TYPES:
            p.append(f"unknown action_type {self.action_type!r}")
        if self.safety_class not in SAFETY_CLASSES:
            p.append(f"unknown safety_class {self.safety_class!r}")
        if self.source not in SOURCES:
            p.append(f"unknown source {self.source!r}")
        if self.outcome_if_executed is not None and not self.executed:
            p.append("outcome set on a candidate that was not executed")
        if self.executed and not self.executable:
            p.append("executed but marked not executable")
        return p

    def head_meta(self):
        """Runtime-visible fields only, as the head's `meta` row. Never `source`, never teacher scores."""
        return ([float(self.action_type == a) for a in ACTION_TYPES]
                + [float(self.safety_class == s) for s in SAFETY_CLASSES] + [float(self.executable)])


# Which decision kinds each teacher may answer (PLAN.md section 11 lane table). Outside its lane a teacher returns
# applicable=False: the conductor scored 0.340 on a task it was not built for, so a forced answer is noise, not a vote.
# CODE_RELEVANCE (is this code chunk relevant to the change) is its own kind: it is not EVIDENCE sufficiency, which no
# checkpoint has been trained to answer, so that kind has no teacher and stays unlabelled.
LANES = {"laya-typed": (), "laya-conductor": ("CONTINUE", "ESCALATE"),
         "laya-stop-completion-judge": ("COMPLETION",), "laya-code": ("CODE_RELEVANCE",), "cortex-1": ()}
# Empty lane = no trusted teacher role. laya-typed is a baseline only (accuracy falls 0.740 to 0.532 as distractor options are added;
# measured on typed-decisions, not on coding). cortex-1 is rejected: fixed templates, label-bearing metadata in its training states.
FORMS = ("DISTRIBUTION", "INDEPENDENT_SCORES")


@dataclass
class TeacherOutput:
    """One teacher's answer to one decision, in the same shape the head is trained on (probabilities over candidate_ids).

    abstain_probability and escalate_probability are None unless the teacher really supplies them: Laya's shipped act head
    reads 1.0 on everything (issue #185, measured here), so it is never copied into these fields.
    """
    decision_id: str
    decision_type: str
    teacher_name: str
    model_revision: str
    applicable: bool = True
    form: str = "DISTRIBUTION"                 # exactly one of FORMS; the fields of the other form must stay empty
    candidate_ids: list = field(default_factory=list)   # DISTRIBUTION: candidate_id -> probability, sums to 1
    probabilities: list | None = None
    selected_candidate: str | None = None
    confidence: float | None = None
    item_ids: list = field(default_factory=list)        # INDEPENDENT_SCORES: item_id -> score in [0, 1], no normalisation
    scores: list | None = None
    abstain_probability: float | None = None
    escalate_probability: float | None = None
    calibration_revision: str = "uncalibrated"
    latency_ms: float = 0.0
    raw_ref: str = ""
    meta: dict = field(default_factory=dict)            # e.g. which inputs were truncated, and to how many tokens

    def problems(self):
        p = []
        if self.teacher_name not in LANES:
            p.append(f"unknown teacher {self.teacher_name!r}")
        elif self.applicable and self.decision_type not in LANES[self.teacher_name]:
            p.append(f"{self.teacher_name} is not in its lane for {self.decision_type}")
        if self.form not in FORMS:
            return p + [f"unknown form {self.form!r}"]
        if not self.applicable:
            if self.probabilities is not None or self.selected_candidate is not None or self.scores is not None:
                p.append("not applicable but carries an answer")
            return p
        if self.form == "DISTRIBUTION":
            if self.scores is not None or self.item_ids:
                p.append("DISTRIBUTION form must not carry independent scores")
            if self.probabilities is None or len(self.probabilities) != len(self.candidate_ids):
                p.append("probabilities must match candidate_ids")
            elif abs(sum(self.probabilities) - 1.0) > 1e-6 or min(self.probabilities) < 0:
                p.append("probabilities must be a distribution")
            if self.selected_candidate not in self.candidate_ids:
                p.append("selected_candidate not among candidate_ids")
        else:
            if self.probabilities is not None or self.selected_candidate is not None or self.candidate_ids:
                p.append("INDEPENDENT_SCORES form must not carry a distribution")
            if not self.item_ids or self.scores is None or len(self.scores) != len(self.item_ids):
                p.append("scores must match a non-empty item_ids")
            elif len(set(self.item_ids)) != len(self.item_ids):
                p.append("duplicate item_ids")
            elif min(self.scores) < 0 or max(self.scores) > 1:
                p.append("scores must be in [0, 1]")
        return p


def none_candidate():
    return Candidate(NONE_ID, "None of these options, or the evidence is insufficient", action_type="none",
                     safety_class="NONE", executable=False)


def with_none(cands):
    """The option list a decision is made over: the given candidates plus NONE last. Raises on a bad set."""
    cands = [c for c in cands if c.candidate_id != NONE_ID]
    ids = [c.candidate_id for c in cands]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ids")
    for c in cands:
        if c.problems():
            raise ValueError(f"{c.candidate_id}: {'; '.join(c.problems())}")
    out = cands + [none_candidate()]
    if len(out) < 2 or len(out) > MAX_CANDIDATES:
        raise ValueError(f"a decision needs 1..{MAX_CANDIDATES - 1} candidates plus NONE, got {len(cands)}")
    return out


@dataclass
class State:
    task: str = ""
    goal: str = ""
    hypothesis: str = ""
    repo: str = ""
    commit: str = ""
    recent_actions: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    tests: list = field(default_factory=list)          # [{"name":..., "status": "pass"|"fail"|"error"}]
    compiler: dict = field(default_factory=dict)       # {"errors": [...]}
    runtime: dict = field(default_factory=dict)
    git_diff: dict = field(default_factory=dict)
    memory_evidence: list = field(default_factory=list)
    pending_actions: list = field(default_factory=list)
    acceptance: list = field(default_factory=list)     # [{"name":..., "met": bool}]
    candidates: list = field(default_factory=list)
    superseded_facts: list = field(default_factory=list)
    budget: dict = field(default_factory=lambda: {"context_tokens": 0, "tool_steps": 0})

    def state_hash(self):
        """sha256 of the canonical JSON of the whole state: key order does not matter, content does."""
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

