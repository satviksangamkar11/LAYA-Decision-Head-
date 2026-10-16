"""Shared interface contract for the Evidence-Ladder runtime (Build Step 2). COORDINATOR-OWNED: agents import this file and never edit it.
If an agent needs a change, it reports the gap to the coordinator. Pure Python 3.12 standard library; no third-party imports in runtime code (jsonschema is allowed in tests only).
Records exchanged between components are plain dicts that validate against specs/evidence_ladder_v1/schemas_v1.json (requirement, evidence_record, check_def, scorer_def,
critic_request, critic_response, decision_record, stop_record, budget_record). The decision core is specs/evidence_ladder_v1/contract/contract_lib_v1.py (loaded via
evidence_ladder.common.contract_import.C); extend or wrap it, never copy it."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, TypedDict

Language = Literal["python", "java"]
Track = Literal["python", "java", "fable"]
Arm = Literal["A", "B", "C", "D0", "D1", "E"]
CheckStatus = Literal["pass", "fail", "unknown", "timeout", "error", "missing_evidence", "invalid_spec"]
DecisionType = Literal["next_action", "complete_program", "tool_call", "commit"]


# ---------------------------------------------------------------- errors
class UnsupportedDecisionType(Exception):
    """A scorer was asked about a decision type it is not validated for (and the run is not exploratory)."""

class NotAuthorized(Exception):
    """Inference or a network call was requested without the explicit authorization flag. Raised, never ignored."""

class GraderIsolationError(Exception):
    """Protected grading was requested from a code path that must not see hidden results (controller, repair loop, selection)."""

class BudgetExhausted(Exception):
    pass


# ---------------------------------------------------------------- artifacts and tasks
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Artifact:
    """Immutable candidate. artifact_id is the sha256 of the source text. Repairs create a NEW artifact with parent_id set; nothing is edited in place."""
    artifact_id: str
    task_id: str
    language: Language
    source: str
    parent_id: Optional[str] = None
    origin: str = "sample"          # "sample:<seed>" | "repair:<parent_id>" | "reference" | "known_wrong:<name>"

    @staticmethod
    def make(task_id: str, language: Language, source: str, parent_id: Optional[str] = None, origin: str = "sample") -> "Artifact":
        return Artifact(sha256_text(source), task_id, language, source, parent_id, origin)


@dataclass(frozen=True)
class TaskSpec:
    """What a worker may see. Never contains protected tests, oracles or reference solutions."""
    task_id: str                     # e.g. "P1-01" or "J1-01"
    index_item_id: str               # e.g. "py1:P1-01" (immutable id from the combined index)
    track: Track
    language: Language
    prompt: str
    public_tests: Sequence[Dict[str, Any]]      # [{"input":..., "expected":...}]
    difficulty: str = ""
    mutation_status: Dict[str, Any] = field(default_factory=dict)   # copied from the index; used only for flagging, never for selection


class CheckOutcome(TypedDict, total=False):
    check_id: str
    status: CheckStatus            # only "pass" is a pass (invariant I4)
    detail: str
    counterexample: Dict[str, Any]  # {"input":..., "expected":..., "observed":...}
    observed: Any                   # output of a model-proposed experiment
    duration_s: float
    stderr_tail: str
    failure_class: str              # "", "compile_error", "runtime_error", "timeout", "sandbox_error", "wrong_answer", "missing_output", "invalid_output"
    example_index: int              # run_public: position of the public example this outcome refers to (check_id "public_examples_v1"); absent for parse_v1/limits_v1


class HiddenGrade(TypedDict):
    artifact_id: str
    task_id: str
    passed: bool
    passed_tests: int
    total_tests: int
    failure: Optional[str]
    sha256_before_run: str


# ---------------------------------------------------------------- generation
@dataclass(frozen=True)
class GenerationRequest:
    task_id: str
    prompt: str
    n: int
    seeds: Sequence[int]
    max_new_tokens: int
    purpose: Literal["sample", "repair", "tests", "requirements"] = "sample"
    thinking: bool = True
    prompt_template_id: str = ""


@dataclass(frozen=True)
class Generation:
    text: str                       # full text (thinking + answer); never sent to the critic
    code: str                       # extracted final code block; "" if extraction failed
    extract_status: Literal["ok", "no_code"]
    tokens_prompt: int
    tokens_generated: int
    seed: int
    model: str
    model_revision: str
    prompt_template_id: str
    prompt_sha256: str
    wall_seconds: float = 0.0
    logprob_summary: Optional[Dict[str, float]] = None   # e.g. DeepConf-style group confidence; optional


# ---------------------------------------------------------------- protocols
class ExecutionBackend(Protocol):
    """Runs candidate code inside the sandbox. Has NO access to protected tests: run_public uses only TaskSpec.public_tests."""
    def run_public(self, artifact: Artifact, task: TaskSpec) -> List[CheckOutcome]: ...
    # run_public returns, in order: a "parse_v1" outcome (compile/parse), then ONE "public_examples_v1" outcome PER public example (same order as
    # task.public_tests, each with example_index, status, observed, and a counterexample when it fails). No aggregate outcome.
    def run_experiment(self, artifact: Artifact, experiment: Dict[str, Any]) -> CheckOutcome: ...   # experiment = critic_response["experiment"]
    def backend_info(self) -> Dict[str, Any]: ...   # isolation receipt: user, namespaces, limits, toolchain versions


class Grader(Protocol):
    """Protected grading. Only the orchestrator's final grading phase may hold one. Controller, repair and selection code must never receive a Grader
    (enforced by construction: ControllerContext has no grader field) and a Grader refuses calls made while a run is still selecting (GraderIsolationError)."""
    def grade(self, artifact: Artifact) -> HiddenGrade: ...


class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> List[Generation]: ...
    def model_info(self) -> Dict[str, Any]: ...      # repo, revision, dtype, template id; logged with every run


class CriticBroker(Protocol):
    """Staged critic (decision D7 option a): packets are exported for the PC to send; responses are ingested later. Idempotent by packet_id."""
    def build_packet(self, critic_request: Dict[str, Any]) -> Dict[str, Any]: ...      # validates blinding + schema; returns {"packet_id","request","created"}
    def completed(self, packet_id: str) -> Optional[Dict[str, Any]]: ...   # validated critic_response of an already-ingested "ok" packet, else None (second pass reads it here)
    def export_pending(self) -> List[Dict[str, Any]]: ...
    def ingest(self, packet_id: str, http_status: Optional[int], raw_response: Any) -> Dict[str, Any]: ...   # {"state": "ok"|"malformed"|"retry"|"unavailable"|"duplicate", "response": critic_response|None}


class Scorer(Protocol):
    scorer_id: str
    def score(self, decision_type: DecisionType, pool: Sequence[Any], exploratory: bool = False) -> List[float]: ...   # raises UnsupportedDecisionType


class Budget(Protocol):
    def charge(self, kind: Literal["qwen_gen", "qwen_prefill", "glm_in", "glm_out"], tokens: int) -> None: ...   # raises BudgetExhausted when the generated-token cap is crossed
    def exhausted(self) -> bool: ...
    def record(self) -> Dict[str, Any]: ...          # budget_record per schemas_v1


# ---------------------------------------------------------------- controller
@dataclass
class ControllerContext:
    """Everything the controller may use. Deliberately has NO grader and NO protected-test access."""
    generation: GenerationBackend
    execution: ExecutionBackend
    budget: Budget
    arm: Arm
    critic: Optional[CriticBroker] = None
    scorer: Optional[Scorer] = None
    seeds: Sequence[int] = ()
    config: Dict[str, Any] = field(default_factory=dict)   # frozen + tuned operational parameters (k1, caps, ...)


@dataclass
class ProblemResult:
    task_id: str
    arm: Arm
    final_artifact_id: Optional[str]
    answer_label: str                      # VERIFIED | SUPPORTED | PLAUSIBLE | UNVERIFIED | BLOCKED
    decision: Dict[str, Any]               # decision_record
    stop_cause: str                        # stop_record.cause
    evidence_ids: List[str]
    budget_record: Dict[str, Any]
    rounds: int
    journal_hash: str                      # sha256 of the canonical evidence journal; reproducibility receipt
    final_artifact: Optional[Artifact] = None   # the selected candidate (so the orchestrator can grade it); None only when nothing was produced
    stop_details: str = ""                 # e.g. "critic_pending" when stop_cause == "api_unavailable"


class Controller(Protocol):
    def run_problem(self, task: TaskSpec, ctx: ControllerContext) -> ProblemResult: ...
