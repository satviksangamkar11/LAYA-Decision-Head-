"""Executable checks for invariants I1-I8 (ARCHITECTURE section 3). assert_invariants(result, journal, ctx) runs at the end of every controller run and raises
InvariantViolation listing every broken invariant. It reads only the journal (the receipt) and the result, so it can also be replayed offline.
static_no_grader_imports() is the static half of I8."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

from evidence_ladder.common.contract_import import C
from .components import MissingComponentError
from .decision import decide_stage1
from .evidence import NOT_PASS, map_status

MODEL_SOURCES = {"model_test", "critic_experiment", "candidate_agreement", "scorer"}
FORBIDDEN_WORDS = ("grader", "protected")
CONTROLLER_DIR = Path(__file__).resolve().parent


class InvariantViolation(Exception):
    def __init__(self, failures: Dict[str, str]):
        super().__init__("; ".join("%s: %s" % kv for kv in sorted(failures.items())))
        self.failures = failures


def static_no_grader_imports(root: Path = CONTROLLER_DIR) -> List[str]:
    """Every import statement (module path and imported names) under evidence_ladder/controller, tests included, must avoid 'grader' and 'protected'. -> offending lines."""
    bad = []
    for p in sorted(Path(root).rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                names = [n.module or ""] + [a.name for a in n.names]
            for nm in names:
                if any(w in nm.lower() for w in FORBIDDEN_WORDS):
                    bad.append("%s:%d imports %s" % (p.name, n.lineno, nm))
    return bad


def _first(journal, kind):
    e = journal.entries(kind)
    return e[0]["data"] if e else None


def check_i1(result, journal) -> str:
    """Label comes from evidence only: recompute it from the journal and compare. VERIFIED needs a trusted covering check on every gating row."""
    tiers = journal.entries("oracle_tier")      # [EL-ARCH-1 B1] the oracle tier's label ceiling is real: no answer above it (BLOCKED is never capped)
    if tiers:
        from .tiers import check_ceiling
        msg = check_ceiling(result.answer_label, tiers[-1]["data"]["tier"])
        if msg:
            return msg
    reqs = (_first(journal, "requirements") or {}).get("reqs")
    if result.final_artifact_id is None or reqs is None:
        return "" if result.answer_label != "VERIFIED" else "VERIFIED without a candidate"
    reg = journal._registry
    env = (_first(journal, "run_start") or {})["environment_revision"]
    d = decide_stage1(reqs, journal.evidence(result.final_artifact_id), result.final_artifact_id, result.final_artifact_id, env, reg, decision_type="complete_program")
    if d["answer_label"] != result.answer_label or result.decision["answer_label"] != result.answer_label:
        return "label %s does not match the evidence-derived label %s" % (result.answer_label, d["answer_label"])
    if result.answer_label == "VERIFIED":
        for r in reqs:
            if C.can_gate(r) and r["priority"] in ("hard", "functional"):
                if not any(e["requirement_id"] == r["requirement_id"] and e["status"] == "pass" and C.authoritative(e, reg) for e in journal.evidence(result.final_artifact_id)):
                    return "VERIFIED but %s has no trusted pass" % r["requirement_id"]
    # [EL-ARCH-1 A5] `derived_by` is a literal written by C.decide, so it proves nothing. What is checked instead: the decision handed back IS the decision the controller journaled, and
    # its rows, candidate and authorization equal a fresh recomputation from the journal evidence (the label alone is not enough).
    dec = [e["data"] for e in journal.entries("decision")]
    if not dec or dec[-1] != result.decision:
        return "result.decision is not the decision the controller journaled"
    for k in ("candidate_id", "requirement_rows"):
        if result.decision.get(k) != d.get(k):
            return "decision %s differs from the evidence-derived decision" % k
    if result.decision.get("authorized_to_execute") and not d.get("authorized_to_execute") and d.get("decision_type") == result.decision.get("decision_type"):
        return "decision authorizes execution that the evidence does not"
    return ""


def check_i2(journal) -> str:
    """Rankings are lexicographic: C.rank reproduces the logged order and no later key (scorer, confidence, cluster) reorders across trusted levels."""
    for e in journal.entries("ranking"):
        items, order = e["data"]["items"], e["data"]["order"]
        if [c["id"] for c in C.rank(items)] != order:
            return "logged order differs from C.rank at seq %d" % e["seq"]
        by = {c["id"]: c for c in items}
        for a, b in zip(order, order[1:]):
            if (by[a]["trusted_passes"], by[a]["validated_passes"]) < (by[b]["trusted_passes"], by[b]["validated_passes"]):
                return "a later key reordered across evidence levels at seq %d" % e["seq"]
    return ""


def check_i3(journal) -> str:
    reg = journal._registry
    for ev in journal.evidence():
        c = reg.checks[ev["check_id"]]
        model = ev["source"] in MODEL_SOURCES or c["trust_class"] == "diagnostic"
        if model and (ev["independence"] != "model_derived" or C.authoritative(ev, reg)):
            return "model-derived record %s counts as authority" % ev["evidence_id"]
        if ev["independence"] == "model_derived" and C.authoritative(ev, reg):
            return "model_derived record %s is authoritative" % ev["evidence_id"]
    reqs = (_first(journal, "requirements") or {}).get("reqs", [])
    for r in reqs:
        if r["kind"] == "derived_hypothesis" and C.can_gate(r) and r["status"] != "validated":
            return "unvalidated derived hypothesis %s can gate" % r["requirement_id"]
    # [EL-ARCH-1 A4] every record names a requirement of the controller's own list, and a model-derived PASS exists only where the routing table (fixed, journaled) lists that check as soft
    # support for that requirement: a critic/model-test pass can never sit on a contract/example/hidden row, so it cannot lift a row above UNVERIFIED.
    ids = {r["requirement_id"] for r in reqs}
    soft: Dict[str, set] = {}
    routed = journal.entries("routing")      # a run-level journal always has them (assert_invariants refuses one without); a bare unit-test journal is judged on the first two rules only
    for rt in routed:
        for row in rt["data"]["table"]:
            soft.setdefault(row["requirement_id"], set()).update(row["soft_checks"])
    for ev in journal.evidence():
        if ids and ev["requirement_id"] not in ids:
            return "evidence %s names a requirement outside the controller's list" % ev["evidence_id"]
        c = reg.checks[ev["check_id"]]
        if (ev["source"] in MODEL_SOURCES or c["trust_class"] == "diagnostic" or ev["independence"] == "model_derived") and ev["status"] == "pass" and routed and ev["check_id"] not in soft.get(ev["requirement_id"], set()):
            return "model-derived pass %s on %s is outside the soft-support route (it would lift a row)" % (ev["evidence_id"], ev["requirement_id"])
    return ""


def check_i4(journal) -> str:
    for s in list(NOT_PASS) + ["", "PASS", "ok", None, "passed"]:
        if map_status({"status": s}) == "pass":
            return "status mapper turns %r into pass" % (s,)
    for e in journal.entries("evidence"):
        if e["data"]["evidence"]["status"] == "pass" and e["data"]["outcome"].get("status") != "pass":
            return "record %s is pass but its outcome is %r" % (e["data"]["evidence"]["evidence_id"], e["data"]["outcome"].get("status"))
    return ""


def check_i5(result, journal, limits=None) -> str:
    stop = (journal.entries("stop") or [None])[-1]
    if stop is None:
        return "no stop record"
    d = stop["data"]
    if d["rounds"] > d["max_rounds"] or d["attempts"] > d["max_attempts"]:
        return "rounds/attempts above the caps"
    if result.stop_cause == "decided" and d["budget_hit"]:
        return "decided after budget exhaustion"
    if limits is not None:      # [EL-ARCH-1 A3] the per-problem limits are real: counters never above the cap; wall/tokens above the cap only when the problem stopped as budget
        b = result.budget_record
        over = [(n, u, c) for n, u, c in (("max_actions", d["attempts"], limits.max_actions), ("max_rounds", d["rounds"], limits.max_rounds)) if u > c]
        if result.stop_cause != "budget":
            over += [(n, u, c) for n, u, c in (("max_qwen_tokens", b["qwen_gen_tokens"] + b["qwen_prefill_tokens"], limits.max_qwen_tokens),
                                               ("max_glm_tokens", b["glm_tokens_in"] + b["glm_tokens_out"], limits.max_glm_tokens), ("max_wall_seconds", b["wall_seconds"], limits.max_wall_seconds)) if u > c and result.stop_cause != "decided"]
        if over:
            return "limit exceeded without a budget stop: %s" % over
        if result.stop_cause == "budget" and result.stop_details.startswith("limit:") and not result.stop_details.split()[0].split(":")[1] in limits.as_dict():
            return "stop_details names an unknown limit"
    try:
        C.validate("budget_record", result.budget_record)
    except Exception as ex:      # noqa: BLE001
        return "budget_record invalid: %s" % str(ex)[:120]
    return ""


def check_i6(journal) -> str:
    """Observe before commit: anything routed to commit/execute, and every finalize, has earlier sandbox evidence for that very candidate."""
    seen = set()
    for e in journal.entries():
        k, d = e["kind"], e["data"]
        if k == "evidence":
            seen.add(d["evidence"]["candidate_id"])
        elif k == "decision" and (d["next_step"] in ("commit", "execute") or d["authorized_to_execute"]) and d["candidate_id"] not in seen:
            return "decision for %s committed before any evidence" % d["candidate_id"]
        elif k == "finalize" and d["candidate_id"] not in seen:
            return "finalize for %s without earlier evidence" % d["candidate_id"]
        elif k == "step_record" and d["action"].get("executed") and d["action"].get("allowed") is not True:      # [EL-ARCH-1 B5] nothing runs that authorize_action did not allow
            return "step %s executed an action of class %s that the action-risk table did not authorize" % (d["step"], d["action"].get("class"))
        elif k == "action_auth" and d["class"] in ("secret", "protected_write", "unknown") and d["allowed"]:
            return "action class %s was authorized" % d["class"]
    return ""


def check_i7(journal) -> str:
    regs = journal.entries("registry")
    s = [r["data"] for r in regs if r["data"]["phase"] == "start"]
    e = [r["data"] for r in regs if r["data"]["phase"] == "end"]
    if not s or not e:
        return "registry hash not recorded at both ends"
    if s[0]["hash"] != e[-1]["hash"] or s[0]["scorer"] != e[-1]["scorer"]:
        return "registry or scorer changed during the run"
    return "" if journal.verify_chain() else "journal chain broken"


def check_i8(journal, ctx) -> str:
    for k in list(vars(ctx).keys()):
        if any(w in k.lower() for w in FORBIDDEN_WORDS):
            return "controller context has a field named " + k
    for cid, c in journal._registry.checks.items():
        if c["grader_only"] and c["runtime_visible"]:
            return "grader-only check %s is runtime visible" % cid
    return ""


def assert_invariants(result, journal, ctx, limits=None) -> List[str]:
    checks = {"I1": lambda: check_i1(result, journal), "I2": lambda: check_i2(journal), "I3": lambda: check_i3(journal), "I4": lambda: check_i4(journal),
              "I5": lambda: check_i5(result, journal, limits), "I6": lambda: check_i6(journal), "I7": lambda: check_i7(journal), "I8": lambda: check_i8(journal, ctx)}
    failures = {}
    if not journal.entries("routing"):      # [EL-ARCH-1 A5] the router is a required component: a run that never journaled a routing table did not use it
        raise MissingComponentError("router", "the journal holds no routing table")
    for name, fn in checks.items():
        msg = fn()
        if msg:
            failures[name] = msg
    if failures:
        raise InvariantViolation(failures)
    return sorted(checks)
