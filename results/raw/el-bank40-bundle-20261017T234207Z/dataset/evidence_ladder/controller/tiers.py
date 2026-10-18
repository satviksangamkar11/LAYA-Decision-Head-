"""[EL-ARCH-1 B1] Oracle-tier router (architecture section 5, R1.2). Pure standard library; decided by CODE from observable facts, never by a model.

  O1  a trusted executable oracle exists for the goal (tests given and present)                       loop: generate-execute-repair      ceiling VERIFIED
  O2  partial oracle: examples in the request, compiler/type-check config, an unmodified regression    loop: generate-execute-discriminate-repair
      suite                                                                                              ceiling SUPPORTED (VERIFIED only for the requirements a trusted check covers)
  O3  nothing observable                                                                              loop: best effort + coverage review; ask the user on a blocking ambiguity
                                                                                                       ceiling PLAUSIBLE / BLOCKED
Facts are re-observed after every evidence-adding step (R1.2) but may be upgraded only by what existed at intake and is still byte-identical: a test file the agent wrote (or edited)
during the run is a model artifact and never counts (R1.2: "it cannot be upgraded by artifacts the model writes"). The tier and its reason are journaled (entry `oracle_tier`) at intake and
whenever it changes; `cap_label` / `check_ceiling` make the ceiling real (invariants.check_tier)."""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TIERS = ("O1", "O2", "O3")
CEILING = {"O1": "VERIFIED", "O2": "SUPPORTED", "O3": "PLAUSIBLE"}
LOOP = {"O1": "generate-execute-repair", "O2": "generate-execute-discriminate-repair", "O3": "best-effort-coverage-review"}
LABEL_RANK = {"VERIFIED": 4, "SUPPORTED": 3, "PLAUSIBLE": 2, "UNVERIFIED": 1, "BLOCKED": 0}      # BLOCKED is a refusal: never above a ceiling, never capped
TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|[^/]*\.test\.[jt]sx?|[^/]*_test\.go|tests?\.py)$")
ASSERTY = re.compile(r"\bassert\b|unittest|def test_|\bexpect\(|pytest|self\.assert")
TYPE_CONFIG = ("mypy.ini", ".mypy.ini", "pyrightconfig.json", "tsconfig.json", "Cargo.toml", "go.mod")
# example pairs observable in the request text: f(args) == v | f(args) -> v | f(args) returns v | Input: ... Output: ...
EX_PATTERNS = (re.compile(r"[A-Za-z_]\w*\([^()\n]*\)\s*(==|->|=>|returns?|should return|gives)\s*\S+"),
               re.compile(r"Input:\s*\S.*?\n\s*(Expected )?Output:\s*\S", re.S | re.I),
               re.compile(r"\be\.g\.[^\n]*\S+\s*(->|=>|=)\s*\S+", re.I))
VAGUE = re.compile(r"\b(somehow|better|nicer?|improve|clean up|make it good|optimi[sz]e|as appropriate|whatever|etc\.?)\b", re.I)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def snapshot(workspace) -> Dict[str, str]:
    """{relative path: sha256} of the workspace files at intake (the trusted baseline for fact upgrades)."""
    ws = Path(workspace)
    return {p.relative_to(ws).as_posix(): _sha(p.read_bytes()) for p in sorted(ws.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts}


def count_examples(text: str) -> int:
    return max(len(list(p.finditer(text))) for p in EX_PATTERNS)      # overlapping patterns must not double count


def observe_facts(workspace, user_text: str, test_cmd: Optional[str], protected: Sequence[str], baseline: Optional[Dict[str, str]] = None,
                  compiler_cmd: Optional[str] = None, regression_cmd: Optional[str] = None) -> Dict[str, Any]:
    """Observable facts only. `baseline` = snapshot() taken at intake; when given, a file counts as a trusted oracle only if it is in the baseline AND unchanged now."""
    ws = Path(workspace)
    now = snapshot(ws) if ws.is_dir() else {}
    prot = {Path(p).as_posix() for p in protected}

    def trusted_file(rel: str) -> bool:
        return rel in now and (baseline is None or baseline.get(rel) == now[rel])

    given = [r for r in sorted(prot) if trusted_file(r) and ASSERTY.search((ws / r).read_text("utf-8", "replace"))]
    tests_given = bool(test_cmd and test_cmd.strip()) and bool(prot) and len(given) == len(prot)
    other_tests = [r for r in sorted(now) if TEST_FILE.search(r) and r not in prot and trusted_file(r) and (baseline is None or r in baseline)]
    regression = [r for r in other_tests if ASSERTY.search((ws / r).read_text("utf-8", "replace"))]
    typed = [n for n in TYPE_CONFIG if trusted_file(n)]
    if trusted_file("pyproject.toml") and "[tool.mypy]" in (ws / "pyproject.toml").read_text("utf-8", "replace"):
        typed.append("pyproject.toml[tool.mypy]")
    if trusted_file("setup.cfg") and "[mypy]" in (ws / "setup.cfg").read_text("utf-8", "replace"):
        typed.append("setup.cfg[mypy]")
    ex = count_examples(user_text)
    return {"tests_given": tests_given, "protected_test_files": given, "examples_in_request": ex, "type_or_compile_check": bool(compiler_cmd) or bool(typed),
            "type_config": typed, "compiler_cmd": compiler_cmd, "regression_suite": regression, "regression_cmd": regression_cmd if regression else None,
            "model_artifacts_ignored": sorted(r for r in now if baseline is not None and (r not in baseline or baseline[r] != now[r]) and (TEST_FILE.search(r) or r in prot))}


def classify(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Facts -> {"tier", "reason", "loop", "ceiling", "covered", "ask_user"}. Pure and deterministic."""
    if facts["tests_given"]:
        tier, reason = "O1", "trusted executable oracle: %s" % ", ".join(facts["protected_test_files"])
    else:
        why = []
        if facts["examples_in_request"]:
            why.append("%d example(s) in the request" % facts["examples_in_request"])
        if facts["type_or_compile_check"]:
            why.append("type/compile check (%s)" % (facts["compiler_cmd"] or ", ".join(facts["type_config"])))
        if facts["regression_suite"]:
            why.append("regression suite (%s)" % ", ".join(facts["regression_suite"]))
        tier, reason = ("O2", "partial oracle: " + "; ".join(why)) if why else ("O3", "no tests, examples, type/compile check or regression suite observed")
    covered = [k for k, on in (("tests", facts["tests_given"]), ("examples", bool(facts["examples_in_request"])), ("type_or_compile", facts["type_or_compile_check"]),
                              ("regression", bool(facts["regression_suite"]))) if on]
    return {"tier": tier, "reason": reason, "loop": LOOP[tier], "ceiling": CEILING[tier], "covered": covered, "ask_user": tier == "O3"}


def ambiguities(user_text: str, workspace) -> List[str]:
    """Blocking ambiguities of an O3 request, detected by code: (a) the request names no file/identifier of the workspace; (b) it states no verifiable criterion (only vague wording)."""
    out = []
    names = set()
    ws = Path(workspace) if workspace is not None else None
    if ws is not None and ws.is_dir():
        for p in ws.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                names.add(p.name)
                names.add(p.stem)
                if p.suffix == ".py":
                    try:
                        names |= {n.name for n in ast.parse(p.read_text("utf-8")).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                    except (SyntaxError, ValueError):
                        pass
    if ws is not None and not any(re.search(r"(?<![\w.])%s(?![\w])" % re.escape(n), user_text) for n in names if len(n) > 2):
        out.append("target_unknown: the request names no file, function or class of the workspace")
    if VAGUE.search(user_text) or len(user_text.split()) < 6:
        out.append("no_verifiable_criterion: the request states no checkable success condition")
    return out


def cap_label(label: str, tier: str) -> str:
    """The best label a tier allows; BLOCKED passes through unchanged."""
    if label == "BLOCKED" or LABEL_RANK[label] <= LABEL_RANK[CEILING[tier]]:
        return label
    return CEILING[tier]


def check_ceiling(label: str, tier: str) -> str:
    return "" if cap_label(label, tier) == label else "label %s is above the %s ceiling %s" % (label, tier, CEILING[tier])


def route_tier(facts: Dict[str, Any], user_text: str = "", workspace=None) -> Dict[str, Any]:
    c = classify(facts)
    c["facts"] = {k: facts[k] for k in ("tests_given", "examples_in_request", "type_or_compile_check", "regression_suite")}
    c["ambiguities"] = ambiguities(user_text, workspace) if c["ask_user"] else []
    c["blocked_until_answered"] = bool(c["ask_user"] and c["ambiguities"])
    return c


def journal_tier(journal, step: str, routed: Dict[str, Any], last: Optional[Dict[str, Any]] = None) -> bool:
    """Journal `oracle_tier` at intake (last is None) and whenever tier/reason changes (R1.2: the router is re-run after evidence-adding steps). When an O3 request is blocked, also
    journal `ask_user` with the exact ambiguities. -> True when an entry was written."""
    if last is not None and last.get("tier") == routed["tier"] and last.get("reason") == routed["reason"]:
        return False
    journal.append("oracle_tier", {"step": step, "tier": routed["tier"], "reason": routed["reason"], "loop": routed["loop"], "ceiling": routed["ceiling"], "covered": routed["covered"],
                                   "facts": routed["facts"], "changed_from": None if last is None else last.get("tier")})
    if routed["blocked_until_answered"]:
        journal.append("ask_user", {"step": step, "tier": routed["tier"], "ambiguities": routed["ambiguities"], "label": "BLOCKED", "next_step": "block"})
    return True


def o3_outcome(task_id: str, routed: Dict[str, Any]) -> Dict[str, Any]:
    """What a controller reports for an O3 task it must not run as verified work: BLOCKED + the questions (blocking ambiguity), else best effort capped at PLAUSIBLE."""
    if routed["blocked_until_answered"]:
        return {"task_id": task_id, "answer_label": "BLOCKED", "next_step": "ask_user", "questions": routed["ambiguities"], "ceiling": routed["ceiling"]}
    return {"task_id": task_id, "answer_label": "PLAUSIBLE", "next_step": "report", "questions": [], "ceiling": routed["ceiling"]}
