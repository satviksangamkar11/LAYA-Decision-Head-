"""[EL-ARCH-1 C6] Evaluation-only oracles must be unreachable from runtime code. This module is the strengthened half of I8 that the name scan in controller/invariants.py cannot do:
  * static_oracle_scan(): AST scan of EVERY runtime module under evidence_ladder/ (not only controller/): no import of the protected-grader module, no string literal naming the hidden-case files.
    Allowed: the grader module itself, test code, and the files in ALLOWED_MENTIONS (deny-list patterns that only REFUSE such text).
  * deep_context_check(): walks the object graph of a ControllerContext (types, attributes, dict/list/tuple members, depth-bounded) and reports any grader/gate/hidden-grade object or any string
    that names the hidden-case files; a field NAME scan alone misses a grader tucked inside a backend or a config value.
  * (tests) ProtectedGrader must not read any case file before the selection gate closed: the constructor and an early grade() touch nothing (phaseC_tests).
HONEST STATUS (to be repeated in every report): the exposed development cases are NOT sealed. They are public in Git history and sit at a fixed path; isolation is import hygiene plus
the selection gate, not secrecy. Numbers from them are development evidence only."""
from __future__ import annotations

import ast
import inspect
import types
from pathlib import Path
from typing import Any, Dict, List, Set

PKG = Path(__file__).resolve().parents[1]                       # evidence_ladder/
GRADER_MODULE = PKG / "execution" / "protected.py"
FORBIDDEN_IMPORT_PARTS = ("execution.protected", "execution import protected", ".protected")
FORBIDDEN_LITERALS = ("protected" + "_cases", "hidden" + "_cases", "oracles" + ".py")   # built by concatenation so this checker does not match itself
ALLOWED_MENTIONS = {"critic_batch/validator.py": "deny-list patterns that reject such text in critic packets",
                    "execution/receipts.py": "path + sha256 pins of grader files for integrity checks; it hashes them, it parses no case"}
FORBIDDEN_TYPE_NAMES = {"ProtectedGrader", "GatedGrader", "SelectionGate", "HiddenGrade"}
NOT_SEALED_STATEMENT = ("The exposed development cases are NOT sealed: they were committed to Git history, are public, and sit at a fixed path. Runtime isolation here is import hygiene "
                        "and the selection gate, not secrecy. Every number from them is development evidence only; a held-out claim needs hidden tests that were never published.")


def _is_runtime(p: Path, root: Path) -> bool:
    rel = p.relative_to(root).as_posix()
    return p.resolve() != GRADER_MODULE and "/tests/" not in "/" + rel and not rel.startswith("tests/") and not rel.startswith("testing/")


def static_oracle_scan(root: Path = PKG) -> List[str]:
    """-> offending 'file:line what' strings. Runs over every runtime module under evidence_ladder/."""
    bad: List[str] = []
    for p in sorted(Path(root).rglob("*.py")):
        if not _is_runtime(p, Path(root)):
            continue
        rel = p.relative_to(root).as_posix()
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for n in ast.walk(tree):
            names: List[str] = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mod = ("." * n.level) + (n.module or "")
                names = [mod] + [mod + "." + a.name for a in n.names]
            for nm in names:
                low = nm.lower()
                if low.endswith("protected") or ".protected." in low or "execution.protected" in low or "execution import protected" in low:
                    bad.append("%s:%d imports %s" % (rel, n.lineno, nm))
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and rel not in ALLOWED_MENTIONS:
                for lit in FORBIDDEN_LITERALS:
                    if lit in n.value:
                        bad.append("%s:%d string names %s" % (rel, n.lineno, lit))
    return bad


def deep_context_check(ctx: Any, max_depth: int = 6) -> List[str]:
    """-> findings. Objects are identified by type name and module, strings by content; cycles are cut."""
    found: List[str] = []
    seen: Set[int] = set()

    def walk(o: Any, path: str, d: int) -> None:
        if d > max_depth or id(o) in seen:
            return
        seen.add(id(o))
        # modules, classes and functions are code, not data: following them walks the whole stdlib (Kaggle's Python reached os.stat.UF_HIDDEN via
        # qwen -> dataclasses -> inspect -> os). A grader CLASS held directly is still flagged by name.
        if isinstance(o, type) and o.__name__ in FORBIDDEN_TYPE_NAMES:
            found.append("%s is the class %s.%s" % (path, o.__module__, o.__name__))
            return
        if isinstance(o, (types.ModuleType, type)) or inspect.isroutine(o):
            return
        t = type(o)
        if t.__name__ in FORBIDDEN_TYPE_NAMES or t.__module__.endswith("execution.protected"):
            found.append("%s is a %s.%s" % (path, t.__module__, t.__name__))
            return
        if isinstance(o, str):
            for lit in FORBIDDEN_LITERALS:
                if lit in o:
                    found.append("%s names %s" % (path, lit))
            return
        if isinstance(o, (bytes, int, float, bool, type(None))):
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k, path + "{key}", d + 1)
                walk(v, "%s[%r]" % (path, k), d + 1)
        elif isinstance(o, (list, tuple, set, frozenset)):
            for i, v in enumerate(o):
                walk(v, "%s[%d]" % (path, i), d + 1)
        elif callable(o) and not hasattr(o, "__dict__"):
            return
        else:
            for k, v in list(getattr(o, "__dict__", {}).items()):
                if any(w in k.lower() for w in ("grader", "protected", "oracle", "hidden")):
                    found.append("%s.%s: attribute name" % (path, k))
                walk(v, path + "." + k, d + 1)

    walk(ctx, "ctx", 0)
    return found


def check_i8_deep(ctx: Any) -> str:
    """Drop-in extension for invariants.check_i8: '' or a message. Phase B's controller calls it next to the name scan (see PHASE_C_REPORT)."""
    f = deep_context_check(ctx)
    return "controller context reaches evaluation-only material: " + "; ".join(f[:5]) if f else ""
