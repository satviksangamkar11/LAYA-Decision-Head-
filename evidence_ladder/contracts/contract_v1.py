"""[P4A] Task contract format v1 + trusted checks (V1.5). Versioned: a change of meaning is contract_v2, never an edit of v1.

A contract (JSON) says what a task permits and promises, and where each statement comes from:
  {"contract_version": "task_contract_v1", "task_id": "...",
   "provenance": {"source": "author" | "derived:<parser id>" | "model:<id>", "written_before_candidates": true, "author": "...", "created_utc": "..."},
   "input":  <schema>,  "output": <schema or null>,
   "edges":  [<input>, ...]                      # inputs the source explicitly permits (no expected values)
   "properties_module": {"path": "...", "sha256": "...", "spec_key": "..."} | null   # optional author code: props(inp, out) -> [(name, True|False|None)], valid(inp)
  }
Schema (closed mini-language; anything else is a ContractError):
  {"type": "int", "min": a, "max": b} | {"type": "str", "pattern": regex, "min_len": a, "max_len": b} | {"type": "bool"} | {"type": "null"}
  {"type": "list", "items": <schema>, "min_len": a, "max_len": b} | {"type": "object", "fields": {name: <schema>}, "exact": true}
  {"type": "any_of", "options": [<schema>, ...]} | {"type": "any"}   (any = no constraint; validity then rests on the author's valid())

TRUST (user rule 2026-10-20): author contracts and deterministic derivations from explicit task declarations are AUTHORITATIVE; model-proposed contracts are
DIAGNOSTIC only (shadow), whatever they pass, because passing the public examples does not prove a property. No trustworthy contract -> no verdict (uncertainty kept).

Checks on an executed candidate (outputs per input): HARD failure = a crash on a VALID input, an output that violates the declared output schema on a VALID input,
or a property returning False on a VALID input. Invalid inputs are never evidence; uncertain inputs (valid() is None) are skipped. A whole-run crash in a sequential
runner is attributed to the first input without a result (P3 limitation fix), and only counts if that input is valid."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

VERSION = "task_contract_v1"
AUTHORITATIVE = ("author", "derived")


class ContractError(ValueError):
    pass


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def check_schema(s: Dict[str, Any]) -> None:
    t = s.get("type") if isinstance(s, dict) else None
    allowed = {"any": set(), "int": {"min", "max"}, "str": {"pattern", "min_len", "max_len"}, "bool": set(), "null": set(), "list": {"items", "min_len", "max_len"},
               "object": {"fields", "exact"}, "any_of": {"options"}}
    if t not in allowed:
        raise ContractError("unknown schema type %r" % (t,))
    extra = set(s) - {"type"} - allowed[t]
    if extra:
        raise ContractError("schema %s has unknown keys %s" % (t, sorted(extra)))
    if t == "list":
        check_schema(s["items"])
    elif t == "object":
        for v in s["fields"].values():
            check_schema(v)
    elif t == "any_of":
        for v in s["options"]:
            check_schema(v)
    elif t == "str" and "pattern" in s:
        re.compile(s["pattern"])


def matches(s: Dict[str, Any], x: Any) -> bool:
    t = s["type"]
    if t == "any":
        return True
    if t == "int":
        return _is_int(x) and s.get("min", x) <= x <= s.get("max", x)
    if t == "str":
        return isinstance(x, str) and s.get("min_len", 0) <= len(x) <= s.get("max_len", len(x)) and ("pattern" not in s or re.fullmatch(s["pattern"], x) is not None)
    if t == "bool":
        return isinstance(x, bool)
    if t == "null":
        return x is None
    if t == "list":
        return isinstance(x, list) and s.get("min_len", 0) <= len(x) <= s.get("max_len", len(x)) and all(matches(s["items"], v) for v in x)
    if t == "object":
        f = s["fields"]
        return isinstance(x, dict) and set(f) <= set(x) and (not s.get("exact", True) or set(x) == set(f)) and all(matches(f[k], x[k]) for k in f)
    return any(matches(o, x) for o in s["options"])          # any_of


@dataclass
class Contract:
    task_id: str
    source: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    edges: List[Any] = field(default_factory=list)
    props: Optional[Callable[[Any, Any], List[Tuple[str, Optional[bool]]]]] = None
    extra_valid: Optional[Callable[[Any], Optional[bool]]] = None     # author code may narrow validity (True/False/None); never widens the schema
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def authoritative(self) -> bool:
        return self.source.split(":")[0] in AUTHORITATIVE and bool(self.raw.get("provenance", {}).get("written_before_candidates", self.source.startswith("derived")))

    def valid(self, inp: Any) -> Optional[bool]:
        if not matches(self.input_schema, inp):
            return False
        if self.extra_valid is not None:
            return self.extra_valid(inp)
        return True

    def check(self, inp: Any, out: Any) -> List[Tuple[str, Optional[bool]]]:
        res: List[Tuple[str, Optional[bool]]] = []
        if self.output_schema is not None:
            res.append(("output_schema", matches(self.output_schema, out)))
        if self.props is not None:
            res += list(self.props(inp, out))
        return res


def load(doc: Dict[str, Any], base: Path = Path(".")) -> Contract:
    if doc.get("contract_version") != VERSION:
        raise ContractError("not a %s contract" % VERSION)
    src = (doc.get("provenance") or {}).get("source", "")
    if not src or src.split(":")[0] not in ("author", "derived", "model"):
        raise ContractError("contract provenance.source must be author | derived:<id> | model:<id>")
    check_schema(doc["input"])
    if doc.get("output") is not None:
        check_schema(doc["output"])
    props = extra = None
    pm = doc.get("properties_module")
    if pm:
        p = (base / pm["path"]).resolve()
        if hashlib.sha256(p.read_bytes()).hexdigest() != pm["sha256"]:
            raise ContractError("properties module %s differs from its pinned sha256" % pm["path"])
        spec = importlib.util.spec_from_file_location("contract_props_%s" % doc["task_id"].replace("-", "_"), p)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        obj = mod.SPECS[pm["spec_key"]] if pm.get("spec_key") else mod          # a module may hold several task specs (e.g. SPECS[task_id])
        props, extra = getattr(obj, "props", None), getattr(obj, "valid", None)
    c = Contract(doc["task_id"], src, doc["input"], doc.get("output"), list(doc.get("edges") or []), props, extra, doc)
    bad = [e for e in c.edges if c.valid(e) is not True]
    if bad:
        raise ContractError("edge inputs not valid under the contract's own schema: %s" % bad[:2])
    return c


def self_check(c: Contract, public_tests: Sequence[Dict[str, Any]]) -> List[str]:
    """Consistency with the task's own public examples: every public input valid and every stated check true on its expected output.
    NECESSARY, NOT SUFFICIENT: passing it never makes a model-proposed contract authoritative."""
    probs = []
    for k, t in enumerate(public_tests):
        if c.valid(t["input"]) is not True:
            probs.append("public input %d not valid" % k)
        probs += ["public example %d violates %s" % (k, n) for n, ok in c.check(t["input"], t["expected"]) if ok is False]
    return probs


def evaluate(c: Contract, inputs: Sequence[Any], results: Sequence[Optional[Tuple[bool, Any]]], whole_run_crashed: bool = False) -> Dict[str, Any]:
    """results[k] = (ok, value) for input k, or None when the runner produced no result for it.
    -> {"hard": [...], "uncertain": [...], "mode": "authoritative"|"diagnostic", "checked_valid_inputs": n}"""
    hard, unc, nvalid = [], [], 0
    first_missing = next((k for k, r in enumerate(results) if r is None), None)
    for k, x in enumerate(inputs):
        v = c.valid(x)
        if v is not True:
            continue
        nvalid += 1
        r = results[k] if k < len(results) else None
        if r is None:
            if whole_run_crashed and k == first_missing:
                hard.append({"input": k, "check": "crash_on_valid_input", "detail": "sequential runner stopped at this input"})
            else:
                unc.append("no result for valid input %d" % k)
            continue
        ok, val = r
        if not ok:
            hard.append({"input": k, "check": "crash_on_valid_input", "detail": str(val)[:120]}); continue
        hard += [{"input": k, "check": "property:" + n} for n, good in c.check(x, val) if good is False]
    return {"hard": hard, "uncertain": unc, "mode": "authoritative" if c.authoritative else "diagnostic", "checked_valid_inputs": nvalid}


# ---------------------------------------------------------------- deterministic parser for explicit declarations (bank style: Input {"n": integer 1..10000, ...})
PARSER_ID = "decl_v1"
_FIELD = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*([^,}]+?)(?=,\s*"|\s*}\s*\.|\s*}\s*$|\s*}\s)')
_TYPES = [
    (re.compile(r"^(?:integer|int)\s+(-?\d+)\.\.(-?\d+)$"), lambda m: {"type": "int", "min": int(m.group(1)), "max": int(m.group(2))}),
    (re.compile(r"^(?:nonnegative|non-negative) (?:integer|int)$"), lambda m: {"type": "int", "min": 0}),
    (re.compile(r"^positive (?:integer|int)$"), lambda m: {"type": "int", "min": 1}),
    (re.compile(r"^(?:integer|int)$"), lambda m: {"type": "int"}),
    (re.compile(r"^string$"), lambda m: {"type": "str"}),
    (re.compile(r"^(?:integer|int) list$"), lambda m: {"type": "list", "items": {"type": "int"}}),
    (re.compile(r"^(?:boolean|bool)$"), lambda m: {"type": "bool"}),
]


def derive_from_text(task_id: str, prompt: str) -> Optional[Dict[str, Any]]:
    """Input schema from an explicit `Input {"field": <type>, ...}` declaration. Returns a contract dict ONLY if every declared field maps to a known type;
    otherwise None (no partial guesses). Output is left undeclared (the parser does not read prose)."""
    m = re.match(r'\s*Input\s*(\{.*?\})[.;]?\s', prompt + " ")
    if not m:
        return None
    body = m.group(1)
    fields = {}
    for name, decl in _FIELD.findall(body):
        decl = decl.strip()
        for rx, mk in _TYPES:
            mm = rx.match(decl)
            if mm:
                fields[name] = mk(mm); break
        else:
            return None
    declared = re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', body)
    if not fields or set(declared) != set(fields):
        return None
    return {"contract_version": VERSION, "task_id": task_id, "provenance": {"source": "derived:" + PARSER_ID, "written_before_candidates": True},
            "input": {"type": "object", "fields": fields, "exact": True}, "output": None, "edges": []}
