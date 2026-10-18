"""[EL-ARCH-1 B6] L2 validated model-written checks, deterministic discriminating inputs and behaviour clustering beyond the public examples (architecture sections 4 and 6, stages 5-7).
Pure standard library. Nothing here approves anything: validated L2 checks and clusters are RANKING / repair-feedback evidence (model-derived, capped by A4); a trusted check still decides.

L2 validation (validate_checks) keeps a model-written check only if ALL hold, each failure is dropped WITH its reason (never silently):
  shape       {"input": object, "expected": any JSON, "quote": str}                       strict parse, prose rejected (parse_checks)
  traceable   `quote` is a verbatim span of the request text (>= 8 chars) -> char_start/char_end recorded
  L3          if its input equals a public example's input, the expected output must equal that example's (a copy adds nothing and is dropped as duplicate)
  consistent  at least one TRUSTED reference (a candidate with no trusted failure that passes the public examples) passes the check; a check that every trusted reference fails
              contradicts L3/L5 and is dropped (a model-written check never overrules trusted evidence)
  non-vacuous at least one deterministic MUTANT of a passing reference fails the check (a check that nothing can fail tests nothing); `beyond_examples` says whether it also kills a mutant
              that survives the public examples
Clustering: candidates are grouped by their outputs on the public examples PLUS the deterministic derived inputs PLUS validated L2 inputs PLUS the critic experiment input (when run)."""
from __future__ import annotations

import ast
import copy
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from evidence_ladder.common.interfaces import sha256_text

THINK_CLOSE = "</think>"
FENCE = "`" * 3
MIN_QUOTE = 8


def canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


# ------------------------------------------------------------------ strict parse of model-written checks
def parse_checks(text: str, max_checks: int = 8) -> Dict[str, Any]:
    i = text.rfind(THINK_CLOSE)
    if i < 0:
        return {"status": "no_think_close", "checks": [], "rejected": [{"index": None, "reason": "thinking never finished (no </think>)"}]}
    ans = text[i + len(THINK_CLOSE):].strip()
    if ans.startswith(FENCE) and ans.endswith(FENCE) and ans.count(FENCE) == 2:
        first, _, rest = ans[3:].partition("\n")
        if first.strip().lower() in ("", "json"):
            ans = rest[:-3].strip()
    try:
        arr = json.loads(ans)
    except ValueError as e:
        return {"status": "not_json_array", "checks": [], "rejected": [{"index": None, "reason": "answer is not exactly one JSON value: %s" % str(e)[:80]}]}
    if not isinstance(arr, list):
        return {"status": "not_json_array", "checks": [], "rejected": [{"index": None, "reason": "top-level JSON value is not an array"}]}
    checks, rej = [], []
    for n, a in enumerate(arr):
        if not (isinstance(a, dict) and set(a) == {"input", "expected", "quote"} and isinstance(a["input"], dict) and isinstance(a["quote"], str)):
            rej.append({"index": n, "reason": 'item is not exactly {"input": object, "expected": value, "quote": string}'})
        elif len(checks) >= max_checks:
            rej.append({"index": n, "reason": "beyond max_checks=%d" % max_checks})
        else:
            checks.append({"index": n, "input": a["input"], "expected": a["expected"], "quote": a["quote"]})
    return {"status": "ok" if checks else "no_valid_check", "checks": checks, "rejected": rej}


# ------------------------------------------------------------------ deterministic mutants
_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mult, ast.Mod: ast.Mult, ast.Div: ast.Mult}


class _One(ast.NodeTransformer):
    """Apply exactly the k-th applicable mutation (in traversal order)."""

    def __init__(self, k: int):
        self.k, self.i, self.kind = k, 0, ""

    def _hit(self, kind: str) -> bool:
        hit = self.i == self.k
        self.i += 1
        if hit:
            self.kind = kind
        return hit

    def visit_Compare(self, node):
        self.generic_visit(node)
        for j, op in enumerate(node.ops):
            if type(op) in _CMP and self._hit("compare:%s" % type(op).__name__):
                node.ops[j] = _CMP[type(op)]()
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in _BIN and self._hit("binop:%s" % type(node.op).__name__):
            node.op = _BIN[type(node.op)]()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self._hit("boolop"):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool) and self._hit("const+1"):
            return ast.copy_location(ast.Constant(node.value + 1), node)
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._hit("not-removed"):
            return node.operand
        return node


def mutants(source: str, limit: int = 8) -> List[Dict[str, str]]:
    """Deterministic mutants of a Python source: [{"id","kind","source"}]. Mutants identical to the original or unparsable are skipped; at most `limit`, spread evenly over all applicable mutations."""
    try:
        base = ast.parse(source)
    except SyntaxError:
        return []
    norm = ast.unparse(base)
    allm, k = [], 0
    while True:
        t = _One(k)
        tree = t.visit(copy.deepcopy(base))
        if t.i <= k:
            break
        ast.fix_missing_locations(tree)
        try:
            src = ast.unparse(tree)
            ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            src = None
        if src and src != norm:
            allm.append({"id": "m%d" % k, "kind": t.kind, "source": src})
        k += 1
        if k > 400:
            break
    if len(allm) <= limit:
        return allm
    step = len(allm) / float(limit)
    return [allm[int(j * step)] for j in range(limit)]


# ------------------------------------------------------------------ deterministic discriminating inputs
def _variants(v: Any) -> List[Any]:
    if isinstance(v, bool):
        return [not v]
    if isinstance(v, int):
        return [0, 1, -1, v + 1, v * 2]
    if isinstance(v, float):
        return [0.0, -v, v * 2]
    if isinstance(v, str):
        return ["", "a", v + v, v[::-1], v.upper()]
    if isinstance(v, list):
        out = [[], list(reversed(v)), v + v, v[:1]]
        try:
            out.append(sorted(v))
        except TypeError:
            pass
        return out
    if isinstance(v, dict):
        return [{}]
    return []


def derive_inputs(public_tests: Sequence[Dict[str, Any]], n: int = 6) -> List[Dict[str, Any]]:
    """Inputs for the solve(data) object, derived from the public example inputs by fixed rules (boundary / empty / duplicate / reverse variants of one field at a time).
    Public inputs are excluded; the order is deterministic and interleaved over (example, field) so a small n still covers different fields."""
    seen = set()
    for t in public_tests:
        x = t.get("input")
        x = _load(x)
        if isinstance(x, dict):
            seen.add(canon(x))
    streams: List[List[Dict[str, Any]]] = []
    for t in public_tests:
        x = _load(t.get("input"))
        if not isinstance(x, dict):
            continue
        for key in sorted(x):
            streams.append([dict(x, **{key: nv}) for nv in _variants(x[key])])
    out = []
    j = 0
    while len(out) < n and any(j < len(s) for s in streams):
        for s in streams:
            if j < len(s) and len(out) < n:
                c = canon(s[j])
                if c not in seen:
                    seen.add(c)
                    out.append(s[j])
        j += 1
    return out


def _load(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


# ------------------------------------------------------------------ validation of model-written checks
Runner = Callable[[str, Dict[str, Any]], Dict[str, Any]]      # (source, input) -> {"status": "pass"|..., "observed": value}


def passes(run: Runner, source: str, check: Dict[str, Any]) -> Optional[bool]:
    """True/False, or None when the run was infrastructure trouble (timeout / error status without an observation)."""
    out = run(source, check["input"])
    if out.get("status") == "pass" and "observed" in out:
        return canon(out["observed"]) == canon(check["expected"])
    if out.get("status") in ("fail",) or (out.get("status") == "error" and out.get("failure_class") in ("runtime_error", "compile_error", "wrong_answer")):
        return False
    return None


def validate_checks(checks: Sequence[Dict[str, Any]], request_text: str, public_tests: Sequence[Dict[str, Any]], trusted_refs: Sequence[Dict[str, str]], run: Runner,
                    mutant_limit: int = 8) -> Dict[str, Any]:
    """trusted_refs = [{"id","source"}] candidates with no trusted failure that pass the public examples. -> {"validated": [...], "dropped": [{"index","reason"}], "mutants": n}."""
    pub = {canon(_load(t.get("input"))): canon(_load(t.get("expected"))) for t in public_tests}
    validated, dropped, n_mut = [], [], 0
    mut_cache: Dict[str, List[Dict[str, str]]] = {}
    for ck in checks:
        ix = ck["index"]
        q = ck["quote"]
        at = request_text.find(q) if len(q) >= MIN_QUOTE else -1
        if at < 0:
            dropped.append({"index": ix, "reason": "quote is not a verbatim span of the request (or shorter than %d chars)" % MIN_QUOTE})
            continue
        key = canon(ck["input"])
        if key in pub:
            dropped.append({"index": ix, "reason": ("contradicts_L3: same input as a public example, different expected" if pub[key] != canon(ck["expected"])
                                                   else "duplicate_of_public_example")})
            continue
        if not trusted_refs:
            dropped.append({"index": ix, "reason": "no trusted reference candidate to check consistency against"})
            continue
        res = [(r, passes(run, r["source"], ck)) for r in trusted_refs]
        good = [r for r, p in res if p is True]
        if not good:
            dropped.append({"index": ix, "reason": "every trusted reference fails or errors on it (a model-written check never overrules trusted evidence)"})
            continue
        ref = good[0]
        ms = mut_cache.setdefault(ref["id"], mutants(ref["source"], mutant_limit))
        n_mut = max(n_mut, len(ms))
        killed, beyond = [], False
        for m in ms:
            p = passes(run, m["source"], ck)
            if p is False:
                killed.append(m["id"])
                if not beyond and _survives_examples(run, m["source"], public_tests):
                    beyond = True
        if not killed:
            dropped.append({"index": ix, "reason": "vacuous: no mutant of the passing reference fails it (%d mutants tried)" % len(ms)})
            continue
        validated.append({"index": ix, "input": ck["input"], "expected": ck["expected"], "quote": q, "char_start": at, "char_end": at + len(q), "reference": ref["id"],
                          "kills_mutants": killed, "beyond_examples": beyond, "passed_by": sorted(r["id"] for r in good), "failed_by": sorted(r["id"] for r, p in res if p is False)})
    return {"validated": validated, "dropped": dropped, "mutants": n_mut}


def _survives_examples(run: Runner, source: str, public_tests: Sequence[Dict[str, Any]]) -> bool:
    for t in public_tests:
        inp = _load(t.get("input"))
        if not isinstance(inp, dict):
            return False
        if passes(run, source, {"input": inp, "expected": _load(t.get("expected"))}) is not True:
            return False
    return True


# ------------------------------------------------------------------ clustering
def cluster(outputs: Dict[str, List[Tuple[str, Any]]]) -> Dict[str, Any]:
    """outputs = {candidate id: [(input key, status/observed canonical), ...]} -> {"signature": {id: sig}, "clusters": {sig: [ids]}, "n": number of clusters}."""
    sig = {cid: sha256_text(canon(o))[:16] for cid, o in sorted(outputs.items())}
    groups: Dict[str, List[str]] = {}
    for cid, s in sig.items():
        groups.setdefault(s, []).append(cid)
    return {"signature": sig, "clusters": {k: sorted(v) for k, v in sorted(groups.items())}, "n": len(groups)}
