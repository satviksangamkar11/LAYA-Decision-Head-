"""[LAYA-INT-1 step E] ONE allow-list of critic experiment kinds, shared by the critic prompt/response parser (glm.py via GlmConfig.allowed_kinds) and the controller
executor (controller/agentic.py critic_stage). Pure stdlib, no network, no keys.

plan(exp, ...) turns a parsed critic experiment into ONE sandbox shell command or raises Rejected(code, detail): unsupported kind, unsafe payload, protected/hidden path,
network, shell metacharacters, unknown function. Nothing is ever executed here and nothing is silently dropped: the caller journals every Rejected."""
from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

SUPPORTED_KINDS: Tuple[str, ...] = ("function_call", "command")       # what the per-action executor can run in the WSL sandbox
SCHEMA_KINDS: Tuple[str, ...] = ("stdin_input", "function_call", "command", "state_query")      # what critic_response allows; the rest is REJECTED explicitly
PAYLOAD_HELP = {
    "function_call": "ONE call expression such as name(1, 'a') of a function defined in the candidate code above; literal arguments only (numbers, strings, lists, dicts, True/False/None)",
    "command": "either the exact test command already shown in the observed outputs, or 'python3 file.py arg ...' for a non-test .py file shown above; no shell operators, no quotes, no network",
}
MAX_PAYLOAD = 2000
META = re.compile("[;&|<>`$(){}\\\\\n\r\0*?~!#\"']")
SAFE_ARG = re.compile(r"^[A-Za-z0-9_.,:=+-]+$")
PY_FILE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.py$")
UNSAFE_WORD = re.compile(r"hidden|protected|grader|secret|\.env\b|/etc|/proc|/dev|/sys|/mnt|/root|curl|wget|\bssh\b|\bscp\b|\bnc\b|\bsudo\b|://|\.\.", re.I)


class Rejected(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__("%s: %s" % (code, detail))
        self.code, self.detail = code, detail


def _public_functions(ws: Path, protected: Sequence[str]) -> Dict[str, str]:
    """{function name: module stem} for top-level defs of the non-protected .py files in the workspace root."""
    out: Dict[str, str] = {}
    for p in sorted(Path(ws).glob("*.py")):
        if p.name in protected or not p.stem.isidentifier():
            continue
        try:
            tree = ast.parse(p.read_text("utf-8"))
        except (SyntaxError, ValueError, OSError):
            continue
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
                out.setdefault(n.name, p.stem)
    return out


def _plan_function_call(payload: str, ws: Path, protected: Sequence[str]) -> Dict[str, Any]:
    if "\n" in payload or "\r" in payload or "\0" in payload:
        raise Rejected("unsafe_payload", "function_call payload must be a single line")
    try:
        call = ast.parse(payload.strip(), mode="eval").body
    except SyntaxError:
        raise Rejected("unsafe_payload", "function_call payload is not one Python expression")
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
        raise Rejected("unsafe_payload", "function_call payload must be a call of a plain function name")
    fns = _public_functions(ws, protected)
    if call.func.id not in fns:
        raise Rejected("unknown_function", "%r is not a public function of a non-test workspace module" % call.func.id)
    try:
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {}
        for k in call.keywords:
            if k.arg is None:
                raise ValueError("**kwargs")
            kwargs[k.arg] = ast.literal_eval(k.value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        raise Rejected("unsafe_payload", "arguments must be literals only")
    fn, mod = call.func.id, fns[call.func.id]
    script = "import sys; sys.path.insert(0, '.'); from %s import %s; print(repr(%s(*%r, **%r)))" % (mod, fn, fn, args, kwargs)
    return {"kind": "function_call", "cmd": "python3 -c " + shlex.quote(script), "display": payload.strip()}


def _plan_command(payload: str, ws: Path, protected: Sequence[str], test_cmd: Optional[str]) -> Dict[str, Any]:
    p = payload.strip()
    if test_cmd and p == test_cmd.strip():
        return {"kind": "command", "cmd": p, "display": p, "note": "the task's own test command, already visible in the observed outputs"}
    if not p:
        raise Rejected("unsafe_payload", "empty command")
    m = META.search(p)
    if m:
        raise Rejected("shell_metacharacter", "character %r is not allowed" % m.group(0))
    toks = shlex.split(p)
    if toks[0] not in ("python3", "python") or len(toks) < 2 or not PY_FILE.match(toks[1]):
        raise Rejected("unsupported_command", "only 'python3 file.py [args]' or the exact test command is supported")
    names = {Path(x).name for x in protected} | set(protected)
    for t in toks:
        if t in names or Path(t).name in names or Path(t).name.startswith("test_"):
            raise Rejected("protected_path", "token %r names a protected or test file" % t)
        w = UNSAFE_WORD.search(t)
        if w:
            raise Rejected("unsafe_path_or_network", "token %r matches %r" % (t, w.group(0)))
    if not (Path(ws) / toks[1]).is_file():
        raise Rejected("unknown_file", "%r is not a file of the workspace" % toks[1])
    for a in toks[2:]:
        if not SAFE_ARG.match(a):
            raise Rejected("unsafe_payload", "argument %r is not a plain token" % a)
    if len(toks) > 10:
        raise Rejected("unsafe_payload", "too many arguments")
    return {"kind": "command", "cmd": " ".join(shlex.quote(t) for t in toks), "display": p}


def plan(exp: Any, workspace, protected: Sequence[str] = (), test_cmd: Optional[str] = None) -> Dict[str, Any]:
    """-> {"kind", "cmd", "display"} for a supported, safe experiment. Raises Rejected(code, detail) otherwise (never executes anything)."""
    if not isinstance(exp, dict) or not isinstance(exp.get("kind"), str) or not isinstance(exp.get("payload"), str):
        raise Rejected("malformed_experiment", "experiment must be an object with string kind and payload")
    kind, payload = exp["kind"], exp["payload"]
    if kind not in SUPPORTED_KINDS:
        raise Rejected("unsupported_kind", kind if kind in SCHEMA_KINDS else "unknown kind %r" % kind[:40])
    if len(payload) > MAX_PAYLOAD:
        raise Rejected("unsafe_payload", "payload longer than %d" % MAX_PAYLOAD)
    ws = Path(workspace)
    return _plan_function_call(payload, ws, protected) if kind == "function_call" else _plan_command(payload, ws, protected, test_cmd)


def judge(expected: Optional[str], run: Dict[str, Any]) -> Tuple[str, str]:
    """-> (status, observed). status in pass|fail|unknown: unknown when the critic gave no expectation for the candidate or the run was killed. Diagnostic only."""
    observed = ("exit %d\n%s%s" % (run["exit_code"], run["stdout"], run["stderr"]))[-600:]
    if expected is None or run.get("killed"):
        return "unknown", observed
    want = expected.strip()
    stdout = run["stdout"].strip()
    ok = run["exit_code"] == 0 and (want == stdout or want.strip("'\"") == stdout.strip("'\"") or want in stdout)
    return ("pass" if ok else "fail"), observed
