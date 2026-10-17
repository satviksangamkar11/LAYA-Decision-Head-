"""[EL-ARCH-1 B2] Deterministic INVESTIGATE stage (architecture stage 2): inspect the source files, run the EXISTING tests in the sandbox, parse the failing location / assertion, record the
observation as L5 evidence in the journal and put a short report into the next context. NO model is involved: the parser is a fixed set of regular expressions and `ast`.
The test run itself goes through AgenticRun._observe, i.e. the same sandbox call, the same protected_tests_v1 / protected_files_unchanged_v1 evidence and the same label logic as every other
end-state observation (so the baseline failure is an authoritative record for exactly the pre-edit workspace state)."""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

FRAME = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')
PYTEST_FRAME = re.compile(r"^(\S+?\.py):(\d+): (\w[\w.]*)", re.M)
EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)|AssertionError|SyntaxError)\b:?\s*(.*)$")


def _rel(file: str, files: Sequence[str]) -> Optional[str]:
    """Map a path printed by the interpreter (absolute in the sandbox, or relative) to a workspace-relative file, by the longest matching suffix."""
    f = file.replace("\\", "/")
    best = None
    for r in files:
        if f == r or f.endswith("/" + r):
            if best is None or len(r) > len(best):
                best = r
    return best


def inspect_sources(workspace, protected: Sequence[str]) -> List[Dict[str, Any]]:
    ws = Path(workspace)
    prot = {Path(p).as_posix() for p in protected}
    out = []
    for p in sorted(x for x in ws.rglob("*") if x.is_file() and "__pycache__" not in x.parts and ".git" not in x.parts):
        rel = p.relative_to(ws).as_posix()
        raw = p.read_bytes()
        rec = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "lines": raw.count(b"\n") + (0 if raw.endswith(b"\n") or not raw else 1), "protected_test": rel in prot, "defs": []}
        if p.suffix == ".py":
            try:
                rec["defs"] = [{"name": n.name, "line": n.lineno} for n in ast.walk(ast.parse(raw.decode("utf-8"))) if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
            except (SyntaxError, ValueError):
                rec["syntax_error"] = True
        out.append(rec)
    return out


def parse_failure(output: str, exit_code: int, files: Sequence[str], protected: Sequence[str]) -> Dict[str, Any]:
    """-> {"kind", "exception", "message", "frames", "test_location", "code_location"}. kind: none (exit 0) | assertion | syntax | exception | timeout | unparsed."""
    if exit_code == 0:
        return {"kind": "none", "exception": "", "message": "", "frames": [], "test_location": None, "code_location": None}
    text = output.replace("\r\n", "\n")
    frames = [(m.group(1), int(m.group(2)), m.group(3) or "") for m in FRAME.finditer(text)]
    if not frames:
        frames = [(m.group(1), int(m.group(2)), "") for m in PYTEST_FRAME.finditer(text)]
    exc, msg = "", ""
    lines = text.splitlines()
    for ln in reversed(lines):                     # the last exception-looking line is the one that ended the run
        m = EXC_LINE.match(ln.strip())
        if m and not ln.startswith(" "):
            exc, msg = m.group(1), m.group(2)
            break
    if exc == "" and exit_code in (124, 137, -9):
        kind = "timeout"
    elif exc == "AssertionError" or (not exc and re.search(r"^E\s+assert ", text, re.M)):
        kind = "assertion"
        exc = exc or "AssertionError"
    elif exc in ("SyntaxError", "IndentationError", "TabError"):
        kind = "syntax"
    elif exc:
        kind = "exception"
    else:
        kind = "unparsed"
    prot = {Path(p).as_posix() for p in protected}
    mapped = [(_rel(f, files), ln, fn) for f, ln, fn in frames]
    mapped = [(r, ln, fn) for r, ln, fn in mapped if r]
    test_loc = next(((r, ln, fn) for r, ln, fn in reversed(mapped) if r in prot), None)
    code_loc = next(((r, ln, fn) for r, ln, fn in reversed(mapped) if r not in prot), None)

    def loc(t):
        return None if t is None else {"file": t[0], "line": t[1], "function": t[2]}
    return {"kind": kind, "exception": exc, "message": msg[:300], "frames": [{"file": r, "line": ln, "function": fn} for r, ln, fn in mapped][-6:], "test_location": loc(test_loc), "code_location": loc(code_loc)}


def _line(ws: Path, rel: str, n: int) -> str:
    try:
        return ws.joinpath(rel).read_text("utf-8").splitlines()[n - 1].strip()
    except (OSError, IndexError, ValueError):
        return ""


def link_assertion(workspace, failure: Dict[str, Any], sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Which non-test function does the failing assertion call? ast of the failing test line (read from the workspace, not from the traceback) matched against the defs of the sources."""
    t = failure.get("test_location")
    if not t:
        return []
    src = _line(Path(workspace), t["file"], t["line"])
    t["text"] = src[:300]
    try:
        tree = ast.parse(src.lstrip())
    except SyntaxError:
        return []
    called = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if nm and nm not in called:
                called.append(nm)
    out = []
    for s in sources:
        if s["protected_test"] or not s["path"].endswith(".py"):
            continue
        for d in s["defs"]:
            if d["name"] in called:
                out.append({"function": d["name"], "file": s["path"], "line": d["line"]})
    return out


def report_text(test_cmd: str, exit_code: int, sources: List[Dict[str, Any]], failure: Dict[str, Any], targets: List[Dict[str, Any]]) -> str:
    L = ["Investigation (deterministic, no model):"]
    L.append("- files: " + "; ".join("%s (%d lines%s%s)" % (s["path"], s["lines"], ", protected test" if s["protected_test"] else "",
                                                             ", defs: " + ", ".join("%s@%d" % (d["name"], d["line"]) for d in s["defs"]) if s["defs"] else "") for s in sources))
    L.append("- test run `%s`: exit %d" % (test_cmd, exit_code))
    if failure["kind"] == "none":
        L.append("- the existing tests already pass")
    else:
        t = failure["test_location"]
        L.append("- failure: %s%s%s" % (failure["exception"] or failure["kind"], (": " + failure["message"]) if failure["message"] else "",
                                       (" at %s:%d: %s" % (t["file"], t["line"], t.get("text", ""))) if t else ""))
        c = failure["code_location"]
        if c:
            L.append("- code location of the failure: %s:%d in %s" % (c["file"], c["line"], c["function"] or "?"))
        if targets:
            L.append("- code under test: " + ", ".join("%s (%s:%d)" % (x["function"], x["file"], x["line"]) for x in targets))
    return "\n".join(L)


# [EL-ARCH-1 A2.2] bounded real context for the proposer. Sizes are fixed here (truncation is journaled as `investigate_context`); hidden / evaluation-only files never enter it.
MAX_FILE_CHARS, MAX_TOTAL_CHARS, MAX_OUTPUT_CHARS, MAX_REQUEST_CHARS, WINDOW, MAX_SIGNATURES = 1400, 4200, 800, 700, 12, 12
HIDDEN_PATTERNS = ("*hidden*", "*grader*", "*oracle*", "*expected_*", "*secret*", ".env*", ".git/*", "*/.git/*")
CONSTRAINT = re.compile(r"\b(do not|don't|must not|must|only|never|without|keep|should not|not modify|leave)\b", re.I)


class ContextLeak(Exception):
    """a line of a hidden / evaluation-only file reached the proposal context"""


def is_hidden(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel.lower(), p) for p in HIDDEN_PATTERNS)


def signatures(path: Path) -> List[str]:
    try:
        tree = ast.parse(path.read_text("utf-8"))
    except (SyntaxError, ValueError, OSError):
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append("%sdef %s(%s)%s  # line %d" % ("async " if isinstance(n, ast.AsyncFunctionDef) else "", n.name, ast.unparse(n.args), (" -> " + ast.unparse(n.returns)) if n.returns else "", n.lineno))
        elif isinstance(n, ast.ClassDef):
            out.append("class %s  # line %d" % (n.name, n.lineno))
    return out


def snippet(path: Path, focus: Optional[int], cap: int):
    """-> (numbered text, meta). The whole file when it fits in `cap` characters, else a window around the focus line (or the head); meta records what was cut."""
    lines = path.read_text("utf-8", errors="replace").splitlines()
    total = sum(len(x) + 1 for x in lines)
    lo, hi = 0, len(lines)
    if total > cap:
        lo = max(0, (focus - 1 if focus else 0) - (WINDOW if focus else 0))
        hi = min(len(lines), (focus - 1 + WINDOW + 1) if focus else len(lines))
        while hi > lo + 1 and sum(len(x) + 9 for x in lines[lo:hi]) > cap:
            hi -= 1
    text = "\n".join("%4d| %s" % (i + 1, lines[i]) for i in range(lo, hi))
    return text, {"lines_total": len(lines), "shown": [lo + 1, hi] if lines else [0, 0], "truncated": lo > 0 or hi < len(lines), "chars_total": total}


def build_context(run, sources, failure, targets, output: str):
    """The proposer-facing INVESTIGATE report: request + constraints, file paths and signatures, bounded source snippets (code under test first), the failing test line and output.
    Hidden / evaluation-only files are excluded by name, the protected test is shown only as its failing line. -> (text, meta); meta is journaled as `investigate_context`."""
    t = run.task
    ws = Path(t.workspace)
    prot = {Path(p).as_posix() for p in t.protected}
    hidden = [s["path"] for s in sources if is_hidden(s["path"])]
    vis = [s for s in sources if s["path"] not in hidden]
    L = ["Investigation (deterministic, no model):"]
    L.append("- request (verbatim, first %d chars): %s" % (MAX_REQUEST_CHARS, t.user_text[:MAX_REQUEST_CHARS].replace("\n", " ")))
    cons = [x.strip()[:200] for x in re.split(r"(?<=[.!?])\s+|\n", t.user_text) if CONSTRAINT.search(x)][:5]
    L.append("- constraints: " + ("; ".join(cons) if cons else "none stated") + "; protected files (never modify): " + ", ".join(sorted(prot)) + ("; allowed scope: " + ", ".join(t.scope) if t.scope else ""))
    L.append("- files: " + "; ".join("%s (%d lines%s)" % (s["path"], s["lines"], ", protected test" if s["protected_test"] else "") for s in vis))
    L.append("- test run `%s`: exit %d" % (t.test_cmd, run.last_exec["exit_code"]))
    if failure["kind"] == "none":
        L.append("- the existing tests already pass")
    else:
        tl = failure["test_location"]
        L.append("- failure: %s%s%s" % (failure["exception"] or failure["kind"], (": " + failure["message"]) if failure["message"] else "", (" at %s:%d: %s" % (tl["file"], tl["line"], tl.get("text", ""))) if tl else ""))
        cl = failure["code_location"]
        if cl:
            L.append("- code location of the failure: %s:%d in %s" % (cl["file"], cl["line"], cl["function"] or "?"))
        if targets:
            L.append("- code under test: " + ", ".join("%s (%s:%d)" % (x["function"], x["file"], x["line"]) for x in targets))
        L.append("- test output (last %d chars):\n%s" % (MAX_OUTPUT_CHARS, output.strip()[-MAX_OUTPUT_CHARS:]))
    focus = {x["file"]: x["line"] for x in targets}
    if failure.get("code_location"):
        focus.setdefault(failure["code_location"]["file"], failure["code_location"]["line"])
    order = sorted((s for s in vis if not s["protected_test"] and s["path"].endswith(".py")), key=lambda s: (s["path"] not in focus, s["path"]))
    meta = {"files": [], "excluded_hidden": sorted(hidden), "protected_shown_as": "failing line only", "caps": {"file": MAX_FILE_CHARS, "total": MAX_TOTAL_CHARS, "output": MAX_OUTPUT_CHARS, "request": MAX_REQUEST_CHARS},
            "output_truncated": len(output.strip()) > MAX_OUTPUT_CHARS, "request_truncated": len(t.user_text) > MAX_REQUEST_CHARS}
    left = MAX_TOTAL_CHARS
    for s in order:
        sig = signatures(ws / s["path"])
        cap = min(MAX_FILE_CHARS, left)
        if cap <= 200:
            meta["files"].append({"path": s["path"], "shown": None, "truncated": True, "reason": "total_cap"})
            continue
        txt, m = snippet(ws / s["path"], focus.get(s["path"]), cap)
        left -= len(txt)
        sig_txt = "; ".join(sig[:MAX_SIGNATURES]) + ("; (+%d more)" % (len(sig) - MAX_SIGNATURES) if len(sig) > MAX_SIGNATURES else "")
        L.append("- source %s%s:\n%s" % (s["path"], (" | signatures: " + sig_txt) if sig else "", txt))
        meta["files"].append(dict(m, path=s["path"], sha256=s["sha256"], signatures=len(sig), signatures_truncated=len(sig) > MAX_SIGNATURES))
    meta["truncated"] = meta["output_truncated"] or meta["request_truncated"] or any(f["truncated"] for f in meta["files"])
    text = "\n".join(L)
    for h in hidden:      # provenance guard: no non-trivial line of a hidden file may appear in the context
        for ln in (ws / h).read_text("utf-8", errors="replace").splitlines():
            if len(ln.strip()) >= 12 and ln.strip() in text:
                raise ContextLeak("a line of the hidden file %s reached the proposal context" % h)
    meta["context_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    meta["chars"] = len(text)
    return text, meta


def classify_progress(prev: Optional[Dict[str, Any]], cur: Dict[str, Any], tests_pass: bool) -> str:
    """[A2.4] progress | repetition | regression | changed, from two parsed failures. repetition = same exception, message and test location as the previous observation;
    regression = a failing assertion became a syntax error / uncaught exception (the code no longer even runs the check)."""
    if tests_pass:
        return "progress"
    if prev is None:
        return "changed"
    same = lambda f: (f["kind"], f["exception"], f["message"], (f.get("test_location") or {}).get("file"), (f.get("test_location") or {}).get("line"))      # noqa: E731
    if same(prev) == same(cur):
        return "repetition"
    if prev["kind"] == "assertion" and cur["kind"] in ("syntax", "exception"):
        return "regression"
    return "changed"


def repair_text(k: int, action_text: str, result_text: str, cur: Dict[str, Any], cls: str, blocked_total: int, code_snippet: str) -> str:
    L = ["Repair report (deterministic, step %d):" % k, "- action taken: %s" % action_text[:300], "- its result: %s" % result_text.strip()[:300].replace("\n", " | ")]
    t = cur.get("test_location")
    L.append("- failure now: %s%s%s [%s]" % (cur["exception"] or cur["kind"], (": " + cur["message"]) if cur["message"] else "", (" at %s:%d" % (t["file"], t["line"])) if t else "", cls))
    if cls == "repetition":
        L.append("- the same failure as before: the same action will not be accepted again unless the files or the failure change (%d repeated action(s) blocked so far)" % blocked_total)
    elif cls == "regression":
        L.append("- REGRESSION: the code no longer reaches the failing assertion; undo or fix the last change")
    if code_snippet:
        L.append("- current code at the failure location:\n" + code_snippet)
    return "\n".join(L)


def investigate(run) -> Dict[str, Any]:
    """Run the stage on an AgenticRun after begin(): journal `investigate_sources`, observe the baseline end state (L5 evidence), journal `investigate`, append the report to the next context,
    re-run the router and the tier. -> the structured result (also stored on run.investigation)."""
    ws = run.task.workspace
    sources = inspect_sources(ws, run.task.protected)
    run.note("investigate_sources", {"files": sources})
    files = [s["path"] for s in sources]
    # the baseline run: same sandbox call and evidence as any end-state observation; the location (when there is a failure) is recorded in the evidence outcome detail
    state = run._observe("investigate", detail_fn=lambda r: _detail(r, files, run.task.protected))
    r = run.last_exec
    out = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
    failure = parse_failure(out, int(r["exit_code"]), files, run.task.protected)
    targets = link_assertion(ws, failure, sources)
    res = {"state_id": state["state_id"], "exit_code": int(r["exit_code"]), "failure": failure, "targets": targets, "tests_pass": state["tests_pass"], "label": state["label"]}
    run.note("investigate", {k: res[k] for k in ("state_id", "exit_code", "failure", "targets", "label")})
    text, meta = build_context(run, sources, failure, targets, out)
    run.note("investigate_context", meta)      # [A2.2] what the proposer was shown, with every truncation and every excluded file
    run.messages += [{"role": "user", "content": text}]
    run.route("investigate", state["state_id"])
    run.retier("investigate")
    run.investigation = res
    run.prev_failure = failure
    return res


def _detail(r: Dict[str, Any], files: Sequence[str], protected: Sequence[str]) -> str:
    f = parse_failure((r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or ""), int(r["exit_code"]), files, protected)
    t = f.get("test_location") or f.get("code_location")
    return "" if f["kind"] == "none" else "%s%s" % (f["kind"], (" at %s:%d" % (t["file"], t["line"])) if t else "")
