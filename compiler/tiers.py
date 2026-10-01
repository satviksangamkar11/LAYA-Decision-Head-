"""Label evidence tiers for ACTION decisions (A4 leakage rules v1 and v1.1). Label side only: nothing here may feed a state or a candidate.

STRONG    an edit or create on a file named in the final patch, in a trajectory with resolved == 1, and the edit did not error.
WEAK      a read of a patch file before its first edit, or any other action in a resolved trajectory. Not assumed correct.
UNKNOWN   no usable evidence (every action in an unresolved trajectory). Absence of success is not failure: no negative is created.
NEGATIVE  contradicting evidence about the CHOSEN action itself: the editor returned ERROR, the shell said command not found (126 or 127), or
          the agent undid the edit later. It gives no positive label and does not make any other candidate correct. A test run that fails
          (exit 1) is normal, informative behaviour and is not negative.
"""
import json
import re

from compiler.candidates import classify, first_call, norm_path

EXIT_RE = re.compile(r"\[Command finished with exit code (-?\d+)\]")
DIFF_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)


def patch_files(model_patch):
    return {m.group(2) for m in DIFF_RE.finditer(model_patch or "")}


def _in_patch(path, files):
    return bool(path) and any(path == f or path.endswith("/" + f) for f in files)


def _path(args):
    return norm_path(args.get("path"))


def tier_for(events, t, resolved, files):
    """(tier, reason) for the action at events[t]; the result is events[t+1] when it exists."""
    tool, args = first_call(events[t])
    kind = classify(tool, args)
    res = events[t + 1]["content"] if t + 1 < len(events) and events[t + 1]["role"] == "tool" else ""
    res = res or ""
    path = _path(args)
    if tool == "str_replace_editor" and res.lstrip().startswith("ERROR"):
        return "NEGATIVE", "the editor returned ERROR"
    if tool == "execute_bash":
        m = EXIT_RE.findall(res)
        if m and int(m[-1]) in (126, 127):
            return "NEGATIVE", f"shell exit code {m[-1]} (command not found or not executable)"
    if kind == "edit" and path:
        for e in events[t + 1:]:
            c = first_call(e) if e["role"] == "assistant" else None
            if c and c[0] == "str_replace_editor" and c[1].get("command") == "undo_edit" and _path(c[1]) == path:
                return "NEGATIVE", "the edit was undone later"
    if not resolved:
        return "UNKNOWN", "unresolved trajectory"
    if kind == "edit" and _in_patch(path, files):
        return "STRONG", "edit of a file in the final patch, trajectory resolved"
    if kind == "view" and _in_patch(path, files):
        later = any((c := first_call(e)) and classify(*c) == "edit" and _path(c[1]) == path for e in events[t + 1:] if e["role"] == "assistant")
        if later:
            return "WEAK", "read of a patch file before its edit"
    return "WEAK", "action in a resolved trajectory"
