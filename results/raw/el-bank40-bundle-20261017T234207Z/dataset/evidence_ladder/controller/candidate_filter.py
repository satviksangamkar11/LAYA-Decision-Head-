"""[EL-ARCH-1 F1 / T23-T24] Deterministic candidate filter in front of ranking, implementing EL_ARCH_INTERFACE_FREEZE_v1.md exactly. Pure standard library, no I/O, no clock, no model.

    filter_options(options, files, *, max_options=5, path_scope=None) -> FilterResult

options  list of CandidateIn in generation order: {"tool", "arguments", "origin": {"seed","index"}, "raw"}; a bad item arrives as {"malformed": True, "origin", "raw"} and is rejected alone.
files    {workspace-relative path: current text} (None = Edit applicability cannot be checked; no_op_edit / unchanged_diff still are).
path_scope  None (workspace only: no absolute / '..' / secret paths) | list of allowed workspace-relative path prefixes | {"allow": [prefixes] | None, "protected": [paths]}.
FilterResult = {"accepted": [{"option_id","tool","arguments","action_key","origin"}], "rejected": [{"option_id","origin","rule","reason"}],
                "abstain": None | {"code": "no_options_proposed" | "all_options_rejected", "reason"}, "counts": {"in","accepted","rejected"}}
option_id = sha256 of generation.render_call(tool, arguments) (the text the journal hashes as option_sha256). Accepted options keep input order; nothing is edited, merged or padded to reach k.
Rule codes (closed set): malformed_option unknown_tool bad_arguments path_outside_scope protected_path no_op_edit unchanged_diff old_string_absent old_string_not_unique duplicate
sequential_plan beyond_max_options  (+ repeated_failure, added by the controller).
Mapping notes: `bad_arguments` is only for structurally wrong or empty arguments. The filter does NOT reject by risk class: unknown / network / irreversible actions are accepted and
reach the controller's policy + authorization gate (risk.authorize_action). A missing file counts as `old_string_absent`; secret and outside-workspace paths are `path_outside_scope`; a write to a protected file is `protected_path`.

SEQUENTIAL-PLAN RULE: inside the ordered list, a Read(P) followed later by an Edit of the same path P and, after that Edit, by any Bash is one plan written as alternatives (Read -> Edit -> Bash).
The Read stays; that Edit and every later Bash are rejected `sequential_plan` (they presuppose earlier steps). Read -> Edit without a later Bash is not a plan. The matching prompt is the
versioned template qwen_actions_v2 (generation.ACTION_INSTR_V2); qwen_actions_v1 is never edited."""
from __future__ import annotations

import json
import posixpath
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evidence_ladder.common.interfaces import sha256_text
from evidence_ladder.controller import risk
from evidence_ladder.controller.generation import ACTION_TOOLS, render_call

RULES = ("malformed_option", "unknown_tool", "bad_arguments", "path_outside_scope", "protected_path", "no_op_edit", "unchanged_diff", "old_string_absent", "old_string_not_unique",
         "duplicate", "sequential_plan", "beyond_max_options", "repeated_failure")


def _norm(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/"))


def _scope(path_scope: Any):
    if isinstance(path_scope, Mapping):
        return path_scope.get("allow"), tuple(path_scope.get("protected") or ())
    return path_scope, ()


def action_key(tool: str, args: Mapping[str, Any]) -> str:
    a = dict(args)
    if "file_path" in a:
        a["file_path"] = _norm(a["file_path"])
    return tool + ":" + json.dumps(a, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _check_structure(c: Mapping[str, Any]) -> Optional[tuple]:
    if c.get("malformed") or not isinstance(c.get("tool"), str) or not isinstance(c.get("arguments"), dict):
        return "malformed_option", "item could not be parsed into {tool, arguments}"
    tool, args = c["tool"], c["arguments"]
    if tool not in ACTION_TOOLS:
        return "unknown_tool", "tool %r is not one of %s" % (tool, sorted(ACTION_TOOLS))
    if set(args) != ACTION_TOOLS[tool] or not all(isinstance(v, str) for v in args.values()):
        return "bad_arguments", "arguments of %s are not exactly %s with string values" % (tool, sorted(ACTION_TOOLS[tool]))
    if any(not args[k].strip() for k in ("file_path", "command") if k in args) or (tool == "Edit" and not args["old_string"]):
        return "bad_arguments", "a required argument is empty"
    return None


def _check_scope(tool: str, args: Mapping[str, Any], allow, protected) -> Optional[tuple]:
    if tool == "Bash":      # shell commands are judged by the controller's policy gate (risk.authorize_action), which journals the denial
        return None
    cl = risk.classify({"name": tool, "arguments": dict(args)}, ".", protected)
    cls = cl["class"]
    if cls in (risk.SECRET, risk.OUTSIDE):
        return "path_outside_scope", "%s: %s" % (cls, cl["reason"])
    if cls == risk.PROT:
        return "protected_path", cl["reason"]
    if allow is not None and "file_path" in args:
        p = _norm(args["file_path"])
        if not any(p == _norm(x).rstrip("/") or p.startswith(_norm(x).rstrip("/") + "/") for x in allow):
            return "path_outside_scope", "%s is outside the allowed path scope" % p
    return None


def _check_edit(args: Mapping[str, str], fmap: Optional[Mapping[str, str]]) -> Optional[tuple]:
    old, new = args["old_string"], args["new_string"]
    if old == new:
        return "no_op_edit", "old_string == new_string: the edit changes nothing"
    if old.replace("\r\n", "\n") == new.replace("\r\n", "\n"):
        return "unchanged_diff", "the edit only changes line endings"
    if fmap is None:
        return None
    text = fmap.get(_norm(args["file_path"]))
    if text is None:
        return "old_string_absent", "no such file in the workspace snapshot: %s" % args["file_path"]
    n = text.count(old)
    if n == 0:
        return "old_string_absent", "old_string does not occur in %s" % args["file_path"]
    if n > 1:
        return "old_string_not_unique", "old_string occurs %d times in %s" % (n, args["file_path"])
    if text.replace(old, new, 1) == text:
        return "unchanged_diff", "applying the edit leaves the file unchanged"
    return None


def _plan_steps(valid: List[Dict[str, Any]]) -> Dict[int, str]:
    """valid = structurally valid candidates (with "i" = input position). -> {input position: reason} of the plan steps to reject."""
    out: Dict[int, str] = {}
    for a, c in enumerate(valid):
        if c["tool"] != "Read":
            continue
        p = _norm(c["arguments"]["file_path"])
        j = next((j for j in range(a + 1, len(valid)) if valid[j]["tool"] == "Edit" and _norm(valid[j]["arguments"]["file_path"]) == p), None)
        if j is None or not any(v["tool"] == "Bash" for v in valid[j + 1:]):
            continue
        why = "follows Read %s -> Edit %s in the same list: a plan step, not an independent next action" % (p, p)
        out[valid[j]["i"]] = why
        out.update({v["i"]: why for v in valid[j + 1:] if v["tool"] == "Bash"})
        break
    return out


def filter_options(options: Sequence[Mapping[str, Any]], files: Optional[Mapping[str, str]] = None, *, max_options: int = 5, path_scope: Any = None) -> Dict[str, Any]:
    allow, protected = _scope(path_scope)
    fmap = None if files is None else {_norm(k): v for k, v in files.items()}
    rej: Dict[int, Dict[str, Any]] = {}

    def reject(i: int, rule: str, reason: str, oid: Optional[str]) -> None:
        rej[i] = {"option_id": oid, "origin": dict(options[i].get("origin") or {}), "rule": rule, "reason": reason}

    valid: List[Dict[str, Any]] = []
    for i, c in enumerate(options):
        bad = _check_structure(c)
        if bad:
            raw = c.get("raw")
            reject(i, bad[0], bad[1], sha256_text(raw) if isinstance(raw, str) and raw else None)
        else:
            valid.append(dict(c, i=i, option_id=sha256_text(render_call(c["tool"], c["arguments"]))))
    plan = _plan_steps(valid)
    accepted: List[Dict[str, Any]] = []
    seen = set()
    for c in valid:
        i = c["i"]
        if i in plan:
            reject(i, "sequential_plan", plan[i], c["option_id"])
            continue
        bad = _check_scope(c["tool"], c["arguments"], allow, protected) or (_check_edit(c["arguments"], fmap) if c["tool"] == "Edit" else None)
        if bad:
            reject(i, bad[0], bad[1], c["option_id"])
        elif c["option_id"] in seen:
            reject(i, "duplicate", "identical to an earlier option", c["option_id"])
        elif len(accepted) >= max_options:
            reject(i, "beyond_max_options", "beyond max_options=%d" % max_options, c["option_id"])
        else:
            seen.add(c["option_id"])
            accepted.append({"option_id": c["option_id"], "tool": c["tool"], "arguments": dict(c["arguments"]), "action_key": action_key(c["tool"], c["arguments"]),
                             "origin": dict(c.get("origin") or {})})
    rejected = [rej[i] for i in sorted(rej)]
    abstain = None
    if not accepted:
        if not options:
            abstain = {"code": "no_options_proposed", "reason": "no options proposed"}
        else:
            codes: Dict[str, int] = {}
            for r in rejected:
                codes[r["rule"]] = codes.get(r["rule"], 0) + 1
            abstain = {"code": "all_options_rejected", "reason": "no valid option after filtering: %d of %d options rejected (%s)" % (
                len(rejected), len(options), ", ".join("%s=%d" % kv for kv in sorted(codes.items())))}
    return {"accepted": accepted, "rejected": rejected, "abstain": abstain, "counts": {"in": len(options), "accepted": len(accepted), "rejected": len(rejected)}}
