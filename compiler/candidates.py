"""Action normalisation and the distractor generator (A4 leakage rules v1.1, rule 4).

Every candidate, true or not, goes through render() so that "looks synthetic" is not a feature. Candidate text names the action and its target
(tool, command, path, shell command) and never the content of an edit or a thought. The distractor pool is a deterministic function of
events[:t] only; the count, selection and order depend on the state hash only. The true action is added afterwards; if its text is already in
the pool it is removed from the pool, so it appears once.
"""
import json
import random
import re

from compiler.state_builder import build_state
from state.schema import Candidate, with_none

KINDS = ("view", "edit", "run_tests", "explore", "run_other", "think", "plan", "finish")
TEST_RE = re.compile(r"\b(pytest|py\.test|tox|nosetests|unittest|npm (run )?test|yarn test|go test|cargo test|mvn test|make test)\b")
EXPLORE_RE = re.compile(r"^\s*(cd [^&;]+(&&|;)\s*)?(ls|find|grep|egrep|rg|cat|head|tail|sed -n|git (status|diff|log|show|grep|blame)|tree|wc|file|which|pwd|echo|python -c|pip (list|show)|env|printenv)\b")
FILE_RE = re.compile(r"(?:/workspace/)?[\w.\-]+/[\w./\-]+\.(?:py|js|ts|c|cc|cpp|h|go|rs|java|rb|md|txt|toml|cfg|ini|ya?ml|json)")
SAFETY = {"view": "READ_ONLY", "explore": "READ_ONLY", "think": "READ_ONLY", "plan": "READ_ONLY", "finish": "READ_ONLY",
          "edit": "WRITE_LOCAL", "run_tests": "EXEC", "run_other": "EXEC"}


def first_call(event):
    """(tool name, args dict) of the first tool call of an assistant event, or None for a plain message."""
    calls = event.get("tool_calls") or []
    if not calls:
        return None
    f = calls[0].get("function") or {}
    try:
        args = json.loads(f.get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {}
    return f.get("name") or "", args if isinstance(args, dict) else {}


def norm_path(p):
    return re.sub(r"^/workspace/", "", str(p or ""))


def classify(tool, args):
    """The action kind (the NEXT_ACTION_TYPE label), derived deterministically from the tool and its arguments."""
    if tool == "str_replace_editor":
        return "view" if args.get("command") == "view" else "edit"
    if tool == "execute_bash":
        cmd = str(args.get("command", ""))
        return "run_tests" if TEST_RE.search(cmd) else "explore" if EXPLORE_RE.search(cmd) else "run_other"
    return {"think": "think", "task_tracker": "plan", "finish": "finish"}.get(tool, "run_other")


def render(tool, args):
    kind = classify(tool, args)
    if tool == "str_replace_editor":
        return f"{args.get('command', '')} {norm_path(args.get('path'))}".strip(), kind
    if tool == "execute_bash":
        return "bash: " + " ".join(str(args.get("command", "")).split())[:200], kind
    return {"think": "think", "task_tracker": "task_tracker", "finish": "finish"}.get(tool, tool), kind


def to_candidate(cid, text, kind, tool, args, true):
    path = norm_path(args.get("path")) if tool == "str_replace_editor" else ""
    return Candidate(cid, text, action_type="edit" if kind == "edit" else "finish" if kind == "finish" else "tool_call", tool=tool,
                     target={"file": path} if path else {}, arguments={"cmd": args["command"]} if tool == "execute_bash" and "command" in args else {},
                     source="TRAJECTORY" if true else "COUNTERFACTUAL", executable=True, safety_class=SAFETY[kind], executed=true)


def distractor_pool(events, t):
    """Ordered, de-duplicated list of (text, kind, tool, args) built from events[:t] only: earlier actions (stale repeats, most recent first),
    then view and edit of every file seen in the prefix, a test run, and the generic think, plan and finish actions."""
    pool, seen = [], set()

    def add(tool, args):
        text, kind = render(tool, args)
        if text not in seen:
            seen.add(text)
            pool.append((text, kind, tool, args))

    prefix = events[1:t]
    for e in reversed(prefix):
        c = first_call(e) if e["role"] == "assistant" else None
        if c:
            add(*c)
    files, last_test = [], None
    for e in prefix:
        blob = (e["content"] or "") + " " + " ".join(json.dumps(c.get("function", {}).get("arguments", "")) for c in (e["tool_calls"] or []))
        for f in FILE_RE.findall(blob):
            f = norm_path(f)
            if f not in files:
                files.append(f)
        c = first_call(e) if e["role"] == "assistant" else None
        if c and classify(*c) == "run_tests":
            last_test = c[1]
    for f in files[:8]:
        add("str_replace_editor", {"command": "view", "path": f})
        add("str_replace_editor", {"command": "str_replace", "path": f})
    add("execute_bash", last_test or {"command": "pytest -q"})
    add("think", {})
    add("task_tracker", {})
    add("finish", {})
    return pool


def make_candidates(events, t, budget_chars=32000):
    """Candidate set for the ACTION decision at t, with the true action from events[t]. Returns None if the pool is too small to build >= 2 options."""
    true = first_call(events[t])
    if true is None:
        return None
    state = build_state(events, t, budget_chars)
    seed = int(state["state_hash"][:16], 16)
    rng = random.Random(seed)
    ttext, tkind = render(*true)
    full_pool = distractor_pool(events, t)
    covered = any(p[0] == ttext for p in full_pool)   # could a deployable, state-only generator have proposed the true action?
    pool = [p for p in full_pool if p[0] != ttext]
    k = min(2 + seed % 12, len(pool))
    if k < 1:
        return None
    chosen = rng.sample(pool, k)
    items = sorted(chosen + [(ttext, tkind, true[0], true[1])], key=lambda p: p[0])   # canonical order first: position must not reveal the label
    rng.shuffle(items)
    cands = [to_candidate(f"c{i}", x[0], x[1], x[2], x[3], x[0] == ttext) for i, x in enumerate(items)]
    return {"candidates": with_none(cands), "label_index": next(i for i, x in enumerate(items) if x[0] == ttext), "true_kind": tkind,
            "k_distractors": k, "pool_size": len(pool), "candidate_coverage": covered, "state_hash": state["state_hash"],
            "manifest_hash": state["manifest_hash"], "cut_stage": state["manifest"]["stage"]}

