"""State builder for ACTION decisions (A4 leakage rules v1 and v1.1).

build_state(events, t, budget_chars) renders the state a model may see at decision point t from events[1:t] ONLY. It never reads events[t:]:
the event at t is the label (its preamble states the intent) and everything after it is the future. The system message (index 0) is constant
text and is not rendered. Event dicts must have exactly the keys content, name, role, tool_call_id, tool_calls; anything else raises LeakError.
No episode field (resolved, patch, exit_status, ...) can enter, because nothing but the event list is accepted.

Budget in characters (tokenizer-independent; a per-backbone check converts to tokens later). If the full render does not fit, cuts are applied in
this fixed order, each declared, each leaving a visible marker, each depending only on (prefix, t, budget) and never on the label:
  1. stepwise caps on every event except the issue and the latest PROTECTED_TAIL events: CAPS = 2000, 800, 300 chars (head + tail kept);
     the latest events use TAIL_CAP while they are protected;
  2. whole oldest events dropped (after the issue, before the protected tail), one marker for all of them;
  3. the issue trimmed to ISSUE_CAP, then the protected tail to TAIL_CAP_MIN;
  4. StateOverflow is raised. Events are never removed because they look irrelevant to the action or the outcome.
The manifest records exactly what was cut; the twin tests require it to be identical for trajectories that share a prefix.
"""
import hashlib
import json

BUILDER_VERSION = "1"
EVENT_KEYS = {"content", "name", "role", "tool_call_id", "tool_calls"}
ROLES = {"system", "user", "assistant", "tool"}
CAPS = (2000, 800, 300)
PROTECTED_TAIL = 2
TAIL_CAP, TAIL_CAP_MIN, ISSUE_CAP = 4000, 1000, 3000


class LeakError(ValueError):
    """An event carries a field or role the allowlist does not know."""


class StateOverflow(ValueError):
    """The protected core does not fit the budget after every permitted cut."""


def _check(e, i):
    if not isinstance(e, dict) or set(e) != EVENT_KEYS:
        raise LeakError(f"event {i}: keys {sorted(e) if isinstance(e, dict) else type(e).__name__} are not exactly {sorted(EVENT_KEYS)}")
    if e["role"] not in ROLES:
        raise LeakError(f"event {i}: unknown role {e['role']!r}")


def _trim(s, cap, where, trims):
    s = s or ""
    if len(s) <= cap:
        return s
    head = cap // 2
    removed = len(s) - cap
    trims.append({"event": where[0], "part": where[1], "removed_chars": removed})
    return s[:head] + f"[... {removed} chars omitted ...]" + s[len(s) - (cap - head):]


def _args(call):
    a = (call.get("function") or {}).get("arguments")
    try:
        return json.dumps(json.loads(a), sort_keys=True, ensure_ascii=False) if isinstance(a, str) else json.dumps(a, sort_keys=True, ensure_ascii=False)
    except (ValueError, TypeError):
        return str(a)


def _render(e, i, cap, trims):
    role = e["role"]
    if role == "user":
        return "USER: " + _trim(e["content"], cap, (i, "content"), trims)
    if role == "tool":
        return f"RESULT({e['name'] or ''}): " + _trim(e["content"], cap, (i, "content"), trims)
    parts = ["ASSISTANT: " + _trim(e["content"], cap, (i, "content"), trims)]
    for k, c in enumerate(e["tool_calls"] or []):
        parts.append(f"CALL {(c.get('function') or {}).get('name', '')} " + _trim(_args(c), cap, (i, f"call{k}"), trims))
    return "\n".join(parts)


def _assemble(events, idx, caps_for, drop_k):
    """idx: indices of events[1:t]; caps_for(i) -> cap for event i; the first dropped drop_k non-issue events become one marker."""
    trims, out = [], []
    keep = idx[:1] + idx[1 + drop_k:]
    for n, i in enumerate(keep):
        out.append(_render(events[i], i, caps_for(i), trims))
        if n == 0 and drop_k:
            out.append(f"[... {drop_k} earlier events omitted ...]")
    return "\n\n".join(out), trims


def build_state(events, t, budget_chars=8000):
    if not 2 <= t <= len(events):
        raise ValueError(f"t={t} out of range for {len(events)} events")
    _check(events[0], 0)
    idx = list(range(1, t))
    for i in idx:
        _check(events[i], i)
    if events[1]["role"] != "user":
        raise LeakError("event 1 must be the user's issue")
    tail = set(idx[-PROTECTED_TAIL:]) if len(idx) > 1 else set()
    manifest = {"t": t, "budget_chars": budget_chars, "stage": "none", "cap": None, "dropped_events": 0, "trims": []}

    def done(text, trims, stage, cap=None, dropped=0):
        manifest.update(stage=stage, cap=cap, dropped_events=dropped, trims=trims, chars=len(text))
        mj = json.dumps(manifest, sort_keys=True)
        return {"text": text, "manifest": manifest, "state_hash": hashlib.sha256(text.encode()).hexdigest(),
                "manifest_hash": hashlib.sha256(mj.encode()).hexdigest(), "builder_version": BUILDER_VERSION}

    big = 10 ** 9
    text, trims = _assemble(events, idx, lambda i: big, 0)
    if len(text) <= budget_chars:
        return done(text, trims, "none")
    for cap in CAPS:                                            # stage 1
        text, trims = _assemble(events, idx, lambda i, c=cap: TAIL_CAP if i in tail else (big if i == idx[0] else c), 0)
        if len(text) <= budget_chars:
            return done(text, trims, "caps", cap)
    cap = CAPS[-1]
    for k in range(1, len(idx) - len(tail)):                    # stage 2: drop oldest whole events (the issue and the tail stay)
        text, trims = _assemble(events, idx, lambda i: TAIL_CAP if i in tail else (big if i == idx[0] else cap), k)
        if len(text) <= budget_chars:
            return done(text, trims, "drop", cap, k)
    k = max(0, len(idx) - len(tail) - 1)                        # stage 3: issue and tail trimmed too
    text, trims = _assemble(events, idx, lambda i: TAIL_CAP_MIN if i in tail else (ISSUE_CAP if i == idx[0] else cap), k)
    if len(text) <= budget_chars:
        return done(text, trims, "core", cap, k)
    raise StateOverflow(f"t={t}: the protected core needs {len(text)} chars, budget {budget_chars}")
