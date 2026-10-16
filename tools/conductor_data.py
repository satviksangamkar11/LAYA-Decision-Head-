"""Builds Laya-conductor fine-tune rows (`should_continue`) from nebius/SWE-rebench-openhands-trajectories (PLAN.md 13.7).

Row format = the official conductor notebook's: {"state": {...}, "gold": {"should_continue": {"probabilities": {"false": p, "true": q}}}, ...meta}.
`should_continue` = P(agent stopped mid-task with work left) = the plan's `done` label negated: FALSE only at the final step of a resolved
trajectory; TRUE at every earlier step and at the final "finish" of an unresolved one.

Why last_observation: at the final step the response ("the fix is complete...") reads alike whether the run later resolved or not, so text alone cannot
separate a resolved finish from a premature one. The tool result just before the step (observable at t, never the step itself or later) is the evidence.

laya.common.build_sequence serializes the state with json.dumps and keeps the HEAD of it (truncate_left=False), so a long field silently
pushes later fields out. Hence: last_response first, all fields cut to fixed char budgets (tuned by tools/check_conductor_rows.py).
"""
import random

TASK_CHARS = 600        # head of the first user message (the issue)
RESP_CHARS = 1000       # TAIL of the assistant message (its conclusion / last action)
OBS_CHARS = 900         # TAIL of the last tool result before this step (test output etc.): the only execution evidence the state carries
K_MID = 2               # random intermediate steps per trajectory, plus the last and the one before it


def _text(m):
    t = m.get("content") or ""
    for c in m.get("tool_calls") or []:
        f = c.get("function", {})
        t += f"\n[tool {f.get('name')}] {f.get('arguments') or ''}"
    return t


def _unwrap(c):
    """SWE-smith stores observations as the string repr of [{'type': 'text', 'text': ...}]; return the text."""
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    if isinstance(c, str) and c.startswith("[{'type'"):
        import ast
        try:
            return "".join(x.get("text", "") for x in ast.literal_eval(c))
        except Exception:
            return c
    return c or ""


def _repo_of(instance_id):
    """SWE-smith ids look like owner__repo.<hash>.<bug>; the repo is what the split must key on."""
    return instance_id.split(".")[0].replace("__", "/", 1)


SOURCES = {   # source -> (outcome column, repo column or None)
    "nebius": ("resolved", "repo"),
    "swesmith_tool": ("resolved", None),
    "jetbrains_mini": ("resolved", None),
    "nanoswe": ("verified", None),
}


def adapt(rec, source):
    """One raw dataset row -> the common form make_rows reads: {trajectory, resolved, trajectory_id, repo, source}.
    Common form: roles user (first = the task), assistant (one per step, tool calls as `tool_calls` or an `action` string), tool (what came back)."""
    import json as _j
    outcome, repo_col = SOURCES[source]
    if source == "nebius":
        return {**rec, "source": source}
    msgs = rec["messages"]
    if isinstance(msgs, str):
        msgs = _j.loads(msgs)
    if source == "swesmith_tool":
        out = [{"role": m["role"], "content": _unwrap(m.get("content")) if m["role"] != "assistant" else (m.get("content") or ""),
                "tool_calls": [{"function": {"name": "action", "arguments": m["action"]}}] if m["role"] == "assistant" and m.get("action") else None}
               for m in msgs if m["role"] != "system"]
    elif source == "jetbrains_mini":    # mini-swe-agent: assistant THOUGHT + bash block, command output comes back as a user message
        seen_user, out = False, []
        for m in msgs:
            if m["role"] == "system":
                continue
            role = m["role"]
            if role == "user":
                role, seen_user = ("user" if not seen_user else "tool"), True
            out.append({"role": role, "content": m["content"], "tool_calls": None})
    else:                                # nanoswe: OpenAI-style, unverified beyond role=user first; unknown roles fail loudly in make_rows' caller
        out = [{"role": m["role"], "content": m.get("content") or "", "tool_calls": m.get("tool_calls")} for m in msgs if m["role"] != "system"]
        assert {m["role"] for m in out} <= {"user", "assistant", "tool"}, {m["role"] for m in out}
    iid = rec["instance_id"]
    return {"trajectory": out, "resolved": bool(rec[outcome]), "trajectory_id": rec.get("traj_id") or rec.get("traj_hash") or f"{source}:{iid}",
            "repo": rec[repo_col] if repo_col else _repo_of(iid), "instance_id": iid, "source": source}


def make_rows(rec, rng, split):
    """rec: one dataset row (trajectory list of dicts). Returns a list of training rows for this trajectory."""
    msgs = rec["trajectory"]
    task = next((m.get("content") or "" for m in msgs if m["role"] == "user"), "")[:TASK_CHARS]
    steps = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if len(steps) < 3:
        return []
    last = len(steps) - 1
    pick = {last, last - 1} | set(rng.sample(range(0, last - 1), min(K_MID, last - 1)))
    resolved = int(rec["resolved"])
    rows = []
    for s in sorted(pick):
        stopped_with_work_left = not (s == last and resolved == 1)
        p_true = 1.0 if stopped_with_work_left else 0.0
        obs = next((m.get("content") or "" for m in reversed(msgs[:steps[s]]) if m["role"] == "tool"), "")
        rows.append({
            "state": {"last_response": _text(msgs[steps[s]])[-RESP_CHARS:], "last_observation": obs[-OBS_CHARS:], "task": task},
            "gold": {"should_continue": {"probabilities": {"false": 1.0 - p_true, "true": p_true}}},
            "trajectory_id": rec["trajectory_id"], "repo": rec["repo"], "source": rec.get("source", "nebius"), "split": split,
            "step": s, "n_steps": len(steps), "is_last": s == last, "resolved": resolved,
            # plan targets kept separate (PLAN.md 13.3): the conductor trains on should_continue only; `done` and `value` are stored for the internal head and are never merged
            "done": int(s == last and resolved == 1), "value": resolved,
        })
    return rows


def demo():
    rng = random.Random(0)
    msg = lambda r, c: {"role": r, "content": c}
    traj = [msg("system", "s"), msg("user", "fix bug")] + [msg("assistant", f"step {i}") for i in range(6)]
    for resolved in (1, 0):
        rows = make_rows({"trajectory": traj, "resolved": resolved, "trajectory_id": "t", "repo": "r"}, rng, "train")
        by_last = {r["step"]: r["gold"]["should_continue"]["probabilities"]["true"] for r in rows}
        assert by_last[5] == (0.0 if resolved else 1.0), by_last          # final step: FALSE only when resolved
        assert all(v == 1.0 for s, v in by_last.items() if s != 5), by_last  # every earlier step: TRUE
        assert {4, 5} <= set(by_last)
    assert all("last_observation" in r["state"] for r in rows)
    t2 = traj[:3] + [{"role": "tool", "content": "3 passed"}] + traj[3:]
    r2 = make_rows({"trajectory": t2, "resolved": 1, "trajectory_id": "t", "repo": "r"}, rng, "train")
    assert any(r["state"]["last_observation"] == "3 passed" for r in r2) and r2[0]["state"]["last_observation"] in ("", "3 passed")
    assert make_rows({"trajectory": traj[:4], "resolved": 1, "trajectory_id": "t", "repo": "r"}, rng, "train") == []
    sm = {"instance_id": "owner__repo.abc.bug_1", "resolved": True, "traj_id": "x", "messages": [{"role": "system", "content": "s"},
          {"role": "user", "content": "[{'type': 'text', 'text': 'fix it'}]"}] + sum([[{"role": "assistant", "content": f"t{i}", "action": f"ls {i}"},
          {"role": "tool", "content": "[{'type': 'text', 'text': 'out" + str(i) + "'}]"}] for i in range(4)], [])}
    a = adapt(sm, "swesmith_tool")
    assert a["repo"] == "owner/repo" and a["trajectory"][0]["content"] == "fix it" and a["trajectory"][2]["content"] == "out0"
    r = make_rows(a, rng, "train"); assert r and r[0]["source"] == "swesmith_tool" and r[-1]["state"]["last_observation"].startswith("out")
    jb = {"instance_id": "o__r.h.b", "resolved": False, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}] +
          sum([[{"role": "assistant", "content": f"THOUGHT {i}"}, {"role": "user", "content": f"<returncode>0</returncode>{i}"}] for i in range(4)], [])}
    a = adapt(jb, "jetbrains_mini"); assert [m["role"] for m in a["trajectory"][:4]] == ["user", "assistant", "tool", "assistant"], a["trajectory"][:4]
    assert make_rows(a, rng, "train")[-1]["gold"]["should_continue"]["probabilities"]["true"] == 1.0     # unresolved: final is TRUE
    print("conductor_data demo ok")


if __name__ == "__main__":
    demo()
