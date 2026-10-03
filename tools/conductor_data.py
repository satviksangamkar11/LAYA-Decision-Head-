"""Builds Laya-conductor fine-tune rows (`should_continue`) from nebius/SWE-rebench-openhands-trajectories (PLAN.md 13.7).

Row format = the official conductor notebook's: {"state": {...}, "gold": {"should_continue": {"probabilities": {"false": p, "true": q}}}, ...meta}.
`should_continue` = P(agent stopped mid-task with work left) = the plan's `done` label negated: FALSE only at the final step of a resolved
trajectory; TRUE at every earlier step and at the final "finish" of an unresolved one.

laya.common.build_sequence serializes the state with json.dumps and keeps the HEAD of it (truncate_left=False), so a long field silently
pushes later fields out. Hence: last_response first, both fields cut to fixed char budgets (tuned by tools/check_conductor_rows.py).
"""
import random

TASK_CHARS = 900        # head of the first user message (the issue)
RESP_CHARS = 1600       # TAIL of the assistant message (its conclusion / last action)
K_MID = 2               # random intermediate steps per trajectory, plus the last and the one before it


def _text(m):
    t = m.get("content") or ""
    for c in m.get("tool_calls") or []:
        f = c.get("function", {})
        t += f"\n[tool {f.get('name')}] {f.get('arguments') or ''}"
    return t


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
        rows.append({
            "state": {"last_response": _text(msgs[steps[s]])[-RESP_CHARS:], "task": task},
            "gold": {"should_continue": {"probabilities": {"false": 1.0 - p_true, "true": p_true}}},
            "trajectory_id": rec["trajectory_id"], "repo": rec["repo"], "split": split,
            "step": s, "n_steps": len(steps), "is_last": s == last, "resolved": resolved,
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
    assert make_rows({"trajectory": traj[:4], "resolved": 1, "trajectory_id": "t", "repo": "r"}, rng, "train") == []
    print("conductor_data demo ok")


if __name__ == "__main__":
    demo()
