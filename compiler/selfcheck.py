"""Checks for compiler.state_builder:  uv run python -m compiler.selfcheck

Synthetic twins and injections prove the leakage rules at the code level; the real-data part runs on the 300-row sample if it is on disk.
"""
import copy
import json
import os

from compiler.candidates import classify, distractor_pool, make_candidates, render
from compiler.tiers import patch_files, tier_for
from compiler.state_builder import LeakError, StateOverflow, build_state
from state.schema import NONE_ID


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    assert ok, name


def ev(role, content="", name=None, calls=None):
    return {"content": content, "name": name, "role": role, "tool_call_id": None, "tool_calls": calls}


def call(name, **args):
    return [{"function": {"name": name, "arguments": json.dumps(args)}, "id": "x", "type": "function"}]


def trajectory(future, label="run pytest -q", n=12):
    e = [ev("system", "You are an agent."), ev("user", "Issue: parse() raises IndexError on empty input.")]
    for k in range(n):
        e += [ev("assistant", f"step {k} thinking", calls=call("execute_bash", command=f"ls dir{k}")), ev("tool", f"out {k} " + "x" * 700, name="execute_bash")]
    t = len(e)
    e += [ev("assistant", label, calls=call("execute_bash", command=label))]   # the label event at t
    e += future
    return e, t


FUT_A = [ev("tool", "all tests passed FINAL_PATCH diff --git a/x b/x RESOLVED=true", name="execute_bash"), ev("assistant", "done", calls=call("finish"))]
FUT_B = [ev("tool", "FAILED 3 tests", name="execute_bash"), ev("assistant", "I give up", calls=call("finish"))]
A, t = trajectory(FUT_A)
B, tb = trajectory(FUT_B, label="edit src/parser.py now")
assert t == tb

# 1. twins: same prefix, different label action and different futures -> identical state, hash and cut manifest, with and without cuts
for budget in (50000, 6000, 2500):
    a, b = build_state(A, t, budget), build_state(B, t, budget)
    check(f"twins with different labels and futures give identical state, hash and cut manifest (budget {budget})",
          a["text"] == b["text"] and a["state_hash"] == b["state_hash"] and a["manifest"] == b["manifest"] and a["manifest_hash"] == b["manifest_hash"],
          f"stage {a['manifest']['stage']}, {a['manifest']['dropped_events']} events dropped, {len(a['manifest']['trims'])} trims, {len(a['text'])} chars")

# 2. the label event and the future never enter, and events[t:] is not even read
s = build_state(A, t, 50000)["text"]
check("label event and future are absent from the state", all(w not in s for w in ("run pytest -q", "FINAL_PATCH", "RESOLVED", "I give up", "all tests passed")))
poisoned = A[:t] + [None] * (len(A) - t)
check("events[t:] is never read (replaced by None, same state)", build_state(poisoned, t, 50000)["state_hash"] == build_state(A, t, 50000)["state_hash"])

# 3. injection: anything outside the allowlist is rejected
for name, bad in (("episode field on an event", dict(A[3], resolved=True)), ("patch field on an event", dict(A[3], model_patch="diff")),
                  ("unknown role", dict(A[3], role="oracle")), ("missing key", {k: v for k, v in A[3].items() if k != "name"})):
    E = copy.deepcopy(A); E[3] = bad
    try:
        build_state(E, t, 50000); check(f"injection rejected: {name}", False)
    except LeakError:
        check(f"injection rejected: {name}", True)
E = copy.deepcopy(A); E[1] = ev("assistant", "not an issue")
try:
    build_state(E, t); check("event 1 must be the user's issue", False)
except LeakError:
    check("event 1 must be the user's issue", True)

# 4. determinism and the cut policy
check("same trajectory, t and budget give the same state twice", build_state(A, t, 6000)["state_hash"] == build_state(A, t, 6000)["state_hash"])
full = build_state(A, t, 50000)
check("no cuts when the full render fits", full["manifest"]["stage"] == "none" and not full["manifest"]["trims"])
caps = build_state(A, t, 9000)
check("stage 1 trims long results with a visible marker and a manifest entry",
      caps["manifest"]["stage"] == "caps" and "chars omitted" in caps["text"] and caps["manifest"]["trims"] and len(caps["text"]) <= 9000)
drop = build_state(A, t, 2500)
check("stage 2 drops the oldest events with one marker; the issue and the latest two events stay",
      drop["manifest"]["stage"] in ("drop", "core") and drop["text"].count("earlier events omitted") == 1 and "Issue: parse()" in drop["text"]
      and "ls dir11" in drop["text"] and "ls dir0" not in drop["text"] and len(drop["text"]) <= 2500)
try:
    build_state(A, t, 150); check("overflow raises after every permitted cut", False)
except StateOverflow:
    check("overflow raises after every permitted cut", True)
sizes = [len(build_state(A, t, b)["text"]) for b in (50000, 9000, 6000, 2500)]
check("a smaller budget never gives a longer state", sizes == sorted(sizes, reverse=True), str(sizes))

# 5. real trajectories (300-row sample), if present
path = "data/raw/swerebench-sample-20261001.jsonl"
if os.path.exists(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:60]
    stages, n, bad = {}, 0, 0
    for r in rows:
        evs = r["trajectory"]
        for i in range(2, len(evs), 6):
            if evs[i]["role"] != "assistant":
                continue
            a = build_state(evs, i, 8000)
            b = build_state(evs[:i] + [None] * (len(evs) - i), i, 8000)
            bad += a["state_hash"] != b["state_hash"] or a["manifest_hash"] != b["manifest_hash"]
            stages[a["manifest"]["stage"]] = stages.get(a["manifest"]["stage"], 0) + 1
            n += 1
    check("real trajectories: the state never depends on events[t:] (prefix-only twin)", bad == 0, f"{n} decisions from {len(rows)} rows, stages {stages}")
else:
    print("SKIP  real-data twin test (sample file not on disk)")
# 6. candidates: the pool and the candidate set depend on the state only
pa, pb = distractor_pool(A, t), distractor_pool(B, t)
check("distractor pool is identical for twins with different labels and futures", [p[0] for p in pa] == [p[0] for p in pb] and len(pa) > 3)
check("the pool never reads events[t:]", [p[0] for p in distractor_pool(A[:t] + [None] * (len(A) - t), t)] == [p[0] for p in pa])
ca, cb = make_candidates(A, t), make_candidates(B, t)
check("distractor count and state hash are identical for twins (they depend on the state only)", ca["k_distractors"] == cb["k_distractors"] and ca["state_hash"] == cb["state_hash"])
ta, tb_ = render("execute_bash", {"command": "run pytest -q"})[0], render("execute_bash", {"command": "edit src/parser.py now"})[0]
da = {c.text for c in ca["candidates"]} - {ta, "None of these options, or the evidence is insufficient"}
db = {c.text for c in cb["candidates"]} - {tb_, "None of these options, or the evidence is insufficient"}
check("distractors are drawn from the pool", da <= {p[0] for p in pa} and db <= {p[0] for p in pb})
check("the true action appears exactly once and NONE is last", sum(c.text == ta for c in ca["candidates"]) == 1 and ca["candidates"][-1].candidate_id == NONE_ID
      and ca["candidates"][ca["label_index"]].text == ta)
true_c = ca["candidates"][ca["label_index"]]
dis_c = next(c for i, c in enumerate(ca["candidates"][:-1]) if i != ca["label_index"])
check("true and distractor share one rendering and one metadata shape; only the source and executed flags differ, and outcomes stay empty",
      len(true_c.head_meta()) == len(dis_c.head_meta()) and true_c.source == "TRAJECTORY" and dis_c.source == "COUNTERFACTUAL"
      and true_c.executed and not dis_c.executed and true_c.outcome_if_executed is None and dis_c.outcome_if_executed is None)
check("action kinds", [classify("execute_bash", {"command": c}) for c in ("pytest -q tests/", "ls -la", "pip install -e .")] == ["run_tests", "explore", "run_other"]
      and classify("str_replace_editor", {"command": "view"}) == "view" and classify("str_replace_editor", {"command": "create"}) == "edit")

if os.path.exists(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:100]
    n = bad = short = 0
    pos, kinds, longest, in_prefix, ks = [], {}, 0, 0, []
    for r in rows:
        evs = r["trajectory"]
        for i in range(2, len(evs), 3):
            if evs[i]["role"] != "assistant" or not evs[i]["tool_calls"]:
                continue
            m = make_candidates(evs, i)
            if m is None:
                short += 1
                continue
            m2 = make_candidates(evs[:i] + [evs[i]] + [None] * (len(evs) - i - 1), i)
            bad += [c.text for c in m["candidates"]] != [c.text for c in m2["candidates"]]
            cs = m["candidates"][:-1]
            pos.append((len(cs), m["label_index"]))
            kinds[m["true_kind"]] = kinds.get(m["true_kind"], 0) + 1
            longest += len(cs[m["label_index"]].text) == max(len(c.text) for c in cs)
            in_prefix += cs[m["label_index"]].text in {p[0] for p in distractor_pool(evs, i)} or False
            ks.append(len(cs))
            n += 1
    # exact test per option count (a normalised-quarter binning was invalid: with 3 options the second quarter is unreachable)
    from collections import Counter
    chi = df = 0.0
    for size in sorted({s for s, _ in pos}):
        obs = Counter(p for s, p in pos if s == size)
        cnt = sum(obs.values())
        if cnt >= 5 * size:
            chi += sum((obs.get(p, 0) - cnt / size) ** 2 / (cnt / size) for p in range(size))
            df += size - 1
    limit = df + 3 * (2 * df) ** 0.5
    check("real data: candidate sets never depend on events after t (only the true event is read)", bad == 0, f"{n} decisions, {short} skipped (pool too small)")
    check("real data: the true action's position is uniform within each option count (pooled chi-square below df + 3 sd)", chi < limit, f"chi2 {chi:.1f}, df {df:.0f}, limit {limit:.1f}")
    print(f"INFO  true action is the longest candidate text in {longest / n:.1%} of decisions (chance about {sum(1 / k for k in ks) / n:.1%}); true-action kinds {dict(sorted(kinds.items()))}")
    print(f"INFO  true action text already in the distractor pool (a stale repeat): {in_prefix / n:.1%}; candidates per decision: min {min(ks)}, median {sorted(ks)[len(ks) // 2]}, max {max(ks)}")
# 8. split: depends on the repository only; owner families stay together in OOD; no lineage crosses a split
from compiler import split
check("split depends on the repository string only and is deterministic", split.split_of("a/b") == split.split_of("a/b") and split.split_of("a/b") in ("train", "cal", "test", "ood"))
fam = [r for r in ("zzz/a", "zzz/b", "zzz/c", "zzz/d") if split.split_of(r) == "ood"]
check("an OOD owner family is OOD for every repository it owns", len(fam) in (0, 4))
if os.path.exists(path):
    srows = [json.loads(l) for l in open(path, encoding="utf-8")]
    sm = split.manifest(srows)
    sizes = {s: sum(1 for v in sm["repos"].values() if v == s) for s in ("train", "cal", "test", "ood")}
    check("manifest: every repository in exactly one split, lineage and OOD-family checks hold", sum(sizes.values()) == len(sm["repos"]), f"{sizes}, sha256 {sm['sha256'][:12]}")

# 7. credit tiers (label side)
PATCH = "diff --git a/src/parser.py b/src/parser.py\n--- a/src/parser.py\n+++ b/src/parser.py\n@@ -1 +1 @@\n-x\n+y\n"
files = patch_files(PATCH)
check("patch_files reads the diff headers", files == {"src/parser.py"})
def traj(action, result, later=()):
    e = [ev("system", "s"), ev("user", "issue"), ev("assistant", "", calls=action), ev("tool", result, name="x")]
    return e + list(later)
EDIT = call("str_replace_editor", command="str_replace", path="/workspace/repo__x/src/parser.py", old_str="a", new_str="b")
OK = "The file /workspace/repo__x/src/parser.py has been edited."
def tr(evs, resolved, f=files): return tier_for(evs, 2, resolved, f)[0]
check("STRONG: edit of a patch file in a resolved trajectory", tr(traj(EDIT, OK), True) == "STRONG")
check("WEAK: edit of a non-patch file in a resolved trajectory", tr(traj(call("str_replace_editor", command="str_replace", path="/workspace/repo__x/docs/a.md"), OK), True) == "WEAK")
check("UNKNOWN: any action in an unresolved trajectory, and no negatives are invented", tr(traj(EDIT, OK), False) == "UNKNOWN" and tr(traj(call("execute_bash", command="ls"), "ok\n[Command finished with exit code 0]"), False) == "UNKNOWN")
check("NEGATIVE: editor ERROR, even in a resolved trajectory", tr(traj(EDIT, "ERROR:\nNo replacement was performed"), True) == "NEGATIVE")
check("NEGATIVE: shell command not found", tr(traj(call("execute_bash", command="pytestt -q"), "bash: pytestt: command not found\n[Command finished with exit code 127]"), True) == "NEGATIVE")
check("a failing test run is NOT negative", tr(traj(call("execute_bash", command="pytest -q"), "1 failed\n[Command finished with exit code 1]"), True) == "WEAK")
undo = [ev("assistant", "", calls=call("str_replace_editor", command="undo_edit", path="/workspace/repo__x/src/parser.py")), ev("tool", "undone", name="x")]
check("NEGATIVE: the edit was undone later", tr(traj(EDIT, OK, undo), True) == "NEGATIVE")
view = call("str_replace_editor", command="view", path="/workspace/repo__x/src/parser.py")
later_edit = [ev("assistant", "", calls=EDIT), ev("tool", OK, name="x")]
check("WEAK with a reason: a read of a patch file before its edit", tier_for(traj(view, "file text", later_edit), 2, True, files) == ("WEAK", "read of a patch file before its edit"))
if os.path.exists(path):
    import collections
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    cnt, by_res = collections.Counter(), collections.Counter()
    for r in rows:
        pf = patch_files(r["model_patch"])
        for i, e in enumerate(r["trajectory"]):
            if i >= 2 and e["role"] == "assistant" and e["tool_calls"]:
                tname = tier_for(r["trajectory"], i, bool(r["resolved"]), pf)[0]
                cnt[tname] += 1; by_res[(bool(r["resolved"]), tname)] += 1
    tot = sum(cnt.values())
    print(f"INFO  tiers over all {tot} decisions of the 300 rows: " + ", ".join(f"{k} {v} ({v / tot:.1%})" for k, v in sorted(cnt.items())))
    res_tot = sum(v for (rz, _), v in by_res.items() if rz)
    print(f"INFO  inside resolved trajectories ({res_tot} decisions): STRONG {by_res[(True, 'STRONG')] / res_tot:.1%}, WEAK {by_res[(True, 'WEAK')] / res_tot:.1%}, NEGATIVE {by_res[(True, 'NEGATIVE')] / res_tot:.1%}")
print("ALL CHECKS PASSED")
