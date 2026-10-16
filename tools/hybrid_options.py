"""Evidence-tiered REAL option pools per state (design: PLAN.md section 15). CPU only, stdlib only; nothing is executed or invented.
A "state" is a shared action-prefix of several runs of the SAME instance (observations are not compared: approximation). At each such state:
  GOLD  the real action of a run that resolved the task from this state
  T1    real action of a failing-only branch, executed from this state, followed by an error / revert / same-command retry (weak evidence; an error after an action does not prove it caused the failure)
  T2    real action of a failing-only branch, executed from this state, with no such sign
  T3    real sibling, outcome unknown, never called wrong: another resolved branch's action, or an action from nearby steps of the gold run (executed_here=False)
  NONE  always present, never a label
`soft_target` is a fixed evidence prior (TIER_W, not tuned, not an outcome). Provenance and tier are stored per option; executed_here is True only for branches that really ran from this state.
    python tools/hybrid_options.py selftest | counts SAMPLE.jsonl
"""
import hashlib, json, re, sys
from collections import Counter, defaultdict

TIER_W = {"GOLD": 1.0, "T1": 0.02, "T2": 0.10, "T3": 0.25}
ACT_CHARS = 400
ERR = re.compile(r"exit code (?!0\b)\d+|Traceback \(most recent|\bError\b|FAILED|command not found|No such file")
REV = re.compile(r"git (checkout|stash|reset|restore)|undo_edit")
PATH = re.compile(r"[\w./-]+\.(?:py|js|ts|md|txt|toml|cfg|json|ya?ml|rs|go|java|c|h|cpp)")


def norm(a):
    t = a.split()
    p = PATH.findall(a)
    return (t[0] if t else "", t[1] if len(t) > 1 and not t[1].startswith("{") else "", p[0].split("/")[-1] if p else "")


def _jl(x):
    try:
        return json.loads(x) if isinstance(x, str) else x
    except Exception:
        return None


FUNC = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.S)


def _xml_act(c):
    """SWE-smith rows without `action` carry the call inside the message text as <function=NAME><parameter=K>V</parameter></function>."""
    m = FUNC.search(c or "")
    if not m:
        return ""
    name, p = m.group(1), dict(PARAM.findall(m.group(2)))
    if name == "bash":
        return "bash " + p.get("command", "")
    if name == "str_replace_editor" and "command" in p:
        cmd, path = p.pop("command"), p.pop("path", "")
        return ("str_replace_editor %s %s %s" % (cmd, path, json.dumps(p, ensure_ascii=False) if p else "")).strip()
    return "%s %s" % (name, json.dumps(p, ensure_ascii=False))


def _act(m):
    if isinstance(m.get("action"), str) and m["action"].strip():
        return m["action"].strip()
    tc = m.get("tool_calls") or []
    if tc:
        f = tc[0].get("function", tc[0])
        return "%s %s" % (f.get("name"), f.get("arguments"))
    return _xml_act(_txt(m))


def _txt(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


TEST = re.compile(r"pytest|unittest|tox|nosetests|manage\.py test|runtests")
FAILED = re.compile(r"(\d+) failed")
EDIT_A = re.compile(r"str_replace_editor\s+(str_replace|create|insert)\s+(\S+)")
EDIT_CMD = re.compile(r'"command":\s*"(str_replace|create|insert)"')
EDIT_PATH = re.compile(r'"path":\s*"([^"]+)"')


def _tail2(p):
    return "/".join(p.strip().split("/")[-2:])


def _edit_path(a):
    m = EDIT_A.search(a)
    if m:
        return _tail2(m.group(2))
    if EDIT_CMD.search(a):
        q = EDIT_PATH.search(a)
        return _tail2(q.group(1)) if q else None
    return None


def patch_files(patch):
    """File names touched by a unified diff, as last-two-path-component tails."""
    return {_tail2(x) for x in re.findall(r"^diff --git a/(\S+)", patch or "", re.M)}


def steps_of(messages):
    """Raw chat messages (OpenHands or SWE-smith tool format) -> [{a, err, rev, edit, test, tfail}]; one entry per assistant action. Unparseable input -> None."""
    msgs = _jl(messages)
    if not isinstance(msgs, list):
        return None
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        a = _act(m)
        if not a:
            continue
        o = next((_txt(x) for x in msgs[i + 1:i + 3] if x.get("role") in ("tool", "user")), "")
        is_test = bool(TEST.search(a))
        f = FAILED.findall(o[-3000:]) if is_test else []
        out.append(dict(a=a[:ACT_CHARS], paths=sorted({_tail2(x) for x in PATH.findall(o[:4000] + " " + a)})[:20], err=bool(ERR.search(o[:3000])), rev=bool(REV.search(a)), edit=_edit_path(a), test=is_test, tfail=(int(f[-1]) if f else (0 if is_test and re.search(r"\d+ passed", o[-3000:]) else None))))
    return out


def signal_counts(groups):
    """groups: {instance: [{resolved, steps, patch_files}]}. Per step-level signal: [hits, steps] in resolved and in unresolved runs.
    A signal is only informative if its rate differs between the two. S3 needs a resolved run of the same instance (its patch files = known right location)."""
    sig = defaultdict(lambda: {"resolved": [0, 0], "unresolved": [0, 0]})
    for runs in groups.values():
        good = set().union(*[r.get("patch_files") or set() for r in runs if r["resolved"]]) if any(r["resolved"] for r in runs) else None
        for r in runs:
            k = "resolved" if r["resolved"] else "unresolved"
            seen = Counter(); edits = Counter(); prev_fail = None
            for st in r["steps"]:
                hits = {"S1_error_after": st["err"], "S2_revert": st["rev"], "S6_action_repeated": seen[st["a"]] >= 1}
                seen[st["a"]] += 1
                if st["edit"]:
                    edits[st["edit"]] += 1
                    hits["S5_same_file_edited_3rd_time"] = edits[st["edit"]] == 3
                    if good is not None:
                        hits["S3_edit_outside_resolved_patch_files"] = st["edit"] not in good
                    if r.get("patch_files") is not None:
                        hits["S4_edit_not_in_own_final_patch"] = st["edit"] not in r["patch_files"]
                if st["test"] and st["tfail"] is not None:
                    hits["S7_test_run_failures_up"] = st["tfail"] > 0 and (prev_fail is None or st["tfail"] > prev_fail)
                    hits["S8_test_run_all_pass"] = st["tfail"] == 0
                    prev_fail = st["tfail"]
                for n, v in hits.items():
                    sig[n][k][1] += 1
                    sig[n][k][0] += bool(v)
    out = {}
    for n, d in sig.items():
        r_, u_ = d["resolved"], d["unresolved"]
        rr, ur = r_[0] / max(r_[1], 1), u_[0] / max(u_[1], 1)
        out[n] = dict(resolved=r_, unresolved=u_, rate_resolved=round(rr, 4), rate_unresolved=round(ur, 4), lift=(round(ur / rr, 2) if rr else None))
    return out


def next_edit_stats(groups):
    """B': natural candidate sets. For each edit step of a RESOLVED run whose file is in its own final patch, candidates = file paths seen in earlier actions/observations of that run.
    Gold = the edited file. Other candidates are real files with unknown relevance (never called wrong). Returns counts and set-size quantiles."""
    n = cov = 0
    sizes = []
    for runs in groups.values():
        for r in runs:
            if not r["resolved"] or not r.get("patch_files"):
                continue
            seen = set()
            for st in r["steps"]:
                if st["edit"] and st["edit"] in r["patch_files"]:
                    n += 1
                    cov += st["edit"] in seen
                    if st["edit"] in seen:
                        sizes.append(len(seen))
                seen.update(st.get("paths") or [])
    sizes.sort()
    q = lambda f: sizes[int(f * (len(sizes) - 1))] if sizes else None
    return dict(edit_decisions=n, gold_in_candidates=cov, set_size_q10_q50_q90=[q(0.1), q(0.5), q(0.9)])


def build_rows(group, match="exact"):
    """group: runs of one instance, [{tid, resolved, steps:[{a, err, rev}]}]. Returns decision rows (see module doc)."""
    rows = []
    seen_nodes = set()

    def walk(runs, d, prefix):
        if len(runs) < 2:
            return
        live = [r for r in runs if len(r["steps"]) > d]
        by_a = defaultdict(list)
        for r in live:
            by_a[r["steps"][d]["a"] if match == "exact" else norm(r["steps"][d]["a"])].append(r)
        if len(by_a) > 1 and (prefix, d) not in seen_nodes:
            seen_nodes.add((prefix, d))
            rows.extend(node_rows(by_a, d))
        for a, rs in by_a.items():
            walk(rs, d + 1, prefix + (a,))

    def node_rows(by_a, d):
        out = []
        for g_key, g_runs in by_a.items():
            gold_runs = [r for r in g_runs if r["resolved"]]
            if not gold_runs:
                continue
            g = gold_runs[0]
            g_act = g["steps"][d]["a"]
            opts = [dict(text=g_act[:ACT_CHARS], tier="GOLD", executed_here=True, provenance=dict(tid=g["tid"], step=d, run_resolved=True, model=g.get("model")))]
            for key, rs in by_a.items():
                if rs is g_runs:
                    continue
                a = rs[0]["steps"][d]["a"]
                res = [r for r in rs if r["resolved"]]
                if res:
                    opts.append(dict(text=a[:ACT_CHARS], tier="T3", executed_here=True, provenance=dict(tid=res[0]["tid"], step=d, run_resolved=True, kind="resolved_branch", model=res[0].get("model"))))
                    continue
                u = rs[0]
                s = u["steps"][d]; nxt = u["steps"][d + 1] if d + 1 < len(u["steps"]) else None
                signs = [k for k, v in (("error", s["err"]), ("revert", s["rev"]), ("retry", nxt is not None and nxt["a"] == s["a"])) if v]
                opts.append(dict(text=a[:ACT_CHARS], tier="T1" if signs else "T2", executed_here=True,
                                 provenance=dict(tid=u["tid"], step=d, run_resolved=False, n_failing_runs=len(rs), signs=signs, model=u.get("model"))))
            texts = {o["text"] for o in opts}
            for k in (d - 2, d - 1, d + 1, d + 2):
                if 0 <= k < len(g["steps"]) and g["steps"][k]["a"][:ACT_CHARS] not in texts:
                    texts.add(g["steps"][k]["a"][:ACT_CHARS])
                    opts.append(dict(text=g["steps"][k]["a"][:ACT_CHARS], tier="T3", executed_here=False, provenance=dict(tid=g["tid"], step=k, run_resolved=True, kind="same_traj_nearby")))
            for o in opts:
                o["near_duplicate_of_gold"] = o["tier"] != "GOLD" and norm(o["text"]) == norm(g_act)
            w = [TIER_W[o["tier"]] for o in opts]
            for o, x in zip(opts, w):
                o["soft_target"] = round(x / sum(w), 4)
            out.append(dict(decision_id="%s:%d:%s" % (g["tid"], d, hashlib.sha1(g_act.encode()).hexdigest()[:8]), match=match, instance=g.get("instance"), state_ref=dict(tid=g["tid"], step=d), options=opts))
        return out

    walk(group, 0, ())
    return rows


def summarise(rows):
    c = Counter(); depth = []
    for r in rows:
        tiers = Counter(o["tier"] for o in r["options"] if not o["near_duplicate_of_gold"])
        c["rows"] += 1
        c["rows_with_T1"] += tiers["T1"] > 0
        c["rows_with_T2_no_T1"] += tiers["T2"] > 0 and not tiers["T1"]
        c["rows_T3_only"] += not tiers["T1"] and not tiers["T2"]
        c["rows_at_step0"] += r["state_ref"]["step"] == 0
        for o in r["options"]:
            c["options_" + o["tier"] + ("_near_dup" if o["near_duplicate_of_gold"] else "")] += 1
        depth.append(r["state_ref"]["step"])
    c["depth_median"] = sorted(depth)[len(depth) // 2] if depth else None
    return dict(c)


def selftest():
    S = lambda *a: [dict(a=x, err=e, rev=False) for x, e in a]
    group = [dict(tid="r1", resolved=True, instance="i", steps=S(("ls", 0), ("edit a.py", 0), ("run t", 0))),
             dict(tid="u1", resolved=False, instance="i", steps=S(("ls", 0), ("edit b.py", 1), ("edit b.py", 1))),
             dict(tid="u2", resolved=False, instance="i", steps=S(("ls", 0), ("cat c.py", 0), ("x", 0))),
             dict(tid="r2", resolved=True, instance="i", steps=S(("pwd", 0), ("y", 0)))]
    rows = build_rows(group)
    assert any(r["state_ref"]["step"] == 0 for r in rows), "root divergence ls vs pwd must give a state"
    n1 = [r for r in rows if r["state_ref"]["step"] == 1]
    assert len(n1) == 1 and n1[0]["options"][0]["tier"] == "GOLD"
    tiers = {o["tier"]: o for o in n1[0]["options"]}
    assert tiers["T1"]["provenance"]["tid"] == "u1" and "error" in tiers["T1"]["provenance"]["signs"] and tiers["T1"]["executed_here"]
    assert tiers["T2"]["provenance"]["tid"] == "u2"
    assert any(o["tier"] == "T3" and not o["executed_here"] for o in n1[0]["options"])
    assert abs(sum(o["soft_target"] for o in n1[0]["options"]) - 1) < 1e-3
    raw = [{"role": "user", "content": "t"}, {"role": "assistant", "content": "", "action": "ls"}, {"role": "tool", "content": "Traceback (most recent call last)"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]}]
    st = steps_of(raw)
    assert [x["a"] for x in st] == ["ls", "bash {}"] and st[0]["err"] and steps_of("not json") is None
    G = {"i": [dict(resolved=True, patch_files={"pkg/a.py"}, steps=[dict(a="x", err=False, rev=False, edit="pkg/a.py", test=False, tfail=None), dict(a="pytest", err=True, rev=False, edit=None, test=True, tfail=0)]),
               dict(resolved=False, patch_files={"pkg/b.py"}, steps=[dict(a="y", err=False, rev=False, edit="pkg/b.py", test=False, tfail=None), dict(a="pytest", err=True, rev=False, edit=None, test=True, tfail=2)])]}
    sc = signal_counts(G)
    assert sc["S3_edit_outside_resolved_patch_files"]["unresolved"] == [1, 1] and sc["S3_edit_outside_resolved_patch_files"]["resolved"] == [0, 1]
    assert sc["S7_test_run_failures_up"]["unresolved"] == [1, 1] and sc["S8_test_run_all_pass"]["resolved"] == [1, 1]
    e = steps_of([{"role": "assistant", "content": "", "action": "str_replace_editor str_replace /testbed/pkg/a.py --old x"}, {"role": "tool", "content": ""}, {"role": "assistant", "content": "", "action": "python -m pytest t.py"}, {"role": "tool", "content": "== 3 failed, 4 passed =="}])
    assert e[0]["edit"] == "pkg/a.py" and e[1]["test"] and e[1]["tfail"] == 3
    assert patch_files("diff --git a/src/x/y.py b/src/x/y.py\n") == {"x/y.py"}
    x = steps_of([{"role": "user", "content": "task"}, {"role": "assistant", "content": "Let me look.\n\n<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/testbed/pkg/a.py</parameter>\n</function>"}, {"role": "user", "content": "OBS"}, {"role": "assistant", "content": "<function=bash>\n<parameter=command>python -m pytest t.py</parameter>\n</function>"}, {"role": "user", "content": "1 failed"}])
    assert len(x) == 2 and x[0]["a"].startswith("str_replace_editor view /testbed/pkg/a.py") and x[1]["a"] == "bash python -m pytest t.py" and x[1]["test"] and x[1]["tfail"] == 1
    ne = next_edit_stats({"i": [dict(resolved=True, patch_files={"pkg/a.py"}, steps=[dict(a="x", paths=["pkg/a.py", "pkg/b.py"], err=False, rev=False, edit=None, test=False, tfail=None), dict(a="e", paths=[], err=False, rev=False, edit="pkg/a.py", test=False, tfail=None)])]})
    assert ne == dict(edit_decisions=1, gold_in_candidates=1, set_size_q10_q50_q90=[2, 2, 2]), ne
    nm = build_rows(group, match="norm")
    assert all(r["match"] == "norm" for r in nm)
    print("selftest ok", summarise(rows))


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    else:
        runs = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
        by = defaultdict(lambda: defaultdict(list))
        for r in runs:
            by[r["src"]][r["instance"]].append(dict(tid=r["tid"], resolved=r["resolved"], steps=r["steps"], instance=r["instance"]))
        print(json.dumps({s: dict(instances=len(g), instances_mixed=sum(len({x["resolved"] for x in v}) == 2 for v in g.values()),
                                  **summarise([row for v in g.values() for row in build_rows(v)])) for s, g in by.items()}, indent=1))
