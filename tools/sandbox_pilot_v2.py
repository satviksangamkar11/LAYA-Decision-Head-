"""Sandbox verified-label pilot v2, late decision points, ONE task (user GO + narrowing, chat 2026-10-15).
Pre-registration: configs/thresholds_sandbox_pilot_v2.toml, configs/fidelity_v4.toml, SANDBOX_LABEL_PILOT_CONTRACT_v2.md.
Runs as root in WSL (python3, stdlib). Reuses tools/sandbox_pilot_v1.py (Pool, ask, Env, editor emulation, evaluate, rollout, transcript, kinds),
tools/wsl_feasibility_run_v2.py (sandbox, view emulation, hashing, disk guard) unchanged; v1 files are not edited.

  run      <out> <iid>     qualify -> candidates -> round 1 -> (early stop | round 2 -> labels); every phase resumes from files
  qualify  <out> <iid>
  scan     <out>           secret scan of every file in <out>
"""
import json, re, shutil, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, "/mnt/d/local model/tools")
import sandbox_pilot_v1 as P1  # noqa: E402  (sets R.BASE = /var/lib/fable-runs/pilot-v1: the v1 environments are reused if intact)
from sandbox_pilot_v1 import R, W, append, now, dname  # noqa: E402

V1 = Path("/mnt/d/local model/results/raw/sandbox-pilot-v1-20261014T162937Z")
SPLIT = "sandbox-pilot-v2"
EDITS = ("create", "str_replace", "insert", "undo_edit")


def log(out, *a):
    with open(out / "RUN.log", "a", encoding="utf-8") as f:
        f.write(now() + " " + " ".join(str(x) for x in a) + "\n")
    print(now(), *a, flush=True)


# ------------------------------------------------------------------ fidelity v4
DUR = [re.compile(r"\b\d+:\d{2}:\d{2}(\.\d+)?\b"), re.compile(r"\b\d+(\.\d+)?\s?(s|sec|secs|seconds|ms)\b")]


def dur_mask(t):
    for p in DUR:
        t = p.sub("<DUR>", t or "")
    return t


def compare_v4(cmd, ours_exit, ours_out, rec_exit, rec_out):
    c = P1.compare_v3(ours_exit, ours_out, rec_exit, rec_out)
    if c["match"] or not P1.TEST.search(cmd or ""):
        return c
    c4 = P1.compare_v3(ours_exit, dur_mask(ours_out), rec_exit, dur_mask(rec_out))
    return dict(c4, rule="v4") if c4["match"] else c


def _selftest():
    rec = "==== 3 passed in 0:00:01.52 ====\nPASSED t::a"
    assert compare_v4("python -m pytest", 0, rec.replace("0:00:01.52", "0:00:02.10"), 0, rec)["match"]
    assert not compare_v4("ls", 0, rec.replace("0:00:01.52", "0:00:02.10"), 0, rec)["match"]
    assert not compare_v4("pytest", 0, rec.replace("3 passed", "2 passed"), 0, rec)["match"]
    assert norm_action(dict(tool="execute_bash", command=" ls  -la ")) == norm_action(dict(tool="execute_bash", command="ls -la"))


# ------------------------------------------------------------------ replay with editor emulation
def editor_env(r, conda, d):
    e = P1.Env.__new__(P1.Env)
    e.r, e.conda, e.d, e.hist, e.env_secs = r, conda, d, P1.collections.defaultdict(list), 0.0
    return e


def replay_v2(task, snapshot, conda, label, keep=False):
    d = dname(task)
    r = R.new_run(snapshot, d)
    R.chown(r)
    ed = editor_env(r, conda, d)
    rows, obs = [], []
    for st in task["prefix"]:
        name, a = st["name"], st["arguments"]
        row = dict(replay=label, step=st["index"], tool=name, command=(a.get("command") or "")[:300], path=a.get("path"))
        if name in ("think", "task_tracker", "finish"):
            rows.append(dict(row, verdict="NO_ENV_EFFECT"))
            obs.append(st["recorded"] or "")
            continue
        if name == "execute_bash":
            if a.get("is_input") in (True, "true"):
                rows.append(dict(row, verdict="MISMATCH", reason="is_input"))
                break
            rec, rec_exit = R.strip_footer(st["recorded"])
            R.chown(r)
            code, o, secs = R.run(R.mounts_of(r, conda, d), R.WRAP % a.get("command", ""), timeout=P1.CMD_TIMEOUT)
            c = compare_v4(a.get("command"), code, o, rec_exit, rec)
            row.update(ours_exit=code, recorded_exit=rec_exit, seconds=secs)
            obs.append(P1.fmt_bash(code, o))
        elif name == "str_replace_editor" and a.get("command") == "view":
            o = R.view(a, r, conda, d)
            if o is None:
                rows.append(dict(row, verdict="MISMATCH", reason="path outside workspace"))
                break
            c = P1.compare_v3(0, o, 0, st["recorded"] or "")
            obs.append(o)
        elif name == "str_replace_editor" and a.get("command") in EDITS:
            o, err = ed.editor(a)
            rec_err = (st["recorded"] or "").lstrip().startswith("ERROR")
            c = dict(match=err == rec_err, exact=None, signature=None, rule="edit_emulation", ours_error=err, recorded_error=rec_err)
            obs.append(o)
        else:
            rows.append(dict(row, verdict="MISMATCH", reason="unexpected tool"))
            break
        row.update(exact=c.get("exact"), rule=c.get("rule"), verdict="MATCH" if c["match"] else "MISMATCH")
        if not c["match"]:
            row["diff"] = R.masked_diff(st["recorded"] or "", obs[-1])
        rows.append(row)
        if not c["match"]:
            break
    h, full, n = R.ws_hash(r / "ws" / d)
    if not keep:
        shutil.rmtree(r, ignore_errors=True)
        r = None
    return rows, dict(replay=label, workspace_sha256=h, full_tree_sha256_info=full, files=n), obs, r


# ------------------------------------------------------------------ qualification (a)-(c) + oracle + presolved diagnostic
def load(out, iid):
    return json.loads((out / ("prefix-%s.json" % iid)).read_text(encoding="utf-8"))


def intact(out, task):
    iid, T = task["instance_id"], task["task"]
    tdir = R.BASE / iid
    snap, conda = tdir / "ws" / dname(task), tdir / "conda"
    if not snap.is_dir() or not conda.is_dir():
        return dict(ok=False, why="missing")
    git = lambda *c: subprocess.run(["git", "-c", "safe.directory=*", "-C", str(snap), *c], capture_output=True, text=True).stdout.strip()
    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    r = R.new_run(snap, dname(task))
    R.chown(r)
    code, fr, _ = R.run(R.mounts_of(r, conda, dname(task)), "pip freeze > /tmp/freeze.txt", timeout=120)
    freeze = (r / "tmp" / "freeze.txt").read_text() if (r / "tmp" / "freeze.txt").exists() else ""
    shutil.rmtree(r, ignore_errors=True)
    v1res = json.loads((V1 / ("result-%s.json" % iid)).read_text())
    want = v1res["setup"]["pip_freeze_sha256"]
    rec = dict(head_ok=head == T["base_commit"], status_clean=status == "", freeze_sha256=W.sha(freeze), v1_freeze_sha256=want, freeze_exit=code)
    rec["ok"] = rec["head_ok"] and rec["status_clean"] and rec["freeze_sha256"] == want
    return rec


def qualify(out, iid):
    q = out / "QUALIFY.jsonl"
    if q.exists() and any(json.loads(l)["instance_id"] == iid for l in open(q)):
        log(out, "qualify: already done for", iid)
        return
    task = load(out, iid)
    T = task["task"]
    ic, d = T["install_config"], dname(task)
    tdir = R.BASE / iid
    snap, conda = tdir / "ws" / d, tdir / "conda"
    t0 = time.monotonic()
    res = dict(instance_id=iid, repo=T["repo"], base_commit=T["base_commit"], trajectory_id=task["trajectory_id"], n_actions=task["n_actions"],
               decision_position=task["decision_position"], prefix_steps=len(task["prefix"]), started=now())
    if not R.disk_ok("qualify-start", out, iid):
        res.update(qualified=False, reason="DISK_GUARD_ABORT")
        return append(out / "QUALIFY.jsonl", res)
    res["reuse"] = intact(out, task)
    log(out, "reuse check", res["reuse"])
    if not res["reuse"]["ok"]:
        res.update(qualified=False, reason="v1 environment not intact; rebuild needed (not automated in v2: stop and ask)")
        return append(out / "QUALIFY.jsonl", res)
    # network-off probe
    r = R.new_run(snap, d)
    R.chown(r)
    pc, po, _ = R.run(R.mounts_of(r, conda, d), "python -c \"import socket; socket.create_connection(('1.1.1.1', 80), timeout=5)\" ; a=$?; getent hosts github.com; b=$?; "
                      "test $a -ne 0 && test $b -ne 0", timeout=60)
    shutil.rmtree(r, ignore_errors=True)
    res["network_off_probe"] = dict(socket_and_dns_failed=pc == 0)
    # initial test condition at base + test_patch, twice (as v2)
    f2p, p2p_all = T["FAIL_TO_PASS"], T["PASS_TO_PASS"]
    p2p = p2p_all if len(p2p_all) <= 20 else P1.random.Random(P1.SEED).sample(p2p_all, 20)
    tests = []
    for k in (1, 2):
        r = R.new_run(snap, d)
        ap = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(r / "ws" / d), "apply", "-"], input=T["test_patch"].encode(), capture_output=True)
        R.chown(r)
        code, o, secs = R.run(R.mounts_of(r, conda, d), "cd /workspace/%s && %s %s" % (d, ic["test_cmd"], " ".join(P1.shlex.quote(x) for x in f2p + p2p)),
                              timeout=1800, mem="4G")
        shutil.rmtree(r, ignore_errors=True)
        st = R.test_ids_status(o, f2p + p2p)
        tests.append(dict(run=k, applied=ap.returncode == 0, exit=code, seconds=secs, f2p_failed=sum(st[i] != "PASSED" for i in f2p), f2p_total=len(f2p),
                          f2p_not_reported=sum(st[i] == "NOT_REPORTED" for i in f2p), p2p_passed=sum(st[i] == "PASSED" for i in p2p), p2p_total=len(p2p), status=st))
    res["tests"] = [{k: v for k, v in t.items() if k != "status"} for t in tests]
    res["tests_ok"] = all(t["applied"] and t["f2p_failed"] == t["f2p_total"] and t["f2p_not_reported"] == 0 and t["p2p_passed"] == t["p2p_total"] for t in tests) \
        and tests[0]["status"] == tests[1]["status"]
    log(out, "initial test condition ok:", res["tests_ok"])
    # (a) + (c): replays A, B, C (C kept as the decision-point state)
    hashes, allrows, obs, keep = [], [], None, None
    for lab in "ABC":
        if not R.disk_ok("replay-" + lab, out, iid):
            res.update(qualified=False, reason="DISK_GUARD_ABORT")
            return append(out / "QUALIFY.jsonl", res)
        rows, h, o, rdir = replay_v2(task, snap, conda, lab, keep=lab == "C")
        allrows += rows
        hashes.append(h)
        if lab == "C":
            obs, keep = o, rdir
        log(out, "replay", lab, "match", sum(x["verdict"] == "MATCH" for x in rows), "mismatch", sum(x["verdict"] == "MISMATCH" for x in rows),
            "v4-only", sum(x.get("rule") == "v4" for x in rows))
    with open(out / "REPLAY.jsonl", "a", encoding="utf-8") as f:
        for x in allrows:
            f.write(json.dumps(dict(instance_id=iid, **x)) + "\n")
    n_env = len([s for s in task["prefix"] if s["name"] not in ("think", "task_tracker", "finish")])
    res["replay"] = {lab: dict(matched=sum(x["verdict"] == "MATCH" for x in allrows if x["replay"] == lab),
                               mismatched=sum(x["verdict"] == "MISMATCH" for x in allrows if x["replay"] == lab),
                               v4_only_matches=sum(x.get("rule") == "v4" for x in allrows if x["replay"] == lab),
                               edit_emulated=sum(x.get("rule") == "edit_emulation" for x in allrows if x["replay"] == lab)) for lab in "ABC"}
    res["prefix_env_steps"] = n_env
    res["hashes"] = hashes
    res["a_prefix_ok"] = all(res["replay"][l]["mismatched"] == 0 and res["replay"][l]["matched"] == n_env for l in "ABC")
    res["c_hash_ok"] = hashes[1]["workspace_sha256"] == hashes[2]["workspace_sha256"]
    res["hash_equal_ABC"] = len({h["workspace_sha256"] for h in hashes}) == 1
    sdir = R.BASE / iid / "state-v2"
    shutil.rmtree(sdir, ignore_errors=True)
    shutil.move(str(keep), str(sdir))
    state = dict(state_dir=str(sdir), workspace_sha256=hashes[2]["workspace_sha256"])
    # (b) oracle + presolved diagnostic
    orc = dict(instance_id=iid, started=now())
    if res["a_prefix_ok"] and res["c_hash_ok"]:
        e = P1.Env(task, state)
        orc["restore_hash_ok"] = e.hash == state["workspace_sha256"]
        acts = []
        for st in task["remaining"]:
            if st["name"] in ("think", "task_tracker", "finish"):
                acts.append(dict(step=st["index"], tool=st["name"], skipped=True))
                continue
            a = dict(tool=st["name"], **{k: v for k, v in st["arguments"].items() if k not in ("security_risk", "is_input", "timeout")})
            o, info = e.execute(a)
            acts.append(dict(step=st["index"], tool=st["name"], command=(a.get("command") or "")[:200], path=a.get("path"), info=info, observation_head=o[:300]))
        orc["remaining_actions"] = acts
        orc["evaluation"] = P1.evaluate(task, e)
        e.close()
        e = P1.Env(task, state)
        pre = P1.evaluate(task, e)
        e.close()
        orc["presolved_evaluation"] = pre
        orc["resolved_at_decision_point"] = pre["resolved"]
        orc["oracle_ok"] = orc["restore_hash_ok"] and orc["evaluation"]["resolved"] is True and not any(x.get("info", {}).get("env_failure") for x in acts)
    else:
        orc["oracle_ok"], orc["skipped"] = False, "prefix or hash failed"
    orc["ended"] = now()
    append(out / "ORACLE.jsonl", orc)
    res["b_oracle_ok"] = orc["oracle_ok"]
    res["resolved_at_decision_point"] = orc.get("resolved_at_decision_point")
    reasons = [m for ok, m in ((res["network_off_probe"]["socket_and_dns_failed"], "network probe did not fail"), (res["tests_ok"], "initial test condition not met"),
                               (res["a_prefix_ok"], "(a) prefix replay mismatch"), (res["c_hash_ok"], "(c) B/C hashes differ"),
                               (res["b_oracle_ok"], "(b) oracle: recorded remaining actions do not resolve")) if not ok]
    res.update(qualified=not reasons, reason="; ".join(reasons) or None, wall_seconds=round(time.monotonic() - t0, 1), ended=now())
    if res["qualified"]:
        g = subprocess.run(["grep", "-rnIE", P1.BEHAV, "--include=*.py", "--exclude-dir=.git", str(snap)], capture_output=True, text=True).stdout.splitlines()
        (out / ("STATE-%s.json" % iid)).write_text(json.dumps(dict(instance_id=iid, **state, behaviour_grep_repo_hits=len(g),
                                                                   behaviour_grep_sample=[x.replace(str(snap), "")[:200] for x in g[:40]], observations=obs), indent=1))
    append(out / "QUALIFY.jsonl", res)
    log(out, "QUALIFY", iid, res["qualified"], res["reason"], "presolved", res["resolved_at_decision_point"])


# ------------------------------------------------------------------ candidates
SCHEMA_S = ('Required format, exactly one line: <action>{"tool": "execute_bash", "command": "<read-only command>"}</action> or '
            '<action>{"tool": "str_replace_editor", "command": "view", "path": "<absolute path>"}</action>. Example: '
            '<action>{"tool": "execute_bash", "command": "cd /workspace/%s && grep -rn \\"def main\\" ."}</action>')
SCHEMA_T = ('It must NOT be a view / ls / cat / grep / find (read-only) action. Required format, exactly one line: '
            '<action>{"tool": "execute_bash", "command": "<command that runs tests>"}</action> or '
            '<action>{"tool": "str_replace_editor", "command": "str_replace", "path": "<absolute path>", "old_str": "<exact text>", "new_str": "<new text>"}</action> '
            '(or "command": "create" with "file_text", or "insert" with "insert_line" and "new_str"). Examples: '
            '<action>{"tool": "execute_bash", "command": "cd /workspace/%s && python -m pytest tests/ -q"}</action> ; '
            '<action>{"tool": "str_replace_editor", "command": "str_replace", "path": "/workspace/%s/pkg/mod.py", "old_str": "return a", "new_str": "return b"}</action>')
TYPE = dict(S="READ/SEARCH", T="TEST-OR-EDIT")


def norm_action(a):
    ws = lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v
    keys = ("tool", "command", "path", "old_str", "new_str", "file_text", "insert_line", "view_range")
    return json.dumps({k: ws(a.get(k)) for k in keys if a.get(k) is not None}, sort_keys=True)


def gen_slot(pool, iid, slot, base, rec_action, d):
    q = (P1.ASK_S if slot == "S" else P1.ASK_T) + " " + ((SCHEMA_S % d) if slot == "S" else (SCHEMA_T % (d, d)))
    msgs = base + [dict(role="user", content=q)]
    attempts, chosen, reasked, equivalent = [], None, False, None
    for k in range(1, 5):  # pre-registered: up to 4 attempts, errors fed back
        calls = []
        r = P1.ask(pool, "%s__v2gen%s__att%d" % (iid, slot, k), msgs, calls)
        att = dict(attempt=k, calls=calls)
        attempts.append(att)
        if r is None:
            att["error"] = "model unavailable"
            continue
        content = r[0] or ""
        act = P1.parse_action(content)
        kind = act and P1.kind_of(act)
        ok = act is not None and ((slot == "S" and kind == "read") or (slot == "T" and kind in ("test", "edit")))
        att.update(parsed=act is not None, kind=kind, finish=r[1], content_tail=content[-600:], type_ok=ok)
        if ok and norm_action(act) == norm_action(rec_action) and not reasked:
            att["equivalent_to_recorded"] = True
            equivalent = act
            reasked = True
            fb = "Propose a different action of the required %s type (not this one). Reply with exactly one <action>{json}</action>." % TYPE[slot]
        elif ok:
            chosen = act
            break
        elif act is None:
            m = P1.ACT.findall(content)
            fb = ("Could not parse your action: %s. " % ("no <action>...</action> block found" if not m else "the text inside <action></action> is not a valid JSON "
                  "object with a known tool")) + "Reply with exactly one <action>{json}</action> of the required %s type." % TYPE[slot]
        else:
            fb = "That is a %s action; the required type is %s. Reply with exactly one <action>{json}</action> of the required type." % (kind.upper(), TYPE[slot])
        msgs = msgs + [dict(role="assistant", content=content), dict(role="user", content=fb)]
    dup = chosen is not None and norm_action(chosen) == norm_action(rec_action)
    return chosen, attempts, dict(reasked_for_equivalence=reasked, duplicate_of_recorded=dup)


def candidates(out, iid):
    path = out / "CANDIDATES.jsonl"
    if path.exists() and any(json.loads(l)["instance_id"] == iid for l in open(path)):
        log(out, "candidates: already done")
        return
    task, st = load(out, iid), P1.state_of(out, iid)
    base = P1.transcript(out, task, st["observations"])
    rec = P1.recorded_action(task)
    pool = P1.Pool(out)
    res = {}
    th = [threading.Thread(target=lambda s=s: res.__setitem__(s, gen_slot(pool, iid, s, base, rec, dname(task)))) for s in "ST"]
    [t.start() for t in th]
    [t.join() for t in th]
    cands = [dict(cid="%s__R" % iid, slot="R", source="recorded run %s position %d" % (task["trajectory_id"], task["decision_position"]), action=rec,
                  kind=P1.kind_of(rec), attempts=[], flags={})]
    for s in "ST":
        act, att, flags = res[s]
        cands.append(dict(cid="%s__%s" % (iid, s), slot=s, source="GLM-5.3 (z-ai/glm-5.3, glm_max64)", action=act, kind=act and P1.kind_of(act), attempts=att, flags=flags))
    for c in cands:
        if c["action"] is None:
            c.update(valid=False, invalid_reason="no parsable action of the required type in 4 attempts")
        else:
            e = P1.Env(task, st)
            c["restore_hash_ok"] = e.hash == st["workspace_sha256"]
            o, info = e.execute(c["action"])
            e.close()
            c.update(valid=bool(c["restore_hash_ok"]) and not info.get("exec_error") and not info.get("env_failure"), dry_run=dict(info, observation_head=o[:1500]),
                     invalid_reason=("cannot execute: " + o[:200]) if info.get("exec_error") else None)
        c.update(instance_id=iid, repo=task["task"]["repo"], base_commit=task["task"]["base_commit"], state_sha256=st["workspace_sha256"],
                 source_dataset="nebius/SWE-rebench + nebius/SWE-rebench-openhands", licence="CC-BY-4.0", split=SPLIT,
                 action_text=P1.render(c["action"]) if c["action"] else None, n_attempts=len(c["attempts"]))
        append(path, c)
        log(out, "CANDIDATE", c["cid"], c["kind"], "valid=%s" % c["valid"], "attempts=%d" % c["n_attempts"], (c["action_text"] or "")[:200])


# ------------------------------------------------------------------ staged rollouts
def rollout_round(out, iid, ks):
    task, st = load(out, iid), P1.state_of(out, iid)
    cands = [json.loads(l) for l in open(out / "CANDIDATES.jsonl") if json.loads(l)["instance_id"] == iid]
    done = {json.loads(l)["rollout_id"] for l in open(out / "ROLLOUTS.jsonl")} if (out / "ROLLOUTS.jsonl").exists() else set()
    base = P1.transcript(out, task, st["observations"])
    pool = P1.Pool(out)
    th = [threading.Thread(target=P1.rollout, args=(out, pool, task, st, c, k, base)) for c in cands if c["valid"] for k in ks
          if "%s__r%d" % (c["cid"], k) not in done]
    assert len(th) + len(done) <= 9, "scope cap"
    log(out, "rollouts round", ks, "starting", len(th))
    [t.start() for t in th]
    [t.join() for t in th]


def rollouts_of(out):
    return [json.loads(l) for l in open(out / "ROLLOUTS.jsonl")] if (out / "ROLLOUTS.jsonl").exists() else []


def labels(out, iid):
    path = out / "LABELS.jsonl"
    if path.exists():
        return
    cands = {c["cid"]: c for c in (json.loads(l) for l in open(out / "CANDIDATES.jsonl"))}
    by = P1.collections.defaultdict(list)
    for r in rollouts_of(out):
        by[r["cid"]].append(r)
    for a, b in (("R", "S"), ("R", "T"), ("S", "T")):
        A, B = cands["%s__%s" % (iid, a)], cands["%s__%s" % (iid, b)]
        ra, rb = by[A["cid"]], by[B["cid"]]
        pa, pb = sum(bool(x["resolved"]) for x in ra), sum(bool(x["resolved"]) for x in rb)
        dup = a == "R" and bool(B["flags"].get("duplicate_of_recorded"))
        if not (A["valid"] and B["valid"]):
            lab, why = "INVALID", "invalid candidate"
        elif len(ra) != 3 or len(rb) != 3 or not all(x["clean"] for x in ra + rb):
            lab, why = "UNKNOWN", "not 3+3 clean rollouts"
        elif abs(pa - pb) >= 2:
            lab, why = ("A_BETTER" if pa > pb else "B_BETTER"), "r_a=%d/3 r_b=%d/3" % (pa, pb)
        elif (pa, pb) in ((3, 3), (0, 0)):
            lab, why = "EQUAL_ON_TESTED_OUTCOME", "both %d/3: equal on the measured test outcome (not 'equally good')" % pa
        else:
            lab, why = "UNKNOWN", "r_a=%d/3 r_b=%d/3 (difference < 2/3)" % (pa, pb)
        append(path, dict(label_id="%s__%s_vs_%s" % (iid, a, b), instance_id=iid, repo=A["repo"], base_commit=A["base_commit"], state_sha256=A["state_sha256"],
                          a=A["cid"], b=B["cid"], a_kind=A["kind"], b_kind=B["kind"], a_action=A["action_text"], b_action=B["action_text"], r_a=pa, r_b=pb,
                          label=lab, reason=why, duplicate_pair=bool(dup), non_edit_comparison=A["kind"] in ("read", "test") or B["kind"] in ("read", "test"),
                          better_worse=lab in ("A_BETTER", "B_BETTER") and not dup, scope="late-task decision (10 recorded actions remain)",
                          provenance="sandbox rollouts (GLM-5.3 continuation, 15 steps, test_patch at evaluation only, 2 evaluation runs)",
                          source="nebius/SWE-rebench + nebius/SWE-rebench-openhands", licence="CC-BY-4.0", split=SPLIT))


def scan(out):
    hits = []
    for p in out.rglob("*"):
        if p.is_file():
            m = R.SECRET.search(p.read_text("utf-8", "replace"))
            if m:
                hits.append(str(p.relative_to(out)))
    (out / "SECRET_SCAN.json").write_text(json.dumps(dict(ts=now(), files=sum(1 for p in out.rglob("*") if p.is_file()), hits=hits), indent=1))
    log(out, "secret scan hits:", hits)


def run(out, iid):
    log(out, "run start", iid)
    qualify(out, iid)
    if not (out / ("STATE-%s.json" % iid)).exists():
        log(out, "not qualified: stop for this id")
        return scan(out)
    candidates(out, iid)
    valid = [json.loads(l) for l in open(out / "CANDIDATES.jsonl") if json.loads(l)["instance_id"] == iid and json.loads(l)["valid"]]
    if len(valid) < 2:
        log(out, "fewer than 2 valid candidates: stop")
        return scan(out)
    rollout_round(out, iid, (1,))
    r1 = [r for r in rollouts_of(out) if r["rollout"] == 1]
    if not any(r["resolved"] for r in r1):
        log(out, "EARLY STOP: no candidate resolved in round 1")
        return scan(out)
    rollout_round(out, iid, (2, 3))
    labels(out, iid)
    log(out, "labels written")
    scan(out)


if __name__ == "__main__":
    P1._selftest()
    _selftest()
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd == "selftest":
        print("selftest ok")
    elif cmd == "run":
        run(out, sys.argv[3])
    elif cmd == "qualify":
        qualify(out, sys.argv[3])
    elif cmd == "scan":
        scan(out)
