"""Sandbox verified-label pilot v1 (user-approved, chat 2026-10-14). Pre-registration: configs/fidelity_v3.toml, configs/thresholds_sandbox_pilot_v1.toml.
Runs as root in WSL Ubuntu-24.04 (python3, stdlib only). Reuses the v2 feasibility runner (tools/wsl_feasibility_run_v2.py: setup, qualification,
sandbox argv, editor `view` emulation, hashing, disk guard, cleanup), compare_v2 (tools/fable_wsl_runner.py) and the NVIDIA HTTP client
(tools/judge_qual_v1.http / key_files).

  v3recheck  <out>                          googleapis step-6 re-evaluation under compare_v3 from saved v2 records (no execution)
  qualify    <out> <prefix_json>            v2 qualification under compare_v3 on a rebuilt environment; if QUALIFIED, builds the decision-point state
  candidates <out> <iid,...>                recorded + 2 GLM-5.3 candidates per state, dry-run validity
  rollouts   <out> <iid,...>                3 rollouts per valid candidate (continuation + evaluation x2), then labels
  cleanup    <out> <iid>
"""
import collections, copy, json, os, random, re, shlex, shutil, subprocess, sys, threading, time, uuid
from pathlib import Path

sys.path.insert(0, "/mnt/d/local model/tools")
import wsl_feasibility_run_v2 as R  # noqa: E402
import fable_wsl_runner as W  # noqa: E402
import judge_qual_v1 as J  # noqa: E402

R.BASE = Path("/var/lib/fable-runs/pilot-v1")
V2 = Path("/mnt/d/local model/results/raw/wsl-feasibility-v2-20261014T151858Z")
SEED = 20261014
SETTINGS = dict(model="z-ai/glm-5.3", temperature=0.5, top_p=1.0, reasoning_effort="max", stream=True, max_tokens=65536)  # glm_max64
INFLIGHT, RPM, COOL = 10, 39, 60.0  # user, chat 2026-10-14 (mid-pilot): 10 in flight per key; was 5 for the first 12 rollouts (starlette, googleapis)
STEPS, ENV_CAP, WALL_CEIL, CMD_TIMEOUT, OBS_MAX, CALL_ATTEMPTS = 15, 1200, 4 * 3600, 300, 20000, 3
SEM_EXEC, SEM_EVAL = threading.Semaphore(3), threading.Semaphore(1)  # WSL has 7 GiB; evaluation runs with MemoryMax 4G

# ------------------------------------------------------------------ fidelity v3
LS = re.compile(r"^([-dlcbps][rwxsStT-]{9}[.+@]?)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+(?:,\s*\d+)?)\s+([A-Z][a-z]{2}\s+\d{1,2}\s+(?:\d{1,2}:\d{2}|\d{4})|"
                r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s+[+-]\d{4})?)\s+(.+)$", re.M)


def ls_norm(t):
    return LS.sub(lambda m: "%s %s <OWNER> <GROUP> %s <MTIME> %s" % (m[1], m[2], m[5], m[7]), (t or "").replace("\r", ""))


def compare_v3(ours_exit, ours_out, rec_exit, rec_out):
    """compare_v2 after owner / group / timestamp of ls -l lines are masked (configs/fidelity_v3.toml)."""
    return dict(W.compare_v2(ours_exit, ls_norm(ours_out), rec_exit, ls_norm(rec_out)), rule="v3")


def _selftest():
    rec = "total 28\ndrwxr-xr-x  7 root root 4096 Sep  1 01:30 .\n-rw-r--r--  1 root root    0 Sep  1 01:30 __init__.py"
    ok = "total 28\ndrwxr-xr-x  7 replay systemd-journal 4096 Oct 14 15:24 .\n-rw-r--r--  1 replay systemd-journal    0 Oct 14 15:24 __init__.py"
    assert compare_v3(0, ok, 0, rec)["match"] and not W.compare_v2(0, ok, 0, rec)["match"]
    for bad in (ok.replace("4096", "4097"), ok.replace("-rw-r--r--", "-rwxr-xr-x"), ok.replace("__init__", "_init_"), ok.replace("total 28", "total 32"),
                ok.replace("  7 replay", "  8 replay")):
        assert not compare_v3(0, bad, 0, rec)["match"], bad
    assert not compare_v3(1, ok, 0, rec)["match"]


# ------------------------------------------------------------------ helpers
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_task(out, iid):
    for p in (out / "walk" / ("prefix-%s.json" % iid), V2 / ("prefix-%s.json" % iid)):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(iid)


def dname(task):
    return task["workspace_dir"].rstrip("/").split("/")[-1]


def trunc(t):
    return t if len(t) <= OBS_MAX else t[:5000] + "\n[... %d characters omitted ...]\n" % (len(t) - 20000) + t[-15000:]


ENV_FAIL = re.compile(r"^(bwrap: |Failed to start transient scope|systemd-run: |Job for .* failed)", re.M)


# ------------------------------------------------------------------ v3 recheck from saved v2 records
def apply_unified(rec_lines, diff):
    """rebuild the replayed text from the recorded text + the saved n=0 unified diff (recorded -> replayed)"""
    hunks, cur = [], None
    for l in diff.splitlines():
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", l)
        if m:
            cur = dict(a=int(m[1]), b=int(m[2] if m[2] is not None else 1), plus=[])
            hunks.append(cur)
        elif cur is not None and l.startswith("+") and not l.startswith("+++"):
            cur["plus"].append(l[1:])
    out = list(rec_lines)
    for h in reversed(hunks):
        s = h["a"] - 1 if h["b"] else h["a"]
        out[s:s + h["b"]] = h["plus"]
    return out


def v3recheck(out):
    iid = "googleapis__python-storage-1357"
    task = json.loads((V2 / ("prefix-%s.json" % iid)).read_text())
    res2 = json.loads((V2 / ("result-%s.json" % iid)).read_text())
    rows = [json.loads(l) for l in open(V2 / "REPLAY.jsonl") if json.loads(l)["instance_id"] == iid]
    rep = dict(instance_id=iid, rule="compare_v3", v2_result_reason=res2["reason"], replays={})
    for lab in "ABC":
        rr = [r for r in rows if r["replay"] == lab]
        mism = [r for r in rr if r["verdict"] == "MISMATCH"]
        assert len(mism) == 1 and mism[0]["step"] == 6
        st = task["prefix"][6]
        rec, rec_exit = R.strip_footer(st["recorded"])
        diff = mism[0]["diff"]
        complete = len(diff) < 1500 and diff.count("\n") + 1 < 24
        ours = "\n".join(apply_unified(rec.splitlines(), diff))
        c3, c2 = compare_v3(mism[0]["ours_exit"], ours, rec_exit, rec), W.compare_v2(mism[0]["ours_exit"], ours, rec_exit, rec)
        rep["replays"][lab] = dict(earlier_steps_matched=sum(r["verdict"] == "MATCH" for r in rr), step6_diff_complete=complete, step6_v2_match=c2["match"],
                                   step6_v3=c3, step6_is_last_prefix_step=st["index"] == len(task["prefix"]) - 1)
    ok = all(v["step6_diff_complete"] and v["step6_v3"]["match"] and not v["step6_v2_match"] and v["earlier_steps_matched"] == 5 for v in rep["replays"].values())
    other = dict(setup=res2["setup"]["ok"], network_probe_failed=res2["network_off_probe"]["socket_and_dns_failed"], tests_ok=res2["tests_ok"],
                 hash_equal_BC=res2["hash_equal_BC"])
    rep.update(other_v2_checks=other, prefix_env_steps=res2["prefix_env_steps"], qualified_v3=ok and all(other.values()),
               method="re-evaluated from saved records; no execution (diff complete, so the replay stage was not re-run)")
    (out / "v3recheck-googleapis.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ qualification + decision-point state
BEHAV = r"st_mtime|getmtime|st_ctime|getctime|st_uid|st_gid|getuid|getgid|getpwuid|chown|utime|os\.stat|lstat|\.stat\(\)|\.owner\(\)|\.group\(\)"


def build_state(task, snapshot, conda):
    """replay the prefix into a persistent run dir (kept as the decision-point state); returns record + observations"""
    d = dname(task)
    r = R.new_run(snapshot, d)
    R.chown(r)
    obs, steps = [], []
    for st in task["prefix"]:
        name, a = st["name"], st["arguments"]
        if name in ("think", "task_tracker", "finish"):
            obs.append(st["recorded"] or "")
            continue
        rec, rec_exit = R.strip_footer(st["recorded"]) if name == "execute_bash" else (st["recorded"] or "", 0)
        if name == "execute_bash":
            R.chown(r)
            code, o, secs = R.run(R.mounts_of(r, conda, d), R.WRAP % a.get("command", ""), timeout=CMD_TIMEOUT)
            c = compare_v3(code, o, rec_exit, rec)
            obs.append(fmt_bash(code, o))
        else:
            o = R.view(a, r, conda, d)
            c = compare_v3(0, o or "", 0, rec)
            obs.append(o or "")
        steps.append(dict(step=st["index"], tool=name, match=c["match"], exact=c["exact"]))
    h, full, n = R.ws_hash(r / "ws" / d)
    dest = R.BASE / task["instance_id"] / "state"
    shutil.move(str(r), str(dest))
    return dict(state_dir=str(dest), workspace_sha256=h, files=n, cwd=(dest / "state" / "cwd").read_text().strip(), steps=steps,
                all_match=all(s["match"] for s in steps)), obs


def qualify(out, prefix_json):
    task = json.loads(Path(prefix_json).read_text(encoding="utf-8"))
    iid = task["instance_id"]
    R.compare_v2 = compare_v3  # the v2 procedure, judged with fidelity v3
    sys.argv = [sys.argv[0], prefix_json, str(out)]
    R.main()  # setup + probe + tests x2 + replays A/B/C; writes result-<iid>.json, REPLAY.jsonl, DISK.jsonl, SETUP log; no cleanup
    res = json.loads((out / ("result-%s.json" % iid)).read_text())
    if not res.get("qualified"):
        R.cleanup(iid, out)
        return
    tdir = R.BASE / iid
    snapshot, conda = tdir / "ws" / dname(task), tdir / "conda"
    rec, obs = build_state(task, snapshot, conda)
    rec["hash_equal_qualification_BC"] = rec["workspace_sha256"] == res["hashes"][1]["workspace_sha256"] == res["hashes"][2]["workspace_sha256"]
    g = subprocess.run(["grep", "-rnIE", BEHAV, "--include=*.py", "--exclude-dir=.git", str(snapshot)], capture_output=True, text=True).stdout.splitlines()
    tp = [l for l in task["task"]["test_patch"].splitlines() if l.startswith("+") and re.search(BEHAV, l)]
    rec["behaviour_grep"] = dict(pattern=BEHAV, repo_hits=len(g), repo_hits_sample=[x.replace(str(snapshot), "")[:200] for x in g[:40]], test_patch_hits=tp[:20])
    rec["v2_hash_B"] = (json.loads((V2 / ("result-%s.json" % iid)).read_text())["hashes"][1]["workspace_sha256"]
                        if (V2 / ("result-%s.json" % iid)).exists() else None)
    (out / ("STATE-%s.json" % iid)).write_text(json.dumps(dict(instance_id=iid, **rec, observations=obs), indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "behaviour_grep"}, indent=1), rec["behaviour_grep"]["repo_hits"])


# ------------------------------------------------------------------ NVIDIA pool (fixed call logic)
class Pool:
    def __init__(self, out):
        self.dir = out / "responses"
        self.dir.mkdir(exist_ok=True)
        stale = list(self.dir.glob("*.started"))
        if stale:
            sys.exit("in-flight markers from a stopped run (a person decides): %s" % [p.name for p in stale])
        self.ledger = out / "CALLS.jsonl"
        self.files = J.key_files()
        assert self.files, "no key files"
        self.cv = threading.Condition()
        self.sent = {f.name: collections.deque() for f in self.files}
        self.cool = {f.name: 0.0 for f in self.files}
        self.busy = {f.name: 0 for f in self.files}

    def _acquire(self):
        with self.cv:
            while True:
                t = time.time()
                best = None
                for f in self.files:
                    q = self.sent[f.name]
                    while q and t - q[0] >= 60:
                        q.popleft()
                    if t >= self.cool[f.name] and len(q) < RPM and self.busy[f.name] < INFLIGHT and (best is None or self.busy[f.name] < self.busy[best.name]):
                        best = f
                if best:
                    self.sent[best.name].append(t)
                    self.busy[best.name] += 1
                    return best
                self.cv.wait(2)

    def call(self, rid, payload):
        p = self.dir / (rid + ".json")
        if p.exists():  # never resent
            return json.loads(p.read_text(encoding="utf-8"))
        mark = self.dir / (rid + ".started")
        while True:
            f = self._acquire()
            alias = "KEY_%s" % f.name[15:-4]
            mark.write_text(now())
            t0 = time.time()
            try:
                status, headers, text = J.http(f, J.BASE + "/chat/completions", payload)
                polls = 0
                while status == 202 and polls < 120:
                    q = headers.get("NVCF-REQID") or headers.get("nvcf-reqid")
                    if not q:
                        break
                    time.sleep(5)
                    polls += 1
                    status, headers, text = J.http(f, J.BASE + "/status/" + q)
            finally:
                with self.cv:
                    self.busy[f.name] -= 1
                    self.cv.notify_all()
            secs = round(time.time() - t0, 2)
            usage = None
            if status == 200:
                try:
                    usage = json.loads(text).get("usage")
                except ValueError:
                    pass
            with self.cv:
                append(self.ledger, dict(ts=now(), id=rid, alias=alias, status=status, seconds=secs, polls=polls, usage=usage))
                if status == 429:
                    ra = headers.get("Retry-After") or headers.get("retry-after")
                    self.cool[f.name] = time.time() + max(COOL, float(ra) if ra and ra.replace(".", "").isdigit() else 0)
                    mark.unlink(missing_ok=True)
                    continue  # rejected before processing: re-queued on another key
            rec = dict(id=rid, status=status, seconds=secs, polls=polls, alias=alias, ts=now(), settings={k: v for k, v in payload.items() if k != "messages"}, raw=text)
            tmp = p.with_suffix(".tmp%d" % threading.get_ident())
            tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, p)
            mark.unlink(missing_ok=True)
            return rec


def reply_of(rec):
    """-> (content, finish_reason, reasoning_chars) or None when not HTTP 200"""
    if rec["status"] != 200:
        return None
    try:
        ch = json.loads(rec["raw"])["choices"][0]
    except (ValueError, KeyError, IndexError):
        return None
    return ch["message"].get("content") or "", ch.get("finish_reason"), ch["message"].get("reasoning_chars")


def ask(pool, rid, msgs, log):
    """one step: up to CALL_ATTEMPTS requests (new id each) until an HTTP 200 answer -> (content, finish) or None (model unavailable)"""
    for a in range(1, CALL_ATTEMPTS + 1):
        rec = pool.call("%s__a%d" % (rid, a), dict(SETTINGS, messages=msgs))
        log.append(dict(id="%s__a%d" % (rid, a), status=rec["status"], seconds=rec["seconds"]))
        r = reply_of(rec)
        if r is not None:
            return r
    return None


# ------------------------------------------------------------------ prompt + actions
TOOLS = """

<TOOLS>
You act through exactly ONE action per reply. End every reply with the action as a JSON object inside <action></action> tags, e.g.
<action>{"tool": "execute_bash", "command": "ls -la"}</action>
Available tools:
1. execute_bash: {"tool": "execute_bash", "command": "<bash command>"} runs the command in a persistent bash shell (the working directory persists between
   commands). Each command is killed after 300 seconds. No interactive input. The network is OFF and no packages can be installed.
2. str_replace_editor: {"tool": "str_replace_editor", "command": "view" | "create" | "str_replace" | "insert" | "undo_edit", "path": "<absolute path>", ...}
   view: optional "view_range": [start, end] (end -1 = to the end of the file); a directory is listed 2 levels deep.
   create: "file_text" (the file must not exist). str_replace: "old_str" (must occur exactly once) and "new_str".
   insert: "insert_line" (the text is inserted after this line; 0 = top) and "new_str". undo_edit: reverts the last edit of "path".
3. think: {"tool": "think", "thought": "..."} logs a thought; it does not change anything.
4. finish: {"tool": "finish", "message": "..."} ends the task when you are done.
</TOOLS>"""
ASK_S = ("Propose your NEXT action. It must be a READ/SEARCH action: a str_replace_editor `view`, or an execute_bash command that only reads or "
         "searches (for example ls, cat, grep, find, sed -n, head, git log / diff / show) and changes no file. Reply with exactly one <action>{json}</action>.")
ASK_T = ("Propose your NEXT action. It must be a TEST-OR-EDIT action: either run tests (an execute_bash command that runs pytest, python -m unittest or a "
         "test script) or edit a file (str_replace_editor create, str_replace or insert). Reply with exactly one <action>{json}</action>.")
ACT = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)
TEST = re.compile(r"pytest|py\.test|unittest|\btox\b|\bnox\b|runtests|run_tests|manage\.py test|make test|python[0-9.]* \S*test\S*\.py")  # v1 / select_v2 TEST
READ_CMDS = {"ls", "cat", "head", "tail", "grep", "egrep", "rg", "find", "sed", "wc", "tree", "file", "stat", "pwd", "git", "awk", "nl", "less", "more",
             "echo", "which", "cd", "python", "python3", "sort", "uniq", "cut", "diff", "realpath", "dirname", "basename", "du", "xargs"}


def is_read(action):
    if action.get("tool") == "str_replace_editor":
        return action.get("command") == "view"
    if action.get("tool") != "execute_bash":
        return False
    c = action.get("command") or ""
    if re.search(r"(?<![0-9&])>(?!&)|\btee\b|-delete\b|-exec\s+rm|\bsed\s+-i|\brm\b|\bmv\b|\bcp\b|\btouch\b|\bmkdir\b|\bchmod\b", re.sub(r"[0-9]?>\s*/dev/null|2>&1", "", c)):
        return False
    for seg in re.split(r"&&|\|\||;|\|", c):
        w = seg.strip().split()
        if not w:
            continue
        if w[0] not in READ_CMDS:
            return False
        if w[0] == "sed" and "-n" not in w:
            return False
        if w[0] == "git" and (len(w) < 2 or w[1] not in ("log", "show", "diff", "status", "grep", "ls-files", "blame")):
            return False
        if w[0] in ("python", "python3") and not (len(w) > 1 and w[1] == "-c" and not re.search(r"open\(|write|remove|unlink|shutil|subprocess|system", seg)):
            return False
    return True


def kind_of(action):
    if action.get("tool") == "str_replace_editor" and action.get("command") in ("create", "str_replace", "insert", "undo_edit"):
        return "edit"
    if action.get("tool") == "execute_bash" and TEST.search(action.get("command") or ""):
        return "test"
    if is_read(action):
        return "read"
    return "other"


def parse_action(content):
    m = ACT.findall(content or "")
    if not m:
        return None
    try:
        a = json.loads(m[-1])
    except ValueError:
        return None
    if not isinstance(a, dict) or a.get("tool") not in ("execute_bash", "str_replace_editor", "think", "finish"):
        return None
    if a["tool"] == "execute_bash" and not isinstance(a.get("command"), str):
        return None
    if a["tool"] == "str_replace_editor" and (a.get("command") not in ("view", "create", "str_replace", "insert", "undo_edit") or not isinstance(a.get("path"), str)):
        return None
    return a


def render(action):
    return "<action>%s</action>" % json.dumps(action, ensure_ascii=False)


def recorded_action(task):
    dec = task["decision"]
    return dict(tool=dec["name"], **{k: v for k, v in dec["arguments"].items() if k not in ("security_risk", "is_input", "timeout")})


def transcript(out, task, state_obs):
    head = json.loads((out / "heads" / ("%s.json" % task["instance_id"])).read_text(encoding="utf-8"))  # system + first user message, cached on Windows
    msgs = [dict(role="system", content=head["system"] + TOOLS), dict(role="user", content=head["user"])]
    for st, o in zip(task["prefix"], state_obs):
        a = dict(tool=st["name"], **{k: v for k, v in st["arguments"].items() if k not in ("security_risk", "is_input", "timeout")})
        msgs += [dict(role="assistant", content=render(a)), dict(role="user", content="OBSERVATION:\n" + trunc(o))]
    return msgs


def fmt_bash(code, out):
    out = out.rstrip("\n")
    if code in (137, -9):
        return out + "\n[The command was killed after %d seconds.]" % CMD_TIMEOUT
    return out + "\n[The command completed with exit code %d.]" % code


class Env:
    """one rollout's workspace: a fresh copy of the decision-point state"""

    def __init__(self, task, state_rec):
        self.d = dname(task)
        self.conda = R.BASE / task["instance_id"] / "conda"
        self.r = R.BASE / "runs" / uuid.uuid4().hex
        shutil.copytree(state_rec["state_dir"], self.r, symlinks=True)
        self.hash = R.ws_hash(self.r / "ws" / self.d)[0]
        self.hist = collections.defaultdict(list)
        self.env_secs = 0.0

    def execute(self, a):
        """-> (observation, info dict with env_failure / exec_error flags)"""
        t0 = time.monotonic()
        try:
            if a["tool"] == "think":
                return "Your thought has been logged.", dict()
            if a["tool"] == "execute_bash":
                with SEM_EXEC:
                    t1 = time.monotonic()
                    R.chown(self.r)
                    code, o, secs = R.run(R.mounts_of(self.r, self.conda, self.d), R.WRAP % a["command"], timeout=CMD_TIMEOUT)
                    self.env_secs += time.monotonic() - t1
                envf = bool(ENV_FAIL.search(o[:500])) and code not in (0,)
                return trunc(fmt_bash(code, o)), dict(exit=code, seconds=secs, env_failure=envf, killed=code in (137, -9))
            t1 = time.monotonic()
            o, err = self.editor(a)
            self.env_secs += time.monotonic() - t1
            return trunc(o), dict(exec_error=err)
        except Exception as e:  # runner bug / infrastructure: environment failure, never the agent's fault
            return "ERROR: internal", dict(env_failure=True, exception="%s: %s" % (type(e).__name__, str(e)[:300]))

    def editor(self, a):
        p, cmd = a["path"], a["command"]
        if cmd == "view":
            o = R.view(a, self.r, self.conda, self.d)
            return (o, False) if o is not None else ("ERROR:\nInvalid `path` parameter: %s. The path is outside /workspace." % p, True)
        h = R.host_path(p, self.r, self.conda, self.d) if p.startswith("/workspace/") else None
        if h is None:
            return "ERROR:\nInvalid `path` parameter: %s. Only files under /workspace can be edited." % p, True
        if cmd == "create":
            if h.exists():
                return "ERROR:\nInvalid `path` parameter: %s. File already exists at: %s. Cannot overwrite files using command `create`." % (p, p), True
            if not isinstance(a.get("file_text"), str):
                return "ERROR:\nParameter `file_text` is required for command: create.", True
            h.parent.mkdir(parents=True, exist_ok=True)
            self.hist[p].append(None)
            h.write_text(a["file_text"], encoding="utf-8")
            R.chown(self.r)
            return "File created successfully at: %s" % p, False
        if not h.is_file():
            return "ERROR:\nInvalid `path` parameter: %s. The path %s does not exist. Please provide a valid path." % (p, p), True
        txt = h.read_text(encoding="utf-8", errors="replace")
        if cmd == "undo_edit":
            if not self.hist[p]:
                return "ERROR:\nNo edit history found for %s." % p, True
            prev = self.hist[p].pop()
            h.unlink() if prev is None else h.write_text(prev, encoding="utf-8")
            return "Last edit to %s undone successfully." % p, False
        if cmd == "str_replace":
            old, new = a.get("old_str"), a.get("new_str") or ""
            if not isinstance(old, str) or not isinstance(new, str):
                return "ERROR:\nParameter `old_str` is required for command: str_replace.", True
            k = txt.count(old)
            if k != 1:
                return ("ERROR:\nNo replacement was performed, old_str `%s` did not appear verbatim in %s." % (old, p) if k == 0 else
                        "ERROR:\nNo replacement was performed. Multiple occurrences of old_str `%s` in %s. Please ensure it is unique." % (old, p)), True
            line = txt[:txt.index(old)].count("\n")
            new_txt = txt.replace(old, new, 1)
        elif cmd == "insert":
            try:
                n = int(a.get("insert_line"))
            except (TypeError, ValueError):
                return "ERROR:\nParameter `insert_line` is required for command: insert.", True
            lines = txt.split("\n")
            if not 0 <= n <= len(lines) or not isinstance(a.get("new_str"), str):
                return "ERROR:\nInvalid `insert_line` parameter: %s. It should be within the range of lines of the file: [0, %d]." % (a.get("insert_line"), len(lines)), True
            new_txt = "\n".join(lines[:n] + a["new_str"].split("\n") + lines[n:])
            line = n
        self.hist[p].append(txt)
        h.write_text(new_txt, encoding="utf-8")
        R.chown(self.r)
        ls = new_txt.split("\n")
        s = max(0, line - 4)
        snip = "\n".join("%6d\t%s" % (i + 1, ls[i]) for i in range(s, min(len(ls), line + 4 + (a.get("new_str") or "").count("\n") + 1)))
        return ("The file %s has been edited. Here's the result of running `cat -n` on a snippet of %s:\n%s\nReview the changes and make sure they are as "
                "expected. Edit the file again if necessary." % (p, p, snip)), False

    def close(self):
        shutil.rmtree(self.r, ignore_errors=True)


# ------------------------------------------------------------------ candidates
def state_of(out, iid):
    return json.loads((out / ("STATE-%s.json" % iid)).read_text())


def candidates(out, ids):
    pool = Pool(out)
    jobs = {}
    for iid in ids:
        task, st = load_task(out, iid), state_of(out, iid)
        msgs = transcript(out, task, st["observations"])
        for slot, q in (("S", ASK_S), ("T", ASK_T)):
            jobs[(iid, slot)] = (task, msgs + [dict(role="user", content=q)])
    res = {}

    def gen(key):
        iid, slot = key
        log, chosen = [], None
        for a in (1, 2):  # pre-registered: up to 2 requests per slot
            rid = "%s__gen%s__try%d" % (iid, slot, a)
            r = ask(pool, rid, jobs[key][1], log)
            if r is None:
                continue
            act = parse_action(r[0])
            k = act and kind_of(act)
            ok = act is not None and ((slot == "S" and k == "read") or (slot == "T" and k in ("test", "edit")))
            log[-1].update(parsed=act is not None, kind=k, finish=r[1], content_tail=(r[0] or "")[-600:])
            if ok:
                chosen = act
                break
        res[key] = (chosen, log)

    th = [threading.Thread(target=gen, args=(k,)) for k in jobs]
    [t.start() for t in th]
    [t.join() for t in th]
    for iid in ids:
        task, st = load_task(out, iid), state_of(out, iid)
        rec = recorded_action(task)
        cands = [dict(cid="%s__R" % iid, slot="R", source="recorded run %s step %d" % (task["trajectory_id"], task["decision"]["index"]), action=rec,
                      kind=task["decision"]["kind"], gen_calls=[])]
        for slot in "ST":
            act, log = res[(iid, slot)]
            cands.append(dict(cid="%s__%s" % (iid, slot), slot=slot, source="GLM-5.3 (z-ai/glm-5.3, glm_max64)", action=act, kind=act and kind_of(act),
                              gen_calls=log, duplicate_of_recorded=act == rec))
        for c in cands:  # dry run on a fresh restore: can the candidate execute?
            if c["action"] is None:
                c.update(valid=False, invalid_reason="no parsable action of the required type in 2 attempts")
            else:
                e = Env(task, st)
                c["restore_hash_ok"] = e.hash == st["workspace_sha256"]
                o, info = e.execute(c["action"])
                e.close()
                c.update(valid=not info.get("exec_error") and not info.get("env_failure"), dry_run=dict(info, observation_head=o[:1500]),
                         invalid_reason="cannot execute: " + o[:200] if info.get("exec_error") else None)
            c.update(instance_id=iid, repo=task["task"]["repo"], base_commit=task["task"]["base_commit"], state_sha256=st["workspace_sha256"],
                     source_dataset="nebius/SWE-rebench + nebius/SWE-rebench-openhands", licence="CC-BY-4.0", split="sandbox-pilot-v1",
                     action_text=render(c["action"]) if c["action"] else None)
            append(out / "CANDIDATES.jsonl", c)
            print(c["cid"], c["kind"], c["valid"], (c["action_text"] or "")[:200])


# ------------------------------------------------------------------ rollouts + evaluation
def test_files(patch):
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", patch, re.M)) | set(re.findall(r"^diff --git a/\S+ b/(\S+)", patch, re.M)))


def evaluate(task, env):
    T = task["task"]
    ic, d = T["install_config"], env.d
    f2p, p2p_all = T["FAIL_TO_PASS"], T["PASS_TO_PASS"]
    p2p = p2p_all if len(p2p_all) <= 20 else random.Random(SEED).sample(p2p_all, 20)
    runs = []
    for k in (1, 2):
        r = R.BASE / "runs" / uuid.uuid4().hex
        shutil.copytree(env.r, r, symlinks=True)
        repo = r / "ws" / d
        git = lambda *c, inp=None: subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), *c], input=inp, capture_output=True)
        for f in test_files(T["test_patch"]):
            if git("checkout", T["base_commit"], "--", f).returncode != 0:
                (repo / f).unlink(missing_ok=True)
        ap = git("apply", "-", inp=T["test_patch"].encode())
        if ap.returncode != 0:
            shutil.rmtree(r, ignore_errors=True)
            runs.append(dict(run=k, test_patch_applied=False, err=ap.stderr.decode()[:300]))
            continue
        R.chown(r)
        cmd = "cd /workspace/%s && %s %s" % (d, ic["test_cmd"], " ".join(shlex.quote(x) for x in f2p + p2p))
        with SEM_EVAL:
            code, o, secs = R.run(R.mounts_of(r, env.conda, d), cmd, timeout=1800, mem="4G")
        shutil.rmtree(r, ignore_errors=True)
        st = R.test_ids_status(o, f2p + p2p)
        runs.append(dict(run=k, test_patch_applied=True, exit=code, seconds=secs, killed=code in (137, -9), env_failure=bool(ENV_FAIL.search(o[:500])),
                         f2p_passed=sum(st[i] == "PASSED" for i in f2p), f2p_total=len(f2p), p2p_passed=sum(st[i] == "PASSED" for i in p2p), p2p_total=len(p2p),
                         resolved=all(st[i] == "PASSED" for i in f2p + p2p), status=st, tail=o[-400:]))
    setup_fail = any(not x["test_patch_applied"] for x in runs)
    flaky = not setup_fail and runs[0]["status"] != runs[1]["status"]
    return dict(runs=[{k: v for k, v in x.items() if k != "status"} for x in runs], eval_setup_failure=setup_fail, flaky=flaky,
                eval_timeout=any(x.get("killed") for x in runs), eval_env_failure=any(x.get("env_failure") for x in runs),
                resolved=None if setup_fail or flaky else runs[0]["resolved"] and runs[1]["resolved"])


def rollout(out, pool, task, st, cand, k, base_msgs):
    iid = task["instance_id"]
    rid = "%s__r%d" % (cand["cid"], k)
    t0 = time.monotonic()
    rec = dict(rollout_id=rid, cid=cand["cid"], instance_id=iid, slot=cand["slot"], kind=cand["kind"], rollout=k, candidate_valid=cand["valid"], started=now())
    calls, steps, env = [], [], None
    try:
        if not R.disk_ok("rollout " + rid, out, iid):
            rec.update(env_failure="DISK_GUARD_ABORT")
            return rec
        env = Env(task, st)
        rec["restore_hash"], rec["restore_hash_ok"] = env.hash, env.hash == st["workspace_sha256"]
        if not rec["restore_hash_ok"]:
            rec.update(env_failure="decision-point hash mismatch at restore")
            return rec
        o, info = env.execute(cand["action"])
        rec["candidate_execution"] = dict(info, observation_head=o[:600])
        if info.get("env_failure"):
            rec.update(env_failure="sandbox failure executing the candidate")
            return rec
        msgs = base_msgs + [dict(role="assistant", content=render(cand["action"])), dict(role="user", content="OBSERVATION:\n" + o)]
        end = "step_budget"
        for s in range(1, STEPS + 1):
            if env.env_secs > ENV_CAP:
                end = "timeout_env_cap"
                break
            if time.monotonic() - t0 > WALL_CEIL:
                end = "timeout_wall_ceiling"
                break
            r = ask(pool, "%s__s%02d" % (rid, s), msgs, calls)
            if r is None:
                end = "model_unavailable"
                rec["env_failure"] = "model unavailable after %d attempts" % CALL_ATTEMPTS
                break
            content, fin, _ = r
            a = parse_action(content)
            if a is None:
                o, info = "ERROR: no valid action found. End your reply with exactly one <action>{json}</action> using one of the tools.", dict(parse_error=True)
            elif a["tool"] == "finish":
                steps.append(dict(step=s, action=a, finish=fin))
                end = "finish"
                break
            else:
                o, info = env.execute(a)
                if info.get("env_failure"):
                    rec["env_failure"] = "sandbox failure at step %d: %s" % (s, info.get("exception") or o[:200])
            steps.append(dict(step=s, action=a, finish=fin, info={k_: v for k_, v in info.items()}, observation_head=o[:400]))
            if rec.get("env_failure"):
                end = "env_failure"
                break
            msgs += [dict(role="assistant", content=content or ""), dict(role="user", content="OBSERVATION:\n" + o)]
        else:
            end = "step_budget"
        if env.env_secs > ENV_CAP and end != "timeout_env_cap":
            end = "timeout_env_cap"
        rec["end"] = end
        rec["timeout"] = end.startswith("timeout")
        rec["final_hash"] = R.ws_hash(env.r / "ws" / env.d)[0]
        rec["git_diff_stat"] = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(env.r / "ws" / env.d), "diff", "--stat"], capture_output=True, text=True).stdout[-800:]
        if not rec.get("env_failure"):
            if not R.disk_ok("eval " + rid, out, iid):
                rec["env_failure"] = "DISK_GUARD_ABORT"
            else:
                rec["evaluation"] = evaluate(task, env)
                if rec["evaluation"]["eval_env_failure"]:
                    rec["env_failure"] = "sandbox failure during evaluation"
        return rec
    except Exception as e:
        rec["env_failure"] = "runner exception: %s: %s" % (type(e).__name__, str(e)[:300])
        return rec
    finally:
        if env:
            env.close()
        ev = rec.get("evaluation") or {}
        rec.update(steps_used=len(steps), model_calls=len(calls), model_seconds=round(sum(c["seconds"] for c in calls), 1),
                   env_seconds=round(env.env_secs, 1) if env else 0, wall_seconds=round(time.monotonic() - t0, 1), ended=now(),
                   env_failure=rec.get("env_failure"), timeout=rec.get("timeout", False),
                   setup_failure=ev.get("eval_setup_failure", False), flaky=ev.get("flaky", False), resolved=ev.get("resolved"))
        rec["clean"] = not (rec["env_failure"] or rec["timeout"] or rec["setup_failure"] or rec["flaky"] or ev.get("eval_timeout")) and rec["resolved"] is not None
        (out / "TRANSCRIPTS").mkdir(exist_ok=True)
        (out / "TRANSCRIPTS" / (rid + ".json")).write_text(json.dumps(dict(rollout_id=rid, candidate=cand["action"], steps=steps, calls=calls), ensure_ascii=False, indent=1))
        append(out / "ROLLOUTS.jsonl", rec)
        print(now(), rid, rec.get("end"), "resolved=%s" % rec["resolved"], "env_fail=%s" % rec["env_failure"], "calls=%d" % rec["model_calls"],
              "wall=%.0f" % rec["wall_seconds"], flush=True)


def rollouts(out, ids):
    pool = Pool(out)
    cands = [json.loads(l) for l in open(out / "CANDIDATES.jsonl")]
    done = {json.loads(l)["rollout_id"] for l in open(out / "ROLLOUTS.jsonl")} if (out / "ROLLOUTS.jsonl").exists() else set()
    th = []
    for iid in ids:
        task, st = load_task(out, iid), state_of(out, iid)
        base = transcript(out, task, st["observations"])
        for c in (c for c in cands if c["instance_id"] == iid and c["valid"]):
            for k in (1, 2, 3):
                if "%s__r%d" % (c["cid"], k) not in done:
                    th.append(threading.Thread(target=rollout, args=(out, pool, task, st, c, k, base)))
    assert len(th) + len(done) <= 36, "scope cap"
    print("starting %d rollouts" % len(th), flush=True)
    [t.start() for t in th]
    [t.join() for t in th]


def labels(out, ids):
    cands = {c["cid"]: c for c in (json.loads(l) for l in open(out / "CANDIDATES.jsonl"))}
    rows = [json.loads(l) for l in open(out / "ROLLOUTS.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cid"]].append(r)
    path = out / "LABELS.jsonl"
    assert not path.exists(), "create-only"
    for iid in ids:
        cs = [cands["%s__%s" % (iid, s)] for s in "RST"]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            A, B = cs[a], cs[b]
            ra, rb = by[A["cid"]], by[B["cid"]]
            pa, pb = sum(bool(x["resolved"]) for x in ra), sum(bool(x["resolved"]) for x in rb)
            non_edit = A["kind"] in ("read", "test") or B["kind"] in ("read", "test")
            if not (A["valid"] and B["valid"]):
                lab, why = "INVALID", "invalid candidate"
            elif len(ra) != 3 or len(rb) != 3 or not all(x["clean"] for x in ra + rb):
                lab, why = "UNKNOWN", "not all 6 rollouts clean: " + ", ".join(sorted({x.get("end") or str(x.get("env_failure")) for x in ra + rb if not x["clean"]}))
            elif abs(pa - pb) >= 2:
                lab, why = ("A_BETTER" if pa > pb else "B_BETTER"), "r_a=%d/3 r_b=%d/3" % (pa, pb)
            elif (pa, pb) in ((3, 3), (0, 0)):
                lab, why = "EQUAL_ON_TESTED_OUTCOME", "both %d/3" % pa
            else:
                lab, why = "UNKNOWN", "r_a=%d/3 r_b=%d/3 (difference < 2/3)" % (pa, pb)
            append(path, dict(label_id="%s__%s_vs_%s" % (iid, A["slot"], B["slot"]), instance_id=iid, repo=A["repo"], base_commit=A["base_commit"],
                              state_sha256=A["state_sha256"], a=A["cid"], b=B["cid"], a_kind=A["kind"], b_kind=B["kind"], a_action=A["action_text"],
                              b_action=B["action_text"], r_a=pa, r_b=pb, label=lab, reason=why, non_edit_comparison=non_edit,
                              better_worse=lab in ("A_BETTER", "B_BETTER"),
                              provenance="sandbox rollouts (3 per candidate, GLM-5.3 continuation, 15 steps, test_patch at evaluation only, 2 evaluation runs)",
                              source="nebius/SWE-rebench + nebius/SWE-rebench-openhands", licence="CC-BY-4.0", split="sandbox-pilot-v1"))
    print(open(path).read())


if __name__ == "__main__":
    _selftest()
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd == "selftest":
        print("selftest ok")
    elif cmd == "v3recheck":
        v3recheck(out)
    elif cmd == "qualify":
        qualify(out, sys.argv[3])
    elif cmd == "candidates":
        candidates(out, sys.argv[3].split(","))
    elif cmd == "rollouts":
        rollouts(out, sys.argv[3].split(","))
    elif cmd == "labels":
        labels(out, sys.argv[3].split(","))
    elif cmd == "cleanup":
        R.cleanup(sys.argv[3], out)
