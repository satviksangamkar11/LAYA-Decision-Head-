"""K1: bounded integrity inventory of CodeTraceBench (NJU-LINK/CodeTraceBench, MIT) on Kaggle CPU. PRIVATE kernel, internet on.

User approvals (chat, 2026-10-14): fetch ONLY bench_artifacts/full/*.tar.zst (~0.45 GB) + the `full` manifest parquet, inside this kernel (nothing to the PC);
no swe_raw/. CodeTracer parsers pinned to commit 2d302191dd07e7c0c2da6f7a5e9451c7cbb62d34. Counts / ids / hashes / masked snippets only: NO raw
trajectory text is written to /kaggle/working. Archives are never executed or imported; extraction is Python tarfile with filter="data" plus own checks.

Per run: format, completeness, candidate unit counts, evidence level (MISSING_SOURCE, COUNT_MISMATCH, COUNT_MATCH_ONLY, REF_CONSISTENT, REF_CONFLICT,
MULTI_UNIT_AMBIGUOUS). REF_CONSISTENT is NOT independent verification: manifest refs were assigned positionally (backfill_step_refs.py).
MODE = "smoke" (5 runs per format + the 7 earlier sample ids) or "full". EARLY-STOP (user, 2026-10-14): formats listed in BLOCKED (decided from the smoke
result: 0 runs at REF_CONSISTENT, no independent anchor) get only cheap statistics (unit counts, completeness, manifest numbers) in the full run.
"""
import ast
import collections
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import tarfile
import time
import types
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

MODE = "full"           # "smoke" | "full"
# Early-stop (user, 2026-10-14): formats with 0 REF_CONSISTENT runs in the 30-run smoke (results/raw/ctb-k1-inventory-20261013T195552Z-smoke) and no independent anchor.
BLOCKED = ["SWE-agent|swe", "mini-SWE-agent|swe", "OpenHands|swe", "OpenHands|terminal", "Terminus2|terminal"]

REPO = "NJU-LINK/CodeTraceBench"
PIN = "aa213b84ffb6690fc37ca15766d6ca174ec36d4d"
CT_COMMIT = "2d302191dd07e7c0c2da6f7a5e9451c7cbb62d34"
CT_FILES = ["models/trajectory.py", "skills/seed/miniswe/parser.py", "skills/seed/openhands/parser.py",
            "skills/seed/terminus2/parser.py", "scripts/backfill_step_refs.py"]
KNOWN = ["ilp-solver", "polyglot-rust-c", "hf-train-lora-adapter", "simple-sheets-put", "dayjs-1725", "zstd-1532", "django-16485"]
MAIN_FORMATS = ["SWE-agent|swe", "mini-SWE-agent|swe", "OpenHands|swe", "mini-SWE-agent|terminal", "OpenHands|terminal", "Terminus2|terminal"]
OUT, WORK = Path("/kaggle/working"), Path("/tmp/ctb")
PER_FILE_CAP, PER_ARCHIVE_CAP, MAX_MEMBERS, SCAN_CAP = 600 << 20, 3 << 30, 300_000, 50 << 20
UA = {"User-Agent": "ctb-k1-inventory/1.0"}

PATTERNS = {  # tools/secret_scan_parquet_v1.py + JWT; anthropic before openai so one match is one category
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "github_token": r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}",
    "anthropic_key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "openai_style_key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}",
    "slack_token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "huggingface_token": r"\bhf_[A-Za-z0-9]{30,}",
    "nvidia_key": r"\bnvapi-[A-Za-z0-9_-]{20,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "private_key_block": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
    "stripe_live_key": r"\b(?:sk|rk)_live_[0-9a-zA-Z]{20,}",
    "jwt": r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.",
}
SECRET_RX = re.compile("|".join("(?P<%s>%s)" % kv for kv in PATTERNS.items()))
SECRET_NAMES = re.compile(r"^(\.jwt_secret|\.env(\..*)?|id_rsa|id_rsa\.pub|.*\.pem)$")


# ---------------------------------------------------------------- network
def http(url, tries=7, stream_to=None):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                if stream_to is None:
                    return r.read(), dict(r.headers)
                h, n = hashlib.sha256(), 0
                with open(stream_to, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b), h.update(b)
                        n += len(b)
                return (h.hexdigest(), n), dict(r.headers)
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                raise
            time.sleep(min(60, 2 ** i * (3 if e.code == 429 else 1)))
        except Exception as e:  # noqa
            last = e
            time.sleep(min(60, 2 ** i))
    raise last


def tree(path):
    url = "https://huggingface.co/api/datasets/%s/tree/%s%s?limit=1000" % (REPO, PIN, ("/" + path) if path else "")
    out = []
    while url:
        body, hdr = http(url)
        out += json.loads(body)
        m = re.search(r'<([^>]+)>;\s*rel="next"', hdr.get("Link") or hdr.get("link") or "")
        url = m.group(1) if m else None
    return out


# ---------------------------------------------------------------- upstream parsers (shim as tools/ctb_align_check_v1.py)
def load_upstream(src):
    pkg = types.ModuleType("codetracer")
    pkg.__path__ = []
    sys.modules["codetracer"] = pkg
    spec = importlib.util.spec_from_file_location("codetracer.models", Path(src) / "models/trajectory.py")
    models = importlib.util.module_from_spec(spec)
    sys.modules["codetracer.models"] = models
    spec.loader.exec_module(models)
    mods = {}
    for name in ("miniswe", "openhands", "terminus2"):
        s = importlib.util.spec_from_file_location("ct_%s" % name, Path(src) / "skills/seed" / name / "parser.py")
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        mods[name] = m
    return mods


UP = None


def init_worker(src):
    global UP
    UP = load_upstream(src)


# ---------------------------------------------------------------- text normalisation
BASH = re.compile(r"```bash\s*\n(.*?)\n?```", re.S)


def core(s):
    t = (s if isinstance(s, str) else json.dumps(s, sort_keys=True, default=str)).replace("\\n", "\n")
    m = BASH.search(t)
    return re.sub(r"\s+", " ", m.group(1) if m else t).strip()


def match(ref, alts):
    for a in alts:
        if a == ref or (min(len(a), len(ref)) >= 20 and (ref in a or a in ref)):
            return True
    return False


def loose_json(c):
    for t in (c, c.strip().rstrip(",")):
        try:
            return json.loads(t)
        except ValueError:
            pass
    return None


def oh_core(ev):
    c = UP["openhands"]._extract_cmd(ev)
    if c:
        return c
    args = ev.get("args") or {}
    if isinstance(args, dict):
        c = args.get("command") or args.get("code")
        if isinstance(c, str) and c.strip():
            return c
    return ev.get("message") or json.dumps(args, sort_keys=True, default=str)


def ref_core(ref):
    c = (ref or {}).get("content") or ""
    if not isinstance(c, str) or not c.strip():
        return ""
    o = loose_json(c)
    if isinstance(o, dict):
        if "action" in o or "tool_call_metadata" in o or ("args" in o and "id" in o):
            return core(oh_core(o))
        if "role" in o:
            return core(o.get("content") or "")
        if "commands" in o and isinstance(o["commands"], list):
            return core("\n".join(c2.get("keystrokes", "") if isinstance(c2, dict) else str(c2) for c2 in o["commands"]))
    return core(c)


# ---------------------------------------------------------------- safe extraction + secret scan
def safe_extract(arc, dest):
    st = dict(members=0, files=0, dirs=0, bytes=0, rejected=collections.Counter(), truncated=False, error=None)
    try:
        with pa.input_stream(str(arc), compression="zstd") as s, tarfile.open(fileobj=s, mode="r|") as tf:
            for m in tf:
                st["members"] += 1
                parts = PurePosixPath(m.name).parts
                if st["members"] > MAX_MEMBERS or st["bytes"] > PER_ARCHIVE_CAP:
                    st["truncated"] = True
                    break
                if m.name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", m.name) or ".." in parts:
                    st["rejected"]["unsafe_path"] += 1
                elif m.issym() or m.islnk():
                    st["rejected"]["link"] += 1
                elif not (m.isfile() or m.isdir()):
                    st["rejected"]["special_device_or_other"] += 1
                elif m.isfile() and m.size > PER_FILE_CAP:
                    st["rejected"]["file_over_cap"] += 1
                else:
                    tf.extract(m, dest, filter="data")
                    if m.isfile():
                        st["files"] += 1
                        st["bytes"] += m.size
                    else:
                        st["dirs"] += 1
    except Exception as e:  # noqa
        st["error"] = "%s: %s" % (type(e).__name__, str(e)[:150])
    st["rejected"] = dict(st["rejected"])
    return st


def mask(s):
    return s[:4] + "..." + "[len %d]" % len(s)


def scan_dir(root):
    r = dict(files=0, text_scanned=0, binary_skipped=0, scan_truncated=0, hits=collections.Counter(), files_with_hits=0, name_hits=[], samples=[])
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        r["files"] += 1
        rel = str(p.relative_to(root))
        if SECRET_NAMES.match(p.name):
            r["name_hits"].append(rel)
        try:
            with open(p, "rb") as f:
                b = f.read(SCAN_CAP + 1)
        except OSError:
            continue
        if b"\0" in b[:4096]:
            r["binary_skipped"] += 1
            continue
        if len(b) > SCAN_CAP:
            r["scan_truncated"] += 1
            b = b[:SCAN_CAP]
        r["text_scanned"] += 1
        hit = False
        for m in SECRET_RX.finditer(b.decode("latin-1")):
            hit = True
            r["hits"][m.lastgroup] += 1
            if len(r["samples"]) < 3:
                r["samples"].append(dict(pattern=m.lastgroup, file=rel, masked=mask(m.group(0))))
        r["files_with_hits"] += hit
    r["hits"] = dict(r["hits"])
    return r


# ---------------------------------------------------------------- candidate units
def u(steps):  # steps: dict step_id -> list of normalised alternative action strings
    return (len(steps), steps)


def commands_txt(run):
    p = run / "commands.txt"
    if not p.exists():
        return None, None
    s1, s2 = {}, {}
    for line in p.read_text("utf-8", errors="replace").splitlines():
        try:
            v = ast.literal_eval(line)
        except Exception:  # noqa
            continue
        if isinstance(v, str):
            t = v.replace("\r\n", "\n").replace("\r", "\n")
            t = t[:-1] if t.endswith("\n") else t
            if t and t != "C-d":
                s1[len(s1) + 1] = [core(t)]
        if isinstance(v, list) and v and v[-1] == "Enter":
            c = "".join(str(x) for x in v[:-1] if isinstance(x, (str, int, float))).strip()
            if c:
                s2[len(s2) + 1] = [core(c)]
    return u(s1), u(s2)


def msg_text(m):
    c = m.get("content")
    if isinstance(c, list):
        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else ""


def find_tool_calls(o, depth=0):
    if depth > 6:
        return []
    if isinstance(o, dict):
        if isinstance(o.get("tool_calls"), list):
            return o["tool_calls"]
        for k in ("choices", "message", "response"):
            if k in o:
                r = find_tool_calls(o[k], depth + 1)
                if r:
                    return r
    elif isinstance(o, list):
        for x in o:
            r = find_tool_calls(x, depth + 1)
            if r:
                return r
    return []


def tc_alts(tcs):
    alts = []
    for tc in tcs:
        fn = (tc or {}).get("function") or {}
        a = fn.get("arguments") or ""
        o = loose_json(a) if isinstance(a, str) else None
        if isinstance(o, dict) and isinstance(o.get("command"), str):
            alts.append(core(o["command"]))
        alts.append(core("%s %s" % (fn.get("name", ""), a)))
    return alts


def units_swe(root, deep):
    """SWE-style raw runs. Returns (format_name, trajectory_present, units, extra)."""
    units, extra = {}, {}
    tj = sorted(root.rglob("*.traj.json"))
    t = sorted(p for p in root.rglob("*.traj") if p.is_file())
    calls = sorted(root.rglob("tensorblock__*.json"), key=lambda p: (float(p.stem.rsplit("-", 1)[-1]) if re.search(r"-[\d.]+$", p.stem) else 0, p.name))
    extra.update(n_traj_json=len(tj), n_traj=len(t), n_llm_call_files=len(calls))
    if tj:
        msgs = (json.loads(tj[0].read_text("utf-8", errors="replace")).get("messages") or [])
        asst = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
        units["swe_traj_json_assistant"] = u({i + 1: [core(msg_text(m))] for i, m in enumerate(asst)} if deep else {i + 1: [] for i in range(len(asst))})
        return "miniswe_traj_json", True, units, extra
    if t:
        tr = json.loads(t[0].read_text("utf-8", errors="replace")).get("trajectory") or []
        units["swe_traj_entries"] = u({i + 1: [core(s.get("action") or "")] for i, s in enumerate(tr)} if deep else {i + 1: [] for i in range(len(tr))})
        return "sweagent_traj", True, units, extra
    if calls:
        allc, withtc = {}, {}
        for p in calls:
            d = json.loads(p.read_text("utf-8", errors="replace"))
            tcs = find_tool_calls(d.get("response"))
            alts = tc_alts(tcs) if deep else []
            allc[len(allc) + 1] = alts
            if tcs:
                withtc[len(withtc) + 1] = alts
        units["oh_llm_call_files"] = u(allc)
        units["oh_llm_call_files_with_toolcall"] = u(withtc)
        return "openhands_llm_calls", True, units, extra
    return None, False, units, extra


def units_terminal(agent, root, deep):
    units, extra = {}, {}
    trial = [d for d in sorted(root.rglob("*")) if d.is_dir() and ((d / "agent-logs").is_dir() or (d / "sessions").is_dir())]
    run = trial[0] if trial else root
    extra.update(n_trial_dirs=len(trial), has_agent_logs=(run / "agent-logs").is_dir(), has_sessions=(run / "sessions").is_dir())
    c1, c2 = commands_txt(run)
    if c1:
        units["commands_txt_str"], units["commands_txt_enter_lists"] = c1, c2
    present = False
    if agent == "mini-SWE-agent":
        m = UP["miniswe"]
        tjp = run / "agent-logs" / "mini.traj.json"
        extra["mini_traj_json"] = tjp.exists()
        extra["agent_log"] = (run / "sessions" / "agent.log").exists()
        present = tjp.exists()  # "empty agent-logs" = no trajectory file (user brief); commands.txt is still counted as a unit
        if tjp.exists():
            msgs = (json.loads(tjp.read_text("utf-8", errors="replace")).get("messages") or [])
            asst = [x for x in msgs if isinstance(x, dict) and x.get("role") == "assistant"]
            units["mini_traj_assistant_msgs"] = u({i + 1: [core(msg_text(x))] for i, x in enumerate(asst)} if deep else {i + 1: [] for i in range(len(asst))})
        if tjp.exists():
            src, steps = "traj_json", m._steps_from_traj_json(tjp, run)
        else:
            src, steps = None, []
        if not steps and (run / "sessions" / "agent.log").exists():
            src, steps = "agent_log", m._steps_from_agent_log(run / "sessions" / "agent.log", run)
        if not steps:
            src, steps = "commands_txt", m._steps_from_commands_txt(run / "commands.txt", run)
        extra["upstream_source"] = src
        units["upstream_parser"] = u({s.step_id: [core(s.action)] for s in steps} if deep else {s.step_id: [] for s in steps})
    elif agent == "OpenHands":
        m = UP["openhands"]
        ss = run / "sessions" / "sessions"
        events = m._load_events_sharded(ss) if ss.is_dir() else (m._load_events_flat(run / "sessions") if (run / "sessions").is_dir() else [])
        present = bool(events)
        if events:
            ids = [e for e in events if isinstance(e.get("id"), int) and e["id"] != 0]
            allact = [e for e in ids if e.get("source") == "agent" and "action" in e and e.get("action") != "system"]
            extra["oh_action_hist"] = dict(collections.Counter(e["action"] for e in allact))
            extra["n_events"] = len(events)
            units["oh_all_agent_actions"] = u({i + 1: [core(oh_core(e))] for i, e in enumerate(allact)} if deep else {i + 1: [] for i in range(len(allact))})
            obs_cause = {e["cause"] for e in events if e.get("observation") == "run" and isinstance(e.get("cause"), int)}
            runs_ = []
            for e in sorted(ids, key=lambda e: e["id"]):
                a = e.get("args")
                if e.get("action") == "run" and isinstance(a, dict) and isinstance(a.get("command"), str) and a["command"].strip() and e["id"] in obs_cause:
                    runs_.append(e)
            units["oh_run_only_backfill"] = u({i + 1: [core(e["args"]["command"])] for i, e in enumerate(runs_)} if deep else {i + 1: [] for i in range(len(runs_))})
            steps = m.parser.parse(run).steps
            units["upstream_parser"] = u({s.step_id: [core(s.action)] for s in steps} if deep else {s.step_id: [] for s in steps})
    elif agent == "Terminus2":
        m = UP["terminus2"]
        logs = run / "agent-logs"
        eps = []
        if logs.is_dir():
            for p in logs.iterdir():
                if p.is_dir() and p.name.startswith("episode-") and (p / "response.txt").exists():
                    try:
                        eps.append((int(p.name.split("-", 1)[1]), p))
                    except ValueError:
                        pass
        eps.sort()
        present = bool(eps)
        ep_steps, cmd_steps, bad = {}, {}, 0
        for i, (_, p) in enumerate(eps):
            txt = (p / "response.txt").read_text("utf-8", errors="replace")
            try:
                o = json.loads(txt)
                cmds = [c.get("keystrokes", "") if isinstance(c, dict) else str(c) for c in (o.get("commands") or [])]
            except Exception:  # noqa
                cmds, bad = [], bad + 1
            alts = [core(c) for c in cmds if c] + [core(m._extract_action_text(txt))] if deep else []
            ep_steps[i + 1] = alts
            for c in cmds:
                cmd_steps[len(cmd_steps) + 1] = [core(c)] if deep else []
        extra["response_parse_failures"] = bad
        if present:
            units["t2_episodes"] = u(ep_steps)
            units["t2_response_commands"] = u(cmd_steps)
            steps = m.parser.parse(run).steps
            units["upstream_parser"] = u({s.step_id: [core(s.action)] for s in steps} if deep else {s.step_id: [] for s in steps})
    else:  # SWE-agent terminal-style: no upstream parser; only commands.txt
        present = c1 is not None
    return present, units, extra


# ---------------------------------------------------------------- evidence
def judge(units, step_count, flag_ids, refs, deep):
    cnt_eq = [n for n, (c, _) in units.items() if c == step_count and c > 0]
    out = dict(matching_units=cnt_eq)
    if not cnt_eq:
        out["evidence"] = "COUNT_MISMATCH"
        return out
    if not deep:
        out["evidence"] = "BLOCKED"
        return out
    # multi-unit ambiguity at flagged ids
    amb = False
    for s in flag_ids:
        for i in range(len(cnt_eq)):
            for j in range(i + 1, len(cnt_eq)):
                a, b = units[cnt_eq[i]][1].get(s, []), units[cnt_eq[j]][1].get(s, [])
                if a and b and not any(match(x, b) for x in a):
                    amb = True
    out["units_differ_at_flagged"] = amb
    steps = units[cnt_eq[0]][1]
    ok = mism = ambig = 0
    for s in flag_ids:
        r = refs.get(s)
        if not r:
            continue
        ids = [k for k, alts in steps.items() if match(r, alts)]
        if ids == [s]:
            ok += 1
        elif s in ids:
            ambig += 1
        else:
            mism += 1
    out.update(ref_ok=ok, ref_mismatch=mism, ref_ambiguous=ambig, ref_checkable=ok + mism + ambig)
    if amb:
        out["evidence"] = "MULTI_UNIT_AMBIGUOUS"
    elif mism or ambig:
        out["evidence"] = "REF_CONFLICT"
    elif flag_ids and out["ref_checkable"] == len(flag_ids):
        out["evidence"] = "REF_CONSISTENT"
    else:
        out["evidence"] = "COUNT_MATCH_ONLY"
        out["partial_refs"] = bool(out["ref_checkable"])
    return out


# ---------------------------------------------------------------- one run
def process(job):
    t0 = time.time()
    row, blocked = job
    tid, agent = row["traj_id"], row["agent"]
    rec = dict(traj_id=tid, agent=agent, artifact_path=row["artifact_path"], manifest_step_count=row["step_count"], solved=row["solved"], source_kind=row["source_kind"])
    flag = row["flag"]
    rec.update(n_incorrect=len(flag["incorrect"]), n_unuseful=len(flag["unuseful"]), n_flagged=len(flag["ids"]), max_flagged_id=max(flag["ids"], default=0),
               max_stage_end=row["max_stage_end"], n_flagged_with_ref=len(flag["refs_raw"]), step_entries_not_in_id_lists=flag["anomaly"])
    sec = None
    try:
        if not row["artifact_path"] or not row["in_tree"]:
            rec.update(fmt="%s|%s" % (agent, "swe" if row["source_kind"] == "swe" else "terminal" if row["source_kind"] == "terminal" else "unknown"),
                       evidence="MISSING_SOURCE", missing_reason="archive_absent")
            return rec, None
        name = row["artifact_path"].split("/")[-1]
        arc, dest = WORK / "dl" / name, WORK / "x" / tid
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        (digest, nbytes), _ = http("https://huggingface.co/datasets/%s/resolve/%s/%s" % (REPO, PIN, row["artifact_path"]), stream_to=arc)
        rec.update(dl_bytes=nbytes, dl_sha256=digest, listed_bytes=row["listed_size"], size_ok=nbytes == row["listed_size"], sha_ok=digest == row["listed_oid"])
        ex = safe_extract(arc, dest)
        arc.unlink(missing_ok=True)
        rec.update(extract={k: v for k, v in ex.items()})
        sec = scan_dir(dest)
        rec.update(files_extracted=sec["files"])
        swe_marker = any(dest.rglob("swe_raw"))
        deep = tid_deep(row, blocked)
        rec["blocked"] = not deep
        if ex["error"] or ex["files"] == 0:
            rec.update(fmt="%s|%s" % (agent, "swe" if swe_marker or row["source_kind"] == "swe" else "terminal"), evidence="MISSING_SOURCE",
                       missing_reason="extract_error" if ex["error"] else "stub_no_files")
            return rec, sec
        if swe_marker or row["source_kind"] == "swe":
            fmt_name, present, units, extra = units_swe(dest, deep)
            kind = "swe"
        else:
            fmt_name = None
            present, units, extra = units_terminal(agent, dest, deep)
            kind = "terminal"
        rec["fmt"] = "%s|%s" % (agent, kind)
        rec["raw_format"] = fmt_name
        rec.update(extra)
        rec["units"] = {k: v[0] for k, v in units.items()}
        rec["units_equal_step_count"] = [k for k, v in units.items() if v[0] == row["step_count"] and v[0] > 0]
        rec["flagged_exceeds_units"] = {k: max(flag["ids"], default=0) > v[0] for k, v in units.items()}
        if not present:
            rec.update(evidence="MISSING_SOURCE", missing_reason="no_trajectory_file")
            return rec, sec
        refs = {s: ref_core({"content": c}) for s, c in flag["refs_raw"].items()} if deep else {}
        rec.update(judge(units, row["step_count"], flag["ids"], refs, deep))
    except Exception as e:  # noqa
        rec.update(evidence="FAILED", failure="%s: %s" % (type(e).__name__, str(e)[:200]), fmt=rec.get("fmt") or "%s|?" % agent)
    finally:
        shutil.rmtree(WORK / "x" / tid, ignore_errors=True)
        (WORK / "dl" / ((row["artifact_path"] or "").split("/")[-1] or "_")).unlink(missing_ok=True)
        rec["seconds"] = round(time.time() - t0, 2)
    return rec, sec


def tid_deep(row, blocked):
    kind = "swe" if row["source_kind"] == "swe" else "terminal"
    return "%s|%s" % (row["agent"], kind) not in blocked


# ---------------------------------------------------------------- main
def flags_of(r):
    st = r["incorrect_stages"]
    st = json.loads(st) if isinstance(st, str) else st or []
    inc, unu, refs, entry_ids = set(), set(), {}, set()
    for g in st:
        inc |= set(g.get("incorrect_step_ids") or [])
        unu |= set(g.get("unuseful_step_ids") or [])
        for s in g.get("steps") or []:
            entry_ids.add(s["step_id"])
            labs = s.get("labels") or []
            if "incorrect" in labs:
                inc.add(s["step_id"])
            if "unuseful" in labs:
                unu.add(s["step_id"])
            ref = s.get("action_ref") or {}
            if (ref.get("content") or "").strip():
                refs[s["step_id"]] = ref["content"]
    ids = inc | unu
    return dict(incorrect=sorted(inc), unuseful=sorted(unu), ids=sorted(ids), refs_raw=refs, anomaly=len(entry_ids - (inc | unu))), \
        [s.get("labels") for g in st for s in g.get("steps") or []], [sorted(s.keys()) for g in st for s in (g.get("steps") or [])[:1]]


def main():
    T0 = time.time()
    assert hasattr(tarfile, "data_filter"), "tarfile.data_filter missing in this Python"
    (WORK / "dl").mkdir(parents=True, exist_ok=True)
    (WORK / "x").mkdir(parents=True, exist_ok=True)
    inv = dict(mode=MODE, blocked_formats=BLOCKED, python=sys.version, repo=REPO, pinned_revision=PIN, codetracer_commit=CT_COMMIT)
    api, _ = http("https://huggingface.co/api/datasets/%s" % REPO)
    inv["api_head_sha_at_run"] = json.loads(api).get("sha")
    top = tree("")
    inv["top_level_entries"] = [(e["type"], e["path"]) for e in top]
    inv["annotation_dir_in_repo"] = any(e["path"].startswith("agent_failure_analysis") for e in top)
    fetched = []
    man_entry = [e for e in top if e["path"] == "bench_manifest.full.parquet"][0]
    mp_ = WORK / "manifest.parquet"
    (dg, nb), _ = http("https://huggingface.co/datasets/%s/resolve/%s/bench_manifest.full.parquet" % (REPO, PIN), stream_to=mp_)
    fetched.append(dict(path="bench_manifest.full.parquet", bytes=nb, sha256=dg, listed=(man_entry.get("lfs") or {}).get("size", man_entry.get("size")),
                        listed_oid=(man_entry.get("lfs") or {}).get("oid")))
    arts = {e["path"]: e for e in tree("bench_artifacts/full") if e["type"] == "file"}
    inv["artifact_files_listed"] = len(arts)
    inv["artifact_bytes_listed"] = sum((e.get("lfs") or {}).get("size", e.get("size", 0)) for e in arts.values())
    # CodeTracer sources
    src = WORK / "ct"
    ct_hashes = {}
    for f in CT_FILES:
        b, _ = http("https://raw.githubusercontent.com/NJU-LINK/CodeTracer/%s/src/codetracer/%s" % (CT_COMMIT, f))
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_bytes(b)
        ct_hashes[f] = hashlib.sha256(b).hexdigest()
    inv["codetracer_files_sha256"] = ct_hashes
    # manifest rows
    rows, label_vals, step_keys = [], collections.Counter(), collections.Counter()
    for r in pq.read_table(mp_).to_pylist():
        fl, labs, keys = flags_of(r)
        for l in labs:
            label_vals[json.dumps(l)] += 1
        for k in keys:
            step_keys[",".join(k)] += 1
        sp = r.get("source_relpath")
        ap = r.get("artifact_path")
        e = arts.get(ap) if ap else None
        rows.append(dict(traj_id=r["traj_id"], agent=r["agent"], step_count=r["step_count"], solved=r["solved"], artifact_path=ap, in_tree=e is not None,
                         listed_size=(e.get("lfs") or {}).get("size", e.get("size")) if e else None, listed_oid=(e.get("lfs") or {}).get("oid") if e else None,
                         source_kind="swe" if (sp or "").startswith("swe_raw") else ("terminal" if sp else "none"), flag=fl,
                         max_stage_end=max((s["end_step_id"] for s in (r["stages"] or [])), default=0)))
    inv["manifest_rows"] = len(rows)
    inv["manifest_step_entry_label_values"] = dict(label_vals)
    inv["manifest_step_entry_key_sets"] = dict(step_keys)
    if MODE == "smoke":
        pick, per = [], collections.Counter()
        for r in sorted(rows, key=lambda r: (not r["flag"]["ids"], not r["flag"]["refs_raw"], r["traj_id"])):
            k = "%s|%s" % (r["agent"], r["source_kind"])
            if k in MAIN_FORMATS and r["in_tree"] and per[k] < 5:
                per[k] += 1
                pick.append(r)
        ids = {r["traj_id"] for r in pick}
        for kw in KNOWN:
            for r in rows:
                if kw in r["traj_id"] and r["traj_id"] not in ids and r["in_tree"]:
                    pick.append(r)
                    ids.add(r["traj_id"])
                    break
        rows = pick
    inv["runs_selected"] = len(rows)
    jobs = [(r, set(BLOCKED)) for r in rows]
    recs, secs, done = [], [], 0
    with mp.get_context("fork").Pool(4, initializer=init_worker, initargs=(str(src),)) as pool:
        for rec, sec in pool.imap_unordered(process, jobs, chunksize=1):
            recs.append(rec)
            secs.append((rec["traj_id"], sec))
            if rec.get("dl_bytes"):
                fetched.append(dict(path=rec["artifact_path"], bytes=rec["dl_bytes"], sha256=rec["dl_sha256"],
                                    listed=rec["listed_bytes"], size_ok=rec["size_ok"], sha_ok=rec["sha_ok"]))
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print("done %d/%d  %.0fs" % (done, len(jobs), time.time() - T0), flush=True)
    recs.sort(key=lambda r: r["traj_id"])
    # ---- aggregate
    table = collections.defaultdict(lambda: collections.defaultdict(lambda: dict(runs=0, flagged_incorrect=0, flagged_unuseful=0)))
    per_fmt_units = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))  # fmt -> unit -> [computed, equals step_count]
    processed, excl, fails = collections.Counter(), collections.Counter(), []
    for r in recs:
        f, ev = r.get("fmt", "?"), r["evidence"]
        c = table[f][ev]
        c["runs"] += 1
        c["flagged_incorrect"] += r["n_incorrect"]
        c["flagged_unuseful"] += r["n_unuseful"]
        processed[f] += 1
        if ev == "MISSING_SOURCE":
            excl["%s: %s" % (f, r.get("missing_reason"))] += 1
        if ev == "FAILED":
            fails.append(dict(traj_id=r["traj_id"], failure=r["failure"]))
        for k, v in (r.get("units") or {}).items():
            per_fmt_units[f][k][0] += 1
            per_fmt_units[f][k][1] += v == r["manifest_step_count"] and v > 0
    sec_tot = dict(files=0, text_scanned=0, binary_skipped=0, scan_truncated=0, files_with_hits=0, runs_with_hits=0, runs_with_secret_filenames=0)
    hits, samples, namehits = collections.Counter(), [], []
    for tid, s in secs:
        if not s:
            continue
        for k in ("files", "text_scanned", "binary_skipped", "scan_truncated", "files_with_hits"):
            sec_tot[k] += s[k]
        sec_tot["runs_with_hits"] += bool(s["hits"])
        sec_tot["runs_with_secret_filenames"] += bool(s["name_hits"])
        hits.update(s["hits"])
        samples += [dict(traj_id=tid, **x) for x in s["samples"]][:2]
        namehits += [dict(traj_id=tid, path=p) for p in s["name_hits"]]
    inv.update(
        table_format_x_evidence={f: dict(v) for f, v in table.items()},
        runs_processed_per_format=dict(processed), exclusions=dict(excl), failures=fails,
        unit_equals_step_count_per_format={f: {k: dict(computed=v[0], equals_step_count=v[1]) for k, v in d.items()} for f, d in per_fmt_units.items()},
        secret_scan=dict(totals=sec_tot, hits_by_pattern=dict(hits), masked_samples=samples[:60], secret_filename_hits=namehits[:200],
                         n_secret_filename_hits=len(namehits), patterns=list(PATTERNS)),
        extraction_rejections=dict(sum((collections.Counter(r.get("extract", {}).get("rejected", {})) for r in recs), collections.Counter())),
        extraction_errors=sum(1 for r in recs if (r.get("extract") or {}).get("error")),
        size_mismatch_runs=sum(1 for r in recs if r.get("size_ok") is False), sha_mismatch_runs=sum(1 for r in recs if r.get("sha_ok") is False),
        fetched_files=fetched, fetched_total_bytes=sum(x["bytes"] for x in fetched), fetched_file_count=len(fetched),
        INDEPENDENTLY_VERIFIED=dict(
            runs=0, flagged_incorrect=0, flagged_unuseful=0,
            reason="No anchor independent of positional ref assignment exists in the published data: the repo has no annotation files (annotation_dir_in_repo=%s), "
                   "manifest step entries carry only the keys listed in manifest_step_entry_key_sets and labels listed in manifest_step_entry_label_values "
                   "(no reasoning/annotation text quoting a command); action_refs come from backfill_step_refs.py (positional) or are raw event/message excerpts." % inv["annotation_dir_in_repo"]),
        wall_seconds=round(time.time() - T0, 1))
    json.dump(inv, open(OUT / "INVENTORY.json", "w"), indent=1)
    with open(OUT / "RUNS.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(dict(table=inv["table_format_x_evidence"], secrets=inv["secret_scan"]["hits_by_pattern"], wall=inv["wall_seconds"]), indent=1))


if __name__ == "__main__":
    main()
