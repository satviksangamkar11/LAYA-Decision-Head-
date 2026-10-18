"""[EL-ARCH-1 F2c / conformance] Audit a finished (or partial) bank run against the architecture: per task PIPELINE_OK or BROKEN(reasons), one run-level verdict. Reads files only (no model, no sandbox,
no network, no GPU); writes CONFORMANCE.json and CONFORMANCE.md (create-only) next to the run.

    python tools/el_bank_conformance.py <run out dir> [--tasks configs/el_bank80_rest_v1.json | configs/el_bank40_v1.json] [--expect-engine mxc] [--legacy-kaggle] [--no-write] [--out DIR]

Run dir layout (what tools/el_bank_local_run.py and kaggle/el_bank40_run_v1.py write): RUN_SUMMARY.json, TIMELINE.jsonl, tasks/<id>/{result.json, journal.jsonl, controller_timeline.json,
gen_calls.json, critic_requests.json, DONE | NOT_TESTED, [states/, exec_results.jsonl]}.

Checks per executed task (a task with DONE; NOT_TESTED tasks are listed with their reason, ERROR / pending / absent tasks are never silently dropped):
  stages      the controller timeline holds the stage rows of the ladder pipeline in order, each with parseable UTC start and end: intake, generate, execute (public examples), l2_generate +
              l2_validate (or an explicit reason: no candidate passed every example), cluster (L1; or < 2 live candidates), discriminate (critic called / pending packets / api_unavailable, or the
              journaled critic_trigger decision), rank, commit_verify (repair rounds between). INVESTIGATE: the function-level ladder has no workspace to inspect, so it is NOT_APPLICABLE
              here (intake routing + oracle tier are journaled) unless a stage named `investigate` exists, then it must come right after intake.
  utc rows    every journal entry has a timeline entry row with a UTC time (count and seq equal); monotonic times never go backwards
  chain       the journal's hash chain recomputes entry by entry and its head equals result.journal_hash (the receipt hash)
  invariants  result.invariants lists I1..I8 as checked and none failed
  l2          L2 verdict INSUFFICIENT with reason 'no valid test parsed' is BROKEN unless the model truly produced nothing (no answer after </think>, ceiling hit); the saved generation is
              re-parsed with parse_checks_v2 and the rejection counts are shown (a parser/prompt mismatch is thereby visible)
  sandbox     the engine recorded for the task (result.engine, else RUN_SUMMARY.engine / sandbox / jail.engine) is the expected backend; no field named engine/backend/sandbox anywhere in the
              task files names WSL or an unsandboxed runner
  policy      thinking policy recorded; profile sha256 and l2_version recorded (waived with --legacy-kaggle, reported as waived)
  public      no hidden / protected marker in any task file; RUN_SUMMARY.public_only_guard.violations == 0
  exec        every record of tasks/<id>/exec_results.jsonl (one sandbox call each) carries the 8 contract fields (exit_code, killed, stdout, stderr, protected_unchanged, pre_tree_sha256,
              post_tree_sha256, persisted); the file must exist (waived with --legacy-kaggle: function-level Kaggle runs recorded spans, not raw sandbox dicts)
  states      if tasks/<id>/states exists: every state file has an index line, belongs to a journaled candidate, and nothing about states appears in the journal (hash-neutral); `selftest()`
              reruns a ladder fixture with and without the capture hook and compares journal hashes
Run level: PIPELINE_COMPLETE = every executed task PIPELINE_OK and no task silently missing; PIPELINE_BROKEN = at least one executed task BROKEN (exact list of task + stage/check);
PIPELINE_PARTIAL = every executed task OK but tasks are missing / ERROR / pending without a reason. Also reported: not_tested (with reason), missing, error."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULT_KEYS = ("exit_code", "killed", "stdout", "stderr", "protected_unchanged", "pre_tree_sha256", "post_tree_sha256", "persisted")
STAGE_ORDER = ("intake", "generate", "execute", "l2_generate", "l2_validate", "cluster", "discriminate", "rank", "commit_verify")
HIDDEN_MARKERS = ("hidden_cases", "protected_cases", "hidden_tests", "expected_hidden", "grader_answers", "/.secrets", "\\.secrets", "NVIDIA_API_KEY")
FORBIDDEN_ENGINE = re.compile(r"wsl|unsandboxed|no[_-]?sandbox|subprocess|host[_-]?exec", re.I)
ENGINE_KEYS = ("engine", "backend", "sandbox", "engine_label", "sandbox_label", "env_label")
GENESIS = hashlib.sha256("evidence-journal-v1".encode("utf-8")).hexdigest()


def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text("utf-8"))


def _utc_ok(s: Any) -> bool:
    if not isinstance(s, str) or not s:
        return False
    try:
        datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def walk_engine_fields(o: Any, path: str = "") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ENGINE_KEYS and isinstance(v, str):
                out.append((path + "/" + k, v))
            out += walk_engine_fields(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += walk_engine_fields(v, "%s[%d]" % (path, i))
    return out


# ------------------------------------------------------------------ per-check functions: each returns a list of problem strings (empty = OK)
def check_chain(entries: List[Dict[str, Any]], result: Dict[str, Any]) -> List[str]:
    h = GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i:
            return ["journal seq %r at position %d" % (e.get("seq"), i)]
        if e.get("prev") != h:
            return ["hash chain broken at entry %d (%s)" % (i, e.get("kind"))]
        h = _sha(_canon(e))
    if result.get("journal_hash") and h != result["journal_hash"]:
        return ["journal head %s does not equal result.journal_hash %s" % (h[:12], str(result["journal_hash"])[:12])]
    return []


def check_invariants(result: Dict[str, Any]) -> List[str]:
    inv = result.get("invariants") or {}
    if sorted(inv.get("checked") or []) != ["I%d" % i for i in range(1, 9)]:
        return ["invariants checked is %r, not I1..I8" % (inv.get("checked"),)]
    if inv.get("failed"):
        return ["invariants failed: %r" % (inv["failed"],)]
    return []


def stage_rows(tl: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in (tl or {}).get("rows", []) if r.get("row") == "stage"]


def check_stages(entries: List[Dict[str, Any]], tl: Optional[Dict[str, Any]], result: Dict[str, Any], gen_calls: Any) -> Tuple[List[str], Dict[str, Any]]:
    if tl is None:
        return ["no controller_timeline.json"], {}
    rows = stage_rows(tl)
    names = [r["name"] for r in rows]
    first = {n: names.index(n) for n in STAGE_ORDER if n in names}
    info: Dict[str, Any] = {n: {"present": n in first, "start_utc": next((r.get("start_utc_iso") for r in rows if r["name"] == n), None), "end_utc": next((r.get("end_utc_iso") for r in rows if r["name"] == n), None)} for n in STAGE_ORDER}
    probs: List[str] = []
    kinds = [e["kind"] for e in entries]
    live_pass = [c for c in _pass_all(entries).values() if c]
    cl = next((e["data"] for e in entries if e["kind"] == "behaviour_clusters"), None)
    cause, n_cand = result.get("stop_cause"), sum(1 for k in kinds if k == "candidate")
    for n in ("intake", "generate", "execute", "rank", "commit_verify"):
        if n in first:
            continue
        if n == "execute" and n_cand == 0 and cause in ("budget", "no_progress_2", "tool_failure") and result.get("generation_failed_at_ceiling", result.get("ceiling_hits")):
            info[n]["explained"] = "no candidate was extracted (generation hit the ceiling / failed): stop_cause %s" % cause
        elif n == "rank" and cause in ("budget", "tool_failure", "api_unavailable", "missing_information"):
            info[n]["explained"] = "the run stopped before the first ranking: stop_cause %s %s" % (cause, result.get("stop_details", ""))
        elif n == "commit_verify" and cause in ("tool_failure", "missing_information"):
            info[n]["explained"] = "stopped without a commit: stop_cause %s" % cause
        else:
            probs.append("stage %s missing (stop_cause %s)" % (n, cause))
    if "l2_generate" not in first and "l2_checks" not in kinds:
        if live_pass:
            probs.append("stage l2 missing although %d candidate(s) passed every example" % len(live_pass))
        else:
            info["l2_generate"]["explained"] = "no candidate passed every example (no trusted reference)"
    elif "l2_validate" not in first and "l2_generate" in first:
        probs.append("stage l2_validate missing after l2_generate")
    if "cluster" not in first and cl is None:
        if len(live_pass) >= 2:
            probs.append("stage cluster missing although %d candidates passed every example" % len(live_pass))
        else:
            info["cluster"]["explained"] = "fewer than 2 live candidates"
    if "discriminate" not in first:
        if len(live_pass) >= 2 and (cl or {}).get("extended_clusters", 0) >= 2:
            probs.append("stage discriminate missing although >= 2 behaviour clusters exist")
        else:
            info["discriminate"]["explained"] = "no discrimination needed (clusters < 2 or no live candidates)"
    if not ({"critic_trigger", "critic"} & set(kinds)) and result.get("stop_cause") != "api_unavailable" and (cl or {}).get("extended_clusters", 0) >= 2:
        probs.append("two or more clusters but no critic decision: neither a critic call, pending packets nor an explicit api_unavailable is journaled")
    inv_ix = names.index("investigate") if "investigate" in names else None
    info["investigate"] = {"present": inv_ix is not None, "status": "PRESENT" if inv_ix is not None else "NOT_APPLICABLE (function-level ladder: no workspace to inspect; intake routing + oracle_tier journaled)"}
    if inv_ix is not None and "intake" in first and inv_ix < first["intake"]:
        probs.append("investigate before intake")
    repaired = "repair" in names
    if repaired:      # a repair round re-enters the loop: discriminate / rank may precede l2_* (no live candidate after round 1); only the skeleton is ordered
        skel = ("intake", "generate", "execute")
        order = [first[n] for n in skel if n in first]
        late = [n for n in STAGE_ORDER if n in first and n != "commit_verify"]
        if order != sorted(order) or ("commit_verify" in first and any(first[n] > first["commit_verify"] for n in late)) or (first and min(first.values()) != first.get("intake", 0)):
            probs.append("stage order wrong (repair run): first occurrences %s" % {n: first[n] for n in STAGE_ORDER if n in first})
        info["order_rule"] = "repair run: skeleton intake < generate < execute, commit_verify last; discriminate / rank / l2 / cluster may repeat after a repair round"
    else:
        order = [first[n] for n in STAGE_ORDER if n in first]
        if order != sorted(order):
            probs.append("stage order wrong: first occurrences %s" % {n: first[n] for n in STAGE_ORDER if n in first})
    for r in rows:
        if not (_utc_ok(r.get("start_utc_iso")) and _utc_ok(r.get("end_utc_iso"))):
            probs.append("stage %s has no parseable UTC start/end" % r["name"])
            break
    return probs, info


def check_utc_rows(entries: List[Dict[str, Any]], tl: Optional[Dict[str, Any]]) -> List[str]:
    if tl is None:
        return ["no controller_timeline.json"]
    er = [r for r in tl.get("rows", []) if r.get("row") == "entry"]
    if [r.get("seq") for r in er] != list(range(len(entries))):
        return ["timeline has %d entry rows for %d journal entries (seq mismatch)" % (len(er), len(entries))]
    bad = [r["seq"] for r in er if not _utc_ok(r.get("utc_iso"))]
    if bad:
        return ["journal entries without a UTC row: %s" % bad[:5]]
    mono = [r["monotonic_s"] for r in er if r.get("monotonic_s") is not None]
    if mono != sorted(mono):
        return ["monotonic time goes backwards in the entry rows"]
    return []


def _pass_all(entries: List[Dict[str, Any]]) -> Dict[str, bool]:
    ev: Dict[str, List[str]] = {}
    for e in entries:
        if e["kind"] == "evidence" and e["data"]["evidence"]["check_id"] in ("public_examples_v1", "parse_v1"):
            ev.setdefault(e["data"]["evidence"]["candidate_id"], []).append(e["data"]["evidence"]["status"])
    return {c: bool(s) and all(x == "pass" for x in s) for c, s in ev.items()}


def l2_generations(gen_calls: Any) -> List[Dict[str, Any]]:
    out = []
    for c in gen_calls or []:
        if c.get("purpose") == "tests":
            out += c.get("generations", [])
    return out


def check_l2(result: Dict[str, Any], gen_calls: Any, input_keys: Optional[Sequence[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
    l2 = result.get("l2") or {}
    info = {"verdict": l2.get("verdict"), "reason": l2.get("reason"), "parsed": l2.get("parsed")}
    if l2.get("verdict") == "INSUFFICIENT" and "no valid test parsed" in str(l2.get("reason", "")):
        from evidence_ladder.controller import l2 as L2
        gens = l2_generations(gen_calls)
        if not gens:
            return [], dict(info, note="L2 never ran (no tests generation recorded)")
        counts: Dict[str, int] = {}
        usable_text = False
        for g in gens:
            pc = L2.parse_checks_v2(g.get("text", ""), 6, input_keys, require_think_close=("</think>" in g.get("text", "")))
            usable_text = usable_text or pc["status"] != "no_think_close" and pc["counts"] != {"not_json": 1}
            for k, v in pc["counts"].items():
                counts[k] = counts.get(k, 0) + v
        info["rejection_counts_v2"] = counts
        if usable_text and counts.get("no_think_close", 0) < len(gens):
            return ["L2 INSUFFICIENT ('no valid test parsed') although the model produced structured output: re-parse rejection counts %s (parser/prompt mismatch, not the model)" % counts], info
        info["model_produced_nothing"] = True
        info["why"] = "no answer after </think> / ceiling hit in every tests generation"
    return [], info


def check_sandbox(task_dir: Path, result: Dict[str, Any], summary: Dict[str, Any], expect: str, legacy: bool) -> Tuple[List[str], Dict[str, Any]]:
    eng = result.get("engine") or summary.get("engine") or summary.get("sandbox") or (summary.get("jail") or {}).get("engine")
    if eng is None and legacy and (summary.get("jail") or {}).get("bwrap_swapped") is not None:
        eng = "kaggle-bwrap-jail"
    info = {"engine": eng, "expected": expect}
    probs: List[str] = []
    if eng is None:
        probs.append("no sandbox engine recorded")
    elif not (str(eng) == expect or str(eng).startswith(expect)):
        probs.append("sandbox engine %r is not the expected %r" % (eng, expect))
    seen = walk_engine_fields(result)
    for f in ("controller_timeline.json", "critic_requests.json"):
        p = task_dir / f
        if p.exists():
            try:
                seen += walk_engine_fields(_read_json(p))
            except ValueError:
                pass
    for path, v in seen:
        if FORBIDDEN_ENGINE.search(v) and not (str(v) == expect or str(v).startswith(expect)):
            probs.append("forbidden execution backend %r at %s" % (v, path))
    return probs, info


def check_policy(result: Dict[str, Any], summary: Dict[str, Any], entries: List[Dict[str, Any]], legacy: bool, require_l2_v2: bool = False) -> Tuple[List[str], Dict[str, Any]]:
    probs: List[str] = []
    pol = result.get("thinking_policy") or summary.get("thinking_policy")
    sha = (result.get("profile_sha256") or (summary.get("profile") or {}).get("sha256") or (summary.get("model") or {}).get("profile_sha256") or summary.get("profile_sha256"))
    l2v = result.get("l2_version", summary.get("l2_version"))
    if l2v is None and any(e["kind"] == "l2_parse" and e["data"].get("version") == 2 for e in entries):
        l2v = 2
    info = {"thinking_policy": (pol or {}).get("policy") if isinstance(pol, dict) else pol, "profile_sha256": sha, "l2_version": l2v, "waived": []}
    if not pol:
        probs.append("thinking policy not recorded")
    if require_l2_v2 and l2v is not None and int(l2v) != 2:
        probs.append("l2_version is %r, the WSLC Java run requires the L2 v2 prompt/parser" % (l2v,))
    for name, v in (("profile sha256", sha), ("l2_version", l2v)):
        if v is None:
            if legacy:
                info["waived"].append(name)
            else:
                probs.append("%s not recorded" % name)
    return probs, info


def check_public(task_dir: Path, summary: Dict[str, Any], markers: Sequence[str]) -> List[str]:
    probs: List[str] = []
    if (summary.get("public_only_guard") or {}).get("violations"):
        probs.append("RUN_SUMMARY.public_only_guard reports %r violations" % summary["public_only_guard"]["violations"])
    for p in sorted(task_dir.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".src", ".md") and "states" not in p.parts:
            txt = p.read_text("utf-8", errors="replace")
            for m in markers:
                if m in txt:
                    probs.append("hidden/protected marker %r in %s" % (m, p.relative_to(task_dir).as_posix()))
                    break
    return probs


WSLC_RECEIPT_FIELDS = ("backend", "image_digest", "toolchain", "source_sha256")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def check_exec(task_dir: Path, legacy: bool, expect: Optional[str] = None, strict_provenance: bool = False, summary: Optional[Dict[str, Any]] = None) -> Tuple[List[str], Dict[str, Any]]:
    """strict_provenance (WSLC mode): every record also names backend (== expect: any other backend is a SILENT FALLBACK), image_digest sha256:<64 hex> (one digest for the whole task),
    toolchain {javac, java, python} non-empty and source_sha256 (64 hex; equal to RUN_SUMMARY.source_sha256 when that is recorded)."""
    p = task_dir / "exec_results.jsonl"
    if not p.exists():
        return ([] if legacy else ["exec_results.jsonl missing (every sandbox call must be recorded with the 8 contract fields)"]), {"records": 0, "waived": legacy}
    n, probs = 0, []
    digests, backends = set(), set()
    for i, l in enumerate(x for x in p.read_text("utf-8").splitlines() if x.strip()):
        n += 1
        rec = json.loads(l)
        miss = [k for k in RESULT_KEYS if k not in rec]
        if miss:
            probs.append("exec record %d lacks %s" % (i, miss))
            break
        if strict_provenance:
            miss = [k for k in WSLC_RECEIPT_FIELDS if not rec.get(k)]
            if miss:
                probs.append("exec record %d lacks receipt field(s) %s" % (i, miss))
                break
            backends.add(str(rec["backend"]))
            if expect and not str(rec["backend"]).startswith(expect):
                probs.append("silent fallback: exec record %d ran on backend %r, expected %r" % (i, rec["backend"], expect))
                break
            if not re.match(r"^sha256:[0-9a-f]{64}$", str(rec["image_digest"])):
                probs.append("exec record %d: image_digest %r is not sha256:<64 hex>" % (i, rec["image_digest"]))
                break
            digests.add(rec["image_digest"])
            tc = rec["toolchain"] if isinstance(rec["toolchain"], dict) else {}
            bad = [k for k in ("javac", "java", "python") if not str(tc.get(k) or "").strip()]
            if bad:
                probs.append("exec record %d: toolchain lacks %s" % (i, bad))
                break
            if not _HEX64.match(str(rec["source_sha256"])):
                probs.append("exec record %d: source_sha256 is not 64 hex" % i)
                break
            want = (summary or {}).get("source_sha256")
            if want and rec["source_sha256"] != want:
                probs.append("exec record %d: source_sha256 differs from RUN_SUMMARY.source_sha256 (not the committed source)" % i)
                break
    if n == 0:
        probs.append("exec_results.jsonl is empty")
    if len(digests) > 1:
        probs.append("more than one image digest used inside one task: %s" % sorted(digests))
    return probs, {"records": n, "backends": sorted(backends), "image_digests": sorted(digests)}


def check_states(task_dir: Path, entries: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    sd = task_dir / "states"
    if not sd.is_dir():
        return [], {"states": "none (capture off or not run)"}
    files = sorted(sd.glob("*.pt"))
    idx = {}
    ip = sd / "index.jsonl"
    if ip.exists():
        for l in ip.read_text("utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                idx[r["candidate_id"]] = r
    cands = {e["data"].get("id") for e in entries if e["kind"] == "candidate"} | {e["data"]["evidence"]["candidate_id"] for e in entries if e["kind"] == "evidence"}
    probs = []
    for f in files:
        cid = f.stem
        if cid not in idx:
            probs.append("state %s has no index line" % cid[:12])
        if cid not in cands:
            probs.append("state %s does not belong to a journaled candidate" % cid[:12])
    jtxt = "\n".join(_canon(e) for e in entries)
    if '"states/' in jtxt or "hq" in {k for e in entries for k in (e["data"] if isinstance(e["data"], dict) else {})}:
        probs.append("state data appears inside the journal (must stay hash-neutral)")
    return probs[:5], {"states": len(files), "indexed": len(idx)}


# ------------------------------------------------------------------ the audit
def audit_task(task_dir: Path, summary: Dict[str, Any], expect: str, legacy: bool, markers: Sequence[str], input_keys: Optional[Sequence[str]] = None, wslc: bool = False) -> Dict[str, Any]:
    tid = task_dir.name
    rec: Dict[str, Any] = {"task": tid, "verdict": "PIPELINE_OK", "problems": [], "checks": {}}

    def add(name: str, probs: List[str], info: Any = None):
        rec["checks"][name] = {"ok": not probs, "problems": probs, **({"info": info} if info is not None else {})}
        for p in probs:
            rec["problems"].append("%s: %s" % (name, p))
    try:
        result = _read_json(task_dir / "result.json")
        entries = [json.loads(l) for l in (task_dir / "journal.jsonl").read_text("utf-8").splitlines() if l.strip()]
    except (OSError, ValueError) as ex:
        rec.update(verdict="BROKEN", problems=["unreadable result.json / journal.jsonl: %s" % ex])
        return rec
    tl = _read_json(task_dir / "controller_timeline.json") if (task_dir / "controller_timeline.json").exists() else None
    gen_calls = _read_json(task_dir / "gen_calls.json") if (task_dir / "gen_calls.json").exists() else []
    sp, sinfo = check_stages(entries, tl, result, gen_calls)
    add("stages", sp, sinfo)
    add("utc_rows", check_utc_rows(entries, tl))
    add("chain", check_chain(entries, result))
    add("invariants", check_invariants(result))
    lp, linfo = check_l2(result, gen_calls, input_keys)
    add("l2", lp, linfo)
    ep, einfo = check_sandbox(task_dir, result, summary, expect, legacy)
    add("sandbox", ep, einfo)
    pp, pinfo = check_policy(result, summary, entries, legacy, require_l2_v2=wslc)
    add("policy", pp, pinfo)
    add("public_only", check_public(task_dir, summary, markers))
    xp, xinfo = check_exec(task_dir, legacy, expect, wslc, summary)
    add("exec_results", xp, xinfo)
    tp, tinfo = check_states(task_dir, entries)
    add("states", tp, tinfo)
    rec["candidate"] = {"answer_label": result.get("answer_label"), "stop_cause": result.get("stop_cause"), "public": result.get("public"), "note": "candidate result, reported separately: an honest candidate failure is still PIPELINE_OK when every stage was recorded"}
    if rec["problems"]:
        rec["verdict"] = "BROKEN"
    return rec


def expected_tasks(summary: Dict[str, Any], tasks_path: Optional[str]) -> List[str]:
    if tasks_path:
        d = _read_json(Path(tasks_path))
        ids = d if isinstance(d, list) else d.get("tasks") or d.get("task_ids") or []
        return [t["task_id"] if isinstance(t, dict) else t for t in ids]
    order = (summary.get("shards") or {}).get("order") or []
    return [t for w in order for t in w] if order and isinstance(order[0], list) else list(order)


def audit_run(run_dir, expect: str = "mxc", legacy: bool = False, tasks_path: Optional[str] = None, markers: Sequence[str] = HIDDEN_MARKERS, wslc: Optional[bool] = None,
              expect_count: Optional[int] = None) -> Dict[str, Any]:
    """wslc (default: True when `expect` contains 'wslc'): strict execution provenance per record (backend, image digest, toolchain, source hash), zero NOT_TESTED (NO_JDK), zero silent fallback,
    L2 v2 required; expect_count (e.g. 20): exactly that many executed tasks must exist. A summary with engine_status BLOCKED is an explicit PIPELINE_BLOCKED, never a pass."""
    run = Path(run_dir)
    wslc = ("wslc" in expect) if wslc is None else wslc
    summary = _read_json(run / "RUN_SUMMARY.json") if (run / "RUN_SUMMARY.json").exists() else {}
    out: Dict[str, Any] = {"run_dir": str(run), "expect_engine": expect, "legacy_kaggle": legacy, "run_checks": {}, "tasks": {}, "not_tested": {}, "error": [], "missing": []}
    out["run_checks"]["run_summary"] = bool(summary)
    out["run_checks"]["timeline_jsonl"] = (run / "TIMELINE.jsonl").exists()
    tdirs = sorted(p for p in (run / "tasks").iterdir() if p.is_dir()) if (run / "tasks").is_dir() else []
    seen = set()
    for td in tdirs:
        seen.add(td.name)
        if (td / "NOT_TESTED").exists():
            try:
                out["not_tested"][td.name] = _read_json(td / "result.json").get("reason", "NOT_TESTED marker without a recorded reason")
            except (OSError, ValueError):
                out["not_tested"][td.name] = "NOT_TESTED marker without a result"
        elif (td / "DONE").exists():
            out["tasks"][td.name] = audit_task(td, summary, expect, legacy, markers, None, wslc)
        else:
            out["error"].append({"task": td.name, "why": "no DONE and no NOT_TESTED marker (crashed, pending or in progress)"})
    exp = expected_tasks(summary, tasks_path)
    out["missing"] = [t for t in exp if t not in seen]
    broken = {t: r["problems"] for t, r in out["tasks"].items() if r["verdict"] != "PIPELINE_OK"}
    out["broken"] = broken
    out["counts"] = {"expected": len(exp) or None, "executed": len(out["tasks"]), "ok": len(out["tasks"]) - len(broken), "broken": len(broken), "not_tested": len(out["not_tested"]),
                     "error": len(out["error"]), "missing": len(out["missing"])}
    if not out["run_checks"]["run_summary"]:
        out["run_checks"]["problem"] = "no RUN_SUMMARY.json"
    if broken or not out["run_checks"]["run_summary"] or not out["tasks"]:
        out["verdict"] = "PIPELINE_BROKEN" if (broken or not out["run_checks"]["run_summary"]) else "PIPELINE_PARTIAL"
    elif out["missing"] or out["error"]:
        out["verdict"] = "PIPELINE_PARTIAL"
    else:
        out["verdict"] = "PIPELINE_COMPLETE"
    if wslc:
        crit: Dict[str, Any] = {}
        no_jdk = sorted(t for t, r in out["not_tested"].items() if "NO_JDK" in str(r).upper() or "JDK" in str(r).upper())
        crit["zero_not_tested"] = not out["not_tested"]
        crit["zero_not_tested_no_jdk"] = not no_jdk
        crit["zero_silent_fallback"] = not any("silent fallback" in p or "forbidden execution backend" in p or "not the expected" in p for ps in broken.values() for p in ps)
        crit["expected_count"] = expect_count
        crit["executed_count_matches"] = expect_count is None or len(out["tasks"]) == expect_count
        crit["all_receipts_complete"] = not any(p.startswith("exec_results:") for ps in broken.values() for p in ps) and bool(out["tasks"])
        crit["engine_status"] = summary.get("engine_status", "OK" if summary else "UNKNOWN")
        out["wslc"] = {"criteria": crit, "not_tested_no_jdk": no_jdk, "candidate_results": {t: r.get("candidate") for t, r in out["tasks"].items()}}
        if str(crit["engine_status"]).upper().startswith("BLOCKED"):
            out["verdict"] = "PIPELINE_BLOCKED"
            out["blocked_reason"] = summary.get("engine_blocked_reason") or summary.get("engine_status")
        elif out["verdict"] == "PIPELINE_COMPLETE" and not (crit["zero_not_tested"] and crit["zero_not_tested_no_jdk"] and crit["executed_count_matches"]):
            out["verdict"] = "PIPELINE_PARTIAL"
    out["label"] = "architecture conformance of a run (files only); not an accuracy result"
    return out


def to_markdown(rep: Dict[str, Any]) -> str:
    L = ["# CONFORMANCE: %s" % rep["verdict"], "", "run: `%s`; expected engine `%s`%s; %s" % (rep["run_dir"], rep["expect_engine"], " (legacy Kaggle waivers on)" if rep["legacy_kaggle"] else "", rep["label"]), "",
         "counts: %s" % json.dumps(rep["counts"]), ""]
    if rep["broken"]:
        L += ["## BROKEN tasks (exact reasons)"] + ["- **%s**: %s" % (t, "; ".join(p)) for t, p in sorted(rep["broken"].items())] + [""]
    if rep["not_tested"]:
        L += ["## NOT_TESTED (reason)"] + ["- %s: %s" % (t, r) for t, r in sorted(rep["not_tested"].items())] + [""]
    if rep["error"]:
        L += ["## ERROR / pending (no marker)"] + ["- %s: %s" % (e["task"], e["why"]) for e in rep["error"]] + [""]
    if rep["missing"]:
        L += ["## MISSING (expected but no task directory)", ", ".join(rep["missing"]), ""]
    if rep.get("blocked_reason"):
        L += ["## BLOCKED", str(rep["blocked_reason"]), ""]
    if rep.get("wslc"):
        L += ["## WSLC criteria", "```", json.dumps(rep["wslc"]["criteria"], indent=1), "```", "", "## Candidate results (SEPARATE from pipeline status)", "", "| task | answer label | stop cause | public |", "|---|---|---|---|"]
        L += ["| %s | %s | %s | %s |" % (t, (c or {}).get("answer_label"), (c or {}).get("stop_cause"), json.dumps((c or {}).get("public"))[:120].replace("|", "/")) for t, c in sorted(rep["wslc"]["candidate_results"].items())] + [""]
    L += ["## per task", "", "| task | verdict | problems |", "|---|---|---|"] + ["| %s | %s | %s |" % (t, r["verdict"], "; ".join(r["problems"])[:300].replace("|", "/") or "") for t, r in sorted(rep["tasks"].items())]
    return "\n".join(L) + "\n"


def selftest() -> Dict[str, Any]:
    """hash-neutrality of the state capture hook on a ladder fixture: the same run with and without the hook gives the same journal hash (CPU, tiny fake model)."""
    import tempfile
    sys.path.insert(0, str(ROOT / "evidence_ladder/controller/tests"))
    import phaseB_tests as B
    import run_tests as RT
    from evidence_ladder.common.interfaces import ControllerContext
    from evidence_ladder.controller.controller import EvidenceLadderController
    from evidence_ladder.models.state_capture import StateCapture
    import torch

    def run():
        ctl = EvidenceLadderController()
        ctx = ControllerContext(RT.FakeGen([[B.PLUS2, B.DOUBLE]]), B.EvalExec(), RT.FakeBudget(), "D0", None, None, (1, 2, 3, 4), {"k": 2})
        return ctl, ctl.run_problem(B.EVAL_TASK, ctx)
    c0, r0 = run()
    c1, r1 = run()
    sc = StateCapture(lambda c, o: (torch.zeros(3, 4).half(), torch.zeros(1, 3, 4).half(), {"status": "ok"}), Path(tempfile.mkdtemp(prefix="conf-")), depths=(1, 2, 3))
    for c in c1.last_run.cands.values():
        sc.capture("E1", c.id, B.EVAL_TASK.prompt, c.art.source)
    return {"without_states": r0.journal_hash, "with_states": r1.journal_hash, "identical": r0.journal_hash == r1.journal_hash, "states_saved": sc.saved}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--tasks")
    ap.add_argument("--expect-engine", default="mxc")
    ap.add_argument("--legacy", "--legacy-kaggle", dest="legacy_kaggle", action="store_true", help="audit an OLD run without the new provenance keys (profile sha, l2_version, exec_results.jsonl); waivers are reported. Default: strict")
    ap.add_argument("--expect-count", type=int, help="exactly this many executed tasks must exist (WSLC Java run: 20)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    rep = audit_run(a.run_dir, a.expect_engine, a.legacy_kaggle, a.tasks, expect_count=a.expect_count)
    if a.selftest:
        rep["selftest_state_hook_hash_neutral"] = selftest()
    md = to_markdown(rep)
    if not a.no_write:
        od = Path(a.out or a.run_dir)
        od.mkdir(parents=True, exist_ok=True)
        for name, txt in (("CONFORMANCE.json", json.dumps(rep, indent=1, sort_keys=True, default=str)), ("CONFORMANCE.md", md)):
            p = od / name
            if p.exists():
                print("refusing to overwrite %s (create-only); use --out" % p)
                return 2
            p.write_text(txt, encoding="utf-8")
    print(md)
    return 0 if rep["verdict"] == "PIPELINE_COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
