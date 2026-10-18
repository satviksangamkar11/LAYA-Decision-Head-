"""[EL-ARCH-1 F2d / report] Generator for WSLC_JAVA20_FINAL_REPORT.md (create-only). It REFUSES to write a success verdict unless every Phase 5 criterion holds; otherwise the report states the exact
blockers (or WSLC_BLOCKED when the engine itself was unavailable). It reads only files produced by the other tools; it runs nothing.

    python tools/el_wslc_final_report.py --run <run dir> --conformance CONFORMANCE.json --replay REPLAY.json --security security_gates.json --toolchain-probe toolchain_probe.json \\
        --suites suites.json --commit <sha> --out WSLC_JAVA20_FINAL_REPORT.md [--expect-backend mxc-wslc] [--expect-count 20]

Inputs: CONFORMANCE.json (tools/el_bank_conformance.py run with --expect-engine mxc-wslc --expect-count 20 --tasks <manifest>), REPLAY.json (tools/el_bank_replay_check.py), security_gates.json
(checked by tools/el_wslc_security_check.py), toolchain_probe.json ({"languages": {<lang>: {"probe": "PASS"|"FAIL", "versions": {...}, "reason": ...}}}), suites.json
({"parser_contracts"|"invariants"|"limits": {"tests": n, "passed": n}}: the contract / invariant / limit suites run on the committed tree).
Phase 5 criteria (all must hold for WSLC_JAVA20_PASS): (1) N of N tasks reached the real WSLC executor (every executed task has execution records, all on the expected backend); (2) N of N complete
pipeline receipts (conformance verdict PIPELINE_COMPLETE, every task PIPELINE_OK); (3) zero NOT_TESTED_NO_JDK; (4) zero silent fallback; (5) security gates (all five), limits, invariants and parser
contracts pass; (6) replay matches (REPLAY_OK); (7) conformance has zero unexplained missing or broken tasks. Candidate results are reported in their own table, never mixed into the pipeline status.
Supported language profiles = only those whose toolchain probe PASSED; every other language is listed unsupported with the probe's reason."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PASS, NOT_PASS, BLOCKED = "WSLC_JAVA20_PASS", "WSLC_JAVA20_NOT_PASS", "WSLC_BLOCKED"


def _load(p: Optional[str]) -> Optional[Dict[str, Any]]:
    if not p or not Path(p).exists():
        return None
    try:
        return json.loads(Path(p).read_text("utf-8"))
    except ValueError:
        return None


def _suite_ok(s: Optional[Dict[str, Any]], name: str) -> Tuple[bool, str]:
    r = (s or {}).get(name)
    if not isinstance(r, dict) or not r.get("tests"):
        return False, "no result for %s" % name
    return r.get("passed") == r.get("tests"), "%s: %s/%s passed" % (name, r.get("passed"), r.get("tests"))


def evaluate(conf: Optional[Dict[str, Any]], replay: Optional[Dict[str, Any]], sec: Optional[Dict[str, Any]], suites: Optional[Dict[str, Any]], expect_count: int, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    C: List[Dict[str, Any]] = []

    def add(n: int, name: str, ok: bool, detail: str):
        C.append({"n": n, "criterion": name, "ok": bool(ok), "detail": detail})
    tasks = (conf or {}).get("tasks", {})
    wslc = (conf or {}).get("wslc", {}).get("criteria", {})
    reached = [t for t, r in tasks.items() if (r["checks"].get("exec_results", {}).get("info", {}).get("records") or 0) > 0 and r["checks"].get("exec_results", {}).get("ok")]
    add(1, "%d of %d tasks reached the real WSLC executor" % (expect_count, expect_count), conf is not None and len(reached) == expect_count and len(tasks) == expect_count,
        "executed %d, with provenance-complete execution records %d" % (len(tasks), len(reached)))
    ok_tasks = [t for t, r in tasks.items() if r["verdict"] == "PIPELINE_OK"]
    add(2, "%d of %d complete pipeline receipts" % (expect_count, expect_count), conf is not None and conf.get("verdict") == "PIPELINE_COMPLETE" and len(ok_tasks) == expect_count, "conformance verdict %s, %d PIPELINE_OK" % ((conf or {}).get("verdict"), len(ok_tasks)))
    add(3, "zero NOT_TESTED_NO_JDK", conf is not None and bool(wslc.get("zero_not_tested_no_jdk")) and bool(wslc.get("zero_not_tested")), "not_tested: %s" % ((conf or {}).get("not_tested")))
    add(4, "zero silent fallback", conf is not None and bool(wslc.get("zero_silent_fallback")), "every execution record on the expected backend" if wslc.get("zero_silent_fallback") else "fallback or foreign backend detected / no conformance")
    sec_ok = bool(sec and sec.get("ok"))
    lim_ok, lim_d = _suite_ok(suites, "limits")
    inv_ok, inv_d = _suite_ok(suites, "invariants")
    par_ok, par_d = _suite_ok(suites, "parser_contracts")
    add(5, "security gates, limits, invariants and parser contracts pass", sec_ok and lim_ok and inv_ok and par_ok,
        "security gates ok=%s (missing %s, failed %s); %s; %s; %s" % (sec_ok, (sec or {}).get("missing"), (sec or {}).get("failed"), lim_d, inv_d, par_d))
    add(6, "replay matches", replay is not None and replay.get("verdict") == "REPLAY_OK", "replay verdict %s" % (replay or {}).get("verdict"))
    add(7, "conformance has zero unexplained missing or broken tasks", conf is not None and not conf.get("broken") and not conf.get("missing") and not conf.get("error"),
        "broken %s, missing %s, error %s" % (list((conf or {}).get("broken", {})), (conf or {}).get("missing"), [e["task"] for e in (conf or {}).get("error", [])]))
    return C


def build_report(run_dir, conf: Optional[Dict[str, Any]], replay: Optional[Dict[str, Any]], sec: Optional[Dict[str, Any]], probe: Optional[Dict[str, Any]], suites: Optional[Dict[str, Any]],
                 commit: str, expect_count: int = 20, expect_backend: str = "mxc-wslc") -> Tuple[str, str, List[Dict[str, Any]]]:
    run = Path(run_dir)
    summary = _load(str(run / "RUN_SUMMARY.json")) or {}
    crit = evaluate(conf, replay, sec, suites, expect_count, summary)
    blockers = ["(%d) %s: %s" % (c["n"], c["criterion"], c["detail"]) for c in crit if not c["ok"]]
    if str(summary.get("engine_status", "")).upper().startswith("BLOCKED") or ((conf or {}).get("verdict") == "PIPELINE_BLOCKED"):
        verdict = BLOCKED
        blockers.insert(0, "WSLC unavailable: %s" % (summary.get("engine_blocked_reason") or summary.get("engine_status") or (conf or {}).get("blocked_reason")))
    elif blockers:
        verdict = NOT_PASS
    else:
        verdict = PASS
    # identities
    digests, toolchains, backends = set(), set(), set()
    for td in sorted((run / "tasks").glob("*")) if (run / "tasks").is_dir() else []:
        p = td / "exec_results.jsonl"
        if p.exists():
            for l in p.read_text("utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    digests.add(str(r.get("image_digest")))
                    backends.add(str(r.get("backend")))
                    toolchains.add(json.dumps(r.get("toolchain"), sort_keys=True))
    L = ["# WSLC JAVA-20 FINAL REPORT: %s" % verdict, "",
         "Generated by tools/el_wslc_final_report.py (create-only). A success verdict is written only when every Phase 5 criterion below holds; this report is an architecture / execution-pipeline report, not an accuracy claim.", "",
         "## Identity", "", "- commit: `%s`" % commit, "- expected backend: `%s`; backends seen in execution records: %s" % (expect_backend, sorted(backends) or "none"),
         "- WSLC image digest(s): %s" % (sorted(digests) or "none recorded"), "- toolchain(s) read inside the image: %s" % (sorted(toolchains) or "none recorded"),
         "- source_sha256 (committed source tree): `%s`" % summary.get("source_sha256"), "- engine_status: %s" % summary.get("engine_status", "not recorded"), ""]
    L += ["## Phase 5 criteria", "", "| # | criterion | result | detail |", "|---|---|---|---|"] + ["| %d | %s | %s | %s |" % (c["n"], c["criterion"], "PASS" if c["ok"] else "**FAIL**", c["detail"].replace("|", "/")) for c in crit] + [""]
    L += ["## Execution status (%d expected)" % expect_count, "", "| task | pipeline | exec records | backend(s) | image digest | problems |", "|---|---|---|---|---|---|"]
    for t, r in sorted(((conf or {}).get("tasks") or {}).items()):
        e = r["checks"].get("exec_results", {}).get("info", {})
        L.append("| %s | %s | %s | %s | %s | %s |" % (t, r["verdict"], e.get("records"), ",".join(e.get("backends") or []), ",".join(d[:19] for d in e.get("image_digests") or []), "; ".join(r["problems"])[:160].replace("|", "/")))
    for t, why in sorted(((conf or {}).get("not_tested") or {}).items()):
        L.append("| %s | NOT_TESTED | | | | %s |" % (t, str(why)[:160].replace("|", "/")))
    for t in (conf or {}).get("missing") or []:
        L.append("| %s | MISSING | | | | no task directory |" % t)
    L += ["", "## Candidate results (SEPARATE from pipeline status)", "", "A failed candidate is still PIPELINE_OK when every stage was recorded.", "", "| task | answer label | stop cause | public examples |", "|---|---|---|---|"]
    for t, c in sorted((((conf or {}).get("wslc") or {}).get("candidate_results") or {}).items()):
        L.append("| %s | %s | %s | %s |" % (t, (c or {}).get("answer_label"), (c or {}).get("stop_cause"), json.dumps((c or {}).get("public"))[:120].replace("|", "/")))
    L += ["", "## Safety, limits and replay", "", "- security gates: ok=%s; missing %s; failed %s; problems %s" % ((sec or {}).get("ok"), (sec or {}).get("missing"), (sec or {}).get("failed"), (sec or {}).get("problems")),
          "- limits / invariants / parser contracts: %s | %s | %s" % (_suite_ok(suites, "limits")[1], _suite_ok(suites, "invariants")[1], _suite_ok(suites, "parser_contracts")[1]),
          "- replay: %s (counts %s); level B not attempted: %s" % ((replay or {}).get("verdict"), (replay or {}).get("counts"), (replay or {}).get("level_b_not_attempted")), ""]
    sup = sorted(l for l, v in ((probe or {}).get("languages") or {}).items() if v.get("probe") == "PASS")
    uns = {l: v.get("reason") or "probe %s" % v.get("probe") for l, v in ((probe or {}).get("languages") or {}).items() if v.get("probe") != "PASS"}
    L += ["## Language profiles", "", "Supported (toolchain probe PASSED): %s" % (", ".join("`%s` %s" % (l, json.dumps(probe["languages"][l].get("versions"))) for l in sup) or "none (no probe result)"), "",
          "Unsupported (probe not passed or never run): %s" % (", ".join("`%s` (%s)" % kv for kv in sorted(uns.items())) or "none listed"), ""]
    L += ["## Blockers", ""] + (["- " + b for b in blockers] if blockers else ["none"]) + [""]
    return verdict, "\n".join(L) + "\n", crit


def write_report(out_path, **kw) -> Tuple[str, Path]:
    """create-only. There is deliberately no way to force a verdict: PASS comes only out of build_report()'s evaluation of the criteria."""
    verdict, md, _ = build_report(**kw)
    p = Path(out_path)
    if p.exists():
        raise FileExistsError("refusing to overwrite %s (create-only)" % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    return verdict, p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--conformance")
    ap.add_argument("--replay")
    ap.add_argument("--security")
    ap.add_argument("--toolchain-probe")
    ap.add_argument("--suites")
    ap.add_argument("--commit", default="UNKNOWN")
    ap.add_argument("--expect-backend", default="mxc-wslc")
    ap.add_argument("--expect-count", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import el_wslc_security_check as SC
    sec = SC.check(a.security, a.expect_backend) if a.security else None
    try:
        verdict, p = write_report(a.out, run_dir=a.run, conf=_load(a.conformance), replay=_load(a.replay), sec=sec, probe=_load(a.toolchain_probe), suites=_load(a.suites), commit=a.commit,
                                  expect_count=a.expect_count, expect_backend=a.expect_backend)
    except FileExistsError as ex:
        print(str(ex))
        return 2
    print("%s -> %s" % (verdict, p))
    return 0 if verdict == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
