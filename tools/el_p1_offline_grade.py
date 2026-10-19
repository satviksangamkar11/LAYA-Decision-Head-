"""[P1, user GO 2026-10-20] OFFLINE grading of STORED candidates after selection. Read-only on the runs; writes a NEW create-only directory.

What it does, per finished task of the given run directories:
  * reads the frozen candidate pool from the task's hash-chained journal (id, source, origin, parent, lineage), the per-candidate public-example evidence that the
    run recorded, and the run's own final selection (the `decision` / `finalize` record); nothing is re-generated, nothing is re-selected;
  * grades every stored candidate on the DEVELOPMENT grader cases (execution/protected.py, gate CLOSED first) inside the qualified sandbox with the trusted verifier
    (python -> MXC process container, java -> MXC WSLC, as in the original runs; one fresh process per input, verdict host-side);
  * a sandbox / infrastructure failure is retried once and then recorded INFRA_ERROR (never a candidate verdict);
  * stores only counts and the failure class per candidate (no case inputs, no expected values), like HiddenGrade.
Isolation: this is evaluation-plane code under tools/ (not a runtime module); its outputs go to results/raw/ only and are never read by the generator, selector, repair loop or any store.
The development cases are public in Git history: every number here is DEVELOPMENT evidence only.

Usage: PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe tools/el_p1_offline_grade.py <new out dir> <run dir> [<run dir> ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from evidence_ladder.execution import config as K
from evidence_ladder.execution import engine
from evidence_ladder.execution.protected import ProtectedGrader, SelectionGate
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
from evidence_ladder.execution.trusted_verify import VERSION as TV_VERSION

BACKEND = {"python": "mxc", "java": "wslc"}      # the backends the original runs used
ANSWERED = ("SUPPORTED", "VERIFIED", "PLAUSIBLE")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_task(tdir: Path) -> dict:
    cands, pub, decision, finalize = {}, {}, None, None
    for line in (tdir / "journal.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        k, d = r["kind"], r["data"]
        if k == "candidate":
            cands[d["id"]] = {"id": d["id"], "origin": d.get("origin"), "parent": d.get("parent"), "lineage": d.get("lineage"), "source": d["source"]}
        elif k == "evidence" and d["evidence"].get("check_id") == "public_examples_v1":
            e = d["evidence"]
            pub.setdefault(e["candidate_id"], []).append(e["status"])
        elif k == "decision":
            decision = d
        elif k == "finalize":
            finalize = d
    res = json.loads((tdir / "result.json").read_text(encoding="utf-8"))
    sel = decision or finalize or {}
    return {"task": res["task"], "language": res["language"], "difficulty": res.get("difficulty"), "label": res.get("answer_label"), "rounds": res.get("rounds"),
            "seconds": res.get("seconds"), "result_public": res.get("public"), "selected": sel.get("candidate_id"), "selected_label": sel.get("answer_label"),
            "cands": cands, "pub": pub, "journal_sha256": sha(tdir / "journal.jsonl")}


def main(out: Path, runs: list) -> int:
    if out.exists():
        print("refusing: %s exists (create-only)" % out); return 2
    out.mkdir(parents=True)
    gate = SelectionGate(); gate.close()                 # selection ended long ago in the original runs; grading only after the gate is closed
    grader = ProtectedGrader(gate)
    cfg = K.SandboxConfig()
    backends = {}
    fc, ft = open(out / "CANDIDATES.jsonl", "a", encoding="utf-8"), open(out / "TASKS.jsonl", "a", encoding="utf-8")      # written row by row: nothing done is lost if the run stops
    tasks, cand_rows, provenance = [], [], {"runs": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for run in runs:
        run = Path(run)
        provenance["runs"][str(run)] = {}
        for tdir in sorted((run / "tasks").iterdir()):
            if not (tdir / "DONE").exists() or not (tdir / "journal.jsonl").exists():
                continue
            t = load_task(tdir)
            if not t["cands"]:
                continue
            provenance["runs"][str(run)][t["task"]] = t["journal_sha256"]
            lang = t["language"]
            cases = grader._cases(lang, t["task"])          # read only after the gate closed; never leaves this process except as counts
            if lang not in backends:
                b = ProviderExecutionBackend(backend=BACKEND[lang], scratch=out / "_scratch" / lang, config=cfg, trusted=True)
                info = b.start(languages=(lang,))
                backends[lang] = b
                provenance.setdefault("backends", {})[lang] = {k: info[k] for k in ("engine", "backend", "image_digest")} | {"toolchains": info["toolchains"], "engine_version": TV_VERSION}
            b = backends[lang]
            rows = []
            for c in t["cands"].values():
                p = t["pub"].get(c["id"], [])
                row = {"task": t["task"], "candidate": c["id"], "origin": c["origin"], "parent": c["parent"], "lineage": c["lineage"],
                       "public_n": len(p), "public_pass": bool(p) and all(s == "pass" for s in p), "selected": c["id"] == t["selected"]}
                if cases is None:
                    row.update(dev="NOT_EVALUATED", dev_reason="no development cases for this task")
                else:
                    for attempt in range(2):
                        try:
                            res = b._engine_run(lang, c["source"].encode("utf-8"), [x["input"] for x in cases])
                            v, _ = engine.judge_each(lang, res, cases, cfg)
                            if v.failure_class == "sandbox_error":
                                raise RuntimeError("sandbox_error: " + (v.detail or "")[:200])
                            row.pop("dev_reason", None)
                            row.update(dev="PASS" if v.status == "pass" else "FAIL", dev_passed=v.passed, dev_total=v.total, dev_failure=v.failure_class or None)
                            break
                        except Exception as e:      # noqa: BLE001  infrastructure, never a candidate verdict
                            row.update(dev="INFRA_ERROR", dev_reason="%s: %s" % (type(e).__name__, str(e)[:200]))
                    b.exec_log.clear()
                rows.append(row)
                fc.write(json.dumps(row, sort_keys=True) + "\n"); fc.flush()
                print(t["task"], c["id"][:10], c["origin"], "public", row["public_pass"], "dev", row["dev"], row.get("dev_passed"), "/", row.get("dev_total"), flush=True)
            cand_rows += rows
            ev = [r for r in rows if r["dev"] in ("PASS", "FAIL")]
            sel = next((r for r in rows if r["selected"]), None)
            answered = t["label"] in ANSWERED
            tasks.append({"task": t["task"], "language": lang, "difficulty": t["difficulty"], "label": t["label"], "rounds": t["rounds"], "seconds": t["seconds"],
                          "pool": len(rows), "samples": sum(1 for r in rows if str(r["origin"]).startswith("sample")), "repairs": sum(1 for r in rows if str(r["origin"]).startswith("repair")),
                          "evaluated": len(ev), "not_evaluated": len(rows) - len(ev),
                          "public_pass_n": sum(r["public_pass"] for r in rows), "dev_pass_n": sum(r["dev"] == "PASS" for r in rows),
                          "public_and_dev_pass_n": sum(r["public_pass"] and r["dev"] == "PASS" for r in rows),
                          "answered": answered, "selected_public_pass": bool(sel and sel["public_pass"]), "selected_dev": sel["dev"] if sel else None,
                          "result_public_candidates_passing_all": (t["result_public"] or {}).get("candidates_passing_all")})
            ft.write(json.dumps(tasks[-1], sort_keys=True) + "\n"); ft.flush()
    fc.close(); ft.close()
    full = [t for t in tasks if t["not_evaluated"] == 0]
    oracle = [t for t in full if t["dev_pass_n"] > 0]
    chosen = [t for t in full if t["answered"] and t["selected_dev"] == "PASS"]
    misses = [t["task"] for t in oracle if not (t["answered"] and t["selected_dev"] == "PASS")]
    false_supp = [t["task"] for t in full if t["answered"] and t["selected_dev"] == "FAIL"]
    summary = {"label": "DEVELOPMENT evidence only: the development grader cases are public in Git history; offline, after selection; no model calls",
               "tasks": len(tasks), "fully_evaluated_tasks": len(full),
               "oracle_at_k": len(oracle), "chosen_pass": len(chosen), "selection_misses": misses, "generation_gap_tasks": [t["task"] for t in full if t["dev_pass_n"] == 0],
               "answered_but_dev_fail": false_supp,
               "public_pass_single": [t["task"] for t in tasks if t["answered"] and t["public_pass_n"] == 1],
               "public_pass_multi": [t["task"] for t in tasks if t["answered"] and t["public_pass_n"] >= 2],
               "public_count_mismatch_vs_result_json": [t["task"] for t in tasks if t["result_public_candidates_passing_all"] is not None and t["result_public_candidates_passing_all"] != t["public_pass_n"]],
               "infra_errors": sum(r["dev"] == "INFRA_ERROR" for r in cand_rows), "not_evaluated": sum(r["dev"] == "NOT_EVALUATED" for r in cand_rows)}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    provenance["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance["tool_sha256"] = sha(Path(__file__))
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=1, sort_keys=True), encoding="utf-8")
    (out / "SHA256SUMS.txt").write_text("".join("%s  %s\n" % (sha(out / n), n) for n in ("CANDIDATES.jsonl", "TASKS.jsonl", "SUMMARY.json", "PROVENANCE.json")), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), sys.argv[2:]))
