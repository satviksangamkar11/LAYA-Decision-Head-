"""[EL-BANK40] PC-side driver for the 40-task Kaggle run. It NEVER calls Kaggle or NVIDIA: it writes the push order, turns downloaded kernel outputs into the next input datasets, and compiles the results.

  python tools/el_bank40_driver.py plan   <pack dir from tools/pack_el_bank40_v1.py> [--resume]      -> PUSH_ORDER.md next to the pack (exact commands, in order)
  python tools/el_bank40_driver.py pass1-datasets <downloaded pass-1 output dir> <out dir>              -> <out>/pass1_out (tasks/* + manifest, for pass 2 and for resume) and <out>/critic_packets (for tools/pack_evidence_ladder_critic_batch_v1.py)
  python tools/el_bank40_driver.py critic-dataset <downloaded critic kernel output dir> <out dir>        -> <out>/critic_responses (responses.jsonl + MANIFEST, dataset for pass 2)
  python tools/el_bank40_driver.py compile <pass-1 output dir> [<later pass-1 versions> ...] [--pass2 <pass-2 output dir>] [--out DIR] -> BANK40_RESULTS.md / BANK40_RESULTS.json (create-only)

Every Kaggle output is untrusted data: shapes are checked, the thinking policy must be present and identical in every result, a task without DONE is reported as unfinished (never as a result)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_ladder import bank40 as B   # noqa: E402

LABEL = "development results (D1 waiver), public examples only; hidden grading NOT done"
OWNER = "sangeetasangamkar"


class IngestError(RuntimeError):
    pass


def jload(p: Path) -> Any: return json.loads(Path(p).read_text("utf-8"))


def read_run(d: Path) -> Dict[str, Any]:
    """-> {"summary": RUN_SUMMARY, "tasks": {id: result}, "done": set, "events": [...], "gpu": [...]} from a downloaded kernel output directory."""
    d = Path(d)
    if not (d / "RUN_SUMMARY.json").is_file(): raise IngestError("%s has no RUN_SUMMARY.json (not a kernel output)" % d)
    summ = jload(d / "RUN_SUMMARY.json")
    pol = summ.get("thinking_policy")
    if summ.get("verdict") not in ("REFUSED_NO_THINKING_POLICY",):
        B.check_policy(pol or {})                      # a run without an explicit policy is not a result
    tasks, done = {}, set()
    for p in sorted((d / "tasks").glob("*/result.json")):
        r = jload(p)
        if r.get("thinking_policy") != pol: raise IngestError("task %s was run with a different thinking policy than the run summary" % p.parent.name)
        tasks[p.parent.name] = r
        ct = p.parent / "controller_timeline.json"
        if ct.is_file():
            try: r["_controller_timeline_summary"] = jload(ct).get("summary")
            except ValueError: pass
        if (p.parent / "DONE").exists(): done.add(p.parent.name)
    ev = B.read_events(d / "TIMELINE.jsonl") if (d / "TIMELINE.jsonl").is_file() else []
    gpu = [json.loads(l) for l in (d / "timeline" / "gpu_util.jsonl").read_text("utf-8").splitlines() if l.strip()] if (d / "timeline" / "gpu_util.jsonl").is_file() else []
    return {"dir": d, "summary": summ, "tasks": tasks, "done": done, "events": ev, "gpu": gpu}


def merge_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Several pass-1 kernel versions of the SAME run (time guard / resume): later versions hold the earlier DONE tasks too; events and GPU samples are concatenated, the summary of the last
    version is kept with the wall seconds of every version recorded. Policies must be identical."""
    if len(runs) == 1: return runs[0]
    pol = runs[0]["summary"].get("thinking_policy")
    if any(r["summary"].get("thinking_policy") != pol for r in runs): raise IngestError("the kernel versions were run with different thinking policies")
    tasks, done = {}, set()
    for r in runs:
        for tid, res in r["tasks"].items():
            if tid in r["done"] or tid not in done: tasks[tid] = res
        done |= r["done"]
    summ = dict(runs[-1]["summary"]); summ["total_seconds"] = round(sum(r["summary"].get("total_seconds") or 0 for r in runs), 1); summ["kernel_versions_seconds"] = [r["summary"].get("total_seconds") for r in runs]
    return {"dir": runs[-1]["dir"], "summary": summ, "tasks": tasks, "done": done, "events": sorted([e for r in runs for e in r["events"]], key=lambda e: (e["ts"], e["mono"])), "gpu": [g for r in runs for g in r["gpu"]]}


def steps_of(events: Sequence[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    """Per-step rows {stage, start_utc, end_utc, seconds} for one task, from the paired *_start / *_end events (real UTC from the kernel clock)."""
    open_: Dict[tuple, List[Dict[str, Any]]] = {}
    out = []
    for e in events:
        if e.get("task") != task or not e["event"].endswith(("_start", "_end")): continue
        stage = e["event"].rsplit("_", 1)[0]
        key = (e.get("worker"), stage)
        if e["event"].endswith("_start"): open_.setdefault(key, []).append(e)
        elif open_.get(key):
            s = open_[key].pop()
            out.append({"stage": stage, "start_utc": s["ts_utc"], "end_utc": e["ts_utc"], "seconds": e.get("seconds"), "ok": e.get("ok"), **({"purpose": s.get("purpose")} if s.get("purpose") else {})})
    return sorted(out, key=lambda r: r["start_utc"])


def gpu_summary(rows: Sequence[Dict[str, Any]], events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[int, List[int]] = {}
    for r in rows: by.setdefault(int(r["gpu"]), []).append(int(r["util_pct"]))
    gen_by_worker: Dict[str, float] = {}
    for e in events:
        if e["event"] == "generate_end": gen_by_worker[e.get("worker")] = gen_by_worker.get(e.get("worker"), 0.0) + float(e.get("seconds", 0))
    wall = (max(e["ts"] for e in events) - min(e["ts"] for e in events)) if events else 0.0
    return {"per_gpu": {str(g): {"samples": len(v), "mean_util_pct": round(sum(v) / len(v), 1), "p50": B.pct(v, 50), "p95": B.pct(v, 95), "idle_samples_pct": round(100.0 * sum(1 for x in v if x < 5) / len(v), 1)} for g, v in sorted(by.items())},
            "generate_busy_seconds_by_worker": {k: round(v, 1) for k, v in gen_by_worker.items()}, "wall_seconds": round(wall, 1),
            "generate_busy_fraction_by_worker": {k: round(v / wall, 3) if wall else None for k, v in gen_by_worker.items()}}


def row_of(tid: str, res: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pub, cl, disc, l2 = res.get("public") or {}, res.get("clusters") or {}, res.get("discrimination") or {}, res.get("l2") or {}
    return {"task": tid, "language": res.get("language"), "difficulty": res.get("difficulty"), "pass": res.get("pass"), "status": res.get("status"), "candidates": res.get("candidates"),
            "public_examples": {"examples": pub.get("examples"), "best_candidate_passes": pub.get("best"), "candidates_passing_all": pub.get("candidates_passing_all"), "candidates_with_examples": pub.get("candidates_with_examples")},
            "l2": l2.get("verdict"), "l2_detail": {k: l2.get(k) for k in ("validated", "parsed", "dropped", "reason") if l2.get(k) is not None},
            "clusters": {"public": cl.get("public_clusters"), "extended": cl.get("extended_clusters")}, "critic_used": bool(disc.get("triggered")), "critic_state": disc.get("critic"),
            "final_label": res.get("answer_label"), "stop_cause": res.get("stop_cause"), "stop_details": res.get("stop_details"), "rounds": res.get("rounds"),
            "tokens": {"generated": res.get("tokens_generated"), "prompt": res.get("tokens_prompt"), "generate_calls": res.get("generate_calls")}, "ceiling_hits": res.get("ceiling_hits"),
            "invariants_failed": (res.get("invariants") or {}).get("failed"), "start_utc": res.get("start_utc"), "end_utc": res.get("end_utc"), "seconds": res.get("seconds"),
            "worker": res.get("worker"), "steps": steps_of(events, tid), "error": res.get("error"),
            "controller_stage_seconds": {k: v.get("total_s") for k, v in ((res.get("_controller_timeline_summary") or {}).get("stages") or {}).items()}}


def compile_results(pass1: Dict[str, Any], pass2: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Final row per task = the pass-2 result when pass 2 finished that task, else the pass-1 result. Unfinished tasks (no DONE) are listed, never given a label."""
    events = list(pass1["events"]) + (list(pass2["events"]) if pass2 else [])
    final: Dict[str, Dict[str, Any]] = {}
    for tid in sorted(pass1["done"]): final[tid] = pass1["tasks"][tid]
    if pass2:
        for tid in sorted(pass2["done"]): final[tid] = pass2["tasks"][tid]
    rows = [row_of(t, r, events) for t, r in sorted(final.items())]
    unfinished = sorted((set(pass1["tasks"]) | set(pass2["tasks"] if pass2 else [])) - set(final))
    allids = {t["task_id"] for t in []}
    stages = B.summarize(events)
    agg: Dict[str, List[float]] = {}
    for r in rows:
        for k, v in (r.get("controller_stage_seconds") or {}).items():
            if v is not None: agg.setdefault(k, []).append(float(v))
    controller_stages = {k: {"tasks": len(v), "total_s": round(sum(v), 3), "p50_s": B.pct(v, 50), "p95_s": B.pct(v, 95)} for k, v in sorted(agg.items())}
    per_lang: Dict[str, Dict[str, int]] = {}
    for r in rows: per_lang.setdefault(r["language"], {}).setdefault(r["final_label"], 0); per_lang[r["language"]][r["final_label"]] += 1
    pol = pass1["summary"].get("thinking_policy")
    secs = [s for s in (pass1["summary"].get("total_seconds"), pass2["summary"].get("total_seconds") if pass2 else None) if s]
    return {"label": LABEL, "scorer": "none", "thinking_policy": pol, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks_finished": len(rows), "tasks_expected": 40,
            "unfinished_or_errored": unfinished, "pending_not_started": pass1["summary"].get("pending_not_started"), "resumed_tasks": pass1["summary"].get("resumed_tasks"),
            "jail": pass1["summary"].get("jail"), "gpus": pass1["summary"].get("gpus"), "model": pass1["summary"].get("model"), "labels_by_language": per_lang,
            "kernel_wall_seconds": {"pass1": pass1["summary"].get("total_seconds"), "pass2": pass2["summary"].get("total_seconds") if pass2 else None, "total": round(sum(secs), 1)},
            "stage_summary": stages, "controller_stage_seconds_per_task": controller_stages, "model_vs_sandbox": {"model_seconds": stages["model_seconds"], "sandbox_seconds": stages["sandbox_seconds"], **stages["model_vs_sandbox"]},
            "gpu": gpu_summary(pass1["gpu"], pass1["events"]), "critic": {"packets_exported": (pass1["summary"].get("critic_packets") or {}).get("exported"), "pass2_responses": pass2["summary"].get("critic_responses") if pass2 else None},
            "ceiling_hits_total": sum(int(r.get("ceiling_hits") or 0) for r in rows), "rows": rows}


def render_md(c: Dict[str, Any]) -> str:
    pol = c["thinking_policy"] or {}
    L = ["# BANK40 RESULTS", "", "**%s**" % c["label"], "",
         "thinking_policy: `%s`; scorer: none (no LAYA); tasks finished %d / %d; created %s" % (json.dumps(pol, sort_keys=True), c["tasks_finished"], c["tasks_expected"], c["created_utc"]),
         "jail: %s; GPUs: %s; kernel wall seconds: %s" % (json.dumps({k: v for k, v in (c["jail"] or {}).items() if k in ("status", "java_home", "bwrap_swapped")}), c["gpus"], json.dumps(c["kernel_wall_seconds"])),
         "unfinished / errored: %s; pending (time guard): %s; resumed from earlier version: %s" % (c["unfinished_or_errored"], c["pending_not_started"], c["resumed_tasks"]), ""]
    if pol.get("policy") == "prompt_brief": L += ["generations that hit the technical token ceiling (counted as FAILED generations, never as success): %s" % c["ceiling_hits_total"], ""]
    L += ["labels by language: " + json.dumps(c["labels_by_language"], sort_keys=True), "",
          "| task | lang | diff | cand | public ex (best/total; cands all-pass) | L2 | clusters pub/ext | critic | label | stop | tokens gen/prompt | seconds | start UTC | end UTC |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in c["rows"]:
        pe = r["public_examples"]
        L.append("| %s | %s | %s | %s | %s/%s; %s | %s | %s/%s | %s | **%s** | %s | %s/%s | %s | %s | %s |" % (r["task"], r["language"], r["difficulty"], r["candidates"], pe["best_candidate_passes"], pe["examples"], pe["candidates_passing_all"], r["l2"],
                 r["clusters"]["public"], r["clusters"]["extended"], (r["critic_state"] or "yes") if r["critic_used"] else "no", r["final_label"], r["stop_cause"], r["tokens"]["generated"], r["tokens"]["prompt"], r["seconds"], r["start_utc"], r["end_utc"]))
    L += ["", "## Per-stage seconds (all tasks)", "", "| stage | n | total s | p50 s | p95 s |", "|---|---|---|---|---|"]
    for k, s in c["stage_summary"]["stages"].items(): L.append("| %s | %d | %s | %s | %s |" % (k, s["n"], s["total_s"], s["p50_s"], s["p95_s"]))
    if c["controller_stage_seconds_per_task"]:
        L += ["", "## Controller stages (seconds per task, from the controller's own timeline)", "", "| stage | tasks | total s | p50 s | p95 s |", "|---|---|---|---|---|"]
        for k, s in c["controller_stage_seconds_per_task"].items(): L.append("| %s | %d | %s | %s | %s |" % (k, s["tasks"], s["total_s"], s["p50_s"], s["p95_s"]))
    ms = c["model_vs_sandbox"]
    L += ["", "model seconds %s, sandbox seconds %s, model share %s; first event %s, last event %s, timeline wall %s s" % (ms["model_seconds"], ms["sandbox_seconds"], ms["model_share"], c["stage_summary"]["first_utc"], c["stage_summary"]["last_utc"], c["stage_summary"]["wall_seconds"]),
          "", "## GPU utilisation (nvidia-smi every 30 s)", "", json.dumps(c["gpu"], indent=1), "", "## Per-step times (UTC)", ""]
    for r in c["rows"]:
        L.append("**%s** (%s -> %s): " % (r["task"], r["start_utc"], r["end_utc"]) + "; ".join("%s %s..%s (%ss)" % (s["stage"], s["start_utc"][11:], s["end_utc"][11:], s["seconds"]) for s in r["steps"][:40]) + (" ..." if len(r["steps"]) > 40 else ""))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ datasets for the next steps
def pass1_datasets(src: Path, out: Path) -> Dict[str, Any]:
    run = read_run(src)
    if out.exists(): raise IngestError("output dir exists (create-only): %s" % out)
    p1 = out / "pass1_out"
    for tid in sorted(run["done"]):
        shutil.copytree(Path(src) / "tasks" / tid, p1 / "tasks" / tid)
    (p1 / "dataset-metadata.json").write_text(json.dumps({"title": "el-bank40-pass1-out-v1", "id": "%s/el-bank40-pass1-out-v1" % OWNER, "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    cp = Path(src) / "critic_packets"
    if cp.is_dir(): shutil.copytree(cp, out / "critic_packets")
    rec = {"done": len(run["done"]), "unfinished": sorted(set(run["tasks"]) - run["done"]), "critic_packets": (len(list((out / "critic_packets").glob("packets.jsonl"))) if cp.is_dir() else 0)}
    (out / "INGEST.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def critic_dataset(src: Path, out: Path) -> Dict[str, Any]:
    src = Path(src); co = src / "critic_out" if (src / "critic_out").is_dir() else src
    if not (co / "responses.jsonl").is_file(): raise IngestError("no responses.jsonl under %s" % src)
    man = jload(co / "MANIFEST.json")
    n = 0
    for ln in (co / "responses.jsonl").read_text("utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if not r.get("packet_id") or "response" not in r: raise IngestError("a response line has no packet_id/response")
            n += 1
    d = Path(out) / "critic_responses"; d.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(co / "responses.jsonl", d / "responses.jsonl"); shutil.copyfile(co / "MANIFEST.json", d / "MANIFEST.json")
    (d / "dataset-metadata.json").write_text(json.dumps({"title": "el-bank40-critic-responses-v1", "id": "%s/el-bank40-critic-responses-v1" % OWNER, "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    return {"responses": n, "states": man.get("states")}


def plan(pack: Path, resume: bool = False) -> str:
    pk = jload(Path(pack) / "PACK.json")
    ds, k1, k2 = pk["dataset"], pk["kernels"]["pass1"], pk["kernels"]["pass2"]
    pol = json.dumps(pk["thinking_policy"])
    L = ["# EL-BANK40 PUSH ORDER (exact; the coordinator runs these, this driver runs none)", "", "thinking policy in the bundle: `%s`; commit: %s" % (pol, pk["commit"]), "",
         "1. Create/refresh the bundle dataset (private): `kaggle datasets create -p \"%s\"` (first time) or `kaggle datasets version -p \"%s\" -m bank40 bundle`" % (ds, ds),
         "2. Push the GPU kernel (T4 x2, internet off): `kaggle kernels push -p \"%s\"`; if your CLI supports it add `--accelerator NvidiaTeslaT4`; check on the Kaggle page that the session shows 2 GPUs." % k1,
         "3. When COMPLETE: `kaggle kernels output %s/el-bank40-run-pass1-v1 -p <pass1_out_dir>`" % OWNER,
         "4. If RUN_SUMMARY.json says PARTIAL_TIME_GUARD / PARTIAL: `python tools/el_bank40_driver.py pass1-datasets <pass1_out_dir> <ds_dir>`, upload `<ds_dir>/pass1_out` as dataset `el-bank40-pass1-out-v1`, add it to `dataset_sources` of the pass-1 kernel (resume) and push again; finished tasks (DONE) are skipped. Repeat until done.",
         "5. Critic: `python tools/el_bank40_driver.py pass1-datasets <pass1_out_dir> <ds_dir>` then `python tools/pack_evidence_ladder_critic_batch_v1.py <ds_dir>/critic_packets`; upload that dataset and push the EXISTING critic kernel (kaggle/evidence_ladder_critic_v2, internet on, GPU off, Secret NVIDIA_API_KEYS added by the user). Download its output.",
         "6. `python tools/el_bank40_driver.py critic-dataset <critic_output_dir> <ds_dir2>`; upload `<ds_dir2>/critic_responses` as dataset `el-bank40-critic-responses-v1` and `<ds_dir>/pass1_out` as `el-bank40-pass1-out-v1`.",
         "7. Push the pass-2 kernel (CPU; jail on): `kaggle kernels push -p \"%s\"`; download its output into <pass2_out_dir>." % k2,
         "8. Results: `python tools/el_bank40_driver.py compile <pass1_out_dir> --pass2 <pass2_out_dir> --out <results dir>` -> BANK40_RESULTS.md/json.", "",
         "If the jail cannot start on Kaggle the kernel stops with verdict JAIL_UNAVAILABLE and executes nothing (no unsandboxed fallback). Never print or commit key values."]
    text = "\n".join(L) + "\n"
    with open(Path(pack) / "PUSH_ORDER.md", "x", encoding="utf-8") as f: f.write(text)
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan"); a.add_argument("pack"); a.add_argument("--resume", action="store_true")
    b = sub.add_parser("pass1-datasets"); b.add_argument("src"); b.add_argument("out")
    c = sub.add_parser("critic-dataset"); c.add_argument("src"); c.add_argument("out")
    d = sub.add_parser("compile"); d.add_argument("pass1", nargs="+", help="one or more pass-1 output dirs (kernel versions of the same run, oldest first)"); d.add_argument("--pass2"); d.add_argument("--out", default=str(ROOT / "results/raw" / ("el-bank40-results-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))))
    n = ap.parse_args(argv)
    if n.cmd == "plan": print(plan(Path(n.pack), n.resume)); return 0
    if n.cmd == "pass1-datasets": print(json.dumps(pass1_datasets(Path(n.src), Path(n.out)))); return 0
    if n.cmd == "critic-dataset": print(json.dumps(critic_dataset(Path(n.src), Path(n.out)))); return 0
    out = Path(n.out); out.mkdir(parents=True, exist_ok=False)
    comp = compile_results(merge_runs([read_run(Path(x)) for x in n.pass1]), read_run(Path(n.pass2)) if n.pass2 else None)
    (out / "BANK40_RESULTS.json").write_text(json.dumps(comp, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "BANK40_RESULTS.md").write_text(render_md(comp), encoding="utf-8")
    print(json.dumps({"out": str(out), "finished": comp["tasks_finished"], "labels": comp["labels_by_language"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
