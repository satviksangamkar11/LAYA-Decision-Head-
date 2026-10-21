"""CPU-only receipt audit; no generation, execution, hidden grader, or editor calls.
Run from repository root: uv run --no-sync python results/raw/h2lite-handoff-20261021/score_receipts.py
Creates SCORING_RECEIPT.json once; refuses to overwrite it.
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evidence_ladder import v2min as V
from evidence_ladder.common.interfaces import TaskSpec
from evidence_ladder.controller import generation as GEN


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    target = OUT / "SCORING_RECEIPT.json"
    assert not target.exists(), "create-only receipt already exists"
    raw = ROOT / "results/raw/h2lite-search-r1"
    remote = OUT / "remote-output"
    h = raw / "h2_search"
    bundle = ROOT / "results/raw/el-h2lite-bundle-search-20261021T132739Z/dataset"
    man = load(bundle / "BUNDLE_MANIFEST.json")
    assert all(sha(bundle / f) == digest for f, digest in man["files"].items())
    compared = {}
    for p in raw.rglob("*"):
        if p.is_file():
            rel = p.relative_to(raw).as_posix()
            other = remote / rel
            assert other.is_file(), rel
            compared[rel] = {"committed_sha256": sha(p), "download_sha256": sha(other)}
            if rel != "el-h2lite-v1.log":
                assert sha(p) == sha(other), rel
    # The original export's kernel log was empty; the fresh download preserves the full log.
    assert (raw / "el-h2lite-v1.log").stat().st_size == 0
    log = load(remote / "el-h2lite-v1.log")
    assert any("VERDICT OK" in x.get("data", "") for x in log)
    rs, cfg = load(raw / "RUN_SUMMARY.json"), load(h / "RUN_CONFIG.json")
    assert rs["verdict"] == "OK" and rs["exit_code"] == 0
    assert rs["bundle"]["commit"] == man["commit"]
    assert rs["jail"] == "PASS" and rs["qualification_ok"] and rs["model_sha256_ok"]
    qual = load(raw / "qualification/QUAL.json")
    assert qual["all_ok"] and all(x["ok"] for x in qual["checks"].values())
    assert cfg["G"] == 6000 and cfg["phase"] == "search" and cfg["fake"] is False
    assert cfg["v2min"] == V.VERSION == "v2min_v1.2"
    assert cfg["tool_sha256"] == man["files"]["tools/el_h2lite_run.py"]
    search = ["P1-%02d" % n for n in (2,4,6,7,9,12,29,30,35,38,39,11,21,23,26,34,16,28,37,14,20,25)]
    confirm = ["P1-%02d" % n for n in (1,3,5,8,10,13,19,24,17,27,36,40,18,31,32,33,15,22)]
    assert cfg["tasks"] == search and not set(search) & set(confirm)
    names = ["baseline", "general", "coding"]
    assert list(cfg["configs"]) == names
    results, calls, answers, gen = [rows(h / f) for f in ("RESULTS.jsonl", "CALLS.jsonl", "ANSWERS.jsonl", "gen_log.jsonl")]
    expected = {(n,t) for n in names for t in search}
    assert len(results) == len(answers) == 66
    assert {(r["config"],r["task"]) for r in results} == expected
    assert {(r["config"],r["task"]) for r in answers} == expected
    assert len(calls) == len(gen) == 489 and not any(c["reused"] for c in calls)
    for c,g in zip(calls,gen):
        assert c["task"] == g["task_id"] and c["template"] == g["template"]
        assert g["model"] == rs["plan"]["model_repo"] and g["model_revision"] == rs["plan"]["model_revision"]
        for k in ("seed","prompt_sha256","purpose","tokens_generated","tokens_prompt","extract_status"):
            assert c[k] == g[k], k
    props = load(ROOT / "results/raw/h2lite-editor-v1/round1_proposals.json")
    ads = {v["name"]: V.PromptAdapter("h2_" + v["name"], v["texts"]) for v in props["variants"]}
    for v in props["variants"]:
        assert ads[v["name"]].sha256 == v["sha256"] == cfg["configs"][v["name"]][1]
    taskfile = "results/raw/codex-pilot1-20261016/fixed_v2/worker/tasks.jsonl"
    tasks = {r["task_id"]: r for r in rows(ROOT / taskfile)}
    assert sha(ROOT / taskfile) == man["files"][taskfile]
    aidx = {(a["config"],a["task"]): a for a in answers}
    original_summary = load(h / "SUMMARY.json")
    metrics = {}
    task_receipts = []
    for name in names:
        rr = [r for r in results if r["config"] == name]
        cc = [c for c in calls if c["arm"] == name]
        for r in rr:
            t = r["task"]
            assert bool(aidx[name,t]["source"]) == r["accepted"]
            assert r["contract"] in (("pass","fail","inconclusive","unavailable") if r["accepted"] else (None,))
            tc = [c for c in cc if c["task"] == t]
            assert len(tc) >= 4
            assert [c["seed"] for c in tc] == [1,2,3,4] + list(range(104,104+len(tc)-4))
            assert sum(c["tokens_generated"] for c in tc) == r["tokens_generated"]
            assert sum(c["tokens_prompt"] for c in tc) == r["tokens_prompt"]
            tr = tasks[t]
            task = TaskSpec(t, "bank:"+t, "python", "python", tr["prompt"], tr["public_tests"], tr.get("difficulty", ""))
            used = 0
            for j,c in enumerate(tc):
                tid = GEN.strategy_for(j)
                prompt = ads[name].render(tid,task) if name in ads else GEN.render_sample_prompt(tid,task)[0]
                assert c["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
                assert c["template"] == (ads[name].adapted_id(tid) if name in ads else tid)
                cap = 4096 if j < 4 else min(4096,6000-used)
                assert 0 < cap and 0 <= c["tokens_generated"] <= cap
                if j >= 4:
                    used += c["tokens_generated"]
            assert used <= 6000
            if not r["accepted"]:
                assert used == 6000, (name,t,used)
            task_receipts.append(dict(r, generation_calls=len(tc), extraction_failures=sum(c["extract_status"] != "ok" for c in tc), inference_call_wall_seconds=round(sum(c["actual_seconds"] for c in tc),3)))
        accepted = sum(r["accepted"] for r in rr)
        violations = sum(r["accepted"] and r["contract"] == "fail" for r in rr)
        m = dict(accepted=accepted, contract_violations_among_accepted=violations,
                 extraction_failures=sum(c["extract_status"] != "ok" for c in cc),
                 actual_generated_tokens=sum(c["tokens_generated"] for c in cc),
                 generation_calls=len(cc), score=accepted-2*violations,
                 inference_call_wall_seconds=round(sum(c["actual_seconds"] for c in cc),3),
                 configuration_wall_seconds=original_summary[name]["seconds"],
                 task_wall_seconds=round(sum(r["seconds"] for r in rr),1),
                 contract_verdicts=dict(Counter(r["contract"] for r in rr if r["accepted"])),
                 not_accepted=[r["task"] for r in rr if not r["accepted"]], incomplete_tasks=[])
        for k in ("accepted","contract_violations_among_accepted","extraction_failures","actual_generated_tokens","score"):
            assert m[k] == original_summary[name][k], (name,k)
        assert len(cc) == original_summary[name]["samples"]
        metrics[name] = m
    b = metrics["baseline"]
    for m in metrics.values():
        m["token_ratio_vs_B"] = m["actual_generated_tokens"] / b["actual_generated_tokens"]
        m["eligible"] = 4*m["actual_generated_tokens"] <= 5*b["actual_generated_tokens"]
    selected = sorted((n for n in names if metrics[n]["eligible"]), key=lambda n: (-metrics[n]["score"],metrics[n]["actual_generated_tokens"]))[0]
    best_gain = max(metrics[n]["score"]-b["score"] for n in names[1:] if metrics[n]["eligible"])
    assert selected == "baseline" and best_gain < 2
    cf = ROOT / "results/raw/p6-cf-validation-v1"
    cv, cq = load(cf / "VALIDATION_v3_counts.json"), load(cf / "QUALIFIED_rules_v2.json")
    eligible = [p for p,v in cv.items() if v["validated"] and v["tests"] >= 10]
    order = sorted(eligible,key=lambda p: hashlib.sha256(("P6-2026-10-21:"+p).encode()).hexdigest())
    assert len(cv) == 77 and len(order) == 72 and order == cq["qualified_in_rule_order"]
    attempt_counts = Counter(next(i+1 for i,a in enumerate(cv[p]["attempts"]) if a["pass"] == cv[p]["tests"]) for p in order)
    assert attempt_counts == {1:65,2:7}
    assert load(cf / "QUAL_cpp_kaggle.json")["all_runnable_ok"]
    cprivate = ROOT / "private/p6-cf-v1/validate-v3"
    assert sha(cprivate / "VALIDATION_v3.json") == sha(cf / "VALIDATION_v3_counts.json")
    extra_paths = [ROOT / f for f in ("H2LITE_PLAN_v1.md","P6_PROTOCOL_DRAFT_v1.md","P6_CF_QUALIFICATION_RULES_v1.md","P6_CF_QUALIFICATION_RULES_v2.md","P6_CHECKPOINT_2026_10_21.md","results/raw/h2lite-editor-v1/round1_proposals.json","results/raw/h2lite-editor-v1/LEDGER.jsonl")]
    extra_paths += [p for p in cf.iterdir() if p.is_file()]
    extra_paths += [bundle / "BUNDLE_MANIFEST.json", ROOT / "kaggle/el_h2lite_v1.py", ROOT / "tools/el_h2lite_run.py", ROOT / "evidence_ladder/v2min.py", cprivate / "el-p6-cf-validate-v3.log"]
    status = subprocess.run([sys.executable,"-m","kaggle","kernels","status","sangeetasangamkar/el-h2lite-v1"],capture_output=True,text=True,check=True)
    assert "COMPLETE" in status.stdout
    receipt = dict(audited_utc=datetime.now(timezone.utc).isoformat(), git_commit=git("rev-parse","HEAD"),
                   remote_branch_commit=git("rev-parse","origin/v1.5-p3-verification"), bundle_commit=man["commit"],
                   kernel="sangeetasangamkar/el-h2lite-v1", live_status=status.stdout.strip(),
                   run_summary=rs, run_config=cfg, verified_bundle_files=len(man["files"]),
                   committed_vs_remote=compared,
                   source_hashes={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in extra_paths},
                   bundle_current_source_matches={f:sha(ROOT/f)==digest for f,digest in man["files"].items() if (ROOT/f).is_file()},
                   metrics=metrics, task_receipts=task_receipts,
                   selection=dict(selected=selected,h2_winner=None,best_eligible_variant_score_gain=best_gain,stop=True,confirmation_run=False,status="NOT QUALIFIED FOR P6"),
                   gpu=dict(approved_seconds=15120,prior_kernel_wall_seconds=rs["total_seconds"],additional_handoff_seconds=0,total_kernel_wall_seconds=rs["total_seconds"],remaining_seconds=15120-rs["total_seconds"],inference_call_wall_seconds=round(sum(c["actual_seconds"] for c in calls),3),billing_device_seconds=None),
                   codeforces=dict(qualified_count=len(order),qualified_in_rule_order=order,qualified_reference_attempts=dict(attempt_counts),exclusions={p:{"recorded_reason":r,"validation":cv[p]} for p,r in cq["reasons"].items() if r!="QUALIFIED"},output_limit_diagnosis="UNCONFIRMED: aggregate other counts; no per-case status/detail/output-size receipts"),
                   limitations=["No hidden grader used on SEARCH; correctness and false acceptance rates are unmeasured.","Only P1-11, P1-21, P1-30 have contract checks among accepted tasks; other accepted tasks have unavailable contracts.","Call seconds are host wall time around inference, not CUDA event time; Kaggle billing/device count is not recorded.","Per-call max_new_tokens is absent from logs; prompt, seed, observed token counts and reconstructed frozen cap verified, actual request caps supported by hashed runner code.","Exact Kaggle version/session ID absent from exported receipts; bundle commit, UTC interval and remote hashes bind this run.","Original kernel log empty; freshly retrieved log is preserved separately."])
    with target.open("x",encoding="utf-8") as f:
        json.dump(receipt,f,indent=2)
        f.write("\n")
    print(json.dumps({"receipt":str(target),"metrics":metrics,"selection":receipt["selection"],"gpu":receipt["gpu"],"bundle_mismatches":[f for f,ok in receipt["bundle_current_source_matches"].items() if not ok]},indent=2))


if __name__ == "__main__":
    main()
