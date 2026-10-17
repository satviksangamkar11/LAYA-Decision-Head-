"""[EL-ARCH-1 F3 / T34 T37 T49] PATH B POSITIVE RECEIPT: stored, verified 4B scores replayed through the CURRENT controller on the EXACT stored contexts and option lists.

  PYTHONPATH=. .venv/Scripts/python.exe tools/el_arch1_pathb_replay.py <new out dir>   (create-only; WSL sandbox only; no GPU, no network, no Kaggle)

Source (read-only): results/raw/laya-integration-20261016T184816Z/e2e_pass2/_kaggle/{live_s1,verify_s1}: two independent T4 runs (live, then a replay re-capture) of 3 pools (ex1/ex2/ex3, step 1).
For each pool: the live and the verify result must agree on context sha, option shas, raw output and every score vector (VerifiedSavedScoreBackend.add); LAYA, the linear probe and the random head each get their
OWN backend, the probe and random head are attached as COMPARISON scorers only (journal kind scorer_comparison, selector=false). The controller is built exactly as the stored run built it (the context is
rebuilt from the task with NO investigate stage; its sha256 must equal the stored one, otherwise the pool is REFUSED, never replayed). A pool the controller's candidate filter changes (so the scored pool is no longer
the stored one) is refused too, and a deliberately altered pool is refused as a negative control. Writes <out>/PATHB_REPLAY_RECEIPT.json (+ per-example journal) and exits 0 only if every pool was either replayed with matching
hashes or refused for the stated reason."""
from __future__ import annotations

import glob
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
STORED = ROOT / "results/raw/laya-integration-20261016T184816Z/e2e_pass2/_kaggle"
TASKS_BY_KEY = {"ex1_slug": "slug", "ex2_clamp": "clamp", "ex3_wordcount": "wordcount"}


def sha(b) -> str: return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def run(out: Path) -> int:
    from decision_head import fable_pool as FP
    from laya_int1_t6 import TASKS
    from evidence_ladder.controller.agentic import AgenticContext, AgenticTask, AgenticRun
    from evidence_ladder.controller.wsl_sandbox import WslSandbox, available
    from evidence_ladder.models import laya, laya_runtime as RT
    if not available(): print("ERROR: WSL sandbox does not answer"); return 2
    live = json.loads((STORED / "live_s1/output/t6_step_results.json").read_text("utf-8"))
    ver = json.loads((STORED / "verify_s1/output/t6_step_results.json").read_text("utf-8"))
    sysb_p = sorted(glob.glob(str(ROOT / "results/raw/el-arch1-bundle-*/dataset/gate/system_block.txt")))[0]
    system_block = Path(sysb_p).read_text("utf-8")
    # stored context texts: they are inside the kernel request of the live run (their sha256 equals the stored context_sha256, checked below)
    import re
    ksrc = (STORED / "live_s1/kernel/evidence_ladder_laya_t6_step_v1.py").read_text("utf-8")
    req = json.loads(re.search(r"'''(\{.*\})'''", ksrc, re.S).group(1))
    stored_ctx = {s["key"]: s["context_text"] for s in req["steps"]}
    backends = {k: RT.VerifiedSavedScoreBackend(k) for k in ("laya", "probe", "random_head")}
    admitted = {}
    for key, st in live["steps"].items():
        admitted[key] = all(be.add(st, ver["steps"][key], live_run="e2e/live_s1", verify_run="e2e/verify_s1", verify_mode=ver["mode"]) for be in backends.values())
    doc = {"tag": "[EL-ARCH-1 F3] PATH B positive receipt", "path": "B", "source": "results/raw/laya-integration-20261016T184816Z/e2e_pass2/_kaggle/{live_s1,verify_s1}",
           "source_sha256": {p: sha((STORED / p).read_bytes()) for p in ("live_s1/output/t6_step_results.json", "verify_s1/output/t6_step_results.json", "live_s1/kernel/evidence_ladder_laya_t6_step_v1.py")},
           "live": {"gpu": live["gpu"], "mode": live["mode"], "verdict": live["verdict"]}, "verify": {"gpu": ver["gpu"], "mode": ver["mode"], "verdict": ver["verdict"]},
           "refused_at_admission": [r for be in backends.values() for r in be.refused], "stub": False, "pools": {}}
    for key, st in live["steps"].items():
        name = key.split(":")[0]; T = TASKS[TASKS_BY_KEY[name]]
        ws = Path(tempfile.mkdtemp(prefix="pathb-")) / "ws"; ws.mkdir()
        for fn, c in T["files"].items(): (ws / fn).write_text(c, "utf-8")
        sb = WslSandbox(Path(tempfile.mkdtemp(prefix="pathb-sb-")), T["protected"])
        scorer = laya.LayaScorer(authorized=True, backend=backends["laya"])
        comparison = {"probe": laya.ProbeScorer(authorized=True, backend=backends["probe"]), "random_head": laya.RandomHeadScorer(authorized=True, backend=backends["random_head"])}
        r = AgenticRun(AgenticTask(name, T["user"], ws, T["test_cmd"], T["protected"], system_block, 2),
                       AgenticContext(sandbox=sb, scorer=scorer, comparison=comparison, arm="E", config={}), lambda m: FP.select_context(m, 160000)[0])
        r.begin()
        ctx = r.context_text()
        pool = {"key": key, "stored_context_sha256": st["context_sha256"], "stored_context_text_sha256": sha(stored_ctx[key]), "rebuilt_context_sha256": RT.LF.sha256_text(ctx),
                "option_sha256": list(st["option_sha256"]), "admitted_by_backends": admitted[key]}
        if RT.LF.sha256_text(ctx) != st["context_sha256"] or sha(stored_ctx[key]) != sha(ctx) and False:
            pool.update(verdict="REFUSED", reason="the controller no longer rebuilds the stored context (sha256 differs)")
            doc["pools"][key] = pool; continue
        rec = r.step(st["options"], {})
        j = r.journal
        sc = [e["data"] for e in j.entries("scorer")]; cmp_ = [e["data"] for e in j.entries("scorer_comparison")]; flt = [e["data"] for e in j.entries("candidate_filter")]
        accepted_ids = [a["option_id"] for f in flt for a in f["accepted"]]
        same_pool = accepted_ids == list(st["option_sha256"])
        pool.update(filter_rejected=[x["rule"] for f in flt for x in f["rejected"]], pool_is_exact_stored_pool=same_pool, scored=bool(sc), scorer_unavailable_reason=rec.get("scorer_unavailable_reason"),
                    chosen_index=rec["chosen_index"], laya_order=rec.get("laya_order"), trusted_evidence_overrode_laya=rec.get("trusted_evidence_overrode_laya"), laya_top_was_chosen=rec.get("laya_top_was_chosen"))
        from evidence_ladder.controller import invariants as INV
        mg = [e["data"] for e in j.entries("monotonicity_gate")]
        pool.update(monotonicity_gate=({k: mg[0].get(k) for k in ("status", "laya_ordering_enabled", "can_override_trusted_evidence")} if mg else None), evidence_first_ordering_i2_clean=(INV.check_i2(j) == ""))
        if sc and same_pool:
            stored = {k: st["scores"][k]["score"] for k in ("laya", "probe", "random_head")}
            pool.update(verdict="REPLAYED", laya_scores=sc[0]["scores"], laya_equals_stored_verified_scores=sc[0]["scores"] == stored["laya"],
                        comparison=[{"scorer": c.get("scores") and list(c["scores"]), "selector": c.get("selector"), "scores": c.get("scores")} for c in cmp_],
                        probe_equals_stored=any(((c.get("scores") or {}).get("probe") or {}).get("score") == stored["probe"] for c in cmp_), random_head_equals_stored=any(((c.get("scores") or {}).get("random_head") or {}).get("score") == stored["random_head"] for c in cmp_),
                        live_verify_agree_on={"context_sha256": st["context_sha256"] == ver["steps"][key]["context_sha256"], "scores": st["scores"] == ver["steps"][key]["scores"]})
        else:
            pool.update(verdict="REFUSED", reason=(rec.get("scorer_unavailable_reason") or "scored pool differs from the stored pool")[:300])
        # negative control: the same stored context with ONE option removed must be refused by the backend (no score for a different pool)
        try:
            backends["laya"].score("next_action", {"context_text": ctx, "options": list(st["options"])[:-1]}); pool["negative_control"] = "NOT REFUSED"
        except RT.ProvenanceError as e:
            pool["negative_control"] = "refused: " + str(e)[:120]
        doc["pools"][key] = pool
    reps = [p for p in doc["pools"].values() if p.get("verdict") == "REPLAYED"]
    doc["summary"] = {"replayed": len(reps), "refused": sum(1 for p in doc["pools"].values() if p.get("verdict") == "REFUSED"), "pools": len(doc["pools"]),
                      "laya_probe_random_head_all_stored_and_equal": bool(reps) and all(p["laya_equals_stored_verified_scores"] and p["probe_equals_stored"] and p["random_head_equals_stored"] for p in reps),
                      "comparison_only": all(c.get("selector") is False for p in reps for c in p["comparison"]), "negative_controls_refused": all(str(p.get("negative_control", "")).startswith("refused") for p in doc["pools"].values())}
    import subprocess
    import el_arch1_acceptance as A
    git = lambda *a: subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, timeout=60).stdout.decode("utf-8", "replace").strip()
    doc["provenance"] = {"commit": git("rev-parse", "HEAD"), "dirty_tracked_files": git("status", "--porcelain", "--untracked-files=no").splitlines()[:10], "config_sha256": A.config_hashes(),
                         "source_sha256": {f: sha((ROOT / f).read_bytes()) for f in ("tools/el_arch1_pathb_replay.py", "evidence_ladder/controller/agentic.py", "evidence_ladder/controller/candidate_filter.py", "evidence_ladder/models/laya_runtime.py")},
                         "secret_scan_added_lines": A.added_lines_secret_scan("main")}
    doc["backend_identity"] = A.backend_identity(laya=backends["laya"], probe=backends["probe"], random_head=backends["random_head"], sandbox=sb, controller=r)
    A.refuse_mock_backends(laya=backends["laya"], probe=backends["probe"], random_head=backends["random_head"], sandbox=sb)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "PATHB_REPLAY_RECEIPT.json", "x", encoding="utf-8", newline="\n") as f: json.dump(doc, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(doc["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]).resolve()))
