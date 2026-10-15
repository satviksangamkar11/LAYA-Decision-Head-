"""Structural validation of a combined index (no models, no grading runs). Usage: python tools/validate_combined_index_v1.py <index folder>
Checks: unique ids, source hashes, required fields, track/scorer consistency, LAYA validity per decision type, Fable leakage (VAL vs v3 by decision and trajectory),
worker folders free of grader assets, protected/public overlap (informational), mutation status. Writes VALIDATION.json (create-only)."""
import gzip, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = Path(sys.argv[1]).resolve()
rows = [json.loads(l) for l in (D / "INDEX.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
sources = json.loads((D / "SOURCES.json").read_text(encoding="utf-8"))
res, fails = {}, []


def check(name, ok, detail=None):
    res[name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)


ids = [r["item_id"] for r in rows]
check("unique_item_ids", len(ids) == len(set(ids)), len(ids))
by = {}
for r in rows:
    by.setdefault((r["track"], r["eval_role"]), []).append(r)
check("track_counts", len(by[("python", "dev_diagnostic_only")]) == 40 and len(by[("java", "dev_diagnostic_only")]) == 40
      and len(by[("fable", "primary_eval_recorded_action_agreement")]) == 46 and len(by[("fable", "exposed_reference_not_for_laya_evaluation")]) == 335,
      {str(k): len(v) for k, v in by.items()})
bad = [p for p, s in sources.items() if not (ROOT / p).is_file() or hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != s["sha256"]]
check("source_files_exist_and_hash_match", not bad, bad or len(sources))
need = ["item_id", "track", "pack", "split", "eval_role", "exposure", "checks", "oracle_type", "scorer", "laya", "source"]
check("required_fields", all(all(k in r for k in need) for r in rows), None)
check("scorers_supported", {r["scorer"]["name"] for r in rows} <= {"program_correctness", "recorded_action_agreement"}, sorted({r["scorer"]["name"] for r in rows}))
check("track_scorer_match", all((r["track"] in ("python", "java")) == (r["scorer"]["name"] == "program_correctness") for r in rows), None)
check("laya_valid_only_for_next_action", all(r["laya"]["valid_for_decision_type"] == (r["laya"]["decision_type"] == "next_action") for r in rows), None)
check("no_pooled_score_field", not any("pooled" in json.dumps(r).lower() and "never" not in json.dumps(r).lower() for r in rows), None)

# Python / Java structure
for tr, pre, n in (("python", "py1:P1-", 40), ("java", "java1:J1-", 40)):
    rs = by[(tr, "dev_diagnostic_only")]
    want = {f"{pre}{i:02d}" for i in range(1, n + 1)}
    check(f"{tr}_ids_complete", {r["item_id"] for r in rs} == want, None)
    check(f"{tr}_public4_protected40plus", all(r["checks"]["public_examples"] == 4 and r["checks"]["protected_cases"] >= 40 for r in rs),
          {"min_protected": min(r["checks"]["protected_cases"] for r in rs), "total_protected": sum(r["checks"]["protected_cases"] for r in rs)})
    check(f"{tr}_difficulty_mix", sorted(__import__("collections").Counter(r["difficulty"] for r in rs).items()) == [("easy", 10), ("hard", 10), ("medium", 20)], None)
for tr, root in (("python", ROOT / "results/raw/codex-pilot1-20261016/fixed_v2/worker"), ("java", ROOT / "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker")):
    leak = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and any(w in p.name.lower() for w in ("protected", "oracle", "reference", "grade", "selftest"))]
    check(f"{tr}_worker_folder_has_no_grader_assets", not leak, leak)
    pc = root.parent / "grader" / "protected_cases.json"
    check(f"{tr}_protected_file_outside_worker", pc.is_file() and root not in pc.parents, str(pc.relative_to(ROOT)))

# Overlap of public and protected cases (informational: duplicates add no information)
def overlap(pub_p, prot_p, idk):
    pub = json.loads(Path(pub_p).read_text(encoding="utf-8")); prot = json.loads(Path(prot_p).read_text(encoding="utf-8")); n = 0
    pk = {t[idk]: {json.dumps(c["input"], sort_keys=True) for c in t["cases"]} for t in pub}
    for t in prot:
        n += sum(json.dumps(c["input"], sort_keys=True) in pk.get(t[idk], set()) for c in t["cases"])
    return n
res["python_public_protected_input_overlap"] = {"pass": True, "detail": overlap(ROOT / "results/raw/codex-pilot1-20261016/fixed_v2/worker/public_cases.json", ROOT / "results/raw/codex-pilot1-20261016/fixed_v2/grader/protected_cases.json", "task_id")}
res["java_public_protected_input_overlap"] = {"pass": True, "detail": overlap(ROOT / "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker/public_cases.json", ROOT / "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/grader/protected_cases.json", "id")}

# Fable leakage
G = lambda p: [json.loads(x) for x in gzip.open(ROOT / p, "rt", encoding="utf-8")]
v = G("results/raw/laya-fullaction-clean-v4-20261012T130553Z/val_v1_clean/private_labels.jsonl.gz")
a = G("results/raw/laya-fullaction-set-v3-20261012T055420Z/private_labels.jsonl.gz")
vd, ad = {x["decision_id"] for x in v}, {x["decision_id"] for x in a}
vt, at = {x["trajectory_id"] for x in v}, {x["trajectory_id"] for x in a}
check("fable_val_decision_overlap_with_v3_is_zero", not (vd & ad), len(vd & ad))
check("fable_val_trajectory_overlap_with_v3_is_zero", not (vt & at), len(vt & at))
ft = set(json.loads((ROOT / "results/raw/laya-finetune-all-v1-20261012T080145Z/REPORT-20261012T080534Z.json").read_text(encoding="utf-8"))["per_pool"]["decision_id"])
check("fable_val_not_in_finetune_all", not (vd & ft), len(vd & ft))
check("fable_v3_roles_sum_335", len(a) == 335 and sum(1 for x in a) == 335, __import__("collections").Counter(x["role"] for x in a))
check("fable_val_label_role_is_evaluation_only", {x["role"] for x in v} == {"val_evaluation_only"}, None)
check("fable_index_rows_match_label_files", {r["item_id"][len("fable-val:"):] for r in rows if r["item_id"].startswith("fable-val:")} == vd
      and {r["item_id"][len("fable-v3:"):] for r in rows if r["item_id"].startswith("fable-v3:")} == ad, None)
check("fable_v3_exposure_flag_matches_finetune_list", all(("used by laya-finetune-all-v1" in r["exposure"]["adapter_training"]) == (r["item_id"][len("fable-v3:"):] in ft) for r in rows if r["item_id"].startswith("fable-v3:")),
      {"in_finetune_all": len(ft & ad), "not_in": len(ad - ft)})

# Python mutation status
py = by[("python", "dev_diagnostic_only")]
un = sorted(r["item_id"] for r in py if r["checks"]["mutation_adequacy"]["status"] == "unmeasured")
lo = sorted(r["item_id"] for r in py if r["checks"]["mutation_adequacy"]["status"] == "measured_lower_bound" and not r["checks"]["mutation_adequacy"]["meets_0.80"])
wrong = sum(r["checks"]["mutation_adequacy"]["wrong_solutions"]["caught_after"] for r in py); wrong_n = sum(r["checks"]["mutation_adequacy"]["wrong_solutions"]["hand_written_wrong_solutions"] for r in py)
wrong_b = sum(r["checks"]["mutation_adequacy"]["wrong_solutions"]["caught_with_original_cases"] for r in py)
res["python_mutation_status"] = {"pass": True, "detail": {"unmeasured": un, "below_0.80_raw": lo, "wrong_solutions_caught": f"{wrong}/{wrong_n} (original cases: {wrong_b}/{wrong_n})",
        "note": "informational for the dataset; the S6 adequacy gate (>= 0.80 measured) is NOT yet met for these tasks"}}
res["java_mutation_status"] = {"pass": True, "detail": "unmeasured for all 40 (kit has only trivial mutant checks)"}

out = {"index": str(D.name), "index_sha256": hashlib.sha256((D / "INDEX.jsonl").read_bytes()).hexdigest(), "checks": res, "failed": fails,
       "result": "PASS (structure)" if not fails else "FAIL", "scope": "structure, hashes and leakage only; graders were not run, models were not run"}
(D / "VALIDATION.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps({"result": out["result"], "failed": fails, "n_checks": len(res)}, indent=1))
for k, v_ in res.items():
    print(("ok  " if v_["pass"] else "FAIL"), k, "->", str(v_["detail"])[:140])
