"""Compile ACTION decision records from a trajectory sample (A4.7). Stage 1 of 2: no teacher is called here (that needs laya-audit).

Usage: uv run python -m compiler.run SAMPLE.jsonl OUT_NAME
Writes data/decisions/OUT_NAME.jsonl, results/raw/split-manifest-OUT_NAME.json and results/raw/compile-report-OUT_NAME.json. All create-only.
A record stores references and hashes, not the 32k-character state: the builder regenerates the state deterministically from (trajectory, t).
Provenance fields (repo, instance_id, trajectory_id) and label-side fields (tier, resolved) are never model input.
"""
import collections
import hashlib
import json
import sys
import tomllib
from pathlib import Path

from compiler import split, state_builder
from compiler.candidates import FILE_RE, classify, first_call, make_candidates, norm_path
from compiler.state_builder import StateOverflow
from compiler.tiers import patch_files, tier_for

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(sample, name):
    cfg = tomllib.loads((ROOT / "configs" / "compile.toml").read_text(encoding="utf-8"))
    budget = cfg["state"]["budget_chars"]
    rows = [json.loads(l) for l in open(sample, encoding="utf-8")]
    sm = split.manifest(rows)
    out_path, sm_path, rep_path = (ROOT / "data" / "decisions" / f"{name}.jsonl", ROOT / "results" / "raw" / f"split-manifest-{name}.json",
                                   ROOT / "results" / "raw" / f"compile-report-{name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cnt = collections.Counter()
    with open(out_path, "x", encoding="utf-8") as f:
        for r in rows:
            evs, files = r["trajectory"], patch_files(r["model_patch"])
            sp = sm["repos"][r["repo"]]
            for t in range(2, len(evs)):
                if evs[t]["role"] != "assistant" or not evs[t]["tool_calls"]:
                    continue
                try:
                    m = make_candidates(evs, t, budget)
                except StateOverflow:
                    cnt["skipped_overflow"] += 1
                    continue
                if m is None:
                    cnt["skipped_pool_too_small"] += 1
                    continue
                tier, why = tier_for(evs, t, bool(r["resolved"]), files)
                tool, args = first_call(evs[t])
                kind = classify(tool, args)
                path = norm_path(args.get("path")) if tool == "str_replace_editor" else ""
                blob = " ".join((e["content"] or "") + " ".join(json.dumps(c.get("function", {}).get("arguments", "")) for c in (e["tool_calls"] or [])) for e in evs[1:t])
                unseen = bool(path) and path not in blob
                rec = {"decision_id": f"{r['trajectory_id']}:{t}", "decision_type": "ACTION", "split": sp, "t": t, "next_action_kind": kind,
                       "state_hash": m["state_hash"], "manifest_hash": m["manifest_hash"], "cut_stage": m["cut_stage"],
                       "candidates": [{"id": c.candidate_id, "text": c.text, "action_type": c.action_type, "tool": c.tool, "safety_class": c.safety_class}
                                      for c in m["candidates"]],
                       "label_index": m["label_index"], "candidate_coverage": m["candidate_coverage"], "tier": tier, "tier_reason": why, "true_target_unseen_in_state": unseen,
                       "provenance": {"repo": r["repo"], "instance_id": r["instance_id"], "trajectory_id": r["trajectory_id"], "sample_offset": r.get("_offset")},
                       "label_side": {"resolved": bool(r["resolved"])}, "builder_version": state_builder.BUILDER_VERSION, "rules": "A4_LEAKAGE_RULES v1 + v1.1"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt["records"] += 1
                cnt[f"split:{sp}"] += 1
                cnt[f"tier:{tier}"] += 1
                cnt[f"stage:{m['cut_stage']}"] += 1
                cnt[f"kind:{kind}"] += 1
                cnt["covered_by_state_only_generator"] += m["candidate_coverage"]
                cnt["target_unseen"] += unseen
                cnt["target_checked"] += bool(path)
    with open(sm_path, "x", encoding="utf-8") as f:
        json.dump(sm, f, indent=1)
    report = {"sample": str(sample), "sample_sha256": sha(sample), "out": str(out_path), "out_sha256": sha(out_path), "budget_chars": budget,
              "split_manifest_sha256": sm["sha256"], "rules_sha256": {"v1": sha(ROOT / "A4_LEAKAGE_RULES.md"), "v1_1": sha(ROOT / "A4_LEAKAGE_RULES_v1_1.md")},
              "counts": dict(sorted(cnt.items())), "rows": len(rows), "repos": len(sm["repos"])}
    with open(rep_path, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["counts"], indent=1))
    print("out", out_path.name, "| split manifest sha256", sm["sha256"][:16])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
