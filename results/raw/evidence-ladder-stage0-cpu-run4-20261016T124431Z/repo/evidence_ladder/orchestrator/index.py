"""Frozen combined index: verified loading, TaskSpec construction (prompt + public tests only), Fable track object."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from evidence_ladder.common.interfaces import TaskSpec
from evidence_ladder.orchestrator.base import REPO, load_toml, sha_bytes_lf_tolerant

INDEX_DIR = REPO / "results" / "raw" / "combined-index-v1-20261015T221945Z"
EXPECTED_COUNTS = {"python": 40, "java": 40, "fable_val": 46, "fable_exposed": 335}
EXPOSED_FLAG = "exposed_not_for_laya_evaluation"
FABLE_RUN_STATUS = "NOT IMPLEMENTED for stage 1"
WAIVER_NOTE = ("The development-dataset waiver (D1) covers only the post-cutoff dataset rule. It does NOT waive oracle adequacy "
               "(toml [frozen.oracle_adequacy]: mutation kill rate >= 0.80 measured per task).")
_MUT_KEEP = ("status", "kill_rate", "kill_rate_excluding_likely_equivalent", "killed", "mutants", "mutants_tested", "survivors",
             "likely_equivalent_survivors", "meets_0.80", "reason", "sample")


class IndexVerificationError(Exception):
    pass


@dataclass(frozen=True)
class FrozenIndex:
    rows: tuple
    sha256: str
    directory: str


def load_index(index_dir=None) -> FrozenIndex:
    d = Path(index_dir) if index_dir else INDEX_DIR
    raw = (d / "INDEX.jsonl").read_bytes()
    want = {json.loads((d / n).read_text(encoding="utf-8"))["index_sha256"] for n in ("SUMMARY.json", "VALIDATION.json")}
    if len(want) != 1:
        raise IndexVerificationError("SUMMARY.json and VALIDATION.json disagree on index_sha256")
    got = sha_bytes_lf_tolerant(raw)
    if not (got & want):
        raise IndexVerificationError(f"INDEX.jsonl sha256 {sorted(got)} does not equal the frozen {sorted(want)}")
    rows = tuple(json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip())
    c = counts(rows)
    if c != EXPECTED_COUNTS:
        raise IndexVerificationError(f"row counts {c} != expected {EXPECTED_COUNTS}")
    if len({r["item_id"] for r in rows}) != len(rows):
        raise IndexVerificationError("duplicate item_id")
    return FrozenIndex(rows, sorted(want)[0], str(d))


def counts(rows) -> Dict[str, int]:
    out = {"python": 0, "java": 0, "fable_val": 0, "fable_exposed": 0}
    for r in rows:
        if r["track"] in ("python", "java"):
            out[r["track"]] += 1
        elif r["eval_role"] == "primary_eval_recorded_action_agreement":
            out["fable_val"] += 1
        elif r["eval_role"] == "exposed_reference_not_for_laya_evaluation":
            out["fable_exposed"] += 1
    return out


def mutation_state(row: dict, threshold: Optional[float] = None) -> dict:
    """Oracle-adequacy state of one task, from the index. below_threshold: raw kill rate < threshold. unmeasured: no score.
    Every non-measured_ok task is flagged; the D1 waiver never clears a flag."""
    if threshold is None:
        threshold = float(load_toml()["frozen"]["oracle_adequacy"]["mutation_kill_rate_min"])
    m = row["checks"]["mutation_adequacy"]
    kr = m.get("kill_rate")
    if m.get("status") == "unmeasured" or kr is None:
        state = "unmeasured"
    elif kr < threshold:
        state = "below_threshold"
    else:
        state = "measured_ok"
    out = {k: m[k] for k in _MUT_KEEP if k in m}
    out.update(state=state, flagged=state != "measured_ok", threshold=threshold, waiver_applies=False, waiver_note=WAIVER_NOTE)
    return out


def _worker_rows(row: dict) -> Dict[str, dict]:
    p = REPO / row["source"]["files"][0].replace("\\", "/")
    if "worker" not in p.parts or "grader" in p.parts:   # only worker files (prompt + public tests) are ever read here
        raise IndexVerificationError(f"refusing to read non-worker file {p}")
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            out[o.get("task_id") or o["id"]] = o
    return out


def build_tasks(index: FrozenIndex) -> Dict[str, List[TaskSpec]]:
    """{'python': [40 TaskSpec], 'java': [40 TaskSpec]}. Only prompt, signature and public tests are copied from worker files."""
    thr = float(load_toml()["frozen"]["oracle_adequacy"]["mutation_kill_rate_min"])
    cache: Dict[str, Dict[str, dict]] = {}
    out: Dict[str, List[TaskSpec]] = {"python": [], "java": []}
    for r in index.rows:
        if r["track"] not in out:
            continue
        key = r["source"]["files"][0]
        wr = cache.setdefault(key, _worker_rows(r))
        tid = r["item_id"].split(":", 1)[1]
        w = wr[tid]
        prompt = w["prompt"]
        if r["track"] == "python":
            prompt += "\n\nWrite this function: " + w["function_signature"]
        else:
            prompt += f"\n\nWrite class {w['submission_class']} with method {w['submission_method']}."
        pub = [{"input": t["input"], "expected": t["expected"]} for t in w["public_tests"]]
        out[r["track"]].append(TaskSpec(task_id=tid, index_item_id=r["item_id"], track=r["track"], language=r["track"], prompt=prompt,
                                        public_tests=tuple(pub), difficulty=w.get("difficulty", ""), mutation_status=mutation_state(r, thr)))
    return out


@dataclass(frozen=True)
class FableTrack:
    """Separate track object. Metric is recorded-action agreement, never program correctness. Stage-1 running is not implemented."""
    val_items: tuple
    exposed_items: tuple
    run_status: str = FABLE_RUN_STATUS
    run_supported: bool = False

    def run(self, *a, **k):
        raise NotImplementedError("Fable track " + FABLE_RUN_STATUS)


def build_fable_track(index: FrozenIndex) -> FableTrack:
    def meta(r, flag=None):
        d = {"item_id": r["item_id"], "eval_role": r["eval_role"], "trajectory_id": r.get("trajectory_id"), "scorer": r["scorer"]["name"]}
        if flag:
            d["flag"] = flag
        return d
    val = tuple(meta(r) for r in index.rows if r["eval_role"] == "primary_eval_recorded_action_agreement")
    exp = tuple(meta(r, EXPOSED_FLAG) for r in index.rows if r["eval_role"] == "exposed_reference_not_for_laya_evaluation")
    return FableTrack(val, exp)
