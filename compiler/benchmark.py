"""A5 locked benchmark: the test and OOD splits of the compiled decisions with every pre-registered variant (configs/compile.toml).

Usage: uv run python -m compiler.benchmark RECORDS.jsonl SAMPLE.jsonl NAME
Writes data/benchmark/NAME.jsonl (create-only, then made read-only) and results/raw/a5-benchmark-NAME.sha256. A change is a new version.
Variants per decision: perm (all candidates including NONE shuffled), none_first / none_middle / none_last (NONE moved, others keep order),
plus3 / plus6 / plus10 (that many more distractors from the same state-only pool; absent if the pool is too small), and the state hash and
cut stage at 12k / 16k / 24k / 32k characters (the context ablation; baselines that ignore the state are invariant to it).
"""
import hashlib
import json
import os
import random
import stat
import sys
from pathlib import Path

from compiler.candidates import make_candidates
from compiler.state_builder import StateOverflow, build_state

ROOT = Path(__file__).resolve().parents[1]
NONE_TEXT = "None of these options, or the evidence is insufficient"


def texts(cands):
    return [c.text for c in cands]


def variants(base_texts, label, seed):
    """base_texts ends with NONE. Returns {name: (texts, label_index)}."""
    items = base_texts[:-1]
    true = base_texts[label]
    out = {}
    order = list(range(len(base_texts)))
    random.Random(seed ^ 0x9E3779B9).shuffle(order)
    out["perm"] = ([base_texts[i] for i in order], order.index(label))
    for name, pos in (("none_first", 0), ("none_middle", len(items) // 2), ("none_last", len(items))):
        seq = items[:pos] + [NONE_TEXT] + items[pos:]
        out[name] = (seq, seq.index(true))
    return out


def build(records_path, sample_path, name):
    rows = {}
    for l in open(sample_path, encoding="utf-8"):
        r = json.loads(l)
        rows[r["trajectory_id"]] = r
    out_path = ROOT / "data" / "benchmark" / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "x", encoding="utf-8") as f:
        for l in open(records_path, encoding="utf-8"):
            rec = json.loads(l)
            if rec["split"] not in ("test", "ood"):
                continue
            evs, t = rows[rec["provenance"]["trajectory_id"]]["trajectory"], rec["t"]
            base = make_candidates(evs, t)
            bt = texts(base["candidates"])
            assert bt == [c["text"] for c in rec["candidates"]] and base["label_index"] == rec["label_index"], "benchmark base differs from the record"
            seed = int(rec["state_hash"][:16], 16)
            v = {k: {"texts": t_, "label_index": li} for k, (t_, li) in variants(bt, rec["label_index"], seed).items()}
            for extra in (3, 6, 10):
                m = make_candidates(evs, t, extra=extra, cap=False)
                v[f"plus{extra}"] = None if m is None else {"texts": texts(m["candidates"]), "label_index": m["label_index"]}
            ctx = {}
            for b in (12000, 16000, 24000, 32000):
                try:
                    s = build_state(evs, t, b)
                    ctx[str(b)] = {"state_hash": s["state_hash"], "cut_stage": s["manifest"]["stage"]}
                except StateOverflow:
                    ctx[str(b)] = {"state_hash": None, "cut_stage": "overflow"}
            for k, x in v.items():   # every variant keeps the true action exactly once and exactly one NONE
                if x:
                    assert x["texts"].count(NONE_TEXT) == 1 and x["texts"][x["label_index"]] == bt[rec["label_index"]]
            f.write(json.dumps({"decision_id": rec["decision_id"], "split": rec["split"], "tier": rec["tier"], "candidate_coverage": rec["candidate_coverage"],
                                "texts": bt, "label_index": rec["label_index"], "variants": v, "context": ctx}, ensure_ascii=False) + "\n")
            n += 1
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    with open(ROOT / "results" / "raw" / f"a5-benchmark-{name}.sha256", "x", encoding="utf-8") as f:
        f.write(f"{h}  data/benchmark/{name}.jsonl ({n} decisions, test and OOD splits)\n")
    os.chmod(out_path, stat.S_IREAD)
    print(n, "decisions locked;", "sha256", h[:16])


if __name__ == "__main__":
    build(*sys.argv[1:4])
