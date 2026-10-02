"""Checkpoint audit of a finished (or partial) v3 capture role. Structural only: counts, ids, finiteness, ranges, gate results from the capture log, embedding coverage,
artifact hashes. It never trains or scores anything and never computes accuracy. Output create-only: results/raw/capture-<role>-checkpoint-<time>.json.
    PYTHONPATH=. .venv/Scripts/python.exe tools/capture_checkpoint_audit.py ROLE [--partial]"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import collections, hashlib, json, sys, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
role = sys.argv[1]
partial = "--partial" in sys.argv
G = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["capture_v3_gates"]
spec = tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))["capture_v3"]
man = json.load(open(ROOT / f"data/v3/manifest-{role}.json", encoding="utf-8"))
ids = man["decision_ids"]
files = sorted((ROOT / f"data/v3/capture/{role}").glob("*.pt"))
problems, hashes, caps = [], {}, {}
for f in files:
    b = f.read_bytes()
    hashes[f.name] = hashlib.sha256(b).hexdigest()
    c = torch.load(f, map_location="cpu")
    did = c["decision_id"]
    if did in caps:
        problems.append(f"duplicate decision id {did}")
    caps[did] = c
    if c["schema"] != "capture-record-v2" or c["depths"] != spec["depths"] or c["role"] != role:
        problems.append(f"{f.name}: schema/depths/role")
    for k in ("query", "cands"):
        t = c[k].float()
        if not torch.isfinite(t).all():
            problems.append(f"{f.name}: non-finite {k}")
        elif t.abs().max().item() >= 60000:
            problems.append(f"{f.name}: {k} outside fp16 storage range")
    if c["cands"].shape[0] != len(c["texts"]) or c["query"].shape[0] != len(spec["depths"]):
        problems.append(f"{f.name}: shapes")
log_all = [json.loads(l) for l in open(ROOT / f"data/v3/capture-log-{role}.jsonl", encoding="utf-8")]
# entries without gate_detail were written by the smoke run made BEFORE the gate amendment (same log file name then); they are reported, not used
log = [l for l in log_all if "gate_detail" in l or "skipped" in l]
stale_pre_amendment_entries = len(log_all) - len(log)
skipped = [l["decision_id"] for l in log if "skipped" in l]
done = [l for l in log if "seconds" in l]
dup_log = [k for k, v in collections.Counter(l["decision_id"] for l in done).items() if v > 1]
missing = [d for d in ids if d not in caps and d not in skipped]
extra = [d for d in caps if d not in set(ids)]
if missing and not partial:
    problems.append(f"{len(missing)} manifest decisions have neither a capture nor a recorded skip")
if extra:
    problems.append(f"{len(extra)} captures are not in the manifest")
gated = [l for l in done if l["gates"]]
swap = [l["gate_detail"]["content_swap"] for l in gated if "content_swap" in l["gate_detail"]]
shape = [l["gate_detail"]["shape_change"]["min_cosine"] for l in gated if "shape_change" in l["gate_detail"]]
if any(s["max_abs_diff"] > G["content_swap_max_abs_diff"] for s in swap):
    problems.append("a Gate A reading above the registered bound is in the log")
if shape and min(shape) < G["shape_change_cos_min"]:
    problems.append("a Gate B reading below the registered bound is in the log")
exp_gated = [n for n in range(len(ids)) if n < 3 or n % 50 == 0]
gated_done = {l["n"] for l in gated}
te_dir = ROOT / "data/v3/textemb"
te = {}
for f in sorted(te_dir.glob("shard-*.pt")):
    te.update(torch.load(f, map_location="cpu"))
no_emb = sorted({t for c in caps.values() for t in c["texts"] if t not in te})
if no_emb:
    problems.append(f"{len(no_emb)} captured candidate texts have no state-free embedding")
cal_log = [json.loads(l) for l in open(ROOT / "results/raw/cal-access-log.jsonl", encoding="utf-8")]
out = dict(role=role, created=datetime.now(timezone.utc).isoformat(), partial=partial, manifest_n=len(ids), log_entries_pre_amendment_smoke_ignored=stale_pre_amendment_entries, captured=len(caps), skipped_render_overflow=len(skipped),
           missing=len(missing), extra=len(extra), duplicate_ids_in_files=0 if len(caps) == len(files) else len(files) - len(caps), duplicate_ids_in_log=len(dup_log),
           gated_decisions=len(gated), expected_gated_positions_captured=sum(1 for n in exp_gated if n in gated_done),
           expected_gated_positions_total=len(exp_gated), gate_A_targets_tested=sum(s["targets"] for s in swap), gate_A_max_abs_diff=max((s["max_abs_diff"] for s in swap), default=None),
           gate_B_min_cosine=min(shape, default=None), plain_and_repeat_gate_runs=sum(1 for l in done if "plain" in l["gates"]),
           seconds_per_decision=dict(ungated_mean=round(sum(l["seconds"] for l in done if not l["gates"]) / max(1, sum(1 for l in done if not l["gates"])), 2),
                                     gated_mean=round(sum(l["seconds"] for l in gated) / max(1, len(gated)), 2), total_hours=round(sum(l["seconds"] for l in done) / 3600, 2)),
           texts_without_embedding=len(no_emb), cal_access_log_entries=len(cal_log), cal_marker_a5v2_evaluation_used=(ROOT / "results/raw/a5v2-cal-evaluation-used.json").exists(),
           accuracy_or_selection_computed=False, artifact_hash_of_hashes=hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest(),
           problems=problems, passed=not problems)
p = ROOT / f"results/raw/capture-{role}-checkpoint-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
with open(p, "x", encoding="utf-8") as f:
    json.dump(dict(summary=out, file_sha256=hashes), f, indent=1)
print(json.dumps(out, indent=1)); print("->", p)
