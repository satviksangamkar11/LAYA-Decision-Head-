"""Isolated self-test of a Kaggle bundle: for EVERY decision in its manifests rebuild the state and render the pool exactly as the capture runner does (CPU only, tokenizer only, no model),
using only files inside the bundle. Where the RTX 3050 capture already has a file for that decision (same manifest position), also compare pool_hash, state_hash and token counts.
    cd BUNDLE && PYTHONPATH=. PY tools/bundle_selftest.py [--model DIR] [--ref data/v3/capture_ref_root]   (ref root has train/ and cal/ with the 3050 .pt files)
Prints one JSON summary; exit code 1 if any hash or token count differs from the 3050 files. The bundle is only read, except that the CAL loader appends to the bundle's own results/raw/cal-access-log.jsonl."""
import os, sys, json, hashlib, argparse
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
from pathlib import Path
import torch
from transformers import AutoTokenizer
from capture.pool import render_pool
from capture.render import RenderOverflow
from compiler import a5v2, state_builder
from compiler.firewall import load_cal_calibration_only, load_dev

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"D:\local model\models\qwen3-4b-thinking-2507")
ap.add_argument("--ref", default=r"D:\local model\data\v3\capture")
args = ap.parse_args()
ROOT = Path.cwd()
rev = json.load(open(ROOT / "results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
tok = AutoTokenizer.from_pretrained(args.model)
raw = a5v2.load_raw(ROOT / "data/raw/swerebench-sample-20261001.jsonl")
out = {}
for role in ("train", "cal"):
    man = json.load(open(ROOT / f"data/v3/manifest-{role}.json", encoding="utf-8"))["decision_ids"]
    rows = load_dev(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl") if role == "train" else load_cal_calibration_only(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl")
    recs = {r["decision_id"]: r for r in rows}
    assert set(man) <= set(recs), f"{role}: manifest ids missing from the bundle's decision rows"
    s = dict(manifest=len(man), skipped=[], compared=0, mismatches=[], state_hash_ok=0)
    for n, did in enumerate(man):
        rec = recs[did]
        tid, t = did.rsplit(":", 1)
        st = state_builder.build_state(raw[tid]["trajectory"], int(t), 32000)
        assert st["state_hash"] == rec["state_hash"], f"{role} {did}: state rebuild differs"
        s["state_hash_ok"] += 1
        try:
            R = render_pool(rec, raw[tid]["trajectory"], st, tok, rev)
        except RenderOverflow as e:
            s["skipped"].append(did)
            continue
        f = Path(args.ref) / role / ("%05d-%s.pt" % (n, hashlib.sha256(did.encode()).hexdigest()[:8]))
        if f.exists():
            d = torch.load(f)
            bad = [k for k, v in (("pool_hash", R["pool_hash"]), ("prompt_hash", R["prompt_hash"]), ("state_hash", R["state_hash"]), ("n_prefix_tokens", R["n_prefix_tokens"]), ("pool_size", R["pool_size"])) if d[k] != v]
            s["compared"] += 1
            if bad:
                s["mismatches"].append((did, bad))
        if n % 250 == 0:
            print(role, n, "/", len(man), flush=True)
    out[role] = s
print(json.dumps({r: {**v, "skipped": len(v["skipped"]), "skipped_ids": v["skipped"][:10], "mismatches": v["mismatches"][:5], "n_mismatches": len(v["mismatches"])} for r, v in out.items()}, indent=1))
sys.exit(1 if any(v["mismatches"] for v in out.values()) else 0)
