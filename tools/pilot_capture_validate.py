"""End-of-capture validation for the 980-decision pilot. Read-only on the captures; writes results/raw/pilot-capture-validation-*.json (create-only).
Checks: file count vs manifest, skips, timing, all-depth consistency (shapes, finiteness), label/mask sanity, loader self-check,
and bit-identical recapture of a few decisions through the same Capturer path.
    PYTHONPATH=. .venv/Scripts/python.exe tools/pilot_capture_validate.py
"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import hashlib, json, statistics, tomllib
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture.capture import Capturer
from capture.render import render
from compiler import state_builder
from decision_head import pilot_data

ROOT = Path(__file__).resolve().parents[1]
th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["capture_one_record"]
D = th["depths"]
man = json.load(open(ROOT / "results/raw/pilot-manifest-v1.json", encoding="utf-8"))
out = ROOT / ("results/raw/pilot-capture-validation-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
res = {}

files = sorted((ROOT / "data/pilot/capture").glob("*.pt"))
rows = [torch.load(f, map_location="cpu") for f in files]
ids = [r["decision_id"] for r in rows]
res["files"] = len(files)
res["manifest_n"] = man["n"]
res["unique_ids"] = len(set(ids))
res["ids_equal_manifest"] = set(ids) == set(man["decision_ids"])
res["order_equals_manifest_order"] = ids == man["decision_ids"]

log = [json.loads(l) for l in open(ROOT / "data/pilot/capture-log.jsonl", encoding="utf-8")]
skips = [x for x in log if "skipped" in x]
done = [x for x in log if "seconds" in x]
t = [x["seconds"] for x in done]
tok = [x["prefix_tokens"] for x in done]
res["skips"] = len(skips)
res["log_records"] = len(done)
res["seconds"] = dict(mean=round(statistics.mean(t), 2), median=round(statistics.median(t), 2), p95=round(sorted(t)[int(0.95 * len(t))], 2),
                      max=max(t), total_hours=round(sum(t) / 3600, 2))
res["prefix_tokens"] = dict(mean=round(statistics.mean(tok)), median=statistics.median(tok), max=max(tok))
res["throughput_tok_s_mean"] = round(sum(tok) / sum(t), 1)

bad = dict(depth_list=0, shape=0, nonfinite=0, label=0, mask=0, schema=0, tokenizer_rev=0)
rev = json.load(open(ROOT / "results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
Hs = set()
qn, on = [], []
for r in rows:
    n = len(r["order"])
    Hs.add(r["query"].shape[-1])
    bad["schema"] += r["schema"] != "capture-record-v1"
    bad["tokenizer_rev"] += r["tokenizer_revision"] != rev
    bad["depth_list"] += r["depths"] != D
    bad["shape"] += not (r["query"].shape == (len(D), r["query"].shape[-1]) and r["options"].shape == (len(D), n, r["query"].shape[-1]))
    bad["nonfinite"] += not (torch.isfinite(r["query"]).all() and torch.isfinite(r["options"]).all())
    bad["label"] += not (0 <= r["label_index"] < n)
    bad["mask"] += r["mask"] != [1] * n
    qn.append(r["query"].norm(dim=-1).mean(0).item() if False else r["query"].norm(dim=-1).tolist())
    on.append(r["options"].norm(dim=-1).mean((1)).tolist())
res["bad_counts"] = bad
res["hidden_sizes"] = sorted(Hs)
res["query_norm_by_depth_mean"] = [round(statistics.mean(x[i] for x in qn), 1) for i in range(len(D))]
res["option_norm_by_depth_mean"] = [round(statistics.mean(x[i] for x in on), 1) for i in range(len(D))]

d = pilot_data.load(36)
res["loader"] = dict(rows=len(d["y"]), hq=list(d["hq"].shape), ho=list(d["ho"].shape), splits={s: d["split"].count(s) for s in set(d["split"])})

tok_ = AutoTokenizer.from_pretrained(r"D:\local model\models\qwen3-4b-thinking-2507")
recs = {}
for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    x = json.loads(l)
    if x["decision_id"] in set(ids[:1] + ids[490:491] + ids[-1:]):
        recs[x["decision_id"]] = x
raw = {}
for l in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    x = json.loads(l)
    raw[x["trajectory_id"]] = x
model = AutoModelForCausalLM.from_pretrained(r"D:\local model\models\qwen3-4b-thinking-2507", dtype=torch.bfloat16).eval()
C = Capturer(model, D)
rc = []
for did in [ids[0], ids[490], ids[-1]]:
    rec = recs[did]
    tid, tt = did.rsplit(":", 1)
    R = render(rec, state_builder.build_state(raw[tid]["trajectory"], int(tt), 32000), tok_, rev)
    cap, _ = C.packed(R)
    saved = rows[ids.index(did)]
    q = torch.stack([cap[x][0] for x in D])
    o = torch.stack([cap[x][1] for x in D])
    rc.append(dict(decision_id=did, prompt_hash_equal=R["prompt_hash"] == saved["prompt_hash"], max_abs_query=(q - saved["query"]).abs().max().item(),
                   max_abs_options=(o - saved["options"]).abs().max().item()))
res["recapture"] = rc
res["lm_head_calls"] = C.lm_calls
res["gates"] = dict(count=res["files"] == man["n"] == res["unique_ids"], ids=res["ids_equal_manifest"], skips=res["skips"] == 0,
                    consistency=all(v == 0 for v in bad.values()),
                    recapture_bit_identical=all(x["max_abs_query"] == 0 and x["max_abs_options"] == 0 and x["prompt_hash_equal"] for x in rc),
                    no_lm_head=C.lm_calls == 0)
res["verdict"] = "PASS" if all(res["gates"].values()) else "FAIL"
with open(out, "x", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
print("->", out)
