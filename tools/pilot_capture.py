"""Pilot capture: the 980 decisions of results/raw/pilot-manifest-v1.json through frozen Qwen3-4B-Thinking-2507, depths 12/18/24/30/36.
Representation per [capture_one_record] v1 (query = hidden at answer cue; option = fp32 mean over its span; pre-final-norm residual).
Resumable: one file per decision under data/pilot/capture/; existing files are skipped. Overflow/skips are logged, never truncated.
    PYTHONPATH=. .venv/Scripts/python.exe tools/pilot_capture.py
"""
import os
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import hashlib, json, time, tomllib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from capture.capture import Capturer
from capture.render import RenderOverflow, render
from compiler import state_builder

ROOT = Path(__file__).resolve().parents[1]
th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["capture_one_record"]
D = th["depths"]
P = r"D:\local model\models\qwen3-4b-thinking-2507"
rev = json.load(open(ROOT / "results/raw/qwen3-4b-thinking-2507.manifest.json"))["revision"]
man = json.load(open(ROOT / "results/raw/pilot-manifest-v1.json", encoding="utf-8"))
outdir = ROOT / "data/pilot/capture"
outdir.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(P)
want = set(man["decision_ids"])
recs = {}
for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["decision_id"] in want:
        recs[d["decision_id"]] = d
raw = {}
for l in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l)
    raw[r["trajectory_id"]] = r
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16).eval()
C = Capturer(model, D)
log = open(ROOT / "data/pilot/capture-log.jsonl", "a", encoding="utf-8")
t_start, done, skipped = time.time(), 0, []
for n, did in enumerate(man["decision_ids"]):
    f = outdir / ("%04d-%s.pt" % (n, hashlib.sha256(did.encode()).hexdigest()[:8]))
    if f.exists():
        continue
    rec = recs[did]
    tid, t = did.rsplit(":", 1)
    try:
        st = state_builder.build_state(raw[tid]["trajectory"], int(t), 32000)
        R = render(rec, st, tok, rev)
    except (RenderOverflow, state_builder.StateOverflow) as e:
        skipped.append(did)
        log.write(json.dumps({"decision_id": did, "skipped": str(e)}) + "\n"); log.flush()
        continue
    t0 = time.time()
    cap, (ids, spans) = C.packed(R)
    assert [ids[0, a:b].tolist() for a, b in spans] == R["suffix_ids"]
    torch.save({"decision_id": did, "schema": R["schema"], "prompt_hash": R["prompt_hash"], "state_hash": R["state_hash"],
                "order": R["order"], "label_index": R["label_index"], "mask": R["mask"], "meta": R["meta"],
                "depths": D, "query": torch.stack([cap[d][0] for d in D]), "options": torch.stack([cap[d][1] for d in D]),
                "tokenizer_revision": rev, "n_prefix_tokens": R["n_prefix_tokens"]}, f)
    done += 1
    log.write(json.dumps({"decision_id": did, "n": n, "prefix_tokens": R["n_prefix_tokens"], "seconds": round(time.time() - t0, 2)}) + "\n")
    log.flush()
    if done % 20 == 0:
        print(f"{n + 1}/{man['n']} captured={done} skipped={len(skipped)} elapsed={(time.time() - t_start) / 60:.1f} min", flush=True)
assert C.lm_calls == 0
print("DONE captured", done, "skipped", len(skipped), "lm_head_calls", C.lm_calls, "minutes", round((time.time() - t_start) / 60, 1))
