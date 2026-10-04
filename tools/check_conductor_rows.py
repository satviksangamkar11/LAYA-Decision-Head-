"""Local sanity check of the conductor data builder on the 300-row nebius sample (no GPU, no training).
Run with the laya-audit interpreter from the project root:
  laya-audit/.venv/Scripts/python.exe tools/check_conductor_rows.py
Reports label balance and how often the 1,024-token window cuts the state (the task field is the one expected to be cut)."""
import json, random, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laya-audit" / "external-laya"))
from transformers import AutoTokenizer
from laya.common import build_sequence, render_options
from tools.conductor_data import make_rows, demo
from teachers.laya_teachers import CONTINUE_Q

demo()
ROOT = Path(__file__).resolve().parents[1]
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "laya-conductor" / "tokenizer")
cfg = json.load(open(ROOT / "models" / "laya-conductor" / "rl_agent_config.json"))
q = CONTINUE_Q["should_continue"]
rng = random.Random(0)
rows = []
for line in open(ROOT / "data" / "raw" / "swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(line)
    rows += make_rows(r, rng, "train")
c = collections.Counter(); cut = 0; cut_resp = 0; cut_obs = 0; lens = []
for r in rows:
    c[r["gold"]["should_continue"]["probabilities"]["true"]] += 1
    qd = {"t": "noul", "ins": q["instructions"], "crit": {}}
    ids, markers, st = build_sequence(tok, r["state"], qd, cfg["max_len"], cfg["head_max_len"], return_truncation_stats=True)
    assert len(markers) == len(render_options({"t": "noul", "crit": {}})), markers
    lens.append(len(ids)); cut += st["truncated"]
    # does the last_response survive? it is the first field, so check its tail text is inside the kept window
    squash = lambda t: "".join(t.split())
    tail = squash(json.dumps(r["state"]["last_response"], ensure_ascii=False)[1:-1])[-30:]   # state is json.dumps-ed, so compare escaped
    if tail and tail not in squash(tok.decode(ids)): cut_resp += 1
    ot = squash(json.dumps(r["state"]["last_observation"], ensure_ascii=False)[1:-1])[-30:]
    if ot and ot not in squash(tok.decode(ids)): cut_obs += 1
print(f"{len(rows)} rows | label TRUE share {c[1.0]/len(rows):.3f} | state truncated {cut/len(rows):.1%} | last_response tail lost {cut_resp/len(rows):.1%} | last_observation tail lost {cut_obs/len(rows):.1%} | max len {max(lens)}")
