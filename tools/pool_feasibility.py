"""Can the existing distractor_pool support a feature-balanced A5-v2? Rule: configs/thresholds.toml [a5_v2_pool_feasibility] v1. TRAIN covered rows only."""
import collections, json, tomllib
from datetime import datetime, timezone
from pathlib import Path
from compiler.candidates import distractor_pool, first_call, render

ROOT = Path(__file__).resolve().parents[1]
th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))["a5_v2_pool_feasibility"]
raw = {}
for l in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l); raw[r["trajectory_id"]] = r
fb = lambda n: min(n, 3)
agg = collections.defaultdict(lambda: collections.Counter())
sizes, per_type = [], collections.Counter()
for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["split"] != "train" or not r["candidate_coverage"]:
        continue
    tid, t = r["decision_id"].rsplit(":", 1); t = int(t); ev = raw[tid]["trajectory"]
    full = distractor_pool(ev, t)
    texts = [p[0] for p in full]
    hist = collections.Counter(render(*c)[0] for e in ev[1:t] if e["role"] == "assistant" and (c := first_call(e)))
    ttext = r["candidates"][r["label_index"]]["text"]
    if ttext not in texts:
        continue
    tr_i = texts.index(ttext); tkind = full[tr_i][1]; tf = hist.get(ttext, 0)
    typ = "repeat(freq>=1)" if tf >= 1 else "first_time(freq=0)"
    sizes.append(len(full)); per_type[typ] += 1
    m = {"L1": 0, "L2": 0, "L3": 0}
    for i, (x, kind, _, _) in enumerate(full):
        if i == tr_i:
            continue
        a = fb(hist.get(x, 0)) == fb(tf)
        b = a and kind == tkind
        c = b and abs(i - tr_i) <= th["rank_window"]
        m["L1"] += a; m["L2"] += b; m["L3"] += c
    for lv, n in m.items():
        agg[typ][lv + ":feasible"] += int(n >= th["m_min"]); agg[typ][lv + ":zero"] += int(n == 0)
    agg[typ]["n"] += 1
res = {"rule": th, "train_covered_rows": sum(per_type.values()), "pool_size_mean": round(sum(sizes) / len(sizes), 1), "pool_size_median": sorted(sizes)[len(sizes) // 2], "by_type": {}}
for typ, c in agg.items():
    res["by_type"][typ] = dict(n=c["n"], **{lv: dict(feasible_share=round(c[lv + ":feasible"] / c["n"], 4), no_match_share=round(c[lv + ":zero"] / c["n"], 4)) for lv in ("L1", "L2", "L3")})
strict = "L3" if all(v["L3"]["feasible_share"] >= th["min_feasible_share"] for v in res["by_type"].values()) else "L2"
res["strictest_level_meeting_bar"] = strict if all(v[strict]["feasible_share"] >= th["min_feasible_share"] for v in res["by_type"].values()) else "NONE"
res["verdict"] = "EXISTING POOL SUPPORTS v2" if res["strictest_level_meeting_bar"] != "NONE" else "POOL MUST BE EXPANDED BEFORE ANY CAPTURE"
p = ROOT / ("results/raw/pool-feasibility-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f: json.dump(res, f, indent=1)
print(json.dumps(res, indent=1)); print("->", p)
