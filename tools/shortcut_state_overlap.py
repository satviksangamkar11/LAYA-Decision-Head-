"""Shortcut audit (DIAGNOSTIC): does a trivial STATE-AWARE lexical rule already explain the head's accuracy on the pilot rows?
Rule: score_i = share of candidate i's word tokens that also occur in the rendered state text (optionally only the last N events' text).
No training. Rebuilds each state with the project's own builder (events before t only). Output create-only."""
import json, re, collections
from datetime import datetime, timezone
from pathlib import Path
from compiler.state_builder import build_state

ROOT = Path(__file__).resolve().parents[1]
man = json.load(open(ROOT / "results/raw/pilot-manifest-v1.json", encoding="utf-8"))
want = set(man["decision_ids"])
recs = {}
for l in open(ROOT / "data/decisions/action-swerebench-sample-v2.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["decision_id"] in want:
        recs[r["decision_id"]] = r
raw = {}
for l in open(ROOT / "data/raw/swerebench-sample-20261001.jsonl", encoding="utf-8"):
    r = json.loads(l)
    raw[r["trajectory_id"]] = r
tok = lambda s: set(w for w in re.findall(r"[A-Za-z0-9_]+", s.lower()) if len(w) >= 3)
NONE = "None of these options, or the evidence is insufficient"
stats = collections.defaultdict(lambda: collections.Counter())
for did, r in recs.items():
    tid, t = did.rsplit(":", 1)
    st = build_state(raw[tid]["trajectory"], int(t), 32000)["text"]
    S_all, S_tail = tok(st), tok(st[-4000:])
    for name, S, mode in (("overlap_full_state", S_all, "max"), ("overlap_last_4000_chars", S_tail, "max"), ("NOVELTY_least_overlap_full_state", S_all, "min")):
        sc = []
        for c in r["candidates"]:
            tk = tok(c["text"])
            sc.append(-1.0 if c["text"] == NONE else (len(tk & S) / len(tk) if tk else 0.0))
        best = max(range(len(sc)), key=lambda i: (sc[i], -i)) if mode == "max" else min((i for i in range(len(sc)) if sc[i] >= 0), key=lambda i: (sc[i], i))
        for key in ("all", f"split:{r['split']}", f"covered:{r['candidate_coverage']}", f"{r['split']}+covered:{r['candidate_coverage']}"):
            stats[name][key + ":n"] += 1
            stats[name][key + ":hit"] += int(best == r["label_index"])
out = {n: {k[:-4]: round(c[k] / c[k[:-4] + ":n"], 4) for k in c if k.endswith(":hit")} | {k[:-2] + "_n": c[k] for k in c if k.endswith(":n")} for n, c in stats.items()}
out["reference_on_pilot_cal"] = dict(pointer_head=dict(all=0.6158, covered=0.6667), option_only_linear=dict(all=0.8211, covered=0.775),
                                     tfidf_full_train=dict(all=0.3842, covered=0.3083), tfidf_matched=dict(all=0.2895, covered=0.3333))
p = ROOT / ("results/raw/shortcut-state-overlap-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)); print("->", p)
