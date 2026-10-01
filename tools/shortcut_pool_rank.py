"""Shortcut audit (DIAGNOSTIC): on the pilot rows, how well does a no-model rule that uses only the distractor pool's own order do?
Rule R1 'most recent': among the real candidates pick the one that appears EARLIEST in compiler.candidates.distractor_pool(events, t)
(pool order = earlier actions most recent first, then views/edits of seen files, test run, think/plan/finish).
Rule R2 'in pool': pick uniformly among candidates that ARE in the pool (a true novel action is by construction not in it; reported on covered rows only,
where the true action is in the pool). Output create-only."""
import json, collections, random
from datetime import datetime, timezone
from pathlib import Path
from compiler.candidates import distractor_pool

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
c = collections.Counter()
ranks_true, ranks_dis = [], []
for did, r in recs.items():
    tid, t = did.rsplit(":", 1)
    pool = [p[0] for p in distractor_pool(raw[tid]["trajectory"], int(t))]
    texts = [x["text"] for x in r["candidates"]]
    lab = r["label_index"]
    real = [i for i, x in enumerate(r["candidates"]) if x["id"] != "NONE"]
    in_pool = [i for i in real if texts[i] in pool]
    for key in (f"{r['split']}+covered:{r['candidate_coverage']}", "all"):
        c[key + ":n"] += 1
        if in_pool:
            c[key + ":R1_hit"] += int(min(in_pool, key=lambda i: pool.index(texts[i])) == lab)
        c[key + ":true_in_pool"] += int(lab in in_pool)
        c[key + ":n_real_in_pool_mean_x"] += len(in_pool)
        c[key + ":n_real_mean_x"] += len(real)
    if r["candidate_coverage"] and texts[lab] in pool:
        ranks_true.append(pool.index(texts[lab]))
        ranks_dis += [pool.index(texts[i]) for i in in_pool if i != lab]
keys = sorted({k.rsplit(":", 1)[0] for k in c})
out = {k: dict(n=c[k + ":n"], R1_most_recent_acc=round(c[k + ":R1_hit"] / c[k + ":n"], 4), true_action_in_pool=round(c[k + ":true_in_pool"] / c[k + ":n"], 4),
               real_cands_mean=round(c[k + ":n_real_mean_x"] / c[k + ":n"], 2), real_cands_in_pool_mean=round(c[k + ":n_real_in_pool_mean_x"] / c[k + ":n"], 2)) for k in keys}
out["covered_rank_in_pool"] = dict(true_mean=round(sum(ranks_true) / len(ranks_true), 2), true_median=sorted(ranks_true)[len(ranks_true) // 2],
                                   distractor_mean=round(sum(ranks_dis) / len(ranks_dis), 2), distractor_median=sorted(ranks_dis)[len(ranks_dis) // 2])
out["reference_on_pilot_cal_covered"] = dict(pointer_head=0.6667, option_only_linear=0.775, tfidf_full=0.3083, tfidf_matched=0.3333)
p = ROOT / ("results/raw/shortcut-pool-rank-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
with open(p, "x", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)); print("->", p)
