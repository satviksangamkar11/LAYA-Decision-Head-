"""Score a Laya checkpoint on the LocalLLaMA/typed-decisions test split. Offline, sha256-pinned, create-only output.

Usage: python tools/eval_laya_typed_decisions.py MODEL_DIR ROWS_JSON [--sha256 HEX] [--device cuda]
  ROWS_JSON is a JSON list of test rows (id, workflow, state, questions, gold as JSON strings), e.g. fetched
  from the datasets server. Accuracy is the benchmark's: the highest-probability label against the gold label,
  ordinal questions included. Writes results/raw/laya-eval-<model>-<time>.jsonl (one row per decision) and
  prints a summary next to the numbers the model card publishes.
"""
import argparse, json, os, time
from collections import defaultdict

os.environ.setdefault("USE_TF", "0")             # a stray TensorFlow import can deadlock model construction
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # a load must never download
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import laya

PUBLISHED = {"all": 0.766, "noul": 0.857, "choice": 0.733, "score": 0.723,           # laya-typed-decisions card, author-reported
             "invoice_processing": 0.804, "security_incidents": 0.766, "customer_service": 0.764,
             "agent_trace_observability": 0.730}


def ece(conf, ok, bins=15):
    err, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi]
        if idx:
            err += len(idx) / n * abs(sum(conf[i] for i in idx) / len(idx) - sum(ok[i] for i in idx) / len(idx))
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("rows")
    ap.add_argument("--sha256"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rows = json.load(open(a.rows, encoding="utf-8-sig"))
    agent = laya.load(a.model, device=a.device, expected_sha256={"model.safetensors": a.sha256} if a.sha256 else None)
    name = os.path.basename(os.path.normpath(a.model))
    path = f"results/raw/laya-eval-{name}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    hit, conf, lat = defaultdict(list), [], []
    with open(path, "x", encoding="utf-8") as f:
        for r in rows:
            state, questions, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
            t0 = time.perf_counter()
            res = agent.predict(state, questions)
            lat.append(time.perf_counter() - t0)
            for qid, q in questions.items():
                ans, typ = res["answers"][qid], q["type"]
                if typ == "choice":
                    pred, p = ans["choice"], max(ans["probabilities"].values())
                elif typ == "noul":
                    pred, p = ("true" if ans["noul"] >= 0.5 else "false"), max(ans["noul"], 1 - ans["noul"])
                else:
                    pred = max(ans["probabilities"], key=ans["probabilities"].get)
                    p = max(ans["probabilities"].values())
                ok = int(str(pred) == str(gold[qid]["label"]))
                f.write(json.dumps({"id": r["id"], "workflow": r["workflow"], "question": qid, "type": typ, "gold": gold[qid]["label"],
                                    "pred": pred, "p_top": round(p, 4), "correct": ok}) + "\n")
                for k in ("all", typ, r["workflow"]):
                    hit[k].append(ok)
                conf.append(p)
    flat = [x for k in ("all",) for x in hit[k]]
    print(f"rows written: {path}   decisions: {len(flat)}   cases: {len(rows)}")
    for k in ("all", "noul", "choice", "score", "invoice_processing", "security_incidents", "customer_service", "agent_trace_observability"):
        acc = sum(hit[k]) / len(hit[k])
        print(f"  {k:28s} n={len(hit[k]):5d}  accuracy {acc:.3f}   published {PUBLISHED[k]:.3f}   diff {acc - PUBLISHED[k]:+.3f}")
    print(f"  ECE (15 bins, top-1 probability, shipped temperatures): {ece(conf, flat):.3f}   published 0.213")
    print(f"  mean latency per case ({a.device}): {1000 * sum(lat) / len(lat):.1f} ms")


if __name__ == "__main__":
    main()
