"""A5 baselines on the locked benchmark: random, position, longest-text, TF-IDF (candidate text only, no state).

Usage: uv run python -m compiler.baselines RECORDS.jsonl BENCHMARK.jsonl NAME
TF-IDF is a listwise linear model on hashed word 1 and 2-grams, fitted on the TRAIN split only; the CAL split picks the epoch. It sees candidate
text and nothing else, so whatever it scores above chance is shortcut signal in the candidates, not decision quality. Accuracy means
agreement with the trajectory's action (never "correctness"). Output: results/raw/a5-baselines-NAME.json (create-only).
"""
import os

os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")   # AVX-512 torch CPU kernels crash on this machine

import collections
import json
import math
import re
import sys
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NONE_TEXT = "None of these options, or the evidence is insufficient"
D = 1 << 18
AXES = ("normal", "perm", "none_first", "none_middle", "none_last", "plus3", "plus6", "plus10")


def feats(text):
    toks = re.findall(r"[a-z0-9_]+", text.lower())
    return [zlib.crc32(g.encode()) % D for g in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]]


class TfIdf:
    def __init__(self, decisions):
        df, n = collections.Counter(), 0
        for texts, _ in decisions:
            for t in texts:
                df.update(set(feats(t)))
                n += 1
        self.idf = collections.defaultdict(lambda: math.log((n + 1) / 1) + 1, {k: math.log((n + 1) / (v + 1)) + 1 for k, v in df.items()})
        self.bag = torch.nn.EmbeddingBag(D, 1, mode="sum")
        torch.nn.init.zeros_(self.bag.weight)

    def vec(self, text):
        tf = collections.Counter(feats(text))
        w = {k: (1 + math.log(v)) * self.idf[k] for k, v in tf.items()}
        z = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return list(w), [x / z for x in w.values()]

    def pack(self, texts_list):
        idx, wt, off = [], [], []
        for t in texts_list:
            off.append(len(idx))
            i, w = self.vec(t)
            idx += i
            wt += w
        return torch.tensor(idx), torch.tensor(off), torch.tensor(wt)

    def scores(self, texts):
        i, o, w = self.pack(texts)
        return self.bag(i, o, per_sample_weights=w).squeeze(-1)

    def fit(self, train, val, epochs=40):
        flat = [t for texts, _ in train for t in texts]
        i, o, w = self.pack(flat)
        starts, k = [], 0
        for texts, _ in train:
            starts.append(k)
            k += len(texts)
        maxn = max(len(t) for t, _ in train)
        P = torch.full((len(train), maxn), -1, dtype=torch.long)
        for d, (texts, _) in enumerate(train):
            P[d, :len(texts)] = torch.arange(starts[d], starts[d] + len(texts))
        y = torch.tensor([l for _, l in train])
        opt = torch.optim.Adam(self.bag.parameters(), lr=0.1)
        best, best_state = -1, None
        for ep in range(epochs):
            s = self.bag(i, o, per_sample_weights=w).squeeze(-1)
            sp = torch.where(P >= 0, s[P.clamp(min=0)], torch.full((), -1e9))
            loss = torch.nn.functional.cross_entropy(sp, y) + 1e-6 * self.bag.weight.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                acc = sum(int(self.scores(t).argmax()) == l for t, l in val) / len(val)
            if acc > best:
                best, best_state = acc, self.bag.weight.detach().clone()
        self.bag.weight.data.copy_(best_state)
        return best


def items(texts):
    return [i for i, t in enumerate(texts) if t != NONE_TEXT]


def policies(tf):
    return {"first": lambda tx: 0, "last_item": lambda tx: items(tx)[-1],
            "longest_text": lambda tx: max(items(tx), key=lambda i: (len(tx[i]), -i)),
            "tfidf_candidate_only": lambda tx: int(tf.scores(tx).argmax())}


def evaluate(bench, pol, axis):
    agg = collections.defaultdict(lambda: [0.0, 0])
    for r in bench:
        x = {"texts": r["texts"], "label_index": r["label_index"]} if axis == "normal" else r["variants"][axis]
        if x is None:
            continue
        hit = (1.0 / len(x["texts"])) if pol is None else float(pol(x["texts"]) == x["label_index"])   # None = random, expected value
        for key in ("all", f"split:{r['split']}", f"tier:{r['tier']}", f"covered:{r['candidate_coverage']}"):
            agg[key][0] += hit
            agg[key][1] += 1
    return {k: {"accuracy": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(agg.items())}


def main(records, bench_path, name):
    train, val = [], []
    for l in open(records, encoding="utf-8"):
        r = json.loads(l)
        if r["split"] in ("train", "cal"):
            (train if r["split"] == "train" else val).append(([c["text"] for c in r["candidates"]], r["label_index"]))
    bench = [json.loads(l) for l in open(bench_path, encoding="utf-8")]
    tf = TfIdf(train)
    cal_acc = tf.fit(train, val)
    pols = {"random": None, **policies(tf)}
    res = {}
    for pname, pol in pols.items():
        res[pname] = {ax: evaluate(bench, pol, ax) for ax in AXES}
    cov = {s: round(sum(r["candidate_coverage"] for r in bench if r["split"] == s) / sum(1 for r in bench if r["split"] == s), 4) for s in ("test", "ood")}
    e2e = {p: {s: round(res[p]["normal"].get(f"covered:True", {"accuracy": 0})["accuracy"] * cov[s], 4) for s in ("test", "ood")} for p in pols}
    out = {"benchmark": str(bench_path), "decisions": len(bench), "tfidf_cal_accuracy": round(cal_acc, 4), "coverage": cov,
           "end_to_end_note": "accuracy on covered decisions x coverage; an uncovered decision cannot be solved from a state-only candidate set. Uses the pooled covered accuracy for both splits: see per-split keys for exact figures.",
           "end_to_end_pooled": e2e, "results": res}
    with open(ROOT / "results" / "raw" / f"a5-baselines-{name}.json", "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"TF-IDF cal accuracy {cal_acc:.3f} | coverage {cov}")
    print(f"{'baseline':22s}" + "".join(f"{a:>12s}" for a in AXES) + "   (all test+ood decisions)")
    for p in pols:
        print(f"{p:22s}" + "".join(f"{res[p][a]['all']['accuracy']:12.3f}" for a in AXES))
    print("normal axis by split / tier / coverage:")
    for p in pols:
        r = res[p]["normal"]
        print(f"  {p:22s} " + "  ".join(f"{k.split(':')[0][:5]}={k.split(':')[1]} {v['accuracy']:.3f}" for k, v in r.items() if k != "all"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
