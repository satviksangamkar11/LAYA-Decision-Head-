"""Load the pilot capture into the tensors the heads consume. Model-independent: hidden size, depths and option counts come from the files.

Inputs (all frozen before capture): results/raw/pilot-manifest-v1.json, data/pilot/laya-fields-v1.jsonl (strata), data/pilot/capture/*.pt.
Output of `load(depth)`: dict of tensors
    hq   [N, H]            query vector at that depth
    ho   [N, K, H]         option vectors, zero-padded to K = max options
    mask [N, K] bool       real options (NONE is a normal option)
    meta [N, K, 12]        head_meta rows (never `source`)
    y    [N]               label index in the presented order
    split, tier, covered, kind: python lists aligned with rows
Never touches test/OOD: the manifest universe is train + cal only (asserted).
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _files():
    return sorted((ROOT / "data/pilot/capture").glob("*.pt"))


def load(depth, files=None):
    man = json.load(open(ROOT / "results/raw/pilot-manifest-v1.json", encoding="utf-8"))
    ann = {}
    for l in open(ROOT / "data/pilot/laya-fields-v1.jsonl", encoding="utf-8"):
        r = json.loads(l)
        ann[r["decision_id"]] = r
    rows = [torch.load(f, map_location="cpu") for f in (files or _files())]
    assert rows, "no capture files"
    assert all(r["schema"] == "capture-record-v1" and depth in r["depths"] for r in rows)
    di = rows[0]["depths"].index(depth)
    H = rows[0]["query"].shape[-1]
    K = max(len(r["order"]) for r in rows)
    N = len(rows)
    hq, ho = torch.zeros(N, H), torch.zeros(N, K, H)
    mask, meta, y = torch.zeros(N, K, dtype=torch.bool), torch.zeros(N, K, 12), torch.zeros(N, dtype=torch.long)
    keep = {"split": [], "tier": [], "covered": [], "kind": [], "decision_id": []}
    allowed = set(man["decision_ids"])
    for i, r in enumerate(rows):
        assert r["decision_id"] in allowed, "capture row outside the frozen manifest"
        a = ann[r["decision_id"]]
        assert a["split"] in ("train", "cal"), "locked split leaked into the pilot"
        n = len(r["order"])
        hq[i], ho[i, :n] = r["query"][di], r["options"][di]
        mask[i, :n] = True
        meta[i, :n] = torch.tensor(r["meta"])
        y[i] = r["label_index"]
        for k, v in (("split", a["split"]), ("tier", a["tier"]), ("covered", a["candidate_coverage"]),
                     ("kind", a["next_action_kind"]), ("decision_id", r["decision_id"])):
            keep[k].append(v)
    return dict(hq=hq, ho=ho, mask=mask, meta=meta, y=y, **keep)


def selfcheck():
    """Run after the capture is finished (or on a handful of files): shapes, masking, split purity, one-hot label inside the mask."""
    fs = _files()
    d = load(36, fs[:20] if len(fs) >= 20 else fs)
    n = len(d["y"])
    assert d["hq"].shape[0] == n == d["mask"].shape[0] and d["ho"].shape[:2] == d["mask"].shape
    assert (d["y"] < d["mask"].sum(1)).all(), "label outside the real options"
    assert (d["ho"][~d["mask"]] == 0).all(), "padding not zero"
    assert set(d["split"]) <= {"train", "cal"}
    print("OK", n, "rows", tuple(d["hq"].shape), tuple(d["ho"].shape))


if __name__ == "__main__":
    selfcheck()
