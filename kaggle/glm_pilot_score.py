"""Score the GLM-5.3 pilot candidates with every frozen selector (user GO, chat 2026-10-13). Kaggle CPU; nothing is trained except the linear probe's
deterministic refit on TRAIN pools only. NO outcomes or labels are in the inputs: the output is each scorer's probabilities over each pool's options, joined
with the sandbox outcomes on the user's machine.

Inputs (mounted): laya-head-exp5-v1 (pack5.pt: 303 cleaned TRAIN pools + 46 VAL pools, exp4lib.py, head.py), laya-head-swap-v1 (heads.pt: the coding
head's 31 tensors + the 3 frozen Fable-trained adapters of laya-baseline-v1), kernel output laya-head-swap-rebuild (rebuilt/*.pt), kernel output
laya-glm-pilot-capture (out/level1/B.pt real context, A.pt neutral prefix), dataset laya-glm-pilot-capture-v1 (the requests: option order and hashes).

Scorers (fixed beforehand; no run is chosen on VAL): laya_v4 = coding head + adapter, rebuilt seeds 0-2; random_head_v4 = the 9 rebuilt random-head runs;
laya_baseline_v1 = coding head + the 3 frozen adapters (trained on the 334 v3 pools); linear_probe = exp4lib.probe_scores refitted on the 303 TRAIN pools.
Integrity check first: every rebuilt run must reproduce its recorded 46 VAL picks.
"""
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

import os
ROOTS = os.environ.get("SCORE_ROOTS", "/kaggle/input").split(os.pathsep)   # local runs (user, 2026-10-13: no Kaggle) list the input folders here
find = lambda pat: sorted(f for r in ROOTS for f in glob.glob(os.path.join(r, "**", pat), recursive=True))
SRC = Path(find("pack5.pt")[0]).parent
sys.argv = [sys.argv[0], str(SRC), os.environ.get("SCORE_OUT", "/kaggle/working")]
sys.path.insert(0, str(SRC))
import exp4lib as L  # noqa: E402
from head import DecisionHead  # noqa: E402

OUT = Path(os.environ.get("SCORE_OUT", "/kaggle/working"))


def main():
    pack = torch.load(SRC / "pack5.pt", map_location="cpu", weights_only=False)
    heads = torch.load(find("heads.pt")[0], map_location="cpu", weights_only=False)
    cfg = dict(pack["head_cfg"])
    rq = find("requests_level1.jsonl*")[0]   # Kaggle decompresses .gz files in datasets
    req = {r["decision_id"]: r for r in (json.loads(l) for l in (gzip.open if rq.endswith(".gz") else open)(rq, "rt", encoding="utf-8"))}
    lvl = Path(find("out/level1/B.pt")[0]).parent
    pools = {}
    for v in ("B", "A"):
        for x in torch.load(lvl / (v + ".pt"), map_location="cpu", weights_only=False)["records"]:
            u = req[x["decision_id"]]
            assert x["option_sha256"] == u["option_sha256"] and x["order_sha256"] == u["order_sha256"] and len(u["options"]) == x["ho"].shape[0]
            n = x["ho"].shape[0]
            pools.setdefault(v, []).append(dict(decision_id=x["decision_id"], hq=x["hq"], ho=x["ho"], pos=torch.zeros(n, dtype=torch.bool),
                                                conf=torch.zeros(n, dtype=torch.bool), ref=0, option_sha256=u["option_sha256"]))
    assert [p["decision_id"] for p in pools["B"]] == [p["decision_id"] for p in pools["A"]]
    dev = torch.device("cpu")
    prob = lambda h, rows: [s.softmax(-1).tolist() for s in L.scores(h, rows, dev)]
    runs = {}

    def head_from(state):
        h = DecisionHead(**cfg)
        r = h.load_state_dict(state, strict=False)
        assert not r.unexpected_keys
        return h

    rebuilt = sorted(find("rebuilt/*.pt"))
    integrity = []
    for p in rebuilt:
        st = torch.load(p, map_location="cpu", weights_only=False)
        h = head_from(st["state"])
        same = [int(s.argmax()) for s in L.scores(h, pack["val"], dev)] == st["val_picks"]
        integrity.append(dict(file=Path(p).name, reproduces_46_val_picks=same))
        fam = "laya_v4" if st["head"] == "coding" else "random_head_v4"
        runs["%s/%s-s%d" % (fam, st["head"], st["seed"])] = {v: prob(h, pools[v]) for v in pools}
    coding = {k: v.float() for k, v in heads["heads"]["coding"]["tensors"].items()}
    for i, ad in enumerate(heads["baseline_adapters"]):
        h = head_from({**coding, **{k: v for k, v in ad.items() if L.group_of(k) == "adapter" or k.startswith(("mu_", "sd_", "kind_emb"))}})
        runs["laya_baseline_v1/final-s%d" % i] = {v: prob(h, pools[v]) for v in pools}
    train = pack["train"]
    runs["linear_probe/train303"] = {v: [np.exp(s - s.max()) / np.exp(s - s.max()).sum() for s in L.probe_scores(train, pools[v])] for v in pools}
    runs["linear_probe/train303"] = {v: [list(map(float, s)) for s in x] for v, x in runs["linear_probe/train303"].items()}
    out = dict(pools=[dict(decision_id=p["decision_id"], option_sha256=p["option_sha256"], n_options=int(p["ho"].shape[0])) for p in pools["B"]],
               integrity=integrity, all_rebuilt_reproduce=all(x["reproduces_46_val_picks"] for x in integrity), runs=runs,
               note="probabilities in each pool's request option order; variant B real context, A neutral prefix (no context)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SCORES.json").write_text(json.dumps(out, indent=1), "utf-8")
    print("integrity", json.dumps(integrity), flush=True)
    for k, v in runs.items():
        print(k, [int(np.argmax(s)) for s in v["B"]], "| no context", [int(np.argmax(s)) for s in v["A"]], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
