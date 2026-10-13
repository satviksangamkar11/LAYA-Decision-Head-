"""Rebuild with saved weights (user GO, chat 2026-10-13): the 12 VAL-configuration runs of the head-swap experiment (coding head seeds 0-2 and the three
random heads x seeds 0-2; adapter trained on folds 1-4, development fold 0, head frozen). Uses kaggle/head_swap_gpu.py UNCHANGED (imported); nothing
else is trained. GATE: every run must reproduce its recorded 46 cleaned-VAL picks exactly (results/raw/head-swap-run-20261012T222906Z/runs.jsonl);
a run that does not is reported and its weights are marked NOT USABLE. Outputs: rebuilt/<head>-s<seed>.pt (adapter + scaler state, the head's 31
tensors, config, best step), MANIFEST.json with sha256 of every file and the gate per run.
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(glob.glob("/kaggle/input/**/head_swap_gpu.py", recursive=True)[0]).parent if len(sys.argv) < 2 else Path(sys.argv[1])
sys.path.insert(0, str(HERE))
argv = sys.argv[:]
sys.argv = [argv[0]] + argv[2:5]          # local: <rebuild dir> <pack5 dir> <heads dir> <out>; Kaggle: none
import head_swap_gpu as HS  # noqa: E402

RUNS = [("coding", s) for s in HS.SEEDS] + [("random_%d" % r, s) for r in HS.RANDOM_SEEDS for s in HS.SEEDS]


def main():
    out = HS.OUT
    (out / "rebuilt").mkdir(parents=True, exist_ok=True)
    rec = {(r["head"], r["seed"]): r["val_pick"] for r in map(json.loads, open(HERE / "recorded_val_runs.jsonl", encoding="utf-8"))}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU", flush=True)
    if dev.type != "cuda" and not HS.SMOKE:
        raise SystemExit("STOP: no CUDA device; nothing was trained.")
    pack = torch.load(HS.SRC / "pack5.pt", map_location="cpu", weights_only=False)
    heads = torch.load(HS.HEADS / "heads.pt", map_location="cpu", weights_only=False)
    HS.HCFG = dict(pack["head_cfg"])
    rows, val, nval = pack["train"], pack["val"], pack["native_val"][:40] if HS.SMOKE else pack["native_val"]
    fold = lambda ks: [r for r in rows if r["fold"] in ks]
    built, orig = [], HS.build
    HS.build = lambda *a, **k: built.append(orig(*a, **k)) or built[-1]   # keep the trained object; run() loads its best state into it
    man = dict(runs=[], recorded_source="results/raw/head-swap-run-20261012T222906Z/runs.jsonl")
    for name, seed in (RUNS[:2] if HS.SMOKE else RUNS):
        res = HS.run(name, seed, fold({1, 2, 3, 4}), fold({0}), dict(val=val), nval, heads, dev)
        h = built[-1]
        same = res["val_pick"] == rec[(name, seed)]
        diff = sum(a != b for a, b in zip(res["val_pick"], rec[(name, seed)]))
        sd = {k: v.detach().cpu() for k, v in h.state_dict().items() if HS.L.group_of(k) in ("adapter", "pretrained") or k.startswith(("mu_", "sd_", "kind_emb"))}
        p = out / "rebuilt" / ("%s-s%d.pt" % (name, seed))
        torch.save(dict(state=sd, head=name, seed=seed, head_cfg=HS.HCFG, best_step=res["best_step"], recipe=HS.CFG, val_picks=res["val_pick"],
                        exact_match=same, usable=same), p)
        man["runs"].append(dict(head=name, seed=seed, file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), best_step=res["best_step"],
                                exact_match_46=same, picks_differing=diff, usable=same, head_tensors_identical=res["head_tensors_identical"]))
        print("REBUILD %-12s s%d exact %s (differing %d) best %d" % (name, seed, same, diff, res["best_step"]), flush=True)
    man["all_exact"] = all(r["exact_match_46"] for r in man["runs"])
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1), "utf-8")
    print("ALL EXACT", man["all_exact"], flush=True)


if __name__ == "__main__":
    main()
