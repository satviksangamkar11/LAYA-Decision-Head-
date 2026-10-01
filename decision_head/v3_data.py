"""A5-v3 data assembly for the trainer: TRAIN and CAL only. This module has no path to the locked set and no locked loader (the trainer modules are statically
scanned for that in tools/v3_trainer_audit.py); the final evaluation lives in decision_head/v3_evaluation.py.
A "decision" here is a dict built by assemble(): features X (compiler.a5v2, 16 dense + 64 hashed text tokens), the captured query/contextual candidate vectors, the state-free
candidate text embeddings (indices into one shared table, so a text maps to ONE vector whatever decision it appears in) and its candidate sets (the NONE option is captured
but is not part of the scored sets: the registered sets come from the frozen sampler, which never contains it).
Spec: configs/a5_v3_spec.toml [amendment_2026_10_02_c] and [capture_v3]."""
import hashlib
import json
import tomllib
from pathlib import Path

import torch

from capture.pool import NONE_TEXT, SCHEMA
from compiler import a5v2
from compiler.firewall import load_cal_calibration_only, load_dev

ROOT = Path(__file__).resolve().parents[1]
SPEC = tomllib.load(open(ROOT / "configs/a5_v3_spec.toml", "rb"))
DEPTHS = SPEC["capture_v3"]["depths"]
N_TRAIN_SETS = 4
NF = a5v2.ND + a5v2.NH
FROZEN = ROOT / "data/a5v2/odds-a5v2-odds-1.pt"
FROZEN_SHA_PREFIX = "20d5d2b8d62aedfe"
DECISIONS = ROOT / "data/decisions/action-swerebench-sample-v2.jsonl"
RAW_OLD = ROOT / "data/raw/swerebench-sample-20261001.jsonl"


def frozen_odds():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest().startswith(FROZEN_SHA_PREFIX), "frozen sampler artifact changed"
    return torch.load(FROZEN, map_location="cpu")


def load_textemb(textemb_dir):
    d = {}
    for f in sorted(Path(textemb_dir).glob("shard-*.pt")):
        d.update(torch.load(f, map_location="cpu"))
    texts = sorted(d)
    return torch.stack([d[t] for t in texts]), {t: i for i, t in enumerate(texts)}


def load_captures(capdir):
    caps = {}
    for f in sorted(Path(capdir).glob("*.pt")):
        c = torch.load(f, map_location="cpu")
        caps[c["decision_id"]] = c
    return caps


def oof_sets_multi(decs, n=N_TRAIN_SETS):
    """n candidate sets per TRAIN decision, generated OUT OF FOLD with the same fold odds models as the registered audit. Set 0 == a5v2.oof_sets (asserted in the audit)."""
    fold_of = a5v2.folds_by_trajectory(decs)
    out = {}
    for f in range(a5v2.S["cv_folds"]):
        for st in ("repeat", "first_time"):
            m = a5v2.fit_odds([d for d in decs if fold_of[d["tid"]] != f and d["stratum"] == st])
            for d in decs:
                if fold_of[d["tid"]] == f and d["stratum"] == st:
                    out[d["did"]] = [a5v2.generate(d, m)] + [a5v2.generate(d, m, aug=i) for i in range(1, n)]
    return out


def assemble(ids, built, caps, TE, te_idx, role, sets):
    """Join manifest ids with their decision features, captured vectors, state-free embeddings and candidate sets. Every alignment is asserted.
    Returns (decisions, missing_capture_ids). Decisions without a capture are reported, never silently dropped."""
    out, missing = [], []
    for did in ids:
        if did not in built:
            continue                                      # unusable decision (reported by the manifest builder), never a capture
        if did not in caps:
            missing.append(did)
            continue
        d, c = built[did], caps[did]
        P = len(d["texts"])
        assert c["schema"] == SCHEMA and c["depths"] == DEPTHS, f"{did}: capture schema/depths"
        assert c["texts"][:-1] == d["texts"] and c["texts"][-1] == NONE_TEXT, f"{did}: captured candidate texts differ from the rebuilt pool"
        assert c["pool_hash"] == d["pool_hash"] and c["state_hash"] == d["sh"], f"{did}: pool/state hash mismatch"
        assert c["cands"].shape[0] == P + 1 and c["cands"].shape[1] == len(DEPTHS) and c["query"].shape == (len(DEPTHS), c["cands"].shape[2]), f"{did}: capture shapes"
        assert NONE_TEXT not in d["texts"], f"{did}: NONE is in the sampled pool"
        assert all(t in te_idx for t in d["texts"]), f"{did}: a candidate text has no state-free embedding"
        for s in sets[did]:
            assert s["order"][s["true_pos"]] == d["ti"] and len(set(s["order"])) == len(s["order"]) and max(s["order"]) < P, f"{did}: candidate set invalid"
        out.append(dict(did=did, tid=d["tid"], stratum=d["stratum"], role=role, X=d["X"], ti=d["ti"], texts=d["texts"], q=c["query"], cx=c["cands"][:P],
                        te_idx=torch.tensor([te_idx[t] for t in d["texts"]]), TE=TE, sets=sets[did], pool_hash=d["pool_hash"], state_hash=d["sh"]))
    return out, missing


def _built_train():
    raw = a5v2.load_raw(RAW_OLD)
    built = {}
    for r in load_dev(DECISIONS):
        if r["candidate_coverage"]:
            d, _ = a5v2.build_decision(r, raw[r["decision_id"].rsplit(":", 1)[0]]["trajectory"])
            if d is not None:
                built[d["did"]] = d
    return built


def load_train(manifest_ids, capdir, textemb_dir):
    built = _built_train()
    sets = oof_sets_multi(list(built.values()))
    TE, te_idx = load_textemb(textemb_dir)
    return assemble(manifest_ids, built, load_captures(capdir), TE, te_idx, "train", sets)


def load_cal(manifest_ids, capdir, textemb_dir):
    """CAL: temperature fitting only. The decisions are never handed to a gradient step (train_loop asserts role == train)."""
    raw = a5v2.load_raw(RAW_OLD)
    want, built = set(manifest_ids), {}
    for r in load_cal_calibration_only(DECISIONS):
        if r["candidate_coverage"] and r["decision_id"] in want:
            d, _ = a5v2.build_decision(r, raw[r["decision_id"].rsplit(":", 1)[0]]["trajectory"])
            if d is not None:
                built[d["did"]] = d
    fz = frozen_odds()
    sets = {k: [a5v2.generate(d, fz[d["stratum"]])] for k, d in built.items()}
    TE, te_idx = load_textemb(textemb_dir)
    return assemble(manifest_ids, built, load_captures(capdir), TE, te_idx, "cal", sets)


def make_batch(items, h0s=None):
    """items: list of (decision, set_index). Padded float32 tensors. h0s: optional {(did, k): score vector} frozen-H0 scores for the residual training."""
    B, K = len(items), max(len(d["sets"][k]["order"]) for d, k in items)
    E = items[0][0]["q"].shape[-1]
    nd = len(DEPTHS)
    b = dict(x=torch.zeros(B, K, NF), te=torch.zeros(B, K, nd, E), cx=torch.zeros(B, K, nd, E), q=torch.zeros(B, nd, E), mask=torch.zeros(B, K, dtype=torch.bool),
             y=torch.zeros(B, dtype=torch.long), h0=torch.zeros(B, K))
    for i, (d, k) in enumerate(items):
        s = d["sets"][k]
        o = torch.tensor(s["order"])
        n = len(o)
        assert int(o.max()) < len(d["texts"]) and all(d["texts"][j] != NONE_TEXT for j in s["order"]), "the NONE option must never be scored"
        b["x"][i, :n], b["te"][i, :n], b["cx"][i, :n] = d["X"][o], d["TE"][d["te_idx"][o]].float(), d["cx"][o].float()
        b["q"][i], b["mask"][i, :n], b["y"][i] = d["q"].float(), True, s["true_pos"]
        if h0s is not None:
            b["h0"][i, :n] = h0s[(d["did"], k)]
    return b
