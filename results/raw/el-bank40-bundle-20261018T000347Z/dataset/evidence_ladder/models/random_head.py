"""[EL-ARCH-1 C1] Random-head control, done properly. A randomly initialised network of the SAME architecture as the shipped scorer (decision_head.head.DecisionHead, hidden 2560, d 1024,
2 context layers, 3 depths, MLP scorer), every learned tensor from a fixed seed, nothing trained, nothing loaded: neither the 31 pretrained tensors nor any Fable-adapter tensor.
Scores for a pool are the logits of that untrained network, one score vector per seed (RANDOM_SEEDS) and their mean.

What this control is and is not (stated, not hidden):
  * It answers: does a head that has learned NOTHING already rank like LAYA on these pools? It is an untrained FLOOR, not a trained-from-scratch baseline (training a head is out of scope).
  * It is NOT the old control that kaggle/evidence_ladder_laya_gen_v1.py builds (trained adapter final-s0 + fresh random head). That one stays in the receipts as the
    'adapter_s0_with_random_head MISMATCH CONTROL' (VAL top-1 0.1698); it mixes a trained adapter with an untrained head, so it tests neither part.
  * The only thing optionally taken from training is the feature scaler (mu/sd buffers): per-dimension mean and std of the TRAIN features, data statistics rather than learned weights.
    Without it raw residual states have outlier dimensions that dominate any untrained network. scaler_from=None keeps the identity scaler (also valid; report both)."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence

from evidence_ladder.models import laya_forward as LF

ROLE = "untrained_random_init_floor_control"
RANDOM_SEEDS = (11, 12, 13)           # fixed; not 0, so no collision with the random_head_seed=0 of the old mismatch control
OLD_CONTROL_LABEL = "adapter_s0_with_random_head_MISMATCH_CONTROL"
SCALER_BUFFERS = ("mu_q", "mu_o", "sd_q", "sd_o")


def build_random_head(seed: int, scaler_from: Optional[str] = None):
    """Same constructor as laya_forward.build_head, but NOTHING is loaded except (optionally) the four scaler buffers of an adapter file (see module doc)."""
    import torch
    from decision_head.head import DecisionHead
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        head = DecisionHead(LF.HIDDEN, d=1024, layers=2, k_layers=len(LF.DEPTHS), scorer="mlp", dropout=0.0)
    with torch.no_grad():
        head.kind_emb.weight.zero_()      # as in build_head: the choice decision uses no kind embedding
    if scaler_from is not None:
        st = torch.load(scaler_from, map_location="cpu", weights_only=False)["state"]
        for k in SCALER_BUFFERS:
            head.get_buffer(k).copy_(st[k].float())
    return head.eval()


def state_sha256(head) -> str:
    h = hashlib.sha256()
    for k, v in sorted(head.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def build_random_heads(seeds: Sequence[int] = RANDOM_SEEDS, scaler_from: Optional[str] = None) -> Dict[str, Any]:
    return {"random-s%d" % s: build_random_head(s, scaler_from) for s in seeds}


def random_head_scores(heads: Mapping[str, Any], hq, ho) -> Dict[str, Any]:
    """hq [3,H], ho [N,3,H] -> {'score': mean over seeds, 'per_seed_logits', 'role'} (same shape LAYA's scorer returns)."""
    per = {n: LF.head_logits(h, hq, ho) for n, h in heads.items()}
    n = ho.shape[0]
    return {"score": [sum(per[k][i] for k in per) / len(per) for i in range(n)], "per_seed_logits": per, "role": ROLE, "seeds": sorted(int(k.split("-s")[1]) for k in per)}
