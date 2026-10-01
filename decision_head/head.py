"""Internal decision head over the frozen hidden states of any local LLM. Design record: DECISION_HEAD.md.
`hidden` has no default on purpose: it is read from the backbone's config, and a head is fitted per backbone.

Two surveyed heads meet here. PostHog/jeeves model/head.py is a pointer: linear q and k maps and a
scaled dot product (layers=0 reproduces it). Upstream Laya's DecisionModel (laya/common.py) runs a
2-layer pre-norm transformer over its markers and scores each with an MLP; we run the same layers
over [query, option_1..option_N] only, then score with the pointer, the MLP, or both. There is no
positional encoding anywhere, so the head is permutation-equivariant by construction. Order
dependence that remains lives in the backbone representations; the capture design removes it
(shared prefix, one independent suffix per candidate) and `perm_consistency` is the fallback.

Inputs, all captured from the frozen backbone (K = number of captured layers, H = hidden size):
  hq   [B, K, H]     query: hidden state at the last prompt position
  ho   [B, N, K, H]  options: mean over each option's own text tokens (a head on a fresh marker
                     token does not learn on a frozen model; a span mean does, see DECISION_HEAD.md)
  mask [B, N]        True for real options. NONE is an ordinary option, so N >= 2.
  prim [B]           0 choice, 1 score, 2 noul        (PRIMS)
  kind [B]           decision kind, index into KINDS  (default ACTION)
  meta [B, N, M]     optional candidate metadata. Runtime-visible fields only (action type, tool,
                     safety class, executable). Never the candidate source: it leaks the label.
K=1 inputs may drop the K axis. Features are standardised with statistics fitted on the training
split only (`fit_scaler`): raw residual states have a few huge outlier dims.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRIMS = ("choice", "score", "noul")
KINDS = ("ACTION", "EVIDENCE", "CONTINUE", "RECOVERY", "PATCH", "COMPLETION", "ESCALATE", "ABSTAIN", "CODE_RELEVANCE")
BUCKETS = ((2, 2), (3, 5), (6, 10), (11, 10**9))  # option-count buckets, as upstream Laya's temperature_by_options
NEG = -1e4  # finite on purpose: -inf makes masked terms NaN in backward


def bucket_of(n_options):
    return torch.tensor([next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= int(n) <= hi) for n in n_options])


class DecisionHead(nn.Module):
    def __init__(self, hidden, d=1024, layers=2, k_layers=1, n_meta=0, heads=None, dropout=0.1,
                 scorer="both", clip=8.0):
        super().__init__()
        assert scorer in ("pointer", "mlp", "both")
        assert layers > 0 or scorer != "mlp", "an MLP scorer without context layers never sees the query"
        self.cfg = dict(hidden=hidden, d=d, layers=layers, k_layers=k_layers, n_meta=n_meta,
                        heads=heads or max(1, d // 64), dropout=dropout, scorer=scorer, clip=clip)
        self.scorer, self.clip, self.k = scorer, clip, k_layers
        # feature scaler: identity until fit_scaler runs
        for name in ("mu_q", "mu_o"):
            self.register_buffer(name, torch.zeros(k_layers, hidden))
        for name in ("sd_q", "sd_o"):
            self.register_buffer(name, torch.ones(k_layers, hidden))
        self.mix_q, self.mix_o = nn.Parameter(torch.zeros(k_layers)), nn.Parameter(torch.zeros(k_layers))  # softmax over layers
        self.in_q, self.in_o = nn.Linear(hidden, d), nn.Linear(hidden, d)
        self.prim_emb, self.kind_emb = nn.Embedding(len(PRIMS), d), nn.Embedding(len(KINDS), d)
        self.meta = nn.Linear(n_meta, d) if n_meta else None
        if layers:
            layer = nn.TransformerEncoderLayer(d, self.cfg["heads"], 4 * d, dropout, batch_first=True, norm_first=True)
            self.ctx = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.out_norm = nn.LayerNorm(d)
        else:
            self.ctx, self.out_norm = None, nn.Identity()
        if scorer in ("pointer", "both"):
            self.pq, self.pk = (nn.Linear(d, d), nn.Linear(d, d)) if layers else (nn.Identity(), nn.Identity())
        if scorer in ("mlp", "both"):
            self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # abstain, escalate. Upstream Laya's act head read ~1.0 on almost every input and ran against
        # correctness (AUROC 0.30, laya issue #185): this one is trained on real labels and must beat
        # plain max-probability confidence on held-out data, or it is not shipped.
        self.gate = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, 2))

    @torch.no_grad()
    def fit_scaler(self, hq, ho, mask):
        """Per-layer, per-dim mean and std over the TRAINING split only. Option rows count only where mask is True."""
        hq, ho = self._k_axis(hq, ho)
        self.mu_q.copy_(hq.mean(0))
        self.sd_q.copy_(hq.std(0).clamp_min(1e-6))
        real = ho[mask]  # [R, K, H]
        self.mu_o.copy_(real.mean(0))
        self.sd_o.copy_(real.std(0).clamp_min(1e-6))

    @staticmethod
    def _k_axis(hq, ho):
        return (hq if hq.dim() == 3 else hq[:, None]), (ho if ho.dim() == 4 else ho[:, :, None])

    def _embed(self, x, mu, sd, mix, lin):
        x = ((x - mu) / sd).clamp(-self.clip, self.clip)  # [..., K, H]
        x = (x * mix.softmax(0)[:, None]).sum(-2)         # learned mix over captured layers
        return lin(x)

    def forward(self, hq, ho, mask, prim, kind=None, meta=None):
        assert ho.size(1) >= 2, "every decision has NONE plus at least one candidate"
        hq, ho = self._k_axis(hq, ho)
        kind = torch.zeros_like(prim) if kind is None else kind
        t = (self.prim_emb(prim) + self.kind_emb(kind))[:, None]
        q = self._embed(hq, self.mu_q, self.sd_q, self.mix_q, self.in_q)[:, None] + t
        o = self._embed(ho, self.mu_o, self.sd_o, self.mix_o, self.in_o) + t
        if self.meta is not None:
            o = o + self.meta(meta)
        if self.ctx is not None:
            x = self.out_norm(self.ctx(torch.cat([q, o], 1), src_key_padding_mask=~F.pad(mask, (1, 0), value=True)))
            q, o = x[:, :1], x[:, 1:]
        logits = 0.0
        if self.scorer in ("pointer", "both"):
            logits = logits + (self.pk(o) @ self.pq(q).transpose(1, 2)).squeeze(-1).float() / math.sqrt(o.size(-1))
        if self.scorer in ("mlp", "both"):
            logits = logits + self.mlp(o).squeeze(-1).float()
        logits = logits.masked_fill(~mask, NEG)
        p = logits.detach().softmax(-1)
        n = mask.sum(-1).float()
        ent = -(p * p.clamp_min(1e-9).log()).sum(-1) / n.log()
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, n / 255.0], -1)
        return logits, self.gate(torch.cat([q.squeeze(1).float(), feats], -1))


def decision_loss(logits, mask, target, ordinal=None, gate_logits=None, gate_target=None,
                  w_sph=0.75, w_rps=1.0, w_gate=0.1):
    """Proper scoring rules against a target distribution (one-hot, teacher soft labels, or a mix).

    log score (soft cross-entropy) + spherical (weight 0.75) + ranked probability score on the rows
    flagged ordinal (weight 1.0): the weights upstream Laya's recipe uses (docs/finetune.md). The
    hard/soft mixing weight is a dataset-side hyperparameter fixed before training. gate_target is
    [B, 2] floats (abstain, escalate).
    """
    lp = F.log_softmax(logits, -1)
    p = lp.exp()
    loss = -(target * lp).sum(-1).mean()
    loss = loss + w_sph * (1 - (p * target).sum(-1) / p.norm(dim=-1).clamp_min(1e-9)).mean()
    if ordinal is not None and ordinal.any():
        rps = ((p.cumsum(-1) - target.cumsum(-1)) ** 2).sum(-1) / (mask.sum(-1) - 1).clamp_min(1)
        loss = loss + w_rps * (rps * ordinal).sum() / ordinal.sum()
    if gate_logits is not None:
        loss = loss + w_gate * F.binary_cross_entropy_with_logits(gate_logits, gate_target)
    return loss


def perm_consistency(logits_a, logits_b, mask):
    """Symmetric KL between two captures of one decision, both already in canonical option order."""
    la, lb = F.log_softmax(logits_a, -1), F.log_softmax(logits_b, -1)
    return (((la.exp() * (la - lb)).sum(-1) + (lb.exp() * (lb - la)).sum(-1)) / 2).mean()


@torch.no_grad()
def fit_temperature(logits, mask, target, prim, min_n=100):
    """Softmax temperatures [len(PRIMS), len(BUCKETS)] by NLL grid search, fitted on the CALIBRATION split only.

    One temperature per (primitive, option-count bucket) as upstream Laya ships; a cell with fewer
    than min_n rows falls back to its primitive's overall fit, then to 1.0. Same grid as
    PostHog/jeeves calibrate.py. Loops over the grid: the all-at-once version needs 351 x rows x
    options floats and crashed a CPU kernel at 20k rows on this machine.
    """
    grid = torch.linspace(-1.5, 2.0, 351).exp()

    def fit(sel):
        nll = torch.stack([-(target[sel] * F.log_softmax(logits[sel] / g, -1)).sum() for g in grid])
        return grid[nll.argmin()]

    T = torch.ones(len(PRIMS), len(BUCKETS))
    bk = bucket_of(mask.sum(-1))
    for pi in range(len(PRIMS)):
        rows = prim == pi
        if rows.sum() < min_n:
            continue
        T[pi] = fit(rows)
        for bi in range(len(BUCKETS)):
            cell = rows & (bk == bi)
            if cell.sum() >= min_n:
                T[pi, bi] = fit(cell)
    return T


def probs(logits, mask, prim, T):
    return (logits / T[prim, bucket_of(mask.sum(-1))][:, None]).softmax(-1)


def ece(p, y, bins=15):
    """Expected calibration error on the top-1 probability (15 bins, as the laya-code and Laya evals)."""
    conf, pred = p.max(-1)
    ok = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.float().mean() * (conf[m].mean() - ok[m].mean()).abs()
    return float(err)
