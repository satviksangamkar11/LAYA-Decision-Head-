"""A5-v3 heads. H0 is state-free BY SIGNATURE: forward(x, te, mask) has no argument through which a trajectory-conditioned vector could enter.
The residual branch is the only module that receives the contextual state (query + contextual candidate vectors). H2's score = frozen H0 score + residual.
Spec: configs/a5_v3_spec.toml [amendment_2026_10_02_c]."""
import torch
import torch.nn as nn

FEATURE_SD_FLOOR = 0.1     # sparse hashed-token columns: a token never seen in training must not explode after standardisation
EMBED_SD_FLOOR = 1e-3
NEG = float("-inf")


class H0Net(nn.Module):
    """16 dense + 64 hashed features, plus state-free candidate embeddings at every depth: each part standardised and projected separately, then a 2-layer MLP."""

    def __init__(self, nf, emb, nd, proj, dropout):
        super().__init__()
        self.fp = nn.Linear(nf, proj)
        self.tp = nn.ModuleList(nn.Linear(emb, proj) for _ in range(nd))
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear((1 + nd) * proj, proj), nn.ReLU(), nn.Dropout(dropout), nn.Linear(proj, 1))
        for n, shape in (("mu_f", (nf,)), ("sd_f", (nf,)), ("mu_t", (nd, emb)), ("sd_t", (nd, emb))):
            self.register_buffer(n, torch.zeros(shape) if n.startswith("mu") else torch.ones(shape))

    def set_scalers(self, st):
        self.mu_f.copy_(st["mu_f"]), self.sd_f.copy_(st["sd_f"]), self.mu_t.copy_(st["mu_t"]), self.sd_t.copy_(st["sd_t"])

    def forward(self, x, te, mask):
        z = [self.fp((x - self.mu_f) / self.sd_f)] + [p((te[:, :, d] - self.mu_t[d]) / self.sd_t[d]) for d, p in enumerate(self.tp)]
        return self.mlp(torch.cat(z, -1)).squeeze(-1).masked_fill(~mask, NEG)


class ResidualNet(nn.Module):
    """Pointer interaction of the standardised query with the contextual candidate vectors, one projection pair per used depth.
    treatment: single (one fixed depth), mean (equal weight over the depths), learned (softmax mixture). wk starts at zero so H2 starts exactly at H0."""

    def __init__(self, emb, nd, proj, treatment, dropout, single_depth_index):
        super().__init__()
        assert treatment in ("single", "mean", "learned")
        self.treatment, self.proj = treatment, proj
        self.used = [single_depth_index] if treatment == "single" else list(range(nd))
        self.wq = nn.ModuleList(nn.Linear(emb, proj, bias=False) for _ in self.used)
        self.wk = nn.ModuleList(nn.Linear(emb, proj, bias=False) for _ in self.used)
        for w in self.wk:
            nn.init.zeros_(w.weight)
        self.mix = nn.Parameter(torch.zeros(len(self.used))) if treatment == "learned" else None
        self.drop = nn.Dropout(dropout)
        for n in ("mu_q", "sd_q", "mu_c", "sd_c"):
            self.register_buffer(n, torch.zeros(nd, emb) if n.startswith("mu") else torch.ones(nd, emb))

    def set_scalers(self, st):
        self.mu_q.copy_(st["mu_q"]), self.sd_q.copy_(st["sd_q"]), self.mu_c.copy_(st["mu_c"]), self.sd_c.copy_(st["sd_c"])

    def forward(self, q, cx, mask):
        s = []
        for j, d in enumerate(self.used):
            qq = self.drop(self.wq[j]((q[:, d] - self.mu_q[d]) / self.sd_q[d]))
            kk = self.drop(self.wk[j]((cx[:, :, d] - self.mu_c[d]) / self.sd_c[d]))
            s.append((kk * qq.unsqueeze(1)).sum(-1) / self.proj ** 0.5)
        s = torch.stack(s)
        w = torch.softmax(self.mix, 0) if self.mix is not None else torch.full((len(self.used),), 1.0 / len(self.used))
        return (w.view(-1, 1, 1) * s).sum(0).masked_fill(~mask, 0.0)


def h2_logits(h0_score, residual, mask):
    """H2 = frozen H0 score + residual state score."""
    return (h0_score + residual).masked_fill(~mask, NEG)
