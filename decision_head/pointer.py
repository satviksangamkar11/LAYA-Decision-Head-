"""Pointer head for variable candidate lists (Tier-2, registered in configs/thresholds.toml [tier2_action_pointer] v1).
score_i = q . k_i / sqrt(D) + meta_bias(meta_i); q = Wq(standardise(hq)), k_i = Wk(standardise(ho_i)). Per-candidate scoring, so permutation
invariance holds by construction (tested in selfcheck). Model-independent: hidden size comes from the data."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerHead(nn.Module):
    def __init__(self, hidden, D, n_meta=12, dropout=0.1):
        super().__init__()
        self.D = D
        self.wq, self.wk = nn.Linear(hidden, D, bias=False), nn.Linear(hidden, D, bias=False)
        self.meta = nn.Linear(n_meta, 1)
        nn.init.zeros_(self.meta.weight), nn.init.zeros_(self.meta.bias)
        self.drop = nn.Dropout(dropout)
        for n in ("mu_q", "sd_q", "mu_o", "sd_o"):
            self.register_buffer(n, torch.zeros(hidden) if n.startswith("mu") else torch.ones(hidden))

    def fit_scalers(self, hq, ho, mask):
        self.mu_q.copy_(hq.mean(0)), self.sd_q.copy_(hq.std(0) + 1e-6)
        o = ho[mask]
        self.mu_o.copy_(o.mean(0)), self.sd_o.copy_(o.std(0) + 1e-6)

    def forward(self, hq, ho, mask, meta):
        q = self.drop(self.wq((hq - self.mu_q) / self.sd_q))            # [B, D]
        k = self.drop(self.wk((ho - self.mu_o) / self.sd_o))            # [B, K, D]
        s = (k * q.unsqueeze(1)).sum(-1) / self.D ** 0.5 + self.meta(meta).squeeze(-1)
        return s.masked_fill(~mask, float("-inf"))


def nll(logits, y):
    return F.cross_entropy(logits, y).item()


def selfcheck():
    torch.manual_seed(0)
    h = PointerHead(32, 16, dropout=0.0).eval()
    hq, ho = torch.randn(4, 32), torch.randn(4, 6, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1] * 6], dtype=torch.bool)
    meta = torch.randn(4, 6, 12)
    h.fit_scalers(hq, ho, mask)
    a = h(hq, ho, mask, meta)
    perm = torch.stack([torch.randperm(6) for _ in range(4)])
    idx = perm.unsqueeze(-1)
    b = h(hq, ho.gather(1, idx.expand(-1, -1, 32)), mask.gather(1, perm), meta.gather(1, idx.expand(-1, -1, 12)))
    assert torch.allclose(a.gather(1, perm), b, atol=1e-5, equal_nan=False) or torch.allclose(
        a.gather(1, perm).nan_to_num(-1e9), b.nan_to_num(-1e9), atol=1e-5), "not permutation invariant"
    assert torch.isinf(a[~mask]).all() and torch.isfinite(a[mask]).all()
    print("OK pointer head: permutation-invariant, masked slots -inf")


if __name__ == "__main__":
    selfcheck()
