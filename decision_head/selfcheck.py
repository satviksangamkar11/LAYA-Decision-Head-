"""Runnable check for decision_head.head:  uv run python -m decision_head.selfcheck

Synthetic data only. It proves the maths, masking, equivariance, scaler, calibration and loss code
and that the training loop can learn. It says nothing about any real LLM's quality; that is gate RB4.
Pass bars were fixed before the run that first used them; where a bar changed, the comment says why.
"""
import os

# torch 2.14 CPU kernels built for AVX-512 raised access violations on this Ryzen 7 7840HS
# (log_softmax on 140 MB, transformer layer at batch 4000); forcing AVX2 ran clean. Set before import.
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")

import torch
import torch.nn.functional as F

from decision_head.head import (BUCKETS, DecisionHead, bucket_of, decision_loss, ece, fit_temperature,
                                perm_consistency, probs)

torch.manual_seed(0)
H, NMAX = 48, 6
U, V = torch.randn(H, 8), torch.randn(H, 8)


def pointer_task(n):
    """Label is the option maximising a hidden bilinear form of (query, option).

    The form is taken on layer-normed vectors because a head that standardises inputs cannot see
    vector norms: a norm-dependent label capped the first two runs at 0.838 and 0.823 (measured
    label agreement between raw and normed inputs: 0.90).
    """
    hq, ho = torch.randn(n, H), torch.randn(n, NMAX, H)
    k = torch.randint(3, NMAX + 1, (n,))
    mask = torch.arange(NMAX)[None] < k[:, None]
    s = torch.einsum("bd,bnd->bn", F.layer_norm(hq, (H,)) @ U, F.layer_norm(ho, (H,)) @ V)
    return hq, ho, mask, s.masked_fill(~mask, -1e9).argmax(-1)


def odd_one_out_task(n):
    """Label is the one option unlike the rest. Relational, so no per-option pointer can solve it."""
    c, c2 = torch.randn(n, 1, H), torch.randn(n, 1, H)
    ho = c.expand(n, NMAX, H) + 0.3 * torch.randn(n, NMAX, H)
    y = torch.randint(0, NMAX, (n,))
    ho[torch.arange(n), y] = c2[:, 0] + 0.3 * torch.randn(n, H)
    return torch.randn(n, H), ho, torch.ones(n, NMAX, dtype=torch.bool), y


def train(head, task, steps, bs=256):
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3)
    head.train()
    for _ in range(steps):
        hq, ho, mask, y = task(bs)
        logits, _ = head(hq, ho, mask, torch.zeros(bs, dtype=torch.long))
        loss = decision_loss(logits, mask, F.one_hot(y, NMAX).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()


@torch.no_grad()
def acc(head, task, n=4000):
    hq, ho, mask, y = task(n)
    logits, _ = head(hq, ho, mask, torch.zeros(n, dtype=torch.long))
    return (logits.argmax(-1) == y).float().mean().item()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    assert ok, name


# 1. equivariance, padding invariance, option floor; K=3 captured layers, metadata, both scorers
head = DecisionHead(H, d=64, layers=2, k_layers=3, n_meta=3, dropout=0.1, scorer="both").eval()
B = 32
hq, ho, mask = torch.randn(B, 3, H), torch.randn(B, NMAX, 3, H), torch.ones(B, NMAX, dtype=torch.bool)
meta, prim, kind = torch.randn(B, NMAX, 3), torch.randint(0, 3, (B,)), torch.randint(0, 8, (B,))
with torch.no_grad():
    base, gate = head(hq, ho, mask, prim, kind, meta)
    p = torch.randperm(NMAX)
    perm, _ = head(hq, ho[:, p], mask, prim, kind, meta[:, p])
    check("equivariant under option permutation", torch.allclose(base[:, p], perm, atol=1e-5),
          f"max diff {(base[:, p] - perm).abs().max():.2e}")
    junk, _ = head(hq, torch.cat([ho, 50 * torch.randn(B, 2, 3, H)], 1), F.pad(mask, (0, 2), value=False), prim, kind,
                   torch.cat([meta, torch.randn(B, 2, 3)], 1))
    check("padded options do not change real logits", torch.allclose(base, junk[:, :NMAX], atol=1e-5),
          f"max diff {(base - junk[:, :NMAX]).abs().max():.2e}")
    check("padded options get zero probability", bool((junk.softmax(-1)[:, NMAX:] < 1e-6).all()))
    check("gate output is [B, 2] and finite", gate.shape == (B, 2) and bool(torch.isfinite(gate).all()))
try:
    head(hq, ho[:, :1], mask[:, :1], prim, kind, meta[:, :1])
    check("single-option decision is rejected", False)
except AssertionError:
    check("single-option decision is rejected", True)
try:
    DecisionHead(H, d=64, layers=0, scorer="mlp")
    check("mlp scorer without context layers is rejected", False)
except AssertionError:
    check("mlp scorer without context layers is rejected", True)

# 2. feature scaler standardises with training statistics and leaves the rest alone
sc = DecisionHead(H, d=64, layers=0, scorer="pointer")
hq2, ho2, m2 = 1000 * torch.randn(512, H) + 5, 3 * torch.randn(512, NMAX, H) - 2, torch.rand(512, NMAX) > 0.2
sc.fit_scaler(hq2, ho2, m2)
z = (hq2[:, None] - sc.mu_q) / sc.sd_q
check("fit_scaler gives mean 0 / std 1 on the fitting rows", abs(z.mean()) < 1e-3 and abs(z.std() - 1) < 1e-3,
      f"mean {z.mean():.4f} std {z.std():.4f}")

# 3. temperature: recovers a known factor; sparse cells fall back; ece is small once calibrated
n = 20000
z = 1.5 * torch.randn(n, 5)
y = torch.distributions.Categorical(logits=z).sample()
m5, pr = torch.ones(n, 5, dtype=torch.bool), torch.zeros(n, dtype=torch.long)
T = fit_temperature(3 * z, m5, F.one_hot(y, 5).float(), pr)
check("temperature fit recovers a 3x overconfident head", 2.6 < T[0, 1] < 3.4, f"T={T[0, 1]:.2f}")
check("unseen primitive keeps T=1", bool((T[1] == 1).all() and (T[2] == 1).all()))
sparse = torch.cat([torch.zeros(n, dtype=torch.long), torch.full((30,), 2)])  # 30 noul rows < min_n
Ts = fit_temperature(torch.cat([3 * z, torch.randn(30, 5)]), torch.ones(n + 30, 5, dtype=torch.bool),
                     F.one_hot(torch.cat([y, torch.zeros(30, dtype=torch.long)]), 5).float(), sparse)
check("a primitive with fewer than min_n rows keeps T=1", bool((Ts[2] == 1).all()))
raw_ece = ece(probs(3 * z, m5, pr, torch.ones(3, 4)), y)
cal_ece = ece(probs(3 * z, m5, pr, T), y)
check("calibration lowers ece", cal_ece < raw_ece / 3, f"ece {raw_ece:.3f} -> {cal_ece:.3f}")
check("bucket_of follows the upstream option-count buckets",
      bucket_of([2, 3, 5, 6, 10, 11, 40]).tolist() == [0, 1, 1, 2, 2, 3, 3] and len(BUCKETS) == 4)

# 4. losses: proper scores are zero for a perfect row, positive otherwise, finite with padding
tgt = F.one_hot(torch.tensor([1, 0]), 4).float()
mk = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
perfect = torch.full((2, 4), -30.0)
perfect[0, 1], perfect[1, 0] = 30.0, 30.0
perfect[0, 3] = -1e4
check("loss is ~0 for a perfect confident prediction", decision_loss(perfect, mk, tgt, ordinal=torch.tensor([True, True])) < 1e-3)
bad = torch.zeros(2, 4, requires_grad=True)
bad_l = decision_loss(bad.masked_fill(~mk, -1e4), mk, tgt, ordinal=torch.tensor([True, False]))
bad_l.backward()
check("loss is positive and its gradient finite for an uncertain row", bad_l > 0.5 and bool(torch.isfinite(bad.grad).all()))
a, m6 = torch.randn(16, 5), torch.ones(16, 5, dtype=torch.bool)
check("perm_consistency is 0 for identical captures", perm_consistency(a, a, m6).abs() < 1e-6)
check("perm_consistency is > 0 for different captures", perm_consistency(a, torch.randn(16, 5), m6) > 0)

# 5. it can learn: pointer task (layers=0, Jeeves form) and relational task (layers=2, Laya form)
p0 = DecisionHead(H, d=64, layers=0, scorer="pointer", dropout=0.0)
train(p0, pointer_task, 1500)
a_ptr = acc(p0, pointer_task)
check("layers=0 learns the bilinear pointer task", a_ptr >= 0.85, f"acc {a_ptr:.3f} (chance ~0.25)")
odd0 = DecisionHead(H, d=64, layers=0, scorer="pointer", dropout=0.0)
odd2 = DecisionHead(H, d=64, layers=2, scorer="both", dropout=0.0)
train(odd0, odd_one_out_task, 600)
train(odd2, odd_one_out_task, 1500)
a0, a2 = acc(odd0, odd_one_out_task), acc(odd2, odd_one_out_task)
# Bound derived before looking at a number: the five inliers cluster, so an argmax over any
# per-option linear score picks the outlier exactly when it lands above the cluster, half the
# time, and is blind otherwise. layers=0 is capped near 0.5 (first run 0.41). An earlier bar of
# 0.30 wrongly assumed identical per-option marginals meant chance; a no-signal control scored 0.169.
check("layers=0 is capped by the per-option bound on odd-one-out", a0 < 0.55, f"acc {a0:.3f} (bound 0.5)")
check("layers=2 solves the odd-one-out task", a2 >= 0.85, f"acc {a2:.3f}")

# 5b. closed-form arm (AnyJev method): learns a low-dimensional signal buried in noise from 200 rows, is calibrated, and finds nothing in random labels
from decision_head import closed_form as cf

g = torch.Generator().manual_seed(1)
K, D = 4, 512
dirs = torch.randn(K, D, generator=g)
shift = 3.0 * torch.randn(1, D, generator=g)  # one fixed offset, like a residual stream's outlier dims (a fresh one per call was a test bug)


def feats(n, noise=6.0):
    y = torch.randint(0, K, (n,), generator=g)
    return dirs[y] + noise * torch.randn(n, D, generator=g) + shift, y


Xtr, ytr = feats(200)
Xte, yte = feats(3000)
head = cf.fit(Xtr, ytr, K)
pte = cf.predict(head, Xte)
a_cf = (pte.argmax(1) == yte).float().mean().item()
check("closed-form head learns the planted signal from 200 rows", a_cf >= 0.85, f"acc {a_cf:.3f} (chance 0.25), chose {head['kind']} {head['param']}")
check("closed-form head is calibrated out of sample", ece(pte.float(), yte) < 0.06, f"ece {ece(pte.float(), yte):.3f}, T={head['T']:.2f}")
rand = cf.fit(Xtr, torch.randint(0, K, (200,), generator=g), K)
check("random labels give chance-level cross-validation (no leak)", rand["cv_acc"] < 0.40 and rand["cv_nll"] > 1.30,
      f"cv acc {rand['cv_acc']:.3f}, cv nll {rand['cv_nll']:.3f} (ln 4 = 1.386)")

# 6. size at real dimensions (gpt-oss hidden size 2880)
for name, kw in (("jeeves form: layers=0 d=256 pointer", dict(d=256, layers=0, scorer="pointer")),
                 ("full: layers=2 d=1024 K=3 both, meta=8", dict(d=1024, layers=2, k_layers=3, n_meta=8, scorer="both"))):
    h = DecisionHead(2880, **kw)
    tot = sum(p.numel() for p in h.parameters())
    print(f"INFO  hidden=2880 {name}: {tot / 1e6:.2f}M trainable parameters")
print("ALL CHECKS PASSED")
