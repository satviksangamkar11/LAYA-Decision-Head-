"""Closed-form decision head: the no-gradient arm. Method from nokia-applied-research/AnyJev docs/method_v3.md section 2.2.

Features are the residual-stream vector at the last prompt position at one block about 2/3 down the model, standardised with
statistics of the fitting rows. A linear head is solved in closed form (shrunk LDA or ridge to one-hot, both through N x N
systems), the configuration is chosen by 5-fold out-of-fold NLL, and the temperature is fitted on those out-of-fold scores.
One head per fixed question layout and per backbone; it does not transfer to another question. The trained set-attention head in
head.py is the version for variable candidate lists. Float64: N is a few hundred, so this is seconds on a CPU.
"""
import torch
import torch.nn.functional as F

GRID = torch.linspace(-1.5, 2.0, 351).exp()


def _ridge(X, y, K, lam):
    mu, Y = X.mean(0), F.one_hot(y, K).double()
    Xc, Yc = X - mu, Y - Y.mean(0)
    W = Xc.T @ torch.linalg.solve(Xc @ Xc.T + lam * torch.eye(len(X), dtype=X.dtype), Yc)
    return W, Y.mean(0) - mu @ W


def _lda(X, y, K, a):
    n, d = X.shape
    m = torch.stack([X[y == k].mean(0) if (y == k).any() else X.mean(0) for k in range(K)])      # [K, d]
    R = X - m[y]
    tau = (R * R).sum() / (n - 1) / d
    s0, c = a * tau, (1 - a) / (n - 1)                       # S = s0 I + c R^T R
    M = m.T                                                  # [d, K]
    inner = torch.linalg.solve(s0 / c * torch.eye(n, dtype=X.dtype) + R @ R.T, R @ M)
    W = (M - R.T @ inner) / s0                               # Woodbury: S^-1 M
    prior = torch.stack([(y == k).double().mean().clamp_min(1e-6) for k in range(K)])
    return W, -0.5 * (m * W.T).sum(1) + prior.log()


CONFIGS = [("lda", a) for a in (0.3, 0.6, 0.9)] + [("ridge", l) for l in (0.1, 1.0, 10.0, 100.0)]


def _solve(kind, p, X, y, K):
    return (_lda if kind == "lda" else _ridge)(X, y, K, p)


def _nll(s, y, T):
    return F.cross_entropy(s / T, y).item()


def _best_T(s, y):
    return GRID[torch.tensor([_nll(s, y, g) for g in GRID]).argmin()].item()


def fit(X, y, K, folds=5, seed=0):
    """X [N, d] float, y [N] class indices. Returns a dict head: kind, param, W, b, mu, sd, T, cv_nll, cv_acc."""
    X, y = X.double(), y.long()
    n = len(X)
    assert n >= max(8, 2 * K), "need at least max(8, 2K) labelled rows"
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fold = torch.arange(n)[perm] % folds
    best = None
    for kind, p in CONFIGS:
        oof = torch.zeros(n, K, dtype=torch.double)
        for f in range(folds):
            tr, te = fold != f, fold == f
            W, b = _solve(kind, p, Z[tr], y[tr], K)
            oof[te] = Z[te] @ W + b
        T = _best_T(oof, y)
        nll = _nll(oof, y, T)
        if best is None or nll < best[0]:
            best = (nll, kind, p, T, (oof.argmax(1) == y).double().mean().item())
    nll, kind, p, T, acc = best
    W, b = _solve(kind, p, Z, y, K)
    return dict(kind=kind, param=p, W=W, b=b, mu=mu, sd=sd, T=T, cv_nll=nll, cv_acc=acc)


def predict(head, X):
    """Probabilities [N, K] from features [N, d]; the temperature fitted out of fold is applied."""
    return (((X.double() - head["mu"]) / head["sd"]) @ head["W"] + head["b"]).div(head["T"]).softmax(-1)
