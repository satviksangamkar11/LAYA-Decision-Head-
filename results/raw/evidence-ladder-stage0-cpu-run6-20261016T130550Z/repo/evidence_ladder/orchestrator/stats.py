"""Paired statistics (stdlib only): outcome table, exact McNemar, Holm, seeded paired bootstrap."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

SAMPLE_SIZE_ESTIMATE = 470   # estimate (unmeasured), S6 section 7: 5-point gain, discordant share 0.15, alpha .05, power .8


def paired_table(x: Sequence[int], y: Sequence[int]) -> Dict[str, int]:
    """x = challenger, y = baseline, same task order. gained: x solved & y not. lost: y solved & x not."""
    if len(x) != len(y):
        raise ValueError("unequal lengths")
    t = {"gained": 0, "lost": 0, "both": 0, "neither": 0}
    for a, b in zip(x, y):
        t["both" if a and b else "gained" if a else "lost" if b else "neither"] += 1
    return t


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar = two-sided binomial sign test on the discordant pairs, p capped at 1."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(p: Dict[str, float]) -> Dict[str, float]:
    order = sorted(p, key=lambda k: (p[k], k))
    m, out, run = len(order), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def bootstrap_diff_ci(x: Sequence[int], y: Sequence[int], resamples: int = 2000, seed: int = 0, clusters: Optional[Sequence] = None, level: float = 0.95):
    """Paired difference in solve rate (x - y), percentile interval, resampling clusters (default: each task is a cluster)."""
    n = len(x)
    if n == 0:
        raise ValueError("no pairs")
    groups: Dict = {}
    for i in range(n):
        groups.setdefault(clusters[i] if clusters is not None else i, []).append(x[i] - y[i])
    g = [groups[k] for k in sorted(groups, key=str)]
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        s = c = 0
        for _ in g:
            grp = g[rng.randrange(len(g))]
            s += sum(grp)
            c += len(grp)
        diffs.append(s / c)
    diffs.sort()
    a = (1 - level) / 2
    return {"estimate": sum(x[i] - y[i] for i in range(n)) / n, "lo": diffs[int(math.floor(a * resamples))],
            "hi": diffs[int(math.ceil((1 - a) * resamples)) - 1], "resamples": resamples, "seed": seed, "n_pairs": n}


def sample_size_warnings(n: int) -> List[str]:
    w = []
    if n < 30:
        w.append(f"n={n} pairs is very small: intervals are wide and p-values are close to meaningless")
    if n < SAMPLE_SIZE_ESTIMATE:
        w.append(f"n={n} is below the ~{SAMPLE_SIZE_ESTIMATE} problems estimated (unmeasured) for a 5-point paired gain at discordant share 0.15; "
                 "underpowered for smaller effects")
    return w
