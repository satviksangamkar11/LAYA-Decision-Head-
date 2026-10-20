"""P5 evaluation-only INDEPENDENT simple / brute-force implementations of C01-C20 (different algorithms from reference_solutions.py), used only to cross-check the reference on small inputs.
`brute_<task_id>(data)` returns the answer, or raises Infeasible when the input is too large for the brute-force method. Written from the statements only. Pure standard library."""
import itertools
import json
import re


class Infeasible(Exception):
    pass


def brute_C01(data):
    covered = set()
    for a, b in data["ranges"]:
        if b - a > 200000 or len(covered) > 500000:
            raise Infeasible("range too wide")
        covered.update(range(a, b + 1))
    return len(covered)


def brute_C02(data):
    s = data["s"]
    rounds = 0
    while s:
        t = re.sub(r"\(\)|\[\]|\{\}", "", s)             # ONE simultaneous pass: removes the innermost adjacent pairs only (no cascading inside the pass)
        if t == s:
            return -1                      # nothing could be removed and s is not empty: not balanced
        s = t
        rounds += 1
    return rounds                          # one round removes every innermost adjacent pair: rounds == maximum nesting depth


def brute_C03(data):
    return "".join(m.group(2) + str(len(m.group(1))) for m in re.finditer(r"((.)\2*)", data["s"], re.S))


def brute_C04(data):
    m = data["m"]
    n = len(m)
    res = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            res[j][n - 1 - i] = m[i][j]
    return res


def brute_C05(data):
    nums, t = data["nums"], data["target"]
    if len(nums) > 400:
        raise Infeasible("quadratic")
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == t:
                return [i, j]
    return []


def brute_C06(data):
    rest = list(data["nums"])
    out = []
    while rest:
        best = None
        for v in set(rest):
            key = (rest.count(v), -v)
            if best is None or key > best[0]:
                best = (key, v)
        v = best[1]
        out.extend([v] * rest.count(v))
        rest = [x for x in rest if x != v]
    return out


def brute_C07(data):
    words = data["words"]
    for k in range(len(words[0]), -1, -1):
        p = words[0][:k]
        if all(w.startswith(p) for w in words):
            return p


def brute_C08(data):
    s = data["s"]
    toks = [("M", 1000), ("CM", 900), ("D", 500), ("CD", 400), ("C", 100), ("XC", 90), ("L", 50), ("XL", 40), ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1)]
    total = 0
    while s:
        for sym, v in toks:
            if s.startswith(sym):
                total += v
                s = s[len(sym):]
                break
        else:
            raise ValueError("not a roman numeral: " + s)
    return total


def brute_C09(data):
    n = data["n"]
    places = [["", "M", "MM", "MMM"], ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"], ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"],
              ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]]
    return places[0][n // 1000] + places[1][n // 100 % 10] + places[2][n // 10 % 10] + places[3][n % 10]


def brute_C10(data):
    m = data["m"]
    if not m or not m[0]:
        return []
    R, C = len(m), len(m[0])
    seen = [[False] * C for _ in range(R)]
    r = c = d = 0
    dr, dc = [0, 1, 0, -1], [1, 0, -1, 0]
    out = []
    for _ in range(R * C):
        out.append(m[r][c])
        seen[r][c] = True
        nr, nc = r + dr[d], c + dc[d]
        if not (0 <= nr < R and 0 <= nc < C and not seen[nr][nc]):
            d = (d + 1) % 4
            nr, nc = r + dr[d], c + dc[d]
        r, c = nr, nc
    return out


def brute_C11(data):
    iv = data["intervals"]
    best = 0
    for s, _ in iv:
        best = max(best, sum(1 for a, b in iv if a <= s < b))
    return best


_OCT = r"(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
_IP = re.compile(r"%s(\.%s){3}" % (_OCT, _OCT), re.ASCII)


def brute_C12(data):
    return _IP.fullmatch(data["s"]) is not None


def brute_C13(data):
    a = list(data["nums"])
    if len(a) > 7:
        raise Infeasible("permutations")
    perms = sorted(set(itertools.permutations(a)))
    if not perms:
        return []
    i = perms.index(tuple(a))
    return list(perms[(i + 1) % len(perms)])


def brute_C14(data):
    nums, t = data["nums"], data["target"]
    for i, v in enumerate(nums):
        if v >= t:
            return i
    return len(nums)


def brute_C15(data):
    nums = data["nums"]
    if max(nums) > 3000 or len(nums) > 6:
        raise Infeasible("scan")
    g = next(d for d in range(min(nums), 0, -1) if all(v % d == 0 for v in nums))
    m = max(nums)
    k = m
    while not all(k % v == 0 for v in nums):
        k += m
        if k > 10 ** 8:
            raise Infeasible("lcm too large for scanning")
    return [g, str(k)]


def brute_C16(data):
    if not data["text"]:
        return []
    counts = {}
    for w in data["text"].split(" "):
        counts[w] = counts.get(w, 0) + 1
    out = []
    while counts and len(out) < data["k"]:
        top = max(counts.values())
        w = min(x for x, c in counts.items() if c == top)
        out.append(w)
        del counts[w]
    return out


def brute_C17(data):
    s = data["s"]
    if len(s) > 120:
        raise Infeasible("cubic")
    return sum(1 for i in range(len(s)) for j in range(i + 1, len(s) + 1) if s[i:j] == s[i:j][::-1])


def brute_C18(data):
    n, base = data["n"], data["base"]
    digs = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    p = 1
    while p * base <= n:
        p *= base
    out = []
    while p:
        q = 0
        while n >= p:
            n -= p
            q += 1
        out.append(digs[q])
        p //= base
    return "".join(out)


def brute_C19(data):
    """exact search: time-step BFS over (remaining counts, steps since each task type last ran)"""
    tasks, n = data["tasks"], data["n"]
    kinds = sorted(set(tasks))
    if len(tasks) > 8 or n > 4:
        raise Infeasible("state space")
    counts = tuple(tasks.count(k) for k in kinds)
    if not any(counts):
        return 0
    inf = n + 1                                 # 'ready' marker: n+1 steps since the last run
    start = (counts, tuple(inf for _ in kinds))
    frontier, t = {start}, 0
    while frontier:
        t += 1
        nxt = set()
        for rem, since in frontier:
            ready = [i for i in range(len(kinds)) if rem[i] > 0 and since[i] > n]
            options = ready if ready else [None]
            for pick in options:
                r2 = list(rem)
                s2 = [min(inf, x + 1) for x in since]
                if pick is not None:
                    r2[pick] -= 1
                    s2[pick] = 1
                if not any(r2):
                    return t
                nxt.add((tuple(r2), tuple(s2)))
        frontier = nxt
    raise AssertionError("unreachable")


def brute_C20(data):
    return [int(x) for x in re.findall(r"-?\d+", json.dumps(data["data"]))]


BRUTES = {"C%02d" % i: globals()["brute_C%02d" % i] for i in range(1, 21)}
