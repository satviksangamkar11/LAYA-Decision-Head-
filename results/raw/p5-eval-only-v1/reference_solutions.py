"""P5 evaluation-only reference solutions for tasks C01-C20, written from the task statements ONLY (configs/p5_contract_tasks_v1/tasks.jsonl). One `solve_<task_id>(data)` per task; `data` is the task's
input dictionary. Pure standard library. Not part of any runtime bundle: evaluation material."""
from collections import Counter
from math import gcd
from bisect import bisect_left


def solve_C01(data):
    """distinct integers covered by the union of closed ranges: sort and merge"""
    rs = sorted((a, b) for a, b in data["ranges"])
    total, cur_end = 0, None
    for a, b in rs:
        if cur_end is None or a > cur_end:
            total += b - a + 1
            cur_end = b
        elif b > cur_end:
            total += b - cur_end
            cur_end = b
    return total


def solve_C02(data):
    pairs = {")": "(", "]": "[", "}": "{"}
    st, depth = [], 0
    for ch in data["s"]:
        if ch in "([{":
            st.append(ch)
            depth = max(depth, len(st))
        else:
            if not st or st[-1] != pairs[ch]:
                return -1
            st.pop()
    return -1 if st else depth


def solve_C03(data):
    s = data["s"]
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str(j - i))
        i = j
    return "".join(out)


def solve_C04(data):
    m = data["m"]
    return [list(r) for r in zip(*m[::-1])] if m else []


def solve_C05(data):
    nums, target = data["nums"], data["target"]
    pos = {}
    for idx, v in enumerate(nums):
        pos.setdefault(v, []).append(idx)
    for i, v in enumerate(nums):
        lst = pos.get(target - v)
        if lst:
            k = bisect_left(lst, i + 1)
            if k < len(lst):
                return [i, lst[k]]
    return []


def solve_C06(data):
    c = Counter(data["nums"])
    return sorted(data["nums"], key=lambda v: (-c[v], v))


def solve_C07(data):
    words = data["words"]
    first = words[0]
    n = min(len(w) for w in words)
    k = 0
    while k < n and all(w[k] == first[k] for w in words):
        k += 1
    return first[:k]


def solve_C08(data):
    val = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s, total = data["s"], 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and val[ch] < val[s[i + 1]]:
            total -= val[ch]
        else:
            total += val[ch]
    return total


def solve_C09(data):
    n = data["n"]
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, sym in table:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def solve_C10(data):
    m = data["m"]
    if not m or not m[0]:
        return []
    top, bottom, left, right = 0, len(m) - 1, 0, len(m[0]) - 1
    out = []
    while top <= bottom and left <= right:
        out.extend(m[top][left:right + 1])
        top += 1
        out.extend(m[r][right] for r in range(top, bottom + 1))
        right -= 1
        if top <= bottom:
            out.extend(m[bottom][c] for c in range(right, left - 1, -1))
            bottom -= 1
        if left <= right:
            out.extend(m[r][left] for r in range(bottom, top - 1, -1))
            left += 1
    return out


def solve_C11(data):
    iv = data["intervals"]
    starts = sorted(a for a, b in iv)
    ends = sorted(b for a, b in iv)
    rooms = best = j = 0
    for s in starts:
        while j < len(ends) and ends[j] <= s:      # half-open: a meeting ending at s has freed its room
            j += 1
            rooms -= 1
        rooms += 1
        best = max(best, rooms)
    return best


def solve_C12(data):
    s = data["s"]
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p or any(ch not in "0123456789" for ch in p):
            return False
        if len(p) > 1 and p[0] == "0":
            return False
        if int(p) > 255:
            return False
    return True


def solve_C13(data):
    a = list(data["nums"])
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i >= 0:
        j = len(a) - 1
        while a[j] <= a[i]:
            j -= 1
        a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return a


def solve_C14(data):
    return bisect_left(data["nums"], data["target"])


def solve_C15(data):
    g = 0
    l = 1
    for v in data["nums"]:
        g = gcd(g, v)
        l = l * v // gcd(l, v)
    return [g, str(l)]


def solve_C16(data):
    words = data["text"].split(" ") if data["text"] else []
    c = Counter(words)
    return sorted(c, key=lambda w: (-c[w], w))[:data["k"]]


def solve_C17(data):
    s, count = data["s"], 0
    for center in range(2 * len(s) - 1):
        l, r = center // 2, center // 2 + center % 2
        while l >= 0 and r < len(s) and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
    return count


def solve_C18(data):
    n, base = data["n"], data["base"]
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(digits[r])
    return "".join(reversed(out))


def solve_C19(data):
    tasks, n = data["tasks"], data["n"]
    if not tasks:
        return 0
    c = Counter(tasks)
    mx = max(c.values())
    n_mx = sum(1 for v in c.values() if v == mx)
    return max(len(tasks), (mx - 1) * (n + 1) + n_mx)


def solve_C20(data):
    out = []

    def walk(x):
        if isinstance(x, list):
            for y in x:
                walk(y)
        else:
            out.append(x)
    walk(data["data"])
    return out


SOLVERS = {"C%02d" % i: globals()["solve_C%02d" % i] for i in range(1, 21)}
