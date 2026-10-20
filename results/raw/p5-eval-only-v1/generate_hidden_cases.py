"""Deterministically rebuilds hidden_cases.json and CHECK.json for the P5 contract tasks C01-C20 (evaluation-only material, written from the statements in configs/p5_contract_tasks_v1/tasks.jsonl only).

    python generate_hidden_cases.py <out dir>            # writes hidden_cases.json + CHECK.json; refuses to overwrite (create-only)
    python generate_hidden_cases.py <out dir> --verify   # rebuilds in memory and compares with the files already in <out dir> (byte identical or exit 1)

Per task (~30 cases): hand-written edge cases the statement permits + seeded random cases (seed 20261020, one random.Random per task). Expected values come from reference_solutions.py; every case
that the independent brute force can handle must agree with it (disagreements make the build fail: nothing is written). Public examples are excluded from the hidden cases. Property checks
(statement invariants such as 'decoding gives s back') run on every case. Only inputs the statements permit are generated."""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
TASKS = ROOT / "configs" / "p5_contract_tasks_v1" / "tasks.jsonl"
SEED = 20261020
TARGET = 30

import brute_force as BF                                    # noqa: E402
import reference_solutions as RS                            # noqa: E402


def _rng(tid):
    return random.Random("%d|%s" % (SEED, tid))


def _ri(r, lo, hi, n):
    return [r.randint(lo, hi) for _ in range(n)]


# ------------------------------------------------------------------ inputs per task: (edge inputs, random generator(rng) -> input)
def c01():
    edge = [{"ranges": [[5, 5]]}, {"ranges": [[1, 2], [3, 4]]}, {"ranges": [[1, 2], [1, 2]]}, {"ranges": [[-5, 5], [-2, 2]]}, {"ranges": [[0, 0], [0, 0], [0, 0]]},
            {"ranges": [[10, 20], [1, 5], [3, 12]]}, {"ranges": [[-10 ** 9, 10 ** 9]]}, {"ranges": [[-10 ** 9, -10 ** 9 + 4], [10 ** 9 - 4, 10 ** 9]]}, {"ranges": [[1, 1], [2, 2], [3, 3], [5, 5]]},
            {"ranges": [[i * 3, i * 3 + 1] for i in range(200)]}, {"ranges": [[3, 9], [1, 2], [4, 6], [8, 12], [20, 21]]}]

    def rnd(r):
        n = r.randint(1, 8)
        out = []
        for _ in range(n):
            a = r.randint(-20, 20)
            out.append([a, a + r.randint(0, 12)])
        return {"ranges": out}
    return edge, rnd


def c02():
    def bal(r, pairs):
        s = ""
        for _ in range(pairs):
            o = r.choice("([{")
            s += o + bal(r, 0) if False else o
        return s
    edge = [{"s": "()"}, {"s": "("}, {"s": ")"}, {"s": "([)]"}, {"s": "{[()]}"}, {"s": "(" * 500 + ")" * 500}, {"s": "()" * 300}, {"s": "}{"}, {"s": "(()"}, {"s": "())"}, {"s": "[](){}"},
            {"s": "{[(" + ")]}" + "()"}, {"s": "((()))(())()"}, {"s": "[" * 3}, {"s": "(" * 100 + "[" * 100 + "]" * 100 + ")" * 100}]

    def gen_bal(r, depth_budget):
        out, st = [], []
        for _ in range(r.randint(0, 14)):
            if st and r.random() < 0.5:
                out.append({"(": ")", "[": "]", "{": "}"}[st.pop()])
            else:
                c = r.choice("([{")
                st.append(c)
                out.append(c)
        while st:
            out.append({"(": ")", "[": "]", "{": "}"}[st.pop()])
        return "".join(out)

    def rnd(r):
        k = r.random()
        if k < 0.4:
            return {"s": gen_bal(r, 5)}
        if k < 0.7:
            s = list(gen_bal(r, 5) or "()")
            s[r.randrange(len(s))] = r.choice("()[]{}")
            return {"s": "".join(s)}
        return {"s": "".join(r.choice("()[]{}") for _ in range(r.randint(0, 10)))}
    return edge, rnd


def c03():
    edge = [{"s": "a"}, {"s": "aa"}, {"s": "ab"}, {"s": "a" * 100}, {"s": "a" * 10 + "b" * 10}, {"s": "abab"}, {"s": "abcdefghijklmnopqrstuvwxyz"}, {"s": "z" * 1000}, {"s": "aabbbcccc"},
            {"s": "a" * 9 + "b" + "a" * 10}, {"s": "q" * 99 + "r" * 101}]

    def rnd(r):
        return {"s": "".join(r.choice("abc") for _ in range(r.randint(0, 30)))}
    return edge, rnd


def c04():
    def mat(r, n, lo=-100, hi=100):
        return [[r.randint(lo, hi) for _ in range(n)] for _ in range(n)]
    r0 = random.Random("%d|C04edge" % SEED)
    edge = [{"m": [[-7]]}, {"m": [[0, 0], [0, 0]]}, {"m": [[1, 1], [1, 1]]}, {"m": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]}, {"m": mat(r0, 5, 0, 2)}, {"m": mat(r0, 30)},
            {"m": [[10 ** 9, -10 ** 9], [0, 5]]}]

    def rnd(r):
        return {"m": mat(r, r.randint(1, 8), -9, 9)}
    return edge, rnd


def c05():
    edge = [{"nums": [5], "target": 10}, {"nums": [0, 0], "target": 0}, {"nums": [-3, 4, 3, 90], "target": 0}, {"nums": [1, 2, 3, 4, 5], "target": 9}, {"nums": [1, 5, 1, 5], "target": 6},
            {"nums": [2, 2, 2, 2], "target": 4}, {"nums": [-1, -2, -3, -4], "target": -6}, {"nums": [7, 1, 2, 3, 4], "target": 8}, {"nums": list(range(300)), "target": 597}, {"nums": [10 ** 9, -10 ** 9, 5], "target": 0},
            {"nums": [3, 2, 4], "target": 6}, {"nums": [1, 1, 1, 1], "target": 3}, {"nums": [4, 4], "target": 8}, {"nums": [0], "target": 0}, {"nums": [1, 2, 3], "target": 100}]

    def rnd(r):
        n = r.randint(0, 12)
        return {"nums": _ri(r, -6, 6, n), "target": r.randint(-8, 8)}
    return edge, rnd


def c06():
    edge = [{"nums": [3, 3, 3]}, {"nums": [1, 2, 3, 4]}, {"nums": [-1, -1, -2, -2, 0]}, {"nums": [5, 4, 3, 2, 1, 5, 4, 3, 2, 5, 4, 3, 5, 4, 5]}, {"nums": [0]}, {"nums": [2, 1, 2, 1]}, {"nums": [10 ** 9, 10 ** 9, -10 ** 9]},
            {"nums": [7] * 50 + [8] * 50}, {"nums": list(range(40, 0, -1))}, {"nums": [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]}]
    r0 = random.Random("%d|C06edge" % SEED)
    edge.append({"nums": _ri(r0, -3, 3, 500)})

    def rnd(r):
        return {"nums": _ri(r, -5, 5, r.randint(0, 25))}
    return edge, rnd


def c07():
    edge = [{"words": [""]}, {"words": ["abc", "abc"]}, {"words": ["ab", "abc", "abcd"]}, {"words": ["a", "a", "b"]}, {"words": ["abc", "", "abc"]}, {"words": ["a" * 1000, "a" * 999 + "b"]},
            {"words": ["interview", "internet", "interval", "internal"]}, {"words": ["x"]}, {"words": ["", ""]}, {"words": ["abc", "ab", "a", ""]}, {"words": ["car", "cart", "carbon", "care"]}, {"words": ["b", "a"]}]

    def rnd(r):
        pre = "".join(r.choice("ab") for _ in range(r.randint(0, 4)))
        return {"words": [pre + "".join(r.choice("abc") for _ in range(r.randint(0, 4))) for _ in range(r.randint(1, 6))]}
    return edge, rnd


def _roman(n):
    return BF.brute_C09({"n": n})


def c08():
    nums = list(range(1, 21)) + [40, 49, 90, 99, 400, 444, 900, 999, 1000, 1994, 2024, 3888, 3999, 1666, 2999, 3000]
    edge = [{"s": _roman(n)} for n in nums]

    def rnd(r):
        return {"s": _roman(r.randint(1, 3999))}
    return edge, rnd


def c09():
    edge = [{"n": n} for n in list(range(1, 11)) + [14, 19, 40, 49, 90, 99, 400, 900, 999, 1000, 2000, 3000, 3888, 3999, 1666, 2024]]

    def rnd(r):
        return {"n": r.randint(1, 3999)}
    return edge, rnd


def c10():
    r0 = random.Random("%d|C10edge" % SEED)

    def mat(r, R, C, lo=-50, hi=50):
        return [[r.randint(lo, hi) for _ in range(C)] for _ in range(R)]
    edge = [{"m": mat(r0, 1, 1)}, {"m": mat(r0, 1, 7)}, {"m": mat(r0, 7, 1)}, {"m": mat(r0, 2, 2)}, {"m": mat(r0, 2, 3)}, {"m": mat(r0, 3, 2)}, {"m": mat(r0, 3, 4)}, {"m": mat(r0, 4, 3)}, {"m": mat(r0, 5, 5)},
            {"m": mat(r0, 10, 3)}, {"m": mat(r0, 3, 10)}, {"m": [[1, 1], [1, 1]]}, {"m": mat(r0, 12, 12, 0, 1)}, {"m": mat(r0, 1, 1, 0, 0)}]

    def rnd(r):
        return {"m": mat(r, r.randint(1, 9), r.randint(1, 9), -9, 9)}
    return edge, rnd


def c11():
    r0 = random.Random("%d|C11edge" % SEED)
    big = []
    for _ in range(300):
        a = r0.randint(0, 1000)
        big.append([a, a + r0.randint(1, 50)])
    edge = [{"intervals": [[1, 2]]}, {"intervals": [[1, 2], [2, 3], [3, 4]]}, {"intervals": [[0, 5]] * 7}, {"intervals": [[1, 10], [2, 9], [3, 8], [4, 7]]}, {"intervals": [[1, 2], [3, 4], [5, 6]]},
            {"intervals": [[-10, -5], [-7, -3], [-4, 0]]}, {"intervals": [[0, 30], [5, 10], [15, 20], [10, 15]]}, {"intervals": big}, {"intervals": [[1, 4], [2, 3], [3, 5], [4, 6]]},
            {"intervals": [[0, 10 ** 9], [10 ** 9 - 1, 10 ** 9 + 5]]}, {"intervals": [[1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}]

    def rnd(r):
        out = []
        for _ in range(r.randint(0, 10)):
            a = r.randint(-5, 15)
            out.append([a, a + r.randint(1, 8)])
        return {"intervals": out}
    return edge, rnd


def c12():
    strs = ["0.0.0.0", "255.255.255.255", "1.1.1.1", "256.0.0.0", "1.2.3.4.5", "1..2.3", "a.b.c.d", "1.2.3.4 ", " 1.2.3.4", "1.2.3.-4", "00.0.0.0", "0.00.0.0", "1.2.3.04", "", "...", "1.2.3.", ".1.2.3",
            "999.999.999.999", "1.2.3.4a", "192.168.1.256", "255.255.255.256", "1.2.3.255", "0.1.2.3", "1.2.3.0", "1,2,3,4", "1.2.3.4\n", "+1.2.3.4", "1.2.3.4.", "127.0.0.1", "10.0.0.255", "1. 2.3.4", "12345.1.1.1", "249.250.251.252"]
    edge = [{"s": s} for s in strs]

    def rnd(r):
        k = r.random()
        if k < 0.5:
            parts = [str(r.randint(0, 300)) for _ in range(4)]
            if r.random() < 0.3:
                i = r.randrange(4)
                parts[i] = "0" + parts[i]
            return {"s": ".".join(parts)}
        if k < 0.7:
            return {"s": ".".join(str(r.randint(0, 255)) for _ in range(r.choice([3, 4, 4, 5])))}
        return {"s": "".join(r.choice("0123456789..") for _ in range(r.randint(0, 15)))}
    return edge, rnd


def c13():
    edge = [{"nums": [1]}, {"nums": [1, 1]}, {"nums": [2, 1]}, {"nums": [1, 3, 2]}, {"nums": [2, 3, 1, 3, 3]}, {"nums": [5, 4, 3, 2, 1]}, {"nums": [1, 5, 1]}, {"nums": [0, 0, 0]}, {"nums": [-1, 0, -2]}, {"nums": [1, 2, 3, 4, 5, 6, 7]},
            {"nums": [7, 6, 5, 4, 3, 2, 1]}, {"nums": [1, 2, 2, 3]}, {"nums": [3, 3, 2, 2, 1, 1]}, {"nums": list(range(1, 21))}, {"nums": [4, 3, 2, 1] * 3}, {"nums": [-5, -5, -4]}]

    def rnd(r):
        return {"nums": _ri(r, 0, 3, r.randint(0, 7))}
    return edge, rnd


def c14():
    r0 = random.Random("%d|C14edge" % SEED)
    big = sorted(r0.sample(range(-5000, 5000), 1000))
    edge = [{"nums": [5], "target": 5}, {"nums": [5], "target": 4}, {"nums": [5], "target": 6}, {"nums": [1, 3, 5, 6], "target": 0}, {"nums": [1, 3, 5, 6], "target": 1}, {"nums": [1, 3, 5, 6], "target": 6},
            {"nums": [-5, -3, 0, 2], "target": -4}, {"nums": [-5, -3, 0, 2], "target": -6}, {"nums": big, "target": big[0]}, {"nums": big, "target": big[-1]}, {"nums": big, "target": big[500] + 0},
            {"nums": big, "target": 6000}, {"nums": big, "target": -6000}, {"nums": [10 ** 9], "target": -10 ** 9}, {"nums": list(range(0, 100, 2)), "target": 51}]

    def rnd(r):
        n = r.randint(0, 12)
        nums = sorted(r.sample(range(-20, 21), n))
        return {"nums": nums, "target": r.randint(-22, 22)}
    return edge, rnd


def c15():
    r0 = random.Random("%d|C15edge" % SEED)
    edge = [{"nums": [1]}, {"nums": [1000000]}, {"nums": [1000000, 1000000]}, {"nums": [999983, 999979]}, {"nums": [2, 3, 5, 7, 11, 13]}, {"nums": [1, 1, 1]}, {"nums": [6, 6, 6]}, {"nums": [1, 1000000]},
            {"nums": [2 ** i for i in range(0, 20)]}, {"nums": _ri(r0, 1, 1000000, 60)}, {"nums": [720720, 360360, 240240]}, {"nums": [12, 18]}, {"nums": [7, 14, 21, 28]}, {"nums": [999999, 1000000]},
            {"nums": [30, 42, 70, 105]}, {"nums": [8, 9]}]

    def rnd(r):
        return {"nums": _ri(r, 1, r.choice([12, 60, 1000]), r.randint(1, 5))}
    return edge, rnd


def c16():
    r0 = random.Random("%d|C16edge" % SEED)
    vocab = ["a", "b", "c", "d", "e", "f", "g"]
    edge = [{"text": "", "k": 1}, {"text": "a", "k": 3}, {"text": "b a", "k": 2}, {"text": "a a a b b c", "k": 2}, {"text": "c b a", "k": 1}, {"text": "x y z", "k": 3}, {"text": "q q q q", "k": 10},
            {"text": "b b a a c", "k": 2}, {"text": " ".join(r0.choice(vocab) for _ in range(500)), "k": 3}, {"text": "apple banana apple cherry banana apple", "k": 2}, {"text": "z", "k": 1}, {"text": "the day is sunny the the the sunny is is", "k": 4},
            {"text": "aa a aa a b", "k": 3}, {"text": "m n o p q r s t", "k": 8}, {"text": "m n o p q r s t", "k": 100}]

    def rnd(r):
        return {"text": " ".join(r.choice(vocab) for _ in range(r.randint(0, 40))), "k": r.randint(1, 8)}
    return edge, rnd


def c17():
    r0 = random.Random("%d|C17edge" % SEED)
    edge = [{"s": "a"}, {"s": "aa"}, {"s": "ab"}, {"s": "aba"}, {"s": "aaaa"}, {"s": "abcba"}, {"s": "a" * 50}, {"s": "abab"}, {"s": "racecar"}, {"s": "abacabadabacaba"}, {"s": "".join(r0.choice("ab") for _ in range(200))},
            {"s": "xyz" * 20}, {"s": "ab" * 60}, {"s": "z"}, {"s": "aabbaa"}]

    def rnd(r):
        return {"s": "".join(r.choice("abc") for _ in range(r.randint(0, 20)))}
    return edge, rnd


def c18():
    edge = [{"n": 0, "base": b} for b in (2, 10, 36)] + [{"n": 1, "base": 2}, {"n": 1, "base": 36}, {"n": 35, "base": 36}, {"n": 36, "base": 36}, {"n": 37, "base": 36}, {"n": 255, "base": 2}, {"n": 256, "base": 2},
            {"n": 10 ** 18, "base": 36}, {"n": 10 ** 18, "base": 2}, {"n": 2 ** 64, "base": 16}, {"n": 2 ** 64 - 1, "base": 16}, {"n": 100, "base": 10}, {"n": 35, "base": 16}, {"n": 7, "base": 7}, {"n": 48, "base": 7},
            {"n": 12345, "base": 8}, {"n": 999999, "base": 36}, {"n": 3, "base": 3}, {"n": 10, "base": 11}, {"n": 36 ** 5, "base": 36}, {"n": 36 ** 5 - 1, "base": 36}]

    def rnd(r):
        return {"n": r.randint(0, r.choice([50, 5000, 10 ** 9, 10 ** 15])), "base": r.randint(2, 36)}
    return edge, rnd


def c19():
    r0 = random.Random("%d|C19edge" % SEED)
    edge = [{"tasks": ["A"], "n": 0}, {"tasks": ["A"], "n": 5}, {"tasks": ["A"] * 5, "n": 2}, {"tasks": ["A", "B", "C", "D"], "n": 3}, {"tasks": ["A", "A"], "n": 100}, {"tasks": ["A", "A", "B", "B"], "n": 1},
            {"tasks": ["A", "A", "A", "B", "B", "B", "C", "C", "D"], "n": 2}, {"tasks": ["A", "A", "B", "B", "C", "C"], "n": 2}, {"tasks": ["A"] * 4 + ["B"] * 4, "n": 0}, {"tasks": ["A", "B", "A"], "n": 1},
            {"tasks": ["A", "A", "A", "A", "B", "C"], "n": 3}, {"tasks": [r0.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(60)], "n": 4}, {"tasks": [r0.choice("ABC") for _ in range(100)], "n": 10},
            {"tasks": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "n": 50}, {"tasks": ["A"] * 30 + ["B"] * 29, "n": 1}]

    def rnd(r):
        return {"tasks": [r.choice("ABCD") for _ in range(r.randint(0, 7))], "n": r.randint(0, 3)}
    return edge, rnd


def c20():
    def deep(k, leaf):
        x = leaf
        for _ in range(k):
            x = [x]
        return x

    def nest(r, depth):
        if depth <= 0 or r.random() < 0.4:
            return r.randint(-9, 9)
        return [nest(r, depth - 1) for _ in range(r.randint(0, 4))]
    edge = [{"data": 0}, {"data": -3}, {"data": [[]]}, {"data": [[[]]]}, {"data": [1]}, {"data": [[1]]}, {"data": [[1, [2]], 3]}, {"data": deep(100, 1)}, {"data": [[], [], []]}, {"data": [-1, [-2, [-3]]]},
            {"data": [10 ** 12, [-10 ** 12]]}, {"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, {"data": [[[1, 2], [3]], [[4]], 5]}, {"data": [[0], [0, [0, [0]]]]}, {"data": 10 ** 9}, {"data": [list(range(50)), [list(range(50))]]}]

    def rnd(r):
        x = nest(r, 4)
        return {"data": x}
    return edge, rnd


GENS = {"C%02d" % i: globals()["c%02d" % i] for i in range(1, 21)}


# ------------------------------------------------------------------ statement properties (every case)
def _from_roman(s):
    return BF.brute_C08({"s": s})


def props(tid, inp, out):
    d = inp
    if tid == "C01":
        L = [b - a + 1 for a, b in d["ranges"]]
        return out >= max(L) and out <= sum(L)
    if tid == "C02":
        return out == -1 or out <= len(d["s"]) // 2
    if tid == "C03":
        import re
        return "".join(ch * int(n) for ch, n in re.findall(r"([a-z])(\d+)", out)) == d["s"]
    if tid == "C04":
        return len(out) == len(d["m"]) and sorted(v for r in out for v in r) == sorted(v for r in d["m"] for v in r)
    if tid == "C05":
        return out == [] or (out[0] < out[1] and d["nums"][out[0]] + d["nums"][out[1]] == d["target"])
    if tid == "C06":
        return sorted(out) == sorted(d["nums"])
    if tid == "C07":
        return all(w.startswith(out) for w in d["words"])
    if tid == "C08":
        return 1 <= out <= 3999 and _roman(out) == d["s"]
    if tid == "C09":
        return _from_roman(out) == d["n"] and set(out) <= set("IVXLCDM")
    if tid == "C10":
        return len(out) == sum(len(r) for r in d["m"]) and sorted(out) == sorted(v for r in d["m"] for v in r)
    if tid == "C11":
        return 0 <= out <= len(d["intervals"]) and (out >= 1 or not d["intervals"])
    if tid == "C12":
        return isinstance(out, bool)
    if tid == "C13":
        return sorted(out) == sorted(d["nums"])
    if tid == "C14":
        i, n, t = out, d["nums"], d["target"]
        return 0 <= i <= len(n) and (i == 0 or n[i - 1] < t) and (i == len(n) or t <= n[i])
    if tid == "C15":
        g, l = out[0], int(out[1])
        return all(v % g == 0 and l % v == 0 for v in d["nums"]) and isinstance(out[1], str)
    if tid == "C16":
        words = d["text"].split(" ") if d["text"] else []
        return len(out) == len(set(out)) and all(w in words for w in out) and len(out) == min(d["k"], len(set(words)))
    if tid == "C17":
        n = len(d["s"])
        return n <= out <= n * (n + 1) // 2
    if tid == "C18":
        return int(out, d["base"]) == d["n"] and (out == "0" or not out.startswith("0")) and out == out.lower()
    if tid == "C19":
        return out >= len(d["tasks"]) and (d["n"] != 0 or out == len(d["tasks"]))
    if tid == "C20":
        flat = BF.brute_C20(d)
        return out == flat and sum(out) == sum(flat)
    return True


def build():
    pubs = {}
    for l in TASKS.read_text("utf-8").splitlines():
        if l.strip():
            t = json.loads(l)
            pubs[t["task_id"]] = t["public_tests"]
    hidden, check = {}, {}
    for tid in sorted(GENS):
        edge, rnd = GENS[tid]()
        rng = _rng(tid)
        seen = {json.dumps(p["input"], sort_keys=True) for p in pubs[tid]}
        cases, n_edge = [], 0
        for inp in edge:
            k = json.dumps(inp, sort_keys=True)
            if k not in seen:
                seen.add(k)
                cases.append(inp)
                n_edge += 1
        tries = 0
        while len(cases) < TARGET and tries < 2000:
            tries += 1
            inp = rnd(rng)
            k = json.dumps(inp, sort_keys=True)
            if k not in seen:
                seen.add(k)
                cases.append(inp)
        ref, bf = RS.SOLVERS[tid], BF.BRUTES[tid]
        pub_ref = all(ref(json.loads(json.dumps(p["input"]))) == p["expected"] for p in pubs[tid])
        pub_bf = []
        for p in pubs[tid]:
            try:
                pub_bf.append(bf(json.loads(json.dumps(p["input"]))) == p["expected"])
            except BF.Infeasible:
                pub_bf.append(None)
        agree = skip = dis = prop_fail = 0
        out_cases, disagreements = [], []
        for inp in cases:
            exp = ref(json.loads(json.dumps(inp)))
            if not props(tid, inp, exp):
                prop_fail += 1
            try:
                b = bf(json.loads(json.dumps(inp)))
                if b == exp:
                    agree += 1
                else:
                    dis += 1
                    disagreements.append({"input": inp, "reference": exp, "brute": b})
            except BF.Infeasible:
                skip += 1
            out_cases.append({"input": inp, "expected": exp})
        hidden[tid] = out_cases
        check[tid] = {"public_examples": len(pubs[tid]), "public_reproduced_by_reference": pub_ref, "public_reproduced_by_brute_force": (None if None in pub_bf else all(pub_bf)),
                      "cases": len(out_cases), "edge_cases": n_edge, "random_cases": len(out_cases) - n_edge, "reference_vs_brute_agree": agree, "reference_vs_brute_disagree": dis,
                      "brute_force_infeasible_skipped": skip, "statement_property_failures": prop_fail, "disagreements": disagreements[:3]}
    return hidden, check


def dumps(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, indent=1) + "\n"


def main(argv):
    out = Path(argv[1])
    hidden, check = build()
    bad = {t: c for t, c in check.items() if not c["public_reproduced_by_reference"] or c["reference_vs_brute_disagree"] or c["statement_property_failures"]}
    check_doc = {"seed": SEED, "target_cases_per_task": TARGET, "source_tasks": "configs/p5_contract_tasks_v1/tasks.jsonl (contract_props.py / contracts/ / REGISTRY_v1.json were NOT opened)", "tasks": check,
                 "all_ok": not bad, "hidden_cases_sha256": hashlib.sha256(dumps(hidden).encode("utf-8")).hexdigest()}
    if "--verify" in argv:
        same = (out / "hidden_cases.json").read_text("utf-8") == dumps(hidden) and json.loads((out / "CHECK.json").read_text("utf-8")) == json.loads(dumps(check_doc))
        print("VERIFY", "identical" if same else "DIFFERENT")
        return 0 if same else 1
    if bad:
        print("BUILD REFUSED, problems in:", sorted(bad))
        print(json.dumps(bad, indent=1)[:3000])
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, txt in (("hidden_cases.json", dumps(hidden)), ("CHECK.json", dumps(check_doc))):
        p = out / name
        if p.exists():
            print("refusing to overwrite", p)
            return 2
        p.write_text(txt, encoding="utf-8")
    print("written", out, "all_ok", check_doc["all_ok"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
