"""[P3] Specification-grounded runtime checks, v1. Written 2026-10-20 from the TASK TEXT and PUBLIC EXAMPLES ONLY (no development-grader case was read).
Per task:
  valid(inp)  -> True (permitted by the task text) | False (not permitted) | None (uncertain: never used as hard evidence)
  edges       -> extra inputs that the task text explicitly permits (no expected values unless the text states the output outright)
  props(inp, out) -> list of (name, ok) for properties the task text states (or that follow by elementary mathematics from it); ok None = not applicable
Hard-failure evidence = a crash on a valid input, or a stated property violated on a valid input. Anything uncertain is skipped."""
import re

INT = lambda x: isinstance(x, int) and not isinstance(x, bool)
STR = lambda x: isinstance(x, str)
LIST = lambda x: isinstance(x, list)
BOOL = lambda x: isinstance(x, bool)
LOWER = re.compile(r"^[a-z]*$")
DIGITS = re.compile(r"^[0-9]+$")


def _ok(cond):
    return bool(cond)


def ints(xs):
    return LIST(xs) and all(INT(v) for v in xs)


SPECS = {}


def spec(tid):
    def deco(cls):
        SPECS[tid] = cls()
        return cls
    return deco


class Base:
    edges = []

    def valid(self, inp):
        return None

    def props(self, inp, out):
        return []


@spec("J1-17")
class _J117(Base):      # n 1..10000, k 1..1000; survivor number
    edges = [{"n": 10000, "k": 1000}, {"n": 2, "k": 1}, {"n": 2, "k": 2}, {"n": 1, "k": 1000}]
    def valid(self, i): return _ok(set(i) == {"n", "k"} and INT(i["n"]) and INT(i["k"]) and 1 <= i["n"] <= 10000 and 1 <= i["k"] <= 1000)
    def props(self, i, o): return [("int_in_1_n", INT(o) and 1 <= o <= i["n"])]


@spec("J1-20")
class _J120(Base):      # boolean per query in order
    edges = [{"words": ["a"], "queries": [{"type": "prefix", "word": ""}]}, {"words": [], "queries": []}, {"words": ["abc"], "queries": [{"type": "exact", "word": ""}]}]
    def valid(self, i):
        if set(i) != {"words", "queries"} or not LIST(i["words"]) or not LIST(i["queries"]): return False
        if not all(STR(w) and LOWER.match(w) for w in i["words"]): return False
        return _ok(all(isinstance(q, dict) and set(q) == {"type", "word"} and q["type"] in ("exact", "prefix") and STR(q["word"]) and LOWER.match(q["word"]) for q in i["queries"]))
    def props(self, i, o): return [("bool_list_len_queries", LIST(o) and len(o) == len(i["queries"]) and all(BOOL(v) for v in o))]


@spec("J1-31")
class _J131(Base):      # unsigned decimal strings 1..40 digits, leading zeros allowed; normalized product
    edges = [{"a": "9" * 40, "b": "9" * 40}, {"a": "0" * 40, "b": "1"}, {"a": "1" + "0" * 39, "b": "1" + "0" * 39}, {"a": "00000", "b": "0"}]
    def valid(self, i): return _ok(set(i) == {"a", "b"} and all(STR(i[k]) and DIGITS.match(i[k]) and 1 <= len(i[k]) <= 40 for k in "ab"))
    def props(self, i, o):
        norm = STR(o) and DIGITS.match(o or "x") is not None and (o == "0" or not o.startswith("0"))
        zero = (int(i["a"]) == 0 or int(i["b"]) == 0)
        return [("normalized_decimal", bool(norm)), ("zero_iff_factor_zero", (o == "0") == zero if norm else None)]


@spec("P1-13")
class _P113(Base):      # length of the longest substring without repeats
    edges = [{"text": "a"}, {"text": "aA"}, {"text": "abcdefghijklmnopqrstuvwxyz" * 4}]
    def valid(self, i): return _ok(set(i) == {"text"} and STR(i["text"]))
    def props(self, i, o): return [("int_0_len", INT(o) and 0 <= o <= len(i["text"])), ("nonempty_gives_at_least_1", (o >= 1) if (INT(o) and i["text"]) else None)]


@spec("P1-14")
class _P114(Base):      # group anagrams; every word appears exactly once in the groups
    edges = [{"words": [""]}, {"words": ["", ""]}, {"words": ["", "a", ""]}]
    def valid(self, i): return _ok(set(i) == {"words"} and LIST(i["words"]) and all(STR(w) and LOWER.match(w) for w in i["words"]))
    def props(self, i, o):
        shape = LIST(o) and all(LIST(g) and g and all(STR(w) for w in g) for g in o)
        return [("groups_shape", shape), ("multiset_preserved", sorted(w for g in o for w in g) == sorted(i["words"]) if shape else None),
                ("groups_are_anagram_classes", all(len({"".join(sorted(w)) for w in g}) == 1 for g in o) and len({"".join(sorted(g[0])) for g in o}) == len(o) if shape else None)]


@spec("P1-15")
class _P115(Base):      # sorted union of closed intervals; shared endpoints merge, adjacency does not
    edges = [{"intervals": [[5, 5]]}, {"intervals": [[-3, -1], [-1, 0]]}, {"intervals": [[1, 10], [2, 3], [4, 5]]}]
    def valid(self, i): return _ok(set(i) == {"intervals"} and LIST(i["intervals"]) and all(ints(x) and len(x) == 2 and x[0] <= x[1] for x in i["intervals"]))
    def props(self, i, o):
        shape = LIST(o) and all(ints(x) and len(x) == 2 and x[0] <= x[1] for x in o)
        return [("interval_shape", shape), ("sorted_and_separated", all(o[j][1] < o[j + 1][0] for j in range(len(o) - 1)) if shape else None),
                ("empty_iff_empty", (len(o) == 0) == (len(i["intervals"]) == 0) if shape else None)]


@spec("P1-16")
class _P116(Base):      # count 4-connected components of '1'
    edges = [{"grid": ["11", "11"]}, {"grid": ["0000"]}, {"grid": ["1", "0", "1"]}]
    def valid(self, i):
        g = i.get("grid") if set(i) == {"grid"} else None
        if not LIST(g): return False
        if not g: return True
        if not all(STR(r) and set(r) <= {"0", "1"} for r in g) or len({len(r) for r in g}) != 1: return False
        return None if len(g[0]) == 0 else True          # zero-width rows: not settled by the text
    def props(self, i, o): return [("int_nonneg", INT(o) and o >= 0), ("zero_if_no_ones", (o == 0) if (INT(o) and not any("1" in r for r in i["grid"])) else None)]


@spec("P1-17")
class _P117(Base):      # BFS shortest edge count or -1
    edges = [{"n": 2, "edges": [], "source": 0, "target": 1}, {"n": 2, "edges": [[0, 0]], "source": 0, "target": 1}, {"n": 3, "edges": [[0, 1], [0, 1], [1, 2]], "source": 2, "target": 2}]
    def valid(self, i):
        if set(i) != {"n", "edges", "source", "target"} or not INT(i["n"]) or i["n"] < 1: return False
        n = i["n"]
        return _ok(LIST(i["edges"]) and all(ints(e) and len(e) == 2 and 0 <= e[0] < n and 0 <= e[1] < n for e in i["edges"]) and INT(i["source"]) and INT(i["target"]) and 0 <= i["source"] < n and 0 <= i["target"] < n)
    def props(self, i, o): return [("int_ge_minus1", INT(o) and o >= -1), ("same_node_zero", (o == 0) if i["source"] == i["target"] else None)]


@spec("P1-18")
class _P118(Base):      # min coins or -1; amount 0 -> 0
    edges = [{"coins": [1], "amount": 500}, {"coins": [5], "amount": 0}, {"coins": [2, 5], "amount": 1}]
    def valid(self, i): return _ok(set(i) == {"coins", "amount"} and ints(i["coins"]) and all(c > 0 for c in i["coins"]) and INT(i["amount"]) and i["amount"] >= 0)
    def props(self, i, o): return [("int_ge_minus1", INT(o) and o >= -1), ("amount_zero_gives_zero", (o == 0) if i["amount"] == 0 else None)]


@spec("P1-19")
class _P119(Base):      # minimum window substring
    edges = [{"s": "a", "t": "aa"}, {"s": "ab", "t": "b"}, {"s": "", "t": ""}]
    def valid(self, i): return _ok(set(i) == {"s", "t"} and STR(i["s"]) and STR(i["t"]))
    def props(self, i, o):
        sub = STR(o) and (o == "" or o in i["s"])
        cover = None
        if STR(o) and o:
            from collections import Counter
            need, have = Counter(i["t"]), Counter(o)
            cover = all(have[c] >= k for c, k in need.items())
        return [("substring_of_s_or_empty", sub), ("covers_t", cover), ("empty_t_gives_empty", (o == "") if i["t"] == "" else None)]


@spec("P1-21")
class _P121(Base):      # wildcard ? and *, everything else literal
    edges = [{"text": "", "pattern": ""}, {"text": "[a]", "pattern": "[a]"}, {"text": "a.b", "pattern": "a.b"}, {"text": "axb", "pattern": "a.b"}, {"text": "", "pattern": "?"}]
    def valid(self, i): return _ok(set(i) == {"text", "pattern"} and STR(i["text"]) and STR(i["pattern"]))
    def props(self, i, o):
        lit = None if any(ch in i["pattern"] for ch in "?*") else (o == (i["text"] == i["pattern"]))      # no wildcard: literal equality is stated
        return [("bool", BOOL(o)), ("literal_pattern_equality", lit if BOOL(o) else None)]


@spec("P1-22")
class _P122(Base):      # max non-overlapping half-open intervals
    edges = [{"intervals": [[0, 1]]}, {"intervals": [[-5, -4], [-4, -3], [-3, -2]]}, {"intervals": [[1, 3], [1, 3], [1, 3]]}]
    def valid(self, i): return _ok(set(i) == {"intervals"} and LIST(i["intervals"]) and all(ints(x) and len(x) == 2 and x[0] < x[1] for x in i["intervals"]))
    def props(self, i, o): return [("int_0_len", INT(o) and 0 <= o <= len(i["intervals"])), ("nonempty_gives_at_least_1", (o >= 1) if (INT(o) and i["intervals"]) else None)]


@spec("P1-24")
class _P124(Base):      # Excel column number
    edges = [{"label": "AZ"}, {"label": "BA"}, {"label": "ZZZ"}, {"label": "XFD"}]
    def valid(self, i): return _ok(set(i) == {"label"} and STR(i["label"]) and re.match(r"^[A-Z]+$", i["label"]) is not None)
    def props(self, i, o): return [("int_ge_1", INT(o) and o >= 1)]


@spec("P1-25")
class _P125(Base):      # search in rotated strictly increasing list
    edges = [{"nums": [3, 1, 2], "target": 1}, {"nums": [2, 1], "target": 2}, {"nums": [5, 6, 7, 1, 2, 3, 4], "target": 4}]
    def valid(self, i):
        if set(i) != {"nums", "target"} or not ints(i["nums"]) or not INT(i["target"]): return False
        a = i["nums"]
        drops = sum(1 for j in range(len(a) - 1) if a[j] >= a[j + 1])
        return _ok(len(set(a)) == len(a) and (drops == 0 or (drops == 1 and a[-1] < a[0])))
    def props(self, i, o):
        rng = INT(o) and -1 <= o < len(i["nums"])
        return [("index_range", rng), ("index_points_at_target", (i["nums"][o] == i["target"]) if (rng and o >= 0) else None),
                ("minus1_iff_absent", (o == -1) == (i["target"] not in i["nums"]) if rng else None)]


@spec("P1-27")
class _P127(Base):      # smallest topological order or [] on a cycle
    edges = [{"n": 1, "edges": []}, {"n": 3, "edges": [[2, 0]]}, {"n": 12, "edges": [[11, 10]]}]
    def valid(self, i):
        if set(i) != {"n", "edges"} or not INT(i["n"]) or i["n"] < 0: return False
        n = i["n"]
        if not (LIST(i["edges"]) and all(ints(e) and len(e) == 2 and 0 <= e[0] < n and 0 <= e[1] < n for e in i["edges"])): return False
        return None if any(e[0] == e[1] for e in i["edges"]) else True          # self-loops: not settled by the text
    def props(self, i, o):
        n = i["n"]
        perm = LIST(o) and sorted(o) == list(range(n)) if LIST(o) and all(INT(v) for v in o) else False
        shape = LIST(o) and (o == [] or perm)
        pos = {v: k for k, v in enumerate(o)} if perm else {}
        return [("empty_or_permutation", bool(shape) if n > 0 else (o == [])),
                ("respects_edges", all(pos[u] < pos[v] for u, v in i["edges"]) if perm and o else None)]


@spec("P1-28")
class _P128(Base):      # BFS on grid; explicit -1 cases
    edges = [{"grid": [[]]}, {"grid": [[0, 0], [0, 1]]}, {"grid": [[0, 0, 0]]}]
    def valid(self, i):
        g = i.get("grid") if set(i) == {"grid"} else None
        if not LIST(g): return False
        if g == [] or g == [[]]: return True
        return _ok(all(ints(r) and set(r) <= {0, 1} for r in g) and len({len(r) for r in g}) == 1 and len(g[0]) > 0)
    def props(self, i, o):
        g = i["grid"]
        empty = g == [] or all(len(r) == 0 for r in g)
        blocked = (not empty) and (g[0][0] == 1 or g[-1][-1] == 1)
        return [("int_ge_minus1", INT(o) and o >= -1), ("stated_minus1_cases", (o == -1) if (empty or blocked) else None)]


@spec("P1-29")
class _P129(Base):      # kth largest
    edges = [{"nums": [3, 3, 3], "k": 3}, {"nums": [-1, -2], "k": 1}, {"nums": [2, 1], "k": 2}]
    def valid(self, i): return _ok(set(i) == {"nums", "k"} and ints(i["nums"]) and len(i["nums"]) >= 1 and INT(i["k"]) and 1 <= i["k"] <= len(i["nums"]))
    def props(self, i, o): return [("element_of_nums", INT(o) and o in i["nums"])]


@spec("P1-30")
class _P130(Base):      # daily temperatures, strictly warmer
    edges = [{"temps": [5, 5]}, {"temps": [5]}, {"temps": [1, 1, 2]}]
    def valid(self, i): return _ok(set(i) == {"temps"} and ints(i["temps"]))
    def props(self, i, o):
        t = i["temps"]
        shape = ints(o) and len(o) == len(t) and all(0 <= o[j] <= len(t) - 1 - j for j in range(len(t)))
        warm = all(o[j] == 0 or t[j + o[j]] > t[j] for j in range(len(t))) if shape else None
        return [("shape_and_range", shape), ("points_at_strictly_warmer", warm)]


@spec("P1-31")
class _P131(Base):      # edit distance
    edges = [{"a": "abc", "b": "abc"}, {"a": "", "b": "xyz"}, {"a": "a" * 50, "b": "b" * 50}]
    def valid(self, i): return _ok(set(i) == {"a", "b"} and STR(i["a"]) and STR(i["b"]))
    def props(self, i, o):
        a, b = i["a"], i["b"]
        return [("int_range", INT(o) and abs(len(a) - len(b)) <= o <= max(len(a), len(b))), ("zero_iff_equal", (o == 0) == (a == b) if INT(o) else None)]


@spec("P1-32")
class _P132(Base):      # LIS length
    edges = [{"nums": [5]}, {"nums": [3, 2, 1]}, {"nums": list(range(200))}]
    def valid(self, i): return _ok(set(i) == {"nums"} and ints(i["nums"]))
    def props(self, i, o):
        n = len(i["nums"])
        return [("int_range", INT(o) and (o == 0 if n == 0 else 1 <= o <= n))]


@spec("P1-33")
class _P133(Base):      # word break
    edges = [{"text": "a", "words": []}, {"text": "aaaa", "words": ["aaaa"]}, {"text": "ab", "words": ["a"]}]
    def valid(self, i): return _ok(set(i) == {"text", "words"} and STR(i["text"]) and LIST(i["words"]) and all(STR(w) and w for w in i["words"]))
    def props(self, i, o):
        t = i["text"]
        return [("bool", BOOL(o)), ("empty_text_true", (o is True) if t == "" else None),
                ("no_words_nonempty_text_false", (o is False) if (t and not i["words"]) else None),
                ("text_is_a_word_true", (o is True) if t in i["words"] else None)]


@spec("P1-34")
class _P134(Base):      # restricted regex . and *
    edges = [{"text": "a", "pattern": "a"}, {"text": "", "pattern": "a*"}, {"text": "b", "pattern": "a"}, {"text": "abc", "pattern": "a.c"}]
    def valid(self, i):
        if set(i) != {"text", "pattern"} or not STR(i["text"]) or not STR(i["pattern"]): return False
        p = i["pattern"]
        return _ok(not p.startswith("*") and "**" not in p)
    def props(self, i, o):
        p = i["pattern"]
        lit = None if any(ch in p for ch in ".*") else (o == (i["text"] == p))
        return [("bool", BOOL(o)), ("literal_pattern_equality", lit if BOOL(o) else None)]


@spec("P1-35")
class _P135(Base):      # trapping rain water
    edges = [{"heights": [5]}, {"heights": [3, 0, 3]}, {"heights": [0, 0, 0]}]
    def valid(self, i): return _ok(set(i) == {"heights"} and ints(i["heights"]) and all(h >= 0 for h in i["heights"]))
    def props(self, i, o):
        h = i["heights"]
        return [("int_nonneg", INT(o) and o >= 0), ("monotone_zero", (o == 0) if (h == sorted(h) or h == sorted(h, reverse=True)) else None)]


@spec("P1-36")
class _P136(Base):      # Dijkstra, nonnegative weights, null unreachable
    edges = [{"n": 2, "edges": [], "source": 1}, {"n": 3, "edges": [[0, 1, 5], [0, 1, 1]], "source": 0}, {"n": 2, "edges": [[1, 1, 3]], "source": 1}]
    def valid(self, i):
        if set(i) != {"n", "edges", "source"} or not INT(i["n"]) or i["n"] < 1: return False
        n = i["n"]
        return _ok(LIST(i["edges"]) and all(ints(e) and len(e) == 3 and 0 <= e[0] < n and 0 <= e[1] < n and e[2] >= 0 for e in i["edges"]) and INT(i["source"]) and 0 <= i["source"] < n)
    def props(self, i, o):
        shape = LIST(o) and len(o) == i["n"] and all(v is None or (INT(v) and v >= 0) for v in o)
        return [("shape", shape), ("source_zero", (o[i["source"]] == 0) if shape else None)]


@spec("P1-37")
class _P137(Base):      # min path sum right/down with obstacles; null if no path / empty
    edges = [{"grid": [[-1]]}, {"grid": [[5, 1, 1]]}, {"grid": [[1], [-1]]}]
    def valid(self, i):
        g = i.get("grid") if set(i) == {"grid"} else None
        if not LIST(g): return False
        if g == []: return True
        return _ok(all(ints(r) and all(v >= 0 or v == -1 for v in r) for r in g) and len({len(r) for r in g}) == 1 and len(g[0]) > 0)
    def props(self, i, o):
        g = i["grid"]
        dead = g == [] or g[0][0] == -1 or g[-1][-1] == -1
        return [("int_nonneg_or_null", o is None or (INT(o) and o >= 0)), ("stated_null_cases", (o is None) if dead else None)]


@spec("P1-38")
class _P138(Base):      # equal-sum partition
    edges = [{"nums": [0]}, {"nums": [1]}, {"nums": [2, 2]}, {"nums": [1] * 40}]
    def valid(self, i): return _ok(set(i) == {"nums"} and ints(i["nums"]) and all(v >= 0 for v in i["nums"]))
    def props(self, i, o): return [("bool", BOOL(o)), ("odd_total_false", (o is False) if sum(i["nums"]) % 2 == 1 else None)]


@spec("P1-39")
class _P139(Base):      # max product subarray
    edges = [{"nums": [0]}, {"nums": [-3, -4]}, {"nums": [2, -5, -2, -4, 3]}]
    def valid(self, i): return _ok(set(i) == {"nums"} and ints(i["nums"]) and len(i["nums"]) >= 1)
    def props(self, i, o): return [("at_least_max_element", INT(o) and o >= max(i["nums"]))]


@spec("P1-40")
class _P140(Base):      # number of SCCs
    edges = [{"n": 5, "edges": []}, {"n": 2, "edges": [[0, 0]]}, {"n": 3, "edges": [[0, 1], [1, 2], [2, 0]]}]
    def valid(self, i):
        if set(i) != {"n", "edges"} or not INT(i["n"]) or i["n"] < 0: return False
        n = i["n"]
        return _ok(LIST(i["edges"]) and all(ints(e) and len(e) == 2 and 0 <= e[0] < n and 0 <= e[1] < n for e in i["edges"]))
    def props(self, i, o):
        n = i["n"]
        return [("int_range", INT(o) and (o == 0 if n == 0 else 1 <= o <= n)), ("no_edges_n_components", (o == n) if not i["edges"] else None)]
