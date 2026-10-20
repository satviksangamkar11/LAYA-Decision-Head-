"""[P5 contract cohort] Author properties for C01-C20, written from each task's STATEMENT only, before any candidate exists and without any hidden test or reference
solution. Each property restates a sentence of the statement. props(inp, out) -> [(name, True | False | None)]; valid(inp) narrows validity beyond the schema."""
import re

INT = lambda x: isinstance(x, int) and not isinstance(x, bool)
SPECS = {}


def spec(tid):
    def deco(cls):
        SPECS[tid] = cls()
        return cls
    return deco


class Base:
    def valid(self, i): return True
    def props(self, i, o): return []


def _int_list(o): return isinstance(o, list) and all(INT(v) for v in o)


@spec("C01")
class _C01(Base):
    def valid(self, i): return all(a <= b for a, b in i["ranges"])
    def props(self, i, o):
        r = i["ranges"]; lo = max([b - a + 1 for a, b in r], default=0); hi = sum(b - a + 1 for a, b in r)
        return [("int_between_longest_and_sum", INT(o) and lo <= o <= hi)]


@spec("C02")
class _C02(Base):
    def props(self, i, o):
        return [("int", INT(o)), ("ge_minus1_le_half", INT(o) and -1 <= o <= len(i["s"]) // 2), ("empty_zero", (o == 0) if i["s"] == "" else None)]


@spec("C03")
class _C03(Base):
    def props(self, i, o):
        if not isinstance(o, str):
            return [("str", False)]
        parts = re.findall(r"([a-z])(\d+)", o)
        ok = "".join(c + n for c, n in parts) == o and all(not n.startswith("0") for _, n in parts)
        return [("str", True), ("well_formed", ok), ("decodes_to_s", ("".join(c * int(n) for c, n in parts) == i["s"]) if ok else None)]


@spec("C04")
class _C04(Base):
    def valid(self, i): return all(len(r) == len(i["m"]) for r in i["m"])
    def props(self, i, o):
        n = len(i["m"])
        shape = isinstance(o, list) and len(o) == n and all(_int_list(r) and len(r) == n for r in o)
        return [("square_same_size", shape), ("same_multiset", sorted(v for r in o for v in r) == sorted(v for r in i["m"] for v in r) if shape else None)]


@spec("C05")
class _C05(Base):
    def props(self, i, o):
        a, t = i["nums"], i["target"]
        shape = _int_list(o) and len(o) in (0, 2)
        pair = shape and len(o) == 2 and 0 <= o[0] < o[1] < len(a)
        return [("pair_or_empty", shape), ("valid_pair", (pair and a[o[0]] + a[o[1]] == t) if shape and o else None)]


@spec("C06")
class _C06(Base):
    def props(self, i, o):
        perm = _int_list(o) and sorted(o) == sorted(i["nums"])
        if not perm:
            return [("permutation", False)]
        from collections import Counter
        c = Counter(o); keys = [o[k] for k in range(len(o)) if k == 0 or o[k] != o[k - 1]]
        grouped = len(keys) == len(set(keys))
        ordered = all((c[keys[k]], -keys[k]) > (c[keys[k + 1]], -keys[k + 1]) for k in range(len(keys) - 1)) if grouped else None
        return [("permutation", True), ("equal_values_together", grouped), ("frequency_then_value_order", ordered)]


@spec("C07")
class _C07(Base):
    def valid(self, i): return len(i["words"]) >= 1
    def props(self, i, o):
        return [("str", isinstance(o, str)), ("prefix_of_every_word", all(w.startswith(o) for w in i["words"]) if isinstance(o, str) else None)]


_ROM = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _roman_value(s):
    m = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    return sum(-m[c] if k + 1 < len(s) and m[c] < m[s[k + 1]] else m[c] for k, c in enumerate(s))


@spec("C08")
class _C08(Base):
    def valid(self, i): return bool(i["s"]) and _ROM.match(i["s"]) is not None
    def props(self, i, o): return [("int_1_3999", INT(o) and 1 <= o <= 3999)]


@spec("C09")
class _C09(Base):
    def props(self, i, o):
        form = isinstance(o, str) and bool(o) and _ROM.match(o) is not None
        return [("standard_roman_letters", form), ("reads_back_to_n", (_roman_value(o) == i["n"]) if form else None)]


@spec("C10")
class _C10(Base):
    def valid(self, i): return len({len(r) for r in i["m"]}) <= 1
    def props(self, i, o):
        allv = [v for r in i["m"] for v in r]
        ok = _int_list(o) and sorted(o) == sorted(allv)
        return [("every_element_once", ok), ("starts_top_left", (o[0] == i["m"][0][0]) if ok and allv else None)]


@spec("C11")
class _C11(Base):
    def valid(self, i): return all(s < e for s, e in i["intervals"])
    def props(self, i, o):
        n = len(i["intervals"])
        return [("int_range", INT(o) and (o == 0 if n == 0 else 1 <= o <= n))]


@spec("C12")
class _C12(Base):
    def props(self, i, o):
        bad_char = any(ch not in "0123456789." for ch in i["s"])
        return [("bool", isinstance(o, bool)), ("other_chars_false", (o is False) if bad_char else None)]


@spec("C13")
class _C13(Base):
    def props(self, i, o): return [("permutation", _int_list(o) and sorted(o) == sorted(i["nums"]))]


@spec("C14")
class _C14(Base):
    def valid(self, i): return all(i["nums"][k] < i["nums"][k + 1] for k in range(len(i["nums"]) - 1))
    def props(self, i, o):
        a, t = i["nums"], i["target"]
        rng = INT(o) and 0 <= o <= len(a)
        return [("index_range", rng), ("insertion_position", ((o == 0 or a[o - 1] < t) and (o == len(a) or t <= a[o])) if rng else None)]


@spec("C15")
class _C15(Base):
    def valid(self, i): return len(i["nums"]) >= 1 and all(1 <= v <= 1000000 for v in i["nums"])
    def props(self, i, o):
        shape = isinstance(o, list) and len(o) == 2 and INT(o[0]) and o[0] >= 1 and isinstance(o[1], str) and re.fullmatch(r"[1-9][0-9]*", o[1] or "x") is not None
        return [("shape", shape), ("g_divides_all", all(v % o[0] == 0 for v in i["nums"]) if shape else None),
                ("all_divide_l", all(int(o[1]) % v == 0 for v in i["nums"]) if shape else None)]


@spec("C16")
class _C16(Base):
    def valid(self, i): return i["k"] >= 1 and (i["text"] == "" or re.fullmatch(r"[a-z]+( [a-z]+)*", i["text"]) is not None)
    def props(self, i, o):
        words = i["text"].split() if i["text"] else []
        from collections import Counter
        c = Counter(words)
        shape = isinstance(o, list) and all(isinstance(w, str) for w in o)
        return [("list_of_words", shape), ("length_min_k_distinct", (len(o) == min(i["k"], len(c))) if shape else None),
                ("words_from_text_once", (all(w in c for w in o) and len(set(o)) == len(o)) if shape else None),
                ("counts_non_increasing", all(c[o[k]] >= c[o[k + 1]] for k in range(len(o) - 1)) if shape and all(w in c for w in o) else None)]


@spec("C17")
class _C17(Base):
    def props(self, i, o):
        n = len(i["s"])
        return [("int_range", INT(o) and n <= o <= n * (n + 1) // 2)]


@spec("C18")
class _C18(Base):
    def valid(self, i): return i["n"] >= 0 and 2 <= i["base"] <= 36
    def props(self, i, o):
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"[:i["base"]]
        form = isinstance(o, str) and bool(o) and all(ch in digits for ch in o) and (o == "0" or not o.startswith("0"))
        return [("digits_no_leading_zero", form), ("reads_back_to_n", (int(o, i["base"]) == i["n"]) if form else None)]


@spec("C19")
class _C19(Base):
    def valid(self, i): return i["n"] >= 0 and all(isinstance(t, str) and re.fullmatch(r"[A-Z]", t) for t in i["tasks"])
    def props(self, i, o):
        n = len(i["tasks"])
        return [("at_least_len", INT(o) and o >= n), ("n_zero_equals_len", (o == n) if i["n"] == 0 else None)]


def _ints(x):
    return [x] if INT(x) else [v for y in x for v in _ints(y)]


def _nested_ints(x):
    return INT(x) or (isinstance(x, list) and all(_nested_ints(y) for y in x))


@spec("C20")
class _C20(Base):
    def valid(self, i): return _nested_ints(i["data"])
    def props(self, i, o):
        allv = _ints(i["data"])
        return [("flat_int_list", _int_list(o)), ("same_multiset", sorted(o) == sorted(allv) if _int_list(o) else None)]
