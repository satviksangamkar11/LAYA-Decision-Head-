"""[P6 contract cohort] Author properties for D01-D20, written from each task's STATEMENT only, before any candidate exists, without any hidden test or
reference solution. Each property restates a sentence of the statement. props(inp, out) -> [(name, True | False | None)]; valid(inp) narrows validity beyond the schema."""
import re
from collections import Counter

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


def _ints(o): return isinstance(o, list) and all(INT(v) for v in o)


@spec("D01")
class _D01(Base):
    def props(self, i, o):
        n = i["n"]; ok = _ints(o) and len(o) == n + 1
        return [("int_list_len_n_plus_1", ok), ("ends_are_1", (o[0] == 1 and o[-1] == 1) if ok else None),
                ("symmetric", o == o[::-1] if ok else None), ("sum_is_2_pow_n", sum(o) == 2 ** n if ok else None)]


@spec("D02")
class _D02(Base):
    def props(self, i, o):
        n = i["n"]; ok = _ints(o) and len(o) == 2 ** n
        return [("int_list_len_2_pow_n", ok), ("element_i_is_i_xor_i_shr_1", all(o[k] == k ^ (k >> 1) for k in range(len(o))) if ok else None)]


@spec("D03")
class _D03(Base):
    def props(self, i, o):
        a = i["nums"]
        if len(a) < 2:
            return [("zero_when_fewer_than_two", o == 0 and INT(o))]
        return [("int_between_0_and_range", INT(o) and 0 <= o <= max(a) - min(a))]


@spec("D04")
class _D04(Base):
    def props(self, i, o):
        n = i["n"]
        return [("positive_int", INT(o) and o >= 1), ("one_when_zero", (o == 1) if n == 0 else None), ("at_most_4_pow_n", (INT(o) and o <= 4 ** n))]


@spec("D05")
class _D05(Base):
    def props(self, i, o):
        a = i["nums"]; c = Counter(a); maj = [v for v, k in c.items() if 2 * k > len(a)]
        return [("majority_or_null", (o is None) if not maj else (INT(o) and o == maj[0]))]


@spec("D06")
class _D06(Base):
    def valid(self, i): return len(i["nums"]) >= 1
    def props(self, i, o):
        a = i["nums"]; ok = _ints(o) and len(o) == len(a)
        def prod(xs):
            p = 1
            for x in xs: p *= x
            return p
        return [("int_list_same_length", ok), ("product_of_the_others", all(o[k] == prod(a[:k] + a[k + 1:]) for k in range(len(a))) if ok else None)]


@spec("D07")
class _D07(Base):
    def props(self, i, o):
        a, b = i["a"], i["b"]
        return [("int_between_0_and_min_len", INT(o) and 0 <= o <= min(len(a), len(b))), ("equal_strings_give_length", (o == len(a)) if a == b else None)]


@spec("D08")
class _D08(Base):
    def props(self, i, o):
        s = i["s"]
        return [("nonnegative_int", INT(o) and o >= 0), ("empty_is_one", (o == 1) if s == "" else None),
                ("leading_zero_is_zero", (o == 0) if s.startswith("0") else None)]


@spec("D09")
class _D09(Base):
    def props(self, i, o):
        n = len(i["nums"])
        return [("int_between_0_and_pairs", INT(o) and 0 <= o <= n * (n - 1) // 2)]


@spec("D10")
class _D10(Base):
    def props(self, i, o):
        n = i["n"]; ok = _ints(o) and len(o) == n + 1
        return [("int_list_len_n_plus_1", ok), ("element_i_is_popcount", all(o[k] == bin(k).count("1") for k in range(n + 1)) if ok else None)]


@spec("D11")
class _D11(Base):
    def props(self, i, o):
        n = len(i["nums"]); pairs = n * (n - 1) // 2
        return [("int_between_0_and_30_per_pair", INT(o) and 0 <= o <= 30 * pairs), ("zero_for_fewer_than_two", (o == 0) if n < 2 else None)]


@spec("D12")
class _D12(Base):
    def props(self, i, o):
        c = Counter(i["nums"])
        return [("sorted_values_occurring_more_than_once", _ints(o) and o == sorted(v for v, k in c.items() if k > 1))]


@spec("D13")
class _D13(Base):
    def props(self, i, o):
        seen, want = set(), []
        for v in i["nums"]:
            if v not in seen:
                seen.add(v); want.append(v)
        return [("distinct_in_first_occurrence_order", _ints(o) and o == want)]


@spec("D14")
class _D14(Base):
    def props(self, i, o):
        a = i["nums"]; d = len(set(a))
        return [("int_between_0_and_distinct", INT(o) and 0 <= o <= d), ("zero_iff_empty", (o == 0) == (len(a) == 0) if INT(o) else None),
                ("at_least_one_if_nonempty", (o >= 1) if (a and INT(o)) else None)]


@spec("D15")
class _D15(Base):
    def props(self, i, o):
        s, t = i["s"], i["t"]
        return [("bool", isinstance(o, bool)), ("equal_strings_true", (o is True) if s == t else None),
                ("different_lengths_false", (o is False) if len(s) != len(t) else None)]


@spec("D16")
class _D16(Base):
    def props(self, i, o):
        s, r = i["s"], i["rows"]
        return [("string_same_characters", isinstance(o, str) and sorted(o) == sorted(s)),
                ("unchanged_when_one_row_or_rows_ge_len", (o == s) if (r == 1 or r >= len(s)) else None)]


@spec("D17")
class _D17(Base):
    def valid(self, i):
        m, k = i["matrix"], i["k"]; n = len(m)
        if n == 0 or any(len(r) != n for r in m) or not 1 <= k <= n * n:
            return False
        return all(m[r][c] <= m[r][c + 1] for r in range(n) for c in range(n - 1)) and all(m[r][c] <= m[r + 1][c] for r in range(n - 1) for c in range(n))
    def props(self, i, o):
        flat = sorted(v for r in i["matrix"] for v in r)
        return [("kth_smallest_counting_duplicates", INT(o) and o == flat[i["k"] - 1])]


@spec("D18")
class _D18(Base):
    def valid(self, i): return i["path"].startswith("/")
    def props(self, i, o):
        ok = isinstance(o, str) and o.startswith("/")
        segs = o.split("/")[1:] if ok else []
        return [("absolute", ok), ("no_empty_dot_or_dotdot_segments", (o == "/" or all(x not in ("", ".", "..") for x in segs)) if ok else None)]


@spec("D19")
class _D19(Base):
    def props(self, i, o):
        s = i["s"]
        return [("int_between_0_and_len", INT(o) and 0 <= o <= len(s)), ("zero_iff_empty", ((o == 0) == (s == "")) if INT(o) else None),
                ("palindrome_gives_length", (o == len(s)) if s == s[::-1] else None)]


@spec("D20")
class _D20(Base):
    def valid(self, i): return len(i["gas"]) == len(i["cost"])
    def props(self, i, o):
        g, c = i["gas"], i["cost"]; n = len(g)
        def ok_from(s):
            tank = 0
            for k in range(n):
                j = (s + k) % n
                tank += g[j] - c[j]
                if tank < 0:
                    return False
            return True
        want = next((s for s in range(n) if ok_from(s)), -1)
        return [("smallest_feasible_start_or_minus_1", INT(o) and o == want)]
