"""[P5 Step 2] Properties for the QUALIFICATION tasks Q01-Q03 (synthetic, excluded from every evaluation cohort). Same interface as the P5 contract cohort:
props(inp, out) -> [(name, True | False | None)]; valid(inp) narrows validity beyond the schema."""
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


@spec("Q01")
class _Q01(Base):
    def props(self, i, o):
        ok = isinstance(o, list) and all(INT(v) for v in o)
        return [("int_list", ok), ("same_multiset", sorted(o) == sorted(i["nums"]) if ok else None),
                ("non_decreasing", all(o[k] <= o[k + 1] for k in range(len(o) - 1)) if ok else None)]


@spec("Q02")
class _Q02(Base):
    def props(self, i, o):
        return [("string", isinstance(o, str)), ("reversed", o == i["s"][::-1] if isinstance(o, str) else None)]


@spec("Q03")
class _Q03(Base):
    def props(self, i, o):
        return [("int", INT(o)), ("sum", o == i["a"] + i["b"] if INT(o) else None)]
