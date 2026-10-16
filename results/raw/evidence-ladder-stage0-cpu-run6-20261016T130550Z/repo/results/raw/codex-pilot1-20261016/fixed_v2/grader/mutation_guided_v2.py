"""v2: close mutation gaps. For each target task, every mutant that survives the current protected cases is searched against random VALID inputs;
a distinguishing input is added (expected value from the trusted oracle). Mutants with no distinguishing input in N tries are recorded as 'likely_equivalent'
(evidence, not proof). Also appends hand-written cases for the three wrong solutions that escaped (U-shaped islands, two-element rotated arrays, newline in regex).
Trusted tooling. Usage: python3 mutation_guided_v2.py [N=20000]"""
import ast, collections, copy, heapq, json, random, re, signal, sys
from bisect import bisect_left
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from oracles import ORACLES
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
src = (here / "oracles.py").read_text(encoding="utf-8")
tree = ast.parse(src)
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name[1:].isdigit()}
CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}


def sites(fn):
    out = []
    for i, node in enumerate(ast.walk(fn)):
        if isinstance(node, ast.Compare):
            out += [("cmp", i, j) for j, op in enumerate(node.ops) if type(op) in CMP]
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in BIN:
            out.append(("bin", i, 0))
        elif isinstance(node, ast.BoolOp):
            out.append(("bool", i, 0))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            out.append(("unary", i, 0))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                out.append(("flip", i, 0))
            elif isinstance(node.value, int):
                out += [("const+", i, 0), ("const-", i, 0)]
    return out


def mutate(fn, kind, idx, j):
    f = copy.deepcopy(fn); nodes = list(ast.walk(f)); n = nodes[idx]
    if kind == "cmp": n.ops[j] = CMP[type(n.ops[j])]()
    elif kind == "bin": n.op = BIN[type(n.op)]()
    elif kind == "bool": n.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()
    elif kind == "unary":
        for p in nodes:
            for fld, val in ast.iter_fields(p):
                if val is n: setattr(p, fld, n.operand)
                elif isinstance(val, list):
                    for k, v in enumerate(val):
                        if v is n: val[k] = n.operand
    elif kind == "flip": n.value = not n.value
    elif kind == "const+": n.value += 1
    elif kind == "const-": n.value -= 1
    ast.fix_missing_locations(f)
    return ast.unparse(f)


class TO(BaseException):
    pass
def _al(s, f): raise TO()
signal.signal(signal.SIGALRM, _al)


def call(fn, inp):
    try:
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        try: return ("ok", fn(copy.deepcopy(inp)))
        finally: signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException:
        return ("err", None)


def differs(m, inp, key):
    r = call(m, inp); o = ORACLES[key](copy.deepcopy(inp))
    return r[0] == "err" or r[1] != o or type(r[1]) is not type(o)

# ---- valid-input generators ----
def g16(r):
    m = r.randint(0, 6); n = r.randint(1, 6); p = r.choice([0.3, 0.5, 0.7])
    return {"grid": ["".join("1" if r.random() < p else "0" for _ in range(n)) for _ in range(m)]}
def g18(r):
    return {"coins": [r.randint(1, 12) for _ in range(r.randint(1, 4))], "amount": r.randint(0, 30)}
def g19(r):
    return {"s": "".join(r.choice("abc") for _ in range(r.randint(0, 12))), "t": "".join(r.choice("abc") for _ in range(r.randint(0, 4)))}
def g25(r):
    n = r.randint(0, 9); a = sorted(r.sample(range(-20, 21), n)); k = r.randint(0, n) if n else 0
    a = a[k:] + a[:k]
    t = r.choice(a) if a and r.random() < 0.7 else r.randint(-22, 22)
    return {"nums": a, "target": t}
def g26(r, depth=3):
    def gen(d):
        if d == 0 or r.random() < 0.3:
            v = r.randint(-12, 12); return [str(v)], v
        while True:
            lt, lv = gen(d - 1); rt, rv = gen(d - 1); op = r.choice("+-*/")
            if op == "/" and rv == 0: continue
            v = {"+": lv + rv, "-": lv - rv, "*": lv * rv}.get(op) if op != "/" else (abs(lv) // abs(rv)) * (-1 if (lv < 0) ^ (rv < 0) else 1)
            return lt + rt + [op], v
    return {"tokens": gen(depth)[0]}
GEN = {"P1-16": g16, "P1-18": g18, "P1-19": g19, "P1-25": g25, "P1-26": g26}

HAND = {
    "P1-16": [{"grid": ["101", "111"]}, {"grid": ["111", "101"]}, {"grid": ["1011", "1101", "0111"]}, {"grid": ["1", "0", "1"]}, {"grid": ["101"]},
              {"grid": ["110", "011", "110"]}, {"grid": ["10101", "11111", "10001", "11111"]}],
    "P1-25": [{"nums": [3, 1], "target": 1}, {"nums": [2, 1], "target": 1}, {"nums": [5, 1, 2, 3, 4], "target": 1}, {"nums": [3, 4, 5, 1, 2], "target": 1},
              {"nums": [1, 3], "target": 3}, {"nums": [4, 5, 6, 7, 0, 1, 2], "target": 0}, {"nums": [4, 5, 6, 7, 0, 1, 2], "target": 3}, {"nums": [2, 3, 1], "target": 1}],
    "P1-34": [{"text": "a\nb", "pattern": "a.b"}, {"text": "\n", "pattern": "."}, {"text": "a\n", "pattern": "a.*"}, {"text": "\n\n", "pattern": ".*"}, {"text": "ab", "pattern": "a\n*b"}]}

P = here / "protected_cases.json"
data = json.loads(P.read_text(encoding="utf-8"))
by = {t["task_id"]: t["cases"] for t in data}
rng = random.Random(20261016)
report = {}
for task in GEN:
    key = task[-2:]; fn = funcs["p" + key]
    cur = by[task]
    muts = []
    for kind, idx, j in sites(fn):
        try: code = mutate(fn, kind, idx, j)
        except Exception: continue
        ns = {"collections": collections, "heapq": heapq, "re": re, "bisect_left": bisect_left}
        try:
            exec(code, ns); muts.append((kind, idx, ns["p" + key], getattr(list(ast.walk(fn))[idx], "lineno", 0)))
        except Exception:
            continue
    added, equiv, survivors0 = [], [], 0
    for kind, idx, m, line in muts:
        if any(differs(m, c["input"], key) for c in cur):
            continue
        survivors0 += 1
        if any(differs(m, a, key) for a in added):
            continue
        found = None
        for _ in range(N):
            inp = GEN[task](rng)
            if differs(m, inp, key):
                found = inp; break
        if found is None:
            equiv.append({"kind": kind, "line": line})
        else:
            added.append(found)
    hand = []
    for inp in HAND.get(task, []):
        hand.append(inp)
    for inp in added + hand:
        s = json.dumps(inp, sort_keys=True)
        if all(json.dumps(c["input"], sort_keys=True) != s for c in cur):
            cur.append({"input": inp, "expected": ORACLES[key](copy.deepcopy(inp))})
    report[task] = {"mutants": len(muts), "survivors_before": survivors0, "guided_cases_added": len(added), "likely_equivalent": equiv, "tries_per_mutant": N}
for task, inputs in HAND.items():  # P1-34 is not in GEN: add its hand cases separately
    if task in GEN: continue
    key = task[-2:]; cur = by[task]
    for inp in inputs:
        s = json.dumps(inp, sort_keys=True)
        if all(json.dumps(c["input"], sort_keys=True) != s for c in cur):
            cur.append({"input": inp, "expected": ORACLES[key](copy.deepcopy(inp))})
    report[task] = {"hand_cases_added": len(inputs)}
P.write_text(json.dumps([{"task_id": t["task_id"], "cases": by[t["task_id"]]} for t in data], ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
(here.parent / "reports" / "mutation_guided.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
