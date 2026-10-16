"""v2: real mutation adequacy. Plants single-site bugs (operator swaps, off-by-one constants, flipped booleans, removed `not`) in each
reference solution and counts how many the protected cases kill. Survivors include equivalent mutants, so the reported kill rate is a LOWER
bound on adequacy; survivors are listed for review. Trusted tooling: runs only mutated copies of trusted oracles (no model code).
Usage: python3 mutation_adequacy_v2.py <protected_cases.json> <out.json> [--per-mutant-seconds 2]"""
import ast, copy, json, signal, sys, time, collections, heapq, re
from bisect import bisect_left
from pathlib import Path

here = Path(__file__).resolve().parent
src = (here / "oracles.py").read_text(encoding="utf-8")
cases_file, out_file = Path(sys.argv[1]), Path(sys.argv[2])
LIMIT = float(sys.argv[sys.argv.index("--per-mutant-seconds") + 1]) if "--per-mutant-seconds" in sys.argv else 2.0
tree = ast.parse(src)
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("p") and n.name[1:].isdigit()}

CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
       ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}


def sites(fn):
    """Enumerate mutation sites as (kind, index-in-walk, detail)."""
    out = []
    for i, node in enumerate(ast.walk(fn)):
        if isinstance(node, ast.Compare):
            for j, op in enumerate(node.ops):
                if type(op) in CMP:
                    out.append(("cmp", i, j))
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
                out.append(("const+", i, 0)); out.append(("const-", i, 0))
    return out


def mutate(fn, kind, idx, j):
    f = copy.deepcopy(fn)
    nodes = list(ast.walk(f))
    n = nodes[idx]
    if kind == "cmp":
        n.ops[j] = CMP[type(n.ops[j])]()
    elif kind == "bin":
        n.op = BIN[type(n.op)]()
    elif kind == "bool":
        n.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()
    elif kind == "unary":
        # replace the unary node by its operand: find parent and swap
        for p in nodes:
            for fld, val in ast.iter_fields(p):
                if val is n:
                    setattr(p, fld, n.operand)
                elif isinstance(val, list):
                    for k, v in enumerate(val):
                        if v is n:
                            val[k] = n.operand
    elif kind == "flip":
        n.value = not n.value
    elif kind == "const+":
        n.value += 1
    elif kind == "const-":
        n.value -= 1
    ast.fix_missing_locations(f)
    return f


class Timeout(BaseException):
    pass


def _alarm(sig, frm):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)
cases = {t["task_id"][-2:]: t["cases"] for t in json.loads(cases_file.read_text(encoding="utf-8"))}
report = {}
total_m = total_k = 0
t0 = time.time()
for name, fn in sorted(funcs.items()):
    key = name[1:]
    mutants = []
    for kind, idx, j in sites(fn):
        try:
            mf = mutate(fn, kind, idx, j)
            code = ast.unparse(mf)
        except Exception:
            continue
        mutants.append((kind, idx, j, code, getattr(list(ast.walk(fn))[idx], "lineno", 0)))
    killed = 0
    survivors = []
    for kind, idx, j, code, line in mutants:
        ns = {"collections": collections, "heapq": heapq, "re": re, "bisect_left": bisect_left}
        try:
            exec(code, ns)
            m = ns[name]
        except Exception:
            killed += 1  # does not even compile/define: counts as caught
            continue
        dead = False
        for c in cases[key]:
            try:
                signal.setitimer(signal.ITIMER_REAL, LIMIT)
                try:
                    got = m(copy.deepcopy(c["input"]))
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                if got != c["expected"] or type(got) is not type(c["expected"]):
                    dead = True
            except BaseException:
                dead = True
            if dead:
                break
        if dead:
            killed += 1
        else:
            survivors.append({"kind": kind, "line": line, "site": idx})
    report["P1-" + key] = {"mutants": len(mutants), "killed": killed, "kill_rate": round(killed / len(mutants), 3) if mutants else None,
                          "survivors": survivors}
    total_m += len(mutants); total_k += killed
summary = {"cases_file": str(cases_file.name), "tasks": len(report), "mutants": total_m, "killed": total_k,
           "kill_rate_overall": round(total_k / total_m, 4), "seconds": round(time.time() - t0, 1),
           "note": "survivors include equivalent mutants; kill rate is a lower bound on adequacy",
           "tasks_below_0.80": sorted(k for k, v in report.items() if v["kill_rate"] is not None and v["kill_rate"] < 0.80)}
out_file.write_text(json.dumps({"summary": summary, "per_task": report}, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
