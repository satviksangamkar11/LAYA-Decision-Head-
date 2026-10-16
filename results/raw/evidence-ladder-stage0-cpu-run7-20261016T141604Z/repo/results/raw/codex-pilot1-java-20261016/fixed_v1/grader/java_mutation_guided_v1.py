"""Mutation-guided cases for the Java kit. For each target task: enumerate EVERY mutation site of the trusted reference, find mutants that survive the current protected cases,
search random VALID inputs for one where mutant output != reference output (differential run through ReferenceRunner), add it (expected = reference output). Mutants with no
distinguishing input in N random inputs are recorded as likely_equivalent (evidence, not proof). Also adds hand-written boundary cases for J1-01.
Writes fixed_v1/grader/protected_cases.json (kit file untouched). Trusted tooling. Usage: python3 java_mutation_guided_v1.py [N=3000]"""
import json, random, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.argv = [sys.argv[0], "/dev/null", "14"] + sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("jm", Path(__file__).resolve().parent / "java_mutation_v1.py")
src_text = Path(spec.origin).read_text(encoding="utf-8").split("tmp = tempfile.mkdtemp(prefix=\"jmut-\")")[0]  # reuse the library part only
ns = {"__file__": spec.origin}
exec(compile(src_text, spec.origin, "exec"), ns)
KIT, lines, owner, sites, evaluate, cases, same = ns["KIT"], ns["lines"], ns["owner"], ns["sites"], ns["evaluate"], ns["cases"], ns["same"]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
HERE = Path(__file__).resolve().parent
TARGET = ["J1-01", "J1-06", "J1-12", "J1-17", "J1-27", "J1-29", "J1-39"]
rng = random.Random(20261016)


def g01(r):
    return {"text": "".join(r.choice("aAzZ09bB ,@[`{:/") for _ in range(r.randint(0, 7)))}
def g06(r):
    d = [r.randint(0, 9) for _ in range(9)]
    if r.random() < 0.55:
        s = sum((10 - i) * v for i, v in enumerate(d)); c = (-s) % 11; last = "X" if c == 10 else str(c)
    else:
        last = r.choice("0123456789Xx")
    code = "".join(map(str, d)) + last
    if r.random() < 0.2: code = code[:r.randint(1, 9)] + "X" + code[r.randint(1, 9):]
    if r.random() < 0.3: code = code[:r.randint(1, 8)] + "-" + code[r.randint(1, 8):]
    if r.random() < 0.15: code = code[:-1]
    return {"code": code}
def g12(r):
    return {"ip": ".".join(str(r.choice([0, 1, 127, 128, 255, r.randint(0, 255)])) for _ in range(4)), "prefix": r.choice([0, 1, 8, 16, 24, 31, 32, r.randint(0, 32)])}
def g17(r):
    return {"n": r.choice([1, 2, 3, r.randint(1, 40), r.randint(1, 400)]), "k": r.choice([1, 2, 3, r.randint(1, 60)])}
def g27(r):
    hi = r.choice([3, 15, 1000000]); return {"nums": [r.randint(0, hi) for _ in range(r.randint(2, 6))]}
def g29(r):
    return {"nums": [r.randint(0, r.choice([1, 2, 4])) for _ in range(r.randint(1, 9))]}
def g39(r):
    return {"text": "".join(r.choice("abc") for _ in range(r.randint(0, 10)))}
GEN = {"J1-01": g01, "J1-06": g06, "J1-12": g12, "J1-17": g17, "J1-27": g27, "J1-29": g29, "J1-39": g39}
HAND = {"J1-01": [{"text": t} for t in ["Aa", "aA", "Ab", "aB", "az", "za", "0a", "a0", "9a", "Az", "zA", "A0", "9Z", "Zz", "RaceCar", "A man, a plan, a canal: Panama", "ab", "a@a", "a[a", "a`a", "a{a", "0:0", "9/9", "z{"]]}


def build(text, tmp):
    w = Path(tempfile.mkdtemp(dir=tmp))
    (w / "ReferenceSolutions.java").write_text(text, encoding="utf-8")
    shutil.copyfile(KIT / "grader" / "support" / "Json.java", w / "Json.java")
    shutil.copyfile(KIT / "grader" / "ReferenceRunner.java", w / "ReferenceRunner.java")
    p = subprocess.run(["javac", "-proc:none", "-encoding", "UTF-8", "-d", str(w), "Json.java", "ReferenceRunner.java", "ReferenceSolutions.java"], cwd=w, capture_output=True, text=True, timeout=90)
    return w if p.returncode == 0 else None


def run_inputs(w, n, inputs, timeout=15):
    ins = "".join(json.dumps({"task_number": n, "input": i}, ensure_ascii=False, separators=(",", ":")) + "\n" for i in inputs)
    try:
        r = subprocess.run(["java", "-Xmx256m", "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-cp", str(w), "ReferenceRunner"], input=ins, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        return r.stdout.splitlines()
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return out.splitlines()


def first_diff(ref, mut):
    for i, r in enumerate(ref):
        if i >= len(mut) or mut[i] != r:
            return i
    return None


tmp = tempfile.mkdtemp(prefix="jguid-")
ref_w = build("\n".join(lines), tmp)
prot = json.loads((KIT / "grader" / "protected_cases.json").read_text(encoding="utf-8"))
by = {t["id"]: t["cases"] for t in prot}
report = {}


def process(task):
    n = int(task.split("-")[1]); r = random.Random(20261016 + n)
    pool = []
    for i in owner:
        if owner[i] == n:
            pool += [(i, s) for s in sites(lines[i])]
    surv = []
    for i, (a, b, rep) in pool:
        text = "\n".join(lines[:i] + [lines[i][:a] + rep + lines[i][b:]] + lines[i + 1:])
        res = evaluate(task, text, tmp)
        if res == "survived": surv.append((i, a, b, rep, text))
    batch = [GEN[task](r) for _ in range(N)]
    ref_out = run_inputs(ref_w, n, batch)
    added, equiv = [], []
    for i, a, b, rep, text in surv:
        w = build(text, tmp)
        if w is None:
            continue
        try:
            if added and first_diff(run_inputs(ref_w, n, added), run_inputs(w, n, added)) is not None:
                continue
            d = first_diff(ref_out, run_inputs(w, n, batch))
        finally:
            shutil.rmtree(w, ignore_errors=True)
        if d is None: equiv.append({"line": i + 1, "orig": lines[i][a:b], "mutant": rep})
        else: added.append(batch[d])
    return task, {"mutants_total": len(pool), "survivors_before": len(surv), "guided_cases_added": len(added), "likely_equivalent": equiv, "inputs_per_mutant": N}, added


with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(process, TARGET))
exp_ref = {}
for task, rep, added in results:
    report[task] = rep
    n = int(task.split("-")[1])
    allin = added + HAND.get(task, [])
    outs = run_inputs(ref_w, n, allin)
    have = {json.dumps(c["input"], sort_keys=True) for c in by[task]}
    for inp, o in zip(allin, outs):
        k = json.dumps(inp, sort_keys=True)
        if k in have: continue
        have.add(k); by[task].append({"input": inp, "expected": json.loads(o)})
    report[task]["hand_cases"] = len(HAND.get(task, []))
shutil.rmtree(tmp, ignore_errors=True)
outp = HERE / "protected_cases.json"
outp.write_text(json.dumps([{"id": t["id"], "cases": by[t["id"]]} for t in prot], ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
(HERE.parent / "reports" / "java_mutation_guided.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps({t: {k: (v if k != "likely_equivalent" else len(v)) for k, v in r.items()} for t, r in report.items()}, indent=1))
