"""Java mutation adequacy (sampled). Plants single-site bugs in the trusted ReferenceSolutions.java (operator flips, off-by-one literals), compiles each mutant, runs the
task's protected cases and counts kills. Up to PER_TASK valid mutants per task (seeded sample); mutants that do not compile are discarded. Survivors include equivalent mutants,
so the kill rate is a lower bound. Trusted tooling: runs only mutated copies of the trusted reference (unjailed, with timeouts).
Usage: python3 java_mutation_v1.py <out.json> [per_task=14]"""
import json, os, random, re, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "kit" / "Pilot1_Java"
SRC = (KIT / "grader" / "ReferenceSolutions.java").read_text(encoding="utf-8")
PER_TASK = int(sys.argv[2]) if len(sys.argv) > 2 else 14
OWN = {"pal": 1, "leap": 2, "persistence": 3, "roman": 4, "time12": 5, "isbn": 6, "transpose": 7, "caesar": 9, "csv": 11, "network": 12, "spiral": 13, "rotate": 14, "poly": 15,
       "reduce": 16, "josephus": 17, "sumcount": 18, "paths": 19, "trie": 20, "multiply": 21, "sudoku": 22, "critical": 23, "Expr": 24, "restore": 25, "distinctPal": 26, "maxxor": 27,
       "aggregate": 28, "jumps": 29, "unionArea": 30, "bigmul": 31, "inversions": 32, "merge": 32, "BIT": 33, "ranges": 33, "bitRange": 33, "matching": 34, "aug": 34, "balloons": 35,
       "queens": 36, "queensDfs": 36, "paren": 37, "results": 37, "subseq": 38, "lps": 39, "jobs": 40}
lines = SRC.split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith(" };}")) + 1
owner = {}
for i in range(start, len(lines)):
    for m in re.finditer(r"static\s+(?:class\s+)?(?:[\w<>\[\],]+\s+)?(\w+)\s*[({]", lines[i]):
        if m.group(1) in OWN:
            owner[i] = OWN[m.group(1)]; break
offs = [0]
for l in lines:
    offs.append(offs[-1] + len(l) + 1)


def literal_mask(line):
    """True for characters inside string/char literals."""
    mask = [False] * len(line); q = None; i = 0
    while i < len(line):
        c = line[i]
        if q:
            mask[i] = True
            if c == "\\": mask[i + 1 if i + 1 < len(line) else i] = True; i += 1
            elif c == q: q = None
        elif c in "\"'":
            q = c; mask[i] = True
        i += 1
    return mask


def sites(line):
    mask = literal_mask(line); out = []
    for m in re.finditer(r"<=|>=|==|!=|&&|\|\|", line):
        if not mask[m.start()]:
            rep = {"<=": "<", ">=": ">", "==": "!=", "!=": "==", "&&": "||", "||": "&&"}[m.group()]
            out.append((m.start(), m.end(), rep))
    for m in re.finditer(r"(?<![<>=!&|-])[<>](?![<>=-])", line):
        if not mask[m.start()]:
            out.append((m.start(), m.end(), m.group() + "="))
    for m in re.finditer(r"(?<![+=])\+(?![+=])", line):
        if not mask[m.start()]: out.append((m.start(), m.end(), "-"))
    for m in re.finditer(r"(?<![-=])-(?![->=])", line):
        if not mask[m.start()]: out.append((m.start(), m.end(), "+"))
    for m in re.finditer(r"(?<![\w.])\d+(?![\w.])", line):
        if not mask[m.start()]:
            v = int(m.group())
            out += [(m.start(), m.end(), str(v + 1))] + ([(m.start(), m.end(), str(v - 1))] if v > 0 else [])
    return out


cases = {t["id"]: t["cases"] for t in json.loads(Path(os.environ.get("JCASES", KIT / "grader" / "protected_cases.json")).read_text(encoding="utf-8"))}


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool): return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)): return a == b
    if type(a) is not type(b): return False
    if isinstance(a, list): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict): return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    return a == b


def evaluate(task, text, tmp):
    w = Path(tempfile.mkdtemp(dir=tmp))
    try:
        (w / "ReferenceSolutions.java").write_text(text, encoding="utf-8")
        for f in ("Json.java",):
            shutil.copyfile(KIT / "grader" / "support" / f, w / f)
        shutil.copyfile(KIT / "grader" / "ReferenceRunner.java", w / "ReferenceRunner.java")
        p = subprocess.run(["javac", "-proc:none", "-encoding", "UTF-8", "-d", str(w), "Json.java", "ReferenceRunner.java", "ReferenceSolutions.java"], cwd=w, capture_output=True, text=True, timeout=90)
        if p.returncode != 0:
            return "invalid"
        n = int(task.split("-")[1])
        ins = "".join(json.dumps({"task_number": n, "input": c["input"]}, ensure_ascii=False, separators=(",", ":")) + "\n" for c in cases[task])
        try:
            r = subprocess.run(["java", "-Xmx256m", "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-cp", str(w), "ReferenceRunner"], input=ins, capture_output=True, text=True, encoding="utf-8", timeout=8)
        except subprocess.TimeoutExpired:
            return "killed"
        out = r.stdout.splitlines()
        if r.returncode != 0 or len(out) != len(cases[task]):
            return "killed"
        for line, c in zip(out, cases[task]):
            try:
                if not same(json.loads(line), c["expected"]): return "killed"
            except ValueError:
                return "killed"
        return "survived"
    finally:
        shutil.rmtree(w, ignore_errors=True)


def work(task, tmp):
    n = int(task.split("-")[1]); rng = random.Random(20261016 + n)
    pool = []
    for i, l in owner.items():
        if owner[i] == n:
            pool += [(i, s) for s in sites(lines[i])]
    rng.shuffle(pool)
    killed = survived = invalid = 0; surv = []
    for i, (a, b, rep) in pool:
        if killed + survived >= PER_TASK:
            break
        line = lines[i]; new = line[:a] + rep + line[b:]
        text = "\n".join(lines[:i] + [new] + lines[i + 1:])
        r = evaluate(task, text, tmp)
        if r == "invalid": invalid += 1
        elif r == "killed": killed += 1
        else:
            survived += 1; surv.append({"line": i + 1, "orig": line[a:b], "mutant": rep, "context": line[max(0, a - 25):b + 25]})
    tot = killed + survived
    return task, {"valid_mutants_tested": tot, "killed": killed, "survived": survived, "invalid_discarded": invalid, "available_sites": len(pool),
                  "kill_rate": round(killed / tot, 3) if tot else None, "survivors": surv}


tmp = tempfile.mkdtemp(prefix="jmut-")
t0 = time.time()
tasks = sorted(cases)
with ThreadPoolExecutor(max_workers=6) as ex:
    res = dict(ex.map(lambda t: work(t, tmp), tasks))
shutil.rmtree(tmp, ignore_errors=True)
tested = sum(v["valid_mutants_tested"] for v in res.values()); killed = sum(v["killed"] for v in res.values())
summary = {"per_task_sample": PER_TASK, "mutants_tested": tested, "killed": killed, "kill_rate_overall": round(killed / tested, 4), "seconds": round(time.time() - t0),
           "tasks_without_mutable_code": [t for t, v in res.items() if v["valid_mutants_tested"] == 0],
           "tasks_below_0.80": sorted(t for t, v in res.items() if v["kill_rate"] is not None and v["kill_rate"] < 0.80),
           "note": "sampled, seeded; survivors include equivalent mutants; kill rate is a lower bound; sampling error is large for small per-task samples"}
Path(sys.argv[1]).write_text(json.dumps({"summary": summary, "per_task": res}, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
