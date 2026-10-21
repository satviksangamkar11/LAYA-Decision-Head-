"""[P6] Mechanical tests of the P6 runner (CPU only; scripted model; real MXC process container + LcbRunner / trusted verifier). No grading data is read.
 1 cf stdin path: EVERY request (round 1, A, B, V2, later rounds) uses lcb_generic_v1 and exactly the rendered LiveCodeBench stdin prompt; never the bank template;
   the fenced program is extracted and run on the statement's examples through LcbRunner; B accepts the correct program found in a later round
 2 cf V2 identity: same answer, same requests (seed, prompt sha, max tokens, purpose, template), same tokens, every gate verdict 'unavailable', all V2 calls reused
 3 arm A is the seed-1 sample only: an extraction failure at seed 1 gives NO_CODE even when a later sample passes; a wrong seed-1 program stays the answer
   (ANSWERED, public_pass False); A's tokens are its own sample's
 4 the allowlisted declaration parser finds no contract in any of the 77 Codeforces prompts (so 'unavailable' is proven, not assumed)
 5 B (v2min.sample_loop, no gate) == P5 arm B (P2.run_arm) call for call with the LiveCodeBench template
 6 contract cohort: B accepts a public-passing hard-code; V2 rejects it on the authoritative P6 contract and answers with the correct program
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/test_p6_stdin.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import el_p2_generation_compare as P2                                         # noqa: E402
import p6_cf_tasks as CFT                                                      # noqa: E402
from evidence_ladder import v2min as V                                         # noqa: E402
from evidence_ladder.contracts import contract_v1 as CV                       # noqa: E402
from evidence_ladder.controller import generation as GEN                      # noqa: E402
from evidence_ladder.models import qwen as Q                                   # noqa: E402

PY = sys.executable
TMP = Path(tempfile.mkdtemp(prefix="p6t-")).resolve()
fence = lambda code: "<think>\n\n</think>\n\n```python\n%s```\n" % code
CF_GOOD = "t = int(input())\nfor _ in range(t):\n    l, r = map(int, input().split())\n    print(1 if l == r == 1 else r - l)\n"
CF_BAD = "t = int(input())\nfor _ in range(t):\n    input()\n    print(0)\n"
D13_HARD = ("def solve(data):\n    t = {(): [], (3, 1, 3, 2, 1): [3, 1, 2], (7, 7, 7): [7], (-1, 0, -1): [-1, 0]}\n    return t.get(tuple(data['nums']), [])\n")
D13_GOOD = "def solve(data):\n    out = []\n    for v in data['nums']:\n        if v not in out:\n            out.append(v)\n    return out\n"
checks = {}


def run(name, cohort, task, texts):
    sp = TMP / (name + ".json"); sp.write_text(json.dumps(texts))
    out = TMP / name
    p = subprocess.run([PY, "-B", str(ROOT / "tools/el_p6_run.py"), str(out), "--cohort", cohort, "--tasks", task, "--exec", "mxc", "--script", str(sp)],
                       cwd=str(ROOT), env={**__import__("os").environ, "PYTHONPATH": str(ROOT), "ATEN_CPU_CAPABILITY": "avx2"}, capture_output=True, text=True, timeout=1200)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-1500:])
    rd = lambda f: [json.loads(l) for l in (out / f).read_text("utf-8").splitlines() if l.strip()] if (out / f).exists() else []
    res = {r["arm"]: r for r in rd("RESULTS.jsonl")}
    ans = {r["arm"]: r["source"] for r in rd("ANSWERS.jsonl")}
    return res, ans, rd("CALLS.jsonl"), rd("IDENTITY.jsonl")


def check(name, fn):
    try:
        checks[name] = bool(fn())
    except Exception as e:      # noqa: BLE001
        checks[name] = False
        print(name, "ERROR", type(e).__name__, str(e)[:600])


ROW = CFT.load()["2063/A"]
PROMPT_SHA = V.sha(CFT.prompt(ROW))
S1 = run("cf1", "cf", "2063/A", [fence(CF_BAD)] * 6 + [fence(CF_GOOD)])


def t1():
    res, ans, calls, _ = S1
    ok_prompts = all(c["template"] == GEN.LCB_TEMPLATE_ID and c["prompt_sha256"] == PROMPT_SHA for c in calls)
    later = max(c["seed"] for c in calls if c["arm"] == "B") > 4
    return ok_prompts and later and res["B"]["label"] == "SUPPORTED" and ans["B"].strip() == CF_GOOD.strip() and len(calls) > 0


def t2():
    res, ans, calls, idn = S1
    i = idn[0]
    return (i["identical"] and i["all_unavailable"] and len(i["verdicts"]) >= 1 and i["v2_all_reused"] and ans["V2"] == ans["B"]
            and res["V2"]["tokens_generated_logical"] == res["B"]["tokens_generated_logical"])


def t3():
    r_a, a_a, _, _ = run("cfA1", "cf", "2063/A", ["no code here, just words"] + [fence(CF_GOOD)] * 3)
    r_b, a_b, _, _ = run("cfA2", "cf", "2063/A", [fence(CF_BAD)] + [fence(CF_GOOD)] * 3)
    return (r_a["A"]["label"] == "NO_CODE" and a_a["A"] is None and r_a["A"]["public_pass"] is None and r_a["B"]["label"] == "SUPPORTED"
            and r_b["A"]["label"] == "ANSWERED" and a_b["A"].strip() == CF_BAD.strip() and r_b["A"]["public_pass"] is False
            and r_b["A"]["tokens_generated_logical"] == 100 and r_b["A"]["tokens_generated_own"] == 0)


def t4():
    return not [k for k, r in CFT.load().items() if CV.derive_from_text(k, CFT.prompt(r))] and len(CFT.load()) == 77


def t5():
    class Rec:
        language = "python"

        def __init__(self):
            self.k, self.req = 0, []

        def generate(self, rq):
            o = []
            for s in list(rq.seeds)[:rq.n]:
                self.req.append((int(s), V.sha(rq.prompt), int(rq.max_new_tokens), rq.prompt_template_id))
                code = "print('MARK_PASS')\n" if self.k == 7 else "print(%d)\n" % self.k
                self.k += 1
                o.append(Q.make_generation(fence(code), tokens_prompt=10, tokens_generated=300, seed=s, prompt_template_id=rq.prompt_template_id,
                                           prompt_sha256=V.sha(rq.prompt), language="python"))
            return o

    fp = lambda art: [{"check_id": "parse_v1", "status": "pass"}] + [{"check_id": "public_examples_v1", "example_index": 0, "status": "pass" if "MARK_PASS" in art.source else "fail"}]
    task = CFT.task_spec(ROW)
    st = (GEN.LCB_TEMPLATE_ID,)
    a = Rec()
    r1 = []
    for rq in GEN.plan_samples(task, 4, [1, 2, 3, 4], 0, st, 4096):
        r1 += list(a.generate(rq))
    hist = []
    from evidence_ladder.common.interfaces import Artifact
    for k, g in enumerate(r1):
        art = Artifact.make(task.task_id, "python", g.code, None, "sample:%d" % g.seed)
        hist.append({"id": art.artifact_id, "order": k, "round": 1, "seed": g.seed, "public": P2.public_result(fp(art), 1), "source": g.code})
    ra = P2.run_arm(a, fp, task, "fresh", hist, len(r1), 6000, 2, lambda r: None, lambda r: None, strategies=st)
    b = Rec()
    rb = V.sample_loop(b, fp, task, 6000, strategies=st)
    return a.req == b.req and ra["selected"] == (rb["selected"]["id"] if rb["selected"] else None) and all(x[3] == GEN.LCB_TEMPLATE_ID for x in b.req)


def t6():
    res, ans, calls, idn = run("ct", "contract", "D13", [fence(D13_HARD)] + [fence(D13_GOOD)] * 3)
    return (res["B"]["label"] == "SUPPORTED" and ans["B"].strip() == D13_HARD.strip()
            and res["V2"]["label"] == "SUPPORTED" and ans["V2"].strip() == D13_GOOD.strip() and len(res["V2"]["rejected_by_contract"]) == 1
            and "fail" in res["V2"]["gate_verdicts"] and ans["A"].strip() == D13_HARD.strip() and not idn)


for n, f in (("1_cf_stdin_path_all_rounds", t1), ("2_cf_v2_identity_exact", t2), ("3_arm_A_seed1_only", t3), ("4_no_declaration_contract_in_77_prompts", t4),
             ("5_B_equals_P5_armB_lcb_template", t5), ("6_contract_cohort_gate", t6)):
    check(n, f)
print(json.dumps({"checks": checks, "all_ok": all(checks.values())}, indent=1))
sys.exit(0 if all(checks.values()) else 1)
