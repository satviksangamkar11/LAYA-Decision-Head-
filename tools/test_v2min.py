"""[V2-minimal] Mechanical tests (CPU only, scripted model, no generation). Tasks: the Step 2 qualification fixtures Q01-Q04 (excluded from every cohort).
 1 equivalence: v2min.sample_loop(gate=None) makes exactly arm B's calls (seed, prompt sha, max_new) and selects the same candidate as P2.run_arm (pure python)
 2 gate, authoritative registry contract (Q01): public-passing hard-code rejected, sampling continues, correct accepted with contract=pass, no extra sample after it
 3 b_veto reference (Q01, same script): B selects the hard-code, the contract vetoes it -> BLOCKED, no answer
 4 diagnostic-only contract (Q02): contract unavailable -> public-only acceptance, label SUPPORTED with checks.contract=unavailable (never VERIFIED)
 5 allowlisted-parser contract (Q04): the hard-code crashes on derived valid inputs -> rejected; correct accepted
 6 budget exhausted with only hard-codes (Q01) -> BLOCKED, abstain; no call after the budget is reached
 7 public-failing candidates never reach the gate
 8 contract tool failure -> ContractToolError (infrastructure), never a verdict
 9 no result carries the label VERIFIED
11-13 H2-lite prompt adapter: identity adapter = arm B call-for-call; an edited adapter keeps every seed, position, template slot and token limit and changes
      only the prompt text; rule violations (placeholders, +400 characters, solution-format block, exact mapping) are refused
14 an unexpected gate verdict raises GateVerdictError under both policies (fail closed)
10 an incomplete contract check (no valid input checked, or a valid input without a result) is "inconclusive", never "pass"; a violation still fails
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/test_v2min.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import el_p2_generation_compare as P2                                             # noqa: E402
from el_p5_contract_qual import GOOD, HARD                                         # noqa: E402
from evidence_ladder import v2min as V                                             # noqa: E402
from evidence_ladder.common.interfaces import Artifact, TaskSpec                   # noqa: E402
from evidence_ladder.controller import generation as GEN                           # noqa: E402
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend    # noqa: E402
from evidence_ladder.models import qwen as Q                                       # noqa: E402

QD = ROOT / "configs/p5_qual_tasks_v1"
TASKS = {json.loads(l)["task_id"]: json.loads(l) for l in (QD / "tasks.jsonl").read_text("utf-8").splitlines()}
BAD_PUBLIC = "def solve(data):\n    return 'nope'\n"


def task(tid):
    t = TASKS[tid]
    return TaskSpec(tid, "qual:" + tid, "python", "python", t["prompt"], t["public_tests"], t["difficulty"])


class Script:
    """Scripted model: returns the listed programs in call order (cycling the last one); every sample costs `tok` tokens. Records every request."""
    language = "python"

    def __init__(self, programs, tok=100):
        self.programs, self.tok, self.k, self.requests, self.detail = list(programs), tok, 0, [], []

    def generate(self, rq):
        out = []
        for s in list(rq.seeds)[:rq.n]:
            code = self.programs[min(self.k, len(self.programs) - 1)]
            self.k += 1
            self.requests.append((int(s), V.sha(rq.prompt), int(rq.max_new_tokens)))
            self.detail.append({"seed": int(s), "max_new": int(rq.max_new_tokens), "template": rq.prompt_template_id, "prompt": rq.prompt})
            out.append(Q.make_generation("<think>\n\n</think>\n\n```python\n%s```\n" % code, tokens_prompt=50, tokens_generated=self.tok, seed=s,
                                         prompt_template_id=rq.prompt_template_id, prompt_sha256=V.sha(rq.prompt), language="python"))
        return out


checks = {}


def check(name, fn):
    try:
        checks[name] = bool(fn())
    except Exception as e:      # noqa: BLE001
        checks[name] = False
        print(name, "ERROR", type(e).__name__, str(e)[:300])


# ---------------------------------------------------------------- 1 equivalence with arm B (no sandbox: public outcome decided by a marker in the source)
def fake_public(art):
    ok = "MARK_PASS" in art.source
    return [{"check_id": "parse_v1", "status": "pass"}] + [{"check_id": "public_examples_v1", "example_index": i, "status": "pass" if ok else "fail"} for i in range(3)]


def equivalence():
    t = task("Q01")
    progs = ["def solve(d):\n    return %d\n" % i for i in range(9)] + ["def solve(d):\n    return 'MARK_PASS'\n"]
    for budget in (350, 5000):
        a = Script(progs)
        r1 = []
        for rq in GEN.plan_samples(t, 4, [1, 2, 3, 4], 0, tuple(GEN.FROZEN_STRATEGIES), P2.MAX_NEW):
            r1 += list(a.generate(rq))
        hist = []
        for k, g in enumerate(r1):
            art = Artifact.make(t.task_id, "python", g.code, None, "sample:%d" % g.seed)
            hist.append({"id": art.artifact_id, "order": k, "round": 1, "seed": g.seed, "public": P2.public_result(fake_public(art), 3), "source": g.code})
        ra = P2.run_arm(a, fake_public, t, "fresh", hist, len(r1), budget, 2, lambda r: None, lambda r: None, strategies=tuple(GEN.FROZEN_STRATEGIES))
        b = Script(progs)
        rb = V.sample_loop(b, fake_public, t, budget, strategies=tuple(GEN.FROZEN_STRATEGIES))
        if a.requests != b.requests or ra["selected"] != (rb["selected"]["id"] if rb["selected"] else None):
            print("equivalence differs at budget", budget, ra["selected"], rb["selected"] and rb["selected"]["id"], len(a.requests), len(b.requests))
            return False
    return True


check("1_equivalence_with_arm_B", equivalence)

# ---------------------------------------------------------------- H2-lite prompt adapter (pure python)
PROGS = ["def solve(d):\n    return %d\n" % i for i in range(9)] + ["def solve(d):\n    return 'MARK_PASS'\n"]
FROZEN = {tid: V.frozen_template(tid) for tid in GEN.FROZEN_STRATEGIES}
EXTRA = "\nBefore coding, restate the input fields and their types in one line."


def runs(adapter, budget):
    s = Script(PROGS)
    V.sample_loop(s, fake_public, task("Q01"), budget, strategies=tuple(GEN.FROZEN_STRATEGIES), adapter=adapter)
    return s


def adapter_identity():
    ident = V.PromptAdapter("identity", FROZEN)
    return all(runs(None, b).requests == runs(ident, b).requests for b in (350, 5000))


def adapter_h2_same_plan():
    h2 = V.PromptAdapter("h2test", {k: v + EXTRA for k, v in FROZEN.items()})
    for b in (350, 5000):
        x, y = runs(None, b).detail, runs(h2, b).detail
        if [(d["seed"], d["max_new"]) for d in x] != [(d["seed"], d["max_new"]) for d in y]:
            return False
        if [d["template"] + "@h2test" for d in x] != [d["template"] for d in y]:
            return False
        if not all(e["prompt"] == d["prompt"] + EXTRA for d, e in zip(x, y)):      # only the text differs
            return False
    return True


def adapter_rejects():
    bad = [{k: v.replace("{examples}", "") for k, v in FROZEN.items()},                         # placeholder removed
           {k: v + "{extra}" for k, v in FROZEN.items()},                                       # placeholder added
           {k: v + "x" * (V.MAX_EXTRA_CHARS + 1) for k, v in FROZEN.items()},                   # too long
           {k: v.replace("Required solution format:\n{solution_format}", "{solution_format}") for k, v in FROZEN.items()},   # format block dropped
           {"qwen_sample_v2": FROZEN["qwen_sample_v2"]}]                                          # mapping not exact
    for texts in bad:
        try:
            V.PromptAdapter("bad", texts)
            return False
        except ValueError:
            pass
    return True


check("11_adapter_identity_equals_B", adapter_identity)
check("12_adapter_same_seeds_order_budget_only_text_differs", adapter_h2_same_plan)
check("13_adapter_rejects_rule_violations", adapter_rejects)

# ---------------------------------------------------------------- sandbox-backed checks
exe = ProviderExecutionBackend(backend="mxc", scratch=Path(tempfile.mkdtemp(prefix="v2min-")).resolve(), trusted=True)
exe.start(languages=("python",))
gate = V.ContractGate(exe, QD / "REGISTRY_qual_v1.json", ROOT)
rp = lambda t: (lambda art: exe.run_public(art, t))
results = []


def run(policy, tid, progs, budget=2000):
    t = task(tid)
    s = Script(progs)
    r = V.solve(policy, s, rp(t), t, budget, gate)
    r["_requests"] = len(s.requests)
    results.append(r)
    return r


def t2():
    r = run("gate", "Q01", [HARD["Q01"], GOOD["Q01"], GOOD["Q01"], GOOD["Q01"]])
    return (r["label"] == "SUPPORTED" and r["answer"].strip() == GOOD["Q01"].strip() and r["checks"]["contract"] == "pass"
            and len(r["rejected_by_contract"]) == 1 and r["_requests"] == 4)          # round 1 is one batch of 4; accepted inside it, no fresh round


def t2b():
    # acceptance in a fresh round: round 1 all hard-codes, then bad, then correct -> stops right after the correct one
    r = run("gate", "Q01", [HARD["Q01"]] * 4 + [BAD_PUBLIC, GOOD["Q01"], GOOD["Q01"]])
    return r["label"] == "SUPPORTED" and r["checks"]["contract"] == "pass" and r["_requests"] == 6


def t3():
    r = run("b_veto", "Q01", [HARD["Q01"], GOOD["Q01"], GOOD["Q01"], GOOD["Q01"]])
    return r["label"] == "BLOCKED" and r["answer"] is None and r["vetoed"] is not None


def t4():
    r = run("gate", "Q02", [HARD["Q02"], GOOD["Q02"]])
    return r["label"] == "SUPPORTED" and r["checks"] == {"public": "pass", "contract": "unavailable"} and r["answer"].strip() == HARD["Q02"].strip()


def t5():
    r = run("gate", "Q04", [HARD["Q04"], GOOD["Q04"], GOOD["Q04"], GOOD["Q04"]])
    return r["label"] == "SUPPORTED" and r["answer"].strip() == GOOD["Q04"].strip() and len(r["rejected_by_contract"]) == 1


def t6():
    r = run("gate", "Q01", [HARD["Q01"]], budget=350)
    return r["label"] == "BLOCKED" and r["answer"] is None and r["tokens_generated"] - 400 <= 350 + 100 - 1      # 4 round-1 samples + fresh calls until >= budget


def t7():
    seen = []
    g = lambda t, art: (seen.append(art.source), gate(t, art))[1]
    t = task("Q01")
    V.solve("gate", Script([BAD_PUBLIC, BAD_PUBLIC, GOOD["Q01"], GOOD["Q01"]]), rp(t), t, 2000, g)
    return all(s.strip() != BAD_PUBLIC.strip() for s in seen) and len(seen) == 1


class Broken:
    def run_inputs(self, *a, **k):
        raise RuntimeError("simulated sandbox failure")


def t8():
    t = task("Q01")
    try:
        V.solve("gate", Script([GOOD["Q01"]]), rp(t), t, 2000, V.ContractGate(Broken(), QD / "REGISTRY_qual_v1.json", ROOT))
    except V.ContractToolError:
        return True
    return False


for name, fn in (("2_gate_rejects_and_continues", t2), ("2b_stops_after_acceptance_in_fresh_round", t2b), ("3_b_veto_blocks", t3),
                 ("4_unavailable_contract_public_only", t4), ("5_parser_contract_rejects_crash", t5), ("6_budget_exhausted_abstains", t6),
                 ("7_public_failures_never_gated", t7), ("8_tool_failure_raises", t8)):
    check(name, fn)
check("10_incomplete_check_is_inconclusive", lambda: V.verdict({"hard": [], "uncertain": [], "checked_valid_inputs": 0}) == "inconclusive"
      and V.verdict({"hard": [], "uncertain": ["no result for valid input 3"], "checked_valid_inputs": 5}) == "inconclusive"
      and V.verdict({"hard": [{"input": 0}], "uncertain": ["x"], "checked_valid_inputs": 0}) == "fail"
      and V.verdict({"hard": [], "uncertain": [], "checked_valid_inputs": 4}) == "pass")
def unexpected_verdict():
    t = task("Q01")
    for pol in ("gate", "b_veto"):
        try:
            V.solve(pol, Script([GOOD["Q01"]]), rp(t), t, 2000, lambda tk, art: "Fail")
            return False
        except V.GateVerdictError:
            pass
    return True


check("14_unexpected_verdict_raises_both_policies", unexpected_verdict)
check("9_never_VERIFIED", lambda: results and all(r["label"] in ("SUPPORTED", "BLOCKED") for r in results))
print(json.dumps({"version": V.VERSION, "checks": checks, "all_ok": all(checks.values())}, indent=1))
sys.exit(0 if all(checks.values()) else 1)
