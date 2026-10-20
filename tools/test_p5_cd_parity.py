"""[P5 Step 1(b)] C/D replay parity with NO authoritative contract (bank task P1-21, the smoke task where C and D diverged on Kaggle).
Both arms get IDENTICAL inputs:
  * generations: the candidates arm C recorded on Kaggle (results/raw/p5-kaggle-smoke-v2/p5_smoke_bank), served by (purpose, seed) through the runner's own CacheGen;
  * execution observations: one memo shared by both arms; C runs first on the real MXC process container + trusted verifier and every execution call is recorded;
    D receives the recorded result for every identical call (a call D makes that C never made is executed live and reported as a divergence);
  * backend_info is frozen to its first value (the live value carries running call counters, so environment_revision would differ by arm; reported, not a decision input).
Compares, in order: generation requests (purpose, seed, prompt sha256), candidates, ranking, routing, labels, rounds, decision, stop reason, final label.
Ignored as identifiers: config_digest, environment_revision, receipt_hash, produced_at, prev, seq, registry hashes. Writes the report to argv[1] (create-only).
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/test_p5_cd_parity.py <out.json>"""
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import el_p5_run as P5                                                            # noqa: E402
from evidence_ladder.common.interfaces import ControllerContext                  # noqa: E402
from evidence_ladder.controller import generation as GEN                         # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController       # noqa: E402
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend   # noqa: E402
from evidence_ladder.models import qwen as Q                                     # noqa: E402

SMOKE = ROOT / "results/raw/p5-kaggle-smoke-v2/p5_smoke_bank"
TASK = "P1-21"
IGNORE = {"config_digest", "environment_revision", "receipt_hash", "produced_at", "prev", "seq", "hash"}


def recorded_generations():
    """(purpose, seed) -> (source, tokens_prompt, tokens_generated) from arm C's Kaggle journal + call log (repairs in call order)."""
    cands = [json.loads(l)["data"] for l in (SMOKE / "journals/P1-21_C.jsonl").read_text("utf-8").splitlines() if json.loads(l)["kind"] == "candidate"]
    calls = [json.loads(l) for l in (SMOKE / "CALLS.jsonl").read_text("utf-8").splitlines()]
    calls = [c for c in calls if c["task"] == TASK and c["arm"] in ("round1", "C")]
    samples = {int(c["origin"].split(":")[1]): c["source"] for c in cands if c["origin"].startswith("sample:")}
    repairs = [c["source"] for c in cands if c["origin"].startswith("repair:")]
    out, ri = {}, 0
    for c in calls:
        key = (c["purpose"], c["seed"])
        if key in out:
            continue
        if c["purpose"] == "sample":
            out[key] = (samples[c["seed"]], c["tokens_prompt"], c["tokens_generated"])
        else:
            out[key] = (repairs[ri], c["tokens_prompt"], c["tokens_generated"]); ri += 1
    return out


class ReplayGen:
    language = "python"

    def __init__(self, rec):
        self.rec, self.unrecorded = rec, []

    def generate(self, rq):
        s = int(list(rq.seeds)[0])
        src, tp, tg = self.rec.get((rq.purpose, s), ("def solve(data):\n    return None\n", 0, 0))
        if (rq.purpose, s) not in self.rec:
            self.unrecorded.append((rq.purpose, s))
        return [Q.make_generation("<think>\n\n</think>\n\n```python\n%s\n```\n" % src, tokens_prompt=tp, tokens_generated=tg, seed=s,
                                  prompt_template_id=rq.prompt_template_id, prompt_sha256=P5.sha(rq.prompt), language="python")]


class MemoExec:
    """Shared execution memo: identical call -> identical recorded observation for both arms."""
    def __init__(self, inner):
        self.inner, self.memo, self.arm, self.live = inner, {}, "", {}
        self.info = inner.backend_info()

    def backend_info(self):
        return dict(self.info)

    def __getattr__(self, name):
        f = getattr(self.inner, name)
        if not name.startswith("run_"):
            return f

        def call(*a, **k):
            key = hashlib.sha256(repr((name, a, sorted(k.items()))).encode()).hexdigest()
            if key not in self.memo:
                self.live.setdefault(self.arm, []).append(name)
                self.memo[key] = f(*a, **k)
            return copy.deepcopy(self.memo[key])
        return call


def norm(x):
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items() if k not in IGNORE}
    if isinstance(x, list):
        return [norm(v) for v in x]
    return x


def run(arm, gen, exe, task, r1_tokens, strategies):
    gen.arm, exe.arm = arm, arm
    cfg = dict(P5.V1_CFG, strategies=list(strategies))
    if arm == "D":
        cfg["contracts"] = {"registry": None, "root": str(ROOT), "parsers": ["decl_v1"], "n_derived": 6}      # exactly the smoke's bank-cohort D config
    ctx = ControllerContext(gen, exe, Q.TokenBudget(task.task_id, "D0", r1_tokens + P5.G_BUDGET), "D0", None, None, P5.SEEDS, cfg)
    ctl = EvidenceLadderController()
    res = ctl.run_problem(task, ctx)
    return res, list(ctl.last_journal.entries()), ctl


def main():
    out = Path(sys.argv[1])
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    tasks, strategies = P5.load_tasks("bank", [TASK])
    task = tasks[0][0]
    rg = ReplayGen(recorded_generations())
    reqs = []
    gen = P5.CacheGen(rg, reqs.append)
    gen.task = TASK
    exe = MemoExec(ProviderExecutionBackend(backend="mxc", scratch=Path(tempfile.mkdtemp(prefix="p5par-")).resolve(), trusted=True))
    exe.inner.start(languages=("python",))
    gen.arm = "round1"
    r1 = []
    for rq in GEN.plan_samples(task, len(P5.ROUND1), list(P5.ROUND1), 0, strategies, P5.MAX_NEW):
        r1 += list(gen.generate(rq))
    r1_tokens = sum(g.tokens_generated for g in r1)
    rc, jc, ctl_c = run("C", gen, exe, task, r1_tokens, strategies)
    rd, jd, ctl_d = run("D", gen, exe, task, r1_tokens, strategies)
    rq = {a: [(r["purpose"], r["seed"], r["prompt_sha256"]) for r in reqs if r["arm"] == a] for a in ("C", "D")}
    dec_kinds = ("generation_request", "candidate", "evidence", "label", "ranking", "routing", "round", "lineage_stopped", "decision", "stop", "oracle_tier", "requirements")
    sc = [(e["kind"], norm(e["data"])) for e in jc if e["kind"] in dec_kinds]
    sd = [(e["kind"], norm(e["data"])) for e in jd if e["kind"] in dec_kinds]
    first = next(({"index": i, "C": a, "D": b} for i, (a, b) in enumerate(zip(sc, sd)) if a != b), None)
    if first is None and len(sc) != len(sd):
        first = {"index": min(len(sc), len(sd)), "C": sc[len(sd):][:1], "D": sd[len(sc):][:1]}
    other = sorted({e["kind"] for e in jc + jd} - set(dec_kinds))
    struct = {k: [norm(e["data"]) for e in jc if e["kind"] == k] != [norm(e["data"]) for e in jd if e["kind"] == k] for k in other}
    final = {a: {"label": r.answer_label, "stop": r.stop_cause, "rounds": r.rounds, "decision": next((e["data"] for e in j if e["kind"] == "decision"), None)}
             for a, r, j in (("C", rc, jc), ("D", rd, jd))}
    ok = first is None and rq["C"] == rq["D"] and final["C"] == final["D"] and not exe.live.get("D") and not rg.unrecorded
    rep = {"test": "p5_cd_parity_v1", "task": TASK, "result": "PASS" if ok else "FAIL", "requests_equal": rq["C"] == rq["D"], "requests": rq,
           "first_decision_divergence": first, "decision_entries_compared": [len(sc), len(sd)], "final": final,
           "live_execution_calls": {k: len(v) for k, v in exe.live.items()}, "d_calls_not_made_by_c": exe.live.get("D", []), "unrecorded_generations": rg.unrecorded,
           "non_decision_kinds_differ": struct, "d_registry_has_contract_checks": "contract_checks_v1" in ctl_d._registry.checks,
           "d_contract": None if ctl_d.last_run.contract is None else {"authority": ctl_d.last_run.contract.authority, "source": getattr(ctl_d.last_run.contract, "source", None)},
           "note": "environment_revision is computed from backend_info(), which includes running call counters; frozen here, reported separately"}
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    for arm, j in (("C", jc), ("D", jd)):
        out.with_name(out.stem + "_journal_%s.jsonl" % arm).write_text("".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in j), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("result", "requests_equal", "first_decision_divergence", "final", "live_execution_calls", "d_calls_not_made_by_c", "unrecorded_generations",
                                          "non_decision_kinds_differ", "d_contract")}, indent=1, default=str)[:4000])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
