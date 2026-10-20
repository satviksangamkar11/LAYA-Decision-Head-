"""[P5 Step 2] Contract-route qualification on EXCLUDED synthetic tasks (configs/p5_qual_tasks_v1; never the 20 final contract tasks), with a scripted model
(no generation) and the P5 controller config (V1_CFG; D adds the contract cfg exactly as the contract cohort does, registry = the qualification registry).
Candidates per request alternate: a hard-coded program that passes every public example but is wrong elsewhere, and a correct program.
Cases and pass conditions:
  Q01 mixed          (registry, authoritative author contract) : D records an authoritative FAIL on the hard-code and never answers it; the correct program is not blocked
  Q01 hardcode_only  (same, every candidate is the hard-code)  : D gives no SUPPORTED/VERIFIED answer
  Q02 mixed          (registry entry marked diagnostic)        : D's contract evidence is diagnostic only and causes no FAILED row that C does not have
  Q03 mixed          (model-sourced contract, registry says authoritative) : host downgrades it to diagnostic; same condition as Q02
                     (diagnostic failures may still withhold SUPPORTED under the frozen row_label rule, so D's choice MAY differ from C's; reported as
                      strict_same_choice_as_C, which was the first-draft condition and failed on Q03 in the MXC dry run)
  Q04 mixed          (no registry entry; allowlisted parser decl_v1) : the hard-code crashes on derived valid inputs -> authoritative FAIL; D never answers it
  every case, arm C  : no contract check in C's registry and no contract evidence in C's journal
Every execution call goes through the selected sandbox (--exec mxc | kaggle) with the trusted verifier, logged to EXEC.jsonl.
Usage: PYTHONPATH=. python tools/el_p5_contract_qual.py <new out dir> [--exec mxc|kaggle]"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import el_p5_run as P5                                                            # noqa: E402
from evidence_ladder.common.interfaces import ControllerContext, TaskSpec        # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController       # noqa: E402
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend   # noqa: E402
from evidence_ladder.models import qwen as Q                                     # noqa: E402

QD = ROOT / "configs/p5_qual_tasks_v1"
HARD = {"Q01": "def solve(data):\n    t = {(): [], (3, 1, 2): [1, 2, 3], (5, 5, 1): [1, 5, 5], (2,): [2]}\n    return t.get(tuple(data['nums']), [])\n",
        "Q02": "def solve(data):\n    return {'': '', 'ab': 'ba', 'abc': 'cba'}.get(data['s'], '')\n",
        "Q03": "def solve(data):\n    return {(1, 2): 3, (0, 0): 0, (-4, 9): 5}.get((data['a'], data['b']), 0)\n",
        "Q04": "def solve(data):\n    return {0: 0, 3: 9, 12: 144}[data['n']]\n"}
GOOD = {"Q01": "def solve(data):\n    return sorted(data['nums'])\n", "Q02": "def solve(data):\n    return data['s'][::-1]\n",
        "Q03": "def solve(data):\n    return data['a'] + data['b']\n", "Q04": "def solve(data):\n    return data['n'] * data['n']\n"}
CASES = [("Q01", "mixed"), ("Q01", "hardcode_only"), ("Q02", "mixed"), ("Q03", "mixed"), ("Q04", "mixed")]


class Scripted:
    language = "python"

    def __init__(self, tid, mode):
        self.tid, self.mode, self.k = tid, mode, 0

    def generate(self, rq):
        out = []
        for s in list(rq.seeds)[:rq.n]:
            code = HARD[self.tid] if (self.mode == "hardcode_only" or self.k % 2 == 0) else GOOD[self.tid]
            self.k += 1
            out.append(Q.make_generation("<think>\n\n</think>\n\n```python\n%s```\n" % code, tokens_prompt=100, tokens_generated=80, seed=s,
                                         prompt_template_id=rq.prompt_template_id, prompt_sha256=P5.sha(rq.prompt), language="python"))
        return out

    def model_info(self):
        return {"backend": "scripted-qualification"}


def kind(src, tid):
    return None if src is None else ("hardcode" if src.strip() == HARD[tid].strip() else "correct" if src.strip() == GOOD[tid].strip() else "other")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--exec", default="mxc", choices=("mxc", "kaggle"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    out.mkdir(parents=True)
    if a.exec == "kaggle":
        from evidence_ladder.execution.kaggle_port import KaggleJailPort
        inner = ProviderExecutionBackend(KaggleJailPort(), backend="kaggle-jail", scratch=out / "_engine", trusted=True)
    else:
        inner = ProviderExecutionBackend(backend="mxc", scratch=out / "_engine", trusted=True)
    inner.start(languages=("python",))
    logf = open(out / "EXEC.jsonl", "a", encoding="utf-8")
    exe = P5.ExecLog(inner, lambda r: (logf.write(json.dumps(r) + "\n"), logf.flush()))
    T = {json.loads(l)["task_id"]: json.loads(l) for l in (QD / "tasks.jsonl").read_text("utf-8").splitlines()}
    ccfg = {"registry": str(QD / "REGISTRY_qual_v1.json"), "root": str(ROOT), "parsers": ["decl_v1"], "n_derived": 6}
    rows, checks = [], {}
    for tid, mode in CASES:
        t = T[tid]
        task = TaskSpec(tid, "qual:" + tid, "python", "python", t["prompt"], t["public_tests"], t["difficulty"])
        res = {}
        for arm in ("C", "D"):
            exe.who = (lambda _t=tid, _a=arm: (_t, _a))
            cfg = dict(P5.V1_CFG, strategies=list(P5.GEN.FROZEN_STRATEGIES))      # as the contract cohort
            if arm == "D":
                cfg["contracts"] = ccfg
            ctx = ControllerContext(Scripted(tid, mode), exe, Q.TokenBudget(tid, "D0", 100000), "D0", None, None, P5.SEEDS, cfg)
            ctl = EvidenceLadderController()
            r = ctl.run_problem(task, ctx)
            j = ctl.last_journal
            dec = next((e["data"] for e in j.entries("decision")), {})
            srcs = {e["data"]["id"]: e["data"]["source"] for e in j.entries("candidate")}
            ans = srcs.get(dec.get("candidate_id")) if r.answer_label in ("SUPPORTED", "VERIFIED") else None
            notes = [e["data"] for e in j.entries("contract_check")]
            failed_rows = sorted({(kind(srcs.get(e["data"]["candidate_id"]), tid), r) for e in j.entries("label") for r, v in e["data"]["rows"].items() if v == "FAILED"})
            res[arm] = {"label": r.answer_label, "stop": r.stop_cause, "answer": kind(ans, tid), "decision_candidate": kind(srcs.get(dec.get("candidate_id")), tid),
                        "registry_has_contract_check": "contract_checks_v1" in ctl._registry.checks, "failed_rows": failed_rows,
                        "contract_notes": [{"candidate": kind(srcs.get(n.get("candidate") or n.get("candidate_id")), tid), "authority": n.get("authority"), "status": n.get("status"),
                                            "violations": str(n.get("violations", n.get("detail", "")))[:200]} for n in notes],
                        "contract": None if ctl.last_run.contract is None else {"authority": ctl.last_run.contract.authority, "source": ctl.last_run.contract.source}}
            (out / ("journal_%s_%s_%s.jsonl" % (tid, mode, arm))).write_text("".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in j.entries()), encoding="utf-8")
        C, D = res["C"], res["D"]
        c_clean = not C["registry_has_contract_check"] and not C["contract_notes"]
        if tid in ("Q01", "Q04"):
            hard_fail = any(n["candidate"] == "hardcode" and n["authority"] == "authoritative" and n["status"] == "fail" for n in D["contract_notes"])
            good_ok = all(n["status"] != "fail" for n in D["contract_notes"] if n["candidate"] == "correct")
            ok = c_clean and hard_fail and D["answer"] != "hardcode" and good_ok and (mode == "hardcode_only" or D["answer"] == "correct")
            if mode == "hardcode_only":
                ok = c_clean and hard_fail and D["answer"] is None
        else:      # diagnostic route. Rule (frozen contract_lib_v1.row_label): a non-authoritative failure never makes a row FAILED, but it withholds SUPPORTED.
            strict = (D["answer"], D["label"], D["decision_candidate"]) == (C["answer"], C["label"], C["decision_candidate"])
            ok = (c_clean and D["contract"] and D["contract"]["authority"] == "diagnostic" and all(n["authority"] != "authoritative" for n in D["contract_notes"])
                  and set(map(tuple, D["failed_rows"])) <= set(map(tuple, C["failed_rows"])))
            res["strict_same_choice_as_C"] = strict
        key = "%s_%s" % (tid, mode)
        checks[key] = bool(ok)
        rows.append({"case": key, "pass": bool(ok), "C": C, "D": D, "strict_same_choice_as_C": res.get("strict_same_choice_as_C")})
        print(json.dumps({"case": key, "pass": bool(ok), "C": {k: C[k] for k in ("label", "answer")}, "D": {k: D[k] for k in ("label", "answer", "contract")},
                          "D_notes": [(n["candidate"], n["authority"], n["status"]) for n in D["contract_notes"]]}), flush=True)
    ex = [json.loads(l) for l in (out / "EXEC.jsonl").read_text("utf-8").splitlines()]
    rep = {"test": "p5_contract_qual_v1", "exec": a.exec, "env": inner.backend_info().get("engine"), "all_pass": all(checks.values()), "checks": checks, "cases": rows,
           "exec_calls": {m: sum(e["method"] == m for e in ex) for m in sorted({e["method"] for e in ex})},
           "exec_timeouts": sum(1 for e in ex if (e.get("accounting") or {}).get("outcome") not in (None, "ran")),
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "QUAL.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("exec", "env", "all_pass", "checks", "exec_calls", "exec_timeouts")}))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
