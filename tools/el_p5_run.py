"""[P5] Four-arm evaluation run (protocol: P5_PROTOCOL_v1.md). Selection only: NO grader, NO hidden case and NO reference solution is imported here;
grading happens afterwards with tools/el_p5_grade.py.

Arms (same model, same sampling, same sandbox per cohort):
  A plain        : the shared round-1 sample with seed 1 is the answer (one attempt)
  B multi-sample : shared round 1 (seeds 1-4) + fresh samples (P2 'fresh' arm), answer = first candidate with a complete passing public result, cap G beyond round 1
  C V1           : the Evidence-Ladder controller, ladder mode as in the local bank run, GLM critic OFF, learned scorer OFF, token cap = round-1 tokens + G
  D V1.5         : C + task contracts (host registry; allowlisted parser decl_v1); model-proposed contracts never authoritative
Shared generation cache: a request (prompt sha256, seed, max_new_tokens) that an earlier arm already made returns the SAME recorded generation; tokens are
attributed to every arm that uses them ("attributed"), wall time is counted once ("actual").
Cohorts: contract (configs/p5_contract_tasks_v1/tasks.jsonl, MXC process container + trusted verifier) or lcb (LiveCodeBench v6 via the LCB backend on MXC).
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_p5_run.py <new out dir> --cohort contract|lcb --tasks id1,id2,... [--arms A,B,C,D] [--fake]"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from evidence_ladder.common.interfaces import Artifact, ControllerContext, GenerationRequest, TaskSpec        # noqa: E402
from evidence_ladder.controller import generation as GEN                                                   # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController                                # noqa: E402
from evidence_ladder.models import qwen as Q                                                               # noqa: E402
import el_p2_generation_compare as P2                                                                      # noqa: E402

G_BUDGET = 6000
MAX_NEW = 4096
ROUND1 = (1, 2, 3, 4)
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
V1_CFG = {"mode": "ladder", "k": 4, "l2": True, "l2_n": 6, "l2_version": 2, "thinking": False, "discriminate": True, "critic": False, "max_new_tokens": MAX_NEW,
          "max_rounds": 2, "mutant_limit": 8}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class CacheGen:
    """One sample per inner call; identical (prompt, seed, max_new_tokens) requests reuse the recorded generation."""
    def __init__(self, inner, log):
        self.inner, self.log, self.cache, self.arm, self.task = inner, log, {}, "", ""
        self.language = "python"

    def generate(self, rq: GenerationRequest):
        out = []
        for s in list(rq.seeds)[:rq.n]:
            key = (sha(rq.prompt), int(s), int(rq.max_new_tokens), rq.purpose)
            hit = key in self.cache
            t0 = time.monotonic()
            if not hit:
                self.inner.language = self.language
                one = GenerationRequest(rq.task_id, rq.prompt, 1, (int(s),), rq.max_new_tokens, rq.purpose, rq.thinking, rq.prompt_template_id)
                self.cache[key] = list(self.inner.generate(one))[0]
            g = self.cache[key]
            self.log({"task": self.task, "arm": self.arm, "seed": int(s), "purpose": rq.purpose, "template": rq.prompt_template_id, "prompt_sha256": key[0], "reused": hit,
                      "tokens_prompt": g.tokens_prompt, "tokens_generated": g.tokens_generated, "actual_seconds": 0.0 if hit else round(time.monotonic() - t0, 3),
                      "extract_status": g.extract_status})
            out.append(g)
        return out

    def model_info(self):
        return getattr(self.inner, "model_info", lambda: {})()


class ExecLog:
    """Logging only (P5 Step 1 follow-up): one EXEC.jsonl row per execution call: task, arm, method, artifact, wall seconds, check statuses and, on the trusted
    verifier, its time accounting (launch overhead, per-input kill, candidate seconds, harness seconds, outcome) and per-input observations. Changes no result."""
    def __init__(self, inner, log):
        self.inner, self.log, self.who = inner, log, lambda: ("", "")

    def __getattr__(self, name):
        f = getattr(self.inner, name)
        if not name.startswith("run_"):
            return f

        def call(*a, **k):
            tv = getattr(self.inner, "trusted_verifier", None)
            if tv is not None:
                tv.drain_observations()
            t0 = time.monotonic()
            r = f(*a, **k)
            task, arm = self.who()
            art = next((x for x in a if isinstance(x, Artifact)), None)
            row = {"task": task, "arm": arm, "method": name, "artifact": art.artifact_id if art else None, "wall_s": round(time.monotonic() - t0, 3),
                   "statuses": [(o.get("check_id"), o.get("example_index"), o.get("status"), o.get("duration_s")) for o in r] if isinstance(r, list) else None}
            if tv is not None:
                row.update(accounting=dict(tv.accounting), per_input=tv.drain_observations())
            self.log(row)
            return r
        return call


class FakeGen:
    language = "python"

    def generate(self, rq):
        code = "import sys\nprint(sys.stdin.read().strip())\n" if "stdin" in rq.prompt else "def solve(data):\n    return None\n"
        return [Q.make_generation("<think>\n\n</think>\n\n```python\n%s```\n" % code, tokens_prompt=len(rq.prompt) // 4, tokens_generated=min(300, rq.max_new_tokens), seed=s,
                                  prompt_template_id=rq.prompt_template_id, prompt_sha256=sha(rq.prompt), language="python") for s in list(rq.seeds)[:rq.n]]


def load_tasks(cohort: str, ids: List[str], lcb_file=None):
    if cohort == "bank":                 # smoke only: exposed development tasks; contracts only through the allowlisted declaration parser
        import el_bank_local_run as RUN
        return [(TaskSpec(r["task_id"], "bank:" + r["task_id"], "python", "python", r["prompt"], r["public_tests"], r.get("difficulty", "")), None)
                for r in RUN.load_task_ids(ids)], tuple(GEN.FROZEN_STRATEGIES)
    if cohort == "contract":
        recs = {json.loads(l)["task_id"]: json.loads(l) for l in (ROOT / "configs/p5_contract_tasks_v1/tasks.jsonl").read_text("utf-8").splitlines()}
        return [(TaskSpec(i, "p5:" + i, "python", "python", recs[i]["prompt"], recs[i]["public_tests"], recs[i]["difficulty"]), None) for i in ids], tuple(GEN.FROZEN_STRATEGIES)
    import lcb_tasks as L
    T = L.load(Path(lcb_file)) if lcb_file else L.load()
    out = []
    for i in ids:
        pv = T[i]["public_view"]
        out.append((TaskSpec(i, "lcb:" + i, "python", "python", L.render_prompt(pv), [{"input": c["input"], "expected": c["output"]} for c in pv["public_tests"]], pv["difficulty"]),
                    {"format": pv["format"], "fn_name": pv["fn_name"], "public": pv["public_tests"]}))
    return out, (GEN.LCB_TEMPLATE_ID,)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cohort", required=True, choices=("contract", "lcb", "bank"))
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--profile", default="qwen2.5-coder-1.5b-instruct")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--exec", default="mxc", choices=("mxc", "kaggle"), help="one sandbox for ALL arms: MXC process container (PC) or the Kaggle jail port (Kaggle)")
    ap.add_argument("--model-dir"); ap.add_argument("--model-manifest")
    ap.add_argument("--lcb-file", help="public-only LiveCodeBench task file (Kaggle bundle)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        print("refusing: %s exists" % out); return 2
    arms = a.arms.split(",")
    tasks, strategies = load_tasks(a.cohort, a.tasks.split(","), a.lcb_file)
    out.mkdir(parents=True)
    from evidence_ladder.controller.sandbox_provider import make_sandbox
    from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
    from evidence_ladder.execution.lcb_runner import LcbExecutionBackend, LcbRunner
    meta = {t.task_id: m for t, m in tasks if m}
    if a.exec == "kaggle":
        from evidence_ladder.execution.kaggle_port import KaggleJailPort
        port = KaggleJailPort()
        exe = (ProviderExecutionBackend(port, backend="kaggle-jail", scratch=out / "_engine", trusted=True) if a.cohort in ("contract", "bank")
               else LcbExecutionBackend(LcbRunner(port.run, out / "_lcb"), meta))
    else:
        exe = (ProviderExecutionBackend(backend="mxc", scratch=out / "_engine", trusted=True) if a.cohort in ("contract", "bank")
               else LcbExecutionBackend(LcbRunner(make_sandbox(out / "_sandbox", [], backend="mxc").run, out / "_lcb"), meta))
    if hasattr(exe, "start"):
        exe.start(languages=("python",))
    files = {n: open(out / n, "a", encoding="utf-8") for n in ("CALLS.jsonl", "RESULTS.jsonl", "ANSWERS.jsonl", "P2CALLS.jsonl", "P2CANDIDATES.jsonl")}
    w = lambda n: (lambda row: (files[n].write(json.dumps(row) + "\n"), files[n].flush()))
    files["EXEC.jsonl"] = open(out / "EXEC.jsonl", "a", encoding="utf-8")
    exe = ExecLog(exe, w("EXEC.jsonl"))
    if a.fake:
        inner = FakeGen()
    elif a.model_dir:
        import os
        from evidence_ladder.models import backbone as BB
        man = json.loads(Path(a.model_manifest).read_text(encoding="utf-8"))
        verify = {f: m["sha256"] for f, m in man["files"].items() if f.endswith((".safetensors", ".json")) and f != "MANIFEST_pinned.json"}
        os.environ.setdefault(Q.HFQwenBackend.INFERENCE_ENV, "1")
        inner = Q.HFQwenBackend(authorized=True, model_dir=a.model_dir, log_path=str(out / "gen_log.jsonl"), verify_files=verify, profile=BB.get(a.profile), local=True)
    else:
        inner = Q.build_local_backend(a.profile, log_path=str(out / "gen_log.jsonl"), think_policy=None)
    gen = CacheGen(inner, w("CALLS.jsonl"))
    exe.who = lambda: (gen.task, gen.arm)
    contract_cfg = {"registry": str(ROOT / "configs/p5_contract_tasks_v1/REGISTRY_v1.json") if a.cohort == "contract" else None, "root": str(ROOT), "parsers": ["decl_v1"], "n_derived": 6}
    (out / "RUN_CONFIG.json").write_text(json.dumps({"protocol": "P5_PROTOCOL_v1", "cohort": a.cohort, "tasks": [t.task_id for t, _ in tasks], "arms": arms, "G": G_BUDGET, "max_new": MAX_NEW,
                                                     "v1_cfg": V1_CFG, "contracts": contract_cfg, "strategies": list(strategies), "fake": a.fake, "exec": a.exec,
                                                     "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    for task, _ in tasks:
        gen.task = task.task_id
        # ---- shared round 1 (exactly the controller's first sample requests: plan_samples over seeds 1-4)
        gen.arm = "round1"
        r1 = []
        for rq in GEN.plan_samples(task, len(ROUND1), list(ROUND1), 0, strategies, MAX_NEW):
            r1 += list(gen.generate(rq))
        r1_tokens = sum(g.tokens_generated for g in r1)
        def answer(arm, code, label, extra):
            row = {"task": task.task_id, "arm": arm, "label": label, "answer_sha256": sha(code) if code else None, **extra}
            w("RESULTS.jsonl")(row); w("ANSWERS.jsonl")({"task": task.task_id, "arm": arm, "source": code})
            print(json.dumps(row), flush=True)
        t_task = time.monotonic()
        if "A" in arms:
            g1 = r1[0]
            answer("A", g1.code if g1.extract_status == "ok" else None, "ANSWERED" if g1.extract_status == "ok" else "NO_CODE",
                   {"tokens_generated": g1.tokens_generated, "tokens_prompt": g1.tokens_prompt})
        if "B" in arms:
            gen.arm = "B"; t0 = time.monotonic()
            rp = lambda art, _t=task: exe.run_public(art, _t)
            hist = []
            for k, g in enumerate(r1):
                if g.extract_status == "ok" and g.code.strip():
                    art = Artifact.make(task.task_id, "python", g.code, None, "sample:%d" % g.seed)
                    hist.append({"id": art.artifact_id, "order": k, "round": 1, "seed": g.seed, "public": P2.public_result(rp(art), len(task.public_tests)), "source": g.code})
            row = P2.run_arm(gen, rp, task, "fresh", hist, len(r1), G_BUDGET, 2, w("P2CALLS.jsonl"), w("P2CANDIDATES.jsonl"), strategies=strategies)
            sel = next((h for h in hist if h["id"] == row["selected"]), None)
            src = sel["source"] if sel else None
            if src is None and row["selected"]:
                src = next((json.loads(l)["source"] for l in (out / "P2CANDIDATES.jsonl").read_text("utf-8").splitlines()
                            if json.loads(l)["task"] == task.task_id and json.loads(l)["id"] == row["selected"]), None)
            answer("B", src, row["label"], {"tokens_generated": r1_tokens + row["arm_tokens_generated"], "tokens_prompt": sum(g.tokens_prompt for g in r1) + row["arm_tokens_prompt"],
                                            "calls": row["calls"] + len(r1), "seconds": round(time.monotonic() - t0, 1)})
        for arm in ("C", "D"):
            if arm not in arms:
                continue
            gen.arm = arm; t0 = time.monotonic()
            cfg = dict(V1_CFG, strategies=list(strategies))
            if arm == "D":
                cfg["contracts"] = contract_cfg
            ctx = ControllerContext(gen, exe, Q.TokenBudget(task.task_id, "D0", r1_tokens + G_BUDGET), "D0", None, None, SEEDS, cfg)      # "D0" = the controller's ladder arm id (schema); P5 arm name kept in RESULTS
            ctl = EvidenceLadderController()
            res = ctl.run_problem(task, ctx)
            j = ctl.last_journal
            dec = next((e["data"] for e in j.entries("decision")), {})
            srcs = {e["data"]["id"]: e["data"]["source"] for e in j.entries("candidate")}
            ok = res.answer_label in ("SUPPORTED", "VERIFIED")
            cc = [e["data"] for e in j.entries("contract_check")]
            (out / "journals").mkdir(exist_ok=True)
            (out / "journals" / ("%s_%s.jsonl" % (task.task_id, arm))).write_text("".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in j.entries()), encoding="utf-8")
            answer(arm, srcs.get(dec.get("candidate_id")) if ok else None, res.answer_label,
                   {"stop": res.stop_cause, "rounds": res.rounds, "candidates": len(srcs), "tokens_generated": (res.budget_record or {}).get("qwen_gen_tokens"),
                    "tokens_prompt": (res.budget_record or {}).get("qwen_prefill_tokens"), "contract_checks": len(cc), "contract_fails": sum(c["status"] == "fail" for c in cc),
                    "contract_authority": sorted({c["authority"] for c in cc}), "seconds": round(time.monotonic() - t0, 1)})
        w("RESULTS.jsonl")({"task": task.task_id, "arm": "_task", "wall_seconds": round(time.monotonic() - t_task, 1), "round1_tokens_generated": r1_tokens})
    (out / "DONE.json").write_text(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    for f in files.values():
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
