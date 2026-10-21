"""[P6] Generation runner (selection only: NO grader, NO grading test, NO hidden case is read here). Output format: P6_OUTPUT_SCHEMA_v1.md.
Per task, all arms share one generation cache with exact reuse (identical (prompt, seed, max_new_tokens, purpose) -> the recorded generation):
  round 1 : GEN.plan_samples over seeds 1-4 (generated once, arm "round1")
  A       : the seed-1 round-1 sample as is (no search for a public-passing replacement; extraction failure = no answer); its public-example result is recorded
  B       : v2min.sample_loop without a gate (= P5 arm B), G = 6,000 generated tokens beyond round 1
  V2      : v2min.sample_loop with the trusted contract gate (per-candidate; reject a proven failure and continue)
Cohorts: cf = Codeforces stdin tasks (configs/p6_cf_public_v1.jsonl; LiveCodeBench generic stdin prompt `lcb_generic_v1` for EVERY request; LcbRunner public
examples; no contract exists, so V2 must equal B: IDENTITY.jsonl records the check per task); contract = D01-D20 (bank templates qwen_sample_v2/v2b,
trusted verifier, host registry configs/p6_contract_tasks_v1/REGISTRY_v1.json + allowlisted parser decl_v1).
Usage: PYTHONPATH=. python tools/el_p6_run.py <new out dir> --cohort cf|contract --tasks id1,id2 [--exec kaggle|mxc] [--model-dir D --model-manifest M] [--fake]"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import el_p5_run as P5                                                          # noqa: E402
import p6_cf_tasks as CFT                                                       # noqa: E402
from evidence_ladder import v2min as V                                          # noqa: E402
from evidence_ladder.common.interfaces import Artifact, GenerationRequest, TaskSpec    # noqa: E402
from evidence_ladder.controller import generation as GEN                       # noqa: E402
from evidence_ladder.models import qwen as Q                                    # noqa: E402

G = 6000
MAX_NEW = 4096
SEEDS1 = (1, 2, 3, 4)
REGISTRY = ROOT / "configs/p6_contract_tasks_v1/REGISTRY_v1.json"
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()


class Cache:
    """One sample per inner call; identical (prompt sha, seed, max_new_tokens, purpose) reuse the recorded generation (the P5 CacheGen rule) and every call is logged."""
    language = "python"

    def __init__(self, inner, log):
        self.inner, self.log, self.cache, self.arm, self.task = inner, log, {}, "", ""

    def generate(self, rq: GenerationRequest):
        out = []
        for s in list(rq.seeds)[:rq.n]:
            key = (sha(rq.prompt), int(s), int(rq.max_new_tokens), rq.purpose)
            hit = key in self.cache
            t0 = time.monotonic()
            if not hit:
                self.inner.language = "python"
                self.cache[key] = list(self.inner.generate(GenerationRequest(rq.task_id, rq.prompt, 1, (int(s),), rq.max_new_tokens, rq.purpose, rq.thinking,
                                                                             rq.prompt_template_id)))[0]
            g = self.cache[key]
            self.log({"task": self.task, "arm": self.arm, "seed": int(s), "purpose": rq.purpose, "template": rq.prompt_template_id, "prompt_sha256": key[0],
                      "max_new_tokens": int(rq.max_new_tokens), "reused": hit, "tokens_prompt": g.tokens_prompt, "tokens_generated": g.tokens_generated,
                      "actual_seconds": 0.0 if hit else round(time.monotonic() - t0, 3), "extract_status": g.extract_status})
            out.append(g)
        return out


class ScriptGen:
    """TESTS ONLY: replays fixed raw outputs in call order (100 generated tokens each)."""
    language = "python"

    def __init__(self, texts):
        self.texts, self.k = list(texts), 0

    def generate(self, rq):
        out = []
        for s in list(rq.seeds)[:rq.n]:
            t = self.texts[min(self.k, len(self.texts) - 1)]
            self.k += 1
            out.append(Q.make_generation(t, tokens_prompt=50, tokens_generated=100, seed=s, prompt_template_id=rq.prompt_template_id,
                                         prompt_sha256=sha(rq.prompt), language="python"))
        return out


def load_tasks(cohort, ids, cf_file):
    if cohort == "cf":
        rows = CFT.load(cf_file)
        return [CFT.task_spec(rows[i]) for i in ids], {i: CFT.lcb_meta(rows[i]) for i in ids}, (GEN.LCB_TEMPLATE_ID,)
    if cohort == "dev":      # TIMING SMOKE ONLY: development-bank tasks (never a P6 task), runtime contracts from the allowlisted declaration parser
        import el_bank_local_run as RUN
        return [TaskSpec(r["task_id"], "bank:" + r["task_id"], "python", "python", r["prompt"], r["public_tests"], r.get("difficulty", "")) for r in RUN.load_task_ids(ids)], {}, tuple(GEN.FROZEN_STRATEGIES)
    recs = {json.loads(l)["task_id"]: json.loads(l) for l in (ROOT / "configs/p6_contract_tasks_v1/tasks.jsonl").read_text("utf-8").splitlines() if l.strip()}
    return [TaskSpec(i, "p6:" + i, "python", "python", recs[i]["prompt"], recs[i]["public_tests"], recs[i]["difficulty"]) for i in ids], {}, tuple(GEN.FROZEN_STRATEGIES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--cohort", required=True, choices=("cf", "contract", "dev")); ap.add_argument("--tasks", required=True)
    ap.add_argument("--exec", default="mxc", choices=("mxc", "kaggle")); ap.add_argument("--cf-file", default=str(CFT.PUBLIC_FILE))
    ap.add_argument("--model-dir"); ap.add_argument("--model-manifest"); ap.add_argument("--profile", default="qwen2.5-coder-1.5b-instruct"); ap.add_argument("--fake", action="store_true")
    ap.add_argument("--script", help="TESTS ONLY: JSON list of raw model outputs replayed in call order (the last one repeats)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    tasks, meta, strategies = load_tasks(a.cohort, a.tasks.split(","), a.cf_file)
    out.mkdir(parents=True)
    from evidence_ladder.controller.sandbox_provider import make_sandbox
    from evidence_ladder.execution.lcb_runner import LcbExecutionBackend, LcbRunner
    from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
    if a.exec == "kaggle":
        from evidence_ladder.execution.kaggle_port import KaggleJailPort
        port = KaggleJailPort()
        exe = (LcbExecutionBackend(LcbRunner(port.run, out / "_lcb"), meta) if a.cohort == "cf"
               else ProviderExecutionBackend(port, backend="kaggle-jail", scratch=out / "_engine", trusted=True))
    else:
        exe = (LcbExecutionBackend(LcbRunner(make_sandbox(out / "_sandbox", [], backend="mxc").run, out / "_lcb"), meta) if a.cohort == "cf"
               else ProviderExecutionBackend(backend="mxc", scratch=out / "_engine", trusted=True))
    if hasattr(exe, "start"):
        exe.start(languages=("python",))
    gate = V.ContractGate(exe, REGISTRY if a.cohort == "contract" else None, ROOT, parsers=("decl_v1",))
    names = ("CALLS.jsonl", "RESULTS.jsonl", "ANSWERS.jsonl", "IDENTITY.jsonl", "EXEC.jsonl")
    files = {n: open(out / n, "a", encoding="utf-8") for n in names}
    w = lambda n: (lambda row: (files[n].write(json.dumps(row) + "\n"), files[n].flush()))
    exe = P5.ExecLog(exe, w("EXEC.jsonl"))
    if a.script:
        inner = ScriptGen(json.loads(Path(a.script).read_text("utf-8")))
    elif a.fake:
        inner = P5.FakeGen()
    else:
        import os
        from evidence_ladder.models import backbone as BB
        man = json.loads(Path(a.model_manifest).read_text(encoding="utf-8"))
        verify = {f: m["sha256"] for f, m in man["files"].items() if f.endswith((".safetensors", ".json")) and f != "MANIFEST_pinned.json"}
        os.environ.setdefault(Q.HFQwenBackend.INFERENCE_ENV, "1")
        inner = Q.HFQwenBackend(authorized=True, model_dir=a.model_dir, log_path=str(out / "gen_log.jsonl"), verify_files=verify, profile=BB.get(a.profile), local=True)
    calls = []
    log_calls = w("CALLS.jsonl")
    cache = Cache(inner, lambda r: (calls.append(r), log_calls(r)))
    exe.who = lambda: (cache.task, cache.arm)
    (out / "RUN_CONFIG.json").write_text(json.dumps({"protocol": "P6_PROTOCOL_v1", "cohort": a.cohort, "tasks": [t.task_id for t in tasks], "arms": ["A", "B", "V2"], "G": G,
                                                     "v2min": V.VERSION, "template": strategies[0] if a.cohort == "cf" else "qwen_sample_v2+v2b", "exec": a.exec, "fake": a.fake or bool(a.script),
                                                     "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                     "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    for task in tasks:
        cache.task = task.task_id
        t_task = time.monotonic()
        rp = lambda art, _t=task: exe.run_public(art, _t)
        n_pub = len(task.public_tests)
        cache.arm = "round1"
        r1 = []
        for rq in GEN.plan_samples(task, len(SEEDS1), list(SEEDS1), 0, strategies, MAX_NEW):
            r1 += list(cache.generate(rq))
        r1_tok = sum(g.tokens_generated for g in r1)
        arm_calls = lambda arm: [c for c in calls if c["task"] == task.task_id and c["arm"] == arm]

        def answer(arm, src, label, extra):
            row = {"cohort": a.cohort, "task": task.task_id, "arm": arm, "label": label, "answer_sha256": sha(src) if src else None, **extra}
            w("RESULTS.jsonl")(row); w("ANSWERS.jsonl")({"cohort": a.cohort, "task": task.task_id, "arm": arm, "source": src})
            print(json.dumps(row), flush=True)
        # ---- A: the seed-1 sample, never replaced
        cache.arm = "A"
        g1 = next(g for g in r1 if g.seed == 1)
        src_a = g1.code if (g1.extract_status == "ok" and g1.code.strip()) else None
        pub_a = V.public_result(rp(Artifact.make(task.task_id, "python", src_a, None, "sample:1")), n_pub)["pass_all"] if src_a else None
        answer("A", src_a, "ANSWERED" if src_a else "NO_CODE", {"public_pass": pub_a, "tokens_generated_logical": g1.tokens_generated, "tokens_generated_own": 0,
                                                              "tokens_prompt_logical": g1.tokens_prompt, "calls": 0, "candidates": 1 if src_a else 0, "rounds": 1,
                                                              "seconds": 0.0, "gate_verdicts": [], "rejected_by_contract": [], "stop": "a_single_sample"})
        res = {}
        for arm, g in (("B", None), ("V2", gate)):
            cache.arm = arm
            t0 = time.monotonic()
            r = V.sample_loop(cache, rp, task, G, strategies=strategies, gate=g)
            sel = r["selected"]
            mine = arm_calls(arm)
            res[arm] = {"r": r, "calls": mine}
            answer(arm, sel["source"] if sel else None, "SUPPORTED" if sel else "BLOCKED",
                   {"public_pass": None, "tokens_generated_logical": r["tokens_generated"], "tokens_generated_own": sum(c["tokens_generated"] for c in mine if not c["reused"]),
                    "tokens_prompt_logical": r["tokens_prompt"], "calls": len(mine), "candidates": len(r["candidates"]), "rounds": r["rounds"],
                    "seconds": round(time.monotonic() - t0, 1), "gate_verdicts": [x["contract"] for x in r["gate_log"]],
                    "rejected_by_contract": [x["candidate"] for x in r["gate_log"] if x["contract"] == "fail"], "stop": "accepted" if sel else "budget"})
        if a.cohort == "cf":
            sig = lambda cs: [(c["seed"], c["prompt_sha256"], c["max_new_tokens"], c["purpose"], c["template"]) for c in cs]
            bsel, vsel = res["B"]["r"]["selected"], res["V2"]["r"]["selected"]
            verdicts = [x["contract"] for x in res["V2"]["r"]["gate_log"]]
            idn = {"task": task.task_id, "same_answer": (bsel or {}).get("source") == (vsel or {}).get("source"),
                   "same_requests": sig(res["B"]["calls"]) == sig(res["V2"]["calls"]),
                   "same_tokens": res["B"]["r"]["tokens_generated"] == res["V2"]["r"]["tokens_generated"],
                   "verdicts": verdicts, "all_unavailable": all(v == "unavailable" for v in verdicts),
                   "v2_all_reused": all(c["reused"] for c in res["V2"]["calls"])}
            idn["identical"] = all(idn[k] for k in ("same_answer", "same_requests", "same_tokens", "all_unavailable", "v2_all_reused"))
            w("IDENTITY.jsonl")(idn)
        w("RESULTS.jsonl")({"cohort": a.cohort, "task": task.task_id, "arm": "_task", "wall_seconds": round(time.monotonic() - t_task, 1), "round1_tokens_generated": r1_tok})
    (out / "DONE.json").write_text(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    for f in files.values():
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
