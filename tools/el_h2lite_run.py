"""[H2-lite, H2LITE_PLAN_v1.md] Generation runner for the search and confirmation phases. Selection only: NO grader is imported here.
Every configuration runs arm B exactly (v2min.sample_loop, no gate; P5 seeds, round 1 = 4 samples, G = 6,000 generated tokens beyond round 1, the shared
generation cache with exact reuse), the frozen model, the same sandbox; a configuration differs ONLY by its PromptAdapter (none = baseline B).
Per task and configuration it records the runtime-visible search signals (§5): accepted (a candidate passed every public example), the decl_v1 runtime contract
verdict on the accepted answer, extraction failures, actual generated tokens, seconds. Answers go to ANSWERS.jsonl for the offline confirmation grade (§6).
Usage: python tools/el_h2lite_run.py <new out dir> --phase search|confirm --configs baseline,general,coding --proposals <round json>[,<round json>...]
       [--exec kaggle|mxc] [--model-dir D --model-manifest M] [--fake]"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import el_p5_run as P5                                                          # noqa: E402
from evidence_ladder import v2min as V                                          # noqa: E402
from evidence_ladder.common.interfaces import Artifact, TaskSpec               # noqa: E402
from evidence_ladder.controller import generation as GEN                       # noqa: E402
from evidence_ladder.models import qwen as Q                                    # noqa: E402

SEARCH = ["P1-%02d" % i for i in (2, 4, 6, 7, 9, 12, 29, 30, 35, 38, 39, 11, 21, 23, 26, 34, 16, 28, 37, 14, 20, 25)]
CONFIRM = ["P1-%02d" % i for i in (1, 3, 5, 8, 10, 13, 19, 24, 17, 27, 36, 40, 18, 31, 32, 33, 15, 22)]
G = 6000
TASKS = ROOT / "results/raw/codex-pilot1-20261016/fixed_v2/worker/tasks.jsonl"


class CountingGen:
    """Wraps the shared-cache generator; counts extraction failures per configuration (runtime-visible)."""
    language = "python"

    def __init__(self, inner):
        self.inner, self.extract_fail, self.samples = inner, 0, 0

    def generate(self, rq):
        out = list(self.inner.generate(rq))
        self.samples += len(out)
        self.extract_fail += sum(1 for g in out if g.extract_status != "ok")
        return out


def adapters(names, proposal_files):
    props = {}
    for f in proposal_files:
        for v in json.loads(Path(f).read_text("utf-8"))["variants"]:
            props[v["name"] if v["name"] not in props else "%s@%s" % (v["name"], Path(f).stem)] = v
    out = {}
    for n in names:
        if n == "baseline":
            out[n] = None
        else:
            v = props[n]
            ad = V.PromptAdapter("h2_" + n, v["texts"])
            assert ad.texts == v["texts"]
            out[n] = ad
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--phase", required=True, choices=("search", "confirm")); ap.add_argument("--configs", required=True)
    ap.add_argument("--proposals", default=""); ap.add_argument("--exec", default="mxc", choices=("mxc", "kaggle"))
    ap.add_argument("--model-dir"); ap.add_argument("--model-manifest"); ap.add_argument("--profile", default="qwen2.5-coder-1.5b-instruct"); ap.add_argument("--fake", action="store_true")
    ap.add_argument("--tasks", default="", help="override the task list (smoke only)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    out.mkdir(parents=True)
    ids = a.tasks.split(",") if a.tasks else (SEARCH if a.phase == "search" else CONFIRM)
    if a.phase == "search" and set(ids) & set(CONFIRM):
        sys.exit("refusing: a confirmation task in the search phase")
    rows = {json.loads(l)["task_id"]: json.loads(l) for l in TASKS.read_text("utf-8").splitlines() if l.strip()}
    tasks = [TaskSpec(i, "bank:" + i, "python", "python", rows[i]["prompt"], rows[i]["public_tests"], rows[i].get("difficulty", "")) for i in ids]
    cfgs = adapters(a.configs.split(","), [p for p in a.proposals.split(",") if p])
    from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
    if a.exec == "kaggle":
        from evidence_ladder.execution.kaggle_port import KaggleJailPort
        exe = ProviderExecutionBackend(KaggleJailPort(), backend="kaggle-jail", scratch=out / "_engine", trusted=True)
    else:
        exe = ProviderExecutionBackend(backend="mxc", scratch=out / "_engine", trusted=True)
    exe.start(languages=("python",))
    gate = V.ContractGate(exe, None, ROOT, parsers=("decl_v1",))           # runtime contracts on the development tasks: allowlisted declaration parser only
    files = {n: open(out / n, "a", encoding="utf-8") for n in ("CALLS.jsonl", "RESULTS.jsonl", "ANSWERS.jsonl")}
    w = lambda n: (lambda row: (files[n].write(json.dumps(row) + "\n"), files[n].flush()))
    if a.fake:
        inner = P5.FakeGen()
    else:
        import os
        from evidence_ladder.models import backbone as BB
        man = json.loads(Path(a.model_manifest).read_text(encoding="utf-8"))
        verify = {f: m["sha256"] for f, m in man["files"].items() if f.endswith((".safetensors", ".json")) and f != "MANIFEST_pinned.json"}
        os.environ.setdefault(Q.HFQwenBackend.INFERENCE_ENV, "1")
        inner = Q.HFQwenBackend(authorized=True, model_dir=a.model_dir, log_path=str(out / "gen_log.jsonl"), verify_files=verify, profile=BB.get(a.profile), local=True)
    cache = P5.CacheGen(inner, w("CALLS.jsonl"))
    (out / "RUN_CONFIG.json").write_text(json.dumps({"plan": "H2LITE_PLAN_v1", "phase": a.phase, "tasks": ids, "configs": {n: (ad.version, ad.sha256) if ad else None for n, ad in cfgs.items()},
                                                     "G": G, "policy": "arm B (v2min.sample_loop, no gate)", "exec": a.exec, "fake": a.fake,
                                                     "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "v2min": V.VERSION,
                                                     "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    summary = {}
    for name, ad in cfgs.items():
        cache.arm = name
        cg = CountingGen(cache)
        acc = viol = tok = 0
        t_cfg = time.monotonic()
        for t in tasks:
            cache.task = t.task_id
            t0 = time.monotonic()
            r = V.sample_loop(cg, lambda art, _t=t: exe.run_public(art, _t), t, G, strategies=tuple(GEN.FROZEN_STRATEGIES), adapter=ad)
            sel = r["selected"]
            verdict = gate(t, Artifact.make(t.task_id, "python", sel["source"])) if sel else None
            acc += sel is not None; viol += verdict == "fail"; tok += r["tokens_generated"]
            row = {"config": name, "task": t.task_id, "accepted": sel is not None, "contract": verdict, "tokens_generated": r["tokens_generated"],
                   "tokens_prompt": r["tokens_prompt"], "candidates": len(r["candidates"]), "rounds": r["rounds"], "seconds": round(time.monotonic() - t0, 1)}
            w("RESULTS.jsonl")(row); w("ANSWERS.jsonl")({"config": name, "task": t.task_id, "source": sel["source"] if sel else None})
            print(json.dumps(row), flush=True)
        summary[name] = {"accepted": acc, "contract_violations_among_accepted": viol, "extraction_failures": cg.extract_fail, "samples": cg.samples,
                         "actual_generated_tokens": tok, "seconds": round(time.monotonic() - t_cfg, 1), "score": acc - 2 * viol,
                         "adapter": (ad.version, ad.sha256) if ad else None}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    (out / "DONE.json").write_text(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
