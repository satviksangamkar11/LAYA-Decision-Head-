"""[P2, PREPARED 2026-10-20, NOT yet approved to run on the GPU] Paired generation comparison at equal generated-token budget.

Arms (same tasks, same shared round 1, same seeds per later round, same sample templates, same budget, same stop and selection rule):
  A fresh       : later rounds = the unchanged sample prompt (qwen_sample_v2 / v2b strategy cycle), new seeds.
  B failure     : the same prompt + a FAILURE NOTE about this arm's earlier candidates: failure class and error text per failing public example index
                  (no earlier code, no observed values).
  C counterex   : the same prompt + ONE counterexample from this arm's best earlier candidate (most public examples passed, ties -> earliest):
                  the public example input, expected and observed values, and "write a new complete solution"; the earlier code is NOT shown.
Round 1 (shared by all arms): the run's STORED first fresh samples (purpose "sample", seeds 1-4) from gen_calls.json, with their recorded token counts and seconds;
a sample that ended without code (J1-33 seed 4 hit the 4,096-token ceiling) is kept in the accounting and contributes no candidate.
Runtime oracle only: public examples, run in the original sandbox (python: MXC process container; java: MXC WSLC) through the trusted verifier v2.
The development grader is NOT imported here; grading happens afterwards, offline, by a separate tool over this run's output.
No learned scorer of any kind (LAYA correctness scoring stays disabled). The Evidence-Ladder controller is not changed and not used: this is an experiment harness.

Per arm and task: stop at the first candidate that passes every public example (selected; label SUPPORTED) or when the arm's generated tokens beyond round 1 reach the
budget (label BLOCKED; a round already started is finished and its overshoot is reported). Every model call is logged: prompt tokens, generated tokens, backend seconds,
measured elapsed seconds, seed, template, prompt sha256, extract status.

Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_p2_generation_compare.py <new out dir> --tasks <run dir>:<task id> [...] [--budget 6000] [--round-n 2]
       [--profile qwen2.5-coder-1.5b-instruct] [--fake]   (--fake = scripted model, no GPU: plumbing check only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidence_ladder.common.interfaces import Artifact, GenerationRequest, TaskSpec
from evidence_ladder.controller import generation as GEN
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend

ARMS = ("fresh", "failure", "counterexample")
BACKEND = {"python": "mxc", "java": "wslc"}
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import el_bank_local_run as RUN      # noqa: E402  (task loader with pinned sources; the runner's code is not changed)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def public_result(exe, art: Artifact, task: TaskSpec) -> Dict[str, Any]:
    outs = exe.run_public(art, task)
    ex = [o for o in outs if o.get("check_id") == "public_examples_v1" and "example_index" in o]
    parse = next((o for o in outs if o.get("check_id") == "parse_v1"), {})
    return {"pass_all": bool(ex) and all(o.get("status") == "pass" for o in ex), "passed": sum(o.get("status") == "pass" for o in ex), "total": len(task.public_tests),
            "parse": parse.get("status"), "examples": [{"i": o["example_index"], "status": o.get("status"), "failure_class": o.get("failure_class", ""),
                                                        "observed": str(o.get("observed", ""))[:300], "error": str(o.get("stderr_tail") or o.get("detail") or "")[-300:]} for o in ex]}


def failure_note(history: List[Dict[str, Any]]) -> str:
    """B: classes and error text of this arm's failing public examples so far; no code, no observed values."""
    lines = []
    for h in history:
        for e in h["public"]["examples"]:
            if e["status"] != "pass":
                err = (e["error"].strip().splitlines() or [""])[-1][:160]
                lines.append("- an earlier attempt failed public example %d: %s%s" % (e["i"] + 1, e["failure_class"] or e["status"], (" (" + err + ")") if err else ""))
        if h["public"]["parse"] not in (None, "pass"):
            lines.append("- an earlier attempt did not compile/load (%s)" % h["public"]["parse"])
    uniq = list(dict.fromkeys(lines))[:8]
    return ("\n\nNotes from earlier attempts (their code is not shown):\n" + "\n".join(uniq) + "\nWrite a new, complete solution that avoids these failures.") if uniq else ""


def counterexample_note(history: List[Dict[str, Any]], task: TaskSpec) -> str:
    """C: one public counterexample from the best earlier candidate; no code."""
    if not history:
        return ""
    best = sorted(history, key=lambda h: (-h["public"]["passed"], h["order"]))[0]
    fail = next((e for e in best["public"]["examples"] if e["status"] != "pass"), None)
    if fail is None:
        return ""
    t = task.public_tests[fail["i"]]
    obs = fail["observed"] or (fail["failure_class"] or fail["status"])
    err = (fail["error"].strip().splitlines() or [""])[-1][:160]
    if obs.startswith("<") and err:
        obs += " " + err                                  # a crash: the observed behaviour is the error itself
    return ("\n\nA previous attempt (code not shown) failed this example:\ninput: %s\nexpected: %s\nobserved: %s\nWrite a new, complete solution that handles it."
            % (json.dumps(t.get("input"))[:600], json.dumps(t.get("expected"))[:300], obs[:300]))


def stored_round1(run: Path, tid: str) -> List[Dict[str, Any]]:
    calls = json.loads((run / "tasks" / tid / "gen_calls.json").read_text(encoding="utf-8"))
    out = []
    for c in calls:
        if c["purpose"] != "sample":
            continue
        for g in c["generations"]:
            out.append({"seed": g["seed"], "template": c["prompt_template_id"], "prompt_sha256": c["prompt_sha256"], "code": g["code"], "extract_status": g["extract_status"],
                        "tokens_prompt": g["tokens_prompt"], "tokens_generated": g["tokens_generated"], "backend_seconds": g.get("wall_seconds"), "elapsed_seconds": None, "stored": True})
    return out


class FakeGen:
    """--fake: scripted model for plumbing checks (no GPU). Always answers with a program that fails, so every arm spends its budget."""
    langs: Dict[str, str] = {}

    def generate(self, rq: GenerationRequest):
        from evidence_ladder.models.qwen import make_generation
        lang = self.langs[rq.task_id]
        code = ("def solve(data):\n    return None\n" if lang == "python" else
                "import java.util.*;\npublic final class Solution {\n    public static Object solve(Map<String,Object> data) { return null; }\n}\n")
        return [make_generation("<think>\n\n</think>\n\n```%s\n%s```\n" % (lang, code), tokens_prompt=len(rq.prompt) // 4, tokens_generated=700, seed=s,
                                prompt_template_id=rq.prompt_template_id, prompt_sha256=sha(rq.prompt), language=lang, wall_seconds=0.0) for s in rq.seeds[:rq.n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=6000, help="generated tokens per arm per task beyond the shared round 1")
    ap.add_argument("--round-n", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--profile", default="qwen2.5-coder-1.5b-instruct")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        print("refusing: %s exists (create-only)" % out); return 2
    out.mkdir(parents=True)
    contract = {"arms": ARMS, "budget_generated_tokens_beyond_round1": a.budget, "round_n": a.round_n, "max_new_tokens": a.max_new_tokens, "profile": a.profile,
                "fake_model": a.fake, "seeds": "round r>=2 uses seeds 100+%d*r .. (same for every arm)" % a.round_n, "selection": "first candidate passing every public example",
                "scorer": "none (LAYA correctness scoring disabled)", "oracle": "public examples only; development grader not imported", "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "CONTRACT.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
    gen = FakeGen() if a.fake else __import__("evidence_ladder.models.qwen", fromlist=["x"]).build_local_backend(a.profile, log_path=str(out / "gen_log.jsonl"), think_policy=None)
    exes: Dict[str, Any] = {}
    calls_f = open(out / "CALLS.jsonl", "a", encoding="utf-8")
    cands_f = open(out / "CANDIDATES.jsonl", "a", encoding="utf-8")
    res_f = open(out / "RESULTS.jsonl", "a", encoding="utf-8")
    for spec in a.tasks:
        run_s, tid = spec.rsplit(":", 1)
        run = Path(run_s)
        rec = RUN.load_task_ids([tid])[0]
        task = TaskSpec(tid, ("py1:" if rec["language"] == "python" else "java1:") + tid, rec["language"], rec["language"], rec["prompt"], rec["public_tests"], rec.get("difficulty", ""))
        lang = task.language
        if a.fake:
            gen.langs[tid] = lang
        if lang not in exes:
            exes[lang] = ProviderExecutionBackend(backend=BACKEND[lang], scratch=out / "_engine" / lang, trusted=True)
            exes[lang].start(languages=(lang,))
        exe = exes[lang]
        r1 = stored_round1(run, tid)
        r1_hist = []
        for k, g in enumerate(r1):
            calls_f.write(json.dumps({"task": tid, "arm": "shared", "round": 1, **{x: g[x] for x in ("seed", "template", "prompt_sha256", "extract_status", "tokens_prompt", "tokens_generated", "backend_seconds", "elapsed_seconds", "stored")}}) + "\n")
            if g["extract_status"] != "ok" or not g["code"].strip():
                continue
            art = Artifact.make(tid, lang, g["code"], None, "sample:%d" % g["seed"])
            pub = public_result(exe, art, task)
            r1_hist.append({"id": art.artifact_id, "order": k, "round": 1, "seed": g["seed"], "public": pub, "source": g["code"]})
        calls_f.flush()
        r1_gen, r1_prompt = sum(g["tokens_generated"] for g in r1), sum(g["tokens_prompt"] for g in r1)
        for arm in ARMS:
            hist = [dict(h) for h in r1_hist]
            for h in hist:
                cands_f.write(json.dumps({"task": tid, "arm": arm, "round": 1, "id": h["id"], "seed": h["seed"], "public_pass": h["public"]["pass_all"], "public_passed": h["public"]["passed"], "source": h["source"]}) + "\n")
            used_gen = used_prompt = 0
            secs = 0.0
            sel = next((h for h in hist if h["public"]["pass_all"]), None)
            rnd = 1
            while sel is None and used_gen < a.budget:
                rnd += 1
                seeds = [100 + a.round_n * rnd + j for j in range(a.round_n)]
                start_index = len(r1) + (rnd - 2) * a.round_n
                note = "" if arm == "fresh" else (failure_note(hist) if arm == "failure" else counterexample_note(hist, task))
                reqs = GEN.plan_samples(task, a.round_n, seeds, start_index, max_new_tokens=a.max_new_tokens)
                for rq in reqs:
                    if note:
                        rq = GenerationRequest(rq.task_id, rq.prompt + note, rq.n, rq.seeds, rq.max_new_tokens, rq.purpose, rq.thinking, rq.prompt_template_id + "+" + arm)
                    t0 = time.monotonic()
                    gens = list(gen.generate(rq))
                    el = time.monotonic() - t0
                    for g in gens:
                        used_gen += int(g.tokens_generated); used_prompt += int(g.tokens_prompt)
                        calls_f.write(json.dumps({"task": tid, "arm": arm, "round": rnd, "seed": g.seed, "template": rq.prompt_template_id, "prompt_sha256": sha(rq.prompt), "extract_status": g.extract_status,
                                                  "tokens_prompt": g.tokens_prompt, "tokens_generated": g.tokens_generated, "backend_seconds": round(g.wall_seconds or 0.0, 3),
                                                  "elapsed_seconds": round(el / max(1, len(gens)), 3), "stored": False, "note": note}) + "\n")
                        if g.extract_status != "ok" or not g.code.strip():
                            continue
                        art = Artifact.make(tid, lang, g.code, None, "%s:r%d:s%d" % (arm, rnd, g.seed))
                        if any(h["id"] == art.artifact_id for h in hist):
                            continue
                        te = time.monotonic()
                        pub = public_result(exe, art, task)
                        secs += time.monotonic() - te
                        h = {"id": art.artifact_id, "order": len(hist), "round": rnd, "seed": g.seed, "public": pub, "source": g.code}
                        hist.append(h)
                        cands_f.write(json.dumps({"task": tid, "arm": arm, "round": rnd, "id": h["id"], "seed": g.seed, "public_pass": pub["pass_all"], "public_passed": pub["passed"], "source": g.code}) + "\n")
                        if sel is None and pub["pass_all"]:
                            sel = h
                    calls_f.flush(); cands_f.flush()
            row = {"task": tid, "language": lang, "arm": arm, "label": "SUPPORTED" if sel else "BLOCKED", "selected": sel["id"] if sel else None, "selected_round": sel["round"] if sel else None,
                   "candidates": len(hist), "rounds": rnd, "round1_tokens_generated": r1_gen, "round1_tokens_prompt": r1_prompt, "arm_tokens_generated": used_gen, "arm_tokens_prompt": used_prompt,
                   "budget": a.budget, "overshoot": max(0, used_gen - a.budget), "sandbox_seconds": round(secs, 2)}
            res_f.write(json.dumps(row) + "\n"); res_f.flush()
            print(json.dumps(row), flush=True)
    for f in (calls_f, cands_f, res_f):
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
