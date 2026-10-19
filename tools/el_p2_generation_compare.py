"""[P2] Paired generation comparison at equal generated-token budget (amended contract 2026-10-20, conditional GO with five corrections).

Arms (same tasks, same shared round 1, same seeds per later round, same sample templates, same budget, same stop and selection rule):
  fresh                    : later rounds = the unchanged sample prompt (frozen qwen_sample_v2 / v2b cycle), new seeds.
  failure                  : the same prompt + a FAILURE NOTE about this arm's earlier candidates: failure class and error text per failing public example
                             (no earlier code, no observed values).
  counterexample_guided    : the same prompt + ONE public counterexample from this arm's best earlier candidate (input, expected, observed or error);
                             the earlier code is NOT shown (so this is guided resampling, not repair).
Round 1 (shared): the run's STORED first fresh samples (purpose "sample", seeds 1-4) from gen_calls.json, with their recorded tokens and seconds; a sample
without code (J1-33 seed 4 hit the 4,096 ceiling) stays in the accounting and contributes no candidate.

Corrections of the conditional GO (each checked by tools/test_el_p2_runner.py):
  1. the generation backend's language is set for every task and checked before any call (Java answers are extracted as Java);
  2. one sample per model request, so nothing is generated after the first public pass (no partially used batch exists);
  3. strict budget: each request asks for min(4096, budget - used) new tokens, so an arm never exceeds 6,000 generated tokens beyond round 1;
  4. PREFLIGHT before any generation: every task's rendered sample prompts (which contain the public examples) must equal the stored round-1 prompt sha256s;
  5. a public pass needs parse == pass and exactly one passing outcome for every public example index 0..n-1 (complete, unique).
Runtime oracle: public examples only (python: MXC process container; java: MXC WSLC; trusted verifier v2). The development grader is NOT imported here;
grading happens afterwards, offline, by tools/el_p2_grade.py. No learned scorer (LAYA correctness scoring disabled). The controller is not changed or used.

Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_p2_generation_compare.py <new out dir> --tasks <run dir>:<task id> [...] [--budget 6000] [--round-n 2] [--fake]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evidence_ladder.common.interfaces import Artifact, GenerationRequest, TaskSpec
from evidence_ladder.controller import generation as GEN

ARMS = ("fresh", "failure", "counterexample_guided")
BACKEND = {"python": "mxc", "java": "wslc"}
MAX_NEW = 4096
ROOT = Path(__file__).resolve().parents[1]


class PreflightError(RuntimeError):
    pass


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def public_result(outs: List[Dict[str, Any]], n: int) -> Dict[str, Any]:
    """Correction 5: success only with parse pass and exactly one passing outcome for each example index 0..n-1."""
    ex = [o for o in outs if o.get("check_id") == "public_examples_v1" and "example_index" in o]
    parse = next((o for o in outs if o.get("check_id") == "parse_v1"), {})
    idx = [o["example_index"] for o in ex]
    complete = sorted(idx) == list(range(n)) and n > 0
    ok = complete and parse.get("status") == "pass" and all(o.get("status") == "pass" for o in ex)
    return {"pass_all": ok, "complete": complete, "passed": sum(o.get("status") == "pass" for o in ex), "total": n, "parse": parse.get("status"),
            "examples": [{"i": o["example_index"], "status": o.get("status"), "failure_class": o.get("failure_class", ""), "observed": str(o.get("observed", ""))[:300],
                          "error": str(o.get("stderr_tail") or o.get("detail") or "")[-300:]} for o in ex]}


def failure_note(history: List[Dict[str, Any]]) -> str:
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
    if not history:
        return ""
    best = sorted(history, key=lambda h: (-h["public"]["passed"], h["order"]))[0]
    fail = next((e for e in sorted(best["public"]["examples"], key=lambda e: e["i"]) if e["status"] != "pass"), None)
    if fail is None:
        return ""
    t = task.public_tests[fail["i"]]
    obs = fail["observed"] or (fail["failure_class"] or fail["status"])
    err = (fail["error"].strip().splitlines() or [""])[-1][:160]
    if obs.startswith("<") and err:
        obs += " " + err
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


def preflight_prompts(task: TaskSpec, r1: List[Dict[str, Any]]) -> Dict[str, str]:
    """Correction 4: the prompts this run would render (task text + public examples + solution format) must equal the stored round-1 prompts."""
    seen = {}
    for g in r1:
        mine = sha(GEN.render_sample_prompt(g["template"], task)[0])
        if mine != g["prompt_sha256"]:
            raise PreflightError("%s: rendered %s prompt sha %s != stored %s" % (task.task_id, g["template"], mine[:12], g["prompt_sha256"][:12]))
        seen[g["template"]] = mine
    if not seen:
        raise PreflightError("%s: no stored round-1 sample" % task.task_id)
    return seen


def set_language(gen: Any, lang: str) -> None:
    """Correction 1."""
    gen.language = lang
    if getattr(gen, "language", None) != lang:
        raise PreflightError("generation backend language is %r, task needs %r" % (getattr(gen, "language", None), lang))


def run_arm(gen: Any, run_public: Callable[[Artifact], List[Dict[str, Any]]], task: TaskSpec, arm: str, r1_hist: List[Dict[str, Any]], n_r1: int,
            budget: int, round_n: int, log_call: Callable[[Dict[str, Any]], None], log_cand: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    lang = task.language
    hist = [dict(h) for h in r1_hist]
    for h in hist:
        log_cand({"task": task.task_id, "arm": arm, "round": 1, "id": h["id"], "seed": h["seed"], "public_pass": h["public"]["pass_all"], "public_passed": h["public"]["passed"], "source": h["source"]})
    used_gen = used_prompt = 0
    model_s = sandbox_s = 0.0
    calls = ceiling_hits = budget_truncated = dups = 0
    sel = next((h for h in hist if h["public"]["pass_all"]), None)
    rnd = 1
    while sel is None and used_gen < budget:
        rnd += 1
        note = "" if arm == "fresh" else (failure_note(hist) if arm == "failure" else counterexample_note(hist, task))
        for j in range(round_n):
            if sel is not None or used_gen >= budget:          # corrections 2 and 3: no request after a selection or at the ceiling
                break
            seed = 100 + round_n * rnd + j
            template = GEN.strategy_for(n_r1 + (rnd - 2) * round_n + j)
            prompt = GEN.render_sample_prompt(template, task)[0] + note
            mx = min(MAX_NEW, budget - used_gen)
            rq = GenerationRequest(task.task_id, prompt, 1, (seed,), mx, "sample", True, template + ("" if arm == "fresh" else "+" + arm))
            if getattr(gen, "language", None) != lang:
                raise PreflightError("backend language changed during the run")
            t0 = time.monotonic()
            g = list(gen.generate(rq))[0]
            el = time.monotonic() - t0
            calls += 1
            used_gen += int(g.tokens_generated); used_prompt += int(g.tokens_prompt); model_s += el
            hit = int(g.tokens_generated) >= mx
            ceiling_hits += int(hit and mx == MAX_NEW); budget_truncated += int(hit and mx < MAX_NEW)
            log_call({"task": task.task_id, "arm": arm, "round": rnd, "seed": seed, "template": rq.prompt_template_id, "prompt_sha256": sha(prompt), "max_new_tokens": mx,
                      "extract_status": g.extract_status, "tokens_prompt": g.tokens_prompt, "tokens_generated": g.tokens_generated, "backend_seconds": round(g.wall_seconds or 0.0, 3),
                      "elapsed_seconds": round(el, 3), "hit_max_new": hit, "stored": False, "note": note})
            if g.extract_status != "ok" or not g.code.strip():
                continue
            art = Artifact.make(task.task_id, lang, g.code, None, "%s:r%d:s%d" % (arm, rnd, seed))
            if any(h["id"] == art.artifact_id for h in hist):
                dups += 1
                continue
            te = time.monotonic()
            pub = public_result(run_public(art), len(task.public_tests))
            sandbox_s += time.monotonic() - te
            h = {"id": art.artifact_id, "order": len(hist), "round": rnd, "seed": seed, "public": pub, "source": g.code}
            hist.append(h)
            log_cand({"task": task.task_id, "arm": arm, "round": rnd, "id": h["id"], "seed": seed, "public_pass": pub["pass_all"], "public_passed": pub["passed"],
                      "public_complete": pub["complete"], "source": g.code})
            if pub["pass_all"]:
                sel = h
    return {"task": task.task_id, "language": lang, "arm": arm, "label": "SUPPORTED" if sel else "BLOCKED", "selected": sel["id"] if sel else None,
            "selected_round": sel["round"] if sel else None, "candidates": len(hist), "rounds": rnd, "calls": calls, "arm_tokens_generated": used_gen, "arm_tokens_prompt": used_prompt,
            "budget": budget, "budget_left": budget - used_gen, "ceiling_hits_4096": ceiling_hits, "budget_truncated": budget_truncated, "duplicates": dups,
            "model_seconds": round(model_s, 2), "sandbox_seconds": round(sandbox_s, 2)}


class FakeGen:
    """--fake: scripted model (no GPU). Answers with a failing program in the task's language; tokens = min(700, max_new_tokens)."""
    language = "python"

    def generate(self, rq: GenerationRequest):
        from evidence_ladder.models.qwen import make_generation
        code = ("def solve(data):\n    return None\n" if self.language == "python" else
                "import java.util.*;\npublic final class Solution {\n    public static Object solve(Map<String,Object> data) { return null; }\n}\n")
        return [make_generation("<think>\n\n</think>\n\n```%s\n%s```\n" % (self.language, code), tokens_prompt=len(rq.prompt) // 4, tokens_generated=min(700, rq.max_new_tokens),
                                seed=s, prompt_template_id=rq.prompt_template_id, prompt_sha256=sha(rq.prompt), language=self.language, wall_seconds=0.0) for s in rq.seeds[:rq.n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--round-n", type=int, default=2)
    ap.add_argument("--profile", default="qwen2.5-coder-1.5b-instruct")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--task-records", help="public-only task records (jsonl: task_id, language, difficulty, prompt, public_tests); default = the sha-pinned bank sources via el_bank_local_run")
    ap.add_argument("--exec", default="local", choices=("local", "kaggle"), help="local: MXC process container / MXC WSLC; kaggle: the Kaggle jail (chroot + uid drop + seccomp)")
    ap.add_argument("--model-dir", help="model directory (Kaggle); default = the profile's local pinned snapshot")
    ap.add_argument("--model-manifest", help="MANIFEST_pinned.json whose sha256s the model files must match (with --model-dir)")
    a = ap.parse_args()
    if a.task_records:
        recs = {d["task_id"]: d for d in map(json.loads, Path(a.task_records).read_text(encoding="utf-8").splitlines())}
        load_ids = lambda ids: [recs[i] for i in ids]
    else:
        sys.path.insert(0, str(ROOT / "tools"))
        import el_bank_local_run as RUN      # task loader with sha-pinned sources (unchanged)
        load_ids = RUN.load_task_ids
    out = Path(a.out).resolve()
    if out.exists():
        print("refusing: %s exists (create-only)" % out); return 2
    # ---- PREFLIGHT for every task before any model call (correction 4); any failure stops the whole run
    plan = []
    for spec in a.tasks:
        run_s, tid = spec.rsplit(":", 1)
        rec = load_ids([tid])[0]
        task = TaskSpec(tid, ("py1:" if rec["language"] == "python" else "java1:") + tid, rec["language"], rec["language"], rec["prompt"], rec["public_tests"], rec.get("difficulty", ""))
        r1 = stored_round1(Path(run_s), tid)
        preflight_prompts(task, r1)
        plan.append((task, r1))
    out.mkdir(parents=True)
    contract = {"arms": ARMS, "budget_generated_tokens_beyond_round1": a.budget, "round_n": a.round_n, "max_new_tokens_per_sample": MAX_NEW, "profile": a.profile, "fake_model": a.fake,
                "seeds": "round r>=2: 100+round_n*r+j, identical across arms", "requests": "one sample per request", "selection": "first candidate with a complete passing public result",
                "scorer": "none (LAYA correctness scoring disabled)", "oracle": "public examples only; development grader not imported", "preflight": "all %d task prompts equal the stored round-1 prompts" % len(plan),
                "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "CONTRACT.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
    if a.fake:
        gen = FakeGen()
    elif a.model_dir:
        from evidence_ladder.models import qwen as Q, backbone as BB
        man = json.loads(Path(a.model_manifest).read_text(encoding="utf-8"))
        verify = {f: m["sha256"] for f, m in man["files"].items() if f.endswith((".safetensors", ".json")) and f != "MANIFEST_pinned.json"}
        os.environ.setdefault(Q.HFQwenBackend.INFERENCE_ENV, "1")
        gen = Q.HFQwenBackend(authorized=True, model_dir=a.model_dir, log_path=str(out / "gen_log.jsonl"), verify_files=verify, profile=BB.get(a.profile), local=True)
    else:
        gen = __import__("evidence_ladder.models.qwen", fromlist=["x"]).build_local_backend(a.profile, log_path=str(out / "gen_log.jsonl"), think_policy=None)
    contract["exec"], contract["model_dir"] = a.exec, a.model_dir or "profile local snapshot"
    (out / "CONTRACT.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
    exes: Dict[str, Any] = {}
    files = {n: open(out / n, "a", encoding="utf-8") for n in ("CALLS.jsonl", "CANDIDATES.jsonl", "RESULTS.jsonl")}

    def w(name):
        return lambda row: (files[name].write(json.dumps(row) + "\n"), files[name].flush())
    t_run = time.monotonic()
    for task, r1 in plan:
        lang = task.language
        set_language(gen, lang)                                                     # correction 1
        if lang not in exes:
            if a.exec == "kaggle":
                from evidence_ladder.execution.artifacts import ArtifactStore
                from evidence_ladder.execution.backend import SandboxExecutionBackend
                from evidence_ladder.execution import config as K
                exes[lang] = SandboxExecutionBackend(ArtifactStore(out / "_store" / lang), K.SandboxConfig())       # the Kaggle jail (EL_JAIL=kaggle, set by the kernel)
            else:
                from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
                exes[lang] = ProviderExecutionBackend(backend=BACKEND.get(lang, "wslc"), scratch=out / "_engine" / lang, trusted=True)
                exes[lang].start(languages=(lang,))
        exe = exes[lang]
        rp = lambda art, _t=task, _e=exe: _e.run_public(art, _t)
        r1_hist = []
        for k, g in enumerate(r1):
            w("CALLS.jsonl")({"task": task.task_id, "arm": "shared", "round": 1, **{x: g[x] for x in ("seed", "template", "prompt_sha256", "extract_status", "tokens_prompt", "tokens_generated", "backend_seconds", "elapsed_seconds", "stored")}})
            if g["extract_status"] != "ok" or not g["code"].strip():
                continue
            art = Artifact.make(task.task_id, lang, g["code"], None, "sample:%d" % g["seed"])
            r1_hist.append({"id": art.artifact_id, "order": k, "round": 1, "seed": g["seed"], "public": public_result(rp(art), len(task.public_tests)), "source": g["code"]})
        r1_cost = {"round1_tokens_generated": sum(g["tokens_generated"] for g in r1), "round1_tokens_prompt": sum(g["tokens_prompt"] for g in r1),
                   "round1_backend_seconds": round(sum(g["backend_seconds"] or 0 for g in r1), 1), "round1_samples": len(r1), "round1_candidates": len(r1_hist),
                   "round1_public_pass": sum(h["public"]["pass_all"] for h in r1_hist)}
        for arm in ARMS:
            row = run_arm(gen, rp, task, arm, r1_hist, len(r1), a.budget, a.round_n, w("CALLS.jsonl"), w("CANDIDATES.jsonl")) | r1_cost
            w("RESULTS.jsonl")(row)
            print(json.dumps(row), flush=True)
    (out / "DONE.json").write_text(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_seconds": round(time.monotonic() - t_run, 1)}), encoding="utf-8")
    for f in files.values():
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
