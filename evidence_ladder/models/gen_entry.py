"""[EL-ARCH-1 B3] Kaggle-runnable generation entry, logic part (kaggle/el_arch1_generate_v1.py is the thin script that loads the model and calls this). Pure standard library apart from
the backend object it is given. A job list in, an append-only fsynced JSONL out (one record per (job, seed), resumable: finished (job, seed) pairs are never regenerated).

job = {"job_id", "task_id", "purpose" ("sample"|"repair"|"tests"), "prompt_template_id", "prompt", "prompt_sha256", "seeds": [int], "max_new_tokens", "thinking": true}
record = {"job_id", "task_id", "seed", "purpose", "prompt_sha256", "generation": {Generation fields}}   (the format evidence_ladder.controller.generation.ReplayGenerationBackend reads)
Every job's prompt hash is re-checked here (the PC pins it when it builds the job), so Kaggle cannot generate from a prompt the controller did not ask for."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping

from evidence_ladder.common.interfaces import GenerationRequest, sha256_text
from evidence_ladder.models import qwen as Q


class JobError(ValueError):
    pass


def make_job(request: GenerationRequest, job_id: str) -> Dict[str, Any]:
    """PC side: turn a GenerationRequest the controller built into a Kaggle job with its prompt hash pinned."""
    return {"job_id": job_id, "task_id": request.task_id, "purpose": request.purpose, "prompt_template_id": request.prompt_template_id, "prompt": request.prompt,
            "prompt_sha256": sha256_text(request.prompt), "seeds": [int(s) for s in request.seeds[:request.n]], "max_new_tokens": int(request.max_new_tokens), "thinking": bool(request.thinking)}


def check_job(job: Mapping[str, Any]) -> None:
    for k in ("job_id", "task_id", "purpose", "prompt_template_id", "prompt", "prompt_sha256", "seeds", "max_new_tokens"):
        if k not in job:
            raise JobError("job lacks %r" % k)
    if sha256_text(job["prompt"]) != job["prompt_sha256"]:
        raise JobError("job %s: prompt does not match its pinned sha256" % job["job_id"])
    if job["purpose"] not in ("sample", "repair", "tests"):
        raise JobError("job %s: unknown purpose %r" % (job["job_id"], job["purpose"]))
    if not job["seeds"] or any(not isinstance(s, int) for s in job["seeds"]):
        raise JobError("job %s: seeds must be a non-empty list of ints" % job["job_id"])


def done_pairs(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {(r["job_id"], int(r["seed"])) for r in (json.loads(l) for l in f if l.strip())}


def run_jobs(backend, jobs: Iterable[Mapping[str, Any]], out_path: str) -> Dict[str, Any]:
    jobs = list(jobs)
    for j in jobs:
        check_job(j)
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise JobError("duplicate job_id")
    done = done_pairs(out_path)
    generated = skipped = errors = 0
    err_list: List[Dict[str, Any]] = []
    for j in jobs:
        for seed in j["seeds"]:
            if (j["job_id"], seed) in done:
                skipped += 1
                continue
            req = GenerationRequest(j["task_id"], j["prompt"], 1, (seed,), int(j["max_new_tokens"]), j["purpose"], bool(j.get("thinking", True)), j["prompt_template_id"])
            try:
                g = backend.generate(req)[0]
            except Exception as ex:      # noqa: BLE001  a failed generation is recorded, the run goes on (results already written are never lost)
                errors += 1
                err_list.append({"job_id": j["job_id"], "seed": seed, "error": "%s: %s" % (type(ex).__name__, str(ex)[:200])})
                continue
            if g.prompt_sha256 != j["prompt_sha256"]:
                raise JobError("backend generated from a different prompt than job %s" % j["job_id"])
            rec = {"job_id": j["job_id"], "task_id": j["task_id"], "seed": seed, "purpose": j["purpose"], "prompt_sha256": j["prompt_sha256"], "generation": Q.generation_to_dict(g)}
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + chr(10))
                f.flush()
                os.fsync(f.fileno())
            generated += 1
    return {"jobs": len(jobs), "generated": generated, "skipped_already_done": skipped, "errors": errors, "error_list": err_list, "model": backend.model_info()}
