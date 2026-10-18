"""[Qwen2.5-Coder-1.5B local run] GPU smoke: load the pinned model resident on the RTX 3050 through models.qwen.build_local_backend, run real bank-style generations (qwen_sample_v2 prompt of a real
bank task rendered by controller.generation.render_sample_prompt), then long-context probes. Measures load time, tokens/s, VRAM peak, host RAM peak and the lowest free system RAM.
Writes a create-only receipt  results/raw/qwen25coder15b-smoke-<UTC>/RECEIPT.json  (no secrets; model text is stored only as sha256 + a 300-char head).
    PYTHONPATH=. .venv/Scripts/python.exe evidence_ladder/models/qwen25_smoke.py
GPU use is user-approved for this task (chat, 2026-10-18)."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PID = "qwen2.5-coder-1.5b-instruct"
TASKS = ROOT / "results" / "raw" / "codex-pilot1-20261016" / "fixed_v2" / "worker" / "tasks.jsonl"


def nvsmi():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        u, t, temp, clk = [x.strip() for x in o.split(",")]
        return {"used_mib": int(u), "total_mib": int(t), "temp_c": int(temp), "sm_clock_mhz": int(clk)}
    except Exception as e:      # noqa: BLE001
        return {"error": str(e)[:100]}


class Mem:
    def __init__(self):
        import psutil
        self.ps, self.me = psutil, psutil.Process()
        self.peak_rss, self.min_avail, self._stop = 0, 10 ** 15, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.me.memory_info().rss)
            self.min_avail = min(self.min_avail, self.ps.virtual_memory().available)
            time.sleep(0.05)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()


def main() -> int:
    import torch
    from evidence_ladder.common.interfaces import GenerationRequest, TaskSpec, sha256_text
    from evidence_ladder.controller import generation as GEN
    from evidence_ladder.models import qwen as Q
    out = ROOT / "results" / "raw" / ("qwen25coder15b-smoke-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True)
    rows = [json.loads(l) for l in TASKS.read_text("utf-8").splitlines() if l.strip()]
    row = next(r for r in rows if r["task_id"] == "P1-11")
    task = TaskSpec(row["task_id"], "py1:" + row["task_id"], "python", "python", row["prompt"] + "\n\nSignature: " + row["function_signature"], row["public_tests"])
    rec = {"probe": "qwen25coder15b_smoke_v1", "profile": PID, "task": row["task_id"], "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "nvidia_smi_before": nvsmi()}
    with Mem() as mem:
        be = Q.build_local_backend(PID, log_path=str(out / "gen_log.jsonl"))
        rec["model_info_before_load"] = be.model_info()
        t = time.time()
        be._verify_snapshot()
        rec["snapshot_verify_seconds"] = round(time.time() - t, 2)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        model, tok = be._load()
        torch.cuda.synchronize()
        rec["load_seconds"] = round(time.time() - t, 2)
        rec["vram_after_load_mib"] = round(torch.cuda.memory_allocated() / 2 ** 20)
        rec["vram_peak_during_load_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        rec["nvidia_smi_after_load"] = nvsmi()
        rec["rss_peak_after_load_mib"] = round(mem.peak_rss / 2 ** 20)
        # ---- real bank-style generations (3 seeds, 1024 new tokens)
        reqs = GEN.plan_samples(task, 4, (1, 2, 3, 4), 0, GEN.FROZEN_STRATEGIES, 1024)
        gens, tps = [], []
        torch.cuda.reset_peak_memory_stats()
        for rq in reqs:
            for g in be.generate(rq):
                ok = False
                try:
                    ast.parse(g.code)
                    ok = bool(g.code.strip())
                except SyntaxError:
                    pass
                rate = g.tokens_generated / g.wall_seconds if g.wall_seconds else None
                tps.append(rate)
                gens.append({"template": g.prompt_template_id, "seed": g.seed, "prompt_tokens": g.tokens_prompt, "new_tokens": g.tokens_generated, "seconds": round(g.wall_seconds, 2),
                             "tokens_per_second": round(rate, 2) if rate else None, "extract_status": g.extract_status, "code_parses": ok, "text_sha256": sha256_text(g.text), "text_head": g.text[len(Q.NONTHINK_PREFIX):][:300],
                             "starts_with_closed_empty_think": g.text.startswith(Q.NONTHINK_PREFIX)})
        rec["generations"] = gens
        rec["tokens_per_second_mean"] = round(sum(x for x in tps if x) / max(1, len([x for x in tps if x])), 2)
        rec["vram_peak_generation_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        # ---- determinism of the recorded seed (same prompt+seed twice)
        a = be.generate(GenerationRequest("det", reqs[0].prompt, 1, (11,), 64, "sample", True, reqs[0].prompt_template_id))[0]
        b = be.generate(GenerationRequest("det", reqs[0].prompt, 1, (11,), 64, "sample", True, reqs[0].prompt_template_id))[0]
        rec["same_seed_same_text"] = a.text == b.text
        # ---- long contexts: prefill N tokens of filler + the real task prompt, generate 64 tokens
        filler = ("def helper_%d(a, b):\n    # utility used by the surrounding project\n    return (a * %d + b) %% 97\n\n")
        longs = []
        for target in (3500, 8000):
            body = ""
            i = 0
            while len(tok(body, add_special_tokens=False)["input_ids"]) < target - 600:
                body += filler % (i, i + 2)
                i += 1
            prompt = body + "\n" + reqs[0].prompt
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                g = be.generate(GenerationRequest("long", prompt, 1, (5,), 64, "sample", True, "qwen_sample_v2"))[0]
                torch.cuda.synchronize()
                longs.append({"target_context_tokens": target, "prompt_tokens": g.tokens_prompt, "new_tokens": g.tokens_generated, "seconds": round(time.time() - t, 2), "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20),
                              "status": "ok"})
            except torch.cuda.OutOfMemoryError as e:
                longs.append({"target_context_tokens": target, "status": "OOM", "error": str(e)[:200]})
                torch.cuda.empty_cache()
            if longs[-1]["status"] != "ok":
                break
        rec["long_context"] = longs
        rec["nvidia_smi_end"] = nvsmi()
        rec["model_info_after"] = be.model_info()
        rec["host_rss_peak_mib"] = round(mem.peak_rss / 2 ** 20)
        rec["host_min_free_ram_mib"] = round(mem.min_avail / 2 ** 20)
    with open(out / "RECEIPT.json", "x", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: rec[k] for k in ("load_seconds", "snapshot_verify_seconds", "vram_after_load_mib", "vram_peak_generation_mib", "tokens_per_second_mean", "host_rss_peak_mib", "host_min_free_ram_mib", "same_seed_same_text", "long_context")}, indent=1))
    print("RECEIPT", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
