"""[LAYA-INT-1 · Aim: integration, not performance] END-TO-END orchestrator: 3 tiny sandboxed coding tasks, up to 2 steps each, through the Evidence-Ladder controller in per-action mode
(evidence_ladder/controller/agentic.py). One process drives everything; every Kaggle round is cached on disk so a re-run reuses finished rounds.

  Kaggle T4 (default account sangeetasangamkar):  Qwen proposes 3-5 actions (greedy) -> capture (training recipe) -> 3 adapters + head   [LIVE round]
  Kaggle T4, second kernel version/process:       re-captures the SAME saved proposals and re-scores them                                  [VERIFY round, replay]
  PC:  VerifiedSavedScoreBackend (live == verify, else refused) behind the real LayaScorer interface -> AgenticRun.step (policy, WSL dry-runs, L0 ranking, WSL execution,
       protected-test evidence, labels) -> receipts -> finish() with assert_invariants I1-I8.
  Pass 2 (D): a fresh run over the same saved Kaggle outputs must give identical journal hashes.

  PYTHONPATH=. .venv/Scripts/python.exe tools/laya_int1_e2e.py <out dir under results/raw/laya-integration-*/> [--steps 2] [--investigate] [--require scorer,critic]
Everything is create-only below <out dir>. No key is read here; the critic (step E) is wired separately."""
import hashlib, json, os, re, shutil, subprocess, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TAG = "[LAYA-INT-1 · Aim: integration, not performance]"
OWNER = "sangeetasangamkar"
KERNEL = OWNER + "/evidence-ladder-laya-t6-step-v1"
sys.path.insert(0, str(ROOT / "tools"))
from laya_int1_t6 import TASKS  # noqa: E402  (task definitions only)

EXAMPLES = [("ex1_slug", "slug"), ("ex2_clamp", "clamp"), ("ex3_wordcount", "wordcount")]


def sha(b): return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()
def wslpath(p): p = str(Path(p).resolve()); return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
def write_new(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f: f.write(text)


from evidence_ladder.controller.wsl_sandbox import WslSandbox  # noqa: E402  (shared class; documented WSL command lives there)


def system_block():
    b = sorted((ROOT / "results/raw").glob("el-laya-int1-bundle-*"))[-1]
    return (b / "gate/system_block.txt").read_text("utf-8")


def render(messages):
    from decision_head import fable_pool as FP
    text, info = FP.select_context(messages, 160000)
    assert text is not None, info
    return text


# ------------------------------------------------------------------ Kaggle rounds (cached)
def kaggle(*args, t=120):
    try:
        p = subprocess.run([shutil.which("kaggle") or "kaggle", *args], capture_output=True, text=True, timeout=t)     # bounded: subprocess timeout (the shell `timeout` is a different program on Windows)
    except subprocess.TimeoutExpired:
        return "ERROR: kaggle call timed out after %ds" % t
    return (p.stdout + p.stderr).strip()


def kaggle_round(out: Path, tag: str, mode: str, items):
    """items = [{"key", "step", "context_text", "saved_raw_output"?}]. -> (result json, run id). Cached under <out>/_kaggle/<tag>/."""
    d = out / "_kaggle" / tag; res = d / "output" / "t6_step_results.json"
    if res.exists(): return json.loads(res.read_text("utf-8")), json.loads((d / "run.json").read_text("utf-8"))["run_id"]
    req = {"run_id": "%s/%s" % (out.name, tag), "mode": mode, "steps": items, "max_new_tokens": 700, "with_controls": True}
    src = (ROOT / "kaggle/evidence_ladder_laya_t6_step_v1.py").read_text("utf-8")
    assert "__STEP_REQUEST_JSON__" in src and "'''" not in json.dumps(req)
    if (d / "kernel").exists(): shutil.rmtree(d / "kernel")          # a previous attempt that never got a result
    write_new(d / "kernel" / "evidence_ladder_laya_t6_step_v1.py", src.replace("__STEP_REQUEST_JSON__", json.dumps(req, ensure_ascii=False)))
    write_new(d / "kernel" / "kernel-metadata.json", json.dumps({"id": KERNEL, "title": "evidence-ladder-laya-t6-step-v1", "code_file": "evidence_ladder_laya_t6_step_v1.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": ["leminhhung0101/qwen3-4b-thinking-2507", OWNER + "/evidence-ladder-laya-int1-bundle-v1"],
        "kernel_sources": [], "competition_sources": []}, indent=2))
    msg = kaggle("kernels", "push", "-p", str(d / "kernel"))
    m = re.search(r"version (\d+)", msg)
    if not m: raise RuntimeError("push failed: " + msg[-300:])
    run_id = "%s@v%s/%s" % (KERNEL, m.group(1), tag)
    write_new(d / "run.json", json.dumps({"run_id": run_id, "push_message": msg[-200:], "pushed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    status = ""
    for _ in range(100):
        status = kaggle("kernels", "status", KERNEL, t=60).splitlines()[-1] if True else ""
        if any(x in status for x in ("403", "404", "ERROR", "CANCEL")): raise RuntimeError("kaggle status: " + status)
        if "COMPLETE" in status: break
        time.sleep(30)
    else: raise RuntimeError("kaggle round timed out: " + status)
    (d / "output").mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", KERNEL, "-p", str(d / "output"), t=300)
    if not res.exists(): raise RuntimeError("no result file in kernel output")
    return json.loads(res.read_text("utf-8")), run_id


# ------------------------------------------------------------------ the controller run
INVESTIGATE = False
EXTRA_CONFIG = {}      # e.g. {"require": ["scorer", "critic"]} for final acceptance runs (a missing component then raises instead of being waived)


def build_runs(out: Path, label: str):
    from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask
    from evidence_ladder.models import laya, laya_runtime as RT
    backend = RT.VerifiedSavedScoreBackend("laya")
    scorer = laya.LayaScorer(authorized=True, backend=backend)
    assert not scorer.disabled, scorer.meta.disabled_reason
    runs = {}
    for name, task in EXAMPLES:
        T = TASKS[task]; ws = out / label / name / "workspace"
        for fn, c in T["files"].items(): write_new(ws / fn, c)
        at = AgenticTask(task_id=name, user_text=T["user"], workspace=ws, test_cmd=T["test_cmd"], protected=T["protected"], system_text=system_block(), max_steps=2)
        ctx = AgenticContext(sandbox=WslSandbox(out / label / "_sandbox", T["protected"]), scorer=scorer, arm="E", seeds=(), config={"max_steps": 2, **EXTRA_CONFIG})
        r = AgenticRun(at, ctx, render); r.begin()
        if INVESTIGATE:      # [EL-ARCH-1 B2] deterministic investigate stage (no model): baseline test run, parsed failure, report in the next context
            from evidence_ladder.controller import investigate as INV
            INV.investigate(r)
        runs[name] = r
    return runs, backend


def critic_phase(out: Path, label: str, runs, prior=None):
    """Step E through the controller's critic stage: blinded packets (broker validation + critic_batch validator), sent from THIS PC by PcSender with the project key system
    tools/nvidia_keys.py. Key values are never read here: if the key system cannot be constructed (its env vars are not set) the phase STOPS, writes the exact error and the controller
    journals critic 'unavailable' with that reason. No workaround (.secrets is never opened)."""
    from evidence_ladder.models.broker import StagedCriticBroker
    from evidence_ladder.models import glm
    from evidence_ladder.critic_batch import validator as V
    if prior is not None and (out / "run1" / "critic_ledger.jsonl").exists():
        (out / label).mkdir(parents=True, exist_ok=True); shutil.copyfile(out / "run1" / "critic_ledger.jsonl", out / label / "critic_ledger.jsonl")     # pass 2 never resends: it reads run1's answers
    broker = StagedCriticBroker(out / label / "critic_ledger.jsonl", config=glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}), max_retries=1)
    info = {"packets": {}, "key_system": None, "sent": 0}
    for n, r in runs.items():
        r.ctx.critic = broker
        state = r.critic_stage()
        pk = broker.packets
        req = r.critic_request()
        pid = next(k for k, v in pk.items() if v["request"]["request_id"] == req["request_id"])
        V.validate_packet({"packet_id": pid, "request": pk[pid]["request"]})            # raises PacketRejected on any forbidden content
        info["packets"][n] = {"packet_id": pid, "first_state": state, "validator": "ok", "request_sha256": sha(json.dumps(pk[pid]["request"], sort_keys=True))}
    if prior is not None:           # pass 2: same outcome as pass 1 without any new call
        ks = prior["key_system"]
        if ks["status"] == "BLOCKED":
            for n, r in runs.items():
                r.note("critic", {"state": "unavailable", "packet_id": info["packets"][n]["packet_id"], "reason": "key system unavailable: %s: %s" % (ks["error_type"], ks["error"][:300])})
        else:
            for n, r in runs.items(): info["packets"][n]["second_state"] = r.critic_stage()
        info["key_system"] = ks
        return info
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import nvidia_keys as NK
        rot = NK.KeyRotator(out / label / "nvidia_key_ledger.jsonl", cap=len(runs) * 3, limit_scope="key", per_key_batch=39)
        info["key_system"] = {"status": "constructed", "aliases": list(NK.ALIASES)}
    except Exception as ex:      # noqa: BLE001  exact reason is the evidence (the message names variables, never values)
        info["key_system"] = {"status": "BLOCKED", "error_type": type(ex).__name__, "error": str(ex)[:400]}
        for n, r in runs.items():
            r.note("critic", {"state": "unavailable", "packet_id": info["packets"][n]["packet_id"], "reason": "key system unavailable: %s: %s" % (type(ex).__name__, str(ex)[:300])})
        return info
    from evidence_ladder.models.pc_sender import KeyScheduler, PcSender
    sender = PcSender(broker, KeyScheduler(list(NK.ALIASES)))

    def transport(packet, key_label):
        try:
            body = rot.request(packet["packet_id"], packet["payload"], NK.http_transport(glm.GlmConfig().endpoint, timeout=290))
            return 200, body
        except NK.AllKeysUnavailable as e:
            return 429, {"error": str(e), "headers": {"retry-after": "60"}}
        except Exception as e:      # noqa: BLE001
            return None, {"error": type(e).__name__}
    info["send"] = sender.send_pending(transport)
    for n, r in runs.items(): info["packets"][n]["second_state"] = r.critic_stage()
    return info


def drive(out: Path, label: str, max_steps: int, prior_critic=None):
    if (out / label).exists(): shutil.rmtree(out / label)          # derived data (workspaces, receipts): rebuilt deterministically from the cached Kaggle rounds in _kaggle/
    runs, backend = build_runs(out, label)
    for step in range(1, max_steps + 1):
        active = [n for n, r in runs.items() if not r.decided]
        if not active: break
        items = [{"key": "%s:%d" % (n, step), "step": step, "context_text": runs[n].context_text()} for n in active]
        live, live_id = kaggle_round(out, "live_s%d" % step, "propose", items)
        ver_items = [dict(it, saved_raw_output=live["steps"][it["key"]]["raw_output"]) for it in items]
        ver, ver_id = kaggle_round(out, "verify_s%d" % step, "replay", ver_items)
        for it in items:
            k = it["key"]; lv, vr = live["steps"].get(k, {}), ver["steps"].get(k, {})
            if lv.get("status") == "ok" and vr.get("status") == "ok": backend.add(lv, vr, live_run=live_id, verify_run=ver_id, verify_mode=ver["mode"])
        for it in items:
            n = it["key"].split(":")[0]; lv = live["steps"].get(it["key"], {})
            opts = lv.get("options", []) if lv.get("status") == "ok" else lv.get("options", [])
            meta = dict(lv.get("generation", {}), kaggle_seconds=live.get("total_seconds", 0) / max(len(items), 1))
            rec = runs[n].step(opts, meta)
            rec.update(qwen_raw_output=lv.get("raw_output"), qwen_raw_output_sha256=lv.get("raw_output_sha256"), kaggle_live_run=live_id, kaggle_verify_run=ver_id,
                       kaggle_device=live.get("gpu"), kaggle_versions=live.get("versions"), proposals_status=lv.get("status"), capture_provenance=lv.get("provenance"), features_sha256=lv.get("features_sha256"),
                       scores_all_kinds=lv.get("scores"), control_label="probe/random head: controls; random head = MISMATCH CONTROL, NOT a valid performance comparison")
            write_new(out / label / n / ("step%d_receipt.json" % step), json.dumps(rec, indent=1, ensure_ascii=False, default=str))
            print(json.dumps({"label": label, "run": n, "step": step, "options": len(opts), "chosen": rec["chosen_index"], "laya_order": rec["laya_order"], "allowed": rec["allowed_set"],
                              "id_choice": rec["id_order_choice"], "laya_changed": rec["laya_changed_the_choice_vs_id_order"], "tests_pass": rec["tests_pass"], "label_now": rec["label"],
                              "scorer_unavailable": rec["scorer_unavailable_reason"]}), flush=True)
    critic_info = critic_phase(out, label, runs, prior_critic)
    if label == "run1": write_new(out / "E_critic_receipt.json", json.dumps(dict(critic_info, tag=TAG), indent=1))
    results = {}
    for n, r in runs.items():
        res = r.finish()
        results[n] = {"answer_label": res.answer_label, "stop_cause": res.stop_cause, "journal_hash": res.journal_hash, "rounds": res.rounds, "invariants": r.invariants_checked,
                      "budget_record": res.budget_record, "steps": len(r.steps)}
        write_new(out / label / n / "journal.json", json.dumps(r.journal.entries(), indent=1, ensure_ascii=False, default=str))
        write_new(out / label / n / "steps.json", json.dumps([{"step": s["step"], "decision_id": s["decision_id"], "chosen_index": s["chosen_index"], "tests_pass": s["tests_pass"], "label": s["label"]} for s in r.steps], indent=1))
        write_new(out / label / n / "result.json", json.dumps(results[n], indent=1, default=str))
    return runs, results, backend, critic_info


def main():
    global INVESTIGATE
    out = Path(sys.argv[1]); steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 2
    INVESTIGATE = "--investigate" in sys.argv
    if "--require" in sys.argv: EXTRA_CONFIG["require"] = sys.argv[sys.argv.index("--require") + 1].split(",")
    out.mkdir(parents=True, exist_ok=True)
    runs1, res1, be, crit1 = drive(out, "run1", steps)
    # pass 2 (D): same saved Kaggle outputs, fresh workspaces and controllers -> identical journal hashes
    runs2, res2, _, _c = drive(out, "run2", steps, prior_critic=crit1)
    cmp_ = {n: {"journal_hash_run1": res1[n]["journal_hash"], "journal_hash_run2": res2[n]["journal_hash"], "identical": res1[n]["journal_hash"] == res2[n]["journal_hash"]} for n in res1}
    summary = {"tag": TAG, "examples": res1, "replay_pass": cmp_, "all_replay_identical": all(v["identical"] for v in cmp_.values()), "refused_imports": be.refused}
    write_new(out / "E2E_SUMMARY.json", json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, default=str)[:2000])


if __name__ == "__main__":
    main()
