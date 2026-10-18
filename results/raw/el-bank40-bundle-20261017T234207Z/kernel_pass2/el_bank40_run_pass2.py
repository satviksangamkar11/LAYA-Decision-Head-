"""[EL-BANK40] Kaggle kernel: the 40 bank tasks (20 Python P1-*, 20 Java J1-*) through the FULL function-level pipeline (EvidenceLadderController, ladder mode: investigate, k candidates, public examples,
L2 tests, L1 behaviour clustering, discrimination via the blinded critic hand-off, rank, repair, verify) with Qwen3-4B-Thinking-2507 on T4 GPUs and the Kaggle execution jail. Internet OFF, NO secrets,
NO LAYA (scorer: none). DEVELOPMENT results under the recorded D1 waiver, public examples only; hidden grading is NOT done.

TEMPLATE: tools/pack_el_bank40_v1.py copies this file into the kernel folder and sets RUN_MODE ("pass1" | "pass2"). The bundle dataset (code + 40 public task records + run_config.json) is separate.
  pass1  GPU kernel. The parent verifies the bundle and the model, builds the Kaggle jail rootfs (fail closed: no unsandboxed fallback), shards the tasks over the visible GPUs (one worker process per
         GPU, CUDA_VISIBLE_DEVICES per worker), samples GPU utilisation, merges the timelines. Each finished task writes tasks/<id>/{result.json, journal.jsonl, gen_calls.json, critic_requests.json,
         glm/} and, LAST and atomically, DONE. A re-run (previous output mounted as an input dataset) skips every task that has DONE. A time guard stops starting tasks after the guard hour and records
         the pending ones. Critic packets of all tasks are exported blinded to critic_packets/ for the separate critic kernel.
  pass2  CPU kernel (the jail still runs candidates). Mounts the pass-1 output and the critic responses; for each task that has packets it replays the RECORDED generations in order (prompt-hash
         checked, nothing regenerated), applies the responses and finishes the task.
The thinking policy is read from the bundle's run_config.json and is REQUIRED (evidence_ladder.bank40.check_policy); it is copied into RUN_SUMMARY and every result. Always exits 0 with RUN_SUMMARY.json."""
import glob, hashlib, json, os, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path

RUN_MODE = "pass2"                       # replaced by the packer: "pass1" | "pass2"
T0 = time.time()
OUT = Path("/kaggle/working")
KERNEL = "el_bank40_run_v1"
QWEN_SHA = {"model-00001-of-00003.safetensors": "98178c1cddbca8ad76102afdbf76e4df79b64b82377ad4dca34c755cac00571a",
            "model-00002-of-00003.safetensors": "57b5bd1f667d420350f169342a90b50e5ffc2aa202ba1819da1af3367ff2261f",
            "model-00003-of-00003.safetensors": "e3cccb84e8926f3f18c334465639939a68af310b0729938fccc2de66a9a31953",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}      # same pins as kaggle/evidence_ladder_gpu_gen_v1.py
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
GENERATED_TOKEN_CAP = 100000


def find_bundle():
    for p in sorted(glob.glob("/kaggle/input/**/BUNDLE_MANIFEST.json", recursive=True)):
        m = json.load(open(p, encoding="utf-8"))
        if m.get("bundle") == "el_bank40_v1":
            return os.path.dirname(p), m
    return None, None


# ====================================================================== task worker (one process per GPU; pass2: one CPU process)
def worker(idx: int, todo_file: str) -> int:
    bundle, man = find_bundle()
    sys.path.insert(0, bundle)
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=os.environ.get("EL_KJAIL_JDK", ""), EVIDENCE_LADDER_ALLOW_INFERENCE="1", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    from evidence_ladder import bank40 as B
    from evidence_ladder.common.interfaces import Artifact, ControllerContext, TaskSpec, sha256_text
    from evidence_ladder.controller import generation as GEN, invariants as INV
    from evidence_ladder.controller.controller import EvidenceLadderController
    from evidence_ladder.execution.artifacts import ArtifactStore
    from evidence_ladder.execution.backend import SandboxExecutionBackend
    from evidence_ladder.execution import config as K
    from evidence_ladder.models import qwen as Q
    from evidence_ladder.models.broker import StagedCriticBroker
    from evidence_ladder.models import glm, experiments as EXP
    from evidence_ladder.orchestrator import isolation
    req = json.load(open(todo_file, encoding="utf-8"))
    policy = B.check_policy(req["thinking_policy"])
    tl = B.Timeline(OUT / "timeline" / ("worker%d.jsonl" % idx), worker="w%d" % idx)
    tasks = {}
    for ln in open(os.path.join(bundle, "bank40_tasks_public.jsonl"), encoding="utf-8"):
        d = json.loads(ln); tasks[d["task_id"]] = d
    pass2 = RUN_MODE == "pass2"
    base_gen, gpu_name = None, None
    if not pass2:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        tl.event("gpu", gpu_name=gpu_name, index=idx, visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
        md = req["model_dir"]
        if policy["policy"] == "model_variant":
            from evidence_ladder.models import backbone as BB
            prof = BB.get(policy["profile_id"])
            base_gen = Q.HFQwenBackend(authorized=True, model_dir=md, profile=prof, language="python", log_path=str(OUT / "timeline" / ("gen_log%d.jsonl" % idx)))
        else:
            kw = dict(think_budget=policy["think_budget"], answer_tokens=policy["answer_tokens"]) if policy["policy"] == "budget" else {}
            base_gen = Q.HFQwenBackend(authorized=True, model_dir=md, language="python", log_path=str(OUT / "timeline" / ("gen_log%d.jsonl" % idx)), **kw)
        base_gen.snapshot_verified = bool(req.get("model_sha_verified"))
        with tl.span("model_load", gpu=idx):
            base_gen._load()
    max_new = (policy["think_budget"] + policy["answer_tokens"]) if policy["policy"] == "budget" else policy["ceiling"]
    cfg0 = {"mode": "ladder", "k": 4, "l2": True, "l2_n": 6, "discriminate": True, "critic": True, "strategies": list(GEN.FROZEN_STRATEGIES), "max_new_tokens": max_new, "max_rounds": 2, "mutant_limit": 8}
    guard = B.TimeGuard(T0 if not req.get("t0") else req["t0"], req.get("guard_h", B.DEFAULT_GUARD_H))
    pending = []
    for tid in req["todo"]:
        if not guard.should_start():
            pending.append(tid); continue
        d = OUT / "tasks" / tid
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)
        rec = {"task": tid, "pass": RUN_MODE, "thinking_policy": policy, "scorer": "none", "worker": idx, "gpu": gpu_name, "label": "development results (D1 waiver), public examples only; hidden grading NOT done"}
        rec["start_utc"] = B.utc_now(); t_start = time.time()
        try:
            td = tasks[tid]
            task = TaskSpec(td["task_id"], ("py1:" if td["language"] == "python" else "java1:") + td["task_id"], td["language"], td["language"], td["prompt"], td["public_tests"], td.get("difficulty", ""))
            exe = B.TimedExec(SandboxExecutionBackend(ArtifactStore(d / "store"), K.SandboxConfig()), tl, tid)
            critic = StagedCriticBroker(d / "glm" / "broker_ledger.jsonl", config=glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=EXP.SUPPORTED_KINDS), max_retries=1)
            cfg = dict(cfg0)
            brief = None
            if pass2:
                calls = json.load(open(Path(req["pass1_dir"]) / "tasks" / tid / "gen_calls.json", encoding="utf-8"))
                gen_inner = B.SequenceReplayGen(calls, {"task": tid})
                resp = req.get("critic_responses") or {}
                cfg["critic_responses"] = resp
                rec["gpu"] = None
            else:
                base_gen.language = td["language"]
                if policy["policy"] == "prompt_brief":
                    brief = B.BriefPromptGen(base_gen, policy["brief_instruction"], policy["ceiling"], tl, tid); gen_inner = brief
                else:
                    gen_inner = base_gen
            recorder = B.RecordingGen(gen_inner, tid)
            gen = B.TimedGen(recorder, tl, tid, gpu=idx if not pass2 else None)
            ctx = ControllerContext(gen, exe, Q.TokenBudget(tid, "D0", GENERATED_TOKEN_CAP), "D0", critic, None, SEEDS, cfg)
            i8 = isolation.deep_context_check(ctx)
            if i8: raise RuntimeError("I8 deep scan found evaluation-only material in the controller context: %s" % i8[:3])
            try:      # the controller's own side-channel timeline (stage map intake/generate/execute/l2/cluster/discriminate/rank/repair/commit_verify); never part of a hash
                from evidence_ladder.controller.timeline import Timeline as CTimeline
                ctl_tl = CTimeline()
            except Exception:      # noqa: BLE001
                ctl_tl = None
            ctl = EvidenceLadderController(timeline=ctl_tl) if ctl_tl is not None else EvidenceLadderController()
            with tl.span("task", task=tid, language=td["language"], shard=idx):
                res = ctl.run_problem(task, ctx)
            j = ctl.last_journal
            try:
                inv = {"checked": list(INV.assert_invariants(res, j, ControllerContext(None, None, None, "D0"))), "failed": None}
            except Exception as e:      # noqa: BLE001
                inv = {"checked": None, "failed": "%s: %s" % (type(e).__name__, str(e)[:300])}
            (d / "journal.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False, default=str, sort_keys=True) + "\n" for e in j.entries()), encoding="utf-8")
            (d / "gen_calls.json").write_text(json.dumps(recorder.calls, ensure_ascii=False), encoding="utf-8")
            (d / "critic_requests.json").write_text(json.dumps({pid: p["request"] for pid, p in critic.packets.items()}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            if ctl_tl is not None:
                try: (d / "controller_timeline.json").write_text(json.dumps({"rows": ctl_tl.rows(), "summary": ctl_tl.summary()}, ensure_ascii=False, default=str), encoding="utf-8")
                except Exception: pass      # noqa: BLE001  a timeline can never fail a task
            v = B.verdicts(res, j)
            gens = [g for c in recorder.calls for g in c["generations"]]
            rec.update(language=td["language"], difficulty=td.get("difficulty", ""), stop_cause=res.stop_cause, stop_details=res.stop_details, answer_label=res.answer_label, rounds=res.rounds,
                       journal_hash=res.journal_hash, journal_chain_ok=j.verify_chain(), invariants=inv, candidates=len(j.entries("candidate")), public=B.public_example_stats(j), budget=res.budget_record,
                       tokens_generated=sum(g["tokens_generated"] for g in gens), tokens_prompt=sum(g["tokens_prompt"] for g in gens), generate_calls=len(recorder.calls),
                       ceiling_hits=(brief.ceiling_hits if brief else 0), generation_failed_at_ceiling=(brief.ceiling_hits > 0 if brief else False),
                       pending_critic_packets=sorted(pid for pid in critic.packets if critic.completed(pid) is None and critic.terminal_state(pid) is None and pid not in (cfg.get("critic_responses") or {})), replay_exhausted=(gen_inner.exhausted() if pass2 else None), **v)
            rec["status"] = "OK"
        except BaseException as e:      # noqa: BLE001  a crash on one task is a recorded result, never a hidden one; no DONE marker, so a re-run retries it
            rec.update(status="ERROR", error="%s: %s" % (type(e).__name__, str(e)[:400]), trace=traceback.format_exc()[-2500:])
            tl.event("task_error", task=tid, error=type(e).__name__)
        rec["end_utc"] = B.utc_now(); rec["seconds"] = round(time.time() - t_start, 1)
        B.write_atomic(d / "result.json", json.dumps(rec, indent=1, ensure_ascii=False, default=str, sort_keys=True))
        if rec["status"] == "OK":
            B.write_atomic(d / "DONE", rec["end_utc"])          # LAST: a task without DONE is unfinished
    B.write_atomic(OUT / "timeline" / ("worker%d.pending.json" % idx), json.dumps(pending))
    tl.event("worker_end", pending=len(pending))
    return 0


# ====================================================================== parent
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    try:
        sys.exit(worker(int(sys.argv[2]), sys.argv[3]))
    except BaseException:      # noqa: BLE001
        traceback.print_exc(); sys.exit(1)

RS = {"kernel": KERNEL, "mode": RUN_MODE, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phases": {}, "notes": [], "scorer": "none",
      "label": "development results (D1 waiver), public examples only; hidden grading NOT done"}


def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
def save():
    (OUT / "RUN_STATE.json").write_text(json.dumps(RS, indent=1, default=str), encoding="utf-8")


def finish(verdict, **kw):
    RS["verdict"] = verdict; RS["total_seconds"] = round(time.time() - T0, 1); RS["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); RS.update(kw)
    save(); (OUT / "RUN_SUMMARY.json").write_text(json.dumps(RS, indent=1, default=str), encoding="utf-8")
    log("VERDICT", verdict); sys.stdout.flush(); os._exit(0)


def main():
    (OUT / "timeline").mkdir(exist_ok=True)
    bundle, man = find_bundle()
    if not bundle: finish("FAILED_NO_BUNDLE")
    sys.path.insert(0, bundle)
    from evidence_ladder import bank40 as B
    tl = B.Timeline(OUT / "timeline" / "main.jsonl", worker="main")
    tl.event("kernel_start", mode=RUN_MODE, python=sys.version.split()[0])
    with tl.span("bundle_verify"):
        bad = [f for f, h in man["files"].items() if not os.path.isfile(os.path.join(bundle, f)) or hashlib.sha256(open(os.path.join(bundle, f), "rb").read()).hexdigest() != h]
    if bad: finish("FAILED_BUNDLE_HASH", bad=bad[:5])
    RS["bundle"] = {"files": len(man["files"]), "manifest_sha256": hashlib.sha256(json.dumps(man["files"], sort_keys=True).encode()).hexdigest(), "code_commit": man.get("commit")}
    # ---- thinking policy: REQUIRED, explicit, recorded
    run_cfg = json.load(open(os.path.join(bundle, "run_config.json"), encoding="utf-8"))
    try:
        policy = B.check_policy(run_cfg.get("thinking_policy") or {})
    except B.RequestRefused as e:
        finish("REFUSED_NO_THINKING_POLICY", error=str(e))
    RS["thinking_policy"] = policy
    # ---- public-only guard on what is actually mounted
    files = {f: open(os.path.join(bundle, f), "rb").read() for f in man["files"]}
    try:
        RS["public_only_guard"] = B.assert_public_bundle(files, [])
    except AssertionError as e:
        finish("REFUSED_BUNDLE_NOT_PUBLIC", error=str(e)[:500])
    tasks = [json.loads(l) for l in open(os.path.join(bundle, "bank40_tasks_public.jsonl"), encoding="utf-8")]
    assert len(tasks) == man["tasks"] == 40, len(tasks)
    # ---- the jail (fail closed)
    java = shutil.which("java")
    JAVA_HOME = str(Path(os.path.realpath(java)).parents[1]) if java else ""
    os.environ.update(EL_JAIL="kaggle", EL_KJAIL_JDK=JAVA_HOME, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    jail = {"java_home": JAVA_HOME, "uid": getattr(os, "getuid", lambda: -1)(), "status": "NOT_RUN"}
    try:
        from evidence_ladder.execution import kaggle_jail as KJ, engine, config as K
        with tl.span("rootfs_build"):
            jail["rootfs"] = KJ.build_rootfs(JAVA_HOME)
        jail.update(selfcheck(KJ, K, bundle))
        engine.py_mod(); engine.java_mod()
        jail["bwrap_swapped"] = bool(engine.py_mod().bwrap_cmd is KJ.bwrap_cmd)
        if jail["status"] != "PASS" or not jail["bwrap_swapped"]: raise RuntimeError("jail self-check failed: %s" % json.dumps({k: v for k, v in jail.items() if k != "rootfs"}, default=str)[:600])
    except BaseException as e:      # noqa: BLE001  FAIL CLOSED: no task runs without the jail, and there is no unsandboxed fallback
        jail["status"] = "FAILED"; jail["error"] = repr(e)[:600]; RS["jail"] = jail; tl.event("jail_failed", error=type(e).__name__)
        finish("JAIL_UNAVAILABLE", note="the jail could not start on Kaggle; nothing was executed and nothing falls back to an unsandboxed run")
    RS["jail"] = {k: v for k, v in jail.items() if k != "rootfs"}; save()
    req = {"thinking_policy": policy, "guard_h": float(os.environ.get("EL_GUARD_H", B.DEFAULT_GUARD_H)), "t0": T0}
    if RUN_MODE == "pass1":
        ng = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
        RS["gpus"] = ng
        if ng < 1: finish("FAILED_NO_GPU")
        md = next((os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if (("4b-thinking-2507" in c.lower()) if policy["policy"] != "model_variant" else (policy["model_glob"].lower() in c.lower()))), None)
        if not md: finish("FAILED_NO_MODEL", policy=policy["policy"])
        req["model_dir"] = md
        with tl.span("model_sha"):
            ok = {}
            for f in (QWEN_SHA if policy["policy"] != "model_variant" else {}):
                x = hashlib.sha256()
                with open(os.path.join(md, f), "rb") as fh:
                    for ch in iter(lambda: fh.read(1 << 24), b""): x.update(ch)
                ok[f] = x.hexdigest() == QWEN_SHA[f]
        RS["model"] = {"dir": md, "sha256_ok": ok, "pinned": bool(ok)}
        if ok and not all(ok.values()): finish("FAILED_MODEL_SHA")
        req["model_sha_verified"] = bool(ok)
        workers = min(ng, 2)
    else:
        resp = {}
        for p in glob.glob("/kaggle/input/**/responses.jsonl", recursive=True):
            for ln in open(p, encoding="utf-8"):
                if ln.strip():
                    r = json.loads(ln)
                    if r.get("packet_id") and r.get("response") is not None: resp[r["packet_id"]] = r["response"]
        gcs = sorted(glob.glob("/kaggle/input/**/tasks/*/gen_calls.json", recursive=True))
        if not gcs: finish("FAILED_NO_PASS1_OUTPUT")
        req.update(pass1_dir=str(Path(gcs[0]).parents[2]), critic_responses=resp)
        RS["critic_responses"] = len(resp)
        workers = 1
    # ---- resume: finished tasks of a previous version are copied in and skipped
    prev = B.finished_task_ids(*[Path(p) for p in glob.glob("/kaggle/input/*")]) if RUN_MODE == "pass1" else {}
    for tid, src in prev.items():
        if not (OUT / "tasks" / tid / "DONE").exists(): shutil.copytree(src, OUT / "tasks" / tid, dirs_exist_ok=True)
    RS["resumed_tasks"] = sorted(prev)
    if RUN_MODE == "pass1":
        order = B.shard_tasks(tasks, workers); todo_all = [[t for t in sh if t not in prev] for sh in order]
    else:
        need = sorted(p.parent.name for p in Path(req["pass1_dir"]).glob("tasks/*/gen_calls.json")
                      if json.load(open(p.parent / "result.json", encoding="utf-8")).get("pending_critic_packets"))
        todo_all = [need]
    RS["shards"] = {"workers": workers, "tasks_per_worker": [len(x) for x in todo_all], "order": todo_all}; save()
    stop_gpu = threading.Event()
    if RUN_MODE == "pass1":
        def sampler():
            while not stop_gpu.is_set():
                o = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                with open(OUT / "timeline" / "gpu_util.jsonl", "a", encoding="utf-8") as f:
                    for ln in o.strip().splitlines():
                        a = [x.strip() for x in ln.split(",")]
                        if len(a) == 3: f.write(json.dumps({"ts_utc": B.utc_now(), "ts": round(time.time(), 3), "gpu": int(a[0]), "util_pct": int(a[1]), "mem_mib": int(a[2])}) + "\n")
                stop_gpu.wait(30)
        threading.Thread(target=sampler, daemon=True).start()
    procs = []
    for i in range(workers):
        tf = OUT / ("worker%d_request.json" % i)
        tf.write_text(json.dumps(dict(req, todo=todo_all[i])), encoding="utf-8")
        env = dict(os.environ);
        if RUN_MODE == "pass1": env["CUDA_VISIBLE_DEVICES"] = str(i)
        else: env["CUDA_VISIBLE_DEVICES"] = ""
        lf = open(OUT / ("worker%d.log" % i), "w", encoding="utf-8")
        procs.append((i, subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(i), str(tf)], env=env, stdout=lf, stderr=subprocess.STDOUT)))
        tl.event("worker_spawn", worker=i, tasks=len(todo_all[i]))
    rcs = {i: p.wait() for i, p in procs}
    stop_gpu.set()
    tl.event("workers_done", rcs=rcs)
    RS["worker_exit_codes"] = rcs
    # ---- merge: timeline, critic packets, summary
    evs = B.read_events(OUT / "timeline" / "main.jsonl", *sorted((OUT / "timeline").glob("worker*.jsonl")))      # not gpu_util / gen_log (other schemas)
    with open(OUT / "TIMELINE.jsonl", "w", encoding="utf-8") as f:
        for e in evs:
            if "event" in e: f.write(json.dumps(e, sort_keys=True) + "\n")
    summ = B.summarize([e for e in evs if "event" in e])
    (OUT / "TIMELINE_SUMMARY.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    if RUN_MODE == "pass1":
        reqs = {}
        for p in sorted((OUT / "tasks").glob("*/critic_requests.json")):
            if (p.parent / "DONE").exists() and json.loads((p.parent / "result.json").read_text("utf-8")).get("pending_critic_packets"):
                r = json.loads(p.read_text("utf-8"))
                for pid in json.loads((p.parent / "result.json").read_text("utf-8"))["pending_critic_packets"]: reqs[pid] = r[pid]
        from evidence_ladder.critic_batch import validator as V
        bad = {}
        for pid in list(reqs):
            try: V.validate_packet({"packet_id": pid, "request": reqs[pid]})
            except Exception as e: bad[pid] = type(e).__name__; reqs.pop(pid)
        V.write_packets(OUT / "critic_packets", reqs)
        RS["critic_packets"] = {"exported": len(reqs), "rejected_blinding": bad}
        tl.event("critic_export", exported=len(reqs), rejected=len(bad))
    done = sorted(p.parent.name for p in (OUT / "tasks").glob("*/DONE"))
    errs = sorted(p.parent.name for p in (OUT / "tasks").glob("*/result.json") if not (p.parent / "DONE").exists())
    pend = sorted({t for sh in todo_all for t in sh} - set(done) - set(errs))
    RS.update(done=len(done), errors=errs, pending_not_started=pend, time_guard_hit=bool(pend), timeline_summary=summ)
    tl.event("kernel_end", done=len(done), errors=len(errs), pending=len(pend))
    finish("OK" if len(done) == 40 or (RUN_MODE == "pass2" and not errs) else ("PARTIAL_TIME_GUARD" if pend else "PARTIAL"))


def selfcheck(KJ, K, bundle):
    """Trivial python in the jail, javac -version in the jail, socket() denied, host canary unreadable. status PASS only if all hold."""
    rec = {"checks": {}}
    w = OUT / "jail_selfcheck"; w.mkdir(exist_ok=True)
    (OUT / "HOST_CANARY.txt").write_text("HOST-CANARY-do-not-read-from-jail\n")
    cand = w / "cand.py"
    cand.write_text("import socket, sys\ndef solve(d):\n    r = {}\n    try:\n        socket.socket(); r['socket'] = 'ALLOWED'\n    except OSError as e:\n        r['socket'] = 'denied'\n    try:\n        open('/kaggle/working/HOST_CANARY.txt').read(); r['canary'] = 'READ'\n    except OSError:\n        r['canary'] = 'unreadable'\n    return r\n")
    runner = w / "runner.py"; shutil.copyfile(K.PY_RUNNER, runner)
    p = subprocess.run(KJ.bwrap_cmd(cand, runner), input=b'{"a": 1}\n', capture_output=True, timeout=60, env={"PATH": "/usr/bin"})
    out = p.stdout.decode(errors="replace")
    rec["checks"]["python_in_jail"] = {"rc": p.returncode, "out": out[-300:]}
    py_ok = p.returncode == 0 and "'denied'" in out.replace('"', "'") and "unreadable" in out
    wj = w / "j"; wj.mkdir(exist_ok=True); os.chmod(wj, 0o755)
    java = os.environ.get("EL_KJAIL_JDK", "")
    ok_j = True
    if java:
        pj = subprocess.run(KJ.java_jail(wj, [java + "/bin/javac", "-version"], rw=True, jdk=java), capture_output=True, timeout=90, env={"PATH": "/usr/bin"})
        rec["checks"]["javac_in_jail"] = {"rc": pj.returncode, "err": (pj.stdout + pj.stderr).decode(errors="replace")[-200:]}
        ok_j = pj.returncode == 0
    else:
        rec["checks"]["javac_in_jail"] = {"rc": None, "err": "no java on this machine"}; ok_j = False
    rec["status"] = "PASS" if py_ok and ok_j else "FAIL"
    return rec


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:      # noqa: BLE001
        RS["error"] = repr(e)[:500]; RS["trace"] = traceback.format_exc()[-4000:]; traceback.print_exc()
        finish("FAILED")
