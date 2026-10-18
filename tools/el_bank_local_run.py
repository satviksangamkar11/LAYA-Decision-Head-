"""[EL-BANK-LOCAL] ONE command runs the bank tasks of a manifest through the FULL function-level pipeline (EvidenceLadderController, ladder mode, L2 v2) LOCALLY and unbroken:
preflight -> every task (generation on the RTX 3050 with a resident NON-thinking model, execution in the MXC sandbox through the provider port, INLINE blinded GLM critic) -> automatic BANK results
and conformance report. Development results under the D1 waiver: public examples only, hidden grading NOT done.

  PYTHONPATH=. .venv/Scripts/python.exe tools/el_bank_local_run.py <out dir> [--tasks configs/el_bank80_rest_v1.json] [--limit N] [--resume] [--wall-cap 3600] [--profile qwen2.5-coder-1.5b-instruct] [--sandbox mxc|wsl]
  PYTHONPATH=. .venv/Scripts/python.exe tools/el_bank_local_run.py <out dir> --selftest           2 python tasks, tiny fake model, REAL MXC + real controller + inline critic over a fake transport; exit != 0 if any stage is missing

PREFLIGHT (refuses with a clear verdict PREFLIGHT_REFUSED and runs nothing): the sandbox backend answers (MXC capability probe), the Java capability is detected (MXC: Java tasks are recorded NOT_TESTED_NO_JDK, never run
elsewhere), free RAM / free VRAM are above configured minimums, the model snapshot is verified. CRITIC: when discrimination triggers a blinded packet the task's packets are sent to z-ai/glm-5.3-flash right there
(keys read from .secrets in memory only, <= 39 sends/key/min, bounded retries, streamed) and the answer is fed back into the controller with the RECORDED generations replayed (prompt-hash checked, nothing
regenerated), so the task completes in the same run. No key file / API failure -> state api_unavailable / critic_failed is recorded and the run CONTINUES; the critic is diagnostic only and can never approve or
change a trusted label. RESILIENCE: per-task atomic results, DONE written LAST, a crash of one task is recorded and the run continues, three engine failures in a row stop the run (ENGINE_UNAVAILABLE, resumable with
--resume), a STOP file ends the run after the current task, RUN_SUMMARY / TIMELINE are rewritten after every task. Output layout = the Kaggle pass-1 layout (+ exec_results.jsonl per task), readable by
tools/el_bank40_driver.py and tools/el_bank_conformance.py. thinking_policy is `model_variant` (non-thinking), ceiling 4096, recorded everywhere with l2_version 2, profile sha256, commit and config hashes."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from evidence_ladder import bank40 as B   # noqa: E402

PROFILE = "qwen2.5-coder-1.5b-instruct"
CEILING = 4096
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
GENERATED_TOKEN_CAP = 100000
WALL_CAP_S = 3600.0
L2_VERSION = 2
MIN_FREE_RAM_GB = 3.0
MIN_FREE_VRAM_GB = 3.5
MIN_FREE_COMMIT_GB = 4.5      # Agent 1 measured: loading the model fails below about 4.5 GB free system commit (GPU allocations count 1:1 against commit on this PC)
SOURCE_SHA256: Optional[str] = None      # set once per process in run()
CRITIC_READ_TIMEOUT_S = 420.0      # first-response wait per critic call (the 120 s of the first run timed out every bank packet: ~122 s per failed attempt)
CRITIC_TOTAL_TIMEOUT_S = 1800.0
MAX_CONSECUTIVE_ENGINE_FAILURES = 3
LABEL = "development results (D1 waiver), public examples only; hidden grading NOT done"
DEFAULT_MANIFEST = ROOT / "configs/el_bank80_rest_v1.json"
CONFIG_FILES = ("configs/el_arch1_limits_v2.toml", "configs/backbones_v1.toml", "configs/el_bank80_rest_v1.json", "configs/critic_concurrency_v1.toml")


def policy_for(profile: str = PROFILE, ceiling: int = CEILING) -> Dict[str, Any]:
    return B.check_policy({"policy": "model_variant", "profile_id": profile, "model_glob": "local:" + profile, "ceiling": ceiling})


def sha(b) -> str: return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:      # noqa: BLE001
        return None


def source_tree_sha256() -> Optional[str]:
    """sha256 over the sorted (path, blob id) listing of the COMMITTED source tree (evidence_ladder, tools, configs); None when git is unavailable."""
    try:
        o = subprocess.run(["git", "ls-tree", "-r", "HEAD", "--", "evidence_ladder", "tools", "configs"], cwd=str(ROOT), capture_output=True, text=True, timeout=60).stdout
        return sha("\n".join(sorted(o.splitlines()))) if o.strip() else None
    except Exception:      # noqa: BLE001
        return None


def config_hashes() -> Dict[str, str]:
    return {f: sha((ROOT / f).read_bytes()) for f in CONFIG_FILES if (ROOT / f).is_file()}


# ------------------------------------------------------------------ tasks (PUBLIC prompt + public tests only)
def load_tasks(manifest_path: Path = DEFAULT_MANIFEST, root: Path = ROOT, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    man = json.loads(Path(manifest_path).read_text("utf-8"))
    if man.get("total") is not None and man["total"] != len(man["tasks"]):
        raise AssertionError("manifest says %d tasks but lists %d" % (man["total"], len(man["tasks"])))
    if man.get("source") and man.get("source_sha256"):      # a derived manifest pins the manifest it was cut from
        if hashlib.sha256((root / man["source"]).read_bytes()).hexdigest() != man["source_sha256"]:
            raise AssertionError("manifest source %s differs from its pinned sha256" % man["source"])
    by: Dict[str, Dict[str, Any]] = {}
    for lang in ("python", "java"):
        src = man["sources"][lang]
        raw = (root / src["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != src["sha256"]:
            raise AssertionError("task source %s differs from the sha256 pinned in the manifest" % src["file"])
        for ln in raw.decode("utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln); by[d.get("task_id") or d["id"]] = {"task_id": d.get("task_id") or d["id"], "language": lang, "difficulty": d.get("difficulty", ""), "prompt": d["prompt"], "public_tests": d["public_tests"]}
    out = []
    for t in man["tasks"]:
        rec = by[t["task"]]
        if rec["language"] != t["language"]: raise AssertionError("manifest language mismatch for " + t["task"])
        out.append(rec)
    if limit is not None: out = out[:limit]
    return out


def load_task_ids(ids: Sequence[str], manifest_path: Path = DEFAULT_MANIFEST, root: Path = ROOT) -> List[Dict[str, Any]]:
    """Public task records for arbitrary bank ids (sources pinned by sha256 in the manifest); used by --selftest to take a task whose semantics the fake model knows."""
    man = json.loads(Path(manifest_path).read_text("utf-8"))
    by: Dict[str, Dict[str, Any]] = {}
    for lang in ("python", "java"):
        src = man["sources"][lang]
        raw = (root / src["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != src["sha256"]:
            raise AssertionError("task source %s differs from the sha256 pinned in the manifest" % src["file"])
        for ln in raw.decode("utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln)
                by[d.get("task_id") or d["id"]] = {"task_id": d.get("task_id") or d["id"], "language": lang, "difficulty": d.get("difficulty", ""), "prompt": d["prompt"], "public_tests": d["public_tests"]}
    return [by[i] for i in ids]


def public_guard(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    blob = ("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in tasks) + "\n").encode("utf-8")
    return B.assert_public_bundle({"bank40_tasks_public.jsonl": blob})         # exactly the public keys, no case file, no key-shaped text


# ------------------------------------------------------------------ resources
def system_resources() -> Dict[str, Any]:
    rec: Dict[str, Any] = {"free_ram_gb": None, "free_vram_gb": None, "free_commit_gb": None, "gpu": None}
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        rec["free_ram_gb"] = round(m.ullAvailPhys / 2 ** 30, 2)
        rec["free_commit_gb"] = round(m.ullAvailPageFile / 2 ** 30, 2)
    except Exception:      # noqa: BLE001
        try:
            rec["free_ram_gb"] = round(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30, 2)
        except Exception:      # noqa: BLE001
            pass
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,name", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0].split(",")
        rec["free_vram_gb"], rec["gpu"] = round(float(o[0]) / 1024, 2), o[1].strip()
    except Exception:      # noqa: BLE001
        pass
    return rec


def check_resources(res: Dict[str, Any], min_ram: float, min_vram: float, min_commit: float = 0.0) -> List[str]:
    why = []
    if min_ram and (res.get("free_ram_gb") is None or res["free_ram_gb"] < min_ram): why.append("free RAM %s GB < required %s GB (close other programs)" % (res.get("free_ram_gb"), min_ram))
    if min_vram and (res.get("free_vram_gb") is None or res["free_vram_gb"] < min_vram): why.append("free VRAM %s GB < required %s GB (another process holds the GPU?)" % (res.get("free_vram_gb"), min_vram))
    if min_commit and (res.get("free_commit_gb") is None or res["free_commit_gb"] < min_commit): why.append("free system commit %s GB < required %s GB (the model load fails below it; close Ableton or other large programs)" % (res.get("free_commit_gb"), min_commit))
    return why


# ------------------------------------------------------------------ wrappers
class ThinkShimGen:
    """The frozen extractor (models/qwen.py extract_code) only reads the code AFTER a closing </think>. A NON-thinking model never writes one, so every candidate would be rejected as no_code. This shim
    prefixes an EMPTY think block (an opening and a closing think tag) to a generation that has no closing tag and rebuilds the Generation (extract_status recomputed); it never changes the model's words. It is
    a no-op when the backend already does this. Counted in the task result (think_shim_count)."""
    PREFIX = "<think>" + chr(10) + "</think>" + chr(10)

    def __init__(self, inner: Any):
        self.inner, self.count = inner, 0

    def __getattr__(self, name): return getattr(self.inner, name)

    def generate(self, request):
        from evidence_ladder.models import qwen as Q
        out = []
        lang = "json" if request.purpose == "tests" else getattr(self.inner, "language", "python")
        for g in self.inner.generate(request):
            if Q.THINK_CLOSE in g.text:
                out.append(g); continue
            self.count += 1
            out.append(Q.make_generation(self.PREFIX + g.text, tokens_prompt=g.tokens_prompt, tokens_generated=g.tokens_generated, seed=g.seed, prompt_template_id=g.prompt_template_id,
                                         prompt_sha256=g.prompt_sha256, language=lang, model=g.model, model_revision=g.model_revision, wall_seconds=g.wall_seconds, logprob_summary=g.logprob_summary))
        return out


class EngineGuard:
    """Wraps the execution backend. An infrastructure error (`infra = True`) is counted and re-raised. The ladder controller turns an execution exception into a recorded tool failure and goes on, which is
    right for a flaky candidate but wrong for a dead engine: run_task compares the counter before/after and refuses to write DONE for a task during which the engine failed (fail closed)."""

    def __init__(self, inner: Any):
        self.inner, self.failures = inner, 0

    def __getattr__(self, name): return getattr(self.inner, name)

    def _wrap(self, fn, *a):
        try:
            return fn(*a)
        except BaseException as e:      # noqa: BLE001
            if getattr(e, "infra", False) or type(e).__name__ in ("WslEngineError", "ProviderEngineError"): self.failures += 1
            raise

    def run_public(self, artifact, task): return self._wrap(self.inner.run_public, artifact, task)

    def run_experiment(self, artifact, experiment): return self._wrap(self.inner.run_experiment, artifact, experiment)


class ExecCallRecorder:
    """Records every ExecutionBackend call EXACTLY as returned to the controller, one JSON line per call (offline replay input of tools/el_bank_replay_check.py):
    {"method": "run_public"|"run_experiment", "artifact_id", "experiment": dict|null, "result": CheckOutcome list|CheckOutcome}. An exception is not recorded (it propagates)."""

    def __init__(self, inner: Any, path: Path):
        self.inner, self.path = inner, Path(path)

    def __getattr__(self, name): return getattr(self.inner, name)

    def _append(self, method: str, artifact, experiment, result) -> None:
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"method": method, "artifact_id": artifact.artifact_id, "experiment": experiment, "result": result}, ensure_ascii=False, default=str) + "\n")

    def run_public(self, artifact, task):
        r = self.inner.run_public(artifact, task); self._append("run_public", artifact, None, r); return r

    def run_experiment(self, artifact, experiment):
        r = self.inner.run_experiment(artifact, experiment); self._append("run_experiment", artifact, experiment, r); return r


class _EngineFailed(RuntimeError):
    pass


_EngineFailed.__name__ = "EngineFailure"      # recorded as error_class so the run-level breaker counts it


# ------------------------------------------------------------------ inline critic (diagnostic only)
def _default_key_loader() -> Dict[str, str]:
    import laya_int1_stepE as SE      # tools/: load_keys() reads .secrets/nvidia_api_key_<n>.txt (n >= 2) into memory; importing it reads nothing and calls nothing
    return SE.load_keys()


class GlmInlineCritic:
    """Sends the blinded packets of ONE task to the critic model right now (BatchRunner / PcSender / broker exactly as tools/el_arch1_glm_live.py) and returns the parsed responses.
    transport=None -> the real streamed NVIDIA transport with keys from key_loader (values never printed, logged or returned); an injected transport (tests, selftest) replaces only the network call.
    Every failure is a RETURNED state (api_unavailable / critic_failed), never an exception: the run continues and the label is never blocked or faked."""

    def __init__(self, transport: Optional[Callable] = None, key_loader: Optional[Callable[[], Dict[str, str]]] = None, max_retries: int = 1,
                 read_timeout_s: float = CRITIC_READ_TIMEOUT_S, total_timeout_s: float = CRITIC_TOTAL_TIMEOUT_S):
        """read_timeout_s = how long ONE streamed call may wait for its first response bytes (reasoning max can take minutes before the first token; the old 120 s was too short for bank-sized
        packets); total_timeout_s = cap for the whole stream. Reasoning effort stays max, <= 39 sends/key/min, max_retries 1 (the retry goes to the next key)."""
        self.transport, self.key_loader, self.max_retries = transport, key_loader, max_retries
        self.read_timeout_s, self.total_timeout_s = float(read_timeout_s), float(total_timeout_s)

    def settings(self) -> Dict[str, Any]:
        return {"read_timeout_s": self.read_timeout_s, "total_timeout_s": self.total_timeout_s, "max_retries": self.max_retries, "reasoning_effort": "max", "max_sends_per_key_per_minute": 39}

    def answer(self, tid: str, requests: Dict[str, Dict[str, Any]], task_dir: Path) -> Dict[str, Any]:
        t0 = time.time()
        base = {"state": "api_unavailable", "responses": {}, "detail": "", "accounting": None, "seconds": 0.0, "model": "z-ai/glm-5.3-flash"}
        try:
            from evidence_ladder.critic_batch import validator as V, batch as BT, accounting as ACC
            from evidence_ladder.critic_batch.config import load_concurrency
            from evidence_ladder.critic_batch.transport import make_transport
            from evidence_ladder.models import glm, experiments as EXP
            keys: Dict[str, str] = {}
            if self.transport is None:
                try:
                    keys = (self.key_loader or _default_key_loader)()
                except Exception as e:      # noqa: BLE001
                    return dict(base, detail="key loader failed: %s (zero calls made)" % type(e).__name__, seconds=round(time.time() - t0, 2))
                if not keys:
                    return dict(base, detail="no readable key file (.secrets/nvidia_api_key_<n>.txt, n >= 2); zero calls made", seconds=round(time.time() - t0, 2))
            pdir = Path(task_dir) / "glm" / "critic_packets"
            if pdir.exists(): shutil.rmtree(pdir)
            V.write_packets(pdir, requests)                    # validates blinding / allow-lists; a bad packet raises here
            cdir = Path(task_dir) / "glm" / "critic_batch"
            if cdir.exists(): shutil.rmtree(cdir)
            cfg = glm.GlmConfig(reasoning_params=dict(BT.REASONING), allowed_kinds=EXP.SUPPORTED_KINDS)
            labels = list(keys) or ["key_1"]
            tr = self.transport or make_transport(keys, read_timeout_s=self.read_timeout_s, total_timeout_s=self.total_timeout_s)
            runner = BT.BatchRunner(pdir, cdir, tr, labels, load_concurrency(), max_retries=self.max_retries, glm_config=cfg)
            sent = runner.run()
            resp: Dict[str, Any] = {}
            rp = cdir / "responses.jsonl"
            if rp.is_file():
                for ln in rp.read_text("utf-8").splitlines():
                    if ln.strip():
                        r = json.loads(ln)
                        if r.get("packet_id") and r.get("response") is not None and r.get("state", r.get("broker_state")) == "ok": resp[r["packet_id"]] = r["response"]
            try:
                acc = ACC.from_broker(runner.broker)
                acc = acc and acc["totals"]
            except Exception:      # noqa: BLE001
                acc = None
            return dict(base, state="applied" if resp else "api_unavailable", responses=resp, accounting=acc, seconds=round(time.time() - t0, 2),
                        detail="" if resp else "no packet reached state ok: %s" % json.dumps(sent["manifest"]["states"], sort_keys=True), packets=len(requests),
                        stop_all=bool(sent["manifest"].get("stop_all_critic_calls")))
        except BaseException as e:      # noqa: BLE001  never let the critic break the run; the exception text is dropped (it could carry secret text)
            if isinstance(e, KeyboardInterrupt): raise
            return dict(base, state="critic_failed", detail="%s (details withheld)" % type(e).__name__, seconds=round(time.time() - t0, 2))


# ------------------------------------------------------------------ one task
def _make_controller(limits, with_timeline: bool = True):
    from evidence_ladder.controller.controller import EvidenceLadderController
    tl = None
    if with_timeline:
        try:
            from evidence_ladder.controller.timeline import Timeline as CTimeline
            tl = CTimeline()
        except Exception:      # noqa: BLE001
            tl = None
    return (EvidenceLadderController(limits=limits, timeline=tl) if tl is not None else EvidenceLadderController(limits=limits)), tl


def run_task(rec_in: Dict[str, Any], gen_backend: Any, exe_backend: Any, out: Path, tl: B.Timeline, *, policy: Dict[str, Any], wall_cap_s: float, state_hook: Optional[Callable] = None,
             worker: int = 0, engine_label: str = "mxc", max_new_tokens: Optional[int] = None, critic_client: Optional[GlmInlineCritic] = None, l2_version: int = L2_VERSION,
             rec_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from evidence_ladder.common.interfaces import ControllerContext, TaskSpec
    from evidence_ladder.controller import generation as GEN, invariants as INV
    from evidence_ladder.controller.limits import load_limits
    from evidence_ladder.models import qwen as Q, glm, experiments as EXP
    from evidence_ladder.models.broker import StagedCriticBroker
    from evidence_ladder.orchestrator import isolation
    tid = rec_in["task_id"]
    d = out / "tasks" / tid
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    info = gen_backend.model_info() if hasattr(gen_backend, "model_info") else {}
    rec: Dict[str, Any] = {"task": tid, "pass": "local", "thinking_policy": policy, "scorer": "none", "worker": worker, "gpu": getattr(gen_backend, "gpu_name", None), "engine": engine_label,
                           "label": LABEL, "wall_cap_s": wall_cap_s, "l2_version": l2_version, "profile_id": info.get("profile_id"), "profile_sha256": info.get("profile_sha256"), "start_utc": B.utc_now(),
                           "task_prompt_sha256": B.sha(rec_in["prompt"]), "commit": git_commit(), "source_sha256": getattr(exe_backend, "source_sha256", None) or SOURCE_SHA256,
                           "toolchain": (getattr(exe_backend, "toolchains", None) or {}).get(rec_in["language"])}
    t_start = time.time()
    try:
        task = TaskSpec(tid, ("py1:" if rec_in["language"] == "python" else "java1:") + tid, rec_in["language"], rec_in["language"], rec_in["prompt"], rec_in["public_tests"], rec_in.get("difficulty", ""))
        if hasattr(gen_backend, "language"): gen_backend.language = rec_in["language"]
        exe = B.TimedExec(ExecCallRecorder(exe_backend, d / "exec_calls.jsonl"), tl, tid)
        shim = ThinkShimGen(gen_backend)
        recorder = B.RecordingGen(shim, tid)
        gen = B.TimedGen(recorder, tl, tid, gpu=0)
        glm_cfg = glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=EXP.SUPPORTED_KINDS)
        critic = StagedCriticBroker(d / "glm" / "broker_ledger.jsonl", config=glm_cfg, max_retries=1)
        cfg = {"mode": "ladder", "k": 4, "l2": True, "l2_n": 6, "l2_version": l2_version, "thinking": False, "discriminate": True, "critic": True, "strategies": list(GEN.FROZEN_STRATEGIES),
               "max_new_tokens": max_new_tokens or policy["ceiling"], "max_rounds": 2, "mutant_limit": 8}
        rec["controller_config"] = cfg
        ctx = ControllerContext(gen, exe, Q.TokenBudget(tid, "D0", GENERATED_TOKEN_CAP), "D0", critic, None, SEEDS, cfg)
        i8 = isolation.deep_context_check(ctx)
        if i8: raise RuntimeError("I8 deep scan found evaluation-only material in the controller context: %s" % i8[:3])
        limits = dataclasses.replace(load_limits("ladder"), max_wall_seconds=float(wall_cap_s))
        ctl, ctl_tl = _make_controller(limits)
        f0 = getattr(exe_backend, "failures", 0)
        with tl.span("task", task=tid, language=rec_in["language"], shard=worker):
            res = ctl.run_problem(task, ctx)
        j = ctl.last_journal
        crit_rec: Dict[str, Any] = {"called": False, "state": "not_needed"}
        label_before = res.answer_label
        pend = [pid for pid in critic.packets if critic.completed(pid) is None and critic.terminal_state(pid) is None]
        if res.stop_details == "critic_pending" and pend:
            crit_rec = {"called": False, "state": "pending", "packets": len(pend)}
            if critic_client is not None:
                reqs = {pid: critic.packets[pid]["request"] for pid in pend}
                with tl.span("glm_critic", task=tid, packets=len(reqs)):
                    ans = critic_client.answer(tid, reqs, d)
                crit_rec = {"called": True, "state": ans["state"], "detail": ans.get("detail", ""), "packets": len(reqs), "responses": len(ans.get("responses", {})), "accounting": ans.get("accounting"),
                            "seconds": ans.get("seconds"), "model": ans.get("model"), "applied_in_controller": False, "settings": critic_client.settings()}
                if ans.get("responses"):
                    try:        # same run: the recorded generations are replayed (prompt-hash checked, nothing regenerated) and the critic answers are applied
                        replay = B.SequenceReplayGen(recorder.calls, {"task": tid})
                        critic2 = StagedCriticBroker(d / "glm" / "broker_ledger_after_critic.jsonl", config=glm_cfg, max_retries=1)
                        cfg2 = dict(cfg, critic_responses=ans["responses"])
                        ctx2 = ControllerContext(B.TimedGen(replay, tl, tid, gpu=None), exe, Q.TokenBudget(tid, "D0", GENERATED_TOKEN_CAP), "D0", critic2, None, SEEDS, cfg2)
                        ctl2, ctl_tl2 = _make_controller(limits)
                        with tl.span("task_after_critic", task=tid, shard=worker):
                            res2 = ctl2.run_problem(task, ctx2)
                        (d / "journal_before_critic.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False, default=str, sort_keys=True) + "\n" for e in j.entries()), encoding="utf-8")
                        if ctl_tl is not None: (d / "controller_timeline_before_critic.json").write_text(json.dumps({"rows": ctl_tl.rows(), "summary": ctl_tl.summary()}, default=str), encoding="utf-8")
                        res, j, ctl_tl, critic = res2, ctl2.last_journal, ctl_tl2, critic2
                        crit_rec["applied_in_controller"] = True
                        crit_rec["label_before_critic"], crit_rec["label_after_critic"] = label_before, res.answer_label
                        ev_ind = sorted({(e.get("independence"), e.get("status")) for e in j.evidence() if e.get("check_id") == "critic_experiment_v1"}, key=str)
                        crit_rec["critic_evidence"] = [{"independence": a, "status": b} for a, b in ev_ind]      # must all be model_derived: the critic is diagnostic and cannot raise a trusted label
                        crit_rec["replay_exhausted"] = replay.exhausted()
                    except Exception as e:      # noqa: BLE001  the first result stands; the failure is recorded, the run continues
                        crit_rec["apply_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        if getattr(exe_backend, "failures", 0) > f0:
            raise _EngineFailed("the execution engine failed %d time(s) during this task; its result is not trustworthy and is not kept as DONE" % (exe_backend.failures - f0))
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
        if state_hook is not None:      # optional internal-state capture (Agent 2's StateCapture): after the decision, never raises, never touches the journal, never runs when disabled
            try:
                sc = state_hook(d / "states")
                if sc is not None and getattr(sc, "enabled", True):
                    for e in j.entries("candidate"):
                        sc.capture(tid, e["data"]["id"], rec_in["prompt"], e["data"]["source"])
            except Exception as e:      # noqa: BLE001
                rec["state_capture_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        v = B.verdicts(res, j)
        gens = [g for c in recorder.calls for g in c["generations"]]
        hits = sum(1 for g in gens if g["tokens_generated"] >= policy["ceiling"])
        left = [] if crit_rec.get("applied_in_controller") else sorted(pid for pid in critic.packets if critic.completed(pid) is None and critic.terminal_state(pid) is None)
        rec.update(language=rec_in["language"], difficulty=rec_in.get("difficulty", ""), stop_cause=res.stop_cause, stop_details=res.stop_details, answer_label=res.answer_label, rounds=res.rounds,
                   journal_hash=res.journal_hash, journal_chain_ok=j.verify_chain(), invariants=inv, candidates=len(j.entries("candidate")), public=B.public_example_stats(j), budget=res.budget_record,
                   tokens_generated=sum(g["tokens_generated"] for g in gens), tokens_prompt=sum(g["tokens_prompt"] for g in gens), generate_calls=len(recorder.calls), ceiling_hits=hits,
                   generation_failed_at_ceiling=hits > 0, think_shim_count=shim.count, pending_critic_packets=left, critic_called=crit_rec["called"], critic=crit_rec, **v)
        rec["status"] = "OK"
    except BaseException as e:      # noqa: BLE001  a crash on one task is a recorded result, never a hidden one; no DONE marker, so --resume retries it
        if isinstance(e, KeyboardInterrupt): raise
        rec.update(status="ERROR", error="%s: %s" % (type(e).__name__, str(e)[:400]), error_class=type(e).__name__, trace=traceback.format_exc()[-2500:])
        tl.event("task_error", task=tid, error=type(e).__name__)
    if hasattr(exe_backend, "drain_exec_log"):
        try:
            logs = exe_backend.drain_exec_log()
            if logs: B.write_atomic(d / "exec_results.jsonl", "".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in logs))
        except Exception:      # noqa: BLE001
            pass
    if rec_extra: rec.update(rec_extra)
    rec["end_utc"] = B.utc_now(); rec["seconds"] = round(time.time() - t_start, 1)
    B.write_atomic(d / "result.json", json.dumps(rec, indent=1, ensure_ascii=False, default=str, sort_keys=True))
    if rec["status"] == "OK":
        B.write_atomic(d / "DONE", rec["end_utc"])          # LAST
    return rec


class ReplayModel:
    """Serves the SAVED generations of a finished task (gen_calls.json) in order, prompt-hash checked; used by --retry-critic so nothing is regenerated and no model is needed."""
    stub = False
    gpu_name = None
    language = "python"

    def __init__(self, calls: Sequence[Dict[str, Any]], info: Dict[str, Any]):
        self._replay, self._info = B.SequenceReplayGen(calls, {"retry": True}), dict(info)

    def model_info(self) -> Dict[str, Any]: return self._info

    def generate(self, request): return self._replay.generate(request)


def needs_critic_retry(d: Path) -> bool:
    """A finished task whose critic did not complete: api_unavailable / critic_failed / pending, or packets still pending in result.json (and the saved generations exist)."""
    try:
        if not ((d / "DONE").is_file() and (d / "gen_calls.json").is_file() and (d / "result.json").is_file()):
            return False
        r = json.loads((d / "result.json").read_text("utf-8"))
        c = r.get("critic") or {}
        return r.get("status") == "OK" and not c.get("applied_in_controller") and (c.get("state") in ("api_unavailable", "critic_failed", "pending") or bool(r.get("pending_critic_packets")))
    except (OSError, ValueError):
        return False


def _free_dir(base: Path, stem: str) -> Path:
    n = 1
    while (base / ("%s.v%d" % (stem, n))).exists(): n += 1
    return base / ("%s.v%d" % (stem, n))


def retry_critic_tasks(out: Path, by: Dict[str, Dict[str, Any]], exe: Any, tl: B.Timeline, critic_client: GlmInlineCritic, *, policy: Dict[str, Any], wall_cap_s: float, engine_label: str,
                       l2_version: int = L2_VERSION) -> List[Dict[str, Any]]:
    """For every finished task whose critic did not complete: re-run ONLY the critic step with the SAVED generations (replayed, prompt-hash checked, nothing regenerated) in a staging dir; if the answer
    was applied, the old task dir is ARCHIVED (tasks_archive/<id>.vN, never deleted) and the new one takes its place atomically with retry provenance in result.json. If the retry fails again, the old task
    stays untouched and the attempt is archived (retry_archive/) and logged. The label can only move through trusted evidence: the critic stays diagnostic (the same controller path as the inline critic)."""
    log: List[Dict[str, Any]] = []
    for d in sorted((out / "tasks").glob("*")):
        tid = d.name
        if tid not in by or not needs_critic_retry(d):
            continue
        t0 = B.utc_now()
        old = json.loads((d / "result.json").read_text("utf-8"))
        entry: Dict[str, Any] = {"task": tid, "started_utc": t0, "previous_critic_state": (old.get("critic") or {}).get("state"), "settings": critic_client.settings()}
        stage = out / "_retry" / ("%s-%d" % (tid, int(time.time())))
        (stage / "tasks").mkdir(parents=True, exist_ok=True)
        try:
            calls = json.loads((d / "gen_calls.json").read_text("utf-8"))
            rep = ReplayModel(calls, {"backend": "replay_of_saved_generations", "profile_id": old.get("profile_id"), "profile_sha256": old.get("profile_sha256"), "snapshot_verified": True})
            extra = {"retry": {"critic_retry": True, "started_utc": t0, "critic_settings": critic_client.settings(), "generations": "replayed from gen_calls.json (nothing regenerated)",
                               "previous": {"critic": old.get("critic"), "pending_critic_packets": old.get("pending_critic_packets"), "answer_label": old.get("answer_label"), "stop_cause": old.get("stop_cause"),
                                            "journal_hash": old.get("journal_hash")}}}
            new = run_task(by[tid], rep, exe, stage, tl, policy=policy, wall_cap_s=wall_cap_s, engine_label=engine_label, critic_client=critic_client, l2_version=l2_version, rec_extra=extra)
            applied = new.get("status") == "OK" and (new.get("critic") or {}).get("applied_in_controller") is True and (stage / "tasks" / tid / "DONE").is_file()
            entry.update(new_status=new.get("status"), new_critic_state=(new.get("critic") or {}).get("state"), applied=applied, label_before=old.get("answer_label"), label_after=new.get("answer_label"))
            if applied:
                arch = out / "tasks_archive"; arch.mkdir(exist_ok=True)
                dest = _free_dir(arch, tid)
                os.replace(str(d), str(dest))                                   # the old evidence is kept, never deleted
                os.replace(str(stage / "tasks" / tid), str(d))
                entry["archived_old_to"] = str(dest.relative_to(out)).replace("\\", "/")
            else:
                ra = out / "retry_archive"; ra.mkdir(exist_ok=True)
                dest = _free_dir(ra, tid + ".failed")
                os.replace(str(stage), str(dest))
                entry["failed_attempt_kept_at"] = str(dest.relative_to(out)).replace("\\", "/")
        except BaseException as e:      # noqa: BLE001  a failed retry never touches the finished task
            if isinstance(e, KeyboardInterrupt): raise
            entry.update(applied=False, error="%s: %s" % (type(e).__name__, str(e)[:300]))
        entry["ended_utc"] = B.utc_now()
        log.append(entry)
        with open(out / "retry_critic_log.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    shutil.rmtree(out / "_retry", ignore_errors=True)
    return log


def not_tested(rec_in: Dict[str, Any], out: Path, tl: B.Timeline, cap: Dict[str, Any], policy: Dict[str, Any], engine_label: str) -> Dict[str, Any]:
    """A task the chosen sandbox cannot run (Java on MXC): recorded explicitly, never executed elsewhere, never faked. No DONE marker (it is not a finished run); a NOT_TESTED marker instead."""
    tid = rec_in["task_id"]
    d = out / "tasks" / tid
    d.mkdir(parents=True, exist_ok=True)
    now = B.utc_now()
    rec = {"task": tid, "pass": "local", "thinking_policy": policy, "scorer": "none", "language": rec_in["language"], "difficulty": rec_in.get("difficulty", ""), "status": "NOT_TESTED",
           "stop_cause": cap["status"], "answer_label": None, "reason": cap["reason"], "engine": engine_label, "label": LABEL, "start_utc": now, "end_utc": now, "seconds": 0.0}
    B.write_atomic(d / "result.json", json.dumps(rec, indent=1, ensure_ascii=False, sort_keys=True))
    B.write_atomic(d / "NOT_TESTED", cap["status"])
    tl.event("task_not_tested", task=tid, reason=cap["status"])
    return rec


# ------------------------------------------------------------------ the run
def merge_timeline(out: Path) -> Dict[str, Any]:
    evs = [e for e in B.read_events(out / "timeline" / "main.jsonl", *sorted((out / "timeline").glob("worker*.jsonl"))) if "event" in e]
    B.write_atomic(out / "TIMELINE.jsonl", "".join(json.dumps(e, sort_keys=True) + "\n" for e in evs))
    summ = B.summarize(evs)
    B.write_atomic(out / "TIMELINE_SUMMARY.json", json.dumps(summ, indent=1))
    return summ


def finalize_reports(out: Path, manifest: Optional[Path], expect_engine: str, expected_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Automatic end of the run: BANK results (tools/el_bank40_driver.py compile) and the conformance audit (tools/el_bank_conformance.py). Never raises: a failure is recorded in the summary."""
    rec: Dict[str, Any] = {}
    try:
        import el_bank40_driver as DR
        run = DR.read_run(out)
        comp = DR.compile_results(run)
        B.write_atomic(out / "BANK_RESULTS.json", json.dumps(comp, indent=1, ensure_ascii=False, default=str))
        B.write_atomic(out / "BANK_RESULTS.md", DR.render_md(comp))
        rec["bank_results"] = {"tasks_finished": comp["tasks_finished"], "labels_by_language": comp["labels_by_language"], "not_tested": comp.get("not_tested")}
    except Exception as e:      # noqa: BLE001
        rec["bank_results_error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    try:
        import el_bank_conformance as CFM
        tp = None
        if expected_ids:      # the conformance tool reads {"tasks": [ids]}; the manifest format of el_bank80_rest_v1.json differs, so the expected ids are written normalised
            B.write_atomic(out / "EXPECTED_TASKS.json", json.dumps({"tasks": list(expected_ids)}))
            tp = str(out / "EXPECTED_TASKS.json")
        rep = CFM.audit_run(str(out), expect_engine, False, tp)
        B.write_atomic(out / "CONFORMANCE.json", json.dumps(rep, indent=1, sort_keys=True, default=str))
        B.write_atomic(out / "CONFORMANCE.md", CFM.to_markdown(rep))
        rec["pipeline_verdict"] = rep.get("verdict")
    except Exception as e:      # noqa: BLE001
        rec["conformance_error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    return rec


def stage_coverage(out: Path) -> Dict[str, Any]:
    """Across the finished tasks of a run: how many reached each stage (from the journals, not from claims). Printed as one line at the end of every run."""
    c = {"executed": 0, "public_pass": 0, "l2_reached": 0, "l2_validated": 0, "clusters_ge2": 0, "critic_triggered": 0, "critic_called": 0, "critic_applied": 0, "repair": 0}
    for d in sorted((Path(out) / "tasks").glob("*")):
        if not (d / "DONE").is_file() or not (d / "journal.jsonl").is_file():
            continue
        c["executed"] += 1
        j = [json.loads(l) for l in (d / "journal.jsonl").read_text("utf-8").splitlines() if l.strip()]
        r = json.loads((d / "result.json").read_text("utf-8"))
        kinds = [e["kind"] for e in j]
        if (r.get("public") or {}).get("candidates_passing_all"): c["public_pass"] += 1
        l2 = [e["data"] for e in j if e["kind"] == "l2_checks"]
        if l2: c["l2_reached"] += 1
        if l2 and l2[0].get("validated"): c["l2_validated"] += 1
        bc = [e["data"] for e in j if e["kind"] == "behaviour_clusters"]
        if bc and (bc[-1].get("extended_clusters") or 0) >= 2: c["clusters_ge2"] += 1
        if any(e["kind"] == "critic_trigger" and e["data"].get("triggered") for e in j): c["critic_triggered"] += 1
        if (r.get("critic") or {}).get("called"): c["critic_called"] += 1
        if (r.get("critic") or {}).get("applied_in_controller"): c["critic_applied"] += 1
        if any(e["kind"] == "candidate" and str(e["data"].get("origin", "")).startswith("repair:") for e in j) or any(e["kind"] == "generation_request" and e["data"].get("purpose") == "repair" for e in j): c["repair"] += 1
    c["line"] = ("STAGE COVERAGE over %d finished task(s): public-pass %d, L2 reached %d (validated %d), clusters>=2 %d, critic triggered %d / called %d / applied %d, repair round %d"
                 % (c["executed"], c["public_pass"], c["l2_reached"], c["l2_validated"], c["clusters_ge2"], c["critic_triggered"], c["critic_called"], c["critic_applied"], c["repair"]))
    return c


def java_capability(sandbox: str) -> Dict[str, Any]:
    from evidence_ladder.execution.provider_engine import java_capability as jc
    return jc(sandbox)


def run(out: Path, tasks: Sequence[Dict[str, Any]], gen_factory: Callable[[Path], Any], exe_factory: Callable[[Path], Any], *, resume: bool = False, wall_cap_s: float = WALL_CAP_S,
        policy: Optional[Dict[str, Any]] = None, state_hook: Optional[Callable[[Path], Any]] = None, gpu_sampler: bool = False, max_new_tokens: Optional[int] = None, sandbox: str = "mxc",
        critic_client: Optional[GlmInlineCritic] = None, resources_fn: Callable[[], Dict[str, Any]] = system_resources, min_free_ram_gb: float = MIN_FREE_RAM_GB,
        min_free_vram_gb: float = MIN_FREE_VRAM_GB, min_free_commit_gb: float = 0.0, require_verified_model: bool = True, manifest: Optional[Path] = None, finalize: bool = True, l2_version: int = L2_VERSION, retry_critic: bool = False) -> Dict[str, Any]:
    out = Path(out).resolve()
    if retry_critic and not resume:
        raise ValueError("--retry-critic is used together with --resume (it works on finished tasks of an existing run)")      # ABSOLUTE everywhere: a relative scratch path made the MXC container look for runner.py in C:\Windows\system32
    policy = B.check_policy(policy or policy_for())
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("output dir is not empty (use --resume to continue, or a new dir): %s" % out)
    (out / "timeline").mkdir(parents=True, exist_ok=True)
    (out / "tasks").mkdir(exist_ok=True)
    tl = B.Timeline(out / "timeline" / "main.jsonl", "main")
    wtl = B.Timeline(out / "timeline" / "worker0.jsonl", "w0")
    critic_client = GlmInlineCritic() if critic_client is None else critic_client
    RS: Dict[str, Any] = {"kernel": "el_bank_local_run", "mode": "local", "started_utc": B.utc_now(), "thinking_policy": policy, "scorer": "none", "label": LABEL, "engine": sandbox, "sandbox": sandbox,
                          "l2_version": l2_version, "commit": git_commit(), "config_sha256": config_hashes(), "out_dir_abs": str(out), "controller_limits": {"max_wall_seconds": wall_cap_s, "profile": "ladder"}, "token_cap": GENERATED_TOKEN_CAP,
                          "seeds": list(SEEDS), "internal_states": "disabled" if state_hook is None else "hook", "preflight": {"min_free_ram_gb": min_free_ram_gb, "min_free_vram_gb": min_free_vram_gb, "min_free_commit_gb": min_free_commit_gb},
                          "critic": {"state": "inline", "model": "z-ai/glm-5.3-flash", "settings": critic_client.settings(), "note": "called inside the run when discrimination triggers a packet; diagnostic only; api_unavailable / critic_failed are recorded and the run continues"},
                          "gpus": 1, "notes": []}
    try:
        RS["public_only_guard"] = public_guard(tasks)
    except AssertionError as e:
        RS.update(verdict="PREFLIGHT_REFUSED", error="tasks are not public-only: %s" % str(e)[:300]); B.write_atomic(out / "RUN_SUMMARY.json", json.dumps(RS, indent=1, default=str)); return RS
    global SOURCE_SHA256
    SOURCE_SHA256 = source_tree_sha256()
    RS["source_sha256"], RS["code_commit"], RS["engine_status"] = SOURCE_SHA256, RS["commit"], "OK"
    langs = sorted({t["language"] for t in tasks})

    def split(caps: Dict[str, Dict[str, Any]]):
        bl = [t["task_id"] for t in tasks if not (caps.get(t["language"]) or {}).get("runnable")]
        return bl
    caps = {l: (java_capability(sandbox) if l == "java" else {"runnable": True, "status": "RUNNABLE", "reason": ""}) for l in langs}      # preliminary (static) until the backend is probed after its start
    RS["java_capability"] = caps.get("java")
    blocked = split(caps)
    finished_before = B.finished_task_ids(out)
    todo, skipped = B.plan_work([t["task_id"] for t in tasks if t["task_id"] not in blocked], finished_before)
    RS["resumed_tasks"] = skipped
    RS["not_tested"] = {tid: caps[by_lang]["status"] for tid in blocked for by_lang in [next(t["language"] for t in tasks if t["task_id"] == tid)]}
    by = {t["task_id"]: t for t in tasks}
    verdict = "RUNNING"
    t0 = time.time()

    def save(final: bool = False):
        done = sorted(p.parent.name for p in (out / "tasks").glob("*/DONE"))
        errs = sorted(p.parent.name for p in (out / "tasks").glob("*/result.json") if not (p.parent / "DONE").exists() and not (p.parent / "NOT_TESTED").exists())
        RS.update(done=len(done), errors=errs, pending_not_started=[t for t in todo if t not in done and t not in errs], verdict=verdict, updated_utc=B.utc_now(), total_seconds=round(time.time() - t0, 1))
        RS["timeline_summary"] = merge_timeline(out)
        if final:
            RS["ended_utc"] = B.utc_now()
        B.write_atomic(out / "RUN_SUMMARY.json", json.dumps(RS, indent=1, default=str))
    stop_gpu = threading.Event()
    exe = gen = raw_exe = None
    consecutive = 0
    try:
        tl.event("run_start", tasks=len(todo), resumed=len(skipped))
        # ---- preflight (nothing is loaded or run before it passes)
        with tl.span("preflight"):
            raw_exe = exe_factory(out)
            if hasattr(raw_exe, "probe_toolchains"): raw_exe.source_sha256 = SOURCE_SHA256
            hello = (raw_exe.start(langs) if hasattr(raw_exe, "probe_toolchains") else (raw_exe.start() if hasattr(raw_exe, "start") else None))
            exe = EngineGuard(raw_exe)
            if hello and hello.get("toolchains") is not None:      # REAL preflight: each language of the manifest was probed INSIDE the backend before the model is loaded
                caps = {l: hello["toolchains"].get(l) or {"runnable": False, "status": "NOT_TESTED_NO_TOOLCHAIN", "reason": "language was not probed"} for l in langs}
                RS["toolchains"], RS["java_capability"] = caps, caps.get("java")
                blocked = split(caps)
                todo, skipped = B.plan_work([t["task_id"] for t in tasks if t["task_id"] not in blocked], finished_before)
                RS["resumed_tasks"] = skipped
                RS["not_tested"] = {tid: caps[next(t["language"] for t in tasks if t["task_id"] == tid)]["status"] for tid in blocked}
            if hello and hello.get("image_digest"): RS["image_digest"] = hello["image_digest"]
            if sandbox == "wslc" and hello and hello.get("engine"): RS["engine"] = hello["engine"]
            RS["jail"] = {"status": "PASS", "engine": (hello or {}).get("engine") or sandbox, "java_home": (hello or {}).get("jdk"), "bwrap_swapped": None,
                          "handshake": {k: (hello or {}).get(k) for k in ("os", "python", "bwrap_version", "java_version", "jail_probe_ran", "backend")}}
            res = resources_fn()
            RS["preflight"]["resources"] = res
            why = check_resources(res, min_free_ram_gb, min_free_vram_gb, min_free_commit_gb)
            if why:
                verdict = "PREFLIGHT_REFUSED"; RS["error"] = "; ".join(why); RS["notes"].append("preflight refused: nothing was loaded or run"); save(final=True); return RS
        for tid in blocked:
            not_tested(by[tid], out, wtl, caps[by[tid]["language"]], policy, RS["jail"]["engine"])
        with tl.span("model_load"):
            gen = gen_factory(out)
        info = gen.model_info() if hasattr(gen, "model_info") else {}
        RS["model"] = info
        RS["profile"] = {"id": info.get("profile_id") or policy["profile_id"], "sha256": info.get("profile_sha256")}
        RS["profile_sha256"] = info.get("profile_sha256")
        if require_verified_model and not info.get("snapshot_verified"):
            verdict = "PREFLIGHT_REFUSED"; RS["error"] = "the model snapshot is not verified (snapshot_verified is not true)"; RS["notes"].append("preflight refused: no task was run"); save(final=True); return RS
        if gpu_sampler:
            def sampler():
                while not stop_gpu.is_set():
                    try:
                        o = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
                        with open(out / "timeline" / "gpu_util.jsonl", "a", encoding="utf-8") as f:
                            for ln in o.strip().splitlines():
                                a = [x.strip() for x in ln.split(",")]
                                if len(a) == 3: f.write(json.dumps({"ts_utc": B.utc_now(), "ts": round(time.time(), 3), "gpu": int(a[0]), "util_pct": int(a[1]), "mem_mib": int(a[2])}) + "\n")
                    except Exception:      # noqa: BLE001
                        pass
                    stop_gpu.wait(30)
            threading.Thread(target=sampler, daemon=True).start()
        save()
        verdict = "OK"
        for tid in todo:
            if (out / "STOP").exists():
                RS["notes"].append("STOP file found: stopped before %s" % tid); verdict = "STOPPED"; break
            r = run_task(by[tid], gen, exe, out, wtl, policy=policy, wall_cap_s=wall_cap_s, state_hook=state_hook, max_new_tokens=max_new_tokens, engine_label=RS["jail"]["engine"],
                         critic_client=critic_client, l2_version=l2_version)
            infra = r.get("status") == "ERROR" and r.get("error_class") in ("EngineFailure", "WslEngineError", "ProviderEngineError")
            consecutive = consecutive + 1 if infra else 0
            if consecutive >= MAX_CONSECUTIVE_ENGINE_FAILURES:
                verdict = "ENGINE_UNAVAILABLE"; RS["notes"].append("%d consecutive engine failures: stopped (fail closed); fix the engine and run again with --resume" % consecutive); save(); break
            save()
        else:
            n_ok = len(list((out / "tasks").glob("*/DONE")))
            verdict = "OK" if n_ok == len(tasks) - len(blocked) else "PARTIAL"
        if retry_critic and verdict in ("OK", "PARTIAL"):      # after the remaining tasks: re-run ONLY the critic step of finished tasks whose critic did not complete (saved generations replayed)
            with tl.span("retry_critic"):
                RS["critic_retry"] = retry_critic_tasks(out, by, exe, wtl, critic_client, policy=policy, wall_cap_s=wall_cap_s, engine_label=RS["jail"]["engine"], l2_version=l2_version)
            save()
    except KeyboardInterrupt:
        verdict = "INTERRUPTED"; RS["notes"].append("interrupted by the user; finished tasks are kept, run again with --resume")
    except BaseException as e:      # noqa: BLE001  setup failure (engine handshake / model load): nothing ran, receipt says why
        if getattr(e, "infra", False) or type(e).__name__ in ("WslEngineError", "ProviderEngineError"): RS["engine_status"] = "BLOCKED: %s" % str(e)[:300]
        verdict = "SETUP_FAILED"; RS["error"] = "%s: %s" % (type(e).__name__, str(e)[:500]); RS["trace"] = traceback.format_exc()[-3000:]
        if getattr(e, "infra", False) or type(e).__name__ in ("WslEngineError", "ProviderEngineError"): verdict = "ENGINE_UNAVAILABLE"
    finally:
        stop_gpu.set()
        tl.event("run_end", verdict=verdict)
        try:
            if raw_exe is not None and hasattr(raw_exe, "backend_info"): RS["engine_info"] = raw_exe.backend_info()
        except Exception:      # noqa: BLE001
            pass
        try:
            if raw_exe is not None and hasattr(raw_exe, "close"): raw_exe.close()
        except Exception:      # noqa: BLE001
            pass
        save(final=True)
        if finalize and verdict not in ("PREFLIGHT_REFUSED", "SETUP_FAILED") and any((out / "tasks").glob("*/result.json")):
            RS["stage_coverage"] = stage_coverage(out)
            print(RS["stage_coverage"]["line"], flush=True)
            RS["reports"] = finalize_reports(out, manifest, "mxc" if sandbox == "mxc" else (str(RS["jail"]["engine"]) if sandbox == "wslc" else "wsl"), [t["task_id"] for t in tasks])
            RS["pipeline_verdict"] = RS["reports"].get("pipeline_verdict")
            save(final=True)
    return RS


def default_gen_factory(profile: str) -> Callable[[Path], Any]:
    def make(out: Path):
        from evidence_ladder.models import qwen as Q
        if not hasattr(Q, "build_local_backend"):
            raise RuntimeError("models.qwen.build_local_backend(profile_id, log_path=None, think_policy=None) is not available yet (Agent 1)")
        be = Q.build_local_backend(profile, log_path=str(Path(out) / "timeline" / "gen_log.jsonl"), think_policy=None)
        be._load()      # load ONCE here (sha256-verifies the pinned snapshot, then the model stays resident for every task); snapshot_verified is only true after this
        return be
    return make


def exe_factory_for(sandbox: str, trusted: bool = False) -> Callable[[Path], Any]:
    """mxc (default): the new MXC sandbox through the provider port (evidence_ladder/execution/provider_engine.py). wsl: the old WSL bwrap engine, only as an explicitly selected comparison. No fallback either way."""
    def make(out: Path):
        if sandbox == "mxc":
            from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
            return ProviderExecutionBackend(backend="mxc", scratch=Path(out).resolve() / "_engine", trusted=trusted)
        if sandbox == "wslc":      # the genuine MXC WSLC container backend (explicit; fail closed when it is not registered or cannot run: no fallback)
            from evidence_ladder.execution.provider_engine import ProviderExecutionBackend
            return ProviderExecutionBackend(backend="wslc", scratch=Path(out).resolve() / "_engine", trusted=trusted)
        if sandbox == "wsl":
            from evidence_ladder.execution.wsl_engine import WslExecutionBackend
            return WslExecutionBackend(repo=ROOT, stderr_log=Path(out) / "timeline" / "wsl_server.stderr.log")
        raise ValueError("--sandbox must be mxc, wslc or wsl (there is no auto mode)")
    return make


# ------------------------------------------------------------------ selftest: a tiny fake model, REAL MXC + real controller + inline critic over a fake transport
class SelfTestGen:
    """Stands in for the model ONLY in --selftest (stub = True). Three cases on real PUBLIC bank tasks:
      P1-02 RICH    4 candidates: the reference, two variants that pass every public example but differ on large values / long lists (3 behaviours -> clusters >= 2, the critic is triggered), and the
                    reference written as a loop; the L2 v2 answer is a valid JSON array of NEW inputs with the reference's outputs and a quote copied verbatim from the statement (L2 validates).
      P1-03 REPAIR  the first round's candidates are wrong (public examples fail), the repair prompt gets the failure, the repaired candidate is correct.
      other         lookup tables of the public examples with different defaults (cheap case)."""
    stub = True
    gpu_name = "selftest-no-gpu"
    language = "python"
    REF02 = "def solve(d):\n    return sum(x for x in d['nums'] if x % 2 == 0)\n"
    VARIANTS02 = [REF02,
                  "def solve(d):\n    return sum(x for x in d['nums'] if x % 2 == 0 and abs(x) < 1000)\n",
                  "def solve(d):\n    return 0 if len(d['nums']) > 50 else sum(x for x in d['nums'] if x % 2 == 0)\n",
                  "def solve(d):\n    t = 0\n    for x in d['nums']:\n        if x % 2 == 0:\n            t += x\n    return t\n"]
    TESTS02 = [([10, 3, -8], 2), ([1000, 2], 1002), ([2] * 60, 120), ([-2, -4], -6), ([0], 0), ([5, 6], 6)]
    REF03 = "def solve(d):\n    t = d['text']\n    return next((i for i, c in enumerate(t) if t.count(c) == 1), -1)\n"
    WRONG03 = "def solve(d):\n    return -1\n"

    def __init__(self, tasks: Sequence[Dict[str, Any]]):
        self.tasks = list(tasks)

    def model_info(self) -> Dict[str, Any]:
        return {"backend": "selftest-fake", "profile_id": "selftest", "profile_sha256": sha("selftest"), "snapshot_verified": True, "stub": True}

    def _task(self, prompt: str):
        return next((t for t in self.tasks if t["prompt"][:200] in prompt), self.tasks[0])

    def generate(self, req):
        from evidence_ladder.models import qwen as Q
        t = self._task(req.prompt)
        tid = t["task_id"]
        table = {json.dumps(c["input"], sort_keys=True): c["expected"] for c in t["public_tests"]}
        out = []
        for s in list(req.seeds)[:req.n]:
            lang = "python"
            if req.purpose == "tests":
                lang = "json"
                if tid == "P1-02":
                    quote = "including negative even integers"
                    assert quote in t["prompt"]
                    items = [{"input": {"nums": n}, "expected": e, "quote": quote} for n, e in self.TESTS02]
                else:
                    quote = t["prompt"].strip()[:40]
                    items = [{"input": c["input"], "expected": c["expected"], "quote": quote} for c in t["public_tests"][:4]]
                text = "```json\n" + json.dumps(items) + "\n```"
            else:
                if tid == "P1-02":
                    code = self.VARIANTS02[s % 4]
                elif tid == "P1-03":
                    code = self.REF03 if req.purpose == "repair" else self.WRONG03
                else:
                    code = "import json\n_T = %r\ndef solve(d):\n    return _T.get(json.dumps(d, sort_keys=True), %d)\n" % (table, s % 2)
                text = "```python\n" + code + "```"
            out.append(Q.make_generation(text, tokens_prompt=50, tokens_generated=60 + s, seed=s, prompt_template_id=req.prompt_template_id, prompt_sha256=B.sha(req.prompt), language=lang, wall_seconds=0.01))
        return out


def selftest_transport(packet, key_label):
    body = {"choices": [{"message": {"role": "assistant", "content": json.dumps({"experiment": {"kind": "function_call", "payload": "{}", "expected_by_candidate": {}}, "implications": "selftest critic: probe the behaviours"}),
                                     "reasoning_content": "selftest"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    return 200, body


BASE_KINDS = ("generation_request", "candidate", "evidence", "ranking", "decision", "stop")
RICH_KINDS = BASE_KINDS + ("l2_parse", "l2_checks", "behaviour_clusters", "critic_trigger", "critic")
RICH_TASK, REPAIR_TASK, CHEAP_TASK = "P1-02", "P1-03", "P1-13"


def _journal(d: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in (d / "journal.jsonl").read_text("utf-8").splitlines() if l.strip()]


def selftest_checks(out: Path, rs: Dict[str, Any], task_ids: Sequence[str]) -> List[str]:
    """-> list of MISSING stages (empty = the whole pipeline ran in this single process). RICH task: L2 VALIDATED, clusters >= 2, discrimination triggered, the inline critic called and applied with only
    model-derived (diagnostic) evidence, UTC rows everywhere. REPAIR task: a repair round whose candidate came from the repair prompt. Every task: the baseline stages and the per-task files."""
    miss: List[str] = []
    for tid in task_ids:
        d = out / "tasks" / tid
        for f in ("DONE", "result.json", "journal.jsonl", "gen_calls.json", "controller_timeline.json", "exec_results.jsonl", "critic_requests.json"):
            if not (d / f).is_file(): miss.append("%s: %s missing" % (tid, f))
        if not (d / "DONE").is_file() or not (d / "journal.jsonl").is_file(): continue
        j = _journal(d)
        kinds = {e["kind"] for e in j}
        r = json.loads((d / "result.json").read_text("utf-8"))
        for k in (RICH_KINDS if tid == RICH_TASK else BASE_KINDS):
            if k not in kinds: miss.append("%s: journal has no %r entry" % (tid, k))
        if r.get("l2_version") != 2: miss.append("%s: l2_version 2 not recorded" % tid)
        if (r.get("invariants") or {}).get("failed"): miss.append("%s: invariants failed: %s" % (tid, r["invariants"]["failed"]))
        ct = json.loads((d / "controller_timeline.json").read_text("utf-8")) if (d / "controller_timeline.json").is_file() else {}
        rows = [x for x in ct.get("rows", []) if x.get("row") == "stage"]
        if not rows or not all(x.get("start_utc_iso") and x.get("end_utc_iso") for x in rows): miss.append("%s: controller timeline stage rows without UTC start/end" % tid)
        if tid == RICH_TASK:
            if (r.get("l2") or {}).get("verdict") != "VALIDATED": miss.append("%s: L2 is %s, not VALIDATED (%s)" % (tid, (r.get("l2") or {}).get("verdict"), (r.get("l2") or {}).get("reason")))
            if ((r.get("clusters") or {}).get("extended_clusters") or 0) < 2: miss.append("%s: clusters %s < 2" % (tid, (r.get("clusters") or {}).get("extended_clusters")))
            if not (r.get("discrimination") or {}).get("triggered"): miss.append("%s: discrimination not triggered" % tid)
            c = r.get("critic") or {}
            if not (r.get("critic_called") and c.get("state") == "applied" and c.get("applied_in_controller")): miss.append("%s: inline critic not applied (%s)" % (tid, c))
            if not c.get("critic_evidence") or any(x["independence"] != "model_derived" for x in c["critic_evidence"]): miss.append("%s: critic evidence is not purely model_derived: %s" % (tid, c.get("critic_evidence")))
            stages = {x["name"] for x in rows}
            for s in ("l2_generate", "l2_validate", "cluster", "discriminate"):
                if s not in stages: miss.append("%s: no timeline stage %r" % (tid, s))
        if tid == REPAIR_TASK:
            if not any(e["kind"] == "candidate" and str(e["data"].get("origin", "")).startswith("repair:") for e in j): miss.append("%s: no repair candidate in the journal" % tid)
            if not any(e["kind"] == "generation_request" and e["data"].get("purpose") == "repair" for e in j): miss.append("%s: no repair generation request" % tid)
            if (r.get("rounds") or 0) < 1: miss.append("%s: rounds %s (no repair round)" % (tid, r.get("rounds")))
    for f in ("RUN_SUMMARY.json", "TIMELINE.jsonl", "TIMELINE_SUMMARY.json", "BANK_RESULTS.json", "CONFORMANCE.json"):
        if not (out / f).is_file(): miss.append("run: %s missing" % f)
    if rs.get("verdict") != "OK": miss.append("run verdict %s" % rs.get("verdict"))
    if rs.get("pipeline_verdict") != "PIPELINE_COMPLETE": miss.append("conformance verdict %s" % rs.get("pipeline_verdict"))
    if rs.get("internal_states") != "disabled": miss.append("states hook is not disabled")
    sc = rs.get("stage_coverage") or {}
    for k in ("l2_validated", "clusters_ge2", "critic_applied", "repair"):
        if not sc.get(k): miss.append("stage coverage: no task reached %s" % k)
    return miss


def selftest(out: Path, manifest: Path = DEFAULT_MANIFEST) -> int:
    cheap = [t for t in load_tasks(manifest) if t["language"] == "python"][0]      # P1-13, the first python task of the real manifest
    tasks = load_task_ids([RICH_TASK, REPAIR_TASK], manifest) + [cheap]
    assert [x["task_id"] for x in tasks] == [RICH_TASK, REPAIR_TASK, CHEAP_TASK], [x["task_id"] for x in tasks]
    fake = SelfTestGen(tasks)
    rs = run(out, tasks, lambda o: fake, exe_factory_for("mxc"), policy=policy_for(), sandbox="mxc", critic_client=GlmInlineCritic(transport=selftest_transport), resources_fn=lambda: {"free_ram_gb": None, "free_vram_gb": None},
             min_free_ram_gb=0, min_free_vram_gb=0, manifest=manifest, finalize=True)
    miss = selftest_checks(Path(out), rs, [t["task_id"] for t in tasks])
    print(json.dumps({"selftest": "FAIL" if miss else "PASS", "missing": miss, "verdict": rs.get("verdict"), "pipeline_verdict": rs.get("pipeline_verdict"), "done": rs.get("done"), "engine": rs.get("jail", {}).get("engine"), "stage_coverage": (rs.get("stage_coverage") or {}).get("line"),
                      "seconds": rs.get("total_seconds")}, indent=1, default=str))
    return 1 if miss else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="local bank run (RTX 3050 + MXC sandbox + inline critic)")
    ap.add_argument("out"); ap.add_argument("--tasks", default=str(DEFAULT_MANIFEST)); ap.add_argument("--limit", type=int); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wall-cap", type=float, default=WALL_CAP_S); ap.add_argument("--profile", default=PROFILE); ap.add_argument("--ceiling", type=int, default=CEILING)
    ap.add_argument("--sandbox", choices=("mxc", "wslc", "wsl"), default="mxc", help="mxc = the MXC process container (default; python only). wslc = the genuine MXC WSLC container backend (python + java after a real toolchain probe). wsl = the old WSL bwrap engine, only as an explicitly selected comparison. No fallback between them")
    ap.add_argument("--trusted-verify", action="store_true", help="use evidence_ladder/execution/trusted_verify.py (one fresh process per input, host-side verdicts); recorded as engine_version; default off = legacy runner")
    ap.add_argument("--selftest", action="store_true", help="2 python tasks, tiny fake model, real MXC + real controller + inline critic over a fake transport; exit != 0 when a stage is missing")
    ap.add_argument("--min-free-ram-gb", type=float, default=MIN_FREE_RAM_GB); ap.add_argument("--min-free-vram-gb", type=float, default=MIN_FREE_VRAM_GB); ap.add_argument("--min-free-commit-gb", type=float, default=MIN_FREE_COMMIT_GB)
    ap.add_argument("--critic-read-timeout", type=float, default=CRITIC_READ_TIMEOUT_S, help="seconds one critic call may wait for its first response bytes (reasoning stays max)")
    ap.add_argument("--critic-total-timeout", type=float, default=CRITIC_TOTAL_TIMEOUT_S, help="seconds one critic stream may last in total")
    ap.add_argument("--retry-critic", action="store_true", help="with --resume: after the remaining tasks, re-run ONLY the critic step (saved generations replayed) of finished tasks whose critic did not complete; old task dirs are archived, never deleted")
    a = ap.parse_args(argv)
    a.out, a.tasks = str(Path(a.out).resolve()), str(Path(a.tasks).resolve())
    if a.selftest:
        return selftest(Path(a.out), Path(a.tasks))
    if a.retry_critic and not a.resume:
        ap.error("--retry-critic is used together with --resume")
    tasks = load_tasks(Path(a.tasks), ROOT, a.limit)
    rs = run(Path(a.out), tasks, default_gen_factory(a.profile), exe_factory_for(a.sandbox, a.trusted_verify), resume=a.resume, sandbox=a.sandbox, wall_cap_s=a.wall_cap, policy=policy_for(a.profile, a.ceiling), gpu_sampler=True,
             min_free_ram_gb=a.min_free_ram_gb, min_free_vram_gb=a.min_free_vram_gb, min_free_commit_gb=a.min_free_commit_gb, manifest=Path(a.tasks),
             critic_client=GlmInlineCritic(read_timeout_s=a.critic_read_timeout, total_timeout_s=a.critic_total_timeout), retry_critic=a.retry_critic)
    print(json.dumps({k: rs.get(k) for k in ("verdict", "done", "errors", "pending_not_started", "not_tested", "pipeline_verdict", "total_seconds")}, default=str))
    return 0 if rs["verdict"] in ("OK", "PARTIAL", "STOPPED") else 2


if __name__ == "__main__":
    sys.exit(main())
