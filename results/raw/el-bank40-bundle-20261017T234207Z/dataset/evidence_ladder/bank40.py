"""[EL-BANK40] Shared, model-free logic of the 40-task Kaggle run (kaggle/el_bank40_run_v1.py, tools/pack_el_bank40_v1.py, tools/el_bank40_driver.py). Pure standard library, no GPU, no network;
everything here is unit-tested on the PC (evidence_ladder/orchestrator/tests/phaseF3b_tests.py).

  THINKING:   one explicit policy (check_policy): "budget" | "prompt_brief" | "model_variant"; a missing or unknown policy is refused, nothing is hard-coded, the chosen policy is written into every result.
  SHARDS:     shard_tasks() deterministic, disjoint, balanced by (language, difficulty) interleave; the same 40 tasks always land in the same shard.
  TIMELINE:   Timeline appends one fsynced JSON line per event (real UTC with milliseconds + monotonic seconds); summarize() gives per-stage p50/p95 and totals; TimedGen/TimedExec wrap the
              generation and execution backends so model time and sandbox time are measured per call.
  RESUME:     finished_task_ids() reads DONE markers (written atomically AFTER result + journal), TimeGuard stops starting tasks near the session limit.
  REPLAY:     RecordingGen records every generate() call in order; SequenceReplayGen serves them back in the same order and refuses any call whose purpose / seeds / prompt hash differ (pass 2 never regenerates).
  PUBLIC ONLY: assert_public_bundle() is the packer's and the kernel's guard: no hidden / protected / grader / oracle file or marker may be in the bundle."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SESSION_LIMIT_H = 12.0
DEFAULT_GUARD_H = 11.0
PASS_MODES = ("pass1", "pass2")
POLICIES = ("budget", "prompt_brief", "model_variant")
CEILING_MIN, CEILING_MAX = 2048, 16384


class RequestRefused(ValueError):
    """The run request has no explicit thinking policy, or the policy is malformed."""


def check_policy(req: Mapping[str, Any]) -> Dict[str, Any]:
    """The generation thinking policy is ONE explicit, recorded setting; nothing is assumed (the user's choice is pending). Missing / unknown / malformed -> RequestRefused.
      budget        {policy, think_budget N (64..8192), answer_tokens M (64..4096)}   think at most N tokens, then force </think> (HFQwenBackend think_budget)
      prompt_brief  {policy, brief_instruction (one line), ceiling (2048..16384)}      NO cutoff on thinking; a one-line brief-reasoning instruction is appended to the prompt; `ceiling` is only a technical
                    safety limit on total new tokens: hitting it is journalled (timeline event ceiling_hit) and counted as a FAILED generation, never as success
      model_variant {policy, profile_id, model_glob, ceiling (2048..16384)}            a different model (profile with enable_thinking False) that does not think"""
    if not isinstance(req, Mapping) or req.get("policy") not in POLICIES:
        raise RequestRefused("thinking_policy is required and must be one of %s (got %r)" % (list(POLICIES), (req or {}).get("policy") if isinstance(req, Mapping) else req))
    pol = req["policy"]
    def intr(k, lo, hi):
        v = req.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi: raise RequestRefused("%s: %s must be an integer in [%d, %d], got %r" % (pol, k, lo, hi, v))
        return v
    if pol == "budget":
        return {"policy": pol, "think_budget": intr("think_budget", 64, 8192), "answer_tokens": intr("answer_tokens", 64, 4096)}
    if pol == "prompt_brief":
        ins = req.get("brief_instruction")
        if not isinstance(ins, str) or not ins.strip() or chr(10) in ins.strip() or len(ins) > 300: raise RequestRefused("prompt_brief: brief_instruction must be one non-empty line (<= 300 chars)")
        return {"policy": pol, "brief_instruction": ins.strip(), "ceiling": intr("ceiling", CEILING_MIN, CEILING_MAX)}
    pid, g = req.get("profile_id"), req.get("model_glob")
    if not isinstance(pid, str) or not pid or not isinstance(g, str) or not g: raise RequestRefused("model_variant: profile_id and model_glob are required")
    return {"policy": pol, "profile_id": pid, "model_glob": g, "ceiling": intr("ceiling", CEILING_MIN, CEILING_MAX)}


def utc_now() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%03dZ" % int((t % 1) * 1000)


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def write_atomic(path, text: str) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ------------------------------------------------------------------ shards
def shard_tasks(tasks: Sequence[Mapping[str, Any]], n_shards: int) -> List[List[str]]:
    """Deterministic and disjoint: tasks sorted by (language, difficulty rank, id) are dealt round-robin, so both shards get the same mix of Python/Java and easy/medium."""
    if n_shards < 1: raise ValueError("n_shards must be >= 1")
    rank = {"easy": 0, "medium": 1}
    order = sorted(tasks, key=lambda t: (t["language"], rank.get(t.get("difficulty", ""), 2), t["task_id"]))
    out: List[List[str]] = [[] for _ in range(n_shards)]
    for i, t in enumerate(order):
        out[i % n_shards].append(t["task_id"])
    return out


# ------------------------------------------------------------------ time guard / resume
class TimeGuard:
    """should_start() is False once elapsed >= guard hours (default 11 h of the 12 h session): no new task starts, the rest is recorded as pending."""

    def __init__(self, t0: float, guard_h: float = DEFAULT_GUARD_H, clock=time.time):
        if not 0 < guard_h <= SESSION_LIMIT_H: raise ValueError("guard_h must be in (0, %s]" % SESSION_LIMIT_H)
        self.t0, self.guard_h, self.clock = t0, guard_h, clock

    def elapsed_h(self) -> float: return (self.clock() - self.t0) / 3600.0

    def should_start(self) -> bool: return self.elapsed_h() < self.guard_h


def finished_task_ids(*roots: Path) -> Dict[str, Path]:
    """-> {task_id: dir} for every <root>/**/tasks/<id>/DONE (the marker is written last, atomically)."""
    out: Dict[str, Path] = {}
    for r in roots:
        r = Path(r)
        if r.is_dir():
            for d in sorted(r.glob("**/tasks/*/DONE")):
                out.setdefault(d.parent.name, d.parent)
    return out


def plan_work(shard: Sequence[str], finished: Iterable[str]) -> Tuple[List[str], List[str]]:
    """-> (todo in shard order, already finished). A re-run never repeats a finished task."""
    fin = set(finished)
    return [t for t in shard if t not in fin], [t for t in shard if t in fin]


# ------------------------------------------------------------------ timeline
class Timeline:
    def __init__(self, path, worker: str = "main", clock=time.time, mono=time.monotonic):
        self.path, self.worker, self.clock, self.mono = Path(path), worker, clock, mono
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, /, **kw: Any) -> Dict[str, Any]:
        t = self.clock()
        rec = {"ts_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%03dZ" % int((t % 1) * 1000), "ts": round(t, 3), "mono": round(self.mono(), 3), "worker": self.worker, "event": name, **kw}
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n"); f.flush(); os.fsync(f.fileno())
        return rec

    def span(self, name: str, /, **kw: Any) -> "_Span":
        return _Span(self, name, kw)


class _Span:
    def __init__(self, tl: Timeline, name: str, kw: Dict[str, Any]): self.tl, self.name, self.kw = tl, name, kw

    def __enter__(self):
        self.t0 = self.tl.mono(); self.start = self.tl.event(self.name + "_start", **self.kw); return self

    def __exit__(self, et, ev, tb):
        self.end = self.tl.event(self.name + "_end", seconds=round(self.tl.mono() - self.t0, 3), ok=et is None, **({"error": type(ev).__name__} if et else {}), **self.kw)
        return False


def pct(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs: return None
    s = sorted(xs); k = (len(s) - 1) * p / 100.0; lo = int(k); hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def read_events(*paths: Path) -> List[Dict[str, Any]]:
    ev = []
    for p in paths:
        if Path(p).is_file():
            ev += [json.loads(l) for l in Path(p).read_text("utf-8").splitlines() if l.strip()]
    return sorted(ev, key=lambda e: (e["ts"], e["mono"]))


def summarize(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per stage (event name without _end): n, total, p50, p95 seconds; model-vs-sandbox split; first/last UTC; wall seconds."""
    by: Dict[str, List[float]] = {}
    for e in events:
        if e["event"].endswith("_end") and "seconds" in e:
            by.setdefault(e["event"][:-4], []).append(float(e["seconds"]))
    stages = {k: {"n": len(v), "total_s": round(sum(v), 3), "p50_s": pct(v, 50), "p95_s": pct(v, 95)} for k, v in sorted(by.items())}
    model = sum(s["total_s"] for k, s in stages.items() if k.startswith("generate"))
    sandbox = sum(s["total_s"] for k, s in stages.items() if k.startswith("sandbox"))
    ts = [e["ts"] for e in events]
    first = min(events, key=lambda e: e["ts"]) if events else None
    last = max(events, key=lambda e: e["ts"]) if events else None
    return {"events": len(events), "first_utc": first and first["ts_utc"], "last_utc": last and last["ts_utc"], "wall_seconds": round(max(ts) - min(ts), 3) if ts else 0.0,
            "stages": stages, "model_seconds": round(model, 3), "sandbox_seconds": round(sandbox, 3),
            "model_vs_sandbox": {"model_share": round(model / (model + sandbox), 4) if model + sandbox else None}}


class TimedGen:
    """Wraps a generation backend: one generate span per call with purpose, seeds, tokens (real UTC start/end)."""

    def __init__(self, inner, tl: Timeline, task_id: str, gpu: Optional[int] = None):
        self.inner, self.tl, self.task_id, self.gpu = inner, tl, task_id, gpu

    def __getattr__(self, name): return getattr(self.inner, name)

    def generate(self, request):
        with self.tl.span("generate", task=self.task_id, purpose=request.purpose, n=request.n, gpu=self.gpu) as sp:
            out = self.inner.generate(request)
            self.tl.event("generate_tokens", task=self.task_id, tokens_prompt=sum(g.tokens_prompt for g in out), tokens_generated=sum(g.tokens_generated for g in out))
        return out


class TimedExec:
    def __init__(self, inner, tl: Timeline, task_id: str):
        self.inner, self.tl, self.task_id = inner, tl, task_id

    def __getattr__(self, name): return getattr(self.inner, name)

    def run_public(self, artifact, task):
        with self.tl.span("sandbox_public", task=self.task_id, artifact=getattr(artifact, "artifact_id", None)):
            return self.inner.run_public(artifact, task)

    def run_experiment(self, artifact, experiment):
        with self.tl.span("sandbox_experiment", task=self.task_id, artifact=getattr(artifact, "artifact_id", None)):
            return self.inner.run_experiment(artifact, experiment)


class BriefPromptGen:
    """policy prompt_brief: appends the one-line brief-reasoning instruction to every prompt, caps total new tokens at the technical ceiling and reports a generation that hit the ceiling
    (ceiling_hit event; counted by the task result as a failed generation). The wrapped backend is unchanged (thinking stays unrestricted inside the ceiling)."""

    def __init__(self, inner, instruction: str, ceiling: int, tl: Optional["Timeline"] = None, task_id: str = ""):
        self.inner, self.instruction, self.ceiling, self.tl, self.task_id, self.ceiling_hits = inner, instruction, int(ceiling), tl, task_id, 0

    def __getattr__(self, name): return getattr(self.inner, name)

    def generate(self, request):
        import dataclasses
        req = dataclasses.replace(request, prompt=request.prompt.rstrip() + chr(10) * 2 + self.instruction, max_new_tokens=min(int(request.max_new_tokens), self.ceiling))
        out = self.inner.generate(req)
        for g in out:
            if g.tokens_generated >= req.max_new_tokens:
                self.ceiling_hits += 1
                if self.tl: self.tl.event("ceiling_hit", task=self.task_id, tokens=g.tokens_generated, ceiling=req.max_new_tokens)
        return out


# ------------------------------------------------------------------ generation record / replay (pass 2 regenerates nothing)
class ReplayMismatch(RuntimeError):
    pass


class RecordingGen:
    def __init__(self, inner, task_id: str):
        self.inner, self.task_id, self.calls = inner, task_id, []

    def __getattr__(self, name): return getattr(self.inner, name)

    def generate(self, request):
        from evidence_ladder.models import qwen as Q
        out = self.inner.generate(request)
        self.calls.append({"call": len(self.calls), "task_id": request.task_id, "purpose": request.purpose, "n": request.n, "seeds": list(request.seeds), "max_new_tokens": request.max_new_tokens,
                           "prompt_template_id": request.prompt_template_id, "prompt_sha256": sha(request.prompt), "generations": [Q.generation_to_dict(g) for g in out]})
        return out


class SequenceReplayGen:
    """Serves the recorded calls in order; any difference in purpose, n, seeds or prompt hash raises ReplayMismatch (never answers a different question, never calls a model)."""

    def __init__(self, calls: Sequence[Mapping[str, Any]], info: Optional[Mapping[str, Any]] = None):
        self.calls, self.i, self._info = list(calls), 0, dict(info or {})

    def generate(self, request):
        from evidence_ladder.models import qwen as Q
        if self.i >= len(self.calls): raise ReplayMismatch("call %d: nothing recorded (the replay asked for a generation pass 1 never made)" % self.i)
        c = self.calls[self.i]
        want = (request.purpose, int(request.n), [int(s) for s in request.seeds][:int(request.n)], sha(request.prompt))
        have = (c["purpose"], int(c["n"]), [int(s) for s in c["seeds"]][:int(c["n"])], c["prompt_sha256"])
        if want != have: raise ReplayMismatch("call %d differs from the recording (purpose/n/seeds/prompt hash)" % self.i)
        self.i += 1
        return [Q.generation_from_dict(g) for g in c["generations"]]

    def model_info(self) -> Dict[str, Any]:
        return {"backend": "sequence_replay_of_pass1", "calls": len(self.calls), **self._info}

    def exhausted(self) -> bool: return self.i == len(self.calls)


# ------------------------------------------------------------------ public-only guard
FORBIDDEN_PATH_PARTS = ("protected" + "_cases", "hidden" + "_cases", "oracl" + "es",      # built from fragments: a runtime module must not even name the held-case files (isolation.static_oracle_scan)
                         "execution/protected.py", "/tests/", "redteam", ".secrets", "kaggle.json", "api_key")
KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9._\-]{16,}|\bsk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|KGAT_[A-Za-z0-9]{6,}")


def assert_public_bundle(files: Mapping[str, bytes], markers: Sequence[bytes] = ()) -> Dict[str, Any]:
    """files: {relative path: bytes}. Raises AssertionError listing every violation: a forbidden path part, a task record with keys other than the public ones, a held-case marker inside any file,
    a key-shaped string. Returns a small receipt."""
    bad: List[str] = []
    for rel, data in files.items():
        low = "/" + rel.replace("\\", "/").lower()
        for part in FORBIDDEN_PATH_PARTS:
            if part in low: bad.append("%s: forbidden path part %r" % (rel, part))
        for m in markers:
            if m and m in data: bad.append("%s: contains a held-case marker" % rel)
        if rel.endswith((".py", ".json", ".jsonl", ".toml", ".txt", ".md", ".java")) and KEY_RE.search(data.decode("utf-8", "replace")): bad.append("%s: key-shaped text" % rel)
    tp = files.get("bank40_tasks_public.jsonl")
    if tp is not None:
        for ln in tp.decode("utf-8").splitlines():
            d = json.loads(ln)
            if set(d) != {"task_id", "language", "difficulty", "prompt", "public_tests"}: bad.append("task record %s has keys %s" % (d.get("task_id"), sorted(d)))
            if not all(set(t) == {"input", "expected"} for t in d["public_tests"]): bad.append("task %s: public test keys" % d.get("task_id"))
    if bad: raise AssertionError("bundle is not public-only: " + "; ".join(bad[:8]))
    return {"files": len(files), "markers_checked": len(markers), "violations": 0}


# ------------------------------------------------------------------ per-task verdict (same definitions as tools/el_arch1_l2_local.py verdicts(), re-implemented here for the bundle)
def verdicts(res, journal) -> Dict[str, Any]:
    l2e = journal.entries("l2_checks"); l2d = l2e[0]["data"] if l2e else None
    if l2d is None: l2v = {"verdict": "INSUFFICIENT", "reason": "L2 stage not reached (no candidate passed the public examples)"}
    elif l2d["validated"]: l2v = {"verdict": "VALIDATED", "validated": len(l2d["validated"]), "parsed": l2d["parsed"], "dropped": l2d["dropped"]}
    else: l2v = {"verdict": "INSUFFICIENT", "parsed": l2d["parsed"], "dropped": l2d["dropped"], "reason": "no valid test parsed" if not l2d["parsed"] else "every parsed test dropped"}
    bc = journal.entries("behaviour_clusters")
    cl = {"computed": bool(bc), "public_clusters": bc[-1]["data"]["public_clusters"] if bc else None, "extended_clusters": bc[-1]["data"]["extended_clusters"] if bc else None}
    tr, cr = journal.entries("critic_trigger"), journal.entries("critic")
    used = bool(tr and tr[-1]["data"]["triggered"])
    disc = {"triggered": used, "critic": (cr[-1]["data"]["state"] if cr else ("pending" if res.stop_details == "critic_pending" else None)) if used else None}
    return {"l2": l2v, "clusters": cl, "discrimination": disc}


def public_example_stats(journal) -> Dict[str, Any]:
    per: Dict[str, Dict[str, int]] = {}
    for e in journal.evidence():
        if e.get("check_id") == "public_examples_v1":
            d = per.setdefault(e["candidate_id"], {"pass": 0, "total": 0}); d["total"] += 1; d["pass"] += 1 if e["status"] == "pass" else 0
    full = [c for c, d in per.items() if d["total"] and d["pass"] == d["total"]]
    return {"candidates_with_examples": len(per), "candidates_passing_all": len(full), "best": max((d["pass"] for d in per.values()), default=0), "examples": max((d["total"] for d in per.values()), default=0)}
