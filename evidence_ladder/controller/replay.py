"""[EL-ARCH-1 B8] Record a per-action run and REPLAY it: re-execute the recorded inputs against the code in the working tree and compare journal hashes entry by entry.
The journal itself (journal.py: append-only, hash chained, no wall time or other nondeterminism in any entry) is the immutable record; this module adds what is needed to run it again:

  RecordedRun(task, ctx, render, files)   delegates to AgenticRun and logs every driver event (begin / investigate / step(options, qwen_meta) / critic_stage / finish) plus every
                                          scorer answer and every sandbox call (command, persist flag, result, and the workspace files after a persisting call)
  replay(record, sandbox=None, critic=None)  rebuilds the workspace from the intake files, re-drives the same events through a fresh AgenticRun and compares:
        sandbox=None   OFFLINE: the recorded sandbox results are served in call order (a command that differs from the recorded one raises ReplayDivergence). Checks the controller logic.
        sandbox=<obj>  LIVE: a real sandbox (e.g. WslSandbox) re-executes everything; identical hashes then also prove the execution is deterministic.
        scorer         recorded scores / errors / provenance are served (the scorer never re-runs: LAYA lives on Kaggle).
  -> {"identical", "recorded_journal_hash", "replayed_journal_hash", "first_divergence", "code_unchanged", "changed_code_files", "git_commit"}
The record also stores sha256 of the controller / models source files and the git commit, so a replay says whether it ran against the very code the record was made with."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .agentic import AgenticContext, AgenticRun, AgenticTask, LimitReached, NoOracle
from .limits import Limits
from .journal import canon

ROOT = Path(__file__).resolve().parents[2]
CODE_DIRS = ("evidence_ladder/controller", "evidence_ladder/models", "evidence_ladder/critic_batch", "evidence_ladder/common")
CODE_FILES = ("configs/el_arch1_limits_v1.toml",)


class ReplayError(Exception):
    pass


class ReplayDivergence(ReplayError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_hashes() -> Dict[str, str]:
    out = {}
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).glob("*.py")):
            out[p.relative_to(ROOT).as_posix()] = _sha(p.read_bytes())
    for f in CODE_FILES:
        if (ROOT / f).exists():
            out[f] = _sha((ROOT / f).read_bytes())
    return out


def git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _tree_files(ws) -> Dict[str, str]:
    ws = Path(ws)
    out = {}
    for p in sorted(ws.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            with open(p, "r", encoding="utf-8", newline="") as f:      # newline="" keeps CRLF / LF exactly as on disk: a replay must hash the same bytes (Windows text mode would rewrite them)
                out[p.relative_to(ws).as_posix()] = f.read()
    return out


def _write_tree(ws, files: Dict[str, str]) -> None:
    ws = Path(ws)
    for p in [x for x in ws.rglob("*") if x.is_file()]:
        p.unlink()
    for rel, txt in files.items():
        q = ws / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        with open(q, "w", encoding="utf-8", newline="") as f:
            f.write(txt)


# ------------------------------------------------------------------ recorders
class RecordingSandbox:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    @property
    def env_label(self):      # [MXC] the environment identity of the wrapped sandbox (default: the WSL label, so existing journal hashes are unchanged)
        return getattr(self.inner, "env_label", "wsl-bwrap-systemd")

    def run(self, workspace, cmd, persist=False, timeout=60):
        try:
            r = self.inner.run(workspace, cmd, persist=persist, timeout=timeout)
        except Exception as ex:      # noqa: BLE001  [D1/S2] a sandbox that raised is part of the record (the controller turns it into ERROR evidence)
            self.calls.append({"cmd": cmd, "persist": bool(persist), "raised": "%s: %s" % (type(ex).__name__, str(ex)[:200])})
            raise
        rec = {"cmd": cmd, "persist": bool(persist), "result": copy.deepcopy(r)}
        if persist:
            rec["files_after"] = _tree_files(workspace)
        self.calls.append(rec)
        return r


class RecordedSandbox:
    def __init__(self, calls: List[Dict[str, Any]], label: Optional[str] = None):
        self.calls, self.i = list(calls), 0
        self.env_label = label or "wsl-bwrap-systemd"      # [MXC] restored from record["sandbox_label"] when the recording backend was not WSL

    def run(self, workspace, cmd, persist=False, timeout=60):
        if self.i >= len(self.calls):
            raise ReplayDivergence("the replay made sandbox call #%d but only %d were recorded" % (self.i + 1, len(self.calls)))
        rec = self.calls[self.i]
        self.i += 1
        if rec["cmd"] != cmd or rec["persist"] != bool(persist):
            raise ReplayDivergence("sandbox call #%d: recorded %r (persist=%s), replay asked %r (persist=%s)" % (self.i, rec["cmd"], rec["persist"], cmd, persist))
        if "raised" in rec:
            raise RuntimeError(rec["raised"].split(": ", 1)[-1])
        if persist and "files_after" in rec:
            _write_tree(workspace, rec["files_after"])
        return copy.deepcopy(rec["result"])


class RecordingScorer:
    def __init__(self, inner):
        self.inner, self.calls = inner, []
        self.scorer_id = getattr(inner, "scorer_id", None)
        self.scorer_def = getattr(inner, "scorer_def", None)

    @property
    def _backend(self):
        return getattr(self.inner, "_backend", None)

    def score(self, decision_type, pool, exploratory=False):
        try:
            s = [float(x) for x in self.inner.score(decision_type, pool)]
        except Exception as ex:      # noqa: BLE001  the failure is part of the record (the controller journals it as scorer_unavailable)
            self.calls.append({"error": "%s: %s" % (type(ex).__name__, str(ex)[:250]), "error_type": type(ex).__name__})
            raise
        last = getattr(getattr(self.inner, "_backend", None), "last", None)
        self.calls.append({"scores": s, "last": json.loads(json.dumps(last, default=str)) if last is not None else None})
        return s


class RecordedScorer:
    def __init__(self, scorer_id, scorer_def, calls):
        self.scorer_id, self.scorer_def, self.calls, self.i = scorer_id, scorer_def, list(calls), 0
        self._backend = SimpleNamespace(last=None)

    def score(self, decision_type, pool, exploratory=False):
        if self.i >= len(self.calls):
            raise ReplayDivergence("the replay scored #%d times but only %d scorer calls were recorded" % (self.i + 1, len(self.calls)))
        rec = self.calls[self.i]
        self.i += 1
        if "error" in rec:
            msg = rec["error"].split(": ", 1)[-1] if rec["error"].startswith(rec.get("error_type", "") + ": ") else rec["error"]
            raise (type(rec["error_type"], (RuntimeError,), {}) if rec.get("error_type") else RuntimeError)(msg)      # same exception TYPE NAME as recorded: the journal's scorer_unavailable reason carries it
        self._backend.last = rec["last"]
        return list(rec["scores"])


# ------------------------------------------------------------------ the recorded run
class RecordedRun:
    """Drive an AgenticRun through this wrapper to make it replayable. The wrapped run is `.run`; call .record() after finish()."""

    def __init__(self, task: AgenticTask, ctx: AgenticContext, render, files: Optional[Dict[str, str]] = None):
        self.files = files if files is not None else _tree_files(task.workspace)       # the intake workspace (before begin / any action)
        self.sb = RecordingSandbox(ctx.sandbox)
        self.sc = RecordingScorer(ctx.scorer) if ctx.scorer is not None else None
        self.cmp = {k: RecordingScorer(v) for k, v in sorted((ctx.comparison or {}).items())}      # [D1] probe / random-head comparison scorers are recorded like the main scorer
        self.cfg = copy.deepcopy(dict(ctx.config))
        ctx2 = AgenticContext(sandbox=self.sb, scorer=self.sc, critic=ctx.critic, budget=ctx.budget, arm=ctx.arm, seeds=ctx.seeds, config=ctx.config, limits=ctx.limits, clock=ctx.clock, comparison=self.cmp or None, timeline=getattr(ctx, "timeline", None))
        self.task, self.ctx = task, ctx2
        self.run = AgenticRun(task, ctx2, render)
        self.events: List[Dict[str, Any]] = []
        self.result = None

    def begin(self):
        self.events.append({"e": "begin"})
        return self.run.begin()

    def investigate(self):
        from . import investigate as INV
        self.events.append({"e": "investigate"})
        return INV.investigate(self.run)

    def step(self, options, qwen_meta=None):
        self.events.append({"e": "step", "options": list(options), "meta": copy.deepcopy(qwen_meta or {})})
        return self.run.step(options, qwen_meta)

    def note(self, kind, data):
        """[D1] a driver-level journal entry (e.g. the stage marker): recorded as an event so a replay writes it at the same place."""
        self.events.append({"e": "note", "kind": kind, "data": copy.deepcopy(data)})
        return self.run.note(kind, data)

    def propose(self, out):
        """[D1] journal one proposal round (output of controller.generation.ActionProposer.propose); the replay re-journals the recorded output."""
        self.events.append({"e": "proposals", "out": copy.deepcopy(out)})
        return self.run.journal_proposals(out)

    def critic_stage(self):
        self.events.append({"e": "critic_stage"})
        return self.run.critic_stage()

    def finish(self):
        self.events.append({"e": "finish"})
        self.result = self.run.finish()
        return self.result

    def record(self) -> Dict[str, Any]:
        """[D1] `limits` is recorded because a non-default limits profile is part of the run; the injected clock is NOT (a wall-time stop is not replayable)."""
        t = self.task
        label = self.sb.env_label
        extra = {} if label == "wsl-bwrap-systemd" else {"sandbox_label": label}      # [MXC] only a non-WSL backend adds the key, so WSL records are byte-identical to before
        return {**extra, "schema": "el_arch1_run_record_v1", "task": {"task_id": t.task_id, "user_text": t.user_text, "system_text": t.system_text, "test_cmd": t.test_cmd, "protected": list(t.protected),
                                                              "max_steps": t.max_steps, "scope": t.scope, "files": self.files},
                "arm": self.ctx.arm, "seeds": [int(s) for s in self.ctx.seeds], "config": self.cfg,
                "limits": self.ctx.limits.as_dict() if self.ctx.limits is not None else None, "events": self.events, "sandbox": self.sb.calls,
                "scorer": None if self.sc is None else {"scorer_id": self.sc.scorer_id, "scorer_def": self.sc.scorer_def, "calls": self.sc.calls},
                "comparison": {k: {"scorer_id": v.scorer_id, "scorer_def": v.scorer_def, "calls": v.calls} for k, v in self.cmp.items()},
                "journal_hash": self.run.journal.journal_hash(), "chain": self.run.journal.chain_hashes(), "kinds": [e["kind"] for e in self.run.journal.entries()],
                "code": code_hashes(), "git_commit": git_commit(), "result": None if self.result is None else {"answer_label": self.result.answer_label, "stop_cause": self.result.stop_cause}}


def replay(record: Dict[str, Any], *, sandbox=None, critic=None, render=None, workdir=None) -> Dict[str, Any]:
    """Re-execute `record` (see module doc) and compare. `render` must be the same context renderer as at record time (default: the plain one the unit tests use)."""
    if record.get("schema") != "el_arch1_run_record_v1":
        raise ReplayError("not a run record")
    if any(e["e"] == "critic_stage" for e in record["events"]) and critic is None:
        raise ReplayError("the record contains a critic stage: pass the critic broker (read-only copy of its ledger)")
    t = record["task"]
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="el-replay-"))
    ws = base / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    _write_tree(ws, t["files"])
    task = AgenticTask(t["task_id"], t["user_text"], ws, t["test_cmd"], list(t["protected"]), t["system_text"], t["max_steps"], t.get("scope"))
    sc = record.get("scorer")
    scorer = RecordedScorer(sc["scorer_id"], sc["scorer_def"], sc["calls"]) if sc else None
    sb = sandbox if sandbox is not None else RecordedSandbox(record["sandbox"], record.get("sandbox_label"))
    cmp_ = {k: RecordedScorer(v["scorer_id"], v["scorer_def"], v["calls"]) for k, v in (record.get("comparison") or {}).items()}
    ctx = AgenticContext(sandbox=sb, scorer=scorer, critic=critic, arm=record["arm"], seeds=tuple(record["seeds"]), config=copy.deepcopy(record["config"]), comparison=cmp_ or None,
                         limits=Limits(**record["limits"]) if record.get("limits") else None)
    run = AgenticRun(task, ctx, render or (lambda m: "ctx"))
    err = None
    try:
        for ev in record["events"]:
            if ev["e"] == "begin":
                run.begin()
            elif ev["e"] == "investigate":
                from . import investigate as INV
                INV.investigate(run)
            elif ev["e"] == "note":
                run.note(ev["kind"], ev["data"])
            elif ev["e"] == "proposals":
                run.journal_proposals(ev["out"])
            elif ev["e"] == "step":
                try:
                    run.step(ev["options"], ev["meta"])
                except (NoOracle, LimitReached):      # the recorded run was refused at this step too; the later events (finish) still replay
                    pass
            elif ev["e"] == "critic_stage":
                run.critic_stage()
            elif ev["e"] == "finish":
                run.finish()
    except NoOracle:
        pass
    except ReplayDivergence as ex:
        err = str(ex)
    chain = run.journal.chain_hashes()
    kinds = [e["kind"] for e in run.journal.entries()]
    first = None
    for i, (a, b) in enumerate(zip(record["chain"], chain)):
        if a != b:
            first = {"seq": i, "recorded_kind": record["kinds"][i], "replayed_kind": kinds[i]}
            break
    if first is None and len(record["chain"]) != len(chain):
        i = min(len(record["chain"]), len(chain))
        first = {"seq": i, "recorded_kind": record["kinds"][i] if i < len(record["kinds"]) else None, "replayed_kind": kinds[i] if i < len(kinds) else None}
    now = code_hashes()
    changed = sorted(k for k in set(now) | set(record["code"]) if now.get(k) != record["code"].get(k))
    out = {"identical": first is None and err is None and run.journal.journal_hash() == record["journal_hash"], "recorded_journal_hash": record["journal_hash"], "replayed_journal_hash": run.journal.journal_hash(),
           "first_divergence": first, "error": err, "mode": "live_sandbox" if sandbox is not None else "offline_recorded_sandbox", "code_unchanged": not changed, "changed_code_files": changed,
           "recorded_git_commit": record.get("git_commit"), "git_commit": git_commit(), "entries": [len(record["chain"]), len(chain)]}
    if workdir is None:
        shutil.rmtree(base, ignore_errors=True)
    return out
