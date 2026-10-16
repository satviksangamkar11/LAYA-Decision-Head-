"""Paired pilot runner. Order is fixed: authorization -> selection (per task and arm, two-phase critic flow) -> freeze + close gate -> protected grading -> report.
Protected grading happens only in grade_all(), through a GatedGrader, after the selection gate is closed; its results never reach a ControllerContext."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evidence_ladder.common.interfaces import Artifact, BudgetExhausted, GraderIsolationError, ProblemResult, sha256_text
from evidence_ladder.orchestrator.arms import ARMS, needs_critic
from evidence_ladder.orchestrator.authorization import require_authorization
from evidence_ladder.orchestrator.base import DEV_LABEL, THRESHOLDS, ConfigError, sha, write_json, write_jsonl
from evidence_ladder.orchestrator.budget import MatchedTokenBudget, resolve_T
from evidence_ladder.orchestrator.pairing import PairedRecords, fresh_context, starting_state_hash
from evidence_ladder.orchestrator.report import make_report, render_markdown


class SelectionGate:
    """Selection is open until close_selection() is called, once, with the hash of the frozen artifacts."""
    def __init__(self):
        self._closed, self.manifest_hash = False, None

    def close_selection(self, manifest_hash: str = None) -> None:
        if self._closed:
            raise GraderIsolationError("selection gate already closed")
        self._closed, self.manifest_hash = True, manifest_hash

    def is_closed(self) -> bool:
        return self._closed


class GatedGrader:
    """Refuses to grade before the gate is closed, and refuses anything that is not a frozen artifact."""
    def __init__(self, grader, gate, frozen_ids):
        self._g, self._gate, self._ids = grader, gate, frozen_ids

    def grade(self, artifact: Artifact):
        is_closed = getattr(self._gate, "is_closed", None)
        if is_closed is None or not is_closed():
            raise GraderIsolationError("protected grading requested while selection is still open")
        if (artifact.task_id, artifact.artifact_id) not in self._ids or sha256_text(artifact.source) != artifact.artifact_id:
            raise GraderIsolationError("artifact is not one of the frozen final artifacts")
        return self._g.grade(artifact)


# ---- thin proxies: count tool runs / API calls / GPU seconds and remember every executed artifact (so the frozen final one can be graded later)
class _Tally:
    def __init__(self):
        self.glm_calls = self.api_errors = 0


class _Gen:
    def __init__(self, inner, budget):
        self._i, self._b = inner, budget

    def generate(self, request):
        out = self._i.generate(request)
        self._b.note_time(gpu_seconds=sum(g.wall_seconds for g in out))
        return out

    def model_info(self):
        return self._i.model_info()


class _Exec:
    def __init__(self, inner, budget, store):
        self._i, self._b, self._s = inner, budget, store

    def run_public(self, artifact, task):
        self._s[(artifact.task_id, artifact.artifact_id)] = artifact
        self._b.note_call("tool_runs")
        return self._i.run_public(artifact, task)

    def run_experiment(self, artifact, experiment):
        self._s[(artifact.task_id, artifact.artifact_id)] = artifact
        self._b.note_call("tool_runs")
        return self._i.run_experiment(artifact, experiment)

    def backend_info(self):
        return self._i.backend_info()

def _has_pending(broker) -> bool:
    """Read-only pending check: the real broker's export_pending() marks packets exported, so it is only called when the broker has no summary."""
    ps = getattr(broker, "pending_summary", None)
    if ps is not None:
        s = ps()
        return bool(s.get("ready") or s.get("waiting_until") or s.get("exported_unresolved"))
    return bool(broker.export_pending())


class _Critic:
    def __init__(self, inner, tally):
        self._i, self._t = inner, tally

    def build_packet(self, req):
        return self._i.build_packet(req)

    def export_pending(self):
        return self._i.export_pending()

    def completed(self, packet_id):
        f = getattr(self._i, "completed", None)
        return f(packet_id) if f is not None else None

    def terminal_state(self, packet_id):
        f = getattr(self._i, "terminal_state", None)
        return f(packet_id) if f is not None else None

    def pending_summary(self):
        f = getattr(self._i, "pending_summary", None)
        return f() if f is not None else None

    def ingest(self, packet_id, http_status, raw_response):
        r = self._i.ingest(packet_id, http_status, raw_response)
        if r["state"] in ("ok", "malformed", "retry"):
            self._t.glm_calls += 1
        if r["state"] == "unavailable" or http_status not in (None, 200):
            self._t.api_errors += 1
        return r


@dataclass
class _Pending:
    task: Any
    arm: str
    broker: Any
    tally: _Tally
    row1: dict


class PilotRunner:
    def __init__(self, *, tasks: Dict[str, list], controller, generation, execution, grader, gate, out_dir, T: Optional[int], k1: Optional[int],
                 master_seed: int, n_total: Optional[int] = None, arms=ARMS, critic_factory: Callable = None, scorer=None, transport: Callable = None,
                 authorization_manifest=None, thresholds_path=THRESHOLDS, dry_run: bool = False, clock: Callable = time.monotonic, synthetic: bool = False):
        if "fable" in tasks:
            raise NotImplementedError("Fable run support is NOT IMPLEMENTED for stage 1")
        self.tasks, self.controller, self.generation, self.execution = tasks, controller, generation, execution
        self._grader, self.gate, self.out = grader, gate, Path(out_dir)
        self.T, self.k1, self.n_total, self.seed, self.arms = T, k1, n_total, master_seed, tuple(arms)
        self.critic_factory, self.scorer, self.transport = critic_factory, scorer, transport
        self.auth, self.thr, self.dry_run, self.clock, self.synthetic = authorization_manifest, thresholds_path, dry_run, clock, synthetic
        self.records = PairedRecords()
        self._artifacts: Dict[tuple, Artifact] = {}
        self._rows_all: List[dict] = []
        self._grades: List[dict] = []
        self._selected = self._graded = False
        self._frozen: Dict[tuple, dict] = {}

    # ------------------------------------------------------------ preflight
    def _preflight(self):
        resolve_T(self.T)                       # ConfigError while T is pending/unset
        if not isinstance(self.k1, int) or self.k1 < 1:
            raise ConfigError("k1 is PENDING (tuned operational parameter); the pilot refuses to start")
        if self.dry_run:
            if self.generation.model_info().get("fake") is not True:
                raise ConfigError("dry_run is only allowed with scripted fake backends (model_info()['fake'] is True)")
        else:
            require_authorization(self.auth, self.thr)   # NotAuthorized unless all four approvals + valid hashes

    # ------------------------------------------------------------ selection
    def _one(self, task, arm, passno, broker, tally) -> tuple:
        if self._graded:
            raise GraderIsolationError("selection after protected grading is forbidden")
        budget = MatchedTokenBudget(task.task_id, arm, self.T)
        store = self._artifacts
        ctx = fresh_context(task, arm, generation=_Gen(self.generation, budget), execution=_Exec(self.execution, budget, store), budget=budget,
                            critic=_Critic(broker, tally) if broker is not None else None, scorer=self.scorer if arm == "E" else None,
                            master_seed=self.seed, k1=self.k1, n_total=self.n_total)
        t0 = self.clock()
        try:
            res = self.controller.run_problem(task, ctx)
        except BudgetExhausted:
            res = ProblemResult(task.task_id, arm, None, "UNVERIFIED", {}, "budget", [], budget.record(), 0, "")
        budget.note_time(wall_seconds=self.clock() - t0)
        budget.note_call("glm_calls", tally.glm_calls)
        budget.note_call("api_errors", tally.api_errors)
        rec = budget.record()
        pending = bool(broker is not None and res.stop_cause == "api_unavailable" and _has_pending(broker))
        row = {"label": DEV_LABEL, "task_id": task.task_id, "track": task.track, "arm": arm, "pass": passno, "budget_record": rec,
               "budget_sidecar": budget.sidecar(), "wall_seconds": rec["wall_seconds"], "gpu_seconds": rec["gpu_seconds"], "tool_calls": rec["tool_runs"],
               "api_calls": rec["glm_calls"], "api_errors": rec["api_errors"], "stop_cause": res.stop_cause, "journal_hash": res.journal_hash,
               "answer_label": res.answer_label, "evidence_ids": list(res.evidence_ids), "final_artifact_id": res.final_artifact_id,
               "rounds": res.rounds, "critic_pending": pending, "mutation_state": task.mutation_status.get("state", "unknown"),
               "starting_state_hash": starting_state_hash(task, self.seed, self.k1)}
        self._rows_all.append(row)
        return row, pending

    def run_selection(self) -> dict:
        if self._selected:
            raise RuntimeError("selection already ran")
        self._preflight()
        pend: Dict[str, _Pending] = {}
        final: Dict[tuple, dict] = {}
        for track in sorted(self.tasks):
            for task in self.tasks[track]:
                for arm in self.arms:
                    broker = self.critic_factory(task, arm) if needs_critic(arm) and self.critic_factory else None
                    tally = _Tally()
                    row, is_pending = self._one(task, arm, 1, broker, tally)
                    final[(task.task_id, arm)] = row
                    if is_pending:
                        pend[f"{task.task_id}|{arm}"] = _Pending(task, arm, broker, tally, row)
        packets = []
        for key in sorted(pend):
            for p in pend[key].broker.export_pending():
                packets.append({"task_id": pend[key].task.task_id, "arm": pend[key].arm, "packet": p})
        if packets:
            write_jsonl(self.out / "critic_packets" / "pass1_pending.jsonl", packets)
        if packets and self.transport is not None:
            by_pid: Dict[str, list] = {}
            for x in packets:      # the same content hash can occur for several arms (each arm has its own broker): route every response to ALL brokers that asked for it
                by_pid.setdefault(x["packet"]["packet_id"], []).append(pend[f"{x['task_id']}|{x['arm']}"])
            unique: Dict[str, dict] = {}
            for x in packets:      # send each distinct packet once (also avoids paying for the same request twice)
                unique.setdefault(x["packet"]["packet_id"], x["packet"])
            for pid, status, raw in self.transport([unique[k] for k in sorted(unique)]):
                for pd in by_pid[pid]:
                    _Critic(pd.broker, pd.tally).ingest(pid, status, raw)
            for key in sorted(pend):                       # pass 2: deterministic replay on a fresh context, same seeds, same broker
                pd = pend[key]
                final[(pd.task.task_id, pd.arm)], _ = self._one(pd.task, pd.arm, 2, pd.broker, pd.tally)
        for (tid, arm), row in sorted(final.items()):
            self.records.add(tid, arm, row)
        self._selected = True
        return {"rows": len(final), "pending_packets": len(packets), "transport_used": bool(packets and self.transport is not None)}

    # ------------------------------------------------------------ freeze, close, grade
    def freeze_and_close(self) -> str:
        if not self._selected:
            raise RuntimeError("selection has not run")
        man = []
        for tid in self.records.task_ids():
            for arm in self.arms:
                row = self.records.get(tid, arm)
                art = self._artifacts.get((tid, row["final_artifact_id"])) if row["final_artifact_id"] else None
                man.append({"task_id": tid, "arm": arm, "track": row["track"], "final_artifact_id": row["final_artifact_id"],
                            "source_sha256": sha256_text(art.source) if art else None, "journal_hash": row["journal_hash"],
                            "recoverable": art is not None})
                self._frozen[(tid, arm)] = {"entry": man[-1], "artifact": art}
        write_jsonl(self.out / "selection" / "selection_log.jsonl", [self.records.get(t, a) for t in self.records.task_ids() for a in self.arms])
        write_jsonl(self.out / "selection" / "selection_log_all_passes.jsonl", self._rows_all)
        write_json(self.out / "selection" / "frozen_artifacts.json", {"label": DEV_LABEL, "artifacts": man})
        h = sha(man)
        self.gate.close_selection(h)           # BEFORE any protected grading
        return h

    def grade_all(self) -> List[dict]:
        if not self._frozen:
            raise GraderIsolationError("nothing frozen: freeze_and_close() has not run")
        ids = {(f["entry"]["task_id"], f["entry"]["final_artifact_id"]) for f in self._frozen.values() if f["artifact"] is not None}
        gg = GatedGrader(self._grader, self.gate, ids)
        out = []
        for (tid, arm), f in sorted(self._frozen.items()):
            e = f["entry"]
            if f["artifact"] is None:    # no answer, or an artifact the controller never executed: counts as a failure and is flagged
                g = {"passed": False, "passed_tests": 0, "total_tests": 0, "sha256_before_run": "",
                     "failure": "no_final_artifact" if e["final_artifact_id"] is None else "final_artifact_not_recoverable", "graded": False}
            else:
                g = dict(gg.grade(f["artifact"]), graded=True)
            out.append(dict(g, task_id=tid, arm=arm, track=e["track"], artifact_id=e["final_artifact_id"]))
        self._graded = True
        self._grades = out
        write_jsonl(self.out / "grading" / "grades.jsonl", [dict(label=DEV_LABEL, **g) for g in out])
        return out

    def build_report(self) -> dict:
        results: Dict[str, Dict[str, Dict[str, dict]]] = {}
        for g in self._grades:
            results.setdefault(g["track"], {}).setdefault(g["task_id"], {})[g["arm"]] = g
        meta = {tr: {t.task_id: t for t in ts} for tr, ts in self.tasks.items()}
        rows = [self.records.get(t, a) for t in self.records.task_ids() for a in self.arms]
        rep = make_report(results, meta, rows, T=self.T, synthetic=self.synthetic, seed=self.seed)
        write_json(self.out / "report.json", rep)
        with open(self.out / "report.md", "w", encoding="utf-8", newline="\n") as f:
            f.write(render_markdown(rep))
        return rep

    def run(self) -> dict:
        info = self.run_selection()
        info["frozen_sha256"] = self.freeze_and_close()
        self.grade_all()
        rep = self.build_report()
        return {"selection": info, "report": rep}
