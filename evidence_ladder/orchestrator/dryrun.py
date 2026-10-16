"""Deterministic end-to-end dry run with scripted fakes. No model, no network, no accuracy claim: every outcome here is invented by the scripts below."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from evidence_ladder.common.interfaces import Artifact, Generation, GenerationRequest, ProblemResult, TaskSpec
from evidence_ladder.orchestrator.base import SYNTHETIC_LABEL, sha, sha_bytes, write_json
from evidence_ladder.orchestrator.pilot import PilotRunner, SelectionGate


def _kind(code: str) -> str:
    return code.split(":", 1)[0]


class FakeGeneration:
    def generate(self, req: GenerationRequest) -> List[Generation]:
        out = []
        for s in list(req.seeds)[:req.n]:
            kind = ("BAD", "TRAP")[s % 7] if s % 7 < 2 else "GOOD"
            code = f"{kind}:{req.task_id}:{s}"
            out.append(Generation(code, code, "ok", 50, 100, s, "fake", "0", req.prompt_template_id, sha(req.prompt), 0.5))
        return out

    def model_info(self):
        return {"fake": True}


class FakeExecution:
    def run_public(self, artifact, task):
        return [{"check_id": "public", "status": "fail" if _kind(artifact.source) == "BAD" else "pass"}]

    def run_experiment(self, artifact, experiment):
        return {"check_id": "experiment", "status": "fail" if _kind(artifact.source) != "GOOD" else "pass"}

    def backend_info(self):
        return {"fake": True}


class FakeGrader:
    def grade(self, artifact):
        ok = _kind(artifact.source) == "GOOD"
        return {"artifact_id": artifact.artifact_id, "task_id": artifact.task_id, "passed": ok, "passed_tests": 10 if ok else 0, "total_tests": 10,
                "failure": None if ok else "wrong_answer", "sha256_before_run": sha(artifact.source)}


class FakeBroker:
    def __init__(self):
        self.packets, self.done = {}, set()

    def build_packet(self, req):
        pid = sha(req)[:16]
        self.packets[pid] = {"packet_id": pid, "request": req, "created": True}
        return self.packets[pid]

    def export_pending(self):
        return [p for k, p in sorted(self.packets.items()) if k not in self.done]

    def ingest(self, packet_id, http_status, raw_response):
        if packet_id in self.done:
            return {"state": "duplicate", "response": None}
        self.done.add(packet_id)
        return {"state": "ok", "response": {"fake": True}}


class FakeScorer:
    scorer_id = "fake"

    def score(self, decision_type, pool, exploratory=False):
        return [0.0 for _ in pool]


class FakeController:
    """Scripted stand-in for Agent 1's controller: just enough behaviour to exercise every orchestrator path."""
    def run_problem(self, task, ctx):
        b, mode, ev = ctx.budget, ctx.config["mode"], []

        def sample(seed):
            g = ctx.generation.generate(GenerationRequest(task.task_id, task.prompt, 1, [seed], 256))[0]
            b.charge("qwen_prefill", g.tokens_prompt)
            b.charge("qwen_gen", g.tokens_generated)
            a = Artifact.make(task.task_id, task.language, g.code, origin=f"sample:{seed}")
            ev.append(a.artifact_id)
            return a

        def pub(a):
            return ctx.execution.run_public(a, task)[0]["status"] == "pass"

        def done(art, label, cause="decided"):
            return ProblemResult(task.task_id, ctx.arm, art.artifact_id if art else None, label, {"decision_id": sha(ev)[:12]}, cause, list(ev),
                                 b.record(), 1, sha({"ev": ev, "arm": ctx.arm, "task": task.task_id}))

        seeds = list(ctx.seeds)
        if mode == "single":
            a = sample(seeds[0])
            return done(a, "PLAUSIBLE" if pub(a) else "UNVERIFIED")
        if mode == "best_of_k":
            arts = [sample(s) for s in seeds[:ctx.config["k"]]]
            pick = next((a for a in arts if pub(a)), arts[-1])
            return done(pick, "PLAUSIBLE" if pub(pick) else "UNVERIFIED")
        arts = [sample(s) for s in seeds[:ctx.config["k1"]]] if mode == "ladder" else [sample(seeds[0])]
        ok = [a for a in arts if pub(a)]
        if mode == "repair" or not ok:
            if not ok:
                bad = arts[0]
                fixed = Artifact.make(task.task_id, task.language, f"GOOD:{task.task_id}:repair", parent_id=bad.artifact_id, origin=f"repair:{bad.artifact_id}")
                ev.append(fixed.artifact_id)
                ok = [fixed] if pub(fixed) else []
            return done(ok[0] if ok else None, "PLAUSIBLE" if ok else "UNVERIFIED")
        if ctx.config.get("scorer") and ctx.scorer:
            b.note_call("scorer_calls")
            ctx.scorer.score("complete_program", ok, exploratory=True)
        if ctx.critic is not None:
            ctx.critic.build_packet({"task": task.task_id, "candidates": [a.artifact_id for a in ok]})
            if ctx.critic.export_pending():
                return done(None, "UNVERIFIED", "api_unavailable")
            b.charge("glm_in", 100)
            b.charge("glm_out", 40)
        good = [a for a in ok if ctx.execution.run_experiment(a, {"kind": "probe"})["status"] == "pass"]
        return done(good[0] if good else None, "VERIFIED" if good else "BLOCKED")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def synthetic_tasks():
    def mk(tid, track, state):
        ms = {"state": state, "flagged": state != "measured_ok"}
        return TaskSpec(tid, f"syn:{tid}", track, track, f"synthetic task {tid}", ({"input": 1, "expected": 1},), "easy", ms)
    return {"python": [mk("SYN-P1", "python", "measured_ok"), mk("SYN-P2", "python", "below_threshold"), mk("SYN-P3", "python", "measured_ok"),
                       mk("SYN-P4", "python", "unmeasured")],
            "java": [mk("SYN-J1", "java", "measured_ok"), mk("SYN-J2", "java", "measured_ok"), mk("SYN-J3", "java", "unmeasured")]}


def make_runner(out_dir, **over) -> PilotRunner:
    kw = dict(tasks=synthetic_tasks(), controller=FakeController(), generation=FakeGeneration(), execution=FakeExecution(), grader=FakeGrader(),
              gate=SelectionGate(), out_dir=Path(out_dir), T=10_000, k1=3, n_total=4, master_seed=20261016, critic_factory=lambda t, a: FakeBroker(),
              scorer=FakeScorer(), transport=lambda packets: [(p["packet_id"], 200, {"fake": True}) for p in packets], dry_run=True,
              clock=FakeClock(), synthetic=True)
    kw.update(over)
    return PilotRunner(**kw)


def run_dryrun(out_dir) -> dict:
    out = Path(out_dir)
    info = make_runner(out).run()["selection"]
    hashes = {str(p.relative_to(out)).replace("\\", "/"): sha_bytes(p.read_bytes()) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "DRYRUN_MANIFEST.json", {"banner": SYNTHETIC_LABEL, "claims_accuracy": False, "selection": info, "file_sha256": hashes})
    hashes["DRYRUN_MANIFEST.json"] = sha_bytes((out / "DRYRUN_MANIFEST.json").read_bytes())
    return hashes


if __name__ == "__main__":
    import json
    print(json.dumps(run_dryrun(sys.argv[1]), sort_keys=True, indent=1))
