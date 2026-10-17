"""EvidenceLadderController: implements interfaces.Controller (run_problem(task, ctx) -> ProblemResult). Pure standard library, no model calls of its own.

Mode (ctx.config["mode"], default "ladder"):
  single     one attempt, no repair; label from the public check only (arm A)
  best_of_k  config["k"] attempts, select by public pass count then candidate id, no repair (arm B)
  repair     k samples + repair rounds from the real public-check counterexample, S5 stop rules, no clusters / critic / scorer (arm C)
  ladder     full Evidence-Ladder (arm D0): per-requirement evidence, C.decide labels, C.rank, progress-based repair, all S5 stop causes, behaviour clusters, journal.
Other config keys: k (default 4), max_rounds (3), max_attempts (32), repair_n (1), max_new_tokens (4096), environment_revision, risk ("sandboxed"),
user_confirmed (False), finalize (True), proposed_requirements ([]), critic (False), critic_experiment (False: ask even with one cluster), critic_responses
({packet_id or request_id: critic_response | ingest result}), scorer (False), scorer_validated_types ([]), exploratory (False).

ctx.critic usage contract (CriticBroker has no "get response" call, so):
  1. the controller builds a blinded critic_request and calls ctx.critic.build_packet(request); packet_id = packet["packet_id"].
  2. a response is "available" if it is in ctx.config["critic_responses"] under packet_id or request_id (the orchestrator puts the broker's ingest() result there for the
     second pass; either the bare critic_response or {"state":"ok","response":{...}}), or if the packet returned by build_packet already carries "response".
  3. otherwise the pass ends with stop_cause "api_unavailable", details "critic_pending"; ctx.critic.export_pending() shows what is waiting. The orchestrator ingests
     and runs a second pass (same task, same seeds, config["critic_responses"] filled). Request ids are content-derived, so the second pass asks for the same packet.
  4. malformed responses are journaled, counted as a wasted call, and the run continues without the critic.
Registry hash and scorer fingerprint are frozen for the run (I7): a change raises RuntimeError."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from evidence_ladder.common.contract_import import C
from evidence_ladder.common.interfaces import (Artifact, BudgetExhausted, ControllerContext, GenerationRequest, ProblemResult, TaskSpec,
                                               UnsupportedDecisionType, sha256_text)
from . import intake as intake_mod
from . import router as router_mod
from .decision import blocked_without_candidate, decide_stage1, with_next
from .components import MissingComponentError, require
from .limits import Limiter, load_limits
from .evidence import Factory, _s, is_tool_failure, map_status, public_records, trim
from .invariants import assert_invariants
from .journal import EvidenceJournal, canon
from .registry import build_registry, digest

MODES = ("single", "best_of_k", "repair", "ladder")
KILL_CAUSES = ("same_failure_twice", "unrepaired_regression")
OK_LABELS = ("VERIFIED", "SUPPORTED")
REPAIR_TEMPLATE = ("Previous candidate: {cid}.\nVerified failure (level {level}): {inp} -> expected {exp}, observed {obs}.\n"
                   "Requirement (quoted): \"{quote}\".\nUnchanged constraints: {constraints}.\n"
                   "Task: propose the smallest corrected change. Do not modify anything outside {scope}.")


class _Cand:
    def __init__(self, art: Artifact, lineage: str, depth: int):
        self.art, self.lineage, self.depth = art, lineage, depth
        self.id = art.artifact_id
        self.summary: Dict[str, Any] = {}
        self.decision: Dict[str, Any] = {}
        self.rows: Dict[str, str] = {}
        self.verified: set = set()
        self.trusted = self.validated = self.fails = 0
        self.sigs: set = set()
        self.first_fail: Optional[Dict[str, str]] = None
        self.repaired = self.demoted = False
        self.scorer = self.deepconf = 0.0


class _Run:
    def __init__(self, ctl: "EvidenceLadderController", task: TaskSpec, ctx: ControllerContext):
        self.ctl, self.task, self.ctx, self.cfg = ctl, task, ctx, dict(ctx.config)
        self.mode = self.cfg.get("mode", "ladder")
        if self.mode not in MODES:
            raise ValueError("unknown mode %r" % (self.mode,))
        self.ladder = self.mode == "ladder"
        self.k = 1 if self.mode == "single" else int(self.cfg.get("k", 4))
        self.max_rounds = int(self.cfg.get("max_rounds", 3)) if self.mode in ("repair", "ladder") else 0
        self.max_attempts = int(self.cfg.get("max_attempts", 32))
        self.risk = self.cfg.get("risk", "sandboxed")
        self.user_confirmed = bool(self.cfg.get("user_confirmed", False))
        self.cands: Dict[str, _Cand] = {}
        self.attempts = self.rounds = self.seed_i = 0
        self.budget_hit = self.tool_failure = self.critic_done = self.scorer_off = self.scorer_noted = False
        self.wasted_critic = 0
        self.entries: List[Dict[str, Any]] = []         # repair-round entries for C.stop_cause
        self.dead: set = set()
        self.started: set = set()
        self.seen_sigs: set = set()
        self.pending_cluster_progress = False
        self._scorer_key = None
        self.limiter = Limiter(ctl.limits, ctl.clock)
        self.limit_detail = ""

    # ------------------------------------------------------------ plumbing
    def limit_check(self) -> bool:
        """[EL-ARCH-1 A3] True (and budget_hit set, details remembered) when a per-problem limit forbids more work: wall time, Qwen/GLM tokens, actions (attempts), rounds."""
        self.limiter.start()
        b = self.ctx.budget.record()
        r = self.limiter.exceeded(int(b.get("qwen_gen_tokens", 0)) + int(b.get("qwen_prefill_tokens", 0)), int(b.get("glm_tokens_in", 0)) + int(b.get("glm_tokens_out", 0)),
                                  self.attempts, self.rounds)
        if r is not None:
            self.budget_hit, self.limit_detail = True, r["details"]
            self.note("limit", {"resource": r["resource"], "cap": r["cap"], **({} if r["resource"] == "max_wall_seconds" else {"used": r["used"]})})      # wall time is not journaled (reproducible hash)
        return r is not None

    def note(self, kind: str, data: Any) -> None:
        self.journal.append(kind, data)

    def check_frozen(self) -> None:
        C.assert_frozen(self.reg_hash0, digest(self.registry))

    def scorer_fp(self) -> Optional[str]:
        s = self.ctx.scorer
        return None if s is None else str(getattr(s, "weights_hash", None) or getattr(s, "scorer_id", ""))

    def next_seeds(self, n: int) -> List[int]:
        base, out = list(self.ctx.seeds), []
        for _ in range(n):
            out.append(int(base[self.seed_i]) if self.seed_i < len(base) else 10000 + self.seed_i)
            self.seed_i += 1
        return out

    def generate(self, purpose: str, prompt: str, n: int):
        n = min(n, self.max_attempts - self.attempts, self.limiter.limits.max_actions - self.attempts)
        if n <= 0:
            return []
        if self.limit_check():
            return []
        if self.ctx.budget.exhausted():
            self.budget_hit = True
            return []
        seeds = self.next_seeds(n)
        template = {"sample": "qwen_sample_v2", "repair": "repair_v1"}.get(purpose, purpose)
        if purpose == "sample":
            # F13 (audit): the generation prompt carries the required solution format (qwen_sample_v2, the template the FC-3 runs used); request_text stays the quoting base of the requirements
            from evidence_ladder.models import qwen as _Q
            prompt, _ = _Q.render("qwen_sample_v2", task_prompt=self.task.prompt, examples=_Q.format_examples(self.task.public_tests),
                                  solution_format=_Q.SOLUTION_FORMATS[self.task.language], language=self.task.language)
        self.note("generation_request", {"purpose": purpose, "n": n, "seeds": seeds, "prompt_sha": sha256_text(prompt)})
        gens = list(self.ctx.generation.generate(GenerationRequest(self.task.task_id, prompt, n, tuple(seeds), int(self.cfg.get("max_new_tokens", 4096)),
                                                                     purpose, bool(self.cfg.get("thinking", True)), template)))
        out = []
        for g in gens:
            out.append(g)                       # the generation exists and cost tokens, so it is kept even if it crossed the cap
            try:
                self.ctx.budget.charge("qwen_prefill", int(g.tokens_prompt))
                self.ctx.budget.charge("qwen_gen", int(g.tokens_generated))
            except BudgetExhausted:
                self.budget_hit = True
                self.note("budget", {"state": "exhausted_during_generation", "kept": len(out), "dropped": len(gens) - len(out)})
                break
        return out

    # ------------------------------------------------------------ candidates and evidence
    def add_candidate(self, gen, parent: Optional[_Cand] = None) -> Optional[_Cand]:
        if gen.extract_status != "ok" or not gen.code.strip():
            self.note("candidate_rejected", {"reason": "no_code", "seed": gen.seed})
            return None
        art = Artifact.make(self.task.task_id, self.task.language, gen.code, parent.id if parent else None,
                            ("repair:%s" % parent.id) if parent else "sample:%d" % gen.seed)
        if art.artifact_id in self.cands:
            self.note("candidate_rejected", {"reason": "duplicate", "id": art.artifact_id})
            return None
        c = _Cand(art, parent.lineage if parent else art.artifact_id, parent.depth + 1 if parent else 0)
        lp = gen.logprob_summary or {}
        c.deepconf = float(lp.get("group_confidence", lp.get("mean", 0.0)) or 0.0)
        self.note("candidate", {"id": c.id, "parent": art.parent_id, "lineage": c.lineage, "origin": art.origin, "source": art.source})
        self.cands[c.id] = c
        self.attempts += 1
        self.evaluate(c)
        return c

    def record(self, recs) -> None:
        for ev, out in recs:
            self.journal.add_evidence(ev, out)

    def run_public(self, art: Artifact):
        try:
            outcomes = list(self.ctx.execution.run_public(art, self.task))
        except BudgetExhausted:
            raise
        except Exception as ex:      # noqa: BLE001  an execution backend failure is a tool failure, never a candidate failure (S5)
            self.note("tool_failure", {"candidate": art.artifact_id, "error": type(ex).__name__ + ": " + str(ex)[:200]})
            self.tool_failure = True
            outcomes = [{"check_id": "public_examples_v1", "status": "error", "failure_class": "sandbox_error", "detail": "backend raised"}]
        self.check_frozen()
        return outcomes

    def evaluate(self, c: _Cand) -> None:
        outcomes = self.run_public(c.art)
        recs, summary = public_records(self.task, c.art, outcomes, self.intake, self.fac)
        self.record(recs)
        c.summary, c.sigs = summary, set(summary["sigs"])
        if summary["tool_failure"]:
            self.tool_failure = True
        c.first_fail = self.first_failure(summary)
        self.relabel(c)

    def first_failure(self, summary) -> Optional[Dict[str, str]]:
        for ex in summary["examples"]:
            if ex["status"] != "pass":
                t = self.task.public_tests[ex["i"]]
                rid = self.intake.example_ids[ex["i"]]
                quote = self.intake.by_id(rid)["source"].get("quote", "")
                obs = ex["observed"] or ex["status"]
                return {"level": "L3", "input": _s(t.get("input", ""), 500), "expected": _s(t.get("expected", ""), 500), "observed": _s(obs, 500), "quote": quote}
        if summary["parse"] not in (None, "pass") or summary["limits"] not in (None, "pass"):
            return {"level": "L5", "input": "(loading the program)", "expected": "the program loads within the limits", "observed": "parse=%s limits=%s" % (summary["parse"], summary["limits"]),
                    "quote": self.intake.by_id("R1")["text"]}
        return None

    def relabel(self, c: _Cand) -> None:
        d = decide_stage1(self.reqs, self.journal.evidence(c.id), c.id, c.id, self.env, self.registry, self.risk, self.user_confirmed, "complete_program")
        c.decision = d
        c.rows = {r["requirement_id"]: r["label"] for r in d["requirement_rows"]}
        c.verified = {k for k, v in c.rows.items() if v == "VERIFIED"}
        c.trusted = len(c.verified)
        c.validated = sum(1 for v in c.rows.values() if v == "SUPPORTED")
        c.fails = sum(1 for v in c.rows.values() if v in ("FAILED", "CONFLICTED"))
        self.note("label", {"candidate_id": c.id, "answer_label": d["answer_label"], "rows": c.rows, "trusted_passes": c.trusted, "validated_passes": c.validated, "fails": c.fails})

    def ok(self, c: _Cand) -> bool:
        return c.decision["answer_label"] in OK_LABELS

    # ------------------------------------------------------------ routing, scoring, ranking
    def route(self, step: str, c: Optional[_Cand] = None) -> None:
        c = c or (self.ranked()[0] if self.cands else None)
        ev = self.journal.evidence(c.id) if c else []
        table = router_mod.route(self.reqs, self.intake.roles, self.registry, ev, c.id if c else None)
        self.note("routing", {"step": step, "candidate_id": c.id if c else None, "ceiling": router_mod.answer_ceiling(table), "table": table})

    def apply_scorer(self) -> None:
        sc = self.ctx.scorer
        if not (self.ladder and self.cfg.get("scorer") and sc is not None) or self.scorer_off:
            return
        exploratory = bool(self.cfg.get("exploratory"))
        if not (exploratory or "complete_program" in self.cfg.get("scorer_validated_types", ())):
            if not self.scorer_noted:               # LAYA is validated for next_action only: not called for complete_program unless exploratory
                self.scorer_noted = True
                self.note("scorer", {"state": "skipped", "reason": "decision type complete_program is not validated and the run is not exploratory"})
            return
        live = sorted((c for c in self.cands.values() if c.fails == 0), key=lambda c: c.id)
        key = tuple(c.id for c in live)
        if len(live) < 2 or key == self._scorer_key:
            return
        try:
            scores = list(sc.score("complete_program", [c.art.source for c in live], exploratory))
        except Exception as ex:      # noqa: BLE001 UnsupportedDecisionType, NotAuthorized, ScorerDisabled: refuse the scorer, never crash the run
            self.scorer_off = True
            self.note("scorer", {"state": "refused", "reason": str(ex)[:200]})
            self.note("scorer_unavailable", {"reason": "%s: %s" % (type(ex).__name__, str(ex)[:200]), "fallback": "scorer_off_for_this_run", "exploratory": exploratory})   # F06: explicit, not silent
            return
        if len(scores) != len(live):
            self.scorer_off = True
            self.note("scorer", {"state": "ignored", "reason": "wrong number of scores"})
            self.note("scorer_unavailable", {"reason": "wrong number of scores", "fallback": "scorer_off_for_this_run", "exploratory": exploratory})
            return
        self._scorer_key = key
        for c, s in zip(live, scores):
            c.scorer = float(s)
        self.note("scorer", {"state": "scored", "scorer_id": sc.scorer_id, "ids": list(key), "scores": [float(s) for s in scores], "exploratory": exploratory})

    def ranked(self, log: Optional[str] = None) -> List[_Cand]:
        """Lexicographic via C.rank: trusted passes, validated passes, cluster size, scorer, confidence, id. Arms A-C use trusted passes (public pass count) then id."""
        if self.ladder:
            self.apply_scorer()
        size: Dict[str, int] = {}
        for c in self.cands.values():
            if c.fails == 0:
                size[c.summary["behaviour"]] = size.get(c.summary["behaviour"], 0) + 1
        items = []
        for c in self.cands.values():
            items.append({"id": c.id, "trusted_passes": c.trusted, "validated_passes": c.validated if self.ladder else 0,
                          "cluster": (-1 if c.demoted else size.get(c.summary["behaviour"], 0) if c.fails == 0 else 0) if self.ladder else 0,
                          "scorer": c.scorer if self.ladder else 0.0, "deepconf": c.deepconf if self.ladder else 0.0})
        order = [x["id"] for x in C.rank(items)]
        if log:
            self.note("ranking", {"step": log, "items": items, "order": order})
        return [self.cands[i] for i in order]

    # ------------------------------------------------------------ critic (ladder only)
    def critic_stage(self):
        """-> None (continue) or (stop_cause, details)."""
        live = sorted((c for c in self.cands.values() if c.fails == 0), key=lambda c: c.id)
        clusters = {c.summary["behaviour"] for c in live}
        if not live or (len(clusters) < 2 and not self.cfg.get("critic_experiment")):
            return None
        ids = [c.id for c in live]
        rid = "crit-" + C.sha256(self.task.task_id + "|" + "|".join(ids))[:16]
        req = {"request_id": rid, "model": "z-ai/glm-5.3-flash", "decision_id": "d-" + rid, "candidate_ids": ids,
               "requirement_quotes": [r["source"]["quote"] for r in self.reqs if r["kind"] == "explicit" and r["source"].get("quote")],
               "candidate_code": {c.id: c.art.source for c in live},
               # broker allow-list: per candidate a LIST of dicts with only input/observed/status/failure_class/detail/public_expected/check_id (never an `expected` key)
               "observed_outputs": {c.id: [{"input": _s(self.task.public_tests[e["i"]].get("input", ""), 500), "observed": e["observed"], "status": e["status"],
                                            "public_expected": _s(self.task.public_tests[e["i"]].get("expected", ""), 500), "check_id": "public_examples_v1"}
                                           for e in c.summary["examples"]] for c in live},
               "question": "distinguish" if len(clusters) >= 2 else "challenge_requirement"}
        C.validate("critic_request", req)
        try:
            packet = self.ctx.critic.build_packet(req)
        except Exception as ex:      # noqa: BLE001 CriticStopped after a payment/quota error: continue without the critic, never retry
            if type(ex).__name__ != "CriticStopped":
                raise
            self.critic_done = True
            self.wasted_critic += 1
            self.note("critic", {"state": "unavailable", "packet_id": None, "ids": ids, "reason": str(ex)[:200]})
            return None
        pid = packet.get("packet_id", rid)
        raw = None
        done = getattr(self.ctx.critic, "completed", None)
        if done is not None:                       # the broker's own record of an ingested "ok" packet wins; config["critic_responses"] is a test fallback
            raw = done(pid)
        raw = raw if raw is not None else (self.cfg.get("critic_responses") or {}).get(pid)
        raw = raw if raw is not None else (self.cfg.get("critic_responses") or {}).get(rid)
        raw = raw if raw is not None else packet.get("response")
        if raw is None:              # a packet that already ended malformed/unavailable is final, not pending
            term = getattr(self.ctx.critic, "terminal_state", None)
            ts = term(pid) if term is not None else None
            if ts == "malformed":
                raw = {"__malformed__": True}
            elif ts == "unavailable":
                raw = {"state": "unavailable"}
        if isinstance(raw, dict) and "state" in raw and "experiment" not in raw:
            if raw["state"] == "ok":
                raw = raw.get("response")
            elif raw["state"] == "malformed":
                raw = {"__malformed__": True}
            elif raw["state"] == "unavailable" or raw.get("stop_all_critic_calls"):
                # the broker gave up on this packet: continue without the critic, count it, never retry it ourselves
                self.critic_done = True
                self.wasted_critic += 1
                self.note("critic", {"state": "unavailable", "packet_id": pid, "ids": ids})
                return None
            elif raw["state"] == "retry":
                raw = None
            else:
                raw = None
        if raw is None:
            ps = getattr(self.ctx.critic, "pending_summary", None)
            self.note("critic", {"state": "pending", "packet_id": pid, "ids": ids, "waiting": (ps().get("ready") if ps is not None else None)})
            return ("api_unavailable", "critic_pending")
        self.critic_done = True
        try:
            if not isinstance(raw, dict) or raw.get("__malformed__"):
                raise ValueError("malformed")
            C.validate("critic_response", raw)
        except Exception as ex:      # noqa: BLE001 jsonschema.ValidationError or malformed input: log, waste, continue without the critic
            self.wasted_critic += 1
            self.note("critic", {"state": "malformed", "packet_id": pid, "error": str(ex)[:200]})
            return None
        self.apply_critic(raw, live, pid)
        return None

    def apply_critic(self, resp: Dict[str, Any], live: List[_Cand], pid: str) -> None:
        exp = resp["experiment"]
        # [EL-ARCH-1 A4] the requirement the evidence attaches to is the CONTROLLER's choice (the "all legal inputs" row); the id the critic names is recorded, never obeyed
        target = self.intake.all_inputs_id
        named = resp.get("requirement_id")
        payload = exp["payload"].strip()
        groups: Dict[Any, List[_Cand]] = {}
        outs = {}
        touched = set()
        for c in live:
            try:
                out = dict(self.ctx.execution.run_experiment(c.art, exp))
            except Exception as ex:      # noqa: BLE001
                self.tool_failure = True
                self.note("tool_failure", {"candidate": c.id, "error": type(ex).__name__ + ": " + str(ex)[:200]})
                continue
            self.check_frozen()
            st = map_status(out)
            outs[c.id] = trim(out)
            # the critic's output is model-derived ranking evidence only
            rec_status = "unknown" if st == "fail" else st          # a model-proposed input can demote in ranking but never records a candidate failure by itself
            self.journal.add_evidence(self.fac.make(c.art, target, "critic_experiment_v1", "critic_experiment", "model_derived", rec_status, "critic experiment: ranking only", out), trim(out))
            if exp["kind"] in ("stdin_input", "function_call") and "observed" in out:
                for i, t in enumerate(self.task.public_tests):
                    if payload == _s(t.get("input", "")).strip() and _s(out["observed"]).strip() != _s(t.get("expected", "")).strip():
                        # the critic proposed the input, the execution produced the evidence: a quoted example is broken (separate record, S2 section 2)
                        cx = {"input": _s(t.get("input", "")), "expected": _s(t.get("expected", "")), "observed": _s(out["observed"])}
                        self.journal.add_evidence(self.fac.make(c.art, self.intake.example_ids[i], "public_examples_v1", "environment_observation", "independent", "fail",
                                                                "critic input reproduces a quoted example; observed output differs", out, cx), trim(out))
                        touched.add(c.id)
            groups.setdefault((st, _s(out.get("observed", ""), 500)), []).append(c)
        for cid in sorted(touched):
            self.relabel(self.cands[cid])
        demoted = []
        passing = [g for k, g in groups.items() if k[0] == "pass"]
        if len(groups) >= 2 and passing:
            for (st, _o), g in groups.items():
                if st != "pass":
                    for c in g:
                        c.demoted = True
                        demoted.append(c.id)
        self.pending_cluster_progress = self.pending_cluster_progress or bool(demoted)
        self.note("critic", {"state": "applied", "packet_id": pid, "ids": [c.id for c in live], "demoted": sorted(demoted), "outcomes": outs, "clusters_after": len(groups)}
                  if named in (None, target) else {"state": "applied", "packet_id": pid, "ids": [c.id for c in live], "demoted": sorted(demoted), "outcomes": outs, "clusters_after": len(groups),
                                                   "critic_named_requirement": str(named)[:40], "attached_to": target})
        self.route("critic")

    # ------------------------------------------------------------ repair
    def repair_target(self) -> Optional[_Cand]:
        for c in self.ranked():
            if c.lineage not in self.dead and not c.repaired and not self.ok(c):
                return c
        return None

    def repair_round(self, target: _Cand) -> None:
        target.repaired = True
        f = target.first_fail or {"level": "L3", "input": "n/a", "expected": "n/a", "observed": "n/a", "quote": self.intake.by_id("R2")["text"]}
        constraints = "; ".join(r["text"] for r in self.reqs if r["kind"] == "contract") + "; keep the entry point `solve` and its signature"
        tmpl = REPAIR_TEMPLATE.format(cid=target.id, level=f["level"], inp=f["input"], exp=f["expected"], obs=f["observed"], quote=f["quote"],
                                      constraints=constraints, scope="`solve` and the helpers it uses")
        prompt = "%s\n\nPrevious candidate code:\n```%s\n%s\n```\n\n%s" % (self.intake.request_text, self.task.language, target.art.source, tmpl)
        gens = self.generate("repair", prompt, int(self.cfg.get("repair_n", 1)))
        children = [c for c in (self.add_candidate(g, target) for g in gens) if c is not None]
        self.rounds += 1
        if target.lineage not in self.started:
            self.started.add(target.lineage)
            self.entries.append({"failure_sigs": sorted(target.sigs), "progress": True, "regression": False, "lineage": target.lineage})
        progress, regression, sigs = False, False, []
        if children:
            best = min(children, key=lambda c: (-c.trusted, -c.validated, c.id))
            progress = bool(best.verified - target.verified) or bool(best.sigs - self.seen_sigs) or (self.ladder and self.pending_cluster_progress)
            regression = bool(target.verified - best.verified)
            sigs = sorted(best.sigs)
            self.seen_sigs |= best.sigs
        self.pending_cluster_progress = False
        self.entries.append({"failure_sigs": sigs, "progress": progress, "regression": regression, "lineage": target.lineage})
        self.note("round", {"n": self.rounds, "target": target.id, "children": [c.id for c in children], "progress": progress, "regression": regression, "sigs": sigs})
        self.route("repair")

    def after_round(self):
        cause = C.stop_cause(self.entries, 0, 1)      # budget_used=0 / cap=1 on purpose: only the NON-budget causes are wanted here; every budget cap is enforced in loop() and limit_check()
        if cause is None:
            return None
        if cause in KILL_CAUSES:
            for lin in sorted({e["lineage"] for e in self.entries} - self.dead):
                if C.stop_cause([e for e in self.entries if e["lineage"] == lin], 0, 1) in KILL_CAUSES:
                    self.dead.add(lin)
                    for e in self.entries:
                        if e["lineage"] == lin:
                            e["failure_sigs"], e["regression"] = [], False
                    self.note("lineage_stopped", {"lineage": lin, "cause": cause})
            if self.repair_target() is not None:
                return None
        return (cause, "")

    # ------------------------------------------------------------ main loop
    def loop(self):
        self.seen_sigs = set().union(*[c.sigs for c in self.cands.values()]) if self.cands else set()
        self.route("samples")
        while True:
            if self.tool_failure:
                return ("tool_failure", "sandbox")
            if self.budget_hit:
                return ("budget", self.limit_detail or "token_cap")
            if self.ladder and self.cfg.get("critic") and self.ctx.critic is not None and not self.critic_done:
                r = self.critic_stage()
                if r:
                    return r
                if self.tool_failure:
                    continue
            ranked = self.ranked("loop")
            if ranked and self.ok(ranked[0]):
                return ("decided", "")
            if self.max_rounds == 0:
                return ("budget", "attempt_cap:" + self.mode)        # stop_record has no 'cap reached' cause; reported as a gap
            if self.rounds >= self.max_rounds:
                return ("budget", "round_cap")
            if self.attempts >= self.max_attempts:
                return ("budget", "attempt_cap")
            if self.limit_check():      # per-problem limits (limits.py) after the legacy cfg caps: a decided problem is never turned into a budget stop
                return ("budget", self.limit_detail)
            target = self.repair_target()
            if target is None:
                if not self.cands and self.rounds < self.max_rounds:     # nothing usable was ever extracted: sample again, counted as a round without progress
                    for g in self.generate("sample", self.intake.request_text, self.k):
                        self.add_candidate(g)
                    self.rounds += 1
                    self.entries.append({"failure_sigs": [], "progress": False, "regression": False, "lineage": "none"})
                    cause = C.stop_cause(self.entries, 0, 1)
                    if cause:
                        return (cause, "")
                    continue
                return ("no_progress_2", "no_repair_target")
            self.repair_round(target)
            r = self.after_round()
            if r:
                return r

    # ------------------------------------------------------------ finalize
    def finalize(self, art: Artifact, expected_hash: Optional[str] = None) -> Dict[str, Any]:
        """Recompute the artifact hash, compare, re-run the trusted public checks through ctx.execution, relabel. Blocks on any mismatch or contradiction."""
        recomputed = sha256_text(art.source)
        want = expected_hash or art.artifact_id
        if recomputed != want or recomputed != art.artifact_id:
            self.note("finalize", {"candidate_id": art.artifact_id, "ok": False, "reason": "hash_mismatch", "recomputed": recomputed})
            return {"ok": False, "reason": "hash_mismatch", "decision": blocked_without_candidate(self.reqs, "d-final-" + art.artifact_id, art.artifact_id, recomputed)}
        c = self.cands.get(art.artifact_id)
        if c is None:
            self.note("finalize", {"candidate_id": art.artifact_id, "ok": False, "reason": "unknown_candidate"})
            return {"ok": False, "reason": "unknown_candidate", "decision": blocked_without_candidate(self.reqs, "d-final-" + art.artifact_id, art.artifact_id, recomputed)}
        outcomes = self.run_public(art)
        recs, summary = public_records(self.task, art, outcomes, self.intake, self.fac)
        self.record(recs)
        self.relabel(c)
        ok = (not self.tool_failure) and summary["all_pass"] and self.ok(c)
        reason = "" if ok else ("tool_failure" if self.tool_failure else "recheck_not_confirmed:" + c.decision["answer_label"])
        self.note("finalize", {"candidate_id": c.id, "ok": ok, "reason": reason, "answer_label": c.decision["answer_label"]})
        self.route("finalize", c)
        return {"ok": ok, "reason": reason, "decision": c.decision}

    def finalize_post_run(self, art: Artifact, expected_hash: Optional[str] = None) -> Dict[str, Any]:
        """finalize() after the run: the already-returned journal (and its journal_hash) is not touched. The check runs in a second journal segment that starts from a
        copy of the evidence and its own chain; its hash is returned as finalize_hash."""
        fin = EvidenceJournal(self.registry)
        for e in self.journal.entries("evidence"):
            fin.add_evidence(e["data"]["evidence"], e["data"]["outcome"])
        fin.append("finalize_segment", {"main_journal_hash": self.journal.journal_hash()})
        main, self.journal = self.journal, fin
        try:
            out = self.finalize(art, expected_hash)
        finally:
            self.journal = main
        self.finalize_journal = fin
        return dict(out, finalize_hash=fin.journal_hash())

    # ------------------------------------------------------------ run
    def execute(self) -> ProblemResult:
        task, ctx = self.task, self.ctx
        require("generation backend", ctx.generation, "every controller arm samples candidates")
        require("execution backend / sandbox", ctx.execution, "no candidate can be checked without it")
        require("budget", ctx.budget, "token and call accounting")
        wanted = [(n, o, k) for n, o, k in (("critic", ctx.critic, "critic"), ("scorer", ctx.scorer, "scorer")) if self.ladder and self.cfg.get(k)]
        need = set(self.cfg.get("require", ()))
        for n, o, _k in wanted:      # [EL-ARCH-1 A5] an arm that asks for the critic/scorer and has none: cfg["require"] (final runs) raises BEFORE any model call; otherwise it is journaled below
            if o is None and n in need:
                raise MissingComponentError(n, "arm config asks for it and cfg['require'] lists it")
        self.limiter.start()
        self.intake = intake_mod.build(task, self.cfg.get("proposed_requirements"))
        self.reqs = self.intake.reqs
        self.registry = build_registry(self.intake.example_ids)
        self.ctl._registry = self.registry
        self.reg_hash0 = digest(self.registry)
        self.journal = EvidenceJournal(self.registry)
        info = ctx.execution.backend_info()
        self.env = self.cfg.get("environment_revision") or ("env-" + C.sha256(canon(info, default=str))[:12])
        self.fac = Factory(self.registry, self.reqs, self.env, self.intake.roles)
        self.note("run_start", {"task_id": task.task_id, "arm": ctx.arm, "mode": self.mode, "seeds": [int(s) for s in ctx.seeds], "environment_revision": self.env,
                                "config_digest": C.sha256(canon(self.cfg, default=str))})
        self.note("registry", {"phase": "start", "hash": self.reg_hash0, "scorer": self.scorer_fp()})
        self.note("requirements", {"reqs": self.reqs, "request_sha": sha256_text(self.intake.request_text), "stop": self.intake.stop, "notes": self.intake.notes})
        for n, o, _k in wanted:
            if o is None:
                self.note("component_missing", {"component": n, "why": "the arm config asks for it but the context has none; the run continues WITHOUT it", "arm": ctx.arm, "waived": True})
        self.route("intake")
        if self.intake.stop:
            cause, details = self.intake.stop, self.intake.stop_details
        else:
            for g in self.generate("sample", self.intake.request_text, self.k):
                self.add_candidate(g)
            cause, details = self.loop()
        return self.finish(cause, details)

    def finish(self, cause: str, details: str) -> ProblemResult:
        task, ctx = self.task, self.ctx
        best: Optional[_Cand] = self.ranked("final")[0] if self.cands else None
        if cause == "decided" and best is not None and self.cfg.get("finalize", True):
            fin = self.finalize(best.art)
            if not fin["ok"]:
                cause, details = "tool_failure", "finalize:" + fin["reason"]
            best = self.ranked("final_after_finalize")[0]
        if best is not None:
            d = best.decision
            if cause == "api_unavailable" and details == "critic_pending":
                d = with_next(d, "discriminate")
            elif d["answer_label"] != "BLOCKED" and not d["authorized_to_execute"]:
                d = with_next(d, "report")
        elif self.intake.stop:
            d = blocked_without_candidate(self.reqs, "d-intake-" + task.task_id)
        else:
            d = with_next(decide_stage1(self.reqs, [], "none", C.sha256(""), self.env, self.registry, self.risk, self.user_confirmed, "complete_program"), "report")
        self.note("decision", d)
        stop = {"problem_id": task.task_id, "arm": ctx.arm, "cause": cause}
        if details:
            stop["details"] = details
        C.validate("stop_record", stop)
        self.note("stop", dict(stop, rounds=self.rounds, max_rounds=self.max_rounds, attempts=self.attempts, max_attempts=self.max_attempts,
                               budget_hit=self.budget_hit, wasted_critic=self.wasted_critic, mode=self.mode))
        end = digest(self.registry)
        self.note("registry", {"phase": "end", "hash": end, "scorer": self.scorer_fp()})
        C.assert_frozen(self.reg_hash0, end)
        result = ProblemResult(task.task_id, ctx.arm, best.id if best else None, d["answer_label"], d, cause,
                               [e["evidence_id"] for e in self.journal.evidence(best.id)] if best else [], ctx.budget.record(), self.rounds, self.journal.journal_hash(),
                               best.art if best else None, details)
        assert_invariants(result, self.journal, ctx, self.limiter.limits)
        self.ctl.last_journal = self.journal
        return result


class EvidenceLadderController:
    """Implements interfaces.Controller. One instance can run many problems; last_journal / last_run describe the most recent one."""

    def __init__(self, limits=None, clock=None):
        """limits: evidence_ladder.controller.limits.Limits (default: profile "ladder" of configs/el_arch1_limits_v1.toml). clock: injected monotonic clock for the wall-time limit."""
        import time as _time
        self.limits = limits if limits is not None else load_limits("ladder")
        self.clock = clock or _time.monotonic
        self._registry = None
        self.last_journal: Optional[EvidenceJournal] = None
        self.last_run: Optional[_Run] = None
        self.finalize_hash: Optional[str] = None

    def run_problem(self, task: TaskSpec, ctx: ControllerContext) -> ProblemResult:
        run = _Run(self, task, ctx)
        self.last_run = run
        return run.execute()

    def finalize(self, artifact: Artifact, expected_hash: Optional[str] = None) -> Dict[str, Any]:
        """Final verification of a committed artifact against the last run: hash recompute + trusted public checks re-run. -> {"ok","reason","decision","finalize_hash"}; ok False means BLOCK.
        The run's journal and journal_hash are unchanged; the check is recorded in a second segment (ctl.last_run.finalize_journal) whose hash is finalize_hash."""
        if self.last_run is None:
            raise RuntimeError("finalize() needs a completed run_problem()")
        out = self.last_run.finalize_post_run(artifact, expected_hash)
        self.finalize_hash = out["finalize_hash"]
        return out
