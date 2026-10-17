"""[LAYA-INT-1] Per-action mode of the Evidence-Ladder controller (architecture R2.1: every action goes through intake -> check -> commit). Pure standard library + the
controller's own parts: Registry, EvidenceJournal (evidence admitted only through Registry.accept), Factory, router, decide_stage1/authorize, next_action ranking, assert_invariants.

One task = a small workspace with a PROTECTED test. Requirements: R1 (explicit, hard) the protected test command exits 0; R2 (contract, hard) the protected files are unchanged.
Per step:
  INTAKE   proposed options (Qwen, from the Kaggle T4 run) are parsed against a fixed tool table (Read / Edit / Bash; risk class per tool)
  CHECK    policy_v1 (rejecting only) + for an Edit an action_dry_run_v1 record: the edit is applied in a SCRATCH copy and the protected test is run in the sandbox
           (real execution, authoritative for R1 of THAT candidate action). dry-run pass -> trusted pass; dry-run fail -> -1 (stays allowed); policy fail -> rejected.
  SCORE    ctx.scorer.score("next_action", {context_text, options}) -> L0 scores; if it raises: journal "scorer_unavailable" with the reason (no silent fallback)
  DECIDE   next_action.decide_next_action: trusted evidence first, LAYA last, never approves; empty allowed set -> abstain
  COMMIT   the chosen action is applied to the workspace (Edit by the controller, Bash in the sandbox with persist)
  OBSERVE  the protected test is run on the REAL end state: evidence for R1 (protected_tests_v1, user_test) and R2 (protected_files_unchanged_v1, certified_check);
           decide_stage1 + authorize() give the label; VERIFIED + authorized -> stop 'decided'
finish() re-runs the trusted checks on the end state (finalize), writes decision/stop/registry entries and runs assert_invariants (I1-I8).
Deviations, stated: actions are applied to the (scratch) task workspace itself, not through C.commit_patch staging; the policy deny-list is a regex in addition to the sandbox boundary;
the critic stage is optional (critic_stage)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evidence_ladder.common.contract_import import C
from evidence_ladder.common.interfaces import ProblemResult
from evidence_ladder.critic_batch import accounting as ACC
from evidence_ladder.models import experiments as EXP
from . import candidate_filter as CF
from . import final_verify as FV
from . import investigate as INV
from . import risk as RISK
from . import router as router_mod
from . import tiers as TIERS
from .components import MissingComponentError, require
from .decision import authorize, blocked_without_candidate, decide_stage1
from .evidence import Factory, trim
from .invariants import assert_invariants, check_i2
from .journal import EvidenceJournal, canon
from .generation import render_call
from .limits import Limiter, load_limits, stop_label
from .next_action import decide_next_action
from .monotonicity import monotonicity_gate

TOOLS = {"Read": ("read_only", {"file_path"}), "Edit": ("reversible_write", {"file_path", "old_string", "new_string"}), "Bash": ("sandboxed_exec", {"command"})}
# [EL-ARCH-1 B5] the regex deny-list is gone: controller/risk.py holds the fixed action-risk table (authorize_action in policy() and at the commit gate)


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def tree_hash(root) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in Path(root).rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(str(p.relative_to(root)).replace("\\", "/").encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


@dataclass
class AgenticTask:
    task_id: str
    user_text: str
    workspace: Path
    test_cmd: str
    protected: List[str]
    system_text: str
    max_steps: int = 2
    scope: Optional[List[str]] = None      # [EL-ARCH-1 B7] optional fnmatch patterns the diff may touch; when set, R3 'the diff stays inside the declared scope' is a hard requirement


@dataclass
class AgenticContext:          # no field name may contain 'grader' or 'protected' (I8 scan)
    sandbox: Any
    scorer: Any = None
    critic: Any = None
    budget: Any = None
    arm: str = "E"
    seeds: Sequence[int] = ()
    config: Dict[str, Any] = field(default_factory=dict)
    limits: Any = None             # [EL-ARCH-1 A3] controller.limits.Limits; default = profile "agentic" of configs/el_arch1_limits_v1.toml
    clock: Any = None              # injected monotonic clock for the wall-time limit (default time.monotonic)
    comparison: Any = None         # [EL-ARCH-1 D1/S0] optional {"probe": scorer, "random_head": scorer} scoring the SAME options; journaled as scorer_comparison, never a selector
    # config["require"] = ["scorer", "critic"]: components this run needs; a missing one writes a journal entry and raises MissingComponentError (A5). Final acceptance runs set it.


class NoOracle(Exception):
    """[EL-ARCH-1 B1] the oracle-tier router found no executable test oracle for per-action mode (tier O2/O3): no action is taken; finish() reports BLOCKED (ask the user) or the tier's ceiling."""


class LimitReached(Exception):
    """A per-problem limit forbids another step. finish() still works and reports stop_cause "budget" with the exhausted resource in stop_details."""


class AgenticBudget:
    def __init__(self, problem_id: str, arm: str = "E"):
        self.r = dict(problem_id=problem_id, arm=arm, qwen_gen_tokens=0, qwen_prefill_tokens=0, glm_tokens_in=0, glm_tokens_out=0, glm_calls=0, scorer_calls=0, tool_runs=0,
                      gpu_seconds=0.0, wall_seconds=0.0, api_errors=0)

    def add(self, **kw):
        for k, v in kw.items():
            self.r[k] += v

    def set(self, **kw):
        for k, v in kw.items():
            self.r[k] = v

    def record(self):
        return dict(self.r)


class _Art:
    def __init__(self, artifact_id): self.artifact_id = artifact_id


def check_defs(with_scope: bool = False) -> List[dict]:
    def chk(cid, kind, trust, covers, limitations):
        return {"check_id": cid, "version": "1", "code_hash": C.sha256(cid + "@1"), "kind": kind, "trust_class": trust, "covers": list(covers), "runtime_visible": True,
                "grader_only": False, "limitations": list(limitations)}
    return [chk("protected_tests_v1", "end_state", "authoritative", ["R1"], ["a single protected test file; passing it does not verify the task for all inputs"]),
            chk("protected_files_unchanged_v1", "certified_invariant", "authoritative", ["R2"], ["hash comparison inside the sandbox run; says nothing about behaviour"]),
            chk("action_dry_run_v1", "end_state", "authoritative", ["R1"], ["evidence about a HYPOTHETICAL end state of one candidate action, run in a scratch copy"]),
            chk("action_policy_v1", "parse", "rejecting_only", [], ["fixed tool table + fixed action-risk table (controller/risk.py); the sandbox is the real boundary"]),
            chk("critic_experiment_v1", "critic_experiment", "diagnostic", [], ["the critic's judgement is not truth"])] + (
        [chk("diff_scope_v1", "certified_invariant", "authoritative", ["R3"], ["compares the file set / hashes of the end state with the intake snapshot against the declared scope patterns; says nothing about behaviour"])] if with_scope else [])


class AgenticRun:
    def __init__(self, task: AgenticTask, ctx: AgenticContext, render_context: Callable[[List[Dict[str, Any]]], str]):
        self.task, self.ctx, self.render = task, ctx, render_context
        require("sandbox backend", getattr(ctx, "sandbox", None), "every action is dry-run and every end state is checked in it")
        if not (task.protected and task.test_cmd):
            raise MissingComponentError("protected test", "the task names no protected test files / test command: nothing could ever verify the end state")
        self.limiter = Limiter(ctx.limits or load_limits("agentic"), ctx.clock or time.monotonic)
        self.limiter.start()
        self.actions = 0
        self.critic_pids: List[str] = []
        self.glm_accounting: Optional[Dict[str, Any]] = None
        self.registry = C.Registry(check_defs(with_scope=task.scope is not None))
        self.reg_hash0 = C.sha256(json.dumps(list(self.registry.checks.values()), sort_keys=True))
        self.journal = EvidenceJournal(self.registry)
        self.budget = ctx.budget or AgenticBudget(task.task_id, ctx.arm)
        ctx.budget = self.budget
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": task.system_text}, {"role": "user", "content": task.user_text}]
        self.steps: List[Dict[str, Any]] = []
        self.entries: List[Dict[str, Any]] = []
        self.step_no = 0
        self.critic_result: Optional[Dict[str, Any]] = None
        self.decided = False
        self.last_state: Optional[str] = None
        self.env = "env-" + C.sha256(canon({"sandbox": getattr(ctx.sandbox, "env_label", "wsl-bwrap-systemd"), "test_cmd": task.test_cmd, "protected": task.protected}))[:12]
        q = task.test_cmd
        i = task.user_text.find(q)
        req1 = {"requirement_id": "R1", "version": 1, "kind": "explicit", "text": "the protected test command exits 0", "priority": "hard", "status": "active",
                "source": {"type": "request_span", "quote": q, "char_start": i, "char_end": i + len(q)}}
        req2 = {"requirement_id": "R2", "version": 1, "kind": "contract", "text": "the protected test files are not modified", "priority": "hard", "status": "active",
                "source": {"type": "task_metadata", "artifact_ref": ",".join(task.protected)}}
        if i < 0:      # [D1/S2 finding] a test command the request does not quote cannot be an 'explicit' requirement (the quote check would demote it to a non-gating hypothesis and a failing test would no longer block VERIFIED): it is a harness-declared contract requirement instead
            req1 = dict(req1, kind="contract", source={"type": "task_metadata", "artifact_ref": q})
        self.reqs = [C.check_requirement(req1, task.user_text), C.check_requirement(req2, task.user_text)]
        self.roles = {"R1": "example", "R2": "contract"}
        if task.scope is not None:      # [EL-ARCH-1 B7] declared diff scope: a hard contract requirement with its own certified check
            req3 = {"requirement_id": "R3", "version": 1, "kind": "contract", "text": "the diff stays inside the declared scope", "priority": "hard", "status": "active",
                    "source": {"type": "task_metadata", "artifact_ref": ",".join(task.scope)}}
            self.reqs.append(C.check_requirement(req3, task.user_text))
            self.roles["R3"] = "contract"
        self.baseline = TIERS.snapshot(task.workspace)       # intake snapshot: the trusted baseline for tier facts (B1) and the diff scope (B7)
        self.tier_last: Optional[Dict[str, Any]] = None
        self.no_oracle: Optional[Dict[str, Any]] = None
        self.last_exec: Optional[Dict[str, Any]] = None
        self.investigation: Optional[Dict[str, Any]] = None
        self.denied = 0
        self.qwen_calls = 0      # [A2.5] proposal rounds handed to step(); model calls = qwen_calls + scorer calls + GLM calls
        self.attempts: Dict[str, Dict[str, Any]] = {}      # [A2.1] repeated-failure memory: key = sha(state hash, normalised action, observed failure) -> what happened
        self.fb_sig: Optional[str] = None      # signature of the last failing observation (None after a passing one)
        self.prev_failure: Optional[Dict[str, Any]] = None      # parsed failure of the last observation (progress / regression / repetition, A2.4)
        self.repeat_blocked = 0
        self.stop_label: Optional[str] = None
        self.tool_error: Optional[str] = None      # [D1/S2] the sandbox / test tool itself failed (raised): ERROR evidence, never a pass, stop cause tool_failure
        self.gate_pools: List[Dict[str, Any]] = []      # [D1/S0] this run's earlier decisions with TRUSTED outcomes (dry-run results); per run so a replay reproduces the journal exactly
        self.final_report: Optional[Dict[str, Any]] = None
        self.fac = Factory(self.registry, self.reqs, self.env, self.roles)

    # ------------------------------------------------------------------ journal helpers
    def note(self, kind: str, data: Any) -> None:
        self.journal.append(kind, data)

    def _missing(self, component: str, why: str) -> None:
        self.note("component_missing", {"component": component, "why": why, "arm": self.ctx.arm})
        raise MissingComponentError(component, why)

    def begin(self) -> None:
        need = set(self.ctx.config.get("require", ()))
        for comp, obj in (("scorer", self.ctx.scorer), ("critic", self.ctx.critic)):
            if obj is None and comp in need:
                self.note("run_start", {"task_id": self.task.task_id, "arm": self.ctx.arm, "mode": "agentic_per_action"})
                self._missing(comp, "config['require'] lists it")
        self.note("run_start", {"task_id": self.task.task_id, "arm": self.ctx.arm, "mode": "agentic_per_action", "seeds": [int(s) for s in self.ctx.seeds], "environment_revision": self.env,
                                "config_digest": C.sha256(canon(self.ctx.config, default=str))})
        self.note("registry", {"phase": "start", "hash": self.reg_hash0, "scorer": self._scorer_fp()})
        self.note("requirements", {"reqs": self.reqs, "request_sha": sha(self.task.user_text), "stop": None, "notes": []})
        if self.ctx.scorer is None and self.ctx.arm == "E":      # not required by this run: stated, not silent (each step also journals scorer_unavailable)
            self.note("component_missing", {"component": "scorer", "why": "arm E has no scorer in the context; steps fall back to the id order", "arm": self.ctx.arm, "waived": True})
        self.route("intake", None)
        self.note("risk_table", {"sha256": RISK.table_hash()})
        routed = self.retier("intake")
        if routed["tier"] != "O1":      # [EL-ARCH-1 B1] per-action mode needs an executable test oracle; O2 -> the ladder (examples) controller, O3 -> ask the user / best effort
            self.no_oracle = routed

    def retier(self, step: str) -> Dict[str, Any]:
        """[EL-ARCH-1 B1/R1.2] observe the facts again (only intake-time, unmodified files count) and journal oracle_tier when it changed."""
        t = self.task
        facts = TIERS.observe_facts(t.workspace, t.user_text, t.test_cmd, t.protected, self.baseline, self.ctx.config.get("compiler_cmd"), self.ctx.config.get("regression_cmd"))
        routed = TIERS.route_tier(facts, t.user_text, t.workspace)
        if TIERS.journal_tier(self.journal, step, routed, self.tier_last):
            self.tier_last = routed
        return self.tier_last or routed

    def limit_check(self) -> Optional[Dict[str, Any]]:
        """[EL-ARCH-1 A3] None, or the exhausted resource {"resource","used","cap","details"}: wall time, Qwen tokens, GLM tokens (ledger-accounted), committed actions, rounds (steps)."""
        if len(self.steps) >= self.task.max_steps:      # [A2.5] the task's own step cap is enforced, not only labelled: no step may start once max_steps steps ran
            r = {"resource": "max_steps", "used": len(self.steps), "cap": self.task.max_steps, "details": "task.max_steps used=%d cap=%d" % (len(self.steps), self.task.max_steps)}
            self.limiter.hit = self.limiter.hit or r
            return r
        b = self.budget.record()
        return self.limiter.exceeded(b["qwen_gen_tokens"] + b["qwen_prefill_tokens"], b["glm_tokens_in"] + b["glm_tokens_out"], self.actions, self.step_no, model_calls=self.model_calls())

    def model_calls(self) -> int:
        b = self.budget.record()
        return self.qwen_calls + b["scorer_calls"] + b["glm_calls"]

    def files_text(self) -> Dict[str, str]:
        """current workspace file texts for the candidate filter (what Edit applicability is judged against); binary / huge files are skipped"""
        out = {}
        ws = Path(self.task.workspace)
        for p in sorted(x for x in ws.rglob("*") if x.is_file() and "__pycache__" not in x.parts and ".git" not in x.parts):
            if p.stat().st_size <= 400000:
                try:
                    out[p.relative_to(ws).as_posix()] = p.read_text("utf-8")
                except (UnicodeDecodeError, OSError):
                    pass
        return out

    def filter_step(self, raw_options: Sequence[str], pre: str) -> Dict[str, Any]:
        """[A2.3 / frozen interface] ONE candidate_filter.filter_options call per decision, before scoring and ranking; `repeated_failure` rejections (A2.1) are added here in the same record
        shape; the whole FilterResult is journaled under kind "candidate_filter". Only the accepted options (canonical text, original order) go on."""
        cands = []
        for i, o in enumerate(raw_options):
            try:
                a = json.loads(o)
                ok = isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("arguments"), dict)
            except ValueError:
                ok = False
            cands.append({"tool": a["name"], "arguments": a["arguments"], "origin": {"seed": None, "index": i}, "raw": o} if ok else {"malformed": True, "origin": {"seed": None, "index": i}, "raw": o})
        fr = CF.filter_options(cands, self.files_text(), path_scope={"allow": None, "protected": list(self.task.protected)})
        kept, extra = [], []
        for a in fr["accepted"]:
            key = C.sha256(canon([pre, a["action_key"], self.fb_sig]))
            prev = self.attempts.get(key)
            if prev is not None:
                extra.append({"option_id": a["option_id"], "origin": a["origin"], "rule": "repeated_failure",
                              "reason": "the same action in the same state with the same observed failure already failed at step %d (failure %s)" % (prev["step"], prev["failure_sig"])})
            else:
                kept.append(a)
        if extra:
            self.repeat_blocked += len(extra)
            fr = dict(fr, accepted=kept, rejected=fr["rejected"] + extra)
            if not kept:
                codes: Dict[str, int] = {}
                for r in fr["rejected"]:
                    codes[r["rule"]] = codes.get(r["rule"], 0) + 1
                fr["abstain"] = {"code": "all_options_rejected", "reason": "no valid option after filtering: %d of %d options rejected (%s)" % (
                    len(fr["rejected"]), len(raw_options), ", ".join("%s=%d" % kv for kv in sorted(codes.items())))}
            fr["counts"] = {"in": len(raw_options), "accepted": len(kept), "rejected": len(fr["rejected"])}
        self.note("candidate_filter", dict(fr, step=self.step_no + 1))
        return fr

    def sync_glm_budget(self) -> Optional[Dict[str, Any]]:
        """[EL-ARCH-1 A1] glm_calls / glm_tokens_* / api_errors of the budget record are SET from the real broker + progress ledgers (never a default 0): calls = sends, tokens = provider usage or
        a flagged estimate. The per-packet detail (calls, retries, failed attempts, tokens, estimated) is journaled once as glm_accounting."""
        c = self.ctx.critic
        if c is None or not self.critic_pids:
            return None
        acc = ACC.from_broker(c, self.critic_pids)
        if acc is None:
            return None
        t = acc["totals"]
        self.budget.set(glm_calls=t["calls"], glm_tokens_in=t["tokens_in"], glm_tokens_out=t["tokens_out"], api_errors=t["failed_attempts"])
        if acc != self.glm_accounting:
            self.glm_accounting = acc
            self.note("glm_accounting", acc)
        return acc

    def _scorer_fp(self) -> Optional[str]:
        s = self.ctx.scorer
        if s is None:
            return None
        d = getattr(s, "scorer_def", None)
        return str(d["weights_hash"]) if d else str(getattr(s, "scorer_id", ""))

    def route(self, step: str, cand: Optional[str]) -> None:
        table = router_mod.route(self.reqs, self.roles, self.registry, self.journal.evidence(cand) if cand else [], cand)
        self.note("routing", {"step": step, "candidate_id": cand, "ceiling": router_mod.answer_ceiling(table), "table": table})

    def context_text(self) -> str:
        return self.render(self.messages)

    # ------------------------------------------------------------------ actions
    def parse(self, text: str):
        try:
            a = json.loads(text)
        except ValueError:
            return None, "not_json"
        if not (isinstance(a, dict) and a.get("name") in TOOLS and isinstance(a.get("arguments"), dict) and set(a["arguments"]) == TOOLS[a["name"]][1]
                and all(isinstance(v, str) for v in a["arguments"].values())):
            return None, "not_in_tool_table"
        return a, ""

    def _safe(self, root: Path, rel: str) -> Optional[Path]:
        if os.path.isabs(rel) or ".." in Path(rel).parts:
            return None
        p = (root / rel).resolve()
        return p if root.resolve() in p.parents else None

    def risk_of(self, act: Dict[str, Any], trusted_precheck: bool = False) -> Dict[str, Any]:
        return RISK.authorize_action(act, str(self.task.workspace), self.task.protected, bool(self.ctx.config.get("user_confirmed", False)), trusted_precheck)

    def policy(self, act: Dict[str, Any]) -> Dict[str, str]:
        """CHECK-stage policy from the fixed action-risk table: classes that can never be authorized fail; irreversible / outside-workspace / network classes fail unless the run was started
        with user_confirmed (the trusted pre-check is demanded again at the commit gate)."""
        r = self.risk_of(act, trusted_precheck=True)      # the pre-check is judged at commit; here only "never" classes and a missing user_confirmed are rejected
        if r["allowed"]:
            return {"status": "pass", "reason": ""}
        return {"status": "fail", "reason": "%s: %s (missing %s)" % (r["class"], r["reason"], ",".join(r["missing"]))}

    def apply_edit(self, root: Path, a: Dict[str, str]):
        p = self._safe(root, a["file_path"])
        if p is None or not p.is_file():
            return "error: no such file", False
        txt = p.read_text("utf-8")
        if txt.count(a["old_string"]) != 1:
            return "error: old_string must occur exactly once (found %d)" % txt.count(a["old_string"]), False
        p.write_text(txt.replace(a["old_string"], a["new_string"]), "utf-8")
        return "edited " + a["file_path"], True

    def _run(self, workspace, cmd, persist=False):
        self.budget.add(tool_runs=1)
        return self.ctx.sandbox.run(workspace, cmd, persist=persist)

    def _run_checked(self, workspace, cmd, tag: str):
        """[D1/S2] a trusted check whose tool breaks (the sandbox raises) is ERROR evidence, not a failure of the candidate and never a pass."""
        try:
            return self._run(workspace, cmd)
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as ex:      # infrastructure errors only (WslSandbox raises these); logic errors and ReplayDivergence still propagate
            self.tool_error = "%s: %s" % (type(ex).__name__, str(ex)[:200])
            self.note("tool_failure", {"tag": tag, "error": self.tool_error})
            return {"exit_code": -1, "killed": False, "stdout": "", "stderr": "sandbox/test tool error: " + self.tool_error, "protected_unchanged": False, "tool_error": True}

    def _evidence(self, cand: str, rid: str, check_id: str, source: str, independence: str, status: str, scope: str, outcome: Dict[str, Any], cx=None):
        ev = self.fac.make(_Art(cand), rid, check_id, source, independence, status, scope, outcome, cx)
        self.journal.add_evidence(ev, trim(dict(outcome, check_id=check_id, status=status)))
        return ev

    # ------------------------------------------------------------------ one step
    def step(self, options: Sequence[str], qwen_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.no_oracle is not None:
            raise NoOracle("tier %s: %s" % (self.no_oracle["tier"], self.no_oracle["reason"]))
        lim = self.limit_check()
        if lim is not None:
            raise LimitReached(lim["details"])
        k = self.step_no + 1
        ws = self.task.workspace
        pre = tree_hash(ws)
        fr = self.filter_step(options, pre)
        self.step_no += 1
        did = "%s:step%d" % (self.task.task_id, k)
        ctx_text = self.context_text()
        raw_sha = [sha(o) for o in options]
        options = [render_call(a["tool"], a["arguments"]) for a in fr["accepted"]]      # canonical text: option_sha256 == option_id of the frozen interface
        fb_before = self.fb_sig
        qm = qwen_meta or {}
        if qm:
            self.qwen_calls += 1
        self.budget.add(qwen_gen_tokens=int(qm.get("new_tokens", 0)), qwen_prefill_tokens=int(qm.get("prompt_tokens", 0)), gpu_seconds=float(qm.get("kaggle_seconds", 0.0)))
        # --- INTAKE + CHECK
        trusted, checks, acts = [], [], []
        for i, o in enumerate(options):
            act, err = self.parse(o)
            acts.append(act)
            if act is None:
                trusted.append({"policy": "fail"}); checks.append({"index": i, "policy": {"status": "fail", "reason": err}}); continue
            cid = C.sha256(canon({"action": act, "pre_state": pre}))
            pol = self.policy(act)
            self._evidence(cid, "R2", "action_policy_v1", "environment_observation", "independent", pol["status"], "policy check of one proposed action", {"check_id": "action_policy_v1", "status": pol["status"], "detail": pol["reason"]})
            t, rec = {"policy": pol["status"]}, {"index": i, "policy": pol, "action_class": TOOLS[act["name"]][0], "candidate_id": cid}
            if act["name"] == "Edit" and pol["status"] == "pass":
                scratch = Path(tempfile.mkdtemp(prefix="agentic-scratch-")) / "ws"
                shutil.copytree(ws, scratch)
                try:
                    msg, ok = self.apply_edit(scratch, act["arguments"])
                    if ok:
                        r = self._run_checked(scratch, self.task.test_cmd, "dry_run")
                        st = "error" if r.get("tool_error") else "pass" if (r["exit_code"] == 0 and r["protected_unchanged"]) else "fail"
                        out = {"check_id": "action_dry_run_v1", "status": st, "observed": ("exit %d\n%s%s" % (r["exit_code"], r["stdout"], r["stderr"]))[-600:], "detail": "edit applied in a scratch copy, protected test run"}
                        self._evidence(cid, "R1", "action_dry_run_v1", "environment_observation", "independent", st, "protected test after this single edit (scratch copy)", out,
                                       None if st == "pass" else {"input": self.task.test_cmd, "expected": "exit 0", "observed": out["observed"][-300:]})
                        if st != "error":      # an ERROR dry-run says nothing about the action: no trusted pass and no trusted fail
                            t["L5_dry_run"] = "pass" if st == "pass" else "fail_nonblocking"
                        rec["dry_run"] = {"status": st, "exit_code": r["exit_code"], "output_sha256": sha(out["observed"])}
                    else:
                        rec["dry_run"] = {"status": "not_applicable", "edit_error": msg}
                finally:
                    shutil.rmtree(scratch.parent, ignore_errors=True)
            trusted.append(t); checks.append(rec)
        self.route("step%d_checked" % k, None)
        # --- SCORE (no silent fallback)
        scores, reason, prov = None, None, None
        try:
            if not options:      # nothing to rank: no scorer call (no model call is spent on an empty pool); the decision abstains with the filter's code
                raise ValueError("no accepted option to score (%s)" % ((fr["abstain"] or {}).get("code", "empty")))
            self.budget.add(scorer_calls=1)
            scores = [float(x) for x in self.ctx.scorer.score("next_action", {"context_text": ctx_text, "options": list(options)})]
            if len(scores) != len(options) or not all(math.isfinite(x) for x in scores):      # [EL-ARCH-1 B4] a malformed hand-off is a recorded scorer failure, never a crash or a silent fallback
                raise ValueError("scorer returned %d score(s) for %d option(s), finite=%s" % (len(scores), len(options), all(math.isfinite(x) for x in scores)))
            prov = getattr(getattr(self.ctx.scorer, "_backend", None), "last", None)
        except Exception as ex:      # noqa: BLE001  a missing/failed/refusing scorer is recorded, never swallowed
            scores = None
            reason = "%s: %s" % (type(ex).__name__, str(ex)[:250])
        # --- COMPARISON scorers (probe / random head on the SAME options; journaled, never passed to the ranking)
        comp = None
        if scores is not None and self.ctx.comparison:
            comp = {}
            for kind, sc in sorted(self.ctx.comparison.items()):
                try:
                    v = [float(x) for x in sc.score("next_action", {"context_text": ctx_text, "options": list(options)})]
                    if len(v) != len(options) or not all(math.isfinite(x) for x in v):
                        raise ValueError("wrong number of scores or non-finite")
                    comp[kind] = v
                except Exception as ex:      # noqa: BLE001  a missing comparison score is recorded, the run goes on (it is not a selector)
                    self.note("scorer_comparison_unavailable", {"decision_id": did, "kind": kind, "reason": "%s: %s" % (type(ex).__name__, str(ex)[:200])})
            comp = comp or None
        # --- monotonicity gate (C3): journaled BEFORE the decision; with so few pools it is INSUFFICIENT_DATA, LAYA ordering of checks stays disabled and I2 is unchanged
        gate = monotonicity_gate(self.gate_pools)
        self.note("monotonicity_gate", {"decision_id": did, "status": gate["status"], "n_pools": gate["n_pools"], "n_informative_pools": gate["n_informative_pools"],
                                        "min_informative_pools": gate["min_informative_pools"], "reason": gate["reason"], "laya_ordering_enabled": gate["laya_ordering_enabled"],
                                        "can_override_trusted_evidence": gate["can_override_trusted_evidence"]})
        # --- DECIDE
        dec = decide_next_action(self.journal, did, list(options), scores, trusted, scorer_id=getattr(self.ctx.scorer, "scorer_id", None),
                                 scorer_provenance=(prov or {}).get("provenance") if isinstance(prov, dict) else None, scorer_unavailable_reason=reason, comparison_scores=comp,
                                 abstain_override=fr["abstain"])
        i2 = check_i2(self.journal)
        known = [i for i in range(len(options)) if "L5_dry_run" in trusted[i]]      # options whose outcome a real execution decided (an Edit that applied)
        if scores is not None and len(known) >= 2:
            self.gate_pools.append({"id": did, "scores": [scores[i] for i in known], "passed": [trusted[i]["L5_dry_run"] == "pass" for i in known]})
        n = len(options)
        allowed = [i for i in range(n) if not any(v == "fail" for v in trusted[i].values())]
        laya_order = sorted(range(n), key=lambda i: (-scores[i], i)) if scores else None
        id_choice = allowed[0] if allowed else None
        laya_choice_no_trusted = next((i for i in laya_order if i in allowed), None) if laya_order else None
        chosen = dec["selected_index"]
        rec = {"step": k, "decision_id": did, "context_sha256": sha(ctx_text), "options": list(options), "option_sha256": [sha(o) for o in options], "checks": checks,
               "trusted": trusted, "scores": scores, "scorer_unavailable_reason": reason, "laya_order": laya_order, "allowed_set": allowed, "id_order_choice": id_choice,
               "laya_choice_among_allowed_ignoring_trusted_evidence": laya_choice_no_trusted, "chosen_index": chosen, "decision": dec, "check_i2": i2 or "ok",
               "laya_changed_the_choice_vs_id_order": (chosen != id_choice) if scores else None,
               "trusted_evidence_overrode_laya": (chosen is not None and laya_choice_no_trusted is not None and chosen != laya_choice_no_trusted),
               "laya_top_was_chosen": (chosen == laya_order[0]) if laya_order and chosen is not None else None, "pre_state": pre,
               "filter": {"counts": fr["counts"], "rejected": fr["rejected"], "abstain": fr["abstain"], "raw_option_sha256": raw_sha}}
        # --- COMMIT
        if chosen is None:
            result_text = "no allowed action"
            self.messages += [{"role": "assistant", "content": "(no allowed action)"}]
        else:
            act = acts[chosen]
            cid_chosen = checks[chosen]["candidate_id"]
            pre_ok = authorize(self.reqs, self.journal.evidence(cid_chosen), cid_chosen, cid_chosen, self.env, self.registry, "irreversible", bool(self.ctx.config.get("user_confirmed", False)))
            gate = self.risk_of(act, trusted_precheck=pre_ok)
            self.note("action_auth", {"decision_id": did, "candidate_id": cid_chosen, "class": gate["class"], "allowed": gate["allowed"], "requires": gate["requires"], "missing": gate["missing"],
                                      "decision_risk": gate["decision_risk"]})
            rec["action_auth"] = gate
        if chosen is not None and not rec["action_auth"]["allowed"]:      # [EL-ARCH-1 B5] the gate in front of every execution; nothing below runs for a denied action
            self.denied += 1
            act = None
            result_text = "denied by the action-risk table: %s (missing %s)" % (rec["action_auth"]["class"], ",".join(rec["action_auth"]["missing"]))
            self.messages += [{"role": "assistant", "content": "(action denied)"}]
            rec["commit_denied"] = True
        elif chosen is not None:
            if act["name"] == "Read":
                p = self._safe(ws, act["arguments"]["file_path"])
                result_text = p.read_text("utf-8")[:1500] if p and p.is_file() else "error: no such file"
            elif act["name"] == "Edit":
                result_text = self.apply_edit(ws, act["arguments"])[0]
            else:
                r = self._run(ws, act["arguments"]["command"], persist=True)
                result_text = ("exit %d\n%s%s" % (r["exit_code"], r["stdout"], r["stderr"]))[:1500]
            cid_msg = "call_%s_%d" % (self.task.task_id, k)
            self.messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": cid_msg, "function": {"name": act["name"], "arguments": json.dumps(act["arguments"])}}]},
                              {"role": "tool", "tool_call_id": cid_msg, "content": result_text}]
            rec["executed_action"] = act
            self.actions += 1
        rec["execution_result"], rec["execution_result_sha256"] = result_text, sha(result_text)
        # --- OBSERVE on the real end state
        state = self._observe("step%d" % k)
        rec.update(state)
        self.messages += [{"role": "user", "content": state["feedback"]}]
        cls = self.progress_after(k, chosen, fr, pre, fb_before, rec, state)
        self.entries.append({"failure_sigs": [] if state["tests_pass"] else [sha(state["feedback"])[:16]], "progress": bool(state["tests_pass"]), "regression": cls == "regression", "lineage": "workspace"})
        self.steps.append(rec)
        # [EL-ARCH-1 B8] one immutable record per step: inputs (context + options), decision (chosen option), evidence (the entries above), action (as executed), outcome (result + end state)
        self.note("step_record", {"step": k, "decision_id": did, "inputs": {"context_sha256": sha(ctx_text), "option_sha256": [sha(o) for o in options], "pre_state": pre},
                                  "decision": {"chosen_index": chosen, "scorer_available": scores is not None, "allowed_set": allowed, "abstain": chosen is None, "filter_counts": fr["counts"]},
                                  "action": {"executed": rec.get("executed_action"), "class": (rec.get("action_auth") or {}).get("class"), "allowed": (rec.get("action_auth") or {}).get("allowed"),
                                             "denied": bool(rec.get("commit_denied"))},
                                  "outcome": {"result_sha256": rec["execution_result_sha256"], "post_state": state["state_id"], "tests_pass": state["tests_pass"], "label": state["label"],
                                              "authorized": state["authorized"], "protected_unchanged": state["protected_unchanged"]}})
        self.route("step%d_observed" % k, state["state_id"])
        self.retier("step%d" % k)
        return rec

    def progress_after(self, k: int, chosen, fr, pre: str, fb_before, rec, state) -> str:
        """[A2.1/A2.4] after the real end-state observation of step k: remember a failed action (repeat blocker), classify the failure against the previous observation
        (progress | repetition | regression | changed), and put a repair report with the REAL failure and the current code at the failing location into the next context. Same controller, same journal."""
        r = self.last_exec or {}
        out = (r.get("stdout", "") or "") + "\n" + (r.get("stderr", "") or "")
        files = sorted(self.files_text())
        failure = INV.parse_failure(out, int(r.get("exit_code", 1)), files, self.task.protected)
        cls = INV.classify_progress(self.prev_failure, failure, state["tests_pass"])
        self.prev_failure = failure
        if chosen is not None and not state["tests_pass"]:
            ak, rec_a = fr["accepted"][chosen]["action_key"], {"step": k, "failure_sig": sha(state["feedback"])[:16]}
            self.attempts[C.sha256(canon([pre, ak, fb_before]))] = rec_a
            if state["state_id"] == pre:      # the action changed nothing: the next proposal round sees this very state with the failure it just observed
                self.attempts[C.sha256(canon([pre, ak, self.fb_sig]))] = rec_a
        if state["tests_pass"]:
            return cls
        loc = failure.get("code_location") or failure.get("test_location")
        snip = ""
        if loc and not INV.is_hidden(loc["file"]) and loc["file"] not in {Path(x).as_posix() for x in self.task.protected} and (Path(self.task.workspace) / loc["file"]).is_file():
            snip = INV.snippet(Path(self.task.workspace) / loc["file"], loc["line"], INV.MAX_FILE_CHARS)[0]
        act = rec.get("executed_action")
        a_txt = render_call(act["name"], act["arguments"]) if act else "(none: %s)" % ("denied" if rec.get("commit_denied") else "abstained")
        text = INV.repair_text(k, a_txt, rec["execution_result"], failure, cls, self.repeat_blocked, snip)
        self.messages += [{"role": "user", "content": text}]
        self.note("repair_report", {"step": k, "classification": cls, "failure": {x: failure[x] for x in ("kind", "exception", "message", "test_location", "code_location")}, "failure_sig": sha(state["feedback"])[:16],
                                    "repeat_blocked_total": self.repeat_blocked, "snippet_sha256": sha(snip), "report_sha256": sha(text)})
        return cls

    def step_from_generation(self, proposer, seeds: Sequence[int] = (0,)) -> Dict[str, Any]:
        """[EL-ARCH-1 B3] ask a generation backend (via controller.generation.ActionProposer) for next-action proposals with the frozen action template, journal what came back (hashes, status,
        every rejection: prose is rejected, never repaired) and run one step on the parsed options. An empty proposal list is a step with no options: the decision abstains."""
        out = proposer.propose(self.task.task_id, self.step_no + 1, self.context_text(), seeds)
        self.journal_proposals(out)
        return self.step(out["options"], out["qwen_meta"])

    def journal_proposals(self, out: Dict[str, Any]) -> None:
        """[D1] the journal entry of one proposal round (split out of step_from_generation so RecordedRun can record and replay it as its own event)."""
        self.note("proposals", {"step": self.step_no + 1, "status": out["status"], "prompt_sha256": out["prompt_sha256"], "raw_sha256": out["raw_sha256"], "option_sha256": [sha(o) for o in out["options"]],
                                "rejected": out["rejected"], "generations": out["generations"]})

    def _observe(self, tag: str, detail_fn: Optional[Callable[[Dict[str, Any]], str]] = None) -> Dict[str, Any]:
        ws = self.task.workspace
        r = self._run_checked(ws, self.task.test_cmd, tag)
        self.last_exec = r
        state_id = tree_hash(ws)
        fb = "Harness test run (`%s`): exit %d\n%s%s" % (self.task.test_cmd, r["exit_code"], r["stdout"][-600:], r["stderr"][-600:])
        st1 = "error" if r.get("tool_error") else "pass" if r["exit_code"] == 0 else "fail"
        self.fb_sig = None if st1 == "pass" else sha(fb)[:16]
        out1 = {"check_id": "protected_tests_v1", "status": st1, "observed": fb[-600:]}
        if detail_fn is not None and st1 != "pass":
            out1["detail"] = detail_fn(r)      # B2: the parsed failing location is part of the L5 observation
        self._evidence(state_id, "R1", "protected_tests_v1", "user_test", "user_provided", st1, "the protected test command on the real end state", out1,
                       None if st1 == "pass" else {"input": self.task.test_cmd, "expected": "exit 0", "observed": fb[-300:]})
        st2 = "error" if r.get("tool_error") else "pass" if r["protected_unchanged"] else "fail"
        self._evidence(state_id, "R2", "protected_files_unchanged_v1", "certified_check", "independent", st2, "sha256 of the protected files before/after the test run", {"check_id": "protected_files_unchanged_v1", "status": st2})
        if self.task.scope is not None:      # [EL-ARCH-1 B7] R3: the end-state diff against the intake snapshot stays inside the declared scope
            ds = FV.diff_scope(self.baseline, TIERS.snapshot(ws), self.task.scope, self.task.protected)
            self._evidence(state_id, "R3", "diff_scope_v1", "certified_check", "independent", "pass" if ds["ok"] else "fail", "files added/changed/removed since intake vs the declared scope",
                           {"check_id": "diff_scope_v1", "status": "pass" if ds["ok"] else "fail", "detail": ",".join(ds["violations"])[:200]},
                           None if ds["ok"] else {"input": ",".join(self.task.scope), "expected": "changes inside scope", "observed": ",".join(ds["violations"])[:200]})
        d = decide_stage1(self.reqs, self.journal.evidence(state_id), state_id, state_id, self.env, self.registry, "sandboxed", False, "commit")
        self.note("label", {"candidate_id": state_id, "tag": tag, "answer_label": d["answer_label"], "rows": d["requirement_rows"]})
        ok = authorize(self.reqs, self.journal.evidence(state_id), state_id, state_id, self.env, self.registry, "sandboxed", False)
        self.decided = ok and not self.tool_error
        self.last_state = state_id
        return {"state_id": state_id, "tests_pass": r["exit_code"] == 0, "feedback": fb, "label": d["answer_label"], "authorized": ok, "protected_unchanged": r["protected_unchanged"]}

    # ------------------------------------------------------------------ critic (optional; diagnostic evidence only, never approves, never averaged with LAYA)
    def critic_request(self) -> Dict[str, Any]:
        """Blinded packet content for the CURRENT end state: the quoted requirement, the non-protected source files, the concrete test output. Never the protected test source,
        never LAYA scores or picks, never Qwen reasoning, never labels (I8)."""
        sid = self.last_state
        ws = self.task.workspace
        NL = chr(10)
        code = NL.join("### %s%s%s" % (p.relative_to(ws).as_posix(), NL, p.read_text("utf-8")) for p in sorted(x for x in ws.rglob("*") if x.is_file() and x.name not in self.task.protected and "__pycache__" not in x.parts))
        fb = next((m["content"] for m in reversed(self.messages) if m["role"] == "user" and str(m["content"]).startswith("Harness test run")), "")
        rid = "crit-" + C.sha256(self.task.task_id + "|" + sid)[:16]
        req = {"request_id": rid, "model": "z-ai/glm-5.3-flash", "decision_id": "d-" + rid, "candidate_ids": [sid],
               "requirement_quotes": [r["source"]["quote"] for r in self.reqs if r["kind"] == "explicit" and r["source"].get("quote")], "candidate_code": {sid: code},
               "observed_outputs": {sid: [{"input": self.task.test_cmd, "observed": fb[-500:], "status": "pass" if fb.splitlines()[0].endswith("exit 0") else "fail", "public_expected": "exit 0", "check_id": "task_tests"}]},      # packet label avoids the word 'protected' (critic_batch validator class protected_cases)
               "question": "challenge_requirement"}
        C.validate("critic_request", req)
        return req

    def critic_stage(self) -> str:
        """-> 'applied' | 'pending' | 'unavailable'. Needs ctx.critic (a CriticBroker). The response is an EXPERIMENT; the controller runs it in the sandbox on a scratch copy of the end state
        and records the outcome as diagnostic evidence (critic_experiment_v1, model_derived): it can never raise a label."""
        if self.ctx.critic is None:
            self.note("critic", {"state": "unavailable", "reason": "no critic broker in the context"})
            if "critic" in set(self.ctx.config.get("require", ())):
                raise MissingComponentError("critic", "config['require'] lists it")
            return "unavailable"
        bb = self.budget.record()
        lim = self.limiter.exceeded(0, bb["glm_tokens_in"] + bb["glm_tokens_out"], 0, 0, model_calls=self.model_calls())      # the critic is neither an action nor a round: only wall time and GLM tokens can stop it
        if lim is not None:        # no new critic work once wall time / GLM tokens are exhausted (the exhausted resource is reported by finish())
            self.note("critic", {"state": "unavailable", "packet_id": None, "reason": "limit:" + lim["resource"]}); return "unavailable"
        req = self.critic_request()
        try:
            packet = self.ctx.critic.build_packet(req)
        except Exception as ex:      # noqa: BLE001  e.g. CriticStopped after a payment/quota error
            self.note("critic", {"state": "unavailable", "packet_id": None, "reason": "%s: %s" % (type(ex).__name__, str(ex)[:200])}); return "unavailable"
        pid = packet["packet_id"]
        if pid not in self.critic_pids:
            self.critic_pids.append(pid)
        raw = self.ctx.critic.completed(pid)
        if raw is None:
            term = getattr(self.ctx.critic, "terminal_state", lambda _p: None)(pid)
            if term is None:
                self.note("critic", {"state": "pending", "packet_id": pid}); self.critic_result = {"state": "pending", "packet_id": pid}; return "pending"
            why = str((getattr(self.ctx.critic, "packets", {}).get(pid) or {}).get("reason", ""))      # e.g. malformed/unsupported_kind: rejected EXPLICITLY, never dropped
            st = "rejected" if term == "malformed" else "unavailable"
            self.critic_result = {"state": st, "packet_id": pid, "executed": False, "reason": term + ":" + why, "label_unchanged_by_critic": True}
            self.note("critic", dict(self.critic_result)); return st
        resp = raw.get("response") if isinstance(raw, dict) and "response" in raw and "experiment" not in raw else raw
        exp = (resp or {}).get("experiment") or {}
        try:
            plan = EXP.plan(exp, self.task.workspace, self.task.protected, self.task.test_cmd)
        except EXP.Rejected as rj:
            self.critic_result = {"state": "rejected", "packet_id": pid, "executed": False, "kind": exp.get("kind"), "reason": "%s: %s" % (rj.code, rj.detail), "reason_code": rj.code,
                                  "payload_sha256": sha(str(exp.get("payload", ""))), "label_unchanged_by_critic": True}
            self.note("critic", dict(self.critic_result)); return "rejected"
        scratch = Path(tempfile.mkdtemp(prefix="agentic-critic-")) / "ws"
        shutil.copytree(self.task.workspace, scratch)
        try:
            r = self._run(scratch, plan["cmd"])
        finally:
            shutil.rmtree(scratch.parent, ignore_errors=True)
        st, observed = EXP.judge((exp.get("expected_by_candidate") or {}).get(self.last_state), r)
        ev = self._evidence(self.last_state, "R1", "critic_experiment_v1", "critic_experiment", "model_derived", st, "critic experiment on a scratch copy of the end state (diagnostic)",
                       {"check_id": "critic_experiment_v1", "status": st, "observed": observed, "detail": str((resp or {}).get("implications", ""))[:300]})
        self.critic_result = {"state": "applied", "candidate_id": self.last_state, "packet_id": pid, "executed": True, "kind": plan["kind"], "experiment": plan["display"], "command": plan["cmd"],
                              "experiment_status": st, "evidence_status": ev["status"], "observed": observed, "observed_sha256": sha(observed), "protected_unchanged": r["protected_unchanged"], "label_unchanged_by_critic": True}
        self.note("critic", {k: v for k, v in self.critic_result.items() if k not in ("observed", "command")})
        return "applied"

    # ------------------------------------------------------------------ finish
    def _finish_no_oracle(self) -> ProblemResult:
        """[EL-ARCH-1 B1] tier O2/O3: no action was taken. BLOCKED, with the ask-user questions (O3) or the pointer to the ladder controller (O2) in the stop details."""
        t = self.no_oracle
        d = blocked_without_candidate(self.reqs, "d-tier-" + self.task.task_id)
        details = ("tier %s: ask_user: %s" % (t["tier"], "; ".join(t["ambiguities"]) or "no blocking ambiguity, best effort only") if t["tier"] == "O3"
                   else "tier %s: no executable test oracle for per-action mode; use the ladder (examples) controller" % t["tier"])
        self.note("decision", d)
        self.sync_glm_budget()
        self.budget.set(wall_seconds=round(self.limiter.elapsed(), 3))
        stop = {"problem_id": self.task.task_id, "arm": self.ctx.arm, "cause": "missing_information", "details": details[:300]}
        C.validate("stop_record", stop)
        self.note("stop", dict(stop, rounds=0, max_rounds=self.task.max_steps, attempts=0, max_attempts=self.task.max_steps, budget_hit=False, wasted_critic=0, mode="agentic"))
        end = C.sha256(json.dumps(list(self.registry.checks.values()), sort_keys=True))
        self.note("registry", {"phase": "end", "hash": end, "scorer": self._scorer_fp()})
        C.assert_frozen(self.reg_hash0, end)
        result = ProblemResult(self.task.task_id, self.ctx.arm, None, d["answer_label"], d, "missing_information", [], self.budget.record(), 0, self.journal.journal_hash(), None, details[:300])
        self.invariants_checked = assert_invariants(result, self.journal, self.ctx, self.limiter.limits)
        return result

    def finish(self) -> ProblemResult:
        if self.last_state is None and self.no_oracle is not None:
            return self._finish_no_oracle()
        if self.last_state is None:
            raise RuntimeError("no step ran")
        before = self.last_state
        fin = self._observe("finalize")        # stage 8: re-run the trusted checks on the REAL end state
        self.note("finalize", {"candidate_id": fin["state_id"], "ok": bool(fin["authorized"]) and fin["state_id"] == before, "reason": "" if fin["state_id"] == before else "state_changed", "answer_label": fin["label"]})
        sid = fin["state_id"]
        d = decide_stage1(self.reqs, self.journal.evidence(sid), sid, sid, self.env, self.registry, "sandboxed", False, "complete_program")
        self.final_report = FV.final_report(self, fin, d, before)      # [EL-ARCH-1 B7] outcome and process fields of the ACTUAL end state
        self.note("final_report", self.final_report)
        cause = C.stop_cause(self.entries, len(self.steps), self.task.max_steps, decided=bool(fin["authorized"]))
        details = ""
        if cause is None:
            cause = "budget" if len(self.steps) >= self.task.max_steps else "no_progress_2"
        if self.tool_error and cause != "decided":      # [D1/S2] a broken test tool is its own stop cause, not a budget or progress verdict
            cause, details = "tool_failure", "sandbox"
        if cause == "budget" and len(self.steps) >= self.task.max_steps:
            details = "task.max_steps used=%d cap=%d" % (len(self.steps), self.task.max_steps)
        self.sync_glm_budget()
        self.budget.set(wall_seconds=round(self.limiter.elapsed(), 3))
        bb = self.budget.record()      # exhausted resources: a refused step/critic (limiter.hit) or wall time / tokens strictly above the cap (counters are only "exhausted" at the cap, not exceeded)
        lim = None if cause == "decided" else (self.limiter.hit or self.limiter.exceeded(bb["qwen_gen_tokens"] + bb["qwen_prefill_tokens"], bb["glm_tokens_in"] + bb["glm_tokens_out"], self.actions, self.step_no, at_cap_counts=True, model_calls=self.model_calls()))
        if lim is not None and (cause == "no_progress_2" or details == ""):      # a per-problem limit is the more precise reason
            cause, details = "budget", lim["details"]
        if d["answer_label"] != "BLOCKED" and not d["authorized_to_execute"]:
            d = dict(d, next_step="report"); C.validate("decision_record", d)
        self.note("decision", d)
        stop = {"problem_id": self.task.task_id, "arm": self.ctx.arm, "cause": cause}
        if details.startswith("limit:"):
            stop["details"] = details.split(" used=")[0] if details.startswith("limit:max_wall") else details      # wall seconds are not journaled (reproducible hash)
        C.validate("stop_record", stop)
        self.stop_label = stop_label(details) if cause == "budget" else None      # [A2.5] names the exhausted limit; the frozen stop_record cause stays "budget"
        self.note("stop", dict(stop, rounds=len(self.steps), max_rounds=self.task.max_steps, attempts=len(self.steps), max_attempts=self.task.max_steps, budget_hit=cause == "budget", wasted_critic=0, mode="agentic",
                               stop_label=self.stop_label))
        end = C.sha256(json.dumps(list(self.registry.checks.values()), sort_keys=True))
        self.note("registry", {"phase": "end", "hash": end, "scorer": self._scorer_fp()})
        C.assert_frozen(self.reg_hash0, end)
        result = ProblemResult(self.task.task_id, self.ctx.arm, sid, d["answer_label"], d, cause, [e["evidence_id"] for e in self.journal.evidence(sid)], self.budget.record(), len(self.steps),
                               self.journal.journal_hash(), None, details)
        self.invariants_checked = assert_invariants(result, self.journal, self.ctx, self.limiter.limits)
        return result
