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
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evidence_ladder.common.contract_import import C
from evidence_ladder.common.interfaces import ProblemResult
from evidence_ladder.critic_batch import accounting as ACC
from evidence_ladder.models import experiments as EXP
from . import router as router_mod
from .components import MissingComponentError, require
from .decision import authorize, decide_stage1
from .evidence import Factory, trim
from .invariants import assert_invariants, check_i2
from .journal import EvidenceJournal, canon
from .limits import Limiter, load_limits
from .next_action import decide_next_action

TOOLS = {"Read": ("read_only", {"file_path"}), "Edit": ("reversible_write", {"file_path", "old_string", "new_string"}), "Bash": ("sandboxed_exec", {"command"})}
DENY = re.compile(r"\brm\s+-[a-z]*r|\bsudo\b|\bcurl\b|\bwget\b|\bssh\b|\bscp\b|\bnc\b|git\s+push|pip\s+install|mkfs|\bdd\s+if=|>\s*/dev/|chmod\s+777|(>|>>|tee|sed\s+-i\S*|mv|cp|rm)\b[^|;&]*\btest_\w+\.py", re.I)


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
    # config["require"] = ["scorer", "critic"]: components this run needs; a missing one writes a journal entry and raises MissingComponentError (A5). Final acceptance runs set it.


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


def check_defs() -> List[dict]:
    def chk(cid, kind, trust, covers, limitations):
        return {"check_id": cid, "version": "1", "code_hash": C.sha256(cid + "@1"), "kind": kind, "trust_class": trust, "covers": list(covers), "runtime_visible": True,
                "grader_only": False, "limitations": list(limitations)}
    return [chk("protected_tests_v1", "end_state", "authoritative", ["R1"], ["a single protected test file; passing it does not verify the task for all inputs"]),
            chk("protected_files_unchanged_v1", "certified_invariant", "authoritative", ["R2"], ["hash comparison inside the sandbox run; says nothing about behaviour"]),
            chk("action_dry_run_v1", "end_state", "authoritative", ["R1"], ["evidence about a HYPOTHETICAL end state of one candidate action, run in a scratch copy"]),
            chk("action_policy_v1", "parse", "rejecting_only", [], ["fixed tool table + regex deny-list; the sandbox is the real boundary"]),
            chk("critic_experiment_v1", "critic_experiment", "diagnostic", [], ["the critic's judgement is not truth"])]


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
        self.registry = C.Registry(check_defs())
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
        self.env = "env-" + C.sha256(canon({"sandbox": "wsl-bwrap-systemd", "test_cmd": task.test_cmd, "protected": task.protected}))[:12]
        q = task.test_cmd
        i = task.user_text.find(q)
        req1 = {"requirement_id": "R1", "version": 1, "kind": "explicit", "text": "the protected test command exits 0", "priority": "hard", "status": "active",
                "source": {"type": "request_span", "quote": q, "char_start": i, "char_end": i + len(q)}}
        req2 = {"requirement_id": "R2", "version": 1, "kind": "contract", "text": "the protected test files are not modified", "priority": "hard", "status": "active",
                "source": {"type": "task_metadata", "artifact_ref": ",".join(task.protected)}}
        self.reqs = [C.check_requirement(req1, task.user_text), C.check_requirement(req2, task.user_text)]
        self.roles = {"R1": "example", "R2": "contract"}
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

    def limit_check(self) -> Optional[Dict[str, Any]]:
        """[EL-ARCH-1 A3] None, or the exhausted resource {"resource","used","cap","details"}: wall time, Qwen tokens, GLM tokens (ledger-accounted), committed actions, rounds (steps)."""
        b = self.budget.record()
        return self.limiter.exceeded(b["qwen_gen_tokens"] + b["qwen_prefill_tokens"], b["glm_tokens_in"] + b["glm_tokens_out"], self.actions, self.step_no)

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

    def policy(self, act: Dict[str, Any]) -> Dict[str, str]:
        n, a = act["name"], act["arguments"]
        if n in ("Read", "Edit") and self._safe(self.task.workspace, a["file_path"]) is None:
            return {"status": "fail", "reason": "path outside the workspace"}
        if n == "Edit" and Path(a["file_path"]).name in self.task.protected:
            return {"status": "fail", "reason": "edit of a protected test file"}
        if n == "Bash" and DENY.search(a["command"]):
            return {"status": "fail", "reason": "command matches the deny-list"}
        return {"status": "pass", "reason": ""}

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

    def _evidence(self, cand: str, rid: str, check_id: str, source: str, independence: str, status: str, scope: str, outcome: Dict[str, Any], cx=None):
        ev = self.fac.make(_Art(cand), rid, check_id, source, independence, status, scope, outcome, cx)
        self.journal.add_evidence(ev, trim(dict(outcome, check_id=check_id, status=status)))
        return ev

    # ------------------------------------------------------------------ one step
    def step(self, options: Sequence[str], qwen_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        lim = self.limit_check()
        if lim is not None:
            raise LimitReached(lim["details"])
        self.step_no += 1
        k = self.step_no
        did = "%s:step%d" % (self.task.task_id, k)
        ctx_text = self.context_text()
        ws = self.task.workspace
        pre = tree_hash(ws)
        qm = qwen_meta or {}
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
                        r = self._run(scratch, self.task.test_cmd)
                        st = "pass" if (r["exit_code"] == 0 and r["protected_unchanged"]) else "fail"
                        out = {"check_id": "action_dry_run_v1", "status": st, "observed": ("exit %d\n%s%s" % (r["exit_code"], r["stdout"], r["stderr"]))[-600:], "detail": "edit applied in a scratch copy, protected test run"}
                        self._evidence(cid, "R1", "action_dry_run_v1", "environment_observation", "independent", st, "protected test after this single edit (scratch copy)", out,
                                       None if st == "pass" else {"input": self.task.test_cmd, "expected": "exit 0", "observed": out["observed"][-300:]})
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
            self.budget.add(scorer_calls=1)
            scores = [float(x) for x in self.ctx.scorer.score("next_action", {"context_text": ctx_text, "options": list(options)})]
            prov = getattr(getattr(self.ctx.scorer, "_backend", None), "last", None)
        except Exception as ex:      # noqa: BLE001  a missing/failed/refusing scorer is recorded, never swallowed
            reason = "%s: %s" % (type(ex).__name__, str(ex)[:250])
        # --- DECIDE
        dec = decide_next_action(self.journal, did, list(options), scores, trusted, scorer_id=getattr(self.ctx.scorer, "scorer_id", None),
                                 scorer_provenance=(prov or {}).get("provenance") if isinstance(prov, dict) else None, scorer_unavailable_reason=reason)
        i2 = check_i2(self.journal)
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
               "laya_top_was_chosen": (chosen == laya_order[0]) if laya_order and chosen is not None else None, "pre_state": pre}
        # --- COMMIT
        if chosen is None:
            result_text = "no allowed action"
            self.messages += [{"role": "assistant", "content": "(no allowed action)"}]
        else:
            act = acts[chosen]
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
        self.entries.append({"failure_sigs": [] if state["tests_pass"] else [sha(state["feedback"])[:16]], "progress": bool(state["tests_pass"]), "regression": False, "lineage": "workspace"})
        self.steps.append(rec)
        self.route("step%d_observed" % k, state["state_id"])
        return rec

    def _observe(self, tag: str) -> Dict[str, Any]:
        ws = self.task.workspace
        r = self._run(ws, self.task.test_cmd)
        state_id = tree_hash(ws)
        fb = "Harness test run (`%s`): exit %d\n%s%s" % (self.task.test_cmd, r["exit_code"], r["stdout"][-600:], r["stderr"][-600:])
        st1 = "pass" if r["exit_code"] == 0 else "fail"
        out1 = {"check_id": "protected_tests_v1", "status": st1, "observed": fb[-600:]}
        self._evidence(state_id, "R1", "protected_tests_v1", "user_test", "user_provided", st1, "the protected test command on the real end state", out1,
                       None if st1 == "pass" else {"input": self.task.test_cmd, "expected": "exit 0", "observed": fb[-300:]})
        st2 = "pass" if r["protected_unchanged"] else "fail"
        self._evidence(state_id, "R2", "protected_files_unchanged_v1", "certified_check", "independent", st2, "sha256 of the protected files before/after the test run", {"check_id": "protected_files_unchanged_v1", "status": st2})
        d = decide_stage1(self.reqs, self.journal.evidence(state_id), state_id, state_id, self.env, self.registry, "sandboxed", False, "commit")
        self.note("label", {"candidate_id": state_id, "tag": tag, "answer_label": d["answer_label"], "rows": d["requirement_rows"]})
        ok = authorize(self.reqs, self.journal.evidence(state_id), state_id, state_id, self.env, self.registry, "sandboxed", False)
        self.decided = ok
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
        lim = self.limiter.exceeded(0, bb["glm_tokens_in"] + bb["glm_tokens_out"], 0, 0)      # the critic is neither an action nor a round: only wall time and GLM tokens can stop it
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
    def finish(self) -> ProblemResult:
        if self.last_state is None:
            raise RuntimeError("no step ran")
        before = self.last_state
        fin = self._observe("finalize")        # stage 8: re-run the trusted checks on the REAL end state
        self.note("finalize", {"candidate_id": fin["state_id"], "ok": bool(fin["authorized"]) and fin["state_id"] == before, "reason": "" if fin["state_id"] == before else "state_changed", "answer_label": fin["label"]})
        sid = fin["state_id"]
        d = decide_stage1(self.reqs, self.journal.evidence(sid), sid, sid, self.env, self.registry, "sandboxed", False, "complete_program")
        cause = C.stop_cause(self.entries, len(self.steps), self.task.max_steps, decided=bool(fin["authorized"]))
        details = ""
        if cause is None:
            cause = "budget" if len(self.steps) >= self.task.max_steps else "no_progress_2"
        if cause == "budget" and len(self.steps) >= self.task.max_steps:
            details = "task.max_steps used=%d cap=%d" % (len(self.steps), self.task.max_steps)
        self.sync_glm_budget()
        self.budget.set(wall_seconds=round(self.limiter.elapsed(), 3))
        bb = self.budget.record()      # exhausted resources: a refused step/critic (limiter.hit) or wall time / tokens strictly above the cap (counters are only "exhausted" at the cap, not exceeded)
        lim = None if cause == "decided" else (self.limiter.hit or self.limiter.exceeded(bb["qwen_gen_tokens"] + bb["qwen_prefill_tokens"], bb["glm_tokens_in"] + bb["glm_tokens_out"], self.actions, self.step_no, at_cap_counts=True))
        if lim is not None and (cause == "no_progress_2" or details == ""):      # a per-problem limit is the more precise reason
            cause, details = "budget", lim["details"]
        if d["answer_label"] != "BLOCKED" and not d["authorized_to_execute"]:
            d = dict(d, next_step="report"); C.validate("decision_record", d)
        self.note("decision", d)
        stop = {"problem_id": self.task.task_id, "arm": self.ctx.arm, "cause": cause}
        if details.startswith("limit:"):
            stop["details"] = details.split(" used=")[0] if details.startswith("limit:max_wall") else details      # wall seconds are not journaled (reproducible hash)
        C.validate("stop_record", stop)
        self.note("stop", dict(stop, rounds=len(self.steps), max_rounds=self.task.max_steps, attempts=len(self.steps), max_attempts=self.task.max_steps, budget_hit=cause == "budget", wasted_critic=0, mode="agentic"))
        end = C.sha256(json.dumps(list(self.registry.checks.values()), sort_keys=True))
        self.note("registry", {"phase": "end", "hash": end, "scorer": self._scorer_fp()})
        C.assert_frozen(self.reg_hash0, end)
        result = ProblemResult(self.task.task_id, self.ctx.arm, sid, d["answer_label"], d, cause, [e["evidence_id"] for e in self.journal.evidence(sid)], self.budget.record(), len(self.steps),
                               self.journal.journal_hash(), None, details)
        self.invariants_checked = assert_invariants(result, self.journal, self.ctx, self.limiter.limits)
        return result
