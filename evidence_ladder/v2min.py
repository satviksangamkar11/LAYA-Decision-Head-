"""V2-minimal (user GO 2026-10-21): fresh multi-sampling (the P5 arm-B path) + the trusted contract gate. No repair, no disagreement selector, no LAYA, no critic.

Two policies over the SAME sampling loop (a faithful copy of tools/el_p2_generation_compare.run_arm, "fresh" arm; tests/test_v2min.py proves call-for-call equality
when no gate is applied):
  "b_veto"  REFERENCE: run B unchanged; if B's selected candidate fails an authoritative contract, return no answer (veto).
  "gate"    PROPOSED: every candidate that passes all public examples is checked by the contract gate before acceptance; a proven failure is rejected and sampling
            continues; the first candidate passing every applicable check is returned at once (no further samples); budget exhausted -> abstain.
Contract gate verdicts: "pass" (authoritative contract checked >= 1 valid input, every valid input produced a result, no violation), "fail" (authoritative
violation: proven failure), "inconclusive" (authoritative contract but an incomplete check: no valid input checked or a valid input without a result; accepted on
public evidence, recorded as such), "unavailable" (no authoritative contract; public-only acceptance). A sandbox / tool failure raises ContractToolError: infrastructure, never a candidate verdict.
Labels: "SUPPORTED" (accepted; `checks` says whether the contract passed or was unavailable) or "BLOCKED" (no answer). Never "VERIFIED"."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evidence_ladder.common.interfaces import Artifact, GenerationRequest, TaskSpec
from evidence_ladder.controller import generation as GEN

VERSION = "v2min_v1.1"      # v1.1: incomplete contract checks are "inconclusive", never "pass"
MAX_NEW = 4096
POLICIES = ("b_veto", "gate")


class ContractToolError(RuntimeError):
    """The contract check could not run (sandbox / tool failure). Infrastructure: stop and report; never treated as a candidate verdict."""


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def public_result(outs: List[Dict[str, Any]], n: int) -> Dict[str, Any]:
    """Same rule as P2/P5 arm B: success only with parse pass and exactly one passing outcome for each example index 0..n-1."""
    ex = [o for o in outs if o.get("check_id") == "public_examples_v1" and "example_index" in o]
    parse = next((o for o in outs if o.get("check_id") == "parse_v1"), {})
    complete = sorted(o["example_index"] for o in ex) == list(range(n)) and n > 0
    ok = complete and parse.get("status") == "pass" and all(o.get("status") == "pass" for o in ex)
    return {"pass_all": ok, "complete": complete, "passed": sum(o.get("status") == "pass" for o in ex), "total": n}


class ContractGate:
    """The frozen V1.5 runtime check (controller.contract_check): contract edges + l2.derive_inputs(public, n_derived), contract_v1.evaluate.
    Authority comes only from the host registry or an allowlisted parser (contracts/registry.trusted_contract)."""

    def __init__(self, execution: Any, registry_path: Optional[Path], root: Path, parsers=("decl_v1",), n_derived: int = 6):
        from evidence_ladder.contracts import registry as CR
        self.execution, self.root, self.parsers, self.n_derived = execution, Path(root), list(parsers), n_derived
        self.registry = CR.load_registry(Path(registry_path)) if registry_path else None
        self._cache: Dict[str, Any] = {}

    def contract(self, task: TaskSpec):
        from evidence_ladder.contracts import registry as CR
        if task.task_id not in self._cache:
            self._cache[task.task_id] = CR.trusted_contract(task.task_id, self.registry, self.root, task.prompt, self.parsers)
        return self._cache[task.task_id]

    def __call__(self, task: TaskSpec, art: Artifact) -> str:
        from evidence_ladder.contracts import contract_v1 as CV
        from evidence_ladder.controller import l2
        ct = self.contract(task)
        if ct is None or not ct.authoritative:
            return "unavailable"
        inputs = list(ct.edges) + l2.derive_inputs(list(task.public_tests), n=self.n_derived)
        try:
            results, term = self.execution.run_inputs(art, task, inputs)
        except Exception as e:      # noqa: BLE001
            raise ContractToolError("contract check could not run for %s: %s: %s" % (task.task_id, type(e).__name__, str(e)[:200])) from e
        return verdict(CV.evaluate(ct, inputs, results, whole_run_crashed=(term == "crash")))


def verdict(ev: Dict[str, Any]) -> str:
    """contract_v1.evaluate result -> gate verdict. 'fail' = a proven violation; 'pass' only when the contract actually checked at least one valid input and
    every valid input produced a result; otherwise 'inconclusive' (incomplete check: not a failure, not a pass — the candidate is accepted on public evidence
    and the record says so)."""
    if ev["hard"]:
        return "fail"
    if ev["checked_valid_inputs"] == 0 or ev["uncertain"]:
        return "inconclusive"
    return "pass"


def _accept(gate: Optional[Callable[[TaskSpec, Artifact], str]], task: TaskSpec, art: Artifact, log: List[Dict[str, Any]]) -> Optional[str]:
    """None = accepted without a gate (B); else the gate verdict, logged. 'fail' means rejected."""
    if gate is None:
        return None
    v = gate(task, art)
    log.append({"candidate": art.artifact_id, "contract": v})
    return v


def sample_loop(gen: Any, run_public: Callable[[Artifact], List[Dict[str, Any]]], task: TaskSpec, budget: int, round_n: int = 2, n_r1: int = 4,
                strategies=None, gate: Optional[Callable[[TaskSpec, Artifact], str]] = None, log_call: Callable[[Dict[str, Any]], None] = lambda r: None,
                log_cand: Callable[[Dict[str, Any]], None] = lambda r: None) -> Dict[str, Any]:
    """Round 1 = GEN.plan_samples over seeds 1..n_r1 (as P5), then fresh rounds of round_n samples (seeds 100 + round_n*rnd + j, templates GEN.strategy_for),
    while no candidate is accepted and the generated tokens beyond round 1 stay below `budget`. With gate=None this is exactly arm B."""
    lang, strategies = task.language, tuple(strategies or GEN.FROZEN_STRATEGIES)
    hist: List[Dict[str, Any]] = []
    gate_log: List[Dict[str, Any]] = []
    r1_gen = r1_prompt = 0
    sel = None

    def consider(g, rnd, seed):
        nonlocal sel
        if g.extract_status != "ok" or not g.code.strip():
            return
        art = Artifact.make(task.task_id, lang, g.code, None, "v2min:r%d:s%d" % (rnd, seed))
        if any(h["id"] == art.artifact_id for h in hist):
            return
        pub = public_result(run_public(art), len(task.public_tests))
        h = {"id": art.artifact_id, "round": rnd, "seed": seed, "public": pub, "source": g.code, "contract": None}
        hist.append(h)
        log_cand({"task": task.task_id, "round": rnd, "seed": seed, "id": h["id"], "public_pass": pub["pass_all"], "source": g.code})
        if pub["pass_all"] and sel is None:
            v = _accept(gate, task, art, gate_log)
            h["contract"] = v
            if v != "fail":
                sel = h

    for rq in GEN.plan_samples(task, n_r1, list(range(1, n_r1 + 1)), 0, strategies, MAX_NEW):
        for g in gen.generate(rq):
            r1_gen += int(g.tokens_generated); r1_prompt += int(g.tokens_prompt)
            log_call({"task": task.task_id, "round": 1, "seed": g.seed, "tokens_generated": g.tokens_generated, "tokens_prompt": g.tokens_prompt})
            if sel is None:
                consider(g, 1, g.seed)
    used_gen = used_prompt = calls = 0
    rnd = 1
    while sel is None and used_gen < budget:
        rnd += 1
        for j in range(round_n):
            if sel is not None or used_gen >= budget:
                break
            seed = 100 + round_n * rnd + j
            template = GEN.strategy_for(n_r1 + (rnd - 2) * round_n + j, strategies)
            prompt = GEN.render_sample_prompt(template, task)[0]
            mx = min(MAX_NEW, budget - used_gen)
            g = list(gen.generate(GenerationRequest(task.task_id, prompt, 1, (seed,), mx, "sample", True, template)))[0]
            calls += 1
            used_gen += int(g.tokens_generated); used_prompt += int(g.tokens_prompt)
            log_call({"task": task.task_id, "round": rnd, "seed": seed, "tokens_generated": g.tokens_generated, "tokens_prompt": g.tokens_prompt, "max_new_tokens": mx})
            consider(g, rnd, seed)
    return {"selected": sel, "candidates": hist, "gate_log": gate_log, "rounds": rnd, "calls_after_r1": calls,
            "tokens_generated": r1_gen + used_gen, "tokens_prompt": r1_prompt + used_prompt, "tokens_generated_after_r1": used_gen, "budget": budget}


def solve(policy: str, gen: Any, run_public: Callable[[Artifact], List[Dict[str, Any]]], task: TaskSpec, budget: int,
          gate: Optional[Callable[[TaskSpec, Artifact], str]], **kw) -> Dict[str, Any]:
    """One task under a policy. Returns label, answer source (or None), the checks that admitted it, and the cost."""
    if policy not in POLICIES:
        raise ValueError("unknown policy %r" % policy)
    t0 = time.monotonic()
    if policy == "gate":
        r = sample_loop(gen, run_public, task, budget, gate=gate, **kw)
        sel = r["selected"]
        contract = sel["contract"] if sel else None
        vetoed = None
    else:
        r = sample_loop(gen, run_public, task, budget, gate=None, **kw)
        sel = r["selected"]
        contract = gate(task, Artifact.make(task.task_id, task.language, sel["source"])) if (sel and gate is not None) else ("unavailable" if sel else None)
        vetoed = sel["id"] if contract == "fail" else None
        if vetoed:
            sel = None
    return {"version": VERSION, "policy": policy, "task": task.task_id, "label": "SUPPORTED" if sel else "BLOCKED",
            "answer": sel["source"] if sel else None, "checks": {"public": "pass" if sel else None, "contract": contract if sel else None},
            "vetoed": vetoed, "rejected_by_contract": [g["candidate"] for g in r["gate_log"] if g["contract"] == "fail"],
            "candidates": len(r["candidates"]), "rounds": r["rounds"], "tokens_generated": r["tokens_generated"], "tokens_prompt": r["tokens_prompt"],
            "seconds": round(time.monotonic() - t0, 2)}
