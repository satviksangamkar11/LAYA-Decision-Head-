"""[EL-ARCH-1 T12 / T13 / T26] LOCAL harness: the REAL 0.6B model writes whole-program candidates and L2 checks for the three toy tasks (ex1_slug, ex2_clamp, ex3_wordcount), the EXISTING ladder controller
(controller/controller.py + l2.py) validates the checks (traceable span, non-vacuous on mutants, consistent with L3/L5), the candidates run in the real WSL sandbox and are clustered by behaviour (L1), and
when >= 2 clusters survive the controller's own discrimination path runs (blinded critic request through the existing critic interface). Nothing here approves anything: L2 tests, clusters and the critic
are ranking evidence only (invariants I1-I8 are checked on every journal and recorded).

Run (coordinator, RTX 3050, no network; the GPU is only used by the generation model):
    PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe tools/el_arch1_l2_local.py results/raw/el-arch1-l2-local-<UTC time> --final [--backbone qwen3-0.6b] [--critic-responses glm_responses.json]
Output dir (must not exist): RECEIPT.json (create-only; backend identity, per-task verdicts), <task>/journal.jsonl, <task>/result.json, glm/ (critic hand-off ledger). A second pass with
--critic-responses {packet_id or request_id: critic response} (made from the local GLM step) re-runs the same tasks with the same seeds so the pending critic packet can be applied.
Verdicts per task (honest, not failures of the harness): L2 = VALIDATED | INSUFFICIENT (+ reason: no valid test parsed / every test dropped / no reference candidate); clusters = n public / n extended;
discrimination = TRIGGERED (critic applied | pending | unavailable) | NOT_TRIGGERED (+ reason).
The script has no mock path: the generation backend is built here (HFQwenBackend, profile qwen3-0.6b, local=True, snapshot hashes verified exactly as tools/el_arch1_acceptance.py does) and
assert_real_backend() refuses anything else. Unit tests inject stubs through run_harness(final=False), whose receipt is labelled LOGIC_ONLY_STUB."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from laya_int1_t6 import TASKS                                                             # noqa: E402  task definitions only (same source as the acceptance driver)
from evidence_ladder.common.interfaces import Artifact, ControllerContext, TaskSpec, sha256_text   # noqa: E402
from evidence_ladder.controller import generation as GEN, invariants as INV, l2 as L2        # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController                    # noqa: E402
from evidence_ladder.models import qwen as Q                                                  # noqa: E402

EXAMPLES = [("ex1_slug", "slug"), ("ex2_clamp", "clamp"), ("ex3_wordcount", "wordcount")]
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
BACKBONE = "qwen3-0.6b"
EXPERIMENT_CHECK = "critic_experiment_v1"
MARK = "@@R@@"
RUNNER = '''import importlib.util, json, sys
spec = json.load(open("cases.json"))
try:
    s = importlib.util.spec_from_file_location("sol", "sol.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); f = m.solve
except BaseException as e:
    print("@@R@@" + json.dumps({"load_error": type(e).__name__ + ": " + str(e)[:300]})); sys.exit(0)
out = []
for inp in spec["inputs"]:
    try:
        v = f(json.loads(json.dumps(inp))); json.dumps(v); out.append({"ok": True, "value": v})
    except BaseException as e:
        out.append({"ok": False, "err": type(e).__name__ + ": " + str(e)[:200]})
print("@@R@@" + json.dumps({"results": out}))
'''


class StubBackendRefused(RuntimeError):
    """The final path accepts only the real local HFQwenBackend."""


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def jdump(o) -> str:
    return json.dumps(o, indent=1, ensure_ascii=False, default=str, sort_keys=True)


def write_new(path, text) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


# ------------------------------------------------------------------ tasks: function-level TaskSpec derived from the driver's task source
def derive_task(name: str, key: str, tasks: Optional[Dict[str, Any]] = None) -> TaskSpec:
    """TASKS[key] (buggy module + protected test file + user text) -> TaskSpec for the ladder: public examples are the asserts of the task's own test file (`assert f(args) == expected`),
    inputs are {parameter name: argument}. The protected test FILE is never given to the model; only its example values are, exactly as a public example."""
    T = (tasks or TASKS)[key]
    mod = next(n for n in T["files"] if not n.startswith("test_"))
    src, tst = T["files"][mod], next(c for n, c in T["files"].items() if n.startswith("test_"))
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))
    params = [a.arg for a in fn.args.args]
    tests = []
    for node in ast.walk(ast.parse(tst)):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and isinstance(node.test.ops[0], ast.Eq):
            call = node.test.left
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == fn.name:
                tests.append({"input": dict(zip(params, [ast.literal_eval(a) for a in call.args])), "expected": ast.literal_eval(node.test.comparators[0])})
    if not tests:
        raise ValueError("no `assert %s(...) == value` examples found for %s" % (fn.name, name))
    prompt = ("%s.\n\nImplement the CORRECT behaviour as `solve(data)`: `data` is a dict with the keys %s (the arguments of `%s(%s)`), and solve must return what the corrected `%s` returns.\n\n"
              "Current implementation (buggy):\n%s" % (T["user"].split(", so ")[0], ", ".join("`%s`" % p for p in params), fn.name, ", ".join(params), fn.name, src))
    return TaskSpec(name, "py1:" + name, "python", "python", prompt, tests)


# ------------------------------------------------------------------ execution backend: candidates run in the real WSL sandbox
class WslFunctionExec:
    """ExecutionBackend for function-level tasks on the qualified WSL sandbox (bubblewrap, no network). One sandbox call runs one source on a list of inputs; results are cached by
    (source, inputs), so a repeated probe costs nothing. A sandbox/infrastructure failure raises (the controller records a tool failure, never a candidate failure)."""

    def __init__(self, scratch: Path, sandbox=None, timeout: int = 30):
        self.scratch = Path(scratch)
        if sandbox is None:
            from evidence_ladder.controller.wsl_sandbox import WslSandbox
            sandbox = WslSandbox(self.scratch / "_sandbox", [])
        self.sandbox, self.timeout, self.calls, self._cache = sandbox, timeout, 0, {}

    def _execute(self, source: str, inputs: List[Any]) -> Dict[str, Any]:
        d = self.scratch / "runs" / uuid.uuid4().hex
        write_new(d / "sol.py", source)
        write_new(d / "runner.py", RUNNER)
        write_new(d / "cases.json", json.dumps({"inputs": inputs}, ensure_ascii=False))
        r = self.sandbox.run(d, "python3 runner.py", False, self.timeout)
        self.calls += 1
        if r.get("killed"):
            return {"timeout": True}
        line = next((l for l in reversed((r.get("stdout") or "").splitlines()) if l.startswith(MARK)), None)
        if line is None:
            return {"load_error": "no result marker (exit %s): %s" % (r.get("exit_code"), (r.get("stderr") or "")[-200:])}
        return json.loads(line[len(MARK):])

    def _run(self, source: str, inputs: List[Any]) -> Dict[str, Any]:
        key = sha(source) + sha(json.dumps(inputs, sort_keys=True, ensure_ascii=False))
        if key not in self._cache:
            self._cache[key] = self._execute(source, inputs)
        return self._cache[key]

    def run_public(self, artifact: Artifact, task: TaskSpec) -> List[Dict[str, Any]]:
        res = self._run(artifact.source, [t["input"] for t in task.public_tests])
        if res.get("timeout"):
            return [{"check_id": "parse_v1", "status": "timeout", "failure_class": "timeout"}] + [
                {"check_id": "public_examples_v1", "example_index": i, "status": "timeout", "failure_class": "timeout"} for i in range(len(task.public_tests))]
        if "load_error" in res:
            return [{"check_id": "parse_v1", "status": "fail", "failure_class": "compile_error", "detail": res["load_error"]}] + [
                {"check_id": "public_examples_v1", "example_index": i, "status": "missing_evidence", "failure_class": "compile_error", "observed": ""} for i in range(len(task.public_tests))]
        out: List[Dict[str, Any]] = [{"check_id": "parse_v1", "status": "pass"}]
        for i, (t, r) in enumerate(zip(task.public_tests, res["results"])):
            base = {"check_id": "public_examples_v1", "example_index": i}
            if r["ok"] and L2.canon(r["value"]) == L2.canon(t["expected"]):
                out.append(dict(base, status="pass", observed=r["value"]))
            else:
                obs = r["value"] if r["ok"] else r["err"]
                out.append(dict(base, status="fail", failure_class="wrong_answer" if r["ok"] else "runtime_error", observed=obs,
                                counterexample={"input": L2.canon(t["input"]), "expected": L2.canon(t["expected"]), "observed": L2.canon(obs)}))
        return out

    def run_experiment(self, artifact: Artifact, experiment: Dict[str, Any]) -> Dict[str, Any]:
        def bad(msg):
            return {"check_id": EXPERIMENT_CHECK, "status": "invalid_spec", "detail": msg, "failure_class": ""}
        if not isinstance(experiment, dict) or experiment.get("kind") not in ("stdin_input", "function_call"):
            return bad("unsupported experiment")
        try:
            obj = json.loads(experiment.get("payload"))
        except (TypeError, ValueError):
            return bad("payload is not valid JSON")
        if not isinstance(obj, dict):
            return bad("payload must be a JSON object")
        res = self._run(artifact.source, [obj])
        if res.get("timeout"):
            return {"check_id": EXPERIMENT_CHECK, "status": "timeout", "failure_class": "timeout"}
        if "load_error" in res:
            return {"check_id": EXPERIMENT_CHECK, "status": "error", "failure_class": "compile_error", "detail": res["load_error"]}
        r = res["results"][0]
        if r["ok"]:
            return {"check_id": EXPERIMENT_CHECK, "status": "pass", "observed": r["value"]}
        return {"check_id": EXPERIMENT_CHECK, "status": "error", "failure_class": "runtime_error", "detail": r["err"]}

    def backend_info(self) -> Dict[str, Any]:
        return {"backend": "wsl_function_exec", "distro": "Ubuntu-24.04", "sandbox": "bubblewrap, no network", "runner_sha256": sha(RUNNER), "calls": self.calls}


# ------------------------------------------------------------------ real backend + guard
def assert_real_backend(be, profile_id: str = BACKBONE) -> Dict[str, Any]:
    """-> the backend identity for the receipt, or StubBackendRefused. Only a real Q.HFQwenBackend in local mode on the named profile with a verified snapshot passes."""
    if not isinstance(be, Q.HFQwenBackend):
        raise StubBackendRefused("final mode needs the real HFQwenBackend, got %s" % type(be).__name__)
    info = be.model_info()
    if info.get("backend") != "hf" or not info.get("local") or info.get("profile_id") != profile_id or not info.get("snapshot_verified"):
        raise StubBackendRefused("backend identity is not the verified local %s snapshot: %s" % (profile_id, {k: info.get(k) for k in ("backend", "local", "profile_id", "snapshot_verified")}))
    return info


def build_real_backend(profile_id: str, out: Path) -> Q.HFQwenBackend:
    """Exactly as tools/el_arch1_acceptance.py worker_gen builds it (profile, local=True, pinned sha256 of model.safetensors + tokenizer.json, model loaded by profile_capture.load_model)."""
    from evidence_ladder.models import backbone as BB, profile_capture as PC
    os.environ["EVIDENCE_LADDER_ALLOW_INFERENCE"] = "1"
    P = BB.get(profile_id)
    man = json.loads((P.model_dir() / "MANIFEST_pinned.json").read_text("utf-8"))
    assert man["revision"] == P.revision, "model manifest revision differs from the profile"
    pins = {f: man["files"][f]["sha256"] for f in ("model.safetensors", "tokenizer.json")}
    model, tok, _dev = PC.load_model(P)
    be = Q.HFQwenBackend(authorized=True, profile=P, local=True, model_dir=str(P.model_dir()), log_path=str(Path(out) / "gen_log.jsonl"), verify_files=pins)
    be._verify_snapshot()
    be._model, be._tok = model, tok
    return be


def make_critic(out: Path):      # does not create out/; the broker creates its own ledger dir lazily
    """The existing critic hand-off (blinded packets, staged, no network from here). None, with the reason, when the GLM modules cannot be imported: the controller then journals the missing component."""
    try:
        from evidence_ladder.models.broker import StagedCriticBroker
        from evidence_ladder.models import glm, experiments as EXP
        cfg = glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=EXP.SUPPORTED_KINDS)
        return StagedCriticBroker(Path(out) / "glm" / "critic_batch" / "broker_ledger.jsonl", config=cfg, max_retries=1), None
    except Exception as e:      # noqa: BLE001
        return None, "%s: %s" % (type(e).__name__, str(e)[:200])


# ------------------------------------------------------------------ one task through the ladder
def controller_config(critic_responses: Optional[Dict[str, Any]] = None, max_new_tokens: int = 2048) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"mode": "ladder", "k": 4, "l2": True, "l2_n": 6, "discriminate": True, "critic": True, "strategies": list(GEN.FROZEN_STRATEGIES), "max_new_tokens": max_new_tokens,
                           "max_rounds": 2, "mutant_limit": 8}
    if critic_responses:
        cfg["critic_responses"] = dict(critic_responses)
    return cfg


def verdicts(res, journal) -> Dict[str, Any]:
    l2e = journal.entries("l2_checks")
    l2d = l2e[0]["data"] if l2e else None
    if l2d is None:
        l2v = {"verdict": "INSUFFICIENT", "reason": "L2 stage not reached (no candidate passed the public examples, so there was no trusted reference)"}
    elif l2d["validated"]:
        l2v = {"verdict": "VALIDATED", "validated": len(l2d["validated"]), "parsed": l2d["parsed"], "dropped": l2d["dropped"], "mutants": l2d["mutants"]}
    else:
        l2v = {"verdict": "INSUFFICIENT", "parsed": l2d["parsed"], "dropped": l2d["dropped"], "parse_rejected": l2d["parse_rejected"],
               "reason": "no valid test parsed from the model" if not l2d["parsed"] else "every parsed test was dropped by the L2 validation"}
    bc = journal.entries("behaviour_clusters")
    cl = {"computed": bool(bc), "public_clusters": bc[-1]["data"]["public_clusters"] if bc else None, "extended_clusters": bc[-1]["data"]["extended_clusters"] if bc else None,
          "members": bc[-1]["data"]["members"] if bc else None, "n_inputs": bc[-1]["data"]["n_inputs"] if bc else None}
    tr, cr = journal.entries("critic_trigger"), journal.entries("critic")
    if tr and tr[-1]["data"]["triggered"]:
        disc = {"verdict": "TRIGGERED", "critic": cr[-1]["data"]["state"] if cr else ("pending" if res.stop_details == "critic_pending" else "unknown"), "trigger": tr[-1]["data"]}
    else:
        disc = {"verdict": "NOT_TRIGGERED", "reason": "fewer than 2 behaviour clusters among live candidates" if bc or tr else "discrimination stage not reached (fewer than 2 live candidates)",
                "trigger": tr[-1]["data"] if tr else None}
    return {"l2": l2v, "clusters": cl, "discrimination": disc}


def run_task(name: str, key: str, gen, exe, critic, out: Path, cfg: Dict[str, Any], tasks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    task = derive_task(name, key, tasks)
    from evidence_ladder.models.qwen import TokenBudget
    ctl, t0 = EvidenceLadderController(), time.time()
    ctx = ControllerContext(gen, exe, TokenBudget(name, "D0", 60000), "D0", critic, None, SEEDS, dict(cfg))
    res = ctl.run_problem(task, ctx)
    j = ctl.last_journal
    try:
        inv = INV.assert_invariants(res, j, ControllerContext(None, None, None, "D0"))
        inv_rec = {"checked": list(inv), "failed": None}
    except Exception as e:      # noqa: BLE001
        inv_rec = {"checked": None, "failed": "%s: %s" % (type(e).__name__, str(e)[:300])}
    d = Path(out) / name
    write_new(d / "journal.jsonl", "".join(json.dumps(e, ensure_ascii=False, default=str, sort_keys=True) + "\n" for e in j.entries()))
    v = verdicts(res, j)
    ev = [e for e in j.evidence() if e.get("check_id") == "qwen_tests_v1"]
    rec = {"task": name, "task_prompt_sha256": sha256_text(task.prompt), "public_examples": len(task.public_tests), "stop_cause": res.stop_cause, "stop_details": res.stop_details,
           "answer_label": res.answer_label, "authorized_to_execute": bool(res.decision.get("authorized_to_execute")), "final_artifact_id": res.final_artifact_id,
           "journal_hash": res.journal_hash, "journal_chain_ok": j.verify_chain(), "n_journal_entries": len(j), "invariants": inv_rec, "seconds": round(time.time() - t0, 1),
           "l2_evidence": [{"candidate_id": e["candidate_id"], "status": e["status"], "independence": e["independence"], "requirement_id": e["requirement_id"]} for e in ev],
           "never_approved_by_l2_clusters_or_critic": res.answer_label != "VERIFIED" and not res.decision.get("authorized_to_execute"),
           "candidates": len([e for e in j.entries("candidate")]), "pending_critic_packets": (critic.export_pending() if critic is not None and hasattr(critic, "export_pending") else None), **v}
    write_new(d / "result.json", jdump(rec))
    return rec


def run_harness(out: Path, gen, exe, critic=None, *, final: bool, profile_id: str = BACKBONE, critic_responses: Optional[Dict[str, Any]] = None,
                critic_note: Optional[str] = None, tasks: Optional[Dict[str, Any]] = None, examples: Sequence[Tuple[str, str]] = tuple(EXAMPLES)) -> Dict[str, Any]:
    out = Path(out)
    if out.exists():
        raise FileExistsError("output dir exists (create-only): %s" % out)
    if final and not hasattr(gen, "generate") and not callable(gen):
        raise StubBackendRefused("final mode needs the real backend")
    if final and hasattr(gen, "generate"):
        assert_real_backend(gen, profile_id)        # refuse a stub BEFORE anything is written
    out.mkdir(parents=True)
    if not hasattr(gen, "generate"):
        gen = gen(out)                              # factory: the real backend writes its generation log inside out/
    ident = assert_real_backend(gen, profile_id) if final else {"backend": "STUB", "label": "LOGIC_ONLY_STUB", "info": getattr(gen, "model_info", lambda: {})()}
    cfg = controller_config(critic_responses)
    per, errors = {}, {}
    for name, key in examples:
        try:
            per[name] = run_task(name, key, gen, exe, critic, out, cfg, tasks)
        except Exception as e:      # noqa: BLE001  a crash on one task is a recorded result, not a hidden one
            errors[name] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300]), "trace": traceback.format_exc()[-1500:]}
    rec = {"harness": "el_arch1_l2_local", "tasks": per, "errors": errors, "stub": not final, "label": "REAL_LOCAL" if final else "LOGIC_ONLY_STUB", "backend": ident,
           "script_sha256": sha((ROOT / "tools" / "el_arch1_l2_local.py").read_bytes()), "controller_config": cfg, "critic": {"kind": type(critic).__name__ if critic is not None else None, "note": critic_note,
           "second_pass": bool(critic_responses)}, "seeds": list(SEEDS), "sandbox": exe.backend_info() if hasattr(exe, "backend_info") else None, "finished_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
           "summary": {n: {"l2": r["l2"]["verdict"], "clusters": r["clusters"]["extended_clusters"], "discrimination": r["discrimination"]["verdict"], "stop": r["stop_cause"],
                           "approved_by_model_evidence": not r["never_approved_by_l2_clusters_or_critic"], "invariants_failed": r["invariants"]["failed"]} for n, r in per.items()}}
    write_new(out / "RECEIPT.json", jdump(rec))
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="EL-ARCH-1 T12/T13/T26 local L2 / clustering / discrimination harness")
    ap.add_argument("out", help="output dir; must not exist")
    ap.add_argument("--final", action="store_true", help="marks the receipt as the final run; the real-backend guard is always on in this CLI")
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--critic-responses", help="JSON {packet_id or request_id: critic response} from the local GLM step (second pass)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        ap.error("output dir exists (create-only): %s" % out)
    from evidence_ladder.controller.wsl_sandbox import available
    if not available():
        ap.error("the WSL sandbox does not answer; refusing to run candidates anywhere else")
    resp = json.loads(Path(a.critic_responses).read_text("utf-8")) if a.critic_responses else None
    critic, note = make_critic(out)
    rec = run_harness(out, lambda o: build_real_backend(a.backbone, o), WslFunctionExec(out / "_work"), critic, final=True, profile_id=a.backbone, critic_responses=resp, critic_note=note)
    print(json.dumps(rec["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
