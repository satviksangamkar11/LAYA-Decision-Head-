"""[EL-ARCH-1 F2d / replay] Offline replay check of a finished (or partial) bank run: rebuild each task's controller decisions from its SAVED artifacts and show identical decisions / journal hashes.
No sandbox, no model, no network, no GPU.

    python tools/el_bank_replay_check.py <run dir> [--public-tasks <bank40_tasks_public.jsonl>] [--require-full] [--no-write] [--out DIR]

LEVEL A (every executed task, journal only): the hash chain is recomputed entry by entry and its head must equal result.journal_hash; every `label` entry is recomputed from the evidence entries before it
with the controller's own decision core (decision.decide_stage1) and must equal the journaled answer label and per-requirement rows; every `ranking` entry is recomputed with the contract's C.rank and must
equal the journaled order; the final `decision` entry is recomputed from all evidence of its candidate and must equal the journaled label and rows.
LEVEL B (full controller replay, when the task directory has what it needs): the SAME EvidenceLadderController is run again on the saved task with the saved generations (gen_calls.json, served by
bank40.SequenceReplayGen, which refuses a request that differs in purpose / n / seeds / prompt hash), the saved execution results (exec_calls.jsonl, one line per ExecutionBackend call exactly as returned;
served by RecordedCalls below) and the saved configuration (result.json controller_config, backend_info, token_cap, wall_cap_s, seeds from the journal), with a staged critic broker (no call is ever made).
It must give the identical journal hash (and so identical decisions, stop cause and invariants: assert_invariants runs inside the controller). Not attempted (explicit reason, never silent) when a needed
file or key is absent. `--require-full` turns NOT_ATTEMPTED into a failure.
Per task REPLAY_MATCH | REPLAY_DIVERGED(first diverging entry) | A-only (level B not attempted: reason). Run: REPLAY_OK | REPLAY_DIVERGED | REPLAY_PARTIAL (level B missing on some tasks, --require-full not given).
Also importable: record_exec(inner, path) wraps an ExecutionBackend and writes exec_calls.jsonl in the shape level B reads."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GENESIS = hashlib.sha256("evidence-journal-v1".encode("utf-8")).hexdigest()


def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ recording / serving of execution calls
class record_exec:
    """wrap an ExecutionBackend: every call is appended to `path` as {"method","artifact_id","experiment","result"} exactly as returned (same shape as the runner's ExecCallRecorder)."""

    def __init__(self, inner, path):
        self.inner, self.path = inner, Path(path)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _append(self, method, artifact, experiment, result):
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"method": method, "artifact_id": artifact.artifact_id, "experiment": experiment, "result": result}, ensure_ascii=False, default=str) + "\n")

    def run_public(self, artifact, task):
        r = self.inner.run_public(artifact, task)
        self._append("run_public", artifact, None, r)
        return r

    def run_experiment(self, artifact, experiment):
        r = self.inner.run_experiment(artifact, experiment)
        self._append("run_experiment", artifact, experiment, r)
        return r


class UnrecordedCall(KeyError):
    pass


class RecordedCalls:
    """ExecutionBackend served from exec_calls.jsonl. A call that was never recorded raises UnrecordedCall: nothing is invented and no sandbox exists here."""

    def __init__(self, lines: Sequence[Dict[str, Any]], info: Dict[str, Any]):
        self._info = copy.deepcopy(info)
        self._q: Dict[str, List[Any]] = {}
        self._i: Dict[str, int] = {}
        for l in lines:
            self._q.setdefault(self._key(l["method"], l["artifact_id"], l.get("experiment")), []).append(l["result"])

    @staticmethod
    def _key(method, artifact_id, experiment) -> str:
        return "%s|%s|%s" % (method, artifact_id, _canon(experiment) if experiment is not None else "")

    def _serve(self, key: str):
        if key not in self._q:
            raise UnrecordedCall("no recorded execution for %s" % key[:90])
        i = self._i.get(key, 0)
        self._i[key] = i + 1
        return copy.deepcopy(self._q[key][min(i, len(self._q[key]) - 1)])

    def run_public(self, artifact, task):
        return self._serve(self._key("run_public", artifact.artifact_id, None))

    def run_experiment(self, artifact, experiment):
        return self._serve(self._key("run_experiment", artifact.artifact_id, experiment))

    def backend_info(self):
        return copy.deepcopy(self._info)


# ------------------------------------------------------------------ level A
def _read_entries(task_dir: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in (task_dir / "journal.jsonl").read_text("utf-8").splitlines() if l.strip()]


def level_a(task_dir: Path) -> Dict[str, Any]:
    from evidence_ladder.common.contract_import import C
    from evidence_ladder.controller.decision import decide_stage1
    from evidence_ladder.controller.registry import build_registry
    out: Dict[str, Any] = {"status": "REPLAY_MATCH", "diffs": [], "checked": {"labels": 0, "rankings": 0, "decision": 0}}
    try:
        entries = _read_entries(task_dir)
        result = json.loads((task_dir / "result.json").read_text("utf-8"))
    except (OSError, ValueError) as ex:
        return {"status": "REPLAY_DIVERGED", "diffs": ["unreadable journal/result: %s" % ex], "checked": {}}
    h = GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i or e.get("prev") != h:
            out["diffs"].append("hash chain broken at entry %d (%s)" % (i, e.get("kind")))
            break
        h = _sha(_canon(e))
    else:
        if result.get("journal_hash") and h != result["journal_hash"]:
            out["diffs"].append("journal head %s != result.journal_hash %s" % (h[:12], str(result["journal_hash"])[:12]))
    reqs = next((e["data"]["reqs"] for e in entries if e["kind"] == "requirements"), None)
    env = next((e["data"]["environment_revision"] for e in entries if e["kind"] == "run_start"), None)
    if reqs is None or env is None:
        out["diffs"].append("requirements / run_start entry missing")
    else:
        ex_ids = sorted({e["data"]["evidence"]["requirement_id"] for e in entries if e["kind"] == "evidence" and e["data"]["evidence"]["check_id"] == "public_examples_v1"
                         and e["data"]["evidence"]["source"] == "user_test"})      # the example rows (intake.example_ids): the all-inputs row's independent record does not make it an example
        reg = build_registry(ex_ids)
        risk = "sandboxed"
        seen: Dict[str, List[Dict[str, Any]]] = {}
        for e in entries:
            k, d = e["kind"], e["data"]
            if k == "evidence":
                seen.setdefault(d["evidence"]["candidate_id"], []).append(d["evidence"])
            elif k == "label":
                cid = d["candidate_id"]
                dec = decide_stage1(reqs, copy.deepcopy(seen.get(cid, [])), cid, cid, env, reg, risk, False, "complete_program")
                rows = {r["requirement_id"]: r["label"] for r in dec["requirement_rows"]}
                out["checked"]["labels"] += 1
                if dec["answer_label"] != d["answer_label"] or rows != d["rows"]:
                    out["diffs"].append("label entry %d (%s): journaled %s %s, recomputed %s %s" % (e["seq"], cid[:10], d["answer_label"], d["rows"], dec["answer_label"], rows))
            elif k == "ranking":
                out["checked"]["rankings"] += 1
                if [x["id"] for x in C.rank(copy.deepcopy(d["items"]))] != d["order"]:
                    out["diffs"].append("ranking entry %d: C.rank gives a different order" % e["seq"])
            elif k == "decision" and d.get("candidate_id") not in (None, "none") and d["candidate_id"] in seen:
                cid = d["candidate_id"]
                dec = decide_stage1(reqs, copy.deepcopy(seen[cid]), cid, cid, env, reg, risk, False, "complete_program")
                out["checked"]["decision"] += 1
                if dec["answer_label"] != d["answer_label"] or dec["requirement_rows"] != d["requirement_rows"]:
                    out["diffs"].append("decision entry %d: journaled %s, recomputed %s" % (e["seq"], d["answer_label"], dec["answer_label"]))
    if out["diffs"]:
        out["status"] = "REPLAY_DIVERGED"
    return out


# ------------------------------------------------------------------ level B
def level_b(task_dir: Path, summary: Dict[str, Any], public: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    tid = task_dir.name
    try:
        result = json.loads((task_dir / "result.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {"status": "NOT_ATTEMPTED", "reason": "no result.json"}
    need = []
    cfg = result.get("controller_config")
    if cfg is None:
        need.append("result.controller_config")
    info = result.get("backend_info")
    if info is None:
        need.append("result.backend_info (needed to reproduce environment_revision)")
    if not (task_dir / "exec_calls.jsonl").exists():
        need.append("exec_calls.jsonl")
    if not (task_dir / "gen_calls.json").exists():
        need.append("gen_calls.json")
    if tid not in public:
        need.append("public task record (--public-tasks)")
    if need:
        return {"status": "NOT_ATTEMPTED", "reason": "missing: " + ", ".join(need)}
    from evidence_ladder import bank40 as B
    from evidence_ladder.common.interfaces import ControllerContext, TaskSpec
    from evidence_ladder.controller.controller import EvidenceLadderController
    from evidence_ladder.controller.limits import load_limits
    from evidence_ladder.models import experiments as EXP, glm, qwen as Q
    from evidence_ladder.models.broker import StagedCriticBroker
    entries = _read_entries(task_dir)
    rs = next(e["data"] for e in entries if e["kind"] == "run_start")
    rec_in = public[tid]
    task = TaskSpec(tid, ("py1:" if rec_in["language"] == "python" else "java1:") + tid, rec_in["language"], rec_in["language"], rec_in["prompt"], rec_in["public_tests"], rec_in.get("difficulty", ""))
    lines = [json.loads(l) for l in (task_dir / "exec_calls.jsonl").read_text("utf-8").splitlines() if l.strip()]
    gen = B.SequenceReplayGen(json.loads((task_dir / "gen_calls.json").read_text("utf-8")))
    exe = RecordedCalls(lines, info)
    tmp = Path(tempfile.mkdtemp(prefix="replay-"))
    critic = StagedCriticBroker(tmp / "ledger.jsonl", config=glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=EXP.SUPPORTED_KINDS), max_retries=1) if cfg.get("critic") else None
    cap = int(result.get("token_cap") or summary.get("token_cap") or 10 ** 9)
    ctx = ControllerContext(gen, exe, Q.TokenBudget(tid, rs["arm"], cap), rs["arm"], critic, None, tuple(int(s) for s in rs["seeds"]), copy.deepcopy(cfg))
    limits = dataclasses.replace(load_limits("ladder"), max_wall_seconds=float(result.get("wall_cap_s") or 10 ** 6))
    ctl = EvidenceLadderController(limits=limits)
    try:
        res = ctl.run_problem(task, ctx)
    except (B.ReplayMismatch, UnrecordedCall) as ex:
        return {"status": "REPLAY_DIVERGED", "reason": "%s: %s" % (type(ex).__name__, str(ex)[:200])}
    want = result.get("journal_hash")
    got = ctl.last_journal.journal_hash()
    out: Dict[str, Any] = {"status": "REPLAY_MATCH" if got == want else "REPLAY_DIVERGED", "recorded_journal_hash": want, "replayed_journal_hash": got, "answer_label": res.answer_label,
                           "stop_cause": res.stop_cause, "invariants_in_replay": getattr(ctl.last_run, "invariants_checked", None) or "asserted by the controller (raises on violation)"}
    if got != want:
        rec_chain = []
        h = GENESIS
        for e in entries:
            h = _sha(_canon(e))
            rec_chain.append(h)
        new = ctl.last_journal.chain_hashes()
        first = next((i for i, (a, b) in enumerate(zip(rec_chain, new)) if a != b), min(len(rec_chain), len(new)))
        out["first_divergence"] = {"seq": first, "recorded_kind": entries[first]["kind"] if first < len(entries) else None, "replayed_kind": ctl.last_journal.entries()[first]["kind"] if first < len(new) else None}
    return out


# ------------------------------------------------------------------ run level
def check_run(run_dir, public_tasks: Optional[str] = None, require_full: bool = False) -> Dict[str, Any]:
    run = Path(run_dir)
    summary = json.loads((run / "RUN_SUMMARY.json").read_text("utf-8")) if (run / "RUN_SUMMARY.json").exists() else {}
    public: Dict[str, Dict[str, Any]] = {}
    if public_tasks and Path(public_tasks).exists():
        for l in Path(public_tasks).read_text("utf-8").splitlines():
            if l.strip():
                t = json.loads(l)
                public[t["task_id"]] = t
    out: Dict[str, Any] = {"run_dir": str(run), "require_full": require_full, "tasks": {}, "label": "offline replay of saved artifacts; no sandbox, model or network was used"}
    for td in sorted(p for p in (run / "tasks").iterdir() if p.is_dir()) if (run / "tasks").is_dir() else []:
        if not (td / "DONE").exists():
            continue
        a = level_a(td)
        pub = dict(public)
        if td.name not in pub and (td / "task_public.json").exists():
            pub[td.name] = json.loads((td / "task_public.json").read_text("utf-8"))
        b = level_b(td, summary, pub)
        status = "REPLAY_DIVERGED" if "REPLAY_DIVERGED" in (a["status"], b["status"]) else ("REPLAY_MATCH" if b["status"] == "REPLAY_MATCH" else "A_ONLY")
        out["tasks"][td.name] = {"status": status, "level_a": a, "level_b": b}
    st = [t["status"] for t in out["tasks"].values()]
    div = sorted(k for k, t in out["tasks"].items() if t["status"] == "REPLAY_DIVERGED")
    nob = sorted(k for k, t in out["tasks"].items() if t["status"] == "A_ONLY")
    out["diverged"], out["level_b_not_attempted"] = div, {k: out["tasks"][k]["level_b"].get("reason") for k in nob}
    out["counts"] = {"tasks": len(st), "match": st.count("REPLAY_MATCH"), "diverged": len(div), "a_only": len(nob)}
    if not st:
        out["verdict"] = "REPLAY_NO_TASKS"
    elif div:
        out["verdict"] = "REPLAY_DIVERGED"
    elif nob:
        out["verdict"] = "REPLAY_DIVERGED" if require_full else "REPLAY_PARTIAL"
        if require_full:
            out["note"] = "--require-full: level B was not attempted for %d task(s)" % len(nob)
    else:
        out["verdict"] = "REPLAY_OK"
    return out


def to_markdown(rep: Dict[str, Any]) -> str:
    L = ["# REPLAY CHECK: %s" % rep["verdict"], "", "run: `%s`; %s" % (rep["run_dir"], rep["label"]), "", "counts: %s" % json.dumps(rep["counts"]), ""]
    if rep["diverged"]:
        L += ["## DIVERGED"] + ["- **%s**: A %s; B %s %s" % (t, rep["tasks"][t]["level_a"]["diffs"][:2], rep["tasks"][t]["level_b"]["status"], rep["tasks"][t]["level_b"].get("first_divergence") or rep["tasks"][t]["level_b"].get("reason", "")) for t in rep["diverged"]] + [""]
    if rep["level_b_not_attempted"]:
        L += ["## level B not attempted (reason)"] + ["- %s: %s" % kv for kv in sorted(rep["level_b_not_attempted"].items())] + [""]
    L += ["## per task", "", "| task | status | level A | level B |", "|---|---|---|---|"] + ["| %s | %s | %s | %s |" % (t, r["status"], r["level_a"]["status"], r["level_b"]["status"]) for t, r in sorted(rep["tasks"].items())]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--public-tasks")
    ap.add_argument("--require-full", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = check_run(a.run_dir, a.public_tasks, a.require_full)
    md = to_markdown(rep)
    if not a.no_write:
        od = Path(a.out or a.run_dir)
        od.mkdir(parents=True, exist_ok=True)
        for name, txt in (("REPLAY.json", json.dumps(rep, indent=1, sort_keys=True, default=str)), ("REPLAY.md", md)):
            p = od / name
            if p.exists():
                print("refusing to overwrite %s (create-only); use --out" % p)
                return 2
            p.write_text(txt, encoding="utf-8")
    print(md)
    return 0 if rep["verdict"] == "REPLAY_OK" else 1


if __name__ == "__main__":
    sys.exit(main())
