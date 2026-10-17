"""[EL-ARCH-1 D1] ACCEPTANCE DRIVER: the three toy tasks (ex1_slug, ex2_clamp, ex3_wordcount) through ONE controller implementation and ONE stage map.

HOW THE TWO CONTROLLER MODES COMPOSE (decision D1-a; no new miniature controller is written, the stage map in every receipt is the proof)
  The toy repos are WORKSPACE tasks (files + a test command + a protected test), i.e. per-action tasks. The controller that owns them is evidence_ladder.controller.agentic.AgenticRun, wrapped
  by controller.replay.RecordedRun (so the whole run is replayable). The eight stages map onto its existing parts:
    1 intake        AgenticRun.begin(): run_start, registry, requirements R1/R2, oracle tier O1/O2/O3 (controller/tiers.py), routing table, action-risk table hash, `ask_user` on O3.
    2 investigate   controller/investigate.py: deterministic read of sources + the existing tests run in the sandbox; parsed failure location appended to the next context (no model).
    3 generate      controller/generation.ActionProposer on a GenerationBackend (real: Qwen3-4B-Thinking-2507 on a Kaggle T4, replayed here with prompt-hash checks): journal `proposals`.
    4 discriminate  AgenticRun.step() CHECK: every proposed action gets the fixed policy check and, for an Edit, a real dry-run of the protected test on a scratch copy (action_dry_run_v1) =
                    behavioural discrimination between candidates by execution (the per-action analogue of the ladder's L2/behaviour clustering; the ladder's own L2 checks and clusters work on
                    function-level `solve` candidates and are not re-implemented for workspaces).
    5 rank          next_action.decide_next_action(): trusted evidence first, LAYA (L0) last, probe / random head journaled as comparison only, monotonicity gate journaled before the decision,
                    abstain when nothing is allowed (I2).
    6 repair        a round >= 2 is a repair round: the next proposal round sees the observed failure of the previous end state (feedback + investigate report) and goes through the SAME
                    generate/discriminate/rank/execute parts (journal marker `stage: repair`).
    4b execute      commit gate (risk table + authorize), the action runs in the WSL sandbox, the protected test is observed on the real end state (L5 evidence), label per round.
    8 commit+verify AgenticRun.critic_stage() (local GLM critic, diagnostic only) then finish(): re-run of the trusted checks on the real end state (finalize), final_report, decision, stop,
                    assert_invariants I1-I8.
  The ladder controller (EvidenceLadderController, function-level programs) is the OTHER mode; here it is not used for the toy repos (O2 tasks are pointed to it by AgenticRun, see variants).
  Every stage boundary is a journal entry `stage` (written through RecordedRun.note, so it is part of the hash chain and of the replay); stage_map() turns the journal into
  {stage: [[first_seq, last_seq], ...]} and is stored in the receipt.

STAGED KAGGLE HAND-OFF (this script never calls Kaggle, NVIDIA or the network)
  The script is RESUMABLE and DETERMINISTIC: every invocation rebuilds all runs from the cached Kaggle outputs under <out>/kaggle_in and stops at the first missing piece, writing exactly what
  the coordinator must push and exiting with code 3 and a WAITING line.   round k: NEED_GEN (kernel el_arch1_generate_v1: Qwen proposals) -> NEED_SCORE (kernel el_arch1_score_v1 twice,
  live + verify: frozen-Qwen capture of the EXACT context and the EXACT option strings, LAYA + probe + random head) -> PC runs the step in the WSL sandbox -> next round only for unsolved examples
  -> NEED_GLM (tools/el_arch1_glm_live.py in the user's terminal) -> finish, replay (offline + live), receipts, exit 0.

MODES   --offline-rehearsal : every Kaggle/GLM answer is produced by a clearly labelled stub (`stub: true` in every record and receipt); for tests and dry runs only.
        --final             : refuses ANY stub (StubRefused), refuses a dirty source tree, requires config["require"]=["scorer","critic"]; receipts are REAL only if everything real was ingested.
  PYTHONPATH=. .venv/Scripts/python.exe tools/el_arch1_acceptance.py <out dir> (--offline-rehearsal | --final --bundle <results/raw/el-arch1-bundle-*>) [--allow-dirty]"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
TAG = "[EL-ARCH-1 D1 · Aim: complete the missing architecture blocks, not performance]"
OWNER = "sangeetasangamkar"
EXAMPLES = [("ex1_slug", "slug"), ("ex2_clamp", "clamp"), ("ex3_wordcount", "wordcount")]
SEEDS = (0, 1)                       # two samples per decision; merged in seed order by ActionProposer
MAX_NEW_TOKENS = 5000                # thinking must close inside this (the strict parser refuses an unfinished think block); 2 seeds x 5000 stays inside the agentic Qwen-token cap
MAX_STEPS = 2                        # one proposal round + one repair round
SYSTEM_BLOCK_SHA256 = "9bf0352e5debcef4997f0fe2b514a408b072c61e83a87249c7aeb4fdaa5d7250"
STAGES = ("intake", "investigate", "generate", "discriminate", "rank", "execute", "repair", "commit_verify")
EXIT_WAIT = 3
CORE_FILES = ("tools/el_arch1_acceptance.py", "tools/el_arch1_stubs.py", "tools/el_arch1_variants.py", "tools/el_arch1_conformance.py", "tools/el_arch1_glm_live.py", "tools/laya_int1_t6.py", "tools/laya_int1_t6_sandbox.py",
              "tools/laya_int1_stepE.py", "tools/pack_el_arch1_bundle_v1.py", "tools/pack_el_arch1_tests_v1.py", "kaggle/el_arch1_generate_v1.py", "kaggle/evidence_ladder_laya_t6_step_v1.py",
              "evidence_ladder/controller/tests/phaseD1_tests.py", "configs/el_arch1_gates_v1.toml")
CODE_PATHS = ("evidence_ladder", "tools", "configs", "kaggle", "specs", "decision_head", "capture")
# the user's own uncommitted TPU/proposal scripts are not part of the evidence (coordinator decision 2026-10-17); excluded from the cleanliness check only
USER_UNRELATED_EXCLUDES = (":(exclude)kaggle/qwen_proposals_v2.py", ":(exclude)kaggle/tpu_qwen30b_smoke.py")
KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]{6,}|Bearer\s+[A-Za-z0-9._\-]{16,}|\bsk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,}|KGAT_[A-Za-z0-9]{6,}")

from el_arch1_stubs import StubScorer, option, raw_generation_text                 # noqa: E402
from laya_int1_t6 import TASKS                                                       # noqa: E402  task definitions only
from evidence_ladder.controller import generation as GEN, invariants as INV, replay as REP   # noqa: E402
from evidence_ladder.controller.agentic import AgenticContext, AgenticTask, LimitReached, NoOracle   # noqa: E402
from evidence_ladder.controller.limits import load_limits                            # noqa: E402
from evidence_ladder.controller.monotonicity import monotonicity_gate                # noqa: E402
from evidence_ladder.controller.wsl_sandbox import WslSandbox, available            # noqa: E402
from evidence_ladder.models import gen_entry as GE, laya, laya_runtime as RT, qwen as Q   # noqa: E402


def sha(b) -> str: return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()
def jdump(o) -> str: return json.dumps(o, indent=1, ensure_ascii=False, default=str, sort_keys=True)
def write_new(path, text) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f: f.write(text)
def write_over(path, text) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, "utf-8")


class StubRefused(RuntimeError):
    """--final met a stand-in for Qwen / LAYA / GLM / the sandbox. Nothing was run."""


class NeedKaggle(Exception):
    def __init__(self, what: str, rnd: int): super().__init__("%s round %d" % (what, rnd)); self.what, self.rnd = what, rnd


class NeedGlm(Exception):
    pass


# ================================================================== stage map (pure; unit-tested)
_RANK_KINDS = {"scorer", "scorer_unavailable", "scorer_comparison", "scorer_comparison_unavailable", "monotonicity_gate", "ranking", "abstain", "action_decision"}
_EXEC_KINDS = {"action_auth", "label", "step_record", "tool_failure"}


def stage_of(entry: Dict[str, Any], phase: Optional[Dict[str, Any]]) -> str:
    """Stage of one journal entry given the last `stage` marker (phase) seen before it."""
    k, d = entry["kind"], entry["data"]
    name = (phase or {}).get("name", "intake")
    if k == "stage":
        return name if name != "step" else "discriminate"
    if name in ("intake", "investigate", "commit_verify"):
        return name
    if name == "repair":
        return "repair"
    if name == "generate":
        return "generate"
    # phase "step": split by what the entry is
    if k == "proposals":
        return "generate"
    if k == "evidence":
        return "discriminate" if d["evidence"]["check_id"] in ("action_policy_v1", "action_dry_run_v1") else "execute"
    if k == "routing":
        return "discriminate" if str(d.get("step", "")).endswith("_checked") else "execute"
    if k in _RANK_KINDS:
        return "rank"
    return "execute"


def stage_map(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """-> {"stages": {stage: [[first_seq, last_seq], ...]}, "repair_substages": {round: {sub: ranges}}, "entries": n}. Every journal entry belongs to exactly one stage; ranges are contiguous runs."""
    phase, per, repair_sub = None, [], {}
    for e in entries:
        if e["kind"] == "stage":
            phase = e["data"]
        st = stage_of(e, phase)
        per.append(st)
        if st == "repair":      # inside a repair round the same parts run; keep their finer labels
            sub = stage_of(e, dict(phase, name="step" if phase.get("part") == "step" else "generate"))
            repair_sub.setdefault(str(phase.get("round")), {}).setdefault(sub, []).append(e["seq"])
    out: Dict[str, List[List[int]]] = {}
    for e, st in zip(entries, per):
        r = out.setdefault(st, [])
        if r and r[-1][1] == e["seq"] - 1: r[-1][1] = e["seq"]
        else: r.append([e["seq"], e["seq"]])
    def ranges(seqs):
        r = []
        for s in seqs:
            if r and r[-1][1] == s - 1: r[-1][1] = s
            else: r.append([s, s])
        return r
    return {"stages": out, "repair_substages": {rd: {s: ranges(v) for s, v in subs.items()} for rd, subs in repair_sub.items()}, "entries": len(entries),
            "reached": [s for s in STAGES if s in out], "not_reached": [s for s in STAGES if s not in out]}


# ================================================================== stub providers (rehearsal only)
class StubKaggle:
    """Writes into <out>/kaggle_in exactly the files a real Kaggle run would produce, every record marked stub:true. Scripted proposals per (example, round)."""

    def __init__(self, root: Path): self.root = root

    def script(self, name: str, task: str, rnd: int) -> List[str]:
        import el_arch1_variants as V
        fx, wr, rd = V.fix(task), V.wrong(task), V.read(task)
        if name == "ex2_clamp":
            return [wr, rd, option("Bash", command="ls"), option("Bash", command="cat clamp.py")] if rnd == 1 else [fx, wr, rd]
        return [fx, wr, rd, option("Bash", command=TASKS[task]["test_cmd"])]

    def gen(self, rnd: int, jobs: List[Dict[str, Any]]) -> None:
        d = self.root / ("round%d" % rnd) / "gen"; d.mkdir(parents=True, exist_ok=True)
        lines = []
        for j in jobs:
            name, step = j["task_id"].split(":step"); task = dict(EXAMPLES)[name]
            text = raw_generation_text(self.script(name, task, int(step)))
            for s in j["seeds"]:
                g = Q.make_generation(text, tokens_prompt=len(j["prompt"]) // 4, tokens_generated=len(text) // 4, seed=s, prompt_template_id=j["prompt_template_id"], prompt_sha256=j["prompt_sha256"],
                                      model="stub:scripted-qwen", model_revision="stub", wall_seconds=1.0)
                lines.append(json.dumps({"job_id": j["job_id"], "task_id": j["task_id"], "seed": s, "purpose": j["purpose"], "prompt_sha256": j["prompt_sha256"], "generation": Q.generation_to_dict(g), "stub": True},
                                        sort_keys=True, ensure_ascii=False))
        write_new(d / "gen_records.jsonl", "\n".join(lines) + "\n")
        write_new(d / "RUN_SUMMARY.json", jdump({"kernel": "el_arch1_generate_v1", "verdict": "DONE", "stub": True, "gpu": "STUB (no GPU)", "summary": {"errors": 0}}))

    def score(self, rnd: int, which: str, items: List[Dict[str, Any]]) -> None:
        d = self.root / ("round%d" % rnd) / ("score_" + which); d.mkdir(parents=True, exist_ok=True)
        steps = {}
        for it in items:
            arr = json.loads(it["saved_raw_output"]); opts = [GEN.render_call(a["name"], a["arguments"]) for a in arr]
            sc = {k: {"score": StubScorer(k).score("next_action", opts), "role": k} for k in ("laya", "probe", "random_head")}
            ctx = it["context_text"]
            steps[it["key"]] = {"step": it["step"], "status": "ok", "context_sha256": RT.LF.sha256_text(ctx), "raw_output": it["saved_raw_output"], "raw_output_sha256": RT.LF.sha256_text(it["saved_raw_output"]),
                                "generation": {"replayed": True}, "options": opts, "option_sha256": [RT.LF.sha256_text(o) for o in opts], "scores": sc,
                                "provenance": RT.make_provenance(device="Tesla T4 (STUB)", context_text=ctx, options=opts), "origin": "live_capture_same_process",
                                "features_sha256": sha("stub-features|" + ctx + "|" + "|".join(opts)), "stub": True}
        write_new(d / "t6_step_results.json", jdump({"kernel": "el_arch1_score_v1", "mode": "replay", "verdict": "DONE", "stub": True, "gpu": "Tesla T4 (STUB)", "versions": {"stub": True},
                                                     "total_seconds": 1.0, "steps": steps}))


def stub_critic_transport(pids: Dict[str, Dict[str, Any]]):
    """In-process fake GLM for the rehearsal: a valid critic_response that proposes the task's own test command as the experiment."""
    def body(obj):
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(obj), "reasoning_content": "stub reasoning"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 900, "completion_tokens": 400}}

    def transport(packet, key_label):
        cmd = pids[packet["packet_id"]]["observed_outputs"][next(iter(pids[packet["packet_id"]]["observed_outputs"]))][0]["input"]
        return 200, body({"experiment": {"kind": "command", "payload": cmd, "expected_by_candidate": {}}, "implications": "STUB critic: re-run the task's own test command"})
    return transport


# ================================================================== the acceptance object
class Acceptance:
    def __init__(self, out: Path, *, final: bool, bundle: Optional[Path] = None, allow_dirty: bool = False, system_block: Optional[str] = None, sandbox_factory=None):
        self.out, self.final, self.bundle = Path(out), final, (Path(bundle) if bundle else None)
        self.mode = "final" if final else "offline_rehearsal"
        self.allow_dirty = allow_dirty
        self.kin = self.out / "kaggle_in"
        self.system_block = system_block if system_block is not None else self._load_system_block()
        self.sandbox_factory = sandbox_factory or (lambda scratch, prot: WslSandbox(scratch, prot))
        self.notes: Dict[str, Any] = {"gen_runs": {}, "score_runs": {}, "glm": {}}
        self.stub_kaggle = None if final else StubKaggle(self.kin)

    # ---------------------------------------------------------------- preflight
    def _load_system_block(self) -> str:
        cands = ([self.bundle / "dataset/gate/system_block.txt"] if self.bundle else []) + sorted((ROOT / "results/raw").glob("el-arch1-bundle-*/dataset/gate/system_block.txt")) + \
                sorted((ROOT / "results/raw").glob("el-laya-int1-bundle-*/gate/system_block.txt"))
        for p in cands:
            if p.is_file() and sha(p.read_bytes()) == SYSTEM_BLOCK_SHA256:
                return p.read_text("utf-8")
        raise RuntimeError("system block with the pinned sha256 not found (build the bundle first)")

    def preflight(self) -> Dict[str, Any]:
        if self.final and self.allow_dirty: raise RuntimeError("--allow-dirty is for the rehearsal only")
        scan = INV.static_isolation_scan()
        if any(scan.values()): raise RuntimeError("static isolation scan found: %s" % scan)
        info: Dict[str, Any] = {"static_isolation_scan": scan, "limits": load_limits("agentic").as_dict(), "wsl": available()}
        if not info["wsl"]: raise RuntimeError("WSL sandbox does not answer: the acceptance run needs the real sandbox")
        try:
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=30).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--"] + list(CODE_PATHS) + list(USER_UNRELATED_EXCLUDES), cwd=str(ROOT), capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            head, dirty = None, ["git unavailable"]
        core = sorted(set(REP.code_hashes()) | set(CORE_FILES))      # every file whose behaviour is part of the evidence must be TRACKED (git status above ignores untracked files)
        try:
            tracked = set(subprocess.run(["git", "ls-files", "--"] + core, cwd=str(ROOT), capture_output=True, text=True, timeout=60).stdout.split())
        except (OSError, subprocess.TimeoutExpired):
            tracked = set()
        untracked_core = [f for f in core if f not in tracked]
        info.update(commit=head, dirty_tracked_source=dirty, untracked_core_files=untracked_core, source_committed=not dirty and not untracked_core and bool(head))
        if self.final and (dirty or untracked_core): raise RuntimeError("--final needs committed source; modified tracked files: %s; core files not tracked: %s" % (dirty[:6], untracked_core[:6]))
        info["code_sha256"] = REP.code_hashes()
        info["driver_sha256"] = sha((ROOT / "tools/el_arch1_acceptance.py").read_bytes())
        return info

    # ---------------------------------------------------------------- stub guard
    def no_stubs(self, **parts) -> None:
        """Final mode: abort if any ingested part declares itself a stand-in. parts: name -> object or dict. Called on the scorers, the critic, the sandbox and on every ingested Kaggle record."""
        if not self.final: return
        for name, obj in parts.items():
            for o in (obj if isinstance(obj, list) else [obj]):
                flagged = (o.get("stub") if isinstance(o, dict) else getattr(o, "stub", False)) or any("stub" in str(x).lower() for x in
                           ([o.get("gpu"), o.get("model"), o.get("scorer_id")] if isinstance(o, dict) else [getattr(o, "scorer_id", "")]))
                if flagged: raise StubRefused("final mode refuses a stub / fake backend: %s" % name)

    # ---------------------------------------------------------------- Kaggle ingest
    def gen_records(self) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        for p in sorted(self.kin.glob("round*/gen/gen_records.jsonl")):
            sm = json.loads((p.parent / "RUN_SUMMARY.json").read_text("utf-8")) if (p.parent / "RUN_SUMMARY.json").exists() else {}
            if sm.get("verdict") != "DONE" or (sm.get("summary") or {}).get("errors"): raise RuntimeError("generation run %s is not a clean DONE: %s" % (p.parent.parent.name, sm.get("verdict")))
            self.notes["gen_runs"][p.parent.parent.name] = {k: sm.get(k) for k in ("verdict", "gpu", "versions", "total_seconds", "snapshot_verified", "stub")}
            recs += [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]
        if self.final:
            self.no_stubs(gen_records=recs, **{"gen_record.generation": [r["generation"] for r in recs]})
            bad = [r["task_id"] for r in recs if r["generation"]["model"] != Q.QWEN_REPO or r["generation"]["model_revision"] != Q.QWEN_REVISION]
            if bad: raise StubRefused("generation records not made by %s@%s: %s" % (Q.QWEN_REPO, Q.QWEN_REVISION[:8], bad[:3]))
        return recs

    def need_gen(self, rnd: int, jobs: List[Dict[str, Any]]) -> None:
        h = self.out / "handoff" / ("round%d" % rnd) / "gen"
        if self.stub_kaggle is not None:
            self.stub_kaggle.gen(rnd, jobs); return
        for j in jobs: GE.check_job(j)
        req = {"authorized": True, "jobs": jobs}
        tmpl = (self.bundle / "kernels/el_arch1_generate_v1.py").read_text("utf-8")
        assert "__GEN_REQUEST_JSON__" in tmpl and "'''" not in json.dumps(req)
        write_over(h / "el_arch1_generate_v1.py", tmpl.replace("__GEN_REQUEST_JSON__", json.dumps(req, ensure_ascii=False)))
        write_over(h / "kernel-metadata.json", json.dumps(self.kernel_meta("el-arch1-generate-v1", "el_arch1_generate_v1.py"), indent=1))
        write_over(h / "WAITING.txt", "round %d: push this kernel, wait for COMPLETE, then:\n  kaggle kernels output %s/el-arch1-generate-v1 -p \"%s\"\nthen re-run the driver.\n"
                   % (rnd, OWNER, (self.kin / ("round%d" % rnd) / "gen")))
        raise NeedKaggle("gen", rnd)

    def kernel_meta(self, slug: str, code: str) -> Dict[str, Any]:
        return {"id": "%s/%s" % (OWNER, slug), "title": slug, "code_file": code, "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": False, "dataset_sources": ["leminhhung0101/qwen3-4b-thinking-2507", OWNER + "/evidence-ladder-el-arch1-bundle-v1"], "kernel_sources": [], "competition_sources": []}

    def need_score(self, rnd: int, items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
        res = {w: self.kin / ("round%d" % rnd) / ("score_" + w) / "t6_step_results.json" for w in ("live", "verify")}
        if self.stub_kaggle is not None:
            for w in ("live", "verify"):
                if not res[w].exists(): self.stub_kaggle.score(rnd, w, items)
        if not all(p.exists() for p in res.values()):
            h = self.out / "handoff" / ("round%d" % rnd)
            req = {"run_id": "%s/round%d" % (self.out.name, rnd), "mode": "replay", "steps": items, "max_new_tokens": 700, "with_controls": True}
            tmpl = (self.bundle / "kernels/el_arch1_score_v1.py").read_text("utf-8")
            assert "__STEP_REQUEST_JSON__" in tmpl and "'''" not in json.dumps(req)
            src = tmpl.replace("__STEP_REQUEST_JSON__", json.dumps(req, ensure_ascii=False))
            msg = ["round %d: push BOTH kernels (two separate kernels = two independent processes), wait for COMPLETE, download each output:" % rnd]
            for w in ("live", "verify"):
                slug = "el-arch1-score-%s-v1" % w
                write_over(h / ("score_" + w) / "el_arch1_score_v1.py", src)
                write_over(h / ("score_" + w) / "kernel-metadata.json", json.dumps(self.kernel_meta(slug, "el_arch1_score_v1.py"), indent=1))
                msg.append("  kaggle kernels output %s/%s -p \"%s\"" % (OWNER, slug, res[w].parent))
            write_over(h / "WAITING_SCORE.txt", "\n".join(msg) + "\nthen re-run the driver.\n")
            raise NeedKaggle("score", rnd)
        out = {}
        for w, p in res.items():
            out[w] = json.loads(p.read_text("utf-8"))
            if out[w].get("verdict") != "DONE": raise RuntimeError("score kernel %s round %d verdict %s" % (w, rnd, out[w].get("verdict")))
        self.no_stubs(**{"score_live": out["live"], "score_verify": out["verify"]})
        if self.final and not str(out["live"].get("gpu", "")).startswith("Tesla T4"): raise RuntimeError("scoring did not run on a Tesla T4: %r" % out["live"].get("gpu"))
        run_ids = {w: "%s/el-arch1-score-%s-v1@round%d" % (OWNER, w, rnd) for w in out}
        self.notes["score_runs"]["round%d" % rnd] = {w: {"run_id": run_ids[w], "gpu": out[w].get("gpu"), "mode": out[w].get("mode"), "versions": out[w].get("versions"), "stub": out[w].get("stub", False)} for w in out}
        return out["live"], out["verify"], run_ids["live"], run_ids["verify"]

    # ---------------------------------------------------------------- building runs
    def render(self, messages):
        from decision_head import fable_pool as FP
        text, info = FP.select_context(messages, 160000)
        assert text is not None, info
        return text

    def build_run(self, name: str, task: str, base: Path, scorer, comparison, critic):
        T = TASKS[task]; ws = base / name / "workspace"
        for fn, c in T["files"].items(): write_new(ws / fn, c)
        at = AgenticTask(task_id=name, user_text=T["user"], workspace=ws, test_cmd=T["test_cmd"], protected=T["protected"], system_text=self.system_block, max_steps=MAX_STEPS)
        sb = self.sandbox_factory(base / "_sandbox", T["protected"])
        self.no_stubs(**{"sandbox:" + name: sb})
        ctx = AgenticContext(sandbox=sb, scorer=scorer, critic=critic, arm="E", seeds=SEEDS, config={"max_steps": MAX_STEPS, "require": ["scorer", "critic"]}, comparison=comparison)
        rr = REP.RecordedRun(at, ctx, self.render)
        rr.begin()
        rr.note("stage", {"name": "investigate"})
        rr.investigate()
        return rr

    def make_critic(self):
        from evidence_ladder.models.broker import StagedCriticBroker
        from evidence_ladder.models import glm, experiments as EXP
        self.glm_dir = self.out / "glm"
        cfg = glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=EXP.SUPPORTED_KINDS)
        return StagedCriticBroker(self.glm_dir / "critic_batch" / "broker_ledger.jsonl", config=cfg, max_retries=1)

    # ---------------------------------------------------------------- one full deterministic pass
    def one_pass(self, label: str) -> Dict[str, Any]:
        base = self.out / "derived" / label
        if base.exists(): shutil.rmtree(base)
        backends = {k: RT.VerifiedSavedScoreBackend(k) for k in ("laya", "probe", "random_head")}
        scorer = laya.LayaScorer(authorized=True, backend=backends["laya"])
        assert not scorer.disabled, scorer.meta.disabled_reason
        comparison = {"probe": laya.ProbeScorer(authorized=True, backend=backends["probe"]), "random_head": laya.RandomHeadScorer(authorized=True, backend=backends["random_head"])}
        self.no_stubs(laya=scorer, probe=comparison["probe"], random_head=comparison["random_head"])
        critic = self.make_critic()
        self.no_stubs(critic=critic)
        runs = {n: self.build_run(n, t, base, scorer, comparison, critic) for n, t in EXAMPLES}
        missing: List[Any] = []
        holder: Dict[str, Any] = {"replay": None}

        class Staged:
            def generate(self_, req):
                if holder["replay"] is not None:
                    try: return holder["replay"].generate(req)
                    except Q.FixtureMissing: pass
                missing.append(req); raise Q.FixtureMissing((req.task_id,))
            def model_info(self_): return {"backend": "staged_kaggle_replay"}
        proposer = GEN.ActionProposer(Staged(), max_new_tokens=MAX_NEW_TOKENS)
        gate_pools: List[Dict[str, Any]] = []
        for rnd in range(1, MAX_STEPS + 1):
            active = [n for n, r in runs.items() if not r.run.decided and len(r.run.steps) == rnd - 1]
            if not active: break
            props: Dict[str, Any] = {}
            for _attempt in (0, 1):                  # second attempt only after the stub wrote the "Kaggle" files; a real run raises NeedKaggle from need_gen
                missing.clear()
                recs = self.gen_records()
                holder["replay"] = GEN.ReplayGenerationBackend(recs) if recs else None
                for n in active:
                    try: props[n] = proposer.propose(n, rnd, runs[n].run.context_text(), SEEDS)
                    except Q.FixtureMissing: props.pop(n, None)
                if not missing: break
                self.need_gen(rnd, [GE.make_job(rq, rq.task_id) for rq in missing])
            if missing: raise RuntimeError("generation records still missing after the hand-off")
            items = [{"key": "%s:%d" % (n, rnd), "step": rnd, "context_text": runs[n].run.context_text(), "saved_raw_output": json.dumps([json.loads(o) for o in props[n]["options"]])}
                     for n in active if len(props[n]["options"]) >= 2]
            if items:
                live, ver, live_id, ver_id = self.need_score(rnd, items)
                for it in items:
                    n, k = it["key"].split(":")[0], it["key"]
                    lv, vr = live["steps"].get(k, {}), ver["steps"].get(k, {})
                    if lv.get("status") == "ok" and vr.get("status") == "ok" and lv.get("options") == props[n]["options"]:      # a pool the kernel parsed differently from the controller is not admitted
                        for be in backends.values(): be.add(lv, vr, live_run=live_id, verify_run=ver_id, verify_mode=ver["mode"])
                self.notes.setdefault("refused_pools", []).extend(backends["laya"].refused)
            for n in active:
                rr, pr = runs[n], props[n]
                rr.note("stage", {"name": "generate" if rnd == 1 else "repair", "round": rnd, "part": "generate"})
                rr.propose(pr)
                rr.note("stage", {"name": "step" if rnd == 1 else "repair", "round": rnd, "part": "step"})
                try:
                    rec = rr.step(pr["options"], dict(pr["qwen_meta"]))
                    known = [i for i, t in enumerate(rec["trusted"]) if "L5_dry_run" in t]
                    if rec.get("scores") and len(known) >= 2:
                        gate_pools.append({"id": rec["decision_id"], "scores": [rec["scores"][i] for i in known], "passed": [rec["trusted"][i]["L5_dry_run"] == "pass" for i in known]})
                except (NoOracle, LimitReached):
                    pass
        return self.finish_pass(runs, critic, gate_pools, backends, base)

    def finish_pass(self, runs, critic, gate_pools, backends, base) -> Dict[str, Any]:
        from evidence_ladder.critic_batch import validator as V
        from evidence_ladder.critic_batch.batch import BatchRunner, REASONING
        from evidence_ladder.critic_batch.config import load_concurrency
        from evidence_ladder.models import glm, experiments as EXP
        reqs, pids = {}, {}
        for n, rr in runs.items():
            rr.note("stage", {"name": "commit_verify"})
            reqs[n] = rr.run.critic_request()
            pids[n] = critic.build_packet(reqs[n])["packet_id"]
            V.validate_packet({"packet_id": pids[n], "request": reqs[n]})          # blinding / allow-list validator; raises PacketRejected
        pending = [n for n in runs if critic.completed(pids[n]) is None and critic.terminal_state(pids[n]) is None]
        if pending:
            pk = self.glm_dir / "critic_packets"
            if not (pk / "PACKETS_MANIFEST.json").exists(): V.write_packets(pk, {pids[n]: reqs[n] for n in runs})
            if self.final:
                write_over(self.glm_dir / "WAITING_GLM.txt", "run in YOUR terminal (reads .secrets through the project key system; this driver never does):\n  & '%s' '%s' '%s'\nthen re-run the driver.\n"
                           % (ROOT / ".venv/Scripts/python.exe", ROOT / "tools/el_arch1_glm_live.py", self.out))
                raise NeedGlm(len(pending))
            cfg = glm.GlmConfig(reasoning_params=dict(REASONING), allowed_kinds=EXP.SUPPORTED_KINDS)      # rehearsal: in-process fake GLM through the real BatchRunner / PcSender / broker
            BatchRunner(pk, self.glm_dir / "critic_batch", stub_critic_transport({pids[n]: reqs[n] for n in runs}), ["key_1", "key_2", "key_3"], load_concurrency(), max_retries=1, glm_config=cfg).run()
            critic = self.make_critic()
            self.notes["glm"]["stub"] = True
        out = {}
        for n, rr in runs.items():
            rr.ctx.critic = critic
            state = rr.critic_stage()
            out[n] = (rr, rr.finish(), state)
        return {"runs": out, "gate_pools": gate_pools, "backends": backends, "base": base, "critic": critic, "pids": pids}

    # ---------------------------------------------------------------- receipts
    def receipts(self, done: Dict[str, Any], pre: Dict[str, Any], variants: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        rec_dir = self.out / "receipts"
        summary: Dict[str, Any] = {"tag": TAG, "mode": self.mode, "stub_run": not self.final, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "preflight": pre, "constants": {
            "seeds": list(SEEDS), "max_new_tokens": MAX_NEW_TOKENS, "max_steps": MAX_STEPS, "require": ["scorer", "critic"]}, "examples": {}, "kaggle": self.notes}
        stub = not self.final
        for n, (rr, res, state) in done["runs"].items():
            run = rr.run
            entries = run.journal.entries()
            sm = stage_map(entries)
            record = rr.record()
            rep_off = REP.replay(record, critic=done["critic"], render=self.render)
            rep_live = REP.replay(record, sandbox=self.sandbox_factory(done["base"] / "_replay_sandbox", record["task"]["protected"]), critic=done["critic"], render=self.render)
            d = rec_dir / n
            write_new(d / "journal.json", jdump(entries)); write_new(d / "record.json", jdump(record))
            steps = [{k: s.get(k) for k in ("step", "decision_id", "chosen_index", "laya_order", "allowed_set", "id_order_choice", "laya_changed_the_choice_vs_id_order", "trusted_evidence_overrode_laya",
                                            "laya_top_was_chosen", "tests_pass", "label", "scorer_unavailable_reason", "executed_action")} for s in run.steps]
            ex = {"answer_label": res.answer_label, "stop_cause": res.stop_cause, "stop_details": res.stop_details, "journal_hash": res.journal_hash, "invariants": run.invariants_checked, "rounds": res.rounds,
                  "budget_record": res.budget_record, "tier": [e["data"]["tier"] for e in run.journal.entries("oracle_tier")][-1], "stage_map": sm, "steps": steps, "critic_state": state,
                  "critic_result": run.critic_result, "glm_accounting": run.glm_accounting, "replay_offline": rep_off, "replay_live_wsl": rep_live,
                  "monotonicity_entries": [e["data"]["status"] for e in run.journal.entries("monotonicity_gate")], "comparison_entries": len(run.journal.entries("scorer_comparison")),
                  "scorer_unavailable_entries": len(run.journal.entries("scorer_unavailable")), "proposals": [e["data"]["status"] for e in run.journal.entries("proposals")],
                  "tool_runs": res.budget_record["tool_runs"], "stub": stub, "files": {"journal": "receipts/%s/journal.json" % n, "record": "receipts/%s/record.json" % n}}
            write_new(d / "example_receipt.json", jdump(ex))
            ex["files"]["receipt"] = "receipts/%s/example_receipt.json" % n
            summary["examples"][n] = ex
        mono = monotonicity_gate(done["gate_pools"])
        summary["monotonicity_gate_all_examples"] = mono
        summary["gates"] = {"invariants": {"status": "PASS" if all(len(e["invariants"]) == 8 for e in summary["examples"].values()) else "FAIL"},
                            "monotonicity": {"status": mono["status"], "laya_ordering_enabled": mono["laya_ordering_enabled"]}, "overall": "INSUFFICIENT_DATA (3 examples; not a performance result)"}
        summary["backends"] = {"qwen": {"kind": "stub" if stub else "hf_kaggle_t4", "stub": stub, "runs": self.notes["gen_runs"]},
                               "laya": {"kind": "stub" if stub else "kaggle_t4_live_plus_verify", "stub": stub, "runs": self.notes["score_runs"]},
                               "probe": {"stub": stub}, "random_head": {"stub": stub}, "glm": {"kind": "stub" if stub else "live_local_key_system", "stub": stub},
                               "sandbox": {"kind": "wsl_bwrap", "stub": False}, "controller": {"kind": "AgenticRun+RecordedRun", "stub": False}}
        summary["variants"] = variants
        summary["external_logs"] = {str(p.relative_to(self.out)): sha(p.read_bytes()) for p in sorted(list(self.kin.rglob("*_push.txt")) + list((self.out / "glm").glob("GLM_LIVE_SUMMARY_*.json")))}      # push messages (kernel versions) saved by the coordinator + the live GLM summary
        leaks = [str(p.relative_to(self.out)) for p in sorted(self.out.rglob("*")) if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".md", ".py") and KEY_RE.search(p.read_text("utf-8", "replace"))]
        summary["secret_scan_hits"] = leaks
        write_new(rec_dir / "ACCEPTANCE_SUMMARY.json", jdump(summary))
        write_new(rec_dir / "FINAL_SHA256SUMS.txt", "\n".join("%s  %s" % (sha(p.read_bytes()), p.relative_to(rec_dir).as_posix()) for p in sorted(rec_dir.rglob("*")) if p.is_file()) + "\n")
        return summary

    # ---------------------------------------------------------------- top level
    def run(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        pre = self.preflight()
        if (self.out / "receipts" / "ACCEPTANCE_SUMMARY.json").exists(): print("receipts already exist (create-only): nothing to do"); return 0
        variants = None
        vs = self.out / "VARIANTS_SUMMARY.json"
        if not vs.exists():
            import el_arch1_variants as V
            V.run_all(self.out)
        variants = json.loads(vs.read_text("utf-8")); variants = {"summary_file": "VARIANTS_SUMMARY.json", "all_ok": variants["all_ok"], "stub_components": variants["stub_components"], "n": sum(len(v) for v in variants["variants"].values())}
        try:
            done = self.one_pass("pass")
        except NeedKaggle as e:
            print("WAITING_FOR_KAGGLE %s round %d -- see %s" % (e.what, e.rnd, self.out / "handoff" / ("round%d" % e.rnd))); return EXIT_WAIT
        except NeedGlm as e:
            print("WAITING_FOR_GLM %d packet(s) -- see %s" % (e.args[0], self.out / "glm" / "WAITING_GLM.txt")); return EXIT_WAIT
        s = self.receipts(done, pre, variants)
        print(json.dumps({"mode": self.mode, "examples": {n: (e["answer_label"], e["stop_cause"], e["invariants"] and len(e["invariants"]), e["replay_live_wsl"]["identical"]) for n, e in s["examples"].items()},
                          "gates": s["gates"], "variants_ok": variants["all_ok"], "secret_scan_hits": s["secret_scan_hits"]}, default=str))
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--offline-rehearsal", action="store_true"); g.add_argument("--final", action="store_true")
    ap.add_argument("--bundle"); ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if a.final and not a.bundle: ap.error("--final needs --bundle <results/raw/el-arch1-bundle-*>")
    return Acceptance(Path(a.out), final=a.final, bundle=a.bundle, allow_dirty=a.allow_dirty).run()


if __name__ == "__main__":
    sys.exit(main())
