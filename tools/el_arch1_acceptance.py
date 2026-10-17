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

LOCAL MODE (--mode local --backbone <profile id>; defaults leave everything above unchanged)  the same driver, the same controller and stage map, with LOCAL backends instead of the Kaggle hand-off:
  generation  = the backbone profile's model (configs/backbones_v1.toml, e.g. qwen3-0.6b, greedy) in a separate GPU process (--local-worker gen), which also EXPOSES that model's hidden states
                (same recipe positions, profile depths) as DIAGNOSTIC data;
  LAYA        = the 4B profile's states captured on the local GPU by per-layer streaming (--local-worker score, two independent processes = live + verify), used ONLY after the pre-registered device
                parity gate (--local-worker parity, configs/el_arch1_local_parity_v1.toml) passed; otherwise LAYA is explicitly unavailable (scorer_unavailable journalled with the numbers);
  critic      = still the staged GLM hand-off (WAITING_FOR_GLM). Stubs are still refused. One GPU process at a time.

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
from evidence_ladder.models import backbone as BB, gen_entry as GE, laya, laya_runtime as RT, qwen as Q   # noqa: E402


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


# ================================================================== [EL-ARCH-1 A3.6/A3.8] provenance + backend guards (pure helpers; unit-tested in orchestrator/tests/phaseF3_tests.py)
FORBIDDEN_BACKEND_MODULES = ("el_arch1_stubs", "unittest.mock", "mock")
CONFIG_GLOBS = ("configs/el_arch1_*.toml", "configs/backbones_v1.toml", "configs/*limits*.toml", "configs/thresholds*.toml")


def backend_identity(**parts) -> Dict[str, str]:
    """name -> 'module.Class' of the object that actually ran (written into the receipt: it names the backend, not a flag)."""
    return {k: "%s.%s" % (type(o).__module__, type(o).__qualname__) for k, o in parts.items()}


def refuse_mock_backends(**parts) -> None:
    """A3.8: in the acceptance path no part may be an instance of a stub/mock class, whatever flag it carries. Raises StubRefused naming the part and class."""
    for name, o in parts.items():
        cls = type(o)
        mods = (cls.__module__ or "").split(".")
        if cls.__module__ in FORBIDDEN_BACKEND_MODULES or "mock" in mods or "Mock" in cls.__name__ or cls.__name__.startswith("Stub"):
            raise StubRefused("acceptance path refuses mock backend %s = %s.%s" % (name, cls.__module__, cls.__qualname__))


def pool_admissible(item: Dict[str, Any], live: Dict[str, Any], verify: Dict[str, Any], proposed_options: Sequence[str]) -> Tuple[bool, str]:
    """A3.4 / T34 / T49: a stored or freshly captured score pool is admitted ONLY for the exact context text and the exact option list the controller proposed in this decision
    (context sha256 AND the ordered option list). Anything else is a refusal with its reason; there is no fallback pairing of candidates with scores computed for other candidates."""
    if live.get("status") == "too_few_proposals" or verify.get("status") == "too_few_proposals":
        return False, "pool has %d option(s): below the LAYA minimum of 3 (the adapters were trained on 3-5 option pools and the frozen scoring kernel refuses fewer); LAYA is not asked" % len(live.get("options") or [])
    if live.get("status") != "ok" or verify.get("status") != "ok":
        return False, "a scoring result is not ok (live=%s, verify=%s)" % (live.get("status"), verify.get("status"))
    if live.get("context_sha256") != RT.LF.sha256_text(item["context_text"]):
        return False, "context text differs from the one the scores were computed for"
    if list(live.get("options") or []) != list(proposed_options):
        return False, "proposed options differ from the scored options (order and text must match)"
    return True, "exact context sha256 and option list match"


def config_hashes() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for g in CONFIG_GLOBS:
        for f in sorted(ROOT.glob(g)):
            out[f.relative_to(ROOT).as_posix()] = sha(f.read_bytes())
    return out


def _git(*args, timeout=60) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, timeout=timeout).stdout.decode("utf-8", "replace")


def added_lines_secret_scan(base_ref: Optional[str] = None, diff_text: Optional[str] = None) -> Dict[str, Any]:
    """A3.6: key-shaped text on ADDED lines (working tree + index vs HEAD, and the commits since merge-base with `base_ref` when given). Never returns the matching text, only file names and counts.
    diff_text is injectable for tests."""
    if diff_text is None:
        diffs = [_git("diff", "-U0", "HEAD"), _git("diff", "-U0", "--cached")]
        if base_ref:
            mb = _git("merge-base", base_ref, "HEAD").strip()
            if mb: diffs.append(_git("diff", "-U0", mb, "HEAD"))
        diff_text = chr(10).join(diffs)
    hits, cur = {}, None
    for line in diff_text.splitlines():
        if line.startswith("+++ "): cur = line[4:].removeprefix("b/"); continue
        if line.startswith("+") and not line.startswith("+++") and KEY_RE.search(line): hits[cur] = hits.get(cur, 0) + 1
    return {"hits": sum(hits.values()), "files": sorted(hits), "base_ref": base_ref}


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


class UnavailableBackend:
    """Local run whose device parity gate did not pass: every score request raises with the numbers, so the controller journals scorer_unavailable (never an id-order fallback dressed up as LAYA)."""

    def __init__(self, reason: str): self.reason, self.refused = reason, []

    def score(self, decision_type, pool): raise RT.ProvenanceError(self.reason)


# ================================================================== the acceptance object
class Acceptance:
    def __init__(self, out: Path, *, final: bool, bundle: Optional[Path] = None, allow_dirty: bool = False, system_block: Optional[str] = None, sandbox_factory=None,
                 local: bool = False, backbone: Optional[str] = None, replay_from: Optional[Path] = None):
        self.out, self.final, self.bundle = Path(out), final, (Path(bundle) if bundle else None)
        self.local = local
        self.replay_from = Path(replay_from) if replay_from else None      # PATH B: replay of a stored, verified 4B run (generation records + score kernels + GLM ledger), exact contexts only
        self.real = final or local or bool(replay_from)                       # real backends (no stubs): the final-mode refusal rules that are not about T4 / committed source apply to a local run too
        if local and not backbone: raise ValueError("--mode local needs --backbone <profile id>")
        self.gen_profile = BB.get(backbone) if backbone else None
        self.state_profile = BB.get()                    # the profile the frozen adapters + head consume (4B)
        self.laya_off: Optional[str] = None              # local run: reason LAYA is unavailable (parity gate failed)
        self.mode = "local" if local else ("replay_4b" if replay_from else ("final" if final else "offline_rehearsal"))
        self.allow_dirty = allow_dirty
        self.kin = self.out / "kaggle_in"
        self.system_block = system_block if system_block is not None else self._load_system_block()
        self.sandbox_factory = sandbox_factory or (lambda scratch, prot: WslSandbox(scratch, prot))
        self.notes: Dict[str, Any] = {"gen_runs": {}, "score_runs": {}, "glm": {}}
        self.stub_kaggle = None if self.real else StubKaggle(self.kin)

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
        self.final_path = bool(self.final or ((self.local or self.replay_from) and not self.allow_dirty))      # A3.6: a local run is the final path unless --allow-dirty (development) was given
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
        if self.replay_from:      # seed the new run directory with the stored evidence (create-only, once); nothing is regenerated, rescored or re-asked
            src = self.replay_from
            for sub in ("kaggle_in", "glm/critic_batch", "glm/critic_packets"):
                if (src / sub).is_dir() and not (self.out / sub).exists(): shutil.copytree(src / sub, self.out / sub)
            info["replay_from"] = {"dir": str(src), "kaggle_in_sha256": {p.relative_to(src).as_posix(): sha(p.read_bytes()) for p in sorted((src / "kaggle_in").rglob("*")) if p.is_file()}}
        info.update(commit=head, dirty_tracked_source=dirty, untracked_core_files=untracked_core, source_committed=not dirty and not untracked_core and bool(head))
        info["secret_scan_added_lines"] = added_lines_secret_scan("main")
        if self.final_path and info["secret_scan_added_lines"]["hits"]: raise RuntimeError("secret scan on added lines found %d hit(s) in %s" % (info["secret_scan_added_lines"]["hits"], info["secret_scan_added_lines"]["files"]))
        info["final_path"] = self.final_path
        if self.final_path and (dirty or untracked_core): raise RuntimeError("the final path needs committed source (use --allow-dirty only for development); modified tracked files: %s; core files not tracked: %s" % (dirty[:6], untracked_core[:6]))
        info["code_sha256"] = REP.code_hashes()
        info["driver_sha256"] = sha((ROOT / "tools/el_arch1_acceptance.py").read_bytes())
        info["config_sha256"] = config_hashes()
        return info

    # ---------------------------------------------------------------- stub guard
    def no_stubs(self, **parts) -> None:
        """Final mode: abort if any ingested part declares itself a stand-in. parts: name -> object or dict. Called on the scorers, the critic, the sandbox and on every ingested Kaggle record."""
        if not self.real: return
        for name, obj in parts.items():
            for o in (obj if isinstance(obj, list) else [obj]):
                if not isinstance(o, dict): refuse_mock_backends(**{name: o})
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
        if self.real:
            gp = self.gen_profile or BB.get()            # final (Kaggle) mode: the 4B profile = the previous hard-coded Q.QWEN_REPO / Q.QWEN_REVISION
            self.no_stubs(gen_records=recs, **{"gen_record.generation": [r["generation"] for r in recs]})
            bad = [r["task_id"] for r in recs if r["generation"]["model"] != gp.repo or r["generation"]["model_revision"] != gp.revision]
            if bad: raise StubRefused("generation records not made by %s@%s: %s" % (gp.repo, gp.revision[:8], bad[:3]))
        return recs

    def need_gen(self, rnd: int, jobs: List[Dict[str, Any]]) -> None:
        h = self.out / "handoff" / ("round%d" % rnd) / "gen"
        if self.stub_kaggle is not None:
            self.stub_kaggle.gen(rnd, jobs); return
        for j in jobs: GE.check_job(j)
        if self.local:
            self.run_worker("gen", rnd, {"jobs": jobs, "profile": self.gen_profile.id}, self.kin / ("round%d" % rnd) / "gen"); return
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
        if self.local:        # two independent GPU processes (live + verify) = the two Kaggle kernels of the hand-off; the frozen-4B capture streams layer by layer
            for w in ("live", "verify"):
                if not res[w].exists(): self.run_worker("score", rnd, {"items": items, "which": w}, res[w].parent)
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
        if self.local: self.notes["score_runs"]["round%d" % rnd] = {w: {"gpu": out[w].get("gpu"), "mode": out[w].get("mode"), "versions": out[w].get("versions"), "stub": False, "local": True} for w in out}
        if (self.final or self.replay_from) and not str(out["live"].get("gpu", "")).startswith("Tesla T4"): raise RuntimeError("scoring did not run on a Tesla T4: %r" % out["live"].get("gpu"))
        run_ids = {w: ("local/el-arch1-score-%s@round%d" % (w, rnd)) if self.local else "%s/el-arch1-score-%s-v1@round%d" % (OWNER, w, rnd) for w in out}
        if not self.local: self.notes["score_runs"]["round%d" % rnd] = {w: {"run_id": run_ids[w], "gpu": out[w].get("gpu"), "mode": out[w].get("mode"), "versions": out[w].get("versions"), "stub": out[w].get("stub", False)} for w in out}
        return out["live"], out["verify"], run_ids["live"], run_ids["verify"]

    # ---------------------------------------------------------------- local run hooks (--mode local)
    def run_worker(self, kind: str, rnd: int, req: Dict[str, Any], outdir) -> None:
        """One local GPU worker = one separate process (the GPU is free again when it exits). It reads a request json and writes its results under outdir."""
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        h = self.out / "handoff" / ("parity" if kind == "parity" else "round%d" % rnd); rp = h / ("%s%s_request.json" % (kind, "_" + req["which"] if "which" in req else ""))
        write_over(rp, json.dumps(req, ensure_ascii=False))
        env = dict(os.environ, PYTHONPATH=str(ROOT), ATEN_CPU_CAPABILITY="avx2", PYTHONUNBUFFERED="1")
        logp = outdir / "worker.log"
        with open(logp, "w", encoding="utf-8") as lf:
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--local-worker", kind, str(rp), str(outdir)], cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc: raise RuntimeError("local worker %s failed (rc %s); tail of %s:\n%s" % (kind, rc, logp, logp.read_text("utf-8", "replace")[-1500:]))

    def local_parity(self) -> None:
        """Pre-registered device parity gate (configs/el_arch1_local_parity_v1.toml). PASS -> the RTX 3050 is registered for the 4B profile for THIS process only, from a versioned file that
        carries the sha256 of the passed result; anything else -> LAYA is unavailable in this run and says why."""
        d = self.out / "parity"; res = d / "PARITY_RESULT.json"
        try:
            if not res.exists(): self.run_worker("parity", 0, {}, d)
            doc = json.loads(res.read_text("utf-8"))
        except (RuntimeError, OSError, ValueError) as e:
            self.notes["parity"] = {"verdict": "NOT_RUN", "error": str(e)[:600]}
            self.laya_off = "scorer unavailable: the local device parity gate could not run (%s)" % str(e)[:300]
            return
        self.notes["parity"] = {k: doc.get(k) for k in ("verdict", "device", "device_prefix", "worst_cos_min", "worst_max_abs_diff", "n_records", "thresholds_sha256", "total_seconds", "peak_gib")}
        self.notes["parity"].update(file=res.relative_to(self.out).as_posix(), sha256=sha(res.read_bytes()))
        self.notes["device"] = doc.get("device")
        if doc["verdict"] == "PASS":
            reg = d / "el_arch1_local_device_v1.toml"
            if not reg.exists():
                write_new(reg, '# [EL-ARCH-1 local run] device registration for THIS LOCAL RUN ONLY (versioned; written only after the pre-registered parity gate PASSED)\n[registration]\nprofile_id = "%s"\n'
                               'device_prefix = "%s"\nparity_result = "%s"\nparity_result_sha256 = "%s"\n' % (self.state_profile.id, doc["device_prefix"], res.resolve().relative_to(ROOT).as_posix(), sha(res.read_bytes())))
            BB.register_device(self.state_profile, doc["device"], reg)
        elif doc["verdict"] == "FAIL":
            self.laya_off = ("scorer unavailable: local device parity gate FAILED (worst cos_min=%.6f vs >= 0.9999, worst max_abs_diff=%.4f vs <= 0.5, %d record(s) failed of %d; device %s)"
                             % (doc["worst_cos_min"], doc["worst_max_abs_diff"], sum(not r["passed"] for r in doc["rows"]), doc["n_records"], doc["device"]))
        else:
            self.laya_off = "scorer unavailable: the local device parity gate errored (%s)" % str(doc.get("error"))[:300]

    # ---------------------------------------------------------------- building runs
    def render(self, messages):
        from decision_head import fable_pool as FP
        text, info = FP.select_context(messages, 160000)
        assert text is not None, info
        return text

    def run_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"max_steps": MAX_STEPS, "require": ["scorer", "critic"]}
        if self.local:      # profile id + hash are part of the journalled run configuration (the default Kaggle path adds nothing, so its journal hashes are unchanged)
            cfg["backbones"] = {"generation": {"id": self.gen_profile.id, "sha256": self.gen_profile.sha256}, "states": {"id": self.state_profile.id, "sha256": self.state_profile.sha256}, "mode": "local"}
        return cfg

    def build_run(self, name: str, task: str, base: Path, scorer, comparison, critic):
        T = TASKS[task]; ws = base / name / "workspace"
        for fn, c in T["files"].items(): write_new(ws / fn, c)
        at = AgenticTask(task_id=name, user_text=T["user"], workspace=ws, test_cmd=T["test_cmd"], protected=T["protected"], system_text=self.system_block, max_steps=MAX_STEPS)
        sb = self.sandbox_factory(base / "_sandbox", T["protected"])
        self.no_stubs(**{"sandbox:" + name: sb})
        ctx = AgenticContext(sandbox=sb, scorer=scorer, critic=critic, arm="E", seeds=SEEDS, config=self.run_config(), comparison=comparison)
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
        backends = {k: (UnavailableBackend(self.laya_off) if self.laya_off else RT.VerifiedSavedScoreBackend(k)) for k in ("laya", "probe", "random_head")}
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
        # PATH A (local) uses the one-independent-action prompt + per-option salvage + candidate filter (ActionProposerV2); the Kaggle / PATH B replay keeps the frozen v1 proposer (stored prompts must reproduce)
        proposer = (lambda b, **k: GEN.ActionProposerV2(b, prefilter=False, **k) if self.use_v2() else GEN.ActionProposer(b, **k))(Staged(), max_new_tokens=MAX_NEW_TOKENS)      # local: prefilter=False, the controller's single filter_options call journals every rejection
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
                    try:
                        props[n] = proposer.propose(n, rnd, runs[n].run.context_text(), SEEDS, **self.filter_ctx(runs[n]))
                        props[n]["step_options"] = self.step_options(props[n])
                        props[n]["scored_options"] = self.accepted_preview(runs[n], props[n]["step_options"]) if self.use_v2() else props[n]["options"]
                    except Q.FixtureMissing: props.pop(n, None)
                if not missing: break
                self.need_gen(rnd, [GE.make_job(rq, rq.task_id) for rq in missing])
            if missing: raise RuntimeError("generation records still missing after the hand-off")
            items = [{"key": "%s:%d" % (n, rnd), "step": rnd, "context_text": runs[n].run.context_text(), "saved_raw_output": json.dumps([json.loads(o) for o in props[n]["scored_options"]])}
                     for n in active if len(props[n]["scored_options"]) >= 2]
            if items and not self.laya_off:
                live, ver, live_id, ver_id = self.need_score(rnd, items)
                for it in items:
                    n, k = it["key"].split(":")[0], it["key"]
                    lv, vr = live["steps"].get(k, {}), ver["steps"].get(k, {})
                    ok_pool, why_pool = pool_admissible(it, lv, vr, props[n]["scored_options"])      # a pool the kernel parsed differently from the controller, or for another context, is not admitted
                    if ok_pool:
                        for be in backends.values(): be.add(lv, vr, live_run=live_id, verify_run=ver_id, verify_mode=ver["mode"])
                    else:
                        self.notes.setdefault("refused_pools", []).append({"key": k, "reason": why_pool, "refused_by": "pool_admissible"})
                        for be in backends.values(): be.refused.append({"context_sha256": RT.LF.sha256_text(it["context_text"]), "reason": why_pool})      # the controller's scorer_unavailable journal entry then carries the real cause
                self.notes.setdefault("refused_pools", []).extend(backends["laya"].refused)
            for n in active:
                rr, pr = runs[n], props[n]
                rr.note("stage", {"name": "generate" if rnd == 1 else "repair", "round": rnd, "part": "generate"})
                rr.propose(pr)
                rr.note("stage", {"name": "step" if rnd == 1 else "repair", "round": rnd, "part": "step"})
                try:
                    rec = rr.step(pr["step_options"], dict(pr["qwen_meta"]))
                    known = [i for i, t in enumerate(rec["trusted"]) if "L5_dry_run" in t]
                    if rec.get("scores") and len(known) >= 2:
                        gate_pools.append({"id": rec["decision_id"], "scores": [rec["scores"][i] for i in known], "passed": [rec["trusted"][i]["L5_dry_run"] == "pass" for i in known]})
                except (NoOracle, LimitReached):
                    pass
        return self.finish_pass(runs, critic, gate_pools, backends, base)

    def use_v2(self) -> bool:
        """The fixed pipeline (one-independent-action prompt qwen_actions_v2, per-option salvage, the controller's single candidate filter): every fresh run, local or Kaggle --final.
        PATH B replay (--mode replay4b) and the offline rehearsal keep the frozen v1 proposer so stored prompts reproduce."""
        return bool(self.local or (self.final and not self.replay_from))

    def filter_ctx(self, rr) -> Dict[str, Any]:
        """Arguments of the candidate filter for ActionProposerV2: the CURRENT workspace texts (so unappliable edits are rejected against what is really on disk) and the protected files. Empty for v1."""
        if not self.use_v2(): return {}
        ws = rr.run.task.workspace
        files = {f.relative_to(ws).as_posix(): f.read_text("utf-8", "replace") for f in sorted(Path(ws).rglob("*")) if f.is_file() and f.suffix in (".py", ".txt", ".md", ".toml", ".json") and f.stat().st_size < 200_000}
        return {"files": files}

    def step_options(self, pr) -> List[str]:
        import el_arch1_variants as V
        return V.options_from_proposal(pr)

    def accepted_preview(self, rr, step_options) -> List[str]:
        """The options the controller's own filter will accept for this decision (same pure function, same files, same scope): the scorer must score exactly THOSE (frozen interface: scorer inputs = accepted
        options). A repeated_failure rejection added by the controller later makes the pool differ and the scorer is then explicitly unavailable (pool_admissible)."""
        from evidence_ladder.controller import candidate_filter as CF
        cands = []
        for i, o in enumerate(step_options):
            try:
                a = json.loads(o); ok = isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("arguments"), dict)
            except ValueError:
                ok = False
            cands.append({"tool": a["name"], "arguments": a["arguments"], "origin": {"seed": None, "index": i}, "raw": o} if ok else {"malformed": True, "origin": {"seed": None, "index": i}, "raw": o})
        fr = CF.filter_options(cands, rr.run.files_text(), path_scope={"allow": None, "protected": list(rr.run.task.protected)})
        return [GEN.render_call(a["tool"], a["arguments"]) for a in fr["accepted"]]

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
            if self.real:
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
        summary: Dict[str, Any] = {"tag": TAG, "mode": self.mode, "stub_run": not self.real, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "preflight": pre, "constants": {
            "seeds": list(SEEDS), "max_new_tokens": MAX_NEW_TOKENS, "max_steps": MAX_STEPS, "require": ["scorer", "critic"]}, "examples": {}, "kaggle": self.notes}
        stub = not self.real
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
        if self.local:
            summary["backends"]["qwen"] = {"kind": "local_hf", "stub": False, "profile": self.gen_profile.as_dict(), "profile_sha256": self.gen_profile.sha256, "runs": self.notes["gen_runs"]}
            summary["backends"]["laya"] = {"kind": "local_4b_stream" if not self.laya_off else "unavailable", "stub": False, "profile": self.state_profile.as_dict(), "profile_sha256": self.state_profile.sha256,
                                           "unavailable_reason": self.laya_off, "runs": self.notes["score_runs"]}
            summary["local"] = {"mode": "LOCAL", "device": self.notes.get("device"), "generation_model": self.gen_profile.repo, "generation_profile": {"id": self.gen_profile.id, "sha256": self.gen_profile.sha256},
                                "state_profile": {"id": self.state_profile.id, "sha256": self.state_profile.sha256}, "parity": self.notes.get("parity"), "laya_available": not self.laya_off,
                                "qwen_diagnostic_states": "qwen3-0.6b states are DIAGNOSTIC; NOT consumable by the frozen LAYA head; data for a future adapter" if not self.gen_profile.is_head_compatible else None}
        summary["variants"] = variants
        any_rr = next(iter(done["runs"].values()))[0]
        summary["backend_identity"] = backend_identity(laya=done["backends"]["laya"], probe=done["backends"]["probe"], random_head=done["backends"]["random_head"], critic=done["critic"], sandbox=any_rr.ctx.sandbox,
                                                       controller=any_rr.run, recorded_run=any_rr)       # A3.8: the receipt NAMES the class that ran (module.Class), not just a stub flag
        mp = [p for p in (summary.get("backends") or {}).values()]
        summary["provenance"] = {"commit": pre.get("commit"), "final_path": pre.get("final_path"), "source_committed": pre.get("source_committed"), "config_sha256": pre.get("config_sha256"),
            "source_sha256": pre.get("code_sha256"), "driver_sha256": pre.get("driver_sha256"), "secret_scan_added_lines": pre.get("secret_scan_added_lines"),
            "models": {"generation": ({"repo": self.gen_profile.repo, "revision": self.gen_profile.revision} if self.gen_profile else {"repo": BB.get().repo, "revision": BB.get().revision}),
                       "states": {"repo": self.state_profile.repo, "revision": self.state_profile.revision, "profile_sha256": self.state_profile.sha256}, "critic": __import__("evidence_ladder.models.glm", fromlist=["GLM_MODEL"]).GLM_MODEL},
            "journal_hashes": {n: res.journal_hash for n, (rr, res, st) in done["runs"].items()},
            "receipt_sha256": {n: sha((rec_dir / n / "example_receipt.json").read_bytes()) for n in done["runs"]},
            "journal_file_sha256": {n: sha((rec_dir / n / "journal.json").read_bytes()) for n in done["runs"]}}
        glm_cb = self.out / "glm" / "critic_batch"
        if glm_cb.is_dir():       # A3.1: ledger reconciliation of the real GLM step
            from evidence_ladder.critic_batch.reconcile import reconcile
            summary["glm_reconcile"] = reconcile(glm_cb)
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
        if self.local or self.replay_from:      # the fault-injection variants are a separate acceptance item (--variants-only), not part of the 3-example path runs
            variants = {"summary_file": None, "all_ok": None, "stub_components": None, "n": 0, "skipped": "path run: variants are reported separately (VARIANTS_SUMMARY.json of the variants run)"}
            if self.local: self.local_parity()
        elif not vs.exists():
            import el_arch1_variants as V
            V.run_all(self.out)
        if not (self.local or self.replay_from): variants = json.loads(vs.read_text("utf-8")); variants = {"summary_file": "VARIANTS_SUMMARY.json", "all_ok": variants["all_ok"], "stub_components": variants["stub_components"], "n": sum(len(v) for v in variants["variants"].values())}
        try:
            done = self.one_pass("pass")
        except NeedKaggle as e:
            print("WAITING_FOR_KAGGLE %s round %d -- see %s" % (e.what, e.rnd, self.out / "handoff" / ("round%d" % e.rnd))); return EXIT_WAIT
        except GEN.PromptMismatch as e:
            if not self.replay_from: raise
            # PATH B refusal (T34/T49): the stored 4B proposals/scores belong to the exact stored contexts; the current controller builds a different context, so NOTHING is replayed and nothing is paired.
            write_over(self.out / "REPLAY_REFUSED.json", jdump({"path": "B", "verdict": "REFUSED", "reason": str(e)[:600], "stored_run": str(self.replay_from), "commit": pre.get("commit"),
                                                                 "meaning": "the stored context/prompt hashes do not reproduce under the current controller code; stored 4B scores are valid only for the stored contexts and options, so no score is used"}))
            print("PATH_B_REFUSED -- %s" % str(e)[:200]); return 4
        except NeedGlm as e:
            print("WAITING_FOR_GLM %d packet(s) -- see %s" % (e.args[0], self.out / "glm" / "WAITING_GLM.txt")); return EXIT_WAIT
        s = self.receipts(done, pre, variants)
        print(json.dumps({"mode": self.mode, "examples": {n: (e["answer_label"], e["stop_cause"], e["invariants"] and len(e["invariants"]), e["replay_live_wsl"]["identical"]) for n, e in s["examples"].items()},
                          "gates": s["gates"], "variants_ok": variants["all_ok"], "secret_scan_hits": s["secret_scan_hits"]}, default=str))
        return 0


# ================================================================== local GPU workers (each runs in its own process, one at a time; torch is imported only here)
def _bundle_dir() -> Path:
    c = [p for p in sorted((ROOT / "results/raw").glob("el-laya-int1-bundle-*")) if (p / "gate/feat_v3_B.pt").is_file()]
    if not c: raise RuntimeError("no el-laya-int1-bundle-* with gate/ and models/ found")
    return c[-1]


def _versions() -> Dict[str, Any]:
    import torch, transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}


def _gate_data(B: Path):
    import torch
    FE = {n: {r["decision_id"]: r for r in torch.load(B / ("gate/feat_%s_B.pt" % n), map_location="cpu", weights_only=False)["records"]} for n in ("v2", "v3", "val")}
    RQ = {n: {d["decision_id"]: d for d in (json.loads(l) for l in open(B / ("gate/req_%s.jsonl" % n), encoding="utf-8"))} for n in ("v2", "v3", "val")}
    return FE, RQ, json.loads((B / "gate/labels.json").read_text("utf-8"))


def worker_gen(req: Dict[str, Any], out: Path) -> int:
    """Local generation with the profile's model (greedy for qwen3-0.6b) through the SAME GenerationRequest/run_jobs path as the Kaggle entry, then the model's hidden states are EXPOSED:
    for every job the proposed options + the context are captured with the profile's depths (same recipe positions) and stored as DIAGNOSTIC data."""
    import torch
    from evidence_ladder.models import profile_capture as PC
    os.environ["EVIDENCE_LADDER_ALLOW_INFERENCE"] = "1"
    t0 = time.time(); P = BB.get(req["profile"]); rs: Dict[str, Any] = {"kernel": "el_arch1_local_gen", "profile": P.as_dict(), "profile_sha256": P.sha256, "stub": False, "versions": _versions()}
    try:
        man = json.loads((P.model_dir() / "MANIFEST_pinned.json").read_text("utf-8"))
        assert man["revision"] == P.revision, "model manifest revision differs from the profile"
        pins = {f: man["files"][f]["sha256"] for f in ("model.safetensors", "tokenizer.json")}
        model, tok, dev = PC.load_model(P); rs["gpu"] = dev
        be = Q.HFQwenBackend(authorized=True, profile=P, local=True, model_dir=str(P.model_dir()), log_path=str(out / "gen_log.jsonl"), verify_files=pins)
        be._verify_snapshot(); be._model, be._tok = model, tok
        torch.cuda.reset_peak_memory_stats()
        summary = GE.run_jobs(be, req["jobs"], str(out / "gen_records.jsonl"))
        recs = [json.loads(l) for l in (out / "gen_records.jsonl").read_text("utf-8").splitlines() if l.strip()]
        ntok, nsec = sum(r["generation"]["tokens_generated"] for r in recs), sum(r["generation"]["wall_seconds"] for r in recs)
        cap_of = {j["job_id"]: j["max_new_tokens"] for j in req["jobs"]}
        rs.update(summary=summary, snapshot_verified=be.snapshot_verified, generation_tokens=ntok, generation_seconds=round(nsec, 2), tokens_per_second=round(ntok / nsec, 2) if nsec else None,
                  gpu_peak_gib_generation=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                  per_generation=[{"task_id": r["task_id"], "seed": r["seed"], "tokens": r["generation"]["tokens_generated"], "seconds": r["generation"]["wall_seconds"],
                                   "hit_token_cap": r["generation"]["tokens_generated"] >= cap_of[r["job_id"]]} for r in recs])
        # ---- exposed hidden states (diagnostic)
        cap = PC.make_capture_fn(P, model, tok); sd = out / ("states_" + P.id); sd.mkdir(exist_ok=True); index = []
        label = "%s states: DIAGNOSTIC; NOT consumable by the frozen LAYA head; data for a future adapter" % P.id
        suf = "\n\n" + GEN.ACTION_INSTR
        for j in req["jobs"]:
            r0 = next((r for r in recs if r["job_id"] == j["job_id"] and r["seed"] == j["seeds"][0]), None)
            if r0 is None or not j["prompt"].endswith(suf): index.append({"job_id": j["job_id"], "status": "no_generation"}); continue
            opts = GEN.parse_actions(r0["generation"]["text"])["actions"]
            if len(opts) < 2: index.append({"job_id": j["job_id"], "status": "fewer_than_2_options", "n_options": len(opts)}); continue
            ctx = j["prompt"][:-len(suf)]; hq, ho, info = cap(ctx, opts)
            if hq is None: index.append({"job_id": j["job_id"], "status": info.get("status")}); continue
            f = sd / (j["job_id"].replace(":", "_") + ".pt")
            torch.save({"label": label, "hq": hq, "ho": ho, "options": opts, "context_sha256": RT.LF.sha256_text(ctx), "capture": info, "profile_id": P.id, "profile_sha256": P.sha256,
                        "provenance": RT.make_provenance(device=dev, context_text=ctx, options=opts, profile=P), "diagnostic_only": True}, f)
            index.append({"job_id": j["job_id"], "status": "ok", "file": f.name, "file_sha256": sha(f.read_bytes()), "n_options": len(opts), "hq_shape": list(hq.shape), "ho_shape": list(ho.shape),
                          "features_sha256": info["features_sha256"], "capture_seconds": info["seconds"]})
        rs.update(states_label=label, states_index=index, gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), verdict="DONE" if not summary["errors"] else "FAILED")
    except BaseException as e:      # noqa: BLE001
        import traceback; rs.update(verdict="FAILED", error=repr(e)[:500], trace=traceback.format_exc()[-3000:]); print(rs["trace"])
    rs["total_seconds"] = round(time.time() - t0, 1); write_over(out / "RUN_SUMMARY.json", jdump(rs)); print("VERDICT", rs["verdict"]); return 0


def _parse_raw(raw: str):
    i = raw.find("["); items, dropped = [], 0
    try: arr, _ = json.JSONDecoder().raw_decode(raw[i:]) if i >= 0 else ([], 0)
    except ValueError: arr = []
    for a in arr if isinstance(arr, list) else []:
        if isinstance(a, dict) and a.get("name") in GEN.ACTION_TOOLS and isinstance(a.get("arguments"), dict) and set(a["arguments"]) == GEN.ACTION_TOOLS[a["name"]] and all(isinstance(v, str) for v in a["arguments"].values()):
            s = GEN.render_call(a["name"], a["arguments"])
            if s not in items: items.append(s)
            else: dropped += 1
        else: dropped += 1
    return items[:5], dropped


def worker_score(req: Dict[str, Any], out: Path) -> int:
    """Frozen-4B capture (profile qwen3-4b-thinking-2507, per-layer streaming on the local GPU) of the EXACT context and option strings, then the 3 adapters + pretrained head + probe + random heads.
    Same step schema as kaggle/evidence_ladder_laya_t6_step_v1.py (replay mode: the proposals are the saved ones)."""
    import torch
    from evidence_ladder.models import profile_capture as PC, probe as PR, random_head as RH
    t0 = time.time(); B = _bundle_dir(); P = BB.get(); rs: Dict[str, Any] = {"kernel": "el_arch1_local_score", "mode": "replay", "which": req.get("which"), "profile_id": P.id, "profile_sha256": P.sha256, "stub": False, "steps": {}}
    try:
        rs["versions"] = _versions(); LF = RT.LF
        heads = {n: LF.build_head(B / "models/pretrained_31.safetensors", B / ("models/%s.pt" % n)) for n in LF.SEEDS}
        FE, RQ, LAB = _gate_data(B); rows = []
        for nm in ("v2", "v3"):
            for did, sr in FE[nm].items():
                if did in LAB["v3"]:
                    lab = LAB["v3"][did]; ix = {o: i for i, o in enumerate(RQ[nm][did]["options"])}
                    rows.append(dict(decision_id=did, ho=torch.stack([sr["ho"][ix[o]] for o in lab["options"]]), pos=torch.tensor([s == "positive" for s in lab["status"]])))
        fit = PR.refit_probe(rows, expect_pools=PR.TRAIN_POOLS); rand = RH.build_random_heads(scaler_from=str(B / "models/final-s0.pt"))
        rs.update(probe_train_pools=len(rows), probe_fingerprint=fit["fingerprint"], probe_order_sha256=fit["order_sha256"], random_head_state_sha256={n: RH.state_sha256(h) for n, h in rand.items()})
        del rows, FE, RQ
        model, tok, dev = PC.load_model(P); rs["gpu"] = dev
        live = RT.LiveScorer(PC.make_capture_fn(P, model, tok), RT.FeatureScorer(heads, probe=fit["probe"], random_head=rand), dev, P)
        torch.cuda.reset_peak_memory_stats()
        for st in req["items"]:
            raw = st["saved_raw_output"]; opts, dropped = _parse_raw(raw)
            r = {"step": st["step"], "context_sha256": LF.sha256_text(st["context_text"]), "raw_output": raw, "raw_output_sha256": LF.sha256_text(raw), "generation": {"replayed": True}, "options": opts,
                 "option_sha256": [LF.sha256_text(o) for o in opts], "dropped_proposals": dropped}
            if len(opts) < 3: r["status"] = "too_few_proposals"
            else:
                o = live.score_texts(st["context_text"], opts)
                r.update(status="ok", scores=o["scores"], provenance=o["provenance"], origin=o["origin"], capture=o["capture"], features_sha256=o["capture"]["features_sha256"], score_seconds=o["seconds"])
            rs["steps"][st["key"]] = r
        rs.update(gpu_peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), verdict="DONE")
    except BaseException as e:      # noqa: BLE001
        import traceback; rs.update(verdict="FAILED", error=repr(e)[:500], trace=traceback.format_exc()[-3000:]); print(rs["trace"])
    rs["total_seconds"] = round(time.time() - t0, 1); write_over(out / "t6_step_results.json", jdump(rs)); print("VERDICT", rs["verdict"]); return 0


def worker_parity(req: Dict[str, Any], out: Path) -> int:
    """configs/el_arch1_local_parity_v1.toml: local-GPU streaming 4B capture vs the stored T4 training features of the 3 TRAIN + 3 VAL smoke pools. ALL must pass; the bar is fixed in that file."""
    import tomllib, torch
    import torch.nn.functional as F
    from evidence_ladder.models import profile_capture as PC
    t0 = time.time(); cfgp = ROOT / "configs/el_arch1_local_parity_v1.toml"; G = tomllib.loads(cfgp.read_text("utf-8"))["gate"]
    P = BB.get(G["profile_id"]); doc: Dict[str, Any] = {"gate": "el_arch1_local_parity_v1", "profile_id": P.id, "profile_sha256": P.sha256, "device_prefix": G["device_prefix"], "thresholds": {k: G[k] for k in ("min_cosine", "max_abs_diff")},
                                                         "thresholds_sha256": sha(cfgp.read_bytes()), "rows": [], "versions": _versions()}
    try:
        B = _bundle_dir(); FE, RQ, LAB = _gate_data(B); LF = RT.LF
        heads = {n: LF.build_head(B / "models/pretrained_31.safetensors", B / ("models/%s.pt" % n)) for n in LF.SEEDS}
        model, tok, dev = PC.load_model(P); doc["device"] = dev; cap = PC.make_capture_fn(P, model, tok); torch.cuda.reset_peak_memory_stats()
        find = lambda did: next(n for n in ("v2", "v3", "val") if did in FE[n])
        def ordered(did, hq, ho):
            lab = next(LAB[k][did] for k in ("v3", "val") if did in LAB[k]); ix = {o: i for i, o in enumerate(RQ[find(did)][did]["options"])}
            return hq, torch.stack([ho[ix[o]] for o in lab["options"]])
        for did in list(G["train_ids"]) + list(G["val_ids"]):
            n = find(did); sr, rq = FE[n][did], RQ[n][did]; role = "train" if did in G["train_ids"] else "val"
            assert sr["option_sha256"] == rq["option_sha256"] and sr["context_sha256"] == rq["context_sha256"], "stored record and request disagree: " + did
            hq, ho, info = cap(rq["context_text"], rq["options"])
            if hq is None: doc["rows"].append({"decision_id": did, "role": role, "status": info["status"], "passed": False, "info": info}); continue
            cm = RT.compare_features((hq, ho), (sr["hq"], sr["ho"]), {"min_cosine": G["min_cosine"], "max_abs_diff": G["max_abs_diff"]})
            per = [{"depth": P.depths[k], "hq_cos": F.cosine_similarity(hq[k].double(), sr["hq"][k].double(), 0).item(), "ho_cos_min": F.cosine_similarity(ho[:, k].double(), sr["ho"][:, k].double(), -1).min().item(),
                    "max_abs": max((hq[k].float() - sr["hq"][k].float()).abs().max().item(), (ho[:, k].float() - sr["ho"][:, k].float()).abs().max().item())} for k in range(len(P.depths))]
            a, b = ordered(did, hq, ho), ordered(did, sr["hq"], sr["ho"]); rec = {}
            for hn, h in heads.items():
                x, y = LF.head_logits(h, *a), LF.head_logits(h, *b)
                rec[hn] = {"same_top": max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__), "max_abs_logit_diff": max(abs(u - v) for u, v in zip(x, y))}
            ok = cm["passed"] and info["status"] == "ok" and info["depth_keys"] == list(G["require_depth_keys"]) and info["lm_head_calls"] == G["require_lm_head_calls"]
            doc["rows"].append({"decision_id": did, "role": role, "status": "ok", "n_options": ho.shape[0], "hq_shape": list(hq.shape), "ho_shape": list(ho.shape), "dtype": str(hq.dtype),
                                "cos_min": cm["cos_min"], "max_abs_diff": cm["max_abs_diff"], "passed": bool(ok), "bit_identical": bool(torch.equal(hq, sr["hq"]) and torch.equal(ho, sr["ho"])), "per_depth": per,
                                "info": info, "recorded_not_gating": rec})
        ok_rows = [r for r in doc["rows"] if r["status"] == "ok"]
        doc.update(n_records=len(doc["rows"]), worst_cos_min=min([r["cos_min"] for r in ok_rows] or [float("nan")]), worst_max_abs_diff=max([r["max_abs_diff"] for r in ok_rows] or [float("nan")]),
                   peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), verdict="PASS" if len(doc["rows"]) == len(G["train_ids"]) + len(G["val_ids"]) and all(r["passed"] for r in doc["rows"]) else "FAIL")
    except BaseException as e:      # noqa: BLE001
        import traceback; doc.update(verdict="ERROR", error=repr(e)[:500], trace=traceback.format_exc()[-3000:]); print(doc["trace"])
    doc["total_seconds"] = round(time.time() - t0, 1); write_over(out / "PARITY_RESULT.json", jdump(doc)); print("VERDICT", doc["verdict"]); return 0


def run_worker_process(kind: str, req_path: str, outdir: str) -> int:
    os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
    return {"gen": worker_gen, "score": worker_score, "parity": worker_parity}[kind](json.loads(Path(req_path).read_text("utf-8")), Path(outdir).resolve())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--offline-rehearsal", action="store_true"); g.add_argument("--final", action="store_true")
    ap.add_argument("--bundle"); ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--backbone", help="backbone profile id for the generating model (configs/backbones_v1.toml); only with --mode local")
    ap.add_argument("--mode", choices=["kaggle", "local", "replay4b"], default="kaggle")
    ap.add_argument("--variants-only", action="store_true", help="run the fault-injection variants (all bases, real controller + WSL sandbox, scripted model text) into <out>/VARIANTS_SUMMARY.json and exit")
    ap.add_argument("--from-run", help="PATH B: a stored acceptance run directory (kaggle_in + glm) to replay on exact stored contexts/options; only with --mode replay4b")
    ap.add_argument("--local-worker", nargs=3, metavar=("KIND", "REQUEST", "OUTDIR"), help="internal: run one local GPU worker (gen|score|parity) in this process")
    a = ap.parse_args(argv)
    if a.local_worker: return run_worker_process(*a.local_worker)
    if not a.out: ap.error("an output directory is required")
    if a.variants_only:
        import el_arch1_variants as V
        Path(a.out).mkdir(parents=True, exist_ok=True)
        s = V.run_all(Path(a.out))
        print(json.dumps({"all_ok": s["all_ok"], "n": sum(len(v) for v in s["variants"].values())}))
        return 0 if s["all_ok"] else 1
    if a.mode == "replay4b":
        if not a.from_run: ap.error("--mode replay4b needs --from-run <stored acceptance dir>")
        return Acceptance(Path(a.out), final=False, allow_dirty=a.allow_dirty, replay_from=Path(a.from_run), system_block=None, bundle=Path(a.bundle) if a.bundle else None).run()
    if a.mode == "local": return Acceptance(Path(a.out), final=False, allow_dirty=a.allow_dirty, local=True, backbone=a.backbone or ap.error("--mode local needs --backbone")).run()
    if a.backbone: ap.error("--backbone is used with --mode local")
    if not (a.final or a.offline_rehearsal): ap.error("one of --offline-rehearsal / --final / --mode local is required")
    if a.final and not a.bundle: ap.error("--final needs --bundle <results/raw/el-arch1-bundle-*>")
    return Acceptance(Path(a.out), final=a.final, bundle=a.bundle, allow_dirty=a.allow_dirty).run()


if __name__ == "__main__":
    sys.exit(main())
