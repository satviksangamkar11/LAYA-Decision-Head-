"""[EL-ARCH-1 F2 / PATH B helper] Read-only access to the STORED verified Qwen3-4B + LAYA captures of the real T4 run
`results/raw/laya-integration-20261016T184816Z/e2e_pass2/_kaggle/{live_s1,verify_s1}` (step 1 of ex1_slug / ex2_clamp / ex3_wordcount, 4 options each).

PATH B rule: a stored score is used ONLY on the exact stored context text and the exact stored option list. `VerifiedSavedScoreBackend` keys by (sha256(context_text), tuple(sha256(option)));
a pool that differs by one option (e.g. the candidate filter removed an unappliable edit) is refused (ProvenanceError) and the controller journals `scorer_unavailable`. Nothing here writes a file.

    from evidence_ladder.controller import pathb_stored as PB
    live, backends, system_block = PB.load_stored_4b()                      # backends = {"laya": ..., "probe": ..., "random_head": ...} (VerifiedSavedScoreBackend each; live + replay-verify must agree)
    run, rec = PB.run_exact_pool("ex1_slug:1", live, backends, system_block, sandbox)   # one AgenticRun.step on the stored pool; NO INVESTIGATE report (the stored context has none)
    run.journal.entries("scorer")   # LAYA scores (L0); probe / random_head are comparison-only (`scorer_comparison`, selector False)

Exact context: messages = [system block, user request]; text = decision_head.fable_pool.select_context(messages, 160000). `stored_context_text(key, system_block)` rebuilds it and
`context_matches(...)` states whether its sha equals the stored `context_sha256` (True for all three step-1 pools on the current tree, measured 2026-10-18).
The system block is read from the newest `results/raw/el-arch1-bundle-*/dataset/gate/system_block.txt` (the file the stored run was launched with)."""
from __future__ import annotations

import glob
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
STORED = ROOT / "results/raw/laya-integration-20261016T184816Z/e2e_pass2/_kaggle"
TASK_OF = {"ex1_slug": "slug", "ex2_clamp": "clamp", "ex3_wordcount": "wordcount"}


class StoredCaptureMissing(Exception):
    pass


def _tasks():
    sys.path.insert(0, str(ROOT / "tools"))
    from laya_int1_t6 import TASKS
    return TASKS


def system_block() -> str:
    c = sorted(glob.glob(str(ROOT / "results/raw/el-arch1-bundle-*/dataset/gate/system_block.txt")))
    if not c:
        raise StoredCaptureMissing("no system_block.txt of the stored run")
    return Path(c[-1]).read_text("utf-8")


def load_stored_4b() -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """-> (live results doc, {"laya","probe","random_head": VerifiedSavedScoreBackend}, system block). Raises StoredCaptureMissing; a pool the backend refuses raises AssertionError."""
    from evidence_ladder.models import laya_runtime as RT
    lp, vp = STORED / "live_s1/output/t6_step_results.json", STORED / "verify_s1/output/t6_step_results.json"
    if not (lp.exists() and vp.exists()):
        raise StoredCaptureMissing("stored 4B captures not on this machine: %s" % STORED)
    live, ver = json.loads(lp.read_text("utf-8")), json.loads(vp.read_text("utf-8"))
    backends = {k: RT.VerifiedSavedScoreBackend(k) for k in ("laya", "probe", "random_head")}
    for key in live["steps"]:
        for be in backends.values():
            assert be.add(live["steps"][key], ver["steps"][key], live_run="e2e/live_s1", verify_run="e2e/verify_s1", verify_mode=ver["mode"]) is True, (key, be.refused)
    return live, backends, system_block()


def stored_context_text(key: str, sysblock: str) -> str:
    from decision_head import fable_pool as FP
    T = _tasks()[TASK_OF[key.split(":")[0]]]
    return FP.select_context([{"role": "system", "content": sysblock}, {"role": "user", "content": T["user"]}], 160000)[0]


def context_matches(key: str, live: Dict[str, Any], sysblock: str) -> bool:
    from evidence_ladder.common.interfaces import sha256_text
    return sha256_text(stored_context_text(key, sysblock)) == live["steps"][key]["context_sha256"]


def run_exact_pool(key: str, live: Dict[str, Any], backends: Dict[str, Any], sysblock: str, sandbox=None, workdir: Optional[Path] = None):
    """Build the toy repo of `key`, begin() an AgenticRun with the stored-score LAYA scorer (+ probe / random-head comparison scorers) and run ONE step on the stored options.
    -> (AgenticRun, step record). `sandbox` default: the qualified WslSandbox. The run is never investigated (PATH B contexts are the stored ones)."""
    from decision_head import fable_pool as FP
    from evidence_ladder.models import laya
    from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask
    T = _tasks()[TASK_OF[key.split(":")[0]]]
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="pathb-"))
    ws = base / "ws"
    ws.mkdir(parents=True)
    for fn, c in T["files"].items():
        (ws / fn).write_text(c, "utf-8")
    if sandbox is None:
        from evidence_ladder.controller.sandbox_provider import make_sandbox      # [MXC] LAYA_SANDBOX_BACKEND (default wsl)
        sandbox = make_sandbox(base / "_sandbox", T["protected"])
    scorer = laya.LayaScorer(authorized=True, backend=backends["laya"])
    cmp_ = {"probe": laya.ProbeScorer(authorized=True, backend=backends["probe"]), "random_head": laya.RandomHeadScorer(authorized=True, backend=backends["random_head"])}
    task = AgenticTask(key.split(":")[0], T["user"], ws, T["test_cmd"], T["protected"], sysblock, 2)
    run = AgenticRun(task, AgenticContext(sandbox=sandbox, scorer=scorer, arm="E", config={}, comparison=cmp_), lambda m: FP.select_context(m, 160000)[0])
    run.begin()
    return run, run.step(live["steps"][key]["options"], {})
