"""[EL-ARCH-1 D1/S5] CONFORMANCE REPORT GENERATOR. Reads the receipts of one acceptance run directory (tools/el_arch1_acceptance.py output: receipts/, variants/, VARIANTS_SUMMARY.json, kaggle_in/, glm/)
and writes CONFORMANCE.json + CONFORMANCE.md (create-only) with ONE ROW PER COMPONENT: the 8 stages, I1-I8, L0-L5 evidence and scope, O1/O2/O3, R1/R2 (+D5), Qwen / frozen adapters / LAYA / probe /
random head, real controller / WSL sandbox / local GLM / repair loop, budget enforcement / safety / reporting, and every original audit finding F01-F28.
  python tools/el_arch1_conformance.py <acceptance out dir> [--tests-results <Kaggle TEST_RESULTS.json>]

STATUS (exactly one per row)
  REAL PASS       the receipts show the component working AND nothing it needs was a stub/mock/fake (the acceptance summary, each example receipt and each Kaggle record are all checked for `stub`)
  LOGIC-ONLY      the receipts show it working but a needed backend (or the whole run, for a --offline-rehearsal) was a stand-in: NEVER upgraded to REAL PASS
  IMPLEMENTED     the code exists (source symbol resolved below) but this run did not exercise it, or it is a code-level property
  NOT APPLICABLE  outside the scope of this closure, with the reason
  BLOCKED         a receipt the row needs is missing, the receipt shows a failure, a source symbol is missing, or the work is not done (the note names the smallest fix)
RULES IN CODE: a missing receipt is BLOCKED; a component whose `needs` include a backend flagged stub is LOGIC-ONLY; a rehearsal run (summary.stub_run) makes every receipt-based row LOGIC-ONLY.
Source locations are resolved at generation time from the static map SRC below ("path::Class.method" -> path:line by AST); an unresolvable symbol turns the row BLOCKED."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("intake", "investigate", "generate", "discriminate", "rank", "execute", "repair", "commit_verify")
REAL, LOGIC, IMPL, NA, BLOCKED = "REAL PASS", "LOGIC-ONLY", "IMPLEMENTED", "NOT APPLICABLE", "BLOCKED"
STATUSES = (REAL, LOGIC, IMPL, NA, BLOCKED)
A = "evidence_ladder/controller/agentic.py"
CT = "evidence_ladder/controller/controller.py"
INV = "evidence_ladder/controller/invariants.py"


def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ source resolution (static map -> file:line)
def resolve(sym: str) -> Optional[str]:
    path, _, name = sym.partition("::")
    p = ROOT / path
    if not p.is_file(): return None
    if not name: return "%s:1" % path
    tree = ast.parse(p.read_text("utf-8"))
    node: Any = tree
    for part in name.split("."):
        nxt = next((n for n in ast.iter_child_nodes(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.name == part), None)
        if nxt is None:      # module-level assignment
            nxt = next((n for n in ast.iter_child_nodes(node) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == part for t in n.targets)), None)
        if nxt is None:
            nxt = next((n for n in ast.iter_child_nodes(node) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == part), None)
        if nxt is None: return None
        node = nxt
    return "%s:%d" % (path, node.lineno)


# ------------------------------------------------------------------ the receipts of one run
class Missing(Exception):
    pass


class Receipts:
    def __init__(self, d: Path, tests_results: Optional[Path] = None):
        self.dir = Path(d)
        self.summary = self._j("receipts/ACCEPTANCE_SUMMARY.json", required=False)
        self.variants = self._j("VARIANTS_SUMMARY.json", required=False)
        self.tests_results = json.loads(Path(tests_results).read_text("utf-8")) if tests_results and Path(tests_results).is_file() else None
        self.tests_path = str(tests_results) if tests_results else None
        self._used: List[str] = []

    def _j(self, rel, required=True):
        p = self.dir / rel
        if not p.is_file():
            if required: raise Missing(rel)
            return None
        return json.loads(p.read_text("utf-8"))

    # --- accessors that record which files a row relied on
    def need(self, *rels: str) -> None:
        for r in rels:
            if not (self.dir / r).is_file(): raise Missing(r)
            self._used.append(r)

    def examples(self) -> Dict[str, Dict[str, Any]]:
        if self.summary is None: raise Missing("receipts/ACCEPTANCE_SUMMARY.json")
        self._used.append("receipts/ACCEPTANCE_SUMMARY.json")
        return self.summary["examples"]

    def journal(self, n: str) -> List[Dict[str, Any]]:
        rel = "receipts/%s/journal.json" % n
        self.need(rel)
        return json.loads((self.dir / rel).read_text("utf-8"))

    def kinds(self, n: str) -> List[str]: return [e["kind"] for e in self.journal(n)]

    def variant(self, name: str) -> Dict[str, Dict[str, Any]]:
        if self.variants is None: raise Missing("VARIANTS_SUMMARY.json")
        self._used.append("VARIANTS_SUMMARY.json")
        out = {}
        for b in self.variants["variants"].get(name, {}):
            rel = "variants/%s/%s/outcome.json" % (name, b)
            self.need(rel)
            out[b] = json.loads((self.dir / rel).read_text("utf-8"))
        if not out: raise Missing("variants/%s/*" % name)
        return out

    def variant_ok(self, *names: str) -> Tuple[bool, str]:
        bad = [(n, b) for n in names for b, o in self.variant(n).items() if o["violations"]]
        return (not bad), ("violations in %s" % bad) if bad else "%s: exact expected stop cause / label / journal entries on all 3 bases" % ",".join(names)

    def is_rehearsal(self) -> bool:
        """True for anything that is not a --final run: a rehearsal (stub_run) or a summary that does not say mode == final."""
        return bool(self.summary and (self.summary.get("stub_run") or self.summary.get("mode") != "final"))

    def stubbed(self, backend: str) -> bool:
        """True if ANY marker says this backend was a stand-in: summary flag, per-example flag, Kaggle/GLM record flags. (A rehearsal run as a whole is handled separately by is_rehearsal().)"""
        s = self.summary or {}
        b = (s.get("backends") or {}).get(backend)
        if b is None or b.get("stub") is not False: return True
        if backend in ("qwen", "laya", "probe", "random_head"):
            runs = (s.get("kaggle") or {}).get("gen_runs" if backend == "qwen" else "score_runs", {})
            flat = [v for r in runs.values() for v in (r.values() if backend != "qwen" else [r]) if isinstance(v, dict)]
            if not flat or any(v.get("stub") or "stub" in str(v.get("gpu", "")).lower() for v in flat): return True
        if backend == "glm" and (s.get("kaggle") or {}).get("glm", {}).get("stub"): return True
        return any(e.get("stub") for e in s.get("examples", {}).values())

    def sha_of(self, rel: str) -> Optional[str]:
        p = self.dir / rel
        return sha(p.read_bytes()) if p.is_file() else None

    def used(self) -> List[Dict[str, Any]]:
        return [{"path": r, "sha256": self.sha_of(r)} for r in sorted(set(self._used))]

    def reset(self) -> None: self._used = []


# ------------------------------------------------------------------ component specs
class Spec:
    def __init__(self, cid, group, title, src, needs=(), test: Optional[Callable] = None, static: Optional[Tuple[str, str]] = None, na: Optional[str] = None):
        self.id, self.group, self.title, self.src, self.needs, self.test, self.static, self.na = cid, group, title, list(src), tuple(needs), test, static, na


def any_example(R, pred, what) -> Tuple[bool, str]:
    ex = R.examples()
    hit = [n for n, e in ex.items() if pred(n, e)]
    return bool(hit), "%s in %s" % (what, ",".join(hit) if hit else "no example")


def all_examples(R, pred, what) -> Tuple[bool, str]:
    ex = R.examples()
    bad = [n for n, e in ex.items() if not pred(n, e)]
    return not bad, ("%s in all %d examples" % (what, len(ex))) if not bad else "%s FAILED in %s" % (what, bad)


def kinds_have(R, kind, minimum=1) -> Tuple[bool, str]:
    cnt = {n: R.kinds(n).count(kind) for n in R.examples()}
    return all(v >= minimum for v in cnt.values()), "journal entry %r per example: %s" % (kind, cnt)


def stage_test(stage):
    def t(R):
        ex = R.examples()
        for n, e in ex.items():      # the stage map in the summary must agree with the example's own journal file (a missing journal is a missing receipt)
            j = R.journal(n)
            sm = e["stage_map"]
            if sm["entries"] != len(j) or sum(b - a + 1 for rs in sm["stages"].values() for a, b in rs) != len(j):
                return False, "stage map of %s does not cover its journal (%d entries)" % (n, len(j))
        return any_example(R, lambda n, e: stage in e["stage_map"]["stages"], "stage %s reached (ranges %s)" % (stage, {n: e["stage_map"]["stages"].get(stage) for n, e in ex.items()}))
    return t


def evidence_records(R, pred) -> int:
    c = 0
    for n in R.examples():
        c += sum(1 for e in R.journal(n) if e["kind"] == "evidence" and pred(e["data"]["evidence"]))
    return c


def invariant_test(inv):
    def t(R):
        ok, note = all_examples(R, lambda n, e: inv in e["invariants"], "assert_invariants ran %s" % inv)
        v_ok, v_note = R.variant_ok("blocked_action", "budget_rounds", "invalid_model_evidence")
        return ok and v_ok, note + "; " + v_note
    return t


def tiers_test(tier, variants):
    def t(R):
        if variants:
            ok, note = R.variant_ok(*variants)
            return ok and all(o["tier"] == tier for v in variants for o in R.variant(v).values()), note
        return all_examples(R, lambda n, e: e["tier"] == tier, "tier %s" % tier)
    return t


def spec_list() -> List[Spec]:
    S: List[Spec] = []
    add = lambda *a, **k: S.append(Spec(*a, **k))
    # ---- 8 stages
    add("ST1", "stage", "1 INTAKE (requirements R1/R2, tier, routing table, risk table hash)", [A + "::AgenticRun.begin", "evidence_ladder/controller/tiers.py::route_tier"], ["controller", "sandbox"], stage_test("intake"))
    add("ST2", "stage", "2 INVESTIGATE (deterministic: read sources, run existing tests, parse failure)", ["evidence_ladder/controller/investigate.py::investigate"], ["controller", "sandbox"], stage_test("investigate"))
    add("ST3", "stage", "3 GENERATE (frozen action prompt, strict parse, Qwen3-4B-Thinking on Kaggle T4)", ["evidence_ladder/controller/generation.py::ActionProposer", "evidence_ladder/models/qwen.py::HFQwenBackend"], ["controller", "qwen"], stage_test("generate"))
    add("ST4", "stage", "4/5 DISCRIMINATE (policy + real dry-run of every proposed action in a scratch copy)", [A + "::AgenticRun.step"], ["controller", "sandbox"], stage_test("discriminate"))
    add("ST5", "stage", "6 RANK (trusted evidence first, LAYA L0 last, monotonicity gate journaled, abstain)", ["evidence_ladder/controller/next_action.py::decide_next_action", "evidence_ladder/controller/monotonicity.py::monotonicity_gate"], ["controller", "laya"], stage_test("rank"))
    add("ST6", "stage", "4 EXECUTE (commit gate, WSL execution, protected test on the real end state)", [A + "::AgenticRun._observe", "evidence_ladder/controller/wsl_sandbox.py::WslSandbox"], ["controller", "sandbox"], stage_test("execute"))
    def repair_t(R):
        ok, note = stage_test("repair")(R)
        return (ok, note) if ok else (None, "no example needed a repair round in this run (all solved in round 1): the stage was not exercised")
    S.append(Spec("ST7", "stage", "7 REPAIR (round >= 2 from the observed failure, same parts)", [A + "::AgenticRun.step", CT + "::_Run.repair_round"], ("controller", "qwen", "laya"), repair_t))
    add("ST8", "stage", "8 COMMIT + VERIFY (finalize on the real end state, final_report, decision, stop, I1-I8)", [A + "::AgenticRun.finish", "evidence_ladder/controller/final_verify.py::final_report"], ["controller", "sandbox"], stage_test("commit_verify"))
    # ---- invariants
    for i, nm in enumerate(["authority", "monotone safety", "no self-authorisation", "unknown is not pass", "bounded", "observe before commit", "frozen scorers + immutable journal", "sealed evaluation (runtime isolation)"], 1):
        add("I%d" % i, "invariant", "I%d %s" % (i, nm), [INV + "::check_i%d" % i, INV + "::assert_invariants"], ["controller", "sandbox"], invariant_test("I%d" % i))
    # ---- evidence levels and scope
    add("L0", "evidence", "L0 scores/opinions (LAYA, probe, random head, critic) order effort only", ["evidence_ladder/controller/next_action.py::decide_next_action"], ["controller", "laya"],
        lambda R: kinds_have(R, "ranking"))
    add("L1", "evidence", "L1 behavioural agreement (ladder controller clusters on derived inputs)", [CT + "::_Run.discriminate_stage", CT + "::_Run.ckey"], static=(IMPL, "ladder-mode (function-level) feature; unit-tested LOGIC-ONLY (phaseB t_b6); the toy per-action run discriminates by real dry-run instead (ST4)"))
    add("L2", "evidence", "L2 validated weak-model checks (quote span, mutants, consistency)", ["evidence_ladder/controller/l2.py::validate_checks"], static=(IMPL, "ladder-mode feature; unit-tested LOGIC-ONLY (phaseB t_b6); no Qwen-written checks were requested in the toy run"))
    add("L3", "evidence", "L3 examples quoted from the request (tier O2 evidence)", ["evidence_ladder/controller/tiers.py::count_examples"], ["controller"], lambda R: R.variant_ok("examples_only_o2"))
    add("L4", "evidence", "L4 certified checks (protected_files_unchanged_v1, diff_scope_v1) inside their scope", [A + "::check_defs"], ["controller", "sandbox"],
        lambda R: (evidence_records(R, lambda e: e["source"] == "certified_check" and e["status"] == "pass") > 0, "certified_check pass records: %d" % evidence_records(R, lambda e: e["source"] == "certified_check" and e["status"] == "pass")))
    add("L5", "evidence", "L5 real environment observation (protected test on the real end state)", [A + "::AgenticRun._observe"], ["controller", "sandbox"],
        lambda R: (evidence_records(R, lambda e: e["check_id"] == "protected_tests_v1" and e["status"] == "pass") >= len(R.examples()), "protected_tests_v1 pass records: %d" % evidence_records(R, lambda e: e["check_id"] == "protected_tests_v1" and e["status"] == "pass")))
    add("Lscope", "evidence", "evidence scope + provenance (requirement/check version, artifact hash, environment, scope text) on every record", ["evidence_ladder/controller/evidence.py::Factory"], ["controller"],
        lambda R: (all(all(e["data"]["evidence"].get(k) for k in ("scope", "artifact_hash", "environment_revision", "requirement_version", "check_version")) for n in R.examples() for e in R.journal(n) if e["kind"] == "evidence"), "all evidence records carry scope/artifact_hash/environment_revision/versions"))
    # ---- tiers, R1, R2, D5
    add("O1", "tier", "O1 trusted executable oracle -> generate-execute-repair, ceiling VERIFIED", ["evidence_ladder/controller/tiers.py::classify"], ["controller"], tiers_test("O1", None))
    add("O2", "tier", "O2 partial oracle (examples) -> pointer to the ladder controller, ceiling SUPPORTED", ["evidence_ladder/controller/tiers.py::classify", A + "::AgenticRun._finish_no_oracle"], ["controller", "sandbox"], tiers_test("O2", ["examples_only_o2"]))
    add("O3", "tier", "O3 no oracle -> best effort / ask_user on blocking ambiguity, BLOCKED", ["evidence_ladder/controller/tiers.py::route_tier"], ["controller", "sandbox"], tiers_test("O3", ["no_tests_o3", "missing_information"]))
    add("R1.1", "revision", "R1.1 evidence per requirement (rows), weakest-row label", ["evidence_ladder/controller/decision.py::decide_stage1"], ["controller"], lambda R: kinds_have(R, "label"))
    add("R1.2", "revision", "R1.2 router re-run after evidence-adding steps; tier upgraded only by pre-existing unmodified files", ["evidence_ladder/controller/tiers.py::observe_facts", A + "::AgenticRun.retier"], ["controller"],
        lambda R: (all(R.kinds(n).count("routing") >= 3 for n in R.examples()), "routing entries per example: %s" % {n: R.kinds(n).count("routing") for n in R.examples()}))
    add("R1.3", "revision", "R1.3 repair continues on progress (stop causes)", [CT + "::_Run.after_round"], static=(IMPL, "ladder-mode logic, unit-tested; the per-action loop stops by solved / task.max_steps / limits (variants budget_*)"))
    add("R2.1", "revision", "R2.1 every action goes intake -> check -> commit (step_record per step)", [A + "::AgenticRun.step"], ["controller", "sandbox"], lambda R: kinds_have(R, "step_record"))
    add("R2.2", "revision", "R2.2 evidence levels are not a total order; scope-covering records only", ["evidence_ladder/controller/router.py::route"], ["controller"], lambda R: kinds_have(R, "routing", 2))
    add("R2.3", "revision", "R2.3 approval depends on risk (fixed action-risk table, user_confirmed + trusted pre-check)", ["evidence_ladder/controller/risk.py::authorize_action"], ["controller", "sandbox"],
        lambda R: R.variant_ok("blocked_action", "blocked_confirmed_without_precheck"))
    add("D5", "revision", "D5 critic = z-ai/glm-5.3-flash, blinded packets, diagnostic evidence only", [A + "::AgenticRun.critic_stage", "evidence_ladder/critic_batch/validator.py::validate_packet"], ["controller", "sandbox", "glm"],
        lambda R: all_examples(R, lambda n, e: e["critic_state"] in ("applied", "rejected", "unavailable") and (e["critic_result"] or {}).get("label_unchanged_by_critic", True), "critic stage ran, label unchanged by the critic"))
    # ---- models
    add("M-QWEN", "model", "Qwen3-4B-Thinking-2507 proposals (Kaggle T4, strict parser, prompt-hash replay)", ["evidence_ladder/models/qwen.py::HFQwenBackend", "kaggle/el_arch1_generate_v1.py"], ["qwen"],
        lambda R: any_example(R, lambda n, e: "ok" in e["proposals"], "a proposal round with status ok"))
    add("M-ADAPT", "model", "3 frozen Fable adapters + pretrained head scoring on correctly captured Qwen states", ["evidence_ladder/models/laya_runtime.py::VerifiedSavedScoreBackend", "evidence_ladder/models/laya_runtime.py::LiveScorer"], ["laya"],
        lambda R: (any(e["data"].get("provenance") for n in R.examples() for e in R.journal(n) if e["kind"] == "scorer"), "scorer entries carry capture provenance (model revision, recipe, depths, device, capture code hashes)"))
    add("M-LAYA", "model", "LAYA scorer used as L0 only; scorer_unavailable journaled, never a silent fallback", ["evidence_ladder/models/laya.py::LayaScorer"], ["laya"],
        lambda R: (sum(R.kinds(n).count("scorer") + R.kinds(n).count("scorer_unavailable") for n in R.examples()) >= len(R.examples()), "scorer / scorer_unavailable entries present"))
    add("M-PROBE", "model", "linear probe on the SAME pools (refit on exactly the 334 TRAIN pools, fingerprinted), comparison only", ["evidence_ladder/models/probe.py::refit_probe"], ["probe"],
        lambda R: any_example(R, lambda n, e: any(x["data"]["scores"].get("probe") for x in R.journal(n) if x["kind"] == "scorer_comparison"), "scorer_comparison with probe scores"))
    add("M-RAND", "model", "random head control (untrained seeded heads, same architecture) on the SAME pools, comparison only", ["evidence_ladder/models/random_head.py::build_random_heads"], ["random_head"],
        lambda R: any_example(R, lambda n, e: any(x["data"]["scores"].get("random_head") for x in R.journal(n) if x["kind"] == "scorer_comparison"), "scorer_comparison with random_head scores"))
    # ---- runtime
    add("RT-CTRL", "runtime", "real controller (AgenticRun in RecordedRun), journal replays identically offline", ["evidence_ladder/controller/replay.py::RecordedRun", "evidence_ladder/controller/replay.py::replay"], ["controller"],
        lambda R: all_examples(R, lambda n, e: e["replay_offline"]["identical"] and e["replay_offline"]["code_unchanged"], "offline replay identical against the committed code"))
    add("RT-WSL", "runtime", "WSL (systemd + bubblewrap) sandbox: every action and every end state executed there; live replay identical", ["evidence_ladder/controller/wsl_sandbox.py::WslSandbox"], ["sandbox"],
        lambda R: all_examples(R, lambda n, e: e["replay_live_wsl"]["identical"] and e["tool_runs"] > 0, "live WSL replay identical, sandbox runs > 0"))
    add("RT-GLM", "runtime", "local GLM critic through the project key system: response enters the controller, safe experiment executed", ["tools/el_arch1_glm_live.py::main", A + "::AgenticRun.critic_stage"], ["glm", "controller", "sandbox"],
        lambda R: any_example(R, lambda n, e: e["critic_state"] == "applied" and (e["glm_accounting"] or {}).get("totals", {}).get("calls", 0) >= 1, "critic applied with >= 1 accounted GLM call"))
    add("RT-REPAIR", "runtime", "repair loop with real Qwen proposals (round >= 2 only for unsolved examples)", [A + "::AgenticRun.step"], ["controller", "qwen", "laya"], repair_t)
    # ---- budget, safety, reporting, journal
    add("BUDGET", "cross", "budget enforcement: rounds / Qwen tokens / wall time stop as `budget` with the resource named; GLM accounting from ledgers", ["evidence_ladder/controller/limits.py::Limiter", "evidence_ladder/critic_batch/accounting.py::from_broker"], ["controller", "sandbox"],
        lambda R: R.variant_ok("budget_rounds", "budget_tokens", "budget_wall"))
    add("SAFETY", "cross", "safety: irreversible / network / protected-write proposals never run; confirmation alone does not authorize; protected files read-only", ["evidence_ladder/controller/risk.py::table_hash"], ["controller", "sandbox"],
        lambda R: R.variant_ok("blocked_action", "blocked_confirmed_without_precheck", "abstain_all_rejected", "scope_conflict"))
    add("FAULTS", "cross", "fault injection: O2/O3, blocked action, missing information, broken test tool (ERROR, not pass), abstention, invalid model evidence, budget exhaustion", ["tools/el_arch1_variants.py::VARIANTS"], ["controller", "sandbox"],
        lambda R: (R.need("VARIANTS_SUMMARY.json") or True) and (bool(R.variants and R.variants["all_ok"]), "all %d variant runs matched their expected outcome" % sum(len(v) for v in (R.variants or {"variants": {}})["variants"].values())))
    add("REPORT", "cross", "reporting: gates read INSUFFICIENT_DATA (never PASS) on 3 examples; NOT-sealed statement; outcome vs process fields; secret scan 0", ["evidence_ladder/orchestrator/gates.py::section_gates", "evidence_ladder/orchestrator/isolation.py::NOT_SEALED_STATEMENT"], ["controller"],
        lambda R: ((R.summary or {}).get("gates", {}).get("monotonicity", {}).get("status") == "INSUFFICIENT_DATA" and (R.summary or {}).get("secret_scan_hits") == [] and all(("final_report" in R.kinds(n)) for n in R.examples()),
                   "monotonicity INSUFFICIENT_DATA, secret scan hits 0, final_report per example"))
    add("JOURNAL", "cross", "immutable hash-chained journal + replay against the committed code (source committed, hashes recorded)", ["evidence_ladder/controller/journal.py::EvidenceJournal"], ["controller"],
        lambda R: (bool((R.summary or {}).get("preflight", {}).get("source_committed")) and all(e["replay_offline"]["code_unchanged"] for e in R.examples().values()), "source_committed and code_unchanged on replay"))
    add("TRUSTED-CONFLICT", "cross", "an actual trusted-evidence conflict (LAYA top option fails real execution, or tests pass while the diff scope fails)", [A + "::AgenticRun.step"], ["controller", "sandbox", "laya"],
        lambda R: (any(s.get("trusted_evidence_overrode_laya") for e in R.examples().values() for s in e["steps"]) or R.variant_ok("scope_conflict")[0],
                   "real LAYA-vs-trusted override seen: %s; scope_conflict variant ok: %s" % (any(s.get("trusted_evidence_overrode_laya") for e in R.examples().values() for s in e["steps"]), R.variant_ok("scope_conflict")[0])))
    # ---- original audit findings F01-F28
    F = lambda i, title, src, needs=(), test=None, static=None, na=None: S.append(Spec("F%02d" % i, "finding", title, src, needs, test, static, na))
    F(1, "T6 end-to-end loop completed with receipts", [A + "::AgenticRun.finish"], ["controller", "qwen", "laya", "sandbox"], lambda R: all_examples(R, lambda n, e: len(e["invariants"]) == 8, "end-to-end finish with I1-I8"))
    F(2, "laya_integration (T5/T7) suite run with T4_scores.json", ["evidence_ladder/models/tests/laya_integration_tests.py"], [], None, None, None)
    F(3, "driver is the Evidence-Ladder controller, not a mini controller", [A + "::AgenticRun", "tools/el_arch1_acceptance.py::Acceptance"], ["controller"], lambda R: all_examples(R, lambda n, e: "step_record" in R.kinds(n) and "registry" in R.kinds(n), "AgenticRun journal (registry, routing, step_record)"))
    F(4, "real trusted-vs-LAYA conflict (not injected)", [A + "::AgenticRun.step"], ["controller", "laya", "sandbox"], lambda R: (any(s.get("trusted_evidence_overrode_laya") for e in R.examples().values() for s in e["steps"]), "a real LAYA-top option failed real execution and was overridden"))
    F(5, "NONE/abstain: empty allowed set abstains, never a random pick", ["evidence_ladder/controller/next_action.py::decide_next_action"], ["controller", "sandbox"], lambda R: R.variant_ok("abstain_all_rejected", "abstain_no_options", "blocked_action"))
    F(6, "no silent LAYA -> id-order fallback (scorer / scorer_unavailable journaled)", [A + "::AgenticRun.step"], ["controller"], lambda R: (sum(R.kinds(n).count("scorer") + R.kinds(n).count("scorer_unavailable") for n in R.examples()) >= len(R.examples()), "journaled per step"))
    F(7, "probe and random-head controls as sibling slots on identical pools", ["evidence_ladder/orchestrator/comparison.py::comparison_report"], ["probe", "random_head", "laya"], lambda R: any_example(R, lambda n, e: e["comparison_entries"] >= 1, "scorer_comparison entries"))
    F(8, "arm E wired into the stage-1 (whole-program) controller", ["evidence_ladder/orchestrator/program_scorer.py::ProgramPoolScorer"], static=(IMPL, "wired and unit-tested LOGIC-ONLY (phaseC); not part of the toy per-action acceptance run, no real program-pool scores"))
    F(9, "Kaggle -> PC trust: two independent runs agree on hashes + scores; run ids recorded", ["evidence_ladder/models/laya_runtime.py::VerifiedSavedScoreBackend"], ["laya"],
      lambda R: (bool((R.summary or {}).get("kaggle", {}).get("score_runs")) and not (R.summary or {}).get("kaggle", {}).get("refused_pools"), "live+verify runs recorded, no pool refused"))
    F(10, "live-mode provenance is self-declared", ["evidence_ladder/models/laya_runtime.py::LiveScorer"], static=(BLOCKED, "NOT FIXED: no attestation computed by the capture implementation; mitigated only by the independent verify run (F09). Smallest fix: LiveScorer takes an attestation (shard sha256 list, device name) produced inside the kernel"))
    F(11, "evidence-producing code frozen: committed source, hashes in receipts", ["evidence_ladder/controller/replay.py::code_hashes"], ["controller"], lambda R: (bool((R.summary or {}).get("preflight", {}).get("source_committed") and (R.summary or {}).get("preflight", {}).get("commit")), "source committed, HEAD %s" % ((R.summary or {}).get("preflight", {}).get("commit") or "?")[:12]))
    F(12, "hidden dev cases NOT sealed: relabelled + static oracle scan", ["evidence_ladder/orchestrator/isolation.py::static_oracle_scan"], static=(IMPL, "relabelled (NOT_SEALED_STATEMENT in reports), runtime import hygiene + deep context scan; real sealing (unpublished hidden tests) is NOT done"))
    F(13, "real Qwen -> controller -> repair -> critic loop ran", [A + "::AgenticRun.finish"], ["controller", "qwen", "laya", "glm", "sandbox"], lambda R: all_examples(R, lambda n, e: e["critic_state"] == "applied" and "ok" in e["proposals"], "Qwen proposals ok and critic applied"))
    F(14, "behaviour clustering beyond public examples + critic trigger logged", [CT + "::_Run.discriminate_stage"], static=(IMPL, "ladder-mode, unit-tested LOGIC-ONLY (phaseB t_b6); not exercised by the toy per-action run"))
    F(15, "critic pass cannot raise a row / choose its target requirement", [CT + "::_Run.apply_critic", "evidence_ladder/controller/evidence.py::cap_model_status"], ["controller", "sandbox"], lambda R: R.variant_ok("invalid_model_evidence", "unsafe_critic_experiment"))
    F(16, "per-action literals and tautologies removed (recomputed I1, fixed risk table)", [INV + "::check_i1", "evidence_ladder/controller/risk.py::authorize_action"], ["controller", "sandbox"], lambda R: R.variant_ok("blocked_action", "blocked_confirmed_without_precheck"))
    F(17, "tiers, L2, monotonicity gate, gates, wall-time cap implemented", ["evidence_ladder/controller/monotonicity.py::monotonicity_gate", "evidence_ladder/orchestrator/gates.py::section_gates"], ["controller", "sandbox"],
      lambda R: (all(R.examples()[n]["monotonicity_entries"] for n in R.examples()) and R.variant_ok("budget_wall")[0], "monotonicity_gate journaled before each decision; wall-time variant ok"))
    F(18, "ensemble top-1 unmeasured", [], na="a performance measurement; outside EL-ARCH-1 (completion of architecture blocks, not performance)")
    F(19, "T7 tolerance sensitivity floor unmeasured", [], na="LAYA-INT-1 T7 measurement; outside EL-ARCH-1")
    F(20, "Qwen-proposed options captured and scored (not stored Fable pools)", ["kaggle/evidence_ladder_laya_t6_step_v1.py"], ["laya", "qwen"], lambda R: (bool(R.summary and R.summary["kaggle"]["score_runs"]), "scoring kernels ran on proposals of this run"))
    F(21, "proposals in thinking mode parsed by the strict parser (no closed-think prefill)", ["evidence_ladder/controller/generation.py::parse_actions"], ["qwen"], lambda R: any_example(R, lambda n, e: "ok" in e["proposals"], "strict-parsed proposal round"))
    F(22, "toy tasks prove wiring, not usefulness", [], na="stated limit of the three-example closure; no performance claim is made")
    F(23, "stop_cause(…,0,1) hard-coded budget args; duplicate repair templates", [CT + "::_Run.after_round"], static=(IMPL, "documented as intentional in code (budget enforced in loop()/limit_check()); the two repair templates are NOT merged"))
    F(24, "decision chain v4-v6 consistent", [], na="a consistency note, no code")
    F(25, "PLAN_LAYA_INT_1 'no NVIDIA calls' superseded", [], na="a documentation note")
    F(26, "positive verification of earlier receipts", [], na="a positive finding")
    F(27, "fable_capture v1 pin never the training path", [], na="a documented correction")
    F(28, "WSL execution suite 21/21 and Kaggle-jail results unverified", ["evidence_ladder/execution/backend.py"], static=(BLOCKED, "not re-run in this closure; smallest fix: run the 21-test WSL execution suite on the PC and attach its result"))
    return S


def grade(spec: Spec, R: Receipts, tests_results_ok: Optional[Tuple[bool, str]] = None) -> Dict[str, Any]:
    row = {"id": spec.id, "group": spec.group, "title": spec.title, "status": None, "note": "", "source": [], "receipts": [], "stub_backends": []}
    locs = [(s, resolve(s)) for s in spec.src]
    row["source"] = [l for _, l in locs if l]
    miss = [s for s, l in locs if l is None]
    if spec.na:
        row.update(status=NA, note=spec.na); return row
    if miss:
        row.update(status=BLOCKED, note="source symbol not found: %s" % miss); return row
    if spec.id == "F02":      # needs a Kaggle test-run result
        if R.tests_results is None: row.update(status=BLOCKED, note="missing receipt: Kaggle TEST_RESULTS.json (pass --tests-results); smallest fix: run the el-arch1 tests kernel"); return row
        ok = bool(R.tests_results.get("all_exit_zero")) and "laya_integration" in R.tests_results.get("suites", {})
        row.update(status=REAL if ok else BLOCKED, note="Kaggle TEST_RESULTS.json all_exit_zero=%s" % R.tests_results.get("all_exit_zero"), receipts=[{"path": R.tests_path, "sha256": sha(Path(R.tests_path).read_bytes())}])
        return row
    if spec.static:
        row.update(status=spec.static[0], note=spec.static[1]); return row
    R.reset()
    try:
        ok, note = spec.test(R)
    except Missing as m:
        row.update(status=BLOCKED, note="missing receipt: %s" % m); return row
    except Exception as ex:      # noqa: BLE001
        row.update(status=BLOCKED, note="receipt unreadable: %s: %s" % (type(ex).__name__, str(ex)[:120])); return row
    row["receipts"] = R.used()
    if ok is None:
        row.update(status=IMPL, note=note); return row
    if not ok:
        row.update(status=BLOCKED, note="FAILED: " + note); return row
    stubs = [b for b in spec.needs if R.stubbed(b)]
    if R.is_rehearsal(): stubs = sorted(set(stubs) | {"rehearsal_run"})
    row["stub_backends"] = stubs
    if stubs:
        row.update(status=LOGIC, note=note + " | exercised with stand-ins: %s" % ", ".join(stubs))
    else:
        row.update(status=REAL, note=note)
    return row


def generate(d: Path, tests_results: Optional[Path] = None) -> Dict[str, Any]:
    R = Receipts(d, tests_results)
    rows = [grade(s, R) for s in spec_list()]
    for r in rows: assert r["status"] in STATUSES
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    return {"tag": "[EL-ARCH-1 D1] conformance", "run_dir": str(d), "mode": (R.summary or {}).get("mode"), "rehearsal": R.is_rehearsal(), "commit": ((R.summary or {}).get("preflight") or {}).get("commit"),
            "counts": counts, "rows": rows}


def render_md(rep: Dict[str, Any]) -> str:
    L = ["# %s" % rep["tag"], "", "run: `%s` · mode: %s%s · commit: %s" % (rep["run_dir"], rep["mode"], " (REHEARSAL: stand-ins everywhere, nothing here is a REAL PASS)" if rep["rehearsal"] else "", rep["commit"]),
         "", "counts: " + ", ".join("%s %d" % (k, v) for k, v in rep["counts"].items()), "", "| id | group | component | status | source | receipts | note |", "|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        L.append("| %s | %s | %s | **%s** | %s | %s | %s |" % (r["id"], r["group"], r["title"].replace("|", "/"), r["status"], "<br>".join(r["source"]) or "-",
                                                              "<br>".join("%s `%s`" % (x["path"], (x["sha256"] or "")[:12]) for x in r["receipts"]) or "-", r["note"].replace("|", "/")))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--tests-results")
    a = ap.parse_args(argv)
    rep = generate(Path(a.dir), Path(a.tests_results) if a.tests_results else None)
    d = Path(a.dir)
    for name, txt in (("CONFORMANCE.json", json.dumps(rep, indent=1, ensure_ascii=False)), ("CONFORMANCE.md", render_md(rep))):
        with open(d / name, "x", encoding="utf-8", newline="\n") as f: f.write(txt)
    print(json.dumps(rep["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
