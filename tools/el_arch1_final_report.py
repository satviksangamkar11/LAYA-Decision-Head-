"""[EL-ARCH-1 F3 / T47 T48 T49] ARCHITECTURE_FINAL_REPORT.md GENERATOR. The coordinator runs it after the final runs; nothing here executes a model.

  PYTHONPATH=. .venv/Scripts/python.exe tools/el_arch1_final_report.py --path-a <PATH A run dir> [--path-b <PATH B run dir>] [--variants <dir with VARIANTS_SUMMARY.json>]
        [--suite-results <SUITE_RESULTS.json from tools/el_arch1_run_suites.py>] [--out ARCHITECTURE_FINAL_REPORT.md]

Rows: ONE per T-id of EL_ARCH_TRACEABILITY.md (T01-T08 are the invariants I1-I8), columns  Component | Implementation | Real test | Negative test | Receipt | Verdict.
Verdict is one of PASS / FAIL / NOT_APPLICABLE / INSUFFICIENT_DATA and is computed by verdict() below, nowhere else:
  NOT_APPLICABLE     the registry says so, with its reason
  FAIL               a referenced test failed in the recorded suite run, or the receipt check ran and failed
  INSUFFICIENT_DATA  anything missing or weaker than a real pass: no real-test reference, no negative-test reference, no receipt reference, a receipt that shows a stand-in (mock-only), a component the
                     run did not exercise, a statistical gate with fewer than MIN_EXAMPLES_FOR_STATISTICS examples, a development run that is not the final path
  PASS               real test AND negative test AND receipt, all resolved, all passing, nothing stubbed
References are RESOLVED, not trusted: a test name must exist (AST of the test files) and appear in the suite-results file as part of a suite that ran with exit 0; a variant name must be in
VARIANTS_SUMMARY.json with every base ok; a receipt must be a file that exists (its sha256 is printed). The two paths are separate sections and separate verdicts (PATH A: local 0.6B live
controller; PATH B: frozen 4B LAYA replay on exact stored contexts and options); a row that needs both is PASS only if both pass. The toy-repo protected tests are development tests (UNSEALED)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import el_arch1_conformance as CF                                  # noqa: E402

MIN_EXAMPLES_FOR_STATISTICS = 30
VERDICTS = ("PASS", "FAIL", "NOT_APPLICABLE", "INSUFFICIENT_DATA")
TEST_FILES = ["evidence_ladder/controller/tests/phase%s_tests.py" % x for x in ("A", "B", "D1", "F1", "F1b", "F2")] + [
    "evidence_ladder/orchestrator/tests/phaseC_tests.py", "evidence_ladder/orchestrator/tests/phaseF3_tests.py", "evidence_ladder/critic_batch/tests/stepE_tests.py",
    "evidence_ladder/critic_batch/tests/run_tests.py", "evidence_ladder/models/tests/run_tests.py", "evidence_ladder/models/tests/backbone_tests.py", "evidence_ladder/models/tests/laya_integration_tests.py",
    "evidence_ladder/controller/tests/run_tests.py", "evidence_ladder/orchestrator/tests/run_tests.py", "evidence_ladder/controller/tests/phaseF1b_tests.py"]
ROWMAP_FILES = ("EL_ARCH_ROWMAP_A1.json", "EL_ARCH_ROWMAP_A2.json")
# Conformance rows whose failure CONTRADICTS a requirement (a safety / integrity / replay / budget check that ran and failed). Any other failed receipt check is a POSITIVE path the final-path receipts did not
# exercise (e.g. no example reached VERIFIED) and reads INSUFFICIENT_DATA, not FAIL.
CONTRADICTION_IDS = {"I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "RT-CTRL", "SAFETY", "BUDGET", "FAULTS", "JOURNAL", "F05", "F15", "F16"}
POSITIVE_GAP = "positive path not exercised in the final-path receipts"
NEG_WORDS = re.compile(r"negative|refus|reject|never|cannot|blocked|insufficient|detect|tamper|forged|unsafe|not_|no_", re.I)


def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()


def E(conf=(), real=(), neg=(), vreal=(), vneg=(), paths="A", stat=False, na=None, receipt=(), src=(), owner=""):
    return dict(conf=list(conf), real=list(real), neg=list(neg), vreal=list(vreal), vneg=list(vneg), paths=paths, stat=stat, na=na, receipt=list(receipt), src=list(src), owner=owner)


A_ = "evidence_ladder/controller/agentic.py"
# ------------------------------------------------------------------ the registry: T-id -> how it is judged. (text of each requirement is read from EL_ARCH_TRACEABILITY.md)
REGISTRY: Dict[str, Dict[str, Any]] = {
    "T01": E(["I1"], ["t_b7_b8_full_run_on_the_real_wsl_sandbox_and_replay_wsl"], ["test_a5_i1_checks_the_journaled_decision_not_a_literal", "t_b1_I1_rejects_a_label_above_the_tier_ceiling"], owner="A2"),
    "T02": E([], ["test_agentic_real_executed_evidence_outranks_the_laya_top_choice"], ["test_top_laya_failing_trusted_evidence_is_never_selected", "test_invariant_negative_control_detects_a_reordering_across_evidence"], paths="B", receipt=["pathb_conflict"], owner="A2"),
    "T03": E(["I3"], ["test_a4_critic_named_requirement_is_ignored_and_example_row_pass_is_capped"], ["test_a4_critic_pass_cannot_lift_unverified_or_blocked_and_forged_records_fail_i3"], vneg=["invalid_model_evidence"], owner="A2"),
    "T04": E(["I4"], ["test_judge_statuses"], [], vneg=["broken_test_command", "broken_sandbox"], owner="A2+A3"),
    "T05": E(["I5", "BUDGET"], ["test_a3_agentic_wall_time_actions_rounds_tokens_stop_as_budget_with_the_resource_named"], [], vneg=["budget_rounds", "budget_tokens", "budget_wall", "budget_actions", "budget_steps"], owner="A2+A3"),
    "T06": E(["I6", "SAFETY"], ["t_b5_authorization_per_class"], ["t_b5_blocked_actions_never_reach_the_sandbox_and_unknown_is_denied", "t_b5_confirmed_irreversible_is_still_blocked_at_the_commit_gate_without_a_precheck"],
             vneg=["blocked_action", "blocked_confirmed_without_precheck", "unsafe_action_filtered"], owner="A2"),
    "T07": E(["I7", "JOURNAL"], ["test_journal_is_reproducible"], [], src=["tools/el_arch1_acceptance.py::config_hashes"], owner="A2+A3"),
    "T08": E(["I8"], ["t_static_controller_imports_stay_clean_and_modules_import_without_torch"], [], owner="A2+A3"),
    "T09": E(["L5"], ["t_b2_investigate_three_toy_repos_wsl"], [], owner="A2"),
    "T10": E(["L4"], [], [], owner="A2"),
    "T11": E(["L3"], [], [], vreal=["examples_only_o2"], owner="A2"),
    "T12": E(["L2"], ["t_b6_mutants_inputs_and_strict_check_parse"], ["t_b6_validate_checks_every_rule_has_a_reason"], receipt=["l2"], owner="A2"),
    "T13": E(["L1"], ["t_b6_ladder_l2_discrimination_clustering_and_critic_trigger_logic_only"], [], receipt=["l2"], owner="A2"),
    "T14": E(["L0"], ["test_l0_orders_equal_evidence_candidates_and_never_approves"], ["test_trusted_evidence_outranks_any_laya_score", "test_real_scores_never_override_trusted_evidence"], paths="AB", receipt=["pathb_conflict"], owner="A2"),
    "T15": E(["R1.1"], [], [], owner="A2"),
    "T16": E(["REPORT"], [], [], owner="A2+A3"),
    "T17": E(["O1"], ["t_b1_three_examples_route_to_O1"], [], owner="A2"),
    "T18": E(["O2"], ["t_b1_agentic_o2_without_test_oracle_points_to_the_ladder"], [], vreal=["examples_only_o2"], vneg=[], owner="A3+A2"),
    "T19": E(["O3"], ["t_b1_agentic_o3_asks_the_user_and_takes_no_action"], [], vreal=["no_tests_o3", "missing_information"], owner="A3+A2"),
    "T20": E(["R1.2"], ["t_b1_model_written_tests_never_upgrade_the_tier"], [], owner="A2"),
    "T21": E(["ST1"], [], [], owner="A2"),
    "T22": E(["ST2"], ["t_b2_investigate_records_L5_observation_and_context_logic_only"], [], owner="A2"),
    "T23": E(["ST3", "M-QWEN"], ["t_b3_action_proposer_feeds_the_agentic_step_logic_only"], [], vreal=["malformed_json", "sequential_plan"], owner="A1+A2"),
    "T24": E([], [], [], vreal=["malformed_json", "sequential_plan", "duplicate_options"], vneg=["unsafe_action_filtered", "empty_pool"], owner="A1+A2"),
    "T25": E(["RT-WSL"], ["t_b7_b8_full_run_on_the_real_wsl_sandbox_and_replay_wsl"], [], owner="A2"),
    "T26": E(["ST4", "D5"], [], ["test_controller_rejects_unsafe_experiment_explicitly_and_never_runs_it"], vneg=["unsafe_critic_experiment"], owner="A2+A3"),
    "T27": E(["ST5"], ["test_agentic_laya_orders_what_trusted_evidence_leaves_and_may_change_the_id_order_choice"], ["test_trusted_evidence_outranks_any_laya_score"], paths="AB", receipt=["pathb_conflict"], owner="A2"),
    "T28": E(["ST7", "RT-REPAIR"], [], [], owner="A2"),
    "T29": E(["ST8"], [], [], owner="A2"),
    "T30": E(["R2.1"], [], [], owner="A2"),
    "T31": E(["R2.3"], ["t_b5_authorization_per_class"], [], vneg=["blocked_confirmed_without_precheck"], owner="A2"),
    "T32": E(["F06"], ["test_scorer_unavailable_is_explicit_never_silent", "test_agentic_missing_scorer_is_journaled_not_silent_and_step_limit_stops"], [], vneg=["missing_scorer", "missing_scorer_required"], owner="A2+A3"),
    "T33": E(["M-LAYA"], ["test_laya_scorer_interface_wiring"], [], owner="A2"),
    "T34": E([], ["test_verified_saved_scores_need_two_agreeing_runs_and_feed_the_laya_scorer"], ["t_T34_T49_pool_admission_needs_exact_context_and_options_never_cross_paths", "test_saved_score_backend_refuses_unproven_records"],
             paths="B", receipt=["pathb_laya"], src=["tools/el_arch1_acceptance.py::pool_admissible", "tools/el_arch1_pathb_replay.py::run"], owner="A3+A2"),
    "T35": E([], ["test_4b_provenance_accepted_and_legacy_record_without_profile_accepted"], ["test_head_incompatible_profile_states_are_refused_with_a_clear_message", "test_layascorer_refuses_06b_states_with_the_reason_on_record",
              "test_guard_refuses_wrong_model_revision_dtype_depths_device_recipe_hash"], receipt=["suites"], src=["evidence_ladder/models/laya.py::LayaScorer"], owner="A3"),
    "T36": E([], ["test_genuine_batch_is_accepted_and_scored"], ["test_fabricated_tensors_with_perfect_metadata_are_caught_by_recapture_not_metadata", "test_single_fabricated_record_anywhere_is_caught_at_100_percent"],
             receipt=["suites"], paths="AB", owner="A3"),
    "T37": E([], ["test_probe_and_random_head_interfaces"], ["test_saved_score_backend_refuses_unproven_records"], paths="B", receipt=["pathb_probe_random"], src=["tools/el_arch1_pathb_replay.py::run"], owner="A3"),
    "T38": E(["F17"], [], [], stat=True, paths="AB", owner="A2"),
    "T39": E([], [], [], receipt=["suites"], owner="A2"),
    "T40": E([], ["t_T40_packet_blinding_positive_and_negative", "test_i8_packet_is_blinded"], ["t_T40_packet_blinding_positive_and_negative", "test_blinding_rejections"], receipt=["suites"],
             src=["evidence_ladder/critic_batch/validator.py::validate_packet"], owner="A3"),
    "T41": E(["D5"], ["test_controller_executes_supported_experiment_and_label_unchanged", "t_T41_experiment_types_validated_and_only_allowed_ops_planned"],
             ["test_unsupported_kinds_rejected", "test_unsafe_commands_rejected", "t_T41_critic_never_authorises_label_unchanged"], vneg=["unsafe_critic_experiment", "invalid_model_evidence"], owner="A3+A2"),
    "T42": E([], ["t_T42_reconcile_the_real_local_ledger", "t_T42_success_run_through_real_runner_reconciles_with_actual_tokens"], ["t_T42_negative_tampered_ledgers_fail_reconciliation",
              "t_T42_transport_failures_are_classified_retries_bounded_no_key_in_files", "t_T42_utf8_stream_roundtrip_and_mojibake_negative"], receipt=["glm_reconcile"], src=["evidence_ladder/critic_batch/reconcile.py::reconcile"], owner="A3"),
    "T43": E([], ["t_T43_config_and_source_hashes_present_and_change_with_content"], ["t_T43_dirty_tree_is_refused_on_the_final_path_only", "t_T43_secret_scan_counts_only_added_lines"], receipt=["provenance"],
             paths="AB", src=["tools/el_arch1_acceptance.py::Acceptance.preflight", "tools/el_arch1_acceptance.py::added_lines_secret_scan"], owner="A3+C"),
    "T44": E([], ["t_T44_mock_backends_refused_in_the_acceptance_path_and_real_ones_named"], ["t_T44_mock_backends_refused_in_the_acceptance_path_and_real_ones_named"], receipt=["backend_identity"],
             paths="AB", src=["tools/el_arch1_acceptance.py::refuse_mock_backends"], owner="A3"),
    "T45": E(["RT-CTRL", "RT-WSL"], ["test_agentic_replay_gives_identical_journal_hashes", "t_b8_replay_offline_logic_only_and_step_records"], ["t_b8_replay_offline_logic_only_and_step_records"], paths="A", owner="A3+C"),
    "T46": E(["FAULTS"], ["t_T46_malformed_sequential_duplicate_empty_unsafe_missing_scorer_variants"], ["t_T46_negative_a_variant_that_does_not_reach_its_expected_state_is_reported"],
             vreal=["malformed_json", "sequential_plan", "duplicate_options", "empty_pool", "unsafe_action_filtered", "missing_scorer", "broken_test_command", "budget_rounds", "examples_only_o2", "no_tests_o3", "scope_conflict"],
             src=["tools/el_arch1_variants.py::VARIANTS"], owner="A3"),
    "T47": E(["REPORT"], ["t_T47_statistical_rows_stay_insufficient_with_three_examples"], ["t_T48_generator_refuses_pass_without_real_negative_and_receipt"], stat=True, paths="A", src=["evidence_ladder/orchestrator/gates.py::section_gates"], owner="A3"),
    "T48": E([], ["t_T48_one_row_per_traceability_id_and_per_invariant"], ["t_T48_generator_refuses_pass_without_real_negative_and_receipt"], receipt=["self"], src=["tools/el_arch1_final_report.py::verdict"], owner="A3"),
    "T49": E([], ["t_T34_T49_pool_admission_needs_exact_context_and_options_never_cross_paths"], ["t_T34_T49_pool_admission_needs_exact_context_and_options_never_cross_paths"], receipt=["paths_separate"],
             paths="AB", src=["tools/el_arch1_acceptance.py::pool_admissible"], owner="A3+C"),
    "T50": E([], ["test_4b_profile_equals_the_former_hardcoded_constants", "test_06b_profile_values_and_proportional_depths"], ["test_guard_refuses_wrong_model_revision_dtype_depths_device_recipe_hash"], receipt=["suites"], owner="C"),
}
MIN_EX = MIN_EXAMPLES_FOR_STATISTICS


# ------------------------------------------------------------------ the one place a verdict is decided
def verdict(d: Dict[str, Any]) -> Tuple[str, str]:
    """d: real/neg/receipt = lists of RESOLVED references; conf_ok = True/False/None (None: the run did not exercise it); stat; na; failed = failing test names; stub = a stand-in was used."""
    if d.get("na"): return "NOT_APPLICABLE", d["na"]
    if d.get("failed"): return "FAIL", "failing test(s): %s" % ", ".join(d["failed"][:4])
    if d.get("conf_ok") is False: return "FAIL", "the receipt check ran and contradicts the requirement"
    if d.get("missing"): return "INSUFFICIENT_DATA", "declared test(s) not found or not run: %s" % ", ".join(d["missing"][:4])      # a declared-but-missing test is never silently dropped
    gaps = [n for n, k in (("no resolved real-test reference", d.get("real")), ("no resolved negative-test reference", d.get("neg")), ("no receipt reference", d.get("receipt"))) if not k]
    if gaps: return "INSUFFICIENT_DATA", "; ".join(gaps)
    if d.get("stub"): return "INSUFFICIENT_DATA", "the receipt shows a stand-in (mock-only is never a PASS)"
    if d.get("positive_gap"): return "INSUFFICIENT_DATA", POSITIVE_GAP + (": " + d["positive_gap"] if isinstance(d["positive_gap"], str) else "")
    if d.get("conf_ok") is None: return "INSUFFICIENT_DATA", "the run did not exercise this component (code exists, no receipt of it working)"
    if d.get("stat"): return "INSUFFICIENT_DATA", "statistical gate with fewer than %d examples" % MIN_EXAMPLES_FOR_STATISTICS
    return "PASS", "real test + negative test + receipt, all resolved and passing"


# ------------------------------------------------------------------ inputs
def traceability(path: Path = ROOT / "EL_ARCH_TRACEABILITY.md") -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for line in path.read_text("utf-8").splitlines():
        m = re.match(r"\|\s*(T\d\d)\s*\|(.+)", line)
        if m:
            cells = [c.strip() for c in m.group(2).split("|")]
            out[m.group(1)] = {"requirement": cells[0], "owner": cells[1] if len(cells) > 1 else "", "acceptance": cells[2] if len(cells) > 2 else ""}
    return out


def traceability_ids() -> List[str]: return list(traceability())


def test_index() -> Dict[str, str]:
    """test function name -> 'path::name' for every test function in the known suites (first definition wins)."""
    idx: Dict[str, str] = {}
    for rel in TEST_FILES:
        p = ROOT / rel
        if p.is_file():
            for n in ast.walk(ast.parse(p.read_text("utf-8"))):
                if isinstance(n, ast.FunctionDef) and (n.name.startswith("test_") or n.name.startswith("t_")):
                    idx.setdefault(n.name, "%s::%s" % (rel, n.name))
    return idx


def auto_tagged(tid: str, idx: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Tests that carry the T-id in their name (convention for the new F1/F2/F3 tests): negative if the name says so, real otherwise; 'positive_and_negative' counts for both."""
    real, neg = [], []
    for n in idx:
        if re.search(r"(^|_)%s(_|$)" % tid, n) or ("_" in n and re.search(r"_%s_" % tid, n)):
            isneg = bool(NEG_WORDS.search(n.replace(tid, "")))
            if not isneg or "positive" in n: real.append(n)
            if isneg: neg.append(n)
    return real, neg


def load_rowmaps(root: Path = ROOT, files: Sequence[str] = ROWMAP_FILES) -> Dict[str, Dict[str, Any]]:
    """Merge the rows the other agents declare: {T-id: {real:[file::test], negative:[file::test], receipt:[{path, what}], notes}} (several files: lists are concatenated)."""
    out: Dict[str, Dict[str, Any]] = {}
    for fn in files:
        fp = Path(root) / fn
        if not fp.is_file(): continue
        for tid, e in json.loads(fp.read_text("utf-8")).items():
            m = out.setdefault(tid, {"real": [], "negative": [], "receipt": [], "notes": []})
            m["real"] += e.get("real", []); m["negative"] += e.get("negative", []); m["receipt"] += e.get("receipt", [])
            if e.get("notes"): m["notes"].append(str(e["notes"]))
    return out


GAP_MARKERS = re.compile(r"no positive receipt|not exercised|no example reached|no receipt reference|negative case only|no LAYA scores", re.I)


class Suites:
    def __init__(self, path: Optional[Path]):
        self.path, self.doc = path, (json.loads(path.read_text("utf-8")) if path and path.is_file() else None)

    def status(self, test_ref: str) -> str:
        """'passed' | 'failed' | 'unrun' for 'path::name': the file's own suite must have been run (tools/el_arch1_run_suites.py) and parsed."""
        if not self.doc: return "unrun"
        file, _, name = test_ref.partition("::")
        for s in self.doc["suites"]:
            if s["file"] == file and s.get("parsed"):
                return "failed" if name in s["failed"] else "passed"
        return "unrun"

    def state(self, test_ref: str) -> str:
        """'passed' | 'failed' | 'unrun' | 'missing' (the function does not exist in that file)."""
        file, _, name = test_ref.partition("::")
        p = ROOT / file
        if not p.is_file() or not any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(ast.parse(p.read_text("utf-8")))): return "missing"
        return self.status(test_ref)

    def receipt(self) -> Optional[Dict[str, str]]:
        return {"path": str(self.path), "sha256": sha(self.path.read_bytes())} if self.path and self.path.is_file() else None


class PathRun:
    def __init__(self, label: str, d: Optional[Path], variants_dir: Optional[Path], l2: Optional[Path] = None):
        self.label, self.dir, self.l2 = label, d, (Path(l2) if l2 else None)
        self.conf = CF.generate(d, None, variants_dir) if d and (d / "receipts").is_dir() else None
        self.rows = {r["id"]: r for r in (self.conf or {"rows": []})["rows"]}
        self.summary = (CF.Receipts(d, None, variants_dir).summary if d and (d / "receipts").is_dir() else None) or {}
        vd = variants_dir or d
        vp = (vd if vd and vd.is_file() else (vd / "VARIANTS_SUMMARY.json")) if vd else None      # a file path or a directory containing VARIANTS_SUMMARY.json
        self.variants = json.loads(vp.read_text("utf-8")) if vp and vp.is_file() else None
        self.variants_path = vp
        self.n_examples = len(self.summary.get("examples", {}))

    def variant_ok(self, names: Sequence[str]) -> Tuple[Optional[bool], List[str]]:
        if not names: return None, []
        if not self.variants: return None, []
        vs = self.variants["variants"]
        okn = [n for n in names if n in vs and all(x["ok"] for x in vs[n].values())]
        bad = [n for n in names if n in vs and not all(x["ok"] for x in vs[n].values()) and not any(x.get("pending") for x in vs[n].values())]      # a pending variant is "not yet", never a FAIL
        return (False if bad else (True if len(okn) == len(names) else None)), okn

    def receipt_check(self, kind: str, suites: "Suites", other: Optional["PathRun"]) -> Tuple[Optional[bool], Optional[str]]:
        s = self.summary
        if kind in ("pathb_laya", "pathb_probe_random", "pathb_conflict"):
            rp = self.dir / "PATHB_REPLAY_RECEIPT.json" if self.dir else None
            if (not rp or not rp.is_file()) and other and other.dir: rp = other.dir / "PATHB_REPLAY_RECEIPT.json"      # a PATH A row may cite the PATH B receipt as its conflict evidence
            if not rp or not rp.is_file(): return None, None
            d = json.loads(rp.read_text("utf-8")); s2 = d["summary"]
            ok = s2["replayed"] >= 1 and s2["negative_controls_refused"] and d.get("stub") is False and all(p.get("laya_equals_stored_verified_scores") for p in d["pools"].values() if p.get("verdict") == "REPLAYED")
            if kind == "pathb_conflict": ok = ok and any(p.get("trusted_evidence_overrode_laya") for p in d["pools"].values() if p.get("verdict") == "REPLAYED")
            if kind == "pathb_probe_random": ok = ok and s2["laya_probe_random_head_all_stored_and_equal"] and s2["comparison_only"]
            return ok, "PATHB_REPLAY_RECEIPT.json %s" % sha(rp.read_bytes())[:12]
        if kind == "l2":
            rp = self.l2 if self.l2 and self.l2.is_file() else (self.l2 / "RECEIPT.json" if self.l2 else None)
            if not rp or not rp.is_file(): return None, None
            d = json.loads(rp.read_text("utf-8"))
            return (None if d.get("errors") else (d.get("stub") is False)), "%s %s%s" % (rp.name, sha(rp.read_bytes())[:12], " (errors: %s)" % ",".join(d["errors"]) if d.get("errors") else "")
        if kind in ("provenance", "backend_identity") and self.label == "B":
            rp = self.dir / "PATHB_REPLAY_RECEIPT.json" if self.dir else None
            if not rp or not rp.is_file(): return None, None
            d = json.loads(rp.read_text("utf-8")); pv = d.get("provenance") or {}
            if kind == "provenance": return (bool(pv.get("commit") and pv.get("source_sha256") and pv.get("config_sha256"))) or None, "PATHB_REPLAY_RECEIPT.json provenance"
            bi = d.get("backend_identity")
            return (bool(bi) and not [k for k, v in bi.items() if re.search(r"stub|mock", v, re.I)]) or None, "PATHB_REPLAY_RECEIPT.json backend_identity"
        if kind == "suites": r = suites.receipt(); return (True if r else None), (r["path"] + " " + r["sha256"][:12] if r else None)
        if kind == "self": return True, "ARCHITECTURE_FINAL_REPORT.md (this file)"
        if not s and kind != "paths_separate": return None, None
        f = "receipts/ACCEPTANCE_SUMMARY.json"
        if kind == "provenance":
            p = s.get("provenance")
            if not p: return None, None
            good = bool(p.get("commit") and p.get("config_sha256") and p.get("source_sha256") and p.get("journal_hashes") and p.get("receipt_sha256") and (p.get("secret_scan_added_lines") or {}).get("hits") == 0)
            return (True if good and p.get("final_path") else (None if good else False)), f + " provenance"
        if kind == "backend_identity":
            bi = s.get("backend_identity")
            if not bi: return None, None
            bad = [k for k, v in bi.items() if re.search(r"stub|mock", v, re.I)]
            return (not bad), f + " backend_identity"
        if kind == "glm_reconcile":
            g = s.get("glm_reconcile")
            return (g["ok"] if g else None), (f + " glm_reconcile" if g else None)
        if kind == "paths_separate":
            a, b = (self, other) if self.label == "A" else (other, self)
            if not a or not b or not a.summary or not b.dir: return None, None
            rb = b.dir / "PATHB_REPLAY_RECEIPT.json"
            b_mode = b.summary.get("mode") or ("pathb_replay" if rb.is_file() else None)
            return (b_mode is not None and a.summary.get("mode") != b_mode and a.dir != b.dir), "%s | PATH B: %s" % (f, b_mode)
        return None, None


def resolve_row(tid: str, ent: Dict[str, Any], idx: Dict[str, str], suites: Suites, runs: Dict[str, PathRun], rowmaps: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    ar, an = auto_tagged(tid, idx)
    rm = (rowmaps or {}).get(tid, {"real": [], "negative": [], "receipt": [], "notes": []})
    # registry names -> refs via the index (a name that is not in any known test file is MISSING, not dropped); rowmap refs are 'file::name' already
    def ref(n): return idx.get(n) or (n if "::" in n else "?::" + n)
    real_refs = list(dict.fromkeys([ref(n) for n in ent["real"] + ar] + rm["real"]))
    neg_refs = list(dict.fromkeys([ref(n) for n in ent["neg"] + an] + rm["negative"]))
    st = {r: suites.state(r) if not r.startswith("?::") else "missing" for r in set(real_refs + neg_refs)}
    real_names, neg_names = [r.split("::")[1] for r in real_refs], [r.split("::")[1] for r in neg_refs]
    failed = sorted(r.split("::")[1] for r, s in st.items() if s == "failed")
    missing = sorted(r for r, s in st.items() if s == "missing")
    real = [r for r in real_refs if st[r] == "passed"]
    neg = [r for r in neg_refs if st[r] == "passed"]
    stats = {r.split("::")[1]: s for r, s in st.items()}
    gap_note = next((n for n in rm["notes"] if GAP_MARKERS.search(n)), None)
    rm_receipts = []
    for rc in rm["receipt"]:
        pth, what = str(rc.get("path", "")), str(rc.get("what", ""))
        ok = pth in ("A", "B") or (ROOT / pth).is_file() or any(r.dir and (r.dir / pth).is_file() for r in runs.values())
        rm_receipts.append((ok, "%s: %s" % (pth, what[:100])))
    row: Dict[str, Any] = {"tid": tid, "owner": ent["owner"], "paths": {}, "real_declared": real_names, "neg_declared": neg_names, "unrun": sorted(n for n, s in stats.items() if s == "unrun"), "missing": missing, "notes_from_owners": rm["notes"]}
    sections = {"A": ["A"], "B": ["B"], "AB": ["A", "B"]}[ent["paths"]]
    src = list(ent["src"])
    for p in sections:
        pr = runs.get(p)
        d: Dict[str, Any] = dict(real=list(real), neg=list(neg), receipt=[], conf_ok=None, stat=ent["stat"], na=ent["na"], failed=failed, stub=False, missing=missing, positive_gap=gap_note)
        notes: List[str] = []
        oks: List[Optional[bool]] = []
        if pr is None or not (pr.conf or pr.variants or (pr.dir and pr.dir.is_dir())):
            d["conf_ok"] = None; notes.append("no run directory for PATH %s" % p)
        else:
            for cid in ent["conf"]:
                r = pr.rows.get(cid)
                if r is None:
                    if pr.conf is None and ent["receipt"]: continue      # this path has no acceptance summary (PATH B replay); its evidence is the receipt kind(s) of the row
                    oks.append(None); continue
                src += r["source"]
                if r["status"] == CF.REAL: oks.append(True)
                elif r["status"] == CF.LOGIC: oks.append(None); d["stub"] = True; notes.append("%s LOGIC-ONLY (%s)" % (cid, ",".join(r["stub_backends"])))
                elif r["status"] == CF.NA: oks.append(True)
                elif r["status"] == CF.IMPL: oks.append(None); notes.append("%s IMPLEMENTED, not exercised" % cid)
                else:
                    contradiction = r["note"].startswith("FAILED") and cid in CONTRADICTION_IDS
                    oks.append(False if contradiction else None); notes.append("%s %s: %s" % (cid, r["status"], r["note"][:90]))
                    if r["note"].startswith("FAILED") and not contradiction: d["positive_gap"] = d["positive_gap"] or r["note"][:90]
                d["receipt"] += ["%s %s" % (x["path"], (x["sha256"] or "")[:12]) for x in r["receipts"]]
                if r["status"] == CF.REAL and not d["real"]: pass
            vr, vok = pr.variant_ok(ent["vreal"])
            vn, vnok = pr.variant_ok(ent["vneg"])
            for names, ok, okn, key in ((ent["vreal"], vr, vok, "real"), (ent["vneg"], vn, vnok, "neg")):
                if names and ok is False: oks.append(False); notes.append("variant(s) failed")
                elif names and ok: d[key] = d[key] + ["variant:" + n for n in okn]; d["receipt"].append("%s %s" % (pr.variants_path.name, sha(pr.variants_path.read_bytes())[:12]))
            for rk in ent["receipt"]:
                ok, ref = pr.receipt_check(rk, suites, runs.get("B" if p == "A" else "A"))
                oks.append(ok)
                if ref: d["receipt"].append(ref)
                if ok is None: notes.append("receipt %s missing or not final-path" % rk)
            for ok_rc, txt in rm_receipts:      # receipts the owners declare: they must exist; they do not turn an un-exercised positive path into a pass (see gap_note)
                if ok_rc: d["receipt"].append(txt)
                else: notes.append("declared receipt missing: " + txt)
            if not ent["conf"] and not ent["receipt"] and not ent["vreal"] and not ent["vneg"]:
                if rm_receipts and all(o for o, _ in rm_receipts): oks = [True]
                else: oks = [None]; notes.append("no receipt check is registered for this row")
            d["conf_ok"] = False if False in oks else (True if oks and all(o is True for o in oks) else None)
            if pr.n_examples and pr.n_examples < MIN_EX and ent["stat"]: d["stat"] = True
        v, why = verdict(d)
        row["paths"][p] = {"verdict": v, "why": why, "notes": notes, "real": d["real"], "neg": d["neg"], "receipt": list(dict.fromkeys(d["receipt"]))}
    row["src"] = list(dict.fromkeys(src))
    vs = [x["verdict"] for x in row["paths"].values()]
    row["verdict"] = "FAIL" if "FAIL" in vs else ("INSUFFICIENT_DATA" if "INSUFFICIENT_DATA" in vs else ("NOT_APPLICABLE" if all(v == "NOT_APPLICABLE" for v in vs) else "PASS"))
    return row


def generate(path_a: Optional[Path], path_b: Optional[Path], variants_dir: Optional[Path], suite_results: Optional[Path], l2: Optional[Path] = None, rowmaps: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tr, idx, suites = traceability(), test_index(), Suites(suite_results)
    rowmaps = load_rowmaps() if rowmaps is None else rowmaps
    runs = {"A": PathRun("A", path_a, variants_dir, l2), "B": PathRun("B", path_b, variants_dir, l2)}
    rows = []
    for tid, t in tr.items():
        ent = REGISTRY.get(tid)
        if ent is None:      # a traceability row nobody registered is never silently dropped
            rows.append({"tid": tid, "owner": t["owner"], "paths": {}, "real_declared": [], "neg_declared": [], "unrun": [], "src": [], "verdict": "INSUFFICIENT_DATA", "why": "no registry entry (row not wired into the generator)"}); continue
        rows.append(resolve_row(tid, ent, idx, suites, runs, rowmaps))
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
    return {"rows": rows, "traceability": tr, "counts": counts, "runs": {k: {"dir": str(v.dir) if v.dir else None, "mode": v.summary.get("mode"), "examples": v.n_examples, "commit": (v.summary.get("preflight") or {}).get("commit")} for k, v in runs.items()},
            "suite_results": suites.receipt(), "rowmaps": sorted(rowmaps)}


def render(rep: Dict[str, Any]) -> str:
    L = ["# ARCHITECTURE FINAL REPORT (EL-ARCH-1)", "", "Generated by `tools/el_arch1_final_report.py` from the receipts named below. This is an ARCHITECTURE closure on three toy tasks: it makes no accuracy claim and no statistical claim.",
         "The toy-repo protected tests are development tests (**UNSEALED**: NOT_SEALED_STATEMENT in evidence_ladder/orchestrator/isolation.py). A row is PASS only with a resolved real test, a resolved negative test and a receipt; anything weaker is INSUFFICIENT_DATA.",
         "VERDICT RULES. FAIL is reserved for a recorded test that failed or a receipt check that CONTRADICTS the requirement (a safety / integrity / replay / budget check that ran and failed). A positive path the final-path receipts did not exercise (for example no example reached VERIFIED) reads INSUFFICIENT_DATA with the reason 'positive path not exercised in the final-path receipts', never FAIL. A declared test that does not exist or was not run is INSUFFICIENT_DATA with its name, never a silent PASS. Tests declared by Agents 1 and 2 (EL_ARCH_ROWMAP_A1/A2.json) are merged and must resolve in the suite results.", ""]
    L += ["counts (combined verdict per T-id): " + ", ".join("%s %d" % (k, v) for k, v in rep["counts"].items()), "", "runs: " + json.dumps(rep["runs"]), "suite results: " + json.dumps(rep["suite_results"]), ""]
    for p, title in (("A", "PATH A: local 0.6B live controller (LAYA only when it scored the exact accepted options, else explicit unavailable-LAYA fallback)"), ("B", "PATH B: frozen 4B LAYA replay (exact stored contexts and options; a mismatch is a refusal)")):
        L += ["## " + title, "", "| Component | Implementation | Real test | Negative test | Receipt | Verdict |", "|---|---|---|---|---|---|"]
        for r in rep["rows"]:
            x = r["paths"].get(p)
            if x is None: continue
            req = rep["traceability"][r["tid"]]["requirement"].replace("|", "/")
            f = lambda xs: "<br>".join(xs) or "-"
            real = f(x["real"] + ([] if x["real"] else ["(none resolved; declared: %s)" % ", ".join(r["real_declared"][:2])]) )
            if x["receipt"] or x["real"]: real += "<br>(dev test, UNSEALED)" if r["paths"] and any(w in req.lower() for w in ("ex1", "ex2", "toy", "protected")) else ""
            L.append("| **%s** %s (%s) | %s | %s | %s | %s | **%s** %s |" % (r["tid"], req[:140], r["owner"], f(r["src"][:4]), real, f(x["neg"] or ["(none resolved)"]), f(x["receipt"] or ["-"]), x["verdict"], "- " + x["why"] + ("; " + "; ".join(x["notes"][:2]) if x["notes"] else "")))
        L.append("")
    L += ["## Combined verdict per T-id", "", "| T-id | verdict | PATH A | PATH B |", "|---|---|---|---|"]
    for r in rep["rows"]:
        L.append("| %s | **%s** | %s | %s |" % (r["tid"], r["verdict"], r["paths"].get("A", {}).get("verdict", "-"), r["paths"].get("B", {}).get("verdict", "-")))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-a"); ap.add_argument("--path-b"); ap.add_argument("--variants"); ap.add_argument("--suite-results"); ap.add_argument("--l2", help="el-arch1-l2-local-*/RECEIPT.json (or its directory)"); ap.add_argument("--out", default=str(ROOT / "ARCHITECTURE_FINAL_REPORT.md"))
    a = ap.parse_args(argv)
    rep = generate(Path(a.path_a) if a.path_a else None, Path(a.path_b) if a.path_b else None, Path(a.variants) if a.variants else None, Path(a.suite_results) if a.suite_results else None, Path(a.l2) if a.l2 else None)
    out = Path(a.out)
    with open(out, "x", encoding="utf-8", newline="\n") as f: f.write(render(rep))      # create-only
    with open(out.with_suffix(".json"), "x", encoding="utf-8", newline="\n") as f: json.dump(rep, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(rep["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
