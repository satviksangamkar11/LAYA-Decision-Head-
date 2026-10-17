# EL_ARCH Agent 1 results (candidate layer), 2026-10-17

Files: `evidence_ladder/controller/generation.py`, `candidate_filter.py` (new), `tests/phaseF1_tests.py` (new, 15 tests). Interface: EL_ARCH_INTERFACE_FREEZE_v1.md, implemented as frozen.
Runs (PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2, .venv python): phaseF1 15/15, phaseB 36/36, phaseA 16/16, run_tests.py 25/25.
phaseD1: 15/21 on the merged tree; all 6 failures are outside my files: `AgenticRun` has no `invariants_checked` (tools/el_arch1_variants.py:354 vs agentic.py), and earlier syntax errors (literal newline in a string) seen in tools/el_arch1_acceptance.py / el_arch1_conformance.py. NOT mine; Agent 2/3 to fix.

## T23 GENERATE parse / template  : DONE (candidate side)
- A1.1 per-option salvage: `generation.parse_actions_v2` (generation.py ~L88). Tests: a11_salvage_keeps_valid_siblings_and_rejects_the_broken_one (positive), a11_negative_strict_parser_and_prose_stay_strict, a11_filter_rejects_a_malformed_item_alone (phaseF1_tests.py).
- A1.2 prompt `qwen_actions_v2` (ACTION_INSTR_V2, one independent next action, quote-escaping instruction); `qwen_actions_v1` untouched (a12_new_prompt_template_is_versioned_and_v1_is_frozen). `ActionProposerV2` (filter per generation, template id recorded). a15_proposer_v2_end_to_end.
- NOT DONE here: "real local 0.6B backend" run (needs Agent 2/3 + the coordinator's final run).

## T24 CANDIDATE FILTER: DONE
- Signature `filter_options(options, files, *, max_options=5, path_scope=None)` -> {accepted, rejected, abstain, counts}; option_id = sha256(render_call). a16_result_shape_ids_and_purity.
- Rule codes with positive+negative tests: malformed_option (a11), unknown_tool/bad_arguments/path_outside_scope/protected_path/duplicate (a14_rejects_bad_structure_scope_and_dedupes, a14_path_scope_prefixes_and_negative), no_op_edit/unchanged_diff/old_string_absent/old_string_not_unique (a13_*), sequential_plan (a12_sequential_plan_keeps_first_step_only / a12_negative_independent_alternatives_are_not_a_plan), beyond_max_options + no padding (a14_count_cap_order_and_never_pads), abstain codes (a15_abstain_codes_and_negative). `repeated_failure` is the controller's.
- No rejection by risk class (coordinator decision): irreversible / network / unknown Bash is ACCEPTED, tested in a14_negative_risky_actions_are_accepted_for_the_authorization_gate. Path scope applies only to the structured `file_path` of Read/Edit; Bash paths go to the policy gate (b5 test expects that: phaseB now 36/36).
- Not covered: a real-controller test of each rule code (Agent 2's wiring is in agentic.py:258; I did not write that test).
- unchanged_diff is reachable only for line-ending-only edits or an edit that leaves the file text unchanged; `old_string_absent` also covers a missing file.

## T12 / T13 / T26 real-model harness: BUILT, NOT RUN on the real model (3050 reserved for the coordinator)
New files only: `tools/el_arch1_l2_local.py`, `evidence_ladder/controller/tests/phaseF1b_tests.py`. No other file edited.
What it does (per ex1_slug / ex2_clamp / ex3_wordcount; `derive_task` builds the function-level TaskSpec from `laya_int1_t6.TASKS`: public examples = the asserts of the task's test file, the test file itself is never shown to the model):
real HFQwenBackend (profile qwen3-0.6b, local=True, pinned sha256 of model.safetensors + tokenizer.json, loaded via profile_capture.load_model, built as `worker_gen` in el_arch1_acceptance.py does) -> EXISTING `EvidenceLadderController` ladder mode (k=4, frozen strategies qwen_sample_v2/v2b, l2, discriminate, critic) -> candidates and mutants run in the real WSL sandbox through `WslFunctionExec` (one bubblewrap call per source x input list, cached) -> journal.jsonl + result.json per task + RECEIPT.json (create-only; backend identity from `model_info()` incl. profile sha, script sha256, controller config, seeds, sandbox info, per-task L2 / clusters / discrimination verdicts, invariants I1-I8 result, `never_approved_by_l2_clusters_or_critic`).
Critic: the existing `StagedCriticBroker` (blinded packet, no network). If discrimination triggers, the run ends `api_unavailable / critic_pending` with the packet listed in the receipt; after the local GLM step, re-run the same command with `--critic-responses <json {packet_id or request_id: response}>` (same seeds) to apply it. If GLM modules cannot be imported the receipt says so and the controller journals the missing component (never faked).
Honest outcomes are valid: L2 INSUFFICIENT (no valid test parsed / every test dropped, with reasons) and discrimination NOT_TRIGGERED (< 2 clusters).
Guards: `assert_real_backend` accepts only an `HFQwenBackend` instance on the local verified snapshot (isinstance + model_info identity); the CLI has no mock flag; `run_harness(final=False)` (tests only) labels the receipt LOGIC_ONLY_STUB; output dir must not exist; receipt files are create-only.

Tests (phaseF1b_tests.py, 10/10 incl. 1 real-WSL sandbox protocol test; the rest are LOGIC-ONLY: scripted generation text, in-process execution; they prove plumbing and ladder behaviour, not the 0.6B):
- T12 positive f1b_valid_l2_check_two_clusters_and_discrimination_triggered (valid check validated); negative f1b_vacuous_and_untraceable_checks_are_rejected_with_reasons (vacuous + non-verbatim quote dropped), f1b_no_valid_l2_text_and_one_cluster_are_valid_receipt_results; cannot raise a row: f1b_nothing_is_approved_by_l2_clusters_or_critic (L2 evidence model_derived, label != VERIFIED, not authorized).
- T13 positive same test (1 public cluster, 2 extended clusters from derived inputs); negative f1b_a_wrong_candidate_set_with_equal_behaviour_does_not_trigger_discrimination.
- T26 positive (TRIGGERED, critic pending, blinded packet exported); negative: critic only pending/applied as evidence, never approval (same invariants check). Not tested here: applying a real GLM response (Agent 2/3 own that path; `--critic-responses` just feeds the controller's existing `critic_responses` hook).
- Guards: f1b_stub_backend_refused_in_final_mode_and_existing_dir_refused, f1b_receipt_is_create_only_and_names_the_backend, f1b_a_crash_in_one_task_is_recorded_not_hidden, f1b_tasks_are_derived_from_the_driver_source, f1b_sandbox_runner_protocol_wsl.

Command for the coordinator (needs WSL answering, the 3050 free, the 0.6B snapshot in models/qwen3-0.6b):
    PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe tools/el_arch1_l2_local.py results/raw/el-arch1-l2-local-<UTC YYYYMMDDTHHMMSSZ> --final
Expected runtime: estimate (unmeasured) 15-40 min total: per task up to 4 candidates + 1 L2 request (<= 2048 new tokens each, greedy 0.6B, tens of seconds each) and roughly 60-150 cached WSL sandbox calls (a few seconds each). Greedy decoding gives one text per prompt, so the 4 samples collapse to at most 2 distinct candidates (v2 and v2b prompts); one cluster is therefore a likely honest result for some tasks.
Untested: the real CLI path (model load, snapshot verify, StagedCriticBroker construction) since it needs the GPU; `python tools/el_arch1_l2_local.py --help` runs.

## Update: duplicate rule + prefilter (coordinator request)
- `ActionProposerV2` now parses with `generation.parse_candidates_v2` (no judging) and lets `filter_options` decide: a duplicate is always rule `duplicate`, a broken array item is `malformed_option` (a17_duplicate_surfaces_as_duplicate_never_as_malformed_option). Cross-seed merge rejections also carry `rule` (duplicate / beyond_max_options).
- `ActionProposerV2(backend, prefilter=True)`: default unchanged. `prefilter=False` never calls `filter_options`; `propose()` returns the CandidateIn items in generation order in `out["candidates"]` (malformed items `{"malformed": True, "origin", "raw"}` and duplicates untouched), `options == []`, no `filter_results`/`abstain` (a17_prefilter_false_hands_raw_items_through_untouched, a17_prefilter_true_still_filters_and_is_the_default).
- Runs: phaseF1 18/18, phaseF1b 10/10, phaseB 36/36, phaseF2 43/43, phaseA 16/16, run_tests 25/25.
