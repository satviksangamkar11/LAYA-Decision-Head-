# [EL-ARCH-1 Phase A] foundation report

No commits, no pushes, no Kaggle, no NVIDIA calls, .secrets/ not read; frozen LAYA-INT-1 receipts untouched.

Tests (local .venv Python 3.12, PYTHONPATH=., ATEN_CPU_CAPABILITY=avx2): contract 23/23 | controller 25/25 | **new phaseA 16/16** | orchestrator 25/25 | models 30/30 | critic_batch 28/28 |
replay 26/26 | stepE 12/12 | laya_integration 24/25 (the one failure is `test_real_scores_never_override_trusted_evidence`: `T4_scores.json` ships only with the Kaggle bundle and is missing locally; unrelated to Phase A, left to the coordinator's Kaggle run).
WSL execution suite (21) not run: no execution code was touched.

## A1 GLM accounting - DONE
- NEW `evidence_ladder/critic_batch/accounting.py`: `account()` / `from_broker()` compute calls (sends), ok, retries, failed attempts and tokens in/out per packet and in total from the broker ledger + progress ledger. Provider usage when present; otherwise tokens are ESTIMATED (4 chars/token of the request messages / streamed content+reasoning) and flagged `estimated:true`; unanswered attempts count 0 tokens.
- `controller/agentic.py`: `AgenticRun.sync_glm_budget()` SETS glm_calls, glm_tokens_in/out, api_errors of the budget record from those ledgers (called in `finish()`) and journals `glm_accounting` once; `critic_stage` records the packet ids; `wall_seconds` is the real elapsed time.
- `tools/laya_int1_stepE.py`: summary gains `glm_accounting` for future runs.
- Tests: `test_a1_accounting_from_the_live_step_e_ledgers_readonly` (live ledgers: 6 sends, 3 ok, 3 retry, 3 failed, estimated tokens), `test_a1_result_json_budget_equals_ledger_counts` (copy of the live ledgers; budget_record equals an independent count from the file), `test_a1_provider_usage_is_used_when_present_and_not_flagged`.

## A2 transport error class - DONE
- `models/_util.py`: `describe_error()` (exception type, category connect|read|tls|stream|truncated|timeout|http|other, phase, errno, redacted message), `redact_message()` (header values, bearer tokens, key-shaped strings, long opaque tokens, URL query strings, literal secrets), `error_info()` (also classifies pre-A2 raw answers).
- `critic_batch/transport.py`: HTTP error bodies redacted; exceptions are still RAISED (existing contract and tests) but carry the phase. `models/pc_sender.py`: both send paths put `error_class` into the raw record. `models/broker.py`: `ingest` keeps `reason="transport_failure"` and adds `error` (persisted in the broker ledger ingest event, returned to callers). `critic_batch/batch.py`: `error` also in the progress `ingested` event.
- Tests: `test_a2_describe_error_classes`, `test_a2_injected_transport_errors_are_classified_and_leak_no_key` (ConnectionResetError, ssl.SSLError, socket.timeout, URLError, truncated stream, HTTP 503; no key-shaped text in the ledger).

## A3 enforced limits - DONE
- NEW `configs/el_arch1_limits_v1.toml` (profiles agentic and ladder, written reason; frozen thresholds untouched). NEW `controller/limits.py` (`Limits`, `Limiter` with injected clock, `load_limits`, `LimitsConfigError`).
- `agentic.py`: `AgenticContext.limits/clock`; `step()` raises `LimitReached` before doing work when wall time / Qwen tokens / GLM tokens / actions / rounds are exhausted; the critic stage is refused on wall time or GLM tokens; `finish()` reports `stop_cause="budget"` and `stop_details="limit:<resource> used=<n> cap=<n>"` (the journal stop note carries `details` only when a limit fired; wall seconds are not journaled). `controller.py`: `EvidenceLadderController(limits=None, clock=None)` (default profile ladder), `limit_check()` in `generate()` and in the loop (after the legacy cfg caps, so a decided problem is never turned into a budget stop); `after_round`'s budget_used=0/cap=1 is now documented as intentional (only non-budget causes are wanted there). `invariants.check_i5(..., limits)` makes the caps real.
- Tests: `test_a3_limits_file_is_explicit_and_loud`, `test_a3_agentic_wall_time_actions_rounds_tokens_stop_as_budget_with_the_resource_named`, `test_a3_glm_token_limit_blocks_the_critic_stage_and_i5_catches_overshoot`, `test_a3_ladder_controller_enforces_actions_rounds_wall_and_tokens`.

## A4 model-derived evidence capped - DONE
- `controller/evidence.py`: `Factory(..., roles)`; model-derived records (independence model_derived, model sources, diagnostic checks) keep `pass` only where the router's fixed table `SOFT_BY_ROLE` lists that check for the controller's own role of that requirement, else they are stored `unknown`; a requirement id not in the controller's list raises `ModelEvidenceError`. `controller.py apply_critic`: the requirement a critic names is recorded (`critic_named_requirement`) but never obeyed; the evidence attaches to the controller's all-inputs row. `agentic.py`: critic evidence (on R1) is stored `unknown` (or `fail`), never `pass`; `critic_result.evidence_status` shows the stored status. `invariants.check_i3` now also rejects (a) evidence naming a requirement outside the controller's list and (b) a model-derived pass outside the journaled soft route (catches records forged past the Factory).
- Tests: `test_a4_unknown_or_invented_requirement_is_refused_before_it_exists`, `test_a4_critic_named_requirement_is_ignored_and_example_row_pass_is_capped`, `test_a4_critic_pass_cannot_lift_unverified_or_blocked_and_forged_records_fail_i3`.
- **One existing test edited (intended behaviour change):** `controller/tests/run_tests.py::model_tests_never_verify` fed model passes to the contract and example rows and expected SUPPORTED, which was exactly the leak. It now passes `roles` and asserts UNVERIFIED overall with only R2 (all-inputs) SUPPORTED. Test count unchanged (25).

## A5 I1-I8 real / MissingComponentError - DONE
- NEW `controller/components.py` (`MissingComponentError`, `require`). Always required (raise): sandbox backend, protected test + test command (agentic), generation/execution/budget (ladder), check registry (router), registry (journal; still a `JournalError`), a journaled routing table (assert_invariants). Scorer/critic: `config["require"] = ["scorer","critic"]` raises (agentic: after a `component_missing` journal entry; ladder: before any model call). When not listed, the missing component is journaled as `component_missing` with `waived:true` and the run continues; kept because the laya_integration and replay tests and the pilot run arm E without a scorer. **Final acceptance runs must set `require`.**
- `invariants.check_i1`: the `derived_by` literal no longer decides anything; the result decision must equal the journaled decision and its candidate, rows and authorization must equal a recomputation from the evidence.
- Tests: `test_a5_agentic_missing_components_raise_or_journal_and_stop`, `test_a5_ladder_controller_missing_components_raise`, `test_a5_registry_router_journal_and_invariants_need_their_parts`, `test_a5_i1_checks_the_journaled_decision_not_a_literal`.

## Effect on frozen receipts
None regenerated or touched. Journal hashes of NEW runs differ from old ones only when something new happens (a limit fires, a component is missing or waived, the critic stage ran against a ledger: `glm_accounting` entry). The no-event path writes the same entries as before. Kaggle/PC bundles must ship `configs/el_arch1_limits_v1.toml` (controllers load it at construction and fail loudly if absent).

## Files
Changed: evidence_ladder/controller/{agentic,controller,evidence,invariants,journal,router}.py, evidence_ladder/controller/tests/run_tests.py, evidence_ladder/critic_batch/{batch,transport}.py, evidence_ladder/models/{_util,broker,pc_sender}.py, tools/laya_int1_stepE.py.
New: evidence_ladder/controller/{components,limits}.py, evidence_ladder/controller/tests/phaseA_tests.py, evidence_ladder/critic_batch/accounting.py, configs/el_arch1_limits_v1.toml, this report.
