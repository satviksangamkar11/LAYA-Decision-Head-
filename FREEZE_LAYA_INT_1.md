# FREEZE: LAYA-INT-1 integration milestone (2026-10-17)

Tag: [LAYA-INT-1 · Aim: integration, not performance]. Frozen at git commit `0ad42eae` (branch `laya-int-1-stage0-critic-2026-10-17`, pushed). Plan: PLAN_LAYA_INT_1.md. Full state: HANDOFF_2026_10_17_LAYA_INT_1.md (addenda 1 and 2).

## What is frozen (do not rerun Qwen, do not retrain the head)
- Qwen3-4B-Thinking-2507 (rev 768f209d), 3 frozen adapters final-s0/s1/s2 + pretrained LAYA head (hashes: results/raw/laya-baseline-v1-20261012T215242Z/MANIFEST.json), capture pins v2 (configs/laya_capture_code_pins_v2.toml + results/raw/laya-integration-20261016T184816Z/CORRECTION_NOTE.md).
- Receipts (sha256 first 16): smoke run results/raw/laya-integration-20261016T184816Z/smoke/RUN_SUMMARY.json 431648d3dfbb38e9; integration run 2 results/raw/laya-integration-20261016T184816Z/integration_run2/ (T1-T4, T7); e2e pass 2 results/raw/laya-integration-20261016T184816Z/e2e_pass2/E2E_SUMMARY.json 1161c62af541b737; Kaggle tests 182/182 results/raw/laya-integration-20261016T184816Z/tests_kernel_run/TEST_RESULTS.json c02da658195e1984; Step E live results/raw/laya-integration-20261016T184816Z/stepE_20261017T070508Z/STEPE_SUMMARY.json 86ea21ea8961b620, FINAL_SHA256SUMS 5b1a6523c12c9402; architecture audit results/raw/architecture-audit-20261016T211246Z/AUDIT.md e0e4b302a12b8fc8.
- Result: 3 toy examples (ex1_slug BLOCKED, ex2_clamp VERIFIED, ex3_wordcount BLOCKED) through Kaggle-T4 Qwen + capture + 3 adapters + LAYA -> real controller (agentic.py) -> WSL execution -> feedback -> identical replay hashes 3/3 -> local GLM-5.3-flash critic -> experiments executed in WSL -> diagnostic evidence; I1-I8 pass; original receipts unchanged.

## Caveats that stay attached to this milestone (state them whenever it is cited)
1. Execution preceded the final commit: the e2e pass 2 and Step E live runs came from the working tree; the code was committed afterwards (800f6a39, 0ad42eae). Equality of committed and executed code was not separately proven.
2. Three tiny toy tasks: wiring evidence only. Not Stage 2, not a benchmark, not performance. Performance NOT TESTED.
3. Budget accounting gap: Step E result.json files show glm_calls = 0 and zero GLM tokens (hard default in agentic.py:78) although the critic ledger records 6 sends. Reconcile before any equal-compute comparison.
4. The first attempt of all 3 critic requests failed as transport_failure within seconds; cause undiagnosed; one retry each succeeded.
5. Qwen proposals were format-forced (prefilled JSON, thinking closed) after prose failures.
6. Random-head control is a mismatch control, not a valid comparison; probe/random-head are in no arm.
7. Hidden dev cases are public in Git history (unsealed); the unsafe-experiment rejection was exercised only with mock responses, not live.
8. The architecture is NOT complete: see PLAN_EL_ARCH_COMPLETION_1.md.
