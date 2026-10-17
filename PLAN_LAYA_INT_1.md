# LAYA-INT-1: Qwen → LAYA integration verification

**Aim (fixed, user 2026-10-17):** prove the architecture is wired correctly: real hidden states of the frozen Qwen3-4B-Thinking-2507 reach the
3 frozen adapters, the pretrained LAYA head runs its real forward pass, and the Evidence-Ladder controller uses the scores only as permitted.
**Not the aim:** proving LAYA improves accuracy. That is a later, separate plan (performance evaluation).

Every message and report about this work starts with: `[LAYA-INT-1 · Aim: integration, not performance]`.

## Architecture under test

```
Qwen 4B proposes candidate next actions
   → capture Qwen hidden states (training recipe: last prefix-token state after "Next action:" + mean state of each option's tokens
     as independent suffixes; depths 18/24/30; training device class and dtype)
   → each frozen adapter (final-s0, final-s1, final-s2) separately → pretrained LAYA head → one score per option (no NONE class)
   → Evidence-Ladder controller: scores are L0 only (order checks, tie-break); never approve; never override L5/L3 (I2)
   → WSL sandbox executes the allowed action → test results → new context → next decision
```

## Pass criteria (all with saved receipts under `results/raw/laya-integration-<UTC>/`)

| Test | Must happen | Status |
|---|---|---|
| T1 real capture | frozen Qwen runs and produces actual hidden states (shapes, depths, dtype, device recorded) | PENDING |
| T2 training parity | captured features match the training recipe and saved reference features (thresholds pre-registered in a new versioned file before the run) | PENDING |
| T3 adapter | the 3 adapters receive the correct context and option tensors (order, shapes, pooling, as the training code path) | PENDING |
| T4 LAYA | real head forward pass, finite valid scores; recorded scores reproduced on stored pools; permutation-stable | PENDING |
| T5 controller | scores reach the controller and affect only permitted decisions (L0); a top-LAYA candidate failing trusted evidence is never selected | PENDING |
| T6 end-to-end | one decision → execution → feedback loop completes (≤ 3 steps), reproducible receipts (second run gives identical hashes) | PENDING |
| T7 provenance rejection | verified by the runtime path, not by stored metadata: live mode captures itself from text; replay mode re-captures a random subset; wrong model, dtype, depths, recipe and fabricated tensors with valid-looking metadata are refused | PENDING |

**Verdicts, reported separately:** INTEGRATION PASS (T1–T7 all pass) · PERFORMANCE: NOT TESTED in this plan.

## Steps

1. Study: exact checkpoints (sha256 from `results/raw/laya-baseline-v1-20261012T215242Z/MANIFEST.json`), exact capture recipe, training device class; verified vs claimed. **In progress.**
2. Implement the LAYA forward pass behind `evidence_ladder/models/laya.py` (L0 only), with the runtime provenance guard. **In progress.**
3. Pre-register parity thresholds (new versioned file, written before any run).
4. Kaggle GPU parity and capture kernel on the default account `sangeetasangamkar` (T1–T4, T7).
5. Pure-logic tests (T5, T7 refusal logic, interface) as a new suite in the Kaggle test kernel; the existing 157 tests stay green.
6. One small end-to-end loop (T6): Qwen + capture + LAYA on Kaggle, execution in the local WSL sandbox, staged per step. If impractical: stop and report options (no silent switch to the Kaggle jail).
7. Coordinator verifies every receipt, then declares INTEGRATION PASS or FAIL.
8. Freeze the working architecture (commit, hashes). Performance evaluation is a separate plan after that.

## Rules

- No training, no adapter changes, no encoder changes; Qwen frozen.
- No 320-answer generation for this plan. LAYA vs probe is not a gate (probe and random-head are reported controls).
- No NVIDIA calls; nothing on `satviksangamkar54321`; no hidden or protected data or keys in any bundle; every Kaggle CLI call under `timeout`; Rs 0.
- Commits are local and secret-scanned; push only when the user asks.

## Related, separate plans (not mixed in)

- Stage-1 performance pilot (arms A–E, GLM critic, k candidates per task): later.
- GLM batch critic: built and tested with fake data (Kaggle 157/157, commit `7c5099be`); live batch not run.

## Status and required corrections (user 2026-10-17)

| item | status |
|---|---|
| 3-pool smoke (S1-S6, Kaggle T4) | **PASS** (receipts `results/raw/laya-integration-20261016T184816Z/smoke/`, commit c76a1423) |
| Full T1-T7 integration | **PENDING** |
| Performance | NOT TESTED |

Required before final acceptance:
1. **Capture-code mismatch:** pin the implementation actually used for training (capture/capture.py + capture_resident.py) via a documented, versioned correction (new pins file + correction note + evidence); no silent bypass.
2. **T7 replay-mode re-capture covers 100%** of the stored records used in the integration tests (sampling only as an option for large batches); a test proves one fabricated record anywhere is caught.
3. **T6 real multi-step decision:** >= 2 steps where LAYA's ranking influences an allowed choice (compare with id-order fallback), WSL sandbox execution and feedback, plus a step where trusted evidence conflicts with LAYA's top choice and the trusted option wins (real scorer output).
Random-head control is labelled "mismatch control, NOT a valid performance comparison"; fixing it belongs to the later performance experiment.
Freeze LAYA-INT-1 only when T1-T7 pass with these three addressed. No new generation workload, no training.

## Handoff
HANDOFF_2026_10_17_LAYA_INT_1.md written 2026-10-17 (full state, decisions, audit blockers, next steps). LAYA-INT-1: smoke PASS; full integration PENDING; performance NOT TESTED.

## Freeze
LAYA-INT-1 frozen: FREEZE_LAYA_INT_1.md. Next milestone: PLAN_EL_ARCH_COMPLETION_1.md (EL-ARCH-1), not started.
