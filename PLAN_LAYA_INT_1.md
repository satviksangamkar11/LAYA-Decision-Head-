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
