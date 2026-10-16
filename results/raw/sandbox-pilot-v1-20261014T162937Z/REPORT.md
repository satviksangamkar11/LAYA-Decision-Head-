# Sandbox verified-label pilot v1: result (2026-10-15, 02:30 IST)

Pre-registration: `configs/fidelity_v3.toml`, `configs/thresholds_sandbox_pilot_v1.toml` (copies in this folder).
Run record: the pilot agent ran qualification, candidates and 12 rollouts (22:00-23:10 IST) and was stopped by an API session limit; the
main session resumed with `tools/sandbox_pilot_v1.py rollouts` (it found 0 rollouts left: 12 = 2 tasks x 2 valid candidates x 3) and ran
`labels`. Concurrency changed to 10 in flight per key mid-run (user); no rollout was affected (all 12 had started before).

## Against the pre-registered criteria

| criterion | required | measured | |
|---|---|---|---|
| qualified starting states | >= 3 | 2 (starlette_exporter-89; googleapis python-storage-1357 under fidelity v3). dwave-system-373: setup failed (exit 2). The qualification walk was not continued. | FAIL |
| defensible BETTER/WORSE comparisons | >= 3 | 0 (2 EQUAL_ON_TESTED_OUTCOME "both 0/3", 4 INVALID) | FAIL |
| ... including a non-edit comparison | >= 1 | 0 | FAIL |
| environment-failure rollouts | <= 20% | 0 of 12 (0%); every rollout clean, offline, start-state hash verified | pass |

NVIDIA calls: 186, all HTTP 200, Rs 0. Rollout wall time 878-2,115 s each. Secret scan: see secret_scan output if present; no credentials stored.

## Why no labels (diagnosis)

1. **The continuation budget cannot finish these tasks.** All 12 rollouts ended at the 15-step budget, 0 resolved. The original
   Qwen3-Coder runs that RESOLVED these tasks used 64-86 tool calls (python-storage, 4 runs) and 37-76 (starlette, 7 runs); unresolved runs
   47-100. With decision points at step ~7, a 15-step continuation reaches roughly 20-30% of a typical solving run, so every candidate scores
   0/3 and every valid pair is EQUAL ("both 0/3"). The label rule worked as designed (no negatives invented); the design gave it nothing to
   separate. This was a contract design error (15 steps), not an execution failure.
2. **The GLM test-or-edit candidate (slot T) was INVALID on both tasks** ("no parsable action of the required type in 2 attempts"), so
   4 of 6 pairs are INVALID.

## Verdict: NO-GO for scaling this design. Reproducibility (fidelity v3, hashes, offline sandbox, 0% environment failures) is confirmed.
