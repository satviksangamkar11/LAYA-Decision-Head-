# P5 protocol v1 (frozen before any evaluation candidate is generated) — 2026-10-20

Status: FROZEN when committed. Any change = P5_PROTOCOL_v2 with a written reason, made before the final run, never after seeing final results.
Purpose: measure what the Evidence-Ladder adds to the same model, and what V1.5's trusted verification adds to V1. It does NOT measure the LAYA head / hidden states
(an experimental path kept separately, not part of this claim).

## Model and settings (all arms)
Qwen2.5-Coder-1.5B-Instruct, pinned revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, non-thinking, sampling T 0.7 / top_p 0.8 / top_k 20 / repetition_penalty 1.1,
max 4,096 new tokens per sample. GLM critic OFF. LAYA learned scoring OFF. One sandbox for every arm of a cohort (Python: MXC process container; Java only if WSLC is re-qualified).

## Arms
| Arm | What it is | Budget |
|---|---|---|
| A plain | one sample (seed 1), answer = that sample | 1 sample |
| B multi-sample | fresh samples, answer = first sample passing every public example | generated-token cap G per task |
| C V1 | existing Evidence-Ladder controller, ladder mode, repair as in V1 (critic off) | same cap G |
| D V1.5 | C + trusted contract checks (contract_v1, host registry), model-proposed checks shadow only | same cap G |
B, C and D share their first-round samples (same seeds and prompts): identical requests reuse the exact recorded output; requests that differ are generated.
G = 6,000 generated tokens beyond the first round of 4 samples (the P2 budget).

## Cohorts (reported separately, never pooled)
1. Benchmark: LiveCodeBench v6 (`data/livecodebench/v6/test6.jsonl`, sha256 bb4c364f...), easy + medium only, excluding the 4 adapter-validation problems
   (abc388_a, abc389_a, 3747, 3753) and the smoke problems. N is fixed after the smoke step by the rule: the largest N <= 91 whose estimated GPU time is <= 10 hours
   at the smoke-measured rate, minimum 40; the N problems are taken in a seeded random order (seed 20261020) and LOCKED (ids + hash) before the final run.
   Official comparison rules (adapter `evidence_ladder/execution/lcb_runner.py`). Contract coverage is reported (expected ~0).
2. Contract tasks: 20 newly authored tasks (bank format, solve(data)). Statement + contract written and hashed BEFORE any candidate exists; hidden tests and a
   reference solution written SEPARATELY from the statement only (without the contract), then hashed. Contracts are authoritative only through the host registry.

## Evaluation isolation
Hidden / private tests and reference solutions are never in a prompt, a runtime check, a retrieval store or a training file. Grading happens offline after each arm's
selection is final. Smoke problems are disjoint from final problems.

## Measures (task-level, paired across arms)
Independently correct final answers (solve rate); false SUPPORTED (answered but wrong); correct answers wrongly blocked (D vs C); abstention rate; median seconds per task;
generated / prompt / total tokens per task; sandbox seconds; trusted-contract coverage (share of tasks with an authoritative contract).
Uncertainty: exact counts + paired 95% intervals (bootstrap over tasks, 10,000 resamples, seed 1). No significance claim beyond that.

## Success criteria (pre-registered)
- D vs C, contract cohort: false SUPPORTED lower; correct answers wrongly blocked = 0 observed; verified solves not lower.
- D vs C, benchmark cohort: verified solves not lower; median wall-time overhead <= 10%.
- "Beats plain Qwen" (C or D vs A): claimed only if the paired interval of the solve-rate difference excludes 0; cost reported alongside.
- Infrastructure failures stay separate from candidate failures (0 unresolved).
- An inconclusive interval does not block freezing the interfaces; it limits the claim.
Freeze V1.5 if all D-vs-C criteria hold. If not: identify contract coverage / validation error / infrastructure / generation capability and change only that part.

## Smoke step (before the final run)
8-10 problems disjoint from the final set (6 LiveCodeBench easy/medium + 3 contract-style bank tasks not in the 20): all four arms end to end, grading works,
budgets respected. Reported: pilot solve rate of arm A, seconds per task, planned N and GPU hours. The final run needs the user's GO after that report.
