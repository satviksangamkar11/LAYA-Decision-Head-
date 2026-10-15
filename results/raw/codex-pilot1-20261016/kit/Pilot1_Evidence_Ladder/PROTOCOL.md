# Pilot 1 Protocol v1 — freeze this BEFORE model calls

## Goal
Measure whether a weak Qwen coder solves more Python function tasks when given public tests, feedback and the Evidence-Ladder controller, not merely a larger token allowance or a stronger external critic.

## Tasks and scoring
- 40 original development prompts: 10 easy, 20 medium, 10 hard. Labels uncalibrated; report by stratum.
- Four public tests visible to the worker per task, 40 other deterministic development tests in the grader; tests were authored with references and are NOT an independent sealed holdout.
- Task success = all 40 protected tests pass for a fixed candidate at the same artifact revision. Each task contributes exactly 1/40 to the overall rate. Include missing outputs, timeouts and crashes as failures.
- Do not choose the best candidate by reading protected results. Freeze selection before grading. If a task is run again after viewing protected results, mark it DEV/CONTAMINATED and exclude from a held-out claim.
- Reference oracles were exercised against all 44 examples per task; mutation smoke checks only establish that trivial constant-null solutions fail. Manually inspect ambiguous tasks and compare with independent implementations before external testing.

## Arms and attribution
A Qwen first attempt.
B Qwen best-of-k chosen by PUBLIC checks, with at least the generated-token budget used by the critic-bearing arm.
C Qwen with real public-check feedback and correction.
D0 C + deterministic Evidence-Ladder requirement and evidence controller; NO GLM.
D1 D0 + GLM-5.3-Flash when its experiment trigger fires. Log GLM calls, API/retry status and full output usage.
E D1 + frozen Qwen→LAYA adapter, strictly as an exploratory out-of-domain transfer condition; NOT a Stage-1 gate for pretrained LAYA.

Do not claim an arm was run if its actual component is absent. E in Stage 1 is NOT validation of the next-action scorer. Stage 2 is needed for that.

## Budget protocol
- Log Qwen tokens, GLM tokens, total generated tokens, all calls, external API quotas, GPU-seconds, test executions, total wall time per task.
- Before starting choose a common total-generated-token ceiling based on DEV and hold it for all arms; because model prices/throughput differ, report token match and wall-time match as two different analyses.
- Do not rely on the chosen pilot to estimate statistical significance of a 5 percentage point gain: one task is 2.5 points. This is a diagnostic pilot.
- Keep all task IDs and seeds fixed. Qwen and GLM weights frozen. Critic statements are hypotheses and never count as execution evidence. No training.
- Rs 0 rule: fail closed before paid usage. This package itself has no network calls.

## Protected grade boundary
Generation environment receives worker/ only. The grader environment receives candidate files after candidate selection is frozen. A separate kernel alone does not protect hidden inputs from hostile candidate code executing in that kernel; use a dedicated sandbox, external evaluation service or restricted runner for real sealed grading. This package's local scorer is intentionally UNSAFE and suitable only for code you trust.

## Pilot exit decision
- Report A,B,C,D0,D1 separately. Essential comparisons: C vs D0 (controller), D0 vs D1 (GLM); A vs B (extra tokens).
- Verify comparison was paired by task and budget logs were collected. Report exact point estimates, task gains and losses, and wide bootstrap intervals.
- Stage 1 transfer E cannot accept/reject LAYA. Stage 2 measures LAYA only on real agent next-action decisions with compatible adapter features.
- This is a DEVELOPMENT pilot. Do not call any gain a generalization improvement without an independent, uncontaminated final benchmark.
