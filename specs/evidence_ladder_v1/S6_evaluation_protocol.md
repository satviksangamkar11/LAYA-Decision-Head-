# S6. Evaluation protocol (Evidence-Ladder v1, P0 + R3, 2026-10-16)

Schema for cost accounting: `schemas_v1.json#/$defs/budget_record`. Everything in sections 1-3 and 5-8 is **frozen** by this document. Section 4 lists what is tuned.
The frozen values are copied into `configs/thresholds_evidence_ladder_v1.toml` (sha256 recorded) once D1 is resolved, before any run. After that, changes need a new versioned file with a written reason.

## 1. Arms (same problems, same starting state, paired)

| arm | system | isolates |
|---|---|---|
| A | Qwen, one attempt, public examples visible in the prompt | floor |
| B | Qwen, best-of-k by public examples, using the full per-problem budget | extra compute alone |
| C | Qwen, samples + repairs from real failures (public examples), no controller beyond S5's stop rules, no critic | verification and repair |
| D0 | C + Evidence-Ladder controller (S1-S5), no GLM | the architecture |
| D1 | D0 + GLM-5.3-flash critic | the external critic |
| E | D1 + adapter/LAYA in the ranking slot | LAYA (stage 1: **exploratory transfer test**; stage 2: confirmatory, next-action) |

- **Controls beside E:** the probe in the same slot, and a random-head adapter in the same slot. Same candidate pools.
- **Common random numbers:** arms B-E share the same first k1 samples per problem (same seeds), so differences come from the method, not from luck in sampling.
- **Hidden tests never select anything at runtime.** They only grade the final answer, after the run, in the grading kernel (S4 section 3).

## 2. Primary metric

- **Verified task completion:** the final answer passes every hidden test. Graded once, per arm, per problem.
- Secondary (reported, not gated):
  - false-approval rate: label VERIFIED or SUPPORTED but the hidden tests fail;
  - false-block rate;
  - repair success rate;
  - critic challenge precision: executed experiments that removed a wrong cluster, divided by critic calls;
  - stop-cause distribution;
  - cost per solved problem.

## 3. Budget (matched-token design; stated as such, not as equal cost)

- Every arm gets the same per-problem cap **T** on total generated tokens, counting Qwen tokens plus GLM output tokens.
  - B spends T on samples.
  - D1's GLM tokens come out of the same T.
- Recorded separately per problem (`budget_record`): Qwen generated and prefill tokens, GLM in/out tokens and calls, scorer calls, tool runs, GPU seconds, wall seconds, API errors.
- **Sensitivity analysis:** repeat the B-vs-D1 comparison matched on wall time.
- API usage is reported separately. D1 results are always labelled "with an external API critic".

## 4. Thresholds: frozen versus tuned

- **Frozen now:**
  - the primary metric;
  - the arms;
  - T (set from the pilot's cost measurement, written into the toml before the main run, then frozen);
  - task eligibility (section 5);
  - error definitions (S5 stop causes, S2 statuses);
  - the safety rules (S4);
  - the statistical rule (section 7).
- **Tuned on development data, then frozen before the holdout:** k1, when to sample more, repair lineages, the critic trigger, the budget split. Tuning uses grouped cross-validation on DEV only.

## 5. Dataset (decision D1: unresolved)

Requirements for any stage-1 dataset (all must hold; checked and recorded before the pilot):
1. Problems published after 2025-07-31 (after Qwen3-4B-Thinking-2507's release). GLM-5.3-flash may still have seen them. D1 results carry that caveat.
2. Executable tests with expected outputs available offline, hidden tests separate from public examples, and a special checker where outputs are not unique.
3. Grading works without Docker (Kaggle Python, g++), inside the S4 jail.
4. Licence permits this research use. Gated access terms are accepted by the user, not by the agent.
5. Reference solutions available for the adequacy check (section 6). If they are missing, the dataset is usable only with the adequacy check marked NOT MEASURED, and that weakens every claim.
6. No silent substitution of an older release.

The first candidate to investigate is LiveCodeBench Pro (`quater_2025_7_9` split), per the user's decision. It is mostly competitive programming and C++, so the headroom screen and the pilot decide whether it is usable. Findings: `results/raw/research-20261016-verify/D_stage1_dataset.md`.

Status 2026-10-16. Read from public Hugging Face API metadata only: nothing downloaded, no terms accepted.
- `QAQAQAQAQ/LiveCodeBench-Pro` exists.
  - Gated ("auto"), apache-2.0 tag.
  - It has a split `quater_2025_7_9` with **144 problems**. (The research agent wrongly reported this split as not found.)
  - Fields: problem_id, title, difficulty, statement, platform, link, time and memory limits. **No tests and no reference solutions in this dataset.**
- `QAQAQAQAQ/LiveCodeBench-Pro-Testcase` is not gated and has no licence tag.
  - 706 per-problem zip files, last modified 2025-09-21.
  - **The highest contest id is 2121.** Mapping contest ids to dates is an estimate: about June 2025. If that holds, the July-September 2025 problems may have **no test cases** in this repository. This is unverified until the 144 problem ids are compared with the zip list, which needs the user to accept the gate.
- The official judge needs Docker (go-judge). We would need our own grader inside the S4 jail, including any special checkers in the zips.
- No candidate dataset found so far meets every requirement in this section. D1 stays **unresolved**.

Split: problems grouped by contest. DEV (pilot + tuning) and HOLDOUT (read once per arm at the end). The split manifest is hashed before any generation.

## 6. Oracle adequacy (the evaluation's own tests)

Per problem:
- every reference solution passes all hidden tests;
- a trivial wrong program fails;
- mutation adequacy: generated mutants of the reference solution that are not equivalent, killed by the hidden tests, divided by all non-equivalent mutants. It must be >= **0.80** (frozen). Problems below it are excluded and counted.
- Flaky tests (status differs between two runs) exclude the problem.

## 7. Statistics

- **Design:** paired by problem. Outcome per arm per problem is solved (1) or not (0).
- **Confirmatory contrasts (stage 1), Holm-corrected at family alpha 0.05:**
  - C vs B (does verification beat extra compute);
  - D0 vs C (does the controller add value);
  - D1 vs D0 (does the critic add value).
- **Reported without a gate:**
  - B vs A;
  - E vs D1 (exploratory in stage 1; probe and random-head versions beside it);
  - each arm vs the simplest arm that beat B.
- **Tests:** exact McNemar per contrast; paired bootstrap 95% intervals (2,000 resamples, resampled by contest cluster).
- **Seeds:** sampling seeds fixed and recorded per problem; arms share seeds (section 1).
- **Sample size** (estimate, unmeasured): to detect a 5-point paired gain with discordant share 0.15, alpha 0.05 two-sided and power 0.8, about 470 problems are needed (standard paired-proportion formula; my calculation). The pilot measures the real discordant share, and the main-run size is fixed from it before the main run. If the eligible pool is smaller, the study is reported as underpowered for effects below the detectable size, and that size is stated.
- **Headroom screen:** if best-of-k by hidden tests (oracle) minus pass@1 is below 5 points on DEV, a selection method has little room. The screen is reported and the user decides whether to continue. It is a heuristic, not a verdict.

## 8. Gates

| stage | question | rule |
|---|---|---|
| stage 1 | does the controller improve Qwen? | D0 > C, Holm-adjusted p < 0.05, CI lower bound > 0 |
| stage 1 | does GLM add value? | D1 > D0, same rule; reported as external-model assistance |
| stage 1 | safety | false-approval rate of D0 and D1 not above C's |
| stage 1 | LAYA | **exploratory only; no keep/drop decision** |
| stage 2 | does the adapter + LAYA improve next-action selection and task completion? | E > D1 (or > D0 if D1 is not kept), plus E > probe and E > random-head in the same slot |

## 9. Stress set (part of DEV; reported separately)

- a problem with a deliberately wrong statement example;
- a broken test tool (must give `error`, never `pass`);
- planted wrong candidates that pass the public examples;
- an ambiguous statement;
- misleading critic output (a malformed or out-of-scope experiment must be rejected).

## 10. Pilot (after the P0 exit gate and the user's GO)

- **Stage 0, Kaggle CPU:** feasibility checks FC-1 (jail), FC-2 (GLM reasoning parameter), FC-3 (Qwen generation throughput, which sets T), FC-4 (hidden tests encrypted and decrypted in memory only); all S1-S5 tests; S4 red-team tests.
- **Stage 0b:** about 40 DEV problems, every arm.
  - Measures headroom, discordant share, cost per arm and stop causes.
  - Tunes nothing beyond section 4's list.
  - No confirmatory claims from pilot problems; they are excluded from the main analysis.
