# H2-lite round 1: verified results and stopping decision

**Decision: retain baseline B; no eligible H2-lite winner. Stop the frozen search. H2-lite is NOT QUALIFIED FOR P6.** Confirmation was not reached and was not run. No hidden grading was performed on SEARCH.

## A. Experimental integrity

The working branch is `v1.5-p3-verification`. `git pull --ff-only origin v1.5-p3-verification` succeeded and reported already up to date at `fb0424ba8d6decf4ac5ee3d515fd446c4a6a9d78`. That commit already contained the round-1 exports; this handoff recovered and audited them rather than regenerating them. The inference bundle records commit `a198b3de58fef1b3f0ef4239f69d950723e9d931`.

Run: [sangeetasangamkar/el-h2lite-v1](https://www.kaggle.com/code/sangeetasangamkar/el-h2lite-v1), private. A fresh authenticated status check reported `KernelWorkerStatus.COMPLETE`. `RUN_SUMMARY.json` records start **2026-10-21 13:28:38 UTC**, end **16:02:05 UTC**, verdict `OK`, and runner exit code 0. The recovered full kernel log independently contains `VERDICT OK`.

Verified configuration:

- Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`. All 489 generation-log entries name this model/revision. The kernel records successful model-file SHA256 validation and a `Tesla T4` fp16 check.
- Policy: unchanged B sampling, `v2min_v1.2`, no acceptance gate during selection; `decl_v1` contracts checked on the selected answer for scoring. G = 6,000 generated tokens **after** the four round-1 samples, rather than a 6,000-token total cap.
- All 130 files in the local staged bundle match its manifest. Every bundled file also present in the current checkout matches its bundle hash. The runner's recorded code hash matches the bundle.
- Exact frozen SEARCH task order and membership: P1-02, 04, 06, 07, 09, 12, 29, 30, 35, 38, 39, 11, 21, 23, 26, 34, 16, 28, 37, 14, 20, 25. No CONFIRMATION task appears in the result set.
- All **66 unique task/configuration pairs**, 66 selected-answer records and **489 calls** are present. Every call reconciles with its generation log, including task, seed, prompt hash, template, extraction status and token counts. No cache reuse occurred, so attributed and newly generated token totals coincide.
- Each task/configuration begins with seeds 1–4 and uses the prescribed continuation seeds starting at 104. Prompt hashes were reconstructed from the actual statements, examples and frozen/adapted templates. Observed outputs fit the reconstructed per-call caps; every unaccepted task exhausted the post-round-1 budget. The logs omit the requested `max_new_tokens` field, so actual request caps are supported by the hashed runner code rather than independently logged request fields.
- Existing jail self-check PASS and all ten jail-port qualification checks were reused. No environment qualification or successful C++ validation was rerun.

Canonical exports are under [`results/raw/h2lite-search-r1/`](results/raw/h2lite-search-r1/). The scoring inputs are `h2_search/RESULTS.jsonl`, `CALLS.jsonl`, `gen_log.jsonl`, `ANSWERS.jsonl`, `RUN_CONFIG.json`, `SUMMARY.json`, and `DONE.json`. The proposals are [`results/raw/h2lite-editor-v1/round1_proposals.json`](results/raw/h2lite-editor-v1/round1_proposals.json). The staged bundle is `results/raw/el-h2lite-bundle-search-20261021T132739Z/dataset/`.

The fresh download matches all 13 non-kernel-log original files byte for byte. The original `el-h2lite-v1.log` was empty; the new nonempty copy is preserved at [`results/raw/h2lite-handoff-20261021/remote-output/el-h2lite-v1.log`](results/raw/h2lite-handoff-20261021/remote-output/el-h2lite-v1.log). Original exports were not overwritten. The first download encountered a console encoding error after downloading the data; a UTF-8 retry recovered the kernel log. This was a local download issue, not an experiment failure.

Key SHA256 hashes (the [machine-readable receipt](results/raw/h2lite-handoff-20261021/SCORING_RECEIPT.json) contains the full inventory):

| Artifact | SHA256 |
|---|---|
| Frozen H2 plan | `95519025acc80c8677f10ff506f09d0c08be2d35d980114168bce92da2fb403f` |
| Bundle manifest | `248ae5327aa43ed9394e0fe6ef9d54aedbf6877ab7a9ffa647145cf2acebfcc4` |
| Proposal JSON file | `5c21119083859b0d0ab841ffab2ea3cb0cc9c5dcbe3610e18b2e0c969132b579` |
| General adapter text hash | `c64b5fd440fce73c23c8fa302cddfaba9687b41d6c9217b901bb32771ce89f1f` |
| Coding adapter text hash | `cfca62403f58339db3ee0a95fc9d9ffcd7479e608682ee6d85ce9e2b934e3a22` |
| RUN_SUMMARY.json | `7babd468fce2f52a6f52963c81a4b0b6c2477900d4074bcd1d24f88c1e5a3158` |
| RUN_CONFIG.json | `71731d0a2b88d092fcacc7b46a039e7bfeba3fd41b32b0fefbbfa25bd952dc61` |
| RESULTS.jsonl | `4818da0dc9f7109998c8df9a64ccf11cdd75c34800eb974f8428b0af823fa5a6` |
| CALLS.jsonl | `36a5eb355449f48032dffd00845ad670fd78b39a7c4295a5be0f70dc3729df99` |
| gen_log.jsonl | `7bab39802267befe98f068892e61cc16019f6bf470fc0ab4509d38ecfc8b6933` |
| ANSWERS.jsonl | `cf24c975cc6c6ae0f0e6f183b26ee4e48d788d68c35f8a6bbe0437f522987265` |
| Recovered full kernel log | `e382b49a0877a1a324cb54397b5a8fec372b5d4bd54b337533ca5e5143eb544c` |

Evidence limits: exact Kaggle session/version ID is absent from the exports; the run is bound by bundle commit, UTC interval, model revision and matching remote artifact hashes. No independent public-check replay was needed: acceptance uses the completed runner receipts. Device-active CUDA timing, allocated GPU count and provider billing are not recorded. SEARCH correctness and false-acceptance rates remain unmeasured. Detailed jail self-check results are represented by the kernel's PASS receipt; port qualification has a separate detailed JSON receipt.

## B. Round-1 results

Computed independently from task and call receipts, then checked against the original summary. All three configurations completed all 22 tasks. Unaccepted answers are completed budget-exhausted outcomes, not missing tasks or infrastructure failures.

| Metric | Baseline B | General | Coding |
|---|---:|---:|---:|
| Public accepted / 22 | 18 | 19 | 18 |
| Contract violations among accepted | 0 | 0 | 0 |
| Extraction failures, generation outputs | 5 | 11 | 4 |
| Score: accepted − 2 × violations | **18** | **19** | **18** |
| Actual generated tokens | 53,595 | 111,354 | 57,776 |
| Token ratio vs B | 1.000000 | 2.077694 | 1.078011 |
| Extra tokens vs B | 0 | 57,759 | 4,181 |
| Generation calls | 174 | 131 | 184 |
| Inference-call wall seconds | 2,136.652 | 4,335.716 | 2,201.695 |
| Configuration wall seconds | 2,273.4 | 4,468.8 | 2,386.6 |
| GPU device-active seconds | N/A | N/A | N/A |
| Accepted: contract pass / unavailable | 3 / 15 | 3 / 16 | 3 / 15 |
| Completed but not publicly accepted | 4 | 3 | 4 |
| Missing/incomplete tasks | 0 | 0 | 0 |
| Unresolved recorded infrastructure failures | 0 | 0 | 0 |
| Token-eligible | **YES** | **NO** | **YES** |

Inference-call wall time is the sum of `CALLS.actual_seconds`: host time around actual backend generation, including backend overhead and initial loading where applicable. It is not a CUDA profiler measurement. Configuration wall time also includes public/contract execution and bookkeeping; sandbox time was not separately logged.

Contracts passed for P1-11, P1-21 and P1-30 in all three configurations. Other accepted answers have `unavailable` contracts, so zero observed violations does not certify universal contract compliance.

B and coding accepted the same task IDs. Both did not accept P1-23, P1-26, P1-37 and P1-20. General did not accept P1-23, P1-26 and P1-34: relative to B it gained public acceptance on P1-37/P1-20 and lost it on P1-34, for a net +1. These are public-test outcomes only.

## C. Selection and stopping

The token threshold is **1.25 × 53,595 = 66,993.75 tokens**; with integer counts the maximum eligible total is 66,993.

1. General scores 19 but generates 111,354 tokens, making it ineligible. Its token increase is **107.77%**.
2. Coding scores 18, ties B and costs 4,181 more tokens (**7.80%**). The frozen tie-break selects B.
3. The best eligible variant improves B's score by **0**, below the required +2. Even ignoring token ineligibility, general's +1 would fail the continuation threshold.

**No H2-lite winner; stop after round 1.** No further editor calls or prompt modifications belong to this frozen search. The existing ledger contains one GLM call and two proposals; neither the maximum three rounds nor the maximum six variants overrides the early-stop rule. No confirmation is forced.

The user's later message permits unlimited calls and requests use of 14 keys with 10 concurrent requests per key and a one-minute per-key cooldown on limits. That changes prospective API authorization, not the measurements already obtained under plan v1. A scope clarification was requested about a separately labeled exploratory search. Its outcome must not replace this frozen result or be presented as a pre-registered continuation. No additional call was made for this report.

## D. Confirmation

**Not run: search admission failed.** This is not a measured confirmation failure, and no confirmation PASS is claimed.

| Metric | B | H2-lite |
|---|---:|---:|
| Independently solved / 18 | N/A | N/A |
| False acceptances | N/A | N/A |
| Contract violations | N/A | N/A |
| Generated tokens | N/A | N/A |
| GPU seconds | N/A | N/A |

All five confirmation conditions (solve difference ≥2, false acceptances ≤B+1, contract violations ≤B, tokens ≤1.25×B, no unresolved infrastructure failure) are **NOT EVALUATED**, because no winning prompt reached this stage. Overall qualification status is **NOT QUALIFIED FOR P6** on the search decision.

## E. Compute accounting

| Recorded usage | Seconds | Hours |
|---|---:|---:|
| Already consumed: round-1 kernel wall time | 9,207.0 | 2.5575 |
| Additional GPU runtime during handoff | 0 | 0 |
| Total measured kernel wall time | **9,207.0** | **2.5575** |
| Original approved runtime allowance | 15,120.0 | 4.2000 |
| Arithmetic remainder of original allowance | **5,913.0** | **1.6425** |

The three configurations account for 9,128.8 wall seconds, including 8,674.063 seconds inside generation calls. The whole kernel is the conservative measured session-runtime charge above. The receipt identifies a T4 on device 0 but does not enumerate provisioned devices or record Kaggle billed GPU-hours; exact billed device-hours are **N/A**, and 2.5575 hours must not be advertised as a verified billing figure. The remaining 1.6425 hours is a runtime-accounting remainder, not a fresh quota check or permission to bypass stopping. The later budget-waiver message is recorded in C; it does not affect consumed runtime. Codeforces qualification was CPU-only and is not H2-lite GPU consumption. No monetary charge is established by these receipts.

## F. P6 readiness and Codeforces accounting

Existing `VALIDATION_v3_counts.json`, `QUALIFIED_rules_v2.json`, `validate-v3_RUN_SUMMARY.json`, `QUAL_cpp_kaggle.json`, the private v3 execution log and historical runner code were inspected. The private validation JSON matches the committed counts file byte for byte. No grading inputs, expected outputs or accepted reference source were exported into this report.

**72 of 77 candidates qualify**, exactly matching the saved list in `sha256("P6-2026-10-21:" + problem_id)` order. Of these, **65** pass with reference 1 and **seven** with reference 2: 2022/A, 2028/A, 2035/A, 2043/A, 2050/B, 2060/A and 2067/A. The kernel reports 75 validated references, but three of those problems have fewer than ten tests and remain excluded. Existing C++ qualification in the same real Kaggle jail reports g++ 13.3, `all_runnable_ok=true`, and a 1,048,576-byte capture limit.

All five exclusions are preserved:

| Problem | Tests | Preserved reason | Actual receipt evidence |
|---|---:|---|---|
| 2033/A | 2 | insufficient_tests | Reference 2 passes 2/2; still below ten tests |
| 2039/A | 2 | insufficient_tests | Reference 2 passes 2/2; still below ten tests |
| 2044/A | 3 | insufficient_tests | Reference 1 passes 3/3; still below ten tests |
| 2036/C | 27 | other | Each of three references: 9 pass, 18 other; no wrong-answer/timeout/runtime/compile failures recorded |
| 2053/B | 27 | other | Each of three references: 16 pass, 11 other; no wrong-answer/timeout/runtime/compile failures recorded |

**Output-limit diagnosis: UNCONFIRMED.** Historical runner code at bundle commit `bca45ad91c1d961b722a74af2590ba261eabbb02` maps native non-ok outcomes to `invalid_output`; the validation wrapper collapses that status, other unrecognized statuses and some exceptions to `other`, discarding per-case detail. The private v3 log retains only the same aggregate tuples. The configured port capture limit is 1 MiB; native classification has a 4 MiB overflow ceiling and checks for output reaching the capture limit. These mechanisms make a capture limitation plausible, but the saved receipts contain no exact per-case failure signature or observed output sizes proving it. Do not treat the earlier commit message's output-limit explanation as a demonstrated diagnosis. No revalidation was run, no exclusion was changed and neither rules version was edited.

P6 is therefore ready only at the cohort-accounting level: **72 qualified candidates, not a timed/final frozen N**. H2-lite does not qualify as a third arm. The existing draft's prospective arms are B and V2-minimal. The 20-task contract cohort and private blind-set verification remain as previously recorded; blind cases were neither opened nor rerun in this handoff.

P6 protocol approval/freeze, a separately authorized timing smoke, determination of N under the protocol's timing rule, and a separate final P6 GO remain pending. Neither final P6 generation/evaluation nor a timing smoke was performed. The two excluded problems need more diagnostic evidence only if someone later seeks to establish their root cause; their uncertainty does not invalidate the other 72 qualification receipts.

## G. What we learned

The general instructions got one more public-test acceptance overall, but used more than twice B's tokens. The coding instructions got the same public acceptance count and the same accepted task set, while costing 7.8% more tokens. Neither meets the frozen improvement rule.

We observed no contract violations where a contract was available. Most accepted tasks had no applicable declared contract, and SEARCH was intentionally not independently graded. Consequently, this experiment does **not** tell us whether the new prompts increased actual correctness or false confidence. `SUPPORTED` means the runtime checks supported an answer; it does not mean the answer is correct.

H2-lite is not worth including as a qualified P6 arm under this plan. This is a negative result for these two proposals on this development split, at the prescribed paired settings and cap. It does not prove that all prompt editing is ineffective, establish general model accuracy, or justify changing the scoring rule after seeing outcomes.

## Deliverables and change scope

The create-only [scoring audit](results/raw/h2lite-handoff-20261021/score_receipts.py) produced the [SCORING_RECEIPT.json](results/raw/h2lite-handoff-20261021/SCORING_RECEIPT.json), including per-task call counts, token totals, scores, eligibility, source hashes, remote comparisons and qualified ordering. Its assertions passed without generation, sandbox execution or hidden grading. The original experiment code and all frozen evidence remain unchanged. The recovered full kernel log is added evidence; redundant fresh copies stay local. Pre-existing unrelated workspace edits were excluded from the reporting commit.
