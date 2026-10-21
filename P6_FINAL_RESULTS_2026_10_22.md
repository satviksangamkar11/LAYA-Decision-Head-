# P6 final results (2026-10-22): development results, frozen protocol

**Status of these results**
- **Protocol:** `P6_PROTOCOL_v1.md` with the logging-only v1.1 amendment. Deployment `P6_DEPLOYMENT_v1.md`; lock `configs/p6_final_lock_v1.json` plus
  `configs/p6_final_lock_v1_CORRECTION_1.md` (tests-manifest hash; grading only).
- **Analysis:** `tools/el_p6_analyze.py`, frozen before generation; output in `results/raw/p6-final-v1/ANALYSIS.json`.
- **Nature of the evidence:** development results on a 1.5B model (Qwen2.5-Coder-1.5B-Instruct, revision 2e1fd397). Codeforces grades use generated tests
  validated by the dataset authors and by our accepted references, NOT complete official tests, with a lenient 6 s per test.

## Run
- **Kernels:** two Kaggle T4 kernels, both verdict OK:
  - k1: 8,481.9 s, 32 Codeforces tasks;
  - k2: 12,570.2 s, 40 Codeforces tasks + D01–D20.
- **Completeness:** 92/92 tasks complete (`COMPLETENESS.json`); the runner sha matches the lock; no infrastructure failures.
- **Grading:**
  - Codeforces: Kaggle CPU, 216 answers, all tests, 6 s per test, timeouts = failures;
  - contract: PC MXC trusted verifier against the private blind set (hash-checked).
- **Infrastructure attempts before the successful grading run** (neither graded anything):
  1. answers-file path (fixed in the grader: lookup by name with a sha256 check);
  2. the lock's tests-manifest hash came from a CRLF working copy (Correction 1).

## Question 1: Codeforces, 72 problems (Oct 2024 – Feb 2025)
| Arm | Solved | Wrong | No answer | Logical tokens |
|---|---|---|---|---|
| A (seed-1 sample) | **1** | 71 | 0 | 15,407 |
| B (multi-sampling, first public pass, G = 6,000) | **8** | 7 (false acceptances) | 57 | 410,966 |

- **B − A:** +7 tasks (+9.7 points), paired bootstrap 95% CI [+2.8, +16.7]; it excludes 0, so **criterion met: multi-sampling solves more than one sample.**
- **Cost:** about 26.7× the logical tokens, about 50,000 generated tokens per extra correct answer. A and B are deliberately NOT equal compute.
- **Identity check:** V2 = B exactly on 72/72 tasks (same answers, requests and tokens; every gate verdict `unavailable`). This is an implementation check, not a
  result.
- **How B's answers failed:**
  - B gave no answer on 57/72 tasks: no sample passed the statement's examples within the budget.
  - B's 7 wrong answers: 4 wrong output, 3 timeout.
- **How A's answers failed:** 71 failures in total: 36 wrong output, 26 runtime error, 9 timeout.
- **Solved:**
  - B: 2065/A, 2050/A, 2036/A, 2034/A, 2060/C, 2062/A, 2042/A, 2060/D;
  - A: 2065/A.

## Question 2: contract tasks D01–D20
| Arm | Solved | Wrong | No answer |
|---|---|---|---|
| A | 9 | 11 | 0 |
| B | 16 | 3 (D08, D10, D15) | 1 (D18) |
| V2 (B + trusted contract gate) | 16 | 3 | 1 |

- **Criteria (pre-registered):** V2 false acceptances <= B (3 <= 3), correct lost = 0, V2 solved >= B (16 >= 16): **all met.**
- **But the gate changed nothing.** All 20 tasks were unchanged from B to V2. The gate returned `pass` on all 19 public-passing answers, including the 3 wrong ones.
  It prevented no false acceptance; it also cost no correct answer.
  - Reason: on these tasks the contracts do not cover the bugs that the hidden cases catch, and a rejecting-only check cannot reject what its contract allows.
  - So "criteria met" here means "no harm", not "benefit".
- **B − A on contract tasks:** +7 (+35 points), 95% CI [+15, +55].

## What this says
1. Multi-sampling with public-example filtering helps on both cohorts, at a large token cost.
2. The trusted contract gate as built is safe but ineffective on this set: it rejected nothing.
3. **The Codeforces failures are mostly generation failures.** On 57/72 problems B found no sample that passes even the statement's examples. This agrees
   with the P5 audit (selection gap 1 task per cohort). Selection-only improvements (P7: consensus or differential selection) can act only on the 15
   answered tasks plus any correct candidate hidden in the logged pools.
   - How large that room is (oracle@k over `CANDIDATES.jsonl`) is **not yet measured**: it needs the logged candidates graded, which is the first P7 step.

## Limitations
- One 1.5B model.
- Codeforces tests are not the official complete tests, and the time limit is lenient.
- The problems were released after the model, but they are not proven absent from its training data.
- The contract cohort has 20 tasks.
- The LAYA head is OFF.
- These are development results; P6's tasks are now development data for P7.
