# P1 offline grading of stored candidates (2026-10-20)

**Label: DEVELOPMENT evidence only.** The development grader's cases are public in Git history (not a sealed benchmark). Grading ran offline, after selection, on stored code only:
no Qwen, no GLM, no new candidates, no package install, no training, no Kaggle, no controller change. Original runs and receipts are unchanged (journal hashes in `PROVENANCE.json`).
Outputs here hold counts and failure classes only, never case inputs or expected values.

## What was graded
- 36 finished tasks: 28 Python (`el-bank-local-20261018T2106`), 5 hard Java (`el-bank-java5hard-wslc-20261018T213058`), 3 Java from the stopped 20-task run (`el-bank-java20-wslc-20261018T211547`: J1-17, J1-20, J1-23; that run is NOT a completed 20-task benchmark).
- 162 stored candidates: 141 fresh samples, 21 repair children. Pool per task 3-6 (lineage per candidate in `CANDIDATES.jsonl`).
- Grading A (`CANDIDATES.jsonl`, `tools/el_p1_offline_grade.py`): trusted verifier (one fresh process per input, verdict host-side), Python on MXC process container, Java on MXC WSLC, as in the original runs. 0 infrastructure errors, 0 not evaluated.
- Grading B (`REFERENCE.jsonl`, `tools/el_p1_reference_grade.py`): the reference engine the grader was built on (WSL bubblewrap, one run per candidate), every candidate.
- A and B agree on 160 of 162 candidates. The 2 disagreements are both J1-17 candidates: A says timeout after 9 of 40 cases, B says 40 of 40 pass. Cause: the trusted verifier starts one JVM per input on WSLC (about 2.7 s each) and hits the time limit; this is a grading-harness limit, not a candidate failure. Final values use B for those 2 rows (`FINAL.json` `reconciled_rows`).
- Count reconciliation (asked before grading): of the 23 answered tasks, 9 had exactly one public-passing candidate and 14 had two or more (23 total). My earlier "10 and 16" was wrong. The journal counts match every `result.json` (`public_count_mismatch_vs_result_json` is empty).

## Results (36 tasks, `FINAL.json`)
| Measure | Value |
|---|---|
| Empirical Oracle@k (at least one stored candidate passes the development grader) | **17 / 36** |
| Chosen-pass (system answered and its answer passes) | **16 / 36** |
| **Selection misses** (a passing candidate existed, the answer did not pass) | **1: P1-21** |
| **Generation gap** (no stored candidate passes) | **19 / 36** |
| Answered (SUPPORTED) but fails the development grader | **7 / 23**: P1-14, P1-21, P1-24, P1-27, P1-28, P1-34, J1-31 |
| Public-passing candidates that fail the development grader | 10 / 51 |
| Repair children that pass | 1 / 21 (P1-19) |
| Python: tasks / oracle / chosen-pass | 28 / 16 / 15 |
| Java: tasks / oracle / chosen-pass | 8 / 1 / 1 |
| Medium: tasks / oracle / chosen-pass | 21 / 12 / 11 |
| Hard: tasks / oracle / chosen-pass | 15 / 5 / 5 |

The one selection miss: P1-21 had 2 public-passing candidates, 1 of them passes the grader; the system answered with the other (48 / 66 cases).
Of the 7 wrong SUPPORTED answers, 6 come from tasks where NO stored candidate passes (a stronger runtime oracle would have turned them into BLOCKED, not into solves);
only P1-21 had a correct alternative. Examples of the wrong answers: J1-31 crashes on very large numbers (`NumberFormatException`, the Long-range case); P1-14 fails 1 of 48 cases; P1-28 passes 5 of 58.

## Per task
| task | lang | diff | label | pool (samples+repairs) | public-pass | grader-pass | answer passes | seconds |
|---|---|---|---|---|---|---|---|---|
| P1-13 | py | medium | SUPPORTED | 4 (4+0) | 3 | 3 | yes | 88 |
| P1-14 | py | medium | SUPPORTED | 4 (4+0) | 1 | 0 | no (47/48) | 75 |
| P1-15 | py | medium | SUPPORTED | 4 (4+0) | 3 | 3 | yes | 68 |
| P1-16 | py | medium | SUPPORTED | 4 (4+0) | 1 | 1 | yes | 85 |
| P1-17 | py | medium | SUPPORTED | 4 (4+0) | 2 | 2 | yes | 477 |
| P1-18 | py | medium | SUPPORTED | 4 (4+0) | 4 | 4 | yes | 63 |
| P1-19 | py | medium | SUPPORTED | 5 (4+1) | 1 | 1 | yes (repair) | 132 |
| P1-20 | py | medium | BLOCKED | 4 (4+0) | 0 | 0 | - | 111 |
| P1-21 | py | medium | SUPPORTED | 4 (4+0) | 2 | 1 | **no: selection miss** (48/66) | 227 |
| P1-22 | py | medium | SUPPORTED | 4 (4+0) | 1 | 1 | yes | 100 |
| P1-23 | py | medium | BLOCKED | 6 (4+2) | 0 | 0 | - | 97 |
| P1-24 | py | medium | SUPPORTED | 4 (4+0) | 1 | 0 | no (7/40) | 58 |
| P1-25 | py | medium | SUPPORTED | 3 (3+0) | 3 | 2 | yes | 75 |
| P1-26 | py | medium | BLOCKED | 6 (4+2) | 0 | 0 | - | 71 |
| P1-27 | py | medium | SUPPORTED | 4 (4+0) | 3 | 0 | no (38/49) | 386 |
| P1-28 | py | medium | SUPPORTED | 4 (4+0) | 1 | 0 | no (5/58) | 100 |
| P1-29 | py | medium | SUPPORTED | 4 (4+0) | 2 | 2 | yes | 251 |
| P1-30 | py | medium | SUPPORTED | 4 (4+0) | 3 | 3 | yes | 89 |
| P1-31 | py | hard | BLOCKED | 5 (4+1) | 0 | 0 | - | 137 |
| P1-32 | py | hard | SUPPORTED | 4 (4+0) | 4 | 4 | yes | 75 |
| P1-33 | py | hard | SUPPORTED | 4 (4+0) | 1 | 1 | yes | 86 |
| P1-34 | py | hard | SUPPORTED | 3 (3+0) | 1 | 0 | no (62/66) | 408 |
| P1-35 | py | hard | SUPPORTED | 4 (4+0) | 4 | 4 | yes | 96 |
| P1-36 | py | hard | BLOCKED | 6 (4+2) | 0 | 0 | - | 113 |
| P1-37 | py | hard | BLOCKED | 5 (4+1) | 0 | 0 | - | 122 |
| P1-38 | py | hard | SUPPORTED | 4 (4+0) | 4 | 4 | yes | 126 |
| P1-39 | py | hard | SUPPORTED | 4 (4+0) | 3 | 3 | yes | 81 |
| P1-40 | py | hard | BLOCKED | 5 (4+1) | 0 | 0 | - | 126 |
| J1-31 | java | hard | SUPPORTED | 6 (4+2) | 1 | 0 | no (0/40, crash on large numbers) | 392 |
| J1-32 | java | hard | BLOCKED | 6 (4+2) | 0 | 0 | - | 125 |
| J1-33 | java | hard | BLOCKED | 5 (3+2) | 0 | 0 | - | 296 |
| J1-34 | java | hard | BLOCKED | 5 (4+1) | 0 | 0 | - | 124 |
| J1-35 | java | hard | BLOCKED | 5 (4+1) | 0 | 0 | - | 138 |
| J1-17 | java | medium | SUPPORTED | 4 (4+0) | 2 | 2 | yes | 464 |
| J1-20 | java | medium | BLOCKED | 6 (4+2) | 0 | 0 | - | 116 |
| J1-23 | java | medium | BLOCKED | 5 (4+1) | 0 | 0 | - | 183 |
Costs: task wall seconds from `result.json` (model, sandbox and critic together); per-call token counts are not in these task receipts, so token-matched comparison starts in P2 (budget accounting).

## What this means for V2.1 (decision table agreed before the run)
1. **Generation is the main bottleneck: 19 of 36 tasks have no correct candidate in the stored pool** (Java 7 of 8). Priority 1.
2. **Public examples are a weak oracle: 7 of 23 answered tasks are wrong** on the development grader (10 of 51 public-passing candidates fail). Priority 2: runtime oracle strengthening (property checks, distinguishing inputs). Its main effect will be to turn wrong SUPPORTED answers into honest BLOCKED / UNVERIFIED (6 of the 7 have no correct alternative), and to catch the P1-21 case (two public-passing candidates that differ).
3. **Selection is not the bottleneck: 1 miss in 36 tasks** (P1-21). No learned ranker or judge is justified by this data.
4. **Repair as built rarely helps: 1 of 21 repair children passes.** Consistent with the anchoring finding; P2 should compare fresh resampling, failure-aware resampling and counterexample-only repair at equal budget.
5. **Harness issue to fix before WSLC is used for Java grading:** the trusted verifier's one-JVM-per-input launch times out on longer case lists (J1-17). Needs a per-input time accounting change or a single-JVM trusted runner; until then Java grading uses the reference engine.

Limits: development evidence on 36 tasks (15 hard, 8 Java); pools of 3-6 candidates are not independent random samples (repair children depend on parents), so Oracle@k describes opportunity in the saved pools, not the model's limit.

Files: `TASKS.jsonl`, `CANDIDATES.jsonl` (grading A), `REFERENCE.jsonl` (grading B), `FINAL.json` (reconciled metrics), `SUMMARY.json` (grading A only, before reconciliation), `PROVENANCE.json`, `SHA256SUMS*.txt`.
