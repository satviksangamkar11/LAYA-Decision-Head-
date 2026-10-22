# Infrastructure re-grade with the hardened cf grader (2026-10-22)

The grader now records any grader or sandbox exception as `infra_error`, never as a candidate runtime_error. Only the rows whose first failure was
runtime_error were re-graded, with unchanged tests, answers and settings.

| Set | Rows | Verdict | infra_error | Grades changed |
|---|---|---|---|---|
| P6 final answers (all A) | 26 | OK | 0 | 0 (all FAIL, first failure runtime_error) |
| P7 logged candidates | 652 | OK | 0 | 0 (all FAIL, first failure runtime_error) |

Every runtime_error in the original grades is a real candidate crash. The P6 grades, `results/raw/p6-final-v1/ANALYSIS.json` and the P7 headroom
counts stand unchanged; no re-analysis is needed.
