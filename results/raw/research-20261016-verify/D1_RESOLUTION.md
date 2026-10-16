# D1 check: does LiveCodeBench-Pro have tests for post-2025-07-31 problems? (2026-10-16)

Method (public metadata only; nothing downloaded from the gated dataset, no terms accepted):
- `QAQAQAQAQ/LiveCodeBench-Pro` (gated): split `quater_2025_7_9` has 144 problems; fields: statements, limits, no tests (HF API card).
- `QAQAQAQAQ/LiveCodeBench-Pro-Testcase` (not gated): 706 zip files named `<contest><problem>.zip`; contest ids 1983 to 2121 (106 distinct ids; saved in `data/lcbpro_testcase_meta.json`).
- Codeforces public API contest list (`data/cf_contest_list.json`): start dates for those ids.

Result: the latest contest start among the covered ids is 2025-06-23 (id 2112); contest 2121 started 2025-06-17; 2122 started 2025-07-19; none of the covered contests starts on or after 2025-07-31.
Conclusion: the July-September 2025 split has no test cases in the public test repository. Using it would require our own judge and our own tests, which defeats the point of an external set.
Other candidates (see D_stage1_dataset.md) fail the date, the offline-grading or the licence rule. D1 stays unresolved externally; the proposal is a recorded waiver for development banks.
