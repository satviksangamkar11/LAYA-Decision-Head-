# P6 Codeforces cohort — qualification rules v1 (written and committed BEFORE the retrieval / validation results exist) — 2026-10-21

Inputs: the 77-problem candidate pool (`data/p6_candidates/candidates77.json`: rating <= 1200, contest >= 2024-10-01, single-answer, non-interactive) and the
outputs of the Kaggle CPU kernel `el-p6-cf-tests-v1` (`kaggle/el_p6_cf_tests_v1.py`; pinned open-r1/codeforces `fbe3f6e9…`, codeforces-submissions `07e7bf31…`).
Exclusions follow ONLY these technical criteria; model performance is never looked at (no model has run on these problems).

A candidate QUALIFIES only if all hold:
1. Provenance: metadata row present (contest id, index, contest start, statement fields, examples, time limit), Codeforces URL derivable, dataset license recorded.
2. Grading coverage: at least 10 tests (generated tests validated by the dataset authors, plus complete official tests where present).
3. Grading path validated in the trusted jail: an accepted PYTHON human solution (the first of up to 3, in submission-id order) passes EVERY test through the
   LiveCodeBench stdin runner (official comparison rules, 6 s per test).
Exclusion reasons, recorded per problem (one of):
- `missing_provenance` (rule 1),
- `insufficient_tests` (< 10 tests),
- `no_python_reference` (no accepted Python solution: the jail runs Python only, so the grading path cannot be validated there),
- `reference_timeouts` (every attempted reference fails only by timeouts: execution requirement unsupported at 6 s for Python),
- `grading_mismatch` (a reference gets wrong answers or runtime errors on some test: invalid grading data or a checker would be needed).

Final N (after a separately approved timing smoke): the qualified problems ordered by sha256("P6-2026-10-21:" + problem id), first N taken, N set by the
10-hour rule of the P6 protocol. Grading data stays private (Kaggle private output / git-ignored local copy), never in git, GLM context or a GPU bundle.
