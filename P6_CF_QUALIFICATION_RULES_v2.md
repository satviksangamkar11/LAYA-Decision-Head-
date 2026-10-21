# P6 Codeforces cohort — qualification rules v2 (written and committed BEFORE any C++ validation result) — 2026-10-21

Supersedes `P6_CF_QUALIFICATION_RULES_v1.md` (kept) ONLY in rule 3. Reason (recorded results, not model performance): the accepted-submission data of the
pinned `open-r1/codeforces-submissions` holds Python solutions for only 2 of the 77 candidates, but accepted C++ solutions for all 77 (2–3 each;
`results/raw/p6-cf-validation-v1/REFCOUNT.json`). Under v1 rule 3, 72 candidates were excluded as `no_python_reference` for lack of a reference, not for any
grading fault. The multi-language trusted sandbox (branch `multilang-sandbox-2026-10-21`, commit bca45ad9; regression in the main checkout: 15/15 suites,
V2-minimal tests, LiveCodeBench adapter 13/13, identical with and without it) can now run C++ under the same guarantees as Python.

Rules 1 (provenance) and 2 (at least 10 tests) are unchanged. Rule 3 v2:
3. Grading path validated in the trusted jail: an accepted human solution in a language QUALIFIED on the real Kaggle jail in the same kernel run (Python; C++
   compiled `g++ -O1 -std=c++17` as the qualified toolchain), taken in submission-id order across those languages, first of up to 3, passes EVERY test through the
   LiveCodeBench stdin runner (official comparison rules, 6 s per test). A reference that does not compile under the qualified toolchain is a failed attempt
   (the next one is tried); it is not by itself a reason to exclude the problem.
   The C++ toolchain must pass the multi-language qualification (`tools/el_multilang_qual.py --backend kaggle-jail --languages cpp`) inside the same Kaggle kernel
   before any validation; otherwise the run stops and is reported (no fallback).
Exclusion reasons: `missing_provenance`, `insufficient_tests`, `no_reference` (no accepted solution in a qualified language), `reference_timeouts` (every
attempted reference fails only by timeouts), `reference_compile_errors` (every attempted reference fails to compile), `grading_mismatch` (a reference gets a
wrong answer or a runtime error on some test). Mixed failures are recorded per attempt; the problem's reason is `grading_mismatch` if any attempt got a wrong
answer or runtime error, else the dominant remaining reason.
Final N (unchanged): qualified problems ordered by sha256("P6-2026-10-21:" + problem id), first N, N from the 10-hour rule after a separately approved timing smoke.
