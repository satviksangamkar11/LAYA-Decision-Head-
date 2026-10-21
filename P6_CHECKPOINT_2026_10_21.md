# P6 / H2-lite checkpoint (2026-10-21) — no Kaggle run, no model generation

## 1. H2-lite editor proposals (GLM, 1 of 6 calls used)
`results/raw/h2lite-editor-v1/` (request, response, ledger, `round1_proposals.json`). One call, 881 s, 3,904 prompt / 27,414 completion tokens (26,735 reasoning).
The editor saw only the two frozen templates, the adapter rules and the 22 SEARCH-split development tasks (statements + public examples); no outcome table
existed. Both proposals pass the `PromptAdapter` rules. They are proposals only: the winner is chosen by the pre-registered search, not by GLM.
| Variant | Idea | Added characters (v2 / v2b) | sha256 |
|---|---|---|---|
| general | short plan-then-verify: state the output shape, plan, check the plan against every public example | +201 / +71 | `c64b5fd4…` |
| coding | read only from the input dict, never hard-code example results, return the native type, never print | +351 / +271 | `cfca6240…` |

## 2. Codeforces cohort (option C): retrieval feasibility
- Candidate pool: 77 problems (rating ≤ 1200, contests from 2024-10-01, single-answer, non-interactive) from 39 contests.
- The validated tests are stored as single-row-group parquet files: per contest `generated_tests/test_cases_<contest>.parquet` (8–307 MB each) or
  `verifiable_tests/` (40 shards, 9.3 GB). Range requests cannot isolate one problem's rows (the test columns are one chunk per file).
- **All 39 contests: 6,504 MB. Within the remaining 222 MB: 9 contests, 16 candidate problems.**
- The official tests inside the 27.5 MB metadata file are insufficient (official tests ≤ 400 characters only): median 1 per problem, complete for 3 of 77.
- Downloaded so far: 27.5 MB (metadata file) + < 1 MB of parquet footers.
- Deduplication: no overlap with the 239 earlier tasks (LiveCodeBench v6, development bank, P5 contract and qualification tasks); highest word-set Jaccard 0.35,
  no title match.
- Grader validation (statement / stdin / time limit / expected output through the trusted grader): NOT done — it needs the tests and one accepted solution per
  problem.
- Eligible count today: 77 candidates pass dedup; 0 are frozen (coverage and grader validation pending).

## 3. P6 contract cohort (frozen before any candidate, commit f8f35dfe)
20 new tasks D01–D20 (`configs/p6_contract_tasks_v1/`): statements, author contracts, properties, registry, `-text` pins; all authoritative; all consistent with
their public examples; written by the coordinator (not GLM). Pin checker: 125 pins, 0 mismatches.

## 4. P6 blind grading set (private, outside git and Kaggle)
Author: a fresh-context subagent with access to `private/p6-eval-v1/STATEMENTS.jsonl` and its brief only; reported no accidental exposure.
- 914 cases (20 tasks; 31–55 each, except D02 = 11 and D04 = 16, which are every valid input the statements allow: n 0..10 and n 0..15).
- Author's validator: `all_ok`, reference == alternative == expected on every case, all public examples reproduced, stated properties checked, generator
  byte-identical, no unresolved case.
- Coordinator checks (no case opened): 9/9 file hashes; validator re-run in a scratch copy → all_ok, byte-identical; the coordinator's own implementations
  agree with all 914 expected values (verification only, no repair); **no expected value violates its frozen contract and every input is valid under it**
  (so the contracts do not reject these correct outputs).
- Ambiguities decided by the author (PROVENANCE.md): D18 segments of three or more dots excluded; D03 duplicates give gap 0; D08 "0" alone never a piece;
  D15 code-point comparison; D20 smallest start completing all n moves.
- Hashes: `cases_p6.json` `d8c57f33fd16ddeb0f0b85a9613a2cefb1ac8d60cc2387110faaeb723e4fd780`; `CHECK_p6.json` `9f7d1631…1591b`; `SHA256SUMS_p6.txt` lists all nine.

## 5. GPU estimate (unmeasured; T4; from P5 timings)
- H2-lite search + confirmation: ≈ 4.2 h (§7 of the plan).
- P6, three arms: Codeforces ≈ 400 s per problem (B ≈ 200 s; V2-minimal reuses B's generations where no contract exists; H2-lite generates separately):
  77 problems ≈ 8.6 h; 16 problems ≈ 1.8 h. Contract cohort: 20 × ≈ 200 s ≈ 1.1 h. Total ≈ 9.7 h (77) or ≈ 2.9 h (16).

## 6. Decision needed
How to obtain the Codeforces tests: (a) a Kaggle CPU kernel downloads the 39 contest files on Kaggle, keeps only the candidates' tests and outputs them
(public data; the PC download stays < 250 MB; the same kernel can validate the grader with one accepted human solution per problem); (b) raise the PC
allowance to ≈ 6.5 GB; or (c) keep only the 16 problems that fit 250 MB.
