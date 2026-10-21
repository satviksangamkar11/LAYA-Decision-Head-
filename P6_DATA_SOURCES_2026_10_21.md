# P6 fresh-task sources — investigation (2026-10-21)

Scope (user GO): find recent contest tasks with reliable independent grading, deduplication and clear provenance; at most 250 MB downloaded at first.
No problem statement, example or test was read; only source documentation and per-problem METADATA (dates, ratings, test counts).
Downloaded: `data/p6_candidates/open-r1-codeforces-verifiable-test/test-00000-of-00001.parquet` (27.5 MB, sha256 `fb1389c7…e221e3`; git-ignored).

## 1. What exists for 2026
| Source | 2026 problems | Reliable independent grading available? |
|---|---|---|
| LiveCodeBench | no release after v6 (v6 ends 2025-04-06; a "v7?" issue from Dec 2025 has no answer) | — |
| AtCoder | 2026 contests exist | **no**: AtCoder has suspended publication of its official test cases (announcement atcoder.jp/posts/1377, undated, no resumption plan) |
| Codeforces | 2026 contests exist | **no**: complete official tests are not published (long tests are truncated) |
| LeetCode | 2026 contests exist | **no**: tests are not published |
**Finding: no 2026 source currently offers official, independently validated test data.** A 2026 cohort would need test suites authored by us
(an independent author, dual implementations, like the blind contract set): workable, but weaker grading than tests validated against many human solutions.

## 2. Best candidate with validated tests: `open-r1/codeforces`, `verifiable` config, `test` split
- Provenance: Hugging Face `open-r1/codeforces` (cc-by-4.0 on the page header, ODC-By in the card); every row carries `contest_id`, `index`, `contest_start`
  (problem URL derivable: codeforces.com/contest/<contest_id>/problem/<index>).
- Grading: tests generated with edge-case generators, outputs kept only when SEVERAL correct human solutions agree, ranked by how many incorrect
  submissions they reject; a problem is "verifiable" only if ≥ 3 human solutions pass its official tests. Multi-answer problems use a generated checker
  validated on correct human solutions.
- Metadata of the test split (422 problems): contests 2024-08-04 → 2025-02-11; all `stdio`; 401 with validated generated tests, 21 with complete official tests.
- Fresh for the model: Qwen2.5-Coder-1.5B weights are from 2024-09-18. Not 2026, but after the model; never used in this project.
- Usable subset (metadata only): rating ≤ 1200, contest ≥ 2024-10-01, single-answer (no checker), non-interactive → **77 problems from 39 contests**
  (ratings: 800 × 40, 900 × 9, 1000 × 7, 1100 × 11, 1200 × 10). With the cut at 2024-09-19: 85 problems.
- Obstacle: the validated tests for the test split are in `verifiable_tests/`, 40 shards, **9.3 GB** in total (largest 782 MB), split by row, not by contest.
  Staying within 250 MB requires fetching only the selected problems' rows (parquet range reads of the `problem_id` column first, then only the needed row
  groups, or the Hugging Face dataset query API). Not attempted yet.
- Fit with our runner: stdin programs, exactly LiveCodeBench's stdin format (lcb_runner already supports it); Codeforces time limits differ from our 6 s;
  multi-answer problems are excluded rather than adding a checker.
- Deduplication still to do: by source id and normalized statement text against LiveCodeBench v6 (all 175; v6 holds LeetCode and AtCoder problems),
  all P5 tasks and the development bank.

## 3. Options
| Option | Freshness | Grading reliability | Cost / risk |
|---|---|---|---|
| A. open-r1 Codeforces test split (77 problems, Oct 2024 – Feb 2025) | after the model, not 2026 | strong: validated against several human solutions | selective test retrieval must fit 250 MB (feasibility check first) |
| B. 2026 statements (AtCoder / Codeforces / LeetCode) + our own blind-authored test suites | 2026 | moderate: independent dual implementations, no human-solution validation | heavy authoring for ~80 problems; statement copyright: store privately |
| C. A for the benchmark cohort + the planned 20 new contract tasks with a blind set | as A | as A + our blind set | as A |

## 4. Recommendation
Option C, i.e. A as the benchmark cohort, subject to one feasibility step: fetch only the `problem_id` index of the 40 test shards (range reads, a few MB),
then estimate the bytes needed for the 77 problems' tests; proceed only if it fits the 250 MB limit (or ask for more). If the user requires 2026 problems,
option B is the only route and its grading would be weaker; that is a trade-off for the user.
The P6 contract cohort (20 new tasks + blind set) is independent of this choice and follows the role rules of the P6 draft.
