# Stage 1 dataset verification: post-July-2025 coding benchmarks (research only)

Date of research: 2026-10-16. Scope: web research only (WebSearch / WebFetch). No dataset downloaded, no terms accepted, no login, no code executed.
Evidence tags: [read] = page fetched and read; [snippet] = seen only in search-result text, not fetched.
Requirement: problems published after 2025-07-31; executable tests; hidden tests preferred; grading without Docker; Python preferred, C++ acceptable.

## Bottom line

No benchmark checked so far meets all of the requirements. The official LiveCodeBench has no release after v6, and v6 ends April 2025. The only candidates with post-July-2025 material (LiveCodeBench-Pro, CodeAssay, AutoCodeBench) each fail on one or more of: data access, licence, Docker, language, or unverified dates. Items marked NOT FOUND below were not verifiable from the pages reachable through the tools.

---

## 1. LiveCodeBench (official)

| Check | Verdict | Evidence |
|---|---|---|
| Latest release is release_v6, end date Apr 2025 | CONFIRMED (with conflict) | GitHub README: "release_v6 ... problems released between May 2023 and Apr 2025 containing 1055 problems" [read] https://github.com/LiveCodeBench/LiveCodeBench |
| HF dataset card lists the same latest tag | DIFFERENT | HF card table ends at release_v5 (Jan 2025, 880 problems) [read] https://huggingface.co/datasets/livecodebench/code_generation_lite . The HF card's own tag list does not include v6 when fetched. |
| Any release_v7 or later | NOT FOUND | No v7 in the README [read]. Open issue #139 "When will Version 7 be released?" (opened 2025-12-23) has no maintainer reply: "no new dataset has been released in over six months" [read] https://github.com/LiveCodeBench/LiveCodeBench/issues/139 |
| Official post-July-2025 problems | NOT FOUND | No such window in any fetched source [read]. |
| Issue tracker | CONFIRMED (partial) | Open issues include #153 (2026-09-05, "--not_fast silently ignores --start_date/--end_date and --release_version") and #145 ("Clarification on LiveCodeBench v6 dataset size (454 vs 175 tasks)") [read] https://github.com/LiveCodeBench/LiveCodeBench/issues . Closed issues not visible. |
| Licence | CONFIRMED (code) | "MIT license" on the GitHub repo page [read]. Dataset licence on HF is only the tag "cc" [read]. |
| Docker / sandbox | NOT FOUND | Neither term appears in the README [read]. |
| Dataset size note | DIFFERENT | Problem count for v6 = 1055 (GitHub, lighteval mirror [snippet]); HF card lists v5 = 880 [read]. |

Caveat: the HF card shows test cases are "hidden test cases" with "pruned and sampled tests" in the lite version [read]. Whether private tests are shipped in the HF files is not stated.

## 2. LiveCodeBench-Pro

| Check | Verdict | Evidence |
|---|---|---|
| Split "quater_2025_7_9" (or any quarter split name) | NOT FOUND | No source names this split [snippet]. BenchLM mentions quarter-specific leaderboards such as "25Q2" [snippet] https://benchlm.ai/benchmarks/livecodebench-pro . The string may be a typo. |
| Problems per post-July-2025 split | NOT FOUND | Not stated anywhere reachable. |
| Difficulty labels | DIFFERENT / unofficial | Official text: "medium-difficulty problems" and "hard problems" [read] https://arxiv.org/abs/2506.11928 . Elo bands (Easy <=2000, Medium 2000-3000, Hard >3000) appear only on a third-party aggregator [snippet] https://www.alphaxiv.org/benchmarks/university-of-washington/livecodebench-pro . |
| Sources: Codeforces / ICPC / IOI | CONFIRMED | "composed of problems from Codeforces, ICPC, and IOI" [read] https://arxiv.org/abs/2506.11928 |
| Problem count 584 | NOT FOUND in fetched text | Stated only in a search snippet [snippet]. The arXiv abstract page fetched did not give a count. |
| Dataset ID and access | CONFIRMED (gated) | Official GitHub names HF "QAQAQAQAQ/LiveCodeBench-Pro" and "QAQAQAQAQ/LiveCodeBench-Pro-Testcase" [read] https://github.com/GavinZhengOI/LiveCodeBench-Pro . The main card is gated: "You need to agree to share your contact information to access this dataset" [read] https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro . Licence shown as apache-2.0 [read]. Card has an empty README; splits, fields and sources not stated [read]. |
| Test cases (inputs + expected outputs) in the dataset | NOT FOUND | The "-Testcase" repo is 15.3 GB, has no dataset card ("No dataset card yet") and the viewer finds no supported data files [read] https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro-Testcase . The repo has 4,132 downloads last month [read]. Whether it contains the tests is not stated. |
| Public judge or external submission | DIFFERENT | The toolkit runs a local judge server; the README lists "docker (for running the judge server)" and go-judge sandboxing [read] https://github.com/GavinZhengOI/LiveCodeBench-Pro . Results are also emailed to the maintainer for a leaderboard [read]. |
| Docker required by official judge | CONFIRMED (yes) | Same README, prerequisite list [read]. |
| Special checkers | NOT FOUND | Not stated. |
| Reference solutions | NOT FOUND | Not stated. |
| Small-model (<=8B) scores | NOT FOUND | Abstract reports only the best model: "53% pass@1 on medium-difficulty problems and 0% on hard problems" [read] https://arxiv.org/abs/2506.11928 . No small-model figures. |
| Post-July-2025 problem dates | NOT FOUND | Collection window not stated in the fetched abstract [read]. |

## 3. Alternatives (up to 6)

Only candidates that were actually fetched or seen in search results are listed.

| # | Name / URL | Date window | Problems | Language | Tests included / hidden? | Licence / access | Small-model scores | Fit |
|---|---|---|---|---|---|---|---|---|
| A | CodeAssay, arXiv 2608.03535 https://arxiv.org/abs/2608.03535 | Submitted 4 Aug 2026 (v2 17 Sep 2026) [read]; problem dates NOT FOUND | 185 [read] | Python only [read] | Public tests for generation/repair plus hidden tests "reserved for grading" [snippet]; withheld status not explicit [read] | Licence NOT FOUND; data/code URL NOT FOUND [read] | None; "seven proprietary LLMs" [read] | Best Python + hidden-test + date match on paper. Data availability unknown. |
| B | AutoCodeBench, tencent/AutoCodeBenchmark https://huggingface.co/datasets/tencent/AutoCodeBenchmark | Paper published 12 Aug 2025 [read]; problem dates NOT FOUND; problems are synthesized, not contest-dated [read] | 3,920 full; 1,586 Lite; 1,000 Complete [read] | 20 languages [read] | Yes: public `demo_test_func`, private `full_test_func` [read]; `canonical_solution` per record [read] | Licence "terms of the LICENSE file", type not named [read]; gated NOT FOUND | NOT FOUND | Fails no-Docker: sandbox is Docker image `hunyuansandbox/multi-language-sandbox:v1` with `--cap-add=NET_ADMIN` [read]. Dates unverified. |
| C | AetherCode, m-a-p/AetherCode https://huggingface.co/datasets/m-a-p/AetherCode | v1_2024 contests 2024-01 to 2025-05; v1_2025 Jan-May 2025 [read] | 400 public + 56 private [read] | Checkers are C++ testlib [read]; Python support NOT FOUND | Public set has "complete test cases and checkers"; private set has none [read] | CC-BY-4.0 [read] | NOT FOUND | Fails date (all problems before 2025-07-31). Listed for completeness. |
| D | LiveBench coding, livebench/coding https://huggingface.co/datasets/livebench/coding | Question dates 2023-11 to 2024-03; livebench release 2024-06 to 2024-07; removal 2025-04-02 [read] | 128 rows [read] | Python [read] | Public and private test cases, stored as strings (`input`, `output`, `testtype`) [read] | Licence NOT FOUND [read] | NOT FOUND | Fails date. GitHub says latest release is 2025-04-25 [read] https://github.com/LiveBench/LiveBench . Docker only for agentic tasks [read]. |
| E | LiveOIBench, arXiv 2510.09595 https://arxiv.org/abs/2510.09595 | Contests 2023-2025 [read]; post-July-2025 share NOT FOUND | 403 [read] | C++ vs Python NOT FOUND | About 60 official tests per problem; public status NOT FOUND [read] | arXiv licence CC BY-NC-SA 4.0 on the paper; data licence NOT FOUND [read] | 34 models; GPT-OSS-120B at 60th percentile; no small-model result [read] | Offline evaluation claimed [read]. Non-commercial licence and language unknown. |
| F | open-r1/codeforces https://huggingface.co/datasets/open-r1/codeforces [snippet only] | Covers contests up to 2025; eval split late 2024 to early 2025 [snippet] | >10k problems [snippet] | Not verified | Official tests plus separately downloaded generated tests (~110 GB) [snippet] | cc-by-4.0 [snippet] | NOT FOUND | Fails date for eval split. Too large for a quick Kaggle run. |

Also checked and excluded for date reasons: OJBench (arXiv 2506.16395, June 2025, 232 NOI/ICPC problems) [snippet]; HumanEvalNext (arXiv 2503.05860, March 2025, 164 problems, function-level Python) [snippet]. Also SWE-bench Live and FeatBench need Docker per task [snippet]. LeetCodeDataset (April 2025) predates the window [snippet].

## 4. Ranking for our need (top 3)

1. **CodeAssay (arXiv 2608.03535)**. Python, hidden tests, submitted after the cutoff, 185 tasks. Main risk: no data or code URL and no licence were found, so the tests may not be released at all. Check the paper's data statement before any use.
2. **LiveCodeBench-Pro (QAQAQAQAQ/LiveCodeBench-Pro)**. Official source of post-cutoff contest problems, Codeforces/ICPC/IOI. Main risks: the quarter split is unverified; the judge needs Docker; the language is likely C++; test-case access and licence are not confirmed; the HF gate needs user consent that I could not do.
3. **AutoCodeBench (tencent/AutoCodeBenchmark)**. Largest set, Python among 20 languages, public and private tests and canonical solutions included. Main risks: synthesized problems with no contest dates, a Docker sandbox, and an unnamed licence. Post-cutoff status is unverified.

Fallback, not a contamination-safe choice: LiveCodeBench release_v6 (Python, public and private tests, MIT code). Its problems end April 2025, so it cannot support the post-July-2025 requirement. Use it only as a non-contamination sanity baseline.

## 5. Checklist

| Item | Verdict | Tag |
|---|---|---|
| 1a. LCB latest release v6, end Apr 2025 | CONFIRMED (GitHub); DIFFERENT (HF card latest = v5) | read |
| 1b. Any release_v7 or later | NOT FOUND | read |
| 1c. Official post-July-2025 LCB problems | NOT FOUND | read |
| 1d. Issue tracker reviewed | CONFIRMED (#139, #145, #153 titles) | read / snippet |
| 2a. LCB-Pro split "quater_2025_7_9" and split names | NOT FOUND | snippet |
| 2b. Problems per post-July-2025 split | NOT FOUND | snippet |
| 2c. Difficulty labels | DIFFERENT (official: medium/hard only; Elo bands third-party) | read / snippet |
| 2d. Sources Codeforces/ICPC/IOI | CONFIRMED | read |
| 2e. Licence | DIFFERENT (HF card: apache-2.0; official dataset licence NOT FOUND) | read |
| 2f. Gated access | CONFIRMED (gated, terms via HF page) | read |
| 2g. Test cases (inputs + outputs) in dataset | NOT FOUND | read |
| 2h. Public judge vs external submission | DIFFERENT (local judge server, Docker) | read |
| 2i. Docker required | CONFIRMED (yes) | read |
| 2j. Special checkers | NOT FOUND | read |
| 2k. Reference solutions | NOT FOUND | read |
| 2l. Small-model (<=8B) scores | NOT FOUND | read |
| 2m. 584 problems | NOT FOUND in fetched text | snippet |
| 3. Six alternatives with post-cutoff problems and offline tests | DIFFERENT (none meets all criteria; 1 partial fit: CodeAssay, data access unknown) | read / snippet |
