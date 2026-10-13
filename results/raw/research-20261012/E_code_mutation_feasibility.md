# E. Deterministic code mutations as outcome-labelled negatives: feasibility (2026-10-12)

Scope: research and read-only counting. No tests, sandbox, WSL, installs or dataset downloads were run.
Local counts come from `count_sites.py` (scratch copy, output `site_counts_train.json` in the scratchpad).
Only rows with `split == "train"` were used (118 of 141). No `cal` folder was opened.

## Part 1. Web research

### 1a. Mutation tools by language

| Language | Candidate | Licence (source) | Scope and notes (source) |
|---|---|---|---|
| Python | cosmic-ray | MIT (https://github.com/sixty-north/cosmic-ray) | Python 3 only. Did not find whether it can be limited to one file or a line range; offline status not stated. |
| Python | mutmut | BSD-3-Clause (https://github.com/boxed/mutmut) | Python only. Needs `fork()`, so Unix or WSL only on Windows. Default mutates only called functions. Line filtering via coverage (`mutate_only_covered_lines`). No diff filter. |
| Python | MutPy | not checked | Status unclear: one 2024 study could not run it, an older blog calls it unmaintained (https://emerging-researchers.org/projects/413-3/ ; https://medium.com/@boxed/mutmut-a-python-mutation-testing-system-9b9639356c78). |
| JS/TS | StrykerJS | Apache-2.0 (https://github.com/stryker-mutator/stryker-js) | "Mutation testing for JavaScript and friends"; TypeScript via the `typescript-checker` plugin, which drops type-error mutants (https://stryker-mutator.io/blog/typescript-support/). `perTest` coverage analysis runs only tests covering a mutant (https://stryker-mutator.io/blog/typescript-coverage-analysis-support/). The official supported-languages page returned HTTP 404, so the claims rest on blog posts. |
| Go | gremlins | Apache-2.0 per search snippet (https://github.com/go-gremlins/gremlins) | Pre-1.0, CLI `unleash`, incremental PR mode, tuned for small modules, slow on large ones. |
| Go | go-mutesting (jonbaldie fork) | MIT per pkg.go.dev (https://pkg.go.dev/github.com/jonbaldie/go-mutesting/v2) | v2.6.13. Has `--git-diff-lines`, a diff-only filter, and a baseline file. This is the only Go option found that restricts to a diff. |
| Java | PIT | Apache-2.0 per secondary sources (https://pitest.org/quickstart/mutators/) | Mutates bytecode, not source (quoted on the official page). Poor fit for a recorded source diff. Default group: CONDITIONALS_BOUNDARY, INCREMENTS, INVERT_NEGS, MATH, NEGATE_CONDITIONALS, VOID_METHOD_CALLS; STRONGER and ALL add more. |
| PowerShell | PSMutant | MIT (https://www.powershellgallery.com/packages/PSMutant) | Pester 5 and PowerShell 7.2+. AST-based, mutates copies in a sandbox, has an equivalents config, parallel workers. Source repo: https://github.com/Fortigi/PSMutant. |
| Ruby | mutant | Commercial; free only for open-source projects, licence per repository (https://www.rubydoc.info/gems/mutant/0.9.0) | Older free forks exist but were not checked for licence. No open alternative surfaced in search. |
| Any language | universalmutator | not confirmed (GitHub and PyPI licence text not retrieved) | Regex rules (match + replacement) for many languages incl. Python, JS, Go, Rust. Optional `--comby` mode gives fewer invalid mutants. Rule-file syntax and offline behaviour not confirmed (https://github.com/agroce/universalmutator ; https://www.cefns.nau.edu/~adg326/icse18t.pdf ; https://pypi.org/project/universalmutator/ returned an error page). |

Key point for this project: every tool above assumes a repository plus a test command. Our inputs are a recorded new-file body (114 of 117 Write calls create a file that is absent in the workspace) plus a few Edit calls. Most options would need a shim. A small in-house mutator on the recorded text, one site per mutant, is likely simpler than adapting any of them.

### 1b. Research on mutants as training data and mutant properties

- SWE-smith (NeurIPS 2025; author list not retrieved): https://arxiv.org/abs/2504.21798 ; full text https://arxiv.org/html/2504.21798v2 ; poster https://neurips.cc/virtual/2025/poster/121828
  - Bug synthesis methods and counts (Table 1, total 50,137; shares computed by me): LM Modify 17,887 (35.7%), Procedural 15,641 (31.2%), Combine 10,092 (20.1%), LM Rewrite 4,173 (8.3%), PR Mirror 2,344 (4.7%).
  - Yield (share of candidates that break at least one test): Combine 96.9%, LM Modify 56.0%, Procedural 40.2%, LM Rewrite 35.0%, PR Mirror 33.8%. Filter: keep only patches that break at least one passing test; test runtime cap 2 minutes.
  - Ablation (Table 4, 1,000 instances per strategy, Qwen 2.5 Coder 7B, SWE-bench Verified resolve %): LM Modify 5.7, LM Rewrite 8.8, Procedural 8.6, PR Mirror 9.2. The paper says LM Modify shows a "steep drop-off"; the other three are inside error bars.
  - (i) Trivially detectable share: not reported. The prompt tells the LM to avoid syntax errors, but no count is given.
  - (ii) Equivalent-mutant rate: not reported. SWE-smith keeps any test-breaking patch, so it does not need equivalence labels.
  - (iii) Shortcut learning: not studied in the paper. The paper trains an agent, not a classifier on candidates.
- Tufano et al., "Learning How to Mutate Source Code from Bug-Fixes" (ICSME 2019): https://arxiv.org/pdf/1812.10772.pdf
  - Reports that over 98% of generated mutants are lexically and syntactically correct, and that learned mutants resemble real fixes in 9% to 45% of cases (search snippet; full text not read). Trivially broken share is therefore below 2% for learned mutants; this is not the same as for rule-based mutants.
  - DeepMutation tool paper: https://arxiv.org/abs/2002.04760
- Just et al., "Are mutants a valid substitute for real faults in software testing?" (FSE 2014): https://people.cs.umass.edu/~rjust/publ/JustJIEHF2014b-abstract.html
  - 357 real faults in 5 applications; mutant detection correlates with real-fault detection after controlling for coverage. Supports mutants as proxies, not as proof of realism.
- Equivalent mutants:
  - arXiv 2408.01760 (LLM-based equivalent-mutant detection, https://arxiv.org/pdf/2408.01760) cites earlier prevalence estimates of about 4% to 39% of mutants being equivalent in real development. It also cites a median of 2.97% across 7 projects from a more recent analysis (that source was not identified in the search output).
  - Code Defenders study: fewer than 10% of human-written mutants are equivalent (same search source).
  - Trivial Compiler Equivalence (Papadakis et al., ICSE 2015): https://discovery-pp.ucl.ac.uk/id/eprint/1499169/1/Jia_Trivial_Compiler_mutation-testing-papadakis-icse15.pdf . On large programs it removes more than 7% of mutants as equivalent and 21% as duplicates; a human check found it catches about 30% of equivalent mutants. So it is a cheap filter, not a complete one.
  - EMS (Kushigian et al., ISSTA 2024): https://homes.cs.washington.edu/~rjust/publ/equi_mutants_ems_issta_2024.pdf . Reports counts (8,776 equivalents in 325 s on 19 Java projects, vs TCE 2,124 in 2,938 h), not recall. No single detection rate across studies exists.
  - Papadakis et al., "Mutation testing advances: an analysis and survey", Advances in Computers 112 (2019), pp. 275–378. Listed as an unresolved problem in later papers; full text not retrieved, so no URL is given.
- Redundancy: Guimaraes et al., "Optimizing Mutation Testing by Discovering Dynamic Mutant Subsumption Relations" (ICST 2020): https://conf.researchr.org/details/icst-2020/icst-2020-papers/23/Optimizing-Mutation-Testing-by-Discovering-Dynamic-Mutant-Subsumption-Relations . MuJava-M generated 64.43% fewer mutants with 100% effectiveness on 20 of 32 targets. Mutant subsumption graphs are undecidable in general; practical use is dynamic (test-based).
- Learned mutants as bug injectors, realism: muBERT (Degiovanni and Papadakis, ICST workshop 2022): https://arxiv.org/pdf/2203.03289 . Detected 27 of 40 Defects4J faults vs PIT 26. Follow-up https://arxiv.org/pdf/2301.03543 (689 versions): muBERT found 47 faults PIT missed, PIT found 13 muBERT missed.
- Synthetic-to-real shift: "Challenging Bug Prediction and Repair Models with Synthetic Bugs" (BugFarm), https://arxiv.org/pdf/2310.02407 : recall falls when models trained on real bugs are tested on synthetic BugFarm bugs. "On Distribution Shift in Learning-based Bug Detectors", https://arxiv.org/pdf/2204.10049 : models trained mainly on synthetic bugs do not capture the real bug distribution.
- Shortcut learning in general (NLU background, not bug detection): https://arxiv.org/abs/2208.11857
- Tool comparison for Python: Guerino et al., SBQS 2024, https://sol.sbc.org.br/index.php/sbqs/article/view/32948 (CosmicRay, MutPy, MutMut, Mutatest; scores close, MutPy slightly ahead on dynamic tests). Pytation vs Cosmic Ray (arXiv 2601.19088, https://arxiv.org/pdf/2601.19088): 89% vs 67% mutation score, about 3.5% cross-killing. Pytest-gremlins comparison page (written by a competing tool author): https://pytest-gremlins.readthedocs.io/en/latest/guide/comparison/

What is NOT known from these sources: the trivially-detectable share for rule-based mutants (only learned mutants have a number, above); equivalent-mutant rates for operators like those we would use; whether models trained on mutant negatives learn shortcuts (no source found; the closest, BugFarm and 2204.10049, reports a train-test shift, not a shortcut test). Where a claim is unsourced it is labelled as such.

### 1c. Best tool per language (for this project's three languages)

- Python: cosmic-ray (MIT, actively maintained per secondary source) for operators; or the in-house AST mutator. mutmut needs WSL here and is function-scoped.
- TypeScript: StrykerJS with the typescript-checker plugin; its per-test coverage helps speed. Setup assumes a project directory.
- Go: go-mutesting (MIT) because of its diff filter; gremlins is the alternative, but it is pre-1.0.
- Other languages in the brief (Java, PowerShell, Ruby): PIT, PSMutant, mutant (commercial). Not needed for our data.

## Part 2. Local counts (TRAIN rows only, my own script)

Source: `results/raw/fable-fidelity-report-20261010/consistent_matches.jsonl` (141 rows; split train 118, val 23). Decisions: `results/raw/fable-replay-states-go-py-ts-strict-20261009T220928Z/<key>/decision.json`. All 118 states found.

Language and change shape:

| Language | Decisions | Write only | At least one Edit |
|---|---|---|---|
| py | 61 | 51 | 10 |
| ts | 53 | 44 | 9 |
| go | 4 | 3 | 1 |
| all | 118 | 98 | 20 |

- Recorded calls per decision: 1 call for 106, 2 for 6, 3 for 3, 4 for 3. No other tool names occur.
- Write calls per decision: median 1, 10th to 90th percentile 0 to 1, max 4. Edit calls: median 0, p90 1, max 2.
- Write targets: 114 did not exist in the workspace (new files); 3 overwrote an existing file. All 22 Edit targets existed.
- New text size (characters of Write content plus Edit new_string): all median 2,704 (p10 520, p90 6,785, min 78, max 13,807); py median 2,428 (293 to 6,855); ts median 2,787 (873 to 6,452); go median 5,385 (1,701 to 6,563, n=4).

Mutation sites in recorded new text (regex, comment lines removed, approximate):

| Set | n | median | p10 | p90 | max | share with >=5 sites |
|---|---|---|---|---|---|---|
| all | 118 | 40 | 2.7 | 99.1 | 301 | 103/118 = 87.3% |
| py | 61 | 30 | 2 | 96 | 222 | 52/61 = 85.2% |
| ts | 53 | 47 | 7.4 | 96 | 301 | 48/53 = 90.6% |
| go | 4 | 62.5 | 18.6 | 126.7 | 154 | 3/4 = 75% |

Per-type site medians over all 118: comparison 8 (p90 30), boolean 2.5 (p90 10), not/! 3 (p90 11), numeric literal 7 (p90 23), return 5.5 (p90 18), +/- binary 1 (p90 7), True/False/None/null/undefined 3 (p90 12).

Other facts that matter for the design:
- Per-test outcome ids: 69 of 118 rows (ts 53, py 16) have per-test ids; 49 (py 45, go 4) only have whole-suite pass or fail.
- For Write calls, the "site" count covers the whole file body, not the change. For 114 new-file writes the whole file is new, so mutants can land in code no test runs, which changes the meaning of a PASS.

Regex limits: numbers inside strings and data files are counted; `-`/`+` only when spaced; comments are removed only when on their own line or after ` //`. These are counts of candidate sites, not mutants.

## Part 3. Assessment

Estimate (unmeasured): the median decision has about 40 raw sites. With 1 to 3 operators per site, that gives roughly 40 to 120 raw mutants. After the filters below, I estimate 5 to 15 mutants per decision remain, and about 3 to 10 of those are verified worse (test-failing) alternatives. Across 118 TRAIN decisions that is roughly 350 to 1,200 negatives, before any cap. About 13% of decisions (15 of 118) have fewer than 5 sites and would yield almost nothing. The estimate depends on the test-failure rate, which is unmeasured.

Main risks and safeguards:

1. Trivially broken mutants teach a syntax or import detector.
   - Safeguard: keep only mutants that parse (ast.parse for py; the TS and Go parsers via their own tools) and import. Report the share that crash at import as a separate bucket. Cap that bucket (for example at 20% of negatives). Use text-span replacement, not an AST round-trip, so formatting does not differ from the original (an ast.unparse round-trip would make every mutant trivially distinguishable by style).
2. Equivalent mutants labelled as worse.
   - Safeguard: a mutant is "worse" only if it fails a test that the recorded action passes. Dedupe by normalized AST and bytecode (TCE-style) and drop no-op mutants. Mutants that pass all tests are "equivalent on tests" only when the mutated line is covered by a passing test; otherwise UNKNOWN. Never use "passes" as a proof of equivalence.
3. Tests too weak, so a broken mutant still passes.
   - Safeguard: measure line coverage of the recorded action on its tests. Mutants on uncovered lines are UNKNOWN, not equivalent. Use per-test ids where present (69 of 118 rows); for the 49 rows without them, use only the pass or fail outcome and report them separately.
4. Mutants that do not resemble real mistakes.
   - Safeguard: pick operators from the observed errors in real fixes (for example Tufano et al. style, or a small hand-written list). Report the operator mix and compare it with the operator mix of the recorded Fable changes. Keep the realism claim separate from the label claim.
5. Whole-file Writes: most sites lie outside the recorded change.
   - Safeguard: for new files, restrict mutants to a random sample of sites per decision (cap about 8) and stratify by operator and position. For Edit calls, mutate only inside new_string. Do not claim a mutant is "the agent's mistake" when it sits outside the agent's edit.
6. Options-only shortcut (the detector learns the mutation signature).
   - Safeguard: run an options-only probe (candidate text only, no state or outcome) and compare its accuracy with chance. Per CLAUDE.md, chance plus 10 points is a warning level only. Also run a probe that sees only the operator token. Group the split by decision_id, so mutants from one decision never sit on both sides of a fold.
7. Label kind mixing.
   - Safeguard: store these as "execution-verified worse, synthetic" with their own kind. They are not Fable imitation labels, and they must not be mixed into the imitation pools (CLAUDE.md, primary training route). Training on them is outside this task and needs the user's GO.
8. Python-only and Windows constraints.
   - mutmut is Unix or WSL only; cosmic-ray scope is not confirmed. Any test run belongs in the sandbox, not here.
9. Ruby, Java and PowerShell do not occur in these 118 rows and are not needed.

Open items: check the mutator licences for universalmutator and MutPy; confirm Stryker's per-file mutate option; the equivalence rate for our own operators is unmeasured and should be measured on a pilot before any scale-up.

## Sources (all URLs above; none invented)

- https://arxiv.org/abs/2504.21798 , https://arxiv.org/html/2504.21798v2 , https://neurips.cc/virtual/2025/poster/121828
- https://arxiv.org/pdf/1812.10772.pdf , https://arxiv.org/abs/2002.04760
- https://people.cs.umass.edu/~rjust/publ/JustJIEHF2014b-abstract.html
- https://arxiv.org/pdf/2408.01760
- https://discovery-pp.ucl.ac.uk/id/eprint/1499169/1/Jia_Trivial_Compiler_mutation-testing-papadakis-icse15.pdf
- https://homes.cs.washington.edu/~rjust/publ/equi_mutants_ems_issta_2024.pdf
- https://conf.researchr.org/details/icst-2020/icst-2020-papers/23/Optimizing-Mutation-Testing-by-Discovering-Dynamic-Mutant-Subsumption-Relations
- https://arxiv.org/pdf/2203.03289 , https://arxiv.org/pdf/2301.03543
- https://arxiv.org/pdf/2310.02407 , https://arxiv.org/pdf/2204.10049 , https://arxiv.org/abs/2208.11857
- https://sol.sbc.org.br/index.php/sbqs/article/view/32948 , https://arxiv.org/pdf/2601.19088
- https://pytest-gremlins.readthedocs.io/en/latest/guide/comparison/
- https://github.com/sixty-north/cosmic-ray , https://github.com/boxed/mutmut
- https://github.com/stryker-mutator/stryker-js , https://stryker-mutator.io/blog/typescript-support/ , https://stryker-mutator.io/blog/typescript-coverage-analysis-support/
- https://github.com/go-gremlins/gremlins , https://pkg.go.dev/github.com/jonbaldie/go-mutesting/v2
- https://pitest.org/quickstart/mutators/
- https://www.powershellgallery.com/packages/PSMutant , https://github.com/Fortigi/PSMutant
- https://www.rubydoc.info/gems/mutant/0.9.0
- https://github.com/agroce/universalmutator , https://www.cefns.nau.edu/~adg326/icse18t.pdf
