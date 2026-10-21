# P5 Step 2 — post-hoc diagnostics on the saved final run (2026-10-21)

Exploratory analysis of `results/raw/p5-final-v1` (no model generation; CPU execution on the PC's MXC sandbox). Hidden grades only DESCRIBE saved candidates;
nothing here feeds back into runtime selection, contracts or the reported P5 score, and none of it is a benchmark result. Script: `tools/el_p5_diagnostics.py`;
counts and task ids: `results/raw/p5-diagnostics-v1/DIAG.json`.

## A. Trusted contract checks applied to B (contract cohort, blind v3 grades)
| B's 18 final answers | Contract check passes | Contract check fails |
|---|---|---|
| correct (16) | **16** | **0** (no correct answer rejected) |
| wrong (2) | C05 (not caught) | **C13 (caught)** |

All 38 public-passing B candidates: correct 35 pass / 0 fail; wrong 2 pass / 1 fail. No tool failures.
**B + trusted check (reject = no answer) → 16 solved, 1 false SUPPORTED, versus D's 13 solved, 1 false SUPPORTED.** C05's wrong answer (shared by all four
arms) satisfies its contract: the contract is incomplete there, as expected of a rejecting-only check.

## B. Ladder abstention audit (LiveCodeBench, 58 tasks where C and D gave no answer; D = C on every task)
- Stop cause: `budget` 57 (the round limit, max_rounds = 2, reached far below the token cap), `same_failure_twice` 1.
- **On all 58, the ladder never produced a candidate that passed the public tests.** Selection failures: 0. Generation failures: 58.
- On 11 of those 58, B's saved pool contains a candidate that passes hidden grading (B solved them): abc394_d, 3811, 3817, 3809, 3704, abc399_b, abc392_b,
  abc391_b, abc394_c, abc396_b, abc388_b.
So the ladder's loss is in candidate SEARCH within its round limit, not in choosing among candidates.

## C. Equal-spend replay of B (exploratory counterfactual)
B's saved calls in their real order (shared round 1 included), whole candidates only, cut at the tokens C actually reported per task; B's own answer rule.
| Cohort | C (actual) | B at C's spend | B (actual) |
|---|---|---|---|
| LiveCodeBench: solved / answered / false SUPPORTED | 18 / 27 / 9 | **18** / 29 / 11 | 27 / 47 / 20 |
| tokens | 110,716 | 84,965 | 327,255 |
| Contract (blind): solved / answered / false SUPPORTED | 13 / 15 / 2 | **15** / 17 / 2 | 16 / 18 / 2 |
| tokens | 29,347 | 11,042 | 33,691 |

**On LiveCodeBench, at a per-task ceiling derived from C's actual expenditure, B and C each solved 18 problems, and B spent FEWER tokens (84,965 vs
110,716)** because the replay cuts at whole candidates and B stops at its first public pass; B then makes slightly more false SUPPORTED claims (11 vs 9).
(Correction, user 2026-10-21: this is not an equal-expenditure comparison, so it neither proves equal-budget superiority for either strategy nor shows that
spending alone caused B's original nine-problem advantage.) The ladder stopped with most of its token cap unused (round limit). On the contract cohort, B under
the same per-task ceiling solves 2 more with fewer tokens (small n: 20 tasks).

## D. Disagreement feasibility (distinguishing inputs)
| Cohort | Tasks with ≥ 2 distinct public-passing candidates | With a disagreement on derived inputs | Authoritatively resolved | Disagreements that contain a correct class | Output-voting (plurality) picks a correct class |
|---|---|---|---|---|---|
| LiveCodeBench | 16 | 7 | **0** (no contracts) | 4 | 3 (exploratory, not authoritative) |
| Contract | 10 | **0** | 0 | 0 | 0 |

On the contract cohort the public-passing candidates never disagree on contract-valid derived inputs: distinguishing inputs add nothing there; the useful signal
is the contract check itself (section A). On LiveCodeBench, disagreements are rare (7 of 85 tasks) and cannot be resolved by any trusted evidence available
at runtime; stdin derived inputs are of unknown validity. A selector that abstains on unresolved disagreement would cost up to 4 correct answers there.

## What this says for the decision rule
| Condition (user, Step 2 GO) | Finding |
|---|---|
| B's correct answers survive trusted checks | **yes**: 16/16 answers and 35/35 public-passing correct candidates pass |
| B's incorrect answers are detectably rejected | **partly**: 1 of 2 (C13 caught, C05 not) |
| Disagreement testing offers useful additional evidence | **no, on this data**: 0 disagreements on the contract cohort; 7 unresolvable ones on LiveCodeBench |

Readings (exploratory, from data the final evaluation already used):
1. "B's search + D's trusted check" dominates D here (16 vs 13 solved, same 1 false SUPPORTED) and loses no correct answer.
2. The repair ladder's deficit is generation within a 2-round limit, not selection; under C's per-task ceiling, fresh sampling ties it on LiveCodeBench
   (with fewer tokens spent). P5 therefore does not show that fresh sampling beats repair per token, nor the reverse; it shows the ladder stopped early.
   Contract checks cost CPU execution (no extra model generation in this retrospective test); they are not free.
3. The distinguishing-input selector of the V2 document is not supported by this data; it should not be built first.
Any design chosen from these readings needs a fresh evaluation set for an unbiased claim.
