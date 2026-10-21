# H2-lite feasibility plan v1 (DRAFT for review) — 2026-10-21

Status: DRAFT (updated 2026-10-21: adapter implemented, roles separated). No GLM call, no model generation, no GPU run and no download is authorized by this document. It freezes the experimental contract for review.

## 1. Question
Can better offline instructions make the UNCHANGED Qwen2.5-Coder-1.5B-Instruct (revision `2e1fd397…`) produce more independently correct programs for the same
actual generation cost? Idea borrowed from Harness² (arXiv 2610.14224): a strong model edits the harness around a frozen model. Here the edit surface is limited to
prompt text, and selection uses runtime-visible signals only. This is a feasibility study on development tasks; it makes no claim. If H2-lite passes its gate it
enters P6 as a conditional arm; if not, it is discarded and V2-minimal remains the only successor.

## 2. Data: the development bank and its family split (fixed here, before any run)
The 40 Python development tasks `P1-01`…`P1-40` (`results/raw/codex-pilot1-20261016/fixed_v2/worker/tasks.jsonl`, sha256 `dd5d945c…`; 10 easy, 20 medium,
10 hard). They carry no family label; families are assigned by algorithm type below and the split is by family (no family on both sides).
These tasks are development material (their graders are public in git); nothing measured on them is a claim.

| Family | Tasks | Split |
|---|---|---|
| Arrays / numeric | P1-02, 04, 06, 07, 09, 12, 29, 30, 35, 38, 39 | SEARCH |
| Parsing / patterns / stacks | P1-11, 21, 23, 26, 34 | SEARCH |
| Grids | P1-16, 28, 37 | SEARCH |
| Hashing / data structures / binary search | P1-14, 20, 25 | SEARCH |
| Strings | P1-01, 03, 05, 08, 10, 13, 19, 24 | CONFIRMATION |
| Graphs | P1-17, 27, 36, 40 | CONFIRMATION |
| Dynamic programming | P1-18, 31, 32, 33 | CONFIRMATION |
| Intervals | P1-15, 22 | CONFIRMATION |

SEARCH: 22 tasks (5 easy, 12 medium, 5 hard). CONFIRMATION: 18 tasks (5 easy, 8 medium, 5 hard).
P1-13, P1-21 and P1-31 appeared in the P5 smoke runs; they stay where their family puts them (development tasks, already exposed).

## 3. What may be edited (the only harness surface)
- The text of the two frozen sampling templates `qwen_sample_v2` (`evidence_ladder/models/qwen.py`) and `qwen_sample_v2b`
  (`evidence_ladder/models/qwen_v2b.py`), supplied as a versioned `PromptAdapter` (`evidence_ladder/v2min.py`, v2min_v1.2): template slot ids become
  `<frozen id>@<adapter version>`, the adapter carries a sha256 of its texts; the frozen templates are never modified.
- Implemented and mechanically proven (tools/test_v2min.py, checks 11-13): an identity adapter reproduces arm B call for call; an edited adapter keeps every
  seed, position, template slot and token limit and changes only the prompt text; rule violations are refused at construction.
- Allowed: task-independent instruction text (how to read the input object, edge cases to consider in general, how to structure the answer).
  Each edit at most +400 characters over the template it replaces; the placeholders `{task_prompt}`, `{examples}`, `{solution_format}` stay and the required
  solution-format block (`SOLUTION_FORMATS["python"]`) is unchanged.
- NOT allowed: task-specific facts or hints, example solutions, any text derived from a hidden grade, a skills system, memory or an experience library,
  sampling parameters, budget, number of samples, seeds, extraction, public-check rules, tools, critics, repair.

## 4. The editor (offline only) and role independence
GLM on the NVIDIA endpoint already used by this project, called offline under the existing logging and key rules; nothing from the editor runs at
inference. Before any call: check the current model id, quota and Rs 0 status (not verified yet). The editor sees ONLY: the current template text, the
search-split statements and public examples, and the runtime-visible outcome table of each variant (section 5). It never sees a hidden grade, a confirmation
task, or the P5/P6 tasks. Per editor round it proposes exactly two variants: one general (how to approach any task) and one coding-specific (input handling,
edge cases), the Harness² invariant / domain pair.
Role independence (user, 2026-10-21): the editor (GLM) never generates, sees or inspects any P6 task, P6 contract, P6 grading case or P6 result; the P6
contract tasks are written by the coordinator, and the P6 blind grading set by a separate fresh-context author with no access to the editor, contracts or results.

## 5. Search procedure (search split only)
- Policy for every variant: arm B (`v2min.sample_loop` with no gate = P5 arm B, call-for-call) with the variant's `PromptAdapter` (none for the baseline), budget G = 6,000 generated tokens beyond the 4 round-1
  samples, the same seeds for every variant (paired), Kaggle T4, sequential, the trusted verifier on the Kaggle jail.
- Runtime-visible score per variant, over the 22 search tasks (no hidden grade is computed during the search):
  `accepted` (tasks with a candidate passing every public example), `contract_violations_among_accepted` (decl_v1 parser contracts, the runtime gate),
  `extraction_failures`, `actual_generated_tokens`.
- Selection rule (fixed): maximize `accepted − 2 × contract_violations_among_accepted`; ties → fewer actual tokens. A variant using more than 1.25 × the
  baseline's actual tokens is not eligible.
- Rounds and stopping rule: at most 3 editor rounds (at most 6 variants + baseline). Stop early when a round's best variant does not exceed the current best
  score by at least 2. The search output is ONE frozen variant (or none, if no variant beats the baseline score).

Public-test acceptance is a diagnostic and search signal, not evidence of correctness (P5: 20 of B's 47 SUPPORTED claims on LiveCodeBench were wrong).

## 6. One-shot confirmation gate (confirmation split, written before any run)
The frozen variant and unchanged B run ONCE on the 18 confirmation tasks, paired seeds, same policy and budget. Graded by the development grader
(`ProtectedGrader`, development evidence only). No prompt change after this step, whatever the result.

H2-lite PASSES only if all hold:
1. solved(H2) − solved(B) ≥ +2 (a meaningful improvement on 18 tasks; no significance claim);
2. false acceptances (accepted but grader FAIL): H2 ≤ B + 1;
3. contract violations among accepted answers: H2 ≤ B;
4. actual generated tokens: H2 ≤ 1.25 × B;
5. no infrastructure failure left unresolved.
Otherwise H2-lite is discarded and P6 runs without it. Pass or fail, the confirmation result is recorded unchanged.

## 7. Cost estimate (from recorded timings; unmeasured for this design)
Recorded on Kaggle T4, bank-format tasks: shared round 1 ≈ 30–98 s per task (P5 smoke bank; P5 final contract shard mean 42 s); B's own sampling mean 37 s,
median 6.4 s (P5 final contract shard). Taken as ≈ 80 s per task per variant.
- Search: 22 tasks × (baseline + ≤ 6 variants) × 80 s ≈ 3.4 GPU h (worst case; early stop reduces it).
- Confirmation: 18 tasks × 2 × 80 s ≈ 0.8 GPU h.
- Total ≈ 4.2 T4 GPU h in at most 4 sequential Kaggle sessions; editor: ≤ 6 GLM calls plus retries. Estimate (unmeasured).

## 8. Isolation
Hidden graders are used only in section 6, after the variant is frozen. P5 tasks, P5 grading sets, P6 tasks and the contract cohorts never enter the search.
All outputs are create-only under `results/raw/h2lite-*`.
