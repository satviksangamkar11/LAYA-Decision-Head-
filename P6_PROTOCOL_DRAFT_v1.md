# P6 protocol — DRAFT v1 (for review; not frozen) — 2026-10-21

Status: DRAFT (updated 2026-10-21: fixed-budget comparisons, dataset criteria, role independence). Becomes `P6_PROTOCOL_v1.md` (frozen, hashed) only after user approval and after the H2-lite gate is decided. No generation, GPU run or download
is authorized by this draft. P5, V1.5 and arm B stay frozen and unchanged.

## 1. Question
On genuinely fresh tasks, at an identical per-task generation cap and with actual spending recorded: does V2-minimal (fresh sampling + trusted contract gate)
reduce false acceptances without losing solves relative to B, and does H2-lite (if it passed its development gate) solve more than V2-minimal at comparable
actual spend?

## 2. Arms (all use the same frozen model, Qwen2.5-Coder-1.5B-Instruct `2e1fd397…`, the same sampling settings and the same sandbox)
| Arm | Configuration | Status |
|---|---|---|
| B | P5 arm B, frozen (`tools/el_p2_generation_compare.run_arm`, fresh arm) | always |
| V2-minimal | `evidence_ladder/v2min.py` `v2min_v1.2`, policy `gate` (per-candidate trusted contract gate; `inconclusive` / `unavailable` recorded; unexpected verdicts fail closed) | always |
| H2-lite | V2-minimal with the frozen H2-lite `PromptAdapter` (version + sha256 locked) | ONLY if it passed the one-shot gate of `H2LITE_PLAN_v1.md` §6; otherwise P6 runs with B and V2-minimal, no redesign |
LAYA scorer OFF, critic OFF, no repair, no training. V1.5 (arm D of P5) is not re-run; its P5 result stands.

## 3. Fresh tasks (sourced only after this protocol is frozen)
1. Benchmark cohort: recent (2026) contest problems, easy + medium. Requirements (user, 2026-10-21):
   - reliable independent grading: official complete test data (or an equivalent independently produced test set), never model-written tests;
   - deduplication against LiveCodeBench v6 (all 175), every P5 task and the development bank (by source id, and by normalized statement text);
   - clear provenance per problem: contest, date, URL, license / terms of use;
   - initial downloads at most 250 MB in total; the source is chosen BEFORE looking at any problem's content, from the investigation report
     (`P6_DATA_SOURCES_2026_10_21.md`).
   Ids, order (seeded) and hashes are locked before any candidate exists.
2. Contract cohort: 20 NEW tasks (bank format), statements + contracts authored and hashed before any candidate (as P5); a NEW blind private grading set
   authored from the statements only by a fresh-context author with no access to contracts, earlier sets or results, validated, hashed before inference, kept
   outside git and Kaggle. No P5 contract task, grading set or qualification fixture is reused.
3. N rule: the largest N (benchmark cohort) whose estimated GPU time is ≤ 10 h at the rate measured in a short smoke on tasks disjoint from the final set,
   minimum 40; 20 contract tasks.
   Rough estimate (unmeasured): B ≈ 200 s per LiveCodeBench task on T4 (P5: B median 149 s + shared round 1); V2-minimal reuses B's generations exactly
   where no contract exists (shared cache, as P5); H2-lite generates separately → ≈ 400 s per task with three arms ⇒ N ≈ 85 would need ≈ 9.4 h.

## 4. Compute: fixed pre-registered budgets (user, 2026-10-21)
- Identical per-task maximum: G = 6,000 generated tokens beyond the 4 shared round-1 samples (P5 value), for every arm. A second fixed cap may be added
  ONLY before freezing, and only if the GPU budget allows (each cap is actually run; claims come only from caps that were run).
- Recorded per task and arm: actual generated tokens (logical: every generation the arm used, shared ones included), actual inference seconds, sandbox /
  contract-check seconds, wall-equivalent seconds (P5 method), calls.
- Claims are made ONLY at the pre-registered fixed cap(s), as run, with actual tokens and time reported next to every solve and false-acceptance number.
- Retrospective replays (truncating saved calls at other spending levels) are EXPLORATORY only and never support a claim. They are also inexact: the last
  call before a cap gets a smaller token limit (`min(4096, budget − used)`), so a lower-cap run is not exactly a truncated higher-cap run.
- A shared generation cache with exact reuse (P5 method): identical requests (prompt, seed, max tokens) return the recorded output; each arm is charged
  for every generation it uses.

## 5. Grading (offline, after every arm's selection is final)
Benchmark cohort: official public + private tests. Contract cohort: the new blind private set (primary; named and hash-checked, fail closed, no fallback).
SUPPORTED is a claim, never counted as correct.

## 6. Measures (task-level, paired)
Solved; false acceptance (accepted but grading FAILS); abstention; correct answers wrongly blocked (B correct, V2-minimal no answer); contract verdict mix
(pass / fail / inconclusive / unavailable); actual tokens; wall-equivalent time. Paired 95% intervals: bootstrap over tasks, 10,000 resamples, seed 1.

## 7. Success criteria (to be frozen with the protocol)
- V2-minimal vs B, contract cohort: false acceptances lower or equal; correct answers wrongly blocked = 0 observed; solves not lower.
- V2-minimal vs B, benchmark cohort: solves not lower; expected identical where no contract exists (reported, not claimed).
- H2-lite vs V2-minimal (if entered): "more solves" claimed only if the paired interval of the solve-rate difference excludes 0 at the fixed cap;
  false acceptances and actual tokens reported alongside.
- Infrastructure failures separate from candidate failures; 0 unresolved.
- Negative or inconclusive results are reported as such.

## 8. Execution (after freeze and a separate GO)
Kaggle T4, the qualified jail port and trusted verifier, shards one after another (one jail user), fresh smoke on disjoint tasks first, then the final run;
the analysis script is written and committed before any final result.

## 9. Role independence
The H2-lite editor (GLM) never generates, sees or inspects any P6 task, contract, grading case or result. The 20 P6 contract tasks (statements + contracts)
are written by the coordinator and hashed before any candidate. The P6 blind grading set is written from the statements only by a separate fresh-context author
with no access to contracts, the editor, earlier sets or results; it is validated, hashed before inference and kept outside git and Kaggle.

## 10. Open items for the user
- Which fresh benchmark source (§3.1) and approval of its download.
- Authoring the 20 new contract tasks and the new blind grading set (§3.2).
- Whether the H2-lite arm exists (decided by its gate).
