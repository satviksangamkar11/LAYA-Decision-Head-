# P6 protocol — DRAFT v1 (for review; not frozen) — 2026-10-21

Status: DRAFT. Becomes `P6_PROTOCOL_v1.md` (frozen, hashed) only after user approval and after the H2-lite gate is decided. No generation, GPU run or download
is authorized by this draft. P5, V1.5 and arm B stay frozen and unchanged.

## 1. Question
On genuinely fresh tasks, at an identical per-task generation cap and with actual spending recorded: does V2-minimal (fresh sampling + trusted contract gate)
reduce false acceptances without losing solves relative to B, and does H2-lite (if it passed its development gate) solve more than V2-minimal at comparable
actual spend?

## 2. Arms (all use the same frozen model, Qwen2.5-Coder-1.5B-Instruct `2e1fd397…`, the same sampling settings and the same sandbox)
| Arm | Configuration | Status |
|---|---|---|
| B | P5 arm B, frozen (`tools/el_p2_generation_compare.run_arm`, fresh arm) | always |
| V2-minimal | `evidence_ladder/v2min.py` `v2min_v1.1`, policy `gate` (per-candidate trusted contract gate; `inconclusive` / `unavailable` recorded) | always |
| H2-lite | V2-minimal with the frozen H2-lite templates instead of `qwen_sample_v2`/`v2b` | ONLY if it passed the one-shot gate of `H2LITE_PLAN_v1.md` §6; otherwise P6 runs with B and V2-minimal, no redesign |
LAYA scorer OFF, critic OFF, no repair, no training. V1.5 (arm D of P5) is not re-run; its P5 result stands.

## 3. Fresh tasks (sourced only after this protocol is frozen)
1. Benchmark cohort: a LiveCodeBench release newer than v6, problems dated after 2025-04-06 (v6's last date), easy + medium, none of the 91 v6 problems
   used in P5. If no newer release exists, another public benchmark of the same format with problems dated after the model's release, chosen BEFORE looking at
   any of its problems. The download needs the user's approval (size stated first). Ids, order (seeded) and hashes locked before any candidate exists.
2. Contract cohort: 20 NEW tasks (bank format), statements + contracts authored and hashed before any candidate (as P5); a NEW blind private grading set
   authored from the statements only by a fresh-context author with no access to contracts, earlier sets or results, validated, hashed before inference, kept
   outside git and Kaggle. No P5 contract task, grading set or qualification fixture is reused.
3. N rule: the largest N (benchmark cohort) whose estimated GPU time is ≤ 10 h at the rate measured in a short smoke on tasks disjoint from the final set,
   minimum 40; 20 contract tasks.
   Rough estimate (unmeasured): B ≈ 200 s per LiveCodeBench task on T4 (P5: B median 149 s + shared round 1); V2-minimal reuses B's generations exactly
   where no contract exists (shared cache, as P5); H2-lite generates separately → ≈ 400 s per task with three arms ⇒ N ≈ 85 would need ≈ 9.4 h.

## 4. Matched compute (pre-registered)
- Identical per-task maximum: G = 6,000 generated tokens beyond the 4 shared round-1 samples (P5 value), for every arm.
- Recorded per task and arm: actual generated tokens (logical: every generation the arm used, shared ones included), actual inference seconds, sandbox /
  contract-check seconds, wall-equivalent seconds (P5 method), calls.
- Comparisons are made in exactly two pre-registered ways:
  1. PRIMARY: as run, at the identical cap.
  2. COST-MATCHED (all three arms are prefix-replayable sampling loops): for each pair of arms and each task, the arm with the larger actual spend is
     replayed from its saved calls in their real order, whole candidates only, under its own acceptance rule, cut at the other arm's actual spend for that task.
     Also reported: an anytime curve from the same replay at per-task budgets of round 1 + {1,500, 3,000, 6,000} tokens.
  No other spending level is used for any claim.
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
- H2-lite vs V2-minimal (if entered): "more solves" claimed only if the paired interval of the solve-rate difference excludes 0 at the COST-MATCHED
  comparison; false acceptances and actual tokens reported alongside.
- Infrastructure failures separate from candidate failures; 0 unresolved.
- Negative or inconclusive results are reported as such.

## 8. Execution (after freeze and a separate GO)
Kaggle T4, the qualified jail port and trusted verifier, shards one after another (one jail user), fresh smoke on disjoint tasks first, then the final run;
the analysis script is written and committed before any final result.

## 9. Open items for the user
- Which fresh benchmark source (§3.1) and approval of its download.
- Authoring the 20 new contract tasks and the new blind grading set (§3.2).
- Whether the H2-lite arm exists (decided by its gate).
