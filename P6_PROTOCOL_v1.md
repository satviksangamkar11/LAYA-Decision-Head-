# P6 protocol v1 (FROZEN before any P6 model generation) — 2026-10-21

Supersedes `P6_PROTOCOL_DRAFT_v1.md` (kept). Any change after freezing = v2 with a written reason, made before the final run, never after seeing final results.

## 1. Two questions, reported separately
| Cohort | Arms | Question |
|---|---|---|
| Codeforces, held-out Oct 2024 – Feb 2025 (up to 72 qualified problems) | A, B | How many additional independently correct solutions does multi-sampling produce over one sample, and at what token cost? |
| same | B, V2 | Diagnostic identity check only (no contract exists there; V2 must equal B exactly; any mismatch = implementation failure, never a result) |
| 20 contract tasks D01–D20 | A, B, V2 | Does the trusted contract gate prevent false successes without losing correct solutions? (A descriptive) |
H2-lite is excluded (failed its development admission, `H2LITE_RESULTS_2026_10_21.md`). The LAYA head is OFF. V1.5 / the repair ladder are not re-run.

## 2. Model, sampling, arms
Qwen2.5-Coder-1.5B-Instruct, revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, the frozen sampling settings, the same seeds for every arm.
- Shared round 1: `GEN.plan_samples` over seeds 1–4, generated once; one shared generation cache with exact reuse (identical prompt, seed, max tokens,
  purpose -> the recorded generation).
- **A**: the seed-1 round-1 sample as is. No search for a public-passing replacement; an extraction failure is no answer. Its public-example result is recorded
  separately. Its cost is its own sample's generated tokens.
- **B**: `evidence_ladder/v2min.py` `sample_loop` without a gate (proven call-for-call equal to P5 arm B, also with the LiveCodeBench template:
  `tools/test_p6_stdin.py` check 5); first candidate passing every public example; G = 6,000 generated tokens beyond round 1.
- **V2**: the same loop with the trusted contract gate, `v2min_v1.2` (reject a proven violation and continue; `inconclusive` / `unavailable` admit on public
  evidence and are recorded; unexpected verdicts fail closed).
- Prompts: Codeforces — EVERY request uses the unchanged LiveCodeBench generic stdin prompt (`lcb_generic_v1`; statement composed by `tools/p6_cf_tasks.py`
  from the public statement file `configs/p6_cf_public_v1.jsonl`; public tests = the statement's own examples). Contract tasks — the bank templates
  `qwen_sample_v2` / `v2b`. Proven by `tools/test_p6_stdin.py` (6/6).
- Runner: `tools/el_p6_run.py`; outputs as `P6_OUTPUT_SCHEMA_v1.md`.

## 3. Tasks
- Codeforces: the 72 problems qualified under `P6_CF_QUALIFICATION_RULES_v2` (`results/raw/p6-cf-validation-v1/QUALIFIED_rules_v2.json`), in
  sha256("P6-2026-10-21:" + id) order. The 3 problems with too few tests and the 2 with an unresolved runner failure stay excluded.
- Contract: D01–D20 (`configs/p6_contract_tasks_v1/`, frozen before any candidate).

## 4. N rule (conservative; set after the timing smoke, before the final run)
- Codeforces rate r_cf = 1.5 × max(timing-smoke mean wall seconds per Codeforces task over all arms, 284 s) — 284 s is the P5 measured LiveCodeBench per-task rate.
- Contract rate r_ct = 1.5 × max(timing-smoke mean wall seconds per development task, 100 s).
- N = the largest N <= 72 with N × r_cf + 20 × r_ct <= 10 h (minimum 40); the first N problems in the hash order. If all 72 fit, N = 72.
- The lock `configs/p6_final_lock_v1.json` (schema in `P6_OUTPUT_SCHEMA_v1.md`) records N, the ids, and every hash below; created before the final run.

## 5. Grading (offline, after every arm's selection is final)
- Codeforces: on Kaggle CPU (`kaggle/el_p6_cf_grade_v1.py`), against the validated tests exported by `el-p6-cf-tests-v1` (manifest hash in the lock),
  Python through LcbRunner stdin with the LiveCodeBench rule of 6 s per test, ALL tests run; a timeout is a failure (no extra time for anyone). These are
  generated tests validated by the dataset authors with several human solutions and by accepted C++ references in our jail — NOT complete official Codeforces
  tests. The 6 s limit is more lenient than Codeforces' typical 1–2 s; reported as such.
- Contract: on the PC (`tools/el_p6_grade_contract.py`), MXC trusted verifier, the private blind set `private/p6-eval-v1/cases_p6.json` (914 cases), sha256
  `d8c57f33fd16ddeb0f0b85a9613a2cefb1ac8d60cc2387110faaeb723e4fd780` checked against the lock (fail closed).
- Only answers move to grading; grading tests never enter inference, a prompt, the generation bundle, git or the GLM context.

## 6. Measures and analysis (`tools/el_p6_analyze.py`, committed before any P6 generation)
Per cohort and arm: solved (independently graded PASS), wrong answers, no answer, false acceptances (label SUPPORTED but graded FAIL; A's wrong answers are
reported separately and carry no claim), logical tokens, own tokens, wall seconds. Public acceptance and graded correctness are always reported separately;
logical tokens and physical GPU time are never mixed. Paired bootstrap 95% intervals over tasks (10,000 resamples, seed 1).
- Codeforces: B − A solve difference with its interval, B/A logical-token ratio; identity check (all tasks identical and all verdicts `unavailable`).
- Contract: V2 − B and B − A; task-level B→V2 transitions (wrong answer prevented, wrong answer repaired, correct lost, correct gained, unchanged); gate verdict mix.

## 7. Success criteria (pre-registered; literal)
- Codeforces: "multi-sampling solves more than one sample" only if the interval of B − A excludes 0 and the point is > 0; always reported with the token ratio.
  A vs B is deliberately NOT equal compute.
- Contract: V2 false acceptances <= B; correct answers lost = 0; V2 solves >= B. 20 tasks: task-level changes are reported; no broad accuracy claim from a small
  difference.
- Infrastructure: missing rows or grades are implementation/infrastructure failures, separate from candidate failures; 0 unresolved.
- Negative or inconclusive results are reported as such.

## 8. Execution
Kaggle T4, qualified jail port and trusted verifier, one jail user at a time, shards one after another. Timing smoke first (tasks outside the final set:
Codeforces 2033/A, 2039/A, 2044/A — generation mechanics only; development tasks P1-13, P1-21, P1-31 — contract-gate overhead), each step with its own GO.

## 9. Limitations stated in advance
Development-grade evidence on a 1.5B model; Codeforces problems are post-release for the model but not proven absent from its training data; the contract cohort
is small; the grading time limit is lenient; the LAYA head and any learned scorer are not evaluated.
