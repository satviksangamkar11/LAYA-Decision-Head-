# P6 protocol v1.1 — logging-only amendment (2026-10-21, before any P6 model generation)

Amends `P6_PROTOCOL_v1.md` and `P6_OUTPUT_SCHEMA_v1.md`. Neither frozen file is edited. Everything in v1 stays as written: arms, prompts, seeds, budget G, N rule,
grading, measures, criteria.

## Reason (written before any P6 generation)
P7 (execution-based selection) needs the complete generated candidate pool. Arm B stops at its first public pass, and the v1 runner records only candidates
an arm evaluated. In P5, 104 of 2,104 generated samples were never saved as candidates, so they cannot be audited (`results/raw/p5-headroom-audit-v1`).

## Change (logging only)
`tools/el_p6_run.py` writes two additional files. No decision, request, budget, gate verdict or token count changes.
- `CANDIDATES.jsonl`: one row per NEW generation, written at the generation-cache boundary at the time of generation. Row fields:
  - request identity: "task", "generated_by_arm", "seed", "purpose", "template", "prompt_sha256", "max_new_tokens";
  - result: "extract_status", "tokens_generated", "code_sha256", "code" (null when no code was extracted), "text" (raw model output).
- `CANDIDATE_USE.jsonl`: one row per task, giving which seeds each arm actually evaluated. Row fields:
  - "task";
  - "A": [1] if seed 1 had code, otherwise [];
  - "B" and "V2": [{"seed", "public_pass", "contract"}];
  - "generated_seeds".

  A generated seed found in no arm's list is `not_evaluated`. Any later public check of such a seed happens offline, after generation, and is
  never an arm's result.
- `RUN_CONFIG.json` gains "amendment": "P6_PROTOCOL_v1.1 (logging only)".

## Proof of unchanged behaviour
`tools/test_p6_logging_v11.py` runs the frozen v1 runner (commit 027c91ea) and the amended runner on the same scripted model outputs, in four scenarios:
Codeforces later-round acceptance, seed-1 acceptance with early stop, seed-1 extraction failure, and contract-gate rejection. It checks:
1. CALLS, RESULTS, ANSWERS and IDENTITY are identical (timing fields excluded);
2. CANDIDATES has exactly one row per non-reused call, with the same request identity;
3. after an early stop, the unevaluated seeds 2–4 are logged and marked not_evaluated.

Result: 9/9 pass. `tools/test_p6_stdin.py` still passes 6/6.

## Data rule
`CANDIDATES.jsonl` holds model outputs only (no grading data). Using it to design a P7 selector makes P6's tasks development data for P7. A P7 confirmation
claim needs a separate blind set.
