# Evidence-Ladder — Pilot 1 (development)

**Scope:** 40 original Python function tasks, 10 easy / 20 medium / 10 hard (subjective buckets), four public examples per task and 40 additional deterministic protected test vectors per task. These tasks are **not** a recent external dataset and cannot serve as an independent held-out final benchmark. Public problems and sample cases were authored together with the reference oracles, so true oracle independence remains unproven. This pack is intended for a cheap local/Kaggle development smoke test and integration check.

**What is and is not included**
- INCLUDED: worker prompts, public examples, question bank, reference oracles, grader test vectors, reference self-test, explicit-unsafe local development grader, paired score comparison tool, experiment protocol.
- NOT INCLUDED: an actual trained/running Qwen system, NVIDIA API/GLM client, LAYA adapter connection, a working Evidence-Ladder controller for D0/D1, strong OS sandbox, independent human validation, or any measured model success scores.
- Zero cost only if you run with your existing free quotas and check provider limits; this package makes no external API calls.

## Files
- `worker/QUESTIONS.md`: 40 questions and example inputs/outputs.
- `worker/tasks.jsonl`: machine-readable prompts and four public tests for each.
- `worker/SOLUTION_FORMAT.md`: how to produce submissions and distinguish arms.
- `grader/protected_cases.json`: 40 extra tests per task; must not be mounted in worker's environment.
- `grader/oracles.py`: reference implementations; grader-only.
- `grader/SELFTEST.py`: checks reference outputs and kills simple constant-null mutants.
- `grader/grade.py`: **UNSAFE DEVELOPMENT ONLY** runner; no real process isolation.
- `grader/compare.py`: paired estimates and exploratory 95% bootstrap intervals.
- `PROTOCOL.md`: evaluation rules and stage gate.

## Quick start (your PC, without model inference)
```sh
python grader/SELFTEST.py
```

For a run with submitted solutions (ONLY with code you trust), put `P1-01.py` ... `P1-40.py` under `submissions/A/`, then:

```sh
python grader/grade.py --solutions submissions/A --arm A --out reports/A.json --unsafe-local
```

On hosted Kaggle, do not put `grader/` files in the generation environment. A separate grader kernel is **not a security sandbox**: candidate code running in that kernel may read any files the kernel can access. For real protected grading, use a properly isolated execution service or a trusted benchmark evaluator that prevents candidate access to grading assets. The provided local grader explicitly requires `--unsafe-local` for this reason.

Compare only *real* arms:
```sh
python grader/compare.py reports/A.json reports/B.json reports/C.json reports/D0.json reports/D1.json --out reports/compare.json
```

The E arm is an OPTIONAL, non-decisive function-level transfer probe. The pretrained LAYA head's actual ACTION evaluation belongs to Stage 2. Use independent sealed benchmark tasks for final claims.

No model was run by creating this package.
