# Java Pilot 1 — Evidence-Ladder development kit

A **new set of 40 Java 21 coding tasks** distinct from the previously created Python Pilot 1, split 10 easy / 20 medium / 10 hard; 4 public examples and 40 additional development grader cases each = **1,760 cases**. Difficulty designations are uncalibrated estimates, and task content was authored for this pilot, not extracted from a post-2025 benchmark.

### Files
- `worker/QUESTIONS.md`: all 40 prompts with Java JSON signature and public examples.
- `worker/tasks.jsonl`: JSONL per-task descriptions and public tests.
- `worker/SOLUTION_FORMAT.md`: Java class/method contract.
- `worker/support/{Json,Runner}.java`: lightweight JSON runner; only Java standard library.
- `grader/protected_cases.json`: additional expected-value vectors; development only.
- `grader/ReferenceSolutions.java`: Java implementations, one per task; grader only.
- `grader/SELFTEST.py`: recompiles references, recomputes all expectations and runs trivial mutant checks.
- `grader/grade.py`: **UNSAFE DEVELOPMENT-ONLY** local Java grader.
- `grader/compare.py`: basic paired comparative report.
- `PROTOCOL.md`: A/B/C/D0/D1/E plan, security and evaluation limitations.

### Validate dataset without AI calls
```bash
python3 grader/SELFTEST.py
```

### Grade already trusted Java submissions locally (unsafe)
```bash
python3 grader/grade.py --solutions submissions/A --arm A --out reports/A.json --unsafe-local
```

For each task place code under `submissions/A/J1-01.java` etc, defining `public final class Solution` and `public static Object solve(Map<String,Object> data)`. The development grader requires `java` and `javac` version 21. For **untrusted** generated code, run in a properly isolated environment instead; the provided grader explicitly does not secure grading secrets or protect the host.

### Compare actual arm reports
```bash
python3 grader/compare.py reports/A.json reports/B.json reports/C.json reports/D0.json reports/D1.json --out reports/compare.json
```

**Status:** all reference programs generated and compilable; all 1,760 outputs have also been cross-checked against independently coded Python implementations, and the grader was smoke-tested with correct/incorrect Java submissions. No weak model, GLM API, Kaggle or LAYA inference has been run. This is a prototype test bank, **not independent sealed evaluation**.
