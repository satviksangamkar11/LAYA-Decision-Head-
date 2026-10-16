# qwen_sample_v2: reason (2026-10-16)

`qwen_sample_v1` is unchanged (its template hash stays pinned). The Stage 0 FC-3 run
(`results/raw/evidence-ladder-stage0-gpu-run1-20261016T135840Z/STAGE0_RESULTS.json`) used v1 on 5 public tasks and only 1 of 5 samples passed its public examples. The saved candidates
(`store_fc3/*.src`) show why: v1 contains the task statement and examples but not the solution format, so the model wrote stdin/stdout programs (and a Java class named ShiftText) instead of
`def solve(data)` / `public final class Solution { public static Object solve(Map<String,Object> data) }`. That is a prompt defect, not a model-quality measurement.

v2 adds one block, `Required solution format:`, filled from `SOLUTION_FORMATS[language]` in `evidence_ladder/models/qwen.py`. The text restates the task kits' own
`SOLUTION_FORMAT.md` rules (Python: `results/raw/codex-pilot1-20261016/fixed_v2/worker/SOLUTION_FORMAT.md`; Java: `results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker/SOLUTION_FORMAT.md`).
It contains no hidden-test information. Tasks in this pilot take one input dictionary and return a value, so there is one format per language.
