# Correction to commit 633ad39d message: unsaved samples (2026-10-21)

The commit message says "unsaved samples only on solved tasks". That is wrong. The counts from both AUDIT.json files:

| Cohort | Tasks B did not solve: unsaved generations (code extracted) | Tasks B solved: unsaved (code extracted) |
|---|---|---|
| LiveCodeBench | 81 (41) | 8 (8) |
| Contract | 15 (13) | 0 |

Why this does not change the classification of the unsolved tasks:
- On a task B did not solve, nothing was ever accepted, so `v2min`/P5 `consider()` ran on every generation.
- A generation is left unsaved only if no code was extracted, or if its code is identical to an already-saved candidate (same artifact id).
- So the unsaved generations with code on unsolved tasks are exact duplicates of graded candidates.

Only the 8 LiveCodeBench samples on SOLVED tasks are true early-stop leftovers whose code was not kept. The earlier statement that the 104 unsaved P5
samples were lost "because of the early stop" is also corrected: most were extraction failures or duplicates.
