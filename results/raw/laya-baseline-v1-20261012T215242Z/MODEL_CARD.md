# LAYA research baseline v1 (frozen 2026-10-13)

**What it is.** A scorer that ranks candidate next actions of a coding agent. Pipeline: frozen Qwen3-4B-Thinking-2507 reads the coding situation and each
candidate → a small trained adapter (5.2 M parameters) maps those states into the pretrained LAYA `Coding_Decision_Agent` head (31 tensors, unchanged) → one
score per candidate. Three adapters (seeds 0, 1, 2) are frozen; each is scored alone. Hashes and paths: `MANIFEST.json` in this folder.

**Status.** Research baseline, not a proven product. Nothing here was retrained or copied; the files are referenced in place and their hashes are fixed.

**Measured (agreement with Fable's recorded action, not task success)**

| Test | Baseline | Linear probe (same Qwen states) | Chance |
|---|---|---|---|
| 53 VAL pools, unseen trajectories (development set) | 79.2% (67.9 to 89.3) | 94.3% | 29.3% |
| Same, coding context removed | 64.2% | 81.1% | |
| Options-only text detector | 67.9% | | |
| In-dataset 5-fold, 334 TRAIN pools | 80.2% | 86.2% | |

**What is not known**
- Whether it helps an agent pass tests (never measured).
- Whether it works with candidates from other agents or with another backbone.
- Why it should be used over the linear probe, which scored higher on every Fable test. The probe is the mandatory baseline in every evaluation.
- Part of every score can be authorship style: Fable's action is written by Fable, the alternatives by other models.

**Next evaluations (user, 2026-10-13):** mutation pilot on 20 TRAIN states (diagnostic; these states were trained on), then candidates from another agent on
the 23 fail-to-pass VAL decisions, labelled by sandbox test results. CAL / TEST stay protected.
