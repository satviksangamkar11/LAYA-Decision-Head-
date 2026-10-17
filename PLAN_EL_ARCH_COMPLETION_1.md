# PLAN EL-ARCH-1: Evidence-Ladder Architecture Completion (opened 2026-10-17, nothing started)

Tag for every message and report: `[EL-ARCH-1 · Aim: complete the missing architecture blocks, not performance]`.
Authority: ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md (+ R1/R2/D5), results/raw/architecture-audit-20261016T211246Z/AUDIT.md, FREEZE_LAYA_INT_1.md (frozen; its receipts and the frozen LAYA/Qwen stay untouched).
Rules: no accuracy/improvement claims; no retraining; frozen thresholds stay frozen; every block gets real (not mock-only) receipts; Kaggle T4 for Qwen/capture, WSL sandbox, local GLM critic via the project key system (user-approved command in the user's terminal), Rs 0.

## Blocks to complete (from the audit; order = dependency order)
| # | Block | Audit rows | Done when |
|---|---|---|---|
| 1 | Reconcile budget accounting (glm_calls/tokens from the critic ledger into result.json) | F-new | result.json matches the ledger; test |
| 2 | Diagnose the first-attempt critic transport failures | F-new | root cause recorded, fixed or documented |
| 3 | Investigate stage in the real controller (read code/state, run existing tests, Qwen proposes, LAYA ranks) | 2, 20, 41 | one repo-style task runs investigate -> repair -> validate through the controller, real receipts |
| 4 | Oracle tiers O1/O2/O3 in the router | 9 | tier chosen by code per task; tests |
| 5 | Probe + random-head (correct control) as comparison arms; arm E wired into the stage-1 controller | 28, F07, F08 | arms A-E run on identical pools/budgets, receipts |
| 6 | LAYA monotonicity guard before ordering checks | 31 | guard + test |
| 7 | Irreversible-risk action classes (fixed table, not a deny regex) | 17, 22 | table + tests |
| 8 | Live-mode provenance hardening (fake capture_fn cannot pass) | 33, F10 | test with a fake capture refused by independent re-capture |
| 9 | Behaviour clustering beyond public examples; critic trigger rate | F14, 5 | clusters on experiment outputs; trigger logged |
| 10 | Evaluation gates in the report: G-0 headroom, equal-compute keep/drop, false-approval cap | 40 | gate verdicts in report.py |
| 11 | Seal or relabel hidden dev cases | 19, 38, F12 | documented policy or sealed set |
| 12 | Freeze committed code before every final run (commit hash in receipts) | F11 | each receipt records the commit |

Order of work: 1, 2 (small), 12 (process), then 3, 4, 5 (core), then 6-11. Performance evaluation (LAYA vs probe vs controls, task success, GLM benefit, the 80-task baseline) is a separate plan that starts only after blocks 3-5 pass.
