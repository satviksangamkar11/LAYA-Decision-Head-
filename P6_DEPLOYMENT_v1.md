# P6 deployment v1: two Kaggle kernels (2026-10-22, before the final run)

Deployment only. The protocol (`P6_PROTOCOL_v1.md` + the logging-only v1.1 amendment) is unchanged: same arms, prompts, seeds, budget G = 6,000, model
revision, N = 72 (frozen rule, `results/raw/p6-smoke-v1/SMOKE_TIMING.json`). Only the split of tasks across Kaggle sessions is decided here.

## Why
A single kernel needs up to 9.35 h by the frozen planning rates, close to Kaggle's 12 h session limit. Two kernels lower the risk of losing the whole run.
User (chat, 2026-10-22): at most 4 hours on the current account (`sangeetasangamkar`); the second account (`satviksangamkar54321`, the user's own) may be
used for the rest, to finish fast. The user was told that Kaggle's terms forbid one person running several accounts to get extra GPU, and that this risks
a ban; the decision is the user's.

## Split (recorded in `configs/p6_final_lock_v1.json` "deployment")
| Kernel | Account | Tasks | Planning estimate (426 s / cf task, 150 s / contract task) | Cap |
|---|---|---|---|---|
| final_k1 | sangeetasangamkar | Codeforces final_ids[0:32] | 13,632 s = 3.79 h + setup | 14,100 s whole session (stops, keeps outputs, records incomplete, no restart) |
| final_k2 | satviksangamkar54321 | Codeforces final_ids[32:72] + D01–D20 | 17,040 + 3,000 = 20,040 s = 5.57 h + setup | none (Kaggle's 12 h) |

Kernel 1 stops at 14,100 s, leaving margin for Kaggle's container start. Any task it leaves incomplete runs later in a third kernel on the second
account. That kernel uses the same bundle code and only the unfinished ids. A partial task's outputs are discarded and the task is rerun whole.

## Checks before launch (`tools/p6_make_lock.py`, fails closed)
- **Coverage:** the 92 task ids of the two kernels are distinct and equal the 72 final Codeforces ids plus D01–D20.
- **Isolation of the smoke tasks:** the smoke tasks (2033/A, 2039/A, 2044/A) are outside the final set.
- **Grading data:** the private blind set's sha256 equals its frozen value. It stays on this PC; neither bundle contains a grading test, the private set,
  the Codeforces test export or a key (`pack_el_p6_v1.py` asserts).
- **Identical code:** both bundles are packed from the same commit; the runner sha256, model revision and model manifest are recorded in the lock.

## After the run
1. Download both kernels' outputs to `results/raw/p6-final-v1/<kernel>/`.
2. Merge, then completeness check:
   - every one of the 92 tasks has rows for A, B, V2 and `_task`, and an answers row per arm;
   - each Codeforces task has an identity row;
   - CANDIDATES/CANDIDATE_USE rows are present.
3. Only then grade:
   - Codeforces on Kaggle CPU (`kaggle/el_p6_cf_grade_v1.py`);
   - contract tasks on the PC (`tools/el_p6_grade_contract.py`).
4. Then run the frozen analysis (`tools/el_p6_analyze.py`).
