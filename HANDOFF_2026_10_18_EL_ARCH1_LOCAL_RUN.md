# HANDOFF 2026-10-18: Evidence-Ladder pipeline, Kaggle bank run, MXC sandbox, local run

Read first: `ARCHITECTURE_AND_RUN_PLAN_2026_10_18.md` (what is built), `FINDINGS_FROZEN_EL_ARCH1_2026_10_18.md` (frozen findings), `MXC_REPLACEMENT_REPORT.md` + `MXC_MIGRATION.md` (sandbox). Rules: CLAUDE.md (section "User rules 2026-10-18").

## Branches (nothing merged to main)
- `el-arch1-final-closure-2026-10-18`: PUSHED to GitHub (report 42/0/8, receipts, research).
- `mxc-execution-provider-2026-10-18`: local only (MXC provider, benchmark, probe).
- `el-arch1-integration-2026-10-18`: local only, current work (timeline, bank-40 runner, brief-thinking, local runner, 1.5B backend, conformance). Latest commits: 87d3b7a8 and the relative-path fix (uncommitted until noted).
- Push of any of these needs the user's explicit yes in the chat.

## What ran, where the evidence is
| Run | Evidence |
|---|---|
| Toy 3/3 VERIFIED (Kaggle 4B, thinking budget 512) | results/raw/el-arch1-kaggle4b-lowthink-20261017T213212Z, ARCHITECTURE_FINAL_REPORT_4B_LOWTHINK.md |
| Kaggle bank, 32/40 tasks | results/raw/el-bank40-pass1-out-20261018T1752, results/raw/el-bank40-results-20261018T1758/BANK40_RESULTS.md |
| MXC verification (provider tests, 75 variants on MXC, Kaggle probe) | results/raw/el-mxc-verify-variants-*, results/raw/mxc-kaggle-probe-20261018, MXC_REPLACEMENT_REPORT.md |
| 1.5B model smoke (22 tok/s, 2.9 GiB VRAM) | results/raw/qwen25coder15b-smoke-20261018T142341Z |
| Local run (28 Python tasks, MXC, Qwen2.5-Coder-1.5B): PIPELINE_COMPLETE, 28/28 PIPELINE_OK, SUPPORTED 21 / BLOCKED 7, critic applied 3/3 | results/raw/el-bank-local-20261018T2106/ (BANK_RESULTS.md, CONFORMANCE.md, RUN_SUMMARY.json, retry_critic_log.jsonl, tasks_archive/) |
| Backup of code, models, trained heads and adapters | C:\LAYA_BACKUP_2026-10-18 (BACKUP_README.txt) |

## Commands
- Selftest (real MXC, CPU, about 1 min): `PYTHONPATH=. .venv/Scripts/python.exe tools/el_bank_local_run.py "D:/local model/results/raw/el-bank-selftest-<UTC>" --selftest`
- Real local run: `PYTHONPATH=. .venv/Scripts/python.exe tools/el_bank_local_run.py "D:/local model/results/raw/el-bank-local-<UTC>"` (ABSOLUTE path; `--resume` continues a run; `--sandbox wsl` only as an explicit comparison). Needs free RAM >= 3.0 GB, VRAM >= 3.5 GB, commit >= 4.5 GB; close heavy apps first.
- Audit any run: `tools/el_bank_conformance.py <run dir> [--legacy]` (strict by default).
- Internal-state probe after a run with capture enabled: `tools/el_bank_state_probe.py <run dir>`.
- Kaggle bank run: pack `tools/pack_el_bank40_v1.py`, steps in `tools/el_bank40_driver.py plan <pack>`; critic notebook needs Internet ON and the Secret `NVIDIA_API_KEYS` attached (the J1-05 critic call there is still open).

## Open items
1. Java: no JDK inside MXC; decide WSL exception or leave NOT_TESTED. 2. Critic read timeout for large packets. 3. Hidden-test grading was never done (all results are development, public examples only). 4. The 8 Python bank tasks never started on Kaggle are covered by the local run; the 40 not selected earlier are not run. 5. Commit the relative-path fix (provider_engine.py, el_bank_local_run.py) and the doc updates, then push only if the user says so.
