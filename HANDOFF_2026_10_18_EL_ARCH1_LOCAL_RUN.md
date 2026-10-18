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

## UPDATE 2026-10-19: WSLC backend, trusted verifier, 20 Java tasks (FROZEN STATE)
Branch `el-arch1-wslc-java20-2026-10-18` (base `el-arch1-local-run-2026-10-18`); commits 37cf87d3 (WSLC + verifier + runner), 78e5bd40 (runner order), b41f5129 (receipt labels). Not pushed, not merged.
- **MXC WSLC backend DONE and qualified on this PC:** `make_sandbox(scratch, protected, 'wslc')` (`evidence_ladder/controller/wslc_sandbox.py`, `sandbox_provider.py`, `tools/mxc_bridge/bridge.mjs`, `configs/wslc_image_v1.toml`); image built locally from Ubuntu-24.04 (no pull, tar sha256 5d7bc6d2...929ff, Python 3.13.11, JDK 26.0.1, Go 1.26.4); env_label/backend `mxc-wslc`; 8-field contract; infra failure raises; no fallback. Gates (`results/raw/wslc-gates-20261019/security_gates.json`): host files blocked, network blocked, timeout tree-kill, output cap, memory/CPU limits, secret env absent, protected files unchanged. **NOT enforced: process-count (pids) limit** (200 forks succeeded; fork containment = memory limit + timeout kill). Tests: phaseWSLC 10/10, phaseF2d 24/24 (incl. real WSLC), phaseF3d 15/15, Java parity vs the WSL engine 6/6.
- **Trusted verifier DONE** (`evidence_ladder/execution/trusted_verify.py`, engine `trusted_verify_v1`, flag `--trusted-verify`, default OFF, used in the 20-task run): one fresh process per input, candidate emits only raw output, verdict compared host-side, expected values never inside the sandbox; forged results/markers/checksums/stdout PASS never admitted. Limit: a hostile candidate can still emit any value for its input.
- **Runner:** `tools/el_bank_local_run.py --sandbox wslc [--trusted-verify] [--limit N]`; order = preflight (RAM >=3.0 / VRAM >=3.5 / commit >=4.5 GB, unchanged) -> model load -> WSLC engine + toolchain probe -> tasks. The WSLC VM costs ~1.7 GB RAM: run `wsl --shutdown` and close heavy apps (Codex/ChatGPT app, Ableton) before starting or preflight refuses.
- **Manifest** `configs/el_bank_java20_wslc_v1.json`: J1-17, J1-20, J1-23..J1-40 (10 medium, 10 hard), pinned sha256, disjoint from the Kaggle Java set. Learned scorer stays `none` (1.5B states cannot enter the frozen 4B LAYA head; user confirmed 2026-10-19).
- **Smoke (1 Java task J1-17, `results/raw/el-bank-java-smoke3-20261018T205908`):** real WSLC run, SUPPORTED (2 of 4 candidates pass 10/10 examples), critic called+applied, 833 s. Conformance said BROKEN only because receipts carried the factory name `wslc` instead of `mxc-wslc`; fixed in b41f5129 (not re-smoked; the full run is the check).
- **20-task run:** started 2026-10-18 21:16 UTC in `results/raw/el-bank-java20-wslc-20261018T211547`; result NOT in at freeze time. After it: `tools/el_bank_conformance.py <dir> --expect-engine mxc-wslc [--no-write]`, `tools/el_wslc_final_report.py`, then `WSLC_JAVA20_FINAL_REPORT.md`, commit receipts, push a new branch (user approved the push; no merge to main).
- **Limits to state in any report:** development results, public examples only, hidden grading not done; "hard" is a manifest label (the 1.5B model solved 6/10 hard Python tasks on public examples); LAYA scoring off; WSLC ~2.7 s per execution; one WSLC session at a time (concurrent users get HRESULT 0x80070020).
