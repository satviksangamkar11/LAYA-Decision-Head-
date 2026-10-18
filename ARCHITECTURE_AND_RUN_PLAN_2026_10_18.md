# EVIDENCE-LADDER ARCHITECTURE + THE NEW PIPELINE + WHAT MUST BE RUN (2026-10-18)

Source of truth for the architecture: `ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md` (invariants I1-I8, ladder L0-L5, tiers O1-O3, eight stages). This file lists what changed since, and exactly which of those changes the local bank run must use.
Status words: DONE (committed or in the tree, tests green) / IN PROGRESS (an agent is working) / DECISION (needs the user).

## 1. The pipeline (function-level bank tasks, ladder controller)
```
bank task (public prompt + public examples ONLY)
  1 INTAKE      requirements, caps, oracle tier
  2 INVESTIGATE
  3 GENERATE    local model, k=4 candidates, thinking policy recorded
  4 FILTER      deterministic: drop duplicates / unusable output (candidate_filter, frozen interface v1)
  5 EXECUTE     public examples in the SANDBOX  (new: MXC provider; Kaggle used the chroot jail)
  6 L2 + L1     model-written tests validated by mutants (L2) and behaviour clusters (L1)   [L2 parser fix IN PROGRESS]
  7 DISCRIMINATE two+ clusters -> blinded critic packet (GLM) -> diagnostic only
  8 RANK + REPAIR + VERIFY   trusted evidence first; a learned score never approves (I2); label SUPPORTED/VERIFIED/...
  every step: UTC timestamp (Timeline side channel, never in journal hashes), append-only hash-chained journal, replayable
```

## 2. Changes since the 2026-10-16 architecture, and whether the local run uses them
| # | Change | Where | State | Used by the local bank run |
|---|---|---|---|---|
| 1 | Candidate filter, frozen interface v1 (per-option rules, abstain codes, stable ids) | controller/candidate_filter.py, generation.py | DONE, committed | yes |
| 2 | Per-step wall-clock Timeline (side channel, hash-neutral) | controller/timeline.py | DONE, committed | yes (TIMELINE.jsonl + summary) |
| 3 | Enforced caps with named stop labels (step/round/token/wall/model-call) | controller/limits.py, agentic.py | DONE, committed | yes (per-task wall cap config) |
| 4 | **MXC sandbox provider** (Microsoft MXC processcontainer behind the unchanged `run(workspace, cmd, persist, timeout)` port; fail closed, no fallback; `LAYA_SANDBOX_BACKEND=mxc`) | controller/sandbox_provider.py, mxc_sandbox.py, win_job.py, mxc_dialect/ | DONE, committed on `mxc-execution-provider-2026-10-18` (verdict MXC_REPLACEMENT_PARTIAL; WSL stays default and rollback) | **yes, required by the user** |
| 5 | Bank-engine adapter over the sandbox port (python3 runner in MXC) | execution/ (new adapter) | IN PROGRESS (Agent 3) | yes |
| 6 | Kaggle chroot+seccomp jail (MXC's Linux backend cannot start on Kaggle: probe) | execution/kaggle_jail.py | DONE, used on Kaggle only | no (local) |
| 7 | Bank-40 runner, packer, driver, results compiler | evidence_ladder/bank40.py, kaggle/el_bank40_run_v1.py, tools/*bank40* | DONE, committed; ran 32/40 on Kaggle (4B) | reused (policy check, resume markers, output layout) |
| 8 | Local runner for the remaining 48 tasks | tools/el_bank_local_run.py | IN PROGRESS (Agent 3) | the entry point |
| 9 | **Qwen2.5-Coder-1.5B-Instruct** profile (non-thinking, fp16 resident, 6 GB GPU), loader v2, empty-think wrapper, `build_local_backend` | configs/backbones_v2.toml, models/qwen.py, backbone.py | DONE in tree (tests green); download in progress; smoke test pending (Agent 1) | yes |
| 10 | Thinking policies recorded in every result (`budget` / `prompt_brief` / `model_variant`) | bank40.check_policy | DONE | `model_variant` (no cutoff, ceiling 4096) |
| 11 | **L2 parse fix** (Kaggle run: L2 INSUFFICIENT on 32/32, 'no valid test parsed') | controller/l2.py | IN PROGRESS (Agent 2) | yes once fixed |
| 12 | **Internal (hidden) state capture + diagnostic probe** (1.5B states, depths 14/19/23; comparison-only, never approves; frozen 4B LAYA head refuses them) | models/state_capture.py, tools/el_bank_state_probe.py | IN PROGRESS (Agent 2) | yes (capture hook in the runner) |
| 13 | Critic: staged packets exported, no GLM call in the local run | critic_batch | DONE | packets listed in RUN_SUMMARY |
| 14 | Report generator + traceability T01-T50, two paths A/B | tools/el_arch1_final_report.py | DONE | afterwards |

## 3. What must be RUN, in order
1. Finish the model download, write MANIFEST_pinned.json, GPU smoke test (speed, VRAM, RAM). Agent 1.
2. Agent 3: bank-engine adapter on MXC + local runner + tests; Agent 2: L2 fix + state capture hook.
3. Integrate: all suites together (CPU), secret scan, commit on one branch.
4. Real run, local GPU, MXC sandbox: the Python tasks of `configs/el_bank80_rest_v1.json`; JAVA see decision below.
5. Post-run: `tools/el_bank40_driver.py compile` (tables with UTC per step), `tools/el_bank_state_probe.py` (internal-state result), findings note.

## 4. Decisions and constraints
- DECISION (Java): no JDK on the Windows host and MXC's container runs native Windows programs only. Options: install a portable Windows JDK and test MXC with it (recommended); WSL for Java as a labelled exception; or skip Java (NOT_TESTED_NO_JDK).
- Free RAM on the PC is about 1 GB while the model needs about 3 GB to load: close heavy apps before the real run. The backend loads shards straight to the GPU (`device_map="cuda"`).
- One GPU process at a time. Development results only (D1 waiver): public examples, no hidden grading; the 4B LAYA head is not used (comparison-only scorers, if any).
- Not changed: frozen checkpoints, datasets, adapters, prompts v1/v2, the evidence requirements, the journal format.
