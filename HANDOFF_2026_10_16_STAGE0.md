# Stage 0 handoff, 2026-10-16

Branch `evidence-ladder-runtime-step2`. Checkpoint commits: `6498df5` (first checkpoint), `f0ffe01c` (Stage 0 work, local, not pushed).
Pilot GO: `results/raw/evidence-ladder-decisions-20261016T113958Z/DECISIONS.json` (sha256 0d30ecc5...). Spend Rs 0.

## Gates (verified from saved outputs)

| gate | status | evidence (sha256) | limits / open |
|---|---|---|---|
| FC-1 Kaggle jail | **PASS on its tests, design weaker than approved** | `results/raw/evidence-ladder-stage0-cpu-run6-20261016T130550Z/STAGE0_RESULTS.json` (69f84d31...), `redteam_v2_kaggle.json` (eb3dbe6e...) | jail = chroot + uid 65534 + seccomp + rlimits (Kaggle's own seccomp blocks unshare, so no pid/net/mount namespaces; no bwrap, apt offline). Self-check PASS; red-team 7/7 executed and blocked; 40/40 Python references, 37/37 wrong caught, 40 Java references; execution suite 20/21 in the jail (the bwrap-namespace receipt test fails by design). Open gaps: AF_UNIX allowed, x32 ABI not filtered, shared /tmp: hardening in progress |
| FC-1 WSL fallback | PASS | `results/raw/evidence-ladder-wsl-regression-20261016T140403Z/wsl_execution_suite_stdout.txt` (05cfb224...) | bubblewrap namespaces; 21/21 after the agent's engine/config changes |
| FC-2 GLM | PASS | `results/raw/evidence-ladder-critic-v1-run8-20261016/FC2.json` (a6cda193...), `KEY_CHECK.json` (59ee3f74...) | `reasoning_effort` max and low accepted by z-ai/glm-5.3-flash; 108-134 s per tiny call; 14 keys from Kaggle Secrets; not repeated |
| FC-3 Qwen speed | MEASURED | `results/raw/evidence-ladder-stage0-gpu-run1-20261016T135840Z/STAGE0_RESULTS.json` (82ca0235...) | T4 fp16, HF generate: single stream 14.6 -> 7.3 tok/s (mean 8.24), batch of 4: 42.7 tok/s; 1,933-7,968 tokens per sample, none truncated. Public-example pass 1/5 caused by a prompt defect (template v1 lacks the solution format). Weights: public dataset leminhhung0101/qwen3-4b-thinking-2507, shard sha256 match revision 768f209d. Proposed T in `configs/thresholds_evidence_ladder_v1_T.toml` (not frozen): 8,192 per sample, 32,768 per problem. Estimate (unmeasured): about 51 GPU-hours for 40 problems x 6 arms with HF generate |
| FC-4 hidden tests | PASS | cpu-run6 `FC4_separation_in_kaggle` | gate open raises; selection stdin public inputs only; jail fs scan for hidden bytes found nothing; grading stdin has no expected values; reference 47/47 after close. Flipped-grader check covered by the Kaggle replay suite (26/26), not re-run in this kernel |
| Critic isolation | PASS (current version) | `results/raw/evidence-ladder-critic-isolation-check-20261016T124143Z/CRITIC_ISOLATION.json` | no exec primitives; live metadata has no dataset, kernel or model sources. Re-check when the pilot version receives packets |

## Kaggle objects
- `evidence-ladder-stage0-run-v1` (CPU, v6 is the evidence run), `evidence-ladder-stage0-run-v1-gpu` (T4, v1; only its FC-3 phase counts), `evidence-ladder-stage0-probe-v1` (v1), dataset `evidence-ladder-stage0-bundle-v1` (3.2 MB, no secrets).
- `evidence-ladder-critic-v1` (v8 evidence run; internet on, 14 secrets attached by the user).

## Code added or changed (in f0ffe01c)
`evidence_ladder/execution/kaggle_jail.py` (new), `engine.py` (EL_JAIL=kaggle selects it; WSL path unchanged), `config.py` (EL_KJAIL_JDK), `evidence_ladder/models/qwen.py` (pad_token_id; real HF path ran for the first time), `kaggle/evidence_ladder_stage0_v1.py`, `kaggle/el_stage0_probe_push/`, `tools/pack_evidence_ladder_stage0_v1.py`, `configs/thresholds_evidence_ladder_v1_T.toml`. The frozen `configs/thresholds_evidence_ladder_v1.toml` still matches its sha256 (8b5e17cd...).

## Commands
- Stage 0 bundle: `.venv/Scripts/python.exe tools/pack_evidence_ladder_stage0_v1.py`, then `timeout 300 .venv/Scripts/kaggle.exe datasets version -p <bundle>/dataset -m <msg> --dir-mode zip`, `timeout 120 ... kernels push -p <bundle>/kernel`, poll `kernels status` under `timeout 60`, `kernels output ... -p results/raw/<new dir>`.
- WSL regression: `MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -u root -e bash -c 'export PATH=/opt/toolchains/bin:$PATH; cd "/mnt/d/local model" && python3 evidence_ladder/execution/tests/run_tests.py'`.
- Critic: `cd kaggle/evidence_ladder_critic_v1 && kaggle kernels push -p .` (secrets stay attached in the Kaggle editor).

## Blockers (Stage 0 is BLOCKED)
1. Kaggle jail hardening (x32 filter, AF_UNIX deny, per-run /tmp, honest Kaggle receipt test): running. Then the user decides whether the layered Kaggle jail is acceptable, or candidate execution uses the WSL bubblewrap fallback.
2. Generation prompt defect: new template `qwen_sample_v2` and an FC-3 re-measure: running.
3. T, k1 and the generation backend (HF generate at ~8 tok/s makes the full pilot about 51 GPU-hours, estimate unmeasured).
No coding comparison arm starts before every mandatory gate passes in the chosen execution configuration.
