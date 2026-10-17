# HANDOFF 2026-10-17: LOCAL full pipeline, small model (Qwen3-0.6B), 3 examples

Tag: `[EL-ARCH-1 · FULL ARCHITECTURE CLOSURE: LOCAL RUN]`. Read after `HANDOFF_2026_10_17_LAYA_INT_1.md` and `FREEZE_LAYA_INT_1.md`. Source commit of the run: `437df8e5` (branch `evidence-ladder-runtime-step2`, local; not pushed).
Run folder: `results/raw/el-arch1-local-20261017T160410Z/` (final receipts in `receipts/`, `FINAL_SHA256SUMS.txt` 10 files, all OK; `ACCEPTANCE_SUMMARY.json` sha256 c3975bde...; `LOCAL_REPORT.md` 1ad736b7...; parity `parity/PARITY_RESULT.json` e0082327...).

## 1. What was asked and decided (user, chat 2026-10-17)
- "Qwen3-0.6B download, then run the full pipeline locally, use the local GPU", "only 3 examples" (ex1_slug, ex2_clamp, ex3_wordcount).
- "Do not rebuild the pipeline, just extend the connection for the second model as well. Later we make this universal."
- Decision record v7: `results/raw/evidence-ladder-decisions-20261017T155133Z/DECISIONS.json`. Earlier relevant: v4 (local NVIDIA critic calls, key system read by the script in the user's terminal), v6 (Kaggle T4 for the reference run). Spend Rs 0.
- Download (user said yes to the >1 GB): `Qwen/Qwen3-0.6B` rev `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0, 1.52 GB, `models/qwen3-0.6b/` (LFS sha256 of model.safetensors and tokenizer.json verified; `MANIFEST_pinned.json`). Hidden 1024, 28 layers.

## 2. The one hard limit
The 3 frozen adapters + pretrained LAYA head read **Qwen3-4B** states (hidden 2560, depths 18/24/30). Qwen3-0.6B states (hidden 1024, 28 layers) cannot enter them; the provenance guard refuses them. So this run uses Plan B:
1. Qwen3-0.6B proposes the actions locally (RTX 3050).
2. Its hidden states are exposed with the same recipe positions and stored as DIAGNOSTIC data (for a future 0.6B adapter; training one needs a separate GO).
3. LAYA, probe and random-head score the same options from **4B states captured locally** (per-layer streaming on the 3050), allowed only because a pre-registered device parity gate passed.

## 3. How it is built (extension, no rebuild)
- ONE new concept: the **backbone profile**: `configs/backbones_v1.toml` + `evidence_ladder/models/backbone.py` (id, repo, pinned revision, local path, hidden size, layers, capture depths, dtype, device policy stream|resident, device classes, recipe id, cue, option prefix, `head_compatible`, generation settings, enable_thinking; its sha256). 4B = depths 18/24/30 as trained (also embedded and cross-checked); 0.6B = depths 14/19/23, `head_compatible` = none.
- Capture, generation backend, provenance guard, driver and receipts read the profile (list of former hard-coded constants: LOCAL_REPORT.md). Capture reuses the existing `capture/capture.py` classes (byte-identical, pins unchanged) through `evidence_ladder/models/profile_capture.py`.
- The existing driver `tools/el_arch1_acceptance.py` gained `--mode local --backbone <id>` (defaults unchanged). No new driver/controller.
- Adding a third model later = one toml entry (+ an adapter set if head-compatible). Proof: a dummy third profile in `evidence_ladder/models/tests/backbone_tests.py` (14 tests) exercises capture shapes, order independence, limits and provenance without a model.
- Defect fixed in committed code: `controller/replay.py` re-raised every recorded scorer error as RuntimeError (replay journals differed whenever LAYA refused a pool); now re-raises the recorded type; regression test.

## 4. Measured timeline of the full run (local clock 2026-10-17, from receipts and file times)
| Step | Time |
|---|---|
| Device parity gate, 4B streaming on the RTX 3050, 6 pools | **61.7 s** (21:34 -> 21:35) |
| Round 1 generation, Qwen3-0.6B (13,052 tokens) | **549.9 s (9.2 min)**, 24.2 tok/s, GPU peak 1.13 GiB (about 244 s of it was a duplicate greedy seed, then fixed) |
| Round 1 LAYA scoring, live + verify (two independent processes, 4B streaming) | about 1 min each (3.4-9 s per decision) |
| Round 2 generation (6,109 tokens) | **265.0 s (4.4 min)**, 24.0 tok/s |
| Round 2 scoring, live + verify | about 1 min each |
| Driver pass 1 (investigate, risk table, WSL execution, repair, stops at WAITING_FOR_GLM) | finished 21:51 |
| **Pipeline so far, GPU work, parity -> end of pass 1** | **about 17 min (21:34 -> 21:51)** |
| Live GLM critic step (3 packets, `z-ai/glm-5.3-flash`, max reasoning, streamed) | **120 s wall** (93 s, 111 s, 120 s per packet), 3 sends, 0 retries, secret scan: no hits |
| Final driver pass (ingest GLM, stage maps, I1-I8, replay offline + live WSL, report) | under 1 minute (22:30) |
| **Whole run, parity start -> final receipts** | about 56 min wall, of which about 20 min were machine time; the rest was waiting for the GLM approval step |
Building the extension (agent work) took about 38 min on top. For comparison the Kaggle T4 reference run (Qwen3-4B) needed 55 + 50 min of generation plus queue waits (round 2 scoring has been queued over 3 hours).

## 5. Results (real, no prompt tuning, greedy, strict action parser)
Device parity gate (4B fp16 streaming on RTX 3050 vs stored T4 features, 6 pools): **PASS**, worst cosine 0.99998999, worst max-abs 0.125 (one fp16 spacing; not bit-identical). All 3 adapters keep the same top choice on every pool. Registration is for this process only; unregistered devices and fabricated tensors are still refused (T7 cases re-run).
| Example | Step 1 | Step 2 | Final | Journal hash |
|---|---|---|---|---|
| ex1_slug | 0.6B thinking loop hit the 5000-token cap, 0 proposals; LAYA not asked (`scorer_unavailable` journalled) | same | **BLOCKED** (budget) | ab1c86b2... |
| ex2_clamp | 3 valid options, LAYA order [1,0,2], chose a no-op Edit (old string = new string) | 4 options, LAYA [1,3,0,2], same no-op Edit | **BLOCKED** (budget); tests still fail | 2b70e3aa... |
| ex3_wordcount | answer not valid JSON, 0 options | 4 options, LAYA [1,3,0,2], no-op Edit | **BLOCKED** (budget) | c8579469... |
- LAYA's order changed the choice versus the id-order fallback in 3 of 3 steps where it was asked. No LAYA top choice was overridden by trusted evidence in this run (no real conflict happened; that was shown in the 4B Kaggle run, ex1 step 1).
- The critic (GLM) answered all 3 packets; it changed no label (diagnostic only).
- Gates: invariants **PASS** (I1-I8 x 3), monotonicity gate INSUFFICIENT_DATA (LAYA ordering of checks stays disabled), overall INSUFFICIENT_DATA ("3 examples; not a performance result"). Replay offline + live WSL identical for all 3. `secret_scan_hits`: none.
- None of the three tasks was solved. The 0.6B model fails mainly through thinking loops and format errors. This is wiring evidence only, not performance.

## 6. Caveats to keep attached
1. The local device registration rests on 6 pools only.
2. The run's code was committed (`437df8e5`) before the GLM step and final pass, but the GPU phases ran before that commit (same source; not separately proven).
3. 0.6B states exist for only 3 of 6 decisions (ex2 step 1 and 2, ex3 step 2); the others had fewer than 2 valid options.
4. The 4B streaming load needs about 7 GB commit; with large apps open it can fail (the gate then reports NOT_RUN and LAYA is marked unavailable, never silent).
5. Kaggle bundles must be rebuilt before any new push (qwen.py and laya_*.py now import backbone.py); already-pushed bundles are unaffected.
6. Probe and random-head are comparison scorers only (no selector, no keep/drop).
7. The 4 Kaggle round-2 scoring notebooks (two accounts) were still QUEUED at 22:30; that is the separate T4 reference run (`el-arch1-accept-20261017T101325Z`).

## 7. Commands (local)
- Whole local run: `PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe tools/el_arch1_acceptance.py results/raw/<new out dir> --mode local --backbone qwen3-0.6b` (stops at WAITING_FOR_GLM after the GPU work).
- Live GLM step (user's own PowerShell tab, user approves): `& 'D:\local model\.venv\Scripts\python.exe' 'D:\local model\tools\el_arch1_glm_live.py' 'D:\local model\results\raw\<out dir>'`, then re-run the driver command above.
- Tests: all suites run locally with `.venv` Python 3.12, `PYTHONPATH=.`, `ATEN_CPU_CAPABILITY=avx2` (backbone 14, controller 25, phaseA 16, phaseB 36, phaseD1 21, orchestrator 25, phaseC 25, models 30, critic_batch 28, stepE 12, replay 26 all green; `laya_integration` 24/25 locally because `T4_scores.json` exists only in the Kaggle bundle).

## 8. Next steps / open items
- [ ] Conformance report for the local run (`tools/el_arch1_conformance.py <out> --tests-results <TEST_RESULTS.json>`), then update `EL_ARCH_1_CHECKLIST.md` (D2-D7, Final).
- [ ] Decide on the Kaggle T4 reference round 2 (queued) or close the local run as the 3-example closure (labelled LOCAL, 0.6B).
- [ ] Universal step: train an adapter for Qwen3-0.6B on the stored Fable pools (needs capture of the 334 TRAIN pools with the profile and a separate user GO; no encoder/Qwen training).
- [ ] Better 0.6B proposals (thinking loops, JSON format) are a generation-side question, not a wiring one; no prompt tuning was done.
- [ ] Push to the branch `laya-int-1-stage0-critic-2026-10-17` only when the user asks (secret scan 0 first).
