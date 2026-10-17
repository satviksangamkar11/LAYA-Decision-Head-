# EL-ARCH-1 LOCAL RUN report (mode=LOCAL, device RTX 3050 6GB, generation model Qwen3-0.6B, LAYA states from local 4B streaming)
Out dir: results/raw/el-arch1-local-20261017T160410Z (driver log: driver_run1.log).
STATUS: stopped at WAITING_FOR_GLM (3 packets). The final receipts/ACCEPTANCE_SUMMARY.json are written by re-running the driver after GLM.

GLM command (coordinator runs it in the user's terminal), then re-run the driver with the same command line (all GPU work is cached; the re-run needs no GPU):

    & 'D:\local model\.venv\Scripts\python.exe' 'D:\local model\tools\el_arch1_glm_live.py' 'results\raw\el-arch1-local-20261017T160410Z'
    PYTHONPATH=. .venv/Scripts/python.exe tools/el_arch1_acceptance.py results/raw/el-arch1-local-20261017T160410Z --mode local --backbone qwen3-0.6b

A throwaway rehearsal of that final step (fake in-process GLM, directory deleted) ran clean: invariants I1-I8 x 3, replay offline + live WSL identical x 3, all 8 stages reached x 3.

## Design (user correction: extend, do not rebuild)
ONE new concept, the backbone profile: configs/backbones_v1.toml + evidence_ladder/models/backbone.py (id, repo, revision, local path, hidden, layers, depths, dtype, local device policy stream|resident,
device classes, recipe id, cue, option prefix, head_compatible, generation settings, enable_thinking; sha256 of the profile).
4B = 18/24/30 as trained, head_compatible=[laya_fable_adapters_v1]. 0.6B = 14/19/23 (round(d/36*28)), head_compatible=[] (diagnostic only).
The 4B entry is also embedded in backbone.py and cross-checked against the toml (a difference raises).
Profile id + sha256 are written into every provenance record, the run config of the replayable record (record.json "backbones"), and ACCEPTANCE_SUMMARY (backends.*.profile_sha256, local.*).
Driver: tools/el_arch1_acceptance.py gained `--mode local --backbone <id>` (+ internal `--local-worker`); defaults unchanged (Kaggle / final / rehearsal paths untouched, suites green). No new controller, no new driver.

Former hard-coded constants, now profile-driven:
- laya_forward: DEPTHS, HIDDEN, QUERY_CUE
- laya_runtime: QWEN_REPO, QWEN_REVISION, RECIPE_ID, DEVICE_CLASS; provenance dtype / hidden / cue / option_prefix / depths and every comparison in check_metadata; check_tensors shape [3,2560]; make_provenance fields; LiveScorer
- qwen.py: QWEN_REPO / REVISION, sampling params and do_sample, the thinking-only rule, fp16 dtype, repo/revision stamped on Generations, the /kaggle gate (local=True lifts it explicitly)
- acceptance driver: generation-record repo/revision check
- probe.py / random_head.py default dims (through laya_forward)
Still literal by design: the frozen Kaggle kernels (unedited), capture-code pins (recipe-level), laya.py build_meta compatibility dict, profile_capture LIM (same limits as the kernels).
Adding a model = one toml entry (+ an adapter set if head_compatible). Proof: a dummy third profile in backbone_tests exercises capture shapes, option-order independence, limits and provenance without a model.

## L1 small-model state exposure: DONE
Same recipe positions (last prefix token after "\n\nNext action:"; mean over each option's own tokens as an independent suffix; packed pass with the capture/capture.py classes; lm_head never run) via
profile_capture.make_capture_fn: no second capture implementation; capture/capture.py and capture_resident.py are byte-identical, pins unchanged.
Real-model checks (backbone_gpu_tests.py -> L1_qwen06b_states_gpu_checks.json, sha256 ffc595aa05243b52..., PASS): shapes [3,1024] / [N,3,1024] at depths 14/19/23, fp16 finite, bit-identical on repeat, option-order independence (rows follow options), packed == plain forward, and the guard refuses with "DIAGNOSTIC ... head_compatible".
Stored per decision under kaggle_in/round*/gen/states_qwen3-0.6b/*.pt (hq, ho, options, provenance with profile id + sha, label "qwen3-0.6b states: DIAGNOSTIC; NOT consumable by the frozen LAYA head; data for a future adapter"):
ex2_clamp step1 (b3edd5aafe41ee24), ex2_clamp step2 (c9970650f5751999), ex3_wordcount step2 (56acbcfdfd81d1b2). ex1 (both rounds) and ex3 step1 gave fewer than 2 valid options, so no states.

## L2 local generation backend: DONE
HFQwenBackend(profile=qwen3-0.6b, local=True) through the same GenerationRequest / run_jobs / ActionProposer / ReplayGenerationBackend (prompt-hash checks) path; frozen qwen_actions_v1 prompt, strict parser, greedy, enable_thinking=true (the 4B runs always think), cap 5000 new tokens.
Snapshot sha256 of model.safetensors and tokenizer.json verified against MANIFEST_pinned.json.
Measured: 24.2 tok/s round 1 (13052 tokens, 539 s), 24.0 tok/s round 2 (6109 tokens); GPU peak 1.13 GiB.
NOTE: round 1 generated seed 1 again although decoding is greedy (identical text; about 244 s and the tokens double counted in the budget record); fixed (greedy cache) before round 2, where seed 1 costs 0 tokens.

## L3 LAYA on the local run: DONE (parity gate PASSED)
configs/el_arch1_local_parity_v1.toml (sha256 1628b70f9bccedba...) was written BEFORE any 4B capture on the 3050: cos >= 0.9999 and max_abs <= 0.5 on ALL records.
Result parity/PARITY_RESULT.json (sha256 e0082327d2f210b7...): PASS on all 6 pools (3 TRAIN + 3 VAL), device "NVIDIA GeForce RTX 3050 6GB Laptop GPU", fp16 per-layer streaming, peak GPU 0.64 GiB, 61.7 s.
Worst cos_min 0.99998999 (VAL bash-logtop:2), worst max_abs_diff 0.125 (one fp16 spacing; not bit-identical, no row above 0.125).

| pool | role | cos_min | max_abs |
|---|---|---|---|
| bash-dotlink:2 | train | 0.9999979 | 0.0625 |
| py-klondike:1 | train | 0.9999982 | 0.0625 |
| go-dashfeed:4 | train | 0.9999960 | 0.0781 |
| bash-logtop:2 | val | 0.9999900 | 0.125 |
| bash-retryrun:3 | val | 0.9999973 | 0.125 |
| bash-retryrun:4 | val | 0.9999973 | 0.125 |

Recorded, not gating: all 3 adapters keep the same top choice on every pool; max |logit diff| local vs stored 0.0004 .. 0.079.
Registration: parity/el_arch1_local_device_v1.toml (names the passed result by sha256; backbone.register_device re-checks it; per profile, per process only).
T7 refusal cases re-run with the extended list (backbone_tests): unregistered or other devices still refused, a FAIL or tampered result cannot register, fabricated tensors with valid metadata on the registered device fail the re-capture comparison, wrong revision refused.
Live + verify = two independent worker processes; scores and feature hashes equal, so VerifiedSavedScoreBackend admitted them. Streaming 4B capture about 3.4-9 s per decision.

## L4 driver: DONE (GLM pending).  L5 outcomes (real, prompts not tuned; strict parser, greedy, max 2 steps)
| example | step 1 | step 2 | result |
|---|---|---|---|
| ex1_slug | thinking loop hit the 5000-token cap, no </think>, 0 proposals | same | no LAYA call (scorer_unavailable journalled), no action, BLOCKED (max_steps) |
| ex2_clamp | 4 proposals, 1 invalid (tool "Summary") -> 3 options; LAYA order [1,0,2]; chose Edit clamp.py old==new (no-op) | 4 options, LAYA [1,3,0,2], same no-op Edit | tests still fail, BLOCKED |
| ex3_wordcount | answer not valid JSON (unescaped quote) -> 0 options | 4 options, chose a no-op Edit | BLOCKED |
None solved. Probe and random head are journaled as comparison only. Qwen generated tokens per the budget records (rehearsal): ex1 15000 (10000 of them the round-1 duplicate seed), ex2 1731, ex3 2430.
GPU peaks: generation 1.13 GiB, 4B scoring 0.23-0.24 GiB, parity 0.64 GiB.

## Files changed (committed files, minimal) / added
Changed: evidence_ladder/models/laya_forward.py, laya_runtime.py, qwen.py; evidence_ladder/controller/replay.py (defect fix below); tools/el_arch1_acceptance.py; tools/pack_el_arch1_bundle_v1.py (EXTRA += configs/backbones_v1.toml).
Added: configs/backbones_v1.toml, configs/el_arch1_local_parity_v1.toml, evidence_ladder/models/backbone.py, profile_capture.py, tests/backbone_tests.py (14 tests), tests/backbone_gpu_tests.py.
Untouched: capture/*, kaggle/*, adapters, kaggle/qwen_proposals_v2.py, kaggle/tpu_qwen30b_smoke.py.

Defect found in committed replay.py: RecordedScorer re-raised every recorded scorer error as RuntimeError, so the journal's scorer_unavailable reason ("ProvenanceError: ...") differed on replay whenever LAYA refused a pool (replay identical=false).
Fix: re-raise with the recorded exception type name (3 lines); regression test in backbone_tests. The sandbox "raised" path has the same pattern (not hit).

Suites after the changes: controller 25, phaseA 16, phaseB 36, phaseD1 21, orchestrator 25, phaseC 25, critic_batch 28, stepE 12, models 30, laya_integration 24/25 (the known T4_scores.json one), backbone 14: green.

## Open risks
- Final receipts do not exist until GLM ran and the driver was re-run.
- The local device registration rests on 6 pools.
- 0.6B greedy thinking loops (ex1) and format errors dominate the outcomes.
- Kaggle bundles must be rebuilt (packer EXTRA edited) before any new Kaggle push, because qwen.py / laya_*.py now import backbone.py; already-pushed bundles are unaffected.
- The 4B load needed about 7 GB commit with about 8 GB headroom (other apps running); it worked, but a larger concurrent app could make it fail (the parity gate would then report NOT_RUN and LAYA would be unavailable, never silent).
- No commits, no Kaggle, no NVIDIA, .secrets not read.
