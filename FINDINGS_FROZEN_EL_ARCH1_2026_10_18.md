# FROZEN FINDINGS: EL-ARCH-1 final architecture closure (2026-10-17 to 2026-10-18)

Status: FROZEN. Nothing below is edited later; new work goes into a new dated file. Every number here is backed by a file under `results/raw/` (paths given); anything not measured says so.
Scope: architecture verification on ex1_slug, ex2_clamp, ex3_wordcount only. This is NOT a model-accuracy result; the overall gate reads INSUFFICIENT_DATA (3 examples) by design.

## 1. What was built (all committed on branch `evidence-ladder-runtime-step2`, then pushed as a new branch)
- Candidate filter (frozen interface `EL_ARCH_INTERFACE_FREEZE_v1.md`): per-option JSON salvage, sequential-plan rejection, no-op / unchanged / unappliable edit rejection, duplicate removal, repeated-failure block (state hash + normalised action + failure), abstain with a coded reason. No rejection by risk class (risk stays in the controller's fixed table). New prompt template `qwen_actions_v2`; `qwen_actions_v1` untouched.
- Controller: real source snippets + failing output in INVESTIGATE with a truncation record and a hidden-data guard; repair report fed back; enforced step / round / token / wall-time / model-call caps with named stop labels (`step_cap` ...); coverage table, `outcome` / `process` fields.
- Accounting and provenance: GLM reconcile (sends, retries, actual tokens), UTF-8 guards, usage request (`stream_options` accepted by the endpoint), driver records commit + config + source hashes, refuses a dirty tree and any secret hit on the final path, refuses mock backends.
- Two final paths kept apart: PATH A (live model, LAYA only on the exact accepted pool) and PATH B (stored 4B scores replayed only on exact stored contexts; mismatch = refusal).
- Optional thinking budget for the 4B Kaggle generation (`--think-budget N`, kernel `el_arch1_generate_lowthink_v1`, own dataset and kernel slugs).
- Report generator `tools/el_arch1_final_report.py` (verdict comes from receipt content, never from notes), traceability matrix T01-T50, fault-injection variants (25 variants x 3 examples = 75 runs).

## 2. Evidence (measured)
| Run | Where | Result | Path |
|---|---|---|---|
| Local, Qwen3-0.6B thinking, 4B LAYA by local streaming | RTX 3050 | all 3 BLOCKED at the 2-step cap; ex1 and ex2 step 1 hit the 5,000-token thinking cap (0 options); ex3 had 2 options so LAYA was correctly not asked (below its 3-option minimum); replay identical offline + WSL x3; GLM 3/3 with actual tokens (1,494 in / 2,693 out) | `results/raw/el-arch1-final-local-20261017T191233Z` (receipts; superseded receipt sets kept) |
| Kaggle, Qwen3-4B-Thinking, full thinking (cap 5,000), fixed pipeline, round 1 generation only | Tesla T4 | 3,115 s, 27,244 tokens, only 3 of 6 samples closed their thinking inside the cap; later steps not run | `results/raw/el-arch1-kaggle4b-20261017T204104Z` |
| **Kaggle, Qwen3-4B-Thinking, thinking budget 512, fixed pipeline, full run** | Tesla T4 (generation 390 s; scoring live + verify about 2 min each) | **ex1, ex2, ex3 all VERIFIED in round 1, stop `decided`**, tests pass, replay identical offline + live WSL x3, invariants I1-I8 PASS, GLM critic 3/3 (diagnostic; ex3 critic returned `fail` and the label stayed VERIFIED because the protected test passed), committed source `f31009ce`, secret scan 0 | `results/raw/el-arch1-kaggle4b-lowthink-20261017T213212Z` |
| PATH B replay (stored 4B scores, exact contexts) | WSL, CPU | ex1 and ex2 replayed (LAYA, probe, random head equal to stored, comparison-only); ex1 LAYA top option failed its executed trusted check and was overridden; ex3 refused (filtered pool no longer equals the stored pool) | `results/raw/el-arch1-pathb-20261017T201512Z` |
| Fault-injection variants | WSL, scripted model text, real controller + sandbox | 75 of 75 ok | `results/raw/el-arch1-final-variants-20261017T201524Z` |
| L2 / clustering harness, real 0.6B | RTX 3050 | ran without error after the I8 fix; 0 candidates (all samples thinking loops), L2 INSUFFICIENT, discrimination NOT_TRIGGERED | `results/raw/el-arch1-l2-local-20261017T201454Z` |
| Suites | CPU/WSL | 369 of 369 at the C2 run (370 after the last report-generator test); the later run is `el-arch1-final-suites-c3-*` | `results/raw/el-arch1-final-suites-c2-202610180140` |
| Report | | PASS 42 / FAIL 0 / INSUFFICIENT_DATA 8 on the local run (rows INSUFFICIENT: T01, T09, T27 A (no VERIFIED on that run), T12, T13, T26 (no candidates), T38 A+B and T47 (statistical rule, 3 examples)) | `ARCHITECTURE_FINAL_REPORT.md` |
Note: the 4B low-thinking run exercised the paths the 0.6B run could not (success path, VERIFIED, LAYA on an accepted pool of 3 to 5 options, filter rejections in a real receipt). It is not yet folded into the generated report rows; that is open item 1 below.

## 3. Why runs differ in time (measured)
- Generation decodes one token at a time: about 9 tokens/s for the 4B model on a T4 (27,244 tokens in 3,115 s), about 24 tokens/s for the 0.6B on the RTX 3050. Scoring is one forward pass over existing text: about 2 minutes per kernel.
- The full-thinking 4B model runs to the 5,000-token cap in about half of its samples; with a 512-token budget the same model closed 6 of 6 and finished in 390 s.
- Kaggle queue: about 16 minutes before the full-thinking generation started on 2026-10-18; the earlier round-2 scoring waited over 3 hours. At most 2 GPU sessions at once ("Maximum batch GPU session count of 2").
- Scalability limit is generation throughput, not the architecture (CPU parts take seconds). Levers (unmeasured): batching, a serving engine, more GPUs, a thinking budget.

## 4. Defects found and fixed during this closure
- Pipeline contract gaps: no-op and unappliable edits executed; repeated failed edit re-run; investigate gave symbol names only; one bad JSON item discarded the whole list; plan steps scored as competing options; "budget" label hid the real cap; step cap was logged but not enforced.
- Found by running for real: the L2 harness crashed on its own create-only check; the deep I8 scan flagged ordinary model attribute names (`hidden_size`) as evaluation-only data; the journal gave a generic reason for a pool with fewer than 3 options; a credential-shaped test fixture tripped the secret scan; the Kaggle CLI needs `-p .` run from inside the dataset folder on Windows.
- Frozen kernel fact: the scoring kernel refuses fewer than 3 options (the adapters were trained on 3 to 5 option pools).

## 5. Corrections to earlier statements (kept on record)
- I wrote that saved GLM responses contained mojibake. That was wrong: the files hold correct UTF-8; I had read them with the Windows default code page. Agent 3's finding stands; guards were added anyway.
- The `evidence_ladder/models` path is matched by a `.gitignore` rule; new files there need `git add -f`.

## 6. Open items and decisions (not done)
1. Fold the 4B low-thinking Kaggle run into the final report as its own PATH A section, then declare or withhold `CODING_ARCHITECTURE_INTEGRATION_PASS` (statistical rows stay INSUFFICIENT_DATA by rule; rows T12, T13, T26 still need a run where the model yields 2+ candidates).
2. Full-thinking Kaggle run: only round 1 generated; remaining steps not run (decision: not needed for the closure).
3. The sandbox: user question 2026-10-18 (necessary? remove? free options?). Research agents wrote `research/EL_RESEARCH_A..C_*.md`; the decision is open. My position: keep the "observe before commit" step, replace the WSL backend behind the existing sandbox port.
4. Plugin direction ("connect any model anywhere"): research in `research/EL_RESEARCH_C_any_model_plugin.md`. LAYA needs hidden states, so it applies to open-weight models only; API models would run without the learned scorer.
5. `git push` of this work was requested; target: a NEW branch, never `main`.

## 7. Reproduce
- Suites: `PYTHONPATH=. .venv/Scripts/python.exe tools/el_arch1_run_suites.py <new dir>`
- Local run: `tools/el_arch1_acceptance.py <new dir> --mode local --backbone qwen3-0.6b` (GLM step: `tools/el_arch1_glm_live.py <dir>`, then re-run).
- Kaggle 4B: pack with `tools/pack_el_arch1_bundle_v1.py`, upload the dataset (run `kaggle datasets version -p .` from inside the dataset folder), run `tools/el_arch1_acceptance.py <new dir> --final --bundle <bundle> [--think-budget 512]`, push the kernels it writes, download outputs, re-run.
