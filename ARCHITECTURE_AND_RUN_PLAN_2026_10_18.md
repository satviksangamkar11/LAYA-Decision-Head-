# EVIDENCE-LADDER ARCHITECTURE: THE CURRENT PIPELINE, HOW IT RUNS, HOW TO PLUG A MODEL IN (2026-10-18)

Source of truth for the design: `ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md` (invariants I1-I8, evidence ladder L0-L5, oracle tiers O1-O3, eight stages, section 13 = revision R3). This file is the working description of what is BUILT and RUNNING now.
Aim (user, 2026-10-18): not accuracy. The pipeline must run properly, end to end, in ONE run, and must work as an adapter/plugin on top of any model; accuracy improvement is the later purpose of that plugin.
Status words: DONE (committed or in the tree, tests green) / IN PROGRESS / DECISION (needs the user). Frozen findings: `FINDINGS_FROZEN_EL_ARCH1_2026_10_18.md`.

## 1. The pipeline (function-level tasks, ladder controller)
```
task (public prompt + public examples ONLY; hidden material never reaches the runtime, invariant I8)
 1 INTAKE        requirements, caps, oracle tier                       (INVESTIGATE = not applicable to function-level tasks: there is no repo to read)
 2 GENERATE      k=4 candidates from ANY model through the backend interface; thinking policy recorded
 3 FILTER        deterministic prune (candidate_filter, frozen interface v1)
 4 EXECUTE       public examples inside the SANDBOX through the provider port
 5 L2 / L1       model-written tests (must quote a verbatim span of the request, validated by mutants) / behaviour clusters
 6 DISCRIMINATE  >= 2 clusters -> blinded packet -> critic (GLM) called INLINE in the same run; diagnostic only
 7 RANK/REPAIR   trusted evidence first; a learned score never approves (I2); repair from the real failure text
 8 VERIFY        label SUPPORTED / VERIFIED / PLAUSIBLE / UNVERIFIED / BLOCKED; append-only hash-chained journal
every step: UTC timestamp (Timeline side channel, hash-neutral); replayable; per-task atomic results with a DONE marker
```
One command runs all of it and then audits itself: `tools/el_bank_local_run.py <ABSOLUTE out dir> [--tasks manifest] [--resume] [--sandbox mxc|wsl] [--selftest]` -> preflight -> tasks -> BANK_RESULTS.md/json -> CONFORMANCE.md/json -> stage-coverage line in RUN_SUMMARY.

## 2. The sandbox (the replaced one)
- Port, unchanged for every caller: `sandbox.run(workspace, cmd, persist=False, timeout=60)` -> `{exit_code, killed, stdout, stderr, protected_unchanged, pre_tree_sha256, post_tree_sha256, persisted}`. Infrastructure failure RAISES (never an invented pass).
- Backends (explicit, NEVER a silent fallback): **mxc** = Microsoft MXC processcontainer (desktop hosts; `LAYA_SANDBOX_BACKEND=mxc`), **wsl** = WSL2 + bubblewrap (rollback/comparison), **kaggle jail** = chroot + uid drop + seccomp (Kaggle only: MXC's Linux backend cannot start there, probe 2026-10-18). Selection and diagnostics: `evidence_ladder/controller/sandbox_provider.py`, `mxc_sandbox.py`, `win_job.py`, `mxc_dialect/`; migration/rollback: `MXC_MIGRATION.md`; report: `MXC_REPLACEMENT_REPORT.md` (verdict MXC_REPLACEMENT_PARTIAL).
- The bank (function-level) engine runs through the same port: `evidence_ladder/execution/provider_engine.py` (python3 runner in MXC). Journal environment id carries the backend label (`mxc-processcontainer`).
- Verified on this PC (2026-10-18, by the coordinator): real MXC tests 21/21; a command inside MXC cannot read the project, has no network and no secret-like environment variables; the 75 fault-injection runs pass on MXC (72 labelled mxc, 3 are the deliberately broken-sandbox stub); MXC is 20-61% faster than WSL at p95 when warm.
- Limits (verified): the MXC container has no POSIX shell (python3 only + a small posix-lite subset) and `java.exe` dies inside it (0xC0000142), so JAVA tasks cannot run under MXC: recorded NOT_TESTED_NO_JDK, run on WSL only if explicitly selected. Absolute paths are required (a relative scratch path made the container start in C:\Windows\System32: fixed in the runner).

## 3. Components and where they live
| Component | Files | State |
|---|---|---|
| Candidate filter, frozen interface v1 | controller/candidate_filter.py, EL_ARCH_INTERFACE_FREEZE_v1.md | DONE |
| Timeline (UTC per step, hash-neutral) | controller/timeline.py | DONE |
| Caps with named stop labels | controller/limits.py, agentic.py | DONE |
| Sandbox port + MXC provider | controller/sandbox_provider.py, mxc_sandbox.py, win_job.py, mxc_dialect/ | DONE (branch mxc-execution-provider-2026-10-18) |
| Bank engine over the port | execution/provider_engine.py (mxc), wsl_engine.py (comparison), kaggle_jail.py (Kaggle) | DONE |
| L2 v2 (prompt and parser that agree; v1 frozen) | controller/l2.py, cfg `l2_version: 2` | DONE |
| Thinking policies recorded in every result | bank40.check_policy (budget / prompt_brief / model_variant) | DONE |
| Model backends | models/qwen.py (`build_local_backend`, non-thinking wrapper NONTHINK_PREFIX), models/backbone.py + configs/backbones_v1.toml + backbones_v2.toml | DONE |
| Internal-state capture + diagnostic probe | models/state_capture.py, tools/el_bank_state_probe.py | DONE (hook off by default; comparison-only) |
| Inline critic (GLM, key system, bounded retries, degrade to api_unavailable) | tools/el_bank_local_run.py, critic_batch/ | DONE |
| One-run local runner + selftest | tools/el_bank_local_run.py | DONE; selftest PASS on real MXC (rich case) |
| Conformance audit per task and per run | tools/el_bank_conformance.py (strict default, `--legacy` for old runs) | DONE |
| Kaggle bank run (4B) | evidence_ladder/bank40.py, kaggle/el_bank40_run_v1.py, tools/pack_el_bank40_v1.py, tools/el_bank40_driver.py | DONE; 32/40 tasks ran |
| Report generator, traceability T01-T50 | tools/el_arch1_final_report.py, EL_ARCH_TRACEABILITY.md | DONE |

## 4. Plugging a model in (the plugin direction)
A model is one entry plus one backend call; nothing else in the pipeline is model-specific.
1. Register a backbone profile in a NEW versioned `configs/backbones_vN.toml`: repo, pinned revision, local path, hidden size, layers, capture depths, dtype, device policy (`resident` fits a 6 GB card for small models), device classes, sampling parameters, `enable_thinking` (true / false / absent), `head_compatible` (empty unless an adapter was trained on that model's states).
2. Put the pinned snapshot under `models/<name>/` with `MANIFEST_pinned.json` (sha256 of every file); the backend verifies it on first load.
3. Call `models.qwen.build_local_backend(profile_id, log_path, think_policy)`: returns a backend with `generate(request)` and `model_info()` (profile id + sha, snapshot_verified, sampling). Non-thinking models get an empty think block in front of their raw answer so every parser sees a closed block.
4. Capabilities that vary per model: hidden states (needed only by the learned scorer; without them the learned score is simply absent and everything else still runs), thinking control (budget / prompt / none), context length.
5. For API-only models the same controller, filter, sandbox, critic and journal apply; the only missing piece is the learned scorer. The frozen 4B LAYA head can never read another model's states (the provenance guard refuses them).
Research on packaging (library + model adapter first, then an OpenAI-compatible proxy, then host hooks; MCP deferred): `research/EL_RESEARCH_C_any_model_plugin.md`.

## 5. Results so far (development results: public examples only, hidden grading NOT done)
- Toy tasks ex1/ex2/ex3: 3/3 VERIFIED on Kaggle (Qwen3-4B, thinking budget 512); replay identical offline and in WSL; invariants I1-I8 pass; report 43 PASS / 0 FAIL / 7 INSUFFICIENT_DATA.
- Kaggle bank run (4B, prompt_brief, no thinking cutoff, ceiling 8192): 32 of 40 tasks in 11.25 h (30 SUPPORTED, 1 UNVERIFIED, 1 BLOCKED); L2 INSUFFICIENT on all 32 because of OUR prompt/parser mismatch (fixed in v2); model time 99.2%, sandbox 0.8%.
- Local run, Qwen2.5-Coder-1.5B-Instruct on the RTX 3050 (22 tok/s, 2.9 GiB VRAM), MXC sandbox, one command: 28 Python tasks + 20 Java NOT_TESTED_NO_JDK. FINAL (run `results/raw/el-bank-local-20261018T2106/`, 2026-10-18): conformance PIPELINE_COMPLETE, 28 Python tasks executed and 28 PIPELINE_OK, 20 Java NOT_TESTED_NO_JDK; labels SUPPORTED 21 / BLOCKED 7 (public examples only); stage coverage: public-pass 21, L2 reached 21 (validated 0: the 1.5B model invents quotes and the span rule drops every check), clusters>=2 in 3, critic triggered 3 / called 3 / applied 3 (two of them only after the critic read timeout was raised from 120 s to 420 s and the retry-critic step re-ran them from the saved generations), repair rounds 8; time 3,920 s task time: generation 78%, critic about 20% (5 calls, median 242 s), MXC sandbox 1.6%; per-step UTC timeline in TIMELINE.jsonl.

## 6. Open items and constraints
- DECISION (Java): WSL as a labelled exception in a second run, or leave NOT_TESTED. A JDK on the host does not help MXC.
- Critic: first-response timeout raised to 420 s (default) and `--retry-critic` added; large packets still take 4-12 min at max reasoning.
- Memory: this PC has little free RAM/commit; the runner refuses to start below 3.0 GB free RAM, 3.5 GB free VRAM, 4.5 GB free commit; load the model once, keep other heavy apps closed, one GPU process at a time.
- Not changed: frozen checkpoints, datasets, adapters, prompts v1/v2/v3, the evidence requirements, the journal format.
