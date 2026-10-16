# CLAUDE.md — jev-airllm

Read all of this before doing anything. These rules override convenience and speed.
If a rule blocks you, stop and ask. Never work around it.

Specs, in order of authority for what they cover:

- [FINAL_GPT_OSS_INTERNAL_LAYA_EXACT_IMPLEMENTATION_RUNBOOK.md](FINAL_GPT_OSS_INTERNAL_LAYA_EXACT_IMPLEMENTATION_RUNBOOK.md)
  ("the runbook") — decision layer, data, training, policy, Harness integration, evaluation.
  Followed phase by phase. Its gates are named **RB0–RB5** here to avoid clashing with PLAN.md.
- [PLAN.md](PLAN.md) — **runtime only**: S0, S1, R1–R4, S2 and gates G0–G3, G5. The runbook
  needs a working 120B runtime (its phase 15) and does not say how to build one.
  PLAN.md's Jev material is superseded: findings 6–10, Track J, G4, D1–D2, G8 and every
  mention of OpenJev/SemIf/AutoJev as the decision layer. G6–G7 (5 seeded bugs, protected
  tests, with and without the head) are kept as the agentic evaluation.

PLAN.md section 11 is the working step list (D0–D12) for the decision layer, merged from the
runbook and the user's pasted design chat. **Current focus (user, 2026-10-01): the internal head
only;** its design record is [DECISION_HEAD.md](DECISION_HEAD.md), and D9–D12 wait for RB4. **Decision-layer work never touches the runtime:**
the MXFP4 weights, the 6 GB streaming design and the expert loader stay as PLAN.md sections
2–8 define them.

Where the runbook and PLAN.md disagree on the decision layer, the runbook wins. Where either
disagrees with this file, this file wins.

## User rules 2026-10-11 (chat) — override everything below where they differ

- **NVIDIA calls:** the 800 / 1,000-call cap applied to the first hard-pool experiment only. There is no overall call cap now. Every
  send is still logged, completed ids are never resent, and spend stays Rs 0.
- **Throughput:** push every key to its limit while staying reliable: several requests may be in flight per key (this replaces
  "one worker per key"); stay at or below 39 sends per key per minute; on any rate-limit response that key cools for one minute and
  then resumes. Measured: non-streamed requests are cut at 300 s, so long calls must be streamed.
- **Fixed call logic (user, 2026-10-11, late):** up to 5 requests in flight per key (10 was tried and half were rejected), <= 39 sends per key per minute, a rate-limit response cools that key one minute and the request is re-queued; full text in AGENT_README.md "FIXED NVIDIA CALL LOGIC".
- **Call logic update (user, 2026-10-16, chat):** up to 10 requests in flight per key (replaces 5; the 2026-10-11 measurement at 10 saw about half rejected with 429), <= 39 sends per key per minute and a 1-minute cooldown on a rate-limit response are unchanged; all NVIDIA keys go into ONE Kaggle Secret `NVIDIA_API_KEYS` (one key per line) read only by the separate critic kernel (`evidence_ladder/models/kaggle_keys.py`).
- **Outside judges:** GLM-5.3, Gemma 4 31B and Kimi K3 on the same NVIDIA endpoints may review blinded packets (no labels sent).
  "Use max quality": do not lower reasoning effort to gain speed. No judge is qualified yet
  (`results/raw/judge-qual-v1-20261010T192150Z`); generated pools are not accepted for training until the review process is.
- **Nine-family menu (user, 2026-10-10):** nine action families with Web separate; versioned module `decision_head/fable_menu_v1.py`
  and `configs/laya_action_family_menu_v1.json`; the frozen eight-option trainer and capture limits and the 31 pretrained tensors are
  never modified. A single Bash command is never labelled by its last segment. Family routing is stage 1 only and is never reported
  as concrete-action imitation.
- **The linear probe on the frozen Qwen state is a mandatory baseline** for every head result (pilot: probe 67.1%, head 59.0%).
- **Still need the user's GO:** any further Kaggle capture, any fine-tuning run, and using previously examined trajectories for training.
- **Late 2026-10-11 decisions:** (1) GLM-5.3 full reasoning + Gemma 4 31B, both must accept, is the review pair; each judge's
  individual gate failure stays on record; the pair passed its own gate on a 28-pool pilot. (2) Previously examined canonical TRAIN
  trajectories MAY be used for training (trajectory and action-overlap safeguards), never for unbiased-evaluation claims. (3) Gemma
  reviews first, GLM only eligible survivors with the same packets, plus sampled Gemma-rejected pools and controls. (4) Combined
  fresh + reused generation is conditional on the current review results and audit, and runs in stages. (5) Development uses grouped
  cross-validation; canonical VAL is kept for later independent evaluation; check development capacity before the large batch.
  (6) Chance + 10 points on shortcut baselines is a warning level only. Remaining steps: AGENT_README.md "FROZEN NEXT PLAN".
- **Second judge (user, 2026-10-11, chat: "try 5 6 models and fix second judge"):** Gemma's endpoint is down; `meta/muse-glimmer-30b`
  (`muse_max`) passed the unchanged pair gate with GLM and reviews pools Gemma has not reviewed. Each pool records its actual judges; the set
  is never described as reviewed by one pair, and results are reported per judge cohort. Dataset v2:
  `results/raw/laya-fullaction-set-v2-20261011T134826Z` (123 pools). The Kaggle launch still needs the user's GO.
- **STANDING RULE (user, 2026-10-13, chat, restated: "if any computation needed use kaggle, make rule"): ANY computation runs on Kaggle, not on this PC**:
  feature capture, training, alignment, scoring, probes, smoke tests of model code, long analysis. This PC is for reading files, writing code and documents,
  packing Kaggle bundles, Kaggle CLI calls and NVIDIA API calls. The only local exception is one the user names for a specific task (so far: the isolated
  WSL sandbox test runs, approved 2026-10-13, because Kaggle has no equivalent sandbox). If Kaggle is queued or unavailable, say so and wait or ask;
  never fall back to the PC silently. Uploads still need the data to be approved for upload.
- **Compute on Kaggle (user, 2026-10-13, chat): when a job needs CPU or GPU compute (training, alignment, capture, long analysis), run it on Kaggle, not on this PC.**
  This supersedes the 2026-10-12 local-GPU line below for compute-heavy work. A local run that was almost finished on 2026-10-13 was allowed to complete.
  If Kaggle is queued or unavailable, say so and wait or ask; do not fall back to the PC silently. Uploads to Kaggle still need the data to be approved for upload.
- **Local GPU (user, 2026-10-12, chat: "use GPU on PC, remove rule of not using GPU"):** the RTX 3050 may be used; this lifts "No GPU work on the
  local RTX 3050" below. Qwen state capture still stays on one device class (features from two devices are not mixed without a parity check).
- **Fine-tuning on all Fable pools (user, 2026-10-12, chat):** GO to fine-tune the pretrained head on every qualified pool of dataset v3
  (`tools/laya_finetune_all_v1.py`); the open issues (no untouched test pools, authorship shortcut) are handled later and stay on record.
- **Objective restated (user, 2026-10-12, chat):** the work is to strengthen the pretrained LAYA decision head ITSELF so it can help smaller models choose
  better actions. Qwen is only a feature provider and is never trained. **No encoder is trained or touched for now** (encoders may change later; the user will
  explore that separately). Fable is the teacher. Teacher-selected actions are never described as verified optimal.
- **Evaluation data (user, 2026-10-12):** canonical CAL and TEST stay protected; the request builder's guard that keeps their contexts out of model requests stays
  in place. VAL is the development evaluation set (inspected closely; never trained on). The outcome evaluation uses the 23 fail-to-pass VAL decisions.
- **Approved, not started (2026-10-12):** mutation pilot on 20 TRAIN states (execution-verified labels, kept separate from imitation pools, with no-context and
  options-only controls); near-miss recovery pilot on 20-30 decisions. NOT approved: the full 150-decision recovery, any further head fine-tuning run beyond
  `tools/laya_headonly_v1.py` as run, any download of outside decision datasets (the Jev rule still needs the user's explicit decision), more Qwen generation.
- **Docs:** `README.md` is the project overview; the dated working log is `AGENT_README.md`; consolidated findings: HANDOFF section 16.
- Status and remaining steps: [HANDOFF_2026_10_11_STATUS.md](HANDOFF_2026_10_11_STATUS.md).

## Primary training route (user, 2026-10-10, chat) — overrides everything below where they differ

- **Supervised imitation of Fable** is the primary objective: fine-tune the EXISTING pretrained LAYA head (frozen Qwen,
  trainable adapter, then adapter + pretrained head as validation justifies) to select the action Fable recorded.
- **Two label kinds, never mixed up:** (1) *imitation label*: the recorded Fable action is the positive target; the other
  3–5-pool options are non-selected alternatives, NOT verified wrong; this is "choose like Fable", not proof of optimality.
  (2) *execution-verified preference*: BETTER/WORSE/EQUAL_ON_ORACLE/UNKNOWN from same-state sandbox runs
  (tools/fable_wsl_runner.py, tools/fable_test_oracle.py); a separate, optional objective. The 2026-10-03 rule "labels come
  from real outcomes" now applies only to kind (2).
- Pools: 3–5 distinct options including the recorded action; versioned imitation builder/validator; frozen v2/v3 contracts,
  diagnostic exclusions and canonical TRAIN/CAL/TEST boundaries unchanged. `objective="recorded"`, `unknown_weight=0.5`
  provisional, compared with 1.0 on held-out validation only (never TEST). Measure candidate recall@3/@5 separately from
  ranking accuracy; the teacher action is never assumed available at inference. No NVIDIA calls or fine-tuning until
  cleaned counts and real-feature compatibility are in.
- **NVIDIA keys (user, 2026-10-10, final rule):** all five keys (`.secrets/nvidia_api_key_2..6.txt`, several accounts) are used in
  round-robin through `tools/nvidia_keys.py`. This replaces the earlier "one account" rule for NVIDIA only. **Update (user, 2026-10-10,
  chat): a 429 no longer stops the run; it cools that key only and the request switches to the next key (`limit_scope="key"`); when every
  key is cooling the runner waits for the earliest one. Provider limit 40 requests/minute per key (user, 2026-10-10): send 39 on a key,
  then switch to the next; after the last key restart at the first (`KeyRotator(per_key_batch=39)`). Parallel runs (user, 2026-10-10): one worker per
  key, <= 39 sends per key per rolling minute, a 429 cools that key 60 s; the key pool is every `.secrets/nvidia_api_key_<n>.txt` (n >= 2), 10 keys on 2026-10-10.** Global call cap and no-resend rules stay. User was told multi-account
  use may conflict with the provider's terms.

## Current direction (user, 2026-10-04, chat) — overrides everything below where they differ

- **New head, same architecture as the internal head, started from the pretrained Laya `Coding_Decision_Agent`** (`models/coding-decision-agent`, no starting-point comparison), reading the frozen Qwen3-4B-Thinking-2507 states. This amends the 13.7 rule ("only the conductor is fine-tuned, the internal head is untouched") **for this new head only**; the old internal head code (`head.py`, `decision_head/`, `head_v3_t4`) stays untouched. Plan: PLAN.md sections 14–16, DECISION_HEAD.md section 20.
- **Capture:** the Qwen sampler on Kaggle finished with **45,500 of 46,826 decisions** (91 shards, none dropped; `results/raw/pool-states-v1-final-20261004-185656.json`). The last 1,326 were not captured and will not be re-run. The set was uploaded as the private Kaggle dataset `laya-head-states-v1`.
- **Negatives:** hybrid real options (tiers T1/T2/T3 plus Qwen proposals, soft targets, same renderer) are approved **to build and count**; recapture waits for the gates (`configs/thresholds_pool_v2.toml`). A generated proposal that equals or nearly equals the original choice is ignored and counted as a hit; a different one is kept with a middle score and never labelled wrong. Nothing is fabricated.
- **Compute: TPU only** (Kaggle TPU v5e-8, vllm-tpu) for the new pipeline, including the state recapture once the TPU parity test passes (`configs/thresholds_tpu_capture_v1.toml`). Qwen3-30B-A3B in thinking mode (Kaggle Model `qwen-lm/qwen-3/transformers/30b-a3b/1`, Apache-2.0, mounted inside Kaggle, no local download) generates the option proposals; the 61 GB download inside Kaggle was approved by the user in chat. All rows (train, cal, test) of the new set must be captured the same way. Spend stays Rs 0. No GPU work on the local RTX 3050.
- Status of every step and the remaining steps: PLAN.md ("Remaining steps") and RUNBOOK.md.

## Earlier direction (user, 2026-10-03)

**The live plan is PLAN.md section 13** (outcome-labelled one-week plan). A5-v3 imitation and the LOCKED capture are scrapped (never evaluate the 2,220-decision LOCKED set).
Fixed rules: the Qwen3-4B-Thinking-2507 backbone is never fine-tuned; labels come from real outcomes; `policy/hard_rules.py` and real tests override the head; the head ships only if the
three-arm agent test passes. **Still pending, so still blocked by the rules below until the user says yes in writing:** the frontier-trace downloads over 1 GB (SWE-rebench-openhands was approved and downloaded; Laya fine-tuning was approved; the agent runs on Kaggle),
dropping per-step registration paperwork, the agent platform choice and the pass margin. (`autojev/` and the JDownloader installer were deleted on 2026-10-03 with the user's yes.) Spend stays Rs 0.
**2026-10-04 (user, chat): the Laya conductor is the base and the only thing fine-tuned (PLAN.md 13.7); the internal head is not touched.**
The 2026-10-02 direction below stays valid for backbone, Qwen-only and compute policy.

## Earlier direction (user, 2026-10-02) — overrides the 120B-first and heretic-first wording below

The decision head is **backbone-agnostic: it must work for any local model**. **Qwen models only for now; gpt-oss
(120b and 20b-heretic) is parked and not touched** until the user says so. The experiment backbone is frozen
`Qwen3-4B-Thinking-2507` (revision `768f209d9e`, `models/qwen3-4b-thinking-2507`). No shortcuts. AGENT_README.md,
DECISION_HEAD.md and PLAN.md are updated as work lands, with what is done and what is not.
**The live plan is PLAN.md section 12** (phases 0-5, gates, stop rules, budget): outcome-labelled targets, Laya's
proper-scoring loss and temperature, free GPU first (Kaggle, Colab), staged paid GPU only as fallback and for LoRA.

**Compute policy (user, 2026-10-02):** renting a GPU is allowed in principle and free Kaggle/Colab come first.
**Update 2026-10-03 (user, in chat): Path A: every role (states AND state-free text embeddings) is captured on the Kaggle T4; amendment (e) is registered in configs/a5_v3_spec.toml and configs/thresholds.toml. Kaggle is the provider and the private upload of TRAIN and CAL decision rows and their trajectories (outcome fields stripped) to `laya-phase2-train-cal` is approved; the Phase-2 capture runs on Kaggle T4 x2 (`configs/phase2_device_rule_v1.toml`). Spend stays Rs 0. LOCKED raw data, the >1 GB output download, and any paid GPU are NOT approved.** Original wording follows. **No cap, provider or upload has been approved yet, so spend stays Rs 0 and nothing leaves this machine until the user
approves them in writing.** Proposed cap: Rs 6,000 in two steps (PLAN.md 12.4). (Multi-account rule removed by the user, 2026-10-10:
several Kaggle accounts may be used.)

## Mission

Build the highest-quality local coding agent that runs on 6 GB VRAM / 15 GB RAM:

- **Reasoner (primary):** `openai/gpt-oss-120b` — MoE, native MXFP4 experts, streamed by
  AirLLM with a custom expert-slice loader so a token reads only its routed experts.
- **Decision layer:** an internal pointer-style decision head over gpt-oss hidden states,
  trained with the backbone frozen. Laya-derived checkpoints are teachers and specialists
  only, never a runtime dependency (runbook sections 0 and 1).
- **Truth:** protected tests and real execution.

**Quality is the only acceptance criterion.** Speed and memory savings ship only if they
pass the same gate as everything else.

## Frozen decisions — changing any needs explicit user approval

- Primary model is gpt-oss-120b (Apache 2.0). Its released MXFP4 weights are the model:
  **no re-quantization and no KV quantization** on top of them unless a gate proves it safe.
- gpt-oss stays frozen. Selective PEFT (runbook phase 26) starts only after the head-only
  result is measured, and only if the user asks.
- DeepSeek-R1-Distill-Llama-70B is an optional stretch goal, started only after the system
  is closed and only if the user asks.
- If gpt-oss-120b fails a gate after real attempts to fix it, **stop and ask**. Do not
  silently swap models.
- Decision layer is the runbook's internal head. **User decision 2026-10-01: no AutoJev, no
  OpenJev/SemIf readout, no hosted Jev; Laya only.** `sarvam-jev/` stays in place, untouched.
- **Laya fine-tuning is allowed from 2026-10-03 (user, chat); none is started.** Earlier rule, now lifted: Laya checkpoints used as released, no training (user decision
  2026-10-01; runbook phase 14, Coding-Laya, is dropped). Post-hoc temperature fitting on
  their outputs does not change weights and is allowed.
- Runtime is AirLLM streaming plus our own expert loader. AirLLM's built-in
  `compression="4bit"/"8bit"` is not used (bitsandbytes, and it disables prefetch).
- Evidence order: hard safety and permissions > tool and schema invariants > protected
  tests / real execution > calibrated internal head > specialist advice > heuristics
  (runbook section 22.1).
- **Spend is zero until the user approves a cap in writing** (see Compute policy above). Then only within that
  cap, staged, with the stop rules of PLAN.md 12.3. No paid APIs. This replaces the earlier absolute ban on
  rentals and the runbook's "use a sufficiently provisioned machine" is allowed only through that approval.
- **All roles (TRAIN, CAL, LOCKED) are captured on the same device class and dtype.** A new device (T4 fp16,
  rented GPU) means recapturing CAL and passing the hard capture gates again. LOCKED is read once; a new head
  after that read needs a fresh LOCKED-2.

## Never

- Never claim quality is preserved without a recorded gate result in `results/`.
- Never change numerics silently (dequant precision, kernels, dtypes, attention path).
  Every such change must pass the equivalence gate.
- Never use lossy or output-changing speedups (lossy speculation, dropping tokens, skipping
  experts, lowering top-k).
- Never loosen or edit a threshold after seeing results. Add a new versioned file with a
  written reason instead.
- Never skip tests because a model is confident. Never pick a random candidate when the
  decision layer is uncertain.
- Never let the agent write to the protected evaluator or to pre-existing tests.
  Agent-written tests are not evidence of correctness.
- Never let a learned score override hard safety, permissions or objective execution
  invariants.
- Never treat head or teacher probabilities as truth. They are conditional on the options
  supplied.
- Never fabricate a counterfactual outcome. A branch that was not executed is stored with
  `executed=false` and no outcome. Teacher preference is kept separate from outcome.
- Never edit the locked benchmark. A change is a new benchmark version and voids every
  comparison against the old one.
- Never put locked-test or OOD provenance into train or calibration. Calibration data is
  never used for gradient updates.
- Never unfreeze gpt-oss because the head is "not good enough" on one split. Inspect leakage,
  marker quality, candidate quality, calibration and state selection first (RB4).
- Never assume a decision-marker token exists. Inspect the gpt-oss tokenizer first.
- Never let live session data change production weights. It goes to a versioned offline
  corpus and ships only through the locked benchmark and regression suite.
- Never quote a number you did not measure. Label unmeasured figures
  `estimate (unmeasured)`.
- Never modify `sarvam-jev/` or `reference/`. Copy what you need, keep the MIT notice.
- Never run `git tag`, `git merge`, `git rebase` or `git reset` unless the user explicitly asks. Read-only git is fine.
  **User decision 2026-10-16: the rule against `git commit` and `git push` is removed.** Commit and push to a new branch is allowed;
  the secret scan on added lines (must be 0) still gates every push, and nothing is merged into `main` without the user's ask.

## Ask first (state name, source, size or cost, then wait for a clear yes)

- Any download over 1 GB: gpt-oss-20b 12.8 GB, remainder of gpt-oss-120b (index total
  65.25 GB), `mvilacad/laya-conductor` 1.57 GB, `tampajohn/laya-stop-completion-judge`
  1.57 GB, and every runbook trajectory dataset (sizes unmeasured).
- Deleting or moving anything outside this project folder. Deleting model files or caches
  inside it also needs a yes.
- Installing packages (use `uv add`, never ad hoc `pip install`).
- Spending money (state provider, cap and hours first).
- Uploading code or data to Kaggle, Colab or any rental (LOCKED raw data only when the LOCKED capture starts),
  `git push`, `SWE-rebench-openhands` trajectories, the `greghavens` Fable 5 / GPT-5.6 Sol trace datasets (size unmeasured), `Qwen3-Reranker-4B` (about 8 GB).
- Anything that sends code or data off this machine. The DeepSeek Harness must be wired only
  to the local gpt-oss runtime; any configuration that calls a hosted model needs a yes.

## Quality gate rules

- Thresholds live in `configs/thresholds.toml`, committed **before** any comparison run.
- **"Same as reference"** means: our runtime's teacher-forced logits on identical inputs stay
  within a pre-registered margin, judged against the measured noise floor (reference vs
  itself at different batch shapes). Exact bit equality is not claimed.
- Reference is transformers' native `modeling_gpt_oss` computation, a different code path
  from our expert loader. Test on real-weight layer slices first, then on full-model samples.
- Margins come from behavioral impact (top-1 agreement, KL), not the noise floor alone.
- Regression means: do not ship, investigate, fix, re-test.
- Runbook gates, each recorded in `results/` before the next phase starts:
  RB0 env + model manifest + experiment ID · RB1 locked benchmark hashed and read-only ·
  RB2 every raw record has provenance and license · RB3 split manifest hashed and leakage
  proof · RB4 frozen-backbone head trained and diagnosed · RB5 hard-policy, split-leakage,
  journal-replay and candidate-masking tests pass.
- Head quality gates (runbook section 6.2): choice accuracy, macro-F1, log loss, Brier,
  top-k recall, ECE; noul AUROC, AUPRC, false-approval rate; permutation delta, OOD delta,
  long-context delta. Baselines: gpt-oss alone and each Laya teacher, on the locked benchmark.
- Hard-policy false-approval rate must be 0 on the safety bucket.

## Decision-layer rules

- Every decision includes a NONE / insufficient-evidence option and an abstain output.
- Never decide from a single candidate. Filter impossible options deterministically first.
  Candidates per decision: about 5–15 (runbook 10).
- Abstain when the margin is below the calibrated threshold. Abstaining triggers more
  evidence gathering, capped at 3 rounds, then escalate to the user.
- Important choices: permutation stability. Critical choices: pairwise comparisons and
  execution evidence.
- The head sets what to try next. It never overrides a protected test result.
- Each Laya specialist answers only the decision it was trained for (runbook 13, 23.1).
- Label authority: real execution > maintainer/human evidence > multi-teacher agreement with
  execution support > specialist teacher > heuristic. Store every disagreement.
- gpt-oss must be prompted in its harmony format. Tool calls are validated JSON with retries.

## Environment

- Python 3.12 via `uv`. Use `uv run <cmd>`. System Python is 3.14 and must not be used.
  Runbook commands that say `py -3.11` or `pip install` are replaced by `uv`.
- Runbook package directories (`compiler/`, `decision_head/`, `state/`, `policy/`, ...) are
  created inside this project root, one at a time, when a phase needs them. The runbook's
  full tree is not scaffolded up front.
- **G0 passed (2026-10-01):** the project `.venv` has `torch 2.14.0+cu126`, CUDA is available
  on the RTX 3050 (`results/raw/g0-cuda-*.json`). A separate venv, `laya-audit/.venv`, has
  `torch 2.14.1+cu126` for the Laya clone.
- **CPU torch crashes on this machine:** the AVX-512 kernels in torch 2.14 CPU builds raised
  access violations on the Ryzen 7 7840HS; `ATEN_CPU_CAPABILITY=avx2` ran clean. Any CPU-side
  torch job sets it before importing torch.
- **Memory:** on 2026-10-01 the commit limit was 51.7 GiB with 2.35 GiB free; Ableton Live 12
  Suite alone held about 20 GB of commit. That caused `MemoryError` in downloads. Ask the user
  to close it before long downloads or 120B runs; never kill their applications.
- Hardware (measured 2026-09-30): RTX 3050 6GB Laptop (sm_86), Ryzen 7 7840HS (AVX-512),
  15.3 GB RAM, one 477 GB Micron NVMe (rated ~3.5 GB/s read) holding C: (291 GB) and
  D: (181 GB). Free on 2026-10-01: D: 31.6 GB, C: 33.7 GB. Model files go under
  `JEV_MODELS_DIR` (see `configs/p0.toml`), never in git.
- Models on disk (2026-10-01), pinned in `configs/models.yaml`: gpt-oss-120b, all 15 shards,
  verified (sha256, headers, index); `models/laya-typed-decisions`, `models/laya-conductor`,
  `models/laya-stop-completion-judge`, each sha256-verified and loaded offline.
  `gpt-oss-20b-heretic` is BF16 (no MXFP4 blocks), so it cannot test the expert loader;
  `laya-audit/external-laya` is Laya v0.3.22 (commit 6d942c9). D: has 19.5 GB free.
  Also on disk: `models/qwen3-4b-thinking-2507` (7.6 GB, the experiment backbone) and `models/laya-code`.
  Capture commands need `PYTHONPATH=.` (e.g. `PYTHONPATH=. .venv/Scripts/python.exe tools/capture_pool_v3.py cal`).
  Never start a second capture process on the same role while one runs.

## Known risks (all figures are estimate (unmeasured) until recorded in `results/`)

- Runbook phase 15 assumes an 80 GB machine. Here, hidden states come from our own runtime.
  PLAN.md's ~470 tok/s prefill would make 10k states of 12k tokens (120M tokens) about 71
  hours. Size the dataset from the measured prefill rate (G5), not from the runbook. If the
  head cannot be trained within that budget, stop and ask. Renting needs the approval above.
- Qwen3-4B capture here, measured: 8,163 tokens and 10.2 s per decision on the RTX 3050, compute-bound
  (`results/raw/capture-cost-model-20261002-205000.json`). The frozen state may add nothing over
  candidate/history features for next-action imitation (0.663 vs 0.667, `pool-audit-A5v1-*.json`), so paid
  steps sit behind stop rules.
- The DeepSeek Harness (runbook phase 24) is a TypeScript project we have not read or run.
- The runbook's cited Laya teachers all exist on the Hugging Face API (checked 2026-10-01,
  Apache-2.0). Their revisions are not yet pinned in `configs/models.yaml`.

## Evidence and reporting

- Every headline number in docs must be backed by a row-level file under `results/raw/`.
- Benchmark outputs are create-only. Scripts refuse to overwrite. Use a new filename.
- Baselines get the same prompt quality as the system under test. An unfair baseline is a
  bug.
- Report failures faithfully with the actual output. If a step was skipped, say so. If you
  do not know, say so.

## Phase discipline

- Do not pass a gate on hope. Do not start the next step until the gate result is recorded.
- Order: runtime first (PLAN.md S0, S1, G0–G3), then runbook phases 0–14 on CPU and disk,
  then runbook phase 15 onward once the runtime and its measured throughput exist.
- Prove on the small proxy (gpt-oss-20b, same architecture) before spending 120b time.
  Whether a real-weight 2-layer slice of the 120B replaces the 20b download is an open
  question for the user; until answered, this rule stands.
- Create directories only when a step needs them. No speculative scaffolding.
- "Closed system" means all of these are recorded in `results/`:
  1. gpt-oss-120b runs end to end through AirLLM + our expert loader here, the equivalence
     gate passes, and real seconds per token and prefill tokens per second are measured.
  2. The runbook section 36 acceptance contract holds.
  3. The head is compared with gpt-oss alone and the Laya teachers on the locked benchmark,
     and the agent fixes 5 seeded bugs in a toy repo, judged by protected tests, with and
     without the head.
  Nothing beyond it (PEFT, 70B, extra speedups) starts before then.

## Benchmark increments (user, 2026-10-01)

- After every **major step** (new backbone, capture change, head variant, loss or calibration change,
  state-budget change, runtime change, agent change), rerun the same fixed benchmark and record the
  delta against the previous step in `results/raw/increment-<step>-<time>.json` (create-only).
- Fixed benchmark: the locked A5 set (`data/benchmark/a5-v1.jsonl`, hash in `results/raw/`) for the head,
  plus the small protected-task set for the agent once it exists. Report accuracy on covered decisions
  per tier, ECE, and the best baseline (TF-IDF 0.323 covered) side by side.
- Keep a running ledger `results/increments.md`: step, what changed, benchmark before, after, delta,
  and whether the gain exceeds the paired-bootstrap interval. A step that does not move the benchmark
  is recorded as such, not dropped. The benchmark is never changed to make a step look better.
