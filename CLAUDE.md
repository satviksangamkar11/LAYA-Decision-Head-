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

## Current direction (user, 2026-10-01) — overrides the 120B-first wording below

The decision head is **backbone-agnostic: it must work for any local model**. The first backbone is
`gpt-oss-20b-heretic`. **gpt-oss-120b is not touched** until the user says so. `gpt-oss-20b-heretic`
is read-only and used only when the head's capture step needs it. No shortcuts. README.md,
DECISION_HEAD.md and PLAN.md are updated as work lands, with what is done and what is not.

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
- **Laya checkpoints are used as released. No Laya training or fine-tuning** (user decision
  2026-10-01; runbook phase 14, Coding-Laya, is dropped). Post-hoc temperature fitting on
  their outputs does not change weights and is allowed.
- Runtime is AirLLM streaming plus our own expert loader. AirLLM's built-in
  `compression="4bit"/"8bit"` is not used (bitsandbytes, and it disables prefetch).
- Evidence order: hard safety and permissions > tool and schema invariants > protected
  tests / real execution > calibrated internal head > specialist advice > heuristics
  (runbook section 22.1).
- **Total spend is zero.** No cloud, no paid APIs, no rentals. This overrides the runbook's
  "use a sufficiently provisioned machine" for hidden-state extraction (see Known risks).

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
- Never run `git commit`, `git push`, `git tag`, `git merge`, `git rebase` or `git reset`
  unless the user explicitly asks. Read-only git is fine.

## Ask first (state name, source, size or cost, then wait for a clear yes)

- Any download over 1 GB: gpt-oss-20b 12.8 GB, remainder of gpt-oss-120b (index total
  65.25 GB), `mvilacad/laya-conductor` 1.57 GB, `tampajohn/laya-stop-completion-judge`
  1.57 GB, and every runbook trajectory dataset (sizes unmeasured).
- Deleting or moving anything outside this project folder. Deleting model files or caches
  inside it also needs a yes.
- Installing packages (use `uv add`, never ad hoc `pip install`).
- Spending money.
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

## Known risks (all figures are estimate (unmeasured) until recorded in `results/`)

- Runbook phase 15 assumes an 80 GB machine. Here, hidden states come from our own runtime.
  PLAN.md's ~470 tok/s prefill would make 10k states of 12k tokens (120M tokens) about 71
  hours. Size the dataset from the measured prefill rate (G5), not from the runbook. If the
  head cannot be trained within that budget, stop and ask. Do not rent hardware.
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
