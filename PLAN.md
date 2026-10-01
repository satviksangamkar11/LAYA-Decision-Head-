# PLAN â€” local coding agent: gpt-oss-120b + OpenJev on 6 GB VRAM / 15 GB RAM

Rules are in [CLAUDE.md](CLAUDE.md). This file is the spec and the step list.

## 1. Objective and definition of done

Best-quality local coding agent on this machine, at zero spend. Quality is the only
acceptance criterion; speed is a bonus that must pass the same gates.

**The system is closed when all five are recorded in `results/`:**

1. gpt-oss-120b runs end to end through AirLLM + our expert loader on this machine, and the
   equivalence gate (section 4) passes.
2. A Jev reader is chosen by measurement, calibrated, and its gate metrics recorded.
3. The agent fixes 5 seeded bugs in a toy repo, judged by protected tests, with and without
   Jev.
4. Decision-level evaluation (Jev vs baselines on labeled candidates) is recorded.
5. Real seconds per token and prefill tokens per second are measured and reported.

Optional, only after closing and only on request: 70B stretch goal, AutoJev quantization,
extra speedups.

## 2. Architecture

```
gpt-oss-120b (harmony format)            reasoner / candidate generator
  MXFP4 experts on disk, BF16 dense parts
  AirLLM layer hooks + OUR expert loader (reads only routed experts' slices)
Jev decision layer (OpenJev / SemIf readout)
  reader = gpt-oss itself | AutoJev-27B BF16 | small SemIf model   (chosen at G4)
Agent controller   candidates -> deterministic filter -> Jev (NONE option, abstain) -> act
Tools              files, search, shell, git, tests; git worktree per candidate
Protected evaluator  read-only to the agent; the source of truth
```

Evidence order: hard facts > protected tests > Jev > model confidence.
Jev sets search priority. Execution establishes correctness.

## 3. Findings (verified unless marked)

| # | Finding | Consequence |
|---|---|---|
| 1 | gpt-oss-120b: 36 layers, hidden 2880, 128 experts, top-4, 57.6 GB (14 shards). Only experts are MXFP4; attention, router, embeddings, lm_head stay BF16. | Native precision: no quantization gap to prove. |
| 2 | Per token: ~1.9 GB experts (144 expert reads) + ~3.1 GB attention and lm_head (BF16, keepable resident). (arithmetic from config, estimate) | Decode ~0.5-2 s/token estimate (unmeasured) vs 21-24 s for the dense 70B. |
| 3 | KV ~36 KiB/token BF16 (18 full-attention layers; sliding layers keep 128 tokens). (arithmetic, estimate) | 131k context ~4.7 GB. No KV compression needed. |
| 4 | Local transformers 5.17 has `GptOssExperts` with fused `[E, H, 2I]` tensors; forward takes `router_indices`. AirLLM 4.0.0 per-expert streaming needs an expert ModuleList and has no gpt-oss adapter. | We write an expert-slice loader; without it a token reads all 57.6 GB. |
| 5 | Project venv has `torch 2.14.0+cpu`. `cu126` index has `2.14.0+cu126`. | Step S0. |
| 6 | Hosted Jev needs an API key; no limits/privacy/pricing docs. | Not used. |
| 7 | OpenJev is renamed SemIf (TheoLeeCJ/openjev, MIT): one forward pass, softmax over option-letter logits. Direct logits on Qwen3.5-4B scored 0.813 authored balanced accuracy; temperature scaling took ECE 0.068 -> 0.038. | Reader design is proven on small models; unproven on gpt-oss. |
| 8 | `sarvam-jev/` already adapts it (core.py, direct.py, shared.py, metrics.py) incl. shared-state prefill and slot discovery. | Copy with MIT notice, do not rewrite. |
| 9 | In sarvam-jev, a chat template lowered the readout score (0.516 -> 0.404) and BF16 noise flipped argmaxes on near-indifferent models. | Test harmony format explicitly; measure argmax stability. |
| 10 | AutoJev-27B: dense BF16 ~49 GiB, readout head, temperature-scaled, 84.6% on its own text set (not coding). | Reader option (b); ~15 s per pass when streamed. |
| 11 | gpt-oss-120b vendor numbers: SWE-bench Verified 62.4%, GPQA Diamond 80.8, Codeforces 2463 (no tools). Harness-dependent. | Not evidence for our tasks; our own tests decide. |
| 12 | Prefill/decision passes over long prompts touch every expert (~57 GB, ~17 s at 3.5 GB/s). | Batch all questions of a state into one pass with a shared prefix. |

## 4. Quality contract

- **Equivalence gate (runtime):** teacher-forced logits from our runtime vs transformers'
  native `modeling_gpt_oss` on identical inputs, top-1 agreement and KL within margins
  pre-registered in `configs/thresholds.toml` after measuring the noise floor.
  Order: real-weight 2-layer slice on GPU -> full-model samples.
- **Noise floor:** reference vs itself at different batch shapes.
- **MXFP4 -> BF16 dequant** is exact; our BF16 matmuls are not bit-identical to vendor
  kernels. Measured, never assumed.
- **Jev gates:** balanced accuracy (authored-144 and our own coding-decision set), ECE,
  Brier, argmax stability under BF16 noise and option permutation.
- **Agent gate:** protected tests only.

## 5. Budgets (all estimates until measured)

| Resource | Budget |
|---|---|
| Disk | 57.6 GB (120b) + 12.8 GB (20b proxy, deletable) + 54 GB only if AutoJev is the reader |
| VRAM 6 GB | resident dense ~3.1 GB + expert working set ~0.3 GB + activations + KV of active layers |
| RAM 15.3 GB | OS ~4, Python ~1.5, embedding table ~1.2, KV up to ~4.7 GB at 131k, rest expert cache |
| Decode | ~0.5-2 s/token |
| Prefill | ~470 tokens/s (one ~57 GB read per ~8k-token chunk) |
| Jev pass | ~17 s per state batch (own reader) or ~15 s (AutoJev) |

## 6. Flow â€” fastest path to a closed system

```
START
  |
  S0  finish CUDA torch (uv add cu126)  --G0: torch.cuda.is_available() True
  |
  S1  AirLLM smoke test, tiny random Llama on GPU (no big download)  --G1: generates
  |
  +---------------------------------+----------------------------------+
  |  TRACK R: RUNTIME               |  TRACK J: JEV (no GPU needed)    |
  |                                 |                                  |
  |  R1 download gpt-oss-20b        |  J1 clone openjev/SemIf fixtures |
  |     (12.8 GB, approval)         |     (authored144, JevBench)      |
  |  R2 read safetensors layout;    |  J2 copy adapted core/shared/    |
  |     measure expert-slice read   |     metrics from sarvam-jev      |
  |     speed on NVMe               |  J3 harmony prompt builder +     |
  |     --G2: go / tune / stop      |     answer-slot discovery for    |
  |  R3 expert loader + AirLLM      |     gpt-oss tokenizer + unit     |
  |     gpt-oss adapter (20b)       |     tests                        |
  |  R4 equivalence tests, noise    |                                  |
  |     floor, commit thresholds    |                                  |
  |     --G3: equivalence passes    |                                  |
  +---------------+-----------------+----------------+-----------------+
                  |                                  |
                  +----------------+-----------------+
                                   |
  J4  Jev readout on authored144 via runtime with gpt-oss-20b
      (direct vs shared-state, permutation avg, temperature fit)
      --G4: accuracy/ECE/stability meet pre-registered targets
      fail -> reader (b) AutoJev-27B BF16, then (c) small SemIf model on CPU
                                   |
  S2  download gpt-oss-120b (57.6 GB, approval, free disk)
      repeat R4 on real 120b samples; measure s/token and prefill tok/s
      --G5: meets the practicality threshold the user accepts
      fail -> tune I/O and caches; still fail -> STOP AND ASK (fallback ladder)
                                   |
  A1  tools + git worktrees + protected evaluator + harmony tool-call loop
      (validator + retries); control run WITHOUT Jev on toy repo, 5 seeded bugs
      --G6: bugs fixed, judged by protected tests
                                   |
  A2  wire Jev decision boundary (NONE option, abstain, 3-round cap, escalate)
      rerun the same 5 bugs
      --G7: no regression vs control
                                   |
  D1  build labeled decision set from candidate patches + protected-test outcomes
  D2  score Jev vs baselines (random, first, model's pick); refit temperature and
      abstain thresholds on a held-out split
      --G8: beats baselines on held-out data, else Jev stays advisory/off
                                   |
  C1  record everything in results/, update docs; user commits
                                   |
                                 CLOSED
                                   |
                (optional, on request: 70B stretch, AutoJev quantization, speedups)
```

## 7. Gates (pre-registered targets go in `configs/thresholds.toml` before each run)

| Gate | Passes when | If it fails |
|---|---|---|
| G0 | `torch.cuda.is_available()` True | fix install |
| G1 | AirLLM generates on the GPU under Windows | fix env |
| G2 | expert-slice read speed recorded and above the go/no-go threshold | tune (unbuffered I/O, RAM expert cache); else stop and ask |
| G3 | logits within margin of native reference (slice, then samples) | fix loader; never widen margin after the fact |
| G4 | Jev readout meets accuracy, ECE, stability targets | next reader |
| G5 | measured s/token and prefill within the accepted threshold | stop and ask |
| G6-G7 | 5/5 seeded bugs fixed, no regression with Jev | debug |
| G8 | Jev beats baselines on held-out decisions | Jev advisory or off |

## 8. Layout (create only when a step needs it)

```
configs/p0.toml, thresholds.toml
reference/            clones of openjev etc. (gitignored, read-only)
airllm_gptoss/        expert loader, AirLLM adapter, harmony helpers
jev/                  readout (adapted from sarvam-jev), reader backends, calibration
agent/                controller, tools, worktree runner, validator
certify/              equivalence + Jev metrics, gate runner
results/raw/          row-level evidence for every reported number
```

## 9. Unverified register

- ~0.5-2 s/token, ~470 tok/s prefill, ~17 s Jev pass, NVMe ~3.5 GB/s.
- Checkpoint tensor layout for fused MXFP4 experts (blocks/scales) and slice read cost.
- Random ~8 MB read throughput on Windows NVMe.
- AirLLM 4.0.0 + transformers 5.17 + gpt-oss on Windows.
- Readout accuracy of gpt-oss in harmony format.
- Whether the local `openjev` fixtures and the SemIf renaming match what sarvam-jev used.
- Numerical difference between our BF16 expert matmuls and vendor MXFP4 kernels.

## 10. Approvals needed as we go

S0 CUDA torch download (~2-3 GB, previously approved, install unfinished) Â· R1 gpt-oss-20b
(12.8 GB) Â· J1 small repo clone Â· S2 gpt-oss-120b (57.6 GB, needs free disk) Â· any AutoJev
download (54 GB) only if reader (b) is chosen.

## 11. Decision layer: internal Laya/Jeeves-style head (supersedes the Jev material above)

Sources: the runbook, plus the design chat the user pasted on 2026-10-01. Sections 2â€“8 above
(runtime, MXFP4 weights, 6 GB streaming, Track R, G0â€“G3, G5) are **unchanged and untouched**.
Nothing in this section may change numerics, quantization or the expert loader.

**Current focus (user, 2026-10-01): the internal head only.** The design record, parameter
inventory, ablation plan and working order are in [DECISION_HEAD.md](DECISION_HEAD.md). D9â€“D12
below (policy, coordinator, specialists, retrieval, memory, Harness, release) are deferred until
RB4 has a result. The head code is `decision_head/head.py` with `selfcheck.py` (synthetic data
only). Real hidden states still depend on the runtime gates G1â€“G3 and G5, which are unchanged.

**Flow.** gpt-oss reasons and generates 5â€“15 candidates -> a decision pass reuses the KV cache
and adds only the candidate tokens plus a decision marker -> internal head scores candidates
(`s_i = qÂ·k_i/sqrt(d) + b_i`, `P = softmax(s/T)`, T fitted on calibration data) -> calibration
-> coordinator -> deterministic hard policy -> tool -> real environment -> journal -> state.
Marker token is chosen by tokenizer inspection and ablation, never assumed.

**Authority.** hard rules and permissions > execution results (tests, compiler, runtime) >
calibrated head > specialists > heuristics. A probability never overrides reality.

**Weights** (sizes measured from the HF API unless noted)

| Weight | Role | Trained |
|---|---|---|
| gpt-oss-120b, MXFP4 as released | reasoner, hidden-state source | no, frozen |
| internal head: Wq, Wk, metadata, abstain, escalate, type embeddings | decision layer | **yes (only training job)** |
| laya-typed-decisions 0.785 GB, cortex-1-large 0.785 GB | decision teachers | no |
| laya-conductor 1.569 GB | effort/continue specialist | no |
| laya-stop-completion-judge 1.569 GB | completion specialist | no |
| laya-code 0.785 GB | code relevance / rerank | no |

Not used: AutoJev, OpenJev/SemIf readout, the `laya` base (it only served Coding-Laya training).
Out of scope until asked: Coding-Laya training (user decision), selective PEFT, vision,
browser, Gyra and MacJev specialists, multimodal extension.

**Steps** (gates RB0â€“RB5 from CLAUDE.md; each recorded in `results/` before the next step)
- D0 RB0: pin env, register every weight with exact revision and sha256, experiment ID.
- D1 decision types (ACTION, EVIDENCE, CONTINUE, RECOVERY, PATCH, COMPLETION, ESCALATE,
  ABSTAIN), candidate and state JSON schemas, NONE and abstain on every decision.
- D2 RB1: locked benchmark (decision core, safety, recovery, long-state, OOD, permutation,
  counterfactual), hashed and read-only. Metrics and thresholds committed first.
- D3 RB2: ingest trajectories and git data with provenance and license (each download over
  1 GB needs approval; sizes unmeasured).
- D4 RB3: episodes at real decision boundaries, decisions, candidates, hard negatives, safe
  counterfactuals (`executed=false` when not run), teacher outputs per its own scope, label
  reconciliation (execution first), dedup, repo/issue/commit-disjoint splits.
- D5 benchmark the existing Laya and Cortex checkpoints on the locked set as released,
  against gpt-oss alone. This is the "external Laya" arm and the teacher choice.
- D6 needs G3: tokenizer inspection and marker ablation.
- D7 needs G3 and G5: capture query and option hidden states through our runtime, dataset
  size set from the measured prefill rate.
- D8 RB4: train the head with the backbone frozen, with permutation-consistency training,
  then fit calibration on the calibration split only.
- D9 hard policy, coordinator, conditional specialists, tree-sitter + BM25 + laya-code
  retrieval, memory graph with evidence and supersession.
- D10 evaluation arms: A gpt-oss alone, B gpt-oss + external Laya, C gpt-oss + internal head,
  D C + conditional specialists. Metrics: decision (accuracy, NLL, Brier, ECE, AUROC,
  permutation stability), robustness perturbations, 5 seeded bugs judged by protected tests.
  Goal C > A, then D > C, on held-out tasks. RB5 CI tests must pass.
- D11 DeepSeek Harness plugin on the documented events, local runtime only; decision journal;
  live data goes to an offline corpus and never changes weights.
- D12 release package with every manifest and hash.

**Order against the runtime:** D0â€“D5 need no GPU runtime and can run now. D6â€“D8 wait for G3
and G5. Closed system is defined in CLAUDE.md.

### Closure path for the decision layer, 20b-heretic first (user, 2026-10-01; nothing in the runbook is excluded)

"Closed" = the runbook section 36 contract with `gpt-oss-20b-heretic` as the frozen backbone: locked benchmark hashed;
pinned models; head trained on backbone states with the backbone frozen; calibration a separate artifact; deterministic
hard policy authoritative; specialists conditional, provenance-tracked and task-scoped; outcome journal; offline-gated
data growth; Harness integration by plugin, no fork; reproducible release. The head is backbone-agnostic, so 120B is a
later re-run of the same pipeline. Time figures are estimate (unmeasured).

Teacher and judge lanes (each answers only its own question; outputs recalibrated on our data and stored with provenance):

| Decision | Head | Teacher / judge | Authority above it |
|---|---|---|---|
| ACTION, RECOVERY | primary | no trusted teacher: laya-typed is a baseline only, cortex-1 rejected (section 12 of DECISION_HEAD.md) | hard policy, execution |
| CODE_RELEVANCE | primary | laya-code (independent per-chunk scores, not a distribution) | none |`n| EVIDENCE sufficiency | none | no checkpoint trained for it; evidence-gate and supersession not found; stays unlabelled | none |
| CONTINUE, effort | primary | laya-conductor | completion invariants |
| COMPLETION | primary | laya-stop-completion-judge (shipped state_pack) | tests and acceptance invariants |
| PATCH | primary | none | compiler and tests |
| ESCALATE, ABSTAIN | gate (must beat plain confidence) | conductor effort | policy |
| Safety, injection | none | none; hard rules only | hard rules always |

Two parallel tracks, fastest first:

- **Track A, no backbone needed:** A1 schemas and state/candidate contract; A2 deterministic hard policy and completion
  invariants with tests; A3 one teacher wrapper (lane-scoped, calibrated, provenance) over the Laya models on disk, then
  download `laya-code` and `cortex-1-large`; A4 minimal dataset compiler on R2E-Gym (56 MB): episodes, decisions,
  candidates, real-action negatives, shortcut audit, repository-disjoint splits, locked benchmark hash (RB1 to RB3);
  A5 teacher benchmark on the locked set.
- **Track B, backbone:** B1 capture plumbing on a tiny random `GptOssForCausalLM` (CPU); B2 layer-streaming forward for
  heretic: one BF16 layer is about 1.6 GB of experts, so it fits 6 GB VRAM; slice-equivalence test against transformers'
  own layers; measure tokens/s; B3 pilot capture of 1,000 decisions (about 1 h at 500 tokens/s), layer sweep, span vs
  marker, shared-prefix vs single sequence; B4 train, calibrate, compare with baselines (RB4).
- **Assembly after A and B:** coordinator, retrieval (tree-sitter + BM25 + laya-code), state selector and memory graph,
  journal replay, Harness plugin (local runtime only, not read yet), evaluation matrix, release package (RB5).

First vertical slice to close: ACTION, CONTINUE and COMPLETION on R2E-Gym, then widen to the other kinds. Unknowns that
can stop it: frozen-decoder ceiling, throughput of the streaming loop, no sandbox for counterfactual execution.

**Revised fastest path after the GitHub cross-check (supersedes the A4 dataset choice and adds Tier 1):**
1. A3 teacher wrappers (lane-scoped, calibrated).
2. A4 compiler on a few hundred **SWE-rebench-openhands** rows (has `resolved`), not R2E-Gym alone.
3. A5 locked benchmark and teacher baselines (random, first, prior, TF-IDF, each teacher).
4. Only then touch heretic: capture at about 2/3 depth with a depth sweep; fit Tier-1 closed-form heads per fixed decision (minutes); train Tier-2 pointer head; compare Tier 2 against Tier 1 and the letter readout.
5. Assembly, reusing laya-codex for retrieval.


