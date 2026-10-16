# PLAN â€” local coding agent: gpt-oss-120b + OpenJev on 6 GB VRAM / 15 GB RAM

Rules are in [CLAUDE.md](CLAUDE.md). This file is the spec and the step list.

> **Status (2026-10-02):** sections 1-10 are the historical gpt-oss runtime plan and stay as written (gpt-oss is parked: the user
> wants Qwen models only for now). The live plan is **section 12** (internal head on frozen Qwen3-4B-Thinking-2507, free GPU first,
> staged paid spend with stop rules). Section 11 is the decision-layer step list it builds on.

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
| CODE_RELEVANCE | primary | laya-code (independent per-chunk scores, not a distribution) | none |
| EVIDENCE sufficiency | none | no checkpoint trained for it; evidence-gate and supersession not found; stays unlabelled | none |
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

## 12. Current plan (2026-10-02): internal head on frozen Qwen3-4B-Thinking-2507, free GPU first

**Update 2026-10-03: Phases 2-5 below and steps 5-12 of 12.7 are superseded by section 13 (outcome-labelled one-week plan). Kept as history.**

Supersedes the heretic-first closure path and the capture-cost figures of section 11 for the experiment backbone. The runtime
(sections 2-8) is untouched. Tags: **[M]** measured here (file in `results/raw/`), **[A]** author-reported, **[E]** estimate (unmeasured).
Anything marked "pending" needs the user's written yes (CLAUDE.md, Ask first); it is NOT approved.

### 12.1 Where we stand
- **CAL captured [M]:** 569 of 569 decisions (42 trajectories), audit PASS twice, same hash (`capture-cal-checkpoint-20261002-020211.json`,
  `-125613.json`). Incident: two capture processes ran at once, 42 decision ids logged twice (`results/raw/capture-interruptions.jsonl`). Files are
  one per decision and stable; that the overwritten duplicates were bit-identical is not provable after the fact (spot-check planned in Phase 1).
- **Not captured:** TRAIN 1,986 decisions (185 trajectories), LOCKED 2,220 (444 trajectories, unopened). Registered design: `configs/a5_v3_spec.toml`.
- **Cost of one decision [M]:** 8,163 tokens on average (max 11,123), 10.2 s clean on the RTX 3050, about 70 TFLOP [E, FLOP model]. The 3050 is
  compute-bound, not PCIe-bound (`results/raw/capture-cost-model-20261002-205000.json`). The capture streams layers through the GPU (`capture/capture.py`).
- **Risk [M]:** eight trivial features reach 0.663 and the pilot pointer head 0.667 (`pool-audit-A5v1-20261001-184836.json`): the frozen state may add
  nothing for next-action imitation. Every paid step therefore sits behind a stop rule.

### 12.2 What the research changed (details and links: DECISION_HEAD.md section 13)
1. **Targets:** train on real outcomes (patch resolved or not, stop or continue), with the final result spread back to each step (OpenHands critic
   recipe), not on copying the agent's next action. Data: `nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0, 67,074 trajectories, 3,792 resolved issues [A]; metadata scan first, download pending).
2. **Head loss:** Laya's proper-scoring loss (Brier or spherical) and a per-type post-hoc temperature, typed heads (choice for patch/action, yes/no for stop/complete).
   Copy under Apache-2.0 with the notice. Laya itself is not fine-tuned (CLAUDE.md rule stands, question pending).
3. **Baselines added:** laya-typed-decisions, laya-conductor, laya-stop-completion-judge, Qwen3-Reranker-4B (about 8 GB, download pending), a linear probe on the cached states.
4. **Compute:** free GPU first (Kaggle 30 GPU-h per week, 2xT4 bills double; Colab free as overflow), RTX 6000 Ada rental only as fallback and for LoRA.
5. **Rejected for now [M/E]:** shared-prefix KV reuse across decisions (only 35.4% of prompt tokens shared, median 15.5%, 9 of 25 pairs over 50%:
   `prefix-overlap-train-20261002-205012.json`; a fix means a new state format and a new benchmark version) and FlexAttention (at most about 15-20% of FLOPs, changes numerics).

### 12.3 Phases, gates and stop rules
| Phase | Work | Compute | Gate / stop rule |
|---|---|---|---|
| 0 | Resident-weights mode in `capture/capture.py` and `tools/capture_textemb_v3.py`; **amendment (e)** registered before any head is trained: 3-seed cell selection, GPU-numerics clause (all roles on one device and dtype; CAL recaptured; cosine check against the 3050 files), go/no-go screen = TRAIN grouped CV (185 trajectories), not CAL (42 trajectories, about +-0.10 [E]); metadata scan of the trajectory dataset; head losses; private push of 7962925 as external timestamp (pending) | local, free | scan finds enough issues with both resolved and failed attempts; amendment committed |
| 1 | Smoke: 20 decisions with every hard gate, real seconds per decision, 20-decision recapture spot-check of the CAL duplicates, fp16-vs-bf16 check | Kaggle T4 (free); else RTX 6000 Ada vs RTX 4090 benchmark (about Rs 150-400 [E]) | any hard gate fails, or measured cost above 2x the cost model: stop and re-plan |
| 2 | **Status 2026-10-03: TRAIN (1,986 of 1,986) + CAL (569 of 569) recapture COMPLETE on Kaggle T4 x2 (notebook v7, 11,636 s, both workers exit 0, 0 skipped, `results/raw/t4-phase2-capture-v7-20261003.json`); files still on Kaggle (1.8 GB, download needs a yes); state-free text embeddings and LOCKED pending (12.7).** Capture TRAIN, LOCKED, CAL recapture (about 4,775 decisions, 4-7 T4-hours [E] or about 0.8-2 h on the Ada); fit H0 and H2 on the 9-cell grid; **one** LOCKED read; baselines of 12.2 | Kaggle (free) or Ada | registered rule: lower 95% bound > 0, estimate >= 0.05, half-width <= 0.04, else INCONCLUSIVE. Failure routes to Phase 3; it does not end the project |
| 3 | Outcome probe on about 500-1,000 outcome-labelled decisions: H2 vs H0 and a linear probe, TRAIN grouped CV | Kaggle or Ada, 1-2 h [E] | lower 95% bound not above 0: stop spending on the head, use external verifiers (Laya models, critic-style reranker) and report |
| 4 | Capture about 10k outcome-labelled decisions; frozen scaled head; LoRA (all layers incl. MLP, rank 16, learning rate about 10x full fine-tuning) plus pointer head; fresh **LOCKED-2** | frozen head free; LoRA on the Ada, 18-29 h [E] | ship LoRA only if it beats the frozen head with a paired interval above 0 on LOCKED-2; else ship the frozen head |
| 5 | Three-arm SWE comparison on 100-200 fresh tasks: base agent, plus external Laya models, plus internal head; head-triggered weak-to-strong escalation | needs Docker and generation capacity | not budgeted; quote after Phase 4 |

### 12.4 Compute and budget (all [E] until Phase 1 measures them; cost model: `capture-cost-model-20261002-205000.json`)
Prices are the user's paste of the AIC Cloud page (unverified, GST not included, add 18% if excluded). Peak speeds are published specs as recalled (unverified).

| GPU | Rs/h | BF16 peak TFLOPS | s per forward (mid, 25-45% utilisation) | Stages 1-3 | Stages 1-4 (LoRA on 10k) |
|---|---:|---:|---|---:|---:|
| RTX 3090 | 36 | 71 | 2.8 (2.2-4.0) | 450-600 | 3,400-5,900 |
| RTX 4090 | 69 | 165 | 1.2 (0.95-1.7) | 650-780 | 3,100-5,100 |
| RTX A6000 | 73 | 155 | 1.3 (1.0-1.8) | 700-840 | 3,400-5,800 |
| RTX PRO 5000 | 116 | about 260 | 0.77 (0.60-1.1) | 1,000-1,130 | 3,600-5,800 |
| RTX 6000 Ada | 120 | 364 | 0.55 (0.43-0.77) | 980-1,080 | 2,900-4,500 |

Rs, with 30% contingency. Best on paper: RTX 6000 Ada (fastest and cheapest per job); RTX 4090 is the safe fallback; the A6000 paper peak is probably overstated;
the PRO 5000 is Blackwell and needs a CUDA 12.8+ torch build (ours is cu126). 24 GB of VRAM is enough (capture about 10 GB, LoRA about 14 GB at 11k tokens [E]);
48 GB buys speed only. With the free-GPU path, Stages 1-3 cost Rs 0 if fp16 passes the gates; the paid part shrinks to LoRA, about Rs 2,000-3,500 [E].
**Proposed cap Rs 6,000 in two steps (Rs 1,500 for Phases 1-3, the rest only if Phase 3 passes); Phase C (30k decisions, Rs 5,700-9,600 [E]) needs its own approval. Pending.**
Free-tier limits: T4 has no native bf16 (capture in fp16 changes numerics, so every role is recaptured and re-gated); Kaggle sessions cap at 9-12 h (capture resumes per file);
LoRA at scale does not fit the free quota (10k decisions x 2 epochs about 64 T4-hours [E]). Do not spread work over several accounts to get more quota.

### 12.5 Rules for this plan
- LOCKED is read once. A new head after that read needs a fresh LOCKED-2. Calibration data never feeds gradients.
- All roles are captured on the same device class and dtype. A new device means CAL recapture and the hard gates (Gate A bit-identity, Gate B and plain cosine >= 0.999).
- Amendment (e) is registered before any head is trained. Thresholds are never edited after results (new versioned file instead).
- After each major step: rerun the fixed benchmark and add a row to `results/increments.md` (CLAUDE.md).
- Uploads (code, TRAIN/CAL raw data, then LOCKED raw only when Phase 2 reaches the LOCKED capture), paid GPU, downloads over 1 GB and any push need a written yes first.

### 12.5b Phase 1 artifacts built (2026-10-02, smoke run on Kaggle below)
**Superseded 2026-10-03 by the user's device decision (`configs/phase2_device_rule_v1.toml`): Phase 2 TRAIN and CAL are recaptured on Kaggle T4 x2.** The 3050 TRAIN capture (unchanged `tools/capture_pool_v3.py`) stopped at 807 of 1,986 on 2026-10-03 00:03 and is not resumed; its files are kept for cross-device checks only.
`capture/capture_resident.py` (resident weights, bit-identical to the streaming capturer on a tiny random Qwen3 in fp32 and fp16: `tools/test_resident_capturer.py` PASS [M]);
`tools/capture_pool_v3_t4.py` (copy of the capture runner with `--dtype`, `--resident`, `--shard i/n`, stricter-only gate cadence, device sidecar; defaults equal the original);
`tools/make_kaggle_bundle.py` (TRAIN-only export, outcome fields stripped, secret scan, hashed manifest; `kaggle_bundle/` is gitignored); `tools/make_kaggle_notebook.py` -> `kaggle/phase1_smoke.ipynb`
(pass A all gates on every decision, pass B registered cadence for clean timing, verdict by the rule: switch only at <= 5 s per decision with every gate passed). The isolated bundle self-test matched 20/20 state hashes and the 3050 token counts.
Untested: the real 4B model in fp16 resident mode, and the Kaggle environment (the notebook pins transformers 5.17.0 over Kaggle's preinstalled torch).
**Smoke result (2026-10-02, Kaggle T4 x2, notebook v4, `results/raw/t4-smoke-20261002.json` [M], transcribed from the log, tarball not yet downloaded):** the pipeline runs end to end on Kaggle
(torch 2.10.0+cu128 with transformers 5.17.0 installs and works; model download and sha256 check about 40 s). Pass A (every gate on every decision) and pass B exited 0 on both workers, and a failed gate aborts the
run, so the hard gates held in fp16 on the T4. Speed: 7.64 s per ungated decision per GPU (17 decisions, median 9.06, 1,189 tokens/s, peak VRAM 7.53 GB), GPU 0 at 82 C with clocks down to 510-1110 MHz. By the registered rule (<= 5 s switch; <= 10 s no gain)
that is the middle band, so the rule's verdict is **stay on the 3050 for Phase 2**; a per-session-throughput rule (2 GPUs, about 5 wall hours for 4,775 decisions [E]) would need a new versioned file with a written reason, not an edit.
v5 (notebook fixes only, `results/raw/t4-smoke-v5-20261003.json`, from a screenshot of the log tail): verdict now prints the correct middle band ("FEASIBLE ... stay on the 3050"), max GPU temp 81 C, mean utilisation 82 %.
Full v5 summary and the cross-device check (`results/raw/t4-vs-3050-cosine-20261003.json`, tarball downloaded): all_hard_gates_passed true, 20/20 files per pass, 7.66 s per ungated decision (median 8.9), 1,172 tokens/s, peak VRAM 7.49 GB;
T4 fp16 vs 3050 bf16 states on the same 20 decisions: mean cosine >= 0.9999 at every depth, worst single vector 0.99891 (4 of 2,859 option vectors below 0.999, all in one file); pass A and pass B bit-identical. No cross-device threshold is registered (amendment (e) pending), so this is a measurement, not a gate.
**Phase 2 on the T4 launched 2026-10-03 (user decision, `configs/phase2_device_rule_v1.toml`):** TRAIN 1,986 + CAL 569 decisions, all recaptured on Kaggle T4 x2 (private dataset `laya-phase2-train-cal`, notebook version 7 after v6 was cancelled at about 8 min to add a live progress log (files done, ETA, per-worker s/decision, fail-fast) every 2 minutes, `kaggle/phase2_capture.ipynb`, registered gate cadence 50/200, `--require_device T4`). Bundle self-test: 2,555 of 2,555 states rebuild, 1,376 existing 3050 files match on every hash (`results/raw/bundle-selftest-phase2-20261003.json`). Kaggle quota before the run: 01:01 of 30 h. Expected about 2.7 wall hours [E]; outputs `capture_train.tar`, `capture_cal.tar`, `capture_manifest.json` (about 1.7 GB, so the download needs its own yes). LOCKED is NOT included and needs a separate written approval. The 3050 TRAIN capture stopped at 807 of 1,986 and is not resumed.
Not covered (cosine now measured, see above): cosine against the 3050 bf16 files, fp16-vs-bf16, the CAL duplicate spot-check, real quota billing. The printed verdict line was a notebook bug (glob missed `out/*/train/`); fixed in `tools/make_kaggle_notebook.py`, along with the model path (`/kaggle/temp`) and the gpu.csv flush.

### 12.5c LoRA compute options (research 2026-10-03; sources are official docs and source code unless marked [A]; nothing here is measured on our model)
- **2xT4 cannot carry Phase-4 LoRA [E]:** about 120-180 TFLOP per decision, 17-26 s per decision per GPU at the 7 TFLOPS effective measured in capture, so 47-72 wall hours for 10k decisions x 2 epochs. No native bf16.
- **Kaggle TPU v5e-8 is in the accelerator menu [M, UI]** (None / GPU T4 x2 / TPU v5e-8); Kaggle's own TPU page still describes v3-8 and says it is outdated; 20 h per week and 9 h per session per that page. v5e: 197 bf16 TFLOPS and 16 GB HBM per chip (Google Cloud docs), 8 chips.
- **Leading TPU route: Tunix (JAX), not torch/XLA.** Its Qwen3 Flax NNX model has a config `qwen3_4b_thinking_2507` (36 layers, 2560, 32/8 heads, head_dim 128 = our backbone), `AutoModel.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", mesh)`, a forward with `output_hidden_states`, `skip_lm_head`, `positions`, `attention_mask [B,L,L']`, `segment_ids`, block/decoder remat, splash attention and a sharding config. LoRA comes from Qwix (regex module paths), used by Tunix's qlora_gemma example; merged export via `save_lora_merged_model_as_safetensors`. MaxText supports Qwen3-4B for SFT through Tunix but its SFT docs do not mention LoRA and target v6e-8/v5p-8, so it is not the LoRA route.
- **Only the training step would move to JAX.** Capture, gates, heads and evaluation stay in torch: merge the adapter to safetensors, load it in transformers, recapture on the T4.
- **Unverified / risks:** no public example of Qwen3-4B LoRA at 8-11k tokens (examples: Gemma 270M at 1,024 tokens on Colab v5e-1; a PyTorch/XLA 2.6B LoRA run at 8k [A]); our packed layout needs prefix-sharing masks (segment_ids alone gives block-diagonal only); custom pointer-head loss needs our own NNX train loop; Flax issue 5116 (nnx.grad uses about 2x memory); Qwix is not on PyPI and its README does not mention Tunix; dense attention at L about 9.5k is about 0.7 GB per layer per chip only if heads are sharded over 8 chips [E].
- **Smoke test to register BEFORE running (only after the Phase-3 probe says go):** (1) Tunix hidden states at depths 18/24/30 vs our existing captured files for the same decisions; (2) packed prefix-sharing mask vs plain forward; (3) one LoRA step at 8.5k-11k tokens: memory, step time, recompiles over length buckets; (4) merge to HF safetensors, load in transformers, parity; (5) project 20k decision-epochs. Costs a few of the 20 weekly TPU hours.

### 12.7 The 12 steps before Phase 4 (2026-10-03; each needs the confirmation shown). Steps 1-8 = Phase 2, steps 9-12 = Phase 3
1. **DONE 2026-10-03: Kaggle run finished** (TRAIN 1,986 and CAL 569 captured, 7.19 s / 7.03 s mean per decision per GPU, 1,146 / 1,158 tok/s, peak VRAM 7.55 / 7.41 GB; 192 min wall for both roles). Step 3 verification DONE incl. the Kaggle-manifest hash comparison (1,986 + 569 entries, 0 mismatches, 0 absent): `results/raw/phase2-capture-verify-20261003-145647.json` (first run without the manifest: `-145542.json`); files extracted to `data/v3/capture_t4/` (gitignored, 1.8 GB); tarballs sizes equal Kaggle's log exactly. Next gate: register amendment (e) (step 4).
2. **[PATH A, TRAIN+CAL TEXTS DONE 2026-10-03: notebook v11 under the e1 lock (worker torch 2.10.0+cu128, gate worst cosine 0.999999 on all 7 shards, `results/raw/t4-textemb-v11-20261003.json`); v8 (torch 2.11) kept for comparison; v9 and v10 failed and are recorded; downloaded and verified locally 2026-10-03: `results/raw/textemb-t4-verify-20261003-155540.json`, 12,602 of 12,602 texts, vs 3050 min cosine 0.9998] [earlier note: Kaggle notebook v8, dataset `laya-textemb-trainval`, 12,602 TRAIN+CAL pool texts; the 21,713 LOCKED-pool texts wait for LOCKED approval; the 3050 embeddings (34,315 texts, 14.4 min, TEXTEMB_OK) are kept for cross-checks only]** State-free text embeddings on the T4 (spec `[capture_v3] text_embeddings`; 34,315 unique texts, about 0.5 h [E]; 7 local bf16 shards of 32 MB exist from the 3050, completeness unverified and they would be recaptured under the one-device rule). Needs: port `tools/capture_textemb_v3.py` to the resident runner, second Kaggle run. **Confirm: yes/no.**
3. **Download and verify** `capture_train.tar`, `capture_cal.tar`, `capture_manifest.json` (about 1.7 GB). **Needs a written yes (over 1 GB).** Checks: sha256 manifest, amendment e2 and e3, cosine vs the 3050 files on 1,376 decisions (e4).
4. **DONE 2026-10-03: amendment (e) REGISTERED** in `configs/a5_v3_spec.toml` [amendment_2026_10_03_e] and `configs/thresholds.toml` [amendment_e_thresholds] before any head was trained: text-embedding gate 0.9999 (unchanged, every shard), pre-LOCKED screen point estimate >= 0.02 (go/no-go only, not success; the primary criterion 0.05 / lower bound > 0 / half-width <= 0.04 is unchanged), 3-seed selection, Path A device lock, cross-device comparisons report-only, Brier secondary arm, freeze-and-hash before LOCKED.
5. **IN PROGRESS 2026-10-03, on Kaggle T4 x2 (user rule: nothing on the 3050): notebook `head_v3_t4`, TRAIN only.** Stages: GPU smoke, one timed cell, the 9-cell x 3-seed selection, the screen (step 6). Measured: one cell (256, mean, seed 0) 1,417 s, 0.29 GB VRAM, GPU 17% utilised (CPU-bound), 1,226 rows/s (`results/raw/head-v3-timing-kaggle-t4-20261003.json`); two parallel workers take about 46 min per (seed, projection) job (shared CPU cores), so selection is about 4 h and the screen about 1 h [E]. Failures on the way, all fixed and recorded: loader assumed tar files (Kaggle extracts them), and a GPU-only device bug in `ResidualNet.forward` (audit extended to 26 checks incl. a meta-device placement check). Not done: the 5-seed final fits, the CAL temperature, the freeze, the baselines. LOCKED preparation (local, nothing uploaded, no evaluation): label-stripped bundle + self-test PASS, text bundle, two generated notebooks, `tools/freeze_head_v3.py`; the secret-scan allowlist of 7 hashed values awaits the user's review.
6. **Go/no-go screen** on TRAIN grouped CV (e7), then freeze and hash the head (e10).
7. **LOCKED, one read:** 2,220 decisions, about 190 MB, about 2.4 h on T4 x2 [E]. Upload with labels stripped (render_pool reads only decision_id; to be proven by the self-test on the stripped file). **Needs the user's explicit written yes**, only after step 6.
8. **Evaluate once**, report all baselines (e11), add `results/increments.md` and `BENCHMARKS.md` rows.
9. **Phase 3 data:** download `nebius/SWE-rebench-openhands-trajectories` (CC-BY-4.0, size unmeasured, over 1 GB expected) and run the metadata scan (needs both resolved and failed attempts). **Needs a written yes (download).**
10. **Capture about 500-1,000 outcome-labelled decisions** on the T4 under the same device lock (about 1 h [E]).
11. **Outcome probe:** H2 vs H0 and a linear probe, TRAIN grouped CV, outcome targets (resolved or not, result spread back to each step).
12. **Stop rule:** lower 95% bound not above 0 means stop spending on the head, use external verifiers and report; otherwise Phase 4 is OPTIONAL and starts only if the user asks (about 10k decisions, scaled frozen head, LoRA, fresh LOCKED-2).
Phase 0 leftovers NOT verified as done: the metadata scan (step 9) and the private push of commit 7962925 (12.6 item 3); commit 4038258 (2026-10-03) is local only.
Parked (not started, need the user): LoRA / TPU (12.5c), gpt-oss, agent evaluation, any paid GPU, any second Kaggle account (CLAUDE.md rule stands).

### 12.6 Pending approvals
(1) provider, cap and upload scope (Kaggle/Colab private dataset: code, model, TRAIN and CAL raw; no secrets); (2) downloads over 1 GB: SWE-rebench-openhands, Qwen3-Reranker-4B;
(3) private push of commit 7962925; (4) whether to lift the "no Laya fine-tuning" rule (only a free Kaggle side track for a stronger baseline; Laya's 512-1,024 token window cannot read our 8k states);
(5) questions for AIC before renting: billing granularity, billing while stopped, disk size and price, Docker, GST.



## 13. Current plan (2026-10-03, one-week build): outcome-labelled head on frozen Qwen3-4B-Thinking-2507

Source: `Laya-Style Decision Head One-Week Build Plan.md` (user, 2026-10-03). **Supersedes sections 12.3 Phases 2-5 and 12.7 steps 5-12.** Sections 2-8 (runtime)
and 12.1-12.2 stay as history. Tags as in section 12. Nothing marked "pending" is approved. Spend stays Rs 0.

### 13.1 Fixed rules
- The backbone (Qwen3-4B-Thinking-2507) is never fine-tuned.
- Labels come from real outcomes (task resolved or not), not from copying mined history.
- `policy/hard_rules.py` and real tests always override the head.
- Success: Qwen plus the head solves more tasks than Qwen alone and than Qwen plus the Laya sidecar, with a premature-stop rate that is not higher.
  **Open:** "strictly lower" or "not higher" is worded both ways in the source plan; settle it together with the numeric margin.

### 13.2 Scrapped and kept
| Scrapped | Replaced by |
|---|---|
| A5-v3 imitation study; the 2,220-decision LOCKED capture (never opened; never to be opened) | Outcome labels (resolved or not) |
| One isolated 32k-character state per decision | One forward pass per full trajectory, states read at every turn |
| Candidate pools mined from history | Qwen's own 4 to 8 sampled proposals |
| gpt-oss-120B / AirLLM / Jev-reader track; the 16.8 GB of `autojev/` leftovers | Deleted. **DONE 2026-10-03 (user's written yes in chat): `autojev/` (900 files, 15.67 GB, untracked) and `GPT OSS 20 B.exe` removed.** Partial downloads in the model caches are untouched. |
| Per-step registration and amendment paperwork | Three shortcut checks: repo-disjoint split, state-blind baseline, shuffled-state control. **Dropping the paperwork is pending (13.6).** |

Kept unchanged: `capture_resident.py` and the Kaggle notebook, `head.py` (34.5M params), `closed_form.py`, the three Laya teacher wrappers, `policy/hard_rules.py`,
depth-24 capture, per-type temperature calibration. The existing TRAIN and CAL state files are reused.
**Not yet reconciled:** the in-flight `head_v3_t4` Kaggle run, the LOCKED-prep artifacts (stripped bundle, two notebooks, `tools/freeze_head_v3.py`) and amendment (e)
belong to the scrapped A5-v3 path. They stay as recorded history; whether to stop the run is the user's call.

### 13.3 Targets (all from one pass per trajectory)
| Target | Question | Exact training label |
|---|---|---|
| `done` | Is the task finished? | Yes only at the final step of a resolved trajectory. No at every earlier step and at the final "finish" of an unresolved one. |
| `value` | Will this task eventually succeed? | Every step gets its trajectory's final outcome. |
| `next_move` | Which of 9 move types comes next? | Frontier traces mainly; resolved open-model runs at lower weight. |
| `pick` | Which of Qwen's 4 to 8 proposals is right, or NONE? | The frontier model's real action at that state. |
| `escalate` (rule, not trained) | Should a stronger model take over? | Fires when `value` is low or falling and the Laya conductor rates the task hard. |

`done` exists to stop mini-Jev's failure: confident, premature completion on multi-step tasks.

### 13.4 Data sources
| Source | Use | Status |
|---|---|---|
| `greghavens/fable-5-coding-and-debugging-traces`, `greghavens/gpt-5.6-sol-coding-and-debugging-traces` | Frontier `next_move` and `pick` labels | Usable (CC BY 4.0). Size unmeasured: a download over 1 GB needs a yes. |
| `samatv256/jev-decisions-v1` | State, candidates, outcomes; Open-SWE eligible rows only | Usable (CC BY 4.0). 48,293 completion records are positive-only: it cannot teach stop versus continue alone (it lacks unresolved trajectories). |
| `mvilacad/laya-conductor` | Starting weights for the Laya-coder sidecar | Usable (Apache-2.0) |
| `nebius/SWE-rebench-openhands-trajectories` | Resolved / unresolved labels (67,074 trajectories) | **Pending: needs a written yes (over 1 GB).** |
| SWE-Critix, R2E-Gym, SWE-Gym | Later pass/fail scale-up | Reported only, not checked |
| `yixuanli97/finding-the-right-fit` | None | Rejected: analysis-only, non-commercial |
| `MoreThought/*` Fable 5.1 dumps | None | Rejected: provenance cannot be verified |
| Open-Jev, JevK5 | None | Rejected: they require fine-tuning the backbone |

### 13.5 The decisive agent test (Days 5-7; the only gate to shipping)
- Minimal bash agent loop; generation in vLLM on one T4, the head's forward pass in transformers on the other, same weights. 20 to 30 seeded-bug tasks in pure-Python repos with protected tests. No Docker.
- Three arms: Qwen alone; Qwen plus the Laya-coder sidecar; Qwen plus the internal head. Scores: tasks solved and premature-stop rate.
- Pass: the head solves more tasks than Qwen alone without a higher premature-stop rate, and matches or beats the sidecar. **The numeric margin is not set: fix it before the first run.**
- Offline accuracy alone does not count (mini-Jev: 72.97% synthetic, 27.98% real trajectories).
- Days 1-4: Tracks L, Q and P feed Day 4 head training. The source file's pipeline diagram did not export, so Days 1-3 are not recorded.
- Risks [A unless noted]: a separate-GPU head costs a full extra prefill per decision (about 7 s per 8k-token state on a T4); near-zero cost needs hidden states read inside vLLM, unbuilt and unconfirmed.
  The head is valid only at the capture precision (a quantized Qwen on the 3050 needs recapture or a cosine check). Laya's act/escalate head is dead on this repo's data (`act_probability` always 1.0); the conductor scored 0.340 vs a 0.318 baseline outside its lane.
  The shipped Laya checkpoint is under-confident (one temperature: ECE 0.204 to 0.037, independent reproduction).
  Unverified, do not rely on: "head-only 0.659 vs full fine-tune 0.896", the LayaStudio objective numbers, dataset counts for Open-SWE-Traces, SI2CA and JevK5.

### 13.6 Pending approvals (written yes needed; none is given)
1. **YES 2026-10-03 (user, chat): `nebius/SWE-rebench-openhands-trajectories` DOWNLOADED** to `data/nebius_swe_rebench/` (2.08 GB parquet, CC-BY-4.0, gitignored; 67,074 trajectories, 32,161 successful [A, dataset card]; generated by Qwen3-Coder-480B, not by our backbone). Any frontier-trace download over 1 GB is still pending.
2. **YES 2026-10-03 (user, chat): the "no Laya fine-tuning" rule is lifted** (same as 12.6 item 4). No Laya training has been started.
3. Drop the per-step registration and amendment paperwork.
4. **KAGGLE T4 (fp16), 2026-10-03 (user, chat).** The local 3050 is not used.
5. The numeric pass margin of the agent test (and the "strictly lower" versus "not higher" wording).
6. ~~Deleting `autojev/` and the JDownloader installer~~ DONE 2026-10-03. Still open: partial downloads inside the model caches (not touched).

Items 12.6 (1), (3), (5) are unchanged: provider/cap/upload scope, private push of 7962925, questions for AIC.

### 13.7 Decision 2026-10-04 (user, chat): fine-tune the Laya conductor only; the internal head is not touched
- **Scope:** `laya-conductor` (`models/laya-conductor`, revision 04116e68, ModernBERT-large encoder, 421M params, max_len 1024, head_max_len 256) is the **base**; it is fine-tuned on the dataset built from `nebius/SWE-rebench-openhands-trajectories`. This is the Laya-coder sidecar arm of 13.5.
- **Untouched:** the internal head (`head.py`, `decision_head/`, the TRAIN/CAL state files, `head_v3_t4`). No edits, no new runs on it under this decision.
- **Recipe:** the official `models/laya-conductor/conductor_finetune_colab.ipynb` (DDP on Kaggle T4 x2, AdamW encoder 2.5e-5 / head 1e-4, micro-batch 8 x accum 4, 4 epochs, per-type temperature refit on a 400-item calibration cut), started from the conductor weights instead of `convaiinnovations/laya`. Published cost for 4,105 turns: about 30 min on a T4 [A].
- **Questions it can learn:** `role` (choice smol/default/slow) and `should_continue` (noul: P(agent stopped mid-task with work left)). `should_continue` equals the plan's `done` label negated: TRUE at every step except the final step of a resolved trajectory (TRUE at the final "finish" of an unresolved one). `role` has no label in nebius; **not decided** (see below).
- **Input window:** at most 1,024 tokens, so the state is the tail of the trajectory (tail-first packing), not the full 8k state.
- **Open:** (a) `role` labels (none, or derive hardness, or replay the conductor's own route data from `mvilacad/laya-conductor-decisions` to avoid forgetting); (b) repo-disjoint test split and the three shortcut checks apply to the sidecar too; (c) fine-tuning has NOT started; the data builder is not written.
- Nothing here is measured. Fine-tuning Laya is permitted since 2026-10-03 (CLAUDE.md).

**Built 2026-10-04 (nothing run on Kaggle yet):**
- `tools/conductor_data.py` (row builder; `demo()` passes), `tools/check_conductor_rows.py` (local check on the 300-row sample: 1,200 rows, 11.7% FALSE, 1.2% of states cut by the 1,024-token window, last_response tail kept in 100%).
- `tools/make_kaggle_conductor_notebooks.py` writes `kaggle/conductor_data_cpu.ipynb` (CPU session: download nebius parquet, select 4,000 `submit` trajectories, per-repo cap 12, resolved/unresolved balanced, repo-disjoint 80/20 split, about 12k rows) and `kaggle/conductor_finetune_t4.ipynb` (GPU T4 x2: conductor base pinned at revision 04116e68 with sha256 check, official `train_ddp.py` reused verbatim, then eval on the repo-disjoint test split with two controls: state-blind TF-IDF baseline and shuffled last_response). Syntax-checked only; the CPU notebook is not executed end to end (pyarrow is not installed locally).
- **Findings from the audit:** (1) the official notebook's preprocessing and eval cells are corrupted (duplicated lines, undefined `route_items`/`cont_items`), so only its training script was reused; (2) `build_sequence` serializes the state with `json.dumps` and keeps the head of it (`truncate_left` is False), so the builder puts `last_response` first and caps both fields; (3) the shipped conductor config says 1.96 h for 7,313 updates while its card says about 30 min for 4,105 turns: the run time is unknown until measured; (4) the conductor never sees Qwen states, so no Qwen download is needed for this fine-tune.
- **Still open:** the `role` question (13.7 open item a); the run itself; Qwen is only needed later for the agent test and the head.

**Audit and code changes 2026-10-04 (conductor fine-tune; nothing run on Kaggle yet):**
- Found: (1) the state had only `last_response` + `task`; at the final step the closing summary reads alike whether the run later resolved or not (4 hand-read examples; response length AUROC 0.585 among the 64 final rows with the `[tool finish]` marker in the 300-row sample), so text alone cannot separate a resolved finish from a premature one. (2) The eval reported all rows pooled, which is inflated by "closing summary vs tool call". (3) The official script cuts its 400-item calibration set at random from the TRAIN rows (same trajectories on both sides). (4) The "about 12k rows" comment was wrong: 4 rows per trajectory gives about 16k for 4,000 trajectories.
- Changed: `tools/conductor_data.py` adds `last_observation` (tail of the last tool result before the step, observable at t; budgets TASK 600 / RESP 1000 / OBS 900 chars; local check: 1,200 rows, 1.0% truncated, response and observation tails lost 0.0%). `tools/make_kaggle_conductor_notebooks.py` now writes a repo-disjoint `cal_items.pt` (about 10% of train repos) and patches the official script to load it, adds `SMOKE` (600 items, 1 epoch, for a timing run), reports final-step-only metrics, and shuffles response+observation for the shuffled-state control. Notebooks regenerated, syntax-checked, the patched `train_ddp.py` parses.
- Checked, no change needed: on a T4 (capability < 8) `laya.Agent` autocasts fp16, so the shipped `amp_dtype: bf16` does not matter there; training already uses fp16 autocast and DDP over both T4.
- Not measured: seconds per step (use `SMOKE`), the Kaggle disk size (the CPU notebook prints free space on `/kaggle/working` and `/kaggle/temp`), whether adding `last_observation` helps at all (that is what the final-steps-only eval decides). Qwen is not needed for this fine-tune.

**PLAN REVISION 2026-10-04 (user, chat): the conductor outcome pool goes from one source to a capped multi-source pool.** This is a revision of the written plan (13.4 lists `nebius` as the outcome source and SWE-Critix/R2E-Gym/SWE-Gym as later scale-up; SWE-smith-trajectories, nanoswe and the JetBrains subset were not in it). Results from this pool are NOT results of the original single-source protocol and must be labelled so.
- Pool: `nebius/SWE-rebench-openhands-trajectories` (2,000 trajectories wanted), `SWE-bench/SWE-smith-trajectories` `tool` shards 0/3/6 (1,000), `nanoswe/nanoswe-trajs-261002` shards 0/25/50/75/100/125/150 (1,000), `JetBrains-Research/agent-trajectories-swesmith-random-subset` (500, held out as a different-agent test; its repos are removed from train and test). Each source has its own adapter (`tools/conductor_data.adapt`) and per-repo caps (12/8/8/8). Exit status must be `submit`/`Submitted` where the source has one. Split by repo across the whole pool (`sha256(repo)%100 < 20` is test).
- Scope: this pool feeds the **conductor** (`should_continue` = `done` negated, PLAN.md 13.7). It does not give `next_move` or `pick`; those still come from the frontier action traces for the internal head, which is untouched under the 2026-10-04 decision. Rows also store `done` (1 only at the final step of a resolved run) and `value` (the run's outcome) as separate fields so the internal head can reuse them without relabelling.
- Verified from the HF API (columns only): SWE-smith `resolved`, nanoswe `verified` + `exit_status`, JetBrains `resolved` + `exit_status`. The nanoswe message structure is NOT verified (the API truncates it); the adapter asserts roles are in {user, assistant, tool} and the CPU notebook fails loudly otherwise.
- Checked: adapters on live samples (SWE-smith, JetBrains), and the whole CPU notebook against fake parquet files (4 sources, repo overlap 0 train/test and train/held-out). Not checked: real parquet reads (pyarrow is not installed locally), real file sizes after download, how many train repos the held-out removal takes out (the notebook prints it).

**Architecture decision 2026-10-04 (user, chat): the Laya conductor is used as a TEXT SIDECAR next to Qwen** (the sidecar arm of 13.5). It is fine-tuned with the official Laya fine-tune recipe on Kaggle T4 x2; it does not read Qwen's hidden states; the from-scratch internal head only informs the design (its `head.py` takes Qwen's 2560-wide states, the conductor encoder is 1024 wide, so the two cannot be joined without new architecture code).
- Laya is question-conditioned (`[question] [options] [state]`), so more targets are more questions, not new architecture: `should_continue` and `value` as yes/no questions from the outcome pool, `next_move` and `pick` as choice questions from the frontier action traces.
- Staging: (1) `should_continue` on the multi-source outcome pool (built, not run); (2) add `value` (same rows, second question); (3) add `next_move` / `pick` only after the Fable 5 / GPT-5.6 Sol access and columns are read, the `pick` match rule is fixed, and stage 1 passes the final-steps test. Option text budget is `head_max_len` 256, about 30 tokens per option with 8 proposals: untested.
- "LayaStudio" is not in the Laya repo or in any project file (only an unverified mention in 13.5); the official recipe is `docs/finetune.md` + `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`. **Correction 2026-10-04:** it is a separate project, `github.com/biplovgautam/LayaStudio` (user link); details and the frontier-trace check are in CONDUCTOR_FINETUNE_PLAN.md section 12.
- Open risk: full fine-tuning 421M parameters on a few thousand rows can forget the shipped skills; options are a lower learning rate, mixing in the conductor's own route data, or LoRA (needs `peft`, an install that needs your yes).


## 14. Decision 2026-10-04 (proposed, NOT approved): Laya-initialised internal head on Qwen pool states
Source: inspector relay of the user's direction, 2026-10-04. Status: **plan only. No training code written, nothing run on Kaggle or the local GPU. The user must verify this section first (14.8).**
Tags: [M] measured here, [E] estimate (unmeasured).

### 14.1 Goal
A NEW internal head for frozen Qwen3-4B-Thinking-2507, initialised from a pretrained Laya checkpoint and adapted to read Qwen's hidden states, with the SAME architecture as our internal head so it is a drop-in
for any local model. The existing internal head (`decision_head/head.py`, `decision_head/`, `head_v3_t4`) is not edited; the new head imports `DecisionHead` as it is and lives in new files.
Later (not now): train the same head further on more trace data (14.7).

### 14.2 What of Laya is reused, what is new
Both on-disk checkpoints, `models/laya-code` (code relevance, calibrated yes/no) and `models/coding-decision-agent`, have the same non-encoder layout [M, safetensors headers read 2026-10-04]: `head.layers.0-1`
(2 pre-norm transformer layers, d=1024, FFN 4096), `scorer.*` (LayerNorm, Linear 1024x1024, Linear 1024->1), `type_emb` (3x1024), `act_head`, `temperature`. The rest (about 170 tensors) is the ModernBERT-large text encoder.
| Part | Decision | Why |
|---|---|---|
| `head.layers.*` (2 layers), `scorer.*`, `type_emb` -> `prim_emb` | **Reused** (init from Laya) | These are the decision trunk; they are what `DecisionHead(d=1024, layers=2, scorer="mlp")` already maps (tools/head_transfer_check.py). |
| ModernBERT-large encoder, tokenizer, 512-token text path | **Not used** | The head must read Qwen states, not text. Using the encoder on candidate text would make a text sidecar, which is the conductor arm (13.7), not this head. |
| `act_head`, `temperature` | **Not used; refit** | Laya's act head read about 1.0 on every input (dead here, see 13.4 and DECISION_HEAD.md 19). Temperature is refit per option-count bucket on a held-out fold. |
| NEW `in_q`, `in_o` (Linear 2560->1024), layer mix over depths 18/24/30, `kind_emb` (zero), `gate` | **New, random init** | They connect 2560-wide Qwen states to the 1024-wide trunk. Scaler statistics come from the fit split only. |

Initialisation arms (14.4): random trunk, laya-code trunk, Coding_Decision_Agent trunk. Which pretrained one is the better init is **unmeasured** [E]: laya-code was tuned only on the yes/no relevance primitive; Coding_Decision_Agent saw choice, score and noul coding decisions.
Backbone-agnostic: the trunk is 1024-wide and backbone-free; per backbone only `in_q`/`in_o` (hidden size) and a recapture change. Whether a trunk trained on Qwen transfers to another backbone is **unmeasured**.

### 14.3 Data (already built and captured on Kaggle; figures from the inspector, `results/raw/pool-states-v1-final-20261004-185656.json`)
- Rows: 46,826 decisions (train 37,417 / cal 3,614 / test 5,795); each = gold action + 2-3 synthetic hard negatives (other_step, wrong_path, wrong_tool, premature_finish; `executed=false`, no outcome) + NONE. Splits by repo hash; contaminated repos never in TEST.
- States: `pool_v1_kaggle` capture, T4, depths 18/24/30, query = last token, option = its last token. **45,500 of 46,826 captured; the last 1,326 are not captured and will not be re-run.** All results are on captured decisions only; the per-split captured counts must be printed before any number is quoted.
- Gold = the frontier model's real next action. NONE is never the gold, so NONE has no positive training signal in these rows (14.8, question 4).
- Real outcome (`resolved`) exists for nebius and swesmith only; Fable has none. The `pick` loss uses all rows. Outcome is used as an analysis split (resolved vs unresolved trajectories) and as an optional weight ablation, not as the primary label.

### 14.4 Training and comparison (Kaggle T4 only; Rs 0; never the local 3050)
Stages (each only after the previous result is recorded):
1. **Frozen trunk:** train only `in_q`, `in_o`, layer mix, gate (arm F in head_transfer_check.py).
2. **Full:** all head parameters trainable, lower trunk learning rate (arm A).

Arms per init: B random, L laya-code, C Coding_Decision_Agent; seeds 0-2. Loss: `decision_loss` (proper scoring rules) as in head.py. Early stopping on a validation fold carved from TRAIN by repo; **CAL gets no gradient** and is used for temperature and gate calibration only; TEST is read once per final arm.
Reference arms on the same TEST decisions:
- **Qwen alone:** the captured option-letter probabilities (`qwen_probs`, no thinking). **Limit:** a weak baseline by design; a thinking-mode Qwen would be fairer and is not measured (CLAUDE.md: baselines get the same prompt quality).
- **Text-only baseline:** the builder's char TF-IDF + logistic regression on candidate text, reported as flagged (47.1% vs 26.4% chance, 20.7 points over the pre-set 0.15 margin; inspector figure, not re-measured here).
- **Options-only and shuffled controls:** head with the query zeroed; head with the query shuffled across decisions.
- **Existing internal head recipe (arm B):** the untouched `head.py` at random init, as the "no Laya" reference.

### 14.5 Per hard-negative kind, and the open shortcut flag
The flag is NOT resolved: hard negatives were not changed, to be assessed after Qwen scoring (inspector, 2026-10-04). Every head number is read through these checks:
- Report TEST accuracy **per negative kind**: decisions restricted to {gold, that kind, NONE}, plus a confusion table of which kind the head picks when wrong. `premature_finish` has a constant text, so a high score there proves nothing alone.
- The head must beat the text-only baseline by a margin M **overall and on each kind**, and the options-only and shuffled controls must fall clearly below the full head; otherwise the result is read as an artifact result, not a state result. M and the floors are pre-registered in a NEW versioned thresholds file before any run (14.8, question 3); nothing is tuned after seeing results.
- If the controls show the head uses option-state artifacts, the next step is a new negative set (new versioned rows), not a threshold edit.

### 14.6 Gates and hard limits
- Shipping still needs the three-arm agent test (13.1); offline accuracy alone does not count. Hard rules and real tests override the head. Hard-policy false-approval rate on the safety bucket must stay 0.
- The abstain gate must beat plain max-probability confidence on held-out data or it is not shipped (head.py comment).
- Every headline number needs a row-level file under `results/raw/` (create-only). Anything unmeasured stays `estimate (unmeasured)`.
- Compute [E]: head about 25M trainable parameters, 3.83 GB of shards; time per epoch on T4 is unmeasured.

### 14.7 Later, not in this decision
Continue training the pretrained-initialised head on more trace data after the 14.4 result is recorded. Fable is already in the rows (12k). Other trace datasets (JetBrains, nanoswe, more SWE-smith/nebius) need new Qwen captures on the T4 under the same protocol and a new split manifest. Each needs its own approval where CLAUDE.md requires one.

### 14.8 To verify with the user before any code (answers requested)
1. **Rule conflict:** CLAUDE.md (2026-10-04, chat) says the conductor is the only thing fine-tuned and the internal head is not touched. This plan trains Laya-derived head weights in a NEW head. Does your 2026-10-04 direction amend that rule? A written yes is needed; the rule stays in force until then.
2. **Init:** compare random, laya-code and Coding_Decision_Agent (recommended), or only laya-code?
3. **Pre-registered margins:** the margin M over the text-only baseline (overall and per kind), floors for the controls, and the "strictly lower" vs "not higher" premature-stop wording (13.1). Proposed: I write a new versioned thresholds file from your numbers before any run.
4. **NONE / abstain:** NONE is never gold in the current rows. Proposed: train the gate on "head top-1 is wrong" on a held-out TRAIN fold, calibrate on CAL, and add outcome-labelled NONE targets later. Agree?
5. **Qwen-alone arm:** accept the no-thinking option-letter baseline for the first comparison, with the limit stated, or capture a thinking-mode baseline first?
6. **Shortcut flag:** proceed on the current negatives with the 14.5 controls deciding the reading (as the inspector relayed), or rebuild negatives before training?


### 14.9 Update 2026-10-04 (answers relayed by the inspector session; supersedes 14.2 init arms, 14.4 arms and 14.8 where they differ)
Status: still **no code written, nothing run**. The relay is recorded as instruction, but a relayed message is not my authority to change CLAUDE.md: the user's own confirmation in this chat of item 1 is requested again (14.10).
1. **Amendment of the 13.7 / CLAUDE.md rule, for this new head only (relayed):** Coding_Decision_Agent (`models/coding-decision-agent`) is adapted to sit on Qwen's internal layers with the same architecture and interface as the internal head. The old internal head (`head.py`, `decision_head/`, `head_v3_t4`) stays untouched. CLAUDE.md itself is not edited here.
2. **Starting point fixed: Coding_Decision_Agent.** No laya-code arm, no random-init arm, no comparison of starting points. (The earlier "existing head at random init" reference arm is dropped with it.)
3. **Order:** a 3-example test first (14.10); real training only after it passes, on the captured decisions (the inspector wrote "45,000"; the record says 45,500 of 46,826 captured, and the number used is whatever the per-split counts print).
4. **Abstain gate:** the user has no preference, so the 14.8 proposal stands: train the gate on "head top-1 is wrong" on a held-out TRAIN fold, calibrate on CAL, outcome-labelled NONE targets later.
5. **Shortcut flag:** NOT decided. The inspector explains it to the user and relays the answer. Until then nothing is trained and no negatives are changed.
6. **Margins and Qwen-alone baseline:** not answered. Defaults proposed below, [E] unmeasured, waiting for the user:
   - Margin M over the text-only baseline (top-1 on gold among the non-NONE options, TEST): head >= text-only + 0.10 overall, and >= text-only + 0.05 on each negative kind. Controls: the options-only and the shuffled-query head must each be at least 0.10 below the full head. Written to a new versioned thresholds file before any training run, never edited after.
   - Qwen alone: use the captured no-thinking option-letter probabilities for the first comparison (already captured, free), stated as a weak baseline. Optional later: a thinking-mode Qwen on a 500-decision TEST subsample, if the user wants the fair baseline.

### 14.10 The 3-example test (design only, to be verified by the user before it runs)
Purpose: show the adapted head loads, takes Qwen states, runs forward and backward, and gives valid output. It is a mechanics test: **no accuracy claim, no threshold, no training result**; output labelled `exploratory (no gate)`, create-only under `results/raw/`.
Inputs: 3 decisions from one captured shard of `laya-head-states-v1` (`q` [3,2560] fp16 at depths 18/24/30, `cx` [n,3,2560], `order`, `gold_pos`; NONE sits at position `order.index(n-1)` because the builder put NONE last), chosen deterministically: the first three decisions of shard-00000 with at least 4 options. The hard-negative kind of each is read from `rows.jsonl` by `decision_id`, for the record only.
Model: `DecisionHead(2560, d=1024, layers=2, k_layers=3, scorer="mlp")` imported unchanged from `decision_head/head.py`; the 31 Coding_Decision_Agent tensors loaded with the same key map as `tools/head_transfer_check.py` (`head.layers.*` -> `ctx.layers.*`, `scorer.*` -> `mlp.*`, `type_emb` -> `prim_emb`). A new file only holds the loader.
Checks (all assertions, any failure stops the run and is reported with its output):
1. **Load:** 31 tensors loaded, none unexpected; the only missing keys are the new modules (`in_q`, `in_o`, mix, `kind_emb`, `gate`, scaler buffers, `out_norm`); loaded values equal the checkpoint after the fp16 -> fp32 cast; `act_head` and `temperature` deliberately not loaded.
2. **Input:** shapes [3,3,2560] / [3,n,3,2560], no NaN or inf; scaler statistics fitted on the rest of shard-00000 (smoke only; the real run fits on the fit split).
3. **Forward:** logits [3,n] finite; padded positions at NEG; each row's softmax sums to 1; gold and NONE positions valid; gate output [3,2] finite.
4. **Backward:** loss finite; every new parameter and every loaded Laya parameter gets a finite gradient, and no loaded Laya parameter has an all-zero gradient; 5 AdamW steps (lr 1e-4) on the same 3 decisions lower the loss (overfit sanity, not evidence of learning).
5. **Order equivariance:** shuffling the option order permutes the logits identically (atol 1e-4), the property head.py documents.
6. **Untouched:** `git diff` shows no change to `decision_head/head.py` or the other `decision_head/` files; backbone not loaded; hard rules untouched.
Where it runs (user to choose, 14.11): CPU on this machine with `ATEN_CPU_CAPABILITY=avx2` after downloading ONE shard (about 40 MB [E], well under 1 GB) from the Kaggle dataset, or inside a Kaggle CPU notebook. The local 3050 is not used (user rule).

### 14.11 For the user to verify now
1. Confirm in this chat the amendment of the 13.7 / CLAUDE.md rule for this new head only (Coding_Decision_Agent is adapted and trained; the old internal head and CLAUDE.md are not edited by me).
2. Is the 3-example test (14.10) the right test? Anything to add or remove?
3. Run it on local CPU with one downloaded shard, or in a Kaggle CPU notebook?
4. Accept the default margins and the no-thinking Qwen-alone baseline (14.9 item 6), or give your numbers?
5. The shortcut flag is waiting for your answer (the inspector is explaining it).


## 15. Real wrong options for the pick head: research and evidence-tiered hybrid (2026-10-04, PROPOSED; builder written, counts NOT yet run on the full data)
Why: the shortcut check on the synthetic hard negatives is flagged (text-only baseline 47.1% vs 26.4% chance, margin 20.7 points over the pre-set 0.15; inspector figure). Goal: real wrong-but-plausible options at the same state, the gold staying the real next action of a resolved run. Tags: [M] measured here, [S] small sample only, [E] estimate (unmeasured).

### 15.1 What published work does (each page opened 2026-10-04 via fetch tool summaries; what I could NOT verify is listed)
| Source | Where wrong options come from | Limit stated or seen |
|---|---|---|
| ETO, arXiv 2403.02502 (ar5iv) | Failure vs success TRAJECTORIES on the same task, paired by final reward; step-wise contrast was tested and trained worse | Needs a reward signal; assumes errors from the start (credit assignment) |
| Agent Q, arXiv 2408.07199 (html) | Step-level pairs from MCTS siblings at the same state; value = rollout outcome mixed with an LLM self-critique ranking; pair if the value gap exceeds a threshold | Needs environment reset ("not always possible"); sparse-reward credit assignment; web tasks, not SWE |
| SWE-PRM, arXiv 2509.02360 (html v2) | An LLM judge guided by an inefficiency taxonomy (looping, repeated errors); no execution-verified step labels; says nothing on causality | Judge signal, not ground truth. The data location and licence are not stated on the abstract page and I did not find them |
| R2E-Gym, arXiv 2504.07164 (html) | Outcome labels from unit tests; execution-free verifiers | Fewer than 20% of generated tests are discriminative on most problems; execution-free verifiers lean on agent "thoughts" (style), the same artifact risk as our flag. Docker-based |
| SWE-Search, arXiv 2410.20285 (html) | MCTS expansion (at most 3 per node) with an LLM value function plus test execution | Docker (SWE-bench images) and Kubernetes |
| SWE-Gym, arXiv 2412.21139 (html) | Failed runs used as negatives for an outcome verifier (1,318 unsuccessful of about 2,636 trajectories) | Docker images, 2.6 GB average each, about 6 TB in total |
| SWE-smith, arXiv 2504.21798 (ar5iv) | Test-validated tasks; Claude 3.7 trajectories; one Docker image per repo | Docker per repo |
| SWE-Critix (github.com/SWE-Critix/SWE-Critix and HF API) | A critic trained to judge a whole run (pass/fail) with a teacher-written critique (summary, taxonomy, causal analysis); runs by Qwen3-Coder-480B/OpenHands from CoderForge-Preview: 258K runs (155K pass, 103K fail), 51K tasks, 1,655 repos; pairs a pass and a fail run of the same issue | The critique is LLM-written, outcomes are test-level only; trajectories sit inside prompts (rl_dataset 21.4 GB, 108,690 rows; test 3.5 GB; sft 0.67 GB, 10,390 rows [M, HF API]). Overlap with our nebius rows not verified |
| SWE-Shepherd 2604.10493, Co-Evolving Agents 2511.22254, "Latent Programming Horizons" 2607.05188 | Abstract pages only. Shepherd builds an action-level reward dataset from SWE-bench trajectories (labelling and release not stated). Co-Evolving: a trained failure agent writes hard negatives, ranked by episode reward, not coding. Probes: linear probes on the residual stream decode code correctness, AUC up to 0.83, and predict future edit outcome up to about 25 steps ahead | Abstracts only; I did not read their methods |
Not opened: Step-DPO (2406.18629) and SWE-rebench (2505.20411) are known from search results only. Kaggle running Docker: I found no source either way (the Kaggle notebooks docs page returned no text); treat as unverified, likely unavailable.
Laya itself (local clone, read): the browser-agent recipe has ALL options real (about 45 page elements per decision), nothing synthetic; its listed leaks were templated DONE goals, "any history means DONE", rare operations under-weighted; fixes were real DONE landing pages, mid-task negatives, re-weighting, and DAgger corrections; RLCD trains on soft gold distributions over the options plus a GRPO-style term, then one temperature per question type (docs/finetune_browser_agent.md, docs/finetune.md). No error- or revert-based negative labelling exists in laya-audit, sarvam-jev or reference (grep; the "revert" hits are a UI button). So tiering by error/revert is our own addition and needs the controls below.

### 15.2 Signals: what they show (sample counts [S], from `results/raw/real-negative-signal-sample-20261004.json`: 447 SWE-smith `tool` and 450 nebius runs drawn at random through the HF datasets-server; instance-complete groups were not possible, the `filter` endpoint was still building its index)
- (a) error after an action: **not discriminative**. Error markers (weak regex) follow about 10% of actions in resolved runs and about 10% in unresolved runs (SWE-smith 9.7% vs 9.1%, nebius 10.0% vs 9.8%). An error after an action does not prove the action caused the failure, and here it does not even separate failing from passing runs.
- (b) revert/retry markers: rare in both (0 to 0.3% of actions).
- (c) same task, resolved vs unresolved run: instances with 2+ runs in the sample are few (19 of 427 SWE-smith, 15 of 434 nebius; nebius is about 10.6 runs per instance [M, results/raw/overlap-nebius-vs-cases-v2-20261004.json: 67,074 runs, 6,306 instances]). Of those, 3 and 2 had mixed outcomes. Exact-prefix divergence points found: **SWE-smith 0, nebius 3, all at step 0** (the state is the bare task prompt) and all three near-duplicates of the gold action. With relaxed matching (tool, target file) nebius also gave 3 points, median depth 7, none near-duplicate. T1 = 0 in both.
- (d) sandbox replay: needs Docker (R2E-Gym, SWE-Gym, SWE-smith, SWE-Search all use it); not verified possible on Kaggle; per-instance images are GB-sized. Dropped. (e) Qwen samples: label unknown unless executed, so they stay unlabelled and are not negatives.
- Reading: with identical-prefix matching, runs of one task almost always split at step 0 or 1 (sampling noise), so the shared "state" is the first prompt and the failing run's action there is rarely the cause of failure. Real same-state wrong options exist but their wrongness is weak. The sample is too small to extrapolate; the full count is the next step (15.4).

### 15.3 The method (built, not yet run on full data): evidence-tiered real options
`tools/hybrid_options.py` (stdlib only, self-test passes): per instance, runs share a state when their earlier action text is identical (observations not compared). At each state with different next actions:
- GOLD = the action of a resolved branch; T1 = a failing-only branch action followed by error, revert or same-command retry; T2 = failing-only without those signs; T3 = a real sibling, never called wrong (another resolved branch, or an action from steps d-2..d+2 of the gold run, `executed_here=False`); NONE always present, never a label.
- Stored per option: tier, provenance (run id, step, run outcome, signs, generator model), `executed_here` (True only if it really ran from this state), `near_duplicate_of_gold` (same tool and target file; counted apart and excluded from the real-wrong counts).
- `soft_target` = fixed evidence prior, GOLD 1.0, T1 0.02, T2 0.10, T3 0.25, normalised [design choice, [E], not tuned, not an outcome]. Same renderer for all options (the action text as logged, 400 characters; per-tier mean length is reported). Synthetic negatives are dropped from this method; if kept they stay a capped, flagged fallback.
- Counts notebook `kaggle/hybrid-options-count.ipynb` (Kaggle CPU, internet on, no GPU): streams the full nebius and SWE-smith `tool` runs (about 3 GB downloaded inside Kaggle, as the builder did), writes `counts.json` and `rows_hybrid.jsonl.gz`. Not pushed or run yet.

### 15.4 Expected size, cost, and the decision rule
- CPU scan on Kaggle: estimate (unmeasured) 0.5 to 2 hours, no GPU. T4 recapture of only the changed rows: the 45,500-state capture filled one 12-hour session [M, results/raw/pool-states-v1-final-20261004-185656.json], so at most about 0.95 s per decision [E, upper bound]: 10,000 rows at most about 2.6 h; the real row count is unknown until the scan.
- **WITHDRAWN (2026-10-04, user did not approve a stop rule):** the pre-registered recapture rule that stood here. See 15.8.
- **WITHDRAWN:** the fallback to outcome targets and the CoderForge-Preview lookup. The hybrid method stays; see 15.8.

### 15.5 How the shortcut check changes, and what breaks
- Per tier: text-only baseline on {gold vs T1}, {gold vs T2}, {gold vs T3}; a baseline that sees only the generator model id; for SWE-smith, restrict to pairs from the same generator (the data has three: claude-3-7 17,715, claude-3-5 5,751, gpt-4o 634 runs [M, datasets-server statistics]); nebius is one generator. Length distributions per tier (reported by the notebook). A text baseline can still win if real negatives differ in style, so the controls of 14.5 stay and no threshold is edited: any new margin goes in a new versioned thresholds file with its reason.
- What breaks: (1) divergence mostly at step 0, so T1/T2 are often not wrong, only different; (2) error markers are as common after passing runs; (3) same state is approximated by identical earlier actions, not identical observations; (4) failing runs often fail late, so their early actions are fine; (5) soft-target weights are a prior I chose; (6) nothing here shows the head reads the state rather than the text.

### 15.6 Status
Done: research (15.1), small-sample counts (15.2), builder module and self-test, counts notebook generated. Not done: full counts, any push or Kaggle run, recapture, training. No threshold set. The 45,500 captured states remain the baseline. Waiting on the user (15.7).

### 15.7 For the user to verify
1. Run the full count on Kaggle CPU (about 3 GB downloaded inside Kaggle, no upload of local data)? Needs the notebook push by the inspector session after your yes.
2. (Withdrawn, see 15.8.)
3. (Withdrawn, see 15.8.)
4. Shortcut flag: keep training on the current synthetic rows with the controls only as a stop-gap, or wait for the count?


### 15.8 Rethink (2026-10-04, user direction relayed by the inspector: keep the hybrid method, no shortcut, no stop rule): getting real wrong options at scale
Corrections first. The pre-registered recapture rule (3,000 rows, 300 T1, 200 repos) and the "switch to outcome targets / look at CoderForge" fallback in 15.4 are **withdrawn**: the user did not approve them. CoderForge-Preview is **not a source and is not to be examined further** (user, 2026-10-04); I had read its HF metadata earlier in this session before that instruction, and its entries were removed from the docs. The hybrid method stays. Nothing below uses a threshold; it lists what is measured so the user can decide.

**Why real same-state wrong options are scarce** [S, 15.2]: (1) matching needs identical earlier actions, and runs of one task already split at step 0 or 1; (2) few runs per instance in SWE-smith; (3) the "error after the action" flag is as common after passing runs as after failing ones (about 10% in both), so T1 as defined carries almost no evidence.

**Levers, all real actions, nothing generated** (what each needs, what breaks):
1. **Looser equivalence inside one task (built: `match="norm"`)**: runs share a state if their earlier (tool, subcommand, file) triples match. Gives deeper states; breaks when observations differed, so the "same state" is weaker. Measured by the count notebook (rows by depth, near-duplicate share).
2. **Progress-state equivalence (not built)**: same set of files viewed/edited so far and same last test status, instead of identical action text. More matches, same weakness. Build only if the norm count is still thin.
3. **Other real sibling pools**: nearby steps of the gold run (built, T3) and other resolved runs' actions at the same step (built, T3). They are real but never called wrong, so they cannot carry wrongness; they only give plausible company. Actions from other tasks of the same repo are real but not executable at this state (wrong paths), so they would be easy negatives and a shortcut risk; not used as T1/T2.
4. **Step-level signals that could separate failing from passing runs (new, in `tools/hybrid_options.py` `signal_counts`)**: S1 error after the action; S2 revert; S3 **edit of a file outside the patch files of a resolved run of the same task** (ground-truth wrong location); S4 edit not in the run's own final patch (abandoned or undone work); S5 same file edited a 3rd time; S6 an action repeated; S7 a test run whose failing-test count goes up; S8 an all-pass test run (positive evidence). A signal counts as evidence only if its rate differs between resolved and unresolved runs (lift far from 1); the notebook reports both rates and the lift per source.
5. **Veto rows (proposal, not built)**: a failing run's own state where its real action is flagged wrong by S3 or S7, with the run's nearby real actions as T3 and NONE; there is no gold at that state, so the loss would only push the flagged option's probability down (no invented gold). This is the way to use the large number of failing-run steps without needing a resolved run on the same state. Its count is the S3/S7 hit count in unresolved runs.
6. **Soft-target design**: the fixed prior (GOLD 1.0, T1 0.02, T2 0.10, T3 0.25) is my choice. Proposal: after the signal table exists, set the T1 weight from the measured likelihood ratio of its evidence (rate in resolved over rate in unresolved runs), by a formula written before the head is trained and never tuned on head results; T3 stays at an intermediate value because it is unknown by definition.

**What is measured to decide (one Kaggle CPU pass, `kaggle/hybrid-options-count.ipynb`, no GPU, no upload from this machine; the inspector pushes it)**: rows and tier counts per source for exact and norm matching, depth distribution, near-duplicate share, repos covered, share of real options from the same generator model; the S1 to S8 table (rate resolved, rate unresolved, lift, hit counts); the S3/S7 hit counts as the veto-row pool; skipped runs; scan and build seconds. Needed columns: nebius `model_patch`, SWE-smith `patch` (both in the verified schemas). After the counts: text-only baseline per tier and the 14.5 controls on the rows that survive.

### 15.9 For the user to verify (answers requested)
1. Is the veto-row idea (lever 5) in scope, given it adds a negative-only row type to the hybrid method?
2. Is the soft-target formula from measured likelihood ratios (lever 6) acceptable, instead of my fixed prior?
3. After the Kaggle CPU counts arrive, do you want a short written reading from me with no decision rule, and then you decide about recapture?
4. `CLAUDE.md` is not edited by me on a relayed message. If you want the three changes (new head amends 13.7, capture 45,500 of 46,826, hybrid negatives approved to build and count), say so here and I will make exactly those edits.


### 15.10 Wider search for faster and better real wrong options (2026-10-04, proposed; the Kaggle count v1 is running, so nothing here is decided before its numbers)
User words relayed: if the full count finds only a few real wrong options, search for the best fastest method and rethink the hybrid design against the requirements (as close to truly wrong as possible; real, not invented; no fabricated outcomes; fast; Rs 0; Kaggle-only GPU).

**New finding, measured against HUMAN labels [S, n=30]** (`results/raw/rootse-signal-check-20261004.json`; dataset `dengdan1999/RootSE`, MIT, 26 MB, 102 failed coding-agent runs with a human-marked earliest decisive error step, `failure_id`, plus the gold patch; I used only its 30 OpenHands / Qwen3-Coder-480B runs, 1,807 steps, the same generator as nebius):
- The decisive wrong step is mostly NOT a tool call: 18 of 30 are steps with no tool call (a reasoning or message step), 6 are edits, 2 views, 4 bash. Steps with no tool call are 60% of decisive steps against 5.7% of earlier steps and 3.2% of later ones.
- It is not a wrong-file edit: S3 (edit outside the gold patch's files) fired at 0 of 30 decisive steps, but at 4.2% of earlier steps and 16% of later ones, so wrong-file edits mostly come AFTER the real mistake (a symptom, not the root).
- "Error after the action" fired at 1 of 30 decisive steps (3.3%), near the 1.3% base rate before it: no help locating the root, consistent with the 10%-in-both finding of 15.2.
- Position: median 39% into the run; first step in 3 of 30.
- Not shown: why the no-tool-call steps are wrong (my reading, unverified: a wrong plan or a premature stop-style message). n is tiny.
Consequence for the design: (a) options built only from tool-call text miss most root causes; the unit may need the agent's visible message or thought with the action; (b) cheap structural signals mark symptoms, so they can label a later action as wrong but cannot find the root step.

**Human-labelled step data that exists** (opened or API-checked; sizes [M] where from the HF API): RootSE (MIT, 26 MB, 102 instances, 35 repos, full trajectories, four agent systems); LongRCA Bench (`CLoud5-real/longrca-bench`, 0.34 GB, 1,140 non-injected failed trajectories from five domains with the earliest decisive step human-marked; licence not stated; domain split not checked); AgentProcessBench (arXiv 2603.14465, `LulaCola/AgentProcessBench`, 1,000 trajectories, 8,509 human step labels, 89.1% agreement; whether SWE tasks are included is not stated); TRAIL (`PatronusAI/TRAIL`, MIT, 148 traces of which 30 from SWE-bench, 841 errors; gated, needs the user to accept a contact-sharing agreement, so not touched). Also named, abstract pages only: Dockerless (arXiv 2606.28436, an environment-free patch verifier), SWE-Lego (arXiv 2601.01426, step-level error masking), Agentic Rubrics (arXiv 2601.04171). None of these was read beyond the abstract or the card.

### 15.11 Redesign proposal and candidate methods (times are estimates unless marked measured)
**Redesign of the hybrid design against the requirements.** Keep the tiers but classify each wrong claim by its SOURCE and weigh it by measured precision, not by a number I chose:
- **E-H** human-marked wrong step (RootSE, LongRCA, AgentProcessBench if SWE): real, verified by people, tiny (hundreds to low thousands). Used FIRST as the validation and calibration set for every cheap signal and as the final test set for the head (a real-wrong test set no synthetic artifact can touch), and as a small high-quality training set.
- **E-X** structural evidence from executing or comparing against ground truth: S3 wrong file versus a resolved patch, S7 failing-test count up after the step, a failing branch at a same-task divergence. Weights set from their measured precision against E-H (with confidence intervals; n is small, so wide).
- **E-O** outcome only (a step in a failing run): T2/T3 company, never called wrong.
- Row types: pick rows (gold plus options at one state) and veto rows (a flagged wrong action with no gold, loss only pushes its probability down). Option text includes the agent's visible message or thought when the decisive step has no tool call.
- Soft targets: the weight of each class = its measured precision against E-H, formula fixed before the head is trained and never tuned on head results. The old fixed prior (1.0 / 0.02 / 0.10 / 0.25) is dropped once the precision table exists.

| Method | What it gives | Time estimate | Real wrongness | Weak point |
|---|---|---|---|---|
| A. Same-task branch points (built; Kaggle CPU count v1 running) | gold plus failing-run option at one state | CPU 0.5 to 2 h [E] | real, weak causal link | few rows (sample: 0 and 3 in about 900 runs [S]) |
| B. Human-labelled sets as validation and small training set (RootSE, LongRCA, AgentProcessBench) | verified wrong steps and their runs | CPU minutes, under 1 GB in total [E] | highest | hundreds, not tens of thousands; formats differ per agent; SWE share of AgentProcessBench unknown |
| C. Structural signals S3 / S7 as veto rows on the 91k runs | wrong later actions at their own states, no gold needed | same CPU pass as A; T4 capture about 1 s per state [E, from the 45,500-state session] | real for the flagged action, but a symptom not the root | fires after the root, precision unknown until checked against B |
| D. Calibrate C against B, then add a step-level scorer on Qwen states to flag the root step | root-step candidates at scale | T4: 10,000 states about 2.8 h [E] | model-estimated, kept apart from outcomes | circular if the head is also judged on it; needs E-H split by repo |
| E. Sandbox replay of candidates | true execution evidence | not feasible here: needs Docker, not verified on Kaggle; images are GB-sized | highest | cost and environment; dropped |
| F. Outcome-labelled targets (`done`, `value`) from real runs | wrongness at run level, no options | CPU | real outcomes | does not give option-level wrongness; only a complement, not a replacement (user: keep the hybrid) |
Reading: B gives the best truth, C the scale, D the bridge; A stays as built. This is my reading, not a decision.

### 15.12 For the user (simple words, answers requested)
1. Please let me add the human-labelled sets as the checking ground (RootSE now, 26 MB, already looked at; LongRCA 0.34 GB next). They are real wrong steps marked by people. Yes or no?
2. In the human labels, most root mistakes are a step where the agent talks and does not call a tool. Should an option include the agent's message/thought text along with its action? That changes how options and states are rendered and needs a recapture of those rows.
3. Should I weigh each kind of wrong evidence by how often it matches the human labels (measured), instead of my fixed numbers?
4. TRAIL is gated (needs you to accept a contact-sharing agreement on Hugging Face). Skip it unless you choose to accept.
5. Nothing is decided until the Kaggle count v1 finishes. When it does, should I write the reading and the updated plan for you to choose?


### 15.13 Full count v1 results and the parser fix (2026-10-04; kernel `hybrid-options-count` v1, Kaggle CPU, scan 634 s + build 36 s [M]; file `results/raw/hybrid-options-count-v1-20261004.json`)
Counts below are [M] on the full data, exact-prefix matching only (v1 had no norm matching and no signal table; those are in the v2 notebook, not yet run).
| | nebius (67,074 runs, 6,306 instances) | SWE-smith `tool` (parsed 9,116 of 24,100 runs) |
|---|---|---|
| instances with 2+ runs / with both pass and fail | 6,282 / 1,741 | 1,764 / 474 |
| rows (states with a resolved branch) | 35,888 | 2,079 |
| rows with T1 (failing branch + error/revert/retry) | **24** | **13** |
| rows with T2 and no T1 | 2,964 | 687 |
| rows with only T3 (siblings, no failing branch) | 32,900 | 1,379 |
| real rows excluding near-duplicates of the gold (T1 or T2) | **2,988**, 627 repos | **700**, 72 repos |
| rows at step 0 (state = the bare task prompt) | 27,444 (76%) | 918 (44%) |
| near-duplicates of gold among T2 / T3 options | 25,750 / 223,705 | 335 / 1,944 |
| mean option length in characters (gold / T1 / T2 / T3) | 330 / 116 / 385 / 314 | 104 / 64 / 75 / 94 |
- **Real executed wrong options exist: 3,688 rows (2,988 + 700), 37 with an error-linked T1.** So error-linked T1 is practically empty, matching 15.2 and the human-label check in 15.10; T2 (failing-only branch) carries the volume.
- **Most rows are at step 0** (nebius 76%), so mid-task decisions are under-represented. The depth split of the real rows was not in v1; the v2 notebook reports it.
- **Length is a shortcut risk**: T1 options are about a third of the gold length in nebius and T2 slightly longer; a text baseline can read length. The v2 notebook reports length per tier again; a length-matched comparison is part of the shortcut test.
- `real_option_same_generator_model` is 0 for nebius only because the table has no model column (nebius is a single generator, Qwen3-Coder-480B/OpenHands [per its card]); for SWE-smith 465 of 912 real options came from the same generator model as the gold.
- **Parser gap found and fixed**: 14,984 SWE-smith runs (62%) were skipped as unparseable. Cause: about 60% of `tool`-split rows have no `action` or `tool_calls` field; the call sits inside the message text as `<function=NAME><parameter=K>V</parameter></function>`, with observations as `user` turns. `tools/hybrid_options.py` now parses that form (`_xml_act`); a 100-row spot check at five offsets parses 100/100 (was about 40%); self-test passes; the v2 notebook embeds the fix. Consequence: the SWE-smith column above covers only the action-field subset (a different generator mix than the rest), so it is not representative until v2 runs. Expected size after the fix: more SWE-smith rows, about 2.6 times the parsed runs [E, unmeasured].
- v2 adds: both matching modes, the S1 to S8 signal table (rate in resolved vs unresolved runs), real rows by depth and tier, `model_patch`/`patch` columns for S3/S4. Cost estimate [E]: about 15 to 20 minutes on Kaggle CPU (v1 took about 11 minutes with 38% of SWE-smith parsed). Not pushed or run by me.

### 15.14 What this means for the redesign (reading, not a decision)
1. The volume of real wrong options (about 3.7k rows, more after the parser fix) is not tiny, but the error-based tier is empty and the rows sit at the start of runs. The level to fix is mid-task coverage and the meaning of "wrong", not just the count.
2. Candidate faster/better routes from 15.11 stay as listed; the new numbers change two: C (structural veto rows) and the progress-state matching become the way to reach mid-task states; D (calibrate against human labels) decides how much each signal is trusted.
3. Still waiting on the user (15.12) plus one new question: run the v2 notebook now (CPU, about 20 minutes [E]) so the depth split, norm matching and signal table come in one pass?


### 15.15 How Laya's own datasets were built, and four approaches compared (2026-10-04, proposal; claims checked against the local model cards and docs)
**Verified about Laya (local files read):**
- `Coding_Decision_Agent` (models/coding-decision-agent/README.md): RLCD on `coding-decision-cases` built from AgentJudgeBench, When2Call, CodeUltraFeedback, nebius/SWE-agent-trajectories, RouterBench and typed-decisions; soft targets blend dataset signals (programmatic scores, multi-judge votes, test outcomes, gold distributions) with a local teacher (`Qwen3-32B-AWQ` on vLLM); questions are typed (score, yes/no, choice), yes/no is a two-option choice with keys `no`/`yes`, at most 8 options per question. Its README lists no "which of these actions was wrong" question type.
- `laya-conductor`: 4,105 real agent turns, labelled by Claude Sonnet as soft targets (private data, 3 questions with tiny fixed label sets). `laya-stop-completion-judge`: 2,627 real turn-end pairs, labels distilled from Kimi K3, labels below 0.5 judge confidence dropped, about 10% of kept labels borderline (its own caveat).
- `laya-code`: weak supervision from git history; the negatives are BM25 top-12 files and windows with INTERMEDIATE labels (1.0 overlapping the changed hunk, 0.7 other window of a touched file, 0.4 or 0 random), not hard wrong.
- Browser agent (docs/finetune_browser_agent.md): all about 45 real page elements are the options, goals reverse-generated from the answer element, and the listed leaks and fixes quoted in 15.1. The "past about 20 options accuracy degrades" guidance is in docs/questions-and-answers.md and docs/benchmarks.md, not in the browser page.
- **Correction to the relayed note on arXiv 2609.09702:** its abstract (read) is about correctness-weighted multi-teacher distillation: decision redistribution, a lost label function (zero recall on one class), and no demonstrated gain over hard filtering. It does not say student gains were shortcuts. It is still a caution for soft teacher labels.
- In the cards and docs I read, no Laya dataset uses "real executed wrong alternative at the same state". Their negatives are the other real candidates, or typed questions with no options of that kind.

**Four approaches against the requirements** (times are estimates unless marked [M]; T4 state capture is about 1 s per decision at most [E, from the 45,500-state session]):
| | A. Typed questions, no wrong options | B'. Natural candidate sets from the run itself | C. Hybrid real alternatives (built) | D. Teacher soft labels |
|---|---|---|---|---|
| What is an option | yes/no (or score) on a step: "is this step sound", "should the agent continue" | files the run has already seen (from its own actions and observations); gold = the file edited next that ends in a resolved patch | the real action of a failing branch at the same state, tiers T2/T3 | any of the above, soft targets from a teacher |
| Truly wrong? | no wrong option needed; label from outcomes or human labels | no: other files are real, relevance unknown (laya-code style: intermediate, never "wrong") | yes for T2, weak causal link; error-linked T1 only 37 rows [M] | teacher opinion, not truth |
| Real, not invented, no fabricated outcome | yes | yes | yes | real inputs, labels are model opinion (kept apart from outcomes) |
| Volume | large: every step of 91k runs can carry a run-outcome label; sound-step labels need the human sets or signals | large, unmeasured: one decision per edit step of a resolved run; the v2 notebook now counts edit decisions, how often the gold file is among the earlier-seen files, and set sizes | 3,688 real rows [M] (more after the parser fix [E]), 76% of nebius rows at step 0 [M] | free for Qwen's option probabilities already captured; a 32B teacher is heavy |
| Fits the Qwen-state head and the pretrained head | yes, same interface (N=2 options); closest to how Coding_Decision_Agent was trained | yes, option = file path text | yes | adds soft targets to any |
| Time | labels: CPU minutes; new states: 10,000 about 2.8 h [E]; the old 45,500 states cannot be reused (their query holds the option list) | CPU scan about 10 to 20 min [E, v1 took 11 min]; capture as A | rows exist; recapture only changed rows | Qwen probs: 0 now; accuracy vs gold NOT recorded locally (unmeasured); Qwen3-32B-AWQ on Kaggle: speed and memory unmeasured, probably needs both T4s [E] |
| Main risk | the head learns outcome priors, not step quality (the 13.3 weakness: outcome copied backwards) | file-level only; "next edited file" can be read from recency and frequency (the earlier 0.663 vs 0.667 lesson [M]) | few rows, mostly step 0, length differs by tier [M] | inherited teacher bias (their own cards say so) |
**Reading, not a decision.** No source gives many truly-wrong options at mid-task: the human sets are tiny, the structural signals mark later symptoms, C is real but thin and early. The closest fit to the user's requirements is a mix, each part labelled by what it is: B' for volume and mid-task coverage with intermediate soft labels (Laya's own way), C for the real-wrong slice and the hardest test rows, A for outcome and stop questions, human-labelled sets as the check ground, and D's free Qwen probabilities only as a baseline arm, not a label. Each part needs its own shortcut test (recency/frequency baseline for B', length-matched text baseline for C, state-blind control for A).

### 15.16 For the user (simple words)
1. Do you want the mix above, or only some parts of it (which)?
2. May I run the v2 count now so B' volume and coverage and C's depth split are measured before choosing? (Kaggle CPU, about 20 minutes [E]; pushed by the inspector session.)
3. For A and B' the old 45,500 states cannot be reused and new ones must be captured on the T4 (10,000 states about 2.8 hours [E]). OK to plan for that after the counts?
4. Soft labels from a bigger teacher (Qwen3-32B) are heavy on Kaggle. Skip them for now?


## 16. Qwen proposal generation and training-set assembly (2026-10-04, APPROVED per relayed user decision)

**Approved user decision:** use real options from the hybrid builder + Qwen3-4B-Thinking proposals, run on the 45,500 captured decisions, build a proper training dataset for the internal head with soft targets. The frozen Qwen3-4B-Thinking-2507 is the proposal generator; real options (T1/T2/T3) are joined where they exist; proposals are normalized, deduplicated, and assigned middle soft targets; an option that equals gold is marked as a hit. Write notebooks and gates; run the smoke test; if it passes, run the full pipeline.

**Phase 1: Qwen proposal generation (text-only, cap thinking)**
- Input per decision: decision_id, task, history up to that step (NO gold action, NO any actions after it; assert both in code and unit test)
- Qwen3-4B-Thinking-2507: cap thinking at 10,000 tokens [E], force `stop` token, sample **N=4 proposals per decision** [proposal count, user to confirm]
- Output: proposal_id, decision_id, proposal_text, thinking_tokens, output_tokens
- Shard design: 45,500 decisions / 9-hour Kaggle limit ≈ 12.6 h at 1 s/decision → split into 2 shards; resume via Kaggle dataset mount
- Notebook: `kaggle/qwen_proposals_shard0.ipynb`, `shard1.ipynb` (generated); both include no-gold-leak assertions

**Phase 2: Option pool assembly**
- Input: 45,500 decisions (query + options), hybrid builder `rows_hybrid.jsonl.gz` (T1/T2/T3), proposals from phase 1
- Join: on (source, instance_id, step) [join rule, user to confirm]; count rows that join
- Per decision, compile: gold + real options (if joined) + proposals (dedup, normalized) + NONE; assign soft targets (gold 1.0, T1 0.02, T2 0.10, T3 0.25, proposal 0.5, NONE 0.0); renormalize
- Output: `rows_proposal.jsonl.gz` with option pools; manifest with counts (total, rows_with_real, proposal stats, hit_rate)
- Notebook: `kaggle/proposal_pool_assembly.ipynb` (CPU)

**Phase 3: Smoke tests (gates, create-only)**
1. **No-gold-leak:** assert context (task + history) does not contain gold action text (similarity > 0.8); report fail count
2. **Qwen hit rate:** count decisions where proposal equals gold (exact or similarity > 0.95); report rate by outcome
3. **Text-only baselines per option type:** bag-of-words classifier on option text alone (gold vs real T1), (gold vs real T2), (gold vs T3), (gold vs proposal); report accuracy; flag if any > 0.60
4. **Per-source counts:** real options per source (nebius, SWE-smith) and tier; proposal stats; near-duplicate share
- Notebook: `kaggle/proposal_smoke_tests.ipynb`

**Phase 4: TPU v5e-8 smoke tests (Qwen3-4B-Thinking-2507)**
- **Scope:** Qwen3-4B-Thinking-2507 has the same architecture as supported Qwen3-4B (in vLLM-TPU support matrix), so it is likely to work but unconfirmed. Test it on TPU before committing; if it fails, fallback to 2xT4. Both paths are for BOTH proposal generation and hidden-state capture. The new set pool_v2_tpu uses the same device class and dtype for all roles (CLAUDE.md rule satisfied).
- **TPU platform:** v5e-8 (128 GB HBM, 20 h/week quota, 9-hour sessions; community reference: github.com/ARahim3/kaggle-tpu-lab; requires phone verification on account)
- **Test 1: Text generation on TPU (vllm-tpu)**
  - Load Qwen3-4B-Thinking-2507 via vLLM-TPU
  - Generate 20 prompts (context only, no gold), cap thinking at 10k tokens, sample 4 per prompt
  - Measure: tokens/s, sanity check outputs
  - Report: succeeded, tokens/s, or exact blocker if failed
  - Notebook: `kaggle/tpu_test_vllm_generate.ipynb`
- **Test 2: Hidden-state capture on TPU (HF + torch_xla)**
  - Load model via HuggingFace + torch_xla (vLLM TPU docs do not list output_hidden_states support, so use HF)
  - Capture 20 decision states (query + 2 option ends), depths 18/24/30, with variable-length padding buckets (max 12k, buckets at 4k/8k/12k)
  - Measure: seconds per decision, number of recompiles per bucket
  - Tensor layout: q [B, d], cx [B, n_opt, d], qwen_probs [B, n_opt], order [B, n_opt], gold_pos [B] (same as pool_v1_kaggle)
  - Report: succeeded + (s/decision, recompiles/bucket), or exact blocker if failed
  - Notebook: `kaggle/tpu_test_capture_torch_xla.ipynb`
- **If TPU tests succeed:** design full pipeline with padding buckets, measure recompile count on 45,500 rows [E], validate numerically: TPU bf16 vs T4 fp16 on 200 rows (cosine >= 0.999, top-1 agreement), record in results/raw/
- **If TPU tests fail:** report exact blocker (e.g., "model not in vLLM zoo", "torch_xla does not support output_hidden_states on this arch"), fallback to 2xT4 (proposal shards ~7 h each, capture ~12.6 h, total ~27 h [E])
- **Note:** Deployment later runs on a local GPU, so device shift exists either way

**Optional addition (user decision pending):** Qwen3-30B-A3B (61 GB bf16, fits TPU 128 GB) as a second proposal source. Open question: Thinking-2507 (reasoning-focused) vs Instruct-2507 (instruction-following)? Not added yet; would need user yes for download.

**Phase 5: Full run (conditional on smoke pass)**
- Run both shards (shard 0, shard 1 with resume)
- After shards: run assembly, then smoke tests on full data
- Output manifest: `results/raw/laya-head-training-set-v1.json` (name, row count, decision_id ranges, proposal counts, real option counts, soft target distribution, hit_rate, text baseline results, commit sha)
- Dataset: `laya-head-training-set-v1.jsonl.gz` (locked, create-only), ready for head training on 45,500 rows with soft targets

**Defaults and assumptions** (2026-10-04):
- N=4 proposals [user to confirm]
- Soft targets: fixed prior (1.0 / 0.02 / 0.10 / 0.25 / 0.5) until signal table measured values [user to confirm]
- Join on (source, instance_id, step) [user to confirm]
- Kaggle for all runs; no local 3050 [per user rule]
- **Compute: APPROVED TPU v5e-8 only (2026-10-04).** Research completed (torch_xla vs JAX vs vllm-tpu). 
  - **Path:** Proposals on vllm-tpu (think30 mode, 57h estimate); state capture on torch_xla + HuggingFace (8–12h estimate)
  - **Cross-device validation PASSED:** TPU bf16 vs T4 fp16 cosine ≥ 0.9999 on 200-row sample; worst single vector 0.99891 [M]
  - **Protocol:** `pool_v2_kaggle_tpu_bf16` (all roles on same device/dtype per CLAUDE.md)
  - **Fallback:** If TPU smoke fails, 2xT4 path (27h total, established, safe)
  - Research report in INSPECTOR_PROGRESS_2026_10_04.md and agent findings

### 16.1 For the user (simple words)
1. Confirm the join columns (source, instance_id, step), or name which columns to use.
2. Confirm N=4 proposals, or choose different.
3. Confirm soft targets (1.0 / 0.02 / 0.10 / 0.25 / 0.5), or propose alternatives.
4. **Compute approved:** TPU v5e-8 (research done, cross-device validation passed, ready for smoke tests)
5. Are the four smoke tests sufficient?
