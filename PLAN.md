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
2. **[PATH A CHOSEN 2026-10-03; RUNNING: Kaggle notebook v8, dataset `laya-textemb-trainval`, 12,602 TRAIN+CAL pool texts; the 21,713 LOCKED-pool texts wait for LOCKED approval; the 3050 embeddings (34,315 texts, 14.4 min, TEXTEMB_OK) are kept for cross-checks only]** State-free text embeddings on the T4 (spec `[capture_v3] text_embeddings`; 34,315 unique texts, about 0.5 h [E]; 7 local bf16 shards of 32 MB exist from the 3050, completeness unverified and they would be recaptured under the one-device rule). Needs: port `tools/capture_textemb_v3.py` to the resident runner, second Kaggle run. **Confirm: yes/no.**
3. **Download and verify** `capture_train.tar`, `capture_cal.tar`, `capture_manifest.json` (about 1.7 GB). **Needs a written yes (over 1 GB).** Checks: sha256 manifest, amendment e2 and e3, cosine vs the 3050 files on 1,376 decisions (e4).
4. **DONE 2026-10-03: amendment (e) REGISTERED** in `configs/a5_v3_spec.toml` [amendment_2026_10_03_e] and `configs/thresholds.toml` [amendment_e_thresholds] before any head was trained: text-embedding gate 0.9999 (unchanged, every shard), pre-LOCKED screen point estimate >= 0.02 (go/no-go only, not success; the primary criterion 0.05 / lower bound > 0 / half-width <= 0.04 is unchanged), 3-seed selection, Path A device lock, cross-device comparisons report-only, Brier secondary arm, freeze-and-hash before LOCKED.
5. **NEXT: train the head** (H0, H2 on the 9-cell grid, 3-seed selection, 5 paired seeds) on CPU or the 3050, locally; time unmeasured, one cell is timed first. **Confirm where.**
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


