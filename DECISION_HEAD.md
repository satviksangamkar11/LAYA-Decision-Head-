# DECISION_HEAD.md — internal decision head over frozen gpt-oss-120b

Design record for the one component this project trains. Rules: [CLAUDE.md](CLAUDE.md). Step list: [PLAN.md](PLAN.md) section 11.
**Direction (user, 2026-10-01): the head works for any local model; the first backbone is `gpt-oss-20b-heretic` (hidden 2880, 24 layers, 32 experts, BF16 [M]), read-only and only for capture. gpt-oss-120b is parked and untouched.** Everywhere below that says "gpt-oss" or "120B" is the long-run target; the 20b-heretic run comes first, and its learning curve sets the capture budget. Backbone-specific numbers (layers, K) are re-measured per backbone. What is done so far is summarised in [README.md](README.md).

Evidence tags: **[M]** measured here on this machine · **[S]** read in source code or a config file I opened · **[A]** author-reported on a model card or README, not reproduced by us · **[E]** my estimate, unmeasured.

## 0. Status

| Item | State |
|---|---|
| `decision_head/head.py`, `selfcheck.py` | Built. 19 checks pass on **synthetic data only** (project venv, torch 2.14.0+cu126). Says nothing about gpt-oss quality. |
| G0 (CUDA) | PASS, recorded in `results/raw/g0-cuda-*.json` [M] |
| gpt-oss-120b | All 15 shards on disk; sizes, safetensors headers, index tensors and sha256 verified (`results/raw/verify-gpt-oss-120b-deep-20261001-154039.json`) [M]. |
| Laya weights | `laya-typed-decisions`, `laya-conductor` and `laya-stop-completion-judge` on disk at pinned revisions (`configs/models.yaml`), sha256 verified, each loaded offline pinned by sha256 [M]. Typed-decisions reproduces its card on all 2,000 test decisions: 0.771 vs 0.766 published, every slice within 1.4 points (`results/raw/laya-eval-laya-typed-decisions-20261001-153708.jsonl`) [M]. The conductor's six printed routing examples land on the same tier [M]. The judge separates announced-next-step turns (0.75, 0.74) from a direct answer (0.05) but scored a short results-reported case 0.46 and an explicit handback 0.70 [M]. Not downloaded: `laya-code`, `cortex-1-large` (0.785 GB each). |
| Capture, data compiler, training, calibration | Not started. |

## 1. What was read

Primary sources read in full or in the relevant parts: the Laya repo clone at v0.3.22 (`laya/common.py` DecisionModel, `docs/finetune.md`, README, the typed-decisions card and `rl_agent_config.json`), PostHog/jeeves (`model/head.py`, `predictor.py`, `calibrate.py`, `prep/format.py`, README), and the model cards below (raw `README.md` from the Hub, 2026-10-01).

| Repo | What it is | What we take | Caveat |
|---|---|---|---|
| NandhaKishorM/laya, `convaiinnovations/laya*` | ModernBERT encoder + 2-layer transformer head + per-marker MLP scorer; RLCD training | The head design, the loss weights, per-(type, option-count) temperatures, the calibration protocol [S] | Base checkpoints sit near chance zero-shot (0.362 on typed-decisions) [A]. Its own act head carries no signal (issue #185, AUROC 0.30) [A]. |
| PostHog/jeeves (MIT) | Qwen3.5-9B + LoRA r16 + `PointerHead` (q and k linear maps, dot product, one temperature) | The pointer form, the marker-token ablations, the temperature grid [S] | Backbone is LoRA-tuned, not frozen. |
| TheoLeeCJ/openjev (SemIf, MIT), `openjev/openjev` (CC BY-NC) | Direct option-letter logits from a decoder | Shared-prefix scoring, calibration numbers, the authored-144 style gates [A] | OpenJev/SemIf is no longer the decision layer (user decision). |
| chaoliangUNSW Jev-Style 0.8B / 2B | Qwen3.5 decoders, option-letter readout, LoRA r32 / full fine-tune | Split protocol (dev / calibration / test), exact slot positions, FP32 option rows [A] | Tuned backbones. |
| wayfind/metask-jev-4b-policy-mix | Qwen3.5-4B LoRA r16, letter readout | Three option orderings per item as training views; per-kind temperatures [A] | Its RLCD citation (arXiv 2510.01237) is a confidence-routing paper, not RLCD [M, abstract read]. |
| lostargon/Tiny-Jev | Qwen3-0.6B + marker + linear head | Contrastive negation and prompt-injection sets; OOD calibration collapse (ECE 0.004 in-domain vs 0.299 on held-out domains) [A] | Synthetic data. |
| aimeigaoshou/agent-jev (AgentJev) | Qwen3-0.6B without LM head + permutation-equivariant candidate head; executed coding pairs | Shared-prefix cache (609.65 ms to 298.91 ms, token ops 33,547 to 2,551); executed-label recipe [A] | Coding-completion AUROC only 0.589 [A]. |
| com-kotobalabs/open-jev-deberta-v3-large, kotoba-lang/typed-decisions | DeBERTa encoder, span-pooled head | **A head on a fresh marker token stayed at the label prior; a span-mean head learned on 3 of 5 backbones.** Augmentation, consistency-loss and answer-leak measurements. A code-decision task ("which definition does this reference next") scored 0.638 on unseen namespaces against chance 0.18 [A] | Encoders, English. README is Japanese; read in excerpts. |
| mobarmg/jev-schema-scorer | One scalar head per (state, candidate) pair | Opaque option ids half the time; one-hot targets give peaked probabilities [A] | Near chance on some task shapes. |
| GoatHerder/Ariadne-Laya-* | Frozen Laya + a 1,049,600-parameter identity-init affine | Frozen-backbone precedent (75.45% vs official 76.70%); LR 3e-4 to 1e-4 cut seed SD from 6.68 to 0.93; TF-IDF baseline [A] | Encoder, narrow tasks. |
| chaoliangUNSW/MacJev-322M-4K-Laya | Laya retrained on 4K inputs | Objective: SFT + ordinal + replay + permutation consistency; 6,000 calibration rows; raise on overflow [A] | Last 6 layers tuned. |
| mvilacad/laya-conductor | Laya fine-tune, effort routing and continue | A teacher for the CONTINUE and effort decisions only [A] | Labels come from Claude Sonnet, not execution; 66% "slow". |
| tampajohn/laya-stop-completion-judge | Frozen encoder, 26.5M-parameter head | **Frozen encoder + head-only works on a narrow task** (AUROC 0.57 to 0.85), tail-first state packing [A] | One operator's data; over-fires on clarification questions. |
| tindang/laya-code | Code relevance reranker | Git-history weak-supervision recipe, commit-hash splits [A] | Weak file-level labels; its marginal gain was within noise at n=20 [A]. |
| mukti-sys/cortex-1-large | ModernBERT-large + 2-layer head | A candidate teacher for safety gating and SWE triage (author reports 82.32% on 690 decisions) | All author-run; it scores 32.7% on typed-decisions after specialising [A]. |
| cklxx/laya-browser, ShaunSpark/laya-mind2web | Browser-agent heads | "Stopping too early" is about 60% of failures; DAgger-style on-policy labels [A] | Browser domain. |
| Manav2op/verdict-small / -typed-small | 118M bi-encoder, cosine × 20 | Shows cross-encoding beats bi-encoding (0.689 vs 0.766 on typed-decisions) [A] | Not a candidate. |
| jinghao1632/bit-jev, TokenRhythm/NeoHorse, apus-ailab/APUS-OpenJev, AlexWortega/openjev, autotrust/JEV-*, akhilaaa3/Jev-Omni | Other pointer / readout models | **AlexWortega/openjev: each option scored on its own over a shared prefix, 0 of 231 order flips** [A]. bit-jev: teacher-logit distillation at T=2.0 [A]. | No coding-decision evidence. AlexWortega's accuracy falls from 0.833 to 0.467 under one injected line [A]. |

Datasets measured via the Hub API [M]: `LocalLLaMA/typed-decisions` 2 MB, Apache-2.0 (1,200 train / 400 test cases, 6,000 / 2,000 decisions, soft gold); `R2E-Gym/R2EGym-SFT-Trajectories` 3,231 rows, 56 MB, dataset-card license blank (GitHub repo Apache-2.0); `nebius/SWE-rebench-openhands-trajectories` 67,074 rows, 2.08 GB, CC-BY-4.0; `nvidia/SWE-Hero-openhands-trajectories` 34,269 rows, 2.40 GB, CC-BY-4.0; `SWE-bench/SWE-smith-trajectories` 76,002 rows (24,100 + 26,076 + 25,826 in three formats of the same tasks), 3.17 GB, MIT.

**Named in the planning chat but not found or not verified:** Gyra (no matching model on the Hub), Evidence Gate and Supersession checkpoints, Laya Instinct, Vision-Laya, a "~150M Verdict" model (the nearest, `verdict-small`, is a 118M bi-encoder).

## 2. Lessons that change the design

1. **Read option spans, not a marker token.** On encoders, a scoring head on a fresh `[OPT]` marker sat at the label prior in every run of one epoch over 3,000 states, across learning rates, head LRs, Brier weights, autocast and attention settings; the same loop overfit 16 states to loss 0, so the mechanics were right and "the marker's hidden state has no pretrained structure to score" [A, kotoba]. Reading the mean of each option's text tokens fixed it for ModernBERT-base (0.539) and DeBERTa-v3-large (0.787) but **not** for ModernBERT-large (0.39) or DeBERTa-v3-base (0.398), so it is backbone-dependent, not a guarantee. gpt-oss has 11 reserved tokens (ids 200000, 200001, 200004, 200009, 200010, 200011, 200013 to 200017) [M], most likely with untrained embeddings. Jeeves's rare-token result came with LoRA, which could learn to use them; ours is frozen. Marker tokens stay as an ablation arm only, and the span choice is re-measured on real gpt-oss states.
2. **Query = the last prompt position after an answer cue.** Jev-Style and SemIf read exactly that position [A].
3. **Remove option order by construction.** An untuned decoder flips its answer on 18.5% of shuffles, a tuned one on 2.3% (OpenJev); Jev-Style v2 6.0% [A]. Our backbone is frozen, so tuning cannot fix it. Primary design: a shared prefix (state + question) once, then **one independent suffix per candidate** with identical position offsets, so each option state is independent of the others and of the order. AlexWortega/openjev does this structurally (each option scored on its own, the common prefix computed once, branches run as separate batch entries with the prefix cache copied) and reports 0 of 231 label changes with a largest probability change of 1.8e-7 (reversed) and 1.2e-7 (shuffled) [A]. Fallback: all options in one sequence in K orders plus `perm_consistency`; kotoba measured that a consistency KL gave no OOD gain over plain augmentation (0.635 vs 0.648 with augmentation alone) at twice the cost [A].
4. **Both surveyed heads, one module.** `layers=0` is the Jeeves pointer [S]; `layers>0` adds Laya's set attention [S]. No positional encoding, so equivariance holds by construction (verified on synthetic data, max diff 2.7e-7). Why the layers matter: a per-option pointer is capped near 0.5 on a relational task (odd-one-out) while the 2-layer head scores 0.999 [M, synthetic].
5. **Capacity matches what has worked.** Laya's head is about 25M parameters [A, metask] and the completion judge's 26.5M [A]. Ours is 34.5M at the full config [M]. With a few thousand to tens of thousands of decisions, that is regularised by dropout 0.1, weight decay 0.01 and early stopping.
6. **The frozen-decoder result is partly answered (see section 10: AnyJev).** Earlier survey text follows, written before that repo was found: Every surveyed decoder-based decision model tunes the backbone (LoRA r16 to r32 or full). The closest frozen precedents are encoders (the judge, Ariadne). Frozen head-only on a 120B decoder is the main research risk; gate RB4 measures it. Adapting the backbone is not feasible here: caching the late-layer inputs for 2k-token samples is about 11.5 MB per sample per layer-input [E], so 5k samples would exceed the free disk. If RB4 fails, **stop and ask**.
7. **Loss = proper scoring rules.** Soft cross-entropy + spherical (0.75) + ranked probability on ordinal rows (1.0), the weights in Laya's recipe [S]. The judge trained with log + spherical and no RL [A], so Laya's GRPO term is not needed. One-hot targets produce peaked, over-confident probabilities [A, schema-scorer], so targets are soft where a teacher or a spread exists.
8. **Calibration is per (primitive, option-count bucket).** Refitting per (type, option count) moved Laya's ECE 0.466 to 0.081 [A]; its shipped buckets run from 0.10 to 1.98 [S]. Fit on a calibration slice held out **before** training: Laya fitted on training rows and got near-1.0 temperatures (issue #186) [A]. Use at least 2,000 rows (Jev-Style v3) to 6,000 (MacJev) [A], calibrate on the target distribution (Tiny-Jev OOD ECE 0.299) and report out-of-fold ECE (SemIf) [A]. Teacher outputs are recalibrated on our data before they become targets.
9. **The abstain/escalate gate must earn its place.** Laya's act head read about 1.0 on almost every input and ran against correctness, while plain confidence reached AUROC 0.77 [A]. Ours trains on real labels with confidence features and ships only if its held-out AUROC beats max-probability confidence. Initial cost rule from Laya's config [S]: `escalate 0.5`, `cost_wrong_act 3.0`, so act iff P(correct) >= 1 - 0.5/3.0 = 0.833, to be tuned and pre-registered.
10. **Optimisation recipe.** AdamW, head LR 1e-4 (Laya, laya-code, conductor; Ariadne's 3e-4 was unstable, 70.23% mean to 75.28% when lowered [A]), weight decay 0.01, clip 1.0, effective batch 32 to 64, minimum epochs then early stopping with patience 3 on validation soft cross-entropy, 5 seeds (0 to 4), selection rule written before the sweep (Ariadne, MacJev) [A].
11. **Coding supervision cannot come from the Laya teachers.** Base Laya is near chance zero-shot [A], laya-code reaches AUROC 0.713 [A], Cortex-1 drops to 32.7% outside its lane [A]. Measured here: the conductor scores 0.340 on the 2,000 typed-decisions test decisions (random 0.318; choice slice 0.203, below chance) against 0.771 for the checkpoint trained on them [M, `results/raw/laya-eval-laya-conductor-20261001-154307.jsonl`]. Teachers give soft targets only inside their lane (effort, completion, relevance, safety). ACTION, PATCH and RECOVERY labels come from trajectories and execution. A weaker zero-shot teacher also hurts: kotoba's teacher lost to its student in-domain by 15 points on a 77-way task and distilling its hard labels **lowered** in-domain accuracy; the teacher was only worth using where no gold existed [A].
12. **Shortcut audit before training.** A TF-IDF + logistic-regression baseline on candidate text must be near chance or reported (it scored 83.6% on prompt injection against the adapter's 88.8% [A]). Candidate metadata excludes `source`. Negatives are real actions from elsewhere, not templates.
13. **Known failure shapes get their own slices.** Negation and yes/no slot dominance (#156: use a two-option choice with neutral slots and swap the yes/no order), options over about 11 (#394), contrastive pairs, "unknowable" controls scored by the rate answered at p >= 0.9 (Jeeves) [A]. **Prompt injection inside the state:** one adversarial line cut AlexWortega's accuracy on 150 JevBench items from 0.833 to 0.467 and raised the share of deny-worthy shell commands it allowed from 17% to 85%; its authors say a guard built on it belongs next to deterministic checks [A]. Training sets include injected-state contrasts (Tiny-Jev) and the policy layer stays authoritative.
13b. **Leakage hazards when building examples.** Two questions about one state in one example leaked: a yes-noul naming X gave away the choice answer and train loss hit 0.000, so kotoba split them into separate examples [A]. One question per example and per pass. Option-shuffle and paraphrase augmentation on a corpus whose options are identifiers broke the state-option correspondence and the model stayed at chance (loss 1.5 to 1.7) [A]; any augmentation of coding candidates is checked with a learning curve before use. ModernBERT-base also diverged on that code task (loss 1.5 to 6.1), so seed-to-seed instability is expected and 5 seeds are required.
14. **State packing.** Tail-first, tokenise each segment separately so span positions are exact (Jev-Style v3), and raise on overflow instead of truncating silently (MacJev) [A]. Request shape changes accuracy (Jev yes/no 0.843 alone vs 0.788 with other questions [A]), so one question per pass.
15. **Completion is the hardest decision.** About 60% of laya-browser's failures are stopping early; every remedy tried stayed within noise [A]. Tests and invariants stay authoritative (already in CLAUDE.md).
16. **Layers.** The head mixes K captured layers with learned weights. Which K layers, and whether an earlier layer is better, is a sweep on real states. Reading an earlier layer also shortens capture (36 layers to L) [E].

## 3. The head

**Capture contract** (one record per decision per capture): `h_query [K,H]`, `h_opt [N,K,H]` as a mean over each option's own tokens (optionally also the last token), `mask`, `prim`, `kind`, `meta`, target distribution, option order, the prompt hash, model and tokenizer revisions, the layer indices, the lm-head letter logits at the query position (the "gpt-oss alone" baseline, free from the same pass, FP32 rows as Jev-Style does), split label. Stored as bf16 safetensors shards. N <= 16. NONE is a normal candidate.

**Architecture** (`decision_head/head.py`): per-layer feature scaler fitted on the training split → learned softmax mix over K layers → linear to d → add primitive and kind embeddings (and metadata) → set attention over [query, options] (2 pre-norm layers, d // 64 heads, FFN 4d, dropout 0.1) → score each option with a pointer dot product, an MLP, or both → masked softmax. Gate head on the query slot plus [top1, margin, entropy, n/255].

**Primitives:** choice, score (an ordered choice; expected level reported) and noul (a two-option choice with neutral slots). Decision kinds: ACTION, EVIDENCE, CONTINUE, RECOVERY, PATCH, COMPLETION, ESCALATE, ABSTAIN.

**Capture design** (primary, B): prefix KV computed once per decision; one suffix row per candidate against that prefix; the query token at the end of the prefix. The numerics of the backbone are unchanged. Needs a "shared prefix, batched suffix" mode in the runtime's layer loop: a runtime feature to specify before R3, not a change to it. BF16 shared-prefix scoring changed 5 to 6 of 777 argmaxes against fresh scoring in SemIf [A], so the equivalence gate records this noise floor.

## 4. Parameters

### 4a. Frozen backbone (`gpt-oss-120b/config.json`) [S]

Hidden size 2880; 36 layers alternating sliding attention (window 128, even indices) and full attention (odd indices, 18 layers); 64 attention heads, 8 KV heads, head dim 64; 128 experts, 4 per token; vocabulary 201,088; yarn RoPE factor 32, theta 150,000, 131,072 positions; attention bias on. The capture reads layers 0 to L_max (embeddings, attention, router, experts); the final norm and the option-letter rows of `lm_head` serve the baseline. **Nothing is modified, re-quantised or fine-tuned.**

### 4b. Trainable (measured by instantiating the module) [M]

| Component | Parameters |
|---|---|
| Layer mix (q, o) | 6 |
| Input projections 2880 to 1024 (q, o) | 2.95M + 2.95M |
| Set attention, 2 layers, d=1024, FFN 4096 | 25.19M |
| Pointer maps (q, k) | 1.05M + 1.05M |
| MLP scorer | 1.05M |
| Primitive, kind, metadata embeddings | 0.02M |
| Abstain/escalate gate | 0.26M |
| **Total (K=3, scorer both)** | **34.53M** |
| Jeeves form (layers 0, d=256, pointer) | 1.55M |

Scaler buffers (mean and std per layer and dim) are fitted, not trained.

### 4c. Hyperparameters

| Parameter | Value | Source | Status |
|---|---|---|---|
| d / layers / heads / FFN / dropout | 1024 / 2 / 16 / 4096 / 0.1 | Laya `DecisionModel` | ablate 256, 512; layers 0, 1, 2, 4 |
| Scorer | pointer + MLP | Jeeves + Laya | ablate pointer, mlp, both |
| Option input | span mean (+ last token) | kotoba ablation | ablate marker token, last token |
| Captured layers K | 3 | none | **layer sweep on real states first** |
| Option layout | shared prefix, independent suffix | AlexWortega | ablate single sequence in K orders |
| Loss weights | log 1.0, spherical 0.75, RPS 1.0, gate 0.1 | Laya recipe (gate: mine) | ablate CE + Brier |
| Optimiser | AdamW, head LR 1e-4, wd 0.01, clip 1.0 | Laya / Ariadne | fixed; LR 3e-4 known unstable |
| Batch | 32 to 64 effective | Laya / Ariadne | fixed |
| Stopping | min epochs, patience 3, val soft-CE | Ariadne | fixed |
| Seeds | 0 to 4, selection rule first | Ariadne / MacJev | fixed |
| Calibration | per (primitive, bucket), bucket = 2 / 3-5 / 6-10 / 11+, min 100 rows per cell | Laya | fixed; slice held out first |
| Calibration size | >= 2,000 rows, target 6,000 | Jev-Style v3 / MacJev | fixed |
| Candidates per decision | 5 to 15, cap 16 | runbook; Laya degrades past about 11 | fixed |
| Gate rule | act iff P(correct) >= 0.833 | Laya config costs | tune on calibration split |
| Hard / soft target mix | not set | none | pre-register before training |
| Option-order views | 2 to 3 if design A | metask (3) | only for design A |

### 4d. Not parameters

Backbone weights, expert layout, quantisation, KV precision, the tokenizer, the harmony template.

## 5. Data

- **Labels, in authority order:** a dataset's recorded execution outcome (episode-level) > the action actually taken in a resolved run > in-lane Laya teacher soft targets > heuristics. Own execution counterfactuals only where a sandbox exists (none yet on this machine). Be explicit that trajectory-observed actions are not independently executed here.
- **Sources:** R2E-Gym SFT first (56 MB, under the 1 GB rule; the dataset-card license is blank, so check before it counts as provenance under RB2). SWE-Hero, SWE-rebench and SWE-smith each exceed 1 GB and need your approval; SWE-smith's three formats are one set of tasks and must be deduplicated. `LocalLLaMA/typed-decisions` (2 MB) for plumbing, regularisation replay (Jev-Style v2 kept about 50% replay [A]), calibration practice and as an external benchmark; its ceiling is 0.735 teacher self-agreement and scores above that mean learning teacher quirks.
- **Splits:** by repository and commit lineage, never by row. Separate development, calibration and test sets, and a repository-disjoint OOD set. The test split is never used for selection.
- **Negatives:** real actions from other steps or trajectories. Templated negatives let the backbone learn "synthetic".
- **Size [E]:** about 2,000 tokens per decision under design B (1,500 prefix + 12 × 40 suffix). At PLAN's unmeasured 470 tokens/s that is about 1.2 h per 1,000 decisions and about 12 h per 10,000. Storage about 0.22 MB per decision at K=3 in bf16, about 2.2 GB per 10,000 (twice that with a second option stream). Pilot at 1,000, then a learning curve decides how many more.

## 6. Ablations and gates

Order: (1) layer sweep and span vs last token vs marker token; (2) design B vs design A; (3) layers 0 / 1 / 2 / 4 and scorer; (4) K layers; (5) loss variants; (6) gate vs confidence baseline; (7) 5-seed final.

Metrics: accuracy, soft accuracy (vs soft targets), NLL, Brier, ECE (15 bins, calibrated and out-of-fold), AUROC for the gate and for correctness, option-shuffle flip rate, "unknowable answered at p >= 0.9" rate, accuracy by option count and by repository (OOD), long-state slices, negation and injection slices.

Baselines on identical candidates and prompts: random / first / label-frequency prior, TF-IDF + logistic regression, gpt-oss letter-logit readout, Laya typed-decisions and in-lane specialists as released, each with refitted temperatures. Proposed pre-registration form (thresholds go in `configs/thresholds.toml` before any run): the head must beat the best of those on held-out repositories with a paired-bootstrap interval that excludes zero, for 5 seeds, and its gate must beat plain confidence.

## 7. Risks and unknowns

- **Frozen decoder ceiling** (lesson 6). No surveyed repo shows head-only success on a frozen 100B-class decoder.
- **Untrained marker embeddings** (lesson 1) and **prompt-format drift**: the backbone has never seen this layout.
- **Coding teachers are thin.** In-lane specialists only; much of the signal must come from trajectories.
- **Throughput** is unmeasured (G5). Capture hours above are estimates.
- **Machine memory.** At 2026-10-01 14:55 the commit limit was 51.7 GiB with 2.35 GiB free [M]. One process, Ableton Live 12 Suite, held about 20 GB of commit [M]. That caused `MemoryError` in the conductor and judge downloads (the Hub CLI printed the traceback and still exited 0, so downloads are now size-checked after every attempt). After Ableton was closed, commit free rose to 6.88 GiB and the downloads completed [M]. Keep it closed during long downloads or any 120B run.
- **Laya's act head is dead in the shipped checkpoint.** `act_probability` was exactly 1.0 on all five answers of the first typed-decisions test case [M], matching issue #185. The loader also warns that the shipped `choice:11+` temperature (0.1006) is outside [0.5, 5] and clamps it [M].
- **Disk.** D: has 19.5 GB free [M]; about 14.4 GB of it is stale `gpt-oss-20b-heretic` download partials that need a yes to delete.
- **AVX-512 kernels in torch 2.14 CPU builds raise access violations** on this Ryzen; `ATEN_CPU_CAPABILITY=avx2` ran clean [M]. Any CPU-side torch job must set it, or it can crash silently.
- **Author-reported numbers** in section 1 were not reproduced. Laya typed-decisions on its own test split (expected about 0.766) is the first reproduction to run.

## 8. Order of work

1. Done: head module, self-check, tokenizer facts, parameter counts, survey.
2. Capture plumbing on a tiny random `GptOssForCausalLM` (CPU, no download): segment-wise tokenisation, span positions, prefix/suffix batching, and the invariance tests.
3. Download `LocalLLaMA/typed-decisions` (2 MB), reproduce Laya typed-decisions' 0.766 on its test split (verifies the Laya install end to end and fixes the evaluation code).
4. Parse R2E-Gym SFT into decision records, with the shortcut audit.
5. Runtime G1, G2, G3 (PLAN.md), then the pilot capture, layer sweep and learning curve.
6. Train, calibrate, evaluate against the baselines (RB4).

## 9. Reality checks on a "millions of decisions, full teacher ensemble, execute every candidate" plan (2026-10-01)

- **Capture sets the data ceiling.** At about 2,000 tokens per decision and 500 tokens/s [E, unmeasured]: 10k decisions about 11 h, 50k about 2.3 days, 1M about 46 days. So tens of thousands at most; a learning curve (10k, 25k, 50k) decides when to stop.
- **Labels are mostly trajectory-observed.** Executing alternative candidates needs a per-repository sandbox that does not exist here, so counterfactual execution is limited to a small pilot.
- **Teachers are lane-scoped.** Conductor scores 0.340 outside its lane [M]; weaker-teacher distillation lowered in-domain accuracy in kotoba [A]. Evidence Gate, Supersession, Gyra, MacJev and Vision-Laya were not found or verified. Teachers give in-lane soft targets and benchmark baselines, not the main labels.
- **Stage 2 (LoRA on the backbone) is not available locally** for a 41.8 GB BF16 model on 6 GB VRAM. A head-only failure at RB4 means stop and ask.
- **Training is the cheap step**: a 34M-parameter head on cached states takes minutes to hours.

## 10. GitHub cross-check (2026-10-01): what changes the fastest path

**nokia-applied-research/AnyJev (Apache-2.0)** [A, docs read]: a frozen decoder LLM plus a **closed-form head on the residual-stream vector at the last prompt position, read at about 2/3 depth**; no gradients, weights untouched. 100 to 300 labels per question, solved in seconds on a CPU (shrunk LDA or ridge, 5-fold choice, temperature fitted out of fold). On typed-decisions: Qwen3-1.7B at 18/28 blocks 0.730, 4B at 24/36 0.786 (fine-tuned Laya 0.768, measured by them), 8B 0.771, the MoE Qwen3-30B-A3B at 40/48 0.799. Truncating to 2/3 depth was faster (0.68x to 0.70x of a plain forward) and **a middle block beat the last one**. Zero-label L0 (average over option rotations) cut the order-flip rate from 0.230 to 0.073. Limits: one head per question and per model (no transfer), at most 26 options, not tested inside an agent loop.

What it changes here:
1. **Head-only on a frozen decoder has precedent**, MoE included. The risk that remained is the harder claim: one shared head over variable candidate lists must generalise, which AnyJev does not show.
2. **Two-tier plan.** Tier 1: `decision_head/closed_form.py` (built, self-checked on synthetic data) per fixed-layout decision (CONTINUE, COMPLETION, evidence yes/no, small-K choices) on about 300 labelled decisions each, fitted in seconds. Tier 2: the trained set-attention head for variable candidate lists (ACTION, PATCH). Tier 1 is also the baseline arm Tier 2 must beat.
3. **Capture at about 2/3 depth, not the last layer** (heretic: about block 16 of 24), with a depth sweep at 50/60/70/85/100% chosen on calibration data only. That cuts capture cost by about a third [E].
4. **Order debiasing** has a cheap zero-label option (rotations) as an arm next to the shared-prefix design.

**Datasets, by recorded outcome** [M, column names read]: `R2E-Gym SFT` has only `messages` (no outcome flag). `SWE-rebench-openhands` has `resolved`, `exit_status`, `gen_tests_correct`, `pred_passes_gen_tests`, `model_patch`. `SWE-smith` has `resolved` and `patch`. `SWE-Hero` has `model_patch` but no resolved flag. So the execution-grounded pilot comes from SWE-rebench-openhands (a few hundred rows via the datasets server is enough for 1,000 decisions) or SWE-smith, not R2E-Gym alone.

**Machine facts** [M]: no Docker, WSL not installed, so SWE environment images (about 300 to 500 MB each, Docker-based) cannot run here: execution-based verification is limited to the datasets' own recorded results. Node v24.13.1 and git are present, so the DeepSeek Harness (TypeScript) is feasible.

**Reuse instead of rebuilding:** pilotspace/laya-codex (tree-sitter + BM25 + Laya reranker, the retrieval step), SWE-Gym and R2E-Gym (execution-based plus execution-free verifier idea, best-of-n selection), OpenHands critic, ruban-24/switchboard (effort router using Jev or Laya). Not read in depth yet.

## 11. Built for local models in general: what is shared and what is per model (2026-10-01)

The head is a recipe plus a per-model fit, not one set of weights for every model. A head reads one model's own hidden-state space,
so it cannot move to another model (AnyJev says the same: one head per question and per model [A]). Everything upstream of the
hidden state is model-independent.

| Part | Shared by every local model | Per model |
|---|---|---|
| Decision data (episodes, candidates, negatives, labels, locked benchmark) | yes | |
| Teachers and judges (Laya typed, conductor, judge, code, cortex): they read text, not hidden states | yes | |
| Hard policy, completion invariants, state and candidate contracts, calibration code, head code (`hidden` is a parameter) | yes | |
| Prompt text, built with the model's own chat template | | yes |
| Option spans, found by segment-wise tokenisation (exact for any tokenizer) | | yes |
| Capture depth as a fraction of the layer count (start at 0.67, sweep 0.5 to 1.0 on calibration data) | | yes |
| Layer-streaming loader for the model's architecture (transformers class) | | yes |
| Letter-readout baseline (needs single-token option letters, checked per tokenizer) | | yes |
| Hidden size, layer count, revision pin | | yes (read from its config) |
| Trained head, closed-form heads, scaler, temperatures | | **yes, one set per model** |

Cost of adding a model: capture its states for the shared decision set, fit the closed-form heads (minutes), train the pointer head,
calibrate, run the same benchmark. Nothing about data or teachers is redone. Order of backbones: `gpt-oss-20b-heretic` first; a small
second model next to prove nothing is gpt-oss-specific (`Qwen/Qwen3.5-4B` is already in the local Hugging Face cache; not opened).
Rules for the code: no constant in `decision_head/` or `state/` may name a model, a layer index, a hidden size or a token id.

## 12. Interface inspection of laya-typed and cortex-1 (2026-10-01): both are fixed-layout specialists

**laya-typed-decisions** [M]. Input: a state (text or JSON) plus a question dict (type, instructions, criteria id to description). A choice answer is a softmax over the options (sums to 1). The architecture takes variable option lists, but the checkpoint is a specialist for four synthetic workflows. On the 600 typed-decisions choice questions: accuracy 0.740; the answer changes on 5.2% of option-order shuffles; adding distractor options taken from other questions lowers accuracy to 0.663 (+3), 0.612 (+6) and 0.532 (+10), with 0.30 to 0.48 of the probability on options that are obviously wrong. Options share a fixed head budget, so counts above about 5 degrade. Role: a baseline arm for small candidate lists in A6, not a teacher. Its coding accuracy is unmeasured.

**cortex-1-large** [M for probes, S for its code]. Loads and behaves identically to its model card's own code path (same probabilities through laya.Agent and `build_sequence`), so the loader is not the problem. Its contract is a catalog of about a dozen fixed question templates (`should_autopilot`, `risk_score`, `route_task`, `vulnerability_class`, ...). Its training data is hand-written: roughly a dozen high-risk and a dozen low-risk templates cycled with only a ticket reference and metadata line changed, and the training state includes a `metadata` field that holds the label (`risk_tier`, `cwe`, `bug_category`, `ml_root_cause`). Its own validation reports balanced accuracy, sensitivity and specificity of exactly 1.0. Probes: `should_autopilot` returned P(yes)=0.000 for 8 plausible task descriptions under 3 wordings (raw logits about [6.2, -4.0] regardless of input); `vulnerability_class` was right on 0 of 3 snippets at 4, 7 and 11 options; adding only a `risk_tier` metadata field moved P(yes) from 0.000 to 0.993 on a low-risk text; `route_task` changed its answer on 3 of 8 order shuffles. Role: rejected as a teacher. Its published coding results come from author-built sets with the same generators and are not trusted.

Lane registry corrected: `laya-typed` and `cortex-1` now have empty lanes (no trusted teacher role). A3 teacher wrappers are therefore: judge, conductor and laya-code done; laya-typed as a baseline wrapper is open; cortex-1 gets no wrapper.

## 13. Research survey for the plan (2026-10-02): papers, Laya weights, cheap training

Tags: [A] author-reported, from abstracts and search results, not full reads and not reproduced; [M] measured here. Plan: PLAN.md section 12.

**Laya (Apache-2.0).** Source github.com/NandhaKishorM/laya; HF `convaiinnovations/laya` (ModernBERT-large 421M, 512 tokens), `laya-multilingual` (mmBERT-base 322M), `laya-typed-decisions` (421M, 1,024 tokens);
dataset `LocalLLaMA/typed-decisions`. Specialists: `mvilacad/laya-conductor`, `tampajohn/laya-stop-completion-judge`, laya-code (from pilotspace/laya-codex, top 8 of 28 layers, about 2.3 h on an M4 Pro [A]),
`cklxx/laya-browser` (browser-agent head with a write-up, not downloaded). A "Gyra" checkpoint was not found. Training [A]: RLCD (soft cross-entropy against the teacher plus REINFORCE with a group-mean baseline and
proper-scoring rewards), one temperature per type fitted afterwards; 4-5 h for 4 epochs over about 30k questions on free Kaggle 2xT4; the browser head about 2 h on one 16 GB GPU. Lessons [A]: candidates shown in full in the
option list beat every data change; templated labels leak; confidence-gated escalation to a bigger LLM made results worse; `torch.compile` on variable shapes was 6x slower. Laya's window cannot read our 8k-token states, so it stays a baseline.

**Papers and weights.** Hidden-state correctness: arXiv 2606.14530 (Qwen3-4B-Instruct-2507, AUC 0.881, 0.842 after removing prompt-length effects [A]); 2512.22245 (Brier-loss linear probes, cheap calibrated judges).
Verifiers and critics: OpenHands critic 32B (HF OpenHands/openhands-critic-32b-exp-20250417: temporal-difference step labels from the final outcome plus a regression head); SWE-RM 2512.21919; R2E-Gym 2504.07164 (hybrid verifier, 51% vs 43.7/42.8 [A]);
SWE-TRACE 2604.14820. Same decisions as ours: early terminal reward prediction 2609.31995 (stop/continue from prefixes), SWE-Router 2607.00053 (weak-to-strong escalation, releases trajectories).
Data: nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0, 67,074 trajectories, 1,823 repositories, 3,792 resolved issues, 32,161 successful trajectories [A]), SWE-Gym 2412.21139, SWE-smith 2504.21798, SWE-rebench 2505.20411 and V2 2602.23866.
Baseline weights: Qwen/Qwen3-Reranker-4B (Apache-2.0, yes/no last-token logit).

**Training faster or cheaper.** Cached heads on frozen states (already). LoRA on all layers including MLP, learning rate about 10x full fine-tuning (thinkingmachines.ai/blog/lora); Unsloth claims 2x speed and 70% less memory [A], compatibility with our packed block mask untested.
Free GPU: Kaggle 30 GPU-h per week, 2xT4 bills double, 9-12 h sessions; T4 has no native bf16. Rejected for now: shared-prefix reuse (Tree Training 2511.00413, psRL 2608.25683, Hydragen 2402.05099) because only 35.4% of prompt tokens are shared between consecutive decisions
[M, `results/raw/prefix-overlap-train-20261002-205012.json`], and FlexAttention (2412.05496) for at most 15-20% of FLOPs and changed numerics. Section 9's claim that LoRA is not available is superseded: LoRA on a rented 24-48 GB GPU is in the plan (Phase 4).

## 14. Status and corrections, 2026-10-03
- **Backbone frozen throughout; only the head is trained.** LoRA (PLAN.md 12.5c) is parked until the Phase-3 probe and the user's request. The research (Kaggle TPU v5e-8, Tunix/Qwix, PyTorch/XLA) is recorded there with sources.
- **Compute:** capture of TRAIN and CAL on Kaggle T4 x2 (fp16, resident), one device class for all roles. Head training is small (H0 about 1.1M to 5.0M parameters, residual about 2.0M to 7.9M, by projection 128/256/512; parameter counts from `decision_head/v3_model.py`) and the trainer defaults to CPU; its duration is unmeasured.
- **Correction:** the earlier note "2xT4 bills double" is not supported by the Kaggle quota readings (01:01 to 01:13 to 01:20 of 30 h across v6 and v7, consistent with 1:1 wall-clock); an inference from three readings, not a confirmed rule.
- **Open design points for the user** (details in PLAN.md 12.7 and the amendment (e) draft): training target (next-action agreement for A5-v3 vs real outcomes later), loss (cross-entropy primary, Laya Brier as a secondary arm), go/no-go stop rule, where to train.
- **External yardsticks** (BENCHMARKS.md section 8): published outcome critics reach AUC about 0.58 to 0.69 on real data; Laya's 0.766 is on its own training split (in-distribution).

## 15. Laya research, 2026-10-03 (primary sources; no Laya training is done or approved, CLAUDE.md rule and PLAN.md 12.6 item 4 stand)
- **What Laya is** (github.com/NandhaKishorM/laya README, HF convaiinnovations/laya): non-autoregressive decision engine, three primitives (choice, score, noul), trained with RLCD against strictly proper scoring rules, Apache-2.0. Checkpoints: `laya` (ModernBERT-large 421M, 512 tokens), `laya-multilingual` (mmBERT-base 322M, 1,024 to 8,192 tokens), `laya-typed-decisions` (fine-tuned). Stated limits: base checkpoints near chance zero-shot (0.362 / 0.352 vs random 0.318, majority 0.461), weak on high-cardinality choices (50+ options; our pools average 45), calibration needs temperature fitting on domain data.
- **Weights we hold locally (sha256-verified):** laya-typed-decisions, laya-conductor, laya-stop-completion-judge, laya-code, cortex-1-large. Community fine-tunes on HF include mvilacad/laya-conductor (4,105 real agent turns, labels by Claude Sonnet) and hxrikp/laya-session-guard-pilot; dataset LocalLLaMA/typed-decisions and benchmark Luni/laya-jev-benchmark.
- **Official fine-tuning recipe** (laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): 1,200 training cases / 6,000 decisions, 4 epochs, effective batch 64, max length 1,024, fp16 autocast on 2xT4, about 4-6 minutes. Loss = spherical proper-scoring reward (weight 0.75) + GRPO-style policy gradient (group 4, exploration noise 0.4 to 0.1) + soft cross-entropy (1.0). Learning rates encoder 2.5e-5, head 1e-4, cosine; 10% calibration split held out BEFORE training; temperature per question type by LBFGS.
- **Published calibration:** typed-decisions ECE 0.213, Brier 0.061 (README); a pentest-harness paper reports ECE 0.081 after temperature scaling on its own set (arXiv 2609.28940, different task).
- **Not Laya papers:** arXiv 2503.23303 and 2510.01237 were wrongly attributed to Laya by a search summary; they are about a sales-conversion agent and LLM hallucination routing. No Laya methods paper was found.
- **Why Laya cannot just read our states:** 512/1,024-token windows vs 8k-token states; laya-typed scored 0.2277 on A5-v1 with a 2,400-character state (a different task from its training).

## 16. Deep Laya research: ecosystem, recipes, data, papers (2026-10-03; all from primary pages unless marked; summaries by the fetch tool, numbers are authors' claims [A])
**Weights (all Apache-2.0 unless noted):** base encoders ModernBERT-large/base (answerdotai, 8,192 ctx via RoPE theta 160k, arXiv 2412.13663) and mmBERT-base (jhu-clsp, arXiv 2509.06888); Laya `laya` (421M, 512 tok), `laya-multilingual` (322M, 1,024-8,192 tok), `laya-typed-decisions`; **MacJev-322M-4K-Laya** (chaoliangUNSW; 4,096-token context: trained last 6 encoder layers + type embeddings + decision transformer + scorer from the official head; staged 1K to 4K; ordinal loss, verified-output replay, candidate-permutation consistency; long-input yes/no 31% to 89.1%, ECE 0.253 to 0.032); `laya-lora-modernbert-r8` (LoRA r=8 alpha 16 on fused QKV, 540,672 trainable = 0.388%, plus Noul/Choice/Score heads); `laya-modernbert-decision-90pct` (90.61% on its own held-out split of an expanded corpus: in-distribution, not independent). HF lists 100+ Laya derivatives (GGUF, ONNX, CoreML, LiteRT).
**Datasets:** LocalLLaMA/typed-decisions (1,200 train cases / 6,000 decisions, 400 test cases / 2,000 decisions); annelo/laya-marker-corpus (95,817 rows, BIG-bench + tables, mixed licences); callensxavier/laya-coding-curriculum-78k (79,003 rows, security/stubs/energy/Lean; labelling method not stated; NOT agent next-step decisions); Luni/laya-jev-benchmark, pranaysuyash/laya-formatting-fragility (480 cases; Laya order-flip rate 22.5%). Agent outcome data: nebius/SWE-rebench-openhands-trajectories (67,074 trajectories, 32,161 resolved / 34,913 unresolved, 2.08 GB, CC-BY-4.0, avg 64.3 turns, fields resolved/exit_status/model_patch); SWE-Gym/OpenHands-Sampled-Trajectories (6,055 rows, 301 MB, empty card).
**Official recipe** (laya/notebooks, Kaggle 2xT4): 4 epochs, effective batch 64, max length 1,024, fp16 autocast; loss = spherical reward (w 0.75) + GRPO-style policy gradient (group 4, noise 0.4 to 0.1) + soft CE (1.0); LR encoder 2.5e-5, head 1e-4; about 4-6 min for 6,000 decisions; 10% calibration split held out BEFORE training; temperature per type (and per option-count bucket, MIN_BUCKET_N 2,000) by LBFGS; optional histogram binning and `fit_abstention_thresholds`. Training code in the repo is notebooks plus a data builder; no standalone RLCD trainer was found.
**Cautions from the sources:** Laya weak at 50+ options (our pools average 45); confidence AUROC only 0.72 in the Spanish phone study (confidence is a weak gate); fully fine-tuned Laya changed 42% of answers on untrained questions (anth.us); INT8 flipped 24/221 decisions; tuning variants after looking at test results (their v4/v5) inflates scores; cbjev one-pass layout (order flips 7.8% to 0.2%, ECE 0.125 to 0.117) is GPL-3.0, so reimplement the idea, never copy code.
**Papers for the hidden-state approach:** code correctness linearly decodable from Qwen3-4B-Instruct-2507 hidden states, AUC 0.881 (0.842 length-controlled, length-only 0.657; 444 LiveCodeBench tasks, arXiv 2606.14530); agent success probes on Bash/SQL/Python (arXiv 2609.09448); recall-controlled probe cascade, 54.9% to 60.2% fewer tokens at 90% recall, behaviour-only monitors underperform hidden-state probes (arXiv 2607.06503). Critics/PRMs: SWE-PRM 40.0% to 50.6% (arXiv 2509.02360), SWE-TRACE candidate pruning (arXiv 2604.14820), SWE-RM (2512.21919), R2E-Gym (2504.07164), OpenHands critic.

### 16.1 Already-trained Laya heads for coding-agent decisions (search 2026-10-03; cards self-reported [A]; none targets "next action among ~45 candidates over an 8k state")
| Model | Decision | Training data and cost | Reported result | Notes |
|---|---|---|---|---|
| tampajohn/laya-stop-completion-judge (WE HOLD IT) | agent ended its turn with work incomplete | 2,627 real Claude Code turn pairs; head only (26.5M params, encoder frozen); 8 epochs, strictly proper scoring | AUROC 0.85 (base 0.64), acc@0.5 0.87, ECE 0.057; block at p >= 0.75 | <= 1,024 tokens, English; domain shift on interactive chat; needs its state_pack.py tail-first packing |
| laya-code (we hold a copy; card seen: tindang/laya-code) | is this code chunk relevant to the change | about 50,900 weak-supervision pairs from git history, 8 repos; 2.3 h on an M4 Pro, half an epoch | AUROC 0.677 to 0.709, ECE 0.016 to 0.060 | 128-token chunks, low probabilities, threshold near 0.4 |
| mvilacad/laya-conductor (WE HOLD IT) | routing tier for coding-agent turns | 4,105 turns labelled by Claude Sonnet; 4 epochs, about 30 min on a T4 | routing accuracy 87% (base about 37%) | our measurement out of lane: 0.3395 on typed-decisions |
| Jojoarumugam/laya-agentguard | destructive tool call, prompt injection | 1,600 synthetic examples; 6.5 min on an M4 Pro; top 12 of 28 layers + head | accuracy 0.804 / 0.796, ECE 0.107 / 0.129, below its own 0.85 / 0.10 bar | shadow mode only |
| Gowtham-R-2002/gyra | destructive command, hang, injection | not disclosed | caught 18/24 destructive, 29/30 hangs; rule layer needed | misses 6/24 |
| hxrikp/laya-session-guard-pilot | allow / block / review actions | 480 synthetic sessions | 100% on templated, 70.8% content / 54.2% action on separate challenge set | explicitly not for authorisation |
| wxsys/code-oracle-laya-421m | approve / reject a code change from an AST graph | 2,400 mutation samples | no results stated | unverified |
| danielamitay/laya-en-fp32-swev, Mukul-svg/laya-swe | conversion only / agent-memory controller | n/a | no head metrics | not decision heads for our task |
Training-cost range seen in the ecosystem: 6.5 min (1,600 examples, M4 Pro), about 30 min (4,105 turns, T4), 2.3 h (50.9k pairs, half an epoch, M4 Pro). Official 2xT4 recipe: 4-6 min for 6,000 decisions at <= 1,024 tokens. Our long-state case is unmeasured.

## 17. Phase 2 complete, Path A, amendment (e) registered (2026-10-03)
- **Capture:** TRAIN 1,986 and CAL 569 captured on Kaggle T4 x2 (fp16, resident, registered gate cadence), downloaded and verified (`results/raw/phase2-capture-verify-20261003-145647.json`): complete, finite, Kaggle-manifest hashes 2,555 of 2,555, decisions 0-19 bit-identical to the smoke; vs the 3050 files on 1,376 shared decisions: hashes equal, mean cosine >= 0.9999, worst 0.998742 (informational).
- **One device class for every role (Path A):** state-free text embeddings are recaptured on the T4 (TRAIN+CAL pool texts, 12,602; gate cosine >= 0.9999 on every shard); the 34,315-text 3050 artifact (bf16, 14.4 min, TEXTEMB_OK) is kept for cross-checks only. LOCKED states and the 21,713 LOCKED-pool texts are captured on the T4 only after the user approves the LOCKED upload.
- **Amendment (e)** registered before any head is trained: screen = TRAIN grouped CV of H2 minus H0_matched, 3 paired seeds, stop if the point estimate is below 0.02 (go/no-go only); primary LOCKED criterion unchanged (estimate >= 0.05, lower 95% bound > 0, half-width <= 0.04); Brier loss is a secondary arm; head and temperature are hashed and committed before LOCKED is touched.
- **Head design in one line:** H0 = state-free MLP over 80 features and per-depth text embeddings; H2 = frozen H0 score + pointer residual over query and contextual candidate vectors, starting exactly at H0 (decision_head/v3_model.py).
