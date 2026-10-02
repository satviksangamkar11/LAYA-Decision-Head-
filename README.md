# jev-airllm: a decision head for local LLMs

Goal: one internal decision head (a Laya/Jeeves-style pointer head) that reads the hidden states of a **local, frozen
language model** and picks among candidate actions with calibrated probabilities. It is **backbone-agnostic**: the
head takes the hidden size as a parameter and is fitted per model. **Current experiment backbone (user, 2026-10-01): `Qwen3-4B-Thinking-2507`**
(frozen, 8.05 GB, revision `768f209d9e`). `gpt-oss-20b-heretic` and the 120B are parked. Laya is a baseline and in-lane helper, never the ACTION teacher.

Rules: [CLAUDE.md](CLAUDE.md). Design record and survey: [DECISION_HEAD.md](DECISION_HEAD.md). Step list and closure path: [PLAN.md](PLAN.md) section 11.
Tags: **[M]** measured here, **[A]** reported by an author and not reproduced, **[E]** estimate.
Last updated: 2026-10-02 (CAL capture done; research and plan update).

## Current plan (2026-10-02, full detail in [PLAN.md](PLAN.md) section 12)

Goal: the best internal Laya-style decision head on frozen `Qwen3-4B-Thinking-2507`, as cheaply and quickly as possible, with a stop rule before every paid step.
Status: CAL captured (569/569, audited). TRAIN (1,986) and LOCKED (2,220) not captured. No head trained, so **nothing is known yet about head quality**; the pilot
risk is that the frozen state adds nothing over candidate/history features for next-action imitation (0.663 vs 0.667 [M]).

| Phase | What | Compute | Stop rule |
|---|---|---|---|
| 0 | resident-weights capture mode; amendment (e) (3-seed selection, GPU-numerics clause, TRAIN-CV screen); dataset metadata scan; Laya-style proper-scoring loss | local, free | scan or amendment fails |
| 1 | 20-decision smoke with all hard gates, real speed, CAL duplicate spot-check, fp16 check | Kaggle T4 free, else rented Ada/4090 | any gate fails or cost over 2x model |
| 2 | capture TRAIN, LOCKED, CAL recapture; registered H0 vs H2 test; one LOCKED read; baselines (Laya models, Qwen3-Reranker-4B) | Kaggle free or Ada | failure routes to Phase 3 |
| 3 | outcome-label probe (SWE-rebench-openhands) on TRAIN grouped CV | free or Ada | lower bound not above 0: stop spending |
| 4 | scale to about 10k outcome-labelled decisions; LoRA plus pointer head; fresh LOCKED-2 | LoRA on RTX 6000 Ada | ship LoRA only if it beats the frozen head |
| 5 | three-arm SWE comparison with and without the head | needs Docker | not budgeted |

Budget [E]: Rs 0 for Phases 1-3 if fp16 passes on Kaggle; LoRA about Rs 2,000-3,500; proposed cap Rs 6,000 in two steps (pending). **Pending approvals:** provider, cap and
uploads; downloads over 1 GB (SWE-rebench-openhands, Qwen3-Reranker-4B); the private push; whether to lift the "no Laya fine-tuning" rule. Measured so far: 8,163 tokens and
10.2 s per decision on the RTX 3050, compute-bound; consecutive decisions share only 35.4% of prompt tokens, so KV reuse is rejected (`results/raw/capture-cost-model-*.json`,
`prefix-overlap-train-*.json`). Research survey, papers, Laya weights and repos: [DECISION_HEAD.md](DECISION_HEAD.md) section 13.

## Benchmark increments (one row per major step; full ledger: [results/increments.md](results/increments.md))

The fixed benchmark is the locked A5 set (`data/benchmark/a5-v1.jsonl`, 4,396 test and OOD decisions, read-only, hash in `results/raw/`). The headline
metric is accuracy on **covered** decisions against the best baseline there (state-blind TF-IDF, **0.323**). A step that does not move it is recorded as
such. No head has been trained yet, so every row after the baselines shows no change.

| # | Step | Result [M] | A5 covered accuracy | Delta | Evidence |
|---|---|---|---|---|---|
| 0 | Locked benchmark and baselines | random 0.125, first 0.148, longest text 0.257, TF-IDF 0.398 all / **0.323 covered**; laya-typed baseline 0.228 all / 0.239 covered | 0.323 (best baseline) | baseline | `results/raw/a5-baselines-a5-v1.json`, `a5-typed-baseline-a5-v1.json` |
| G2 | 120B expert-read speed (runtime feasibility, not a head step) | best mode 0.87 s per simulated token, 2.18 GB/s, GO (line 1.0 s); I/O only | unchanged | none | `results/raw/g2-expert-read-20261001-132735.json` |
| 1 | Qwen3-4B-Thinking-2507 downloaded; equivalence | v1 rule **FAIL** (top-1 at near-ties, floor was 0); v2 (registered after seeing v1, reason written) **PASS**: streamed vs resident bit-exact on 12 layers, CPU vs GPU-streamed KL 5.5e-4 to 6.6e-4, cosines at least 0.9998 | 0.323 | none | `equiv-qwen3-4b-20261001-145454.json` (v1), `equiv-v2-qwen3-4b-20261001-145942.json` (v2) |
| 2 | Capture throughput | 1,017.6 tok/s at 1x8192 tokens (GO line 1,000, passes by 1.8%); 1,128.9 at 2x4096; peak VRAM 0.8 GiB; weight-transfer share 8% | 0.323 | none | `capture-qwen3-4b-20261001-150329.json` |
| 3 | Capture plumbing on one real decision | all five gates PASS: repeat bit-identical, plain-forward cosine query 1.0 and options at least 0.99993, spans exact, order independence at least 0.99996, lm_head never run | 0.323 | none | `capture-one-20261001-151152.json` |
| 4 | Pilot manifest and Laya per-row fields | 980 decisions (600 train, 380 cal; cal 20 short of target, recorded), test and OOD excluded; Laya-typed on the pilot rows 0.217 all / **0.178 covered**, all states cut by its 640-token window | 0.323 | none | `pilot-manifest-v1.json` (+ `.sha256`), `laya-fields-v1.sha256` |
| 5 | Pilot capture complete and validated | 980 of 980, 0 skips, depths 12/18/24/30/36, 2.98 GPU hours, mean 10.96 s per decision, 670 tok/s mean; 3 recaptured decisions bit-identical | 0.323 | none | `pilot-capture-validation-20261001-182625.json` |
| 6 | Tier-1 closed-form head (`next_action_kind`, sanity arm) | rule registered first; trained on 600 train rows; depth 24 chosen on the 380 pilot-cal rows (lowest NLL); cal accuracy 0.516 vs label-prior 0.213, delta +0.303, 95% trajectory-cluster CI [+0.233, +0.368] excludes 0. All five depths recorded (cal acc 0.500 to 0.539). Not an A5 measurement. | 0.323 | none | `tier1-next-action-kind-20261001-182958.json` |
| 7 | Tier-2 pointer head on ACTION, **pilot only** (rule registered first, `[tier2_action_pointer]` v1) | trained on 600 train rows, 6-cell grid x 5 seeds, selected depth 24 / D 128 on pilot-cal NLL; temperature from out-of-fold train (T = 1.377). **Pilot-cal accuracy 0.616 all / 0.667 covered (n=120 covered)**, NLL 1.254 (1.235 calibrated, ECE 0.095). Baselines on the same rows: TF-IDF full-train 0.384 / 0.308, TF-IDF data-matched (600 rows) 0.290 / 0.333, first 0.118, random 0.121. **Not an A5 measurement and not yet a claim:** the locked benchmark has not been evaluated, and the shortcut audit (row 8) shows a no-model rule already reaches 0.49 on covered rows. | 0.323 | not measured | `tier2-action-pointer-20261001-183544.json` |
| 8 | Shortcut and leakage controls on the pilot (diagnostic) | label-shuffled training 0.108 (chance, no train-time leak); state-shuffled eval 0.616 to 0.234 (the head uses the state-option interaction); metadata-only 0.258; query-zeroed control is degenerate (identical to metadata-only, invalid); **option-only linear scorer with no query reaches 0.821 / 0.775 covered, above the pointer head**, so the query is not what carries the signal. No-model **most-recent-in-pool rule: 0.492 on cal covered, 0.447 on train covered** (true action median pool rank 5 vs 21 for distractors). Word-overlap rules 0.32 (recency-weighted), 0.16 (novelty). **The A5 bar of 0.323 (candidate-only TF-IDF) is too low: a state-aware no-model baseline already exceeds it.** | 0.323 | none | `tier2-controls-20261001-183813.json`, `tier2-control-option-only-20261001-183902.json`, `shortcut-state-overlap-20261001-184015.json`, `shortcut-pool-rank-20261001-184102.json` |
| 9 | Benchmark integrity gate registered; A5-v1 frozen as diagnostic | strongest no-model baseline on pilot cal covered rows is most-frequent-in-history 0.550 (most-recent 0.492, TF-IDF 0.333, metadata-only 0.258, random 0.127; limit random + 0.10 = 0.227): A5-v1 FAILS | 0.323 (historical) | none | `integrity-gate-A5v1-20261001-184428.json` |
| 10 | A5-v2 rules registered; pool audit on the A5-v1 sampler (3,907 train / 816 cal covered rows) | within-decision AUC of the true action: pool rank 0.212, history frequency 0.762, lexical overlap 0.673; an adversarial MLP over eight trivial features reaches 0.663 on cal covered (random 0.149), equal to the pilot pointer head 0.667 | 0.323 (historical) | none | `pool-audit-A5v1-20261001-184836.json` |
| 11 | Pool feasibility for balanced distractors (registered first) | existing pool cannot support per-decision matching: with 3 matches needed, frequency-only 100% (first-time) / 67.7% (repeat), plus action kind 32.4% / 22.2%, plus rank window 1.1% / 3.0%; bar 70% | 0.323 (historical) | none | `pool-feasibility-20261001-185056.json` |
| 12 | A5-v2 sampler `a5v2-odds-1` (spec `a5_v2_spec.toml`, TRAIN only through the firewall, out-of-fold, 5 trajectory-grouped folds) | FAILS the registered gate, not loosened: feature-only attackers on the generated sets A 0.443, B 0.344 (A5-v1 sets: A 0.652, B 0.524; random 0.149; limit 0.249); four AUC violations remain | 0.323 (historical) | none | `a5v2-train-cv-audit-20261001-194503.json` |
| 13 | A5-v3 specification registered; v1 and v2 frozen with their failures | new question: does the frozen state add decision value BEYOND candidate/history features on identical candidate sets (arms H0, H1, H2, Y, S, R) | 0.323 (historical) | none | `configs/a5_v3_spec.toml` |
| 14 | Spec amendment before anything was scored; CPU power simulation | primary claim now **H2 minus H0_matched** (success: lower 95% bound > 0, estimate >= 0.05, half-width <= 0.04, else INCONCLUSIVE); conservative scenario power 0.955 at 400 clusters for a true effect 0.08, 0.46 to 0.53 for exactly 0.05 | 0.323 (historical) | none | `power-sim-v3-20261001-195023.json` |
| 15 | Fresh A5-v3 data downloaded and split by repository hash | 562 fresh rows (438 dropped for already-used repositories); F-dev 118 trajectories / 87 repositories; LOCKED 444 trajectories / 324 repositories, hashed, read-only, unopened | 0.323 (historical) | none | `fresh-v3-provenance.json`, `fresh-v3-locked.sha256` |
| 16 | F-dev CPU feasibility of the frozen sampler (criteria registered first) | USABLE on fresh data: 2,454 covered F-dev decisions, retention 1.000, covered and repeat shares within 1 point of old train; feature-only attackers on F-dev v2 sets A 0.405 / B 0.324 (old 0.443 / 0.344); frozen odds models saved, sha256 20d5d2b8... | 0.323 (historical) | none | `fdev-feasibility-20261001-200544.json` |
| 17 | v3 training design registered; decision manifests built; full-pool capture pipeline built and smoke-tested (not started) | H0 state-free branch vs primary H2 = frozen H0 + residual state branch on cross-fitted scores; 9 registered cells, selection inside TRAIN, 5 paired seeds at the end. Manifests: TRAIN 1,986 / CAL 569 / LOCKED 2,220 (5 per each of 444 trajectories). Smoke test: all hard capture gates pass (independence >= 0.9999 even at a 79-candidate pool); pilot cross-check worst cosine 0.999945 | 0.323 (historical) | none | `v3-manifests.json`, `v3-pilot-crosscheck-20261001-202311.json` |
| 18 | Sampler unified; v3 trainer built and audited (13/14 checks, 1 pending); independence gate stopped the smoke capture (bf16 noise floor, candidates bit-identical when content changes at fixed shape) | No benchmark change. Gate threshold unchanged; amendment awaits the user. Long capture not started. See results/increments.md row 18. |
| 19 | Independence gate amended (content-swap bit-identity + numerical sanity); NONE and early-stopping locks; trainer audit 16/16 | No benchmark change. Capture estimate now roughly 20-24 h (extrapolated from 10 smoke decisions). Awaiting the user's go for the full capture. See results/increments.md row 19. |
| 20 | v3 CAL capture complete (569/569), audited twice | audit PASS twice, same hash 8717cd11...; Gate A max diff 0.0 (62 targets), Gate B min cosine 0.999906. Incident: two capture processes ran at once (42 ids logged twice); files unique and stable, duplicates' bit-identity unprovable (spot-check planned). | 0.323 (historical) | none | `capture-cal-checkpoint-20261002-020211.json`, `-125613.json`, `capture-interruptions.jsonl` |
| 21 | Cost model and plan update | 8,163 tokens and 10.2 s per decision on the 3050 [M], compute-bound; prompt overlap between consecutive decisions 35.4% [M]; GPU costs are estimates. No benchmark change. | 0.323 (historical) | none | `capture-cost-model-20261002-205000.json`, `prefix-overlap-train-20261002-205012.json` |

> **A5-v2 and A5-v3 (2026-10-02):** A5-v2 FAILED the registered gate and was not loosened. Some history/candidate predictability is legitimate behavioural information, so `configs/a5_v3_spec.toml` registers a new question: does the frozen state add value beyond a state-free candidate/history baseline trained on the same rows (primary claim H2 minus H0_matched, minimum effect +0.05, locked set of fresh repository-disjoint trajectories). Fresh data is downloaded and the locked part is closed (row 15). F-dev feasibility passed (row 16). Training design is registered (row 17). Next: the full-pool capture (about 14 to 15 GPU hours, an estimate).

> **A5-v1 status (2026-10-02): FROZEN, DIAGNOSTIC ONLY.** It fails the benchmark-integrity gate (`configs/thresholds.toml` `[benchmark_integrity_gate]`): on pilot cal covered rows a no-model
> *most-frequent-in-history* rule scores 0.550 and *most-recent-in-pool* 0.492, against random 0.127 and a limit of 0.227. The 0.323 TF-IDF reference is historical and is not the bar for any
> head claim. No head result on A5-v1 counts as evidence of decision quality. Next: build A5-v2 with rank- and kind-matched distractors, capture each decision's full pool once, pass the gate on
> train/cal rows only, then register the model family and run the locked evaluation.
> Pool audit on 3,907 train / 816 cal covered rows (`results/raw/pool-audit-A5v1-*.json`): eight trivial candidate/history features with no model and no state predict the true action with 0.663
> accuracy (random 0.149), which equals the pilot pointer head's 0.667. The pilot head showed no signal beyond candidate-pool structure.

## What exists

| Item | State |
|---|---|
| `decision_head/head.py` | The trained head: 34.53M parameters at full size, any hidden size, 9 decision kinds including `CODE_RELEVANCE`. |
| `decision_head/closed_form.py` | Closed-form head (AnyJev's method: shrunk LDA or ridge, 5-fold choice, out-of-fold temperature). The no-gradient baseline arm. |
| `decision_head/selfcheck.py` | 22 checks pass **on synthetic data only** (`uv run python -m decision_head.selfcheck`): maths, masking, order-equivariance, scaler, calibration, losses, learning, closed-form (0.956 from 200 rows, ECE 0.014, chance on random labels). Says nothing about any real LLM. |
| `state/schema.py` | A1: candidate, state and teacher-output contracts. NONE enforced, 2 to 16 options, no outcome without execution, head metadata excludes the candidate source. `TeacherOutput` has two exclusive forms (DISTRIBUTION, INDEPENDENT_SCORES) and a lane registry. |
| `policy/hard_rules.py` | A2: deterministic hard policy and completion invariants. Pattern lists, so they catch what they list; extend, never loosen. |
| `policy/selfcheck.py` | 8 table-driven checks (`uv run python -m policy.selfcheck`): 23 unsafe candidates blocked, 8 safe allowed, lanes name real kinds, teacher forms validated. |
| `teachers/laya_teachers.py`, `teachers/selfcheck.py` | A3, three of five teachers: the judge (COMPLETION), the conductor (CONTINUE, ESCALATE effort) and laya-code (CODE_RELEVANCE: independent per-chunk scores, no softmax, truncation explicit and recorded). Offline, sha256-pinned, out-of-lane returns `applicable=False`. 18 checks pass on the real checkpoints; run with `laya-audit/.venv/Scripts/python.exe -m teachers.selfcheck` from the project root. **A3 is closed:** teachers are the judge, the conductor and laya-code; `teachers/baselines.py` holds `laya-typed` as a BASELINE adapter (never a training target, enforced by `usable_as_target`); cortex-1 is rejected. |
| `capture/render.py`, `capture/capture.py` | Capture records (`capture-record-v1`, frozen): shared prefix plus one independent suffix per candidate, block-masked packed forward, depths 12/18/24/30/36, lm_head never run. Model-independent. |
| `decision_head/pilot_data.py` | Loads the pilot capture into head tensors; asserts the frozen manifest, train and cal only. |
| `compiler/pilot.py`, `configs/pilot_manifest_v1.toml` | The pilot sampling rule (registered before sampling) and the hashed manifest. |
| `configs/thresholds.toml` | Every gate threshold, registered before the run it governs (G2, equivalence v1 and v2, capture throughput, capture-one-record). |
| `tools/` | `verify_hf_download.py`, `eval_laya_typed_decisions.py`, `_manifest.py`, plus the equivalence, throughput, capture, annotation and validation runners. Outputs are create-only. |
| `configs/models.yaml` | Pinned revisions and sha256 for the models we hold. |
| `results/raw/` | Row-level evidence: G0 CUDA pass, download verifications, Laya evaluations. |
| `DECISION_HEAD.md` | Survey of about 35 repos and cards, 16 lessons, parameter inventory, ablation plan, risks, GitHub cross-check. |

## What has been shown

- **G0 passed [M]:** torch 2.14.0+cu126 sees the RTX 3050 6GB.
- **Models on disk, verified [M]:** gpt-oss-120b (15 shards) and gpt-oss-20b-heretic (9 shards) are complete and sha256-verified, both untouched. The five Laya-family teachers are downloaded and verified (`laya-typed-decisions`, `laya-conductor`, `laya-stop-completion-judge`, `laya-code`, `cortex-1-large`).
- **Laya reproduced [M]:** `laya-typed-decisions` scores 0.771 on all 2,000 typed-decisions test decisions against 0.766 published. The conductor scores 0.340 on the same test (random 0.318): each specialist is useful only inside its own lane.
- **Laya's act head is dead [M]:** `act_probability` is exactly 1.0 on every answer. Our abstain/escalate gate must beat plain confidence to ship.
- **Teachers behave as their cards say [M, sanity only]:** conductor 6/6 printed routing examples; judge 0.75 vs 0.05 for announced next step vs direct answer; laya-code relevant chunk 0.43 vs unrelated 0.16, and a chunk scores 0.432 alone and 0.427 in a batch of 5.
- **Frozen-decoder head-only has precedent [A]:** AnyJev (Nokia) fits a closed-form head at about 2/3 depth: Qwen3-4B reaches 0.786 on typed-decisions with 300 labels per question. Heads there do not transfer across questions; our shared head over variable candidate lists is the unproven part.
- **Datasets by recorded outcome [M]:** R2E-Gym SFT has no outcome flag; SWE-rebench-openhands (`resolved`, `exit_status`) and SWE-smith (`resolved`) do. No Docker here, so execution-based checks rely on recorded results.
- **Machine [M]:** AVX-512 torch CPU kernels crash on this CPU (use `ATEN_CPU_CAPABILITY=avx2`); the Hub CLI exits 0 after a `MemoryError`, so downloads are size-checked; D: has about 33 GB free.

- **State builder built and tested [M]** (`compiler/state_builder.py`, `compiler/selfcheck.py`, 17 checks pass): twin trajectories with different labels and futures give identical state, hash and cut manifest at three budgets; the label event and future never enter and `events[t:]` is never read (also true on 1,264 real decisions); episode fields, unknown roles and missing keys are rejected; cuts follow a fixed declared order with visible markers; overflow raises. Budget needs a decision: on 2,147 real decisions the share needing the last-resort core trimming (issue and latest events cut) is 33.5% at 8,000 chars, 3.8% at 12,000, 1.4% at 16,000 and 24,000, 0% at 32,000; oldest-event dropping is needed for 59% to 78% of decisions up to 16,000 chars. The budget is fixed before compilation, not tuned on results.

- **Compiler stage 1 built and run on the 300-row pilot sample [M]** (`compiler/`: state builder, `candidates.py`, `tiers.py`, `split.py`, `run.py`; `configs/compile.toml` fixes the 32,000-character state budget; `uv run python -m compiler.selfcheck` all checks pass). Output `data/decisions/action-swerebench-sample-v1.jsonl` (not in git): 18,894 ACTION records, none skipped, 42 MB; splits by repository train 11,941 / cal 2,557 / test 2,231 / OOD 2,165 (142 / 32 / 29 / 27 repositories); tiers STRONG 706, WEAK 7,138, UNKNOWN 10,807, NEGATIVE 243; cut stage none 2,977, caps 9,747, drop 6,168, core 2. Reports and the split manifest hash are in `results/raw/` (`compile-report-*`, `split-manifest-*`). Known shortcuts to quantify in A5: the true action is the longest candidate text in 24.7% of decisions (chance about 14.9%), and 25.7% of view/edit labels name a file that never appears in the state. Position of the true action is uniform (chi-square 84.6, df 90). **Candidate coverage [M]: only 6,118 of 18,894 true actions (32.4%) are in the pool a state-only generator could propose** (the rest are novel files or commands), so A5 reports accuracy next to coverage. Tier use, the NONE-position axis (first, middle, last), shuffling that includes NONE, and the 12k/16k/24k/32k budget ablation are pre-registered in ``configs/compile.toml``.

- **A5 locked benchmark and baselines [M]** (`compiler/benchmark.py`, `compiler/baselines.py`). `data/benchmark/a5-v1.jsonl`: 4,396 decisions (test 2,231 plus OOD 2,165), read-only, sha256 `f385e40b5f7a5e0b...` in `results/raw/a5-benchmark-a5-v1.sha256`, with every pre-registered variant (shuffle including NONE, NONE first/middle/last, +3/+6/+10 distractors, state hashes at 12k/16k/24k/32k). Baselines in `results/raw/a5-baselines-a5-v1.json`, accuracy = agreement with the trajectory action: random 0.125, first 0.148 (0.000 when NONE is first), last item 0.150, longest text 0.257, **TF-IDF on candidate text only 0.398**. Coverage of true actions by a state-only generator: 31.7% (test), 31.8% (OOD). **The state-blind TF-IDF model reaches 0.398, over 3x random: shortcut signal in the candidates.** It is stronger on uncovered decisions (0.433, the true action is the one candidate not drawn from the pool) than on covered ones (0.323, still 2.6x random; longest-text falls to 0.097). So the primary head metric is accuracy on **covered** decisions, per evidence tier, against the best baseline there (TF-IDF 0.323), with uncovered decisions reported separately; end-to-end is covered accuracy times coverage. Head claims need a paired-bootstrap interval that excludes zero against the best baseline.

- **Step 1 of the A6/A7 plan [M]:** baselines now save per-decision predictions (`results/raw/a5-baselines-a5-v1-perdecision.json`, 1.8 MB) for paired bootstrap; the rerun reproduces the earlier TF-IDF figure exactly (0.3983). The `laya-typed` baseline over all eight axes is run by `compiler/typed_baseline.py` (state rebuilt tail-first at 2,400 characters or the smallest budget whose core fits; any remaining explicit cut is counted); results in `results/raw/a5-typed-baseline-a5-v1.json`.

- **Qwen3-4B equivalence [M]:** rule v1 failed as written (top-1 agreement 0.961 to 0.992 against a 0.99 cap; every disagreement at a reference top-2 gap of at most 0.125; the noise floor was exactly 0, so the rule reduced to absolute caps). v2 was registered afterwards with the reason written; v1 stays recorded as FAIL. v2 passes: streamed equals resident exactly on 12 layers, and CPU vs GPU-streamed KL is 5.5e-4 to 6.6e-4.
- **Capture [M]:** 1,017.6 tok/s on 8,192-token sequences (1.8% above the 1,000 GO line); the real 980-decision run averaged 670 tok/s (longer states, thermal throttling at 86 to 87 C). 2.98 GPU hours, 0 skips.
- **Pilot capture validated [M]:** 980 of 980 files equal the manifest in id and order; all five depths consistent, finite, hidden size 2,560; 3 recaptured decisions bit-identical; lm_head never run. Hidden-state norms grow with depth (query mean 34 at layer 12, 658 at layer 36), so heads must normalise per layer.
- **Laya on the pilot rows [M]:** laya-typed scores 0.217 on all 980 and 0.178 on the 449 covered rows, below the TF-IDF 0.323; all 980 states exceed its 640-token window. It stays a baseline for ACTION.
- **120B read speed [M]:** G2 passed at 0.87 s per simulated token (I/O only).
- **Audit (2026-10-01):** splits are repository-disjoint, no pilot decision or trajectory is in the locked benchmark, all hash sidecars verify, all 33 project `.py` files compile. Open items: RB0 env manifest and experiment id not recorded; `CLAUDE.md`, `PLAN.md` and `configs/p0.toml` still name gpt-oss as primary and heretic as first backbone; `configs/models.yaml` has no Qwen3-4B entry; `autojev/` holds 16.8 GB of dead partial downloads.

## Not done

Evaluation of any head on the locked benchmark, final calibration on a larger frozen calibration split, a recency/pool-rank baseline on the locked benchmark, CONTINUE and COMPLETION decision compilation, and the in-lane teacher outputs for those kinds. **Nothing is known yet about how well the head works on real states.** The benchmark has not moved from 0.323.

## Order (user, 2026-10-01)

Track A first, no model touched: A3 (finish the teachers) then A4 (compiler) then A5 (locked benchmark) then A6 (baselines). Then Track B on heretic: capture at a depth chosen by a sweep, closed-form heads for fixed-layout decisions, then the pointer head for variable candidate lists.

## Next

**Superseded by the current plan above (PLAN.md section 12): Phase 0 first.** The older list below is kept as history.

1. Tier-1 closed-form head on `next_action_kind` at all five depths: fit on train only, pick depth on the 380 pilot calibration rows (pilot-only, not the final calibration), report against the baselines.
2. Tier-2 set-attention pointer head on ACTION (about 34M parameters, frozen backbone, 600 train decisions: strong regularisation and early stopping), compared with TF-IDF 0.323 covered by a paired bootstrap on the locked benchmark. That needs the benchmark decisions captured too (up to about 15 GPU hours for 4,396).
3. After each step, rerun the locked benchmark and add a row to the increments table above and to `results/increments.md`.

## Open decisions for the user

**Plan approvals (2026-10-02):** (1) provider, cap (proposed Rs 6,000) and upload scope for Kaggle/Colab or a rental; (2) downloads over 1 GB: SWE-rebench-openhands, Qwen3-Reranker-4B; (3) lift the "no Laya fine-tuning" rule (only a free side track)? (4) questions for AIC: billing granularity, billing while stopped, disk, Docker, GST. CLAUDE.md, PLAN.md, `configs/p0.toml` and `configs/models.yaml` were updated for the Qwen3-4B backbone on 2026-10-02.

Older items: delete the 16.8 GB of dead `autojev/` partial downloads, the JDownloader installer (`GPT OSS 20 B.exe`) and the partial downloads in the model caches? Record the RB0 env manifest and experiment id now? Update `CLAUDE.md`, `PLAN.md` and `configs/p0.toml` to the Qwen3-4B experiment backbone (they are frozen documents, so this needs your approval)? Add a pinned Qwen3-4B entry to `configs/models.yaml`? Commit the new files and push?






