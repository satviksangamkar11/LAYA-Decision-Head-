# BENCHMARKS: Laya decision-head benchmarks only

Created 2026-10-03. Scope (user, 2026-10-03): **only benchmarks of the Laya decision head**: the A5 decision benchmark, its baselines (including the Laya checkpoints), the head experiments, and the audits that decide whether a head claim can stand.
Runtime, GPU and capture-speed benchmarks are not in this file; they stay in `results/raw/` and `results/increments.md`.
**Tags:** **[M]** measured here, row-level file in `results/raw/`; **[A]** author-reported, not reproduced; **[E]** estimate. No threshold was edited after results (new versioned files instead); failed gates stay recorded as FAIL.

## 1. Headline: where the head stands
**No head has been scored on a locked benchmark.** The only head numbers are on the 980-decision pilot (section 3), and the integrity audits (section 4) show a no-model rule reaches the same level (0.663 vs 0.667). So **no head claim stands yet**. The bar to beat on the registered A5-v1 covered rows was TF-IDF 0.323; A5-v1 itself later FAILED the integrity gate and is diagnostic only.

## 2. A5-v1 decision benchmark: baselines (historical, diagnostic)
Benchmark `data/benchmark/a5-v1.jsonl`, 4,396 decisions (test 2,231, OOD 2,165; covered 1,395, uncovered 3,001). Accuracy [M], `a5-baselines-a5-v1.json`:
| Baseline | All | Covered | Uncovered | Test | OOD |
|---|---:|---:|---:|---:|---:|
| random | 0.1246 | 0.1263 | 0.1238 | 0.1248 | 0.1244 |
| first | 0.1481 | 0.1570 | 0.1440 | 0.1546 | 0.1413 |
| last item | 0.1501 | 0.1677 | 0.1420 | 0.1461 | 0.1543 |
| longest text | 0.2573 | 0.0968 | 0.3319 | 0.2559 | 0.2587 |
| **TF-IDF (candidate only)** | **0.3983** | **0.3233** | 0.4332 | 0.4137 | 0.3824 |
| laya-typed-decisions used as an A5 baseline (`a5-typed-baseline-a5-v1.json`) | 0.2277 | 0.2387 | 0.2226 | 0.2237 | 0.2319 |
TF-IDF calibration accuracy 0.3801; coverage test 0.3165, OOD 0.3182; Laya typed with permuted option order 0.2332 all. Per-decision predictions for paired bootstrap: `a5-baselines-a5-v1-perdecision.json`.

### Laya checkpoints on their own tasks [M] (2,000 rows each, 4 workflows; row-level `laya-eval-*.jsonl`)
| Checkpoint | Overall | choice | noul | score |
|---|---:|---:|---:|---:|
| laya-typed-decisions | **0.7715** (0.766 published [A]) | 0.7433 | 0.8583 | 0.7275 |
| laya-conductor | 0.3395 | 0.2033 | 0.5167 | 0.3088 |
Each Laya specialist is useful only in its own lane. Laya's `act_probability` is exactly 1.0 on every answer, so our abstain/escalate gate must beat plain confidence (README, [M]).

## 3. Head experiments on the 980-decision pilot (PILOT ONLY; the locked set was never touched)
| Step | Result | File |
|---|---|---|
| Tier-1 closed-form head, next_action_kind (depth 24 by NLL on 380 pilot-cal rows) | vs label-frequency prior: cal accuracy delta **+0.303**, 95% cluster-bootstrap CI [+0.233, +0.368]. Sanity arm, not an A5 number | `tier1-next-action-kind-20261001-182958.json` |
| Tier-2 pointer head on ACTION (depth 24, D 128, 5-seed ensemble, T = 1.377 from out-of-fold train) | pilot-cal accuracy 0.616 all / **0.667 covered** (n = 120), NLL 1.254. TF-IDF data-matched 0.290 / 0.333; TF-IDF full-train 0.384 / 0.308 | `tier2-action-pointer-20261001-183544.json` |
| Controls | label-shuffled 0.108; state-shuffled eval 0.234; metadata-only 0.258; option-only linear scorer **0.821 / 0.775 covered (beats the head)**; most-recent-in-pool 0.492 (cal covered) | `tier2-controls-20261001-183813.json`, `tier2-control-option-only-20261001-183902.json`, `shortcut-pool-rank-20261001-184102.json` |
| Laya typed on the pilot rows (README only) | 0.217 on all 980, 0.178 on the 449 covered | no row-level file located (section 7) |

## 4. Does the benchmark prove anything? Integrity and shortcut audits
| Step | Result | File |
|---|---|---|
| Integrity gate on A5-v1 | **FAIL**; A5-v1 frozen as diagnostic. Cal covered (n = 120): most-frequent-in-history **0.550**, most-recent 0.492, TF-IDF 0.333, metadata-only 0.258, random 0.127; limit 0.227. The pilot head (0.667) is only 0.117 above a no-model rule | `integrity-gate-A5v1-20261001-184428.json` |
| Pool audit | within-decision AUC of "is true": pool_rank 0.212, history_frequency 0.762, lexical overlap 0.673 (tolerance 0.5 +/- 0.05). An MLP over **eight trivial features with no state and no model reaches 0.663**, equal to the pilot head's 0.667 | `pool-audit-A5v1-20261001-184836.json` |
| Pool feasibility for A5-v2 | no feature-balancing level passes the 70% bar (frequency-only feasible for 100% first-time but 67.7% repeat rows) | `pool-feasibility-20261001-185056.json` |
| A5-v2 sampler `a5v2-odds-1` | **FAIL** the registered gate, not loosened: attacker A 0.443 overall (limit 0.249; same attacker on v1 sets 0.652); attacker B 0.344 (v1 0.524) | `a5v2-train-cv-audit-20261001-194503.json` |
| F-dev feasibility of the frozen sampler (fresh data) | USABLE, all 7 criteria: 7,446 decisions / 118 trajectories, 2,454 covered, retention 1.000; attackers A 0.405, B 0.324 (random 0.150) | `fdev-feasibility-20261001-200544.json` |

## 5. A5-v3: the question and its sizing
New question (registered before any scoring): does the frozen Qwen3-4B state add decision value **beyond candidate/history features** on identical candidate sets. Primary claim H2 minus H0_matched; success needs the lower 95% cluster-bootstrap bound > 0, estimate >= 0.05 and half-width <= 0.04, otherwise INCONCLUSIVE (`configs/a5_v3_spec.toml`).
| Item | Result | File |
|---|---|---|
| Power simulation (conservative scenario) | power 0.000 at 100 and 200 clusters, **0.955 at 400**, 0.995 at 600; smallest N with power >= 0.80 is 400 locked clusters; a true effect of exactly 0.05 reaches only about 0.46 to 0.53 power | `power-sim-v3-20261001-195023.json` |
| LOCKED set | 444 trajectories / 324 repositories (margin 44 over 400), repository-disjoint from F-dev and the 230 existing repositories; hashed (sha256 b5ac8b31...), read-only, not opened | `fresh-v3-provenance.json`, `fresh-v3-locked.sha256` |
| Manifests | TRAIN 1,986 decisions / 185 trajectories; CAL 569 / 42; LOCKED 2,220 (5 per trajectory, no cluster lost) | `v3-manifests.json` |
| Trainer audit | PASS 16 of 16 (earlier runs failed and found a real seeding bug; kept as the record) | `v3-trainer-audit-20261001-211604.json` |
| Sampler regression | 3,907 of 3,907 TRAIN decisions identical across the old script, the refactor and `compiler/a5v2.py` | `sampler-regression-20261001-204118.json` |

## 6. Not yet measured for the head
No head scored on LOCKED; no H0 / H2 comparison; no linear-probe or outcome-probe result (plan Phase 3); no head trained on Kaggle-T4 states; no baseline for Qwen3-Reranker-4B or the conductor / stop-completion judge on A5-v3 candidate sets; no abstain-gate comparison against plain confidence; no agent evaluation with and without the head.

## 7. Gaps found while writing this file
1. **Laya typed on the pilot rows (0.217 / 0.178):** appears in README with no row-level file located; re-run `tools/eval_laya_typed_decisions.py` or point to the file.
2. **Where the numbers live:** `results/increments.md` is the step-by-step ledger (rows 0 to 21 cover these steps); this file is the head-only summary of the same files.

## 8. External reference points [A] (published by others, not reproduced here, retrieved 2026-10-03)
These are the yardsticks for "what accuracy is realistic". They are NOT comparators for our A5 numbers: our current target is agreement with the agent's next action among a candidate pool; the published critics predict trajectory OUTCOMES.
| Source | Metric | Published value |
|---|---|---|
| Laya typed-decisions (primary: github.com/NandhaKishorM/laya BENCHMARKS.md, section 'typed-decisions: 400 cases, 2,000 decisions') | overall accuracy | 0.766, soft accuracy 0.471, Brier 0.061, ECE 0.213, score MAE 0.242; random 0.318, majority-class 0.461, teacher ceiling 0.735; base checkpoints 0.362 / 0.352 (below majority). **Laya states the 0.766 belongs to the fine-tuned checkpoint on that benchmark's own training split (in-distribution), not held-out generalization.** Per-primitive figures (noul 0.857, choice 0.733, score 0.723) were seen only in a search summary, not in the primary text. Ours reproduced overall: 0.7715 (0.8583 / 0.7433 / 0.7275), section 2: this confirms the checkpoint runs correctly, not that it generalises |
| laya-conductor model card | routing accuracy | 87% (zero-shot base about 37%); different task from typed-decisions (we measured 0.3395 on typed-decisions, out of its lane) |
| OpenHands, "Learning to verify AI-generated code" (2026-03-05) | critic AUC | benchmark-trained critics 0.45 to 0.48 on real-world outcomes (worse than random); production-trained: PR merge 0.58, code survival 0.69 |
| same | best-of-8 on the mixed-outcome subset of SWE-bench Verified | critic 73.8% vs random 57.9% (a filtered subset, not all instances) |
| OpenHands critic model card (Qwen2.5-Coder-32B, TD learning) | SWE-bench Verified | 60.6% with 1 rollout, 66.4% with 5 attempts |
| R2E-Gym (arXiv 2504.07164) | Best@26 on SWE-bench Verified | execution-free verifier 42.8%, execution-based 43.7%, hybrid 51.0% |
| SWE-RM (arXiv 2512.21919), 30B MoE, 3B active | test-time scaling on SWE-bench Verified | Qwen3-Coder-Flash 51.6% to 62.0%; Qwen3-Coder-Max 67.0% to 74.6% |
Reading: realistic outcome-critic discrimination is AUC about 0.6 to 0.7 on real data, and benchmark-trained critics can be at chance on real-world outcomes. A head claim far above that, or any large gain over the no-model rules (most-frequent-in-history 0.550 on cal covered), deserves suspicion until the integrity audits pass.
Sources: https://www.openhands.dev/blog/20260305-learning-to-verify-ai-generated-code , https://huggingface.co/OpenHands/openhands-critic-32b-exp-20250417 , https://arxiv.org/abs/2504.07164 , https://arxiv.org/abs/2512.21919 , https://huggingface.co/mvilacad/laya-conductor , https://github.com/NandhaKishorM/laya (BENCHMARKS.md)
