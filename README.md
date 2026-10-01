# jev-airllm: a decision head for local LLMs

Goal: one internal decision head (a Laya/Jeeves-style pointer head) that reads the hidden states of a **local, frozen
language model** and picks among candidate actions with calibrated probabilities. It is **backbone-agnostic**: the
head takes the hidden size as a parameter and is fitted per model. First backbone: `gpt-oss-20b-heretic`. The 120B model is
parked and **not touched**. No model is touched until the head side (Track A) is ready.

Rules: [CLAUDE.md](CLAUDE.md). Design record and survey: [DECISION_HEAD.md](DECISION_HEAD.md). Step list and closure path: [PLAN.md](PLAN.md) section 11.
Tags: **[M]** measured here, **[A]** reported by an author and not reproduced, **[E]** estimate.
Last updated: 2026-10-01.

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
| `tools/` | `verify_hf_download.py`, `eval_laya_typed_decisions.py`, `_manifest.py`. Outputs are create-only. |
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

## Not done

Teacher outputs attached to the records (stage 2, in laya-audit), the locked benchmark and baselines (A5, A6), any hidden-state capture, training, calibration on our data, and any evaluation on a real LLM. **Nothing is known yet about how well the head works on real states.**

## Order (user, 2026-10-01)

Track A first, no model touched: A3 (finish the teachers) then A4 (compiler) then A5 (locked benchmark) then A6 (baselines). Then Track B on heretic: capture at a depth chosen by a sweep, closed-form heads for fixed-layout decisions, then the pointer head for variable candidate lists.

## Next

A4.0 is done: 20 SWE-rebench-openhands rows read (`A4_LEAKAGE_RULES.md` v1 and `A4_LEAKAGE_RULES_v1_1.md`, both frozen by hash in `results/raw/`). Next is the compiler: state builder (events before `t` only, tail-first, raises on overflow), action normalisation, tier assignment, candidate generator, leakage checks, then teacher-attach as a separate stage in `laya-audit`. Targets: ACTION (primary), NEXT_ACTION_TYPE (sanity arm), COMPLETION (real and weak labels apart).

## Open decisions for the user

Delete the leftover partial downloads in the Laya and 120B caches (about 335 MB)? Allow about 300 SWE-rebench rows via the datasets server (about 83 MB [E]; the full 1.94 GB needs a yes)? Commit the new files (`A4_LEAKAGE_RULES*.md`, baselines, schema changes) and push? Which second backbone after heretic (`Qwen/Qwen3.5-4B` is in the local cache, not opened)?





