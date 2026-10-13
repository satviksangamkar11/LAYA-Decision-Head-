# jev-airllm

A decision head for local coding models. A small trained scorer reads the hidden states of a **frozen** local language model and ranks candidate
next actions (run a command, edit a file, write a file, ...). The aim is a local coding agent that makes better next-step choices on a 6 GB GPU.

**Status (2026-10-13, 03:30 IST):** research prototype. The Qwen model only supplies features and is never trained. On 53 pools from trajectories it
never saw, the head with a trained adapter picks the teacher's action 79.2% of the time (chance 29.3%) and loses 15 points when the coding context is
removed; this is a development result. Training only the head's own 31 tensors reaches at most about 50% out of fold (section 2i), and adding a small
linear read path raises agreement to 71% but the gain sits in the new layer, not in the 31 tensors, and much of it does not need the context (section 2j).
A simple linear probe on the same states scores 85.8 to 94.3% throughout. Behind the same trained adapter, randomly initialised heads beat every pretrained
LAYA head (83% against 77%, section 2k). The first execution test (section 2l) found only 1 informative problem out of 5, so no result yet shows better
coding-task outcomes; the next generator (gpt-oss-20b) failed its API verification and was stopped.

**Update 2026-10-13, 13:40 IST:** see sections 2k-2l and HANDOFF_2026_10_11_STATUS.md 16.15-16.20.

All numbers in this file were measured in this repository unless marked otherwise. Each has a row-level file under `results/raw/`.

| Document | What it is |
|---|---|
| [AGENT_README.md](AGENT_README.md) | Dated working log: status reports, frozen plans, the fixed API call logic |
| [HANDOFF_2026_10_11_STATUS.md](HANDOFF_2026_10_11_STATUS.md) | Full record of the current data and training work, with file paths |
| [CLAUDE.md](CLAUDE.md) | Project rules (what may and may not be done, and by whose decision) |
| [DECISION_HEAD.md](DECISION_HEAD.md) | Design record and literature survey for the head |
| [PLAN.md](PLAN.md) | Runtime plan and step lists |
| [results/increments.md](results/increments.md) | Ledger: one row per major step |

## Contents

- [How it works](#how-it-works)
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [What is not shown](#what-is-not-shown)
- [Reproducing](#reproducing)
- [Repository layout](#repository-layout)
- [Earlier work](#earlier-work)
- [Roadmap](#roadmap)

## How it works

```
coding session so far ──┐
                        ├─► frozen Qwen3-4B ──► hidden states ──► adapter ──► pretrained LAYA head ──► one score per option
candidate next actions ─┘      (never trained)   (layers 18/24/30)  (trained)     (31 tensors, frozen)
```

1. **Capture.** The session context and each candidate action are run through the frozen language model once. Hidden states at three depths are
   saved: one vector set for the context (`[3, 2560]`) and one per option (`[N, 3, 2560]`). The model's output layer is never run.
2. **Adapter.** A small trainable module maps those states into the input space of a pretrained decision head.
3. **Head.** The pretrained head (two context layers and a scorer) gives one score per option. The highest score is the head's choice.
4. **Baseline.** A linear probe (logistic regression on the option states) is trained on the same data and reported next to every head result.

In the model that is kept, only the adapter is trained. The language model is never trained. The head's own tensors were trained only in controlled
experiments (results 2b and 2d), saved as separate files; the released checkpoint file is never written and its hash is checked after every run.

## Models

| Component | Model | Parameters | Trained here |
|---|---|---|---|
| Language model | `Qwen3-4B-Thinking-2507` (revision `768f209d9e`), hidden size 2,560 | about 4 billion | No, frozen |
| Decision head | Laya `Coding_Decision_Agent` (`models/coding-decision-agent`), 31 tensors | 26,248,193 | In experiments only (2b, 2d) |
| Adapter | New, 8 tensors | 5,246,982 | Yes |
| Fixed parts | Kind embedding and gate | 273,154 | No |
| Candidate generator | Nemotron-3-Super 120B (hosted) | n/a | No |
| Review judges | GLM-5.3 (full reasoning) with Gemma 4 31B or Muse Glimmer 30B (hosted) | n/a | No |

The head was originally trained on a different encoder (ModernBERT-large). Feeding it Qwen states is the reason an adapter is needed.

## Dataset

The teacher is **Fable**: recorded coding-agent sessions. For each decision, the pool holds Fable's recorded action plus 2 to 4 alternative actions.

- **Labels.** Fable's action is the positive. Alternatives are labelled *unknown*, never *wrong*: the data teaches "choose like Fable", not
  "this action is verified best".
- **Canonical splits.** TRAIN, VAL, CAL and TEST are separated by trajectory. Only TRAIN has been used for training. VAL is used for evaluation and has
  been inspected closely. CAL (309 decisions): no model has seen it; its edit decisions were replayed in the sandbox once to count usable tests, and a
  guard that keeps CAL and TEST contexts out of model requests was left in place. TEST (713 decisions): never used; one research report quoted three
  short example lines from TEST trajectories, which were redacted.
- **Teacher identity.** 1,294 of the 11,436 rows in the trace file's train portion are marked as attested Fable 5; for the rest it is the uploader's claim.

### Pipeline

| Step | What happens |
|---|---|
| 1. Generate | 8 generator calls per decision (3 conditioned on the tool, 5 blind). The generator never sees Fable's action; a leak check runs on every request. |
| 2. Objective screen | Tool contract, grounding in the session, duplicates, and syntax parsers in a sandbox (Python, Go, Java, PowerShell, JavaScript, TypeScript, Bash, Zsh, Ruby, Rust, C#, C++). |
| 3. Blinded review | Two judges see the pool without labels, with hidden control options. **Both must accept** an alternative. |
| 4. Assembly | Alternatives that equal Fable's action (after normalising the command) are removed. A pool needs Fable's action + 2 to 4 alternatives, one using the same tool. |
| 5. Verification | Every kept alternative is re-checked one by one against both judges' verdicts and the screen. |
| 6. Hand audit | A blind sample is read by hand. |
| 7. Freeze | Hashed manifest with a role per pool. Create-only. |

### Dataset v3 (frozen)

`results/raw/laya-fullaction-set-v3-20261012T055420Z`, manifest sha256 `725cd39d...d8f4`.

| Quantity | Value |
|---|---|
| Eligible TRAIN decisions (coding, tool action) | 1,342 |
| Sent to the generator | 1,074 |
| First-step decisions, **not processed** | 268 |
| Qualified pools | 335 |
| Alternatives, all individually verified | 866 of 866 |
| Yield (pools per generated decision) | 31% |
| Pools reviewed by GLM + Gemma / GLM + Muse | 68 / 267 |
| Pools with captured features | 334 (one exceeds the 24,000-token context limit) |
| Trajectories | 192 |

Roles recorded per pool: `train` 120 pools (94 trajectories), `train_extension` 54 pools (35 trajectories), `heldout_transfer` 161 pools
(98 trajectories that appear in no train pool).

**Where the other decisions were lost** (995 decisions of the five TRAIN batches, `results/raw/research-20261012/C_pool_failure_funnel.md`): 260 before a pool
could be built (114 with too few candidates, 103 with no candidate using the teacher's tool), 23 at the objective screen, 377 at review or assembly.
150 ended with exactly one jointly accepted alternative, one short of a pool.

**Cleaned version (v4)**, `results/raw/laya-fullaction-clean-v4-20261012T130553Z`: 59 of the 866 alternatives (6.8%) were the teacher's action plus only
read-only steps and were removed by a deterministic rule; 31 pools then fell below the pool rule. 304 pools and 779 alternatives remain (303 with
features). The same rule leaves 46 of the 53 VAL pools. Dataset v3 is unchanged. Counting looser cases (a set-up, lint or timing step added or
dropped), about a quarter of alternatives are the teacher's action in effect (24% on VAL, 25.5% in a 60-pool training sample).

**Gate record.** 52 training pools come from two review runs whose own reports mark the pair gate as failed for accounting reasons (the second judge's
replies were saved under the retry name; one batch had only 24 known-bad controls). No known-bad option was accepted by both judges in either. The v3
manifest lists an exception only for the last batch; this gap is recorded in the handoff, not yet in a manifest.

One batch was accepted under a documented **infrastructure exception**: a network outage and an endpoint outage made 42% to 47% of first judge
replies unusable, which fails the written 5% limit. The original check stays recorded as failed. Judge-caused failures were 0 of 352 (GLM) and
5 of 309 (Muse); every reply was recovered by resending.

## Results

"Accuracy" below means: the scorer ranks Fable's recorded action first. It is agreement with the teacher, not task success. Intervals are 95%,
from 2,000 bootstrap resamples of whole trajectories.

### 1. Evaluation on new trajectories (VAL, pre-registered)

The three saved adapters, the probe and a text baseline were scored once on 53 pools from 36 VAL trajectories that were never used for anything.
Chance 29.3%. Source: `results/raw/val-eval-v1-20261012T082055Z`.

| Scorer | Real context | No context |
|---|---|---|
| Head, mean of the three saved adapters | 79.2% (67.9 to 89.3) | 64.2% (52.0 to 76.3) |
| Linear probe | 94.3% (87.3 to 100) | 81.1% (69.0 to 90.9) |
| Options text only | 67.9% (53.9 to 81.1) | n/a |

- **Accuracy holds:** 79.2% on new trajectories against 80.2% inside the dataset. Per seed: 79.2%, 81.1%, 77.4%.
- **Context helps:** real context minus no context is +15.1 points (+6.3 to +24.1) for the head and +13.2 (+5.6 to +22.5) for the probe.
  The models were fitted with context, so part of the drop can be a train/test mismatch.
- **Probe ahead:** probe minus head is +15.1 points (+6.2 to +24.5).
- **Head against text only:** +11.3 points (-4.1 to +28.7), not significant at this size.
- **Audit caveat:** in a blind sample, 5 of 18 kept alternatives were Fable's action in effect (for example the same test command with an extra
  file read). Picking one counts as a miss for every scorer.
- **Diagnosis of the gap (read-only, `results/raw/val-gap-diag-v1-20261012`):** 32 of 134 VAL alternatives (24%) are Fable's action in effect. Counting
  them as correct moves the head to 83.6% and leaves the probe at 94.3%. The probe never picks one of them, so part of its score is recognising the
  teacher's exact wording. In 9 of the 13 pools where the probe wins, at least one of the three adapters is right.

### 2. Fine-tuned head on all pools (in-dataset, 5-fold by trajectory)

334 pools, 192 trajectories, adapter only, chance 28.9%. Source: `results/raw/laya-finetune-all-v1-20261012T080145Z`.

| Pools | Head | Linear probe | Probe minus head |
|---|---|---|---|
| All 334 | 80.2% (75.5 to 84.8) | 86.2% (82.4 to 89.8) | +6.0 (+1.9 to +10.1) |
| Former `train` role, 120 | 84.2% | 89.2% | +5.0 (0.0 to +10.9) |
| Former `heldout_transfer` role, 160 | 81.3% | 83.8% | +2.5 (-2.0 to +6.8) |
| `train_extension`, 54 | 68.5% | 87.0% | +18.5 (+5.6 to +33.3) |

Three final adapters (seeds 0 to 2) are saved in `adapters/`. Each was fitted on 277 pools with 57 used only to pick the stopping point; they
reach 91% to 93% on their fitting pools and 82.5% on the stopping pools. Training took about 4 minutes on a CPU.

The weak `train_extension` figure comes mostly from one fold (20 pools: head 40%, probe 70%); tool type, step, judge pair and context length
do not explain it.

### 2b. Training the head's own tensors (controlled, pre-registered)

Labels were first cleaned in a separate version: 59 of 866 alternatives that strictly equal Fable's action (its steps plus read-only steps) were
removed, leaving 304 training pools and 46 VAL pools. Then two methods were trained on identical data, features, folds, seeds and steps.
Source: `results/raw/laya-joint-v1-20261012T130647Z`.

| Test | Adapter only | Adapter + head tensors | Difference |
|---|---|---|---|
| Cleaned VAL, 46 pools (primary) | 76.1% | 75.4% | -0.7 (-5.9 to +4.3) |
| Original VAL, 53 pools | 75.5% | 76.1% | +0.6 (-4.5 to +5.8) |
| In-dataset 5-fold, 303 pools | 77.9% | 78.9% | +1.0 (-1.8 to +3.8) |

Updating the head's 26 million pretrained parameters gave no measurable gain at this data size, so the adapter-only model is kept.

### 2c. Outcome pilot: can the scorers pick a Qwen-written fix? (pre-registered)

Edit and write decisions were replayed twice in an isolated sandbox: 44 CAL and 39 VAL decisions have a test whose result changes with the teacher's
own action; 27 and 23 of them go cleanly from fail to pass. The pilot used VAL. For the first five decisions the frozen Qwen model wrote four
candidate actions each without seeing the teacher's action. Source: `results/raw/val-outcome-v1-20261012T134540Z`.

| Measure | Result |
|---|---|
| Samples that finished within 6,144 new tokens | 5 of 20 (the other 15 never closed their thinking block) |
| Valid decoded actions | 1 of 20 (a file read); 4 finished samples were rejected by the decoder |
| Code-change candidates | 0 |
| Decisions where any candidate makes the tests pass | 0 of 5 |
| GPU time | 3.75 hours of worker time on 2 x T4 (about 4.0 hours counting start-up) |

No scorer comparison was possible: there was nothing to choose between. On these one-step fix decisions the bottleneck is candidate generation,
not ranking. The pilot was stopped before the remaining 18 decisions. This measures deployment with this particular small model, not the head.

### 2d. Training only the head, with the bridge frozen (pre-registered)

The decisive test for "is the knowledge in the head": the adapter is frozen (bit-identical before and after, checked by hash) and only the head's
31 pretrained tensors train. Source: `results/raw/laya-headonly-v1-20261012T161940Z`. Cleaned VAL, 46 pools, 3 seeds.

| Frozen bridge | Head learning rate | Before | After | Change | Native items before -> after |
|---|---|---|---|---|---|
| Aligned on LAYA's native items only (never saw Fable) | 2e-5 | 39.1% | 54.3% | +15.2 (+6.3 to +24.9) | 71.5% -> 69% |
| Same | 2e-4 (chosen on TRAIN; registered primary) | 39.1% | 52.2% | +13.0 (-2.5 to +26.7) | 71.5% -> 44-58% |
| Fable-trained adapter | 2e-5 | 76.1% | 74.6% | -1.4 (-6.7 to +3.0) | about 30% -> about 30% |
| Same | 2e-4 | 76.1% | 76.1% | 0.0 (-5.1 to +4.9) | about 30% -> about 31% |

The head can absorb part of the teacher's choices through a fixed bridge, but it stays about 22 points below adapter-only training on the same
pools, and a high learning rate destroys its native skill.

### 2e. Where the signal is lost (probes on identical inputs, and two controlled runs)

Source: `results/raw/laya-headonly-diag-v2-20261012T170830Z`. Same 303 training pools and 46 cleaned VAL pools for everything.

| A linear probe reading... | Accuracy |
|---|---|
| raw Qwen option states | 89.1% |
| the output of the bridge that never saw the teacher | 87.0% |
| the same, after the head's pretrained context layers | 58.7% |
| (the head trained through that bridge) | 54.3% |

| Head-only run (same bridge, learning rate 2e-5) | Accuracy | Change |
|---|---|---|
| Reference: 251 pools, 300 steps | 54.3% | - |
| Four times more steps | 55.8% | +1.4 (-8.2 to +11.3) |
| Half the pools | 48.6% | -5.8 (-12.4 to +0.7) |

The bridge passes the signal; the head's own context layers lose it, and at a learning rate that preserves the head's native skill the head fits only
61% of its training pools. More steps do not help; the effect of more data is a trend that this test cannot establish.

### 2f. Audit of the head's input path (read-only)

Source: `results/raw/laya-head-audit-v1-20261012T173011Z`.

| A linear probe reading... (bridge that never saw the teacher) | Accuracy |
|---|---|
| what the head's context layers receive | 87.0% |
| after context layer 1 | 56.5% |
| after context layer 2 | 56.5% |
| the scorer's input | 58.7% |
| what the layers receive and the scorer's input, joined | 84.8% |

After the context layers the options of one pool are nearly the same vector (cosine 0.97; 0.88 in the original model on its own data). No error was
found in masks, shapes or checkpoint handling, and the input scale matches the original model's. Removing the two small differences from the original path (an extra normalisation layer; a kind embedding that turned out to be exactly zero)
changes nothing. The structural difference is the input itself: the
original layers run over hundreds of tokens and build each option's representation from its text, while here each option arrives as one vector.

### 2g. Sequence-input test (prediction written before the data)

Source: `results/raw/laya-seq-v1-20261012T185241Z`. The released head was fed a sequence per decision (16 context slices, 4 slices and a marker per option)
through a new bridge, against the single-vector path, both bridges aligned on the same 600 native items.

| Measure | Sequence input | Single-vector control |
|---|---|---|
| Probe on the bridge output | 80.4% | 89.1% |
| Probe after context layer 1 | 60.9% | 54.3% |
| Cosine between the options of a pool after the layers | 0.985 | 0.980 |
| Native items, accuracy | 56.5% | 62.0% |

The prediction (options stay apart, no drop at layer 1) was not met: this coarse sequence does not stop the collapse. A token-level sequence was not tested.

### 2h. Where the head loses the signal (read-only review)

Source: `results/raw/head-review-run-20261012T192308Z/REPORT-20261012T192246Z.json`. Released head tensors, bridge aligned on native items only, 46 cleaned VAL pools.

| Point inside the head | Linear probe | Mean vector norm |
|---|---|---|
| Bridge output | 87.0% | 42 |
| Layer 1 after attention | 76.1% | 56 |
| Layer 1 after feed-forward | 56.5% | 4,046 |
| Scorer input | 58.7% | 30 |

Blocking attention between tokens changes nothing (probes 54-59%; native accuracy 71.5% to 68.8%). The 0.97 cosine between options is mostly a common
vector (0.48 with it removed). On the fit pools the released head scores worse than equal scores (loss 1.064 against 0.813); head-only training lowered the
loss to 0.751 mainly by flattening its scores (spread 0.73 to 0.28).

### 2i. Four-arm head-only experiment (5-fold cross-validation, Kaggle GPU)

Source: `results/raw/head-exp4-run-20261012T201525Z/REPORT.json`. Bridge frozen and never trained on Fable; only the 31 head tensors train; 3 seeds; 303 TRAIN pools scored out of fold.

| Arm | Out of fold (303) | Minus reference | Native accuracy | Cleaned VAL (46) |
|---|---|---|---|---|
| Released head, no training | 39.9% | - | 71.5% | 39.1% |
| A reference (all tensors 2e-5) | 45.5% | - | 69.0% | 55.1% |
| B+ temperature-scaled start | 47.4% | +1.9 (-5.7 to +9.7) | 44.9% | 49.3% |
| C higher rate on context layer 1 (1e-4) | 50.3% | +4.7 (-0.8 to +10.5) | 63.9% | 48.6% |
| D = C + native replay | 50.4% | +4.8 (+0.9 to +8.7) | 69.7% | 52.9% |
| Linear probe on raw Qwen states | 85.8% | +40.3 | - | 89.1% |

Replay keeps the native skill while layer 1 trains faster, and D is the only arm that meets the registered rule, but the gain is small and the 46 VAL pools do
not confirm it. A single fitted temperature has no finite optimum for the released head on these pools (its loss only reaches the equal-score value).

### 2j. Direct read path: a modified head (5-fold cross-validation, Kaggle GPU)

Source: `results/raw/head-exp5-run-20261012T211558Z/REPORT.json`; full record HANDOFF_2026_10_11_STATUS.md 16.15. One linear layer (1,024 weights, starting at 0)
reads each option's bridge output and adds to the head's score. Bridge frozen; 3 seeds; 303 TRAIN pools scored out of fold. The reproduction gate passed exactly
(previous arm D: 50.4%, native 69.7%).

| Arm | Out of fold (303) | Minus previous D | Native accuracy | Cleaned VAL (46) |
|---|---|---|---|---|
| Previous D: 31 tensors trained, no path | 50.4% | - | 69.7% | 52.9% |
| Read path only, 31 tensors frozen | 71.1% | +20.7 (+15.3 to +26.1) | 68.7% | 68.1% |
| Read path and 31 tensors trained together | 68.4% | +18.0 (+13.3 to +23.1) | 68.8% | 63.8% |
| The same 31 tensors with the path switched off | 41.5% | -8.9 (-13.9 to -4.0) | 69.1% | 45.7% |
| Read path only, half the data | 68.6% | +18.3 | 68.2% | - |
| Linear probe on raw Qwen states | 85.8% | - | - | 89.1% |

- The read path raises agreement with Fable, but **the gain belongs to the new layer**: trained jointly, the 31 pretrained tensors end up worse than without the path
  (-7.5 points against the same recipe without it), close to the released head.
- **Much of the gain does not need the coding context:** trained without context the path scores 75.6% against 71.1% with it on the same 112 pools.
- Native-task accuracy stays at 67-72% in every arm. Half the data is as good as all of it. No checkpoint was saved or adopted.
- These are agreement-with-Fable figures, not coding-task success, and say nothing about other models.

### 2k. Head swap: pretrained LAYA heads against random heads (5-fold cross-validation, Kaggle GPU)

Source: `results/raw/head-swap-run-20261012T222906Z/REPORT.json`; record HANDOFF_2026_10_11_STATUS.md 16.17. Every head frozen, only the adapter trained
(same recipe, folds, seeds); both reproduction gates passed.

| Head | Out of fold (303) | Cleaned VAL (46) |
|---|---|---|
| coding-decision-agent (reference) | 77.2% | 78.3% |
| laya-typed-decisions / laya-code / laya-conductor | 79.0% / 76.9% / 79.6% | 73.9% / 72.5% / 75.4% |
| Same architecture, random weights (3 seeds) | 83.8% / 83.2% / 83.3% | 80.4% / 82.6% / 81.9% |
| Linear probe on raw Qwen states | 85.8% | 89.1% |

- **Random heads beat the pretrained reference by 6.2 points** (99.17% interval -9.0 to -3.7 for coding minus random): under this interface the pretrained
  LAYA head gives no advantage and is a handicap. It does not show that a better-trained general head could never help.
- No specialisation effect (the four pretrained heads are fine-tunes of a common base, within 2.4 points). The probe still beats the reference by 8.6 points.

### 2l. Execution test: do the scorers pick test-passing fixes? (VAL, development-exposed)

Sources: `results/raw/val-outcome-v1-20261012T134540Z/pilot-20261012T224831Z` (OUTCOMES_A3.json, RESULT_A3.json), scores
`results/raw/glm-pilot-score-run-20261013T074323Z`; record HANDOFF_2026_10_11_STATUS.md 16.18-16.20. GLM-5.3 wrote 5 candidate fixes for each of 5 VAL
problems whose recorded test fails before the fix; every candidate ran twice in the isolated sandbox; the frozen scorers ranked the valid candidates.

| | Result |
|---|---|
| Valid candidates / passing among valid | 17 of 25 / 15 of 17 |
| Problems with at least one passing candidate (oracle) | 4 of 5 |
| Problems with both passing and failing valid candidates | **1 of 5** |
| Solved end to end: every learned scorer (LAYA, random head, baseline, probe) / first valid candidate / random among valid / uniform random / GLM sample 1 | 3 / 3 / 3.5 / 3.0 / 2 of 5 |

On the one informative problem every scorer run picked a failing fix: one problem, no conclusion either way. The pilot was not expanded. A second
pilot with gpt-oss-20b stopped at API verification (one gateway timeout, one empty answer), and a third with GLM-5.3 at low reasoning effort stopped the
same way (one gateway timeout, one valid answer); in both, the failing call was the second of two simultaneous calls on one key. A sequential re-check
had no timeout but one of its two answers wrapped the action in a list, which the decoder rejects, so that pilot also stopped.

A registered v3 of the low-effort pilot then ran (strict unwrap of a single-object list, local feature capture after a parity check, no Kaggle):
25 of 25 answered with no timeouts, but only 5 of 25 answers were valid (genuine model format errors), the oracle solved 2 of 5 problems, and only
1 problem was informative, so the expansion gate failed. On that problem every learned scorer picked the passing fix; end to end every learned
scorer solved 2 of 5 against 1 for the first valid candidate. Across both pilots there are two informative problems, one going each way: no
selector, LAYA included, is shown to help or to hurt.

**Mutation diagnostic (2026-10-13, 16.24):** given Fable's verified-passing fix and up to 4 copies broken by a single token, every learned scorer
(LAYA, random head, baseline, probe; with or without context) picked the passing fix 0-2% of the time against 20.5% for a random pick, with
probabilities within ~0.02 of uniform. **Corrected by the diff-focused diagnostic (16.25):** a linear probe trained on the pass / fail labels
(leave one state out) picks the passing fix in 89.5% of states with edit-site features and 100% with the existing whole-option features (random 20.5%,
permutation p = 0.001): the features carry the signal; the imitation-trained scorers do not use it. Open question: correctness or "this text was
mechanically edited" (test with the 36 mutants that still pass their tests). Until then code correctness comes from execution;
the re-plan is in LAYA_DECISION_BRAIN_PLAN_2026_10_13.md section 8, all numbers in ALL_NUMBERS_2026_10_13.md. The sandbox now also replays Go,
C# and Rust (`results/raw/wsl-toolchains-20261013/MANIFEST.md`).

### 3. Transfer to new trajectories (held-out, pre-registered)

Scorers fitted on the 120 `train` pools only, scored once on 160 held-out pools. Five seeds, chance 28.1%.
Source: `results/raw/align-transfer-v1-20261012T055725Z`.

| Scorer | Held-out accuracy | Interval | Per seed |
|---|---|---|---|
| Linear probe | 75.6% | not computed | single fit |
| Head, fresh adapter | 63.1% | 56.8 to 69.4 | 60.0, 61.3, 66.9, 62.5, 65.0 |
| Head, adapter aligned to the original encoder's option distribution | 58.4% | 51.2 to 65.4 | 51.9, 56.3, 60.6, 62.5, 60.6 |
| Head, adapter aligned to the original encoder's vectors | 54.9% | 47.9 to 61.6 | 57.5, 50.0, 47.5, 60.0, 59.4 |

- **Alignment hurts:** aligned minus fresh is -4.75 points (-9.6 to -0.3). The fresh adapter is kept.
- **Probe ahead:** probe minus fresh head is +12.5 points (+7.3 to +17.6).
- **More data helps:** the same 160 pools score 81.3% (table 2) once other pools from the same batches are in training. The two setups differ,
  so this is not a clean 18-point gain.

### 4. Pilot on the first 123 pools (in-dataset, 3-seed means, chance 29.6%)

Source: `results/raw/fullaction-pilot-v1-20261011T182942Z`.

| Scorer | Accuracy | Interval |
|---|---|---|
| Action-history prior | 37.2% | 27 to 46 |
| Pretrained head, no training | 36.6% | 32 to 42 |
| Options text only (no model state) | 62.6% | 54 to 72 |
| Head, adapter only | 77.2% | 71 to 83 |
| Head, adapter + pretrained layers | 76.7% | 70 to 83 |
| Head trained and tested with no context: adapter / joint | 76.7% / 78.3% | 70 to 85 |
| Linear probe, real context | 84.6% | 78 to 91 |
| Linear probe, no context | 84.6% | 78 to 91 |
| Linear probe, context from another state / another trajectory | 89.4% / 85.4% | 79 to 94 |

- Training the pretrained layers as well gave no gain (-0.5 points), so they stay frozen.
- **Context made no measurable difference** on these pools for any scorer.
- Scorers fitted on these 123 pools dropped about 20 points on 48 pools from new trajectories: probe 62.5%, head 57.6%, options text 47.9%
  (chance 26.9%).

### 5. Aligning the adapter to the head's original encoder

Training only the adapter on the head's own native dataset (2,988 items; 600 held out) makes the frozen head work on Qwen states for its native
questions: 71.5% gold accuracy against 78.3% for the original model, up from 19% to 26% unaligned. It did not help on the coding decisions
(table 3).

### 6. Judges

A judge pair must pass a gate on hidden controls before its verdicts count.

| Pair | Known-bad options accepted by both | Fable's action accepted by both | Near-duplicates caught |
|---|---|---|---|
| GLM-5.3 + Gemma 4 31B (28-pool pilot) | 0 of 46 | 86% | 11 of 11 |
| GLM-5.3 + Muse Glimmer 30B (same pilot) | 0 of 46 | 24 of 28 (85.7%) | 10 of 11 |
| GLM-5.3 + Muse, last TRAIN batch | 0 of 316 | 88.4% | 136 of 137 |
| GLM-5.3 + Muse, VAL batch | 0 of 114 | 94 of 104 (90.4%) | 50 of 51 |

Gemma's endpoint went down; of seven substitutes tried, only Muse passed the same gate. Results are reported per judge pair where it matters.

### 7. Known shortcut

A detector that sees **only the option texts** picks Fable's action 65% to 70% of the time (chance 28% to 30%). Alternatives are written by a
different model than the teacher, so part of any score here can come from recognising the teacher's style.

### 8. Other data sources examined (metadata only, by subagents; not independently re-checked)

Reports in `results/raw/research-20261012/`.

| Source or method | Finding |
|---|---|
| Other public repositories of the teacher's sessions | Mostly copies of each other; the distinct originals hold a few hundred sessions (18, 63, 79, 244), some without tool results; four need a login |
| Ready-made decision datasets with alternatives | One 50,000-row set converts without a language model; others partly; useful for general skills, not coding |
| Recorded branches inside our traces (failed action, then retry) | 51 pairs in 11,436 decisions, 11 clearly from the same state: too rare |
| Borrowing alternatives from similar decisions | Plenty of volume, only a surface check of relevance |
| Programmatic mutations of the teacher's code, judged by tests | 118 replayable TRAIN states, 87% with at least five mutation sites; not yet run |

## What is not shown

- **The independent test is small.** 53 pools on 36 trajectories (46 after cleaning): enough to rule out a collapse, too few to rank scorers finely.
- **No evidence about coding-task success.** Everything measures agreement with the teacher. The one outcome pilot produced no usable candidates.
- **The head has not beaten the linear probe** in any comparison.
- **The knowledge mostly sits in the adapter.** Head-only training reaches about 54% where adapter training reaches 76% on the same pools. Adding a read path
  (section 2j) puts the gain in the new layer; the 31 pretrained tensors did not get stronger in any tested recipe.
- **Portability is untested.** Only one language model has been used, so nothing shows the head helps a different model.
- **Value of context is only partly shown.** Removing the context at test time costs 13 to 15 points on VAL, but models trained without
  context matched models trained with it in the 123-pool pilot.
- **Part of every score can be style recognition.** Alternatives are written by a different model than the teacher.
- The locked A5 benchmark from the earlier phase was not run on this head.

## Reproducing

Python 3.12 through `uv`. CPU jobs set `ATEN_CPU_CAPABILITY=avx2` (the scripts do this themselves).

```bash
uv sync
```

Fine-tune on all pools and write the report and adapters (about 4 minutes on a CPU; needs the captured features under `results/raw/`):

```bash
PYTHONPATH=. .venv/Scripts/python.exe tools/laya_finetune_all_v1.py results/raw/laya-fullaction-set-v3-20261012T055420Z results/raw/level12-run-v2-cancelled-20261011T175252Z results/raw/level12-bundle-20261011T151731Z results/raw/v3-context-capture-run-20261012T065106Z results/raw/level12-bundle-20261012T060047Z results/raw/laya-finetune-all-v1-<new stamp>
```

| Task | Script |
|---|---|
| Select decisions and build generator requests | `tools/hard_batch_v3_conv.py` |
| Send generator requests (parallel, rate-limited, logged) | `tools/nemotron_candidate_pilot.py` |
| Build pools, sandbox syntax check, objective screen | `tools/hard_batch_v2_build.py`, `tools/hard_qualify_v2_syntax.py` |
| Blinded review and per-option scoring | `tools/pair_pilot_v1.py`, `tools/judge_qual_v1.py` |
| Assemble strict pools | `tools/hard_batch_v2_assemble.py` |
| Freeze a dataset version | `tools/laya_fullaction_freeze_v3.py` |
| Bundle and capture features on Kaggle | `tools/level12_bundle.py`, `kaggle/level12_t4.py` |
| Baselines and pilot | `tools/fullaction_pilot_v1.py` |
| Adapter alignment and held-out comparison | `tools/adapter_align_v1.py`, `tools/align_transfer_v1.py` |
| Fine-tune on all pools | `tools/laya_finetune_all_v1.py` |
| Score saved models on VAL pools | `tools/val_eval_v1.py` |
| Capture features on a local 6 GB GPU (weights streamed from disk) | `tools/val_capture_local.py` |
| Diagnose the head-versus-probe gap; clean strict equivalents | `tools/val_gap_diag_v1.py`, `tools/laya_clean_v4.py` |
| Adapter-only against adapter + head; head-only with a frozen bridge | `tools/laya_joint_v1.py`, `tools/laya_headonly_v1.py` |
| Sandbox replay check and outcome evaluation | `tools/fable_materialize_states.py`, `tools/fable_wsl_runner.py`, `tools/cal_oracle_check.py`, `tools/cal_outcome_v1.py` |
| Unattended generation-to-scoring chain | `tools/stage2_chain.sh` |

Generation and review need hosted-model API keys in `.secrets/` (not in git). Outputs are create-only: scripts refuse to overwrite a result.

## Repository layout

| Path | Contents |
|---|---|
| `decision_head/` | Head, frozen trainer (`fable_train.py`), feature protocol, pool contracts |
| `capture/` | Hidden-state capture (shared prefix, one independent suffix per option) |
| `tools/` | Pipeline, training and evaluation scripts |
| `kaggle/` | Notebook scripts run on Kaggle GPUs |
| `configs/` | Pinned models and pre-registered thresholds |
| `policy/`, `state/`, `compiler/`, `teachers/` | Hard safety rules, schemas, state builder, teacher adapters from the earlier phase |
| `results/raw/` | Row-level evidence for every reported number (not in git where large) |
| `models/`, `data/` | Model files and datasets (not in git) |
| `progress_ui/` | Small read-only progress page |

## Earlier work

Before the Fable imitation work, the project built and tested (details in [AGENT_README.md](AGENT_README.md) and the ledger):

- **Runtime checks.** Qwen3-4B streamed and resident runs are bit-exact on 12 layers; CPU against GPU KL 5.5e-4 to 6.6e-4. Capture speed
  1,017.6 tokens per second on 8,192-token inputs. The 120B model's expert read passed at 0.87 seconds per simulated token (I/O only).
- **A5 benchmark (4,396 decisions), now diagnostic only.** A text-only baseline reached 0.398 (0.323 on covered decisions), and a rule with no
  model (most frequent action in history) reached 0.550, so the benchmark failed its own integrity gate and was frozen.
- **Pilot pointer head.** 0.667 on covered pilot rows, matched by eight trivial features with no model (0.663): no signal beyond pool structure.
- **Laya baselines.** `laya-typed-decisions` reproduced at 0.771 on its own test (0.766 published) and scored 0.178 on covered pilot rows.

These results are why the current work uses reviewed pools, a mandatory probe baseline and shortcut checks.

## Roadmap

**Status of the plan below (2026-10-13, 13:40 IST):** steps 0-2 done (exp5 documented, baseline frozen, mutation pilot v1.1 usable 19/20); step 3 ran
with GLM-5.3 (section 2l: 1 informative problem of 5, not expanded); the gpt-oss-20b pilot stopped at API verification. Head-specialisation research is
paused (section 2k). Next: the user's decision on a generator that yields mixed passing / failing pools, or the labelled mutant diagnostic.

**Current (user, 2026-10-13):** head-only training and new Fable generation are stopped. (0) document exp5; (1) freeze a research baseline (pretrained head,
three Fable-trained adapters, linear probe, with hashes); (2) mutation pilot on 20 TRAIN states in the local WSL sandbox (a diagnostic: these states were trained
on); (3) next priority, needs a GO for generation: score candidates written by another agent on the 23 fail-to-pass VAL decisions against sandbox test results,
with the probe, random choice and the agent's own first candidate as baselines. Not approved: a second bridge, CAL / TEST use, more Fable data.

Earlier list, proposed on 2026-10-12 (steps 1 and 2 were done as sections 2i and 2j):

1. Longer head-only run with the never-trained-on-teacher bridge at the low learning rate: is 54% a ceiling or was it undertrained?
2. Head-only training with general decisions mixed in (the head's 2,988 native items, already captured): does replay stop forgetting?
3. Mutation pilot on 20 TRAIN states (approved): verified, same-style alternatives without a generator or judges.
4. Recovery pilot on 25 of the 150 near-miss decisions (approved); the full 150 only after the yield is measured.
5. If 1 and 2 are positive: 3,000 rows of an outside decision dataset, captured locally, in the same head-only comparison.
6. A second small language model, to test whether the trained head carries over. Canonical TEST stays untouched until a final evaluation.
