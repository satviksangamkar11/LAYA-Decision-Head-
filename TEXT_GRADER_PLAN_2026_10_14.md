# Text-native LAYA step grader: consolidated recommendation (2026-10-14)

Status: PLAN ONLY. Nothing trained, launched, uploaded or fetched beyond the research samples (~100 MB, temp folders outside the project).
User GO (chat, 2026-10-14): revised defaults: encoder frozen, Open-Jev excluded, Kaggle, prepare small datasets; NO head training until a genuinely
new coding-labelled set exists, CodeTraceBench first.

Evidence (all 2026-10-14, read-only): `results/raw/research-20261014/`
- `CTB_codetracebench_audit.md` (CodeTraceBench)
- `RSE_APB_audit.md` (RootSE, AgentProcessBench)
- `LAYA_runtime_trainer_audit.md` (checkpoint rendering, trainer, head-only path)
- `STEP_QUALITY_DATASETS_search.md` (other step-label datasets; nothing better than CodeTraceBench)

## 1. What the research settled

| Question | Answer (measured unless marked) |
|---|---|
| New coding labels not seen by the checkpoint? | CodeTraceBench `full`: 17,701 incorrect + 1,468 unuseful human (author) step flags; 0 repo / 0 instance overlap with LAYA's nebius SWE-agent cases |
| Label quality | hindsight (annotators saw failing tests); kappa 0.73 only for the error-critical step label; flags dense (median 75% of a flagged stage) |
| `verified` independent? | No: strict subset of `full`, identical labels. We make our own grouped split |
| Positives | none verified. Least doubtful: 26,220 unflagged steps in solved, unflagged runs. Unflagged steps in unsolved runs: excluded |
| Parseable today | 646 SWE-style runs (906 incorrect, 88 unuseful, 18,688 unflagged). Terminal-Bench runs hold ~95% of flags and need per-agent parsers |
| Download | artifacts 0.45 GB (whole repo 3.52 GB, of which 2.89 GB is a duplicate unpacked copy) |
| Independent evaluation | RootSE: 102 decisive error steps (72 non-OpenHands runs with no LAYA / rebench overlap = primary; 30 OpenHands = secondary). AgentProcessBench: 8,509 human step labels, NOT coding, out-of-domain only |
| Runs the original checkpoint? | yes: `kaggle/align_capture_t4.py` `teacher()` (3,588 items, 252 s on T4, 0 failures; 78.3% on its own 600 val items) |
| Head-only training | trainer `train_rlcd.py` has `detach_encoder`; freeze encoder `requires_grad`; few-line change; ~4-6 GB on T4 (estimate) |
| Context budget | 2,048 tokens; question+options capped at 320; state gets the rest and is truncated ON THE RIGHT. Contexts are 20-70x too long |

## 2. Dataset manifest (one mixture)

| Role | Source | Items | Use |
|---|---|---|---|
| Train / dev / test | CodeTraceBench `full`, grouped split (SWE: by repo; Terminal-Bench: by task; all runs of a task together), ~70/10/20 = 2,353 / 267 / 696 runs | negatives = incorrect, unuseful; weak positives = unflagged steps in solved, unflagged runs; everything else excluded | train the head; dev for temperature + early stopping; test once |
| Independent coding eval | RootSE (72 non-OpenHands primary, 30 OpenHands secondary) | 102 decisive steps vs 1,915 earlier steps, paired within run | never trained or calibrated on |
| Out-of-domain eval | AgentProcessBench (+1 vs -1; 0 dropped) | 8,509 steps | reported separately, never as coding evidence |
| Regression only | the checkpoint's own coding-decision-cases val items (the 600 native items) | 600 | the head must not lose its existing skills (78.3% now) |
| Excluded | When2Call, CodeUltraFeedback, AgentJudgeBench, typed-decisions (already in the checkpoint), Open-Jev, the hindsight-judge `swev-prm` set (licence unstated; possible later auxiliary) | | |

## 3. Fixed rendering (one compressor for every arm)

Because the library truncates the state on the right, the order is fixed so the decisive parts can never be cut:
1. the proposed action (clipped to 600 tokens);
2. the task statement (first 400 tokens);
3. the most recent history, newest first: (action, observation clipped to 300 chars) pairs until the budget is full.

Future information is never rendered: later steps, the judged step's own observation, `solved`, `report.json`, `patch.diff`, test output,
`failure_id`, `gold_patch`, agent and model names. The compressor version is frozen before any score is seen; the unchanged checkpoint, the probe,
the baselines and the trained head all read the same rendered text.

Question: a new question type `step_quality` with options `good / unproductive / incorrect` (text options, so no new parameters). The unchanged
checkpoint is also scored on its existing `agent_trace.action` (continue / observe / human_review / stop) and `agent_trace.outcome`
(success / partial / failure / harmful); the zero-shot score is P(not continue) and P(failure or harmful), both pre-registered, Holm-corrected.

## 4. Baselines and shortcut controls (fixed before any model is scored)

- Label-table shortcuts: step position, relative position, agent, model, run length (no text).
- Action-type rules R1-R6 (NEXT_ACTION_FEASIBILITY section 4).
- TF-IDF + logistic regression on the same rendered text.
- Frozen-encoder linear probe (the mandatory probe baseline, here on LAYA's own encoder) on the same items.
- Unchanged checkpoint, zero-shot (section 3).
- Action-only and context-only renderings (is the context used at all?).
- Stratified reporting inside unsolved runs only (92% of incorrect steps sit in unsolved runs, so "unsolved" is the main shortcut).

## 5. Metrics and gates (pre-registered in a new `configs/thresholds_text_grader_v1.toml` before K2)

- Primary: CodeTraceBench test AUROC and AUPRC (negative vs weak positive), plus the same inside unsolved runs; RootSE paired win-rate (decisive
  step scored worse than the earlier steps of the same run), 72 non-OpenHands runs.
- Secondary: macro-F1 of the 3 classes, ECE after a temperature refit on dev, AgentProcessBench AUROC (OOD), regression on the 600 native items.
- Head-only GO-to-use gate: beats the best of {probe, TF-IDF, unchanged checkpoint, shortcuts} on CodeTraceBench test with a paired-bootstrap 95%
  CI above 0, AND RootSE win-rate not lower than the unchanged checkpoint, AND native regression drop <= 2 points.
- If the gate fails: stop. The next choice (encoder fine-tuning on the same data, or a different architecture) is the user's.

## 6. One Kaggle workflow (compute on Kaggle only)

| Step | Where | What | Needs |
|---|---|---|---|
| P1 | PC, code only | write per-agent parsers + the compressor; test on the 7 sample archives already in the temp folder (no new download) | nothing |
| K1 | Kaggle CPU, internet on | fetch CodeTraceBench artifacts (0.45 GB) + RootSE + AgentProcessBench from Hugging Face inside the kernel; parse; gate: rebuilt step count = manifest `step_count` on >= 95% of runs per agent format (formats that fail are dropped and reported); build rendered items, the split, the shortcut + TF-IDF baselines | **user yes** for the 0.45 GB fetch (under 1 GB, but asked anyway) |
| K2 | Kaggle T4 | one pass of the unchanged checkpoint: zero-shot scores + frozen encoder features for the probe, all items, fp16 (every arm on the same T4 fp16 path) | **user yes**: running the frozen LAYA encoder (forward only) |
| GATE | PC, read results | is there a real labelled signal and room above the baselines? If the unchanged checkpoint or probe already meets the gate, head training is not needed | |
| K3 | Kaggle T4 | 1-batch overfit + timing check, then head-only training on CodeTraceBench train, dev for early stopping + temperature; score test, RootSE, APB, native once | **user GO for fine-tuning** |

Estimated time (unmeasured): P1 is the longest step (3 Terminal formats); K1 under 1 h; K2 about 70 ms per item on T4, so ~1-2 h for ~60-100k items;
K3 a few hours.

## 7. Standing-rule items that need the user's decision

1. "No encoder is trained or touched for now" (2026-10-12): K2 runs LAYA's own encoder forward, frozen. Proposed reading: forward use is allowed,
   training stays forbidden. Needs a yes.
2. Mission line "Laya-derived checkpoints are teachers and specialists only, never a runtime dependency": a text-native LAYA grader at runtime
   breaks it. Not needed for K1-K3; needed before deployment.
3. The 0.45 GB CodeTraceBench fetch inside Kaggle (and RootSE / AgentProcessBench, ~70 MB) instead of uploading from this PC.
4. Fine-tuning GO for K3 (after the K2 gate).
(Open-Jev and paid compute: excluded under the revised defaults; no conflict.)

## 8. Honest limits

- Labels are hindsight human flags; "good" is only "not flagged in a solved run". The result is a step critic, not a proof of better task
  completion; that needs the agent test afterwards.
- Premature stopping has almost no step labels anywhere; the grader will not learn it from this data.
- 95% of the flags depend on Terminal-format parsers that are not written yet; if they fail their gate, training data shrinks to ~1k negatives,
  which is enough for evaluation but thin for training.
- Contexts are compressed 20-70x; what the grader cannot see, it cannot judge.

## 9. Amendments after the user's review (2026-10-14); these override sections 2-7 where they differ

1. **First target: coding action-ERROR detection, not next-action choice.** Two tasks, trained and reported separately, never pooled:
   - **A. After-action critic (first target):** state + action + that action's own observation -> incorrect / unproductive / weak-good.
   - **B. Before-action grader (strictly separate experiment):** state + proposed action only; the action's observation is never rendered.
   For both, nothing after the judged step is rendered (later steps, final outcome, test output, patch). The annotators' hindsight is recorded per
   item; the evidence cutoff is the step itself.
2. **Parsers: reuse CodeTracer's own normalizers first** (official repository: OpenHands, mini-SWE-agent, Terminus2). Our own parser only for a
   format whose upstream normalizer fails the alignment test. Alignment test per format: rebuilt step count = manifest `step_count`, and every
   flagged step's `action_ref` text matches the rebuilt action. Formats or runs that fail are excluded, never guessed.
3. **Three label states in the manifest:** explicit negative (incorrect / unuseful), weak positive (unflagged step in a solved run with no flags),
   unknown (everything else; excluded from training and from metrics). Weak positives are never called correct.
4. **Manifest fields per item:** source, task, run, agent, step id, action type, step position, evidence cutoff, render version, label state,
   hindsight flag, truncation (tokens dropped).
5. **`solved` is NOT a baseline.** It is unavailable before the action; it is used only as a leakage diagnostic (if a model's score tracks `solved`
   beyond what the rendered input allows, look for leakage). Fair baselines: position, action type, agent format, R1-R6, TF-IDF, frozen-encoder probe,
   unchanged checkpoint.
6. **Dependence:** flags cluster in contiguous failure episodes, so every CI is a cluster bootstrap over runs (and tasks), and a secondary metric
   counts one item per failure episode (its first flagged step).
7. **Metrics:** precision-recall (AUPRC) on incorrect / unproductive as primary, AUROC, then coverage (share of items rendered without losing the
   proposed action) and truncation, all per task (A / B) and per action type.
8. **Gates before any training:** (i) label alignment passes for every format used; (ii) temporal integrity: an automated check that no rendered
   item contains text from after its cutoff; (iii) the 2,353 / 267 / 696 grouped split passes a group-integrity check (no task, and for SWE no repo,
   in two parts). Any failure stops the plan before training.
9. **Not repeated:** Qwen-adapter, mutation, branching and pretrained-head-on-Qwen experiments. Open-Jev, Azure, more generated trajectories: HOLD.

## 10. P1 result (2026-10-14): alignment is the blocker (results/raw/ctb-align-check-v1-20261014/FINDINGS.md)

- SWE-agent and mini-SWE SWE-style: step counts match (35/35, 9/9), count-only evidence.
- OpenHands SWE-style: 25 vs 23, unit undefined.
- Terminal: mini-SWE and Terminus2 samples lack the step source (published artifacts are incomplete); OpenHands has three conflicting indexings and
  the manifest action_refs come from a positional backfill, so they cannot prove alignment.
- Terminal formats hold 17,865 of 19,169 flags. If they stay unrecoverable, the clean training set is the SWE-style part only (about 906 incorrect +
  88 unuseful, fewer after alignment), enough for an evaluation and a small head experiment, not a strong training set.
- Next: K1 stage 1 = the decisive per-format alignment test over ALL archives on Kaggle CPU (no model), before any rendering.

## 11. K1 and K2 results (2026-10-14); zero-shot testing CLOSED (user)

- K1 (`results/raw/ctb-k1-inventory-20261014T2246Z-full`, Kaggle CPU, 3,316 runs, revision aa213b8): 0 independently verified label-to-action
  mappings. Step counting is consistent for Terminus2 (response commands 914/917) and OpenHands Terminal (all agent actions 690/787), but the
  published data has no anchor independent of the positional reference backfill. 0 admissible CodeTraceBench training labels under the 100% rule.
  Archives contain secret-like strings (1,709 pattern hits, 788 secret-named files); outputs hold none unmasked.
- K2 v1 (genuine full run, version 3 of text-grader-k2-v1; an earlier version was a dryrun on random scores and is discarded): within-run
  metrics confounded by position (the error step was always last in its comparison set). Reading: `results/raw/text-grader-k2-full-v3-pull-20261014T0510Z/READING.md`.
- K2 v2 (position-controlled, `configs/thresholds_text_grader_k2_v2.toml`, version 3 of text-grader-k2-v2, 6,282 new + 18,774 reused scores):
  verdict before and after the action: "no evidence beyond position/action-type controls; stop zero-shot testing". RootSE 71 runs: the unchanged
  checkpoint's 4 native questions give decisive-step local-peak rates 0.29-0.36 (chance 0.33); APB position-residualised AUROC 0.47-0.54.
  Report: `results/raw/text-grader-k2-v2-20261014T0724Z/out/REPORT.md`.
- User reading (2026-10-14): this does not show the encoder is useless, only that this evaluation did not demonstrate coding-error recognition by
  the unchanged checkpoint. Next priority: reliable coding-action labels (CodeTracer author request; else sandbox-verified decisions), then one
  supervised experiment defined but not launched. Architecture change and fine-tuning: HOLD.
