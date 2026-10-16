# LAYA decision head: what has been established (write-up, 2026-10-15)

Goal throughout: make the pretrained LAYA `Coding_Decision_Agent` (ModernBERT-large encoder + decision head) help smaller coding agents choose
better next actions. This document records what was built, what was measured, and why each route stopped. Every number points to a result folder
under `results/raw/`.

## 1. Infrastructure that works (reusable)

| Component | Status | Evidence |
|---|---|---|
| Original LAYA checkpoint on Kaggle T4 x2 (laya 0.3.22, fp16) | scores 25k+ items, 0 failures, probability parity with the library 5e-5 | `text-grader-k2-v2-20261014T0724Z` |
| Fixed renderer k2_render_v1 (2,048 tokens, state budget 1,969, action + task protected from right-truncation, pre/post-action variants, leak checks) | task kept 100%, action intact 90-97% | K2, text-imitation, outcome-head reports |
| Head-only trainer on the frozen encoder (encoder hash checked before/after, gradients only in head + scorer, listwise or binary loss, pre-registered lr overfit rule, early stopping on validation groups only) | runs end to end; the lr rule caught the 2e-5 "head never moves" bug | `text-imitation-v1-*`, `outcome-head-v1-20261015` |
| Evaluation discipline: pre-registration before scoring, repo/task-grouped splits, cluster bootstrap, shortcut and position controls, options-only / observation-only baselines | caught three false positives (dryrun random scores, position confound, authorship shortcut) | K2 v1 READING.md, K2 v2, text-imitation |
| WSL execution sandbox (bubblewrap, offline, no secrets) reproducing complete SWE-rebench workspaces at fixed commits; replay of recorded prefixes; workspace hashes; fidelity rules v3/v4 | 2 of 2 starting states reproduced (v3); 0% environment failures over 12 rollouts | `wsl-feasibility-v2-*`, `sandbox-pilot-v1-*` |
| CodeTraceBench integrity inventory (safe extraction, secret scan, per-format alignment evidence levels) | 3,316 runs inventoried, 0 failures | `ctb-k1-inventory-20261014T2246Z-full` |
| Kaggle operations: private datasets, kernel resume, host-failure handling, browser log recovery for ERROR kernels | used throughout | this session |

## 2. Negative results (all measured, all pre-registered)

| # | Experiment | Result | Folder |
|---|---|---|---|
| 1 | Unchanged LAYA, zero-shot, does it localise the decisive coding error (RootSE) or a wrong step (AgentProcessBench)? | No evidence beyond position / action-type controls. RootSE decisive-step local-peak rate 0.29-0.36 (chance 0.33); APB position-residualised AUROC 0.47-0.54 | `text-grader-k2-v2-20261014T0724Z` |
| 2 | CodeTraceBench (19,169 human step flags): can each flag be tied to its exact action? | 0 independently verifiable mappings; Terminal formats count-consistent but references positional; author request NJU-LINK/CodeTracer#11 | `ctb-k1-inventory-20261014T2246Z-full` |
| 3 | Imitation of Fable's recorded action, LAYA's own frozen encoder + trained head | Gate FAIL: VAL head 32.6%, probe 50.0%, options-only 75.0%; candidate text alone identifies Fable's choice | `text-imitation-v1-20261014T102605Z` |
| 4 | (earlier) Imitation through Qwen states + adapter + head; head swap; mutation scoring; Nebius exact branching | probe always beat the head; mutation scorers at chance; branchpoints are trivial early orientation steps | HANDOFF section 16 |
| 5 | Execution-verified action comparisons with GLM continuations, early decision points | 0 BETTER/WORSE labels: a 15-step continuation cannot finish tasks whose solving runs need 37-86 actions | `sandbox-pilot-v1-20261014T162937Z` |
| 6 | Same at late decision points | 0 qualified states: recorded runs used a 1,000-column tty and clipped editor views our sandbox does not reproduce; late steps are mostly post-fix cleanup | `sandbox-pilot-v2-20261014T210515Z` |
| 7 | Train LAYA's `outcome` head to predict run success from the state + action at steps 3/10/20 (Nebius, 1,741 mixed-outcome tasks) | Every arm at chance on 172 unseen repos: trained head AUROC 0.503, unchanged 0.505, TF-IDF 0.490, probe 0.474; within-task pair AUROC 0.50 for all | `outcome-head-v1-20261015` |

## 3. Why (the common cause)

1. **The supervision did not contain the signal we asked for.** Every label source was one of: a teacher's choice (imitation, dominated by
   authorship and "usual action" priors), hindsight annotations we cannot align to actions, or a run-level outcome. A run-level outcome is decided
   by dozens of later actions; a single early step carries almost none of it, so no model (trained or not, LAYA or TF-IDF) found any (result 7).
2. **The model is not the bottleneck.** In every experiment the frozen-encoder probe or a text baseline matched or beat the trained head, and
   when the label was learnable at all, simple baselines learned the same shortcut. Swapping models, encoders or recipes would not fix the labels.
3. **Real counterfactual labels are expensive here.** Same-state "A vs B" evidence needs long continuations (dozens of steps at ~1-2 min per GLM
   call) and exact environment parity (tty width, tool clipping); each fix costs hours before a single label exists.
4. **Evaluation traps were real and were caught:** a dryrun on random scores, error steps always last in their comparison set, and candidate
   authorship. Without the pre-registered controls, three of the results above would have looked like successes.

## 4. Next direction with fresh eyes

Stop asking a per-step classifier to predict long-horizon outcomes. Move the decision to where the outcome is decided and verifiable:

**Recommended: patch verification ("is this fix likely correct?").** The decision a small agent most needs help with is whether its current
patch solves the issue, so it can keep fixing, test more, or stop. The signal is concentrated there (the patch itself), the labels already exist
with real test execution (Nebius `model_patch` + `resolved`, 67,074 runs on disk), and LAYA has a native question for it
(`code_review.likely_correct`). Two cautions are known in advance: LAYA's code_review family was pre-trained on nebius SWE-agent trajectories,
so evaluation must use the non-overlap repos; and the pre-registered controls (patch length, files touched, TF-IDF, probe) must be beaten.
Order: a cheap zero-shot check of the unchanged checkpoint first (inference only, ~1 h on T4); train only if the label is learnable by some arm.

Other options kept open: CodeTracer step labels if the authors reply (#11); environment-parity sandbox work only if patch verification shows
that execution-level labels are worth their cost.
