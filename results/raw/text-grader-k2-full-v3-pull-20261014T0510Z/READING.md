# K2 full (Kaggle version 3): verification and reading (2026-10-14, 10:45 IST)

Pulled read-only into this folder (the launching chat's folder text-grader-k2-full-v3-20261014 left untouched).

## Real inference: verified
- run_meta.mode = full; 2 workers, each "Tesla T4", fp16, 9,387 items each = 18,774 scored; 0 failures, 0 retries, 0 non-finite;
  max prob difference vs the laya library 5.0e-05; 4,260 s scoring. Not the earlier dryrun.

## Headline (RootSE primary cohort, 72 non-OpenHands runs, pre-action)
| arm | win-rate | percentile | top-1 (chance 0.134) | AUROC (secondary) |
|---|---|---|---|---|
| unchanged LAYA, needs_review (best zero-shot) | 0.704 | 0.647 | 0.239 | 0.643 |
| step index (out-of-fold) | 0.676 | 0.741 | 0.366 | 0.603 |
| action type (out-of-fold) | 0.824 | 0.810 | 0.494 | 0.862 |
| TF-IDF full text (out-of-fold) | 1.000 | 0.944 | 0.718 | 0.870 |
| frozen-encoder probe CLS (out-of-fold) | 0.901 | 0.862 | 0.634 | 0.879 |
zs_action is inverted (0.282); zs_outcome near chance.

## Design flaw (mine, in the K2 brief): the error step is always the LAST item of its set
- RootSE items = earlier steps + the decisive step; AgentProcessBench episode metric = first -1 step vs the +1 steps BEFORE it.
  So "pick the latest step" wins by construction: oof step index is 1.000 on the APB episode metric, and any feature that grows with
  history (rendered length, history tokens) is a position proxy. TF-IDF win-rate 1.000 and the probe numbers are therefore NOT evidence of
  error recognition. The within-run metrics as registered cannot separate "recognises the error" from "recognises the latest step".
- What is still informative: the unchanged checkpoint's zero-shot scores are not fitted, yet even the best (needs_review) is below the
  step-index baseline on percentile and top-1. So no evidence yet that unchanged LAYA localises coding errors beyond position.

## Fix (needs a new pre-registration v2; no change to v1 results)
- Position-controlled comparisons: (a) include RootSE LATER steps as items (each scored with only its own past; they need new Kaggle
  inference: ~3,251 x 2 items), so the decisive step is no longer always last; (b) matched-position pairs: decisive step vs the
  immediately previous and next step; (c) report every arm's score after regressing out step index / history length within run.
- APB: score all labelled steps of a trajectory, including those after the first error (labels propagate, so use first -1 vs +1 steps
  AFTER recovery too where they exist), or use all-step AUROC with position-residualised scores as primary.
