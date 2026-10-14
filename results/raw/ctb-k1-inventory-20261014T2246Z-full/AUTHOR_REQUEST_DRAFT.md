# DRAFT (not posted; needs the user's approval): GitHub issue for NJU-LINK/CodeTracer

**Title:** Request: per-trajectory normalized steps.json used for the CodeTraceBench step annotations

Hello, and thank you for releasing CodeTracer and CodeTraceBench (MIT).

We are using the `full` split (dataset revision `aa213b84ffb6690fc37ca15766d6ca174ec36d4d`) to evaluate step-level error detection, and we
cannot map every annotated `step_id` to a unique action in the published artifacts. Could you share, for the annotated trajectories:

1. the normalized `steps.json` (or equivalent) that the annotators saw, i.e. the exact step list each `step_id` refers to; and
2. if available, the mapping from annotation `step_id` to the raw event / episode / command in each artifact.

What we observed (counts only; parsers from CodeTracer commit `2d302191`):

- **OpenHands (Terminal), e.g. `openhands-Moonshot__Kimi-K2-250905-ilp-solver-73ad79cc`:** `step_count` = 15 and the stages span steps 1-15,
  which matches counting all agent actions (think, edit, run, read, run_ipython, finish). The OpenHands parser yields 7 steps (run and
  run_ipython only), and the `action_ref` fields match a run-only ordering (step 2 -> event 11, step 3 -> event 13, step 5 -> event 23),
  consistent with `scripts/backfill_step_refs.py` assigning references by position. Under the two orderings, flagged steps 2/3/5/6 refer to
  different actions. Across the split, all-agent-action counts equal `step_count` in 690 of 787 OpenHands Terminal runs.
- **Terminus2, e.g. `terminus2-Moonshot__Kimi-K2-250905-hf-train-lora-adapter-685faf4d`:** `step_count` = 59 with flags at steps 32-35, but the
  artifact contains 8 episodes. Across the split, response-command counts equal `step_count` in 914 of 917 runs, but no flagged step carries an
  `action_ref`, so we cannot confirm the mapping.
- **mini-SWE-agent (Terminal), e.g. `miniswe-Qwen__Qwen3-Coder-480B-A35B-Instruct-polyglot-rust-c-20cc77a5`:** `step_count` = 61, but
  `agent-logs/` is empty (no `mini.traj.json`) in 192 of 833 runs, so these steps cannot be rebuilt from the artifact.
- The annotation directory referenced by `annotation_relpath` is not in the dataset repository.

Separately, and for your awareness: some artifacts contain strings matching credential patterns and files such as `sessions/.jwt_secret`.
We have not published or used any of them; you may want to check whether any are real before further redistribution.

Thank you very much for any help.
