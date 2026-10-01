# A4 leakage firewall, v1.1 (addendum to v1; written after reading 20 SWE-rebench-openhands rows read-only)

v1 is frozen by hash (`results/raw/a4-leakage-rules-v1.sha256`) and unchanged. This file adds the trajectory structure and the rules that follow
from it. It is frozen by its own hash (`results/raw/a4-leakage-rules-v1_1.sha256`) before any bulk compilation.

## What the rows look like [M, 20 rows of `nebius/SWE-rebench-openhands-trajectories`, 5.5 MB in total]

- Each `trajectory` is an OpenAI-chat message list. Roles over the 20 rows: system 20, user 20, assistant 1,211, tool 1,191. Every event has the same
  keys: `content`, `name`, `role`, `tool_call_id`, `tool_calls`. Events per trajectory: 85 to 185.
- An assistant event carries `content` (a short preamble, often empty) and `tool_calls` (function name plus JSON arguments). Its result is the next
  `tool` event. Tools used: `str_replace_editor` 494, `execute_bash` 631, `think` 53, `task_tracker` 13, `finish` 20. About 60 decisions per trajectory.
- Episode fields: `resolved` 13 of 20 (so 35% unresolved), `exit_status` is `submit` for all 20, `gen_tests_correct` and `pred_passes_gen_tests` are often None.
  `model_patch` is a full diff (about 28k characters on row 0). Row size is about 277 KB, so 300 rows is about 83 MB [E].

## New leak, caught by reading the structure

The assistant message at step `t` (its `content` **and** its `tool_calls`) is the label. Its preamble ("Let me run the tests...") is produced in the same
turn as the action and states the intent. **The state for the decision at `t` must contain only events with index below `t`**, and never any part of event `t`.
Earlier assistant events, including earlier `think` thoughts, are observable at `t` and may appear.

## Event classes (ACTION decisions)

| Event or field | Class |
|---|---|
| system message (event 0) | excluded: constant text, no information (it contains the words "resolved" and "test" in generic instructions, so it must also stay out of the label-vocabulary scan) |
| first user message (the issue) | OBSERVABLE_AT_T |
| earlier assistant `content` and `tool_calls` | OBSERVABLE_AT_T |
| earlier `tool` results | OBSERVABLE_AT_T, cut tail-first to the token budget (max single result 30k characters; median event 218 characters) |
| `tool_call_id`, `name` of events | excluded: random ids, no information |
| `tools` | excluded: constant list of 5 tools |
| the assistant event at `t` and later events | the label and the future: never in the state |
| `resolved`, `exit_status`, `gen_tests_correct`, `pred_passes_gen_tests`, `model_patch`, `instance_id`, `trajectory_id`, `repo` | as in v1 |

## Rules that follow

1. **State builder:** events `[1, t)` minus the exclusions, rendered by one function, tail-first, with a hard token budget that raises on overflow (no silent truncation).
2. **Action normalisation:** the label action is `(tool, normalised arguments)`. NEXT_ACTION_TYPE is derived deterministically: `str_replace_editor` view, edit (str_replace, create, insert), `execute_bash` classified by command pattern into search, run tests, other, `think`, `task_tracker`, `finish`.
3. **STRONG tier, concrete:** an edit or create on a file that appears in the `model_patch` diff headers, in a trajectory with `resolved == 1`. A read or view of a file in that diff, before its first edit, is WEAK. Test runs and everything else in a resolved trajectory are WEAK. Unresolved trajectories give no positive label.
4. **Candidate generator constraint:** candidates for a decision at `t` may be built only from information at `t`: the true action, the agent's own earlier actions (stale repeats are natural hard negatives), and generic actions instantiated with files and commands seen in the state. **Never from later actions of the same trajectory**: they correlate with the final patch. Every candidate, including the true one, goes through the same rendering function so that "looks synthetic" is not a feature; the candidate-only shortcut baseline measures what is left.
5. **Failure-analysis hooks:** the compiler reports, per run, the share of true actions whose file or command string never appeared in the state (a high share means the label is unguessable by construction and the ACTION benchmark needs the candidate-only baseline next to it).

## Not decided yet

Token budget numbers, the command-pattern table for `execute_bash`, how many distractors per decision, and the tier weights. The budget and patterns are fixed before compilation; the tier weights go to `configs/thresholds.toml` before any training run.
