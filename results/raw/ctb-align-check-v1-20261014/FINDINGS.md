# P1 alignment check on the 7 CodeTraceBench samples (2026-10-14; local CPU exception approved by the user for these 7 only)

Tool: `tools/ctb_align_check_v1.py`; upstream parsers NJU-LINK/CodeTracer @ 2d302191 (MIT), copied to a temp folder, nothing from the archives
executed. Row-level output: `ALIGN.json`. Extra manual reads (event lists, casts, commands.txt) recorded below.

| sample | format | manifest steps | rebuilt | verdict |
|---|---|---|---|---|
| SWE-agent / dayjs-1725 (swe_raw .traj) | own count of `trajectory` entries | 35 | 35 | count matches; no manifest action_ref to check text (count-only evidence) |
| mini-SWE / zstd-1532 (swe_raw .traj.json) | own count of assistant messages | 9 | 9 | count matches; no action_ref (count-only) |
| OpenHands / django-16485 (swe_raw LLM-call logs) | one file per LLM call | 23 | 25 | MISMATCH: step unit not yet defined |
| OpenHands / ilp-solver (Terminal) | upstream parser (run + run_ipython) | 15 | 7 | MISMATCH, and three indexings disagree (below) |
| mini-SWE / polyglot-rust-c (Terminal) | upstream parser | 61 | 8 | UNRECOVERABLE: `agent-logs/` empty (no mini.traj.json); agent.log holds only the tmux launch; commands.txt 10 lines |
| Terminus2 / hf-train-lora-adapter (Terminal) | upstream parser (episodes) | 59 | 8 | UNRECOVERABLE: 8 episodes, 8 commands, log ends at training start; flags sit at steps 32-35 |
| mini-SWE / simple-sheets-put (Terminal) | - | 13 | 0 | stub archive (empty directories) |

OpenHands Terminal, three indexings in one run (events 0-34):
- all agent actions (think, edit, run, read, run_ipython, finish) = 15 = manifest `step_count`, and the manifest `stages` span steps 1-15;
- upstream parser (run + run_ipython) = 7 steps;
- the manifest `action_ref`s were written by upstream `scripts/backfill_step_refs.py`, which assigns refs POSITIONALLY over `run` commands only
  (5 here): step 2 -> event 11, step 3 -> event 13, step 5 -> event 23; flagged step 6 has no ref.
- So `action_ref` text cannot validate alignment (it is produced by the same positional assumption), and labels 2/3/5/6 mean different actions
  under the 15-step and the run-only indexing. AMBIGUOUS -> excluded under the 100% rule until a decisive test passes.

Decisive test for K1 (all archives, Kaggle): per format, compare `step_count` with each candidate unit count, and check whether any flagged step id
exceeds the run-only count (proves all-action indexing). A format is admitted only if one unit matches `step_count` on every admitted run and the
labels cannot be read under a second indexing.

Where the flags are (manifest, `full`):

| agent | part | runs | incorrect | unuseful |
|---|---|---|---|---|
| Terminus2 | terminal | 923 | 6,815 | 718 |
| OpenHands | terminal | 768 | 5,447 | 351 |
| mini-SWE-agent | terminal | 833 | 4,254 | 280 |
| OpenHands | no source path | 146 | 279 | 31 |
| mini-SWE-agent | swe | 191 | 439 | 18 |
| SWE-agent | swe | 127 | 235 | 51 |
| OpenHands | swe | 328 | 232 | 19 |

Also found: an OpenHands Terminal archive contains `sessions/.jwt_secret` -> the K1 secret scan is required before anything is stored or shown.
