# Next-action decisions: feasibility report (2026-10-13)

Read-only. Nothing was downloaded, generated, executed in the sandbox, captured or trained; no Kaggle, no CAL / TEST content. Counts come from files
already on disk; "estimate" marks anything not measured. Goal reminder: a decision component (adapter per model + shared head) that helps a smaller
model finish more tasks with fewer unnecessary actions. The label must teach WHEN reading / searching / testing / editing / asking / stopping is the more
useful next action, not only whether an action is possible. A successful trajectory does not make each of its actions good.

## 1. Replay states and split reconciliation (measured)

| Set | Materialised | Replays exactly (both runs) | Canonical TRAIN | Canonical VAL | CAL / TEST (protected, not used) |
|---|---|---|---|---|---|
| Set 1, 2026-10-09 (Go / Python / TS) | 192 | 141 | 88 | 13 | 9 / 31 |
| + Go repair (go.mod recreated, 2026-10-13) | | +39 | +27 | +12 | |
| VAL set, 2026-10-12 (all languages) | 66 | 46 | - | 46 | |
| **Total distinct** | | | **115 states, 74 trajectories** | **58 states, 41 trajectories** | |

Reconciliation of "88 vs 72-of-118": the 118 were the trace file's own "train" rows among the 141; 72 of them are canonical TRAIN (24 TEST, 13 VAL, 9 CAL).
16 rows the trace file calls "val" are canonical TRAIN. 72 + 16 = 88. The canonical split always comes from the imitation set's split map.

## 2. Same-state alternatives with independently verified outcomes (our data, measured)

| Source | States | Split | Alternatives per state | States with different verified outcomes | What kind of decision |
|---|---|---|---|---|---|
| Mutation screen: untouched state vs Fable's action, same test | 44 | TRAIN | 2 | 21 | "act vs do nothing" (trivial: acting wins) |
| Mutation v1.1: edited copies of Fable's action | 19 | TRAIN | 7-8 edits | 15 (36 pass / 83 fail) | which code variant (synthetic; shortcut-prone, 16.26) |
| VAL oracle: untouched vs Fable's action | 39 useful (23 clean fail-to-pass) | VAL | 2 | 39 | "act vs do nothing" |
| GLM pilots: generated fixes | 10 | VAL | 2-5 valid | 2 | which natural fix |
| **Different action TYPES (read vs search vs test vs edit vs ask vs stop) from one state** | **0** | | | **0** | the decision we now care about |
| Saved Qwen3-30B proposals at each replay state, never executed | 192 (set 1; 4 distinct each) | mixed (canonical TRAIN subset usable) | 4 + Fable's action | unmeasured | natural next actions of any type |

Separation asked for:
- **Trivial validity evidence** (edit applies, target file exists, command runs): derivable for every one of the 115 TRAIN states by execution, but it
  teaches what a program can already check. Use: filters and rule baselines, never the training target.
- **Evidence of task progress** (the recorded test changes from fail to pass, or a new failure appears): measured only for code-change actions so far
  (rows above). For read / search / ask / stop, progress is invisible after one step; it needs either a short continuation (rollout) or natural branching
  (section 3).

Conclusion for our own data: **0 genuinely informative next-action-type decisions exist today.** The cheapest way to create some is to execute the 4
saved proposals + Fable's action at each canonical TRAIN replay state (sandbox, no model calls) and measure progress after a fixed short continuation.

## 3. External sources (from DATASET_SOURCES.md, DATASETS_INTERNAL_HEAD.md, results/raw/research-20261012)

| Source | On disk | Licence | Native label | Same-state evidence | Measured here | Overlap / risk | Estimated usable next-action decisions |
|---|---|---|---|---|---|---|---|
| **nebius SWE-rebench-openhands** | **yes** (2.0 GB) | CC-BY-4.0 | `resolved` per run (real tests) | **many runs per task: natural branching** | 67,074 runs, 6,306 tasks, median 10 runs per task (max 34); **1,741 tasks with both resolved and unresolved runs (19,602 runs, 859 repos)** | generator Qwen3-Coder-480B (card); LAYA's pretraining saw nebius/SWE-agent-trajectories (10.7% of instances overlap with the cases data, measured 2026-10-04) -> repo-disjoint tests | unmeasured: depends on how often two runs share an identical prefix and then take different actions (needs trajectory parsing) |
| Fable 5 traces (DSFFGFG456) | yes | CC-BY-4.0 | none (accepted-only) | 51 recorded branch pairs, 11 clearly same-state (research D) | 12,490 rows | imitation only | ~0 outcome-labelled |
| SWE-smith trajectories (tool format) | no (8 GB repo) | MIT | `resolved` | runs per task unknown | 24,100 runs (9,427 resolved) per datasets-server | download needs GO | unmeasured |
| nanoswe | no (14.5 GB) | MIT | `verified`, `exit_status` | unknown | 1,385,818 rows, unit unverified | download needs GO | unmeasured |
| JetBrains SWE-smith subset | no (26 MB) | MIT | `resolved` | small | 1,465 runs | | small |
| nebius SWE-agent-trajectories | no (2.2 GB) | CC-BY-4.0 | `target` | runs per task | 80,036 runs | **LAYA pretraining source** (contamination for any LAYA-vs-random test) | test only on non-overlapping repos |
| RootSE | lead (26 MB) | MIT | human-marked earliest decisive error step in 102 failed runs | n/a | 30 runs read on 2026-10-04 | | evaluation of "this step went wrong", not training |
| AgentProcessBench | lead | not checked | 8,509 human step labels | n/a | not read | SWE share unknown | evaluation lead |
| samatv256 jev-decisions | no (20.7 GB) | CC-BY-4.0 | positive-only completions | | 12 M decision rows | Jev rule: excluded without the user's explicit decision | excluded |

## 4. Rule baselines and shortcut controls (fixed before any model is evaluated)

Rule baselines (deterministic, from the visible history):
- R1 after an edit with no test run since, run the tests;
- R2 before editing a file not yet read in the session, read it;
- R3 never stop while the last test run failed;
- R4 after a failing test, act on the file named in the traceback;
- R5 never repeat an identical command without an intervening change;
- R6 Markov prior: the most frequent next action type given the previous action type (fitted on TRAIN).

Shortcut controls in every report: options-only and no-context scoring; action-type-only baseline; centre / consensus baseline (lesson of 16.26);
same-author candidate pools (one generator per pool); position-in-trajectory baseline; repo- and trajectory-disjoint splits.

## 5. Smallest defensible next step

**Recommended GO (small, CPU only, Kaggle, no model, no training):** a nebius branching count. On Kaggle CPU, parse the 1,741 mixed-outcome tasks and
count decision points where >= 2 runs of the same task share an identical history (same actions and observations) and then take DIFFERENT next actions,
with the downstream resolve rate per action (Monte Carlo over the runs that continue from it). Report: decision points, distinct repos, action types
involved, runs per branch, and how many branches differ in resolve rate by >= 0.3 with >= 3 runs each. Upload: the parquet columns needed (public
CC-BY-4.0 data already on disk). Runtime estimate (unmeasured): 10-30 min CPU.

Gate fixed now: **>= 300 such decision points from >= 100 repos** -> natural branching can supply outcome-informed next-action labels (Phase 2 builds
them, repo-disjoint). Otherwise -> fall back to executing the saved proposals at the 115 TRAIN replay states with a short fixed continuation (sandbox).

**HOLD:** feature capture, any probe or head training, new generation, downloads, CAL / TEST, the second-Qwen adapter.

Caveats: a shared identical history is rarer than a shared task; downstream resolve rate mixes the action's effect with later steps (needs >= 3 runs
per branch and is still noisy); the runs are from one generator (Qwen3-Coder-480B), so labels describe that agent's continuations.

## 6. Result of the branching count (2026-10-14, 00:10 IST; HANDOFF 16.27)

| stage (EXACT history matches; headline) | branchpoints | tasks | repos |
|---|---|---|---|
| mixed-outcome tasks parsed (19,602 runs, parse 100%, tool calls matched 98.5%) | - | 1,741 | 859 |
| identical history, then >= 2 different actions | 9,980 | 1,738 | 859 |
| ... with >= 2 action TYPES | 5,689 | 1,693 | 833 |
| ... and different resolve rates (single tier) | 3,741 | 1,608 | 793 |
| ... >= 2 runs per branch (repeated) | 1,483 | 1,114 | 594 |
| ... >= 3 runs per branch and gap >= 0.3 (strong) | 256 | 249 | 190 |
| strong, excluding the 361 LAYA-overlap repos | 129 | 126 | 107 |
| verified identical workspace | 0 (no snapshots) | | |

The count passes the discovery threshold, but the audit shows the branches are early, interchangeable orientation moves; the decisions that matter
(edit, test after a change, stop) almost never occur at an exactly shared history. Natural exact branching is not a trustworthy label source as is.
