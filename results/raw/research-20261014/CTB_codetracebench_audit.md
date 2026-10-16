# CodeTraceBench (NJU-LINK/CodeTraceBench, MIT) audit for a text-native action grader

Date 2026-10-14. Read-only. No training, no Kaggle, nothing committed. About 28 MB downloaded in total
(manifests 19 MB, repo file list 4 MB, arXiv HTML 0.7 MB, 7 sample archives 0.15 MB).
Scratch: `C:/Users/Satvik/AppData/Local/Temp/ctb_sample/` (extracted archives, 3.6 MB) and
`C:/Users/Satvik/AppData/Local/Temp/ctb_scripts/` (manifests, repo file list, scripts, arXiv text).
No `zstd` binary on PATH, so archives were decoded with `pyarrow.input_stream(..., compression='zstd')` + `tarfile` in the project venv.
Counts below are computed from `bench_manifest.full.parquet` / `.verified.parquet` unless marked otherwise.
Label sets in `incorrect_stages` are Python-repr strings (single quotes), not JSON: parse with `ast.literal_eval`.

## 0. Size of the repo (resolves the 1.09 GB vs 3.52 GB question)

| Part | Files | Bytes |
|---|---:|---:|
| All files (sum of `siblings[].size`, `?blobs=true`) | 20,447 | 3,524,305,096 (3.52 GB) |
| `bench_artifacts/full/*.tar.zst` (the artifacts only) | 3,291 | 451,524,800 (0.45 GB) |
| `swe_raw/` (unpacked copy of the 646 SWE-bench-style trajectories) | 646 dirs | 2,886,546,085 (2.89 GB) |
| manifests (jsonl/parquet) + reports + card | rest | about 186 MB |
| HF API `usedStorage` | | 1,086,734,335 (cause of the gap: not verified) |

The reviewer's 3.52 GB is the true sum of file sizes. The artifacts alone are 0.45 GB. The 646 SWE-style trajectories
exist twice (once in `swe_raw/`, once inside their artifact tar), so the artifacts alone are enough.
Artifacts present for 3,291 of 3,316 `full` rows (25 OpenHands rows have `artifact_path = None`); 8 of the 1,000 `verified` rows lack one.
Stub archives: 11 artifacts are under 1 KB and hold only empty directories (6 Terminus2, 5 mini-SWE-agent), e.g. `simple-sheets-put` (225 bytes, 13 steps, 3 flagged).

## 1. Dataset card and paper: how were labels produced?

Sources: raw `README.md` (read in full) and the paper arXiv 2604.11641 ("CodeTracer", HTML read; sections 2.2, 2.3, limitations, appendix prompts).

- Card: "4,316 agent trajectories with human-verified step-level annotations", labels `incorrect` and `unuseful`. The manifests hold 3,316 rows
  (the card's own table says 3,316), `verified` is a strict subset of `full` (1,000 of 1,000 ids are in `full`). The paper says 4,354 trajectories, 236 tasks, 3.32K full / 1.06K verified.
  The data has 662 tasks and 1,000 verified. So card, paper and data disagree on totals (4,316 vs 4,354 vs 3,316; 236 vs 662 tasks; 1.06K vs 1,000).
  `bench_manifest.parquet` (2,843 rows, an earlier manifest, all ids in `full`) is also in the repo.
- Who labelled (paper 2.3): "All annotations are performed by the authors." Annotators had the task spec, a reference solution and the execution environment.
  They assigned a workflow stage to every step. For successful runs they marked redundant and trial-and-error steps. For failed runs they used a
  backward-tracing protocol that starts from the failing test output and walks upstream to an "error critical step", with an error type
  (setup issue, dependency failure, mislocalized edit, incorrect hypothesis, verification misinterpretation, unproductive looping).
  So the labels are human (authors), made WITH hindsight of the final failure. They are not LLM labels. The paper's LLM prompts in the appendix are the baselines being evaluated, not the ground truth.
- Agreement: 15% random subset double annotated; Cohen's kappa = 0.73, and it is for the **error-critical step label**, not for per-step `incorrect`/`unuseful`.
  No agreement figure for `unuseful`. No separate verification protocol is described for `verified` versus `full`. The pipeline reports show `verified` was selected
  by filtering ("quality-filtered", card). For the 1,000 ids in both splits the labels are identical (checked: 1000/1000 same).
- Definitions (card/paper): `incorrect` = a wrong state-changing action (mislocalized edit, bad hypothesis, regression, irrelevant change, bad dependency/config);
  `unuseful` = redundant exploration (repeated reads/searches without new evidence, not leveraged later).
- How the per-step lists in the manifest relate to the paper's error-critical chain (the paper reports only critical steps and chains): not documented. Not verified.
- Do unflagged steps count as correct? **Not claimed anywhere.** The card only says flagged steps are problematic. The paper's limitations section says the labels "still involve annotator interpretation".
  The `reasoning` field shown in the card example is NOT in the data (keys are only `stage_id, incorrect_step_ids, unuseful_step_ids, steps`).
- The card says 5 models; the data has 6 (extra `DeepSeek/DeepSeekChat`, 102 rows, 90 flagged, none in `verified`).
- Pipeline reports (`bench_manifest_report.json`): of 3,318 candidate terminal tasks 709 were dropped for short step count and 213 for missing stage labels, i.e. some trajectories lack annotation and were removed rather than kept as "all correct".
  Not verified whether any kept trajectory was only partially annotated; see 630 unsolved trajectories with zero flags in section 6.

## 2. Artifact layout (7 archives extracted, 5 agent formats)

| Agent / kind | Archive | Content | Per-step data | step_id mapping |
|---|---|---|---|---|
| SWE-agent (SWE-style) | dayjs-1725 (26 KB) | `*.traj` (JSON: `trajectory` list of {thought, action, observation, response, state}, `history` messages, `info`), `.patch`, `report.json`, `context.json` | yes, structured | 35 manifest steps = 35 list entries (count matches; flagged steps 34, 35 are the final `exit_format`). index = step_id-1 assumed, not verified further |
| mini-SWE-agent (SWE-style) | zstd-1532 (13 KB) | `*.traj.json`: `messages` (system, user task, then assistant/user pairs), `info` (submission, exit_status), `report.json` | yes | 9 manifest steps = 9 assistant messages (matches). Step k = k-th assistant message, observation = next user message |
| OpenHands (SWE-style) | django-16485 (33 KB) | 25 LLM-call logs `*.json` {messages (full history so far), response (tool call), cost}, `report.json`, `patch.diff`, `eval.sh`, `run_instance.log`, `test_output.txt` | yes (history is replayable per call) | 23 manifest steps = 13 execute_bash + 8 task_tracker + 2 str_replace_editor; the `finish` call and one empty-tool call are not steps (consistent in this sample only) |
| OpenHands (Terminal) | ilp-solver (36 KB) | `sessions/.../events/0..34.json`, `event_cache`, `agent_state.pkl`, `agent.log`, `agent.cast`, `tests.log`, `results.json`, `panes/post-test.txt` | yes (event stream) | 15 manifest steps vs 35 events; flagged step 2 carries `action_ref` id 11. Events include system/user/env events, so the map is not 1:1. Not verified |
| Terminus2 (Terminal) | hf-train-lora-adapter (15 KB) | `agent-logs/episode-N/{prompt.txt (terminal output so far), response.txt (JSON analysis/plan/commands), debug.json}`, `commands.txt`, `sessions/agent.cast` | yes | **mismatch**: 8 episodes vs 59 manifest steps (stage 4 flags 9 of steps 31-47). Mapping not recoverable from the archive in this sample |
| mini-SWE-agent (Terminal) | polyglot-rust-c (12 KB) | only `sessions/agent.log` (raw terminal stream), `agent.cast`, `commands.txt`, `results.json`, `panes/post-*`; no trajectory JSON (`agent-logs/` is empty) | no (raw terminal log only) | 61 manifest steps, no step structure in the archive. Not reconstructable without parsing the terminal log |

Pre-action context: reconstructable with confidence for SWE-agent, mini-SWE-agent and OpenHands SWE-style archives (646 trajectories, all with a `swe_raw` path)
because each holds a message history. Terminus2 (923) and the Terminal OpenHands/mini-SWE trajectories (2,024) need per-agent parsing, and for Terminus2 and mini-SWE Terminal the step index
did not line up in the samples checked. Outcome of the sample audit: "usable text for every step" is verified for 3 of 5 formats, not for the other 2.

Manifest-embedded text: `action_ref` / `observation_ref` (raw event JSON) exist only for flagged steps and only in part:
with an action 3,497 of 4,991 flagged mini-SWE steps, 3,190 of 6,359 OpenHands, 4,001 of 7,533 Terminus2, **0 of 286 SWE-agent**; 10,688 of 19,169 overall (56%).
Median `action_ref` is 1,781 chars (p90 7,265, max 102,539); the OpenHands blobs include the whole model response, so they need extraction before use. The manifest never has unflagged steps.

Information from AFTER the step (keep out of the model input):
- `report.json`, `patch.diff`, `eval.sh`, `run_instance.log`, `test_output.txt`, `results.json` (is_resolved), `panes/post-agent.txt`, `panes/post-test.txt`, `tests.log`, `tests.cast`.
- `.traj` `info` (submission, exit_status) and SWE-agent `context.json` (viewed_files for the whole run; the last response lists PATCH_CONTEXT).
- The full `agent.log` / `agent.cast` / event stream: must be cut at the step.
- Later LLM-call logs and later messages of the same run.
- Manifest fields `solved`, `stages` (annotator-assigned phase boundaries), `incorrect_error_stage_count`, `stage_count`, `step_count`, plus the label lists themselves.
- The label was made with the failing test output in view, so label = hindsight judgement; the model input must contain only the past.

## 3. `source_relpath` / `annotation_relpath`

- `annotation_relpath` (`agent_failure_analysis/step_annotations_all/...`): **0 files** in the repo under that prefix. The raw annotations are not published; the manifest `incorrect_stages` is the only label source.
- `source_relpath`: for the 646 SWE-style rows it is `swe_raw/<group>/<instance>` and all 646 directories exist in the repo (and inside the artifact). For Terminal rows it is a directory path inside the artifact tar (not a repo file). 121 OpenHands Terminal rows have `source_relpath = None` but do have an artifact.

## 4. Label granularity

Flags are per step, but dense inside flagged stages.

| | full | verified |
|---|---:|---:|
| trajectories / flagged trajectories | 3,316 / 1,751 | 1,000 / 405 |
| steps (sum of step_count) / median per trajectory | 147,628 / 30 | 46,539 |
| incorrect / unuseful flagged steps (no step has both) | 17,701 / 1,468 | 1,695 / 224 |
| flagged stages | 4,039 of 22,246 | not computed |
| stages where every step is flagged | 1,331 (33% of flagged stages) | not computed |
| flagged share of a flagged stage (median / mean) | 0.75 / 0.75 | not computed |
| flagged steps per flagged trajectory: min, p25, median, p75, p90, max (mean) | 1, 2, 5, 10, 19, 565 (10.95) | 1, 2, 4, 6, 9, 48 (4.74) |
| flagged share of steps in a flagged trajectory (median / mean) | 17.2% / 23.4% | 8.3% / 11.0% |
| first flagged step: median (p25, p75) | 11 (6, 23) | 22 (14, 33) |
| first flagged step as fraction of trajectory: median (p25, p75) | 0.42 (0.20, 0.69) | 0.65 (0.40, 0.86) |
| first flagged step is step 1 / is within steps 1-3 | 67 / 207 | 0 / 3 |

Flag sets inside a stage are sub-sets of that stage range (checked for all 4,039). `verified` is a lower-density, later-onset subset of `full` (4.1% of its steps are flagged against 13.0%).

## 5. Task overlap

Task counts: 662 tasks in `full` = 380 `owner__repo-NNNN` names + 46 SWE-bench-Pro `instance_...` names + 236 Terminal-Bench names (the paper's 236 tasks is the Terminal count); the `owner__repo-NNNN` regex alone leaves 282 non-matching names because the 46 Pro names are in that bucket. `verified`: 325 `owner__repo-NNNN` names (this regex count; Pro and Terminal not split out).

| Source (by `source_relpath`) | Trajectories | Flagged | Steps | incorrect | unuseful |
|---|---:|---:|---:|---:|---:|
| Terminal-style (Terminus2 923, OpenHands 949, mini-SWE 872, SWE-agent 30) | 2,670 | 1,440 | 127,946 | 16,795 | 1,380 |
| SWE `verified` (OpenHands 212, mini-SWE 14) | 226 | 55 | 7,301 | 148 | 13 |
| SWE `poly` | 208 | 116 | 6,211 | 358 | 48 |
| SWE `multi` | 108 | 71 | 2,360 | 176 | 5 |
| SWE `pro` (`instance_<owner>__<repo>-<hash>` names) | 104 | 69 | 3,810 | 224 | 22 |
| SWE-style total, by agent: mini-SWE 191, OpenHands 328, SWE-agent 127 | 646 | 311 | 19,682 | 906 | 88 |

- Names of the form `owner__repo-NNNN`: 542 rows, 380 distinct instance names. The 104 Pro rows have hash names (`instance_<owner>__<repo>-<sha>`) and are counted separately (46 distinct Pro tasks); 542 + 104 = 646 SWE-style rows.
- Repos: 42 distinct among the `owner__repo-NNNN` names (django 92 tasks, sympy 35, huggingface/transformers 28, mui/material-ui 21, matplotlib 19, sphinx 19, scikit-learn 14, ponylang/ponyc 13, clap-rs/clap 12, keras 11, svelte 10, astropy 10, xarray 9, pytest 9, prettier 8, facebook/zstd 7, vscode 7, vuejs/core 6, and 24 more with 1-4 each) plus 7 SWE-bench-Pro repos (ansible 65 rows, openlibrary 15, qutebrowser 13, element-web 6, flipt 2, vuls 2, NodeBB 1).
- Overlap with the LAYA checkpoint training data: `data/coding-decision-cases/data/*.parquet` has 102,751 rows; `nebius/SWE-agent-trajectories` rows = 12,399 (train 9,644, val 422, test 2,333), 858 distinct repo groups.
  **0 of the 42 CodeTraceBench repos appear in those groups; 0 of 7 Pro repos; 0 of CodeTraceBench task names or instance ids (1,843 instance ids in the LAYA meta) match.** The LAYA `group` field of any source also never equals a CodeTraceBench repo or task name. Case-insensitive check also empty.
  (Overlap of the Terminal tasks with `AgentJudgeBench` groups was checked by task-name equality only: 0.) Underlying benchmark overlap (SWE-bench Verified problems being in the base ModernBERT/LAYA pretraining of other sources) is not verified.

## 6. Usable examples for (pre-action context, proposed action) -> {incorrect, unuseful, unflagged}

Step counts, `full`, split by manifest `solved` (106 rows have `solved = None`; `solved = True` can also mean the card's `solved`, semantics unverified: the dayjs sample is `solved=True` while its `report.json` says `valid: false`):

| Group | incorrect | unuseful | unflagged steps |
|---|---:|---:|---:|
| flagged trajectories, solved=True | 2,009 | 155 | 11,541 |
| flagged trajectories, solved=False | 15,297 | 1,213 | 49,160 |
| flagged trajectories, solved=None | 395 | 100 | 3,727 |
| unflagged trajectories, solved=True (863 traj) | 0 | 0 | 26,220 |
| unflagged trajectories, solved=False (630 traj) | 0 | 0 | 30,645 |
| unflagged trajectories, solved=None (72 traj) | 0 | 0 | 7,166 |
| **Total** | **17,701** | **1,468** | **128,459** |

`verified`: 1,695 incorrect + 224 unuseful = 1,919 flagged; unflagged = 17,481 in flagged trajectories + 27,139 in unflagged ones (595 trajectories) = 44,620.

What is explicit and what is not:
- **Explicit negatives**: the 17,701 `incorrect` (strongest; defined as wrong state-changing actions, from hindsight) and 1,468 `unuseful` (redundant, a weaker kind of "bad"). 92% of the incorrect steps are in unsolved runs.
- **Unflagged is NOT verified good.** Nothing in the card or paper states that unflagged = correct. The most trustworthy "likely fine" steps are the 26,220 in solved trajectories with no flags (annotators there looked for redundancy and found none). The 64,428 unflagged steps inside flagged trajectories are weaker: neighbours of a flagged step, and in a flagged stage a median 25% of steps are unflagged. Unflagged steps in failed trajectories with no flags at all (30,645) are ambiguous (the annotator may have found no error-critical step, or the run lacks annotation); do not use them as positives.
- Reconstruction feasibility cuts the usable pool: the safe-to-parse subset is the 646 SWE-style trajectories (19,682 steps; 906 incorrect, 88 unuseful, plus 18,688 unflagged). Add the 923 Terminus2 and 2,024 other Terminal trajectories only after a per-agent parser passes a step-count check (it failed in the one Terminus2 and one mini-SWE Terminal sample). The Terminal part holds 94% of the flags (16,795 incorrect, 1,380 unuseful).
- Input size: the grader's 2,048-token window is far smaller than the typical step (median `action_ref` 1,781 chars plus median `observation_ref` 944 chars). Context must be a truncated recent window, which will lose the error-chain evidence on which the annotators based hindsight labels.
- Label-noise risks: hindsight labels (a step is "incorrect" partly because the run later failed), authors as sole annotators, kappa 0.73 only for critical steps, dense contiguous flagging (stage median 75%), and flags that cluster in 4,039 stages (one third of those fully flagged), so the label probably correlates with position, stage and `solved` (not tested).
  A baseline that predicts from the position in the trajectory, the step index or the agent/model should be run before any claim.

## 7. Recommended split (not applied; illustrative sizes)

Rules:
1. Group by task, never by trajectory: every (agent, model) run of the same task must land in one part (15 runs per task share the task text and solution).
2. For SWE-style tasks group by repo (`owner__repo`, including the Pro repos) because django alone is 92 tasks and shares code; for Terminal tasks group by `task_name`.
3. Do not use `verified` as the independent set: it is a subset of `full` with identical labels, built by filtering. Use it only as a report slice (it makes clean tasks, lower flag density).
4. Hold out one agent format and one model for a secondary check (for example SWE-agent, 127 trajectories, 70 flagged; or Claude-Sonnet-4, 531 trajectories) rather than for tuning.
5. Keep the held-out part away from LAYA. Since 0 repos overlap with the LAYA training sources, no further deduplication is required for the checkpoint; the held-out part must still not be inspected before the pipeline is frozen.

Illustrative sizes using sha1 hashing of the group key (70 / 10 / 20 percent of groups, salt `ctb-split-v2`; hash splits of few groups are lumpy, so fix the final assignment by hand and stratify by source):

| Part | Groups | Trajectories | Steps | incorrect | unuseful | Flagged trajectories | SWE-style trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 198 | 2,353 | 106,591 | 13,496 | 1,187 | 1,270 | 484 |
| DEV (model selection) | 27 | 267 | 12,703 | 1,120 | 75 | 131 | 39 |
| TEST (once) | 60 | 696 | 28,334 | 3,085 | 206 | 350 | 123 |

A plain task-level hash (not repo-grouped) shares 24 repos between TRAIN and TEST; use the repo-grouped one. The numbers above are the totals before dropping trajectories that cannot be parsed.
Because only the 646 SWE-style archives are verified parseable, a first usable version is "SWE-style only": about 906 incorrect + 88 unuseful flagged steps and 18,688 unflagged steps overall (about 70/10/20 gives roughly 650 / 90 / 180 flagged steps, estimate (unmeasured)).

## 8. Verdict

- Possible: yes for a NEW source of step-level negatives (17,701 incorrect, 1,468 unuseful, human-annotated by the authors), disjoint by repo and task from the LAYA nebius training set, MIT licence, artifacts 0.45 GB.
- Not as a clean 3-class set: "unflagged" is not a verified-good label, the labels are hindsight (use the past-only input), labels are heavily tied to failed runs, and only about 5.2% of flagged steps (994 of 19,169, the SWE-style rows) are in formats confirmed parseable.
- Cheapest next check before any download of the full set (about 0.45 GB for the artifacts, or 3 GB with `swe_raw`): write one parser per agent and verify step counts against `step_count` on the 646 SWE-style archives (and a sample of Terminal ones), then run the position/agent/`solved` shortcut baselines on the manifest alone.
- Not verified in this audit: the relation of the step lists to the paper's error-critical chains; the semantics of `solved`; whether any kept trajectory is only partly annotated; why `usedStorage` is 1.09 GB; step-id mapping for Terminus2, Terminal OpenHands and Terminal mini-SWE beyond the one sample each.
