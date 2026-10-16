# RootSE and AgentProcessBench audit (read-only, 2026-10-14)

Scratch copy: `C:/Users/Satvik/AppData/Local/Temp/rootse_apb/` (rootse 28 MB, apb 39 MB, paper pdf 3 MB; about 70 MB total). Scripts (rs_stats.py, apb_stats.py, rs_len.py, tok.py, ov2.py, ov3.py, ov4.py) are in the same folder.
Nothing was trained, committed, pushed or sent to Kaggle. Tokens were counted with the tokenizer in `models/coding-decision-agent/tokenizer`. Budget assumption: 2,048 minus 320 = 1,728 tokens for the state/context (taken from the task text, not re-verified in the checkpoint config).

## 1. Schema and label meaning

| | A. RootSE (dengdan1999/RootSE) | B. AgentProcessBench (LulaCola/AgentProcessBench) |
|---|---|---|
| Licence / size | MIT, 102 JSON files (28 MB), `viewer: false` | MIT, 4 jsonl files (bfcl 15.0, gaia_dev 12.0, hotpotqa 3.2, tau2 10.0 MB), split `test` only. The HF card is two lines; label meaning comes from the paper (arXiv 2603.14465, KDD 2026) and the GitHub README |
| Unit | one failed coding run (a full agent trajectory) | one trajectory of a tool-using agent (not coding) |
| Fields (measured) | `repo, instance_id, base_commit, repo_language, original_mission_prompt, gold_patch, test_patch, failed_tests{test: {error_description, code}}, original_traj[list of steps], failure_id, failure_reason, agent, model` | `total_index, query_index, sample_index, question, data_source, ground_truth, answer_text, messages, tool_metrics, tools, step_labels, final_label` (+ `task_description` in tau2 and some others; null elsewhere) |
| Step fields | OpenHands: `index, thought, action (dict or ''), response, observation`. SWE-agent: adds `execution_time, state, extra_info`, `response` always ''. ACR: `action` is a list of `{func_name, arguments, call_ok}`. `index` is a string, base 0, equal to list position in all 102 files | `messages` are OpenAI-style (system, user, assistant with `tool_calls`, tool). Only assistant messages are labelled |
| Label | `failure_id` (stored as a string): index of the **earliest decisive error step**, one per run; `failure_reason`: free-text justification. README says both are annotated by "human experts". Number of annotators and agreement: **not given in the README, no paper link: not verified** | `step_labels`: dict, key = **index in `messages`** (all keys verified to be assistant messages, 0 bad keys), value +1 correct and advancing, 0 neutral/exploratory (reasonable but little or unclear effect, e.g. a 404, a redundant restatement), -1 incorrect or harmful (misread tool output, fabricated evidence, policy violation, repeating a failed action). `final_label` in {-1, 0, 1} is a trajectory outcome; its meaning is **not documented**. Counts are -1: 516, 0: 24, +1: 460, while the paper's table gives 528 unsuccessful and 472 successful, so -1/+1 look like fail/success and 0 is unverified |
| Labellers | human, per the README | Paper: 2 human experts (CS degree, >=1 year of LLM experience) per trajectory, discrepancies resolved by discussion. **Step IAA 89.1%, Cohen kappa 0.767.** Annotators got official solutions and reference labels from 3 LLMs (DeepSeek-V3.2, GPT-5.2, Claude 4.5 Sonnet) as guidance. A 100-trajectory re-annotation without LLM references agreed 84.06% with the original labels |
| Special rule | none stated | **Error propagation:** after a -1 step, every later dependent step is also -1 until the agent explicitly corrects the error or moves to an independent subtask. Annotators saw the whole trajectory ("avoid hindsight" is instructed, but an incorrect statement is -1 when it is "relied upon later", so labels depend on later steps) |
| Generators | see section 2 | five models: Qwen3-4B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507, DeepSeek-V3.2, GPT-5-mini, GPT-5 |

## 2. Measured counts

### A. RootSE

| Agent / model | Runs | Steps | Mean steps/run | Steps before decisive step | Repos |
|---|---|---|---|---|---|
| openhands / qwen3-coder-480B-A35B | 30 | 1,807 | 60.2 | 599 | 22 |
| swe-agent / claude45 | 30 | 2,078 | 69.3 | 800 | 7 |
| swe-agent / gemini2 | 12 | 333 | 27.8 | 68 | 6 |
| swe-agent / gpt5 | 11 | 477 | 43.4 | 245 | 5 |
| auto-code-rover / gpt-4-0125 | 10 | 42 | 4.2 | 15 | 3 |
| swe-agent-live / claude-sonnet-4-5 | 5 | 402 | 80.4 | 94 | 3 |
| swe-agent-live / gpt5 | 2 | 51 | 25.5 | 42 | 2 |
| swe-agent-live / gpt5mini | 2 | 78 | 39.0 | 52 | 2 |
| **Total** | **102** | **5,268** | 51.6 | **1,915** | **35 repos, 98 unique instance ids** |

- Languages: python 73, go 24, js 5 runs (matches the README 71.6 / 23.5 / 4.9%). 4 instance ids appear twice under different agents (one openlibrary, two navidrome, one qutebrowser).
- Decisive step: `failure_id` is a 0-based position, always `< len(traj)`; it is the last step in 1 run and step 0 in 3 runs; median position as a fraction of the run is 0.386. There is **one** labelled step per run and no label on any other step.
- Step buckets: 102 decisive, 1,915 earlier, 3,251 later (later steps carry no label of any kind).
- Action form at the decisive step: 18 of 30 OpenHands decisive steps are **no-tool-call steps** (thought/message only), against 34 of 599 earlier OpenHands steps; every other agent has an action at its decisive step. This is a form shortcut. It agrees with `results/raw/rootse-signal-check-20261004.json` (30 runs, 1,807 steps, 599 before, no_tool_call 0.6 at the decisive step vs 0.057 before), which I reproduced.
- All runs failed; there are no passing runs.

### B. AgentProcessBench

| Config | Trajectories | Assistant steps (all labelled) | +1 | 0 | -1 | final_label -1 / 0 / 1 | First-step label |
|---|---|---|---|---|---|---|---|
| bfcl (multi-turn base) | 250 | 2,590 | 1,916 | 104 | 570 | 152 / 7 / 91 | +1 200, -1 45, 0 5 |
| gaia_dev | 250 | 1,628 | 612 | 165 | 851 | 163 / 4 / 83 | +1 184, -1 59, 0 7 |
| hotpotqa (searchR1) | 250 | 734 | 499 | 56 | 179 | 87 / 0 / 163 | +1 220, -1 30 |
| tau2 (airline 85, retail 85, telecom 80) | 250 | 3,557 | 2,320 | 127 | 1,110 | 114 / 13 / 123 | +1 250 |
| **Total** | **1,000** | **8,509** | **5,347 (62.8%)** | **452 (5.3%)** | **2,710 (31.8%)** | 516 / 24 / 460 | |

8,509 equals the paper's "8,509 human-labeled step annotations". Every assistant message has a label (0 unlabelled). Propagation makes -1 runs long (the first BFCL example has -1 on 22 consecutive steps).

## 3. Rebuilding the pre-action context, and leakage fields

| | RootSE | AgentProcessBench |
|---|---|---|
| Can it be rebuilt? | Yes. Context for step k = `original_mission_prompt` + for each step j<k: `thought`, `response`, `action`, `observation`. Candidate action = step k `thought/response/action`. The OpenHands mission is the full OpenHands system prompt plus the issue | Yes. Context for labelled step i = `messages[:i]` (+ `tools` definitions, static per trajectory). Action = `messages[i]` content and `tool_calls` |
| Must be excluded from model input | `failure_id`, `failure_reason`; `gold_patch`, `test_patch`, `failed_tests` (describe the correct fix and the final failing tests); **`observation` of step k itself** (the result of the action being judged, stored in the same record); `execution_time`, `state`, `extra_info` of step k; all steps > k; and identity fields `agent`, `model`, file path (shortcuts) | `messages[i+1:]`; `step_labels`; `final_label`; `answer_text` (final answer); `ground_truth` (gold answer, actions, assertions); **`tool_metrics`** (aggregates over the whole trajectory, including later calls); `task_description` (tau2 purpose/notes) only if treated as part of the task statement. BFCL puts `system` state_info messages in the middle and at the end of `messages`, so cut by index, never by role |
| Leakage inside labels | The label is a single point. Steps before it are **not** verified good (non-decisive errors may exist) | Labels use hindsight and propagation: a step can be -1 because it follows an earlier error or is "relied upon later", so the label is not a function of the pre-action context alone |

## 4. Overlap with the checkpoint training sources and project data

Method: `data/coding-decision-cases/data/*.parquet` (train 83,656, val 4,751, test 14,344 rows) filtered to `source` containing `nebius/SWE-agent`: 12,399 rows, 858 groups (train 9,644, val 422, test 2,333), 1,843 unique `meta.instance_id`. `group` is `owner__repo`. Rebench: `instance_id` and `repo` columns only (67,074 trajectories, 6,306 unique instances, 1,823 repos).

| Check | Result |
|---|---|
| RootSE repos (35) vs nebius/SWE-agent groups (case-insensitive) | **6 repos overlap**: AzureAD/microsoft-authentication-library-for-python, Bachmann1234/diff_cover, CS-SI/eodag, Erotemic/ubelt, ESSS/conda-devenv, Fatal1ty/mashumaro. All six are OpenHands repos (of 22). No SWE-agent, SWE-agent-live or ACR repo (django, sympy, astropy, matplotlib, pytest, sphinx, ansible, vuls, teleport, openlibrary, navidrome, NodeBB, qutebrowser) appears in any case group |
| RootSE instances vs `meta.instance_id` of nebius/SWE-agent cases | **2 instances overlap**: `ESSS__conda-devenv-51` (16 case rows) and `Erotemic__ubelt-64` (14 case rows), 30 rows, all in the **train** split. Both are RootSE OpenHands runs. The other overlapping repos have different issues (mashumaro-116/130 vs case 146; eodag-634 vs 586; diff_cover-413 vs 277/325; msal-530 vs 312). Whether the released checkpoint was trained on those rows is not verified (only the model-card source list is known) |
| RootSE OpenHands vs `data/nebius_swe_rebench` | **30 of 30 instance ids are in rebench** (22 of 22 repos), about 11.9 trajectories per instance (358 total: 112 resolved, 246 not). The 72 non-OpenHands instance ids: 0 in rebench. Rebench is not a LAYA source but is in this project, so any training of ours on rebench contains the source tasks (and probably the source runs) of the 30 RootSE OpenHands items. **Trajectory-level identity was not checked** (one 4,096-row group is about 1 GB uncompressed); the README only states the runs derive from the Nebius release |
| AgentProcessBench vs LAYA sources | The case sources are CodeUltraFeedback 30,000; When2Call 15,000 + 9,000 + 3,652; nebius/SWE-agent 12,399; several AgentJudgeBench subsets (5.4K to 6.6K each); routerbench 2,395; typed-decisions. No BFCL, GAIA, HotpotQA or tau2 source by name. Item-level overlap with AgentJudgeBench / When2Call **not checked** |

## 5. Suitability

| Use | RootSE | AgentProcessBench |
|---|---|---|
| Independent evaluation of a coding action-quality grader | Best domain fit (real coding agents, human-marked decisive error) but small. Items: **102 decisive (bad) steps**. Earlier steps (1,915) are "not decisive", not "good", so use them as weak controls only, or run a paired within-run test: score(decisive) vs scores of earlier steps of the same run (102 pairs; rank of the decisive step among the first N steps). The 30 OpenHands runs are weaker for unbiased claims (rebench overlap, 2 in LAYA train). The 72 non-OpenHands runs have no overlap found | Not coding (BFCL file-system calls, web QA, customer-service dialogue). 8,509 labelled steps. Useful as an out-of-domain check of a generic step grader and as a tool-call-verdict calibration source, not as evidence of coding quality |
| Calibration | Too small (102 positives, 8 agent/model cells) | Large, but labels are correlated within a trajectory (group by trajectory, 1,000 groups; 200 tasks, each task has trajectories from all five models) |
| Training | Not recommended: one label per run, and it would remove the only human-marked coding test | Possible for a generic step head with a coarse target (+1 vs -1, drop 0 = 5.3%); propagated -1 labels are not causal errors. Split by task, not by trajectory |
| Shortcut risks | OpenHands no-action form (18/30 vs 34/599), step position (median 0.386), agent/model identity, run length. Report these baselines with any head result | Position (first step is +1 in 854 of 1,000), per-config base rates (hotpotqa has no 0; tau2 first step always +1), source model, final outcome |
| Items | decisive 102; earlier 1,915; later 3,251 (unlabelled, unusable) | per-step 8,509 (+1 5,347 / 0 452 / -1 2,710). Paper's FirstErrAcc target = first -1 step per trajectory (count of trajectories with any -1 not computed here) |

## 6. Rendered length (chars measured; tokens measured with the checkpoint tokenizer)

Chars per token on a code-heavy RootSE sample: 2.66 (20 steps only). Budget compared: 1,728 tokens. RootSE rendering = mission + all previous thought/response/action/observation (observation clipped = 300 chars each).

### A. RootSE

| Quantity | Mean | Median | p95 | Max | n |
|---|---|---|---|---|---|
| Single action chars, all steps | 832 | 208 | 3,898 | 101,554 | 5,268 |
| Single action chars, decisive step | 2,809 | 2,149 | 6,665 | 14,479 | 102 |
| Single action tokens, decisive step | 791 | 644 | 2,125 | 3,901 | 102 (93.1% fit in 1,728) |
| Pre-action context chars, raw, all steps | 126,541 | 114,066 | 273,960 | 704,971 | 5,268 |
| Same, steps before decisive | 65,258 | 55,974 | 150,499 | 337,311 | 1,915 |
| Same, at the decisive step | 88,157 | 83,827 | 180,152 | 339,482 | 102 |
| Context chars, observations clipped to 300 | 37,598 | 31,266 | 86,946 | 134,025 | 5,268 |
| Mission alone, chars | 8,187 | 5,455 | 16,702 | 45,934 | 102 |
| Mission alone, tokens | 1,992 | 1,368 | 3,970 | 11,363 | 102 (56.9% fit) |

Mission tokens per agent (mean): openhands 4,036 (system prompt included), swe-agent-live 2,131, swe-agent 1,121, auto-code-rover 349.
Conclusion: the full pre-action context is 20 to 70 times the budget even with observations clipped. Using RootSE needs a deterministic compressor (task summary + last 2 to 4 steps + file/test state), which changes what the grader sees and should be pre-registered. The single decisive action usually fits (93%), but at OpenHands decisive steps the candidate is mostly a long thought.

### B. AgentProcessBench (messages before the step, without tool definitions)

| Config | Context tokens mean / median / p95 / max | Steps with context <= 1,728 | Action tokens mean / p95 | Tool-definition tokens (extra) mean |
|---|---|---|---|---|
| bfcl | 7,448 / 7,483 / 10,158 / 12,246 | 0.0% | 73 / 169 | 5,756 |
| gaia_dev | 8,200 / 3,891 / 26,886 / 1,772,281 | 30.0% | 161 / 354 | 586 |
| hotpotqa | 3,323 / 1,290 / 13,624 / 51,660 | 61.6% | 126 / 306 | 98 |
| tau2 | 5,799 / 5,295 / 14,186 / 27,648 | 11.1% | 121 / 346 | 2,633 |

Chars (mean / p95): bfcl context 29,026 / 38,038, action 236 / 627; gaia context 27,224 / 90,992 (max 4.25M), action 516 / 1,309; hotpotqa context 13,300 / 53,782, action 439 / 1,099; tau2 context 21,054 / 47,295, action 408 / 1,277. BFCL and tau2 contexts are dominated by system prompt, state info and policy text. Actions are short (about 75 to 160 tokens on average), so the action fits; the context needs truncation for most steps.

## Not verified

- RootSE annotator count and agreement (no paper or citation in the README).
- Meaning of APB `final_label` = 0 and the exact fail/success mapping.
- Whether the released checkpoint trained on `ESSS__conda-devenv-51` and `Erotemic__ubelt-64` (they are in the cases train split).
- Trajectory-level equality between RootSE OpenHands steps and the rebench rows (trajectory column not loaded, RAM).
- Item-level overlap of APB with AgentJudgeBench / When2Call.
- The 1,728-token budget (from the task text, not read from the checkpoint config).
