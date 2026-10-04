# Dataset and model sources for the outcome-labelled head

Checked 2026-10-03 against the Hugging Face API (`huggingface.co/api/datasets/<id>`) and web search. Nothing here is downloaded.
"Storage" is the HF `usedStorage` figure for the whole repo (all files and revisions), so it can overcount what one download costs. Download size of the needed split: estimate (unmeasured).
Every download over 1 GB needs the user's written yes (CLAUDE.md, "Ask first").

## A. Named in the build plan

| Repo | Use | License | Storage | Status |
|---|---|---|---|---|
| `nebius/SWE-rebench-openhands-trajectories` | solved / unsolved labels, 67,074 runs (plan figure) | CC BY 4.0 | 2.08 GB | NEEDS YES |
| `greghavens/gpt-5.6-sol-coding-and-debugging-traces` | frontier `next_move` and `pick` | CC BY 4.0 | 115.5 GB | NEEDS YES; whole repo is huge, only a slice should be pulled |
| `greghavens/fable-5-coding-and-debugging-traces` | frontier `next_move` and `pick` | not checked | not checked | NOT FOUND under this exact name (API: "Invalid username or password"). Nearest real repo by the same author: `greghavens/fable-5-coding-and-debugging-traces-synthetic-corrections`. Confirm the name with the user; "synthetic corrections" may not be real traces. |
| `samatv256/jev-decisions-v1` | state, candidates, outcomes (Open-SWE rows only) | CC BY 4.0 | 22.2 GB | positive-only completions; NEEDS YES |
| `mvilacad/laya-conductor` | sidecar start weights | Apache-2.0 | 1.57 GB | already on disk (`models/laya-conductor`) |

## B. Found by search, not in the plan (all unverified for fit)

| Repo | Why it could help | License | Storage | Downloads | Last modified |
|---|---|---|---|---|---|
| `nanoswe/nanoswe-trajs-261002` | SWE-smith tool-call runs with a `verified` (resolved) column | MIT | 15.5 GB | 69 | 2026-10-02 |
| `SWE-Lego/SWE-Lego-Synthetic-Data` | 11.5k runs, 9.1k resolved and 2.4k unresolved (per search snippet) | not checked (API call failed) | not checked | n/a | n/a |
| `nvidia/Open-SWE-Traces` | 200k+ runs (SWE-agent, OpenHands) | CC BY 4.0 | 379.7 GB | 26,381 | 2026-09-22 |
| `SWE-bench/SWE-smith-trajectories` | SWE-smith agent runs | MIT | 8.78 GB | 10,708 | 2025-07-19 |
| `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` | 66k mini-swe-agent runs | MIT | 2.49 GB | 2,620 | 2025-11-06 |
| `JetBrains-Research/agent-trajectories-swesmith-random-subset` | small sample with a `resolved` column from SWE-bench evaluation; good for a first test | MIT | 25.7 MB | 63 | 2026-03-28 |
| `nebius/SWE-agent-trajectories` | 80,036 SWE-agent runs | CC BY 4.0 | 2.31 GB | 7,310 | 2024-12-23 |
| `SWE-Gym/OpenHands-Sampled-Trajectories` | OpenHands runs on SWE-Gym | not stated | 301 MB | 1,328 | 2024-12-23 |

## C. Task and environment sources (for the seeded-bug agent test, or scale-up)

| Repo | Use | License | Storage |
|---|---|---|---|
| `SWE-Gym/SWE-Gym` | tasks with tests | MIT | 43.6 MB |
| `R2E-Gym/R2E-Gym-Subset` | executable tasks | Apache-2.0 | 944 MB |
| `R2E-Gym/R2E-Gym-V1` | executable tasks, larger | Apache-2.0 | 2.91 GB |
| `nebius/SWE-rebench` | tasks behind the nebius runs | CC BY 4.0 | 938 MB |
| `SWE-bench/SWE-smith` | synthetic bug tasks | MIT | 4.03 GB |
| `princeton-nlp/SWE-bench_Verified` | held-out benchmark | not stated | 8.4 MB |

## D. Other greghavens trace sets found (not in the plan)

`greghavens/glm-5.2-coding-and-debugging-traces`, `greghavens/kimi-k3-coding-and-debugging-traces`, `greghavens/vcf-9x-version-tagged-sft`. Not checked. Add only if the user asks.

## E. Rejected by the plan

`yixuanli97/finding-the-right-fit` (non-commercial), `MoreThought/*` Fable 5.1 dumps (unverifiable origin), Open-Jev and JevK5 (need backbone fine-tuning).

## F. Already on disk (no download)

- `data/decisions/action-swerebench-sample-v2.jsonl`: 300 trajectories, 18,894 steps, 140 resolved trajectories, 230 repos.
- TRAIN and CAL Qwen states under `data/v3/capture*` and text embeddings under `data/v3/textemb*`.
- Models: `models/qwen3-4b-thinking-2507`, `models/laya-conductor`, `models/laya-typed-decisions`, `models/laya-stop-completion-judge`.

## Open checks before any download
1. Resolve the Fable 5 repo name with the user.
2. Confirm each set has a per-run resolved flag (only the nebius, nanoswe, SWE-Lego and JetBrains sets are known to).
3. License per row must be recorded (RB2) before a source enters train or cal.

## G. Row counts (HF datasets-server `/size`, 2026-10-03)
Unit differs per repo (run, step or task); "run" = one trajectory. Where unit is unverified it is marked.
| Repo | Rows | Unit |
|---|---|---|
| nebius/SWE-rebench-openhands-trajectories | 67,074 | run (plan figure matches) |
| nvidia/Open-SWE-Traces | 265,158 (openhands 80,630; sweagent 77,261; minisweagent 107,267) | run, unverified; the API total of 511,668 counts other configs |
| nebius/SWE-agent-trajectories | 80,036 | run |
| SWE-bench/SWE-smith-trajectories | 76,002 (tool 24,100; xml 26,076; ticks 25,826) | run, same runs in 3 formats, unverified |
| Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k | 65,994 | run |
| nanoswe/nanoswe-trajs-261002 | 1,385,818 | unverified; far above a plausible run count, probably steps or samples |
| samatv256/jev-decisions-v1 | 12,028,080 (train 50,000; validation 354,037; test 354,635) | decision (step); the three splits do not sum to the total |
| SWE-Gym/OpenHands-Sampled-Trajectories | 6,055 | run |
| JetBrains-Research/agent-trajectories-swesmith-random-subset | 1,465 | run |
| greghavens/glm-5.2-coding-and-debugging-traces | 1,821 | unverified |
| greghavens/kimi-k3-coding-and-debugging-traces | 3,956 | unverified |
| greghavens/fable-5-...-synthetic-corrections | 2 | unverified |
| greghavens/gpt-5.6-sol-coding-and-debugging-traces | 0 (server could not parse it) | unknown |
| SWE-Lego/SWE-Lego-Synthetic-Data | not counted (renamed); search snippet says 11.5k | run |
| Task sets: SWE-smith 59,136; nebius/SWE-rebench 27,878; SWE-Gym 2,438; R2E-Gym-V1 8,101; R2E-Gym-Subset 4,578 | | task |

## H. Corrections after reading the dataset cards (2026-10-04)
- `greghavens/fable-5-coding-and-debugging-traces` is NOT a wrong name: the card of its companion (`...-synthetic-corrections`, 1 trajectory, 2 rows, 16 kB) links it as "the original dataset". The unauthenticated API call for it failed ("Invalid username or password"), so it is probably gated or private; its size, columns and licence are unread. The companion holds only narrowly corrected traces that never passed in the original.
- `greghavens/gpt-5.6-sol-coding-and-debugging-traces`: card says 17,939 cumulative next-step rows from 1,474 ACCEPTED trajectories (4 seed-authoring, 629 model-judge); `traces.jsonl` is 1.05 GB. Accepted-only means no unresolved runs, and only part of it is software-bug fixing. Good for `next_move` / `pick` (internal head); not usable for stop-versus-continue because every trajectory succeeded (a source would then predict the label).
- Neither set is in the conductor pool. No "Astra" dataset appears in the plan or in my searches.

## 2026-10-04 additions (leads, not downloaded)
- `SWE-Critix/{alpaca_style_sft_dataset 0.67 GB, rl_dataset 21.4 GB, test_dataset 3.5 GB}`: critic training data (run text in the prompt, test-level 0/1 outcome, LLM-written critique); not used as negatives. Licence on HF is "other"; the GitHub repo says Apache-2.0.

- CoderForge-Preview: user said NO on 2026-10-04 (do not use or examine). Earlier metadata notes were removed.
- Human-labelled wrong-step sets (2026-10-04, leads): `dengdan1999/RootSE` (MIT, 26 MB, 102 failed runs, human-marked earliest decisive error step; 30 OpenHands/Qwen3-Coder runs were read, results in `results/raw/rootse-signal-check-20261004.json`); `CLoud5-real/longrca-bench` (0.34 GB, 1,140 failed runs, licence not stated, not yet read); `LulaCola/AgentProcessBench` (8,509 human step labels, SWE share unchecked, not read); `PatronusAI/TRAIL` (MIT, gated, not touched).
