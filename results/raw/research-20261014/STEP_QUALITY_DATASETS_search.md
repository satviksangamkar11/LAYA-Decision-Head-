# Step-level action-quality datasets for coding/terminal agents: search report

Date: 2026-10-14. Read-only. Method: HF Hub API (`/api/datasets?search=`, `/api/datasets/<id>`), datasets-server `/size`, `/splits`, `/first-rows`, dataset cards, arXiv API abstracts and HTML, GitHub READMEs/trees. No dataset file above ~1 MB was downloaded (one 1.8 KB range-read of a label JSON). No accounts, no gated terms accepted. "UNVERIFIED" = not confirmed from a primary page.

Known/excluded list was honoured (CodeTraceBench, RootSE, AgentProcessBench, TRAIL, When2Call, AgentJudgeBench, CodeUltraFeedback, typed-decisions, longrca-bench, jev-*, ToolPreference, AgentToolDecisions-180K, nebius, SWE-smith). CodeTraceBench confirmed present on HF (NJU-LINK/CodeTraceBench, non-gated, ~11k downloads).

## Ranking (usefulness for a coding action-quality critic)

### 1. tarsur385/swev-prm-action-label-inputs  (best new find; companion tarsur385/swev-trm-trajectories-25models)
- URL: https://huggingface.co/datasets/tarsur385/swev-prm-action-label-inputs ; companion https://huggingface.co/datasets/tarsur385/swev-trm-trajectories-25models
- Licence: NOT STATED on the label repo (no licence tag; card is 563 bytes). Companion TRM repo says MIT. Treat the label repo as unlicensed until the author is asked.
- Gated: no.
- Size (HF blobs API): `action_labels_10pct.json` 810,759 B; `label_inputs.tar.gz` 58,617,199 B (12,473 transcripts: train 10,107, val 2,366); `manifest.json` 2,128,615 B. Companion TRM: 12,473 rows, 929,244,006 B (datasets-server).
- Labels: PER STEP. Card: "Sonnet-judge GOOD/WRONG/NEUTRAL labels for ~10% (1248 trajs, 50383 actions)". I range-read the JSON head: `{"gemini-2.5-pro/astropy__astropy-13579": {"0":"WRONG","1":"GOOD",...}}`, keyed `model/instance_id` then step index.
- Who labelled: LLM judge (Claude Sonnet; exact version UNVERIFIED). Label prompt and the judge's agreement with humans are not published on the card.
- Coding share: 100% (SWE-bench Verified, mini-SWE-agent, bash-only actions).
- Pre-action context: yes. Transcripts are `ISSUE + OUTCOME + numbered [step k] ACTION/RESULT`; the companion TRM set holds full mini-swe-agent message lists with resolved/failed verdict.
- Caveats: (a) the transcript the judge saw includes OUTCOME, so labels are probably hindsight labels (outcome-aware); UNVERIFIED but implied by the card. (b) Only a 10% subsample is labelled. (c) Three classes, not your four (appropriate / redundant / incorrect / premature); NEUTRAL is ambiguous. (d) No redundancy-specific label.
- Overlap risk: SWE-bench Verified tasks (500 instances, task-disjoint train/val in the TRM set). Generators: 25 models (claude-4-opus ... claude-opus-4.6, gemini-2.5-pro/3-flash, glm-4.6/5, gpt-5.x, gpt-oss-120b, kimi-k2, minimax-m2/2.5, o3, o4-mini, qwen coder, deepseek-v3.2). Overlaps any Verified-based eval and any Claude/GPT-generated pool you already hold.
- Related (same author, weak outcome-propagated labels, step text only): https://huggingface.co/datasets/tarsur385/swev-prm-embeddings-qwen36-27b (MIT; 401,509 train + 94,210 val (state, action, rollout-reward) pairs; `slim_train.json` 236 MB; embeddings 8 GB, not useful as text).

### 2. MasterVito/swe-agent-tool-rubrics-860
- URL: https://huggingface.co/datasets/MasterVito/swe-agent-tool-rubrics-860
- Licence: `other` (unspecified). Gated: no. Updated 2026-08-06.
- Size: 860 decision points (`cases.jsonl` ~5.0 MB) + `map_io.jsonl` (~9.6 MB, full judge prompts/responses). datasets-server: 1,720 rows over two configs.
- Labels: PER STEP, PAIRWISE. Each row has 2 different candidate bash commands sampled from the same prefix, `executed_idx`, and `_judge` with rubric scores R1-R3, A1-A4 per candidate, `winner_idx`, weighted_total, justification. 11 deterministic action categories (edit, read, run_code, search, run_test, write_file, locate, vcs, submit, env, shell_misc).
- Who labelled: LLM judge. For 452 `oracle_distill3k` cases the judge SAW THE GOLD PATCH; for 408 `self_pro` cases the model judged itself without gold patch and without observations. The card says GPT-5.6 prompts/responses are in map_io (judge identity otherwise UNVERIFIED).
- Coding share: 100% (479 instances, 198 repos; SWE-rebench-v2 and SWE-bench Pro).
- Pre-action context: yes (problem statement truncated to 1400 chars, up to 5 prior commands with truncated observations, candidate reasoning). 200 problem statements are empty.
- Caveats: tiny; stratified sampling (not natural frequency); `final_solved` is trajectory-level, not step quality; observation missing for `self_pro`. Best used as a held-out probe or rubric template, not as main training mass.
- Overlap risk: SWE-rebench-v2 (nebius lineage) and SWE-bench Pro tasks; policy model behind the "slime minimal harness" trajectories is not named on the card (UNVERIFIED).

### 3. TrajErrBench, SWE-Bench Pro subset (GitHub THU-KEG/TrajDebug, EMNLP 2026 Findings)
- URLs: https://github.com/THU-KEG/TrajDebug (data in `data/trajerrbench/{zh,en}/swebenchpro/`), paper https://arxiv.org/abs/2608.06346
- Licence: repo MIT for code and original annotations; task text/trajectories stay under upstream SWE-Bench Pro terms (data/README says so explicitly). Gated: no. Not on HF (HF search "TrajErrBench" returned nothing).
- Size: 86 failed SWE-Bench Pro trajectories (EN edition 19,844,129 B total from the git tree; avg ~119.7 steps per paper). The other 400 are tau2-bench (non-coding). English edition released 2026-09-07 with labels inherited from the Chinese edition.
- Labels: PER TRAJECTORY, pointing at a STEP: the earliest decisive (critical) error step. Paper says "manually annotated". Steps after the critical step are not labelled good or bad.
- Coding share: 86 of 486 records (18%); coding subset is long-horizon.
- Pre-action context: yes (full unified-schema trajectories).
- Overlap risk: SWE-Bench Pro tasks; file names show claude-opus-4-6 as at least one generator (other models UNVERIFIED). Overlaps MasterVito `self_pro` tasks.
- Use: a small human-labelled evaluation set for "incorrect" (decisive error). Too small to train on.

### 4. mahirlabibdihan/swe-prm-collection (SWE-Shepherd)
- URL: https://huggingface.co/datasets/mahirlabibdihan/swe-prm-collection ; paper https://arxiv.org/abs/2604.10493 ; code https://github.com/mahirlabibdihan/swe-shepherd (a mini-swe-agent fork; the README I fetched is upstream mini-swe-agent text).
- Licence: none stated (no card, no licence tag). Gated: no. Last modified 2026-02-20.
- Size: 33,119 rows (train + test), 1,615,750,955 B. Paper text says "over 15,000 samples", so the HF dump is larger than the paper's figure (discrepancy unexplained).
- Labels: PER STEP scalar reward in [0,1] (fields: `problem_statement`, `history`, `action`, `reward`).
- Who labelled: PROGRAMMATIC heuristics (successful execution, relevant file access, target-file modification, test results, penalty for repetitive actions, discounted cumulative). Paper Table 2: mean reward for resolved tasks 0.4894 vs unresolved 0.4818, i.e. the reward barely tracks success. Treat as a weak/auxiliary signal, mainly useful for the repetition/redundancy axis.
- Coding share: 100% (SWE-bench training split, 1,794 tasks = full test minus Verified, per paper).
- Pre-action context: yes (history + problem statement + candidate action).
- Overlap risk: SWE-bench (non-Verified) tasks; the generating agent is mini-SWE-agent with an LLM (model for the training trajectories UNVERIFIED; gpt-5-mini was used for the evaluation runs).

### 5. mcemri/MAST-Data (MAD)
- URL: https://huggingface.co/datasets/mcemri/MAST-Data ; code https://github.com/multi-agent-systems-failure-taxonomy/MAST
- Licence: CC-BY-4.0. Gated: no.
- Size: 1,642 traces (+19 human-labelled agreement-study traces); 7 MAS frameworks, 8 benchmarks, 5 LLMs.
- Labels: PER TRACE, 14 binary failure modes. Relevant ones: 1.3 Step Repetition (redundant), 3.1 Premature Termination (premature), 3.2 No/Incomplete Verification, 3.3 Incorrect Verification, 2.6 Reasoning-Action Mismatch. Card: "annotations are produced by an LLM judge, not by human labelling".
- Coding share: partial (ProgramDev, ProgramDev-v2, SWE-Bench-Lite, Test-C among 8 benchmarks; exact split UNVERIFIED). Traces are multi-agent (ChatDev, MetaGPT, HyperAgent...), not single-agent bash loops.
- Pre-action context: yes, but trace-level labels give no step index, so it cannot give per-action targets without extra work.
- Overlap risk: GPT-4o / Claude / Qwen / CodeLlama generators; SWE-Bench-Lite tasks.
- Use: taxonomy and a trace-level sanity check, not step training data.

## Considered and rejected (one line each)
- Kevin355/Who_and_When (184 failure tasks; who + decisive step + explanation; non-gated; licence not stated; GAIA/AssistantBench, i.e. not coding). Mirrors under zhaotianyi, Brabhu786, ClearUp, moonlsss are copies.
- davide221/agenterrorbench (200 trajectories, `critical_failure_step`, `step_annotations`; AgentDebug set = ALFWorld/GAIA/WebShop per the TrajDebug repo layout, not coding; licence none).
- SWE-Gym/OpenHands-Verifier-Trajectories (5,272 rows), R2E-Gym/R2EGym-Verifier-Trajectories (5,750), SWE-Gym/MoatlessTools-Agent-Verifier-Train-Data (2,163), Lego-X/SWE_Lego_real_data_Verifier (17,642; Apache-2.0; `score`): trajectory-level resolved/score only, no step labels.
- davongluck/swe-bench-trajectory-quality-subsets (Apache-2.0): trajectory-level quality scoring of nebius/SWE-rebench-openhands (already in your overlap list).
- JetBrains-Research/agent-trajectories-swe-bench-test-minus-verified (MIT, 1,785 trajs): the router picked the better of gpt-5.2 / gpt-5-mini candidates per turn using gold-patch knowledge, but the card shows only the chosen steps (`selected_models`), `resolved` is null. Pairwise preference could be derived only if rejected candidates are stored (UNVERIFIED, not in card). JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro-Trajectories: 25,200 rows, 6.27 GB, licence `other`; step labels UNVERIFIED (not inspected).
- yuki99981/codetrace-agent-1000-instances: 1,000 executable environments, 273 agent interaction records; no step-quality labels. Not the same as NJU CodeTraceBench; do not confuse.
- few-sh/terminal-wrench-trajectories (Apache-2.0, 15,905 rows): hack vs non-hack trajectory labels (reward hacking), not step quality. aksh-n/basharena_action_only_*-monitor-labels: security monitor "point of no return" judge logs (exfiltration side tasks), not quality.
- McGill-NLP/agent-reward-bench, TraceElephant (CC-BY-4.0, 2,510 rows, 597 MB, multi-agent failure attribution; coding share UNVERIFIED, not inspected), microsoft/AgentRx (auto-gated, tau-retail/Magentic, non-coding), Jolandaaa/agentprm (auto-gated, GUI VL agents).
- wckwan/PRM-agent-rl-artifacts (Apache-2.0): RL run artifacts for Search-R1, not coding.
- Papers with no usable public step dataset found: SWE-PRM "When Agents go Astray" (arXiv 2509.02360, taxonomy of redundant exploration/looping/failure to terminate; no HF dataset located), SWE-TRACE (2604.14820), Counterfactual Rollout Replay (2609.33875, labels are counterfactual returns, no released labels found), ToolPRMBench (2601.12294, tool-use not coding; dataset not located on HF). GitHub release of these UNVERIFIED (I did not search GitHub for them).
- HF searches that returned nothing relevant: swe-shepherd, TrajErrBench, ToolPRMBench, process-reward-swe, step-critique, swe-critic, swe-judge, redundant (non-agent hits only), unproductive, agent-inefficiency, failure-attribution.

## Honest verdict
- Nothing found that is clearly better than CodeTraceBench on the combination "coding + per-step flags + human or careful labelling + scale". Everything human-labelled that I found is tiny (TrajErrBench coding subset 86 trajectories; Who&When 184 and non-coding).
- The only comparably large step-level coding supervision found is #1 (50,383 per-action labels, 1,248 SWE-bench Verified trajectories), and it is LLM-judged, probably outcome-aware, licence-unclear, and only 3-class. It complements CodeTraceBench; it is not a replacement. A critic trained on it should be validated against the human-labelled TrajErrBench subset and CodeTraceBench flags, not against itself.
- #4 gives cheap programmatic redundancy/progress signal at 33k rows but is shown by its own paper to correlate poorly with success; use as an auxiliary or filter only.
- Mapping to your four classes (appropriate / unproductive-redundant / incorrect / premature): "premature" has essentially no direct step-level label anywhere found (closest: MAST 3.1 at trace level, and the `submit` category in #2). "Redundant" is closest in #4 (repetition penalty) and MAST 1.3 (trace level). "Incorrect" is covered by #1 WRONG, #3 critical step and CodeTraceBench.
- Overlap warning: almost everything above is built on SWE-bench Verified / SWE-bench (non-Verified) / SWE-rebench / SWE-bench Pro tasks with Claude/GPT/Gemini/open-weight generators. If your current training or eval pools use these task sets, split by `instance_id` before mixing.
