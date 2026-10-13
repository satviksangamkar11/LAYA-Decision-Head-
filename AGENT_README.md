# AGENT_README: working log of the agent sessions (formerly README.md)

> Renamed from README.md on 2026-10-12 (user, chat). This file is the dated working log: status reports, frozen plans and the fixed NVIDIA call
> logic. The project overview is now [README.md](README.md). Original title: ﻿# jev-airllm: a decision head for local LLMs

Goal: one internal decision head (a Laya/Jeeves-style pointer head) that reads the hidden states of a **local, frozen
language model** and picks among candidate actions with calibrated probabilities. It is **backbone-agnostic**: the
head takes the hidden size as a parameter and is fitted per model. **Current experiment backbone (user, 2026-10-01): `Qwen3-4B-Thinking-2507`**
(frozen, 8.05 GB, revision `768f209d9e`). `gpt-oss-20b-heretic` and the 120B are parked. Laya is a baseline and in-lane helper, never the ACTION teacher.

Rules: [CLAUDE.md](CLAUDE.md). Design record and survey: [DECISION_HEAD.md](DECISION_HEAD.md). Step list and closure path: [PLAN.md](PLAN.md) section 11.
Tags: **[M]** measured here, **[A]** reported by an author and not reproduced, **[E]** estimate.
Last updated: 2026-10-12 01:18 IST (status of the Fable imitation data work: [HANDOFF_2026_10_11_STATUS.md](HANDOFF_2026_10_11_STATUS.md)).

## FULL STATUS REPORT - 2026-10-12, 01:18 IST (2026-10-11, 19:48 UTC)

Everything below is measured unless marked otherwise. Full record with file paths: [HANDOFF_2026_10_11_STATUS.md](HANDOFF_2026_10_11_STATUS.md)
sections 14-15.

**Running at this time**
- GLM-5.3 full-reasoning review of the 137 Stage 1 pools: 126 of 137 packets answered (5 requests in flight per key, 14 keys; 65 rate-limit
  rejections in the first minute, none after).
- The read-only progress page `progress_ui/server.py` at http://localhost:8765.
- Nothing else. No Kaggle notebook is running. The fine-tuning pilot and the Stage 1 generation have finished.

**Goal (unchanged).** A reusable LAYA decision head that makes a frozen offline coding model choose better next actions. Teacher: Fable's recorded
actions. Qwen3-4B-Thinking-2507 is frozen and never fine-tuned.

**Pretrained head in use.** Laya `Coding_Decision_Agent` (`models/coding-decision-agent/model.safetensors`, Apache-2.0). 31 decision tensors
(26,248,193 parameters) are copied from it; a new adapter (8 tensors, 5,246,982 parameters) maps frozen-Qwen states into them; 273,154 parameters
stay fixed. Joint training updates 31,495,175 parameters. The checkpoint file is never written (hash checked after the pilot).

**Data inventory (TRAIN, coding, tool decisions).** 1,342 eligible decisions in 281 trajectories. Candidates generated for 603 so far
(79 first experiment, 189 batch v2, 40 + 95 conventions prompt, 200 Stage 1). Not generated: 471 usable + 268 first-step decisions (excluded by
rule). VAL 227, CAL 309, TEST 713 decisions exist and have never been read; they have no candidate pools yet.

**Dataset v2 (frozen, in use).** `results/raw/laya-fullaction-set-v2-20261011T134826Z`: 123 pools, 96 trajectories, 307 alternatives; each pool =
Fable's recorded action (positive) + 2-4 alternatives (unknown, never "wrong"). 70 pools reviewed by GLM + Gemma, 53 by GLM + Muse; recorded per
pool. Every kept alternative verified per option: both judges accepted it, it passed the objective checks (re-checked per option on 2026-10-11),
it is not the same Bash command as Fable's. Hand audits: 30 + 24 options.

**Judges.** GLM-5.3 full reasoning + a second judge, both must accept. Gemma 4 31B passed the pair gate but its endpoint went down; of seven
substitutes only Meta Muse Glimmer 30B passed the same gate (0 of 46 known-bad options accepted by both; Fable's action accepted by both 24 of 28
= 85.7% against the pair gate's 80%; it was never run on the individual 90% qualification). Kimi K3 down; gpt-oss-120b retired on NVIDIA.

**Level 1 features (captured on Kaggle T4, verified here).** 123 pools x 4 context variants = 492 captures
(`results/raw/level12-run-v2-cancelled-20261011T175252Z`): real context, another state of the same trajectory, another trajectory, no context.
**Level 2** (Qwen's own full-context proposals) was NOT completed: the library's generation runs out of memory above about 5,000 prompt tokens and
the corrected loop needs 0.06-0.27 s per token (about 24 hours on two T4s for 100 x 4); deferred by the user.

**Baselines and fine-tuning pilot (123 pools, tested on unseen trajectories, 3-seed means, chance 29.6%).**

| Model | Picks Fable's action | 95% interval |
|---|---|---|
| Action-history prior | 37.2% (36.6% with a fixed tie-break) | 27-46 |
| Pretrained head, no training | 36.6% | 32-42 |
| Options text only | 62.6% | 54-72 |
| Fine-tuned head, adapter only | 77.2% | 71-83 |
| Fine-tuned head, joint | 76.7% | 70-83 |
| Head trained and tested with NO context: adapter / joint | 76.7% / 78.3% | 70-85 |
| Linear probe on Qwen option states, real context | 84.6% | 78-91 |
| Linear probe, NO context | 84.6% | 78-91 |
| Linear probe given another state / another trajectory as context | 89.4% / 85.4% | 79-94 |

Paired results: joint vs adapter-only -0.5 points (no difference); head vs linear probe about -7.5 points (significant); context vs no context:
no difference for any model. Train accuracy 94-97% against 76-81% on test. Repeated action texts across folds explain about 2 points (probe
82.4% on the 108 pools whose action text was never seen). By tool, no-context probe: Bash 92%, Write 87%, Edit 67%, multi-call 71%.
**Conclusions under the pre-registered rules:** keep the pretrained layers frozen at this data size; diagnose the adapter / head before scaling
head training; these pools show no measured benefit from the coding context; none of this is evidence about coding-task success.

**Stage 1 (200 more decisions, session-conventions prompt).** Generation done: 1,600 calls. 142 structurally valid pools, 137 passed the
objective checks (C# 18, Go 14, PowerShell 14, Python 25, Ruby 13, Bash 11, Java 11, Rust 11, Zsh 9, TypeScript 8, others 3; 21 multi-call).
Review in progress (see "Running"). Then Muse, scoring, assembly, audit, freeze.

**Sandbox (WSL Ubuntu-24.04).** Added on 2026-10-11 with the user's approval: Ruby 3.2.3, Zsh 5.9, Rust 1.75.0, .NET SDK 8.0.131 (g++ was present).
Syntax-only parsers now cover Python, Go, Java, PowerShell, JavaScript, Bash, Ruby, Zsh, Rust, C# and C++. Fable's own files pass every new
parser, so they are enforced; 17 Stage 1 alternatives fail one and will be dropped even if both judges accept them. TypeScript is not enforced
(2 of 7 of Fable's files fail the available check). Each option records its verification level; an `Edit` result is not parsed in any language.

**Bugs found and fixed on 2026-10-11.** (1) Scoring kept one verdict per repeated option label: fixed, per-option; dataset v1 (54 pools)
superseded by v2. (2) The objective screen keyed results the same way: fixed; all 307 kept alternatives of v2 re-checked and pass. (3) Kaggle
Level 2: a worker died because the parent process held GPU memory; every stage now runs in its own process and workers restart and resume.

**NVIDIA call logic.** Section below ("FIXED NVIDIA CALL LOGIC"): 14 keys, 5 requests in flight per key, at most 39 sends per key per minute, a
rate-limit response cools that key one minute, nothing completed is resent. Measured: about 60 generation calls a minute.

**Next steps, in order.**
1. Finish Stage 1: GLM (running), Muse on the eligible pools, per-option scoring, pair gate on this batch's controls, assembly with the parser and
   same-as-Fable filters, blind hand audit including C#, Ruby and Rust, freeze as a separate set.
2. Measure Stage 1 separately: yield, options-only and no-context scores by language and tool.
3. Small test of ranking Qwen's own candidates (the only measurement of whether any scorer helps the offline agent); the linear probe is a
   candidate decision layer for it.
4. Decide on the remaining 471 decisions, head diagnosis, and outcome-labelled data only after steps 1-3.
Needs the user's GO: any Kaggle run, any further fine-tuning, generation beyond Stage 1.

## UPDATE - 2026-10-13, 17:30 IST: mutation diagnostic scored; all numbers; re-plan

- 19 TRAIN states, passing fix vs one-token-broken copies, scored on Kaggle: LAYA 0%, random head 0%, baseline 1.8%, probe 0% vs random 20.5%;
  probabilities within ~0.02 of uniform: the features cannot see a one-token bug. Capture v1 aborted on one over-limit pool; capture v2 (another
  session, per-option routing) captured 19/19; scored from an uploaded features dataset because Kaggle hides outputs of ERROR kernels.
- All numbers: ALL_NUMBERS_2026_10_13.md. Re-plan: LAYA_DECISION_BRAIN_PLAN_2026_10_13.md section 8. HANDOFF 16.24; ledger 69.

## UPDATE - 2026-10-13, 15:30 IST: low-effort GLM pilot v3 complete; expansion gate failed

- Offline verification passed (3 saved answers decode; strict unwrap). Generation 25/25 HTTP 200, 0 timeouts (up to 5 in flight per key).
- Only 5 of 25 valid (model format errors at low effort). Oracle 2 of 5; 1 informative problem (py-event-sched:1): gate needs 2, FAILED, no expansion.
- Local capture parity passed (0.999992 / 0.999957; 12/12 top choices); local scoring reproduced Kaggle 128/128. Every learned scorer picked the passing
  fix on the informative problem; end to end 2 of 5 each vs first valid 1. Execution bug (single-call actions dropped) fixed as OUTCOMES_A3b; GLM-max
  pilot unaffected. HANDOFF 16.23; ledger 68.

## UPDATE - 2026-10-13, 14:15 IST: low-effort GLM verification v2 (sequential) stopped on answer format

- Sequential calls on KEY_2: both HTTP 200 (no 504; consistent with, not proof of, the concurrency explanation). Call 1 decodable; call 2 wrapped the
  action in a JSON list (wrong_top_level): stopped, nothing generated. Blocker: answer-format reliability at low effort / temperature 1.0. HANDOFF 16.22; ledger 67.

## UPDATE - 2026-10-13, 14:00 IST: GLM-5.3 low-effort pilot stopped at API verification

- Call 1 HTTP 504 after 302 s; call 2 HTTP 200 with a decodable action (Read + Bash). The registered rule needs both: stopped, nothing generated.
- Both stopped verifications (gpt-oss-20b, GLM-low) failed on the second of two SIMULTANEOUS calls on KEY_2 (504 at the gateway's ~300 s limit), which
  points to the gateway / concurrency rather than the models. A sequential re-check needs a new registration and the user's GO. HANDOFF 16.21; ledger 66.

## UPDATE - 2026-10-13, 13:40 IST: gpt-oss-20b pilot stopped at API verification

- Two verification calls on the TRAIN request py-incident-cli:1: HTTP 504 after 302 s; HTTP 200 with an empty answer (235 reasoning characters). Both
  required checks failed; the pilot stopped as registered. No VAL request sent, nothing generated, executed, captured or scored; local capture not
  attempted; expansion gate not reached. `results/raw/oss20b-verify-20261013T080629Z`; HANDOFF_2026_10_11_STATUS.md 16.20; ledger 65.

## UPDATE - 2026-10-13, 13:30 IST: GLM-5.3 outcome pilot done (not informative enough); Go / C# / Rust replay repaired

- Rebuilt scorers with saved weights: 12 of 12 reproduce their recorded VAL picks exactly (`results/raw/head-swap-rebuild-run-20261012T225237Z`).
- Sandbox languages: .NET 10, Rust 1.96, offline Go module and NuGet caches; Go 41 of 41 repaired states replay exactly (VAL 14 of 14, was 0); C# 2 of 3
  and Rust 1 of 1 match under a new formatting-only rule v2 (`results/raw/wsl-toolchains-20261013/MANIFEST.md`).
- GLM-5.3 pilot (5 VAL problems x 5): 17 of 25 valid, 15 of 17 pass, oracle 4 of 5, 1 informative problem. Every learned selector ties with "first valid
  candidate" (3 of 5 end to end); on the one informative problem every run picked a failing fix: one problem, no conclusion. Amendments 2 and 3 recorded.
  Not expanded. Next generator (intermediate strength, e.g. gpt-oss-20b) waits for the user. Details HANDOFF_2026_10_11_STATUS.md 16.18-16.19; ledger 64.

## UPDATE - 2026-10-13, 04:05 IST: head-swap experiment finished (Kaggle T4, 126 runs): random heads beat the pretrained LAYA heads

- Gates passed (coding seed 0 out of fold 77.89% = recorded; VAL 78.3% vs 76.1%).
- Out of fold: coding 77.2%, typed 79.0%, code 76.9%, conductor 79.6%; random heads 83.2-83.8%; probe 85.8%. coding minus random -6.2 (corrected -9.0 to
  -3.7): pretraining is a handicap under this interface. No specialisation effect; context effect +4.5 (not significant); probe minus coding +8.6.
- By the user's reading fixed beforehand: head-specialisation research stops for now; next priority is coding success (cross-agent execution test).
  Details HANDOFF_2026_10_11_STATUS.md 16.17; ledger row 63.

## UPDATE - 2026-10-13, 04:00 IST: baseline frozen, mutation pilot v1.1 usable, head-swap experiment prepared (NOT uploaded)

- Research baseline v1 frozen: `results/raw/laya-baseline-v1-20261012T215242Z` (hashes, model card).
- Mutation pilot v1: not usable by its rule (14 of 20); cause: `| tail -N` hid the test exit code in 5 TS states. v1.1 with the audited evaluation commands,
  same states, mutants and rule: 19 of 20 usable, 114 WORSE / 36 EQUAL_ON_TESTS / 2 UNKNOWN, no crashes; ts-msgfmt:3 invalid (Fable's action fails its tests).
  Only 72 of the 118 "train" replayable states were canonical TRAIN; CAL / TEST / VAL states were never executed.
- Head-swap experiment prepared (`kaggle/head_swap_gpu.py`, bundle `results/raw/head-swap-bundle-v1-20261012T221311Z`); upload and launch HELD for the user.
  The four pretrained heads are close relatives (17-19% weight difference from coding; typed vs conductor 4.8%).
- Details: HANDOFF_2026_10_11_STATUS.md 16.16; ledger row 62.

## UPDATE - 2026-10-13, 03:30 IST: plan after exp5 (user GO for steps 0, 1, 2)

- Step 0: exp5 documented (HANDOFF_2026_10_11_STATUS.md 16.15, README.md 2j, ledger row 61). Step 1: freeze a research baseline (pretrained head, three
  Fable-trained adapters, linear probe; hashes and model card; a research baseline, not a proven product). Step 2: mutation pilot on 20 canonical TRAIN states in
  isolated WSL sandboxes on this PC (user: no Kaggle needed for a few test runs); a diagnostic only, because these states were trained on.
- Next priority, needs a GO for generation: candidates written by another agent for the 23 fail-to-pass VAL decisions, labelled by sandbox test results, scored by
  the frozen scorer, the probe, random choice and the agent's own first candidate.
- Correction accepted: the linear probe on the same frozen Qwen features can transfer across candidate sources as easily as LAYA; what LAYA adds over it is the
  open question. Not approved: a second bridge, CAL / TEST use, more Fable generation, any training.

## UPDATE - 2026-10-13, 02:46 IST (2026-10-12, 21:15 UTC): direct read-path experiment (exp5) finished (Kaggle T4)

- **Reproduction gate passed exactly** (previous arm D 50.39% out of fold, native 69.67%).
- **The frozen-head read path improved agreement with Fable:** 71.1% out of fold (+20.7, +15.3 to +26.1), cleaned VAL 68.1%. The gain belongs to the new linear
  layer; the probe is still 14.7 points higher (85.8%).
- **Joint training did not improve the 31 pretrained tensors:** with the path off they score 41.5%, below the same recipe without a path (48.9%; -7.5, 98.3% interval
  -14.9 to -0.5) and near the released head (39.9%). Joint model 68.4%, not above the path alone (-2.6, -6.2 to +0.8).
- **Substantial context-independent ability:** trained without context the path scores 75.6% against 71.1% with context on the same 112 pools.
- **Native retention measured:** 67-72% in every arm (released 71.5%). Half data 68.6%, no worse than full.
- **No exp5 checkpoint was adopted or saved.** Fable-choice agreement only; not coding-task success, not cross-model portability.
- Report `results/raw/head-exp5-run-20261012T211558Z/REPORT.json`; details HANDOFF_2026_10_11_STATUS.md 16.15.

## UPDATE - 2026-10-13, 01:45 IST (2026-10-12, 20:15 UTC): four-arm head-only experiment finished (Kaggle T4, 88 runs, 14 minutes)

- **Reproduction gate passed** (arm A: VAL 55.1% against the recorded 54.3%, native 69.0% against 69.1%).
- **Out of fold on 303 TRAIN pools:** released 39.9%; A 45.5%; B+ 47.4%; C (1e-4) 50.3%; C (2e-4) 51.2%; D (C + native replay) 50.4%; linear probe 85.8%.
- **D passes the registered rule:** +4.8 points over A (+0.9 to +8.7) with native 69.7%. C loses native skill (63.9% / 57.6%); B+ destroys it (44.9%).
- **VAL does not confirm:** A 55.1%, D 52.9% (-2.2, -11.1 to +5.7), C 48.6%, B+ 49.3%; probe 89.1%.
- With the higher layer-1 rate the head fits 70.4% of its own pools but 50.3% of unseen ones: the limit is now generalisation.
- Stopped as agreed. Nothing adopted, no checkpoint saved. Details: HANDOFF_2026_10_11_STATUS.md 16.14; report `results/raw/head-exp4-run-20261012T201525Z/REPORT.json`.

## UPDATE - 2026-10-13, 01:20 IST (2026-10-12, 19:50 UTC): four-arm head-only experiment started on Kaggle GPU; new handoff

- **Start here now: HANDOFF_2026_10_13_0120AM_IST.md** (rules, what is running, all results, plan).
- Running: `sangeetasangamkar/laya-head-exp4-run` (GPU T4), script `kaggle/head_exp4_gpu.py`, bundle `results/raw/head-exp4-bundle-v1-20261012T194655Z`.
  Arms: A reference, B calibration control (no training), B+ calibrated start (added by the agent, disclosed), C higher rate on context layer 1, D = C + native replay
  (target: the original model's predicted distribution, not gold labels). 5-fold cross-validation on 303 TRAIN pools, then cleaned VAL and 600 native items.
  Arm A must reproduce the recorded 54.3% first, else stop.
- Found on the fit pools while preparing: the released head's best temperature is unbounded (its loss only reaches the equal-score value); the range is capped at 20.
- After the run: report and stop. No further training without a new GO.

## UPDATE - 2026-10-13, 01:08 IST (2026-10-12, 19:38 UTC): causal check of the layer-1 feed-forward update (Kaggle CPU, read-only)

- Scaling the update down raises the probe after layer 1 from 56.5% to 76.1% (s = 0) and the probe at the scorer input to at most 69.6% (s = 0.1).
- **The head's own choices do not improve at any scale** (VAL 39-43%, TRAIN fit pools 40-41%), and native accuracy falls from 71.5% to 43-47% at s <= 0.03.
- Reading: the signal survives when the update is reduced, but layer 2 and the scorer do not read it; the update itself carries the native skill.
- Controlled training experiment (reference / calibration / differential rate): approved by the user, NOT started. Details: HANDOFF_2026_10_11_STATUS.md 16.13.

## UPDATE - 2026-10-13, 00:53 IST (2026-10-12, 19:23 UTC): head review, steps 1 and 2 (Kaggle CPU, read-only)

- **Where the signal is lost:** bridge output 87.0% -> after the attention of layer 1 76.1% -> after its feed-forward step 56.5%. That step writes an update 96 times
  larger than its input (4,041 against 42).
- **Not attention mixing:** with tokens blocked from seeing each other the probes stay at 54-59% and the native accuracy only moves 71.5% -> 68.8%.
- **Correction:** the "options collapse, cosine 0.97" is mostly one common vector; with it removed the cosine is 0.48 (0.29 at the scorer input; original model 0.18).
- **Head-only training:** the released head starts worse than equal scores (loss 1.064 against 0.813); the descent direction in the context layers is nearly "make the
  options alike" (cosine 0.74-0.997); after training the loss is 0.751 and the score spread fell 0.73 -> 0.28. Unknown weight 0.5 or 1.0: same gradient (0.9999).
- Nothing was trained. Details: HANDOFF_2026_10_11_STATUS.md 16.12; report `results/raw/head-review-run-20261012T192308Z/REPORT-20261012T192246Z.json`.

## UPDATE - 2026-10-13, 00:31 IST (2026-10-12, 19:01 UTC): sequence-input test - prediction not met

- Kaggle sequence capture done (1,187 of 1,187 requests, 46 minutes). The released head was then fed a sequence (16 context slices, 4 slices + a marker per option)
  through a bridge aligned on 600 native items, against a single-vector control aligned the same way.
- **The options still collapse (cosine 0.985 against 0.980) and the probe still drops at context layer 1 (80.4% -> 60.9%; control 89.1% -> 54.3%).** The written
  prediction is not met, so this sequence format does not fix the loss. Small, non-decisive differences favour the sequence arm; its bridge aligned worse (native 56.5% against 62.0%).
- By the staged plan: no larger capture, no head-only training on sequences. Next: review the head's loss and optimisation.
- New rule (user): compute-heavy jobs run on Kaggle, not on the PC. Details: HANDOFF_2026_10_11_STATUS.md 16.11.

## UPDATE - 2026-10-12, 23:30 IST (18:00 UTC): read-only audit of the head's input path

- The loss of signal happens at the head's FIRST pretrained context layer: probe 87.0% on what that layer receives, 56.5% after it. After the layers the options of
  a pool are almost the same vector (cosine 0.97; the original model on its own items 0.88); head-only training makes it worse (0.99).
- No integration error found (masks, shapes, checkpoint); the input scale matches the original model's (about 50), so the earlier scale guess is withdrawn.
- Structural difference: the original layers run over hundreds of tokens and build each option from its own text; ours get one vector per option.
- Joining the pre-context vector to the scorer input restores 84.8%. Data pilots and further fine-tuning stay on hold. Details: HANDOFF 16.9.

## UPDATE - 2026-10-12, 22:54 IST (17:24 UTC): head-only diagnostics - the bridge is not the limit

- Probes on identical inputs and splits (46 cleaned VAL pools): raw Qwen states 89.1%, **output of the never-trained-on-Fable bridge 87.0%, after the head's
  pretrained context layers 58.7%**; the trained head scores 54.3%. The signal reaches the head and is lost inside it.
- Four times more training steps: 55.8% (+1.4, not significant; plateau). Half the data: 48.6% (-5.8, interval touches 0). The head fits only 61% of its own
  training pools at the safe learning rate: under-fitting, not a data shortage.
- Record audit by option position: 0 kept alternatives fail the technical screen. Details: HANDOFF 16.7, 16.8.

## UPDATE - 2026-10-12, 22:30 IST (17:00 UTC): all documents brought up to date

- Consolidated findings, problems on record, approvals and the proposed plan: **HANDOFF_2026_10_11_STATUS.md section 16**. Project overview with every result
  table: **README.md**. Rules added today: CLAUDE.md ("Objective restated", "Evaluation data", "Approved, not started"). Also PLAN.md 19 and DECISION_HEAD.md 25.
- Objective (user): strengthen the pretrained LAYA head itself; Qwen only supplies features; no encoder is trained.
- Headline numbers: adapter on 334 pools 80.2% in-dataset and 79.2% on 53 unseen VAL pools (probe 86.2% / 94.3%); cleaned v4 304 pools; adapter + head 75.4%
  against 76.1%; head only with a frozen never-trained-on-Fable bridge 39.1% -> 54.3% (2e-5) or 52.2% (2e-4); outcome pilot: no usable Qwen candidate.
- Approved and not started: mutation pilot (20 TRAIN states), near-miss recovery pilot (25 decisions). Waiting for the user: longer head-only run, head-only
  with replay of the native items, outside decision data.
- Note: another session edited these documents in parallel today (a second section 15.31, a second ledger row 54); both were renumbered, nothing was removed.

## UPDATE - 2026-10-12, 21:25 IST (15:55 UTC): outcome pilot on VAL - Qwen wrote no usable fix

- Pool changed from CAL to **23 clean fail-to-pass VAL decisions** (the request builder refuses CAL contexts by design; user chose to keep that guard). Original
  registration kept, amendment for the pool only.
- **Pilot, 5 decisions x 4 Qwen-4B samples on Kaggle (about 4.0 GPU hours): 15 of 20 samples hit the 6,144-token limit while still reasoning; 1 valid action
  (a file read); 0 code-change candidates; 0 decisions where any candidate makes the tests pass.** No scorer comparison was possible.
- Candidate generation, not ranking, is the bottleneck on these decisions. Stopped before the remaining 18. Details: HANDOFF 15.31.

## UPDATE - 2026-10-12, 22:03 IST (16:33 UTC): head-only fine-tuning with the bridge frozen

- Objective restated by the user: improve the pretrained LAYA head itself; Qwen only supplies features; no encoder is trained.
- **Only the head's 31 tensors trained, bridge bit-identical in all 12 runs.** With a bridge that never saw Fable: cleaned VAL 39.1% -> 54.3% at head learning rate
  2e-5 (+15.2 points, interval +6.3 to +24.9), native skill kept (71.5% -> 69%); at 2e-4 52.2% (+13.0, -2.5 to +26.7, the registered primary, not significant) with
  native skill collapsing to 44-58%. On top of a Fable-trained bridge: no gain.
- So the head can store part of Fable's choices, but it stays about 22 points below adapter-only (76.1%) on the same pools. Details: HANDOFF 15.32.

## UPDATE - 2026-10-12, 21:30 IST (16:00 UTC): outcome pilot finished; five research reports

- **Outcome pilot (5 VAL decisions, Qwen writes 4 candidates each): 15 of 20 samples never finished thinking within 6,144 tokens; 0 code-change candidates; nothing
  to execute or rank. 3.75 GPU hours used; the other 18 decisions were not generated.** Candidate generation by the small model is the bottleneck.
- Research (`results/raw/research-20261012/`): new Fable repositories are mostly copies (a few hundred distinct sessions); recorded branches are too rare; code
  mutations judged by tests look feasible on 118 replayable TRAIN states; 150 failed TRAIN decisions were one accepted alternative short of a pool.
- Open items: the v3 manifest's gate-exception list is incomplete (52 pools); one report quoted three TEST example lines (redacted). Details: HANDOFF 15.31.

## UPDATE - 2026-10-12, 18:46 IST (13:16 UTC): cleaned labels (v4) and the head-itself experiment

- **Cleaned sets (new versions, v3 untouched):** 59 strict equivalents of Fable's action removed from training (6.8%), 304 pools left; cleaned VAL 46 pools.
- **One controlled experiment, adapter only against adapter + the head's 31 pretrained tensors, same everything else:** cleaned VAL 76.1% against 75.4%
  (difference -0.7 points, interval -5.9 to +4.3); in-dataset +1.0 (-1.8 to +3.8). **No measurable gain from updating the head; adapter-only stays.**
- The probe is still 11 to 14 points ahead. Still pending: the CAL outcome pilot (44 useful decisions, 27 fail -> pass). Details: HANDOFF 15.30.

## UPDATE - 2026-10-12, 18:15 IST (12:45 UTC): CAL replay qualification for the outcome test

- User GO for an outcome-based evaluation on CAL (TEST untouched). Count first, no generation yet.
- **CAL edit / write decisions: 155; reconstructable 72; replay to the recorded test result twice 52; test result changes with Fable's action alone 44
  (35 trajectories); clean fail -> pass 27.** Languages: Python 14, TypeScript 12, PowerShell 9, Java 6, Go 2, Zsh 1.
- Saved Qwen proposals exist for 30 of the 44 but 97% were cut at the length limit. Details: HANDOFF 15.29.

## UPDATE - 2026-10-12, 17:40 IST (12:10 UTC): equivalence audit of the training pools; next step is an outcome-based test

- Blind audit of 60 random training pools (161 alternatives): **6.8% strict equivalents of Fable's action, 25.5% including loose ones**, almost all in pools where
  Fable runs a test command. Saved as a separate annotation; dataset v3 is unchanged.
- For the outcome-based test: only 2 Fable decisions have saved Qwen candidates; 141 decisions replay in the sandbox (118 TRAIN, 23 VAL, 0 CAL).
- Waiting for the user: CAL or VAL decisions, and a Kaggle session for Qwen generation. Details: HANDOFF 15.28.

## UPDATE - 2026-10-12, 17:15 IST (11:45 UTC): why the linear probe beats the head on VAL (read-only diagnosis)

- **24% of VAL alternatives (32 of 134) are Fable's action in effect.** Counting them as correct: head 79.2% -> 83.6%, probe stays 94.3%; the gap goes from
  15.1 to 10.7 points. The original result is unchanged; this is a separate sensitivity figure.
- **The probe never picks an equivalent alternative** (0 of its 53 picks), so part of its score is recognising Fable's exact wording, not a better decision.
- **The head is unstable on close calls:** in 9 of the 13 pools where the probe wins, at least one of the three adapters is right.
- Nothing was trained or changed. Details: HANDOFF 15.27.

## UPDATE - 2026-10-12, 16:50 IST (11:20 UTC): evaluation on new VAL trajectories

- 178 VAL decisions (trajectories disjoint from TRAIN) went through the unchanged pipeline for evaluation only: **53 qualified pools**, 134 alternatives, pair gate
  passed on every check. Nothing was fitted on them.
- **Head (three saved adapters, mean): 79.2% with the real context (in-dataset 80.2%: no collapse), 64.2% with no context: +15.1 points from context
  (interval +6.3 to +24.1).** Linear probe 94.3% / 81.1%; options-only text 67.9%; chance 29.3%. The probe beats the head by 15.1 points.
- Hand audit: 5 of 18 kept VAL alternatives are Fable's action in effect (redundant Read added, or `chmod` dropped); counted as misses for every scorer.
- A local-GPU capture now exists (`tools/val_capture_local.py`) and matches Kaggle (worst cosine 0.99999 on 12 pools). Details: HANDOFF 15.26.

## UPDATE - 2026-10-12, 13:35 IST (08:05 UTC): pretrained LAYA head fine-tuned on all Fable pools

- User GO. **Three fine-tuned adapters are saved** in `results/raw/laya-finetune-all-v1-20261012T080145Z/adapters` (adapter only; the 31 pretrained tensors and Qwen are unchanged). Data: 334 pools, 192 trajectories.
- In-dataset cross-validation (chance 28.9%): **head 80.2%, linear probe 86.2%** (probe ahead by 6.0 points, interval +1.9 to +10.1). On the 160 pools that were
  held out this morning the head went from 63.1% to 81.3% once pools of the same batches were in training; the probe there is 83.8%.
- Not an independent test: every Fable pool is now used, so there is no untouched pool to score on. The local GPU and Kaggle are now allowed by the user; this
  4-minute job ran on the CPU. Details: HANDOFF 15.25.

## UPDATE - 2026-10-12, 12:36 IST (07:06 UTC): held-out comparison done - alignment does not help

- Real-context capture of the later pools finished on Kaggle: 214 of 214 pools (one pool over the 24,000-token limit was excluded, not truncated; two earlier
  kernel versions aborted at start-up with nothing captured).
- **Held-out test (160 pools on 98 trajectories never used for fitting; agreement with Fable; chance 28.1%):** linear probe 75.6%, LAYA head with fresh adapter
  63.1%, head with aligned adapter 58.4% (primary) and 54.9%. **Aligned minus fresh = -4.75 points (interval -9.6 to -0.3): worse, not better.**
- Decision by the pre-registered keep rule: the Fable-only adapter stays, the aligned adapters are dropped. The earlier +5.2 points inside dataset v2 did not transfer.
- The linear probe still beats the LAYA head (+12.5 points, interval +7.3 to +17.6). Details: HANDOFF 15.24.

## UPDATE - 2026-10-12, 11:27 IST (05:57 UTC): Fable dataset v3 frozen

- **`results/raw/laya-fullaction-set-v3-20261012T055420Z`: 335 qualified pools, 866 alternatives**, every option individually verified. This completes the Fable TRAIN decisions under the
  current eligibility rule: 1,074 of 1,342 eligible decisions were sent to the generator; **the 268 first-step decisions were not processed**; VAL, CAL and TEST have no pools.
- **Roles per pool:** train 120 (the pools every model so far was fitted on), train_extension 54 (later pools on train trajectories), **heldout_transfer 161 pools
  on 98 trajectories that occur in no train pool**. A held-out pool is not used to fit, select or tune anything before the transfer comparison.
- The last batch was accepted under a documented infrastructure exception (network and endpoint outages; the response-failure check is recorded as FAILED, not
  passed; judge-caused failures were 0% for GLM and 1.6% for Muse).
- Rules tightened before the freeze: TypeScript and Go alternatives that fail a qualified parser are removed (5 options); a command that only repeats Fable's command
  counts as the same action.
- Running: one Kaggle real-context capture of the 215 later pools; then the pre-registered held-out comparison of the aligned and the fresh adapter (HANDOFF 15.23).

## UPDATE - 2026-10-12, 05:40 IST (00:10 UTC): alignment result and the remaining Fable batch

- **Adapter alignment (pre-registered, HANDOFF 15.20):** training only the adapter on the checkpoint's own dataset makes the frozen pretrained head work on Qwen
  features for its native questions: 70.8% gold accuracy on 600 held-out items against 78.3% for the original model and 19-26% unaligned. Starting Fable
  training from the probability-aligned adapter gives 82.4% on dataset v2 (trajectory-held-out, 3-seed mean) against 77.2% from a fresh adapter: +5.2 points,
  interval +0.6 to +9.8. Caution: three alignment targets were tried, the test is cross-validation inside v2, the linear probe is at 84.6%. Needs
  confirmation on held-out pools with real-context features.
- **Remaining Fable decisions (471):** generated, screened (352 pools), reviewed by GLM + Muse, assembled: **162 strict pools**, 105 trajectories. NOT frozen:
  a network interruption and a Muse outage made 42-47% of first-attempt judge replies unusable (all recovered by resending), which fails the written
  response-failure check; the user decides whether the batch counts. Judge accuracy on its controls is as before (0 of 316 known-bad accepted by both).
- **Totals if accepted:** dataset v2 123 + Stage 1 58 + this batch 162 = 343 pools (before the corrected v2, which loses 3 TypeScript options).
- **Sandbox:** TypeScript is now really parsed; 4 previously kept TypeScript alternatives fail it (3 in v2, 1 in Stage 1) and will be removed at the next freeze.
- Running: nothing. Details: HANDOFF sections 15.16-15.22.

## UPDATE - 2026-10-12, 02:20 IST (2026-10-11, 20:50 UTC): Stage 1 frozen and the transfer result

- **Stage 1 frozen as its own set:** `results/raw/laya-fullaction-stage1-v1-20261011T203622Z`: 59 pools, 52 trajectories, 164 alternatives (GLM + Muse;
  pair gate holds on this batch: 0 of 123 known-bad options accepted by both, Fable's action accepted by both 85.7%, near-duplicates 53 of 53). Languages:
  Go, C#, Python, Ruby, Rust, Java, Bash, PowerShell, Zsh, TypeScript. 41 of the 60 assembled pools are Bash decisions; only 3 Write pools survived.
  Hand audit of 24 options: 13 of 18 kept plausible, 4 unclear, 1 the same as Fable's action (now filtered), 0 implausible.
- **Transfer test (pre-registered; scorers fitted on dataset v2 only, no-context features, tested on the 48 Stage 1 pools from new trajectories; chance 26.9%):**
  linear probe 62.5% (inside v2: 84.6%); adapter-only head 57.6% (76.7%); options text 47.9% (62.6%); action prior 32.1%; untrained pretrained head 41.0%.
  Probe vs head: +4.9 points, not significant. The drop of about 20 points is the same for languages already in v2 and for new languages.
- **What it means:** the 85% inside v2 was partly specific to v2. Scorers trained on 123 pools transfer to new trajectories at about 60%, still well above
  chance. The linear probe is the best single scorer so far; the head has not beaten it. Nothing here measures the value of context or coding-task success.
- Running: nothing. On hold: the remaining 471 TRAIN decisions, four-variant captures, further fine-tuning. Next practical step: the small test of ranking
  Qwen's own candidates.

## FIXED NVIDIA CALL LOGIC (user, chat, 2026-10-11; applies to every generation and judge run)

1. **Keys:** every `.secrets/nvidia_api_key_<n>.txt` with n >= 2 is in the pool (14 keys on 2026-10-11). Adding a file adds a key. Key values are
   read only inside the send call and never written or printed.
2. **Up to 5 requests in flight per key** (measured 2026-10-11: with 10, each key rejected about half with HTTP 429; `PAR_INFLIGHT = 5` in `tools/nemotron_candidate_pilot.py`; `inflight` in `tools/judge_qual_v1.run_jobs`).
3. **At most 39 sends per key per rolling 60 seconds** (the provider's limit is 40); the window is shared by all of that key's in-flight requests.
4. **A rate-limit response (HTTP 429) cools that key only, for one minute** (longer if the response asks for longer). The request was not processed,
   so it goes back to the queue and another key takes it. The other keys keep sending.
5. **No overall call cap.** A per-run cap in the run's plan still stops a runaway run.
6. **Nothing completed is ever sent again:** a request with a saved response is skipped on every restart.
7. **A request that was in flight when a run was stopped** has a `.started` marker and blocks the restart until a person decides; on a deliberate
   stop the markers are renamed `.interrupted`, listed in an `INTERRUPTED_*.txt` file, and those requests are queued again.
8. **Every send is one ledger line** (time, request id, key alias, status, tokens, seconds).
9. Long calls are streamed (non-streamed replies are cut at 300 seconds). Spend stays Rs 0.

Checked by `scratchpad/test_inflight.py` logic on 2026-10-11 with a fake sender (at 10 in flight per key; the limit is a parameter): no send while a key cools, the 39-per-minute
window holds, every request ends with exactly one saved response.

## FROZEN NEXT PLAN (user, 2026-10-11): qualify, expand, capture, train

Full record: [HANDOFF_2026_10_11_STATUS.md](HANDOFF_2026_10_11_STATUS.md) sections 7-9. **No training-qualified dataset exists yet;
Kaggle capture and full-action training are NO-GO until the dataset is qualified.**

Decisions in force (user, chat, 2026-10-11):
- GLM-5.3 (full reasoning, streamed, 65,536-token allowance) + Gemma 4 31B (streamed) review blinded packets; an alternative is
  admitted only if BOTH accept it. Each judge failed its individual gate and those failures stay on record; the PAIR passed its own
  pre-registered gate on a 28-pool pilot (see below).
- Previously examined canonical TRAIN trajectories MAY be used for training (with trajectory and action-overlap safeguards), never
  for claims of unbiased evaluation.
- Gemma reviews first; GLM reviews only the pools Gemma leaves eligible, with the same packets, plus a sample of Gemma-rejected
  pools and the controls.
- One combined generation batch for fresh + reused decisions, CONDITIONAL on the current review results and audit, and run in stages.
- Fable's recorded action is the imitation positive; alternatives stay UNKNOWN. The pretrained head, the trainer and Qwen are not changed.

Measured so far for hard batch v2 [M] (stages must never be added to the 20 earlier strict pools or the 249 nine-family captures):

| Stage | Count |
|---|---:|
| Fresh TRAIN decisions in the batch | 189 |
| Generation calls sent | 1,512 |
| Structurally valid pools | 129 |
| Technically screened pools (contract, grounding, duplicates, real compilers) | 125 |
| Reviewed by the pair (pilot 28 + remaining 97) | 125 (123 scored; 2 lost to unusable GLM replies) |
| **Strict pools (Fable's action + 2-4 jointly admitted alternatives, one same-tool)** | **54 (28.6% of 189 decisions; 51 trajectories; 125 alternatives)** |

Strict pools by language: Python 22, Go 16, TypeScript 14, PowerShell 2. By Fable's tool: Bash 21, Write 17, compound 8, Edit 7, Read 1.
The pair passed its gate again on the remaining 97 pools (0 of 98 known-bad options accepted by both; Fable's action accepted by both 87%).

**SHORTCUT WARNING TRIGGERED [M] (step A2).** An options-only classifier that never sees the task, fitted on these 54 pools with
trajectory-grouped cross-validation, finds Fable's action 65% of the time (character n-grams) and 76% (word tokens) against 31% chance.
The selectors frozen earlier on Fable-native pools stay near chance here (30-41%) because they were fitted on a different kind of
pool; the earlier "near chance on 16 fresh pools" reading under-tested the shortcut. The cue is authorship style (Fable chains
commands with `&&` in 43% of its actions against 18% of alternatives, and its comment prose differs). Judges screen plausibility,
not style. **Scaling generation is on hold.**

**SHORTCUT ELIMINATION V1: NO-GO [M] (user GO 2026-10-11, HANDOFF section 11).** Conservative deterministic normalisation of how
options are rendered does not remove the cue: the strongest refitted state-blind selector scores 76% on the original rendering, 76%
after presentation normalisation, 72% with comments removed and 68% even when file contents are replaced by a structural summary
(chance 31%). The executable originals were never changed. Matching an option's style against earlier actions in the context is NOT a
shortcut (28%, below chance). Remaining cues are in what the generator does, not how it formats: Fable chains shell steps with `&&`
in 41% of its commands against 1.4% of Nemotron's, Nemotron reads files with `cat`/`head` where Fable uses the Read tool, and file
contents differ by author. **Next: change how candidates are authored (not more normalisation), re-run this same check on a small
fresh set, and only then scale.** The 54 pools are kept as a diagnostic set, not as training data.

**SESSION-CONVENTIONS DIAGNOSTIC: INCONCLUSIVE, PARTIAL REDUCTION [M] (user GO 2026-10-11, HANDOFF section 12).** 40 decisions,
320 calls, one paragraph added to the generator's system message (follow the tools and command habits already visible in the
session); frozen protocols, decoders and runner unchanged. At the screened stage, before any judge call: 28 pools; the word-token
selector falls from 71% (old prompt, 125 pools; 65% on size-matched subsets) to 41%, but the character selector is still 54%
against 24% chance, so the strongest detector sits between "promising" (34%) and "NO-GO" (60%). On Bash decisions the word selector
drops to 37%; Write decisions stay identifiable (67%, n = 6). Alternatives now chain with `&&` in 10% of commands (0.9% before;
Fable 31-48%). No judge calls were made, per the rule that judges run only on a promising result. The new set spans 13 languages and
the old one 4, so part of the difference may be population, not prompt.

**ENLARGED SESSION-CONVENTIONS RUN: NO-GO FOR THE PROMPT FIX [M] (user GO for 1,100 calls, 760 used; HANDOFF section 13).** 95 more
decisions in the same languages as the old batch gave 70 screened pools: strongest refitted detector 61% against 23% chance (word
tokens 59%, character patterns 61%), Bash decisions 77%. Alternatives chain with `&&` in only 8% of commands against Fable's 48%. The
40-decision reading (Bash 37%) was a small-sample and population effect. Two offline follow-ups on the 223 screened pools:
requiring Bash alternatives to match Fable's chaining leaves Bash at 65-76%; and in the 37 pools where Nemotron itself wrote Fable's
exact action (so every option is generator-written) the detectors still score 49-54% against 25% chance. **So part of the
state-blind signal is not authorship at all: Fable's action tends to be the typical, obvious one and the alternatives the unusual
ones. That part cannot be removed by any candidate source; "state-blind at chance" is not a reachable gate.** Pending user decision:
judge the data by whether a head WITH context beats the options-only baseline, not by driving options-only to chance.

**CURRENT STATE, 2026-10-11 (HANDOFF section 14).** The user replaced the near-chance gate with three separate checks (candidate
quality; shortcut diagnostic; context value against options-only, action-prior and linear-probe baselines) and set three evaluation
levels: 1 Fable imitation, 2 realistic candidate selection with Qwen's own proposals, 3 task success with and without the head.
- Review of the 98 conventions-prompt pools: GLM done (74 pools still eligible, 22 ruled out, 2 unusable replies). **Gemma's endpoint
  has timed out on every request since about 00:50 UTC (one-word health checks included), so those 74 pools are "GLM-reviewed,
  Gemma pending" and nothing is admitted on one judge.**
- Hand audit of 30 options from batch v2 (blind to the judges): of 18 jointly admitted, 17 plausible and 1 unclear, none
  implausible; of 4 rejected by both, 3 implausible; of 6 disputed, 5 looked plausible (the both-must-accept rule costs good
  alternatives). One admitted TypeScript option has a likely syntax error that the disabled TypeScript parser check did not catch.
- Frozen: `results/raw/laya-fullaction-set-v1-20261011T082326Z/FROZEN_SET.json` (the 54 strict pools, 51 trajectories, 5 trajectory
  folds; canonical VAL / CAL / TEST untouched).
- Prepared, NOT launched: `results/raw/level12-bundle-20261011T082630Z/` (Level 1: 54 pools x 4 context variants; Level 2: 100
  trajectory-separated decisions, 4 Qwen samples each, Fable's next action not in the prompt). **Launching needs the user's GO.**

Pair pilot (28 pools): known-bad options accepted by both judges 0 of 46; Fable's own action accepted by both 24 of 28 (86%);
near-duplicates flagged 11 of 11; unusable first-attempt replies GLM 0 of 28, Gemma 1 of 28; no truncation. GLM's larger allowance
turned 10 of 10 retested empty answers into complete ones (6 to 37 minutes each).

**STATE AFTER THE JUDGE CHANGE, 2026-10-11 about 13:50 UTC (HANDOFF sections 14.1-14.5; this replaces the bullets above where they differ).**
- Gemma's and Kimi's endpoints stayed down (HTTP 504 at 302 s; cause found: overload of NVIDIA's free hosted endpoints). Seven substitute
  second judges ran the same 28 pilot + 20 hard-control packets under the unchanged pair gate. Only **Meta Muse Glimmer 30B**
  (`muse_max`, max reasoning) passed: 0 of 46 known-bad options accepted by both judges, 0 of 17 premature edits accepted alone, Fable's
  action accepted by both 24 of 28, near-duplicates 10 of 11, no unusable reply. It was never run on the individual 90% qualification
  (GLM and Gemma both failed that one; the pair gate is the applicable gate). DeepSeek V4.1 Flash, Nemotron-3-Ultra, gpt-oss-20b,
  DiffusionGemma, Laguna and Nemotron Nano Omni failed; gpt-oss-120b is retired on NVIDIA (HTTP 410).
- **Scoring bug fixed:** option labels repeat inside a pool and scoring kept one verdict per label. Every verdict is still tied to one option
  by its letter, so nothing was re-reviewed; scoring now uses `origin@index`. 3 of 125 kept alternatives of the 54-pool set had one judge only.
- **Dataset v2 frozen:** `results/raw/laya-fullaction-set-v2-20261011T134826Z/FROZEN_SET.json`: **123 pools, 96 trajectories, 307 alternatives**;
  70 pools reviewed by GLM + Gemma, 53 by GLM + Muse (19 trajectories occur in both cohorts; folds are grouped by trajectory, 25/24/24/25/25).
  Never describe the set as reviewed by one pair; report results per cohort. The 54-pool set v1 and bundle `...082630Z` are superseded.
- Hand audit v2 (24 options, blind): GLM + Gemma cohort 9 of 10 plausible; GLM + Muse cohort 7 of 10 plausible, 1 implausible, 1 unclear,
  1 was Fable's command without `| tail -20`. A deterministic same-as-Fable Bash filter was added (2 options and 2 pools removed).
- Shortcut diagnostic on v2 (warning, not a gate): options-only 67.5% vs 29.6% chance (Gemma cohort 71.4%, Muse cohort 65.1%);
  action-history prior 37.0% vs 29.6%.
- Kaggle bundle rebuilt, **NOT uploaded, NOT launched**: `results/raw/level12-bundle-20261011T134843Z/` (Level 1: 123 pools x 4 context
  variants = 492 captures, none over the capture limits; Level 2: 100 decisions). The Level 2 generation code has still never run.
- Next, each needing the user's GO: (1) launch the Kaggle run; (2) baselines on the captured features; (3) controlled fine-tuning, then the
  head-vs-baseline comparison; (4) Level 2: first whether Qwen's own proposals are useful, after training whether the head ranks them better
  than simple methods; (5) Level 3 agent test.

**STATE AFTER THE KAGGLE RUN, 2026-10-11 about 18:15 UTC (HANDOFF sections 14.6-14.11; replaces the bullets above where they differ).**
- User decisions: dataset v2 stays as frozen (70 GLM + Gemma pools, 53 GLM + Muse pools, no re-review); Kaggle launch approved.
- **Level 1 captured and verified [M]:** `results/raw/level12-run-v2-cancelled-20261011T175252Z/` (`LEVEL1_VERIFY.json`): all 123 pools in the
  four context variants (real, same-trajectory other state, other trajectory, neutral prefix) = 492 captures, hashes equal to the upload,
  shapes and finiteness correct, 10 of 10 in-run validation checks passed. Steps A, B5-B6 and C9-C11 below are DONE for these 123 pools.
- **Level 2 (Qwen's own full-context proposals) NOT done, deferred by the user.** The first runs lost a worker to a CUDA error and then showed
  that the library's `generate` runs out of memory above about 5,000 prompt tokens and costs 0.06-0.27 s per token with the corrected loop;
  100 decisions x 4 samples would need about 24 hours on two T4s. Existing evidence stands in for now: saved Qwen proposals contain Fable's
  exact action in 1.4% and its intent in 17% of 1,616 decisions (cut-down context, single call; a baseline, not a full-context measurement).
  This is step D16, which is not a blocker.
- **Next: D12** baselines on the captured features (options-only, action-history prior, linear probe; trajectory-grouped folds; per judge
  cohort and combined), then **D13** the controlled fine-tuning pilot (needs the user's GO), then D14-D15. Expansion (B7-B8) stays on hold
  until D14 shows whether the head with context beats the baselines.

**FINE-TUNING PILOT RESULT, 2026-10-11 19:33 UTC (HANDOFF 15.9; 123 pools, trajectory-held-out, 3-seed means, chance 29.6%).** Pretrained head
without update 36.6%; options text 62.6%; fine-tuned head adapter-only 77.2%, joint 76.7%; the same two recipes trained WITHOUT any context 76.7%
and 78.3%; linear probe on frozen Qwen option states 84.6% with the real context and 84.6% with none. Paired: joint - adapter -0.5 points (no
difference); head - linear probe about -7.5 points (significant); context adds nothing measurable for any model. By the pre-registered rules: keep
the pretrained layers frozen at this data size, and diagnose the adapter / head before scaling head training. Fine-tuning works mechanically; it has
not beaten a linear scorer and has not shown use of the coding state. Stage 1 (200 more decisions): 137 screened pools under GLM + Muse review;
Ruby, Zsh, Rust and C# now have real parser checks in the sandbox.

### Remaining steps

**A. Finish qualifying hard batch v2 (189 fresh TRAIN decisions)**
1. Finish the GLM review of the remaining 97 screened pools (running; Gemma is done) and score it: the strict pool count for the 189 decisions.
2. Shortcut warning: state-blind baselines (pick-longest, metadata-only, options-only) on the admitted pools; chance + 10 points is a WARNING level, not proof of quality.
3. Hand audit of about 30 options, stratified, including options both judges accepted and difficult cases.
4. Freeze the admitted pools with their evidence: candidate text, context hashes, provenance, both judges' verdicts, technical-check results, exclusion reasons; single-tool and multi-call counts kept separate. Disputed and rejected alternatives stay recorded, never labelled wrong.

**B. Split, then expand**
5. Freeze trajectory boundaries and check how many qualified development pools they can yield. Development uses grouped cross-validation over qualified TRAIN pools (every pool held out once, whole trajectories together); canonical VAL is kept back for later independent evaluation; canonical CAL and TEST stay untouched. No claim is made for a language absent from the evaluation data.
6. Coverage audit (language, tool, multi-call) before scaling and again after.
7. One combined generation plan for the remaining fresh decisions (about 250) and the previously examined TRAIN decisions (about 900, training use approved). Staged: a stratified initial portion that includes the missing languages and multi-call decisions, then the rest only if yield, reviewer controls and the shortcut warning hold.
8. Per batch: objective checks, then Gemma, then GLM only on pools Gemma leaves eligible (same packet and option presentation), plus GLM on a sample of Gemma-rejected pools and on the controls so its measured accuracy is not biased.

**C. Capture**
9. Tokenizer check of every qualified pool on this machine (no truncation; over-limit pools excluded with a reason).
10. One guarded Kaggle T4 capture that starts with an in-run validation sample and aborts on a shape, parity, memory or hash failure. **Needs the user's GO.**
11. Verify every captured record: complete context, shapes, finite values, hashes, option order, compatibility with the pretrained head. The 249 nine-family captures do not substitute for this.

**D. Train and evaluate**
12. Baselines on the same pools and folds: majority / prior, state-blind selectors, a context-text model, and a linear scorer on the same Qwen features (mandatory).
13. Pre-defined training experiment: Qwen frozen, adapter warm-up, then joint training with the 31 pretrained tensors at a lower learning rate; non-chosen-option weight 0.5 against 1.0 chosen on development data only; several seeds. **Needs the user's GO.**
14. Evaluate on unseen trajectories: head against the linear scorer, real against shuffled context, by language, tool and compound decisions.
15. Scale-up decision. If the head adds value: broader training, then calibration on CAL, one read of TEST and the agent evaluation. If not: diagnose the adapter / head representation rather than add data.
16. Later integration check (not a blocker for the dataset): repeat the evaluation with candidates produced by the actual local model; Qwen's saved proposals contained Fable's intent in only 17% of decisions.

## STATUS 2026-10-11: where the Fable imitation work stands

Full record, evidence files and remaining steps: [HANDOFF_2026_10_11_STATUS.md](HANDOFF_2026_10_11_STATUS.md) (PLAN.md section 18).
**No training-qualified dataset exists yet and no production fine-tuning has run.**

- **[M] Mechanics proven end to end:** real Qwen features captured on a Kaggle T4 (249/249, no truncation), run through the pretrained
  head with its 31 tensors unchanged, and a 9-run CPU fine-tuning pilot completed with the existing trainer.
- **[M] The pretrained head has not beaten a linear probe.** On nine-family action routing (249 stratified TRAIN decisions, every
  decision held out once): fine-tuned head 59.0% / macro-F1 0.58, linear probe on the frozen Qwen state 67.1% / 0.66. With shuffled
  context the head falls to 8%, so it does use the state.
- **[M] Concrete-action pools:** 20 strictly qualified pools (14 fresh, 6 development). A scale-up batch for 189 fresh TRAIN decisions
  has been generated (1,512 NVIDIA calls); its pools are not built or reviewed yet.
- **[M] Outside judges:** GLM-5.3 and Gemma 4 work as blinded reviewers but neither passes every pre-registered gate alone (GLM too
  lenient on premature edits, Gemma rejects 14-17% of Fable's own actions); Kimi K3 failed the access test.
- **[M] Ruled out by measurement:** symmetric same-trajectory pools, generated-only pools (candidate recall 12-21%), and removing the
  shortcut by stripping content (options-only stays at 60-68%).
- **Next:** decide the judge pair, build and review the batch-v2 pools, run the shortcut gate, then capture and train with the linear
  probe as a mandatory baseline. Kaggle capture and training need the user's GO.

## FROZEN PLAN (2026-10-10): Fable imitation fine-tuning of the pretrained LAYA head

Supersedes the older plans below where they differ. Full plan, every file produced and the remaining steps:
[HANDOFF_2026_10_10_IMITATION_PLAN.md](HANDOFF_2026_10_10_IMITATION_PLAN.md) (PLAN.md section 17). Status: imitation pools built
(TRAIN 2,047 / VAL 305 / CAL 321 / TEST 573), split proof and leakage audit clean, loss checked with placeholder features;
no real Qwen features, no NVIDIA calls and no fine-tuning yet.

## Current plan (2026-10-03, full detail in [PLAN.md](PLAN.md) section 13)

**Direction change 2026-10-03:** the A5-v3 imitation study and the LOCKED capture are scrapped. The head now trains on real outcomes (task resolved or not) with four targets
(`done`, `value`, `next_move`, `pick`; `escalate` is a rule) from one forward pass per trajectory, and ships only if a three-arm agent test passes (Qwen alone, plus Laya sidecar,
plus internal head; 20-30 seeded-bug tasks; numeric margin still to be set). The backbone stays frozen. Pending written approvals: the SWE-rebench-openhands download, lifting
"no Laya fine-tuning", dropping the per-step paperwork, where the agent runs (Kaggle T4 or local 3050), the pass margin. (`autojev/` and the JDownloader installer were deleted 2026-10-03 with the user's yes.) Existing TRAIN and CAL state files are reused.
Nothing is known yet about head quality. The 2026-10-02 plan below is kept as history.

### Superseded: plan of 2026-10-02 (PLAN.md section 12)

Goal: the best internal Laya-style decision head on frozen `Qwen3-4B-Thinking-2507`, as cheaply and quickly as possible, with a stop rule before every paid step.
Status: CAL captured (569/569, audited). TRAIN (1,986) and LOCKED (2,220) not captured. No head trained, so **nothing is known yet about head quality**; the pilot
risk is that the frozen state adds nothing over candidate/history features for next-action imitation (0.663 vs 0.667 [M]).

| Phase | What | Compute | Stop rule |
|---|---|---|---|
| 0 | resident-weights capture mode; amendment (e) (3-seed selection, GPU-numerics clause, TRAIN-CV screen); dataset metadata scan; Laya-style proper-scoring loss | local, free | scan or amendment fails |
| 1 | 20-decision smoke with all hard gates, real speed, CAL duplicate spot-check, fp16 check | Kaggle T4 free, else rented Ada/4090 | any gate fails or cost over 2x model |
| 2 | capture TRAIN, LOCKED, CAL recapture; registered H0 vs H2 test; one LOCKED read; baselines (Laya models, Qwen3-Reranker-4B) | Kaggle free or Ada | failure routes to Phase 3 |
| 3 | outcome-label probe (SWE-rebench-openhands) on TRAIN grouped CV | free or Ada | lower bound not above 0: stop spending |
| 4 | scale to about 10k outcome-labelled decisions; LoRA plus pointer head; fresh LOCKED-2 | LoRA on RTX 6000 Ada | ship LoRA only if it beats the frozen head |
| 5 | three-arm SWE comparison with and without the head | needs Docker | not budgeted |

**Status 2026-10-03:** Phase 2 capture is COMPLETE and VERIFIED on Kaggle T4 x2: TRAIN 1,986 and CAL 569 (7.19 s and 7.03 s per decision per GPU, 192 min wall), all files present and finite, Kaggle-manifest sha256 2,555 of 2,555 equal, smoke decisions 0-19 bit-identical across sessions, T4 vs 3050 states mean cosine >= 0.9999 (worst vector 0.998742; informational). Per the user's Path A, every role stays on the T4: the state-free text embeddings of the TRAIN+CAL pool texts (12,602) are being recaptured on the T4 (notebook v8, gate 0.9999 on every shard); LOCKED states and the LOCKED-pool texts wait for the user's written approval. **Amendment (e) is REGISTERED** (`configs/a5_v3_spec.toml` [amendment_2026_10_03_e], `configs/thresholds.toml` [amendment_e_thresholds]) before any head is trained. The head selection and the pre-LOCKED screen are running on Kaggle T4 x2 (TRAIN only; timing measured: one cell takes 1,417 s and the run is CPU-bound, GPU utilisation 17%); no head result exists yet and LOCKED has not been evaluated. LOCKED preparation is done locally and nothing is uploaded: a label-stripped bundle passed its self-test, and a freeze-and-hash tool exists; the upload needs the user's written approval (amendment e10). Next steps: PLAN.md 12.7. Head-only benchmarks and external references: `BENCHMARKS.md`.
Budget [E]: Rs 0 for Phases 1-3 if fp16 passes on Kaggle; LoRA about Rs 2,000-3,500; proposed cap Rs 6,000 in two steps (pending). **Pending approvals:** provider, cap and
uploads; downloads over 1 GB (SWE-rebench-openhands, Qwen3-Reranker-4B); the private push; whether to lift the "no Laya fine-tuning" rule. Measured so far: 8,163 tokens and
10.2 s per decision on the RTX 3050, compute-bound; consecutive decisions share only 35.4% of prompt tokens, so KV reuse is rejected (`results/raw/capture-cost-model-*.json`,
`prefix-overlap-train-*.json`). Research survey, papers, Laya weights and repos: [DECISION_HEAD.md](DECISION_HEAD.md) section 13.

## Benchmark increments (one row per major step; full ledger: [results/increments.md](results/increments.md))

The fixed benchmark is the locked A5 set (`data/benchmark/a5-v1.jsonl`, 4,396 test and OOD decisions, read-only, hash in `results/raw/`). The headline
metric is accuracy on **covered** decisions against the best baseline there (state-blind TF-IDF, **0.323**). A step that does not move it is recorded as
such. No head has been trained yet, so every row after the baselines shows no change.

| # | Step | Result [M] | A5 covered accuracy | Delta | Evidence |
|---|---|---|---|---|---|
| 0 | Locked benchmark and baselines | random 0.125, first 0.148, longest text 0.257, TF-IDF 0.398 all / **0.323 covered**; laya-typed baseline 0.228 all / 0.239 covered | 0.323 (best baseline) | baseline | `results/raw/a5-baselines-a5-v1.json`, `a5-typed-baseline-a5-v1.json` |
| G2 | 120B expert-read speed (runtime feasibility, not a head step) | best mode 0.87 s per simulated token, 2.18 GB/s, GO (line 1.0 s); I/O only | unchanged | none | `results/raw/g2-expert-read-20261001-132735.json` |
| 1 | Qwen3-4B-Thinking-2507 downloaded; equivalence | v1 rule **FAIL** (top-1 at near-ties, floor was 0); v2 (registered after seeing v1, reason written) **PASS**: streamed vs resident bit-exact on 12 layers, CPU vs GPU-streamed KL 5.5e-4 to 6.6e-4, cosines at least 0.9998 | 0.323 | none | `equiv-qwen3-4b-20261001-145454.json` (v1), `equiv-v2-qwen3-4b-20261001-145942.json` (v2) |
| 2 | Capture throughput | 1,017.6 tok/s at 1x8192 tokens (GO line 1,000, passes by 1.8%); 1,128.9 at 2x4096; peak VRAM 0.8 GiB; weight-transfer share 8% | 0.323 | none | `capture-qwen3-4b-20261001-150329.json` |
| 3 | Capture plumbing on one real decision | all five gates PASS: repeat bit-identical, plain-forward cosine query 1.0 and options at least 0.99993, spans exact, order independence at least 0.99996, lm_head never run | 0.323 | none | `capture-one-20261001-151152.json` |
| 4 | Pilot manifest and Laya per-row fields | 980 decisions (600 train, 380 cal; cal 20 short of target, recorded), test and OOD excluded; Laya-typed on the pilot rows 0.217 all / **0.178 covered**, all states cut by its 640-token window | 0.323 | none | `pilot-manifest-v1.json` (+ `.sha256`), `laya-fields-v1.sha256` |
| 5 | Pilot capture complete and validated | 980 of 980, 0 skips, depths 12/18/24/30/36, 2.98 GPU hours, mean 10.96 s per decision, 670 tok/s mean; 3 recaptured decisions bit-identical | 0.323 | none | `pilot-capture-validation-20261001-182625.json` |
| 6 | Tier-1 closed-form head (`next_action_kind`, sanity arm) | rule registered first; trained on 600 train rows; depth 24 chosen on the 380 pilot-cal rows (lowest NLL); cal accuracy 0.516 vs label-prior 0.213, delta +0.303, 95% trajectory-cluster CI [+0.233, +0.368] excludes 0. All five depths recorded (cal acc 0.500 to 0.539). Not an A5 measurement. | 0.323 | none | `tier1-next-action-kind-20261001-182958.json` |
| 7 | Tier-2 pointer head on ACTION, **pilot only** (rule registered first, `[tier2_action_pointer]` v1) | trained on 600 train rows, 6-cell grid x 5 seeds, selected depth 24 / D 128 on pilot-cal NLL; temperature from out-of-fold train (T = 1.377). **Pilot-cal accuracy 0.616 all / 0.667 covered (n=120 covered)**, NLL 1.254 (1.235 calibrated, ECE 0.095). Baselines on the same rows: TF-IDF full-train 0.384 / 0.308, TF-IDF data-matched (600 rows) 0.290 / 0.333, first 0.118, random 0.121. **Not an A5 measurement and not yet a claim:** the locked benchmark has not been evaluated, and the shortcut audit (row 8) shows a no-model rule already reaches 0.49 on covered rows. | 0.323 | not measured | `tier2-action-pointer-20261001-183544.json` |
| 8 | Shortcut and leakage controls on the pilot (diagnostic) | label-shuffled training 0.108 (chance, no train-time leak); state-shuffled eval 0.616 to 0.234 (the head uses the state-option interaction); metadata-only 0.258; query-zeroed control is degenerate (identical to metadata-only, invalid); **option-only linear scorer with no query reaches 0.821 / 0.775 covered, above the pointer head**, so the query is not what carries the signal. No-model **most-recent-in-pool rule: 0.492 on cal covered, 0.447 on train covered** (true action median pool rank 5 vs 21 for distractors). Word-overlap rules 0.32 (recency-weighted), 0.16 (novelty). **The A5 bar of 0.323 (candidate-only TF-IDF) is too low: a state-aware no-model baseline already exceeds it.** | 0.323 | none | `tier2-controls-20261001-183813.json`, `tier2-control-option-only-20261001-183902.json`, `shortcut-state-overlap-20261001-184015.json`, `shortcut-pool-rank-20261001-184102.json` |
| 9 | Benchmark integrity gate registered; A5-v1 frozen as diagnostic | strongest no-model baseline on pilot cal covered rows is most-frequent-in-history 0.550 (most-recent 0.492, TF-IDF 0.333, metadata-only 0.258, random 0.127; limit random + 0.10 = 0.227): A5-v1 FAILS | 0.323 (historical) | none | `integrity-gate-A5v1-20261001-184428.json` |
| 10 | A5-v2 rules registered; pool audit on the A5-v1 sampler (3,907 train / 816 cal covered rows) | within-decision AUC of the true action: pool rank 0.212, history frequency 0.762, lexical overlap 0.673; an adversarial MLP over eight trivial features reaches 0.663 on cal covered (random 0.149), equal to the pilot pointer head 0.667 | 0.323 (historical) | none | `pool-audit-A5v1-20261001-184836.json` |
| 11 | Pool feasibility for balanced distractors (registered first) | existing pool cannot support per-decision matching: with 3 matches needed, frequency-only 100% (first-time) / 67.7% (repeat), plus action kind 32.4% / 22.2%, plus rank window 1.1% / 3.0%; bar 70% | 0.323 (historical) | none | `pool-feasibility-20261001-185056.json` |
| 12 | A5-v2 sampler `a5v2-odds-1` (spec `a5_v2_spec.toml`, TRAIN only through the firewall, out-of-fold, 5 trajectory-grouped folds) | FAILS the registered gate, not loosened: feature-only attackers on the generated sets A 0.443, B 0.344 (A5-v1 sets: A 0.652, B 0.524; random 0.149; limit 0.249); four AUC violations remain | 0.323 (historical) | none | `a5v2-train-cv-audit-20261001-194503.json` |
| 13 | A5-v3 specification registered; v1 and v2 frozen with their failures | new question: does the frozen state add decision value BEYOND candidate/history features on identical candidate sets (arms H0, H1, H2, Y, S, R) | 0.323 (historical) | none | `configs/a5_v3_spec.toml` |
| 14 | Spec amendment before anything was scored; CPU power simulation | primary claim now **H2 minus H0_matched** (success: lower 95% bound > 0, estimate >= 0.05, half-width <= 0.04, else INCONCLUSIVE); conservative scenario power 0.955 at 400 clusters for a true effect 0.08, 0.46 to 0.53 for exactly 0.05 | 0.323 (historical) | none | `power-sim-v3-20261001-195023.json` |
| 15 | Fresh A5-v3 data downloaded and split by repository hash | 562 fresh rows (438 dropped for already-used repositories); F-dev 118 trajectories / 87 repositories; LOCKED 444 trajectories / 324 repositories, hashed, read-only, unopened | 0.323 (historical) | none | `fresh-v3-provenance.json`, `fresh-v3-locked.sha256` |
| 16 | F-dev CPU feasibility of the frozen sampler (criteria registered first) | USABLE on fresh data: 2,454 covered F-dev decisions, retention 1.000, covered and repeat shares within 1 point of old train; feature-only attackers on F-dev v2 sets A 0.405 / B 0.324 (old 0.443 / 0.344); frozen odds models saved, sha256 20d5d2b8... | 0.323 (historical) | none | `fdev-feasibility-20261001-200544.json` |
| 17 | v3 training design registered; decision manifests built; full-pool capture pipeline built and smoke-tested (not started) | H0 state-free branch vs primary H2 = frozen H0 + residual state branch on cross-fitted scores; 9 registered cells, selection inside TRAIN, 5 paired seeds at the end. Manifests: TRAIN 1,986 / CAL 569 / LOCKED 2,220 (5 per each of 444 trajectories). Smoke test: all hard capture gates pass (independence >= 0.9999 even at a 79-candidate pool); pilot cross-check worst cosine 0.999945 | 0.323 (historical) | none | `v3-manifests.json`, `v3-pilot-crosscheck-20261001-202311.json` |
| 18 | Sampler unified; v3 trainer built and audited (13/14 checks, 1 pending); independence gate stopped the smoke capture (bf16 noise floor, candidates bit-identical when content changes at fixed shape) | No benchmark change. Gate threshold unchanged; amendment awaits the user. Long capture not started. See results/increments.md row 18. |
| 19 | Independence gate amended (content-swap bit-identity + numerical sanity); NONE and early-stopping locks; trainer audit 16/16 | No benchmark change. Capture estimate now roughly 20-24 h (extrapolated from 10 smoke decisions). Awaiting the user's go for the full capture. See results/increments.md row 19. |
| 20 | v3 CAL capture complete (569/569), audited twice | audit PASS twice, same hash 8717cd11...; Gate A max diff 0.0 (62 targets), Gate B min cosine 0.999906. Incident: two capture processes ran at once (42 ids logged twice); files unique and stable, duplicates' bit-identity unprovable (spot-check planned). | 0.323 (historical) | none | `capture-cal-checkpoint-20261002-020211.json`, `-125613.json`, `capture-interruptions.jsonl` |
| 21 | Cost model and plan update | 8,163 tokens and 10.2 s per decision on the 3050 [M], compute-bound; prompt overlap between consecutive decisions 35.4% [M]; GPU costs are estimates. No benchmark change. | 0.323 (historical) | none | `capture-cost-model-20261002-205000.json`, `prefix-overlap-train-20261002-205012.json` |

> **A5-v2 and A5-v3 (2026-10-02):** A5-v2 FAILED the registered gate and was not loosened. Some history/candidate predictability is legitimate behavioural information, so `configs/a5_v3_spec.toml` registers a new question: does the frozen state add value beyond a state-free candidate/history baseline trained on the same rows (primary claim H2 minus H0_matched, minimum effect +0.05, locked set of fresh repository-disjoint trajectories). Fresh data is downloaded and the locked part is closed (row 15). F-dev feasibility passed (row 16). Training design is registered (row 17). Next: the full-pool capture (about 14 to 15 GPU hours, an estimate).

> **A5-v1 status (2026-10-02): FROZEN, DIAGNOSTIC ONLY.** It fails the benchmark-integrity gate (`configs/thresholds.toml` `[benchmark_integrity_gate]`): on pilot cal covered rows a no-model
> *most-frequent-in-history* rule scores 0.550 and *most-recent-in-pool* 0.492, against random 0.127 and a limit of 0.227. The 0.323 TF-IDF reference is historical and is not the bar for any
> head claim. No head result on A5-v1 counts as evidence of decision quality. Next: build A5-v2 with rank- and kind-matched distractors, capture each decision's full pool once, pass the gate on
> train/cal rows only, then register the model family and run the locked evaluation.
> Pool audit on 3,907 train / 816 cal covered rows (`results/raw/pool-audit-A5v1-*.json`): eight trivial candidate/history features with no model and no state predict the true action with 0.663
> accuracy (random 0.149), which equals the pilot pointer head's 0.667. The pilot head showed no signal beyond candidate-pool structure.

## What exists

| Item | State |
|---|---|
| `decision_head/head.py` | The trained head: 34.53M parameters at full size, any hidden size, 9 decision kinds including `CODE_RELEVANCE`. |
| `decision_head/closed_form.py` | Closed-form head (AnyJev's method: shrunk LDA or ridge, 5-fold choice, out-of-fold temperature). The no-gradient baseline arm. |
| `decision_head/selfcheck.py` | 22 checks pass **on synthetic data only** (`uv run python -m decision_head.selfcheck`): maths, masking, order-equivariance, scaler, calibration, losses, learning, closed-form (0.956 from 200 rows, ECE 0.014, chance on random labels). Says nothing about any real LLM. |
| `state/schema.py` | A1: candidate, state and teacher-output contracts. NONE enforced, 2 to 16 options, no outcome without execution, head metadata excludes the candidate source. `TeacherOutput` has two exclusive forms (DISTRIBUTION, INDEPENDENT_SCORES) and a lane registry. |
| `policy/hard_rules.py` | A2: deterministic hard policy and completion invariants. Pattern lists, so they catch what they list; extend, never loosen. |
| `policy/selfcheck.py` | 8 table-driven checks (`uv run python -m policy.selfcheck`): 23 unsafe candidates blocked, 8 safe allowed, lanes name real kinds, teacher forms validated. |
| `teachers/laya_teachers.py`, `teachers/selfcheck.py` | A3, three of five teachers: the judge (COMPLETION), the conductor (CONTINUE, ESCALATE effort) and laya-code (CODE_RELEVANCE: independent per-chunk scores, no softmax, truncation explicit and recorded). Offline, sha256-pinned, out-of-lane returns `applicable=False`. 18 checks pass on the real checkpoints; run with `laya-audit/.venv/Scripts/python.exe -m teachers.selfcheck` from the project root. **A3 is closed:** teachers are the judge, the conductor and laya-code; `teachers/baselines.py` holds `laya-typed` as a BASELINE adapter (never a training target, enforced by `usable_as_target`); cortex-1 is rejected. |
| `capture/render.py`, `capture/capture.py` | Capture records (`capture-record-v1`, frozen): shared prefix plus one independent suffix per candidate, block-masked packed forward, depths 12/18/24/30/36, lm_head never run. Model-independent. |
| `decision_head/pilot_data.py` | Loads the pilot capture into head tensors; asserts the frozen manifest, train and cal only. |
| `compiler/pilot.py`, `configs/pilot_manifest_v1.toml` | The pilot sampling rule (registered before sampling) and the hashed manifest. |
| `configs/thresholds.toml` | Every gate threshold, registered before the run it governs (G2, equivalence v1 and v2, capture throughput, capture-one-record). |
| `tools/` | `verify_hf_download.py`, `eval_laya_typed_decisions.py`, `_manifest.py`, plus the equivalence, throughput, capture, annotation and validation runners. Outputs are create-only. |
| `configs/models.yaml` | Pinned revisions and sha256 for the models we hold. |
| `results/raw/` | Row-level evidence: G0 CUDA pass, download verifications, Laya evaluations. |
| `DECISION_HEAD.md` | Survey of about 35 repos and cards, 16 lessons, parameter inventory, ablation plan, risks, GitHub cross-check. |

## What has been shown

- **G0 passed [M]:** torch 2.14.0+cu126 sees the RTX 3050 6GB.
- **Models on disk, verified [M]:** gpt-oss-120b (15 shards) and gpt-oss-20b-heretic (9 shards) are complete and sha256-verified, both untouched. The five Laya-family teachers are downloaded and verified (`laya-typed-decisions`, `laya-conductor`, `laya-stop-completion-judge`, `laya-code`, `cortex-1-large`).
- **Laya reproduced [M]:** `laya-typed-decisions` scores 0.771 on all 2,000 typed-decisions test decisions against 0.766 published. The conductor scores 0.340 on the same test (random 0.318): each specialist is useful only inside its own lane.
- **Laya's act head is dead [M]:** `act_probability` is exactly 1.0 on every answer. Our abstain/escalate gate must beat plain confidence to ship.
- **Teachers behave as their cards say [M, sanity only]:** conductor 6/6 printed routing examples; judge 0.75 vs 0.05 for announced next step vs direct answer; laya-code relevant chunk 0.43 vs unrelated 0.16, and a chunk scores 0.432 alone and 0.427 in a batch of 5.
- **Frozen-decoder head-only has precedent [A]:** AnyJev (Nokia) fits a closed-form head at about 2/3 depth: Qwen3-4B reaches 0.786 on typed-decisions with 300 labels per question. Heads there do not transfer across questions; our shared head over variable candidate lists is the unproven part.
- **Datasets by recorded outcome [M]:** R2E-Gym SFT has no outcome flag; SWE-rebench-openhands (`resolved`, `exit_status`) and SWE-smith (`resolved`) do. No Docker here, so execution-based checks rely on recorded results.
- **Machine [M]:** AVX-512 torch CPU kernels crash on this CPU (use `ATEN_CPU_CAPABILITY=avx2`); the Hub CLI exits 0 after a `MemoryError`, so downloads are size-checked; D: has about 33 GB free.

- **State builder built and tested [M]** (`compiler/state_builder.py`, `compiler/selfcheck.py`, 17 checks pass): twin trajectories with different labels and futures give identical state, hash and cut manifest at three budgets; the label event and future never enter and `events[t:]` is never read (also true on 1,264 real decisions); episode fields, unknown roles and missing keys are rejected; cuts follow a fixed declared order with visible markers; overflow raises. Budget needs a decision: on 2,147 real decisions the share needing the last-resort core trimming (issue and latest events cut) is 33.5% at 8,000 chars, 3.8% at 12,000, 1.4% at 16,000 and 24,000, 0% at 32,000; oldest-event dropping is needed for 59% to 78% of decisions up to 16,000 chars. The budget is fixed before compilation, not tuned on results.

- **Compiler stage 1 built and run on the 300-row pilot sample [M]** (`compiler/`: state builder, `candidates.py`, `tiers.py`, `split.py`, `run.py`; `configs/compile.toml` fixes the 32,000-character state budget; `uv run python -m compiler.selfcheck` all checks pass). Output `data/decisions/action-swerebench-sample-v1.jsonl` (not in git): 18,894 ACTION records, none skipped, 42 MB; splits by repository train 11,941 / cal 2,557 / test 2,231 / OOD 2,165 (142 / 32 / 29 / 27 repositories); tiers STRONG 706, WEAK 7,138, UNKNOWN 10,807, NEGATIVE 243; cut stage none 2,977, caps 9,747, drop 6,168, core 2. Reports and the split manifest hash are in `results/raw/` (`compile-report-*`, `split-manifest-*`). Known shortcuts to quantify in A5: the true action is the longest candidate text in 24.7% of decisions (chance about 14.9%), and 25.7% of view/edit labels name a file that never appears in the state. Position of the true action is uniform (chi-square 84.6, df 90). **Candidate coverage [M]: only 6,118 of 18,894 true actions (32.4%) are in the pool a state-only generator could propose** (the rest are novel files or commands), so A5 reports accuracy next to coverage. Tier use, the NONE-position axis (first, middle, last), shuffling that includes NONE, and the 12k/16k/24k/32k budget ablation are pre-registered in ``configs/compile.toml``.

- **A5 locked benchmark and baselines [M]** (`compiler/benchmark.py`, `compiler/baselines.py`). `data/benchmark/a5-v1.jsonl`: 4,396 decisions (test 2,231 plus OOD 2,165), read-only, sha256 `f385e40b5f7a5e0b...` in `results/raw/a5-benchmark-a5-v1.sha256`, with every pre-registered variant (shuffle including NONE, NONE first/middle/last, +3/+6/+10 distractors, state hashes at 12k/16k/24k/32k). Baselines in `results/raw/a5-baselines-a5-v1.json`, accuracy = agreement with the trajectory action: random 0.125, first 0.148 (0.000 when NONE is first), last item 0.150, longest text 0.257, **TF-IDF on candidate text only 0.398**. Coverage of true actions by a state-only generator: 31.7% (test), 31.8% (OOD). **The state-blind TF-IDF model reaches 0.398, over 3x random: shortcut signal in the candidates.** It is stronger on uncovered decisions (0.433, the true action is the one candidate not drawn from the pool) than on covered ones (0.323, still 2.6x random; longest-text falls to 0.097). So the primary head metric is accuracy on **covered** decisions, per evidence tier, against the best baseline there (TF-IDF 0.323), with uncovered decisions reported separately; end-to-end is covered accuracy times coverage. Head claims need a paired-bootstrap interval that excludes zero against the best baseline.

- **Step 1 of the A6/A7 plan [M]:** baselines now save per-decision predictions (`results/raw/a5-baselines-a5-v1-perdecision.json`, 1.8 MB) for paired bootstrap; the rerun reproduces the earlier TF-IDF figure exactly (0.3983). The `laya-typed` baseline over all eight axes is run by `compiler/typed_baseline.py` (state rebuilt tail-first at 2,400 characters or the smallest budget whose core fits; any remaining explicit cut is counted); results in `results/raw/a5-typed-baseline-a5-v1.json`.

- **Qwen3-4B equivalence [M]:** rule v1 failed as written (top-1 agreement 0.961 to 0.992 against a 0.99 cap; every disagreement at a reference top-2 gap of at most 0.125; the noise floor was exactly 0, so the rule reduced to absolute caps). v2 was registered afterwards with the reason written; v1 stays recorded as FAIL. v2 passes: streamed equals resident exactly on 12 layers, and CPU vs GPU-streamed KL is 5.5e-4 to 6.6e-4.
- **Capture [M]:** 1,017.6 tok/s on 8,192-token sequences (1.8% above the 1,000 GO line); the real 980-decision run averaged 670 tok/s (longer states, thermal throttling at 86 to 87 C). 2.98 GPU hours, 0 skips.
- **Pilot capture validated [M]:** 980 of 980 files equal the manifest in id and order; all five depths consistent, finite, hidden size 2,560; 3 recaptured decisions bit-identical; lm_head never run. Hidden-state norms grow with depth (query mean 34 at layer 12, 658 at layer 36), so heads must normalise per layer.
- **Laya on the pilot rows [M]:** laya-typed scores 0.217 on all 980 and 0.178 on the 449 covered rows, below the TF-IDF 0.323; all 980 states exceed its 640-token window. It stays a baseline for ACTION.
- **120B read speed [M]:** G2 passed at 0.87 s per simulated token (I/O only).
- **Audit (2026-10-01):** splits are repository-disjoint, no pilot decision or trajectory is in the locked benchmark, all hash sidecars verify, all 33 project `.py` files compile. Open items: RB0 env manifest and experiment id not recorded; `CLAUDE.md`, `PLAN.md` and `configs/p0.toml` still name gpt-oss as primary and heretic as first backbone; `configs/models.yaml` has no Qwen3-4B entry; `autojev/` holds 16.8 GB of dead partial downloads.

## Not done

Evaluation of any head on the locked benchmark, final calibration on a larger frozen calibration split, a recency/pool-rank baseline on the locked benchmark, CONTINUE and COMPLETION decision compilation, and the in-lane teacher outputs for those kinds. **Nothing is known yet about how well the head works on real states.** The benchmark has not moved from 0.323.

## Order (user, 2026-10-01)

Track A first, no model touched: A3 (finish the teachers) then A4 (compiler) then A5 (locked benchmark) then A6 (baselines). Then Track B on heretic: capture at a depth chosen by a sweep, closed-form heads for fixed-layout decisions, then the pointer head for variable candidate lists.

## Next

**Superseded by the current plan above (PLAN.md section 12): Phase 0 first.** The older list below is kept as history.

1. Tier-1 closed-form head on `next_action_kind` at all five depths: fit on train only, pick depth on the 380 pilot calibration rows (pilot-only, not the final calibration), report against the baselines.
2. Tier-2 set-attention pointer head on ACTION (about 34M parameters, frozen backbone, 600 train decisions: strong regularisation and early stopping), compared with TF-IDF 0.323 covered by a paired bootstrap on the locked benchmark. That needs the benchmark decisions captured too (up to about 15 GPU hours for 4,396).
3. After each step, rerun the locked benchmark and add a row to the increments table above and to `results/increments.md`.

## Open decisions for the user

**Current (2026-10-03):** the five pending approvals in PLAN.md 13.6. The lists below are from earlier plans.

**Plan approvals (2026-10-02):** (1) provider, cap (proposed Rs 6,000) and upload scope for Kaggle/Colab or a rental; (2) downloads over 1 GB: SWE-rebench-openhands, Qwen3-Reranker-4B; (3) lift the "no Laya fine-tuning" rule (only a free side track)? (4) questions for AIC: billing granularity, billing while stopped, disk, Docker, GST. CLAUDE.md, PLAN.md, `configs/p0.toml` and `configs/models.yaml` were updated for the Qwen3-4B backbone on 2026-10-02.

Older items: `autojev/` and the JDownloader installer were deleted 2026-10-03 (done). Still open: the partial downloads in the model caches? Record the RB0 env manifest and experiment id now? Update `CLAUDE.md`, `PLAN.md` and `configs/p0.toml` to the Qwen3-4B experiment backbone (they are frozen documents, so this needs your approval)? Add a pinned Qwen3-4B entry to `configs/models.yaml`? Commit the new files and push?







**Status 2026-10-04:** a new head from Coding_Decision_Agent on frozen Qwen3-4B states is planned (PLAN.md 14, DECISION_HEAD.md 20); nothing trained yet. State capture on Kaggle: 45,500 of 46,826 decisions. The synthetic hard negatives are flagged by the shortcut check, so real wrong options were researched and a tiered builder written (PLAN.md 15); its counts have not been run on the full data and recapture waits for them.
