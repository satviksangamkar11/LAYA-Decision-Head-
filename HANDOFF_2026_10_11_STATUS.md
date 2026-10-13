# Status handoff, 2026-10-11: Fable imitation data for the pretrained LAYA head

Goal (unchanged): fine-tune the EXISTING pretrained LAYA `Coding_Decision_Agent` head (31 pretrained tensors + a new adapter over
frozen Qwen3-4B-Thinking-2507 states) to choose coding actions like Fable. Fable's recorded action is the imitation target; other
options are `unknown`, never "proven wrong".

State in one line: **no training-qualified dataset exists yet and no production fine-tuning has run.** Pipeline mechanics are proven
end to end; the open problem is enough qualified same-state alternatives, plus evidence that the pretrained head beats a linear probe.

All figures below are measured [M] in the named file unless marked estimate [E]. Times are UTC.

## 1. What was done on 2026-10-10 and 2026-10-11

| # | Step | Result | Evidence (`results/raw/`) |
|---|---|---|---|
| 1 | Hard-pool review of 37 pools (two blinded Claude passes, full context, controls) | Double-reviewed plausibility of new Nemotron options 43/72 = 59.7%, below the 60% gate. Controls 100%. Agreement 87%, kappa 0.69 | `hard-pools-20261010T150521Z/build-20261010T152617Z/review-v2-20261010T154410Z/REVIEW_RESULT-20261010T154850Z.json` |
| 2 | Offline checkpoint (Codex), rechecked here | 22 admitted pools (16 fresh, 6 development); 5 fresh pools defeat all three state-blind baselines; every recomputed number matched | `hard-checkpoint-codex-20261010-r3/` |
| 3 | Strict qualification of the 22 pools | 20 survive (14 fresh, 6 development) with double-supported, technically clean alternatives; real `javac` / PowerShell 7 / Python / bash parsers confirm 2 broken Java options | `hard-qualify-v1-20261010T163636Z/`, `hard-qualify-v2-syntax-20261010T171340Z/` |
| 4 | Symmetric same-trajectory pools (30 trajectories, 132 decisions) | NO-GO. 35/132 = 26.5% keep two available alternatives (target 30%); an availability rule scores 57.4% against 28.0% chance; in the sandbox 5 of 11 "available" actions failed | `symmetric-pools-v1-20261010T165430Z/` |
| 5 | Candidate-recall audit of saved Nemotron outputs | Exact decision-level recall: v3t@5 12-21%, alt4@4 8%; Write, Edit and Reply 0%; recall does not grow with k; 3 generated-only pools from 69 decisions | `candidate-recall-audit-v1-20261010T171752Z/` |
| 6 | Intent-level checks | Removing content does not remove the shortcut: options-only 78% -> 60% (Qwen alternatives), 68% -> 68% (Fable-native), chance 26-31%. Qwen's saved proposals: exact recall@4 1.4%, intent recall@4 17.0% | `intent-level-checks-v1-20261010T172625Z/` |
| 7 | Hierarchical menu counts on 1,616 eligible TRAIN decisions | Nine families cover everything (Web 2.2%); Read 99.3% / Edit 100% / Write 53.3% of targets already named in context; step + previous-family prior 57.9%; CPU context proxy 64.1% | `hier-menu-checks-v1-20261010T173609Z/` |
| 8 | Nine-option head compatibility (CPU, real smoke features) | Head passes every mechanics check; the frozen trainer and capture cap pools at 8, so nine options got its own versioned module | `menu-head-compat-v1-20261010T174303Z.json` |
| 9 | Nine-family capture pilot on Kaggle T4 | 249/249 captured, hq 3x2560, ho 9x3x2560, fp16, no truncation, parity 10/10, all verification checks pass | `menu-pilot-v1-20261010T175205Z/VERIFY.json` |
| 10 | Controlled fine-tuning pilot (CPU, existing trainer, 9 runs) | 4-fold held-out: fine-tuned head 59.0% / macro-F1 0.58; linear probe on frozen Qwen state 67.1% / 0.66; context-text 45.8% / 0.44. Shuffled context drops the head to 8%. The head does not beat the probe | `menu-finetune-pilot-v1-20261010T183815Z/REPORT.json` |
| 11 | Hard batch v2: generation for 189 fresh TRAIN decisions (user lifted the call cap) | 1,512 sends finished 20:52 (toolcond 567: 562 HTTP 200, 5 timeouts; v3t 945). Pools NOT built yet | `hard-batch-v2-20261010T190132Z/`, `hard-gen2-toolcond-20261010T190132Z/`, `hard-gen2-v3t-20261010T190132Z/` |
| 12 | Track 2: outside blinded judges on the 37 full-context packets + 20 hard-control packets | No configuration passes every pre-registered gate (table in section 2). Kimi failed the access test | `judge-qual-v1-20261010T192150Z/QUAL_REPORT-20261010T211811Z.json` |

Rows 9 and 10 are ACTION-FAMILY ROUTING (stage 1 of a hierarchical selector), not concrete-action imitation. They are not comparable
with any concrete-action number.

## 2. Track 2 result: outside judges (GLM-5.3, Gemma 4 31B, Kimi K3 on NVIDIA endpoints)

Gates fixed before any call (`judge-qual-v1-20261010T192150Z/PREREG.json`): each control type >= 90% on each pass; unusable
responses < 5%; no truncation (the judge must report the packet's last line number).

| Configuration | Fable's action accepted (pass1 / pass2) | Off-task rejected | Near-duplicate flagged | Hard premature edits rejected | Unusable | Same verdict when re-ordered (kappa) | Median seconds | Gates |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3, full reasoning, streamed | 35/35, 32/32 | 33/33, 31/31 | 13/13, 13/13 | 13/17 = 76% | 8.4% | 0.79 | 340-490 | FAIL (hard controls, unusable) |
| GLM-5.3, low reasoning | 36/36, 32/32 | 33/33, 29/29 | 13/13, 11/11 | 11/17 = 65% | 4.6% | 0.66 | 45-55 | FAIL (hard controls) |
| Gemma 4 31B, streamed | 32/37 = 86%, 29/35 = 83% | 34/34, 33/33 | 13/13, 12/13 | 17/17 = 100% | 1.5% | 0.46 | 64-105 | FAIL (rejects Fable's own action too often) |
| Kimi K3 (three settings) | not reached | | | | 8 of 12 access calls failed | | | FAIL (access) |

Findings:
- The old controls are too easy (every judge 100%); the new hard controls separate the judges.
- The two usable judges fail in opposite directions: GLM is lenient (accepts 24-35% of edits that cannot be applied yet), Gemma is
  strict (rejects 14-17% of Fable's own actions). With a both-must-accept rule the pair lets 0 of the gated negatives through.
- Full reasoning matters for broken code: GLM at full reasoning rejects both `javac`-rejected Java options; at low reasoning it accepts both.
- GLM full-reasoning failures are fixable: 11 answers ran out of the 32,768-token budget during reasoning (empty answer), 3 dropped the last brace.
- Temperature 0 gives the same verdict as the documented temperature on 93-96% of options.
- Agreement between GLM (full) and Gemma: 80%, kappa 0.50; both accept 36 of 63 new Nemotron options (57%).
- Not a gate and not used: agreement with the earlier Claude reviews.

Throughput facts measured on these endpoints: non-streamed requests are cut at 300 s (HTTP 504), so long judge calls must be streamed;
more requests in flight per key raises latency (Gemma lost 21 of 131 calls to the 300 s limit at 4 in flight per key, non-streamed);
a rate-limit response followed by a 60 s key cool-down recovered every time; Kimi rejects any `top_p` other than 0.95.

## 3. User decisions recorded in this period

- 2026-10-10: nine action families (Web separate) for the hierarchical menu; versioned nine-option configuration; existing Compound
  mapping kept (a single Bash command is never labelled by its last segment); `reply` mapped to `text_reply`; 250-decision capture pilot.
- 2026-10-11: controlled fine-tuning pilot on the 249 decisions; no larger capture yet.
- 2026-10-11: the NVIDIA call cap (800, then 1,000) applied to the first hard-pool experiment only; there is no overall cap now.
  Go-ahead for about 2,000 calls on 200 fresh TRAIN decisions (1,512 used on 189).
- 2026-10-11: outside judges approved (Track 2); "use max quality"; on any rate-limit response the key cools one minute and resumes;
  the runner should push every key to its limit while staying reliable (several requests in flight per key).

## 4. Remaining steps to a training-qualified concrete-action dataset

1. **Decide the judge configuration (user decision).** No single judge passed. Proposal: Gemma (streamed) + GLM full reasoning, both
   must accept, with a pair-level gate written down before use; raise GLM's token budget so reasoning cannot crowd out the answer.
2. **Build pools from hard batch v2** (offline): existing v1.2.5 rules, then objective checks that judges cannot override
   (tool contract, grounding incl. edit anchors, duplicates, real-compiler syntax in the WSL sandbox).
3. **Two-judge blind review** of those pools with the hard controls mixed in; disputes dropped or sent to a third model.
4. **Hand audit of about 30 options**, including ones both judges accepted.
5. **Shortcut gate**: state-blind baselines within 10 points of chance on at least 50 admitted pools.
6. **Generate for the remaining fresh decisions** (about 250; 440 never-examined TRAIN tool decisions existed, 189 used). Reaching
   300-600 pools [E] needs the roughly 900 decisions of trajectories examined by earlier pilots (training only; user decision).
7. **Freeze a trajectory-level TRAIN / DEV split; capture on the Kaggle T4** (needs the user's GO).
8. **Controlled training with the linear probe as a mandatory baseline** (needs the user's GO), then the agent evaluation of the frozen plan.

Open in parallel (no calls needed): diagnose why the pretrained head trails the linear probe on the 249 family-routing decisions
(checkpoint selection by macro-F1, longer adapter-only phase, richer option texts). A larger family capture stays NO-GO until then.

## 5. New code and configuration (none of it changes a frozen contract)

- `decision_head/fable_menu_v1.py`, `tests/test_fable_menu_v1.py`, `configs/laya_action_family_menu_v1.json` (nine-family menu, pilot status)
- `kaggle/menu_pilot_t4.py`, `tools/menu_pilot_bundle.py`, `tools/menu_pilot_verify.py`, `tools/menu_finetune_pilot.py`, `tools/menu_head_compat_v1.py`
- `tools/hard_qualify_v1.py`, `tools/hard_qualify_v2_syntax.py`, `tools/symmetric_pools_v1.py`, `tools/candidate_recall_audit_v1.py`,
  `tools/intent_level_checks_v1.py`, `tools/hier_menu_checks_v1.py`
- `tools/hard_batch_v2.py` (scale-up generation), `tools/judge_qual_v1.py` (outside judges + max-throughput runner)

Unchanged: the pretrained checkpoint (sha256 `e316f15e...2810a`), `decision_head/fable_train.py`, `fable_features.py`, `fable_pool.py`,
`head.py`, `tools/fable_capture.py`, the v1.2.x imitation builder, canonical TRAIN / VAL / CAL / TEST. Canonical VAL, CAL and TEST were
never read. Nothing was committed to git.

## 6. Corrections to earlier statements made in chat

- WSL has JDK 26.0.1 and PowerShell 7.6.3 under `/opt/toolchains` (not on the default PATH); an earlier "no JDK" statement was wrong.
- Three result folders were cited with wrong timestamps; the correct names are the ones in the table above.
- The 59-64% baselines of row 7 are nine-family numbers on the natural mix; the pilot of rows 9-10 is stratified, so they do not transfer.

## 7. Frozen next plan (user, 2026-10-11): qualify the batch-v2 pools with the GLM + Gemma pair

Decision: **CONDITIONAL GO** for GLM-5.3 (full reasoning) + Gemma 4 31B as an EXPERIMENTAL two-judge pair with a both-must-accept
rule. The pair is NOT qualified yet: each judge failed its individual gate, those results stay recorded as failures, and two failed
judges do not make a qualified pair without their own test.

Counts that must never be added together (three different stages): 20 strict concrete-action pools; 1,512 generation calls for 189
fresh TRAIN decisions (14% of about 1,342 eligible tool decisions; none qualified yet); 249 nine-family captures (a separate experiment).

Steps, in order:
1. Build candidate pools from the completed 1,512-call batch.
2. Objective checks BEFORE any LLM review: tool contract, grounding, duplicates, compilers. Judges cannot override them.
3. Test GLM with a larger output allowance; measure whether unusable responses drop (do not assume).
4. Paired-review pilot on 25-30 representative fresh pools with evidence-backed hard controls.
5. Report pair-level false acceptance, false rejection, response failures, positive-control acceptance and strict usable-pool yield,
   separately from the individual gate results.
6. Rejected and unresolved alternatives stay recorded; they are never relabelled as objectively wrong.
7. Only if the pair passes a separately documented quality gate: review the remaining fresh pools. Otherwise report the specific
   failure and the smallest correction.

Fixed rules: Fable's recorded action is the imitation positive; alternatives stay UNKNOWN. No change to the pretrained LAYA head or
trainer. **NO-GO** until the dataset is qualified: full-scale review without pair validation, Kaggle capture, full-action training.
The number to report: of the 189 decisions, how many have Fable's action plus 2-4 technically valid, independently approved
alternatives; plus generated / structurally valid / technically screened / jointly admitted / strict pool counts and the
shortcut-baseline results.

## 8. Update, late 2026-10-11: batch v2 built and screened, pair pilot passed

| Step | Result | Evidence (`results/raw/hard-batch-v2-20261010T190132Z/build-20261010T212540Z/`) |
|---|---|---|
| Pools built from the 1,512-call batch | 129 structurally valid pools of 189 decisions (68%; 43% in the earlier batch); 60 excluded (34 too few candidates, 22 no same-tool alternative, 4 no same-shape alternative). Before the build, 369 candidates rendering identically to another and 253 rendering like Fable's action were collapsed | `BUILD_REPORT.json` |
| Objective checks before any LLM review | 21 of 470 alternatives failed (Go, Python, PowerShell and shell syntax; ungrounded file references); 125 pools keep Fable's action + at least two clean alternatives incl. one same-tool. Two checks were disabled by the calibration rule because Fable's own actions failed them (edit-anchor 41/42, TypeScript parse 12/15) | `SCREEN_REPORT.json`, `SYNTAX.jsonl` |
| GLM output allowance 32,768 -> 65,536 | 10 of 10 retested packets that had returned an empty answer now return a complete one (385 to 2,250 s each); one still running when this was written | `results/raw/judge-qual-v1-20261010T192150Z/` (`glm_max64__*`) |
| Paired-review pilot, 28 screened pools | PAIR GATE PASSED: joint false acceptance 0/46; Fable's action accepted by both 24/28 = 86%; near-duplicates 11/11; first-attempt failures GLM 0/28, Gemma 1/28 (provider overload, answered on one retry); no truncation. Alternatives: 48 jointly admitted, 19 disputed, 9 rejected by both, 7 duplicates of Fable's action. Strict usable pools 12/28 | `pair-pilot-20261010T212807Z/PAIR_REPORT-20261010T220303Z.json` |
| Review of the other 97 screened pools | started after the pair passed; Gemma 94/97 on the first attempt; GLM in progress | `pair-review-rest-20261010T220327Z/` |

The individual gate failures of GLM and Gemma (section 2) stand. The pair result is separate and applies to the both-must-accept rule only.
Parser repairs counted in the reports: GLM sometimes drops `":` after `duplicate_of` or the last brace; Gemma once put a stray key inside `options`.

## 9. User decisions, late 2026-10-11, and the current remaining steps (supersede sections 4 and 7 where they differ)

- CONDITIONAL GO for the GLM + Gemma pair, both must accept; full-scale review only after pair validation (done: pilot passed).
- YES: previously examined canonical TRAIN trajectories may be used for training, with trajectory and action-overlap safeguards.
- YES with conditions: Gemma first, GLM only on eligible survivors; same packet and option presentation; GLM still sampled on
  Gemma-rejected pools and controls; Gemma's rejections of Fable's own action stay visible as a diagnostic.
- Combined fresh + reused generation: CONDITIONAL GO after the 97-pool results and the audit, executed in stages.
- Split: freeze trajectory boundaries now; development by grouped cross-validation; canonical VAL kept for later independent
  evaluation; solve the development-capacity question BEFORE the large batch.
- One Kaggle run is enough (3-5-option complete-action capture already ran on 5 and on 411 pools); tokenizer check first, in-run
  validation sample, abort on failure.
- Chance + 10 points on the shortcut baselines is a warning level, not proof. A failing test command is not by itself a bad action.

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

## 10. Batch v2 final counts and the shortcut warning (steps A1-A2 done, 2026-10-11)

Evidence: `results/raw/hard-batch-v2-20261010T190132Z/build-20261010T212540Z/strict-20261010T224125Z/STRICT_REPORT.json`,
`pair-review-rest-20261010T220327Z/PAIR_REPORT-*.json`.

| Stage | Count |
|---|---:|
| Fresh TRAIN decisions | 189 |
| Structurally valid pools | 129 |
| Technically screened pools | 125 |
| Scored by both judges | 123 (2 lost to unusable GLM replies) |
| Strict pools | 54 (28.6% of decisions), 125 admitted alternatives, 51 trajectories |

Remaining-97 review: pair gate passed again (joint false acceptance 0/98; Fable's action accepted by both 82/94 = 87%; near-duplicates
33/33; first-attempt failures 3.1% each judge; no truncation). Alternatives: 172 jointly admitted, 65 disputed, 25 rejected by both,
18 duplicates of Fable's action. Yield of screened pools: single-tool 46/101, compound 8/24; Python 22/53, Go 16/33, TypeScript 14/33,
PowerShell 2/6; Bash 21/44, Write 17/36, Edit 7/18, Read 1/3.

Shortcut warning (chance 30.9%, 54 pools):

| State-blind selector | Top-1 | Above chance |
|---|---:|---:|
| Pick longest | 31.5% | +0.6 |
| Metadata-only, frozen earlier on Fable-native pools | 40.7% | +9.9 |
| Options-only text, frozen earlier on Fable-native pools | 29.6% | -1.3 |
| Options-only text, refitted on these pools (grouped 5-fold, character n-grams) | 64.8% | +33.9 (WARNING) |
| Options-only word tokens, refitted (grouped 5-fold) | 75.9% | +45.0 (WARNING) |
| Eleven hand-made surface features, refitted | 49.4% | +18.5 (WARNING) |

Reading: the pair review qualifies plausibility and grounding, and the yield assumption (28%) held. It does not remove the authorship
gap: Fable's action is the only option Fable wrote, and a selector fitted on this pool type recognises it without the task. The frozen
selectors were fitted on a different pool type and do not transfer, so their near-chance scores are not evidence of absence.
By Fable's tool the word-token selector scores Bash 86%, Write 82%, compound 62%, Edit 57%.

Not done: step A3 (hand audit) and A4 (freeze). The combined generation batch is on hold: its condition was that the shortcut
warning holds, and it did not.

## 11. Shortcut elimination v1: NO-GO (user GO 2026-10-11)

Evidence: `results/raw/hard-batch-v2-20261010T190132Z/build-20261010T212540Z/strict-20261010T224125Z/shortcut-elim-v1-20261010T224720Z/`
(`PREREG.json` written first; `FREEZE.json` holds the hashes of the 54 strict pools and the pair reports, which were only read).

Question: after conservative, deterministic normalisation of the RENDERING of every option (Fable's included), can a selector that
never sees the task still find Fable's action? Detectors were refitted on each rendering with trajectory-grouped 5-fold cross-validation.

| Rendering (model-input text only; executable originals untouched) | Pools | Character n-grams | Word tokens | Surface features | Rule "has &&" |
|---|---:|---:|---:|---:|---:|
| R0 original | 54 | 66.7% | 75.9% (65-87) | 44.4% | 48.5% |
| R1 presentation (Unicode punctuation to ASCII, trailing whitespace, blank-line runs) | 54 | 66.7% | 75.9% | 33.3% | 48.5% |
| R2 = R1 + full-line and block comments removed from contents | 54 | 63.0% | 72.2% | 40.7% | 48.5% |
| R3 = R1 + Write/Edit contents replaced by a structural summary | 50 | 63.0% | 68.0% | 37.0% | 49.1% |

Chance is 30.9% (31.2% for R3). Verdict by the pre-registered rule: **NO-GO** (strongest refitted detector on the safe renderings 75.9%).

- Integrity: originals byte-identical (hash check) in every rendering; R1 and R2 keep all 54 pools and all options distinct; R3
  collapses 9 alternatives and loses 4 pools. `&&`, `;`, pipes, arguments and literals were never touched.
- By Fable's tool (word tokens, R0 -> R3): Bash 86% -> 86%; Write 82% -> 51%; Edit 57% -> 70% (n 7 -> 5); compound 63% -> 60%.
  Removing content helps only Write; Bash is untouched because its cue is the command itself.
- Context style matching (no training; similarity of each option to the earlier tool calls in the same context): 27.8% against
  30.9% chance. Matching Fable's style from the history is not a shortcut here.
- Named remaining cues: Fable chains shell steps with `&&` in 26 of its 63 Bash commands (41%) against 6 of 414 decoded Nemotron
  commands (1.4%); the single rule "pick the option containing &&" scores 74% on Bash decisions. Nemotron uses `cat` / `head` / `ls`
  to read where Fable uses the Read tool (10% against 0% of options). Alternatives are multi-call more often (26% against 15%).
  Write contents differ by author.
- A side check was inconclusive: only 8 strict pools belong to decisions where Nemotron independently wrote Fable's exact action
  (48 of 189 decisions did, but few became strict pools).

Smallest change indicated: fix candidate AUTHORSHIP at generation time (the generator should follow the conventions visible in the
session, including command chaining and tool choice), then repeat this exact check on a small fresh confirmation set before any
scaling. More normalisation is not indicated. The hand audit (A3) was not run: it cannot change this verdict.

## 12. Session-conventions generation diagnostic: inconclusive, partial reduction (user GO 2026-10-11)

Evidence: `results/raw/hard-batch-v3-conv-20261010T225509Z/` (`LEDGER.json`, `build-20261010T230858Z/`, `SHORTCUT_SCREENED-*.json`);
scripts `tools/hard_batch_v3_conv.py`, `tools/shortcut_stage_test.py`.

Change tested: one paragraph appended to the system message of every generation request, telling the generator to use the tools and
command habits already visible in the session (its file-reading tool rather than `cat`, chaining dependent steps with `&&` where the
session does). Nothing about Fable's current action is given; the call count is not prescribed. Protocol files, decoders and the
runner are unchanged.

- Selection: 40 decisions never sent to a generator, one per trajectory, trajectory-disjoint from batch v2. No never-examined
  trajectory was left (batch v2 used all 129), so these come from trajectories earlier pilots touched (reuse approved); 13 languages.
- Generator never saw the current action: the existing leak guard blocked 0 of 80 requests; of 106 argument texts of the recorded
  actions, 14 also occur in a request, all of them text already present earlier in the session (4 file paths, 6 edit anchors read
  before the edit, 4 commands run before).
- Sends: 320 (toolcond 120 answered; v3t 185 answered, 15 lost to rate-limit responses that the run's cap counted as sends).
- Pools: 30 structurally valid, 28 technically screened (11 of 107 alternatives failed an objective check).

Shortcut test at the screened stage (same detectors, refitted, trajectory-grouped 5-fold):

| Set | Pools | Chance | Word tokens | Character n-grams | Surface features |
|---|---:|---:|---:|---:|---:|
| Old prompt, batch v2 | 125 | 22.8% | 70.8% (64-78) | 63.2% (55-72) | 34.0% |
| Old prompt, 200 size-matched subsets of 28 (mean; 5th-95th percentile) | 28 | 22.9% | 65.4% (50-80) | 59.0% (46-75) | 41.3% |
| Session-conventions prompt | 28 | 24.1% | 41.1% (23-59) | 53.6% (36-71) | 39.3% |

Verdict by the thresholds fixed beforehand: strongest refitted detector 53.6% -> **inconclusive** (promising needs <= 34%, NO-GO is
>= 60%). The word-token result is below the 5th percentile of the size-matched old subsets, so that part of the reduction is unlikely
to be a sample-size effect; the character result is inside the old range.

- Cues: alternatives chained with `&&` 0.9% -> 10% (Fable 48% / 31% in the two sets); alternatives reading with `cat`/`head`/`ls`
  21% -> 11% (Fable 0%); multi-call share unchanged (27% against 18-19%).
- By Fable's tool (word tokens, new set): Bash 37% (n 15), Write 67% (n 6), compound 20% (n 5).
- Confound: the new set has 13 languages, the old set 4 (Python, Go, TypeScript, PowerShell).
- No judge calls were made: the rule was to call the judges only on a promising result.

## 13. Enlarged session-conventions run and two follow-ups: prompt fix NO-GO; the state-blind signal is partly a typicality prior

Evidence: `results/raw/hard-batch-v3-conv-20261010T231422Z/` (`LEDGER.json`, `build-20261010T235616Z/`, `SHORTCUT_SCREENED-*.json`,
`CHAIN_MATCH_WHATIF-20261011T000033Z.json`).

Run: user approved 1,100 calls; 95 never-generated decisions were available in the batch-v2 trajectories (Python 33, Go 32,
TypeScript 30; at most 2 per trajectory), 760 calls planned, 756 answered (rate-limit responses retried under the larger cap).
71 structurally valid, 70 technically screened pools (69 single-tool).

| Set (screened stage, refitted detectors, grouped 5-fold) | Pools | Chance | Word tokens | Character n-grams | Surface |
|---|---:|---:|---:|---:|---:|
| Old prompt, batch v2 | 125 | 22.8% | 70.8% | 63.2% | 34.0% |
| Old prompt, 200 subsets matched to the new set's size and tool mix (mean; 5th-95th) | 70 | 23.1% | 77.8% (72-84) | 66.1% (61-71) | 50.9% |
| Session-conventions prompt, same languages | 70 | 23.0% | 58.6% (47-69) | 61.4% (51-72) | 55.7% |

Verdict by the fixed thresholds: strongest 61.4% -> **NO-GO**. Word tokens fell below the matched old range, character n-grams did
not. By Fable's tool (word tokens): Bash 77% (n 44), Edit 20% (n 15), Write 25% (n 8). Cues: alternatives chain with `&&` 7.7%
(Fable 47.7%); read with `cat`/`head`/`ls` 15.5% (Fable 4.5%); multi-call 20.5% (Fable 1.4%).

Follow-up 1 (post hoc, offline, all 223 screened pools of the three batches): keep only Bash alternatives with the same `&&`
chaining flag as Fable's command. The single `has &&` rule falls to chance (26%), but the word-token detector stays at 63% overall
and 76% on Bash (65% on the conventions-prompt pools); 46 pools are lost. Chaining is one cue, not the cue.

Follow-up 2 (post hoc, offline): decisions where Nemotron itself produced Fable's exact action (byte-identical rendering): 81 of 324
decisions, 60 by the blind protocol. In the 37 screened pools of those decisions every option is text the generator writes, yet the
detectors score word tokens 54.1% (40-68), character n-grams 48.6% (34-64) against 25.3% chance (surface features 9.5%, below
chance). In the other 186 pools: 70.4% and 65.6% against 22.6%.

Reading: the state-blind signal has two parts. About 15-20 points look like authorship (they disappear when the generator wrote the
positive too). About 25-30 points remain even then: Fable's action is typically the common, expected action for that kind of
moment, and the admitted alternatives are the less usual ones. That is an action prior, the same kind of regularity that gave a
58% step-and-previous-action baseline in the nine-family counts. No candidate source can remove it, including sampling Fable itself.
Caveats: 37 pools, wide intervals; those decisions are a selected, more predictable subset.

Consequence: "state-blind selectors near chance" cannot be the acceptance gate for training data. The usable gate is comparative:
an options-only baseline is fitted on the same pools and folds, and a head with real context must beat it, with the shuffled-context
and other-state-context controls. Authorship can still be monitored separately by comparing same-author pools with the rest.
Not done: no judge calls on the 98 conventions-prompt screened pools; no capture; no training.

## 14. State on 2026-10-11 after the gate change: review, audit, freeze, Kaggle bundle

User decisions (chat, 2026-10-11): the "state-blind near chance" gate is replaced by three separate checks: a candidate-quality
gate, a shortcut DIAGNOSTIC (not a gate), and a context-value gate (the head with real context must beat options-only, a simple
action prior and a linear probe on the same held-out trajectories, with other-state and other-trajectory context controls). The
claim that no generator can remove the residual signal is NOT established (the 37 same-author pools are a selected subset). Three
evaluation levels: 1 imitation, 2 realistic candidate selection with the local model's own proposals (about 100 decisions, action
quality as well as Fable recall, outcomes `UNKNOWN` without reliable execution evidence), 3 task success with and without the head.
One Kaggle session, two separate experiments. No large generation, no large-scale fine-tuning, no claim that LAYA improves coding.

Review of the 98 conventions-prompt screened pools (`.../hard-batch-v3-conv-*/build-*/pair-review-rest-20261011T0048*`):
- Gemma failed first: two jobs at about 40 requests in flight produced 36 timeouts of 42 calls; after stopping them, a quarter of the
  load and then single one-word requests also timed out (HTTP 504 at 302 s) while GLM and Nemotron answered in 1-3 s. The outage
  continued through repeated checks until at least 08:26 UTC. 48 Gemma calls are recorded as failed and need one retry.
- GLM therefore reviewed all 98 first: 96 usable, off-task controls accepted 0/96, Fable's action accepted 94/96, near-duplicates
  flagged 55/57, alternatives accepted 247/331. Pools GLM leaves eligible: 74 (23 + 51); ruled out: 22; unusable: 2
  (`GLM_FIRST.json` in each review folder). Nothing is admitted until Gemma reviews those 74 (+ a sample of the 22).

Hand audit (`.../build-20261010T212540Z/audit-v1/AUDIT_RESULT.json`): 30 options sampled by seed from the 208 reviewed alternatives
of batch v2, judged by this session from excerpts WITHOUT the judges' verdicts, verdicts written before the key was opened.
Jointly admitted 18: 17 plausible, 1 unclear, 0 implausible. Rejected by both 4: 3 implausible, 1 unclear. Disputed 6: 5 plausible,
1 a duplicate of Fable's action in effect. Flagged duplicates 2: 1 implausible (invalid flag), 1 plausible. Findings: no joint false
acceptance seen in this sample; the both-must-accept rule discards alternatives that look fine; one admitted TypeScript Write has a
stray closing brace (the TypeScript parser check was disabled because Fable's own files failed it). Limits: excerpt-based, one
auditor of the same model family as the first reviewers.

Frozen set (`results/raw/laya-fullaction-set-v1-20261011T082326Z/FROZEN_SET.json`): the 54 strict pools (51 trajectories; Bash 21,
Write 17, compound 8, Edit 7, Read 1; Python 22, Go 16, TypeScript 14, PowerShell 2) with file hashes; 5 trajectory-grouped folds
(sizes 11/11/11/10/11); no trajectory in two folds; 4 identical option texts occur in more than one fold (common commands);
canonical VAL, CAL and TEST not read.

Kaggle bundle, prepared only (`results/raw/level12-bundle-20261011T082630Z/`, `kaggle/level12_t4.py`, `tools/level12_bundle.py`):
- Level 1: 54 pools x 4 variants (real context; another state of the same trajectory, available for all 54; another trajectory;
  neutral prefix) = 216 captures; longest prefix 11,865 tokens, longest option 3,150 tokens, none over the frozen limits; in-run
  validation sample aborts the bulk capture on a shape, parity, memory or finiteness failure.
- Level 2: 100 decisions, one per trajectory, 11-12 per action family, 12 languages, 18 with a fidelity-consistent reconstructed
  state; the existing blind next-turn request builder; 4 samples at the model card's thinking defaults (temperature 0.6, top_p 0.95,
  top_k 20), 3,072 new tokens, 5.5-hour wall budget shared by two GPU workers. Of 130+ argument texts of the recorded actions, the
  ones that also occur in a prompt are text already present earlier in the session (URLs, edit anchors, commands run before).
- The kernel has NOT been run; the generation path is untested (no GPU work is allowed on this machine). No labels are in the upload.
- An earlier bundle folder (`level12-bundle-20261011T082553Z`, 79 Level 2 requests) is superseded.

### 14.1 Substitute second judge tried: DeepSeek V4.1 Flash, pair gate FAILED (2026-10-11, about 11:12 UTC)

Gemma and Kimi K3 both return HTTP 504 at about 302 s on one-word requests (`results/raw/gemma-health-*`, `results/raw/judge-alt-health-*`);
the cause found is overload of NVIDIA's free hosted endpoints, not our request. With the user's yes, `deepseek-ai/deepseek-v4.1-flash`
(`deepseek_max`: reasoning_effort max, streamed; probe `results/raw/deepseek-reasoning-probe-*`) judged the same 28 pilot packets and
20 hard-control packets Gemma had judged (`PREREG_DEEPSEEK.json`, `PAIR_REPORT-20261011T111201Z.json` in the pair-pilot folder; run with
`PAIR_PASS2=deepseek_max`, in which report fields named "gemma" mean the pass2 judge).
- Responses 48/48 usable, no truncation. Fable's action accepted by both 27/28. Off-task controls accepted 0/28.
- **Joint false acceptance 4/46 = 8.7% (gate <= 5%): FAIL.** DeepSeek accepted the same 4 premature-edit controls GLM accepts; Gemma rejected all 17.
- **Near-duplicates flagged by either judge 9/11 (gate >= 90%): FAIL.** DeepSeek missed the same 2 that GLM misses; GLM + Gemma caught 11/11.
- Yield would have been 17 strict pools of 28 (GLM + Gemma: 12), because DeepSeek mostly agrees with GLM (7 disputed alternatives vs 19).
Conclusion: DeepSeek's errors are correlated with GLM's, so it does not act as a second check. It is NOT qualified; nothing is admitted
with it. The 74 GLM-eligible pools still wait for Gemma (watcher `tools/gemma_when_ready.py`).

### 14.2 More substitutes tried (2026-10-11): gpt-oss-120b gone, Nemotron-3-Ultra pair gate FAILED

- `openai/gpt-oss-120b` on NVIDIA: HTTP 410 on all 14 keys ("end of life on 2026-09-03"). Only `openai/gpt-oss-20b` is hosted (healthy, not tested as a judge).
- `nvidia/nemotron-3-ultra-550b-a55b` (`nemotron_ultra_max`: reasoning_effort high = documented maximum, reasoning_budget -1, streamed), same 28 + 20
  packets, gate unchanged (`PREREG_NEMOTRON_ULTRA.json`, PAIR_REPORT written 2026-10-11 after 11:12 UTC with judges.pass2 = nemotron_ultra_max):
  48/48 HTTP 200 in 311 s, but **9 replies were not valid JSON (3 of 28 pilot = 10.7%, gate < 5%: FAIL; 6 of 20 hard)** and 8 more needed repair.
  **Joint false acceptance 3/37 = 8.1% (gate <= 5%): FAIL.** Alone it accepted 5 of 11 premature-edit controls (GLM 3 of the same 11, Gemma 0 of 17)
  and 1 of 25 off-task controls; it flagged 6 of 10 near-duplicates. Fable accepted by both 21/25. NOT qualified. Also the generator's family.

### 14.3 Second judge settled: Muse Glimmer 30B passes the pair gate (2026-10-11, about 13:20 UTC)

User (chat): "Try 5 6 models and fix second judge". 19 catalogue chat models were probed (`results/raw/judge-candidates-health-*`,
`judge-candidates-reasoning-*`); most are 404 for our keys or 504. Five ran the same 28 pilot + 20 hard-control packets under the unchanged
pair gate (`PREREG_FIVE_CANDIDATES.json`, one PAIR_REPORT each in the pair-pilot folder, judges.pass2 names the candidate):

| second judge (with GLM-5.3 full reasoning) | joint false accept | premature edits accepted alone | Fable accepted by both | near-dup caught | unusable pilot replies | pair gate |
|---|---|---|---|---|---|---|
| gemma_stream (reference, 2026-10-10) | 0/46 | 0/17 | passed | 11/11 | passed | PASS |
| **muse_max** = meta/muse-glimmer-30b, reasoning_effort max | **0/46** | **0/17** | 24/28 | 10/11 | 0/28 | **PASS** |
| gptoss20_high = openai/gpt-oss-20b | 4/45 | 11/17 | 22/27 | 10/11 | 1/28 | FAIL |
| diffgemma_max = google/diffusiongemma-26b-a4b-it | 0/33 | 0/13 | 7/19 | 7/9 | 7/28, wrong end line | FAIL |
| laguna_max = poolside/laguna-xs-2.1 | - | - | - | - | 28/28 | FAIL |
| nanoomni = nvidia/nemotron-3-nano-omni-30b-a3b-reasoning | - | 5/7 | - | - | 21/28 | FAIL |
| deepseek_max (14.1) | 4/46 | 4/17 | 27/28 | 9/11 | 0/28 | FAIL |
| nemotron_ultra_max (14.2) | 3/37 | 5/11 | 21/25 | 9/10 | 3/28 | FAIL |

Selected by the pre-registered rule: **muse_max** (the only passing candidate). On identical packets Muse and Gemma gave the same verdict on
82 of 98 alternatives (Muse stricter: 11 Gemma-only accepts, 5 Muse-only) and 61 of 68 controls. Muse's JSON needed a counted repair in 10 of 28
replies. Median 99.5 s per packet. GLM + Muse strict yield on the pilot: 10 of 28 pools (GLM + Gemma: 12). Muse is now reviewing the 74
GLM-eligible conventions pools (`MUSE_MAX_AFTER_GLM.json` in each review folder, run with `PAIR_PASS2=muse_max`); the Gemma watcher still runs.

**Scoring bug found (not yet fixed):** origin labels are not unique inside a pool (a merged duplicate keeps the same label, e.g. three options
all `nemotron_toolcond_gen:1`); `pair_pilot_v1.score` keeps one verdict per label and `hard_batch_v2_assemble` keeps the FIRST option with an
admitted label. Controls and Fable's action have unique labels, so every gate figure above is unaffected. Effect on the frozen 54 pools
(`strict-20261010T224125Z/DUP_ORIGIN_AUDIT.json`): 17 of 125 kept alternatives carry a repeated label; 14 were accepted by both judges, **3 were
accepted by one judge only** (go-telewire:1, py-campaign-templates:1, py-table-export:2). Fix needed before any pool from the new review is
admitted: score per option letter, then a NEW frozen-set version (the current FROZEN_SET.json is not edited).

### 14.4 Scoring bug fixed; corrected strict set has 71 pools (2026-10-11, 13:22 UTC)

`tools/pair_pilot_v1.py` now labels every alternative `origin@i` (`relabel`, option matched by its text in the packet) in `score` and `cascade`;
`tools/hard_batch_v2_assemble.py` reads the index from the label. No judge call was resent: the saved GLM + Gemma verdicts of batch v2 were
re-scored (`PAIR_REPORT-20261011T13215*Z.json`; pair gate still passes, joint false acceptance 0/46 and 0/99). Pools: pilot 12 -> 14 strict, rest
42 -> 57 strict (19 pools gained because options hidden behind a repeated label are now counted, 2 lost). Corrected strict set:
`.../build-20261010T212540Z/strict-20261011T132223Z` (**71 pools**; options-only refit detector 70.4% vs 29.7% chance, still a warning).
`FROZEN_SET.json` (54 pools, 3 options admitted on one judge) and the Kaggle bundle `level12-bundle-20261011T082630Z` are SUPERSEDED but not
edited. To do: new frozen-set version (folds by trajectory) from the 71 pools plus whatever GLM + Muse admits from the conventions pools, then a
new bundle. The Level 1 bundle must not be launched from the old 54-pool set.

### 14.5 Dataset v2 frozen; reconciliation of the reviewer's five corrections (2026-10-11, 13:50 UTC)

1. **Muse's gate.** The applicable gate is the pre-registered PAIR gate (`PREREG.json` in the pair-pilot folder: Fable's action accepted by both
   >= 80%, joint false acceptance <= 5%, near-duplicates >= 90%, unusable < 5%, no truncation), the same one GLM + Gemma passed with 86%. Muse:
   24/28 = 85.7% both, 25/28 alone; on the 80 conventions packets 74/80 alone and 73/80 both. The individual 90% qualification
   (`judge-qual-v1`, 37 packets, two passes) was NOT run for Muse; GLM and Gemma both failed it and the user approved the pair rule knowing
   that. JSON: 31 of 110 usable Muse replies lacked only the final closing brace (complete answer, finish_reason stop, every letter present);
   no other repair type occurred; 1 reply unparsable and 1 HTTP 404 (2 packets unusable, not admitted).
2. **Label integrity.** Judges answer by option letter and letters are unique, so every saved verdict maps to exactly one option; `relabel`
   matches each letter to its option text and asserts exactly one match (passed for all four review folders). No verdict was ambiguous, nothing
   was re-reviewed. GLM eligibility recomputed per option (`GLM_FIRST_v2.json`): 3 more pools became eligible and were sent to Muse.
3. **Audit before freeze.** Hand audit v2 (`results/raw/laya-audit-v2-20261011/`, verdicts written before the key was opened): GLM + Gemma
   cohort 9 plausible / 1 implausible of 10 (an unrequested report file); GLM + Muse cohort 7 plausible / 1 implausible (re-reads a file just
   written) / 1 unclear / 1 same as Fable's command without `| tail -20` of 10; of 4 not-admitted options the auditor agreed with the rejection
   in 3 (all three were accepted by Muse and rejected by GLM: Muse is lenient on redundant re-reads). Deterministic filter added to
   `hard_batch_v2_assemble.bash_core` (drops an alternative whose Bash command equals Fable's after removing `cd . &&`, `2>&1`, `| tail/head`):
   2 options dropped, 2 pools lost.
4. **Cohorts.** Each pool carries `review_judges`, `judge_cohort`, `review_dir`, `source_strict_dir`. GLM + Gemma: 70 pools, 63 trajectories
   (Bash 27, Write 21, Edit 11, compound 10, Read 1). GLM + Muse: 53 pools, 52 trajectories (Bash 38, Edit 7, compound 4, Write 2, Web 2;
   10 languages). 19 trajectories occur in both cohorts. Fable's own action was rejected by one judge in 11 of the 123 pools.
5. **Order.** Baselines right after capture; head-vs-baseline only after fine-tuning; Level 2 in two parts (proposal usefulness before
   training, ranking of Qwen's proposals after).

Frozen: `results/raw/laya-fullaction-set-v2-20261011T134826Z/` (FROZEN_SET.json sha256 8971fd56b1df21a364771cc303a4a6fda30b9b9076aa41ecb9245ecf02808390;
`tools/laya_fullaction_freeze_v2.py`): 123 pools, 96 trajectories, 307 alternatives; options 3/4/5 = 77/31/15; folds 25/24/24/25/25, no trajectory
in two folds, 6 option texts occur in more than one fold. Shortcut diagnostics: options-only strongest refitted detector 67.5% (chance 29.6%),
Gemma cohort 71.4%, Muse cohort 65.1%; action-history prior 37.0% (Gemma cohort 36.8%, Muse cohort 38.3%).
Bundle: `results/raw/level12-bundle-20261011T134843Z/` (492 Level 1 captures, longest prefix 18,072 tokens, longest option 3,150, none over
limit; Level 2 100 decisions, prompts 1,337-22,450 tokens; 1.39 MB; no labels). Local checks: bundle builder limits, `tests/test_fable_menu_v1.py`
5 passed, scripts compile. NOT uploaded, NOT run. The Gemma watcher was stopped. Three folders named `INCOMPLETE-strict-*` are failed assembly
attempts and are not used.

### 14.6 User decision and per-option verification (2026-10-11)

User (chat): keep dataset v2 as frozen (70 GLM + Gemma pools, 53 GLM + Muse pools); do NOT re-review the older pools with Muse; no more NVIDIA
calls for reviewer consistency for now; GLM and Muse stay available as the review pair when new pools are needed. Report head results per cohort
and combined. Muse passed the PAIR gate (Fable accepted by both >= 80%; measured 24/28 = 85.7%); this is never to be described as meeting the
individual 90% threshold.

Per-option verification of all 123 pools (`results/raw/laya-fullaction-set-v2-20261011T134826Z/PER_OPTION_VERIFICATION.json`): 307 of 307 kept
alternatives were matched by exact text to their own letter and found PLAUSIBLE and not a duplicate of Fable for BOTH judges of their cohort, are
in the technically screened pool, and are not the same Bash command as Fable; Fable action identical to the screened reference in 123 of 123; 0 failures.
Next: the user GO for the Kaggle launch (bundle `results/raw/level12-bundle-20261011T134843Z`).

### 14.7 Kaggle Level 1 + Level 2 run launched (2026-10-11, 15:20 UTC)

User GO in chat ("Yes I approve"). Before the upload the Level 2 worker start was changed to use a copy of the script in the dataset
(`level12_worker.py`; a Kaggle script kernel may not define `__file__`), and the bundle was rebuilt: `results/raw/level12-bundle-20261011T151731Z`
(request rows byte-identical to bundle `...134843Z` after decompression). Private dataset `sangeetasangamkar/laya-level12-v1` (1.4 MB, no labels),
kernel `sangeetasangamkar/laya-level12-v1-run` version 1, T4 x2, Rs 0. Output is downloaded to `results/raw/level12-run-v1-*` when the run ends.
Nothing is trained in this run.

### 14.8 Kaggle run status (2026-10-11, 17:35 UTC)

Version 1 of `laya-level12-v1-run` sat QUEUED for 42 minutes and was cancelled by the user; version 2 started at once.
- **Level 1 complete (Kaggle log): 492 of 492 captures ok** (123 pools x B/S/X/A), 10 of 10 validation checks passed (repeat identical, plain-vs-packed cosine 1.0, finite), 2,424 s, peak 8.7 GB. Not yet downloaded or verified locally: Kaggle saves the output when the run ends.
- **Level 2: one of the two workers died at its first generate** with `CUDA error: CUBLAS_STATUS_EXECUTION_FAILED` on a small prompt (first prompts 3,076 and 1,413 tokens, so not prompt size). Probable cause (not confirmed): the parent process kept the Level 1 model on GPU 0, leaving too little memory for the worker on that GPU. The other worker kept running.
- Fix (`kaggle/level12_t4.py`, VERSION 2026-10-11b): every stage is its own process (the parent never touches CUDA); Level 2 workers are restartable, resume from their file, retry a crashed decision one sample at a time and skip it after two crashes; control flow tested with fake torch (`scratchpad/mock_l2.py`: crash, restart, skip, resume all as intended).
- Separate kernel `sangeetasangamkar/laya-level2-fix-run` v1 (Level 2 only, even-index rows = the 50 decisions of the dead worker, same prompts, seeds and sampling settings) pushed 17:31 UTC with dataset version 2; bundle `results/raw/level2-fix-bundle-20261011T173018Z`. Version 2 of the first kernel was left running so its Level 1 output is saved.

### 14.9 Level 1 captured and verified; Level 2 generation not working yet (2026-10-11, 18:00 UTC)

- The user cancelled both running kernels. A cancelled Kaggle run DOES keep its output (my earlier statement that it would be lost was wrong).
- **Level 1 verified locally** (`results/raw/level12-run-v2-cancelled-20261011T175252Z/LEVEL1_VERIFY.json`): B/S/X/A each hold 123 records = the 123 frozen pools, context/option/order hashes equal to the uploaded requests, hq (3, 2560) and ho (n_options, 3, 2560) fp16, all finite; no query state identical between the real context and any control (median last-depth cosine to the real context: S 0.914, X 0.942, A 0.873). Kaggle log: 10 of 10 validation checks passed.
- **Level 2 saved rows (5 decisions, not usable as a result):** first kernel, surviving worker: decision 1 (prompt 1,298 tokens) 763 s, four samples finished; decision 2 (4,068 tokens) 2,146 s, four samples hit the 3,072-token limit inside the thinking block. First fix kernel: three decisions (5,187-7,023 prompt tokens) returned ZERO samples within about 1 s: out of memory in both the batched and the one-sample path, and the code recorded them as done (bug).
- So two real problems: about 0.17-0.27 s per generated token, and out-of-memory in `generate` above roughly 5,000 prompt tokens, although Level 1 forwards 18,000-token prefixes at 8.7 GB peak. Cause not established. My time estimate for Level 2 was never measured and was wrong.
- Diagnostic kernel `sangeetasangamkar/laya-level2-diag-run` (`kaggle/level2_diag_t4.py`) pushed: default attention path vs `attn_implementation="sdpa"`, 4 real prompt lengths, 1 and 4 samples, prefill time, seconds per token, peak memory. The corrected Level 2 kernel `laya-level2-fix-run` version 2 (pushed 17:55 UTC, before these rows were read) should be cancelled: it would hit the same memory problem.

### 14.10 Level 2 diagnostics (2026-10-11, 18:05 UTC)

`laya-level2-fix-run` v2 was cancelled by the user before it started. Two diagnostic kernels ran (`kaggle/level2_diag_t4.py`, `kaggle/level2_diag2_t4.py`; outputs `results/raw/level2-diag-run-20261011T175541Z`, `results/raw/level2-diag2-run-20261011T175953Z/manual/diag2.json`).
- Diagnostic 1 (library `generate`, attention already sdpa, cache on): 0.038 s/token at 1,298 prompt tokens, 0.144 s/token at 4,313 (13.6 GB peak); 4 samples at once out of memory at 4,313; ONE sample out of memory at 7,939 (one allocation of 7.51 GiB = heads x L x L float32) and at 13,931 (23.14 GiB).
- Diagnostic 2 (manual loop, prompt prefilled in 1,024-token chunks): no out-of-memory up to 19,986 prompt tokens (peak 11.0-11.5 GB). Seconds per generated token, growing cache / preallocated StaticCache: 1,225 tokens 0.077 / 0.058; 3,777: 0.132 / 0.077; 7,023: 0.213 / 0.110; 12,173: 0.336 / 0.193; 19,986: 0.513 / 0.267. The per-token cost still grows with context length (cause not established; suspected the per-step expansion of the grouped key-value heads).
- Consequence: at the fixed settings (100 decisions, 4 samples, up to 3,072 new tokens) the measured speeds give tens of GPU-hours; it does not fit one Kaggle session. Level 2 is NOT launched; the design decision is with the user.

### 14.11 Level 2 deferred (2026-10-11, about 18:15 UTC)

User (chat): "Is confirming Level 2 needed? Check all the previous Kaggle runs for Qwen quality and skip it." Existing evidence on the Fable decisions:
`results/raw/intent-level-checks-v1-20261010T172625Z/REPORT.json` check 3: 1,616 TRAIN coding decisions, 1,579 with saved Qwen3-4B proposals (1,180 with four):
Fable action among the 4 proposals: exact 1.4%, same intent 17.0% (single-call tool decisions: 2.2% / 25.2%; Write 53% intent, Edit 41%, Read 41%, Bash 6%, replies 0%).
Limit of that evidence: the saved proposals are single-call, forced-answer outputs from a cut-down context, so replies and multi-call actions cannot match by
construction; it is a baseline for the weak protocol, not a measurement of full-context proposals. The new full-context run would cost about 10 GPU-hours for
40 decisions (section 14.10) and does not change the next decision (whether the head with context beats the baselines on Level 1).
Decision recorded: Level 2 is NOT run now; it is revisited only if Level 1 shows the head adds value, and then with a faster generation engine.
The 5 Level 2 decisions saved by the stopped runs are kept as engineering evidence only (2 with samples, 3 empty).


## 15. Complete remaining steps (written 2026-10-11 about 18:40 UTC at the user's request: "the rest steps without missing any, the proven way")

Inventory measured now (`results/raw/fullaction-inventory-20261011.json`): 1,342 eligible TRAIN coding tool decisions (281 trajectories);
hard same-state candidates generated for 403 (79 first experiment, 189 batch v2, 40 + 95 conventions prompt); 123 qualified into dataset v2
(30.5%); 939 not generated, of which 671 usable under the current rule (575 single-tool, 96 multi-call; 235 trajectories; 618 never examined,
53 previously examined) and 268 first-step decisions excluded by the selection rule. Other canonical splits, counts only (not read):
VAL 227 tool decisions / 51 trajectories, CAL 309 / 66, TEST 713 / 141. Reply decisions (274 in TRAIN) are outside the full-action tool pools.
Trainer fact: `decision_head/fable_train.py` validator asserts 5-8 options per pool; the v2 pools have 3-5, so training goes through a versioned
driver that calls the unchanged `train()` with its own rows, as `tools/menu_finetune_pilot.py` did.

**Phase 1 - does the head learn from context on the 123 captured pools (no new data)**
1. Baselines on the Level 1 features, v2 folds, per judge cohort and combined, trajectory bootstrap: majority / action-history prior, options-only
   text selectors, a context-text model, and the mandatory linear probe on the Qwen features (real context).
2. Reference: the pretrained head with its 31 tensors unchanged (no fine-tuning) on the same folds.
3. Versioned full-action training driver (frozen `train()` unchanged; 3-5 options passed directly), with a pre-registration written BEFORE any run:
   configuration, seeds, metrics, and what counts as "the head adds value". CPU dry run on one fold.
4. Controlled fine-tuning pilot (USER GO): Qwen frozen, phase A adapter only, phase B adapter + the 31 pretrained tensors at a lower learning rate;
   `objective="recorded"`, unknown weight 0.5 and 1.0; at least 3 seeds; 5-fold trajectory-grouped cross-validation; CPU on cached features.
5. Evaluation: head vs linear probe vs options-only vs prior; real context vs the three controls (same-trajectory other state, other trajectory,
   neutral prefix); by cohort, tool, language, single vs multi-call; paired bootstrap intervals; one row in `results/increments.md`.
6. Decision: if the head with real context beats the linear probe and options-only beyond the interval, continue to Phase 3 with more data; if not,
   diagnose the adapter / head representation before spending more on data.

**Phase 2 - generate and qualify the remaining candidates (the proven pipeline; may run in parallel with Phase 1)**
7. Three user decisions first: (a) generator prompt for the remaining decisions (original or session-conventions; recorded per pool as a cohort
   either way), (b) whether the 268 first-step decisions are admitted (a rule change), (c) stage sizes.
8. Move the one-off scripts into `tools/` so every batch runs the same code: per-option GLM eligibility, per-option verification, Level 1 capture
   verification, inventory.
9. Stage 1 selection (about 200 decisions): seeded, stratified to include the missing languages (C#, Java, Rust, Ruby, PowerShell) and multi-call
   decisions, at most 2 per trajectory, never-examined first; the 53 previously examined ones flagged "training only".
10. Generation: Nemotron-3-Super, 8 calls per decision (3 tool-conditioned + 5 blind next-turn), every key at <= 39 sends per minute, a
    rate-limit response cools that key 60 s, every send logged, completed ids never resent.
11. Decode and build pools: structural validity, identical renderings collapsed and counted.
12. Objective checks before any judge: tool contract, grounding, duplicates, real parsers in the WSL sandbox under the calibration rule. Open item:
    the TypeScript parser check is disabled by that rule and one admitted TypeScript option had a syntax error.
13. Blinded full-context packets in two letter orders with controls in every packet: Fable's action, an off-task action, a premature edit where one
    exists, a synthetic near-duplicate for Bash.
14. GLM-5.3 full reasoning (streamed, 65,536-token allowance) on every packet; one retry for an unusable reply.
15. GLM eligibility computed per option; Muse Glimmer 30B (max reasoning) on the eligible pools, a seeded 25% sample of the others and the pools
    with an unusable GLM reply; one retry.
16. Scoring per option: admitted only if both judges say plausible and neither marks it a duplicate of Fable's action; one representative per
    duplicate cluster; a pool needs Fable's action plus at least 2 admitted alternatives, one with Fable's tool set.
17. Pair gate re-checked on each batch's own controls (both accept a known-bad option <= 5%, Fable's action accepted by both >= 80%, near-duplicates
    >= 90%, unusable < 5%, no truncation). A batch that fails is not admitted.
18. Assembly with the deterministic same-as-Fable Bash filter; then per-option verification of every kept alternative (as done for v2: 307 of 307).
19. Blind hand audit of a seeded sample per batch (admitted and not admitted), verdicts written before the key is opened.
20. Shortcut diagnostics (options-only and action-history prior, reported separately, per cohort) and a coverage audit (language, tool, multi-call).
21. Stage 2 (the rest of the 671) only if stage 1 yield, controls, audit and diagnostics hold; same steps 10-20.
22. Freeze dataset v3: v2 pools unchanged; trajectories already in v2 keep their fold; new trajectories assigned by the same seeded rule; hashes,
    generator-prompt cohort and judge cohort per pool; v2 stays on disk.
23. Tokenizer check of every new pool; Kaggle T4 Level 1 capture of the new pools in the same four variants with the in-run validation sample
    (USER GO); local verification of every record against the frozen set.

**Phase 3 - full training and independent evaluation**
24. Baselines and training repeated on v3 by grouped cross-validation; one configuration chosen on development data only.
25. Evaluation pools for canonical VAL (227 decisions) built with the identical pipeline (steps 10-20, 23), then ONE evaluation of the frozen
    configuration on VAL. VAL is never used to choose anything.
26. CAL (309 decisions): pools and capture the same way; temperature and the abstain threshold fitted on CAL; no gradient step ever uses CAL.
27. TEST (713 decisions): pools and capture the same way; read once with the frozen head and thresholds.
28. Level 2 (deferred): Qwen's own full-context proposals with a faster generation engine; first whether they contain a good action, then whether
    the trained head ranks them better than simple methods.
29. Level 3: the offline agent on protected tasks with and without the head (three-arm test; hard rules and real tests override the head).
30. Ship decision: the head ships only if the three-arm agent test passes.

**Standing rules through every step**: Fable's action is the positive and alternatives stay unknown; nothing is admitted on one judge; thresholds are
never edited after results (a new versioned file instead); results are create-only under `results/raw/`; spend stays Rs 0; no GPU work on the local
RTX 3050; the pretrained head file, the trainer and Qwen are never modified; README, PLAN, DECISION_HEAD, this handoff and `results/increments.md`
are updated after every major step; no git commit or push unless the user asks.

### 15.1 Both tracks started (2026-10-11, 18:30 UTC; user GO in chat for baselines, the fine-tuning pilot, adapter-only vs joint, Stage 1 generation with the session-conventions prompt; the remaining 471 held; Qwen never modified)

- **Stage 1 generation running:** `results/raw/hard-batch-v3-conv-20261011T182620Z` (mode `stage1` added to `tools/hard_batch_v3_conv.py`: 200 never-generated, never-examined TRAIN decisions, any trajectory, at most 2 per trajectory, 30% multi-call, languages in rotation; 1,600 planned sends, cap 2,000; toolcond on keys 2-6, v3t on keys 7-15). The leak re-check found 144 recorded argument texts that also occur in a request (edit anchors, paths, URLs, commands already present earlier in the session; same pattern as earlier batches).
- **Pilot running:** `tools/fullaction_pilot_v1.py`, output `results/raw/fullaction-pilot-v1-20261011T182942Z` (PREREG.json written before any run, with the outcome rules from the chat; nested selection: checkpoint chosen on an inner development fold, never on the test fold; every head figure is a 3-seed mean).
- **Parameter counts measured from the built head (`PARAMS.json`):** 31 pretrained tensors = 26,248,193 parameters; adapter phase A trains 8 tensors = 5,246,982; 5 tensors (273,154) stay frozen in both phases; phase B trains 39 tensors = 31,495,175.
- **Baselines measured (BASELINES.json; 123 pools, 5 trajectory-grouped folds, chance 29.6%):** action-history prior 37.2%; options text 62.6%; context text 66.7%; **linear probe on Qwen option states with the real context 84.6% (CI 78.4-90.6) and with the neutral prefix, fitted and tested without any context, 84.6% (CI 78.0-90.8)**; the real-context probe scored on wrong-context features: other trajectory 81.3%, neutral 78.9%, other state of the same trajectory 65.9%. So in Qwen feature space the options alone identify the recorded action about 85% of the time and the real context adds nothing measurable for a linear probe.

### 15.2 No-context baseline checked (2026-10-11, about 18:45 UTC; read-only, `results/raw/fullaction-pilot-v1-20261011T182942Z/NOCONTEXT_CHECK.json`)

- The A capture is context-free: its prefix is the fixed text "Candidate next actions for a coding task:" + the query cue, 11 tokens for all 123 pools; no label, reference index or origin was uploaded; the query state is the same for every pool up to fp16 noise (minimum cosine 0.999999); the same option text in two different pools has the same A features (45 pairs, minimum cosine 0.999999) and different B features (minimum 0.944).
- The two probes are equally accurate but not identical: both right on 99 pools, only the real-context probe right on 5, only the no-context probe right on 5, both wrong on 14.
- No position cue: always picking option 1, 2 or 3 gives 27.6%, 30.1%, 31.7% (chance 29.6%). No length cue: Fable action is the longest option in 30.1% of pools.
- The narrow interval of the action-history prior (34-40%) is real, not a bug: in 92% of pools the prior ties across options with the same tool, so a pool scores a fraction (0.5, 0.33, 0.25) and the per-pool spread is 0.14 against 0.48 for a right/wrong outcome.
- The drop of the real-context probe on other-state features (65.9%) is not evidence of state reasoning: it can be plain train/test mismatch.
- Side tool: `progress_ui/server.py` (written by a Haiku sub-agent, read-only, standard library only) shows live call and run counts at http://localhost:8765; preview config `.claude/launch.json`.

### 15.3 Read-only audit while the pilot runs (2026-10-11, about 19:00 UTC; `results/raw/fullaction-pilot-v1-20261011T182942Z/AUDIT2.json`, per-pool chosen option and scores saved for every probe)

- **Probes fitted AND tested on one context variant, identical folds, C grid, inner selection and option order:** real context 84.6% (CI 78.4-90.6), other state of the same trajectory 89.4% (84.2-94.4), another trajectory 85.4% (79.0-91.7), no context 84.6% (78.0-90.8). So a probe given the WRONG context does as well as or better than one given the right context: for a linear probe the context carries no usable signal on these pools, and the earlier 65.9% for the real-context probe scored on other-state features was train/test mismatch.
- Real-context vs no-context probe: same option chosen in 112 of 123 pools; paired difference 0.0 (CI -5.0 to +5.0 points).
- **Overlap of action texts across folds:** Fable action text occurs exactly as a positive in the fit folds for 15 of 123 pools (16 after normalising Bash commands, 42 at >= 0.9 string similarity); it never occurs as a fit-fold alternative; in 15 pools a test alternative is a fit-fold positive. No-context probe: 100% on the 15 exact-seen pools, 82.4% on the 108 unseen ones. Repetition explains about 2 points, not the effect.
- By tool (real / no context / options text): Bash n=65: 90.8 / 92.3 / 78.5; Write n=23: 91.3 / 87.0 / 65.2; Edit n=18: 72.2 / 66.7 / 44.4; multi-call n=14: 64.3 / 71.4 / 14.3. By generator prompt and judge cohort (the same split): original + Gemma 84.3 / 84.3, session-conventions + Muse 84.9 / 84.9: the conventions prompt did not reduce the feature-space cue.
- When the no-context probe is wrong it picks a tool-conditioned alternative 13 times and a blind one 6 times (alternatives overall: 195 and 112).
- Action-history prior: 37.2% is the expected value under random tie-breaking; with a deterministic choice (first of the tied options) 36.6% (CI 27.1-45.7).
- Head runs finished so far (22 of 90, folds 0-1 only, NOT a result): train top1 93-99%, test 71-80%, gap 14-29 points; adapter-only on real context 79.7% (6 runs), joint 72.7% (4 runs).
- The head runs save per-pool correctness for every context variant, not the chosen option index; checkpoints are deleted after scoring.

### 15.4 Runner changed to the fixed call logic (2026-10-11, about 19:10 UTC)

User (chat): stop, save, add all keys, run; then: 10 calls in flight per key, a rate-limit response cools that key one minute; record it as fixed logic in
the README. `tools/nemotron_candidate_pilot.run_parallel` now starts `PAR_INFLIGHT = 10` workers per key with a per-key shared 39-per-minute window and a
per-key cooldown; self-check with a fake sender passed (10 in flight per key, no send during a cooldown, 90 of 90 responses saved once). The Stage 1
tool-conditioned run was stopped twice (5 and then 14 requests in flight; markers set aside and listed in `INTERRUPTED_*.txt`; no saved response resent)
and restarted on all 14 keys. README section "FIXED NVIDIA CALL LOGIC"; CLAUDE.md user rules.

### 15.5 In-flight limit set to 5 (2026-10-11, about 19:12 UTC)

Measured with 10 in flight per key: about half of each key burst was rejected with HTTP 429 (89 rejections against 190 completions in about four minutes). User (chat): limit 5, restart, check no data loss. `PAR_INFLIGHT = 5`; README and CLAUDE.md updated. Integrity check before the restart (`INTEGRITY_*.json` in the toolcond run folder): 370 of 600 responses saved, all readable, all `completed`, none empty, none outside the plan, no temporary file; ledger 370 answers for 370 distinct request ids, no request answered twice; 78 in-flight markers set aside and re-queued; the v3t half is complete (1,000 of 1,000).

### 15.6 Stage 1 generation finished, pools built and screened, GLM review started (2026-10-11, 19:20 UTC)

- Generation complete: v3t 1,000 of 1,000, toolcond 600 of 600 (724 sends including 124 rate-limit rejections; `RUN_DONE.json`). At 5 in flight per key the steady rate was about 60 calls a minute with no rejection after the first minute.
- Build `results/raw/hard-batch-v3-conv-20261011T182620Z/build-20261011T191653Z`: 200 decisions -> 142 structurally valid pools (58 excluded: 35 no same-tool alternative, 20 too small, 3 no same-shape alternative); 329 candidates rendered exactly like the recorded action and 275 like another candidate were collapsed before the build.
- **Screening bug fixed before this batch:** `tools/hard_batch_v2_build.py` keyed check results by origin label, and labels repeat inside a pool, so an option could inherit the result of another option with the same label. `syntax` now writes the option index and `screen` keys by (decision, index). **The earlier builds (dataset v2) were screened with the old keying and must be re-checked per option; not done yet.**
- Screen: 518 alternatives, 30 failed an enforced check; 137 pools technically screened (csharp 18, go 14, powershell 14, python 13 + py 12, ruby 13, bash 11, java 11, rust 11, zsh 9, ts 8, cpp 2, js 1; 21 multi-call). C#, Ruby and Rust have no parser in the sandbox; the TypeScript parser check is still not enforced (2 of 7 Fable files fail it).
- Review packets: `.../pair-review-rest-20261011T191916Z` (137 packets; controls: 137 off-task, 60 near-duplicate, 11 premature edit). GLM full reasoning running at 5 in flight per key on all 14 keys.

### 15.7 Dataset v2 re-checked per option against the objective screen (2026-10-11, about 19:30 UTC)

The sandbox syntax step was re-run per option index for the three earlier builds (`SYNTAX_byindex.jsonl`) and every check recomputed per option (`results/raw/laya-fullaction-set-v2-20261011T134826Z/SCREEN_PER_OPTION_AUDIT.json`): the enforced scopes are the same as in the original screens; **all 307 kept alternatives pass every enforced check; 0 pools affected.** Dataset v2 stands unchanged. (134 options in those builds carried a repeated label, so the old keying could have mattered; it did not for any kept option.)

### 15.8 Rule: syntax-verified vs judge-approved (user, chat, 2026-10-11)

For languages without a parser in the sandbox (C#, Ruby, Rust; also Zsh and C++ file contents) an admitted alternative is JUDGE-APPROVED ONLY. GLM + Muse never stand in for an executable syntax check. To do at Stage 1 scoring / freeze: (1) a per-option sidecar listing which objective scopes actually ran for that option (contract, grounding, parser name) and a per-option level `parser_verified` / `structure_checked_only` (the C# and Java brace-balance check is not a parser) / `judge_only`; (2) the same field back-filled for dataset v2; (3) results reported separately by that level; (4) the independent hand audit of Stage 1 must include C#, Ruby and Rust options in its sample, stratified. Open option for the user: install Ruby, Rust and .NET toolchains in the WSL sandbox so these languages become parser-verified (needs approval: package installs and downloads).

### 15.9 Fine-tuning pilot RESULT (2026-10-11, 19:33 UTC; `results/raw/fullaction-pilot-v1-20261011T182942Z/REPORT-20261011T193329Z.json`; 90 CPU runs, pre-registered, pretrained file unchanged)

123 pools, 5 trajectory-grouped test folds, checkpoint chosen on an inner development fold, every head figure a 3-seed mean; top1 = ranks Fable's
recorded action first; chance 29.6%. Intervals: 2,000 bootstrap resamples of whole trajectories.

| Model | top1 | 95% interval |
|---|---|---|
| Pretrained head, no update | 36.6% | 31.5-41.9 |
| Options text (state-blind) | 62.6% | 53.5-71.6 |
| Head, adapter only (weight 0.5) | 77.2% | 70.8-83.3 |
| Head, joint (weight 0.5) | 76.7% | 70.0-83.1 |
| Head, adapter only, trained and tested WITHOUT context | 76.7% | 69.6-83.5 |
| Head, joint, trained and tested WITHOUT context | 78.3% | 71.9-84.5 |
| Linear probe, real context | 84.6% | 78.4-90.6 |
| Linear probe, no context | 84.6% | 77.9-90.8 |

Paired differences on the same pools: joint - adapter -0.5 points (CI -2.4 to +0.9); joint - linear probe -7.9 (-12.0 to -3.7, significant);
adapter - linear probe -7.3 (-11.6 to -3.0, significant); joint with context - joint without -1.6 (-6.0 to +2.8); adapter with - without +0.5
(-5.1 to +6.2); joint - options text +14.1 (significant); joint - pretrained init +40.1 (significant). Unknown weight 1.0: adapter 80.8%, joint 75.6%
(weight 1.0 - 0.5 for joint: -1.1, not significant). Train top1 93.6-97.2% against 76-81% test. Joint runs selected a phase-B checkpoint in 7-9 of 15
runs. A real-context head scored on wrong-context features falls (other state 61.0%, other trajectory 69.4%, neutral 71.5%), which is train/test
mismatch, since heads trained without context do as well.

Reading against the pre-registered outcome rules: (1) adapter-only matches joint -> prefer keeping the pretrained LAYA layers frozen at this data
size; (2) the linear probe beats both head recipes -> diagnose the adapter / head before any further head scaling; (3) no model shows a benefit from
the real context on these pools. Fine-tuning does work mechanically (36.6% -> 77%). No claim about coding-task success.

### 15.10 Sandbox toolchains added and extended parser checks (2026-10-11, about 19:35 UTC; user approved in chat)

Installed in WSL Ubuntu-24.04 from the Ubuntu repository: ruby 3.2.3, zsh 5.9, rustc 1.75.0, dotnet-sdk 8.0.131 (about 240 MB download); g++ 13.3 was
already present. New syntax-only parsers in `tools/hard_qualify_v2_syntax.py`: `ruby -c`, `zsh -n`, `rustc -Z parse-only`, a Roslyn ParseText tool
(`tools/sandbox_csparse.cs`, built to /opt/toolchains/csparse), `g++ -fsyntax-only`. Each was tested in the sandbox on one valid and one broken file.
Per-option results: `SYNTAX_v2_parsers.jsonl`, `PARSER_LEVELS_v2.jsonl`, `PARSERS_V2_REPORT.json` in each build folder.
- Stage 1 build (481 screened alternatives): Fable's own files pass every new parser (C# 7/7, Ruby 3/3, Rust 5/5, Zsh 5/5), so they are ENFORCED;
  17 alternatives fail one (Zsh 6, C# 5, Rust 3, Ruby 3) and will be dropped at assembly even if both judges accept them. TypeScript stays not
  enforced (5 of 7 Fable files pass `node --check`). Verification level of the alternatives: 107 with parser-verified file content, 20 failing a
  parser, 5 with file content and no parser (2 C++, 2 PowerShell, 1 C#), 349 with no file content (Bash / Read / Edit / Web; an Edit result is not parsed).
- Dataset v2 builds: no kept alternative fails a newly available enforced parser.
- `tools/hard_batch_v2_assemble.py` now drops an admitted option that fails an enforced parser and stores `verification_level` and `parser_scopes_run` per option.
- GLM review of the 137 Stage 1 pools: running at 5 in flight per key (65 rate-limit rejections in the first minute, none after).

### 15.11 README full status report written (2026-10-12 01:18 IST = 2026-10-11 19:48 UTC)

README.md section "FULL STATUS REPORT - 2026-10-12, 01:18 IST" consolidates the state at that time. Running then: the GLM review of the 137 Stage 1 pools (126 answered) and the progress page; nothing else.

### 15.12 Stage 1 scored and assembled (2026-10-11, 20:30 UTC; pair `glm_max64` + `muse_max`)

- GLM: 137 packets, 136 usable first time, the 1 unusable recovered on its single retry (median 520 s per packet; 11 replies needed the counted JSON repair). GLM left 104 pools eligible, 32 not, per option (`GLM_FIRST.json`, now produced by `pair_pilot_v1.py glm_first`).
- Muse: 113 packets (104 eligible + 8 sampled + 1), 110 usable first time, 3 retried once, 112 usable in the end. NOTE: the scorer reports Muse first-attempt failures as 113 of 113 because Muse answers are stored under the run name `pilotr` in GLM-first order; the real first-attempt failure rate is 3 of 113 = 2.7% (gate < 5%). The report flag `pair_passed: false` comes only from that naming artefact.
- Pair gate on this batch controls: known-bad options accepted by both 0 of 123 (112 off-task + 11 premature edits); Fable action accepted by both 96 of 112 = 85.7% (GLM 109, Muse 96); near-duplicates flagged by either 53 of 53; no truncation. All gate checks hold.
- Alternatives in the 112 scored pools: 243 jointly admitted, 121 disputed, 22 rejected by both, 13 duplicates of Fable.
- **Strict pools: 60** (`.../build-20261011T191653Z/strict-20261011T203033Z`; 53 trajectories, 167 alternatives; options 3/4/5 = 23/27/10). 62 before the filters; the same-as-Fable Bash filter dropped 4 options and 2 pools; no jointly admitted option failed an enforced parser (the 17 parser failures were not jointly admitted). Yield 60 of 200 decisions = 30%.
- By language: Go 11, C# 8, Python 9 (7 + 2), Ruby 7, Rust 7, Java 6, Bash 5, PowerShell 4, Zsh 2, TypeScript 1. By Fable tool: Bash 41, multi-call 10, Edit 4, Read 3, Write 3, WebFetch 1 (Write: only 3 of 36 screened pools survive).
- Verification level of the 167 kept alternatives: 159 have no file content (Bash / Read / Edit / Web), 8 have parser-verified file content, 0 are judge-approved-only file content.
- Shortcut diagnostic (text only, warning): options-only refit 55.0% against 27.4% chance (dataset v2: 67.5% against 29.6%). The Qwen-feature probe cannot be run on Stage 1 without a new capture (needs the user GO).
- Still to do for Stage 1: per-option verification report, blind hand audit including C#, Ruby and Rust, freeze as a separate set.

### 15.13 Stage 1 completed, frozen and sent for the short no-context capture (2026-10-11, 20:40 UTC = 2026-10-12 02:10 IST)

User (chat): GO for Stage 1 completion, then the short no-context Kaggle capture (no four-variant capture); fix the pair-gate report; models trained only on v2, tested on trajectory-disjoint Stage 1 pools; the remaining 471 decisions stay on hold.
- **Pair-gate report fixed** (`pair_pilot_v1.score`, `first_attempt`): in GLM-first order the second judge first answers are read from `pilotr` / `pilotr_failed*`. Re-scored: Muse first attempt 110 of 113 usable (2 unparsable, 1 HTTP 502; 2.7%), 112 after one retry; `pair_passed: true`; controls, alternative statuses and pool statuses are identical to the earlier scoring (verdict-derived fields unchanged).
- **Hand audit v3** (`results/raw/laya-audit-v3-stage1-20261012/`, 24 options, verdicts written before the key was opened; kept options stratified C# 4, Ruby 4, Rust 4, other 6): of the 18 kept, 13 plausible, 4 unclear, 1 the same as Fable action in effect (Fable test command plus a separate `echo $?` call), 0 implausible. Of 6 not admitted: 2 implausible, 3 plausible, 1 unclear. Syntax is not judged in the audit; the sandbox parsers do that.
- Filter extended (`bash_core`): a multi-call option that is Fable command plus `echo $?` counts as the same action. Re-assembled: `.../build-20261011T191653Z/strict-20261011T203619Z`, 59 pools (7 options dropped, 3 pools lost to the filter in total).
- **Stage 1 frozen separately:** `results/raw/laya-fullaction-stage1-v1-20261011T203622Z` (59 pools, 52 trajectories, 164 alternatives; folds 12/12/12/12/11). Per-option verification: 164 of 164 alternatives accepted by both judges, in the screened pool, not the same Bash command as Fable; 59 of 59 Fable actions identical to the screened reference. Overlap with v2: 9 trajectories, 11 pools; no decision in both.
- Shortcut diagnostic (text): strongest options-only detector 63.3% (word tokens) against 27.3% chance; action-history prior 32.1%.
- **Kaggle:** `kaggle/level12_t4.py` VERSION 2026-10-12a gained `VARIANTS` and `RUN_LEVEL2`; kernel `sangeetasangamkar/laya-stage1-nocontext-run` v1 (variant A only, no Level 2) pushed 20:38 UTC with dataset version 4; bundle `results/raw/level12-bundle-20261011T203713Z`.
- **Transfer test pre-registered before any Stage 1 feature was downloaded:** `tools/stage1_transfer_v1.py`, PREREG in `results/raw/stage1-transfer-v1-20261011T203921Z`. Note: no fitted probe or head was saved by the pilot (checkpoints are deleted after scoring), so the scorers are REFITTED on v2 only: linear probe on all 123 v2 pools; adapter-only head on v2 folds 1-4 with the checkpoint chosen on v2 fold 0, three seeds.

### 15.14 Stage 1 no-context capture and TRANSFER RESULT (2026-10-11, about 20:50 UTC = 2026-10-12 02:20 IST)

- Kaggle `laya-stage1-nocontext-run` v1: 59 of 59 no-context captures ok, in-run validation passed (`results/raw/stage1-nocontext-run-20261011T203905Z`).
- Transfer test (`results/raw/stage1-transfer-v1-20261011T203921Z/REPORT.json`, pre-registered; scorers fitted ONLY on dataset v2 no-context features, tested on Stage 1; pretrained file unchanged). PRIMARY = the 48 Stage 1 pools on trajectories absent from v2 (chance 26.9%):

| Scorer | inside v2 (cross-validated, no context) | Stage 1, new trajectories | 95% interval |
|---|---|---|---|
| Linear probe on Qwen option states | 84.6% | 62.5% | 47.7-77.3 |
| Head, adapter only (3-seed mean) | 76.7% | 57.6% | 46.3-69.4 |
| Options text | 62.6% | 47.9% | 34.0-62.2 |
| Action-history prior | 37.2% | 32.1% | 27.7-36.4 |
| Pretrained head, no training | 36.6% | 41.0% | 33.3-48.7 |

  Paired on the 48 pools: probe - head +4.9 points (CI -5.3 to +16.3, not significant); probe - options text +14.6 (0.0 to +29.8); head - options text +9.7 (not significant).
  The drop is not confined to new languages: languages already in v2 (Python, Go, TypeScript, PowerShell; 19 pools) probe 63.2%, head 57.9%; new languages (C#, Ruby, Rust, Java, Bash, Zsh; 29 pools) probe 62.1%, head 57.5%. By tool: Bash (32) probe 71.9%, head 60.4%; multi-call (9) probe 33.3%, head 40.7%. Small cells (Java 6: probe 16.7%; Python 5: 100%) are too small to read.
  The 11 pools on v2 trajectories: probe 63.6%, head 54.5%, options text 72.7%.
- Reading: every scorer fitted on v2 loses about 20 points on Stage 1. The 85% measured inside v2 was partly specific to v2 (its tasks, command habits and pool make-up), not a general property; trajectory-grouped folds inside one dataset did not reveal that. All scorers stay well above chance (27%), and the linear probe remains the best single scorer, though its lead over the head is no longer significant. No context was captured, so nothing here speaks to context. Agreement with Fable only.

### 15.15 Remaining Fable decisions: generation started (2026-10-11, about 21:05 UTC = 2026-10-12 02:35 IST)

User (chat): GO for all 471 remaining eligible Fable decisions now with the proven pipeline (session-conventions prompt, 8 calls per decision, GLM + Muse); hold the new batch out for ONE transfer measurement before it joins training; the 268 first-step decisions stay out; NO linear-probe residual change to the head; next training improvement = adapter alignment on ready-made typed decisions (candidates named by the user: AgentToolDecisions-180K, Open-Jev; NOT yet verified, downloaded or approved here; the 2026-10-01 rule "no OpenJev" needs the user to confirm it does not cover using that dataset), starting with 5,000 examples, adapter only, pretrained tensors frozen.
Batch `results/raw/hard-batch-v3-conv-20261011T204850Z` (mode `stage2`): 471 decisions, 203 trajectories, 53 of them examined in earlier offline experiments (training use only); single-tool 435, multi-call 36; Python 176, Go 81, C# 66, PowerShell 45, TypeScript 43, Java 26, Ruby 12, Rust 10, others 12; 3,768 planned sends, cap 5,200; leak re-check 226 of 1,378 recorded argument texts also present in a request (earlier-session text, as before).


### 15.16 Adapter alignment to the original model: decisions and interface findings (2026-10-11, about 21:20 UTC = 2026-10-12 02:50 IST)

User (chat): GO for a small original-encoder alignment experiment with the checkpoint's own dataset `S4MPL3BI4S/coding-decision-cases` (download
approved; start with 1,000-3,000 TRAIN examples across the four families, native rendering); Qwen is never fine-tuned and may be replaced later; the
31 pretrained tensors stay frozen for this experiment (NOT a permanent rule); Open-Jev excluded under the 2026-10-01 rule;
`MaziyarPanahi/AgentToolDecisions-180K` is a later optional source (metadata audit only, no download); no residual, no architecture change, no broad
sweep; a low teacher score on Fable pools must not be used to reject alignment.
- Downloaded: `data/external/coding-decision-cases` (revision 2ebe9040, 114 MB, git-ignored): train 83,656 / val 4,751 / test 14,344 rows; fields id,
  family, source, group, split, state, questions, gold, meta; gold carries the label, a probability distribution and for some questions teacher
  probabilities. The card declares no licence; sources are mixed (CodeUltraFeedback, When2Call, SWE-agent-trajectories, AgentJudgeBench, routerbench).
  Private experiment only.
- Original model on this machine: `laya` 0.3.22 (`laya-audit/external-laya`) loads `models/coding-decision-agent` with its own ModernBERT encoder on
  CPU in 6-9 s and answers the native routing questions with probabilities. Its limits (model card): 2,048 input tokens in total, 320 for options, at
  most 8 options.
- **Interface finding (`laya/common.py` DecisionModel.forward against `decision_head/head.py`):** in the original model the two pretrained head layers
  run over the WHOLE token sequence (up to 2,048 encoder states plus the type embedding) and each option is read at its marker token, which the
  pretrained scorer MLP then scores. In our Qwen head the same two layers run over a short sequence of N+1 vectors (one query vector, one vector per
  option). There is no one-to-one "head input" to match. Two well-defined alignment targets exist: (a) the 1,024-d vector the pretrained scorer receives
  for each option (original: the marker state after the head layers; ours: the option vector after the head layers and `out_norm`), and (b) the option
  probability distribution on the native question. Both will be captured.
- **Teacher probe on Fable pools (compatibility only, `results/raw/teacher-compat-20261012/TEACHER_COMPAT_123.json`):** the original model, asked
  "Which next action should the coding agent take now?" with the pool's options, ranks Fable's action first in 30.1% of the 123 v2 pools (chance 29.6%).
  The inputs had to be cut to its window: 108 of 123 contexts were longer than the state window and 174 options in 69 pools were shortened. So this is
  chance-level on a question it was never trained on with most of the evidence removed; by the user's rule it is not a reason to reject alignment, and it
  is not evidence for it either. 5.8 s per pool on CPU.


### 15.17 Alignment capture launched (2026-10-11, 21:20 UTC = 2026-10-12 02:50 IST)

- Local check of the original model on its native questions (`scratchpad/teacher_hook.py`): one call of the pretrained scorer per question, its input has
  shape (1, number of options, 1024) in the question's option order, and the softmax of its raw outputs matches the reported probabilities up to the
  per-type temperature. So the per-option scorer input and the option logits can be recorded exactly with a forward hook.
- Dataset facts: train families tool_call 45,893, code_review 29,989, routing 7,474, agent_trace 300; question types choice and score; 2-8 options;
  state text median about 2,000 characters, maximum 5,631 (far shorter than Fable contexts).
- Bundle `results/raw/align-bundle-20261011T211922Z` (`tools/align_bundle.py`, seed 20261012): 2,988 TRAIN items and 600 VAL items, one item = one
  (example, question), 166 per (family, question) cell in TRAIN, one item per source group inside a cell; 69 VAL items dropped because their source
  group also occurs in the TRAIN sample; no Fable data in the upload (3.1 MB incl. the laya 0.3.22 package and its licence).
- Kaggle: private dataset `sangeetasangamkar/laya-align-v1`, kernel `sangeetasangamkar/laya-align-capture-run` v1 (`kaggle/align_capture_t4.py`):
  stage `teacher` (original model, hook on the scorer; falls back to the CPU if the T4 cannot run it) and stage `qwen` (frozen Qwen, fable_capture_v1
  representation, prefix = state + "Question: <instructions>" + the protocol's query cue, option = "<key>: <criterion>"), each in its own process, part
  files every 250 items, resumable. Nothing is trained on Kaggle.
- Remaining Fable generation at 21:20 UTC: 2,965 of 3,768 responses saved.
- Still to write: the CPU alignment trainer (adapter only, 31 pretrained tensors frozen; target A = the scorer-input vector per option, target B = the
  option probability distribution), its pre-registration, and the Fable comparison (aligned start against the current start, same data and folds).
  Open point: Stage 1 has only no-context features; a fair with-context Fable comparison across batches needs a real-context capture of Stage 1.

### 15.18 Sandbox language coverage re-checked; TypeScript now really parsed; 4 kept options fail it (2026-10-11, about 21:35 UTC = 2026-10-12 03:05 IST)

User (chat): check that no language is missing in the WSL sandbox for judging the incoming decisions.
- File types written by Fable in the batch now being generated (471 decisions): .py 54, .go 23, .cs 20, .java 16, .ts 16, .ps1 10, .rb 3, .sh 2, .rs 2, .cpp 1, .hpp 1. Every one has a parser. No further toolchain install is needed.
- File types seen in all earlier pools without a parser: .mjs 8, .tsx 6, .psm1 4, .h 3 (plus .md 7 and .txt 4, which are not code). Added, no install: .mjs / .cjs (`node --check`), .psm1 (PowerShell parser), .h (g++ syntax-only as C++). .tsx is still unchecked (it would need the TypeScript compiler package).
- **TypeScript:** `node --check file.ts` rejects every type annotation, so the old .ts check failed valid files and was never enforced. Replaced by Node's own TypeScript parser (`tools/sandbox_tscheck.js`, copied to /opt/toolchains/tscheck; no package installed). Valid file passes, broken file fails, a file using enum / namespace is reported "not checkable" and passes.
- Re-check of the frozen sets with the new TypeScript check (`results/raw/ts-recheck-20261012.json`): all 12 of Fable's own TypeScript files pass, so the check can be enforced. **4 kept alternatives fail it: 3 in dataset v2 (ts-formfield:1, ts-input-mask:3, ts-qs-codec:1) and 1 in Stage 1 (ts-config-merge3:3).** Both judges had accepted all four. ts-qs-codec:1 is the one the first hand audit flagged. Two of the four pools have only 2 alternatives, so they fall below the minimum once the option is removed.
- To do at the next freeze (no edit of the frozen sets): corrected versions of v2 and Stage 1 without these 4 options; the pilot and transfer figures were computed with them in (3 of 307 and 1 of 164 alternatives).

### 15.19 Alignment capture complete; first alignment fold (2026-10-11, about 22:00 UTC = 2026-10-12 03:30 IST; NOT the final result)

- Kaggle `laya-align-capture-run` v1: 3,588 of 3,588 items for both stages; the original model ran on the T4 (252 s, softmax of the recorded logits matches its reported probabilities on all items); Qwen capture 838 s, validation passed (repeat identical, plain-vs-packed cosine 0.999999). Output: `results/raw/align-capture-run-20261011T212044Z`.
- Trainer `tools/adapter_align_v1.py`, pre-registration written before any run in `results/raw/adapter-align-v1-20261011T214159Z/PREREG.json`. Adapter only; the 31 pretrained tensors and Qwen frozen.
- Fold 0 (dry run, one alignment seed): on the 600 held-out native items the ORIGINAL model is right on 78.3% (gold label). The Qwen-fed head after alignment: arm A (scorer-input vectors) agrees with the original model's top option on 69.2%, gold accuracy 64.3%, cosine to its scorer-input vectors 0.984; arm B (option distribution) 76.7% agreement, 70.7% gold accuracy; arm AB 75.2% / 69.5%. So the adapter can make the frozen pretrained head work on Qwen features for its native questions.
- Fable, fold 0 only (25 pools, 3 seeds each): scratch start 76.0%, aligned A 70.7%, B 69.3%, AB 65.3% after the same adapter-only Fable training; before any Fable training the aligned starts all give 48% on these pools. One fold is not a result; the five-fold run is in progress (about 16 minutes per fold).
- Remaining Fable batch: generation complete (3,768 responses, 220 rate-limit rejections re-queued), 352 pools screened of 471 decisions, GLM review running in the unattended chain (`tools/stage2_chain.sh`, log `results/raw/hard-batch-v3-conv-20261011T204850Z/chain.log`).


### 15.20 Adapter alignment RESULT, five folds (2026-10-11, 23:01 UTC = 2026-10-12 04:31 IST; `results/raw/adapter-align-v1-*/REPORT-20261011T230124Z.json`, pre-registered, pretrained file unchanged)

Adapter only in every run; Qwen and the 31 pretrained tensors frozen. Alignment data: 2,988 TRAIN / 600 VAL native items of the checkpoint's own dataset.

**Native questions, 600 held-out items (mean over the 5 fold-specific alignments; chance 29.0%):**

| System | gold accuracy | agrees with the original model's top option |
|---|---|---|
| Original model (own encoder) | 78.3% | - |
| Qwen + UNALIGNED adapter (3 seeds) | 18.7-25.8% | 16.2-24.5% |
| Qwen + adapter aligned on the option distribution (arm B) | 70.8% | 76.2% |
| Qwen + adapter aligned on both (arm AB) | 69.9% | 74.9% |
| Qwen + adapter aligned on the scorer-input vectors (arm A) | 65.0% | 70.2% |

So an adapter alone makes the frozen pretrained head usable on Qwen features for its own tasks: from chance level to within about 8 points of the original.

**Fable ranking, dataset v2, 123 pools, 5 trajectory-held-out folds, real-context features, adapter-only Fable training, 3-seed means (chance 29.6%):**

| Start of Fable training | before Fable training | after | 95% interval | seeds |
|---|---|---|---|---|
| scratch (the pilot's method) | 36.6% | 77.2% | 70.8-83.3 | 71.5 / 78.9 / 81.3 |
| aligned, arm A | 48.8% | 80.0% | 73.5-86.1 | 78.0 / 82.1 / 79.7 |
| aligned, arm B | 47.2% | 82.4% | 75.9-88.5 | 83.7 / 82.9 / 80.5 |
| aligned, arm AB | 48.8% | 77.0% | 70.2-83.2 | 76.4 / 77.2 / 77.2 |
| reference: linear probe (pilot) | - | 84.6% | 78.4-90.6 | - |

Paired against scratch on the same pools: arm B +5.2 points (CI +0.6 to +9.8, excludes 0); arm A +2.7 (-2.0 to +7.1); arm AB -0.3 (-5.5 to +5.1). By cohort, arm B: GLM + Gemma 85.2% (scratch 79.0%), GLM + Muse 78.6% (74.8%). The scratch figure reproduces the pilot's 77.2% exactly.

Reading under the pre-registration: arm B "helps" by the stated rule (interval excludes 0). Cautions that go with it: three arms were tried and one is significant with a lower bound of +0.6; the test is cross-validation inside dataset v2, which overstated generalisation before (84.6% inside v2 became 62.5% on Stage 1 for the probe); one alignment seed; the head is still below the linear probe (82.4% against 84.6%, not tested for significance here). Alignment alone, before any Fable training, already lifts Fable ranking from 36.6% to about 48% and makes the seeds much more alike. Confirmation needs a held-out batch with real-context features (Stage 1 or the new batch), which needs a Kaggle capture.

### 15.21 Network interruption during the GLM review of the remaining batch (2026-10-11, 22:10-22:28 UTC)

GLM first attempt on the 352 packets: 205 usable; 120 HTTP 200 with an empty body (stream cut, finish_reason none; 117 with no content at all); 24 name-lookup failures and 2 closed connections (status -1); 1 HTTP 503. All 147 failures fall in the 22:10-22:28 window; the 147 answers before it were all usable. The chain retries the 147 once (`pilotr`, 5 in flight per key; `retry` in `pair_pilot_v1.py` raised from 1 to 5). Muse reviewed 309 packets (148 GLM-eligible, 14 sampled, 147 with no usable GLM answer). The pair gate's first-attempt response-failure check (< 5%) will read 42% for GLM on this batch; the cause is the local connection, and whether the batch counts is the user's decision. Disk D: 7.9 GB free, RAM 1.1 GB free at 22:36 UTC.


### 15.22 Remaining-decisions batch scored, assembled and audited; NOT frozen, user decisions pending (2026-10-12, 00:10 UTC = 05:40 IST)

Batch `results/raw/hard-batch-v3-conv-20261011T204850Z` (471 decisions): 3,768 generation responses; 352 pools passed the objective screen (full parser set,
TypeScript really parsed); review dir `.../build-20261011T214554Z/pair-review-rest-20261011T214941Z`.
- **Muse outage and resend.** Muse first pass 22:28-22:36 UTC: about 70 replies cut by the same network trouble as GLM. The chain's retry at 22:39-22:42 hit
  HTTP 404 "function not found" on all 14 keys (138 x 404, 6 x 500): the endpoint was unavailable, nothing was processed. At 23:24 UTC Muse answered a
  probe; the 144 packets were sent again (a request rejected before processing is not a used retry, as with a rate-limit rejection). Every attempt is kept
  (`pilotr_failed*`). Final: GLM 351 of 352 usable, Muse 302 of 309.
- **Pair gate on this batch (301 pools scored by both judges):** known-bad options accepted by both 0 of 316 (301 off-task + 15 premature edits; Muse alone
  accepted 1 off-task and 1 premature edit, GLM none); Fable's action accepted by both 266 of 301 = 88.4%; near-duplicates 136 of 137; no truncation.
  **First-attempt response failures: GLM 41.8%, Muse 46.6% (limit 5%): this check FAILS as written.** Cause: the local connection between 22:10 and
  22:30 UTC and the Muse outage, not judge behaviour. `pair_passed` is false for that reason alone. Whether the batch counts is the user's decision.
- **Levels bug fixed:** `PARSER_LEVELS_v2.jsonl` indexed options by their position in the UNSCREENED pool while assembly uses the screened position, so a
  level could be attached to the wrong option whenever screening had dropped one. `PARSER_LEVELS_v3.jsonl` uses the screened index; assembly reads v3.
  Stage 1 re-assembled with v3 and the TypeScript parser: 58 pools (`strict-20261012T000051Z`); the one TypeScript-failing option (ts-config-merge3:3) is
  dropped and its pool lost; no other kept option fails an enforced parser. The frozen Stage 1 set (59 pools) is therefore superseded at the next freeze.
- **New batch strict pools: 162** (`strict-20261012T000054Z`; 105 trajectories, 425 alternatives; 169 before the same-as-Fable filter, which dropped 9
  options and 7 pools). Languages: Python 59, Go 31, C# 21, PowerShell 14, Java 13, TypeScript 10, Ruby 5, Rust 5, Bash 2, JS 1, Zsh 1. Fable tool: Bash 98,
  Write 32, Edit 15, WebFetch 6, multi-call 5, Read 5, WebSearch 1. Yield 162 of 471 = 34%. 88 of the pools (56 trajectories) are on trajectories in
  neither v2 nor Stage 1.
  Verification level of the 425 alternatives: 359 no file content, 59 parser-verified file content, 6 file content without a parser, **1 fails a parser
  that is NOT enforced in this batch** (go-shiplog:2: the Go check is unenforced because 1 of Fable's 20 Go files fails it; likewise PowerShell 9 of 10 and
  C++ 0 of 1). Options-only text detector on these pools: 62.4% against 28.6% chance.
- **Blind hand audit v4** (`results/raw/laya-audit-v4-rest-20261012/`, 24 options, verdicts before the key): of 18 kept: 11 plausible, 3 unclear,
  2 implausible (a clean-up plus an echoed final report; a re-read of a file just written), 2 the same as Fable's action in effect (the test command run
  once where Fable runs it twice; the same test command with a default flag). Of 6 not admitted: 2 implausible, 2 plausible, 2 unclear.
  This batch is weaker than the earlier audits (4 of 18 kept options questionable against 1 of 18 in Stage 1).
- **Open decisions for the user:** (1) accept the batch despite the first-attempt response-failure check; (2) drop the Go option that fails an unenforced
  parser, or make "a parser that at least 95% of Fable's files pass" enforceable; (3) extend the same-as-Fable filter (`X && X`, a default flag);
  (4) real-context Kaggle capture of Stage 1 and this batch to confirm the alignment gain on held-out pools.


### 15.23 Dataset v3 FROZEN with roles; held-out comparison prepared (2026-10-12, 05:57 UTC = 11:27 IST)

User decisions (chat, 2026-10-12): GO on all four items with safeguards. (1) Accept the remaining batch under a documented INFRASTRUCTURE EXCEPTION; the original
gate result stays recorded as failed. (2) Drop the Go option; a parser is enforced when qualified on Fable's own complete files; only complete, supported files are
parsed; partial edits are never judged syntax-invalid. (3) The same-as-Fable filter also covers a repeated command, for the pool's current action only (earlier
retries in the trajectory are untouched). (4) One real-context capture of the later pools. Critical safeguard: models are never scored on pools they trained on;
roles are recorded per pool; the 268 first-step decisions stay "not processed".

- **Infrastructure exception** (`.../pair-review-rest-20261011T214941Z/INFRASTRUCTURE_EXCEPTION.json`): first attempts classified one by one. GLM: 205 usable,
  147 infrastructure (lost connection, empty or cut stream, HTTP 503), 0 judge-caused. Muse: 165 usable, 139 infrastructure (72 HTTP 404, 65 empty or cut
  streams, 2 HTTP 400), 5 judge-caused (1.6%: letters mismatch 2, malformed 2, bad verdict 1). All substantive checks pass. Condition met.
- **Rules in code:** `bash_core` collapses a repeated command (`X && X`, `X; X`); parser levels v4 (`PARSER_LEVELS_v4.jsonl`, `PARSERS_V4_REPORT.json` per build):
  enforced when every Fable file passes, or at least 5 files and at least 95% pass (Go 19 of 20 in the last batch is now enforced; PowerShell 9 of 10 and C++ 0 of 1
  are not). The three v2 builds were re-parsed with the full parser set (`SYNTAX_v3_parsers.jsonl`).
- **Re-assembled strict pools:** batch v2 68 (3 TypeScript options dropped: ts-formfield:1, ts-input-mask:3, ts-qs-codec:1), conventions-40 16, conventions-95 36,
  Stage 1 58 (ts-config-merge3:3 dropped), remaining batch 157 (go-shiplog:2 dropped; the repeated-command filter removed 10 more options).
- **`results/raw/laya-fullaction-set-v3-20261012T055420Z` (FROZEN_SET.json sha256 725cd39dd27f0232de9ebd700a91042cec993505e8fedd8117aa024ceed4d8f4; `tools/laya_fullaction_freeze_v3.py`):**
  **335 pools, 866 alternatives.** Per-option verification: 866 of 866 alternatives accepted by both judges of their cohort, in the screened pool, not the same
  command as Fable's; 335 of 335 Fable actions identical to the screened reference. Judge cohorts: GLM + Gemma 68 pools, GLM + Muse 267.
  Roles, fixed before any model saw the later pools:

  | role | pools | trajectories | alternatives |
  |---|---|---|---|
  | train (the pools behind dataset v2; they keep their v2 fold) | 120 | 94 | see manifest |
  | train_extension (later pools on a train trajectory) | 54 | 35 | see manifest |
  | heldout_transfer (later pools on trajectories in no train pool) | 161 | 98 | 433 |

  No held-out trajectory is in a training role. Options-only text detector: train 70.0%, extension 65.4%, held-out 64.6% (chance 28-30%).
  Eligibility inventory in the manifest: 1,342 eligible; 1,074 sent to the generator; **268 first-step decisions NOT processed**; VAL / CAL / TEST have no pools.
  The blind audits are unchanged by the new filters except that one audited near-copy (test command run twice) is now removed by rule.
- **Real-context capture:** subset `results/raw/laya-fullaction-v3-capture-subset-20261012T055500Z` (215 non-train pools), bundle
  `results/raw/level12-bundle-20261012T055511Z`, kernel `sangeetasangamkar/laya-v3-context-capture-run` v1 (variant B only), started 05:56 UTC. One pool
  (csharp-azure-arm-lro:7) exceeds the 24,000-token prefix limit and will be excluded from the comparison, not truncated.
- **Held-out comparison pre-registered before any held-out feature was downloaded:** `tools/align_transfer_v1.py`, `results/raw/align-transfer-v1-20261012T055725Z/PREREG.json`. Fit = the 120 train
  pools (folds 1-4, selection on fold 0); test = the held-out role; PRIMARY = aligned (option distribution) minus fresh start, 5-seed means, trajectory bootstrap.


### 15.24 Real-context capture of the later pools; held-out comparison: alignment does NOT transfer (2026-10-12, 07:06 UTC = 12:36 IST)

- **Capture.** Kernel `sangeetasangamkar/laya-v3-context-capture-run`. Version 1 aborted in its validation sample with 0 rows: the sample takes the five longest
  contexts and one is the over-limit pool csharp-azure-arm-lro:7 (25,501 tokens, limit 24,000, unchanged); the other five samples passed
  (`results/raw/v3-context-capture-run-20261012T055713Z`). The pool was removed from the upload (subset
  `results/raw/laya-fullaction-v3-capture-subset-20261012T060040Z`, 214 pools; bundle `results/raw/level12-bundle-20261012T060047Z`). Version 2 started before the new
  dataset version existed and aborted the same way (`results/raw/v3-context-capture-run-v2aborted-20261012T060639Z`). **Version 3: 214 of 214 pools captured, variant B,
  all status ok, about 44 minutes** (`results/raw/v3-context-capture-run-20261012T065106Z`).
- **Held-out comparison** (`tools/align_transfer_v1.py`, pre-registered; `results/raw/align-transfer-v1-20261012T055725Z/REPORT-20261012T070534Z.json`).
  Fit on the 120 train pools only (97 fit, 23 for checkpoint selection); scored on 160 held-out pools, 98 trajectories in no train pool (1 pool excluded: over the limit).
  Agreement with Fable's recorded action, 5-seed means, 95% intervals from trajectory bootstrap. Chance 28.1%.

  | scorer | held-out top-1 | 95% interval | each seed |
  |---|---|---|---|
  | linear probe on the frozen Qwen state (mandatory baseline) | 75.6% | - | - |
  | LAYA head, fresh adapter | 63.1% | 56.8 - 69.4 | 60.0, 61.3, 66.9, 62.5, 65.0 |
  | LAYA head, adapter aligned to option distribution (PRIMARY) | 58.4% | 51.2 - 65.4 | 51.9, 56.3, 60.6, 62.5, 60.6 |
  | LAYA head, adapter aligned to scorer-input vectors | 54.9% | 47.9 - 61.6 | 57.5, 50.0, 47.5, 60.0, 59.4 |

  **PRIMARY: aligned minus fresh = -4.75 points, interval -9.6 to -0.3: alignment made held-out agreement WORSE.** Secondary: vector alignment -8.25 (-13.8 to -3.2);
  probe minus fresh head +12.5 (+7.3 to +17.6); probe minus aligned head +17.3 (+10.9 to +23.7). Before any Fable training: fresh 33.0%, aligned 43.1% / 43.8%.
  Same-trajectory extension pools (54, not primary): fresh 71.9%, aligned 61.9% / 62.2%, probe 77.8%.
- **Consequences.** (1) Keep rule applied: the Fable-only (fresh) adapter stays; the aligned adapters are not used; the pretrained head is unmodified. (2) The +5.2 points
  measured inside dataset v2 (15.2x) did not hold on new trajectories; it is withdrawn as evidence that alignment helps. (3) On new trajectories with real context the
  linear probe beats the LAYA head by 12.5 points; the head, as trained here (adapter only, 120 pools), is not yet better than the mandatory baseline.
- **Limits.** One alignment seed; 120 training pools; agreement with Fable, not task success; canonical VAL / CAL / TEST untouched. The held-out role has now been
  read once: any further tuning must not be scored on it as if unseen.


### 15.25 Pretrained LAYA head fine-tuned on all qualified Fable pools (2026-10-12, 08:05 UTC = 13:35 IST)

User (chat, 2026-10-12): GO to fine-tune the pretrained head on all the quality Fable data; the open issues are handled later. Also: the local GPU and Kaggle may
be used (the no-local-GPU rule is lifted). This run used the CPU: the frozen trainer has no GPU path and the whole job took about 4 minutes.

- `tools/laya_finetune_all_v1.py`, `results/raw/laya-finetune-all-v1-20261012T080145Z/PREREG.json` (written before the run), `results/raw/laya-finetune-all-v1-20261012T080145Z/REPORT-20261012T080534Z.json`. 334 of 335 pools (one over the capture limit),
  192 trajectories, 5 trajectory-grouped folds (57 / 66 / 61 / 82 / 68). Adapter only, fresh start, 300 steps, unknown weight 0.5. Pretrained checkpoint unchanged (sha256 checked).
- **Final adapters:** `results/raw/laya-finetune-all-v1-20261012T080145Z/adapters/final-s0.pt`, `final-s1.pt`, `final-s2.pt` (fit 277 pools, checkpoint selection on 57; best steps 150 / 200 / 175; fit top-1
  91.3% / 92.8% / 92.1%; selection-fold top-1 82.5% for each). They hold the adapter tensors and input statistics only.
- **Cross-validation (in-dataset, one seed, chance 28.9%; agreement with Fable):**

  | pools | head | linear probe | probe minus head |
  |---|---|---|---|
  | all 334 | 80.2% (75.5 - 84.8) | 86.2% (82.4 - 89.8) | +6.0 (+1.9 to +10.1) |
  | former train role, 120 | 84.2% | 89.2% | +5.0 (0.0 to +10.9) |
  | former held-out role, 160 | 81.3% | 83.8% | +2.5 (-2.0 to +6.8) |
  | train_extension, 54 | 68.5% | 87.0% | +18.5 (+5.6 to +33.3) |

  Per fold head / probe: 82.5 / 84.2, 81.8 / 89.4, 83.6 / 82.0, 68.3 / 85.4, 88.2 / 89.7.
- **Reading.** On the former held-out pools the head rose from 63.1% (fitted on 120 pools of other batches, 15.24) to 81.3% (fitted on about 200 pools that include
  other trajectories of the same batches). More data of the same kind helps a lot, and the gap to the probe shrinks to 2.5 points there. The probe is still ahead overall.
  Fold 3 and the extension pools are the weak spot (head 68%).
- **Limits (unchanged, to be handled later by the user's decision).** These figures are in-dataset and include pools already read; the earlier in-dataset 85% fell
  to 62% on new batches. No untouched Fable pool remains: an independent claim needs new pools. Alternatives are Nemotron-written (options-only detector 65-70%).
  Nothing here is task success. The A5 benchmark was not run.


### 15.26 Evaluation on canonical VAL trajectories: accuracy holds, context helps, probe still ahead (2026-10-12, 11:20 UTC = 16:50 IST)

User (chat, 2026-10-12): GO for four steps: verify the adapters, diagnose the extension pools, process all eligible VAL decisions for evaluation only, paired
real-context / no-context capture. Safeguards: TEST untouched, pre-register, no seed selection, no retraining. Later the same day: the local GPU and Kaggle may
both be used; Muse may start before the last GLM packet returns.

- **Step 1** `results/raw/laya-finetune-all-v1-20261012T080145Z/ADAPTER_VERIFICATION.json`: the three adapters load, file hashes match, all scores finite on 334
  pools, the pretrained tensors are identical to the checkpoint.
- **Step 2** `.../EXTENSION_DIAGNOSIS.json` (read-only): the weak extension figure sits in one fold (fold 3: 20 pools, head 40%, probe 70%; other folds 76%); 6 of
  17 misses come from two trajectories; reweighting the other pools to the same tool and step mix gives 81.4% against 68.5%. No pool type explains it.
- **Step 3, VAL pools (evaluation only).** `tools/hard_batch_v3_conv.py` mode `val`: 227 eligible VAL decisions, 49 first-step decisions NOT processed (same rule
  as TRAIN), **178 sent**, 51 VAL trajectories, none of them a TRAIN trajectory. Batch `results/raw/hard-batch-v3-conv-20261012T081844Z`.
  Generation 1,424 of 1,424 completed (sends 550 + 1,186 including rate-limit rejections and HTTP 504; another session raised the per-run send budget by 3,000
  after the first pass stopped at it, logged in the ledger; 7 in-flight calls of a run that ended early were marked `.interrupted_run3` and re-sent once).
  Screen 129 pools. GLM 129 of 129 (3 unparseable, retried once), Muse 104 of 104 (started before the last GLM packet returned, at the user's request; that packet was
  treated as unusable, so Muse reviewed it too). **Pair gate passed on every check:** 0 of 114 known-bad accepted by both, Fable's action accepted by both 94 of 104,
  near-duplicates 50 of 51, unusable first replies GLM 2.3% / Muse 0%. Parser levels: no failure among 437 alternatives. **53 strict pools, 36 trajectories,
  134 alternatives; 134 of 134 verified per option.** Set: `results/raw/laya-fullaction-val-v1-20261012T104239Z`.
- **Blind hand audit** `results/raw/laya-audit-v5-val-20261012` (24 options): of 18 kept alternatives 10 plausible, 3 unclear, 0 implausible, **5 the same as
  Fable's action in effect** (Fable's test command plus a redundant Read, or without a leading `chmod +x`); the same-as-Fable filter does not catch these. No rule
  was changed after seeing this (the comparison was pre-registered); a pick of such an option counts as a miss for every scorer.
- **Step 4, capture.** Kaggle `laya-val-capture-run` sat in the queue 30 minutes, then ran: 53 of 53 pools in variants B and A
  (`results/raw/val-capture-run-20261012T111837Z`). Meanwhile a local-GPU capture was built (`tools/val_capture_local.py`: layer weights read from disk per forward,
  because the PC had 0.9 GB free RAM). Parity gate fixed before running: 12 TRAIN pools captured locally against Kaggle: worst cosine 0.99999 (limit 0.999), the
  three adapters' top choice identical on 12 of 12 (`results/raw/local-capture-parity-*/PARITY.json`). The scored features are Kaggle's, as pre-registered.
- **Result** (`tools/val_eval_v1.py`, pre-registration `results/raw/val-eval-v1-20261012T082055Z/PREREG.json`, report `REPORT-20261012T111926Z.json`). 53 pools,
  36 trajectories, chance 29.3%, agreement with Fable, 95% intervals by trajectory bootstrap. Nothing was fitted on VAL.

  | scorer | real context | no context |
  |---|---|---|
  | LAYA head, mean of the three saved adapters | **79.2%** (67.9 - 89.3); seeds 79.2 / 81.1 / 77.4 | 64.2% (52.0 - 76.3); seeds 71.7 / 58.5 / 62.3 |
  | linear probe fitted on the 334 TRAIN pools | **94.3%** (87.3 - 100) | 81.1% (69.0 - 90.9) |
  | options-only text baseline | 67.9% (53.9 - 81.1) | - |

  **Question 1:** in-dataset 80.2%, VAL 79.2%; no collapse (the interval is not below 70%). **Question 2 (primary):** head real minus no context
  **+15.1 points (+6.3 to +24.1)**; top choice changed on 9 / 12 / 16 pools per seed, gains 5 / 12 / 11, losses 1 / 0 / 3. Head minus options-only text +11.3
  (-4.1 to +28.7, not significant). Probe minus head +15.1 (+6.2 to +24.5). Probe real minus no context +13.2 (+5.6 to +22.5).
  By tool: Bash (33) head 87.9% / probe 100% / text 81.8%; Write (5) 80% / 100% / 60%; other (15) 60% / 80% / 40%.
- **Reading.** (1) The fine-tuned head keeps its accuracy on trajectories it never saw. (2) Removing the context costs 15 points for the head and 13 for the probe,
  so the scorers do use the situation; this differs from the 123-pool pilot, where models trained per variant showed no difference. The no-context figure also
  carries a train/test mismatch (the models were fitted with context). (3) The linear probe is still clearly better than the head. (4) A probe at 81% with no
  context means much of the signal is still in the option representation itself (alternatives are Nemotron-written). (5) 53 pools: wide intervals.
  Agreement with Fable, not task success. The A5 benchmark was not run. Canonical CAL and TEST untouched.


### 15.27 Read-only diagnosis of the head-versus-probe gap on VAL (2026-10-12, about 11:45 UTC = 17:15 IST)

User (chat): GO for one short read-only diagnostic; first check whether the gap is decision quality or labelling; no fine-tuning, no change to the head, no new
VAL pools, no capture; the original VAL result stays unchanged and corrected-label figures are kept separate.
`tools/val_gap_diag_v1.py`, `results/raw/val-gap-diag-v1-20261012/` (`EQUIVALENCE_VERDICTS.json` written before any prediction was read, `DIAGNOSIS.json`).
Also recorded: the local capture of all 53 VAL pools matches Kaggle (worst cosine 0.999948, same top choice 53 of 53 for each adapter;
`results/raw/val-capture-local-20261012T111423Z/PARITY.json`).

- **Labels.** Reading only the option texts: **32 of 134 alternatives (24%) are Fable's action in effect**, in 24 of 53 pools. 14 strict (Fable's complete
  action plus a read-only step) and 18 loose (same main command with a set-up, lint or timing step added or dropped). One auditor.
- **Sensitivity (separate from the result of record):**

  | labels | head (3 adapters) | probe | text only | probe minus head |
  |---|---|---|---|---|
  | original | 79.2% | 94.3% | 67.9% | 15.1 |
  | strict equivalents count as correct | 80.5% | 94.3% | 69.8% | 13.8 |
  | strict and loose count as correct | 83.6% | 94.3% | 71.7% | 10.7 |

  About 4 of the 15 points are a labelling artefact; about 11 remain.
- **The probe never picks an equivalent alternative:** 50 exact, 3 other, 0 equivalent, although 24 pools contain one that differs from Fable's action only by a
  redundant read or a dropped `chmod`. In those 24 pools it picks Fable's exact form 95.8% of the time. Telling such pairs apart is not a better coding decision;
  it is recognition of how Fable writes an action. So part of the probe's 94.3% is not decision quality, and it should not be a target for the head to imitate.
- **The head is unstable rather than uniformly wrong.** The probe is right and at least one adapter wrong in 13 pools (25 adapter misses: 5 loose equivalents,
  20 other). In 9 of those 13 at least one adapter is right; all three are wrong in 4. Mean head probability on Fable's action there: 0.38.
- **Where the head is weak:** Fable's action is the shortest option (8 pools): head 54%, probe 88%. Edit and multi-call / other tools (15 pools): head 60%, probe 80%.
  Pools the text-only baseline gets wrong (17): head 67%, probe 94%.
- **Diagnosis.** The 15-point gap has three parts: (1) about 4 points of equivalent alternatives counted as misses; (2) seed-to-seed instability of the adapter on
  close calls; (3) the probe's ability to identify Fable's exact wording. How much of the remaining gap is real decision quality cannot be measured on
  Fable-against-Nemotron pools. Limits: one auditor, 53 pools, VAL has now been inspected closely (later claims need fresh pools).
- **Not done, by instruction:** no retraining, no ensemble tuned on VAL, no relabelling of the frozen set.


### 15.28 Equivalence audit of the training pools; facts for the outcome-based test (2026-10-12, about 12:10 UTC = 17:40 IST)

User (chat): stop improving Fable-matching accuracy for now; next is a small outcome-based evaluation on Qwen-written candidates; keep dataset v3 and the VAL
result unchanged and add equivalence annotations separately; audit a sample of the 866 training alternatives blind; check the saved Qwen candidate pools; no
change to the head and no fine-tuning. User corrections accepted: a loose equivalent (extra `go vet`, missing `chmod`) can behave differently, so the strict
figure is the safer one; the probe's preference for the exact action does not prove it uses no decision signal.

- **Training-set audit** (`results/raw/laya-v3-equivalence-audit-20261012/EQUIVALENCE_AUDIT.json`, sidecar only, dataset v3 untouched). 60 of 335 pools at random,
  161 alternatives, option texts only, no prediction or verdict on screen: **11 strict equivalents (6.8%), 30 loose (18.6%), 41 together (25.5%), in 23 of 60 pools.**
  All but one sit in pools where Fable's action is a Bash command (a test run with a lint, set-up or clean-up step). train 21 of 70, heldout_transfer 17 of 68,
  train_extension 3 of 23. VAL had 14 strict and 18 loose of 134 (23.9%). One auditor.
- **Saved Qwen candidates for Fable decisions:** 2 decisions x 4 samples only (`level12-run-v2-cancelled-20261011T175252Z/out/level2/gen-1.jsonl`); the other
  Level 2 runs saved nothing usable. The proposal folders of 2026-10-06/07 belong to the earlier pipeline (provenance not re-verified). New generation is needed.
- **Replay-ready sandbox states:** 141 decisions with a consistent replay and a known test command (`fable-fidelity-report-20261010/consistent_matches.jsonl`):
  118 TRAIN, 23 VAL, 0 CAL (CAL was never put through the replay check).
- **Open design points put to the user:** outcomes separate only file-changing actions; generation cost on the T4 (0.06-0.27 s per token measured) limits one
  12-hour session to about 30 decisions (estimate); CAL (needs a first read and a replay check) or the 23 VAL decisions. Nothing was started.


### 15.29 CAL replay qualification for the outcome evaluation (2026-10-12, about 12:45 UTC = 18:15 IST)

User (chat): GO - LAYA OUTCOME EVALUATION. CAL (first read today), TEST untouched; replay qualification and the count first, no generation before it; protocol
frozen before candidate outcomes; Kaggle generation capped at 12 GPU hours, pilot of 3-5 decisions x 4 Qwen candidates; compare Qwen's first proposal, the three
saved adapters (predeclared aggregation), the fixed probe and random; score by test outcome; nothing is trained or modified; CAL results are development evidence.

- **Reconstruction (no execution).** CAL has 155 decisions whose recorded action is only Write / Edit calls (`fable-reconstructible-scan-20261009T204227Z`).
  **72 pass the strict rules** (no uncertain shell command before, edited paths known, a later test command, test file known), 53 trajectories:
  py 16, ts 16, go 14, powershell 11, java 10, csharp 2, rust 2, zsh 1. `tools/fable_materialize_states.py` got a `--split` option (TEST is refused) and
  `--min-rows`; states: `results/raw/fable-replay-states-cal-csharp-go-java-powershell-py-rust-ts-zsh-strict-20261012T123831Z`.
- **Sandbox replay, twice** (`fable-fidelity-steps-20261012T123843Z`, `fable-fidelity-gate-20261012T123906Z` and `...T123949Z`; identical verdicts in both runs):
  **52 of 72 reproduce the recorded test result**: ts 16 of 16, py 14 of 16, powershell 11 of 11, java 8 of 10, go 2 of 14 (missing go.mod, as before), zsh 1 of 1,
  csharp 0 of 2, rust 0 of 2. First time Java, PowerShell and Zsh states were replayed.
- **Useful oracle** (`tools/cal_oracle_check.py`, `results/raw/cal-oracle-check-20261012T124119Z`): baseline test on the untouched state against the test after
  Fable's own action alone, each twice. **44 decisions on 35 trajectories: the test result changes** (py 14, ts 12, powershell 9, java 6, go 2, zsh 1);
  1 unstable; 7 unchanged. **27 are a clean fail -> pass (exit 1 -> 0)**; 9 are 0 -> 0 with different output, 7 are 1 -> 1, 1 is 0 -> 1. In 32 of the 44 Fable's
  action is the only file change before the recorded test. Fable's action: Write 31, Edit 13.
- **Saved Qwen proposals for these 44** (`tpu-schema-recovery-independent-20261007/valid_proposals.jsonl.gz`, the 2026-10-07 TPU run; 39,095 retained rows for
  Fable decisions in all): 152 rows, 141 distinct, 93 distinct applicable code changes; 14 decisions have none, 12 have four or more; of the 27 fail -> pass
  decisions 17 have at least two. **148 of the 152 rows stopped at the output-length limit**, so these are weak candidates; which Qwen model wrote them was not
  re-verified. Correction to 15.28: saved Qwen proposals for Fable decisions do exist; the earlier statement covered only the 4B Level 2 runs.
- Nothing was generated, captured or scored. Canonical TEST: ids were counted by split in one scan file (309 rows); no TEST content was read or materialised.


### 15.30 Cleaned labels (v4) and the head-itself experiment: joint training gives no gain (2026-10-12, 13:16 UTC = 18:46 IST)

User (chat): the saved model trains only the adapter, so "the pretrained head was fine-tuned" was not exact; clean the labels (strict equivalents only, separate
version, frozen v3 untouched), then ONE controlled experiment that also updates the head's 31 pretrained tensors, compared fairly with adapter-only; keep the
original head; check native capability before adopting; no sweep; the 268 first-step decisions stay separate; TEST untouched.

- **Cleaning** (`tools/laya_clean_v4.py`, `results/raw/laya-fullaction-clean-v4-20261012T130553Z`). Deterministic strict rule: an alternative is removed when it
  contains every step of Fable's action in order and every extra step is read-only. Against the blind hand verdicts: 23 of 25 hand-strict cases flagged, 0 flagged
  that were not hand-strict, 0 hand-loose flagged. **train_v4: 59 of 866 alternatives removed (6.8%), 31 pools dropped by the unchanged pool rule, 304 pools,
  779 alternatives, 177 trajectories** (303 with features). **Cleaned VAL: 14 removed, 7 pools dropped, 46 pools, 113 alternatives, 32 trajectories.**
- **Experiment** (`tools/laya_joint_v1.py`, pre-registered `results/raw/laya-joint-v1-20261012T130647Z/PREREG.json`, report `REPORT-20261012T131617Z.json`).
  Same data, features, folds, seeds and 300 steps. adapter_only = the recipe so far; joint = adapter for 100 steps, then adapter + the 31 pretrained tensors
  (learning rate 5e-4 / 2e-5). About 10 minutes on the CPU.

  | test | adapter only | joint (head updated) | joint minus adapter only | linear probe |
  |---|---|---|---|---|
  | **cleaned VAL, 46 pools (PRIMARY)** | 76.1% (62.2 - 88.4) | 75.4% (62.2 - 86.5) | **-0.7 points (-5.9 to +4.3)** | 89.1% |
  | original VAL, 53 pools | 75.5% | 76.1% | +0.6 (-4.5 to +5.8) | 86.8% |
  | in-dataset 5-fold, 303 pools | 77.9% | 78.9% | +1.0 (-1.8 to +3.8) | - |

  Per seed on cleaned VAL: adapter only 73.9 / 71.7 / 82.6; joint 76.1 / 76.1 / 73.9. In the joint models all 31 pretrained tensors moved (largest relative
  change 4% to 9%). The checkpoint file is unchanged (hash checked); the six final models are saved under `models/` (428 MB).
- **Decision by the pre-registered rule:** the interval includes 0, so the updated head is not preferred; **adapter-only stays**. The native-capability check on
  the original encoder was not needed and was not run. The joint models are kept on disk, not adopted.
- **Also seen (not pre-registered claims):** cleaning did not raise VAL agreement (adapter-only trained on v4 scores 75.5% on the original VAL against 79.2% for
  the v3 adapters; 31 fewer training pools, not a significant difference); the probe stays about 11 to 14 points ahead; joint seeds vary less (3 seeds only).
- **Reading.** With about 300 pools, updating 26 million head parameters at a low learning rate neither helps nor hurts measurably. This agrees with the
  123-pool pilot. It does not show that the head cannot be specialised: the data is small and VAL resolves only large differences.


### 15.31 Outcome-evaluation pilot on VAL: Qwen produced no usable code-change candidate (2026-10-12, 15:55 UTC = 21:25 IST)

User decisions (chat): pool = the 27 clean fail-to-pass CAL decisions; then, after the request builder refused CAL contexts (a deliberate guard in
`tools/fable_teacher_runner.py`: CAL / TEST prefixes are never copied), **Option B: the 23 verified VAL decisions; CAL and TEST completely protected**; first five on
Kaggle; original pre-registration kept, a versioned amendment for the pool only; no change to weights or scoring rules.

- **VAL qualification** (same checks as CAL in 15.29): 66 reconstructable edit / write decisions, 46 replay to the recorded test twice, 39 change the test result with
  Fable's action alone, **23 clean fail -> pass on 23 trajectories** (py 12, powershell 5, ruby 2, ts 2, bash 1, java 1; `results/raw/val-oracle-check-20261012T133816Z`).
  7 of the 23 are in the 53 inspected VAL pools; 21 were sent to the Nemotron generator.
- **Protocol** `results/raw/val-outcome-v1-20261012T134540Z`: `PREREG_original.json` (the CAL registration, unchanged, hash recorded), `PREREG_AMENDMENT_1.json` (pool only,
  ids and hashes), `MANIFEST.json`. `tools/cal_outcome_v1.py` (prep / decode / execute / score), worker `kaggle/cal_outcome_gen_t4.py`. Pilot = first five in the
  fixed hash order, one per trajectory: py-retry-policy:2, py-role-hierarchy:1, py-tree-filters:3, py-inventory-log:2, bash-retryrun:2. Only these five prompts
  were uploaded (private dataset `laya-cal-outcome-v1`). Leak check: 8 recorded argument texts also occur in the 23 prompts, all `old_string` (file text already
  read in the session); 0 hits for `content` or `new_string`. The failed CAL attempt left `results/raw/cal-outcome-v1-20261012T133201Z` (registration + ids only).
- **Generation** (Kaggle kernel `laya-cal-outcome-gen` v1, 2 x T4, Qwen3-4B-Thinking-2507 fp16, manual loop with a preallocated cache, 4 samples x 5 decisions,
  limit 6,144 new tokens): wall 7,143 s, **about 4.0 GPU hours** of the 12 approved; 0.13-0.14 s per token; peak 9.6 GiB. **15 of 20 samples hit the token limit
  with the reasoning block still open** (the model drafts and re-checks the whole file inside its reasoning; not a repetition loop). 5 finished in 92-162 s.
- **Candidates:** of the 5 finished, 1 decodes (a `Read` of the file that is still to be written) and 4 are the same test-run command without the `reply` field the
  protocol requires (rejected by the unchanged decoder). **Valid samples 1 of 20; code-change candidates 0; executed candidates 0.** Sandbox baselines: all five
  decisions fail before any action, twice (as qualified).
- **Scores** (`pilot-20261012T154832Z/SCORES.json`): decisions with a passing candidate 0 of 5; informative decisions 0 of 5; Qwen's first proposal, the head, the
  probe and random all 0 of 5. No scorer comparison is possible.
- **Reading.** On these one-step fix decisions the bottleneck is candidate generation, not ranking: within 6,144 tokens the frozen 4B model almost never commits to
  an action, and when it does it reads or runs tests where Fable writes the fix. The head and the probe had nothing to choose between. Limits: 5 decisions, one
  prompt format, one token limit; the 4 undecodable answers were also not code changes, so a looser decoder would not change the outcome.
- **Stopped as pre-registered:** no further GPU time; the remaining 18 decisions were not sent. CAL and TEST untouched. Nothing trained or modified.


### 15.31b Outcome pilot on VAL (Qwen writes the candidates) and five research reports (2026-10-12, about 16:00 UTC = 21:30 IST)

- **Pool change.** The request builder refuses to copy CAL / TEST contexts into model-facing requests; the guard was left in place. User (chat): GO Option B, the 23
  clean fail-to-pass VAL decisions (`results/raw/val-oracle-check-20261012T133816Z`: 66 reconstructable, 46 replayable twice, 39 useful, 23 fail -> pass on 23
  trajectories). Original registration `results/raw/cal-outcome-v1-20261012T133201Z/PREREG.json` kept; amendment `results/raw/val-outcome-v1-20261012T134540Z/PREREG_AMENDMENT_1.json`
  (only the pool changes). CAL / TEST contexts were not read into any request. Upload: 5 pilot prompts, private Kaggle dataset `laya-cal-outcome-v1`.
  Fable's recorded text in the 23 prompts: 8 hits, all `old_string` (file text already read in the session); 0 for `content` / `new_string`.
- **Pilot result** (`.../pilot-20261012T155304Z/PILOT_RESULT.json`, `CANDIDATES.json`; generation `gen-pilot-20261012T155304Z`). 5 decisions x 4 samples, frozen Qwen3-4B-Thinking,
  6,144 new tokens, 2 x T4: **3.75 GPU hours** (13,488 s; about 0.13 s per token). **15 of 20 samples did not finish** and never closed their thinking block (the file is
  drafted inside the thinking). 5 finished: 1 valid action (a Read), 4 rejected by the registered decoder (a Bash test run wrapped as `{"calls": [...]}`).
  **0 code-change candidates, 0 sandbox executions, 0 informative decisions; no scorer comparison is possible.** The remaining 18 decisions were not generated (stop rule).
  Conclusion: candidate generation by the small model is the bottleneck, before any ranking.
- **Research by five Haiku subagents** (user request; reports in `results/raw/research-20261012/`, read-only):
  A new Fable repositories: mostly copies of each other; distinct originals hold a few hundred sessions (18, 63, 79, 244), some without tool results; four repositories
  need a login. B ready-made decision datasets: `samatv256/jev-decisions-v1` (50,000-row clean subset) converts without a model; the others partly or not; no card names
  Open-Jev as a source, lineage not verified beyond the cards. C funnel of the five TRAIN batches (995 decisions): 260 lost before build (114 too few candidates, 103 no
  candidate with Fable's tool), 23 at the screen, 712 reviewed, 335 pools; **150 near misses with exactly one jointly admitted alternative**. D recorded branches: 51 pairs
  in 11,436 decisions, 11 clearly same-state: ruled out; retrieval gives volume but only a surface check. E code mutations: 118 replayable TRAIN states (py 61, ts 53, go 4),
  87% with at least 5 mutation sites; estimate 350 to 1,200 verified-worse alternatives; safeguards listed.
- **Slips and open checks.** (1) Report D's "train" filter used the trace file's own split field, which does not separate canonical CAL / TEST; its aggregate counts span
  them and three example lines from TEST trajectories were quoted; they were redacted and a note added. No label, model or score was involved. (2) Report C: 52 TRAIN-role
  pools (conventions-40: 16, conventions-95: 36) come from pair runs whose own report says `pair_passed false`: Muse's replies are all counted as first-attempt
  failures because they were saved under the retry run name (the accounting fault fixed later for Stage 1), and conventions-40 has only 24 known-bad controls;
  0 known-bad options were accepted by both judges in either. FROZEN_SET v3 lists only the remaining-batch exception: **the manifest's exception list is incomplete.**
  (3) Report C says 5 kept alternatives are marked as technical failures in SCREEN_OPTIONS; matching by origin label is ambiguous (labels repeat inside pools), so this
  is NOT confirmed and needs a check by option index. (4) Only 1,294 of the trace file's 11,436 "train" rows carry `model_attested = true`; for the rest "Fable 5" is
  the uploader's claim.


### 15.32 Head-only fine-tuning with a frozen bridge (2026-10-12, 16:33 UTC = 22:03 IST)

User (chat): the objective is the pretrained LAYA head itself (Qwen only supplies features; no encoder is trained or touched); GO for head-only fine-tuning with two
frozen bridges; verify that only the head changes; learning rate chosen on TRAIN only, both reported; forgetting and the no-context control measured.
`tools/laya_headonly_v1.py`; `results/raw/laya-headonly-v1-20261012T161940Z/PREREG.json` (before the run), `REPORT-20261012T163328Z.json`. Train_v4, fit 251 pools,
selection fold 52 pools, 300 steps, adapter learning rate 0, seeds 0-2. In all 12 runs the bridge tensors are bit-identical before and after (sha256) and all 31
pretrained tensors changed. Checkpoint file unchanged.

| bridge | head lr | cleaned VAL before | after (3-seed mean, interval) | after minus before | per seed | no context | LAYA native 600 before -> after |
|---|---|---|---|---|---|---|---|
| A: aligned on native items only, never saw a Fable label | 2e-5 | 39.1% | **54.3%** (43.3 - 66.7) | **+15.2 (+6.3 to +24.9)** | 52.2 / 56.5 / 54.3 | 46.4% | 71.5% -> 69.2 / 69.0 / 69.0 |
| A | 2e-4 (chosen on TRAIN: fold-0 59.6% against 48.7%) | 39.1% | 52.2% (40.9 - 63.5) | +13.0 (-2.5 to +26.7) | 56.5 / 39.1 / 60.9 | 40.6% | 71.5% -> 49.3 / 43.8 / 57.8 |
| B: Fable-trained adapter | 2e-5 (chosen) | 76.1% | 74.6% (61.0 - 86.3) | -1.4 (-6.7 to +3.0) | 69.6 / 69.6 / 84.8 | 65.2% | 32.3 / 28.0 / 32.0 -> 31.2 / 29.0 / 31.0 |
| B | 2e-4 | 76.1% | 76.1% (64.5 - 86.8) | 0.0 (-5.1 to +4.9) | 71.7 / 76.1 / 80.4 | 63.0% | -> 28.5 / 34.2 / 31.3 |

Reference on the same 46 pools: adapter-only 76.1%, joint 75.4%, linear probe 89.1%; chance 29.9%.

- **PRIMARY as registered (Bridge A, the rate chosen on TRAIN = 2e-4): +13.0 points, interval -2.5 to +26.7, not significant.** The other registered rate, 2e-5, gives
  +15.2 (+6.3 to +24.9). Both are reported as required; the TRAIN-fold choice picked the rate that fits TRAIN better but is unstable on VAL and forgets.
- **What is established:** with a bridge that never saw Fable held bit-identical, training only the head's 31 tensors moves Fable agreement from 39% to about 52-54% on
  unseen trajectories. Part of Fable's choices can be stored in the head. At 2e-5 the best checkpoint is the last step in all three seeds (still improving at 300 steps),
  and native accuracy is nearly kept (71.5% -> 69%).
- **What is not:** it stays about 22 points below adapter-only on the same pools. On top of a Fable-trained bridge the head adds nothing. At 2e-4 native accuracy
  collapses to 44-58% (forgetting). Real context beats no context by 8 to 12 points after training (Bridge A).
- **Reading against the pre-agreed table:** "Bridge A improves, general ability retained" holds only for the low rate and only partly (well short of adapter-only);
  "both remain below adapter-only" also holds. So: the head CAN learn Fable decisions through a fixed bridge, but under this setup most of what adapter training
  learns does not get into the head. Longer training at the low rate was not tested.
- Limits: 46 VAL pools; one backbone (nothing here shows portability to another model); the bridge's input standardisation is refitted from Fable features without
  labels; agreement with Fable, not task success. The 12 head files take 1.2 GB (`heads/`); drive D: has 5.9 GB free.


## 16. Consolidated findings and plan (2026-10-12, 22:30 IST = 17:00 UTC)

Written at the user's request ("update all the docs with all the numbers"). Every figure below has its source in sections 15.23 to 15.32. Another session edited the
docs in parallel on 2026-10-12: there are two sections numbered 15.31 (the second is renamed 15.31b here) and the ledger had two rows 54 (the second is now 55).
GPU time of the outcome pilot: 3.75 hours of worker time (13,488 s), about 4.0 hours counting kernel start-up; both figures describe the same run.

**Objective (user, 2026-10-12):** strengthen the pretrained LAYA decision head itself so that it can help smaller models choose better actions. Qwen is only a
feature provider and is never trained. No encoder is trained or touched for now (encoders may change later). Fable is the teacher.

### 16.1 Data
| item | value |
|---|---|
| eligible Fable TRAIN decisions / sent to the generator / first-step, not processed | 1,342 / 1,074 / 268 |
| dataset v3 (frozen) | 335 pools, 866 alternatives, 192 trajectories; 334 with features; roles 120 / 54 / 161 |
| yield | 31% of generated decisions |
| losses among 995 TRAIN decisions (report C) | 260 before build (114 too few candidates, 103 no candidate with Fable's tool), 23 at the screen, 377 at review or assembly; 150 near misses |
| cleaned v4 | 59 strict equivalents removed (6.8%), 31 pools dropped, 304 pools, 779 alternatives, 177 trajectories (303 with features) |
| VAL pools | 178 decisions sent, 129 screened, 53 pools on 36 trajectories, 134 alternatives; 46 pools after cleaning |
| equivalent alternatives (hand audits) | VAL 14 strict + 18 loose of 134 (23.9%); 60 TRAIN pools 11 strict + 30 loose of 161 (25.5%) |
| replayable edit / write decisions | CAL 155 -> 72 reconstructable -> 52 replay twice -> 44 useful -> 27 fail-to-pass; VAL 66 -> 46 -> 39 -> 23 |

### 16.2 Models (agreement with Fable's recorded action)
| experiment | result |
|---|---|
| pretrained head, no training (123 pools) | 36.6%, chance 29.6% |
| original LAYA with its own encoder, as released (123 pools) | 30.1%, chance 29.6%; 108 of 123 contexts over its window |
| adapter on 120 pools -> 160 unseen pools | 63.1%; probe 75.6%; aligned adapters 58.4% and 54.9% (aligned minus fresh -4.75, -9.6 to -0.3) |
| adapter on 334 pools, in-dataset | 80.2%; probe 86.2% (+6.0, +1.9 to +10.1) |
| same adapters on 53 VAL pools | 79.2% (67.9 - 89.3); no context 64.2%; probe 94.3% / 81.1%; text only 67.9% |
| VAL with equivalents counted as correct | head 80.5% (strict) / 83.6% (strict + loose); probe 94.3% unchanged |
| adapter only against adapter + head, cleaned VAL | 76.1% against 75.4% (-0.7, -5.9 to +4.3); probe 89.1% |
| head only, bridge never trained on Fable, cleaned VAL | 39.1% -> 54.3% at 2e-5 (+15.2, +6.3 to +24.9; native 71.5% -> 69%); -> 52.2% at 2e-4 (+13.0, -2.5 to +26.7, registered primary; native -> 44-58%) |
| head only, Fable-trained bridge | 76.1% -> 74.6% / 76.1%: no gain |
| outcome pilot, 5 VAL decisions x 4 Qwen samples | 15 of 20 unfinished, 1 valid action, 0 code changes, nothing to rank |

### 16.3 What the results mean
1. The head with a trained adapter generalises to unseen trajectories (79.2%), and it uses the context (15 points lost without it).
2. Most of what was learned sits in the adapter. Training the head alone through a fixed bridge reaches about 54%, about 22 points short.
3. The head's own tensors can store part of Fable's choices (+13 to +15 points with a bit-identical bridge). At 2e-5 it keeps its native skill and was still
   improving at step 300; at 2e-4 it forgets.
4. More pools coincided with better scores (63% at 120 pools, 79% at 334); the setups differ, so this is supported, not proven.
5. The linear probe is ahead in every comparison (6 to 15 points). It never picks an equivalent alternative on VAL, so part of its lead is recognition of
   Fable's exact wording; how much is real decision quality cannot be measured on Fable-against-Nemotron pools.
6. Aligning the adapter to the original encoder and training adapter + head together gave no gain.
7. Nothing shows better coding outcomes; the small model wrote no usable fix in the pilot.

### 16.4 Problems on record
- v3 manifest: the gate-exception list is incomplete (52 pools from two runs marked failed for accounting reasons; 0 known-bad accepted by both).
- Report D quoted three example lines from TEST trajectories (redacted); its counts span TRAIN, CAL and TEST because the trace file has no CAL / TEST field.
- 1,294 of 11,436 rows of the trace file's train portion are attested Fable 5.
- Unconfirmed: 5 kept alternatives marked as technical failures (label matching is ambiguous).
- CAL: edit decisions replayed in the sandbox for counting only; the request guard was left in place. VAL: development-exposed.
- Disk D: 5.9 GB free; the 12 head files of 15.32 take 1.2 GB. The A5 benchmark was not run.

### 16.5 Approvals and plan
| step | state |
|---|---|
| mutation pilot on 20 TRAIN states | approved, NOT started |
| near-miss recovery pilot on 25 decisions | approved, NOT started; the full 150 is not approved |
| longer head-only run (Bridge A, 2e-5, about 1,200 steps) | proposed, waits for GO |
| head-only with replay of the 2,988 native items | proposed, waits for GO |
| 3,000 rows of an outside decision dataset in the head-only comparison | proposed; needs the user's decision on the Jev rule and a 172 MB download |
| second small language model (portability) | later |
| not planned | any encoder training, the native text model, more Qwen generation, CAL / TEST use |
### 16.6 Record corrections (2026-10-12, about 17:30 UTC)

`results/raw/laya-v3-record-corrections-20261012/CORRECTIONS.json` (the frozen sets are unchanged): (1) the 52 pools of conventions_40 / conventions_95 are now recorded as a gate exception with the cause (accounting of Muse replies; 24 controls in one batch; 0 known-bad accepted by both); (2) the claim that 5 kept alternatives failed the technical screen is NOT confirmed: in each case the failing option shares its origin label with a kept one and was removed at the screen; (3) the TEST quote in research report D; (4) the attestation count.


### 16.7 Head-only diagnostics: the signal is lost inside the head, not in the bridge (2026-10-12, 17:24 UTC = 22:54 IST)

User (chat, plan review): GO for two controlled head-only runs plus a probe on identical frozen inputs and splits; one factor per comparison; paired changes; native
skill and context dependence reported. `tools/laya_headonly_diag_v2.py`; `results/raw/laya-headonly-diag-v2-20261012T170830Z/PREREG.json` (before the run, with the
readings fixed in advance), `REPORT-20261012T172423Z.json`. Cleaned VAL, 46 pools (one pool = 2.2 points). Bridge bit-identical in all six runs; checkpoint unchanged.

**Probes (pretrained head untouched; fitted on the 303 train_v4 pools, scored on the same 46 VAL pools):**

| the probe reads | top-1 | minus the trained head (54.3%) |
|---|---|---|
| raw Qwen option states | 89.1% (78.4 - 97.8) | +34.8 (+22.0 to +47.2) |
| output of Bridge A (never saw Fable) | **87.0%** (75.9 - 96.1) | +32.6 (+19.7 to +44.9) |
| after the pretrained context layers, Bridge A | **58.7%** (41.9 - 74.5) | +4.3 (-13.0 to +19.4) |
| output of Bridge B (Fable-trained) | 82.6% | +28.3 |
| after the pretrained context layers, Bridge B | 73.9% | +19.6 |

**Runs (Bridge A, head learning rate 2e-5, seeds 0-2):**

| run | fit pools | steps / examples seen | VAL | per seed | against the reference | fit accuracy | native 600 |
|---|---|---|---|---|---|---|---|
| reference (15.32) | 251 | 300 / 2,400 | 54.3% | 52.2 / 56.5 / 54.3 | - | 48.7% on the selection fold | 69.2 / 69.0 / 69.0 |
| longer | 251 | 1,200 / 9,600 | 55.8% (42.8 - 68.9) | 54.3 / 60.9 / 52.2 | +1.4 (-8.2 to +11.3); pools gained / lost per seed 5/4, 4/2, 2/3 | 61.4 / 61.8 / 61.8% | 67.8 / 68.2 / 68.8 |
| half data | 116 | 300 / 2,400 | 48.6% (36.1 - 62.7) | 47.8 / 50.0 / 47.8 | -5.8 (-12.4 to +0.7); 0/2, 1/4, 1/4 | 52.6 / 51.7 / 52.6% | 68.0 / 67.7 / 68.3 |

Longer run, selection-fold curve (seed 0): 38% until step 100, 58% at 200, peak 61.5% at 350-450, then flat at 56-58% to step 1,200. Real minus no context after
the longer run: +10.9 (-2.0 to +24.4).

- **Readings fixed before the run:** "probe on the bridge output well above the trained head" holds (+32.6, interval excludes 0): the information reaches the head,
  so the bridge is not the limit. "Longer helps" does not hold: it plateaus. "Half clearly below full": -5.8 points, interval touches 0: a trend, not established.
- **New fact:** the head does not even fit its own training pools at this rate (61% after 1,200 steps), while a linear read-out of the same bridge output scores 87%
  on unseen pools. The pretrained context layers turn a linearly readable signal (87%) into one that is not (59%), and a safe learning rate does not undo that;
  the higher rate (15.32) fits more but forgets the native skill. This is under-fitting inside the head, not over-fitting and not a data shortage.
- **Not established:** why the context layers lose the signal. Candidates to audit (review's row "audit head input contract, loss, masking, normalisation"):
  natively these layers run over a full token sequence and read options at marker tokens, here they get one query vector and N option vectors; the alignment matched
  the head's output distribution on native items, not its internal representation; the scorer reads the post-context vector only.
- Limits: 46 VAL pools; internal probes use seed 0; a probe shows what is linearly readable at a point, not what training could recover; one backbone.

### 16.8 Record audit by option position (2026-10-12)

`results/raw/laya-v3-record-corrections-20261012/SCREEN_RECHECK_BY_POSITION.json`: for the five (decision, label) pairs every option carrying the label was re-checked
on its exact text (bash -n, Python parse, edit anchor in context). In all five the option that reproduces the recorded error was removed at the screen and the kept
option(s) pass: 0 kept options fail. Plan-review error noted: the 161 `heldout_transfer` pools are TRAIN-split pools (training on them was approved on 2026-10-12);
the held-out set is the 46 cleaned VAL pools.


### 16.9 Read-only audit of the head's input path (2026-10-12, about 18:00 UTC = 23:30 IST)

User (chat): GO for the read-only architecture audit first; data pilots and further fine-tuning on hold. `tools/laya_head_audit_v1.py`;
`results/raw/laya-head-audit-v1-20261012T173011Z/` (`PLAN.json` written first, `AUDIT-*.json`, `NATIVE_HEAD_INPUTS.json`, `CODE_COMPARISON.json`). Probes fitted on the
303 train_v4 pools, scored on the 46 cleaned VAL pools. Nothing trained except diagnostic probes; no model file written.

**Probe at every point (Bridge A):**

| point | released head | head-only-trained head (15.32, seed 0) |
|---|---|---|
| bridge output (what the context layers receive) | 87.0% | 87.0% |
| after context layer 1 | **56.5%** | 50.0% |
| after context layer 2 | 56.5% | 52.2% |
| after out_norm (scorer input) | 58.7% | 52.2% |

Other read-outs on the released head: bridge and scorer-input joined 84.8%; scorer input, small MLP 67.4%; scorer input centred within the pool 52.2%; bridge
centred within the pool 80.4%.

**Representation statistics at the option vectors:**

| where | point | mean norm | cosine between options of a pool | variance share within pools |
|---|---|---|---|---|
| ORIGINAL model on 40 native items (median 622 tokens) | head input | 52 | 0.62 | 0.47 |
| | after layer 1 | 8,319 | 0.80 | 0.62 |
| | after layer 2 (scorer input; 10,914 on all 600) | 10,995 | 0.88 | 0.56 |
| ours, Fable VAL pools | bridge | 42 | 0.73 | 0.40 |
| | after layer 1 | 4,046 | 0.91 | 0.30 |
| | after layer 2 | 6,975 | 0.97 | 0.27 |
| ours, native items through the same bridge | bridge | 180 | 0.83 | 0.40 |
| | after layer 2 | 2,873 | 0.94 | 0.34 |
| ours, Fable VAL, after head-only training | after layer 2 / out | 6,995 | 0.99 | 0.14 / 0.11 |

Attention of an option vector in the released layers on our sequence: layer 1 puts 44% on the single context vector, 15% on itself, 17% on each other option;
layer 2: 35%, 22%, 18%.

- **Found:** the drop happens at the FIRST context layer (87.0% -> 56.5%). After the layers the options of one pool are almost the same vector (cosine 0.97 against
  0.88 in the original model on its own items), and head-only training makes this worse (0.99; within-pool variance share 0.11). The pre-context vector still holds
  the signal: joining it to the scorer input restores 84.8%. A non-linear read-out of the scorer input recovers only part (67.4%), so this is not just a loss of
  linear readability.
- **Not found:** no error in masks, shapes, padding or checkpoint handling. **The input scale is not mismatched** (native head input norm about 52; bridge output 42 and
  180); the norm growth to thousands is the original model's own behaviour. (An earlier guess that the residual is swamped because the bridge output is too small
  is therefore withdrawn.)
- **Structural difference (from the two source files):** natively the two layers run over 99 to 1,280 tokens and an option's marker token collects its own option's
  text and the context; here they get one pre-pooled vector for the context and one per option, plus an extra LayerNorm and an untrained kind embedding.
- **Reading (hypothesis consistent with the measurements, not proven):** the pretrained layers were trained to BUILD an option's representation from many tokens; given
  one vector per option and one for the context they mix the options together, and a learning rate low enough to keep the native skill does not undo that.
- Limits: 46 VAL pools; one seed for the trained head; 40 native items for the original model's statistics; a probe shows what is readable, not what is recoverable.

### 16.10 Audit step C: the two small differences do not matter (2026-10-12, about 18:15 UTC = 23:45 IST)

User (chat): GO. `tools/laya_head_audit_v1c.py`, `results/raw/laya-head-audit-v1-20261012T173011Z/AUDIT_C.json`. Released head, Bridge A, nothing trained except probes.

| forward pass | head on cleaned VAL (no Fable training) | native items vs gold | probe after layer 1 / layer 2 / scorer input | cosine between options at the scorer input |
|---|---|---|---|---|
| as is | 39.1% | 71.5% | 56.5 / 56.5 / 58.7% | 0.964 |
| kind embedding not added | 39.1% | 71.5% | 56.5 / 56.5 / 58.7% | 0.964 |
| extra LayerNorm skipped | 39.1% | 71.3% | 56.5 / 56.5 / 56.5% | 0.971 |
| neither | 39.1% | 71.3% | 56.5 / 56.5 / 56.5% | 0.971 |

- The kind embedding has norm 0.0: it is zero-initialised and adds nothing. **Correction:** sections 16.9, the README and `CODE_COMPARISON.json` call it "randomly
  initialised" / "untrained"; it is untrained and exactly zero. The extra LayerNorm is close to identity (mean weight 0.995) and removing it changes native accuracy
  by 0.2 points.
- So neither difference explains the collapse. What remains is the input itself: one pooled vector per option and one for the context, where the layers were
  trained on hundreds of tokens.


### 16.11 Sequence-input test: the prediction is NOT met (2026-10-12, 19:01 UTC = 2026-10-13, 00:31 IST)

User (chat): GO option A, capture on Kaggle. Capture `sangeetasangamkar/laya-seq-capture-run`: 1,187 of 1,187 requests (387 Fable pools, 800 native items), none
excluded, all finite, 2,789 s on a T4, `results/raw/seq-capture-run-20261012T183217Z` (685 MB; bundle `results/raw/seq-bundle-v1-20261012T174308Z`). Format: up to 16
slice means of the context + the last-token vector; per option up to 4 slice means + its last token (marker) + the full mean. Cross-check: the full option mean equals
the existing VAL features (minimum cosine 1.0). Analysis `tools/laya_seq_v1.py`, `results/raw/laya-seq-v1-20261012T185241Z/` (`PLAN.json`, written at the start of the run; the prediction was written
before the capture finished, HANDOFF_2026_10_12_1122PM_IST.md section 7; `REPORT-20261012T190124Z.json`). The user asked for the GPU in place of the CPU, so the bridge training ran on the PC's RTX 3050 (a CPU run stopped
at step 700 is kept as `...-cpu-stopped`). Both bridges aligned on the same 600 native items with the same settings; head tensors identical in both arms and unchanged.

| measure (released head tensors, no Fable training) | sequence input | single-vector control |
|---|---|---|
| native items, accuracy vs gold (200) | 56.5% | 62.0% |
| native items, agreement with the original model | 63.5% | 68.5% |
| bridge alignment: best step / KL to the original model | 700 / 0.287 | 1,500 / 0.212 |
| head's own top-1 on cleaned VAL | 50.0% | 43.5% |
| probe on the bridge output (at the marker / option vector) | 80.4% | 89.1% |
| probe after context layer 1 | 60.9% | 54.3% |
| probe after context layer 2 (scorer input) | 56.5% | 56.5% |
| cosine between the options of a pool: bridge / layer 1 / layer 2 | 0.69 / 0.94 / 0.985 | 0.69 / 0.92 / 0.980 |
| share of variance within pools at layer 2 | 0.49 | 0.20 |

- **Prediction written beforehand:** with sequence input the options stay apart after the context layers (cosine clearly below 0.97, towards the original model's
  0.88) and the probe after layer 1 stays near the probe on the bridge output. **Observed:** cosine 0.985 (no better than the control's 0.980) and the probe still
  falls from 80.4% to 60.9% at layer 1. **The prediction is not met.**
- Smaller differences in favour of the sequence arm, none of them decisive on 46 pools: layer-1 probe 60.9% against 54.3%, the head's own top-1 50.0% against 43.5%,
  within-pool variance share 0.49 against 0.20.
- Against it: the sequence bridge aligned worse on the native items (56.5% against 62.0%; its best step was 700 of 1,500 and the curve is noisy).
- **What this does and does not show.** This coarse sequence (about 36 slice-mean vectors) does not stop the collapse, so "one pooled vector per option" is not, by
  this test, the cause. It does not test a token-level sequence (the original model sees 99 to 1,280 tokens), and both bridges had only 600 native items (the earlier
  bridge had 2,988 and reached 71.5%). One seed.
- **By the staged plan:** stop here. No larger capture and no head-only training on sequences. Next is a review of the head's loss and optimisation, not more data.
- New rule (user, 2026-10-13): compute-heavy jobs run on Kaggle, not on the PC; the run above was allowed to finish locally.


### 16.12 Review of the head: where the signal is lost, and what head-only training did (2026-10-12, 19:23 UTC = 2026-10-13, 00:53 IST)

User (chat, 2026-10-13): "Go complete step 1 and 2". Read-only; run on Kaggle CPU (`sangeetasangamkar/laya-head-review-run`, private dataset `laya-head-review-v1`,
297 MB: features and recorded-option index of 303 TRAIN and 46 cleaned VAL pools, 600 native items, the head with Bridge A, the head-only-trained tensors of seed 0 at
2e-5; no texts, no CAL / TEST). Script `kaggle/head_review_cpu.py`, pack `tools/head_review_pack_v1.py`, bundle `results/raw/head-review-bundle-v1-20261012T191311Z`,
report `results/raw/head-review-run-20261012T192308Z/REPORT-20261012T192246Z.json` (`PLAN.json` with the readings written before the run). A 20-item smoke test ran on the PC first. Checks: the written-out layers
reproduce the head's scores to 5e-7 and the four earlier probe figures exactly. 46 VAL pools: one pool is 2.2 points. The A5 benchmark was not run.

**Step 1: where the signal is lost (released head tensors, Bridge A)**

| point inside the head | probe on cleaned VAL | mean vector norm |
|---|---|---|
| bridge output | 87.0% | 42 |
| layer 1 after attention | 76.1% | 56 (attention update 40) |
| layer 1 after feed-forward | 56.5% | 4,046 (feed-forward update 4,041) |
| layer 2 after attention / after feed-forward | 58.7% / 56.5% | 4,415 / 6,975 |
| scorer input | 58.7% | 30 |
| attention update of layer 1 alone / feed-forward update alone | 39.1% / 58.7% | |

| attention allowed | probe after layer 1 | probe at scorer input | head's own top-1 on VAL | native accuracy (600) |
|---|---|---|---|---|
| normal | 56.5% | 58.7% | 39.1% | 71.5% |
| no token sees another | 58.7% | 56.5% | 41.3% | 68.8% |
| options cannot see the context token | 54.3% | 56.5% | 41.3% | 70.5% |
| options cannot see each other | 56.5% | 58.7% | 41.3% | 70.8% |

| cosine between the options of a pool | as measured | common mean removed | share of variance within pools |
|---|---|---|---|
| Fable VAL, bridge output | 0.73 | 0.46 | 0.40 |
| Fable VAL, after layer 2 | 0.97 | 0.48 | 0.27 |
| Fable VAL, scorer input | 0.96 | 0.29 | 0.57 |
| native items through the same bridge, scorer input | 0.93 | 0.50 | 0.45 |
| ORIGINAL model on its own items, scorer input | 0.87 | 0.18 | 0.57 |

- The feed-forward step of layer 1 writes an update about 96 times larger than what it receives (4,041 against 42); what the bridge delivered is about 1% of the stream
  afterwards. That is where the probe falls (76.1% to 56.5%). The attention step costs a further 11 points before it.
- Mixing between tokens is not the cause: with all mixing removed nothing changes on Fable pools, and the native accuracy only moves from 71.5% to 68.8%. Through this
  bridge the two context layers act almost as a per-option transform.
- **Correction of 16.8 / 16.10 ("the options collapse, cosine 0.97"):** most of that similarity is one common vector shared by every option (norm 6,731 of 6,975).
  With it removed the cosine is 0.48 (0.29 at the scorer input; the original model on its own items 0.18), and the share of variance within pools at the scorer input
  equals the original model's (0.57). The geometry is not collapsed; the content is poor (a pool-centred probe at the scorer input reads 52.2%, section 16.8).

**Step 2: gradients of head-only training (251 fit pools, the trainer's loss, unknown weight 0.5)**

| measure | released tensors | after head-only training (seed 0, 2e-5) |
|---|---|---|
| loss / loss if all scores were equal | 1.064 / 0.813 | 0.751 / 0.813 |
| top-1 on the fit pools | 41.4% | 55.4% |
| spread of the scores within a pool / mean top probability | 0.73 / 0.48 | 0.28 / 0.36 |
| share of the head's gradient in the last scorer layer (`mlp.3`) | 99.1% | 50.4% |
| share in the attention of context layer 1 | 0.7% | 46.6% |
| cosine: descent direction against "make the options more alike", context-layer tensors | 0.74 to 0.997 | -0.67 to 0.70 |
| cosine between the gradients at unknown weight 0.5 and 1.0 | 0.9999 | 0.964 |

- The released head starts WORSE than giving every option the same score (1.064 against 0.813): it is confident and wrong on Fable pools. The cheapest way down is to
  flatten the scores, and for every context-layer tensor that direction is nearly the same as making the options more alike (cosine 0.74 to 0.997).
- After 300 steps the loss is only 0.06 below "all scores equal" and the score spread has fallen from 0.73 to 0.28. Much of the training went into removing
  over-confidence, not into ranking.
- The unknown weight (0.5 or 1.0) does not change the gradient direction; it is not the problem.
- The trainer uses AdamW, which rescales each tensor's step, so the 99% gradient share of the last layer does not by itself mean the context layers stood still.

**Reading.** The limit is the feed-forward step of context layer 1 (it replaces the input with features that do not separate Fable's options), not attention mixing
and not the loss weight. The head-only recipe as run spends its steps on calibration. No fine-tuning was run here; any new run needs the user's GO and goes to Kaggle.


### 16.13 Causal check of the layer-1 feed-forward update (2026-10-12, 19:38 UTC = 2026-10-13, 01:08 IST)

User (chat, 2026-10-13, pasted decision): GO for the small causal feed-forward diagnostic, then a controlled calibration-versus-learning-rate experiment on Kaggle GPU
(not started; see below). Read-only, Kaggle CPU, `sangeetasangamkar/laya-head-ffscale-run`, same private dataset as 16.12, script `kaggle/head_ffscale_cpu.py`, report
`results/raw/head-ffscale-run-20261012T193833Z/REPORT-20261012T193806Z.json` (`PLAN.json` written before the run). The feed-forward update of context layer 1 is multiplied by s; everything after it runs
unchanged. The review named s = 1, 0.5, 0.25, 0; 0.1, 0.03 and 0.01 were added because the update is about 96 times its input. Released head tensors, Bridge A.

| s | update norm (stream before it: 56) | probe after layer 1 | probe at scorer input | head's own top-1, 251 TRAIN fit pools | head's own top-1, cleaned VAL (46) | native accuracy (600) |
|---|---|---|---|---|---|---|
| 1 (as released) | 4,079 | 56.5% | 58.7% | 41.4% | 39.1% | 71.5% |
| 0.5 | 2,039 | 54.3% | 54.3% | 39.8% | 39.1% | 70.8% |
| 0.25 | 1,020 | 58.7% | 63.0% | 40.2% | 39.1% | 68.5% |
| 0.1 | 408 | 67.4% | 69.6% | 39.8% | 43.5% | 68.0% |
| 0.03 | 122 | 71.7% | 65.2% | 40.2% | 43.5% | 46.7% |
| 0.01 | 41 | 73.9% | 63.0% | 39.8% | 41.3% | 42.7% |
| 0 | 0 | 76.1% | 65.2% | 32.3% | 30.4% | 45.3% |

- The update does hide the signal at that point: the probe after layer 1 rises step by step from 56.5% to 76.1% as the update is removed. s = 0.5 and 0.25 change
  little, as predicted (the layers normalise their input and the update still dominates).
- Only part of it reaches the scorer input (at most 69.6%, never the 87.0% of the bridge output), and **the head's own choices do not improve at any s** (39 to 43% on
  VAL, 40 to 41% on the TRAIN fit pools; the TRAIN-chosen setting is s = 1). By the reading fixed beforehand: the signal survives, but layer 2 and the scorer do not
  read it.
- The same update carries the head's native skill: native accuracy falls from 71.5% to 43-47% once it is scaled to 3% or less. It is not a defect to be removed.
- Not started: the controlled training experiment. It cannot use the frozen trainer (one learning rate for all 31 tensors, CPU only), so it needs a new versioned
  training loop whose reference arm must first reproduce the existing 54.3%.


### 16.14 Four-arm head-only experiment with 5-fold cross-validation (2026-10-12, 20:15 UTC = 2026-10-13, 01:45 IST)

User (chat, 2026-10-13): GO four arms, 5-fold cross-validation, Kaggle GPU; calibration as a control, not a training run; the new loop must reproduce the reference or
stop; stop after reporting. Kaggle `sangeetasangamkar/laya-head-exp4-run` on a Tesla T4 (CUDA confirmed in the log before training), private dataset
`laya-head-exp4-v1` (420 MB; no texts, no CAL / TEST), script `kaggle/head_exp4_gpu.py`, pack `tools/head_exp4_pack_v1.py`, bundle
`results/raw/head-exp4-bundle-v1-20261012T194655Z`, report `results/raw/head-exp4-run-20261012T201525Z/REPORT.json` with `PLAN.json` (written before the run), `runs.jsonl` (88 runs, per-pool hits)
and the log. 14 minutes. Only the 31 pretrained head tensors were updated: the adapter was bit-identical before and after in all 88 runs; no non-finite step; no
checkpoint was saved; nothing is adopted. The A5 benchmark was not run.

**Reproduction gate: passed.** Arm A in the reference configuration, three-seed means, new loop on the GPU against the recorded reference: cleaned VAL 55.1% / 54.3%,
development fold 48.7% / 48.7%, native 69.0% / 69.1%, fit pools 55.4% / 55.1% (allowed 4.4 / 4 / 3 / 4 points). The CPU repeat of seed 0 differs from the recorded
run on 1 of 46 VAL pools (torch 2.11 on Kaggle, 2.14 on the PC; cause not confirmed).

**Primary evaluation: out of fold on the 303 TRAIN pools (fit about 190 pools per run)**

| arm | out-of-fold top-1, 303 TRAIN pools | each seed | minus A (95% interval) | top-1 on its own fit pools | native accuracy, mean (min to max over 15 runs) |
|---|---|---|---|---|---|
| released head, no training | 39.9% (34.4 to 45.9) | - | - | - | 71.5% |
| A reference | 45.5% (39.9 to 51.1) | 45.9 / 44.6 / 46.2 | - | 54.1% | 69.0% (67.5 to 70.0) |
| B+ calibrated start (agent's addition) | 47.4% (41.8 to 53.3) | 47.9 / 46.5 / 47.9 | +1.9 (-5.7 to +9.7) | 55.7% | 44.9% (26.5 to 61.3) |
| C, layer 1 at 1e-4 (chosen on the inner folds: 53.9% against 53.3%) | 50.3% (45.3 to 55.5) | 51.2 / 47.9 / 51.8 | +4.7 (-0.8 to +10.5) | 70.4% | 63.9% (57.3 to 67.3) |
| C, layer 1 at 2e-4 | 51.2% (46.5 to 55.8) | 49.5 / 53.8 / 50.2 | +5.6 (+0.8 to +10.9) | 64.5% | 57.6% (41.0 to 68.3) |
| D = C (1e-4) + native replay | 50.4% (45.5 to 55.3) | 51.2 / 49.2 / 50.8 | +4.8 (+0.9 to +8.7) | 61.6% | 69.7% (67.7 to 71.7) |
| linear probe on the raw Qwen option states | 85.8% (81.5 to 89.9) | - | +40.3 (+33.7 to +46.9) | - | - |

- D minus C (1e-4): +0.1 (-4.3 to +4.0). C (1e-4) minus B+: +2.9 (-2.4 to +8.0).

**Additional: cleaned VAL and native validation**

| arm (fit folds 1-4, development fold 0, 3 seeds) | cleaned VAL, 46 pools | each seed | minus A | without context | native (600), each seed |
|---|---|---|---|---|---|
| A reference | 55.1% (43.7 to 67.5) | 54.3 / 56.5 / 54.3 | - | 46.4% | 69.0 / 69.0 / 69.0 |
| B+ | 49.3% (36.7 to 61.5) | 43.5 / 47.8 / 56.5 | -5.8 (-25.2 to +10.1) | 22.5% | 31.0 / 33.3 / 34.2 |
| C, layer 1 at 1e-4 | 48.6% (37.1 to 60.3) | 50.0 / 41.3 / 54.3 | -6.5 (-17.7 to +4.4) | 37.0% | 49.3 / 49.0 / 58.0 |
| D | 52.9% (41.9 to 64.1) | 58.7 / 47.8 / 52.2 | -2.2 (-11.1 to +5.7) | 47.1% | 68.7 / 67.2 / 70.5 |
| linear probe | 89.1% (78.4 to 97.8) | - | - | - | - |

**Arm B, calibration control (no training).** The fitted temperature is at the cap of 20 in every fold: on the fit pools the released head's loss only reaches the
equal-score value as the temperature grows (1.504 at 1, 1.2525 at 20, 1.2517 at 1,000 = equal scores). Out of fold: negative log-likelihood 1.524 to 1.254,
calibration error 0.204 to 0.109, training loss 1.080 to 0.815, top-1 unchanged at 39.9%. On VAL: 1.485 to 1.223, calibration error 0.208 to 0.092, top-1 39.1%.

**Reading**
- By the registered acceptance rule (interval above 0 against A AND native at or above 66%), **D passes on the primary evaluation**: +4.8 points (+0.9 to +8.7),
  native 69.7%. C at 2e-4 has an interval above 0 but loses native skill (57.6%); C at 1e-4 misses both conditions.
- **The gain is small and VAL does not confirm it:** D is 52.9% on cleaned VAL against 55.1% for A (-2.2, -11.1 to +5.7). Four arms were compared with A without a
  correction for multiple comparisons, and D's lower bound is +0.9. This is not the "substantially beyond 54.3%" the user set as the convincing outcome.
- **Replay does what it was added for:** with the same layer-1 rate, native accuracy is 69.7% with replay and 63.9% without (57.6% at the higher rate); on VAL
  configuration 67-70.5% against 49-58%. A calibrated start alone (B+) destroys native skill (44.9% out of fold, 31-34% in the VAL configuration).
- **The limit moved from fitting to generalising:** with the higher layer-1 rate the head ranks 70.4% of its own fit pools correctly (A: 54.1%), but only 50.3% of
  unseen pools. The linear probe on the same inputs reaches 85.8% out of fold.
- Seeds agree out of fold (D is above A in every seed); on the 46 VAL pools the seeds spread by up to 11 points.
- User's decision table: "C improves Fable but damages native; D repairs it -> replay is valuable" applies. The head is still about 35 points below the probe, so by
  the user's stated priority the next question is how the head's layers use the candidate representations, not more rate tuning or more pools. Stopped here.
- The agent's addition B+ and the temperature cap of 20 were disclosed to the user before the run.

### 16.15 Direct read-path experiment, exp5 (2026-10-12, 21:15 UTC = 2026-10-13, 02:46 IST)

User (chat, 2026-10-13): GO for a versioned, modified head = the pretrained LAYA head + one linear layer that reads each option's bridge output and adds
`w . LayerNorm(bridge output)` to its score (1,024 weights, start at 0, bridge detached), 5-fold cross-validation, native replay, Kaggle GPU. Required by the user:
the trained head scored with the read path switched off; no-context and options-only controls; checkpoints chosen on TRAIN folds only; stop after reporting.
Kaggle `sangeetasangamkar/laya-head-exp5-run` (Tesla T4, torch 2.11), dataset `laya-head-exp5-v1` (no texts, no CAL / TEST), script `kaggle/head_exp5_gpu.py`,
pack `tools/head_exp5_pack_v1.py`, bundle `results/raw/head-exp5-bundle-v1-20261012T203417Z`, output `results/raw/head-exp5-run-20261012T211558Z`
(`PLAN.json` written before the run, `REPORT.json`, `runs.jsonl`, log). Results with the read path are results of a MODIFIED head, never of the 31 tensors alone.

**Reproduction gate: passed exactly.** The previous arm D (Dprev) gave 50.39% out of fold and 69.67% native, identical to 16.14. The path rate was chosen on the inner
TRAIN development folds: 1e-3 (72.1%) over 1e-4 (50.1%).

| arm (out of fold, 303 TRAIN pools, 3 seeds) | top-1 | minus Dprev (95%) | own fit pools | native (600), mean |
|---|---|---|---|---|
| released head, no training | 39.9% | - | - | 71.5% |
| Dprev: 31 tensors trained, no path (16.14 arm D) | 50.4% | - | 61.6% | 69.7% |
| D0: Dprev without the temperature-scaled start | 48.9% | -1.4 (-5.4 to +2.8) | 65.3% | 70.4% |
| B: 31 tensors FROZEN, path trained at 1e-4 | 51.6% | +1.2 (-5.4 to +7.7) | 55.7% | 68.4% |
| **B: 31 tensors FROZEN, path trained at 1e-3 (chosen)** | **71.1%** | **+20.7 (+15.3 to +26.1)** | 80.0% | 68.7% |
| C: 31 tensors and path both trained | 68.4% | +18.0 (+13.3 to +23.1) | 86.8% | 68.8% |
| **C with the path switched off (the 31 tensors alone)** | **41.5%** | **-8.9 (-13.9 to -4.0)** | 49.5% | 69.1% |
| E: B on half of each run's fit trajectories | 68.6% | +18.3 (+12.4 to +24.0) | 89.5% | 68.2% |
| linear probe on raw Qwen states | 85.8% | - | - | - |

| comparison | difference | 95% | 98.3% (corrected for the three key comparisons) |
|---|---|---|---|
| C minus Dprev (key) | +18.0 | +13.3 to +23.1 | +12.3 to +24.2 |
| C minus B (key) | -2.6 | -6.2 to +0.8 | -6.7 to +1.5 |
| C with the path off minus Dprev (key) | -8.9 | -13.9 to -4.0 | -15.3 to -3.1 |
| C with the path off minus D0 (same recipe without a path) | -7.5 | -13.6 to -1.6 | -14.9 to -0.5 |
| half minus full (B) | -2.4 | -5.8 to +1.0 | -6.6 to +1.6 |
| probe minus C / probe minus B | +17.4 / +14.7 | both above 0 | both above 0 |

**Shortcut controls.** On the 112 TRAIN pools that have no-context features (about 65 fit pools per run), the path trained and tested WITHOUT context (Bnc) scores
75.6% against 71.1% with context (Bsub); difference -4.5 (-12.4 to +3.2). The chosen B, trained with context and scored without it at test time: 62.8% against 73.2%.

**Cleaned VAL (46 pools; fit folds 1-4, development fold 0; 3 seeds).** Dprev 52.9%, D0 58.0%, B 68.1% (minus Dprev +15.2, +3.8 to +27.4), C 63.8%, C with the path
off 45.7%, B with the path off (= released head) 39.1%; probe 89.1%. Without context: B 57.3%, C 52.9%. Loose equivalents counted: B 70.3%, C 65.9%. Native (600) 67-72%
in every arm. Bridge swap, diagnostic only (same Qwen): B 60.1%, C 60.1%. The untrained pool-centred residual of the review gives 35-41% on TRAIN: no gain.

**Reading (against the table written before the run).**
- The reproduction gate passed.
- **The frozen-head read path raised agreement with Fable** (71.1% against 50.4%). The gain belongs to the NEW linear layer, not to the LAYA tensors, and it is still
  14.7 points below the plain linear probe.
- **Joint training did not improve the 31 pretrained tensors; it made them worse.** With the path off they score 41.5%, 7.5 points below the same recipe without a
  path (D0), close to the released head (39.9%). The learning moved into the new layer. Caveat: they were scored at the checkpoint chosen with the path on, so this
  is a lower bound for this recipe, not proof that the tensors cannot improve.
- **The read path predicts substantially without the coding context:** trained without context it matched (75.6%) the version trained with context (71.1%) on the same
  pools. Much of its gain is recognising the teacher's options, not reading the situation. Small sample (112 pools).
- **Native-task retention was measured:** 67-72% in every arm on the 600 native validation items (released 71.5%).
- Half the data is as good as all of it: more Fable pools are not indicated under this setup.
- The user's convincing outcome (C above both the probe and B, and the 31 tensors better with the path off) is not met on any part. The agent's written prediction
  held.
- **No exp5 checkpoint was adopted or saved.** All frozen checkpoints and raw evidence are kept unchanged.
- Scope: agreement with Fable's recorded choice only; not coding-task success, not portability to another model. The A5 benchmark was not run.
- Next direction (user, 2026-10-13): context-dependent, execution-verified supervision, starting with the mutation pilot; and scoring candidates from another agent
  against test results (plan in AGENT_README.md, 2026-10-13 03:30 IST).

### 16.16 Research baseline frozen, mutation pilot v1 / v1.1, head-swap experiment prepared (2026-10-13, about 04:00 IST)

- **Research baseline v1** (step 1): `results/raw/laya-baseline-v1-20261012T215242Z` (MANIFEST.json with sha256 of the pretrained head, the three Fable-trained
  adapters of laya-finetune-all-v1, the code; MODEL_CARD.md). Files referenced in place, nothing copied or retrained. Scoring rule: each adapter alone, mean of
  the three top-1 values (VAL 79.25 / 81.13 / 77.36). The probe is frozen as recipe + data (refitted deterministically).
- **Canonical split correction:** of the 118 replayable states the research report E called "train" (trace file's split field) only 72 are canonical TRAIN
  (24 TEST, 9 CAL, 13 VAL); 16 trace-"val" states are canonical TRAIN: 88 canonical TRAIN states in total. Only those were executed.
- **Mutation pilot v1** (`tools/mutation_pilot_v1.py`, `results/raw/mutation-pilot-v1-20261012T215655Z`): 43 states screened in sha256 order, 20 selected
  (Fable's action turns a failing test run into a passing one, stable over two runs, nothing else changed before the test). Sandbox re-qualified first
  (`fable-wsl-runner-qualification-20261012T215607Z`, 18 of 18 probes). 152 single-site mutants (cmp, bool, logic, num, arith; comments and strings skipped),
  each run twice: WORSE 78, EQUAL_ON_TESTS 36, UNKNOWN 38; 14 of 20 states with >= 2 non-crash WORSE: **not usable by the rule (15)**.
- **Cause:** 5 of 8 TS states run `node --test ... 2>&1 | tail -N`; the exit code is tail's. 25 of 37 TS UNKNOWN show assertion failures with exit 0, and
  Fable's own action in ts-msgfmt:3 fails tests while exiting 0.
- **v1.1** (`tools/mutation_pilot_v1_1.py`, `results/raw/mutation-pilot-v1.1-20261012T220733Z`; user GO): the audited evaluation commands of
  `fable-fidelity-report-20261010/evaluation_command_audit.jsonl` (display filter removed), the same 20 states and the same 152 mutants (identity checked),
  same rule, invalid states not replaced. ts-msgfmt:3 INVALID (Fable's action fails its tests). WORSE 114, EQUAL_ON_TESTS 36, UNKNOWN 2; no crash among WORSE;
  **19 of 20 states have >= 2 WORSE: usable.** Only change from v1: 36 UNKNOWN -> WORSE. 36 mutants (24%) pass the tests: weak tests or equivalent code, not
  proof of equivalence. Diagnostic only: these states were trained on.
- **Head-swap experiment PREPARED, NOT UPLOADED, NOT LAUNCHED** (held for the user's review): script `kaggle/head_swap_gpu.py` (plan in its PLAN dict and
  in the run's PLAN.json), pack `tools/head_swap_pack_v1.py`, bundle `results/raw/head-swap-bundle-v1-20261012T221311Z` (heads.pt 277 MB; reuses the existing private dataset laya-head-exp5-v1).
  Four on-disk heads (coding-decision-agent = reference, laya-typed-decisions, laya-code, laya-conductor) and three random heads (seeds 1001-1003), all frozen;
  only the 8 adapter tensors train (recipe of tools/laya_joint_v1.py adapter_only). Every checkpoint has 36 non-encoder tensors: 31 transferred, 4 act_head
  (LAYA's abstain/escalate gate, not loaded) and 1 temperature scalar. Reproduction gate: coding head seed 0 out of fold 77.89% +- 4, VAL 76.09% +- 4.4.
  Six primary comparisons at 99.17%; a difference counts only if >= 5 points and the corrected interval excludes 0.
- **Found while packing:** the four pretrained heads are close relatives (relative weight difference from coding: typed 0.170, code 0.188, conductor 0.170;
  typed vs conductor 0.048): fine-tunes of a common base, not independent heads.
- **Zero-training swap measured during the local smoke test** (no training involved; RTX 3050; to be re-measured on Kaggle): the Fable-trained adapters in
  front of each head score on cleaned VAL coding 78.3%, typed 76.8%, code 74.6%, conductor 76.8%; random heads 6.5%, 33.3%, 49.3%. Native figures of the smoke
  test used 40 items only and are not reported.

### 16.17 Head-swap experiment result (2026-10-12, 22:29 UTC = 2026-10-13, 03:59 IST)

User GO (chat, 2026-10-13: "Go") for upload and launch. Private dataset `laya-head-swap-v1` (heads.pt 264 MB) + the existing `laya-head-exp5-v1`; kernel
`sangeetasangamkar/laya-head-swap-run` (Tesla T4, torch 2.11); output `results/raw/head-swap-run-20261012T222906Z` (PLAN.json, REPORT.json, runs.jsonl, log).
126 adapter-only runs of about 4 s; every head's 31 tensors identical before and after training in every run.

**Gates:** coding head seed 0 out of fold 77.89% (recorded 77.89%, exact); 3-seed cleaned VAL 78.3% (recorded 76.1%, allowed 4.4). Both passed.

| head (frozen; adapter trained) | out of fold, 303 TRAIN pools (95%) | each seed | no context (112 pools) vs with context | own fit pools | cleaned VAL (46) | VAL no context |
|---|---|---|---|---|---|---|
| coding-decision-agent (reference) | 77.2% (72.8 to 81.3) | 77.9 / 76.9 / 76.9 | 76.5% vs 81.0% | 92.8% | 78.3% | 60.1% |
| laya-typed-decisions | 79.0% (74.6 to 83.3) | 77.6 / 82.5 / 76.9 | 73.5% vs 81.0% | 94.5% | 73.9% | 65.9% |
| laya-code | 76.9% (72.2 to 81.3) | 78.2 / 76.2 / 76.2 | 72.0% vs 79.8% | 92.9% | 72.5% | 60.1% |
| laya-conductor | 79.6% (75.5 to 83.6) | 79.2 / 81.2 / 78.6 | 74.4% vs 82.7% | 90.8% | 75.4% | 62.3% |
| random head 1001 | 83.8% (79.5 to 87.9) | 84.2 / 83.8 / 83.5 | 82.1% vs 85.7% | 99.3% | 80.4% | 68.8% |
| random head 1002 | 83.2% (78.7 to 87.4) | 83.2 / 82.2 / 84.2 | 81.6% vs 85.4% | 99.0% | 82.6% | 68.1% |
| random head 1003 | 83.3% (78.8 to 87.4) | 82.8 / 82.8 / 84.2 | 79.8% vs 84.8% | 98.6% | 81.9% | 68.8% |
| linear probe on raw Qwen states | 85.8% (81.5 to 89.9) | - | - | - | 89.1% | - |

| primary comparison (family of 6) | difference | 99.17% interval | 95% | rule (>= 5 points and corrected interval excludes 0) |
|---|---|---|---|---|
| coding minus random (mean of 3) | -6.2 | -9.0 to -3.7 | -8.4 to -4.2 | **met, against pretraining** |
| typed minus coding | +1.8 | -1.3 to +4.8 | -0.4 to +4.1 | not met |
| code minus coding | -0.3 | -3.6 to +2.9 | -2.8 to +2.1 | not met |
| conductor minus coding | +2.4 | +0.1 to +4.7 | +0.8 to +4.3 | not met (under 5 points) |
| coding: with minus without context (same pools) | +4.5 | -2.8 to +12.3 | -1.4 to +10.3 | not met |
| probe minus coding | +8.6 | +4.4 to +13.2 | +5.2 to +11.9 | **met** |

- **Zero-training swap** (Fable-trained adapters of laya-baseline-v1 in front of each head; cleaned VAL): coding 78.3%, typed 76.8%, code 74.6%, conductor 76.8%;
  random 6.5%, 33.3%, 49.3%. The coding head's adapter works through the three related heads and not through random ones.
- Native accuracy (600 items) is 20-39% for every model with a Fable-trained adapter, pretrained or random: the adapter trained on Fable does not carry
  the native items. Information only.

**Reading (against the table written before the run).**
- Q1: the pretrained heads do NOT beat random heads. Under adapter-only training, a randomly initialised head of the same architecture scores 6.2 points
  HIGHER than the reference pretrained head (83.4% against 77.2%), with almost no spread over three initialisations (83.2-83.8%), and fits its training pools
  better (99% against 93%). Under this interface and recipe the pretrained head is a handicap. This does not show that a better-trained general head could
  never help (user's correction); it shows these four pretrained heads give no advantage here.
- Q2: no specialisation effect: typed, code and conductor are within 2.4 points of coding. Expected from their weights (fine-tunes of a common base).
- Q3: removing the context costs the reference 4.5 points out of fold (not significant by the rule) and 18 points on VAL (development-exposed).
- Q4: the linear probe beats the reference by 8.6 points (meets the rule). The random-head system comes within 2.4 points of the probe (not tested formally).
- Consistent with the head review (16.12-16.13): the pretrained context layer 1 overwrites the incoming signal; a random head does not carry that learned
  transformation.
- By the user's pre-decided reading, head-specialisation research stops for now; the priority is actual coding success (cross-agent execution test).
  No checkpoint saved or adopted. Agreement with Fable only; one backbone.

### 16.18 Rebuilt scorers, GLM-5.3 generation, Go / C# / Rust replay repair (2026-10-13, about 05:10 IST)

- **Rebuild with saved weights** (user GO): the 12 VAL-configuration runs of the head-swap experiment (coding head seeds 0-2, random heads 1001-1003 x
  seeds 0-2) re-run on Kaggle with the unchanged script; **all 12 reproduce their recorded 46 VAL picks exactly**. Weights, configs and sha256:
  `results/raw/head-swap-rebuild-run-20261012T225237Z` (MANIFEST.json, rebuilt/*.pt). Kernel `laya-head-swap-rebuild`, dataset `laya-head-rebuild-v1`.
- **GLM-5.3 generation for the 5 VAL pilot problems** (user GO): `tools/val_outcome_glm_v1.py`, PREREG inside `results/raw/val-outcome-v1-20261012T134540Z/gen-glm-20261012T224831Z`.
  Finished; not yet decoded, executed or scored (paused by the user to fix the languages first).
- **Language repair in the sandbox** (user approval; details and checksums in `results/raw/wsl-toolchains-20261013/MANIFEST.md`): .NET SDK 10.0.302, Rust
  1.96.0, an offline Go module cache (yaml.v3 + its graph) and an offline NuGet fallback folder. Go: go.mod recreated in the replay copy when the recorded
  output proves the module name: 41 of 41 repaired TRAIN/VAL states replay exactly (VAL: 14 of 14, was 0). C#: 2 of 3 VAL states and Rust 1 of 1 pass the
  same tests as the trace but differ in formatting (path form, parallel test order): MISMATCH under the original rule, MATCH under a new rule v2
  (`compare_v2`, added after seeing them, reported next to v1, never replacing it). csharp-kennelcard:2 lacks its project file: not repairable.
  Sandbox re-qualified: 18 of 18 probes. CAL / TEST states were never executed.

### 16.19 GLM-5.3 outcome pilot on 5 VAL problems: complete, NOT informative enough to expand (2026-10-13)

User GO (chat, 2026-10-13): GLM-5.3 generation for the five pilot problems, WSL execution, Kaggle capture, rebuilt scorers; no automatic expansion.
Files under `results/raw/val-outcome-v1-20261012T134540Z/`: `gen-glm-20261012T224831Z` (PREREG, 25 responses, ledger: 50 sends, 25 rate-limit rejections
re-queued, 2,604 s), `pilot-20261012T224831Z` (CANDIDATES.json, AMENDMENT_2.json, OUTCOMES.json, AMENDMENT_3.json, OUTCOMES_A3.json, RESULT_A3.json).
Capture `results/raw/glm-pilot-capture-run-20261013T073754Z` (Kaggle T4, kernel laya-glm-pilot-capture, the VAL capture script unchanged); scores
`results/raw/glm-pilot-score-run-20261013T074323Z/SCORES.json` (Kaggle CPU, kernel laya-glm-pilot-score; no outcomes uploaded; every rebuilt run
re-checked against its 46 VAL picks: all reproduce).

- Generation: 24 answers finished; 1 hit the 65,536-token limit (INVALID as pre-registered). Decoded with the project's v3t decoder: **17 of 25 samples valid**
  (py-retry-policy:2: 0 valid, 4 placeholder contents and 1 file write via the shell). A decode of the same generation already existed (05:02, origin not
  identified); a fresh decode was byte-identical (candidates, capture bundle, config) and was used.
- **Amendment 2** (written after decoding, before any execution): every GLM candidate bundles its Write / Edit with Bash calls (all of them test runs,
  `bash -n` or `chmod +x`); under the cal_outcome_v1 rule all 17 would have been "not a code change". Now each candidate's calls run in order in the
  sandbox, then the recorded test.
- **Amendment 3** (after the A2 run flagged 10 candidates as tampering, before executing them): the protected list matched by substring (permits.py inside
  `python3 test_permits.py`, the module every fix edits) and the Bash check fired on `2>&1`. Whole-word / TESTISH protected files; tampering = a write
  operation targeting a protected file. The A2 outcomes stay on record.
- Every problem's untouched state reproduces the recorded failure (exit 1, twice). **15 of 17 valid candidates pass** (A3); oracle 4 of 5 problems;
  **1 of 5 problems informative** (bash-retryrun:2: 2 pass, 2 fail).
- Validity filtering: invalid samples are removed by the decoder BEFORE any selector; every learned selector and the random-among-valid baseline chose
  from the same pools. Only "GLM sample 1" was measured end to end (an invalid sample 1 counts as FAIL).

| selector | end to end, 5 problems (invalid = fail) | valid only, 4 problems with >= 2 valid | the one informative problem |
|---|---|---|---|
| oracle | 4 | 4 | 1 |
| random, uniform over all 5 samples | 3.0 | - | - |
| random among valid candidates | 3.5 | 3.5 | 0.5 |
| GLM sample 1 as generated | 2 | - | 0 |
| first valid candidate | 3 | 3 | 0 |
| LAYA v4 (3 rebuilt seeds), random head (9), baseline v1 (3), linear probe | 3 | 3 | 0 (every run) |

- **Reading:** every learned selector ties with "first valid candidate". The 16 failing runs on bash-retryrun are ONE problem, not 16 observations: nothing
  is shown for or against any selector. When a GLM candidate is valid it nearly always passes (15 of 17), so the pilot lacks pools with valid failures.
  That the other 18 problems would also be mostly all-pass is an expectation, not a measurement. By the user's feasibility rule GLM-5.3 is not expanded.
- **Next (user, pending):** a new, separately pre-registered pilot with an intermediate-strength generator (candidate: openai/gpt-oss-20b; endpoint and
  output format to be verified first), five predetermined problems, five candidates each, reporting end-to-end and valid-only success separately; the
  execution-verified mutants only as a labelled diagnostic. No retraining, no new Fable data. NVIDIA calls held until the user decides.

### 16.20 gpt-oss-20b outcome pilot: STOPPED at API verification (2026-10-13, 13:40 IST)

User GO (chat, 2026-10-13): a new, separately registered pilot with openai/gpt-oss-20b (two TRAIN verification calls; 5 registered VAL problems
py-event-sched:1, py-trie-suggest:1, ps-shiftbrief:2, ps-loadsheet:1, ps-fleetpatch:1 x 5 samples; amendments 2 and 3 from the start; local WSL execution;
local Qwen capture only after a parity gate; no Kaggle; expansion gate: >= 2 of 5 mixed valid pools). Tool `tools/val_outcome_oss20b_v1.py` (PREREG inside);
`tools/val_outcome_glm_v1.gen` generalised (defaults reproduce the GLM pilot).

- **Verification failed** (`results/raw/oss20b-verify-20261013T080629Z/VERIFY.json`, raw responses kept): call 1 HTTP 504 after 302 s (no answer); call 2 HTTP 200, finish_reason stop, 235
  reasoning characters and an EMPTY answer (no decodable tool action). Both required checks fail, so the pilot stopped as registered: no VAL request was
  sent, nothing was generated, executed, captured or scored. Settings were not changed and no other generator was tried.
- Not attempted: the local Qwen capture and its parity gate (the scoring step depends on generation). The expansion gate was not reached.
- Pre-flight facts recorded for a later attempt: the 5 problems are canonical VAL, their recorded test commands have no display filter, and the VAL oracle
  check shows each fails before Fable's action and passes after it (exit 1 -> 0).

### 16.21 GLM-5.3 low-effort pilot: STOPPED at API verification (2026-10-13, 14:00 IST)

User GO (chat, 2026-10-13: "use glm itself with low accuracy" -> "Go complete report update"). Registration `tools/val_outcome_glmlow_v1.py` (GLM-5.3,
reasoning_effort low, temperature 1.0, top_p 1.0, streamed, 32,768 tokens; the 5 problems registered for gpt-oss-20b; same gate and rules).
- Verification (`results/raw/glmlow-verify-20261013T082223Z`): call 1 HTTP 504 after 302 s; call 2 HTTP 200, finish stop, a decodable action (Read + Bash). The registered rule needs both, so
  the pilot stopped: no VAL request sent, nothing generated, executed, captured or scored.
- Observation: in both stopped verifications (gpt-oss-20b and GLM-low) the two calls ran at the same time on KEY_2 and the slower one got the gateway's
  504 at ~302 s. This points to the gateway / concurrency, not to the model; a sequential verification would need a new registration (user decision).

### 16.22 GLM-5.3 low-effort verification v2 (sequential): STOPPED on answer format (2026-10-13, 14:15 IST)

User GO (chat, 2026-10-13): a new, separately registered verification with sequential calls; the five-problem pilot only if both pass.
`tools/val_outcome_glmlow_v1.py verify_seq` (PREREG_V2), output `results/raw/glmlow-v2-verify-20261013T092636Z` (VERIFY.json, CONCURRENCY.json, raw responses).
- Call 1 (alone): HTTP 200, 82 s, decodable Read action: passed. Call 2 (alone, after call 1): HTTP 200, 80 s, the same action wrapped in a JSON list:
  the v3t decoder rejects it (wrong_top_level): failed. Pilot stopped; no VAL request sent; decoder and gate unchanged.
- No 504 with sequential calls (2 of 2 answered), consistent with the concurrency explanation of the earlier 504s but not proof of it.
- The new blocker is answer-format reliability at reasoning_effort low / temperature 1.0 (1 of 2 answers decodable).

### 16.23 GLM-5.3 low-effort pilot v3: run complete, expansion gate FAILED (2026-10-13, 15:30 IST)

User GO (chat, 2026-10-13): glmlow-v3 with strict singleton-list normalization verified offline on the saved responses, then the bounded 5-problem pilot;
temperature 1.0 kept; no Kaggle. Registration PREREG_V3 in `tools/val_outcome_glmlow_v1.py` (gen_v3); generation
`results/raw/val-outcome-v1-20261012T134540Z/gen-glmlow3-20261013T100634Z` (raw and normalized text kept, CONCURRENCY.json); outcomes and result in
`results/raw/val-outcome-v1-20261012T134540Z/pilot-20261013T100634Z` (OUTCOMES_A3.json, OUTCOMES_A3b.json, RESULT_A3b.json).
- Offline verification: the 3 saved HTTP-200 answers all decode (one through the unwrap); negative cases stay rejected. No new verification call.
- Generation: 25 of 25 HTTP 200, 0 timeouts (17 rate-limit rejections re-queued), mean 218 s; 4 to 5 requests in flight on a key for 14 of the 25
  answers, so concurrency alone did not cause 504s this time. The unwrap applied to 0 answers.
- **Validity: 5 of 25.** The 20 invalid answers are genuine model format errors at low effort (extra closing brackets, a made-up "[call] {...}" or
  "<tool_call>" syntax, prose before the JSON, bare lists of calls); the decoder was not changed after seeing them.
- Execution bug found and fixed: the amendment-2 parser dropped single-call actions (rendered as a bare object), so one Write candidate was wrongly
  "not a code change" (OUTCOMES_A3.json, kept). Fixed rerun OUTCOMES_A3b.json (same rule). The GLM-max pilot was not affected (all its candidates are lists).
- Every baseline reproduces the recorded failure. py-event-sched:1: 2 valid (1 PASS, 1 FAIL), informative; py-trie-suggest:1: 2 valid, both PASS;
  ps-shiftbrief:2: 1 valid, FAIL; ps-loadsheet:1 and ps-fleetpatch:1: 0 valid. **Oracle 2 of 5; informative 1 of 5: expansion gate FAILED (needs 2).**
- Local capture verified first: parity re-capture of the 4 GLM-max pools (`results/raw/local-capture-parity-glmmax-20261013T100901Z/PARITY.json`):
  worst cosine 0.999992 (real context), 0.999957 (no context), top choices 12 of 12 identical; the local scoring path reproduced the Kaggle scores
  128 of 128. Then the 2 v3 pools with >= 2 valid candidates were captured locally and scored (folders below).

| selector | end to end, 5 problems | the informative problem (py-event-sched:1) |
|---|---|---|
| oracle | 2 | 1 |
| GLM raw sample 1 | 0 (every sample 1 invalid) | 0 |
| first valid candidate | 1 | 0 |
| random among valid / uniform over all 5 samples | 1.5 / 0.6 | 0.5 |
| LAYA v4 (3), random head (9), baseline v1 (3), linear probe | 2 each | 1 (every run, with and without context) |

- Reading: on the informative problem every learned scorer picked the passing fix (in the GLM-max pilot every scorer picked a failing one). Two
  informative problems in total, one each way: still no evidence for or against any selector, and LAYA is never distinguishable from the probe or the
  random head. By the registered gate the pilot stops here. Next: the user's decision.
- Folders: local capture results/raw/local-capture-glmlow3-20261013T101558Z; scores results/raw/glmlow3-score-local-20261013T101735Z; local-vs-Kaggle scoring check results/raw/glm-pilot-score-local-check-20261013T101346Z.
