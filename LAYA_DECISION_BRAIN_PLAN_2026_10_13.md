# LAYA decision brain: architecture and research plan (2026-10-13)

Planning only. Nothing here is running or approved to run; every experiment below needs the user's GO and runs on Kaggle (user rule, 2026-10-13).
Evidence references: HANDOFF_2026_10_11_STATUS.md sections 16.12-16.23.

## 1. What we know (measured)

| Finding | Evidence |
|---|---|
| Behind a trained Qwen adapter, random heads beat every pretrained LAYA head (83% vs 77% Fable agreement); the linear probe is higher still (86%) | 16.17 |
| The pretrained head's first layer overwrites the incoming Qwen signal (probe 87% at its input, 56% after layer 1's feed-forward step) | 16.12-16.13 |
| Head-only training cannot fix that (best 50%); a bolted-on linear layer does the work instead, and much of it needs no context | 16.14-16.15 |
| Fable agreement is partly style recognition (options-only and no-context controls stay high) | 16.15, VAL eval |
| Two execution pilots gave 2 informative problems out of 10, one going each way: no selector is shown to help or hurt | 16.19, 16.23 |
| The pipeline works end to end: sandbox (now Python, TS, PowerShell, Java, Ruby, Bash, Go, C#, Rust), capture, frozen scoring, parity-checked | 16.18, 16.23 |

Conclusion: the pretrained LAYA head, as currently connected (Qwen hidden states -> adapter -> head), adds nothing measurable. The problem is not
data volume (half data is as good as full) and not optimisation; it is the interface between Qwen's representations and what the head was trained on.

## 2. What the decision brain should do

Not just "rank Fable's action first". Five capabilities, each with its own label source and its own test:

| Capability | Question it answers | Label source that is not a shortcut |
|---|---|---|
| Choose | Which candidate action is best now? | execution outcome of each candidate from the same state |
| Predict outcome | Will this action make the failing test pass (or break something)? | sandbox runs (pass / fail / unstable) |
| Ask for evidence | Is there enough information to act, or should the agent read / run first? | states where acting first fails but reading first then acting passes |
| Abstain / escalate | Is every candidate bad (NONE)? | pools where every candidate fails the tests |
| Calibrate | How sure is it? | held-out outcome data (ECE, Brier) |

LAYA's native primitives already map onto this: choice, score (graded), noul (NONE / abstain). The head was designed for these decisions; it was
never tested on them with execution labels.

## 3. Architecture options (to decide before any training)

| Option | How | For the goal "any local model, later Claude / Codex" | Risk |
|---|---|---|---|
| **A. Text route: LAYA with its own encoder** | Candidates and context as text into LAYA's released ModernBERT encoder + head, exactly as pretrained. Model-agnostic by design: works for any Qwen, Claude or Codex because it only reads text | Best fit: one component, no per-model adapter; Claude needs no hidden states | Uses the released encoder (frozen, not trained): needs the user's explicit OK under the "no encoder touched" rule. Context length (2,048 tokens for coding-decision-agent) may truncate long sessions |
| B. Current route: Qwen states -> adapter -> LAYA head | What we have; one adapter per model | Needs hidden states, so not Claude / Codex | Measured: pretraining gives no advantage here |
| C. Head trained from scratch on decision data, same architecture | Random-head result says the architecture learns fine through an adapter | Same as B | Loses the "pretrained LAYA" premise; becomes a new model |
| D. Simple probe as the scorer | Linear read-out on frozen features | Same as B | No structure for abstain / ask-for-evidence; strongest baseline so far |

Recommendation: decide between A and B first with a cheap, read-only comparison (no training): LAYA through its own encoder on text vs the Qwen
route, on the SAME execution-verified pools. Option A is untested in this project for Fable/outcome pools (native accuracy through its own encoder is
71.5% on its own 600 validation items, measured earlier). Options B-D stay as baselines.

## 4. Learning approach (after the architecture choice)

1. **Execution-verified data first.** Sources we can build: (a) mutations of verified fixes (TRAIN, 19 usable states now, ~6 failing variants each;
   scale to the 88 canonical TRAIN replay states, Go / C# / Rust now included); (b) natural candidates from a generator that yields mixed pools;
   (c) "read first vs act first" pairs for the ask-for-evidence capability.
2. **General typed-decision data second**, only licence-filtered sources, deduplicated, Open-Jev excluded (Jev rule); purpose: keep the general skills.
3. **Fable imitation only as an auxiliary signal**, never the headline metric.
4. **Anti-shortcut rules built in:** same-author candidate pools; options-only, no-context and option-order controls mandatory in every report;
   trajectory-disjoint splits; pools with one passing and one failing version of the SAME code where possible (style cannot decide).
5. **What trains:** depends on section 3. Option A: the head (and possibly upper encoder layers, needs approval); option B: adapter, then head.

## 5. How to measure real improvement

- Primary: selection success on UNSEEN problems with informative pools (at least one passing and one failing valid candidate), paired against the
  generator's first valid candidate, the probe and random choice; per problem, not per candidate.
- Size: to detect a 15-point gain over "first valid" with a paired test needs roughly 80-100 informative problems (estimate, unmeasured). The pilots
  gave 2 informative of 10, so problem supply is the bottleneck, not compute.
- Problem supply: VAL has 23 replayable fail-to-pass decisions (development, already used); CAL / TEST are protected for the final assessment; more
  replayable states can be materialised from canonical TRAIN for training only. A held-out source of new problems is needed for a convincing result.

## 6. Connecting to other models (later, only after section 5 shows value)

- Option A needs no per-model work: any agent (local Qwen, Claude Code, Codex) sends context + candidate actions as text to one scoring service.
- Claude Code: a PreToolUse hook sees one proposed action; ranking needs the agent to propose several (or the hook scores "act vs gather evidence").
- Codex: an MCP tool or orchestration wrapper with the same service.

## 7. Milestones (each needs GO; compute on Kaggle)

| # | Milestone | Gate to the next |
|---|---|---|
| 1 | Decide option A vs B: read-only comparison on existing execution-verified pools (no training) | A or B clearly better on informative pools, with controls |
| 2 | Build the execution-verified dataset (mutations at scale + natural candidates), with controls | enough informative TRAIN pools for training; a separate held-out set |
| 3 | Train the chosen route on execution labels (+ general decisions) | beats the probe and "first valid" on held-out informative problems |
| 4 | Final assessment on protected CAL / TEST problems | pre-registered, once |
| 5 | Integration: scoring service, Claude Code hook, Codex MCP | end-to-end agent runs with and without the scorer |

Open decisions for the user: (1) is using LAYA's released encoder frozen (option A) allowed under the "no encoder touched" rule? (2) which held-out
problem source for milestone 3? (3) is the general typed-decision data in scope (licences, Jev rule)?

## 8. Re-plan after all scoring is complete (2026-10-13, 17:30 IST)

All numbers: ALL_NUMBERS_2026_10_13.md. Three facts now decide the plan:

1. **Fable agreement is not the bottleneck and not the goal.** Adapters reach 76-80%, random heads 83%, the probe 86-94%; the pretrained head adds nothing.
2. **The current features cannot see code correctness.** On the mutation diagnostic every scorer stays within ~0.02 of uniform between a working fix and
   its one-token-broken copies (16.24). Mean-pooling a frozen model's states over a whole file erases a one-token change. No head, adapter or probe on
   these features can learn "this patch is correct"; more data or more training will not change that.
3. **Execution already beats every scorer at choosing fixes.** Running the candidates' tests (the oracle) solved 4/5 and 2/5 problems where the best
   learned selector solved 3/5 and 2/5.

What follows for the design:
- **Correctness of code is decided by execution, never by the scorer.** The project's evidence order already says this (protected tests / real
  execution above the head). The practical system for "which fix?" is: generate several candidates, run them in the sandbox, keep the passing ones.
- **LAYA's job is the decisions execution cannot answer cheaply**: what to do next (read, search, run tests, edit, ask), whether there is enough
  evidence to act, when to stop, when every option is bad, and which passing fix to keep when several pass (smaller, safer). These are visible in
  the features (they differ by tool, target and intent, not by one token).
- **If code-level discrimination is ever wanted from a learned scorer**, it needs a different representation, not more training: features of the
  edited region only (diff-focused), or a model that reads context and option together token by token. That is a new capture protocol (versioned).

Proposed next steps (each needs GO; compute on Kaggle; nothing started):
1. **Diff-focused feature check (read-only, small):** recapture the 19 mutation pools with features of the changed region (option tokens around the
   edit, or option minus pool mean at token level) and refit only the linear probe out of fold. Answers in one Kaggle run whether ANY frozen-feature
   scorer can see a one-token bug. If not, close the "learned scorer picks code" line for good.
2. **Action-type decision dataset:** from the Fable trajectories, label next-action decisions by what happened next in the sandbox replay (did the
   test run after this action pass, did a read precede a successful edit), with same-author pools and no-context / options-only controls. This is where
   LAYA's choose / score / none primitives fit.
3. **Execution-in-the-loop selector for fixes:** generator -> sandbox -> pick passing; LAYA only ranks among passing candidates or decides when to stop.
4. The option-A text route (LAYA with its own encoder) stays a candidate for steps 2-3, pending the user's decision on using the released encoder frozen.

## 9. Next-step direction (user, chat 2026-10-13) and the diff-focused diagnostic protocol (FOR REVIEW, not launched)

**Direction.** The mutation result is reported on all 19 states (primary), with the 7 states no scorer trained on reported separately; no 18-state
substitute. The next, optional experiment is a bounded diff-focused feature diagnostic on the same 19 states. If it fails: close the code-patch ranking
line for the current frozen features and prioritise generate candidates -> run sandbox tests -> keep passing candidates, with LAYA used for decisions
such as read / test / ask / stop and for choosing among several passing candidates. Before any broader training, solve the shortage of informative
held-out pools. Correction to the record: capture v2 did not change the total-token limit; it routes a pool whose COMBINED batch exceeds it through
separate per-option forward passes and keeps the individual forward-pass guard (prefix + one option must fit; nothing truncated).

### 9.1 Question

Can ANY scorer on frozen Qwen3-4B features tell Fable's passing fix from its one-token-broken copies when the features describe the edited region
instead of the whole option?

### 9.2 Data (unchanged)

- The 19 pools of `results/raw/mutation-score-bundle-v1-20261013T104646Z`: same contexts, options, option order (sha256), request hashes, private key.
  17 pools of 5 options, 2 of 4. 19 states = 19 distinct trajectories. 7 states were never in any scorer's training pools.
- Labels: v1.1 execution outcomes (Fable's action PASS; mutants WORSE, crash-free). Same backbone revision, layers [18, 24, 30], fp16 storage.

### 9.3 Features (new capture protocol, versioned `diff_capture_v1`)

- **Edit sites, label-free:** tokenise each option exactly as the existing capture does (independent suffix after the same prefix); align every option's
  token ids to the pool's first option in hash order (difflib on token ids); the pool's sites = every token position where any option differs. Every
  option is represented over ALL sites of its pool (its own tokens at each site), so no option's span size reveals which option is the reference.
  The private key is never read by the capture.
- **Span feature:** mean hidden state over each option's site tokens +- w tokens. Primary: w = 8, centred by the pool mean (option minus the mean of
  its pool's options). Secondary (reported, not decisive): w = 8 uncentred; w = 32 centred.
- Taken from the same forward pass as the existing mean-pooled features (packed when the combined batch fits, otherwise per-option passes with the
  individual guard), with real context (B) and with the neutral prefix (A).

### 9.4 Evaluation

- **Linear probe, out of fold, grouped:** leave one state (= one trajectory) out, 19 folds. Training data: the other 18 pools' options, label 1 for the
  passing fix, 0 for each mutant. Same recipe as the project's probe (standardise, logistic regression, class_weight balanced); C from {1e-3, 1e-2, 1e-1}
  chosen by an inner grouped 3-fold CV on the 18 training states only. Prediction: the held-out pool's highest-scoring option.
- **Metric:** per state, 1 if the top choice is the passing fix. Primary: mean over all 19 states. Reported separately: the 7 unseen states.
- **Controls (same folds):** (a) the same probe on the existing mean-pooled features (expected near chance; ties the result to the feature change);
  (b) the no-context (A) span features; (c) option-only baselines: uniform random (1/n), shortest, longest; (d) **lexical baseline:** the same probe
  on a bag of the option's own token ids at the sites (no Qwen), to catch "the probe learned which tokens a mutation produces"; (e) **label-permutation
  null:** 1,000 runs with the positive option chosen at random inside each training pool; p = share of runs with mean top-1 >= the observed one.

### 9.5 Validation checks (stop on any failure, before any evaluation)

1. Request, option and order hashes identical to the bundle; the key file is not among the capture inputs.
2. Whole-option span reproduces the existing mean-pooled features: cosine >= 0.999 against `laya-mutation-features-v2` for every option (B and A).
3. Every option has a non-empty site span; every pool has >= 1 site; all values finite.
4. Deterministic repeat on 2 pools (identical tensors).
5. The one over-limit pool uses per-option passes with each pass within the guard.

### 9.6 Pre-registered criteria (primary feature, all 19 states)

| Outcome | Rule | Decision |
|---|---|---|
| PASS | top-1 >= 50% (>= 10 of 19; chance is about 21%) AND permutation p < 0.05 AND >= 15 points above the lexical baseline | frozen features can see the bug when focused on the edit; consider a diff-focused scorer as a separate, later proposal |
| SHORTCUT | the first two hold but the lexical margin fails | treated as FAIL (token identity, not understanding) |
| FAIL | top-1 < 50% or p >= 0.05 | close the code-patch ranking line for the current frozen features; prioritise generate -> sandbox -> keep passing |

The 7 unseen states and the no-context result are reported, not decisive (too few). With 19 states only a large effect can be detected: 10 of 19 against
a chance of about 0.21 has a one-sided binomial p of about 0.002 (estimate).

### 9.7 Kaggle runtime and cost (estimates, unmeasured for this protocol)

- Capture, T4 GPU: model download and load about 2 min, validation about 6 min, B about 5 min and A about 2 min (the earlier capture of the same pools
  took 260 s and 98 s; site spans come from the same forward pass) -> about 15-20 min.
- Probe evaluation, Kaggle CPU: 19 folds x 3 C x 4 feature sets + lexical baseline: under 2 min; permutation null 1,000 x 19 folds: about 5-15 min.
- Cost Rs 0 (free Kaggle quota; about 0.3 GPU hours of the weekly allowance). No NVIDIA calls.

### 9.8 What needs the user's decision before launch

1. GO for the capture kernel (new versioned capture script) and the CPU evaluation kernel.
2. **Upload of the 19 states' pass / fail labels** (TRAIN, execution-verified) to a private Kaggle dataset: the probe must be trained inside the
   evaluation kernel. Until now these labels stayed on this PC.


### 9.9 Result (2026-10-13, 18:40 IST): PASS, with a correction and an open confound

Primary 89.5% (17/19; unseen 85.7%), p = 0.001, lexical 26.3%: verdict PASS (HANDOFF 16.25). But the existing whole-option features reach 100% with the
same probe, so section 8 point 2 ("the current features cannot see code correctness") is WRONG as written: the features carry the signal; the scorers
trained to imitate Fable do not use it. Before any "learned scorer picks code" work resumes, one cheap control must run: the probe (same CV) on pools
of each original vs its 36 test-PASSING mutants (EQUAL_ON_TESTS). If it still rejects them, it detects edits, not correctness, and the section-8
recommendation (execution decides code correctness) stands; if it accepts them, a pass / fail scorer trained on execution labels becomes a real option.
Needs the user's GO (Kaggle CPU, minutes; features of the passing mutants need one more capture of those pools, about 10 min T4).

### 9.10 Edited-only control result (2026-10-13, 22:00 IST): ONLY WITH ORIGINAL

The 9.9 PASS was the original-as-centre shortcut (an untrained similarity rule finds the original in 19/19 pools). With every candidate edited, the probe
is above chance (73% vs 30%) but no better than the changed tokens alone (67%) or the mutation operator alone (60%), does not need the context, and
falls below those baselines on operator-matched pools. So: **close the code-patch ranking line for the current frozen features.** Priority, as directed:
generate candidates -> run sandbox tests -> keep passing candidates; LAYA for read / test / ask / stop and for choosing among several passing candidates;
before any broader training, solve the shortage of informative held-out pools. Section 8 point 2's practical conclusion (execution decides code
correctness) stands, now for the right reason.

## 10. Next-action decisions: plan (user direction 2026-10-13; feasibility report NEXT_ACTION_FEASIBILITY_2026_10_13.md)

Goal unchanged: a decision component that helps a smaller model finish more tasks with fewer unnecessary actions. The target is WHEN to read / search /
test / edit / ask / stop, labelled by evidence of task progress, never by "the action before a success was right".

1. **Inventory and feasibility (done, read-only):** 115 canonical TRAIN and 58 canonical VAL replay states; 0 informative next-action-TYPE decisions in
   our own data; the on-disk nebius set has 1,741 tasks with both resolved and unresolved runs (19,602 runs, 859 repos), the best natural source of
   same-state branching; 192 replay states carry 4 never-executed Qwen3-30B proposals each.
2. **Build a small reliable decision dataset (next GO):** first the nebius branching count (Kaggle CPU, gate: >= 300 decision points from >= 100 repos);
   then labels from natural branching (repo-disjoint) or, if the gate fails, from executing the saved proposals with a short fixed continuation in the
   sandbox. Validity rules are filters and baselines, never the target. Rollout outcomes may later supervise training (user), not only evaluate.
3. **Small Kaggle comparison:** a simple outcome-trained probe first (the existing scorers were trained for imitation and are not a fair judge), against
   rule baselines R1-R6 and a random-head system with every shortcut control; pretrained LAYA only if the labels prove informative.
4. **Final gate:** task success and tool cost with and without the decision system on independent problems; CAL / TEST protected; VAL is development.

### 10.1 Branching count result (2026-10-14): HOLD on nebius exact branching as a label source

3,741 exact branchpoints with different action types (793 repos), but they are early orientation steps (view vs pwd, pytest --version vs find); the
decisions that matter are absent because histories stop matching once real work starts. Options for the user: (a) controlled continuations from our 115
TRAIN replay states (saved proposals + Fable's action, each continued by one fixed policy in the sandbox): same snapshot, so the environment is
verified identical; (b) an abstracted-state matching tier on nebius (same files changed + same last test result), clearly labelled exploratory and
observational; (c) stop pursuing outcome labels for next actions and keep the rule baselines.

### 10.2 Option A step 1 (2026-10-14): not eligible; pilot not run

- 115 canonical TRAIN replay states (74 trajectories); all 115 snapshots verify (every recorded file present with its recorded sha256).
- The saved proposals cannot supply next-action choices: all 768 of them (192 states; 460 at the 115 TRAIN states) are Write (707) or Edit (61): the
  replay set was built with `min_code_proposals: 4`, i.e. states were SELECTED for having 4 code-edit proposals. No read, search, test or stop proposal
  exists. Every proposal also carries finish_reason `length`; they all parse as JSON, but whether each file body is complete cannot be confirmed
  without execution.
- Result: 0 states with >= 2 meaningfully different action TYPES; the only alternatives are other code edits (the code-patch ranking question, closed
  in 16.26). Following the user's rule (do not manufacture action-type diversity), the 36-continuation pilot did NOT run; no model was called.
- A further limit for any read / search alternative here: the states are trace-reconstructed (only files the original session saw exist), so reading
  a file the session never opened would fail for an artificial reason.
