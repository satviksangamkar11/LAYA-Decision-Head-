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
