> NOTE (main session, 2026-10-12): the trace file marks rows only as train / val; canonical CAL and TEST trajectories are inside its "train" rows. The counts in this report therefore span TRAIN, CAL and TEST trajectories together. 3 example lines from protected trajectories were redacted. Aggregate counts only; no label, model or score was involved.

# Feasibility study: recorded branches (B) and retrieval (D), TRAIN rows only

Date: 2026-10-12. Read-only study. No model calls, no network, no project files changed.
Source: `data/frontier/fable5-DSFFGFG456/traces.jsonl` (12,490 rows).

Split check: the `split` field has two values, `train` (11,436 rows) and `val` (1,054 rows).
Only `train` rows were parsed for analysis. `val` rows were skipped at parse time and never printed, counted or analysed.

All numbers below are measured by scripts in
`C:/Users/Satvik/AppData/Local/Temp/claude/D--local-model/751b39fe-10e2-469e-8229-6be9ea75a314/scratchpad/`
(`s1_collect.py`, `s2_ctx.py`, `s3_analysis.py`, `s6_followups.py`, `s8_similarity.py`).
Heuristic judgements are labelled as such. Nothing here is an estimate of alternative quality.

## 0. Data structure and limits (measured)

- Each row is one decision. `messages` ends at the target assistant message (`messages[target_message_index]`).
  Tool results of a decision appear in the row of the next step of the same trajectory.
  The join key is (trajectory, tool_call_id). Rows per trajectory equal `assistant_steps` in all 2,179 trajectories; no duplicate (trajectory, step) pairs.
- Tool names come in two spellings: `Bash/Read/Edit/Write` and `bash/read/edit/write`.
  All 1,453 lower-case calls have no tool-result message in their trajectories
  (lower-case bash 793, read 420, edit 154, write 86). Two capitalised Write calls also lack a result.
  **So failures in those trajectories are not observable.** The failure analysis covers only the other trajectories.
- `lang` is messy and used as-is: seed-authoring 2,966 rows, en 1,874, python 1,116, non-code 1,030, go 888, ts 865, English 239 (separate from `en`), and more.
- `category` is dominated by seed-authoring: 1,504 of the 3,465 Bash-only decisions.
- Tool-result failure text is short and sits at the start of the result (`Exit code N`, `Error: ...`).
  An earlier raw survey counted 112 "String to replace not found" lines. Those are occurrences repeated across later rows.
  Joined by call id there are only **4 distinct** failing Edit calls.

## 1. Basic counts (train)

- Trajectories: 2,179. Decisions: 11,436. Decisions per trajectory: median 3, max 106.

Recorded action (first-tool category; case variants merged):

| Category | Decisions | Share |
|---|---:|---:|
| Bash | 3,465 | 30.3% |
| multi-call (2 or more tool calls) | 2,532 | 22.1% |
| no tool call (text only) | 2,168 | 19.0% |
| Write | 1,278 | 11.2% |
| Edit | 495 | 4.3% |
| Read | 318 | 2.8% |
| other (non-coding API tools, many distinct names) | 1,180 | 10.3% (derived: total minus the rows above) |

- Multi-call compositions (top): Read only 468, Bash+Read 386, Write only 296, Bash only 231, Edit only 167.
- Calls per decision: 0 calls 2,168; 1 call 6,736; 2 calls 1,702; 3 calls 463; 4 calls 266; 5 or more 101.
- Exact repeated states (identical preceding context within one trajectory): **0**.
- Exact repeated actions (same tool and same arguments within one trajectory): 44.

## 2. Idea B: failed action, then a retry (measured)

Failing calls with a known result (capitalised trajectories only):

| Tool | Failure marker | Calls |
|---|---|---:|
| Bash | `Exit code N`, N not 0 | 115 |
| Edit | `Error: String to replace not found` | 4 |
| Edit | `Error: No changes to make` | 2 |
| Edit | `Error: File has not been read yet` (precondition) | 26 |
| Write | `Error: File has not been read yet` (precondition) | 20 |

Pairing rule: a failing call is paired with the first call in the next 1 to 3 decisions that has the same tool and the same file path (Read/Edit/Write) or the same command family (Bash). Family = first command word, plus the second word for runners (`go test`, `npm run test`, `python -m x`). Identical retries are counted separately and are not alternatives.

| Tool, failure | Alternative pairs | Exact repeats excluded | Precondition pairs |
|---|---:|---:|---:|
| Bash, exit not 0 | 49 | 19 | - |
| Edit, not found | 2 | - | - |
| Edit, not read yet | - | 7 | 4 |
| Write, not read yet | - | 16 | 4 |

- Alternative pairs (precondition excluded): **51**. Decisions with at least one: **51 (0.4% of train decisions)**.
- Retry distance: 1 step 34, 2 steps 9, 3 steps 8.
- The retry itself failed in 6 of the 49 Bash pairs.

Caveat on state (the same-state question), heuristic:

| Status | Pairs | Share | Meaning |
|---|---:|---:|---|
| same_state_clean | 11 | 21.6% | Failing action left the workspace unchanged (by heuristic), and no intervening or sibling change. |
| failed_cmd_may_have_mutated | 25 | 49.0% | Failing Bash command contains a mutating verb (regex); its effect is unknown. |
| changed_between_or_sibling | 15 | 29.4% | A later, intervening or same-decision call changed state (13 Bash, 2 Edit). |

- Write: no usable pair. Every failing Write is the precondition error, which changes nothing. A successful Write always changes the file, so a second Write is never from the same state.
- Purpose check (heuristic): 36 of 49 Bash pairs share at least one file name between failed and retry command (73%). Of the 11 clean pairs, 6 share a file name.
- Similarity of failed vs retry text (difflib ratio): 8 of 51 near-duplicates (ratio at least 0.90, e.g. `head -14` changed to `head -12`), 20 moderate, 23 low. Of the 11 clean pairs, 2 are near-duplicates.
- Failed Bash commands: 11 are test runs, 4 are inspection commands (ls, cat, grep and so on). The rest are other commands.

Examples (each field truncated to 200 characters; from the measured pairs):

1. [Bash, exit 1, same_state_clean] step 33 to 34.
   failed: `cd workspaces/ts-fuellog && node --test test_fuellog.ts 2>&1 | head -14`
   result: `Exit code 1 /bin/bash: line 1: cd: workspaces/ts-fuellog: No such file or directory`
   retry: `cd workspaces/ts-fuellog && node --test test_fuellog.ts 2>&1 | head -12` (near-duplicate, not a real correction)
2. [Bash, exit 1, same_state_clean] step 32 to 33.
   failed: `node probe.ts`
   result: `Exit code 1 node:internal/modules/cjs/loader:1522 throw err; ^ Error: Cannot find module 'probe.ts' ...`
[line redacted 2026-10-12 by the main session: it quoted an example from a canonical TEST trajectory]
3. [Bash, exit 1, same_state_clean] step 65 to 66.
[line redacted 2026-10-12 by the main session: it quoted an example from a canonical TEST trajectory]
[line redacted 2026-10-12 by the main session: it quoted an example from a canonical TEST trajectory]
   retry: same loop with a rewritten Python body (plausible alternative for the same purpose)
4. [Edit, not found, changed_between_or_sibling] step 4 to 5, file `streamstats.py`.
   failed: Edit `old_string` = "mean at flush time." (the retry is at the next step, so the flag comes from a sibling call in the same decision, by heuristic)
   result: `Error: String to replace not found in file. String: mean at flush time. ...`
   retry: Edit of the same file with a different `old_string` and `new_string` (a genuine correction; status is changed only because a sibling call changed the file)
5. [Bash, exit 2, same_state_clean] step 49 to 50.
   failed: `python3 src/generate_traces.py --smoke --only rb-vanpool && python3 -c "..."`
   result: `Exit code 2 python3: can't open file 'workspaces/rs-gatebadge/src/generate_traces.py': [Errno 2] No such file or directory`
   retry: `cd . && python3 src/generate_traces.py --smoke --only rb-vanpool && ...` (the retry's `cd .` is a no-op, so it is not a fix)

## 3. Idea D: retrieval of alternatives from other decisions (measured)

Pool 1: other trajectories with the same `lang` and the same recorded tool category (all decisions):

| Bucket (other trajectories) | Decisions |
|---|---:|
| 0 | 31 |
| 1 | 65 |
| 2 to 4 | 601 |
| 5 to 9 | 523 |
| 10 to 49 | 4,796 |
| 50 or more | 5,420 |

- Median 46. At least 1: 99.7%. At least 2: 99.2%. Bash decisions only: median 43, at least 2 for 99.9%.

Pool 2, Bash grounding. Recorded action = first Bash command of the decision (4,181 decisions with at least one Bash call). Candidate = a Bash command from another trajectory with the same `lang`. Groundable = every file name in the candidate (tokens with an extension) appears in this decision's context (the messages before the target, plus their tool-call arguments). Commands with no file name are vacuously groundable.

| Measure (same `lang`) | Decisions | Share |
|---|---:|---:|
| At least 1 groundable, non-identical alternative | 4,177 | 99.9% |
| At least 2 such alternatives, where each names a present file | 2,683 | 64.2% |
| Identical command from another trajectory is groundable here | 953 | 22.8% |

- Of all groundable candidates, 33.4% of non-identical ones name no file at all (vacuous, e.g. `ls -la`). Identical share among groundable candidates: 0.1%. Test-run share among non-identical groundable candidates: 11.8%.

Pool 3, the brief's "same task family" version (same `lang`, same `category` as the task-family proxy, Bash; candidate must name at least one file present here; identical excluded):

| Measure | Decisions | Share |
|---|---:|---:|
| At least 1 such alternative | 2,378 | 56.9% |
| At least 2 such alternatives | 2,224 | 53.2% |
| At least 2 that are neither inspection-only nor test runs | 2,089 | 50.0% |

- Median other-trajectory pool: 10. Composition of the 1,065,471 candidate (decision, command) pairs: inspection-only 34.2%, test-run 3.9%, substantive 61.9%.

## 4. Assessment

**What is measured:** every count above, on train rows only. Heuristics are the state-change rule (regex on mutating verbs), the test-run and inspection regexes, the command-family rule, the failure markers, and the file-name grounding test.

**What is not measured:** whether any alternative is a valid same-state alternative. No outcome exists for an unexecuted branch, and nothing here runs anything. Nothing below is an estimate of quality.

**Idea B (recorded branches): too sparse to use as a dataset.**
- Clean same-state pairs: 11 (0.1% of train decisions). Upper bound with all retries counted: 51 (0.4%).
- Edit pairs: effectively zero. Only 4 distinct failing Edit calls exist. Both alternative Edit pairs are flagged as changed by a sibling or intervening call.
- Coverage is further reduced because lower-case-named trajectories have no tool results (1,455 calls).
- Main weakness: the same-purpose test is weak. Retries include `cd .`, `head -12` against `head -14`, and retries that touch a different file. A nonzero exit is not always a failure (grep, diff and which can exit 1 on a normal answer). About half of the pairs (25 of 51) involve a Bash command that may have changed state.
- Usable only as a handful of seed examples that need manual or execution-based checks.

**Idea D (retrieval): plentiful candidates, weak validity.**
- Surface grounding is easy. About half of Bash decisions (50.0%) get at least two substantive, file-grounded alternatives from other trajectories with the same language and category.
- Grounding only checks that named files appear in the context. It does not check that the file is relevant, that the command is correct for this state, or that it serves the same purpose. The test is a surface check.
- The pool is skewed toward seed-authoring (1,504 of 3,465 Bash-only decisions), and the `lang` field is inconsistent (`en` and `English` are separate values).
- Under the project rules these would be "retrieved, unverified" options. They could serve as non-selected imitation options, but never as "verified wrong" or as counterfactual outcomes. Verifying any of them would need same-state sandbox runs (`tools/fable_wsl_runner.py`).

**Recommendation.** D is the only idea that yields usable volume, and it is usable only as unverified, non-selected options with a measured validity check added. B does not produce a training-scale set from these traces; at most it is a small probe set. For real same-state alternatives, the sandbox replay is the source to pursue, not these traces.

## 5. Files

- Report: `D:\local model\results\raw\research-20261012\D_branches_and_retrieval.md` (this file).
- Scripts (scratchpad, not in the project): `s1_collect.py` (single pass, train rows only), `s2_ctx.py` (grounding tokens for Bash decisions), `s3_analysis.py` (B and D), `s4_check_join.py` (join check), `s5_inspect_lower.py` (lower-case format check), `s6_followups.py` (purpose check and category pool), `s7_examples.py`, `s8_similarity.py`.
