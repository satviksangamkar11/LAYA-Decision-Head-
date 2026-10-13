# C. Pool failure funnel, TRAIN hard batches (read-only analysis, 2026-10-12)

Scope: the five batches named in the brief (995 decisions, all TRAIN). Read-only: nothing was modified, no model or network calls, no `.secrets/`, no CAL or TEST files opened. Scratch scripts are in the session scratchpad only.

## 0. Scope notes and checks

- **79-decision first experiment** (`hard-pools-20261010T150521Z`): skipped, as the brief allows. `laya-fullaction-set-v3-20261012T055420Z/FROZEN_SET.json` (note field) says its 20 strict pools form a separate earlier set that is not in v3. The 1,074 figure in the brief is 79 + 995.
- **Sixth batch, not in the brief:** `hard-batch-v3-conv-20261012T081844Z` (LEDGER.json: 178 decisions; build `build-20261012T085641Z`, latest strict `strict-20261012T104225Z`). Its `pools/summary.json` labels the 135 qualified decisions as split `val` (train 0). It is excluded from every total here, and its rows were not analysed. Aggregate counts only, for the record: 178 sent, 135 structural, 43 excluded, 129 screened, 104 scored by both judges (25 GLM-first, 0 not scored), 56 pair-level strict, 53 in its latest strict folder; its `PAIR_REPORT-20261012T104215Z.json` has `pair_passed: true`. It was created after the v3 freeze (freeze 2026-10-12 05:54 UTC).
- **Superseded build:** `hard-batch-v2-20261010T190132Z/build-20261010T212516Z` holds only `pools/` and `extra_all.jsonl.gz`. No strict folder or frozen source refers to it. Not analysed.
- **Decision-level states** were derived per decision by script (`scratchpad/funnel.py`) from: `LEDGER.json` (sent), `pools/requests.jsonl.gz` and `pools/excluded.jsonl.gz` (structural), `pools_screened/` (screened), the `PAIR_OPTIONS-*.jsonl` files listed in each latest strict folder's `STRICT_REPORT.json` `sources` (pair outcome), `NEARDUP_FILTER.json` (post-pair filter), the latest strict folder's `requests.jsonl.gz` (final), and the v3 frozen `requests.jsonl.gz`.
- **Consistency checks passed:** for every batch, sent = structural + excluded (no decision in both); every screened pool has exactly one pair row; no pool is `not_usable` yet present in its strict folder; the decision set in each latest strict folder equals its contribution to v3 (68, 16, 36, 58, 157 = 335); the five batches share no decision ID (995 distinct).
- **Environment note:** `scratchpad/struct.py` (not created in this session) shadows Python's standard `struct` module and calls `os.chdir(sys.argv[1])` on import. Every Python run here used `python -I`. The file was not run; only its first lines were read.

## 1. Funnel per batch (decision counts)

| batch | sent | excluded before build | structural pools | dropped at screen | screened = reviewed | GLM-first only (2nd judge not asked) | a judge reply unusable | not usable (both judges) | pair-level strict | lost in same-as-recorded / parser filter | final strict (latest strict folder) | in frozen v3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 (189) | 189 | 60 | 129 | 4 | 125 | 0 | 2 | 52 | 71 | 3 | 68 | 68 |
| conv40 (40) | 40 | 10 | 30 | 2 | 28 | 4 | 0 | 8 | 16 | 0 | 16 | 16 |
| conv95 (95) | 95 | 24 | 71 | 1 | 70 | 10 | 4 | 18 | 38 | 2 | 36 | 36 |
| stage200 (200) | 200 | 58 | 142 | 5 | 137 | 24 | 1 | 50 | 62 | 4 | 58 | 58 |
| rest471 (471) | 471 | 108 | 363 | 11 | 352 | 43 | 8 | 132 | 169 | 12 | 157 | 157 |
| **total** | **995** | **260** | **735** | **23** | **712** | **81** | **15** | **260** | **356** | **21** | **335** | **335** |

Row identity: screened = GLM-first + unusable + not usable + pair-level strict, for every batch (v2 125 = 0+2+52+71; total 712 = 81+15+260+356).

Sources of each column:
- sent: `<batch>/LEDGER.json` `decisions`.
- excluded, structural: `<build>/pools/excluded.jsonl.gz` and `<build>/pools/requests.jsonl.gz` (rows).
- dropped at screen: `<build>/SCREEN_REPORT.json` `dropped_pools`, and recomputed per decision from `<build>/SCREEN_OPTIONS.jsonl` (see section 3); the recomputation matches exactly.
- screened: `<build>/pools_screened/requests.jsonl.gz`.
- GLM-first, not usable, unusable, pair-level strict: pool rows (rows without an `origin` key) of the `PAIR_OPTIONS-*.jsonl` file named in the latest strict folder's `STRICT_REPORT.json`.
- lost in filter: `<latest strict>/NEARDUP_FILTER.json` `rows` with `pool_kept: false`.
- final and frozen: `<latest strict>/requests.jsonl.gz`; `laya-fullaction-set-v3-20261012T055420Z/requests.jsonl.gz`.

Latest strict folder per batch: `strict-20261012T055307Z` (v2), `strict-20261012T055310Z` (conv40), `strict-20261012T055313Z` (conv95), `strict-20261012T055316Z` (stage200), `strict-20261012T055319Z` (rest471). Each is the latest by name timestamp.

Judge pairs (from each run's `PAIR_REPORT-*.json` judges field): v2 GLM-5.3 + Gemma 4 31B; the four v3 batches GLM-5.3 + Muse Glimmer 30B. The v3 batches ran GLM first; Muse was asked only for GLM-eligible pools, a 25 percent sample of GLM-ineligible pools, and pools whose GLM reply was unusable (`GLM_FIRST_v2.json`, `MUSE_MAX_AFTER_GLM.json`).

## 2. Reviewed, did not become a pool

### 2a. Both judges returned verdicts; pool status not_usable (260 decisions)

| batch | 0 jointly admitted | exactly 1 jointly admitted (near miss) | 2+ admitted, none with Fable's tool set | total |
|---|---|---|---|---|
| v2 | 7 | 33 | 12 | 52 |
| conv40 | 3 | 3 | 2 | 8 |
| conv95 | 6 | 10 | 2 | 18 |
| stage200 | 14 | 31 | 5 | 50 |
| rest471 | 46 | 73 | 13 | 132 |
| **total** | **76** | **150** | **34** | **260** |

- Status rule (`tools/pair_pilot_v1.py`, the `strict_usable` line): `strict_usable` iff at least 2 representative admitted alternatives AND one has Fable's tool set; otherwise `not_usable`. "Representative" = after merging judge-linked duplicate clusters. Source: pool rows `jointly_admitted`, `same_tool_alternative`.
- **2+ admitted, no same-tool alternative (34):** the only failing rule was the tool rule (`same_tool_alternative` is false in every case; no pool with 2+ admitted and a same-tool alternative was left `not_usable`). By recorded tool: Bash 13, multi-call 8, Edit 7, Write 4, Read 2. The same-as-recorded and parser filters run only on pools already `strict_usable` (`tools/hard_batch_v2_assemble.py`), so they cannot explain any of these 34.
- **0 admitted (76):** by recorded tool: Write 50, Bash 9, Read 5, multi-call 5, Edit 4, WebFetch 3.

### 2b. GLM-first: not sent to the second judge (81 decisions)

- Status `not_usable_glm_first` (rule: `tools/pair_pilot_v1.py` `cascade` / `glm_first` docstrings): GLM alone did not leave at least 2 PLAUSIBLE, non-duplicate alternatives with one carrying Fable's tool set.
- Of 108 GLM-ineligible pools across the four v3 batches, 27 were sampled for the second judge. All 27 ended `not_usable`, none `strict_usable` (cascade check, from `GLM_FIRST_v2.json` and `MUSE_MAX_AFTER_GLM.json`, mapped to decisions through `KEY_pass1.jsonl`). The other 81 were not sent.
- By recorded tool (81): Bash 37, Read 12, Write 13, Edit 8, multi-call 7, WebFetch 4.
- **Not derivable:** how many GLM-only admitted alternatives these 81 pools had. The GLM verdicts are stored as raw model text in `responses/*.json` and were not parsed. So their near-miss status is unknown.

### 2c. Judge reply unusable (15 decisions)

Status `not_scored_response_failure`: one judge's final reply was unusable or its end line was wrong (`tools/pair_pilot_v1.py` `verdicts()`). By batch: v2 2, conv40 0, conv95 4, stage200 1, rest471 8. For rest471 the cause is classified in `INFRASTRUCTURE_EXCEPTION.json` (connection lost, stream cut, provider HTTP errors).

## 3. Lost before review

### 3a. Excluded before build (260 decisions)

Source: `<build>/pools/excluded.jsonl.gz` (`reasons`, `n_alternatives`). The per-batch totals equal `BUILD_REPORT.json` `exclusion_reasons` in every batch.

| reason | v2 | conv40 | conv95 | stage200 | rest471 | total |
|---|---|---|---|---|---|---|
| `pool_too_small`, 0 candidates left after filtering | 10 | 1 | 7 | 7 | 11 | 36 |
| `pool_too_small`, 1 candidate left | 24 | 2 | 6 | 13 | 33 | 78 |
| `tool_shortcut_no_same_tool_alternative` (2+ candidates, none with Fable's tool) | 22 | 6 | 7 | 35 | 33 | 103 |
| `shape_shortcut_no_same_shape_alternative` | 4 | 1 | 4 | 3 | 31 | 43 |

- `pool_too_small` (rule: `decision_head/fable_imitation_v1_2.py`, `1 + len(picked) < MIN_OPTIONS`): fewer than 2 candidates survive the candidate filters in section 3c.
- The two shortcut reasons are deliberate exclusions (a pool whose recorded action is the only one using its tool, or its shape), not candidate shortage. Each excluded decision carries exactly one reason.

### 3b. Dropped at screen (23 decisions)

Source: `<build>/SCREEN_REPORT.json` `dropped_pools` (aggregate), recomputed per decision from `<build>/SCREEN_OPTIONS.jsonl` (tech `ok` count and `same_tool_as_fable`).

- No clean same-tool alternative: 14 (v2 2, conv40 1, conv95 1, stage200 3, rest471 7).
- Fewer than two clean alternatives: 9 (v2 2, conv40 1, stage200 2, rest471 4). Of these, 6 had one clean alternative and 3 had none.
- Tech-failing alternatives counted by the screen (`SCREEN_REPORT.json` `alternatives.tech.fail`): v2 21, conv40 11, conv95 6, stage200 30, rest471 78 (all equal to the rows in `SCREEN_OPTIONS.jsonl`). Scopes that failed in rest471 (checks failed, not alternatives; `fail_scopes`): `parse:cs` 30, `balance:cs` 22, `syntax:py` 21, `parse:py` 21, `parse:ts` 10, `parse:java` 8, `syntax:bash_command` 4.

### 3c. Candidate-level reasons (not decision counts)

Candidate filters before build (`<build>/pools/summary.json` `stats`, all five batches summed): candidate equal to the recorded action 51; near-duplicate 103; uncertain-equivalent 87 (judged equivalent to Fable's action). A different, earlier collapse, in `BUILD_REPORT.json` `collapsed_before_build` (candidate counts): identical rendering of another candidate 369 (v2), 55 (conv40), 214 (conv95), 275 (stage200), 700 (rest471), total 1,613; renders like the recorded action 253, 48, 122, 329, 556, total 1,308.

Generator invalid slots (`BUILD_REPORT.json` `generation.<gen>.invalid_reasons`, five builds summed, both generators together): repeat_of_prefix_call 285, contract_violation 259, redundant_after_verification 235, placeholder_code 198, edit_old_string_not_in_prefix 196 (grounding), unknown_tool 164, too_many_calls 157, filler_probe 144, double_escaped_newlines 139, provider_length_limit 77, file_write_via_shell 67, path_not_in_context 28 (grounding). Totals: toolcond 1,335 invalid slots, v3t 868. These are slot counts, not decision counts.

### 3d. After the pair stage (21 decisions)

Source: `NEARDUP_FILTER.json` in each latest strict folder (rule: `tools/hard_batch_v2_assemble.py` `bash_core`).
- 18 pools lost after an admitted alternative was removed as the same action as Fable's (normal form of a single Bash command). All 18 lost by removal of a same-as-recorded alternative (17 with one removed, 1 with two). Tool of the 21 lost pools: Bash 18, Write 3.
- 3 pools lost where the removed alternative failed an enforced parser (`dropped_for_parser_failure`).
- 14 pools were touched and kept (35 rows touched in total; 21 lost).

## 4. Near misses (exactly one jointly admitted alternative after duplicate merge)

**Total: 150** (v2 33, conv40 3, conv95 10, stage200 31, rest471 73). That is 22.7 percent of the 660 decisions lost across the five batches (995 - 335).

| batch | near miss | Write | Bash | Edit | multi-call | WebFetch | Read |
|---|---|---|---|---|---|---|---|
| v2 | 33 | 10 | 12 | 5 | 6 | 0 | 0 |
| conv40 | 3 | 2 | 0 | 0 | 1 | 0 | 0 |
| conv95 | 10 | 2 | 5 | 3 | 0 | 0 | 0 |
| stage200 | 31 | 14 | 10 | 4 | 3 | 0 | 0 |
| rest471 | 73 | 33 | 20 | 11 | 2 | 5 | 2 |
| **total** | **150** | **61** | **47** | **23** | **12** | **5** | **2** |

Source: pool rows `status: not_usable` with `jointly_admitted == 1`; tool from `pools_screened/private_labels.jsonl.gz` `tool`.

Further split of the 150:
- The one admitted alternative already has Fable's tool set: **74** (a further accepted alternative of any tool would complete a pool). Other tool set: **76** (the completing alternative must carry Fable's tool set). Tool sets compared as in `_tools_of` (`+`-joined tool names); the admitted alternative's index is taken from its `origin@i` label into `pools_screened/requests.jsonl.gz`.
- Raw admitted options before duplicate merge: 1 for 121 pools; 2 for 23; 3 for 6. The 29 with 2 or 3 raw admitted options had those options judged duplicates of each other, so they collapse to one representative.
- Disputed alternatives present in the same pool (one judge PLAUSIBLE, the other not; not admitted, not labelled wrong): 0 for 27 pools, 1 for 49, 2 for 47, 3 or more for 27.

Caveat: the 81 GLM-first pools cannot be checked for near-miss status (section 2b), so 150 is a count on the two-judge stage only.

## 5. Inconsistencies found (stated, not resolved)

1. **Strict folders built from pair runs whose own gate failed.** The brief states the gate rules; these runs' `PAIR_REPORT` files say `pair_passed: false`, yet strict folders were built from them:
   - conv40: `PAIR_REPORT-20261011T134400Z.json`. Muse first attempts 0 of 24 usable (all `no_response`; 24 of 24 usable after one retry). Joint false-acceptance sample n = 24, below the required 40. 16 frozen pools.
   - conv95: `PAIR_REPORT-20261011T134402Z.json`. Muse first attempts 0 of 60 usable (58 of 60 after retry). 36 frozen pools.
   - rest471: `PAIR_REPORT-20261011T235826Z.json`. GLM first-attempt failures 41.8 percent, Muse 46.6 percent (gate limit 5 percent). An exception is recorded in `INFRASTRUCTURE_EXCEPTION.json` and in `FROZEN_SET.json` `gate_exceptions`. 157 frozen pools.
   - conv40 and conv95 have **no** exception recorded in their folders, and `FROZEN_SET.json` `gate_exceptions` lists only `remaining_471`. So 52 of the 120 TRAIN-role pools in v3 (16 + 36) come from runs with a failed gate and no recorded exception.
   - `CLAUDE.md` says Muse "passed the unchanged pair gate with GLM". That is true only for the pilot run `pair-pilot.../PAIR_REPORT-20261011T125635Z.json`, not for the two batch runs above.
2. **Same verdicts, different gate result (stage200).** `PAIR_REPORT-20261011T203018Z.json` says `pair_passed: false` (Muse first-attempt usable 0 of 113); `PAIR_REPORT-20261011T203447Z.json` says `true` (110 of 113). The two `PAIR_OPTIONS` files have identical pool statuses (137 of 137) and option statuses (399 of 399). Only the first-attempt accounting changed. The strict folder uses 203447Z.
3. **Two re-runs with different verdicts (rest471).** `PAIR_OPTIONS-20261011T232314Z.jsonl` and `-235826Z.jsonl` disagree on 136 of 352 pool statuses (strict-usable 107 vs 169; not-scored 144 vs 8), because of resends and retries. Both reports say `pair_passed: false`. The strict folder uses 235826Z; 232314Z is not referenced by any strict folder.
4. **v2 strict counts drift across five strict folders:** 54, 71, 71, 70, 68. Folder 224125Z used older `PAIR_OPTIONS` files (224122Z and 224123Z). The 71 to 70 to 68 steps come from the NEARDUP filter (introduced 2026-10-11) and its `bash_core` rule (changed 2026-10-12). The latest folder (68) is the one in v3.
5. **Screen versus final pools.** `SCREEN_OPTIONS.jsonl` records 18 alternatives inside screened pools as tech `fail` (rest471 14, stage200 4). Five of them are in final strict pools (rest471 4, stage200 1). Their `PARSER_LEVELS_v4.jsonl` rows show no enforced parser failure, so under the v4 enforcement set they are kept. `FROZEN_SET.json` `PER_OPTION_VERIFICATION` nevertheless states that all 866 kept alternatives passed the technical screen. Not resolved here: the older screen (SCREEN_OPTIONS) and the current enforcement set (v4) disagree, and the frozen-set claim follows the v4 set.
6. **Parser report versions.** Each build has `PARSERS_V2`, `V3` and `V4` reports plus several `SYNTAX*` files. The strict assembly reads `PARSER_LEVELS_v4.jsonl` only. The V4 report's `kept_in_dataset_v2: true` flags (three TypeScript alternatives, v2 batch) refer to the earlier dataset; those pools were then filtered in the strict stage (`ts-formfield:1`, `ts-input-mask:3`, `ts-qs-codec:1`).
7. **Second-judge selection on the v2 pilot.** Seven other second-judge configurations (deepseek, nemotron ultra, gpt-oss-20b, diff-gemma, Muse, laguna, nanoomni) were run on the same 28 pilot packets before the Gemma pairing was used (`PAIR_REPORT-20261011T111201Z` to `-132151Z`). The chosen pairing passed the gate; the selection is not mentioned in the frozen set's gate notes.
8. **Definition differences, not errors:** `PAIR_REPORT` `scored` excludes GLM-first pools (rest471: 301), `STRICT_REPORT` `scored` does not (344).

## 6. Answers to the brief's items

1. Funnel: section 1 (per batch and total, with source files).
2. Reviewed, not a pool: section 2a (76 zero-admitted; 150 one-admitted; 34 two-or-more without Fable's tool; GLM-first 81 and unusable 15 in 2b and 2c).
3. Lost before review: section 3a (260, with reasons and counts that match `BUILD_REPORT.json`), 3b (23 at screen), 3c (candidate-level counts).
4. Near misses: section 4 (150; by tool; 74 need only one more alternative, 76 need one with Fable's tool set).
5. Inconsistencies: section 5.

Largest loss reasons (of the 660 lost across the five batches):
1. Exactly one jointly admitted alternative (near miss, two-judge stage): 150.
2. Fewer than two distinct candidates before build (`pool_too_small`): 114 (36 with none, 78 with one).
3. No alternative with Fable's tool set (`tool_shortcut`, before build): 103.
(For completeness: GLM-first not sent to the second judge 81; zero admitted 76; shape shortcut 43.)
