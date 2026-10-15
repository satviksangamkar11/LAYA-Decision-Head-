# S3. Check and scorer registry (Evidence-Ladder v1, P0 + R3, 2026-10-16)

Schemas: `schemas_v1.json#/$defs/check_def`, `#/$defs/scorer_def`. The registry is a versioned file, hash-checked at the start and end of every run (I7).
A check or scorer missing from the registry cannot produce evidence.

## 1. Stage-1 checks

| check_id | kind | trust_class | covers | runtime_visible | grader_only | limitations |
|---|---|---|---|---|---|---|
| `parse_v1` | parse | rejecting_only | syntax | yes | no | says nothing about behaviour |
| `compile_v1` (C++ if used) | compile | rejecting_only | compiles under the pinned compiler | yes | no | same |
| `public_examples_v1` | public_test | authoritative_in_scope | the statement's own examples only | yes | no | examples are few, so passing them does not verify the problem |
| `limits_v1` | public_test | rejecting_only | time/memory limits on public examples | yes | no | hardware differs from the judge's |
| `qwen_tests_v1` | model_test | diagnostic | none (ranking only) | yes | no | may share the generator's misunderstanding |
| `critic_experiment_v1` | critic_experiment | diagnostic | none; its execution result can reject via S2 section 2 | yes | no | the critic's expected output is not truth |
| `behaviour_cluster_v1` | behaviour_cluster | diagnostic | none (ranking) | yes | no | agreement is not correctness |
| `hidden_tests_v1` | hidden_test | authoritative | **evaluation only** | **no** | **yes** | adequacy must be measured (S6 section 6) |

Stage 1 has no `end_state` or `certified_invariant` checks. Certified checks arrive in stage 3 (decision D4).

## 2. Scorers

### 2.1 Adapter + LAYA head: compatibility contract

| field | value | source |
|---|---|---|
| qwen_repo / qwen_revision | Qwen3-4B-Thinking-2507 / `768f209d9ea81521153ed38c47d515654e938aea` | `decision_head/fable_features.py` PROTOCOL |
| dtype / device_class | fp16 / Kaggle T4 (one device class for every split) | PROTOCOL `feature_dtype`; CLAUDE.md |
| depths / hidden | [18, 24, 30] / 2560 | PROTOCOL |
| query_feature | hidden state at the last prefix token, after the query cue `"\n\nNext action:"` | PROTOCOL |
| option_feature | mean over the option's own tokens, each option an independent suffix of the shared prefix, option prefix `" "` | PROTOCOL |
| renderer / context policy | `fable_render_v1` / `fable_ctx_v1` (agent-trajectory rendering) | PROTOCOL |
| capture code | `tools/fable_capture.py` sha256 `9d8dd379…72`; `decision_head/fable_features.py` sha256 `fe2ae8ea…e9` | computed 2026-10-16 |
| limits | max 8 options, prefix 24,000 tokens, option 4,096 tokens | `tools/fable_capture.py` LIMITS |
| weights | adapters `final-s{0,1,2}.pt`, hashes in `results/raw/laya-baseline-v1-20261012T215242Z/MANIFEST.json`; head = pretrained Coding_Decision_Agent, 31 tensors unchanged | MANIFEST |
| quantization | none (fp16); a quantised local runtime needs a new parity check and adapter (A3 RC-71) | - |
| decision types validated | **`next_action`** only (Fable next-action pools) | training data |
| reproduction gate before any use | cleaned VAL 46: 0.7826 / 0.8043 / 0.7609 | A3 RC-12 |

Rules:
- **The controller refuses** to call LAYA for a decision type that is not in `decision_types_validated`, unless the run is flagged `exploratory`.
- In stage 1 the decision type is `complete_program`, so LAYA's status there is `transfer_test`. Stage-1 arm E is exploratory and cannot decide LAYA's fate (S6).
- Stage 1 also renders candidates with a different renderer than `fable_render_v1` (programs, not trajectory actions). That rendering is versioned as `program_render_v1` and recorded. It is part of why the stage-1 result is a transfer test.
- Stage 2 uses `fable_render_v1` / `fable_ctx_v1` unchanged for next-action decisions. That is the validated use.

### 2.2 Other scorers (always run beside LAYA)

| scorer_id | status | notes |
|---|---|---|
| `probe_v1` (logistic on the same features, C by grouped CV, recipe `kaggle/head_exp4_gpu.py:probe_scores`) | validated for the same decision types it is fitted on | mandatory baseline (CLAUDE.md) |
| `random_head_v1` (adapter + random-initialised head) | control | tests whether the pretrained head matters |
| `deepconf_v1` (token log-prob confidence of the generating trace) | validated as a signal, any decision type | no training |

## 3. Tests (pilot stage 0)

| id | case | expected |
|---|---|---|
| S3-T1 | check not in the registry produces a record | record rejected |
| S3-T2 | LAYA called with `complete_program` in a non-exploratory run | refused (unsupported decision type) |
| S3-T3 | registry or adapter file hash changes during a run | run aborted (I7) |
| S3-T4 | `hidden_tests_v1` referenced from runtime code | import or lookup fails (I8) |
| S3-T5 | capture code hash differs from this table | LAYA disabled for the run |
