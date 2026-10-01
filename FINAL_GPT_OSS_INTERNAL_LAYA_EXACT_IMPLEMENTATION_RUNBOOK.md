**FINAL EXACT IMPLEMENTATION RUNBOOK**

**GPT-OSS 120B + INTERNAL LAYA/JEEVES-STYLE DECISION HEAD**

From zero repository to reproducible local coding-agent runtime — every stage, artifact, command, interface, training gate, and acceptance test.

| **Item**                         | **Final value**                                                                                                                                            |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Primary reasoning model          | GPT-OSS 120B (frozen initially) — 117B total / 5.1B active per token / 128K context                                                                        |
| Final decision mechanism         | Internal pointer-style decision head over GPT-OSS hidden representations                                                                                   |
| Teacher layer                    | Existing Laya-derived checkpoints + coding specialists + real execution outcomes                                                                           |
| Learning path                    | Automatic decision dataset compiler → teacher benchmark → Coding-Laya (only when required) → internal head → calibration → optional selective PEFT         |
| Host                             | DeepSeek Harness via Cordis plugins/events; do not fork the core loop                                                                                      |
| Truth authority                  | Hard invariants + actual compiler/tests/runtime outcomes \> model confidence                                                                               |
| Current local machine constraint | 16 GB RAM / 6 GB GPU; use it for development, compilation, data work and small teachers — not as the target environment for native 120B inference/training |
| Permanent external dependency    | None on Laya. Laya is teacher/validator/reference; final decision head is inside GPT-OSS stack                                                             |

| **THIS REPLACES THE PREVIOUS DOCUMENT** The previous file was a high-level specification. This version is an implementation runbook: each phase has concrete files, commands, inputs/outputs, tests, and explicit “do not proceed” gates. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 0. NON-NEGOTIABLE TARGET

| **Final runtime** GPT-OSS generates reasoning and candidate actions. The internal head scores the candidates from GPT-OSS representations. Deterministic policy constrains what may execute. Specialists are invoked conditionally. The real environment produces the final evidence. No external Laya request is required in production. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

USER TASK

↓

STATE BUILDER

↓

GPT-OSS 120B

├─ reasoning

└─ candidate generation

↓

HIDDEN-STATE CAPTURE

↓

INTERNAL DECISION HEAD

├─ pointer scores

├─ abstain

└─ escalation

↓

CALIBRATION

↓

DETERMINISTIC POLICY

├─ hard safety

├─ permissions

└─ completion invariants

↓

SPECIALISTS (conditional)

↓

TOOL EXECUTION

↓

REAL ENVIRONMENT

├─ compiler

├─ tests

├─ runtime

└─ git state

↓

OUTCOME JOURNAL

↓

NEXT GPT-OSS STEP

# 1. WHAT IS FACT, WHAT IS A DESIGN DECISION, WHAT IS NOT ALLOWED

| **Category**           | **Rule**                                                                                                                                                                                                              |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Verified external fact | GPT-OSS-120B is a 117B-total-parameter MoE with 5.1B active parameters/token, 36 layers, 128 experts total, 4 active experts/token, and 128K context. OpenAI states the native MXFP4 release runs within 80GB memory. |
| Verified external fact | Laya is a non-autoregressive typed decision system with choice/score/noul primitives; current published checkpoints include 421M ModernBERT-large variants.                                                           |
| Verified external fact | DeepSeek Harness currently exposes Cordis extension points including agent/pre-step, agent/request, tools/pre-execute, tools/execute, tools/post-execute, tools/result, agent/turn-stopping and session/event.        |
| Design decision        | Freeze GPT-OSS first and train only the internal decision head. Test selective PEFT only after a head-only gain is proven.                                                                                            |
| Design decision        | Use Laya-derived models as teachers/specialists, not as the permanent runtime decision API.                                                                                                                           |
| Absolute invariant     | Never let a learned score override hard safety, permissions, or objective execution invariants.                                                                                                                       |
| Absolute invariant     | Never fabricate counterfactual outcomes. Non-executable branches may have teacher probabilities but must remain marked non-executed.                                                                                  |

# 2. FINAL REPOSITORY: CREATE THIS BEFORE TRAINING

internal-decision-agent/

├── configs/

│ ├── system.yaml

│ ├── models.yaml

│ ├── decisions.yaml

│ ├── policy.yaml

│ └── splits.yaml

├── gpt_oss/

│ ├── runtime/

│ ├── hidden_state_capture/

│ ├── tokenizer/

│ ├── prompt/

│ └── model_adapter/

├── decision_head/

│ ├── pointer.py

│ ├── candidate_encoder.py

│ ├── choice.py

│ ├── noul.py

│ ├── score.py

│ ├── abstain.py

│ ├── escalation.py

│ ├── markers.py

│ └── calibration.py

├── state/

│ ├── schema.py

│ ├── builder.py

│ ├── selector.py

│ ├── compressor.py

│ └── hashing.py

├── retrieval/

│ ├── tree_sitter.py

│ ├── bm25.py

│ └── laya_code.py

├── memory/

│ ├── schema.py

│ ├── evidence.py

│ ├── supersession.py

│ ├── graph.py

│ └── retrieval.py

├── specialists/

│ ├── conductor.py

│ ├── completion.py

│ ├── code_oracle.py

│ ├── evidence_gate.py

│ ├── gyra.py

│ ├── vision.py

│ └── browser.py

├── policy/

│ ├── hard_rules.py

│ ├── permissions.py

│ ├── confidence.py

│ ├── debias.py

│ ├── coordinator.py

│ └── completion_invariants.py

├── compiler/

│ ├── ingestion.py

│ ├── normalization.py

│ ├── episodes.py

│ ├── decisions.py

│ ├── candidates.py

│ ├── teachers.py

│ ├── outcomes.py

│ ├── negatives.py

│ ├── counterfactuals.py

│ ├── dedup.py

│ └── manifest.py

├── data/

│ ├── raw/

│ ├── normalized/

│ ├── decisions/

│ ├── candidates/

│ ├── outcomes/

│ ├── teachers/

│ ├── negatives/

│ ├── counterfactuals/

│ ├── provenance/

│ └── splits/

├── training/

│ ├── coding_laya/

│ ├── internal_head/

│ ├── calibration/

│ └── adapters/

├── evaluation/

│ ├── locked_decisions/

│ ├── coding/

│ ├── safety/

│ ├── recovery/

│ ├── long_context/

│ ├── calibration/

│ └── downstream_agent/

├── journal/

│ ├── decisions.jsonl

│ ├── outcomes.jsonl

│ ├── episodes.jsonl

│ └── contradictions.jsonl

├── integrations/deepseek_harness/

│ ├── plugin.ts

│ ├── pre_step.ts

│ ├── request.ts

│ ├── tool_guard.ts

│ ├── result.ts

│ ├── turn_stopping.ts

│ └── session_events.ts

├── tests/

└── scripts/

# 3. PHASE 0 — PIN THE ENVIRONMENT

## 3.1 Install base environment

\# PowerShell

mkdir internal-decision-agent

cd internal-decision-agent

git init

py -3.11 -m venv .venv\\

. .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel setuptools

pip install torch transformers safetensors huggingface_hub datasets accelerate

pip install pydantic pyyaml numpy scipy scikit-learn orjson tqdm rich

pip install tree-sitter tree-sitter-language-pack rank-bm25

pip install pytest pytest-cov jsonschema

Use Python 3.10+ for current Laya; pin the exact version actually used in the experiment manifest rather than tracking “latest”.

## 3.2 Capture immutable environment manifest

python --version \> artifacts/python_version.txt

pip freeze \> artifacts/pip_freeze.txt

git rev-parse HEAD \> artifacts/git_base.txt

python scripts/capture_env.py --out artifacts/env_manifest.json

| **GATE 0** You cannot start dataset generation until env_manifest.json, the model manifest, and the experiment ID are recorded. |
|---------------------------------------------------------------------------------------------------------------------------------|

# 4. PHASE 1 — REGISTER MODELS AND CHECKPOINTS

## 4.1 Register GPT-OSS

\# Record the exact Hugging Face revision you use. Do not use floating “main”.

python scripts/register_model.py \\

--name gpt-oss-120b \\

--source-hf \<EXACT_HF_REPO\> \\

--revision \<EXACT_REVISION\> \\

--tokenizer \<EXACT_TOKENIZER_REVISION\>

The model adapter must support: tokenization, chat/harmony prompt construction, generation, hidden-state capture, and deterministic seed/config recording. Do not invent a special decision token before inspecting the tokenizer.

## 4.2 Register Laya teachers

python scripts/register_teacher.py --name laya-typed --hf convaiinnovations/laya-typed-decisions --revision \<PIN\>

python scripts/register_teacher.py --name laya-conductor --hf mvilacad/laya-conductor --revision \<PIN\>

python scripts/register_teacher.py --name laya-code --hf tindang/laya-code --revision \<PIN\>

python scripts/register_teacher.py --name completion --hf tampajohn/laya-stop-completion-judge --revision \<PIN\>

python scripts/register_teacher.py --name cortex-1 --hf mukti-sys/cortex-1-large --revision \<PIN\>

| **Do not merge weights** These specialists remain independent inference teachers/validators. They are not merged into GPT-OSS. Their predictions are stored with provenance and uncertainty. |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 5. PHASE 2 — DEFINE THE DECISION LANGUAGE

decision_type:

ACTION: choose one executable action

EVIDENCE: choose whether evidence is sufficient / which evidence to trust

CONTINUE: continue vs complete/hand back

RECOVERY: choose the next recovery action after failure

PATCH: choose among patch candidates

COMPLETION: determine whether acceptance criteria are satisfied

ESCALATE: stay autonomous vs invoke specialist/human

ABSTAIN: defer because evidence is insufficient

## 5.1 Candidate contract

{

"candidate_id": "cand-...",

"text": "Run the targeted test for auth middleware",

"action_type": "tool_call",

"tool": "shell",

"target": {"repo":"r1", "file":"..."},

"arguments": {"cmd":"pytest tests/test_auth.py -q"},

"expected_effect": "obtain evidence about the failing auth path",

"preconditions": \["working_tree_clean_or_expected_changes_present"\],

"source": "MODEL\|TRAJECTORY\|TEACHER\|RETRIEVAL\|PATCH\|COUNTERFACTUAL",

"executable": true,

"safety_class": "READ_ONLY",

"teacher_scores": {},

"outcome_if_executed": null

}

## 5.2 State contract

{

"task": "...",

"goal": "...",

"hypothesis": "...",

"repo": "...",

"commit": "...",

"recent_actions": \[\],

"tool_results": \[\],

"tests": \[\],

"compiler": {},

"runtime": {},

"git_diff": {},

"memory_evidence": \[\],

"screenshots": \[\],

"pending_actions": \[\],

"candidates": \[\],

"superseded_facts": \[\],

"budget": {"context_tokens": 0, "tool_steps": 0},

"state_hash": "sha256:..."

}

# 6. PHASE 3 — BUILD THE LOCKED BENCHMARK BEFORE TRAINING

## 6.1 Benchmark composition

| **Bucket**     | **Purpose**                                                  | **Rule**                                 |
|----------------|--------------------------------------------------------------|------------------------------------------|
| Decision core  | ACTION / EVIDENCE / CONTINUE / RECOVERY / PATCH / COMPLETION | No training access                       |
| Safety         | destructive commands, secrets, permission boundaries         | Hard rules must be evaluated separately  |
| Recovery       | failed tests, contradictory evidence, repeated failures      | Measure next-step choice                 |
| Long-state     | large repo state with relevant tail evidence                 | Measure state selector + head robustness |
| OOD            | repositories/issues not seen in training                     | Repo/issue-disjoint                      |
| Permutation    | same state, candidate order shuffled                         | No order leakage                         |
| Counterfactual | same state with plausible wrong alternatives                 | Measure rejection                        |

## 6.2 Metrics

choice: accuracy, macro-F1, log loss, Brier, top-k recall, ECE

noul: AUROC, AUPRC, Brier, ECE, false-approval rate

score: MAE, rank correlation, ordinal calibration

agentic: success rate, tool-step efficiency, recovery rate, premature-stop rate

safety: false approval rate = 0 is the target hard-gate property for rules that are deterministic

robustness: permutation delta, OOD delta, long-context delta

## 6.3 Lock it

python scripts/build_locked_benchmark.py --out data/splits/locked_v1

python scripts/hash_manifest.py data/splits/locked_v1/manifest.json \> data/splits/locked_v1/manifest.sha256

\# make benchmark read-only

attrib +R data\splits\locked_v1\\ /S

| **GATE 1** The benchmark manifest hash becomes immutable. Any later benchmark modification creates a new benchmark version and invalidates comparisons. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|

# 7. PHASE 4 — INGEST ALL PUBLIC/PROJECT DATA WITHOUT LEAKAGE

## 7.1 Sources

Primary candidate sources: SWE-rebench/OpenHands trajectories, SWE-Hero, SWE-smith, SWE-Gym, R2E-Gym, project-private trajectories, Git commits/diffs, tests, tool logs, and prior decision corpora. Treat overlapping datasets as potentially duplicated because some releases derive from other public task pools.

## 7.2 Ingestion command

python -m compiler.ingestion \\

--source swe-rebench-openhands=hf:nebius/SWE-rebench-openhands-trajectories \\

--source swe-hero=hf:nvidia/SWE-Hero-openhands-trajectories \\

--source swe-smith=hf:SWE-bench/SWE-smith-trajectories \\

--source swe-gym=git:SWE-Gym/SWE-Gym \\

--source r2e-gym=hf:R2E-Gym/R2EGym-SFT-Trajectories \\

--project-data \<LOCAL_TRAJECTORY_DIR\> \\

--out data/raw/

## 7.3 Provenance fields

source_name

source_revision

dataset_config

row_id

repo_url

repo_sha

issue_id

trajectory_id

commit_before

commit_after

license

local_import_hash

| **GATE 2** Every raw record has a source provenance ID and a license field. Records missing provenance are quarantined, not silently included. |
|------------------------------------------------------------------------------------------------------------------------------------------------|

# 8. PHASE 5 — NORMALIZE INTO EPISODES

python -m compiler.normalization \\

--input data/raw \\

--output data/normalized \\

--schema configs/episode_schema.json

## 8.1 Reconstruct state at every decision boundary

A decision boundary exists immediately before a tool call, before a proposed patch, when a specialist is selected, when the agent considers ending a turn, and after a material contradiction/failure. Capture the state that was actually available at that time — not later information.

## 8.2 Git truth attachment

git rev-parse HEAD

git diff --binary \<parent\> HEAD

git diff --name-status \<parent\> HEAD

git log --format=fuller -n 20

pytest / build / compiler commands and exit codes

# 9. PHASE 6 — EXTRACT DECISIONS AUTOMATICALLY

python -m compiler.decisions \\

--episodes data/normalized \\

--out data/decisions \\

--types ACTION,EVIDENCE,CONTINUE,RECOVERY,PATCH,COMPLETION,ESCALATE,ABSTAIN

## 9.1 Extraction rules

| **Decision** | **Trigger**                                       | **Positive evidence**               |
|--------------|---------------------------------------------------|-------------------------------------|
| ACTION       | tool call follows agent reasoning                 | actual selected tool/arguments      |
| EVIDENCE     | retrieval/test/log selection                      | item used to update state           |
| CONTINUE     | turn continues after tool result                  | next concrete action exists         |
| RECOVERY     | previous action failed or contradicted hypothesis | next corrective action              |
| PATCH        | edit/diff is proposed                             | actual committed/reverted diff      |
| COMPLETION   | agent attempts to stop                            | tests/artifacts/acceptance criteria |
| ESCALATE     | specialist/human/tool-level fallback selected     | actual escalation                   |

## 9.2 Never infer labels from text alone when execution truth exists

Example: “I think the auth fix is correct” is not a success label. The success label comes from tests, build, runtime, or subsequent maintainer/commit evidence. Text is context; environment outcome is stronger supervision.

# 10. PHASE 7 — GENERATE CANDIDATES

python -m compiler.candidates \\

--decisions data/decisions \\

--retrieval tree-sitter,bm25,laya-code \\

--teacher-suggestions enabled \\

--model-alternatives enabled \\

--max-candidates 12 \\

--out data/candidates

Target candidate count is small enough for pointer scoring and detailed auditing. Start around 5–15 candidates per decision; expand only when the benchmark demonstrates that larger candidate sets improve recall.

## Candidate sources

- Observed action from the trajectory.

- Other executable tools available in the state.

- Other files/snippets returned by retrieval.

- Known successful/failed patches from related episodes.

- Teacher-proposed alternatives.

- GPT-OSS-generated alternatives constrained to the action schema.

# 11. PHASE 8 — CREATE HARD NEGATIVES

python -m compiler.negatives \\

--candidates data/candidates \\

--episodes data/normalized \\

--out data/negatives

| **Negative class**       | **Construction**                                            |
|--------------------------|-------------------------------------------------------------|
| Wrong file               | same operation against a nearby but irrelevant file         |
| Stale action             | repeat an action after its evidence became invalid          |
| Duplicate failure        | repeat exact failed operation without changed preconditions |
| Premature patch          | edit before enough evidence                                 |
| Irrelevant investigation | inspect unrelated subsystem                                 |
| Premature completion     | stop while tests/artifacts/criteria remain unsatisfied      |
| Unsafe shortcut          | destructive action when a safe read-only path exists        |

# 12. PHASE 9 — EXECUTE COUNTERFACTUALS WHERE SAFE

python -m compiler.counterfactuals \\

--candidates data/candidates \\

--sandbox docker_or_vm \\

--max-branches-per-state 4 \\

--out data/counterfactuals

| **Critical accounting rule** A non-executed counterfactual receives no claimed outcome. Store executable=false or executed=false explicitly. Teacher preference is separate from objective outcome. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 13. PHASE 10 — RUN TEACHER ENSEMBLE

python -m compiler.teachers \\

--state data/decisions \\

--candidates data/candidates \\

--teacher laya-typed \\

--teacher cortex-1 \\

--teacher laya-conductor \\

--teacher completion \\

--teacher evidence-gate \\

--out data/teachers

Use the specialist only for the decisions it was trained to answer. Do not ask laya-code to solve tool choice or completion; its stated task is code relevance. Do not ask a completion judge to approve a destructive shell command.

## 13.1 Teacher output

{

"decision_id": "...",

"teacher": "laya-typed",

"teacher_revision": "...",

"question_type": "choice",

"distribution": {"A":0.12,"B":0.74,"C":0.14},

"confidence": 0.74,

"temperature": 1.21,

"latency_ms": 28.4

}

# 14. PHASE 11 — RECONCILE LABEL AUTHORITY

priority = \[

REAL_EXECUTION_OUTCOME,

MAINTAINER_OR_HUMAN_EVIDENCE,

MULTI_TEACHER_AGREEMENT_WITH_EXECUTION_SUPPORT,

SPECIALIST_TEACHER,

HEURISTIC

\]

Store every disagreement instead of hiding it. A disagreement record becomes useful hard-negative or abstention data.

python -m compiler.outcomes \\

--decisions data/decisions \\

--counterfactuals data/counterfactuals \\

--teachers data/teachers \\

--out data/outcomes

# 15. PHASE 12 — DEDUPLICATION AND LEAKAGE CONTROL

python -m compiler.dedup \\

--keys repo,issue,commit_before,commit_after,trajectory \\

--near-duplicate-state-similarity 0.98 \\

--out data/provenance/dedup_manifest.json

## Split policy

| **Split**   | **Rule**                                                  |
|-------------|-----------------------------------------------------------|
| train       | repo/issue-disjoint from locked test and OOD              |
| calibration | never used for gradient updates                           |
| validation  | used for early stopping/hyperparameter selection only     |
| test        | immutable benchmark, never inspect for training           |
| OOD         | repository families and task patterns not used in fitting |

| **GATE 3** No training job starts until the split manifest is hashed and the compiler can prove no test/OOD provenance crosses into train/calibration. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|

# 16. PHASE 13 — BENCHMARK EXISTING DECISION MODELS FIRST

python -m evaluation.locked_decisions \\

--model laya-typed \\

--model cortex-1 \\

--model laya-conductor \\

--benchmark data/splits/locked_v1 \\

--out runs/benchmarks/\<RUN_ID\>

Add a small Verdict-class candidate only when its licensing/checkpoint and task interface are verified. The selection criterion is independent coding-decision performance on your locked data — not model size, benchmark marketing, or a universal “best” label.

# 17. PHASE 14 — TRAIN CODING-LAYA ONLY WHEN THE BENCHMARK REQUIRES IT

| **Training decision** Existing specialists are reusable. Train a new Coding-Laya checkpoint only if the locked benchmark shows no existing checkpoint covers the required decision distribution well enough. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 17.1 Dataset for Coding-Laya

python -m training.coding_laya.build_dataset \\

--source data/decisions \\

--labels data/outcomes \\

--teachers data/teachers \\

--negatives data/negatives \\

--out data/coding_laya_dataset

## 17.2 Training

\# Example shape; use the upstream Laya RLCD implementation/config rather than inventing a new objective

python -m training.coding_laya.train \\

--base convaiinnovations/laya \\

--train data/coding_laya_dataset/train.jsonl \\

--val data/coding_laya_dataset/val.jsonl \\

--max-len 1024 \\

--bf16 \\

--output runs/coding_laya/\<RUN_ID\>

## 17.3 Calibration

python -m training.calibration.fit \\

--predictions runs/coding_laya/\<RUN_ID\>/val_predictions.jsonl \\

--method temperature \\

--out runs/coding_laya/\<RUN_ID\>/calibration.json

# 18. PHASE 15 — BUILD GPT-OSS HIDDEN-STATE CAPTURE

## 18.1 First resolve the exact decision marker

Jeeves demonstrates a rare-token marker + pointer-head pattern, but GPT-OSS uses its own tokenizer and Harmony format. Therefore: inspect the actual GPT-OSS tokenizer; test candidate existing/reserved tokens or an internal marker representation; never assume a special decision token exists.

python scripts/inspect_tokenizer.py --model \<GPT_OSS_MODEL\> --search-candidates "\<DECIDE\> \<STATE\> \<Q\> \<OPT\>"

python scripts/run_marker_ablation.py --benchmark data/splits/locked_v1

## 18.2 Capture tensors

required captures:

\- hidden state at query/decision marker

\- hidden state at each option marker

\- decision type embedding input

\- candidate mask

\- attention/position metadata needed to reproduce the representation

\- prompt hash

\- model revision

\- tokenizer revision

\- generation configuration

## 18.3 Cache representations

python -m gpt_oss.hidden_state_capture.run \\

--states data/decisions \\

--model \<PINNED_GPT_OSS_MODEL\> \\

--marker-config configs/decisions.yaml \\

--device \<AVAILABLE_RUNTIME\> \\

--dtype \<VERIFIED_DTYPE\> \\

--out data/gpt_oss_representations/\<MODEL_REV\>/\<RUN_ID\>

| **Hardware reality** Your 16 GB RAM / 6 GB GPU is not the target environment for 120B model weights. OpenAI states GPT-OSS-120B is designed to fit within 80 GB memory in its native MXFP4 form. Use a sufficiently provisioned machine for representation extraction when required; keep your local machine for the rest of the engineering loop. |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 19. PHASE 16 — IMPLEMENT THE INTERNAL POINTER HEAD

\# Minimal mathematical contract

q = Wq @ h_query

k_i = Wk @ h_option_i

score_i = dot(q, k_i) / sqrt(d_head) + bias_i

logits = score_i / temperature

P(i) = softmax(logits)

## 19.1 Trainable parameters — initial head-only run

| **Parameter**                        | **Train?**        | **Reason**                            |
|--------------------------------------|-------------------|---------------------------------------|
| GPT-OSS base weights                 | NO                | Freeze backbone                       |
| Wq                                   | YES               | query projection                      |
| Wk                                   | YES               | candidate projection                  |
| candidate/action metadata projection | YES               | encode action type/tool/decision type |
| abstain head                         | YES               | explicit abstention                   |
| escalation head                      | YES               | specialist/human fallback             |
| decision-type embeddings             | YES               | share one head across decision types  |
| calibration temperature(s)           | YES, separate fit | post-hoc probability calibration      |

## 19.2 Implementation files

decision_head/pointer.py \# q/k projections + scaled dot product

decision_head/candidate_encoder.py \# candidate metadata features

decision_head/choice.py \# softmax over arbitrary candidate set

decision_head/noul.py \# binary/yes-no head

decision_head/score.py \# ordinal expected-score head

decision_head/abstain.py \# abstain probability

decision_head/escalation.py \# escalate probability

decision_head/calibration.py \# temperature / calibration transforms

## 19.3 Pseudocode

def decide(hidden_query, hidden_options, decision_type, metadata):

q = Wq(hidden_query)

keys = Wk(hidden_options)

meta = candidate_encoder(metadata)

scores = (keys \* q.unsqueeze(0)).sum(-1) / sqrt(d)

scores = scores + meta_bias(meta, decision_type)

p = softmax(scores / T)

abstain_p = sigmoid(abstain_head(concat(hidden_query, pooled_meta)))

escalate_p = sigmoid(escalate_head(concat(hidden_query, pooled_meta)))

return p, abstain_p, escalate_p

# 20. PHASE 17 — INTERNAL HEAD TRAINING

## 20.1 Dataset format

{"state_id":"...","decision_type":"ACTION","candidate_ids":\["a","b","c"\],"winner":"b",

"soft_target":{"a":0.08,"b":0.79,"c":0.13},

"executed_outcome":{"a":null,"b":{"success":true},"c":null},

"evidence_strength":0.93, "source_authority":"REAL_EXECUTION_OUTCOME",

"representation_path":"data/gpt_oss_representations/..."}

## 20.2 Loss

L = lambda_hard \* CE(P, y_hard)

\+ lambda_soft \* KL(P \|\| y_soft)

\+ lambda_ord \* ordinal_loss(P, y_score)

\+ lambda_abst \* abstention_loss(P_abstain, y_abstain)

\+ lambda_perm \* permutation_consistency(P, shuffled_options)

Start with hard + soft target supervision and permutation consistency. Add more elaborate objectives only when an ablation demonstrates measurable value.

## 20.3 Training command

python -m training.internal_head.train \\

--backbone frozen \\

--representations data/gpt_oss_representations/\<REV\> \\

--labels data/outcomes \\

--types ACTION,EVIDENCE,CONTINUE,RECOVERY,PATCH,COMPLETION,ESCALATE,ABSTAIN \\

--trainable pointer,metadata,abstain,escalate,decision_type_embeddings \\

--output runs/internal_head/\<RUN_ID\>

| **GATE 4** Do not unfreeze GPT-OSS because the head is merely “not good enough” on one split. First inspect data leakage, marker quality, candidate quality, calibration, and state selection. Backbone adaptation is a separate experiment. |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 21. PHASE 18 — CALIBRATE THE INTERNAL HEAD

python -m training.calibration.fit \\

--predictions runs/internal_head/\<RUN_ID\>/calibration_predictions.jsonl \\

--split calibration \\

--group-by decision_type,candidate_count \\

--out runs/internal_head/\<RUN_ID\>/calibration.json

Measure ECE, Brier, NLL/log loss and reliability curves. Fit only on the calibration split. The calibration file is a first-class artifact and must be versioned with the head.

# 22. PHASE 19 — ADD DETERMINISTIC POLICY ABOVE THE LEARNED SCORE

## 22.1 Authority order

1\. hard safety / permissions

2\. tool invariants / schema validity

3\. objective execution results

4\. calibrated internal head

5\. specialist advice

6\. heuristics

## 22.2 Hard rules

class HardPolicy:

def check(self, candidate, state):

validate_schema(candidate)

validate_permissions(candidate)

deny_forbidden_destructive_patterns(candidate)

deny_secret_exfiltration(candidate)

deny_unknown_tool(candidate)

return ALLOW

A specialist such as Gyra can inform uncertainty around safety-like decisions, but deterministic rules remain authoritative for obvious forbidden actions. This is policy, not learned ranking.

# 23. PHASE 20 — CONDITIONAL SPECIALIST ROUTING

## 23.1 Specialist applicability

| **Specialist**      | **Use only for**                                                        |
|---------------------|-------------------------------------------------------------------------|
| laya-code           | code relevance / retrieval reranking                                    |
| laya-conductor      | effort/continue-style routing decisions                                 |
| completion judge    | turn completion/incomplete-tail decisions                               |
| evidence gate       | evidence sufficiency                                                    |
| supersession        | whether later evidence supersedes earlier evidence                      |
| Gyra                | uncertain safety / guard signals; never sole hard gate                  |
| vision Laya         | visual decision/evidence extraction tasks when screenshots matter       |
| browser specialists | web/browser navigation decisions when browser state is part of the task |

## 23.2 Confidence policy

if hard_policy.denies(candidate): DENY

elif head.confidence \>= T_high and top_margin \>= M_high:

EXECUTE

elif head.confidence \<= T_low or contradiction_detected:

CALL_RELEVANT_SPECIALIST_OR_ABSTAIN

else:

RECHECK_STATE_OR_SCORE_PERMUTED_OPTIONS

# 24. PHASE 21 — RETRIEVAL AND STATE SELECTION

## 24.1 Retrieval pipeline

repo -\> tree-sitter chunks -\> BM25 top-N -\> laya-code rerank -\> selected snippets -\> state builder

The head should not receive all 128K tokens just because GPT-OSS supports 128K. The state selector should build a decision-relevant slice. Start with ~5–15 top code candidates and include tests, errors, git diff, relevant memory, and the current action boundary.

## 24.2 State compression rules

preserve:

\- current objective

\- latest contradictory evidence

\- latest failing/successful test output

\- current git diff

\- changed files

\- pending actions

\- specialist observations

\- memory facts with support links

\- unresolved hypotheses

remove/condense:

\- duplicate tool output

\- superseded hypotheses

\- unrelated repository history

\- repeated unchanged state

# 25. PHASE 22 — MEMORY / EVIDENCE GRAPH

MemoryObject:

id

kind: FACT \| OBSERVATION \| HYPOTHESIS \| DECISION

content

source_event_id

support_ids\[\]

contradict_ids\[\]

supersedes_ids\[\]

confidence

created_at

state_hash

## 25.1 Update algorithm

1\. append new observation

2\. retrieve nearby supporting/contradicting observations

3\. mark superseded facts

4\. never delete raw event

5\. expose only active + relevant history to state builder

6\. hash the resulting state

# 26. PHASE 23 — RUNTIME DECISION LOOP

while not terminal:

state = state_builder.build(session)

candidates = gpt_oss.generate_candidates(state)

hard = hard_policy.filter(candidates, state)

if hard.empty():

return escalate_or_safe_stop()

h = gpt_oss.capture_decision_representations(state, hard)

decision = internal_head.decide(h, hard)

decision = calibrator.apply(decision)

decision = coordinator.apply(decision, specialists, memory)

decision = hard_policy.finalize(decision)

result = tool_runtime.execute(decision.candidate)

journal.append(decision, result)

state = state_builder.update(result)

# 27. PHASE 24 — DEEPSEEK HARNESS INTEGRATION

Do not fork the Harness loop. Implement an adapter plugin that listens to the documented Cordis extension points.

| **Harness event**   | **Our component**                 | **Action**                                           |
|---------------------|-----------------------------------|------------------------------------------------------|
| agent/pre-step      | state builder + confidence policy | assemble current decision state before model step    |
| agent/request       | GPT-OSS adapter                   | inject the canonical state/prompt/tool schema        |
| tools/pre-execute   | hard policy + specialist guard    | allow/deny/ask before dispatch                       |
| tools/execute       | metrics wrapper                   | measure runtime; preserve cancellation semantics     |
| tools/post-execute  | result enrichment                 | attach decision IDs and normalized outcome metadata  |
| tools/result        | outcome journal                   | persist immutable final outcome                      |
| agent/turn-stopping | completion policy                 | block premature stop when acceptance invariants fail |
| session/event       | durable event listener            | persist turn/tool/collision/compaction provenance    |

## 27.1 Plugin skeleton

export const name = 'internal-decision-agent';

export function apply(ctx) {

ctx.on('agent/pre-step', async (step, next) =\> {

const state = await buildState(step);

await decidePreStep(state);

return next();

});

ctx.on('tools/pre-execute', async (exec, next) =\> {

const verdict = await hardPolicyAndDecisionGuard(exec);

if (verdict.kind === 'deny') return verdict;

return next();

});

ctx.on('tools/result', async (exec, result) =\> {

await journalOutcome(exec, result);

});

ctx.on('agent/turn-stopping', async (...args) =\> {

return completionGate(...args);

});

}

| **Harness rule** Use monotonic guards for invariants that must never be reversed by a later listener. Use tools/pre-execute for the extensible policy waterfall; use tools/result for authoritative observation of the final tool outcome. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 28. PHASE 25 — LIVE OUTCOME DATA FEEDBACK

session

-\> decision record

-\> chosen action

-\> tool execution

-\> outcome

-\> label reconciliation

-\> candidate/negative/counterfactual record

-\> offline dataset

-\> locked regression

-\> model release

Live data must never directly mutate production weights. New data flows into a versioned offline corpus. A new model/head is promoted only after the locked benchmark and regression suite pass.

# 29. PHASE 26 — OPTIONAL SELECTIVE GPT-OSS PEFT

| **Only after head-only success is measurable** Run a controlled experiment that unfreezes a small adapter/LoRA parameter set or selected late layers. Keep the original frozen-backbone head checkpoint as the control. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Experiment A: frozen GPT-OSS + internal head

Experiment B: frozen GPT-OSS + head + LoRA

Experiment C: selected late-layer adaptation + head

Compare against GPT-OSS-alone and standalone Coding-Laya teacher.

| **Requirement before PEFT**    | **Why**                                             |
|--------------------------------|-----------------------------------------------------|
| head-only benchmark completed  | establishes causal value of the architecture        |
| data leakage audit passed      | avoids training artifact masquerading as gain       |
| marker ablation completed      | ensures representation interface is valid           |
| permutation/OOD tests passed   | checks the head is not exploiting superficial order |
| reproducible checkpoint + seed | permits real comparison                             |

# 30. PHASE 27 — EVALUATION MATRIX

| **Test**          | **Baseline**                 | **System under test**    | **Pass criterion**                                                |
|-------------------|------------------------------|--------------------------|-------------------------------------------------------------------|
| Decision accuracy | GPT-OSS-alone / Laya teacher | internal head            | statistically meaningful improvement or justified tradeoff        |
| Calibration       | uncalibrated logits          | calibrated head          | lower ECE/Brier on held-out calibration/test                      |
| Permutation       | fixed order                  | shuffled order           | small degradation                                                 |
| OOD               | in-distribution              | OOD                      | measured degradation bounded and documented                       |
| Recovery          | raw GPT-OSS                  | head + policy            | fewer repeated failed actions                                     |
| Completion        | raw stop behavior            | completion gate          | premature stops reduced without suppressing legitimate completion |
| Safety            | none/LLM-only                | hard policy + specialist | hard forbidden cases always blocked                               |
| Long context      | full packed context          | selected state           | same or better decision quality at lower decision-context size    |
| Agentic           | GPT-OSS baseline             | full stack               | higher task success or lower tool waste without unsafe regression |

# 31. PHASE 28 — TEST SUITE YOU MUST IMPLEMENT

tests/

test_state_schema.py

test_state_hash.py

test_candidate_schema.py

test_candidate_masking.py

test_pointer_math.py

test_choice_permutation.py

test_abstain.py

test_escalation.py

test_calibration.py

test_hard_policy.py

test_completion_invariants.py

test_specialist_routing.py

test_dataset_dedup.py

test_split_leakage.py

test_teacher_provenance.py

test_counterfactual_accounting.py

test_journal_replay.py

test_deepseek_harness_plugin.py

test_live_outcome_feedback.py

## 31.1 Required CI commands

pytest -q

pytest --cov=. --cov-report=term-missing

python scripts/check_dataset_leakage.py data/splits/v1

python scripts/check_manifest_hashes.py

python scripts/replay_journal.py --fixture tests/fixtures/sample_episode.jsonl

| **GATE 5** No deployment build can be created if any hard-policy, split-leakage, journal-replay, or candidate-masking test fails. |
|-----------------------------------------------------------------------------------------------------------------------------------|

# 32. PHASE 29 — REPRODUCIBILITY PACKAGE

release/\<RELEASE_ID\>/

model_manifest.json

tokenizer_manifest.json

teacher_manifest.json

head_checkpoint.safetensors

calibration.json

benchmark_manifest.sha256

train_manifest.sha256

environment.json

git_commit.txt

metrics.json

failure_analysis.md

policy_version.txt

harness_plugin_commit.txt

A release is reproducible when another environment can identify exact model revisions, dataset manifests, code commit, calibration artifact, configuration, and evaluation results.

# 33. FINAL END-TO-END EXECUTION ORDER

1.  1\. Create repo and pin Python/dependencies.

2.  2\. Register exact GPT-OSS model + tokenizer revision.

3.  3\. Register exact Laya-derived teacher revisions.

4.  4\. Define decision types and JSON schemas.

5.  5\. Build and hash the locked benchmark.

6.  6\. Ingest trajectories/Git/tests with provenance.

7.  7\. Normalize episodes and reconstruct pre-action states.

8.  8\. Extract decisions automatically.

9.  9\. Generate 5–15 candidate actions per decision where possible.

10. 10\. Generate hard negatives.

11. 11\. Execute safe counterfactuals in sandbox; record only real outcomes.

12. 12\. Run task-appropriate Laya/Cortex/specialist teachers.

13. 13\. Reconcile labels using execution-first authority.

14. 14\. Deduplicate and produce repo/issue-disjoint splits.

15. 15\. Benchmark existing Laya-derived checkpoints.

16. 16\. Train Coding-Laya only if the benchmark shows a gap.

17. 17\. Fit teacher calibration on calibration data.

18. 18\. Build the GPT-OSS decision marker after tokenizer inspection.

19. 19\. Capture query and candidate hidden states.

20. 20\. Train pointer/metadata/abstain/escalation head with GPT-OSS frozen.

21. 21\. Fit calibration for the internal head.

22. 22\. Compare head-only against GPT-OSS-alone and teacher baselines.

23. 23\. Add deterministic hard policy, permissions, and completion invariants.

24. 24\. Add conditional specialist routing.

25. 25\. Add state selection, retrieval, memory, and supersession.

26. 26\. Run the complete runtime loop on offline episodes.

27. 27\. Integrate with DeepSeek Harness using Cordis plugins/events.

28. 28\. Enable live journaling and offline corpus growth.

29. 29\. Only then test selective GPT-OSS PEFT/late-layer adaptation.

30. 30\. Re-run locked benchmark, safety, OOD, permutation, recovery, and completion tests.

31. 31\. Produce release package and freeze all artifact hashes.

# 34. WHAT YOU TRAIN VS WHAT YOU DO NOT TRAIN

| **Component**              | **Initial action**    | **Later action**                                  |
|----------------------------|-----------------------|---------------------------------------------------|
| GPT-OSS 120B               | Freeze                | Selective PEFT only if benchmark justifies it     |
| Internal decision head     | TRAIN                 | Iterate as evidence improves                      |
| Calibration                | FIT                   | Refit per head/checkpoint/data regime             |
| Coding-Laya                | ONLY IF NEEDED        | Refine with newly compiled coding decisions       |
| Laya-code                  | REUSE                 | Optional future distillation into retrieval layer |
| Conductor                  | REUSE                 | Optional distillation                             |
| Completion judge           | REUSE                 | Optional distillation                             |
| Gyra / safety specialist   | REUSE                 | Never replace deterministic hard rules            |
| Vision/browser specialists | REUSE                 | Only when modality/task requires                  |
| DeepSeek Harness loop      | DO NOT RETRAIN/MODIFY | Plugin integration only                           |

# 35. FAILURE MODES AND EXACT RECOVERY

| **Failure**                             | **Immediate check**                                      | **Recovery**                                                             |
|-----------------------------------------|----------------------------------------------------------|--------------------------------------------------------------------------|
| Head worse than GPT-OSS-alone           | candidate quality, marker ablation, leakage, calibration | fix data/interface before PEFT                                           |
| Teacher disagreement high               | decision type mismatch or poor state packing             | narrow specialist scope; preserve disagreement as uncertainty            |
| Permutation sensitivity high            | candidate encoding/order leakage                         | shuffle during training; remove position leakage                         |
| OOD collapse                            | train distribution too narrow                            | add repo-disjoint training data and abstention coverage                  |
| Completion gate blocks valid work       | acceptance criteria too rigid                            | separate required invariants from optional heuristics                    |
| Safety model disagrees with rule        | learned model overrules policy                           | hard rule remains authoritative                                          |
| Live data degrades model                | distribution shift or bad auto-labels                    | quarantine live batch, revert checkpoint, inspect provenance             |
| 120B representation extraction too slow | runtime bottleneck                                       | cache hidden states on adequate accelerator; keep head training separate |

# 36. FINAL ACCEPTANCE CONTRACT

| **The project is FINAL only when all statements below are true.** The locked benchmark is immutable and hashed. GPT-OSS is pinned. The internal head consumes GPT-OSS representations. The head is trained independently of the frozen backbone in the initial release. Candidate options are dynamically scored without a fixed hardcoded class list. Calibration is a separate artifact. Hard safety/permission policy is deterministic and authoritative. Specialist models are conditional, provenance-tracked, and task-scoped. Real execution outcomes are journaled. Dataset growth is offline and gated. DeepSeek Harness is integrated through plugins/events without forking the agent loop. The release package can reproduce the exact code/model/data/calibration configuration. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 37. SOURCE REFERENCES VERIFIED FOR THIS RUNBOOK

| **Source**                          | **URL**                                                                                         | **Use**                                                                              |
|-------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| OpenAI — Introducing GPT-OSS        | https://openai.com/index/introducing-gpt-oss/                                                   | Architecture, 117B/5.1B, 36 layers, 128 experts, 128K context, MXFP4/80GB statement. |
| OpenAI Developers — GPT-OSS-120B    | https://developers.openai.com/api/docs/models/gpt-oss-120b                                      | Current model page and H100 deployment statement.                                    |
| Laya GitHub                         | https://github.com/NandhaKishorM/laya                                                           | Current Laya architecture, APIs, checkpoints, installation.                          |
| Laya typed-decisions                | https://huggingface.co/convaiinnovations/laya-typed-decisions                                   | Typed-decisions checkpoint and reported accuracy.                                    |
| Laya Conductor                      | https://huggingface.co/mvilacad/laya-conductor                                                  | Coding-agent routing/continuation specialist.                                        |
| Laya Code                           | https://huggingface.co/tindang/laya-code                                                        | Code relevance reranker and its stated limitations.                                  |
| Laya Stop-Completion Judge          | https://huggingface.co/tampajohn/laya-stop-completion-judge                                     | Completion specialist and held-out evaluation.                                       |
| Cortex-1 Large                      | https://huggingface.co/mukti-sys/cortex-1-large                                                 | Coding-agent decision specialist; author-reported benchmarks.                        |
| Jeeves GitHub                       | https://github.com/PostHog/jeeves                                                               | Pointer-head/decision-marker architecture precedent and training pipeline.           |
| DeepSeek Harness extension cookbook | https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cookbook/extension-cookbook.md | Cordis plugin extension points and policy waterfall.                                 |
| DeepSeek Harness tool pipeline      | https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/tool-execution-pipeline.md     | Tool execution flow, guards, pre-execute, result events.                             |

# APPENDIX A — MINIMUM CONFIGURATION FILES

## A.1 configs/system.yaml

project: internal-decision-agent

backbone:

name: gpt-oss-120b

trainable: false

decision_head:

trainable: true

hidden_size: \<READ_FROM_MODEL\>

head_dim: 1024

temperature: fit_on_calibration

state:

max_selected_tokens: 12000

candidates:

max_count: 12

policy:

hard_rules_first: true

abstain_enabled: true

specialists:

conditional_only: true

## A.2 configs/models.yaml

models:

\- id: gpt-oss-120b

source: hf

repo: \<PIN\>

revision: \<PIN\>

\- id: laya-typed

repo: convaiinnovations/laya-typed-decisions

revision: \<PIN\>

\- id: laya-conductor

repo: mvilacad/laya-conductor

revision: \<PIN\>

\- id: laya-code

repo: tindang/laya-code

revision: \<PIN\>

# APPENDIX B — THE ONE-LINE MENTAL MODEL

| **Remember this** GPT-OSS asks: “What could we do?” → the internal head asks: “Among these candidates, what should we do?” → deterministic policy asks: “Are we allowed to do it?” → the environment answers: “Did it actually work?” |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
