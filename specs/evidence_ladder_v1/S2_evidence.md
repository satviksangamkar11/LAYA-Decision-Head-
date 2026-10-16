# S2. Evidence-record contract (Evidence-Ladder v1, P0 + R3, 2026-10-16)

Schema: `schemas_v1.json#/$defs/evidence_record`, `#/$defs/decision_record`.

## 1. One record per (requirement, check, artifact)

Every check result becomes an append-only record. It names:
- what was checked: `requirement_id`, `requirement_version`;
- by what: `check_id`, `check_version`;
- on exactly what: `candidate_id`, `artifact_hash`, `environment_revision`;
- where the evidence came from (`source`) and whether that source is independent of the models (`independence`);
- what it establishes and under what conditions (`scope`, `assumptions`);
- a `counterexample` when it fails.

## 2. Independence classes (the oracle-independence contract)

| `source` | `independence` | may approve? |
|---|---|---|
| `environment_observation` (real end state, trusted tool result) | independent | yes, for the observed fact |
| `existing_test`, `user_test` | independent / user_provided | yes, **within the check's `covers` list** (S3) |
| `certified_check` | independent | yes, within its stated scope |
| `request_example` | user_provided | rejects, never approves alone |
| `model_test` (Qwen-written), `critic_experiment` (GLM-proposed) | model_derived | **no** (ranking and repair feedback) |
| `candidate_agreement` | model_derived | no (ranking) |
| `scorer` (LAYA, probe, confidence) | model_derived | no (ordering and ties) |

A `critic_experiment` is an exception only in this sense: if the experiment's input is run and the result shows that a candidate breaks an **explicit**
requirement or a quoted example, that is recorded as a separate record with `source = environment_observation` and `status = fail`. The critic proposes
the input; the execution produces the evidence. The critic's expected output is never treated as truth.

## 3. Version binding and invalidation

- A record applies only if all five match the item being decided:
  - `artifact_hash` (exact bytes of the candidate);
  - `environment_revision` (base state);
  - `requirement_version`;
  - `check_version`;
  - `candidate_id`.
- A repaired candidate is a new candidate with a new hash. **No evidence is carried over.**
- At commit time the controller recomputes the artifact hash and re-runs the trusted checks on the real end state (S4 section 5). A mismatch means BLOCK.

## 4. Per-requirement labels and the answer label

Row label per requirement (computed by the controller only):
- **FAILED:** any applicable trusted record has status `fail`.
- **CONFLICTED:** applicable trusted records disagree, for example `pass` and `fail` from two independent checks on the same artifact.
- **VERIFIED:** an applicable trusted record (approves per section 2) has status `pass`, and none disagree.
- **SUPPORTED:** no trusted record, but model-derived records or quoted examples pass with no failures.
- **UNVERIFIED:** nothing applicable, or only `unknown` / `timeout` / `error` / `missing_evidence` / `invalid_spec` (I4: these never count as pass).

The answer label is computed from the rows of hard and functional requirements:
- VERIFIED if every row is VERIFIED;
- otherwise SUPPORTED if every row is VERIFIED or SUPPORTED;
- otherwise BLOCKED if any row is FAILED or CONFLICTED;
- otherwise UNVERIFIED.

PLAUSIBLE is used only when no requirement has any applicable record.

**CONFLICTED never approves.** The controller investigates by re-running both checks and comparing their versions and scopes. If the conflict stays, it blocks and reports.

## 5. The decision record

`decision_record` is produced by the controller from records and policy (S4). It is never accepted as model output (`derived_by = "controller"`).

## 6. Tests (pilot stage 0)

| id | case | expected |
|---|---|---|
| S2-T1 | stale evidence: candidate A passes, repair produces A' | A' has no evidence until re-checked; approval of A' refused |
| S2-T2 | conflicting trusted checks on one artifact | row CONFLICTED, no approval |
| S2-T3 | model_test passes, no trusted check | row at most SUPPORTED, answer never VERIFIED |
| S2-T4 | `timeout` / `error` records only | row UNVERIFIED |
| S2-T5 | "never negative" proved, "implements policy" has no record | answer not VERIFIED (weakest row) |
