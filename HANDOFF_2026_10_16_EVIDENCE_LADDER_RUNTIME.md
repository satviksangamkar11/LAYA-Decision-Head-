# Handoff: Evidence-Ladder runtime, Build Step 2 (2026-10-16)

Start here. This document is the complete state of the Evidence-Ladder runtime build: what exists, what is verified (with receipts), what was found broken
and fixed, what is still open, and what only the user can decide. No model was run, no Qwen or GLM request was made, nothing was trained.

## 1. Where things are

| item | value |
|---|---|
| Repo | github.com/satviksangamkar11/LAYA-Decision-Head- |
| Build branch | `evidence-ladder-runtime-step2` (pushed up to 39e17b7; local commits after that are listed in section 9) |
| Earlier branch (P0, specs, kits, index) | `evidence-ladder-p0-2026-10-16` at d0466ed |
| Architecture (source of truth) | `ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md` (R1, R2, 12c) |
| Specs | `specs/evidence_ladder_v1/` S1-S6, `schemas_v1.json`, `P0_STATUS_2026_10_16.md` (supersedes the specs README) |
| Frozen thresholds | `configs/thresholds_evidence_ladder_v1.toml` + `.sha256` (pending values listed inside) |
| Combined index | `results/raw/combined-index-v1-20261015T221945Z` (40 Python, 40 Java, 46 Fable VAL, 335 exposed Fable rows) |
| Runtime code | `evidence_ladder/` (common, controller, execution, orchestrator, models, testing) |
| Master checklist | `evidence_ladder/MASTER_CHECKLIST.md` |
| Kaggle test receipts | `results/raw/evidence-ladder-kaggle-tests-20261016/run1..run5/` |

## 2. What the runtime is (one paragraph per package)

- `common/`: the shared interface contract (`interfaces.py`: Artifact, TaskSpec, CheckOutcome, HiddenGrade, Generation, protocols for execution,
  grading, generation, critic broker, scorer, budget, controller; ControllerContext has NO grader by construction) and the loader for the validated
  P0 decision core (`contract_import.py` -> `specs/evidence_ladder_v1/contract/contract_lib_v1.py`).
- `controller/`: the deterministic Evidence-Ladder controller. Requirement intake (explicit / contract / derived hypothesis), per-requirement routing,
  hash-chained evidence journal bound to artifact/environment/requirement/check versions, stale-evidence invalidation, conflict detection, scoped labels,
  risk-aware authorization, progress-based repair and the S5 stop causes, all eight invariants checked at the end of every run. Modes: `single` (arm A),
  `best_of_k` (B), `repair` (C), `ladder` (D0; with critic D1; with scorer E). Stage-1 answers can never be VERIFIED (no trusted check covers
  "correct on all legal inputs"): SUPPORTED is the ceiling.
- `execution/`: the sandbox. Content-addressed immutable artifacts (hash re-checked before every run); Python and Java run in bubblewrap as the
  unprivileged user `nobody` (user/pid/net/ipc/uts namespaces, read-only /usr and JDK, empty /tmp, no host mounts, no network, resource limits,
  group kill); one public-check outcome per example with counterexamples; failure classes; model-proposed experiments (stdin_input, function_call
  only); `ProtectedGrader` is the only module that can read protected cases and refuses while the `SelectionGate` is open. Java pinned to UTF-8.
- `orchestrator/`: the paired pilot framework. Verified index loading, arms A-E, matched-token budgets (T must be given; refuses while pending),
  shared seeds, two-phase critic flow, freeze-then-grade, authorization manifest check (refuses unless D1 waiver, D7, credits and GO are all
  recorded and hashes match), per-track reports (pooling raises), McNemar + Holm + paired bootstrap with the confirmatory contrasts marked
  PRESERVED FOR HOLDOUT, a synthetic dry run that claims no accuracy.
- `models/`: model interfaces, all gated. Qwen prompt templates and code extraction, a fixture generation backend, a real Qwen backend that refuses
  unless authorized AND on Kaggle AND an inference flag is set (its generate path has never run); GLM-5.3-flash request builder with blinding allow-list
  and schema checks, payload builder that refuses until the FC-2 reasoning parameter is set; the staged critic broker (content-hash packet ids,
  idempotent ingestion, bounded retries, Retry-After, stop-all-calls on payment/quota errors, append-only keyless ledger, resume from ledger);
  a PC sender with an injected transport and a fake-clock-tested 5-in-flight / 39-per-minute scheduler; LAYA, probe and random-head scorers with
  frozen metadata checks (no forward pass in this build).
- `testing/`: the integrated replay. Real sandbox outcomes recorded once in WSL (`record_execution_fixtures.py` -> `fixtures/`), replayed by
  hash through the real controller, broker, budgets and pilot runner (`replay_integration.py`, 26 tests).

## 3. Test receipts (actually executed)

| suite | where | result | receipt |
|---|---|---|---|
| contract (S1-S5, rollback, schemas) | Kaggle CPU (Python 3.13.15) | 23/23 | run5/TEST_RESULTS.json |
| controller | Kaggle CPU | 25/25 | run5 |
| orchestrator | Kaggle CPU | 24/24 | run5 |
| models and critic broker | Kaggle CPU | 28/28 | run5 |
| integrated replay | Kaggle CPU | 26/26 (4 defects confirmed in run4, all fixed in run5) | run4, run5 |
| execution and grading | WSL bubblewrap (sandbox exception) | 21/21, re-run by the coordinator from the main checkout | coordinator run, 2026-10-16 |
| sandbox recording for the replay | WSL | 13 cases, 65 public outcomes, 39 experiments, 13 hidden grades | `evidence_ladder/testing/fixtures/` |

Totals: 126/126 on Kaggle plus 21/21 in WSL. Every Kaggle run verified the bundle's file hashes (71 to 78 files).
Earlier receipts kept intact: Python red-team 7/7, references 40/40, wrong solutions 0/37 pass; Java references 40/40 (replayed by hash in the
execution suite).

## 4. Defects found and fixed during this build

| # | defect | how found | fix |
|---|---|---|---|
| 1 | CLAUDE.md rule-removal edit never committed (a commit message claimed it) | baseline check | committed (9d763fd) |
| 2 | two agent worktrees created on an old base commit (fe1a01e) | coordinator check | agents re-based or copied the base files |
| 3 | execution receipt hashes depended on line endings | coordinator re-run 19/20 | hashes over LF-normalized bytes |
| 4 | controller wanted per-example outcomes, execution returned an aggregate | interface review | contract changed; execution emits one outcome per example |
| 5 | controller could not read an ingested critic response | interface review | `CriticBroker.completed(packet_id)` |
| 6 | `.gitignore` rule `models/` silently ignored evidence_ladder/models | Agent 4 | `git add -f` (no .gitignore edit) |
| 7 | LAYA capture-code files untracked in git | Agent 4 | shipped in the Kaggle bundle |
| 8 | fake key text in a test tripped the secret scan; my first scan did not gate a local commit | coordinator | text built at runtime; commit amended before any push; scans now gate with `&&` |
| 9 | controller test fakes returned a default experiment output that broke the quoted example | Kaggle run 1 | fakes fixed |
| 10 | orchestrator: two arms with the same packet id, response routed to only one broker | Kaggle runs 1-2 | route to every broker, send each packet once |
| 11 | malformed/unavailable critic packets looked "pending" forever | replay run 4 | `terminal_state()` + controller mapping |
| 12 | payment stop (402) crashed the second pass | replay run 4 | controller catches CriticStopped and continues without the critic |
| 13 | real LAYA refusal crashed arm E | replay run 4 | controller refuses the scorer instead of crashing |
| 14 | pilot critic proxy lacked `completed`; read-only pending checks were mutating (`export_pending` marks packets exported) | replay run 4 + review | proxy forwards; pending checks use `pending_summary` |
| 15 | arms C/D0/D1/E ignored k1 | replay agent | `cfg.setdefault("k", k1)` |
| 16 | Kaggle kernel copy dropped every nested MANIFEST.json (LAYA looked disabled) | coordinator review | copy all, delete only the bundle manifest |
| 17 | kernel summary print crashed on pretty-printed JSON (status ERROR, results intact) | Kaggle run 5 | parser prefers the whole stdout object |
| 18 | a Kaggle CLI call hung with no timeout and stalled the session | user report | every Kaggle call now runs under `timeout` |

## 5. Known limitations (recorded, not hidden)

- **Kaggle sandbox not proven (FC-1 unrun).** All candidate execution and grading ran in WSL only. No secure-Kaggle-execution claim is made.
- **The protected development cases are public in Git history.** They are a development oracle, not a sealed benchmark.
- **Oracle adequacy gaps.**
  - Raw mutation kill rate below 0.80 (survivors judged likely equivalent by random search, which is evidence, not proof): P1-19, P1-26, J1-17, J1-27.
  - Unmeasured: P1-05, P1-10, P1-14, J1-08, J1-10, J1-20.
  - The development-dataset waiver does NOT waive this rule; the index flags these tasks.
- **Public checks are weak.** 19 of 37 known-wrong Python solutions pass all public examples; only the protected cases catch them.
- **Not implemented in this build:**
  - LAYA forward pass (scorers refuse);
  - Fable-track run support (stage 2);
  - diverse prompt strategies;
  - DeepConf confidence (key read, never exercised);
  - a git-worktree commit for finalize (S4-T7 covered at library level only).
- **Never executed:**
  - the real Qwen generation path;
  - the real GLM transport (no request was ever made);
  - the NVIDIA reasoning-effort parameter (FC-2), which is unknown.
- **Pending values:** token cap T, k1, main-run sample size (set from the pilot).
- **Stop-cause reporting:** cause "budget" is also used for attempt/round caps; read `stop_details` before counting token exhaustion.
- **Theoretical channel:** a candidate can write to the Python result channel, but it cannot forge correct answers without the hidden expected values, which never enter the jail.

## 6. Decisions only the user can make (pilot is blocked until all four are recorded)

| id | decision | current state |
|---|---|---|
| D1 | accept the Python + Java development banks under a recorded waiver (no external post-2025-07-31 set with offline tests exists: LiveCodeBench-Pro test cases stop at contests from 2025-06-23) | waiting |
| D7 | where critic calls run: (a) PC sends staged packets; or the two-kernel Kaggle design: kernel A generates and runs model code with internet OFF and NO secrets, kernel B only sends critic packets with internet ON and the NVIDIA keys as Kaggle Secrets, never running candidate code | waiting; nothing reads keys today |
| credits | NVIDIA trial credits confirmed on the account page (spend stays Rs 0; payment/quota errors stop all critic calls) | waiting |
| GO | explicit pilot GO | not given |
The authorization manifest (`orchestrator/authorization.py`) records these four, hashes them, and the pilot runner refuses to start without it.

## 7. Stage 0 (only after GO)
- **FC-1:** jail feasibility on Kaggle.
- **FC-2:** the GLM reasoning parameter.
- **FC-3:** Qwen throughput, which sets T.
- **FC-4:** protected-test separation in the real environment.
- Then the contract and red-team tests in that environment.
- A mandatory failure stops the pilot.

## 8. How to run things

- Kaggle test bundle (needs the user's yes before every upload):
  1. `python -X utf8 tools/pack_evidence_ladder_tests_v1.py`
  2. from `<bundle>/dataset`: `timeout 300 .venv/Scripts/kaggle.exe datasets version -p . -m "<msg>" --dir-mode zip`
  3. wait for `datasets status` = ready
  4. from `<bundle>/kernel`: `timeout 120 .venv/Scripts/kaggle.exe kernels push -p .`
  5. poll `kernels status` (each call under `timeout 60`)
  6. `kernels output ... -p <run dir>`
  7. read `TEST_RESULTS.json`.
- WSL execution suite (sandbox exception): `MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -u root -e bash -c 'export PATH=/opt/toolchains/bin:$PATH; cd "/mnt/d/local model" && python3 evidence_ladder/execution/tests/run_tests.py'`
- Re-record replay fixtures (WSL, sandbox exception): same pattern with `python3 evidence_ladder/testing/record_execution_fixtures.py`.
- Compute rule: pure-logic tests run on Kaggle only (user, 2026-10-16); only sandbox runs stay in WSL.

## 9. Commits on the build branch
- 9d763fd: start: interface contract, checklist, CLAUDE.md edit.
- fb19d9e: kernel and packer.
- aadec56: the four packages.
- ffe7d1b: run 1 and test fixes.
- afba31a: orchestrator routing fix.
- 39e17b7: run 3 (100/100). Pushed up to here.
- 100deaa: replay suite, run 4, defect fixes 11-16. Local.
- Next commit: run 5 receipts, kernel parser fix, this handoff. Local until the user says push.

## 10. Models used (self-reported by each agent, not independently verified)
- **claude-sonnet-5-5 at medium effort:**
  - the four build agents (controller, execution, orchestrator, models);
  - the replay agent.
- **claude-haiku-5-5:** every small edit. Per the user's rule: Haiku for small edits, Sonnet medium for full code.
