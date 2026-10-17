# Evidence-Ladder Build Step 2: master checklist (coordinator-owned; statuses DONE / FAILED / BLOCKED / PENDING)

Base: branch `evidence-ladder-p0-2026-10-16`, commit d0466edc33b98eb521d5f7df8d352babcc752e3e (equals origin). Build branch: `evidence-ladder-runtime-step2` (local only; not pushed, not merged).
Rules in force: no push or merge without the user's approval; no model inference, no GLM requests, no NVIDIA keys read, no Kaggle launch; spend Rs 0; compute on Kaggle except the WSL sandbox exception.

## Checkpoint 0: baseline
| item | status | evidence |
|---|---|---|
| Branch, HEAD, tree state verified | DONE | HEAD d0466ed = origin; working tree dirty with unrelated edits (.gitignore, DECISION_HEAD.md, PLAN.md, kaggle/*, pyproject.toml, uv.lock) and 715 untracked paths: preserved, untouched |
| CLAUDE.md rule-removal edit | **DEFECT FOUND, FIXED** | the edit was never committed (the blocked command never ran `git add`); commit ba0aca0's message claimed it. Now committed on the build branch |
| Unrelated changes preserved | DONE | only owned paths are staged |
| 23/23 contract tests recorded | DONE | specs/evidence_ladder_v1/contract_test_results.json (sha256 7f562e29...); negative controls: 1 and 4 tests fail when the library is broken |
| Python/Java sandbox receipts recorded | DONE | python red-team 7/7, reference 40/40, wrong 0/37 pass (fixed_v2/reports); java reference 40/40, hostile candidate blocked (fixed_v1/reports) |
| Combined index: 40 Python, 40 Java, 46 Fable VAL, 335 exposed | DONE | combined-index-v1-20261015T221945Z, validator PASS (structure), 50 sources hashed |
| Existing modules inspected | DONE | contract_lib_v1.py, grade_sandboxed_v2.py, runner_v2.py, grade_sandboxed_java_v1.py, tools/nvidia_keys.py (keys NOT read) |
| Shared interface contract | DONE | evidence_ladder/common/interfaces.py (coordinator-owned) |
| File ownership allocated | DONE | below |
| Agents launched in separate worktrees | DONE | 4 agents, model alias "sonnet", effort medium, isolation worktree; each must self-report its model ID (actual configuration verified from the reports, not assumed) |
| Persistent master checklist | DONE | this file |

Known caveats carried into the build: protected development cases are public in Git history (not a sealed benchmark); D1, D7, NVIDIA credits and the pilot GO are not given; the oracle-adequacy gate (>= 0.80 measured) is unmet for 2 Python tasks (P1-19, P1-26), 2 Java tasks (J1-17, J1-27) on the raw count, and unmeasured for P1-05, P1-10, P1-14, J1-08, J1-10, J1-20; the dataset waiver does not waive it.
Compute policy ANSWERED by the user (2026-10-16): pure-logic unit tests run on **Kaggle only**; only isolated SANDBOX test runs stay in WSL under the 2026-10-13 exception. Consequence: controller, orchestrator, models and contract suites run in a Kaggle CPU kernel (kaggle/evidence_ladder_tests_v1.py, packed by tools/pack_evidence_ladder_tests_v1.py); uploading that code bundle to Kaggle needs the user's explicit yes (CLAUDE.md "Ask first"). Execution/sandbox tests cannot run on Kaggle (bubblewrap) and make no Kaggle security claim.

## File ownership (non-overlapping)
| owner | paths |
|---|---|
| Coordinator | evidence_ladder/common/**, evidence_ladder/MASTER_CHECKLIST.md, evidence_ladder/testing/** (integrated replay), integration commits |
| Agent 1 controller | evidence_ladder/controller/** |
| Agent 2 execution and grading | evidence_ladder/execution/** |
| Agent 3 orchestrator | evidence_ladder/orchestrator/** |
| Agent 4 model interfaces and critic broker | evidence_ladder/models/** |

## Agents (status filled in at Checkpoint 1)
| agent | status | integration-ready |
|---|---|---|
| 1 controller | reported 2026-10-16 (model self-report: claude-sonnet-5-5); 22 tests written, NOT executed; resumed for integration fixes | no |
| 2 execution | reported; 20/20 WSL tests; coordinator re-ran in the main checkout: 19/20 then 20/20 after the receipt-hash re-pin (line-ending defect in the agent's pins, not in the sandbox); resumed to emit per-example public outcomes | WSL path yes; Kaggle untested |
| 3 orchestrator | reported; 21 tests written, NOT executed | no (needs Kaggle run) |
| 4 models | reported; ~29 tests written, NOT executed; resumed to add `completed(packet_id)` | no (needs Kaggle run) |

Integration gaps found at review (all assigned): (a) controller expects per-example public outcomes, execution returned an aggregate -> contract fixed, Agent 2 resumed; (b) controller had no way to read an ingested critic response -> `CriticBroker.completed()` added, Agents 4 and 1 resumed; (c) `.gitignore` ignores `models/`, so evidence_ladder/models needs `git add -f`; (d) LAYA capture-code files are untracked in git: added to the Kaggle bundle; (e) ProblemResult lacked final_artifact/stop_details: added.

## Checkpoint 1 result (verified by the coordinator, not by agent claims)
| suite | where | result |
|---|---|---|
| contract (S1-S5, rollback, schemas) | Kaggle CPU, run 3 | 23/23 |
| controller | Kaggle CPU, run 3 | 25/25 (run 1 had 23/25: two test-fixture bugs, fixed) |
| orchestrator | Kaggle CPU, run 3 | 24/24 (run 1 and 2 had 23/24: a REAL routing bug: one critic response reached only one of two arms' brokers; fixed in pilot.py) |
| models and critic broker | Kaggle CPU, run 3 | 28/28 |
| execution and grading | WSL, coordinator re-run | 21/21 (one defect found and fixed: receipt hashes pinned over CRLF-sensitive bytes) |
Bundle integrity: 71 file hashes verified in every Kaggle run; Python 3.13.15 on Kaggle. Receipts: results/raw/evidence-ladder-kaggle-tests-20261016/run{1,2,3}/.
Models actually used (self-reported by each agent): all four build agents and the replay agent claude-sonnet-5-5 at medium effort; the small-edit agents claude-haiku-5-5.

## Checkpoint 2: integrated deterministic replay: DONE
Kaggle run 5: replay 26/26 (run 4 had 22/26: four real defects, all fixed), plus contract 23, controller 25, orchestrator 24, models 28 = 126/126.
Covered: Python and Java correct and known-wrong solutions (real recorded sandbox outcomes), failed check then repair, stale and conflicting evidence,
GLM malformed / 429 / 500 / timeout / 402, budget exhaustion, irreversible refusal, unsupported LAYA decision, reproducibility (identical hashes on
two runs), protected grading never influencing selection (flipped grader gives identical selections), authorization refusal, per-track separation and
mutation flags. Full record: HANDOFF_2026_10_16_EVIDENCE_LADDER_RUNTIME.md.
## Checkpoint 3: authorization: decisions RECORDED, pilot GO NOT given
User confirmed 2026-10-16 ("yes go"): D1 dev-only waiver (does not waive grader adequacy), D7 = `kaggle_secrets_critic_kernel` (separate internet-on critic kernel, keys as Kaggle Secrets, never runs candidate code, no hidden grading material), NVIDIA credits confirmed by the user (not independently checked), pilot_go = false, spend Rs 0. Record: results/raw/evidence-ladder-decisions-20261016T090742Z/DECISIONS.json (sha256 0c80650f...). Schema v2: D7 stores the location string and must equal the runner's `critic_location`. Kaggle run 6: 127/127 (contract 23, controller 25, orchestrator 25 incl. new location test, models 28, replay 26). No authorization manifest written (pilot_go false). Critic kernel not built yet.
## Checkpoint 4: pilot execution: BLOCKED (needs Checkpoint 3)

## Stage 0 checkpoint (2026-10-16, after pilot GO; verified from saved outputs only)
| gate | status | evidence | blocker | next |
|---|---|---|---|---|
| FC-1 Kaggle jail | **ERROR** (not PASS) | results/raw/evidence-ladder-stage0-cpu-run3-20261016T123727Z/STAGE0_RESULTS.json (sha256 e3da5c7f...), redteam_v2_kaggle.json (34c32f59...) | custom jail (evidence_ladder/execution/kaggle_jail.py, no bubblewrap on Kaggle) fails at child setup: FileNotFoundError; 0 of 7 attacks actually executed (all PROCESS_ERROR); 0/40 references ran (SANDBOX_ERROR); Java rootfs lacks libjli.so | fix + rerun (in progress, Stage 0 agent) |
| FC-2 GLM reasoning/API | **PASS** | results/raw/evidence-ladder-critic-v1-run8-20261016/FC2.json (a6cda193...), KEY_CHECK.json (59ee3f74...) | none; note ~108-134 s per tiny call | use reasoning_effort="max", streamed |
| FC-3 Qwen speed | **NOT RUN** | run3 FC3_qwen_throughput = SKIPPED (no CUDA in that kernel) | needs a GPU kernel | separate free GPU kernel |
| FC-4 hidden-test isolation | **ERROR** (reported FAIL by the kernel because jailed grading crashed) | run3 FC4_separation_in_kaggle | partial evidence holds (selection stdin = public inputs only; grading stdin = protected inputs only, no expected values) but jailed grading returned SANDBOX_ERROR and the jail fs scan did not run | rerun after FC-1 works |
Kaggle: stage0 kernel `evidence-ladder-stage0-run-v1` and dataset `evidence-ladder-stage0-bundle-v1` (COMPLETE at checkpoint); critic `evidence-ladder-critic-v1` v8 COMPLETE (14 keys, 2 NVIDIA calls).
Uncommitted agent code at checkpoint: evidence_ladder/execution/{config.py, engine.py} (modified), evidence_ladder/execution/kaggle_jail.py, evidence_ladder/models/qwen.py (modified), kaggle/evidence_ladder_stage0_v1.py, kaggle/el_stage0_probe_push/, tools/pack_evidence_ladder_stage0_v1.py. The WSL execution suite (21/21) has NOT been re-run against the modified engine/config.
**Stage 0: BLOCKED.** No coding pilot while FC-1 is unverified. Spend Rs 0.

## Stage 0 (pilot GO 2026-10-16): kernel A, internet off, no secrets, no NVIDIA calls
Evidence root: results/raw/evidence-ladder-stage0-{probe,cpu-run1..6,gpu-run1}-*. Runs 1-5 are kept failure receipts (jail did not start; java libjli; /dev/null mode; test-marker self-hit).
| item | status | evidence |
|---|---|---|
| FC-1 probe | DONE | evidence-ladder-stage0-probe-20261016T120934Z/PROBE.json: root, seccomp filter present, unshare(user/net/pid/mount) = EPERM, no bwrap, apt offline; setuid/runuser nobody OK, chroot capability present |
| FC-1 jail on Kaggle | **PASS (chroot + uid 65534 + seccomp + rlimits; NO namespaces)** | cpu-run6-20261016T130550Z/STAGE0_RESULTS.json (sha256 69f84d31...): selfcheck PASS, python and java jails start, redteam_run_v2 7/7 (redteam_v2_kaggle.json eb3dbe6e...), 40/40 python references, 37/37 known-wrong caught, java 40 references, 20/21 execution tests |
| FC-1 residual gaps | RECORDED | no pid/net/mount/ipc namespace: network blocked by seccomp socket() EPERM (AF_INET/INET6/RAW/NETLINK) plus kernel internet off; AF_UNIX allowed (abstract host sockets reachable); x32-ABI syscalls not filtered; /tmp inside the rootfs is shared between runs (wiped before each); `isolation_receipt_contents` test (expects bwrap namespaces, reads /proc) FAILS by design here |
| FC-2 | not mine (done by coordinator) | |
| FC-3 Qwen throughput | **MEASURED** | gpu-run1-20261016T135840Z/STAGE0_RESULTS.json (sha256 82ca0235...): Tesla T4, shards sha256 match HF pins; 8.24 tok/s mean single stream (14.6 -> 7.2 as length grows), 42.7 tok/s batch 4, 5593 tokens/sample mean, 7968 max, 678 s/sample. Proposed T: configs/thresholds_evidence_ladder_v1_T.toml (not frozen) |
| FC-3 defect found | OPEN | prompt template qwen_sample_v1 has no solution-format instructions: 4/5 samples were stdin programs, public examples pass 1/5 |
| FC-4 separation | **PASS in Kaggle (cpu-run6)** | gate-open grading raises; selection stdin only public inputs; no protected input in selection stdin; in-jail filesystem scan for 40 raw bytes of the hidden-case file finds nothing; grading stdin = protected inputs only, no expected values; HiddenGrade has counts/boolean only; 47/47 reference graded after gate close. The flipped-grader check is in the replay suite (Kaggle run 5/6, 26/26), not re-run here |
| Contract + red-team tests in this environment | DONE | execution suite 20/21 (the one failure is the bwrap-namespace receipt test) |

## Stage 0 update (after the fix runs; full record HANDOFF_2026_10_16_STAGE0.md)
FC-1 Kaggle: PASS on its tests (run6: self-check, red-team 7/7 executed and blocked, 40/40 py refs, 37/37 wrong, java refs), but the layered jail (chroot+uid+seccomp+rlimits, no namespaces) is weaker than the approved bubblewrap design; hardening running. FC-1 WSL: 21/21 PASS. FC-2 PASS. FC-3 MEASURED (8.2 tok/s single, 42.7 batch-4, T4). FC-4 PASS. Critic isolation PASS (current version). **Stage 0: BLOCKED** (jail hardening + user acceptance, prompt v2, T/k1/backend).

### Stage 0 addendum: jail gaps closed (cpu-run7, results/raw/evidence-ladder-stage0-cpu-run7-20261016T141604Z, STAGE0_RESULTS.json sha256 1338589e...)
| item | status | evidence |
|---|---|---|
| seccomp: arch check KILL, x32 bit KILL | DONE | extra probes: x32 syscall in a forked child killed by signal 31 (SIGSYS); in the main process launcher exit 159 = 128+31 |
| AF_UNIX denied | DONE | socket() of every family returns EPERM (errno 1): abstract connect, path connect, dgram, inet all blocked; host listeners saw 0 connections. Java suite still passes (no AF_UNIX need found) |
| per-run private /tmp | DONE | /tmp is a symlink to a fresh nobody-owned 0700 dir under /runs (0711), removed after the run; flock serialises launchers; leftover uid-65534 daemons are reaped. Cross-run probe: run B sees empty /tmp, cannot list /runs (EACCES), no marker, daemon from run A did not write |
| CPU rlimit | DONE | busy loop died at 30.2 s with exit 137, not by wall kill |
| Kaggle receipt test | DONE | isolation_receipt_contents has a Kaggle branch (uid/gid/groups, no /proc, socket EPERM, x32 SIGSYS, rlimits, mechanisms recorded; "not_provided: namespaces"); the WSL branch is unchanged |
| Re-run results | DONE | selfcheck PASS; redteam_run_v2 7/7; execution suite 21/21 (40 python refs, 37/37 wrong caught, 40 java refs); FC-4 PASS; models tests 30/30; extra probes PASS |
| qwen_sample_v2 | DONE (code), see FC-3 v2 in the report | configs/qwen_sample_v2_reason.md |

FC-3 batch run (gpu-run3, dca5125a6ac0fa62): batch 5 OOM on T4; batch 3 = 16.74 tok/s aggregate, 174.5 s amortised per sample (523 s wall for the batch), 3/3 public pass; batch 2 not recorded. HF generate batching gives about 2x, not 4x (padding to the longest sample).

## Step 4: GLM Kaggle batch critic (2026-10-16): IMPLEMENTED and SYNTHETICALLY VALIDATED; LIVE BATCH NOT DONE
Receipts: results/raw/evidence-ladder-kaggle-tests-20261016/run11/ (TEST_RESULTS.json sha256 47e3d381...; Kaggle CPU, bundle hashes verified). run10 had 26/28 in critic_batch (a test-fixture bug on the cooldown check and a static test tripping on a literal in validator.py); both fixed, run11 green. Suites run11: contract 23, controller 25, orchestrator 25, models 30, critic_batch 28, replay 26 = 157/157, all exit 0.
| item | status | evidence |
|---|---|---|
| CP1 allow-list validator shared by PC export and kernel import | DONE | evidence_ladder/critic_batch/validator.py (aeb17416...); wraps glm.validate_request; rejects hidden tests, protected cases, reference solutions, private labels, LAYA, Qwen think text, secrets (nvapi/hf_/KGAT_/sk-/Bearer/PEM), unexpected line, top-level and observed_outputs fields, id != content hash; every class tested as value and as field name |
| CP1 packet dataset export/import, manifest sha256, rejection recorded | DONE | write_packets / read_packets / export_for_kernel; rejected.json and bundle MANIFEST list class + packet id; rejected packets are never sent (test) |
| CP1 kernel dataset_sources only the packet dataset (static check) | DONE | validator.check_kernel_metadata (+ check_dataset_dir file allow-list), run in the Kaggle test and by the packer; kaggle/evidence_ladder_critic_v2/kernel-metadata.json: dataset_sources = [sangeetasangamkar/evidence-ladder-critic-packets-v1], no kernel/model/competition sources |
| CP2 concurrency reconciliation | DONE | configs/critic_concurrency_v1.toml (4191467...): cap 10 in flight/key (user rule, CLAUDE.md quote inside), 39/min/key, 60 s cooldown; ramp start 2, +1 per key after 5 completions while recent 429 rate < 0.10 (window 20), halve on 429. Frozen thresholds file (says 5, sha 8b5e17cd...) NOT edited. config.load_concurrency refuses if missing, disagreeing with broker.LIMITS, malformed, or if the frozen file hash differs |
| CP2 parallel sender, no wave barrier | DONE | PcSender.send_streaming (submit + wait FIRST_COMPLETED; ingest each answer at once); KeyScheduler ramp optional (default unchanged); existing sender tests stay green. Test: 5 answers persisted while the slowest call still in flight; 14 keys start at 28 in flight, not 140 |
| CP2 429 / 402 / timeout / connection loss / malformed | DONE | Retry-After else 60 s cooldown + requeue + level halved; 402 stops all (also on restart and on PC import); TimeoutError and ConnectionResetError retried; malformed recorded not retried; retries exhausted -> unavailable |
| CP2 transport: streamed, bounded timeouts, keys never logged | DONE (mocked only) | critic_batch/transport.py: socket timeout 120 s per read, 1800 s total; labels only in ledgers; Bearer text redacted; NO live call made |
| CP2 progress ledger, resume, uncertain policy | DONE | progress.jsonl fsync per line (test counts fsyncs), torn last line healed; answer saved BEFORE ingest; completed packets never resent. DELIVERY IS AT-LEAST-ONCE: a packet in flight at a crash is UNCERTAIN, resent ONCE with the same packet_id, and if uncertain a second time becomes unavailable (uncertain_inflight_twice). Responses deduped by packet id. Tested over 3 crashing starts |
| CP3 response bundle + PC importer | DONE | responses.jsonl sorted by packet id + MANIFEST.json sha256 (packet id, attempt, model, usage, timing, state, raw); critic_batch.batch.import_bundle -> StagedCriticBroker.ingest (new StagedCriticBroker.requeue_exported); rejects duplicates, id/hash mismatch, unknown ids, not-completed, tampered bundle; full replay request -> kernel loop -> bundle -> PC ingestion states match |
| CP4 tests | DONE | 28 new tests in the Kaggle suite "critic_batch" (run11), added to kaggle/evidence_ladder_tests_v1.py; tests/pack tool lists updated |
| CP5 secret scan of added lines | DONE | 0 matches for token-shaped nvapi-/hf_/KGAT_/sk- in added lines and new files |
| Kernel pack prepared locally | DONE (not pushed) | tools/pack_evidence_ladder_critic_batch_v1.py --synthetic -> results/raw/evidence-ladder-critic-batch-pack-20261016T182650Z (1 synthetic packet; PACK.json pushed=false). Real packets must come from the PC broker via export_for_kernel |
| Kernel script sha256 | RECORDED | kaggle/evidence_ladder_critic_v2/evidence-ladder-critic-v1.py 3af7305e... (new version of the existing kernel id so attached Secrets carry over) |
| Live batch on Kaggle with real GLM calls | **NOT DONE** | no push, no NVIDIA call, no key read; needs the user's go, the real packet dataset upload, and a ramp watch on the first run |

## LAYA integration milestone (user decision 2026-10-17, chat): integration verification first, performance later
Pass criteria REVISED (user 2026-10-17): T1 real capture · T2 training parity · T3 adapters get correct tensors · T4 real head forward pass, valid scores · T5 controller uses scores only as L0 · T6 one end-to-end loop · T7 provenance rejection verified by the runtime path (live mode captures itself; replay mode re-captures a random subset; fabricated tensors with valid-looking metadata refused by re-capture). Verdicts reported separately: INTEGRATION PASS vs PERFORMANCE (not tested).
Earlier wording: Pass criteria (all six, receipts under results/raw/laya-integration-<UTC>/): T1 frozen head + 3 adapters load (sha256 match) · T2 capture parity with the training recipe (pre-registered thresholds) · T3 valid scores on real pools (recorded scores reproduced, permutation-stable) · T4 controller receives scores as L0 only · T5 LAYA cannot override stronger trusted evidence (I2) · T6 one small end-to-end loop, reproducible receipts.
Not gates for this milestone: LAYA vs probe, task-success gains, GLM accuracy gains (later evaluation). No 320-answer generation for this milestone.
Status: forward pass NOT YET IMPLEMENTED (being built); T1-T6 PENDING.
Update 2026-10-17 [LAYA-INT-1]: capture-code pin CORRECTED by version, not edited in place: v2 pins (capture/capture.py, capture/capture_resident.py, decision_head/fable_features.py) in configs/laya_capture_code_pins_v2.toml; the v1 table (tools/fable_capture.py) stays in laya.py as EXPECTED_CAPTURE_CODE_V1; reason and evidence: results/raw/laya-integration-20261016T184816Z/CORRECTION_NOTE.md. SMOKE PASS (3 TRAIN + 3 VAL pools, receipts smoke/); integration kernel run 2 DONE: T1, T2 (65/65 bit-identical), T3, T4, T7 GPU cases (replay verification re-captures 100%), T2b PASS (integration_run2/); T5 + T7 logic in the Kaggle test kernel: PENDING; T6 end-to-end loops: IN PROGRESS. PERFORMANCE: NOT TESTED.
Update 2026-10-17 [LAYA-INT-1] (agent): per-action mode of the REAL controller built (evidence_ladder/controller/agentic.py: Registry, journal, router, decide_stage1/authorize, next_action ranking, assert_invariants I1-I8 all pass). END-TO-END 3/3 loops completed through it (2 steps max): ex2_clamp VERIFIED (decided), ex1_slug and ex3_wordcount BLOCKED (budget; Qwen's edits did not apply/fix) - receipts results/raw/laya-integration-20261016T184816Z/e2e_pass2/ (hashes in FINAL_SHA256SUMS.txt). Qwen proposals + capture + 3 adapters + head on the Kaggle T4; every imported score pool is accepted only when a second independent Kaggle run (replay, separate kernel version) re-captures and agrees. Replay pass (fresh controllers over the saved outputs): journal hashes identical 3/3. Real executed conflict: ex1_slug step 1, LAYA's top Edit failed its sandbox dry-run, controller chose the next option. Tests: Kaggle kernel evidence-ladder-laya-tests-run-v1: existing 157 green + 25 laya_integration (tests_kernel_run/TEST_RESULTS.json c02da658...). Audit: F01 F02 F03 F04 F06 F13 addressed; F05 waiver configs/laya_none_abstain_waiver_v1.toml + controller abstain. Step E: BLOCKED by the key system (KeyConfigError: NVIDIA_API_KEY_1..5 not set; .secrets never opened); packets built and validated (E_critic_receipt.json). INTEGRATION PASS NOT DECLARED (step E blocked). PERFORMANCE: NOT TESTED.

Step E (local GLM critic) LIVE: 3/3 packets answered (after 1 retry each), experiments executed in WSL, I1-I8 pass, originals unchanged. See HANDOFF addendum 2. LAYA-INT-1 required path demonstrated for 3 toy examples; performance NOT TESTED.
