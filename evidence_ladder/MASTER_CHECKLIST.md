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
