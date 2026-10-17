# MXC replacement of the LAYA sandbox: final implementation report (2026-10-18)

**Verdict: MXC_REPLACEMENT_PARTIAL.** A real MXC execution provider exists, passes every security, evidence and replay test that this host can run, and is faster than WSL in steady state. Three required gates are not proven (listed under "Open blockers"), so the production default stays `wsl` and MXC is opt-in (`LAYA_SANDBOX_BACKEND=mxc`).

Branch `mxc-execution-provider-2026-10-18`, created from `el-arch1-final-closure-2026-10-18`. Nothing merged to main, nothing pushed. Host: Windows build 26300 (Home), Ryzen 7 7840HS, 16 logical CPUs, 15.3 GB RAM, Node 24.13.1, WSL 3.0.3 / Ubuntu 24.04.

## 1. What was inspected (Phase 1)
Every production sandbox path ends in one object method, `sandbox.run(workspace, cmd, persist=False, timeout=60)`:
- `AgenticRun._run` / `_run_checked` (`agentic.py`): Edit dry-run, committed Bash action (persist=True), observe of the real end state (also INVESTIGATE baseline and finalize), critic experiment. Infrastructure errors (`RuntimeError/OSError/ValueError/TimeoutExpired`) become ERROR evidence, never a pass.
- `replay.RecordingSandbox/RecordedSandbox`: record `{cmd, persist, result, files_after}`; live replay re-executes.
- `WslSandbox` -> `tools/laya_int1_t6_sandbox.py` -> `tools/fable_wsl_runner.py` (copy, hash, systemd cgroup + bubblewrap, `--clearenv`, protected files re-bound read-only, `timeout -s KILL`).
- The Python/Java function engine `evidence_ladder/execution/{backend,engine}.py` is a different interface (public examples, bwrap, Kaggle jail) and is NOT part of the controller sandbox; it was not migrated. Its test suite fails 19/21 on Windows before and after this work (Linux/bwrap-bound).
- Journal identity: `agentic.py` hashed the literal `wsl-bwrap-systemd` into the environment id; the label is now taken from the sandbox object with that string as default, so every WSL run hashes exactly as before (tests prove it).
Details: `research/EL_SANDBOX_EXPLAINED.md`, `MXC_MIGRATION.md`.

## 2. MXC facts (Phase 2; docs fetched from github.com/microsoft/mxc, SDK installed from npm)
- `@microsoft/mxc-sdk` 1.0.1 (MIT, Node >= 24, `koffi`, `semver`, `node-pty`; no install scripts; installed with `--ignore-scripts`, 169 MB). Rust, .NET and Node SDKs; V1 API: `run`, `spawn`, container lifecycle, `getPlatformSupport`, `getAvailableBackends`, `probe`.
- Windows host check (real calls): `isSupported: true`, `availableMethods: [processcontainer, isolation_session, hyperlight]`, tier `base-container`, deny-paths and enumerate-paths supported. `wslc` is listed but not available: it needs a cached container image (`alpine:latest` not cached; filling the cache pulls from a registry, which needs approval, and alpine has no Python). `windows_sandbox` and `hyperlight` are "not available in the v1.0 high-level SDK". `isolation_session` demands network allow-all, so it cannot satisfy "no network".
- ProcessContainer behaviour measured: network denied when no `network` policy is declared; host environment inherited unless `inheritDefaultEnvironment:false` (the default INHERITS it, including secrets-shaped variables); `LOCALAPPDATA` is mandatory in an explicit environment; MSYS bash and PowerShell die with 0xC0000142 inside the container (so no POSIX userland); `cmd.exe` and python run; timeout kills the whole process tree; no memory/CPU/process-count fields in the V1 policy; native stdio streaming needs Node >= 24.21 (here 24.13.1, so the output cap is applied after the run).
- Linux (bubblewrap, needs `bwrap` on PATH >= 0.5.0 and unprivileged user namespaces; NOT bundled), macOS Seatbelt, LXC, Hyperlight: read in the docs only. **NOT_TESTED** (wrong OS).
- Kaggle: the earlier FC-1 probe (`results/raw/evidence-ladder-stage0-probe-20261016T120934Z/PROBE.json`) records no `bwrap`, `unshare` = EPERM under seccomp, apt offline. MXC's Linux backend therefore cannot be expected to work there, and the existing `kaggle_jail.py` (chroot + uid drop + seccomp + rlimits) stays the Kaggle backend. `kaggle/mxc_probe_v1.py` (+ `kaggle/mxc_probe_v1/`) re-checks this on the current image; **not run by me**. The user's own one-cell check (root, `bwrap`, `unshare`, `/dev/kvm`, user namespace) is consistent with it and cheaper; either result decides it.

## 3. Implementation (Phases 3, 4, 7)
Exact files:
- NEW `evidence_ladder/controller/sandbox_provider.py` (port, factory, `validate_result`, CLI `python -m evidence_ladder.controller.sandbox_provider`), `mxc_sandbox.py` (provider), `win_job.py` (Job Object limits), `mxc_dialect/{__init__,posix_lite}.py`.
- NEW `tools/mxc_bridge/{bridge.mjs,package.json,package-lock.json,.gitignore,probe.mjs,exp/*}`, `tools/mxc_benchmark.py`, `tools/mxc_suite_runner.py`.
- NEW `evidence_ladder/controller/tests/phaseMXC_tests.py`, `MXC_MIGRATION.md`, this report, `kaggle/mxc_probe_v1*`, `research/EL_SANDBOX_EXPLAINED.md`.
- EDITED (minimal): `agentic.py` (env label, 1 line), `replay.py` (label carried through record/replay), `tools/el_arch1_acceptance.py`, `tools/el_arch1_variants.py`, `tools/el_arch1_l2_local.py`, `evidence_ladder/controller/pathb_stored.py` (use the factory). `wsl_sandbox.py`, `laya_int1_t6_sandbox.py`, `fable_wsl_runner.py` untouched.

Design: Python copies the workspace into `scratch/runs/<uuid>/ws`, hashes it with the WSL tree algorithm, makes protected copies OS-read-only and lists them read-only in the MXC policy, runs through the bridge (containment `processcontainer`, no network block, `inheritDefaultEnvironment:false`, minimal fixed environment, read-write only the run copy and a private home, read-only the Python install), re-hashes, persists only under the WSL rules (protected unchanged, not killed) plus: refuse links, refuse capped output, path-escape guard. The run directory is always deleted. Output paths are rewritten to `/work` and `/tmp` and CRLF to LF so output does not depend on the random run directory. Resource limits come from a Job Object around the bridge: memory 2048 MB, 256 processes, hard CPU cap 4 cores (the WSL profile). Fail-closed rules: no `auto`; `SandboxUnavailable`, `UnsupportedPolicy` (invalid or unenforceable limits, non-relative protected names), `UnsupportedCommand`, `SandboxIntegrityError`, malformed bridge answers -> `RuntimeError`; a dead bridge is restarted only if the request was not delivered. MXC audit mode (`--audit`, learning mode) is never used.
Command dialect (the real limitation): one plain `python3|python <args>` runs directly; the subset of `posix_lite.py` (`echo cat ls pwd mkdir touch rm true false`, `; && ||`, `> >>`) runs inside the container; anything else (pipes, `$VAR`, `grep`, `git`, ...) raises `UnsupportedCommand` where WSL would run it.

## 4. Tests (Phase 5; all run on this host)
| Suite | WSL baseline (before) | WSL after | MXC |
|---|---|---|---|
| controller phaseA / B / D1 / F1 / F1b / F2 / run_tests | 16 / 36 / 21 / 18 / 10 / 45 / 25, all pass | identical | n/a (these build `WslSandbox` directly) |
| critic_batch run_tests, stepE | 28, 12 pass | identical | n/a |
| models run_tests, backbone, lowthink | 30, 14, 3 pass | identical | n/a |
| orchestrator phaseC, phaseF3, run_tests | 25, 30, 25 pass | identical | n/a |
| execution/tests/run_tests.py (Linux bwrap engine) | 2/21 | 2/21 | not migrated |
| 75 fault-injection variants (25 x ex1_slug/ex2_clamp/ex3_wordcount, real controller + sandbox) | 75/75 OK, 51.1 s | 75/75 OK, 50.7 s | **75/75 OK, 35.5 s** |
| NEW `phaseMXC_tests.py` (real MXC, no mock, none skipped) | - | - | **21/21** |
Raw: `results/raw/mxc-baseline-wsl-20261017T224021Z`, `mxc-final-wsl-20261017T231801Z`, `mxc-final-mxc-20261017T231801Z`, `mxc-final-phaseMXC-20261018`.
What `phaseMXC_tests.py` proves on real MXC: result contract and determinism (identical result dict twice, same tree hash as the WSL helper); **output parity with WSL on 11 commands x 3 tasks** (exit codes, hashes, stdout/stderr byte-identical except interpreter-specific traceback text); host secrets and environment invisible (with a native negative control that DOES see them); project, `.secrets` and other user files denied; network (TCP external, loopback, DNS) blocked; protected files cannot be modified, appended, deleted, renamed over, or chmod-ed (7 attacks) while an unprotected write succeeds; `..`, absolute and sibling paths blocked and nothing created outside; input links refused, workload-made link not persisted; timeout kills the tree and a killed run is never persisted; detached background children die with the run and cannot write after persist; memory bomb, fork bomb and CPU cap enforced; failed commands, unknown programs (127), unsupported syntax and malformed bridge answers never look like success; persist/rollback semantics; per-run isolation and cleanup; persistent bridge keeps no state; **the controller reaches VERIFIED on all three tasks with a scorer that prefers the wrong edit (trusted evidence overrides it), invariants I1-I8 checked, offline and live replay identical to the recorded journal hash**. Pure-logic tests cover the dialect parser, factory and the label/hash compatibility.

## 5. Latency (Phase 6; `results/raw/mxc-benchmark-20261017T230215Z`, ABAB interleaved, n = 30 per cell, seconds, no model time)
"mxc_persistent" = bridge kept alive (now the default of `make_sandbox`); "mxc_oneshot" = node started per call.

| case (n=30 each, seconds) | backend | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| cold_first_call | wsl | 0.245 | 0.235 | 0.324 | 0.388 |
| cold_first_call | mxc_oneshot | 1.049 | 1.021 | 1.184 | 1.208 |
| cold_first_call | mxc_persistent | 1.045 | 0.994 | 1.280 | 1.313 |
| warm_echo | wsl | 0.244 | 0.246 | 0.283 | 0.402 |
| warm_echo | mxc_oneshot | 0.351 | 0.339 | 0.428 | 0.527 |
| warm_echo | mxc_persistent | 0.169 | 0.160 | 0.225 | 0.266 |
| python_test | wsl | 0.263 | 0.254 | 0.306 | 0.417 |
| python_test | mxc_oneshot | 0.291 | 0.277 | 0.355 | 0.420 |
| python_test | mxc_persistent | 0.123 | 0.114 | 0.174 | 0.225 |
| python_test_pass | wsl | 0.243 | 0.233 | 0.295 | 0.346 |
| python_test_pass | mxc_oneshot | 0.294 | 0.288 | 0.356 | 0.391 |
| python_test_pass | mxc_persistent | 0.124 | 0.116 | 0.164 | 0.176 |
| workspace_persist | wsl | 0.689 | 0.667 | 0.825 | 0.975 |
| workspace_persist | mxc_oneshot | 0.442 | 0.428 | 0.535 | 0.542 |
| workspace_persist | mxc_persistent | 0.266 | 0.270 | 0.318 | 0.322 |
| sequential_cands | wsl | 2.865 | 2.731 | 3.594 | 4.769 |
| sequential_cands | mxc_oneshot | 3.201 | 3.101 | 4.071 | 4.279 |
| sequential_cands | mxc_persistent | 1.280 | 1.208 | 1.633 | 2.237 |
| full_task | wsl | 9.343 | 1.322 | 1.788 | 241.103 |
| full_task | mxc_oneshot | 1.604 | 1.623 | 1.821 | 1.904 |
| full_task | mxc_persistent | 0.846 | 0.837 | 0.994 | 1.056 |
| cold_first_call, after capability cache (2nd run) | wsl | 0.232 | 0.228 | 0.278 | 0.289 |
| cold_first_call, after capability cache (2nd run) | mxc_oneshot | 0.379 | 0.350 | 0.403 | 1.046 |
| cold_first_call, after capability cache (2nd run) | mxc_persistent | 0.344 | 0.336 | 0.398 | 0.418 |

Breakdown of an MXC python test call (mean ms, persistent): prepare 7, MXC call 113, verify 2-5; one-shot: MXC call 257 (node start ~145). Workspace persist adds ~54 ms. Full controller task: 5 sandbox calls per task on every backend; sandbox time per task 0.79 s (persistent) vs 1.44 s (one-shot). Memory: peak RSS of the MXC process tree 79-137 MB; WSL VM (`vmmemWSL`) stayed at 0.73-0.80 GB for the whole run (it is shared with other WSL use, so not attributable per call). CPU seconds are sampled approximations. The single 241 s WSL sample in `full_task` is one stall in 30 (p50/p95 unaffected); I did not investigate it.
Reading: steady-state p95 improves 20-61 % with the persistent bridge in every warm case; **a cold first call (new object in a process) is slower**: p50 0.34 s vs 0.23 s after the capability probe is cached (1.0 s before the cache); one-shot mode is slower than WSL at p95 for python tests and sequential candidates (+16 %, +13 %). Gate "no material p95 regression" holds for the persistent default except the once-per-process cold start.

## 6. Open blockers (why PARTIAL)
1. **POSIX userland**: ProcessContainer cannot run bash/PowerShell; arbitrary Bash actions (pipes, grep, git, ...) raise `UnsupportedCommand`. The 75 variants and the three tasks pass because their commands are inside the subset, not because bash parity exists. Fix options: WSLC with an approved Python image (needs a registry pull, `wxc-exec.exe --setup-wslc --image <image>`), or accept the dialect for python-test workloads. **Needs your decision.**
2. **NOT_TESTED**: WSLC, Linux bubblewrap, macOS Seatbelt, LXC, Hyperlight (host/OS or approval), and the Kaggle probe (prepared, not run).
3. **Output cap is post-hoc** on Node 24.13.1 (a flood is bounded only by the 2 GB Job limit, which kills the bridge -> fail closed). Node >= 24.21 enables streaming and the bridge switches automatically; not tested here.
4. Tier is `base-container` (not the stronger AppContainer/PSEC tiers); UDP, raw sockets and IPv6 egress were not tested individually.
5. Existing `*_wsl` tests are not parametrised by backend; MXC coverage comes from `phaseMXC_tests.py` and the variants run.
Not changed: frozen checkpoints, datasets, prompts, adapters, scoring logic, evidence requirements, `sarvam-jev/`, `reference/`.
