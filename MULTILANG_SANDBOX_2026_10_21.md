# Multi-language trusted sandbox (2026-10-21)

Branch `multilang-sandbox-2026-10-21` (cut from `v1.5-p3-verification`, the branch the main checkout is on; the pre-created worktree branch was based on a 2026-10-03 commit that has no `evidence_ladder/`).
Results: `results/raw/multilang-sandbox-20261021T142000Z/` (`QUAL.json` = merged table, `QUAL_<backend>.json` = per-check detail). These are development qualification results on one machine, not an evaluation.

## What was added (all additive; python and java code paths untouched)

| file | change |
|---|---|
| `evidence_ladder/execution/trusted_native.py` (new) | trusted verifier path for C, C++, JavaScript, Go, Rust: stdin -> stdout programs, one fresh process per input, host-side verdict |
| `evidence_ladder/execution/trusted_verify.py` | 3 lines in `TrustedVerifier.run`: dispatch to `trusted_native` for those 5 languages only |
| `evidence_ladder/execution/toolchains.py` | new field `trusted_runner="native"` for the 5 languages (`runner` stays `None`, so the frozen test `phaseF3d` and untrusted backends still say `NOT_TESTED_NO_RUNNER`); `-lm` for C; rust links with `-fuse-ld=bfd` (rust-lld needs `/proc/self/exe`, absent in the Kaggle chroot) |
| `evidence_ladder/execution/provider_engine.py` | probe + `_engine_run` accept the 5 languages only when `trusted=True` and the probe passed inside that backend; NTSTATUS exit codes shown in hex in probe reasons |
| `evidence_ladder/execution/lcb_runner.py` | `LcbRunner.run_case/grade(..., language=)`: LiveCodeBench stdin tests in the 5 languages (compile once per program, LCB comparison on the host); default `language="python"` is the old code path; functional (`class Solution`) tests stay Python-only |
| `evidence_ladder/execution/kaggle_jail.py` | `build_rootfs(toolchains=(...))` explicit list; launcher keeps the execute bit of host-staged files, resolves tools through the jail PATH, reads `<rootfs>/.toolchains.json`. Default rootfs and python behaviour unchanged (checked below) |
| `evidence_ladder/controller/mxc_sandbox.py` | opt-in `tool_dirs=` (extra read-only dirs on PATH); default `()` gives the identical policy |
| `tools/el_multilang_qual.py`, `tools/el_multilang_merge.py`, `tools/test_trusted_native.py` (new) | qualification suite (14 checks per language), merger, logic test with a scripted fake port |

## Design (same guarantees as the python verifier)

* Compiled languages: one compile call (`persist=True`; only the file `prog` is read back), then **one fresh sandbox call per input** in a fresh workspace holding a host-written launcher, `in.txt` (the input) and the program. The launcher (`python3 launch.py <rlimits> -- ./prog`) sets rlimits, redirects stdin from `in.txt` and `execvp`s the candidate, so the exit status the port reports is the candidate's own. JavaScript has no compile step (`node cand.js`). A python launcher is used because it is the one command every backend can start (the MXC dialect and the Kaggle rootfs have no `sh`).
* Verdict is host side. There are no result frames: the candidate's stdout is just its answer, so forged frames only make it differ from the expected text. Non-zero exit or kill overrides any output (claim-then-crash fails). Expected values never enter `run()` / sandbox calls (a spy in the suite scans every staged file and command for a canary).
* Compile error / wrong answer / runtime error are candidate outcomes. Exit 127 from the compiler, a compile that persisted nothing, or a port exception raise `TrustedVerifyInfraError`.
* Stated limit: the supervisor sees only what the port captures (WSL helper and MXC keep the last 6000 characters; the jail 1 MiB). An output that reaches that limit is `invalid_output`, never compared (a program with large stdout cannot be judged on WSL/MXC).
* Not tested (not claimed): fork-bomb / memory-bomb behaviour per language (left to the backend gates), a hanging compile (the 60 s compile timeout exists but has no test), JavaScript `solve(json)` function mode (only stdin programs are supported for the 5 languages).

## Result table (QUAL.json)

| language | WSL reference engine | MXC processcontainer | WSLC | Kaggle jail |
|---|---|---|---|---|
| C (gcc 13.3) | qualified | unavailable | unavailable | qualified on a WSL root **proxy only** |
| C++ (g++ 13.3) | qualified | unavailable | unavailable | proxy only |
| JavaScript (node 26.3) | qualified | unavailable | unavailable | proxy only |
| Go (1.26.4) | qualified | unavailable | unavailable | proxy only |
| Rust (1.96.0) | qualified | unavailable | unavailable | proxy only |

"qualified" = all 14 checks passed: honest pass, wrong answer fails, forged frames never pass, claim-then-crash never passes, hang killed, compile/syntax error is a candidate outcome, fresh process per input, socket denied (a host listener saw 0 connections), host canary files unreadable, expected never staged, large output never silently truncated, LCB stdin honest pass / wrong fails / whitespace rules.
A negative control (`bare`, no isolation, run in WSL) fails exactly `socket_denied` and `host_paths_unreadable` for every language, so those two checks do detect a leak. The listener positive control (an unsandboxed connection reaches it) held on every backend run. On the WSL backend the network check is weaker than elsewhere: the listener is on Windows and the sandbox has an empty network namespace, so it shows denial but cannot show that the listener would otherwise be reachable.

Why the other cells are unavailable (probe results, nothing fell back to another backend):
* **MXC**: the Windows host has no gcc, g++, go or rustc. node exists but **cannot start inside the process container**: `0xC0000142` (STATUS_DLL_INIT_FAILED), also when `node.exe` is copied into the workspace or the policy additionally grants the Windows directory, and also when launched directly rather than through python. So JavaScript on MXC stays unavailable unless the container policy changes (not attempted: it would loosen the sandbox).
* **WSLC**: the pinned image tar is present, but the backend's real-run probe fails in this session (`WslcListSessionImages failed: HRESULT 0x80070057`). Not investigated further; the image's toolchains were therefore not probed.
* **Kaggle jail**: no Kaggle kernel was started. The cell is the jail launcher (chroot, uid 65534, NO_NEW_PRIVS, seccomp, flock) run as root inside WSL Ubuntu 24.04 with a rootfs built by `build_rootfs(toolchains=("c","cpp","javascript","go","rust"))` (1.1 GB). Kaggle's image (Ubuntu 22.04, other gcc/node versions, likely no go or standalone rust) is **untested**.

### What a Kaggle kernel must provide (untested)
c/cpp: gcc/g++ with cc1/cc1plus, `as`, `ld`, libc dev files and `/usr/include`; javascript: a `node` binary; go: a GOROOT (or `EL_KJAIL_GOROOT`), builds use `CGO_ENABLED=0`, cold std compile took 2.6 s on the proxy; rust: a standalone sysroot (a distribution rustc whose sysroot is `/usr` is refused) plus `cc`, so "c" is staged with it. `build_rootfs` raises for a missing tool (no silent partial rootfs). Then run `python tools/el_multilang_qual.py <out> --backend kaggle-jail` inside the kernel before any use.

## Regression (python and java behaviour)

| check | before | after |
|---|---|---|
| `tools/test_v2min.py` | rc 0, all_ok | rc 0, all_ok |
| `tools/el_lcb_adapter_check.py` (real MXC, 4 LCB tasks) | 13/13 as expected | 13/13 as expected |
| `tools/run_p4b_regression.sh` | see below | see below |
| `tools/el_kaggle_port_qual.py` (python, default rootfs without toolchains, WSL proxy) | not run before | all 10 checks pass after the jail edits |
| `tools/test_trusted_native.py` | n/a | ok |

P4B note: in this worktree 7 of 15 suites exit non-zero **before any change** (phaseB, phaseD1, phaseC, phaseF3, phaseF3b, phaseF3c, phaseMXC). The causes seen are missing untracked files (`decision_head/fable_pool`, a frozen-thresholds hash, `.secrets` paths), not this work. The after run is compared per suite below.

Per-suite comparison, before vs after (failing test names compared, not just exit codes): all 15 suites have the same exit codes and the same failing test names (phaseB 1, phaseD1 2, phaseC 4, phaseF3 3, phaseF3b 9, phaseF3c 6, phaseMXC 1; every other suite 0 failures). The pre-existing failures should be re-checked in the main checkout, which has the untracked files this worktree lacks.
`evidence_ladder/orchestrator/tests/phaseF3d_tests.py` (the registry test, not in the P4B list) fails 2 tests in this worktree: the java20 manifest test (missing untracked config) and the real WSLC probe (`HRESULT 0x80070057`, the same backend error as in the table); its registry / probe tests pass with the new field. I did not run it before the change.

Reproduce (WSL proxy for the Kaggle jail; run from the worktree root): `bash tools/el_multilang_kj_build.sh` then `bash tools/el_multilang_kj_run.sh <out dir> kaggle-jail`; Windows backends: `python tools/el_multilang_qual.py <out dir> --backend wsl|mxc|wslc`; merge with `python tools/el_multilang_merge.py <out dir>`.
