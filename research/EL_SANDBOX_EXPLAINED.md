# The execution sandbox, explained (read from the code, 2026-10-18)

Tags: **[V]** verified by reading the cited line, **[U]** unverified / inferred. Nothing was run to produce this file.

## 1. Where the sandbox is called, and why

```
 proposed action (Read / Edit / Bash)
        |
        v
  CHECK  policy (risk table, no execution)
        |--- Edit? --> copy workspace to a scratch dir, apply the edit there,
        |              run the PROTECTED TEST in the sandbox  ("dry run")     agentic.py:405-412
        v
  COMMIT gate (authorize) --> chosen action is applied
        |--- Bash action: runs IN the sandbox with persist=True               agentic.py:500
        v
  OBSERVE the REAL end state: protected test command in the sandbox           agentic.py:566-568 (_observe)
        |   (also stage 2 INVESTIGATE baseline run, and stage 8 finalize re-run)
        v
  evidence rows (R1 test passes, R2 protected files unchanged) --> label / decision
        ^
        |--- optional CRITIC experiment: a command from the critic runs in a
             scratch copy of the end state, diagnostic evidence only            agentic.py:~648-652
```

Why: the only trusted evidence that an edit or end state is correct is a real test run (invariant: observe before commit, `invariants.py:167`). Every path goes through two methods on the controller, `_run` (`agentic.py:353-355`, counts a tool run, calls `ctx.sandbox.run`) and `_run_checked` (`agentic.py:357-364`, turns a raising sandbox into ERROR evidence, never a pass). The function-level ladder uses a second, separate execution engine (`evidence_ladder/execution/engine.py`, Python/Java jail) and the harness `tools/el_arch1_l2_local.py` `WslFunctionExec`, which wraps the same `WslSandbox`. [V for the calls above; engine.py internals not re-read for this note]

## 2. How it isolates today

Call chain: `AgenticRun._run` -> `WslSandbox.run` (`wsl_sandbox.py:38-47`) writes a JSON request into a scratch dir, runs `wsl -d Ubuntu-24.04 -u root -- python3 tools/laya_int1_t6_sandbox.py <request.json>` (`:43`, outer timeout = timeout+180), parses the last stdout line as JSON, raises `RuntimeError` on a non-zero exit or empty output (`:45-46`).

Inside WSL, `tools/laya_int1_t6_sandbox.py` `main` (`:24-47`):
1. `shutil.copytree(workspace, run/ws)` into `/var/lib/fable-runs/<uuid>` (`:28`; `fable_wsl_runner.py:28`). The candidate never touches the real workspace unless persisted.
2. Records SHA-256 of every protected file BEFORE (`:30`) and a tree hash (`:31`, `tree()` `:15-21`).
3. `chown -R replay:replay` (`:32`), then runs `W.sandbox_argv(...)` (`:33`).
4. Re-hashes the protected files AFTER (`:34`); `protected_unchanged = before == after` (`:37`). `killed` = exit code 137 or -9 (`:35`). Output is the last 6000 chars of stdout/stderr (`:36`).
5. Persist: only if `persist` AND protected unchanged AND not killed (`:38`): deletes files missing from the copy, copies the copy's files back, skips `__pycache__` (`:39-46`). Then deletes the run dir (`:47`).

`sandbox_argv` (`fable_wsl_runner.py:50-64`):
- `systemd-run --scope` cgroup limits: MemoryMax 2G, no swap, TasksMax 256, CPUQuota 400%, user `replay` (`:51-52`).
- `timeout -s KILL <n>` hard timeout (`:53`).
- `bwrap --unshare-all --die-with-parent --new-session`, uid/gid 999 (`:54`): new PID, mount, IPC, UTS, user, **net** namespaces, so **no network**.
- Mounts: `/usr`, `/etc` and `/opt/toolchains` read-only; `/work` = the workspace copy (read-write); `/tmp` a tmpfs; fresh `/proc`, `/dev`; root remounted read-only (`:55-61`). The Windows drives (`/mnt/*`) and `/root` are not mounted. The runner's own probe list checks that (`network_tcp`, `dns`, `windows_drive_hidden`, `secrets_unreachable`, `root_home_hidden`, `:89-95`). [V that the probes exist; V that they were last run: not checked here]
- Env: `--clearenv`, then only the fixed `ENV` dict (PATH, HOME=/tmp/home, locale, PYTHONHASHSEED=0, toolchain paths, `CI=1`) (`:31-35`, `:62-63`). No host secrets are inherited.
- Protected tests: each is re-bound **read-only** over `/work/<file>` (`:59-60`), and independently hash-checked before/after (above). Both must hold for `protected_unchanged`.
- The command runs as `/bin/bash -c <cmd>` (`:64`).

Result dict (the whole contract the controller reads): `exit_code, killed, stdout, stderr, protected_unchanged, pre_tree_sha256, post_tree_sha256, persisted` (`laya_int1_t6_sandbox.py:36-37`).

## 3. What it does NOT protect against (code-verifiable)

- **Windows-only, WSL-only.** Needs `wsl`, the `Ubuntu-24.04` distro, systemd, bubblewrap, user `replay`, `/opt/toolchains`, the `D:\local model` path baked in (`laya_int1_t6_sandbox.py:21`, `fable_wsl_runner.py:27`). `available()` (`wsl_sandbox.py:23-30`) just runs `true`. Nothing portable. [V]
- **The helper itself runs as root in WSL.** The jailed command does not, but the post-processing does. Persist copies files with `shutil.copyfile` (`laya_int1_t6_sandbox.py:45`) and `tree()` reads with `read_bytes` (`:20`); both follow symlinks, and `rglob` has no symlink check. A command (with persist=True, i.e. a committed Bash action) that creates a symlink inside `/work` could make the root-owned helper read a file outside the jail into the hash, and copy its contents into the real workspace. This is **from code reading, not exercised**; the file would land in the workspace, not in the model's output channel. [U: not tested]
- **Persisted changes are trusted.** After a clean run, everything the command left in `/work` (new files, edited non-protected files, any file named like a hook or config) is copied into the real workspace. Only `protected` files are checked. Git hooks / `.git` are not special-cased: if a workspace contained `.git/hooks`, a persisted command could write there. The toy tasks have no `.git`. [V for the copy; U for real-repo use]
- **Not a VM.** bubblewrap shares the WSL kernel; a kernel exploit escapes. Resource limits are cgroup only (no disk quota; stdout is captured fully in memory by `capture_output=True` before truncation, `:33`). [V]
- **`/etc` is readable inside** (`:56`); the host's WSL `/etc` could contain things you do not want shown to a model. [V mount; U contents]
- **Timeouts:** the hard kill is `timeout -s KILL`; descendants are reaped by the PID namespace (`--unshare-all`). A child that ignores nothing can survive only inside the dead namespace. [U: not tested]
- **Only checks what the policy hash says.** The environment identity in the journal is the literal string `wsl-bwrap-systemd` plus the task (`agentic.py:158`). Replacing the sandbox without keeping or versioning that label changes every journal hash (the controller does not detect which sandbox actually ran). [V]
- The `_safe` path check for Edit/Read is lexical in the controller (`agentic.py` `_safe`); not re-read here. [U]

## 4. The sandbox PORT (what a replacement must implement)

```python
class Sandbox:
    def run(self, workspace, cmd, persist=False, timeout=60) -> dict
```
Callers: `agentic.py:355` (`ctx.sandbox.run(workspace, cmd, persist=persist)`; `timeout` defaulted), `replay.py:86-98` (`RecordingSandbox.run(workspace, cmd, persist, timeout)`), `tools/el_arch1_l2_local.py:117`, plus tools that build `WslSandbox(scratch, protected)` directly (`el_arch1_acceptance.py:297,480`, `el_arch1_variants.py:230`, `laya_int1_e2e.py:103`, `pathb_stored.py:84`). Constructor shape used everywhere: `(scratch_dir, protected_names)`. The acceptance driver already has a factory hook: `sandbox_factory(scratch, protected)` (`el_arch1_acceptance.py:283,297`), which is the natural swap point.

What the controller assumes of the returned dict:
- keys `exit_code` (int), `killed` (bool), `stdout`, `stderr` (str), `protected_unchanged` (bool); `_observe` also relies on a PASS only when `exit_code == 0` (`agentic.py:571-572`, dry run: `exit_code == 0 and protected_unchanged`, `:410`).
- `pre_tree_sha256` / `post_tree_sha256` (receipts; the helper's `tree()` hashing rule: sorted relative path + file digest).
- `persisted` (bool).
- Infrastructure failure = **raise** `RuntimeError / OSError / ValueError / TimeoutExpired` (`agentic.py:361`); anything else propagates as a bug. A backend must never return a fake pass on failure.
- The run must work on a **copy** (workspace unchanged unless `persist` and not killed and protected unchanged), protected files read-only AND hash-verified, no network, scrubbed env.

Replay: `RecordingSandbox` stores per call `{cmd, persist, result (deep copy), files_after (workspace tree if persist)}` or `{cmd, persist, raised}` (`replay.py:86-100`). `RecordedSandbox` (offline replay) serves them in call order and raises `ReplayDivergence` if `cmd` or `persist` differs (`:103-117`). So a new backend needs no replay changes if it returns the same dict, but **offline replay never calls the real backend**; "live" replay (`sandbox=<obj>`) re-executes and compares hashes (`replay.py:8`), which only matches if the new backend produces byte-identical tree hashes and outputs (stdout truncation length 6000 is part of the recorded result).

## 5. If execution were simply removed

- **Evidence lost:** `protected_tests_v1` (R1) and `protected_files_unchanged_v1` (R2) in `_observe`; `action_dry_run_v1` for every Edit; critic experiments; the INVESTIGATE baseline run; final verify. Without them nothing can reach VERIFIED/authorized; stage-1 labels stay UNVERIFIED/SUPPORTED. [V by the call sites above]
- **Invariants/rows that depend on it:** observe-before-commit (I6, `invariants.py:167`), trusted evidence over model preference (I2), tool errors never pass (I4), protected-file integrity (R2).
- **Tests that need it:** every `*_wsl` test (e.g. `phaseF2_tests.py:110,245`; phaseB `t_b7_b8_full_run_on_the_real_wsl_sandbox_and_replay_wsl`), the acceptance driver (`el_arch1_acceptance.py:316` refuses to run without a real sandbox), conformance row `RT-WSL`, final-report rows T01/T25. [V from grep; the exact set was not exhaustively counted]
- Offline replay and logic-only tests keep working because they use recorded or scripted sandboxes.

## 6. Effort estimate for a portable backend (no code)

Rough, **estimate (unmeasured)**: 2-4 working days for a Docker/Podman or plain-subprocess backend that passes the existing contract on Linux/macOS/Windows; more if it must match the WSL tree hashes byte for byte in live replay.

Files to touch/add:
1. NEW `evidence_ladder/controller/sandbox_port.py`: the `run(...)` protocol + result schema check (reject missing keys, never default to pass).
2. NEW backend module (e.g. `docker_sandbox.py`): copy workspace, `--network none`, read-only protected mounts, env scrub, timeout/kill, hash before/after, persist rules identical to `laya_int1_t6_sandbox.py:38-47` (add the symlink guard).
3. `evidence_ladder/controller/wsl_sandbox.py`: keep as one implementation; `available()` becomes part of backend selection.
4. `tools/el_arch1_acceptance.py` (`sandbox_factory` default, `:297`, `:316` WSL check), `tools/el_arch1_variants.py:230`, `tools/laya_int1_e2e.py:103`, `controller/pathb_stored.py:84`, `tools/el_arch1_pathb_replay.py:57`, `tools/el_arch1_l2_local.py:108,311`: select the backend through the factory.
5. `agentic.py:158`: the env label `wsl-bwrap-systemd` should be derived from the backend (versioned), otherwise journals from different backends look identical.
6. Tests: parity tests (same commands, same dict) for each backend, plus the isolation probes of `fable_wsl_runner.py:89-95` run against the new backend.
7. Docs: the architecture notes that say "WSL".

Toolchains (JDK, Go, .NET, Rust under `/opt/toolchains`) only matter for non-Python tasks; the three toy tasks need only `python3`. [U for other languages]

## Checklist
- DONE: call sites, isolation mechanics with line refs, limits, port, replay recording, dependency list, effort.
- NOT DONE / UNVERIFIED: running anything (symlink read/copy-back is code-reading only); `evidence_ladder/execution/engine.py` internals; `_safe` implementation; exhaustive test dependency count; whether the isolation probes pass today.
