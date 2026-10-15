# S4. Execution, approval and isolation policy (Evidence-Ladder v1, P0 + R3, 2026-10-16)

## 1. The controller

- Plain Python. It holds permissions, caps, the registry, the evidence journal and every decision.
- Model output (Qwen, GLM), tool text and retrieved content are data only (I3).
- Per action:
  1. propose (Qwen; LAYA ranks if the decision type is validated);
  2. sandbox execute;
  3. evidence records;
  4. decision record (S2);
  5. commit or the next step.
- Per task:
  1. intake and routing (S1, and the router in the architecture document, section R1.2);
  2. the loop above;
  3. commit;
  4. final check on the real end state;
  5. report.

## 2. Approval by risk (R2.3)

| risk | example | approval rule |
|---|---|---|
| `sandboxed` | running a candidate in an isolated process | always allowed inside caps (it is how evidence is gathered) |
| `reversible_real` | writing the chosen solution file; a git commit in a worktree | every hard/functional row VERIFIED, or the user is shown the coverage table and accepts it |
| `irreversible` | deleting data, sending, deploying, paying | stronger coverage AND explicit user confirmation, whatever the checks say; stage 1 has none |

## 3. Isolation of untrusted code

### 3.1 Two kernels, and isolation inside each

| | generation kernel(s) | grading kernel |
|---|---|---|
| mounted data | problems with public examples only; **no hidden-test dataset attached** | candidates (frozen files with hashes) + encrypted hidden tests |
| runs untrusted code? | yes (public examples, Qwen tests, critic experiments) | yes (hidden tests) |
| internet | off | off |
| output | candidates, evidence records, budgets | per-candidate pass/fail and counts only; written to a separate output never mounted by a generation kernel during a run |

### 3.2 The candidate jail (both kernels)

- Each execution is a child process that drops to an unprivileged uid created at kernel start (`setuid` / `setgid` in `preexec_fn`).
- It gets an empty environment, a fresh temporary working directory owned by that uid, and a closed stdin except for the test input pipe.
- Resource limits: `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_NPROC`, `RLIMIT_FSIZE`, `RLIMIT_NOFILE`. Wall-clock kill of the whole process group.
- Hidden tests (grading kernel only):
  - stored encrypted at rest in the dataset;
  - decrypted only into the grader process's memory, never written to disk;
  - inputs streamed to the candidate's stdin one test at a time; expected outputs never leave grader memory.
  - The key is a Kaggle Secret read by the grader process. The jailed uid cannot read the grader's `/proc/<pid>/environ` or memory (different uid).
- **Feasibility check FC-1 (pilot stage 0):** the Kaggle runtime allows creating a uid and dropping privileges. **Unverified.**
  - If it fails, grading moves to the approved local WSL bubblewrap sandbox (exception approved 2026-10-13), with the same rules.
  - Kaggle is never used without a jail.
- The grader returns no hidden-test details to any generation loop. Hidden results are read only by the evaluation stage (I8).

### 3.3 Red-team tests (pilot stage 0; every one must fail to escape)

| id | candidate tries to | expected |
|---|---|---|
| S4-T1 | open `/kaggle/input/**` hidden-test files | ciphertext only, or permission denied |
| S4-T2 | read `/proc/<grader>/environ` or `/proc/<grader>/mem` | permission denied |
| S4-T3 | open a socket | fails (internet off; jail) |
| S4-T4 | fork bomb, 10 GB allocation, infinite loop, giant file | the matching limit fires; the kernel survives; status not `pass` |
| S4-T5 | write outside its temp dir, or modify another candidate's files | denied |
| S4-T6 | print a fake "PASS" banner | ignored (status comes from the comparison, not from candidate text) |

## 4. Transactions and rollback

- **Stage 1:** a candidate is an immutable file plus its sha256. A repair writes a new candidate whose `parent` is the old one. Nothing is ever edited in place.
- **Stage 2 (repository tasks):**
  - every candidate lives in its own git worktree created from a recorded base commit (`environment_revision`);
  - commit = apply the approved patch to a clean worktree at the same base, re-run the trusted checks there, then fast-forward;
  - any failure discards the worktree, so the base is untouched.
- **S4-T7:** a commit that fails midway leaves the base revision byte-identical (hash check).

## 5. Final check

After commit:
1. recompute the artifact hash;
2. re-run the trusted checks on the real end state;
3. check that the diff scope is inside the allowed files.

Any mismatch blocks the result and triggers rollback.

## 6. GLM-5.3-flash critic calls

- **Model:** `z-ai/glm-5.3-flash` (decision D5, final).
- **When it is called:** only when two or more behaviour clusters survive, or when a hard/functional requirement has no trusted check (stage 1: the first case).
- **Request** (`critic_request`): quoted requirements, candidate code, concrete observed outputs. Never LAYA's pick, Qwen's reasoning, hidden tests or labels.
- **Response** (`critic_response`): must parse against the schema.
  - The controller checks that the experiment is allowed: the kind is allowed for the stage, and the payload size and format are valid.
  - Then it runs the experiment in the jail.
  - An invalid response is logged and counted as a wasted call, never retried with the same request id.
- **Call limits** (user's measured ceilings, unchanged):
  - every `.secrets/nvidia_api_key_<n>.txt` (n >= 2);
  - <= 5 in flight per key, <= 39 sends per key per minute;
  - a 429 cools that key, honouring `Retry-After` when present (60 s default), and the request is re-queued.
- **Logging:** every send is logged; a completed request id is never resent; long calls are streamed (non-streamed calls are cut at 300 s).
- **Spend:**
  - Rs 0. NVIDIA trial terms limit credits and availability.
  - Before the pilot, the user confirms the remaining credits on the NVIDIA account page.
  - Any 402 / 403-for-payment / quota-exceeded response stops all critic calls at once (stop cause `api_unavailable`). Nothing is ever paid.
- **Reasoning setting:** the request parameter for maximum reasoning on glm-5.3-flash is checked in pilot stage 0 (**FC-2**) and recorded. If it is rejected, stop and ask; never silently downgrade.
- **Where calls run: decision D7 (open).**
  - (a) Staged loop. The Kaggle generation kernel stops at the critic step, writes the packets, and the PC sends them with the existing tooling. The answers go back as a new dataset version for the next kernel stage. No key leaves the PC. Slower.
  - (b) Calls from inside Kaggle with the keys stored as Kaggle Secrets. This needs the user's approval, because keys leave the PC. Candidate code must stay jailed and off the network while the kernel has internet on.
  - Recommendation: (a) for the pilot.

## 7. The eight invariants (unchanged) and where they are tested

| invariant | tests |
|---|---|
| I1 label honesty | S2-T3, S2-T5 |
| I2 no reorder across trusted evidence | S5-T3 |
| I3 no self-authorisation | S4-T6, S1-T2 |
| I4 unknown is not pass | S2-T4, S4-T4 |
| I5 bounded | S5-T4 |
| I6 observe before commit | S4-T7 |
| I7 frozen scorers and registry | S3-T3 |
| I8 sealed evaluation | S3-T4, S4-T1, S4-T2 |
