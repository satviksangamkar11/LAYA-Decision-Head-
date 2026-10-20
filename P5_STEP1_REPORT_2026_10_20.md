# P5 Step 1 — evidence report (2026-10-20)

Scope: Step 1(a) jail flakiness, 1(b) C/D replay parity, 1(c) cost accounting. No model generation, no training, no LAYA scoring, no architecture change, no final-cohort task touched. Full P5: still NO-GO.

## 1(a) Jail flakiness: the P1-21 timeout did NOT reproduce

Exact candidate (`449e72bb…`, `re.fullmatch`), exact path (Kaggle jail port + `ProviderExecutionBackend(trusted=True)`, one backend calibrated once, same bundle commit `f808bc61` as the smoke), Kaggle CPU kernel, 4 vCPU. Limits unchanged (10 s, launch overhead calibrated at 0.50 s).

| Condition (load avg 1 min) | Shape | n | Example 4 pass | Timeouts | Per-input s p50 / max | Candidate-total s max |
|---|---|---|---|---|---|---|
| isolated (0.6) | example 4 alone | 20 | 20 | 0 | 0.51 / 0.56 | 0.06 |
| isolated | full public batch | 20 | 20 | 0 | 0.51 / 0.56 | 0.12 |
| 4 busy-loop processes (3.2–3.3) | example 4 alone | 20 | 20 | 0 | 0.89 / 1.09 | 0.59 |
| 4 busy-loop processes | full public batch | 20 | 20 | 0 | 0.92 / 1.12 | 2.09 |

The full batch's examples 1–3 fail every time, which is the candidate's real behaviour. Example 4 passed 80 of 80 runs.

**Finding:** full CPU saturation slows each jail launch about 1.8× but stays far below both timeout limits: a single input is killed only after 10.5 s, and the candidate total limit is 10 s. CPU contention alone does not explain the smoke's timeout.

**What the smoke did not record:** in the smoke, D's timeout cost about 20 s of extra wall time (D 56.7 s against C 36.5 s), so the stall was ≥10 s. The smoke saved no verifier time accounting, so we cannot tell which limit fired.

**Not tested:**
- two concurrent jail users (the smoke ran the LCB and bank shards in parallel, both launching jails);
- a GPU process sharing the host;
- memory or disk pressure.

**Decision rule outcome:** neither tested cause explains it, so P5 stays NO-GO and the execution layer stays under investigation.

Raw data: `results/raw/p5-jail-flake-v1/FLAKE.json` (all 80 rows, per-input seconds, accounting, load average). Kernel: `kaggle/el_p5_jail_flake_v1.py`.

## 1(b) C/D replay parity: PASS

`tools/test_p5_cd_parity.py`, task P1-21.
- **Generations:** arm C's Kaggle-recorded candidates, served by (purpose, seed) through the runner's own CacheGen.
- **Execution:** one shared memo. C ran live on MXC with the trusted verifier, and D received the recorded result for every identical call.
- **Config:** D used the smoke's bank-cohort contract config.

| Check | Result |
|---|---|
| Generation requests (purpose, seed, prompt sha256) | identical (6 each, both repair prompts included) |
| Decision-relevant journal entries (candidates, evidence, labels, ranking, routing, rounds, decision, stop) | 67 = 67, no divergence |
| Final | both BLOCKED, stop `budget`, 2 rounds, identical decision record |
| Execution calls D made that C never made | 0 |

**Cause of the Kaggle divergence:** C observed example 4 for candidate 449e as `pass` and D observed `timeout`. That changed the ranking, so D repaired a different candidate. Given identical observations, the controller is deterministic and D behaves exactly like C.

**Limit of this test:** D had an authoritative contract loaded (`derived:decl_v1`), but no P1-21 candidate passed every public example, so the contract gate never fired. This test covers parity when the contract never fires. It does not cover a contract check that runs and passes.

**Identifier-only defect, recorded:** `environment_revision` is hashed from `backend_info()`, which contains running call counters. It therefore differs by arm and by run order. It does not influence decisions, and it is not changed during the frozen comparison.

Report: `results/raw/p5-cd-parity-P1-21-v3.json` and its two journal files.

## 1(c) Cost accounting

`tools/el_p5_cost_accounting.py` → `results/raw/p5-kaggle-smoke-v2/COST_ACCOUNTING.json`. Three separate measures:
- **Logical tokens:** what each arm used, the budget measure.
- **Actual inference work:** cache misses the arm itself caused.
- **Execution / controller time:** wall time minus inference time.

| Kaggle smoke | Arm | Logical gen tokens | Own (actual) gen tokens | Non-inference s | Wall-equivalent s |
|---|---|---|---|---|---|
| LCB (6) | B | 32,334 | 27,499 | 199 | 1,374 |
| | C | 9,612 | 4,777 | 64 | 518 |
| | D | 9,612 | 0 (all reused) | 63 | 518 |
| Bank (3) | B | 4,267 | 1,050 | 24 | 69 |
| | C | 4,363 | 1,146 | 46 | 272 |
| | D | 4,491 | 718 | 69 | 301 |

**Wall-equivalent** = recorded inference seconds of every generation the arm requested + the arm's own non-inference seconds. It makes D comparable with C despite the shared cache; the protocol's 10% wall criterion is kept, not removed.

**Median D-vs-C wall overhead:** LCB 0.0% (D has no contract there, as expected); bank 5.8%, from 3 tasks and one D timeout. These are smoke numbers only, not the criterion's verdict.

**Comparability caveat:** C and D re-request round 1 through the cache and B does not. The wall-equivalent is therefore only comparable between C and D. B's measure is its own wall time.

## Infrastructure changes proposed (not made)

1. **Log the trusted verifier's time accounting and per-input observations for every execution call into the run directory.** This is logging only; no limit or retry changes. Without it, the next timeout cannot be attributed.
2. **Run shards one after another even when Kaggle gives 2 GPUs,** so only one process uses the jail at a time. The kernel today runs them in parallel when it gets ≥2 GPUs. This is a precaution: contention between two jail users is the untested remaining suspect.
3. **Next test, if approved:** Test A with a second process running jail executions concurrently, to see whether that reproduces the timeout.

## Follow-up (approved 2026-10-20): A2 with a concurrent jail user — REPRODUCED, cause found

`kaggle/el_p5_jail_flake_v2.py` → `results/raw/p5-jail-flake-v2/FLAKE2.json`. Same candidate, path and limits as A; 20 trials per condition; Kaggle CPU, 4 vCPU; launch overhead calibrated 0.60 s.

| Condition (load avg 1 min) | Per-input s p50 / max | Example 4 alone: timeouts | Full public batch: timeouts (outcome) |
|---|---|---|---|
| isolated (0.7) | 0.61 / 0.82 | 0 / 20 | 0 / 20 |
| peer runs trivial jail programs in a loop (1.1) | 0.94 / 1.32 | 0 / 20 | 0 / 20 |
| peer runs a busy-loop program the LCB runner kills at 6 s (1.0) | **7.01 / 7.25** | 0 / 20 | **20 / 20 (`candidate_total_timeout`)** |

**Cause:** `evidence_ladder/execution/kaggle_jail.py:183` runs one jailed program at a time through a global `flock`, by design, because the jail has no PID namespace and reaps leftover processes. A second jail user's long run makes every launch wait for the lock, up to the other run's 6–7 s kill. That wait falls inside the trusted verifier's timed window, so it is charged as candidate time. Two waits exceed the 10 s candidate total, which is the smoke's failure on example index 3. Load average stays near 1: the cause is lock waiting, not CPU.

**Fix:** workload isolation, with no limit, timeout or retry changed. The P5 kernel now always runs shards one after another, so only one process uses the jail at a time. The isolated condition above is that arrangement: 0 timeouts in 40 runs. Within a shard, execution calls are already sequential.

**Effect on the Kaggle smoke:** the smoke ran its two shards in parallel, so any jail observation in it may include lock waiting. Journals with timeout evidence are P1-21 D (2), 3722 C (4) and 3722 D (4). The 3722 timeouts are LCB-runner kills that cannot be attributed after the fact. The smoke's Kaggle answers are therefore not usable for any timing-dependent comparison; they are a plumbing result only.

**Logging (approved):** `tools/el_p5_run.py` now writes `EXEC.jsonl`, one row per execution call, with the verifier's time accounting and per-input observations. A future timeout can be attributed to the limit that fired. It changes no result; fake-model checks on MXC passed for the bank and LCB cohorts.

**Not changed:** the timing semantics (lock wait counted as candidate time) stay as they are. Sequential use makes the wait zero. Excluding the wait from the candidate's time would be a verifier change and needs its own decision.

## Step 2 (GO 2026-10-20): contract-route qualification on excluded tasks — PASS on MXC and on Kaggle

**Setup:**
- **Tasks:** `configs/p5_qual_tasks_v1`, four synthetic tasks Q01–Q04 with their own hash-pinned registry. None of the 20 final contract tasks is used.
- **Model:** scripted (`tools/el_p5_contract_qual.py`), alternating a public-passing hard-coded program with a correct program.
- **Config:** P5 `V1_CFG`. D's contract config matches the contract cohort's except for the registry.
- **Kaggle run:** kernel `el-p5-contract-qual-v1`; the jail port qualification passed first; bundle commit 118ad777.
- **Results:** `results/raw/p5-contract-qual-kaggle-v1/qual/QUAL.json`, plus 10 journals and `EXEC.jsonl`.

| Case | Route | C | D | MXC | Kaggle jail |
|---|---|---|---|---|---|
| Q01, both candidates offered | registry, author contract, authoritative | correct | authoritative FAIL on the hard-code, PASS on correct; answers correct | pass | pass |
| Q01, hard-code only | same | **hard-code (SUPPORTED)** | authoritative FAIL; **BLOCKED, no answer** | pass | pass |
| Q02 | registry entry marked diagnostic | correct | diagnostic notes only; correct | pass | pass |
| Q03 | model-sourced contract, registry says authoritative | hard-code | host downgraded the contract to diagnostic; correct | pass | pass |
| Q04 | no registry entry; parser `decl_v1` | correct | the hard-code crashed on derived valid inputs: authoritative FAIL; correct | pass | pass |

**Across all cases:**
- C never has a contract check in its registry or contract evidence in its journal.
- Kaggle: 108 execution calls (27 `run_public`, 9 `run_inputs`, 72 `run_experiment`), 0 timeouts. The contract `run_inputs` path works through the jail port.

**Finding: a diagnostic contract can change D's choice.**
- On Q03, D chose the correct program while C chose the hard-code.
- Cause: the frozen V1 rule `contract_lib_v1.row_label`. A row is SUPPORTED only if no applicable evidence fails, whatever the source. A diagnostic failure therefore never makes a row FAILED, but it withholds SUPPORTED.
- My first-draft condition, "on the diagnostic routes D chooses exactly what C chooses", failed on Q03. The applied condition is "no FAILED row that C doesn't have"; the strict result stays recorded as `strict_same_choice_as_C`.
- The final cohorts should not be affected: the contract cohort has 20 authoritative contracts and LCB has none.
- Not changed: whether a non-authoritative check may influence choices at all is a design decision for the user.

**Finding: line endings vs. pinned hashes.**
- `core.autocrlf=true`. C01–C20 were written with CRLF and pinned on the CRLF bytes, while git stores LF. All 20 committed blobs differ from their pins; the Windows working copy and the Kaggle bundle, which is packed from the working copy, match.
- A Linux checkout from GitHub would make the registry refuse those contracts. It fails closed, but the failure would stop a run.
- Q01–Q03 are the reverse case: pinned on LF, and they may become CRLF on a later Windows checkout.
- Proposed fix, not made: a `.gitattributes` rule `-text` for `configs/p5_*_tasks_v1/**` and `results/raw/p5-eval-only-v1/**`, with the files re-added so git stores the pinned bytes exactly. The pins stay unchanged.
