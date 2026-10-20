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
