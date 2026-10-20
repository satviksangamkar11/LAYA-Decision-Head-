# P5 protocol v2 (amendment of v1, written before any final candidate is generated) — 2026-10-20

Status: FROZEN when committed. Supersedes `P5_PROTOCOL_v1.md` (kept unchanged for audit) ONLY in the lines marked **[v2]**. Any further change = v3 with a written reason,
made before the final run, never after seeing final results.

## Amendments from v1 and their reasons
| Field | v1 | v2 | Reason |
|---|---|---|---|
| Execution environment (Python) | MXC process container (PC) | **Kaggle jail** (chroot + uid 65534 + seccomp; `evidence_ladder/execution/kaggle_port.py`) with the trusted verifier | compute runs on Kaggle (standing rule); the jail passed the port qualification 10/10 and the contract-route qualification 5/5 (`P5_STEP1_REPORT_2026_10_20.md`) |
| Execution isolation | not specified | **shards run one after the other; one jail user at a time** | Step 1 A2: a concurrent jail user's lock wait is charged as candidate time (20/20 false timeouts); sequential use makes the wait zero |
| Benchmark size | rule only | **85 LiveCodeBench + 20 contract tasks**, ids and order as in `configs/p5_final_lock_v1.json` (unchanged) | the v1 rule applied to the smoke-measured rate gives N = 85, the whole remaining pool |
| Contract-cohort grading | hidden tests + reference written separately from the statement | **primary: a NEW private set** written from the statements only by an author without access to the existing hidden cases, reference solutions or contracts, independently validated, hashed before final inference, kept outside git and Kaggle; **secondary: the existing public set** (`results/raw/p5-eval-only-v1`, publicly accessible since commit f6ab9d29) | the existing set is public on the repository; a public set cannot be called a held-out grading set |
| Model, arms, sampling, G, contracts, success criteria | — | **unchanged** | — |

Clarification (no criterion change): the shared generation cache makes raw arm wall time incomparable (a later arm reuses an earlier arm's generations). The D-vs-C
wall-time criterion is computed on the wall-equivalent time (`tools/el_p5_cost_accounting.py`: recorded inference seconds of every generation the arm used + the arm's
own non-inference seconds); raw times are reported alongside.

## Model and settings (all arms)
Qwen2.5-Coder-1.5B-Instruct, pinned revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, non-thinking, sampling T 0.7 / top_p 0.8 / top_k 20 / repetition_penalty 1.1,
max 4,096 new tokens per sample. GLM critic OFF. LAYA learned scoring OFF. One sandbox for every arm of a cohort: **[v2] the Kaggle jail with the trusted verifier, shards
sequential, one jail user at a time** (Java only if WSLC is re-qualified).

## Arms
| Arm | What it is | Budget |
|---|---|---|
| A plain | one sample (seed 1), answer = that sample | 1 sample |
| B multi-sample | fresh samples, answer = first sample passing every public example | generated-token cap G per task |
| C V1 | existing Evidence-Ladder controller, ladder mode, repair as in V1 (critic off) | same cap G |
| D V1.5 | C + trusted contract checks (contract_v1, host registry), model-proposed checks shadow only | same cap G |
B, C and D share their first-round samples (same seeds and prompts): identical requests reuse the exact recorded output; requests that differ are generated.
G = 6,000 generated tokens beyond the first round of 4 samples (the P2 budget).

## Cohorts (reported separately, never pooled)
1. Benchmark: LiveCodeBench v6 (`data/livecodebench/v6/test6.jsonl`, sha256 bb4c364f...), easy + medium only, excluding the 4 adapter-validation problems
   (abc388_a, abc389_a, 3747, 3753) and the smoke problems. **[v2] N = 85, LOCKED in `configs/p5_final_lock_v1.json` (ids sha256 a18cfc1d…), order unchanged.**
   Official comparison rules (adapter `evidence_ladder/execution/lcb_runner.py`). Contract coverage is reported (expected ~0).
2. Contract tasks: 20 newly authored tasks (bank format, solve(data)). Statement + contract written and hashed BEFORE any candidate exists. Contracts are authoritative
   only through the host registry. **[v2] Grading: primary = the new private set (hashes in `configs/p5_final_lock_v2.json`, content outside git and Kaggle);
   secondary = the existing public set. Both are reported; the success criteria are judged on the primary set.**

## Evaluation isolation
Hidden / private tests and reference solutions are never in a prompt, a runtime check, a retrieval store or a training file. Grading happens offline after each arm's
selection is final. Smoke problems are disjoint from final problems. **[v2] The private set's content never enters git, a Kaggle bundle or any generation process;
only its hashes are published before final inference.**

## Measures (task-level, paired across arms)
Independently correct final answers (solve rate); false SUPPORTED (answered but wrong); correct answers wrongly blocked (D vs C); abstention rate; median seconds per task;
generated / prompt / total tokens per task; sandbox seconds; trusted-contract coverage (share of tasks with an authoritative contract).
Uncertainty: exact counts + paired 95% intervals (bootstrap over tasks, 10,000 resamples, seed 1). No significance claim beyond that.

## Success criteria (pre-registered, unchanged from v1)
- D vs C, contract cohort: false SUPPORTED lower; correct answers wrongly blocked = 0 observed; verified solves not lower.
- D vs C, benchmark cohort: verified solves not lower; median wall-time overhead <= 10%.
- "Beats plain Qwen" (C or D vs A): claimed only if the paired interval of the solve-rate difference excludes 0; cost reported alongside.
- Infrastructure failures stay separate from candidate failures (0 unresolved).
- An inconclusive interval does not block freezing the interfaces; it limits the claim.
Freeze V1.5 if all D-vs-C criteria hold. If not: identify contract coverage / validation error / infrastructure / generation capability and change only that part.
A negative or inconclusive result is reported as such, whatever the infrastructure checks show.

## Disclosures carried into the final report
The Q03 diagnostic-contract effect (a non-authoritative contract can withhold SUPPORTED and change D's choice; Step 2); the Kaggle smoke's parallel-shard jail
contention; the compute estimate is a linear extrapolation (contract-cohort rate borrowed from 3 bank smoke tasks); the public secondary grading set.

## Smoke step
Done (v1). The final run needs the user's GO.
