# P5 final preflight audit (2026-10-20) — no model generation

Scope (user GO): close the grading-data exposure question, lock N by the frozen rule, publish the compute estimate, audit the final Kaggle bundle.
Nothing was changed in the model, budget, arms, contract semantics or grading criteria. The full P5 run is still NO-GO until the user decides.

## 1. Grading-data exposure

| Material | In git / public branch? | Read by |
|---|---|---|
| Contract cohort hidden cases, reference solutions, brute forces, generator (`results/raw/p5-eval-only-v1/`) | **yes**, public on `origin/v1.5-p3-verification` since commit f6ab9d29 (2026-10-20 16:47 IST) | only `tools/el_p5_grade.py` (offline grading); no runtime, generation or packing code reads them (grep of `tools/`, `evidence_ladder/`, `kaggle/`) |
| LiveCodeBench v6 private tests (`data/livecodebench/v6/test6.jsonl`) | **no** (only `PROVENANCE.json` is tracked); the benchmark itself is public on Hugging Face | the grader only; Kaggle bundles carry `private_test_cases = ""` |
| Kaggle bundles (smoke 113550Z, Step 2 133039Z, final 165103Z) | — | none contains an eval-only file (checked) |

**What can and cannot be claimed:**
- The contract cohort's hidden cases are **not a secret holdout**: anyone could read them. They were held out from every prompt, runtime check and Kaggle bundle.
- Order of writing, from git: the contracts were committed before the hidden cases (8ec5586b, then f6ab9d29), and the hidden cases were written from statements only.
- The model is a frozen public checkpoint (weights 2024-09) and cannot have trained on them.
- The remaining exposure risk is people or agents tuning the system after reading them. Since f6ab9d29 no arm, budget, contract or selection rule has changed. Changes since then: the runner logging, sequential shards and the qualification fixtures. None of these reads them.
- Git history cannot be cleaned without a history rewrite, which is not allowed here, and the branch is already public.

**Options (user decision):**
- (a) Keep the cases and disclose them as "publicly accessible, held out from generation" (recommended; costs nothing).
- (b) Author a new private hidden set for C01–C20, written blind from the statements by a separate agent and never committed, then use it for the final grading. This costs one authoring pass. The current set would then be reported as a secondary, public set.

## 2. N locked by the frozen rule

Rule (`P5_PROTOCOL_v1`, cohort 1): the largest N ≤ 91 whose GPU time at the smoke-measured rate is ≤ 10 h, minimum 40, taken as the first N of the seeded order.

- **Rate used:** the slower of the two smoke runs, 284.4 s per task on Kaggle T4 with parallel shards. The local RTX 3050 / MXC run measured 227.8 s.
- **Result:** after the 6 smoke and 4 adapter-validation exclusions, 85 problems remain. 85 × 284.4 s = 6.7 h ≤ 10 h, so **N = 85**, the whole remaining pool.
- **Lock:** `configs/p5_final_lock_v1.json`, with ids sha256 `a18cfc1d…`, the LCB file hash, the contract-cohort hashes, the protocol hash, the model revision, G = 6,000 and arms A–D. Script: `tools/lock_p5_final_v1.py` (create-only).
- **Power:** with arm A solving 0–1 of 6 in the smoke, a paired interval over 85 problems can only detect large differences. The claim is limited by the interval, as the protocol says.

## 3. Compute estimate (unmeasured; linear extrapolation of the smoke means)

| Part | Tasks | Rate (s/task) | Hours |
|---|---|---|---|
| LiveCodeBench | 85 | 284.4 (Kaggle T4) | 6.72 |
| Contract cohort | 20 | 88.1 (3 bank smoke tasks on Kaggle; the contract cohort itself was never run) | 0.49 |
| **Total** | 105 | | **7.21** |

**Assumptions behind the estimate:**
- The Kaggle smoke rate was measured while the two shards competed for the jail lock, so jail waits are included. The final run is sequential, so the real time is probably lower. On the local RTX 3050 / MXC rates the total would be about 5.6 h.
- One Kaggle session (12 h limit) is enough at this estimate. The final kernel runs 6 shards one after another (5 × 17 LCB + 1 × 20 contract), each in its own run folder, so a lost session loses at most the running shard.

## 4. Final bundle audit: PASS

Bundle: `results/raw/el-p5-kaggle-final-bundle-20261020T165103Z` (157 files, commit b3c43f28). It was packed locally and is **not uploaded**. Audit (`tools/audit_p5_bundle.py`) result is in `AUDIT.json`:

| Check | Result |
|---|---|
| manifest hashes | pass |
| no eval-only file (path or name) | pass |
| no line of the reference / brute-force / generator code | pass. 10 coarse hits, all shared idioms (`from collections import Counter`, …) that occur in 3+ bundle files or in files committed before the eval material |
| no hidden input together with its expected value (604 pairs scanned) | pass |
| LCB rows = the 85 locked ids, in order | pass |
| LCB private tests empty | pass |
| contract tasks carry public tests only (20) | pass |
| no P5 grader script | pass |
| no key-shaped string | pass |
| plan covers exactly the lock (85 LCB + 20 contract) | pass |

**Audit history:**
- **First pack (164942Z), failed on two checks.** The coarse first version of the audit flagged the generic idioms and trivial inputs; the tightened checks are stated in the script. It also found one real coincidence: qualification task Q01's public example `{"nums": [2]} → [2]` equals one hidden case of the contract cohort. Both audit outputs are kept in that folder.
- **Fix:** the final bundle now excludes the Step 2 qualification fixtures, which the final run does not use.
- **Not P5 material:** the bundle carries `results/raw/codex-pilot1-…/grader/` sandbox code from the older code set. It is not P5 evaluation material.

## 5. Open decisions before the GPU spend

1. **Sandbox vs. the frozen protocol.** `P5_PROTOCOL_v1` names the **MXC process container** as the Python sandbox, but the final run was planned on the Kaggle jail. That jail passed the port qualification 10/10 and the contract qualification 5/5. Two ways forward:
   - write `P5_PROTOCOL_v2` with the reason, before the run, if Kaggle is used;
   - or run the final locally on MXC (about 5.6 h on the RTX 3050). This needs an explicit exception to the standing compute-on-Kaggle rule.
2. **Exposure handling:** (a) disclose, or (b) a new private hidden set (section 1).
3. **Disclosures that stay:** the Q03 diagnostic-contract effect (D's choice may differ from C's under a non-authoritative contract); the Kaggle smoke's parallel-shard jail contention; and that the contract-cohort rate is extrapolated from bank tasks.
