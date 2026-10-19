# P2 experiment contract: fresh resampling vs failure-aware resampling vs counterexample-only repair (2026-10-20)

Status: PREPARED, NOT APPROVED. No GPU generation has run. Harness: `tools/el_p2_generation_compare.py` (sha256 in the Phase A record).
The Evidence-Ladder pipeline is unchanged. V1.5 / V2.1 are not approved. This is an experiment harness, not a controller change.

## Question
On tasks where the stored first round produced no candidate passing the public examples, does a feedback strategy produce more correct candidates than fresh resampling
at the same generated-token budget, with the same untouched model?

## Fixed design
| Item | Value |
|---|---|
| Model | Qwen2.5-Coder-1.5B-Instruct, profile `qwen2.5-coder-1.5b-instruct` (pinned revision), local RTX 3050, lowest thinking (non-thinking), unchanged sampling (T 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1), max 4,096 new tokens per sample (failure if hit) |
| Tasks (13) | every P1 task with no public-passing stored sample: P1-20, P1-23, P1-26, P1-31, P1-36, P1-37, P1-40 (`el-bank-local-20261018T2106`); J1-32, J1-33, J1-34, J1-35 (`el-bank-java5hard-wslc-20261018T213058`); J1-20, J1-23 (`el-bank-java20-wslc-20261018T211547`) |
| Round 1 (shared, not regenerated) | the STORED fresh samples (seeds 1-4, templates qwen_sample_v2 x2 + v2b x2) from each run's `gen_calls.json`, with their recorded tokens and seconds. J1-33: seed 4 hit the 4,096 ceiling with no code, so it has 3 candidates; its 4,096 tokens stay in the round-1 cost. Round 1 is identical for all arms and is reported separately |
| Arms | A fresh: unchanged sample prompt, new seeds. B failure-aware: same prompt + note of failure class and error text per failing public example of this arm's earlier candidates (no code, no observed values). C counterexample-only: same prompt + one failing public example (input, expected, observed or error) from this arm's best earlier candidate (no code). No arm ever shows earlier code |
| Rounds | 2 samples per round; templates continue the frozen v2/v2b cycle; seeds per round r = 100+2r, 101+2r, identical across arms |
| Budget | 6,000 generated tokens per arm per task beyond round 1 (equal for all arms). A new round starts only while the arm has used less than the budget; a started round finishes and its overshoot is reported. Prompt tokens, total tokens and wall-clock are reported too |
| Stop and selection (runtime, all arms) | stop at the first candidate passing every public example; that candidate is the arm's answer (SUPPORTED); otherwise BLOCKED. No learned scorer, no critic, LAYA correctness scoring disabled |
| Runtime oracle | public examples only, in the original sandboxes (python: MXC process container; java: MXC WSLC) through trusted verifier v2 |
| Evaluation oracle (after the run, offline) | every candidate graded on the development grader with the reference engine (`tools/el_p1_reference_grade.py`-style tool over this run's `CANDIDATES.jsonl`). The grader is never imported by the harness; its results never enter generation, feedback or selection. Development evidence only (cases public in Git) |

## Measurements (per arm, paired by task)
1. Availability: tasks with at least one candidate (any round) passing the development grader. 2. Selected correct: the arm's answer passes the development grader.
3. False SUPPORTED: answer passes public examples but fails the grader. 4. Public solves. 5. Generated / prompt / total tokens and wall seconds (per call, from `CALLS.jsonl`).
6. Solve rate against cumulative generated tokens (curves), so arms are also compared at matched token levels below the cap. 7. Near-duplicate rate (exact duplicates by hash are skipped and counted).
Reporting: per-task table plus totals; with 13 tasks the result is descriptive (no significance claim); "no difference" is a valid outcome.

## Cost and run rules
Estimate (unmeasured): at most 3 arms x 13 tasks x 6,000 tokens / about 22 tokens/s = about 3 hours of GPU, less if arms solve early.
Local preflight unchanged (RAM >= 3.0 GB, VRAM >= 3.5 GB, commit >= 4.5 GB; `wsl --shutdown` and closing heavy apps before the start); one GPU process; one WSLC user.
Output create-only, written row by row (`CALLS.jsonl`, `CANDIDATES.jsonl`, `RESULTS.jsonl`, `CONTRACT.json`); a stopped run keeps every finished row.

## Checked without the GPU (scripted model, `--fake`)
`results/raw/p2-dryrun-fake-*` (python P1-20, java J1-33): all three arms run; stored round 1 reused (J1-33: 3 candidates, 4,839 round-1 tokens incl. the 4,096-token sample);
budget and overshoot accounting; duplicate skipping; notes B and C rendered from public examples only; sandbox calls on MXC and WSLC.

## Not covered by P2
Tasks whose public-passing answer is wrong on the grader (P1-21 and six others): stopping at the first public pass ends every arm identically there; that is P3 (runtime oracle).
