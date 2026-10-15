# A3 "execution-grounded adaptive loop": full architecture with recheck points (2026-10-15)

Status: **DESIGN ONLY. Nothing built or run.** Chosen in `ARCHITECTURE_DESIGN_2026_10_15.md` (Part 3) from 252,000 raw combinations.
Every number below that is not cited to a file is a **proposal** (to be frozen in `configs/thresholds_exec_loop_v1.toml` before any run) or an
**estimate (unmeasured)**.

Goal (user, 2026-10-15): the best possible answers from a small local Qwen model, built WITH the adapter (Qwen states -> LAYA) + the pretrained
LAYA head + loop engineering; no strong model at runtime; compute on Kaggle.

---

## 0. One-paragraph summary

The small model (Qwen3-4B-Thinking-2507, frozen) writes a few attempts at a coding problem. Every attempt is executed: first by parser and lint,
then by the problem's public tests, then by tests Qwen writes itself (CodeT agreement). The adapter + pretrained LAYA head look at Qwen's own
hidden states for the problem and each attempt and do one job with one interface: **point at the attempt most likely to be correct, or at NONE
("no attempt here is correct")**. Tests decide first; LAYA breaks ties, chooses which failed attempt to repair, and its NONE probability decides
whether to spend more samples. The loop stops on a named terminal state. Every executed attempt becomes a real pass/fail label that retrains the
adapter offline. A linear probe, DeepConf, majority vote and a random-head adapter compete for LAYA's slot at every point, and LAYA keeps a slot only
where it wins.

---

## 1. Constraints this design honours (source)

| # | constraint | source | where honoured |
|---|---|---|---|
| K1 | adapter + pretrained LAYA head + loop engineering must be in the system | user 2026-10-15 | M2, M8, M9, M10 |
| K2 | no strong model at runtime; one offline use allowed | user 2026-10-15 | only Qwen at runtime; no teacher anywhere in A3 |
| K3 | Qwen is never trained; no encoder trained | CLAUDE.md 2026-10-12 | Qwen frozen; only the 8 adapter tensors (and, as one arm, the head) train |
| K4 | all compute on Kaggle; PC only for code, docs, packing, CLI | CLAUDE.md 2026-10-13 | every stage is a Kaggle kernel stage |
| K5 | NONE / abstain option in every decision; never decide from one candidate without NONE | CLAUDE.md decision-layer rules | NONE is always in the pool |
| K6 | learned score never overrides hard checks or real test results | CLAUDE.md evidence order | ranking is lexicographic, tests first (M8) |
| K7 | thresholds frozen before results; create-only outputs; row-level files for every number | CLAUDE.md quality gates | section 9, RC-01, RC-63 |
| K8 | linear probe is a mandatory baseline | CLAUDE.md 2026-10-11 | M13 |
| K9 | same device class and dtype for train and runtime states | CLAUDE.md frozen decisions | RC-11, RC-71 |
| K10 | live data never changes deployed weights; goes to a versioned offline corpus | CLAUDE.md "Never" | M11, M12 |
| K11 | uploads and > 1 GB downloads need a yes; spend Rs 0 | CLAUDE.md ask-first | section 12 |

**Open tension (needs your word):** memory note 2026-10-15 "loop engineering, not generation: loop around the frozen Fable adapter system on the
SAME dataset; no new candidates" was for the Fable-pool loop experiment. A3 has Qwen generate its own attempts at runtime (that is how a small
model answers). I read the note as not applying here; confirm (section 12, D5).

---

## 2. Components

Each block: purpose · inputs -> outputs · implementation (reuse first) · parameters · failure modes · recheck points (RC ids, defined in section 8).

### M0. Problem intake
- Purpose: one normalised record per problem; separates what the runtime may see from what only evaluation may see.
- In: raw problem from the chosen set. Out: `problem.json` {problem_id, group_id, statement, signature/entry point, public_tests,
  release_date, source}; `hidden_tests` written to a **sealed** file only the evaluation module opens.
- Implementation: new `tools/exec_loop_prep_v1.py` (local; reading and packing only, allowed on PC).
- Failure modes: hidden tests leaking into the prompt; duplicates across splits; contaminated problems (seen in Qwen training).
- RC-02, RC-03, RC-04, RC-05.

### M1. Qwen runtime: generator + state capture (one model, one process per GPU)
- Purpose: produce attempts and the hidden states the adapter reads.
- Model: Qwen3-4B-Thinking-2507, revision `768f209d9e…` (`models/qwen3-4b-thinking-2507`), fp16 on Kaggle T4 (no_gqa SDPA patch, as before).
- Generation: thinking mode, temperature / top-p / top-k as the Qwen3 report's thinking setting (0.6 / 0.95 / 20) unless the pre-registered
  diversity check (RC-31) says otherwise; `max_new_tokens` = the knee measured on DEV only (RC-30). Token log-probs kept for DeepConf.
- Code extraction: final fenced code block after `</think>`; extraction failure = attempt with status `NO_CODE` (kept, counted, never "pass").
- State capture (reuse `tools/fable_capture.py`, protocol `fable_capture_v1`, unchanged numerics):
  - prefix = neutral header + problem statement + query cue; **hq** = residual at depths 18/24/30 of the last prefix token (3 x 2560, fp16);
  - each attempt's **final code only** (thinking stripped) is an independent suffix after the prefix; **ho** = mean over its own tokens at the same
    depths; NONE option = the fixed text "None of these attempts is correct." captured the same way;
  - limits as before: prefix 24,000 tokens, option 4,096 tokens, 8 options per pool (so <= 7 attempts + NONE per pool; see M8 for > 7).
- Failure modes: thinking traces too long for the time budget; truncated code; capture drift between smoke and full runs.
- RC-10, RC-11, RC-15, RC-16, RC-30, RC-31, RC-32.

### M2. Adapter + pretrained LAYA head (the decision layer)
- Purpose: one scoring interface for every LAYA job: a pointer over {attempt_1..attempt_n, NONE}.
- Architecture (as in `tools/adapter_align_v1.py`): query token = `_embed(hq)` via mix_q/in_q; option tokens = `_embed(ho)` via mix_o/in_o; the
  pretrained context layers (frozen) mix them; out_norm; scorer gives one logit per option; softmax over the pool. No positional encoding, so
  the pool is order-free (RC-14).
- Trainable: 8 adapter tensors (in_q, in_o weights/biases, mix_q, mix_o, out_norm weight/bias). Frozen: the 31 pretrained head tensors (RC-13).
- Outputs used by the loop:
  - `p_i` = P(attempt i is the correct one); `p_none` = P(no attempt in the pool is correct);
  - calibrated with one temperature fitted on DEV folds (RC-43).
- Arms (all evaluated, section 7): A-zero = existing Fable adapters `final-s{0,1,2}.pt` unchanged; A-exec = adapter retrained on execution labels
  (M12, head frozen); A-joint = adapter + head trained; A-rand = adapter retrained with a random-init head (control); 3 seeds each, ensemble =
  mean of probabilities.
- Failure modes: scores track length or style, not correctness (RC-46); miscalibration (RC-43); NONE never chosen or always chosen (RC-45).

### M3. Budget controller (how much to sample)
- Purpose: spend samples where they help (Snell: compute by difficulty).
- Rule (proposal): round 1 draws k1 = 2 attempts. Score pool {a1, a2, NONE}. If an attempt passes all public tests AND `p_none` < tau_stop ->
  go to DECIDE. Else round 2 draws up to k_max_round = 6 more (total <= 8), then rescore. Further attempts come only from repair (M9).
- tau_stop chosen on DEV folds to maximise accuracy at a fixed mean-sample budget (pre-registered target, e.g. mean <= 4 samples).
- Competitor arms in the same slot: fixed k (2, 4, 8); probe on hq (pre-generation P(solve), the 2606.14530 signal); prompt-length baseline;
  DeepConf of round-1 traces.
- Failure: confident wrong stop (p_none low on a wrong attempt that passes weak public tests) -> watched by RC-52/RC-57.

### M4. Sampler
- Draws attempts in parallel batches per GPU; seeds recorded; every attempt gets `attempt_id = problem_id:round:index`.
- **Offline replay trick (fair and cheap):** in the study, draw k_full = 16 attempts per problem once (order fixed by seed); every budget policy is
  replayed by taking the first k of that fixed order. All policies then see identical attempts, and live and replayed runs must agree (RC-58).

### M5. Deterministic filter
- Python parse (`ast.parse`), entry-point present, ruff/pyflakes errors that make the code un-runnable (undefined names), banned imports list
  (network, subprocess) -> status `FILTERED` with reason. Lint style issues are logged only, never used as a reward (E: lint-only reward drove
  short wrong code).
- RC-51.

### M6. Sandbox test runner (inside the Kaggle kernel)
- Each attempt runs in a fresh subprocess, own temp directory, copied test file (tests read-only to the attempt), `resource` limits: CPU time
  per test (proposal 10 s), address space (proposal 2 GB), no child processes beyond 1, wall timeout, process-group kill; kernel internet OFF.
- Statuses: PASS / FAIL(test id, expected, actual, traceback head) / TIMEOUT / MEMORY / CRASH / ERROR(sandbox). **ERROR is never PASS.**
- Three test kinds, never mixed in records: `public` (runtime), `generated` (runtime, by M7), `hidden` (evaluation only, sealed).
- RC-20..RC-25.

### M7. Self-written tests (CodeT dual agreement)
- Qwen (same model, non-thinking or short thinking, pre-registered) writes m = 10 test inputs with expected outputs from the statement only.
- Run every surviving attempt on every generated input. Attempts with identical outputs on all generated inputs form a **behaviour cluster**.
- Cluster score = |cluster| x (number of generated tests the cluster passes). Each attempt also gets an **output signature** (hash of its outputs
  on the generated inputs), used for no-progress detection (M10).
- Generated tests are evidence for ranking only, never evidence of correctness in reports (CLAUDE.md: agent-written tests are not evidence).
- RC-33.

### M8. Ranker (who wins), lexicographic, frozen order
1. public-test pass count (all-pass first);
2. CodeT cluster score;
3. LAYA `p_i` (ensemble, calibrated);
4. DeepConf trace confidence (lowest-group mean log-prob);
5. attempt_id (deterministic last resort; never random, per CLAUDE.md).
- More than 7 attempts: pool in groups of <= 7 + NONE, keep each group's winner, final pool of winners + NONE (order-free head, so grouping is
  only a size limit). `p_none` for M3/M10 = from the final pool.
- Arms in LAYA's slot (step 3): probe, DeepConf, majority over behaviour clusters, A-rand, nothing (ablation). A learned blend (logistic on the
  four signals, fitted on DEV folds) is one extra arm, never the default.
- Hard rule RC-52: no learned score can lift an attempt above one that passes more public tests.

### M9. Repair round
- Trigger: no attempt passes all public tests, round < 3, progress made last round.
- Which attempt to repair (LAYA's second job): highest by (public passes, LAYA `p_i`, DeepConf). Competitors: random failing attempt,
  DeepConf-only, public-passes-only.
- Repair prompt: statement + the chosen attempt's code + the real failing test (input, expected, actual, truncated traceback). No self-critique
  without execution output (C: self-repair without real feedback is weak).
- Repair outputs (r = 2 per round, proposal) join the pool; new ho captured for them; hq reused; everything re-ranked.

### M10. Loop controller (state machine)
```
INTAKE -> ROUND(1): sample k1 -> FILTER -> TEST(public) -> TEST(generated) -> CAPTURE -> SCORE -> CHECK
CHECK:
  DECIDED      if best attempt passes all public tests AND p_none < tau_stop           (or: all budget used and best passes all public tests)
  ROUND(2)     if round == 1 and not DECIDED                                            (sample up to k_max_round more)
  REPAIR       if no attempt passes all public tests and repair_round < 3 and progress
  NO_PROGRESS  if a round adds no new output signature and the best public pass count did not rise
  EXHAUSTED    if repair_round == 3 or sample cap (16) or wall cap per problem reached
  ERROR        if the sandbox fails for this problem
Output per terminal state:
  DECIDED      -> answer = rank-1 attempt, flag VERIFIED_BY_PUBLIC_TESTS
  NO_PROGRESS, EXHAUSTED -> answer = rank-1 attempt, flag UNVERIFIED (never reported as a decision; counted in coverage, not in selective accuracy)
  ERROR        -> no answer, flag ERROR (counts as wrong in forced accuracy)
```
- Progress = a new output signature appeared, or the best public pass count rose.
- Caps (proposal): rounds <= 1 sample round extension + 3 repair rounds; attempts <= 16; wall per problem <= 15 min on T4 (estimate).
- RC-50..RC-58.

### M11. Journal (append-only)
- One JSONL row per attempt and per decision (schemas in section 4), written before the next round starts; shards per 50 problems; resumable
  (never lose done results). Hidden-test results go to a separate sealed file.

### M12. Offline trainer (Tr1: adapter on execution labels)
- Data: DEV problems only. One pool per problem per round = {attempts, NONE}; target = uniform over attempts that pass **hidden** tests (true
  correctness, available for DEV), NONE = 1 if none pass. Loss: listwise cross-entropy with soft targets (the existing objective).
- Splits: DEV in 5 grouped folds (group = problem source/contest); hyper-parameters by inner folds; LR grid {1e-4, 3e-4, 1e-3}, epochs <= 5,
  early stop on inner fold, weight decay 0.01, clip 1.0 (values used in outcome_head_v1; re-registered here).
- Out: adapters per fold and seed (out-of-fold scores for every DEV problem) + one final adapter per seed trained on all DEV for HOLDOUT.
- Overfit check first (32 pools, loss must drop >= 50% in 60 steps, as outcome_head_v1) -> RC-41.

### M13. Competitor harness (every comparison at equal attempts)
| competitor | slot | how |
|---|---|---|
| linear probe | M3, M8, M9 | logistic regression on [hq, ho_i, hq*ho_i] per option, C by grouped CV (recipe `kaggle/head_exp4_gpu.py:probe_scores`) |
| DeepConf | M3, M8, M9 | trace confidence from token log-probs (lowest-group mean) |
| majority vote | M8 | largest behaviour cluster |
| random-head adapter | M8 | A-rand |
| length baseline | M3, M8 | prompt length / code length logistic |
| code-only text | M8 | TF-IDF logistic on attempt code without the problem (shortcut control) |
| no-context | M8 | LAYA with the neutral prefix only (problem removed) |
| oracle | ceiling | any attempt passes hidden tests |

---

## 3. End-to-end data flow (one problem)

```
problem.json --(M1 prefill)--> hq  ------------------------------------------------+
     |                                                                              |
     +--(M4 sample k1)--> attempts --(M5 filter)--> (M6 public tests) --(M7 generated tests, clusters, signatures)
                                                                     |                                     |
                                                       (M1 capture ho per attempt + NONE)                  |
                                                                     v                                     v
                                               (M2 adapter+LAYA: p_i, p_none) ---> (M8 lexicographic rank) ---> (M10 CHECK)
                                                                                                                  |
                     DECIDED / NO_PROGRESS / EXHAUSTED / ERROR  <-----------------------------------------------+
                     ROUND(2): back to M4 ;  REPAIR: M9 -> new attempts -> M5 ...
everything -> M11 journal ; DEV problems' hidden-test labels -> M12 trainer (offline, Kaggle) -> new adapter version (offline only)
```

---

## 4. Data contracts (JSONL, create-only)

- `attempts.jsonl`: problem_id, attempt_id, round, origin (sample | repair:<parent_id>), seed, n_think_tokens, n_code_tokens, code_sha256,
  extract_status, filter_status, public {passed, total, first_fail}, generated {outputs_sha (signature), passed, total}, cluster_id,
  deepconf, p_laya {s0, s1, s2, ens}, p_probe, p_rand, capture {hq_file, ho_index}, wall_s.
- `decisions.jsonl`: problem_id, terminal_state, rounds, n_attempts, chosen_attempt_id, flag, p_none_final, decisive_signal (which rank key
  separated rank 1 from rank 2), arm, thresholds_sha256.
- `hidden_eval.sealed.jsonl` (evaluation only): attempt_id, hidden {passed, total}, status. Read once per evaluation stage, access logged.
- `features/*.pt`: hq per problem, ho per attempt (fp16), with `meta.json` (protocol, revision, sha256 per tensor file).

---

## 5. Splits and data

- Problem set: user choice (section 12, D1). Recommended: LiveCodeBench problems released **after** Qwen3-4B-Thinking-2507's release (2025-07)
  to avoid contamination; fallback MBPP+/HumanEval+ (older, contamination likely, report as such).
- DEV (about 70%): training, threshold fitting, all development, 5 grouped folds. HOLDOUT (about 30%): read once at the end per pre-registered arm.
  Group = contest / source so near-duplicates stay together. Split manifest hashed before any generation (RC-04).
- Labels: hidden tests (true correctness). Public tests are runtime evidence, not labels.

---

## 6. What LAYA does, exactly (three jobs, one interface)

| job | when | LAYA signal | decision it changes | competitor that can take the slot |
|---|---|---|---|---|
| J-stop (budget) | after round 1 | p_none | stop at 2 attempts vs draw more | probe p_none, hq-probe P(solve), DeepConf, fixed k |
| J-pick (tie-break) | every SCORE | p_i | winner among attempts equal on public tests and cluster score | probe, DeepConf, vote, A-rand, nothing |
| J-repair (target) | REPAIR | p_i among failing attempts | which attempt gets the repair prompt | random, DeepConf, public-passes only |

LAYA's value is reported per job as the change in final accuracy when its slot is swapped for each competitor at equal attempts, plus how often
it was the decisive signal (`decisive_signal` field). A job where LAYA does not beat the probe is handed to the probe, and the report says so.

---

## 7. Evaluation

Metrics (all on HOLDOUT once, and out-of-fold on DEV for development):
- pass@1 (first attempt), oracle@k (any attempt passes hidden), selected@k per arm, gap closure = (selected - pass@1) / (oracle - pass@1);
- forced accuracy (every problem answered) and selective accuracy on DECIDED, coverage, **false-accept rate** (DECIDED but hidden fail);
- compute: attempts, generated tokens, executions and wall time per problem; accuracy-vs-attempts curves (budget policies at equal mean attempts);
- per-attempt correctness AUROC and AUPRC, ECE, Brier for p_i and p_none (LAYA arms vs probe vs DeepConf vs length);
- repair: fix rate per round by targeting rule;
- W->R / R->W tables for "stack with LAYA" vs "stack with the slot swapped" (paired, per problem).
Statistics: paired bootstrap clustered by problem group, 2,000 resamples, 95% CI; McNemar on paired correctness. Chance +10 points is a warning
only (CLAUDE.md).

### Gates (pre-registered; values are proposals until frozen)
| gate | question | pass rule |
|---|---|---|
| G-0 headroom | is there anything to select? | oracle@8 - pass@1 >= 5 points on DEV OOF. **Fail = stop selection work, report.** |
| G-A loop | does the full loop beat a single attempt? | forced accuracy (full A3) - pass@1 > 0, CI lower bound > 0, HOLDOUT |
| G-B pick | does LAYA earn the tie-break slot? | stack(LAYA) - stack(probe) >= 0 AND stack(LAYA) - stack(no learned key) CI lower > 0 |
| G-C head | does the pretrained head matter? | A-exec - A-rand CI lower > 0; else report "adapter only, head adds nothing" |
| G-D budget | does J-stop save compute? | same forced accuracy (CI includes 0 or better) at >= 25% fewer mean attempts than fixed k=8, or higher accuracy at equal attempts |
| G-E repair | does J-repair help? | fix rate (LAYA-targeted) - fix rate (random) CI lower > 0 |
| G-S safety | does the loop stay honest? | false-accept rate of A3 <= that of fixed best-of-8 by public tests; 0 sandbox escapes; 0 hidden-test reads at runtime |
| G-Z shortcut | is LAYA reading correctness, not style? | LAYA AUROC > code-only text AUROC and > length AUROC, CI lower > 0 |

---

## 8. Recheck points (all of them)

Format: id · what · how it is checked · pass rule · on failure.

### Phase 0. Before anything runs (local, documents only)
- **RC-00** approvals recorded · user's GO, dataset yes, upload yes quoted in the run's `APPROVALS.md` · all present · do not launch.
- **RC-01** thresholds frozen · `configs/thresholds_exec_loop_v1.toml` written, sha256 recorded in the kernel, kernel aborts on mismatch ·
  hash matches · abort; new versioned file with a written reason, never edit.
- **RC-02** dataset licence, size, hash · licence text saved, sha256 of every file · licence allows research use; size as approved · stop, ask.
- **RC-03** contamination window · every problem's release_date > model release date (if LiveCodeBench) · 100% · drop offending problems, log count.
- **RC-04** split manifest · DEV/HOLDOUT by group, sha256 recorded, no group in both · 0 overlaps · rebuild before any generation.
- **RC-05** hidden tests sealed · runtime modules cannot import the sealed reader (a self-test tries and must fail); prompts scanned for any
  hidden-test string · 0 hits · abort.
- **RC-06** compute plan · Kaggle session limit and quota checked, shards sized to finish within one session · plan fits · re-shard.

### Phase 1. Runtime and capture (Kaggle smoke, about 10 DEV problems)
- **RC-10** model identity · Qwen revision hash, tokenizer hash, dtype fp16, device T4, SDPA no_gqa patch on · all match the manifest · abort.
- **RC-11** capture parity · re-capture 5 stored Fable pools with the same code; min cosine vs stored features · >= 0.99999 (last measured min
  0.9999993) · stop; find the numerics change.
- **RC-12** adapter reproduction · existing adapters `final-s{0,1,2}` on cleaned VAL 46 · 0.7826 / 0.8043 / 0.7609 exactly · stop; the head
  path changed.
- **RC-13** head untouched · sha256 of the 31 pretrained head tensors before and after every training run · unchanged (except arm A-joint,
  whose copy is saved under a new name) · abort run.
- **RC-14** order-free pool · shuffle options 3 times per pool on 20 pools · max |delta p| <= 1e-5 · investigate (a position leak).
- **RC-15** truncation · count attempts over 4,096 tokens and prefixes over 24,000 · reported; > 5% of attempts truncated = warning in report.
- **RC-16** determinism · same seed twice on 3 problems gives identical attempts and features · identical · fix seeding before scaling.

### Phase 2. Sandbox (smoke, before any model code runs at scale)
- **RC-20** network off · an attempt that opens a socket must fail · fails · do not run.
- **RC-21** limits work · infinite loop -> TIMEOUT, 10 GB allocation -> MEMORY, fork loop -> CRASH/limited · each gives the expected status
  and the kernel survives · fix limits.
- **RC-22** tests valid · each problem's reference solution passes all public and hidden tests; an empty function fails · 100% / 100% ·
  drop the problem, log it.
- **RC-23** flaky tests · every attempt run twice on DEV smoke · status agreement 100%; problems with disagreement dropped (counted).
- **RC-24** tests read-only · an attempt that writes to the test file cannot change the result (tests copied per run, hashed before and after) ·
  hashes equal · fix isolation.
- **RC-25** ERROR never PASS · unit check on the status mapper with synthetic outcomes · all ERROR/TIMEOUT/MEMORY/CRASH map to not-pass.

### Phase 3. Generation (DEV)
- **RC-30** thinking budget knee · accuracy vs max_new_tokens on 30 DEV problems; knee = smallest budget within 1 point of the best ·
  fixed before the full run, recorded · never re-tuned on HOLDOUT.
- **RC-31** diversity · distinct output signatures per 8 attempts, mean · >= 2 (proposal); if lower, the pre-registered higher temperature is
  used once, recorded.
- **RC-32** code extraction · NO_CODE rate · reported; > 10% = fix the extraction prompt on DEV before the full run.
- **RC-33** generated-test quality (DEV only, diagnostic) · fraction of Qwen-written tests the reference solution passes · reported; it calibrates
  how much weight CodeT can carry; never used at runtime.
- **RC-34** shard integrity · every shard has a manifest with row count and sha256; resume skips completed attempt_ids · no duplicates, no gaps.

### Phase 4. Training (DEV only)
- **RC-40** grouping · no problem group in both train and evaluation folds; OOF scores exist for every DEV problem · 0 overlaps.
- **RC-41** overfit check · 32 pools, loss drop >= 50% in 60 steps · pass, else the training path is broken; stop.
- **RC-42** hyper-parameters · chosen by inner folds only; the choice log is saved · no choice uses outer-fold or HOLDOUT data.
- **RC-43** calibration · temperature fitted on DEV OOF; ECE reported before and after · ECE reported (no pass threshold; it is a measurement).
- **RC-44** label audit · re-run hidden tests for a random 5% of DEV attempts · 100% agreement · else find the runner bug, relabel.
- **RC-45** class balance · NONE-positive rate and mean passing attempts per pool reported; NONE chosen rate in predictions reported ·
  a scorer that never or always picks NONE fails the gate G-S review.
- **RC-46** shortcut controls · code-only text, length-only, no-context arms trained with the same folds · G-Z rule.
- **RC-47** seeds · 3 seeds per arm; per-seed numbers reported next to the ensemble.

### Phase 5. Inside every loop run (online checks, logged per decision)
- **RC-50** NONE present in every pool; every pool has >= 1 attempt.
- **RC-51** hard filters ran before any learned score (journal order check).
- **RC-52** no learned key reorders attempts with different public pass counts (assert in the ranker; any violation aborts the problem as ERROR).
- **RC-53** EXHAUSTED or NO_PROGRESS is never recorded as DECIDED (assert).
- **RC-54** caps respected (rounds, attempts, wall) (assert).
- **RC-55** journal row complete before the next round (schema check).
- **RC-56** scorer frozen during the loop: adapter file sha256 checked at start and end of the run (Goodhart guard, loop-A LA3).
- **RC-57** confident-wrong watch (DEV only): DECIDED problems that fail hidden tests are listed with their p_none and public pass counts.
- **RC-58** live = replay: 10 problems run live end-to-end and replayed from the fixed 16-attempt order give identical decisions.

### Phase 6. Evaluation
- **RC-60** equal compute: every comparison is at the same attempts (or reports both axes on one curve).
- **RC-61** HOLDOUT read once per pre-registered arm; access log with time and arm; a second read = new HOLDOUT needed.
- **RC-62** clustered CIs for every headline number; W->R / R->W tables for every slot swap.
- **RC-63** every headline number backed by a row-level file under `results/raw/` (create-only).
- **RC-64** failed gates reported with the actual numbers; no gate edited after results.
- **RC-65** `results/increments.md` updated (step, before, after, delta, CI) even when the step moved nothing.
- **RC-66** Goodhart curve: selected@k for k = 1..16 per arm; if it falls with k, the cap is the peak k found on DEV (LB3).

### Phase 7. Port to another Qwen (only after A3 passes on Qwen3-4B, separate GO)
- **RC-70** new model (e.g. Qwen3.5-9B): repeat RC-10..RC-16 with that model; new adapter trained (M12); head shared; all gates re-run.
- **RC-71** deployment parity: the local runtime (6 GB laptop) must produce states the same way as training (same quantisation and dtype).
  Qwen3-4B in fp16 does not fit 6 GB (about 8 GB, estimate), so local use needs a quantised model, and the adapter must be trained on states from
  that same quantised model (parity test as RC-11 on that device class; compute stays on Kaggle).

---

## 9. Files (minimum; reuse first)

| file | new / reuse | role |
|---|---|---|
| `configs/thresholds_exec_loop_v1.toml` | new | every value marked "proposal" here, frozen before the smoke run |
| `tools/exec_loop_prep_v1.py` | new | M0: normalise problems, split manifest, seal hidden tests, pack the Kaggle dataset |
| `kaggle/exec_loop_v1.py` | new | one kernel, stages via env var: smoke, sample+execute, capture, train, replay, repair, holdout-eval |
| `tools/fable_capture.py` | reuse, unchanged | capture protocol |
| `tools/adapter_align_v1.py`, `decision_head/head.py` | reuse | head forward, adapter tensors |
| `kaggle/head_exp4_gpu.py:probe_scores` | reuse | probe |
| `kaggle/context_diag_v1.py` | reuse pattern | one subprocess per GPU, phase workers |
| `results/raw/exec-loop-v1-<UTC>/` | new, create-only | all outputs |

---

## 10. Build phases and their gates

| phase | where | content | gate to continue |
|---|---|---|---|
| P0 | PC | approvals, dataset choice, thresholds file, prep script, split manifest | RC-00..RC-06 |
| P1 | Kaggle | smoke: 10 DEV problems, k = 4, all modules end to end | RC-10..RC-16, RC-20..RC-25, RC-58 (on the smoke set) |
| P2 | Kaggle | full sampling: 16 attempts per problem (DEV + HOLDOUT), execution (public, generated; hidden sealed), capture | RC-30..RC-34 |
| P3 | Kaggle | train adapters / probe / controls on DEV (OOF) | RC-40..RC-47; G-0 (headroom) |
| P4 | Kaggle (CPU) | replay every policy on DEV OOF; fix tau_stop and the cap | G-B, G-C, G-D, G-Z on DEV (development reading only) |
| P5 | Kaggle | repair rounds for DEV and HOLDOUT problems still failing | RC-50..RC-57; G-E on DEV |
| P6 | Kaggle (CPU) | HOLDOUT evaluation, once per pre-registered arm | all gates; RC-60..RC-66 |
| P7 | Kaggle | port to the next Qwen | RC-70, RC-71 (separate GO) |

Compute (estimate, unmeasured): 300 problems x 16 attempts x about 6,000 generated tokens = about 29M tokens; plus capture and execution.
On two T4s this is several sessions. The P1 smoke measures tokens per second and attempts per hour, and P2 is re-planned from that
measurement (RC-06).

---

## 11. Risks and kill rules

| risk | sign | action |
|---|---|---|
| no headroom (Qwen solves everything or nothing) | G-0 fails | stop selection work; report pass@1 and oracle; choose a harder or easier problem window only with a new versioned split |
| probe beats LAYA again | G-B or G-Z fails | the probe takes that slot; LAYA keeps only jobs it wins; reported plainly |
| pretrained head adds nothing | G-C fails | report "adapter only"; still use the head as the architecture's context layers since it costs nothing |
| public tests decide almost everything | `decisive_signal` = LAYA in < 5% of problems | report LAYA's measured marginal effect as small; the budget and repair jobs are where it can still matter |
| Goodhart with k | RC-66 curve falls | cap k at the DEV peak |
| contamination | RC-03 | only post-release problems count for headline numbers |
| Kaggle session limits | RC-06 | shards + resume |

---

## 12. Decisions needed from you

| id | decision | my suggestion |
|---|---|---|
| D1 | problem set | LiveCodeBench problems released after 2025-07 (size and licence checked before download) |
| D2 | where code runs | inside the Kaggle kernel, internet off, resource limits (M6) |
| D3 | GO for P0 (prep script + thresholds file, local) | yes |
| D4 | GO for P1-P2 on Kaggle and the dataset upload | after P0 is reviewed |
| D5 | confirm the 2026-10-15 "loop engineering, not generation" note does not apply to A3 (Qwen writes its own attempts) | confirm |
| D6 | later: allow training Qwen itself (ReST / distillation tiers A10-A11)? | not needed for A3 |

---

## 13. Checklist

| item | status | evidence |
|---|---|---|
| Components M0-M13 specified (inputs, outputs, reuse, failure modes) | DONE | section 2 |
| LAYA interface checked against the code | DONE | pointer over {hq, ho_i} in `tools/adapter_align_v1.py` (lines 95-97); questions in `kaggle/outcome_head_v1.py` (lines 30-45) |
| Data flow and state machine | DONE | sections 2 (M10) and 3 |
| Data contracts | DONE | section 4 |
| Splits, labels, contamination rule | DONE | section 5 |
| LAYA's three jobs and their competitors | DONE | section 6 |
| Metrics, statistics, gates | DONE (values are proposals) | section 7 |
| Recheck points RC-00..RC-71 | DONE | section 8 |
| Files and build phases | DONE | sections 9, 10 |
| Thresholds file frozen | NOT DONE | waits for D3 |
| Any code written or run | NOT DONE | waits for D1-D4 |
| D5 confirmation | WAITING ON USER | section 1 open tension |
