# H3 review: Pilot 1 grader, protocol and comparison (read-only)

Scope: `D:/local model/results/raw/codex-pilot1-20261016/kit/Pilot1_Evidence_Ladder/`, compared with `D:/local model/specs/evidence_ladder_v1/S4_execution_policy.md` and `S6_evaluation_protocol.md`.
Method: Read, Grep, Glob only. No code was run, nothing was modified, no network. Line numbers are as read.

## Summary (10 lines)

1. BROKEN: `grade.py` has no isolation. A candidate can read `protected_cases.json` and `oracles.py` by absolute path, or forge `{"ok": true, ...}` lines on stdout (grade.py L17, L34-35; runner L16). S4 would reject this grader for real grading.
2. BROKEN: SELFTEST's "constant_none_mutants_killed: 40" is a counter, not a measurement. No mutant is executed (SELFTEST L15-16, L23).
3. BROKEN: the S6 section 6 mutation-adequacy rule (>= 0.80) is NOT MEASURED. There is no stress set and no headroom screen.
4. BROKEN: the candidate sha256 is computed after the run (grade.py L47), so it does not attest the bytes that were executed. S4 section 4 requires an immutable candidate with a hash.
5. BROKEN: no budget or token logging exists anywhere in the kit (grade.py L48 has no such field; compare.py takes none). PROTOCOL L24-25 and L35 cannot be met.
6. BROKEN vs S6 section 7-8: compare.py has no McNemar test, no Holm correction, no gate output, and uses 4000 resamples where S6 says 2000 (compare.py L5). No file in the kit mentions McNemar or Holm.
7. WEAK: one 5.0 s wall-clock limit covers all 40 cases of a task (grade.py L12, L29-31, L38). This is undocumented, load-dependent, and its fairness is unmeasured.
8. WEAK, unverified: `cwd='/tmp'` (grade.py L31) on Windows means `C:\tmp`. I could not confirm that path exists. If it is missing, every task becomes PROCESS_ERROR (L39) and the arm scores 0 silently. `env` also drops SYSTEMROOT (L26), which some Windows Python builds need. Neither was tested.
9. WEAK: PROTOCOL L34 names the essential contrasts as C vs D0, D0 vs D1 and A vs B. S6 section 7 makes C vs B, D0 vs C and D1 vs D0 the confirmatory set. The "does verification beat extra compute" contrast (C vs B) is missing from PROTOCOL's exit decision.
10. WEAK, not measured: the dataset is 40 classic problems, authored with their own oracles, with no post-2025-07-31 material (S6 section 5 item 1). Oracle independence, contamination and headroom are all unproven. The kit discloses this honestly (README L3, PROTOCOL L8, QUESTIONS.md L3), but it does not record it as a waiver of S6.

---

## Q1. grade.py correctness

Verdict: the all-40 rule is implemented correctly (L43-46). Several edge-case and robustness issues are WEAK or BROKEN.

- All 40 must pass: OK. `ok` counts only when `received.get('ok') is True` and the value matches (L43). The task passes only if `ok == len(data)` (L46). This matches PROTOCOL L9.
- Type-strict comparison, top level: OK. `type(received.get('value')) is type(expected['expected'])` (L43) rejects int vs float (6.0 vs 6) and bool vs int (True vs 1). The rule that 6.0 is wrong for an int answer is not written in the protocol.
- Type-strict comparison, nested: WEAK. Nested values use `==` (L43), so `[True] == [1]` and `[1.0] == [1]` pass, and so do dict keys `1` and `True`. No task checks this, but the strictness claim is only partial.
- List vs tuple: OK. The runner emits JSON (L16), so tuples become lists before grading.
- Timeout, one limit for all 40 cases: WEAK. The code deliberately runs one subprocess per task (L29-31) and maps `TimeoutExpired` to `TIMEOUT` for the whole task (L38). It is not documented as intended. A single slow case fails the whole task. The 5.0 s default (L12) is not justified anywhere. Wall-clock limits are load-dependent, so results are not reproducible across machines or load levels. Input sizes are not measured here (I did not read protected_cases.json contents), so fairness cannot be judged.
- Missing submission: OK (L21-22).
- Oversized file: OK. 128 KiB is checked before execution (L23-24).
- Runtime error in one case: OK. The runner catches `BaseException` per case (_candidate_runner L17-18), that case becomes `ok:false`, and the task fails at L46.
- Extra lines on stdout: WEAK. Any extra JSON line gives `MISSING_OUTPUT` (L37), a non-JSON line gives `INVALID_OUTPUT` (L36). The task fails, but the label is misleading, and a harmless `print()` inside `solve` fails the entire task. Nothing in SOLUTION_FORMAT.md forbids printing, and it does not warn about this.
- Crash path: BROKEN (robustness). If stdout contains a valid JSON line that is not an object, and the line count still matches, `received.get` (L43) raises AttributeError. The grader then aborts before writing its output (L49), which loses the whole arm's report. Reaching this needs deliberately crafted output (for example, replacing `sys.stdout` or patching `json.dumps`). It is a robustness hole, not a common-path bug.
- Output line splitting: WEAK. `splitlines()` (L34) also splits on U+2028, U+2029 and U+0085. The runner uses `ensure_ascii=False` (runner L16), so a correct answer containing one of those characters is marked INVALID_OUTPUT. This is rare.
- Candidate calls `sys.exit`: OK. Inside `solve`, SystemExit is caught per case (runner L17). At import, it gives a fatal line and exit 1 (runner L10-11), which becomes PROCESS_ERROR (L32).
- Candidate reads stdin itself: WEAK, and not an answer leak. Stdin carries only inputs (grade.py L30), never expected outputs. A candidate that reads stdin does, however, desynchronise the runner's `for line in sys.stdin` (runner L12), which gives wrong or missing answers. The stdin pipe is shared with the runner's loop.
- Candidate mutates its input: OK per case. Each line is parsed freshly (runner L14). Cross-case state is still possible, because all 40 cases share one process and module-level state persists (grade.py L29-31).
- Nondeterminism: WEAK. `PYTHONHASHSEED=0` (grade.py L26) is good. The wall-clock timeout (L12, L31) is not deterministic. Candidate randomness is unseeded. `cwd` is the shared `/tmp` (L31), so candidates can leave files for each other.
- Candidate file hash: BROKEN. `sha256` is computed at L47, after `subprocess.run` (L31). The candidate can rewrite its own file during the run, and the recorded hash then describes the modified file, not the executed one. Protocol L10 ("freeze selection before grading") has no enforcement path at all.
- Result file: WEAK. The report is written once, at the end (L49). Any crash loses it. There is no per-task record flagging a rerun after viewing protected results (PROTOCOL L10 DEV/CONTAMINATED), and no field for budget or tokens (L48).
- Windows `cwd='/tmp'`: WEAK, unverified. I could not confirm whether `C:\tmp` exists (the Glob on `C:/` timed out). If it does not, subprocess raises FileNotFoundError, which is an OSError, and L39 turns it into PROCESS_ERROR for every task. The arm then reports 0 without an error. This must be checked before any run.
- Windows env: WEAK, unverified. `env` is only PATH and PYTHONHASHSEED (L26). Whether the Windows Python child starts with that environment was not tested.

## Q2. Security holes in the unsafe grader

Verdict: BROKEN. The grader is explicitly unsafe, and the flag is the only guard. S4 would not accept it.

The `--unsafe-local` gate is real: grade.py L14-15 calls `p.error(...)`, which exits before any execution. Without the flag nothing runs (fail closed, correct).

Once the flag is passed, a hostile or buggy candidate can:

1. Read the expected answers. `protected_cases.json` is at `folder/'protected_cases.json'` (L17). The candidate runs with `-I -S` (grade.py L28), which does not restrict filesystem reads. An absolute path works.
2. Import the reference implementations. `oracles.py` is in the same folder. A candidate can load it with `importlib.util.spec_from_file_location` (the same mechanism the runner uses, runner L5-6) and delegate to `ORACLES[task]`. That passes every protected case without solving anything. The grader's folder is reachable from the runner's `__file__`, which the candidate can read.
3. Forge output. The grader trusts the candidate's stdout (grade.py L34-35). The candidate can write any JSON lines, patch `json.dumps` or `sys.stdout`, and call `os._exit(0)`, and the grader will score the forged lines as passes. This is the direct equivalent of S4-T6 ("print a fake PASS banner ... ignored"), and it fails here.
4. Read the other arms' submissions, the results, and the project tree. There is no path restriction. Walking up from the kit folder reaches `D:/local model/.secrets/`, which CLAUDE.md says holds the NVIDIA key files.
5. Use the network. There is no socket restriction. PROTOCOL L28 says "This package itself has no network calls", which is true of the package but not of the candidate. A candidate can call out. This contradicts the Rs 0 and "internet off" requirements (S4 section 3.1).
6. Write or delete anywhere the user can write. That includes `protected_cases.json` itself, which is read once per run (grade.py L17) and so would poison later arms. The candidate can also overwrite its own file after `sha256`, or kill the parent before the report is written.
7. Exhaust resources. There are no `RLIMIT_*` settings (S4 section 3.2 requires `RLIMIT_CPU`, `AS`, `NPROC`, `FSIZE`, `NOFILE`). Only the wall clock applies (L31). `subprocess.run` kills only the direct child, not the process group (S4 section 3.2 requires a process-group kill). Grandchildren survive.
8. Run as the same user, with no uid drop (S4 section 3.2 requires `setuid` in `preexec_fn`).
9. Read hidden expectations in plain text. S4 section 3.2 requires encryption at rest and decryption only in grader memory. Here the expected values sit in plain JSON on disk.

S4 check, requirement by requirement:

- 3.2 uid drop: absent (grade.py L28, no `preexec_fn`).
- 3.2 empty environment: partly (L26 keeps PATH).
- 3.2 fresh per-run temp dir: absent (L31 uses shared `/tmp`).
- 3.2 closed stdin except the test pipe: met (L30-31).
- 3.2 RLIMITs: absent (L26-31).
- 3.2 process-group kill: absent (L31, direct child only).
- 3.2 hidden tests encrypted and in memory only: absent (L17, plaintext on disk).
- 3.3 S4-T1 (open hidden files): fails.
- 3.3 S4-T2 (read grader env or memory): not prevented.
- 3.3 S4-T3 (socket): fails.
- 3.3 S4-T4 (fork bomb, allocation, giant file): not limited.
- 3.3 S4-T5 (write outside temp): fails.
- 3.3 S4-T6 (fake PASS): fails (see point 3 above).

Conclusion: the grader would not pass S4 section 3 or any red-team test. The README (L32) and PROTOCOL (L31) say this themselves. The `--unsafe-local` flag is correctly required, but the README quick-start (L26-29) shows the flag in the usual command, so it is easy to use for a real run by accident.

## Q3. SELFTEST.py

Verdict: BROKEN. The "constant-null mutants killed" claim is not backed by the code.

What it actually does:

- Loads `ORACLES` (L5) and `protected_cases.json` (L6). It loads `worker/public_cases.json` (L7). That file exists (Glob of `worker/`), so the public check runs.
- For each protected case, asserts `ORACLES[key](input) == expected` (L11-12). For each public case, asserts the same (L17-22). That is 44 oracle-vs-expected checks per task, which is what PROTOCOL L11 describes.
- Lines L14-16 are the mutant part. L15 asserts only that at least one expected output is not None. L16 increments `mutants` unconditionally, once per task.
- L23 prints `constant_none_mutants_killed: mutants` (= 40). That is the number of tasks, not the number of mutants killed.

What it does not test:

- No candidate or mutant is executed. SELFTEST never imports `grade.py` or `_candidate_runner.py`.
- No wrong program is run against the tests. A trivial constant-None mutant would be killed only if grade.py were run on it, and that never happens here.
- The `==` at L12 and L21 is not type-strict, so int/float or bool/int disagreements in the oracles are not caught.
- Nothing checks that the grader itself works.
- Nothing checks that oracles and expected outputs were produced independently. The kit does not record where the expected values came from. If they were generated by the oracles, L12 passes by construction. I could not check this from the files.

Mutation adequacy (S6 section 6, >= 0.80): NOT MEASURED. There are no generated mutants, no hidden-test kill rate, no flakiness rerun (S6 section 6 requires two runs to agree), and no check that a trivial wrong program fails. The README claim "kills simple constant-null mutants" (README L16) and PROTOCOL L11 ("mutation smoke checks only establish that trivial constant-null solutions fail") are not supported by code that runs such a mutant.

## Q4. compare.py

Verdict: the bootstrap, pairing and sign convention are OK. Holm, McNemar, budget and the gate are missing (BROKEN against S6 section 7-8).

- Paired bootstrap: OK. The same resampled task ids index both arms (L9-10), so the pairing is preserved. The percentile interval indices (L11) are correct.
- Resampling unit: OK for this kit. It resamples tasks (L9). S6 section 7 says "by contest cluster". Tasks here have no contests, so the task is the unit, which is acceptable, but it should be stated.
- Resample count: deviates. L5 uses `reps=4000`. S6 section 7 says 2,000.
- Seed: fixed (L5, MANIFEST L12). That is reproducible, but nothing ties it to per-problem sampling seeds (S6 section 1 and 7 require those, and the kit has no generator).
- Pairs (L22): (A,B), (B,C), (C,D0), (D0,D1), (D1,E). Output labels are `right minus left` (L29). So the output contains "B minus A", "C minus B", "D0 minus C", "D1 minus D0", "E minus D1". Those match S6's confirmatory contrasts (C vs B, D0 vs C, D1 vs D0) and reported contrasts (B vs A, E vs D1) in direction.
- Sign convention: OK. `gain_points = (sum(y) - sum(x)) / n` with x = left and y = right (L29). A positive value means the right arm is better. Gained = left 0 and right 1 (L29). Lost = left 1 and right 0 (L29).
- Intersection of task ids (L28): WEAK. Tasks missing from one arm are dropped silently. `paired_tasks` is reported, but there is no warning and no check that both arms cover all 40 tasks, which PROTOCOL L35 requires.
- Arm keying (L20): WEAK. `runs[d['arm']]` overwrites silently if two reports share an arm name. `grade.py` defaults the arm to `unknown` (grade.py L10).
- Missing C or D0: WEAK. Only a warning (L21), and the run continues.
- Missing: BROKEN vs S6. No McNemar exact test. No Holm correction across the three confirmatory contrasts. No p-values at all. No gate decision (S6 section 8 needs "D0 > C, Holm-adjusted p < 0.05, CI lower bound > 0"). No "each arm vs the simplest arm that beat B". No probe or random-head controls beside E (S6 section 1). No headroom screen (S6 section 7). Grep of the whole kit confirms McNemar, Holm and headroom appear nowhere.
- Budget: BROKEN vs PROTOCOL L35. There is no budget input to compare.py and no budget field in grade.py's output (L48).

## Q5. PROTOCOL.md and README.md against S6 (and S4)

Agreements:

- 40 original tasks, four public tests, separate protected tests (PROTOCOL L8; README L3, L14; MANIFEST L10-11).
- Arm names and roles A to E (PROTOCOL L14-19; SOLUTION_FORMAT L13-19; S6 section 1 table).
- Primary metric: all hidden tests pass (PROTOCOL L9; S6 section 2).
- Failures, timeouts and crashes count as failures (PROTOCOL L9; grade.py L21-39, L46).
- Pairing by task (PROTOCOL L35; S6 section 7).
- E is not a gate and is not stage-1 acceptance for LAYA (PROTOCOL L21, L36; README L39; S6 section 8).
- Hidden tests are graded after the run and do not select anything (S6 section 1, last bullet; PROTOCOL L10).
- Protected-grade boundary and need for a real sandbox (PROTOCOL L30-31; README L32; S4 section 3).
- Rs 0 (PROTOCOL L28; S4 section 6 spend).
- DEVELOPMENT pilot, no generalization claim without an independent benchmark (PROTOCOL L37; README L3; MANIFEST L14).

Differences and contradictions:

- B budget: PROTOCOL L15 says "at least the generated-token budget used by the critic-bearing arm". S6 section 3 says the same cap T for every arm. "At least" is weaker and can exceed T. T is also never set in the kit (PROTOCOL L25 says "based on DEV" but no number is recorded).
- A composition: SOLUTION_FORMAT L14 says "Qwen + tools, first attempt". S6 section 1 says "one attempt, public examples visible in the prompt". "+ tools" is not in S6. PROTOCOL L14 omits the public-examples detail.
- C: S6 section 1 includes "no controller beyond S5's stop rules". PROTOCOL L16 does not.
- D1 trigger: PROTOCOL L18 says "when its experiment trigger fires" without defining it. S4 section 6 defines the trigger (two or more clusters survive, or a hard requirement has no trusted check). The kit does not reference S4 section 6.
- GLM model name: PROTOCOL L18 and SOLUTION_FORMAT L18 say "GLM-5.3-Flash". S4 section 6 says `z-ai/glm-5.3-flash` (decision D5). CLAUDE.md's judge rules name "GLM-5.3 full reasoning". These should be reconciled. S4 FC-2 (maximum reasoning parameter) is not mentioned in the kit.
- E controls: S6 section 1 requires a probe and a random-head adapter in the same slot. PROTOCOL L19 has neither.
- Confirmatory contrasts: PROTOCOL L34 lists C vs D0, D0 vs D1 and A vs B as "essential". S6 section 7 lists C vs B, D0 vs C and D1 vs D0 as confirmatory, with B vs A reported only. The exit decision therefore leaves out the key "verification vs extra compute" contrast (C vs B) and promotes A vs B, which S6 does not treat as a gate.
- Statistics: PROTOCOL L35 says "wide bootstrap intervals" and says nothing about Holm or McNemar. S6 section 7 requires both. compare.py has neither.
- Gate: S6 section 8 defines stage-1 gates. PROTOCOL has no gate and no safety (false-approval) rule. The kit has no labels, so false-approval is not computable here.
- Sample size: S6 section 7 says about 470 problems for a 5-point gain (estimate, unmeasured), and says an underpowered pilot must be reported as such. PROTOCOL L26 says "diagnostic pilot" and mentions the 2.5-point task step, but gives no detectable-effect statement. This is partly OK.
- Headroom: S6 section 7 requires a best-of-k oracle minus pass@1 screen below 5 points. Absent from PROTOCOL and from the kit.
- Stress set: S6 section 9 requires planted wrong candidates, a broken test tool, a misleading statement and a malformed critic experiment. Absent.
- Stage-0 feasibility checks (FC-1 to FC-4; S6 section 10, S4 section 3.2): absent. The kit runs the grader unverified on the user's Windows machine (see Q1 WEAK items).
- Dataset: S6 section 5 requires problems published after 2025-07-31 and gives the LiveCodeBench Pro candidate (D1 unresolved). The kit uses original tasks, which breaks item 1 of that list (README L3: "not a recent external dataset"; QUESTIONS.md L3: "not freshly released after 2025"). The kit does not say this is a waiver of S6 section 5, and does not reference the D1 decision.
- Common random numbers: S6 section 1 requires arms B to E to share the first k1 samples. The kit has no generation code, so this cannot be checked.

Claims in the kit that no file in the kit backs:

- PROTOCOL L11 "Manually inspect ambiguous tasks and compare with independent implementations": no independent implementation and no inspection record exists in the kit. This is stated as a future step, which is honest, but it is not done.
- README L16 and SELFTEST L23 "kills simple constant-null mutants" / "constant_none_mutants_killed": no mutant runs (see Q3).
- PROTOCOL L10 "mark it DEV/CONTAMINATED": no mechanism in grade.py.
- PROTOCOL L24 and L35 "budget logs were collected": no log, no field and no reader exists.
- PROTOCOL L28 "Rs 0 rule: fail closed before paid usage": there is no such check in the code. The package makes no calls, but the candidate can (see Q2 point 5).
- README L8 "Zero cost only if you run with your existing free quotas": no cost record.
- README L3 "Public problems and sample cases were authored together with the reference oracles": a self-description, consistent with the files, but not a verifiable record.
- PROTOCOL L11 "exercised against all 44 examples per task": backed by SELFTEST L10-22 (44 checks per task, public file present). Backed as equality, not as independence.
- SOLUTION_FORMAT L14 "+ tools": no tool use exists anywhere in the kit.
- "tests were authored with references" (PROTOCOL L8): consistent with the files but not documented.
- Not checked by me: "40 questions" in QUESTIONS.md (not read) and the 40 protected cases per task (I read only the first four lines of tasks.jsonl, which show `protected_count: 40`, and counted one JSON line in protected_cases.json, so the per-task count was not verified).

Backed and checked: difficulty split 10 easy / 20 medium / 10 hard (task_list.json: P1-01..10 easy, P1-11..30 medium, P1-31..40 hard). "not calibrated" is stated (tasks.jsonl line 1: `difficulty_basis`).

## Q6. Dataset provenance and independence

What the kit shows: the task prompts, public examples, oracle code (`oracles.py`) and protected expected values were all produced by one assistant, in one kit (README L3, PROTOCOL L8, QUESTIONS.md L3). The kit discloses this.

(a) Oracle independence: NOT ESTABLISHED.

- No second author, no second implementation and no record of review exist.
- The expected values' origin is not recorded. If they came from `oracles.py`, then SELFTEST L12 passes by construction, and every oracle bug becomes a "correct" answer in all arms.
- Oracle-versus-spec agreement could not be fully checked. I read `oracles.py` in full, but the prompts for P1-05 to P1-40 were outside the instruction to read only the first lines of tasks.jsonl. So I checked the code only, not the prompts. Spec-dependent points to check against the prompts: p07 (L30-33, assumes ascending input); p09 (L43, returns None under two distinct values); p20 (L118-124, cap 0 handling); p22 (L139-140, touching intervals treated as non-overlapping); p36 (L249-255, assumes non-negative weights); p37 (L259, returns None when a corner is blocked); p38 (L271-273, the `q+x<=s//2` prune is valid only for non-negative numbers); p40 (L283-296, recursive DFS; deep graphs exceed Python's default recursion limit and raise, so SELFTEST would crash rather than pass).
- I found no outright code bug in the oracles I checked against the task titles. The title-level check is weak evidence and does not replace a spec check.

(b) Contamination: PARTIAL, not measured.

- The titles in task_list.json are widely circulated textbook and competitive-programming problems. From the titles alone, I recognise many as standard set pieces: maximum subarray (P1-12), LRU cache (P1-20), Levenshtein distance (P1-31), LIS (P1-32), trapping rain water (P1-35), Dijkstra (P1-36), strongly connected components (P1-40), coin change (P1-18), word break (P1-33), equal subset partition (P1-38), maximum product subarray (P1-39), and so on. This is my recognition from titles only. I did not search the web (no network), and I did not read the prompts beyond the first four.
- The kit does not measure memorisation. Given the premise that Qwen has very likely seen such problems, a pass on these tasks says little about reasoning. Estimate (unmeasured).
- S6 section 5 item 1 (post-2025-07-31 material) is not met, and the kit does not say so as a decision.

(c) Headroom: NOT MEASURED, and likely small.

- Estimate (unmeasured): a 4B thinking model is likely near ceiling on the 10 easy tasks, and plausibly high on many classic medium and hard tasks. If pass@1 is already high, best-of-k minus pass@1 may be below the 5-point screen in S6 section 7, so a selection or controller effect would be hard to detect here.
- The kit has no headroom screen, so this cannot be confirmed.

What the pilot can show:

- Whether the harness and grader run end to end, once the Windows issues (`/tmp`, env) are checked.
- Rough per-arm pass counts on 40 development tasks, and the discordant pattern (gained and lost tasks) between arms, if the arms are ever run.
- That the grading flow fails closed without `--unsafe-local`.

What it cannot show:

- Generalisation, or any claim beyond this development set (the kit says so).
- That the controller (D0) or the critic (D1) helps. Each task is 2.5 points, the bootstrap intervals on 40 tasks will be wide, and no McNemar or Holm test is implemented.
- Oracle correctness (no independent check).
- LAYA's effect (E is an out-of-domain transfer probe, and the kit has no controls for it).
- Cost or budget matching (no logs exist).
- Security: the grader cannot be trusted for protected grading (see Q2).

---

## Final checklist

Format per CLAUDE.md "Always checklists": DONE / NOT DONE / BLOCKED / WAITING ON USER, with evidence.

- DONE: read README.md, PROTOCOL.md, MANIFEST.json, worker/SOLUTION_FORMAT.md, grader/grade.py, grader/_candidate_runner.py, grader/SELFTEST.py, grader/compare.py, S6 and S4 (evidence: line references above).
- DONE: read worker/tasks.jsonl first 4 lines and counted its 40 lines; read worker/task_list.json; read grader/oracles.py (beyond the requested list, needed for Q3 and Q6).
- DONE: confirmed existence of worker/public_cases.json and worker/QUESTIONS.md (Glob).
- DONE: Q1 to Q6 answered with file and line evidence and verdicts.
- DONE: grep of the kit for McNemar, Holm, headroom, mutation, budget, seed, stress, jail, sandbox, independent (no hits for McNemar, Holm, headroom or stress).
- NOT DONE: no code was run, by instruction. So no grade.py or SELFTEST smoke run was executed; all findings are from reading.
- NOT DONE: prompts for P1-05 to P1-40 were not read (scope was the first lines of tasks.jsonl), so oracle-vs-spec matching is UNVERIFIED for those tasks.
- NOT DONE: protected_cases.json contents not read (only a one-line count), so input sizes and the expected-value origin are UNVERIFIED.
- NOT DONE: QUESTIONS.md and public_cases.json contents not read.
- BLOCKED: whether `C:\tmp` exists. The Glob on `C:/` timed out twice. Needs a direct check by the user.
- BLOCKED: Windows child-process environment (SYSTEMROOT) behaviour. Needs a run.
- WAITING ON USER: decision on whether to fix the grader (add a real sandbox, per S4 section 3) before any arm is graded; decision on the S6 section 5 dataset waiver; decision on the confirmatory-contrast list in PROTOCOL L34.
- NOT DONE (by design, not required of this review): no edits to any kit file, no network, no model calls.

Output file: `D:/local model/results/raw/codex-pilot1-20261016/review/H3_grader_protocol.md`
