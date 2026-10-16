# Evidence-Ladder — Java Pilot 1 experiment protocol

**Purpose:** A Java-domain DEVELOPMENT smoke test of the A/B/C/D0/D1 arms. These are original tasks with assistant-authored reference solutions, *not* a fresh published sealed benchmark. Language is Java 21; 10 easy, 20 medium, 10 hard. Run on local JDK or Kaggle only if JDK 21 can be provided under the stated zero-spend constraints.

## Separation and reliability
- Worker receives `Pilot1_Java_Worker.zip` only: 40 prompts, 4 public examples each, JSON runner protocol and Java support code.
- Grader receives `Pilot1_Java_Grader.zip` only in a distinct trusted grading environment: 40 extra cases per task, reference solutions, grading scripts. These grader examples are **development tests**, not truly independent or externally sealed oracles.
- The included Python CLI grader **IS NOT A SECURITY SANDBOX**. It compiles and executes submitted Java, which may access any file the process can access. `--unsafe-local` is explicit acknowledgement. Use only on code you trust; for untrusted generated code use genuine OS/container isolation, no grading data or keys visible inside candidate process, and a separate scoring process.
- Under no circumstances provide grader reference answers or protected test vectors to Qwen or GLM during a measured run. If a worker must receive failure feedback, expose only approved public/probe diagnostic results from a separate development oracle, not the final grader cases.
- If testing under different environments, fix locale, Java version, memory, timeout, and input encoding before comparing arms.

## Candidate submission
- One `Solution.solve(Map<String,Object>)` per task, saved as `J1-01.java`, etc. No additional files or external libraries.
- Compile each solution independently with Java 21; judge compiles all 40 candidates in separate temporary directories.
- `grade.py` tests 4 public plus 40 private DEVELOPMENT vectors per problem. Task solved iff all 44 pass. Null outputs, compile errors, crashes and timeouts fail.

## Experimental arms
- A: Qwen + public tools, one attempt.
- B: Qwen + extra generation budget, best-of-k selected using PUBLIC tests only.
- C: Qwen + public execution feedback and bounded repair.
- D0: C + Evidence-Ladder controller, **without GLM**.
- D1: D0 + NVIDIA-hosted GLM critic proposing runnable discriminating checks. This measures incremental **strong external model** value; never pretend this is local-only performance.
- E: optional LAYA transfer test on whole-program ranking; **not a head-quality gate**. LAYA's relevant action-selection benefit belongs to Stage 2.

## Measurement and acceptance
- Primary outcome: percent of tasks with all approved grader tests passing. Per-task paired outcomes and development confidence intervals.
- Report public test pass rate separately, compile failures, runtime errors, repair attempts, case pass rate, unknowns, wasted/invalid runs, token counts per model, CUDA time, wall time, tool calls and cost.
- Set tokens/time ceilings from DEVELOPMENT work. A token-matched GLM arm is not a model-cost matched arm. No additional vendor charges; check free trial and account limits first.
- Reserve a future **new, task-disjoint sealed external benchmark** for generalization claims. Never tune thresholds or model prompts on its protected cases.
- Do NOT credit Java Pilot 1 with validating pretrained LAYA next-action ranking. Do NOT call a repaired answer verified until an independently executed, goal-relevant oracle approves its actual artifact.
- No models, API calls, LAYA adapter inference or Kaggle jobs are bundled or executed here.

## Grader quality gate
- Compile the 40 Java reference solutions with `javac`.
- Recompute all 1,760 expected results against the stored vectors.
- Reject a constant-null bad implementation for each task.
- Run `grader/INDEPENDENT_CHECKS.py` (1,760 case expectations checked against separately implemented Python algorithms). This reduces implementation errors but does not establish independence of task selection, examples, or oracle adequacy; external audit remains necessary before using the bank as sealed evidence.
