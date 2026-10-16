# Java Pilot 1 validation report

- **Task mix:** 10 easy, 20 medium, 10 hard; 40 new task titles, no repeat of original Python Pilot 1 prompt/topic.
- **Fixtures:** 4 worker-visible samples per task + 40 grader-only development fixtures per task = **1,760 distinct inputs per task set** (unique within each task).
- **Java reference:** `javac` 21 successfully compiled `ReferenceSolutions.java` and `ReferenceRunner.java`. `SELFTEST.py` replayed **1,760/1,760** Java expected outputs.
- **Independent diagnostic:** `INDEPENDENT_CHECKS.py` independently recomputed **1,760/1,760** expected outputs using separate Python code and standard libraries, with no mismatches.
- **Trivial mutant test:** a constant-null implementation fails at least one vector in **40/40** tasks; more subtle mutants are not systematically measured.
- **Real Java grader smoke:** A correct Java leap-year submission passes all **44/44** J1-02 vectors. An always-false implementation fails and only passes **20/44**.
- **Unperformed:** No Qwen/GLM/LAYA inference. No API calls. No verified secure sandbox. No official external or sealed evaluation. No independent human review of task specifications.

## Threats to validity

1. Reference programs, specifications and fixture selection were authored in the same development workflow; independent computational cross-checking does not make the dataset an uncontaminated holdout.
2. Difficulty labels are qualitative, not calibrated to Qwen Java performance.
3. The prototype grader compiles and runs arbitrary Java in a child process on its own host. `--unsafe-local` is not security isolation, and a separate Kaggle notebook does not make untrusted candidate code safe when grader assets share the process filesystem.
4. Only fixed input/output vectors are included; these do not prove correctness on all admissible inputs.

**Safe intended use:** test worker prompts, Java parsing/compilation, controller dataflow and development scoring. Use a true isolated executable environment and an independently curated holdout for final accuracy claims.
