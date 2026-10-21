# Correction 1 to configs/p6_final_lock_v1.json (2026-10-22, after generation, before any Codeforces grade)

Field: cf.tests_manifest_sha256.
- **Lock value:** 89d48ef7e96d1fb5…, the hash of the Windows working copy of results/raw/p6-cf-validation-v1/p6_cf_MANIFEST.json after git's LF-to-CRLF
  conversion. It is not the hash of the manifest itself.
- **Correct value:** 584cd46a1d266911be194ca9eb0789d6ff63a065596d450a324bcac43ce25955, the bytes of the Kaggle export manifest. Shown by two independent copies:
  - private/p6-cf-v1/p6_cf/MANIFEST.json, downloaded from Kaggle and never git-managed;
  - the committed (LF) git blob of the same file.

  Both hash to this value.
- **How it was found:** the Kaggle CPU grader refused with REFUSED_TESTS_MANIFEST_MISMATCH before grading anything (results/raw/p6-final-v1/grade_cf_attempt2/RUN_SUMMARY.json).
- **Scope:** the corrected value is used only in the grade plan (results/raw/p6-final-v1/grade_cf_input/P6_GRADE_PLAN.json). The tests themselves are unchanged
  and are still hash-checked file by file against that manifest.
- **What is not affected:** no generation input, task, arm or threshold. The lock file itself is not edited (create-only).
