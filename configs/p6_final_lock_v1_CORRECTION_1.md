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

## Audit record (added 2026-10-22, after grading)
- **Authority:** this document is authoritative for the single field cf.tests_manifest_sha256. Every other lock field stands as written.
- **Corrected grade plan:** el-p6-grade-input-v1/P6_GRADE_PLAN.json, sha256 d39aa5739c0047f2c00f7ea9fc9b747ded67d9249368b431a09b0e0c88742d5b
  (bytes as served by Kaggle). Its tests_manifest_sha256 = 584cd46a1d266911be194ca9eb0789d6ff63a065596d450a324bcac43ce25955.
- **P6 Codeforces grading shards:** one, kernel el-p6-cf-grade-v1 version 3, verdict OK, 216/216 answers. The grader refuses unless the Kaggle manifest hashes to the
  plan's value, so a completed run proves that manifest was used. Attempts 1 and 2 graded nothing (file path; refused manifest).
- **Every later grading run uses the same manifest hash:** P7 candidate grading el-p7-cf-cand-grade-v1, and the infra re-grades el-infra-regrade-p6-k1 and
  el-infra-regrade-p7-k1.
