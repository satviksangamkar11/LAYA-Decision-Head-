# P6 run-output schema v1 (fixed before the runner, graders and analysis are written) — 2026-10-21

One run directory per shard (`p6_<shard>/`), written by `tools/el_p6_run.py`. All files JSON / JSON lines, UTF-8.

- `RUN_CONFIG.json`: {"protocol": "P6_PROTOCOL_v1", "cohort": "cf" | "contract", "tasks": [...], "arms": ["A","B","V2"], "G": 6000, "v2min": "<VERSION>",
  "template": "lcb_generic_v1" (cf) | "qwen_sample_v2+v2b" (contract), "exec": "kaggle"|"mxc", "fake": bool, "tool_sha256": "...", "started_utc": "..."}
- `RESULTS.jsonl`, one row per (task, arm):
  {"cohort", "task", "arm": "A"|"B"|"V2",
   "label": A: "ANSWERED" | "NO_CODE"; B and V2: "SUPPORTED" | "BLOCKED",
   "answer_sha256": sha256 of the answer source or null,
   "public_pass": bool or null (A only: did the seed-1 answer pass every public example; recorded, never used to replace it),
   "tokens_generated_logical": generated tokens of every generation the arm USED (shared ones included),
   "tokens_generated_own": generated tokens of the cache misses the arm itself caused (A: 0, its sample comes from the shared round 1),
   "tokens_prompt_logical", "calls": int, "candidates": int, "rounds": int, "seconds": float,
   "gate_verdicts": list of verdicts on public-passing candidates (V2 only; [] otherwise), "rejected_by_contract": list of artifact ids (V2), "stop": "accepted"|"budget"|"a_single_sample"}
  plus one row per task with "arm": "_task": {"cohort", "task", "arm": "_task", "wall_seconds", "round1_tokens_generated"}.
- `ANSWERS.jsonl`: {"cohort", "task", "arm", "source": str or null}. The ONLY file a grader reads from the run.
- `CALLS.jsonl` (shared cache log): {"task", "arm", "seed", "purpose", "template", "prompt_sha256", "max_new_tokens", "reused", "tokens_prompt", "tokens_generated",
  "actual_seconds", "extract_status"}.
- `IDENTITY.jsonl` (cf cohort only): {"task", "identical": bool, "same_answer": bool, "same_requests": bool, "same_tokens": bool, "verdicts": [...], "all_unavailable": bool}.
- `EXEC.jsonl`: per execution call (P5 ExecLog format).
- `DONE.json`: {"finished_utc"}.

Grades (written by the graders, create-only, one file per run directory):
- cf: `GRADED_cf.jsonl` (Kaggle CPU, validated tests of the v1 export, LcbRunner stdin, 6 s per test, all tests run):
  {"task", "arm", "grade": "PASS"|"FAIL"|"NO_ANSWER", "passed", "total", "first_fail": status or null, "timeouts": int}
- contract: `GRADED_contract_private.jsonl` (PC, MXC trusted verifier, private blind set private/p6-eval-v1/cases_p6.json, sha256 checked against the P6 lock):
  {"task", "arm", "grade": "PASS"|"FAIL"|"NO_ANSWER", "passed", "total", "first_fail"}

P6 lock (`configs/p6_final_lock_v1.json`, created after the timing smoke, create-only):
{"protocol": "P6_PROTOCOL_v1", "protocol_sha256_committed": "...", "cf": {"candidates_ordered": [72 ids in sha256("P6-2026-10-21:"+id) order], "N": int,
 "final_ids": [...first N...], "tests_manifest_sha256": "<sha256 of the v1 export MANIFEST.json>", "statements_sha256": "..."},
 "contract": {"ids": ["D01".."D20"], "frozen_sha256_file": "configs/p6_contract_tasks_v1/FROZEN_SHA256.txt",
 "grading": {"location": "private/p6-eval-v1/cases_p6.json", "cases_sha256": "d8c57f33fd16ddeb0f0b85a9613a2cefb1ac8d60cc2387110faaeb723e4fd780"}},
 "model": {...}, "G": 6000, "arms": ["A","B","V2"]}
