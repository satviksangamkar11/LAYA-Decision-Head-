# CONFORMANCE: PIPELINE_BROKEN

run: `D:\local model\results\raw\el-bank-java-smoke3-20261018T205908`; expected engine `mxc-wslc`; architecture conformance of a run (files only); not an accuracy result

counts: {"expected": 1, "executed": 1, "ok": 0, "broken": 1, "not_tested": 0, "error": 0, "missing": 0}

## BROKEN tasks (exact reasons)
- **J1-17**: exec_results: silent fallback: exec record 0 ran on backend 'wslc', expected 'mxc-wslc'

## WSLC criteria
```
{
 "zero_not_tested": true,
 "zero_not_tested_no_jdk": true,
 "zero_silent_fallback": false,
 "expected_count": null,
 "executed_count_matches": true,
 "all_receipts_complete": false,
 "engine_status": "OK"
}
```

## Candidate results (SEPARATE from pipeline status)

| task | answer label | stop cause | public |
|---|---|---|---|
| J1-17 | SUPPORTED | decided | {"best": 10, "candidates_passing_all": 2, "candidates_with_examples": 4, "examples": 10} |

## per task

| task | verdict | problems |
|---|---|---|
| J1-17 | BROKEN | exec_results: silent fallback: exec record 0 ran on backend 'wslc', expected 'mxc-wslc' |
