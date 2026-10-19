# CONFORMANCE: PIPELINE_COMPLETE

run: `D:\local model\results\raw\el-bank-java5hard-wslc-20261018T213058`; expected engine `mxc-wslc`; architecture conformance of a run (files only); not an accuracy result

counts: {"expected": 5, "executed": 5, "ok": 5, "broken": 0, "not_tested": 0, "error": 0, "missing": 0}

## WSLC criteria
```
{
 "zero_not_tested": true,
 "zero_not_tested_no_jdk": true,
 "zero_silent_fallback": true,
 "expected_count": null,
 "executed_count_matches": true,
 "all_receipts_complete": true,
 "engine_status": "OK"
}
```

## Candidate results (SEPARATE from pipeline status)

| task | answer label | stop cause | public |
|---|---|---|---|
| J1-31 | SUPPORTED | decided | {"best": 10, "candidates_passing_all": 1, "candidates_with_examples": 6, "examples": 10} |
| J1-32 | BLOCKED | budget | {"best": 1, "candidates_passing_all": 0, "candidates_with_examples": 6, "examples": 5} |
| J1-33 | BLOCKED | budget | {"best": 0, "candidates_passing_all": 0, "candidates_with_examples": 5, "examples": 5} |
| J1-34 | BLOCKED | budget | {"best": 2, "candidates_passing_all": 0, "candidates_with_examples": 5, "examples": 5} |
| J1-35 | BLOCKED | budget | {"best": 2, "candidates_passing_all": 0, "candidates_with_examples": 5, "examples": 5} |

## per task

| task | verdict | problems |
|---|---|---|
| J1-31 | PIPELINE_OK |  |
| J1-32 | PIPELINE_OK |  |
| J1-33 | PIPELINE_OK |  |
| J1-34 | PIPELINE_OK |  |
| J1-35 | PIPELINE_OK |  |
