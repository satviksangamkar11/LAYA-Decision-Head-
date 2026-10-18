# BANK40 RESULTS

**development results (D1 waiver), public examples only; hidden grading NOT done**

thinking_policy: `{"ceiling": 4096, "model_glob": "local:qwen2.5-coder-1.5b-instruct", "policy": "model_variant", "profile_id": "qwen2.5-coder-1.5b-instruct"}`; scorer: none (no LAYA); tasks finished 2 / 40; created 2026-10-18T15:19:41Z
jail: {"status": "PASS", "java_home": null, "bwrap_swapped": null}; GPUs: 1; kernel wall seconds: {"pass1": 1.5, "pass2": null, "total": 1.5}
unfinished / errored: []; pending (time guard): []; resumed from earlier version: []
sandbox: mxc; NOT_TESTED (not run, never moved to another sandbox): {}

labels by language: {"python": {"BLOCKED": 2}}

| task | lang | diff | cand | public ex (best/total; cands all-pass) | L2 | clusters pub/ext | critic | label | stop | tokens gen/prompt | seconds | start UTC | end UTC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1-13 | python | medium | 2 | 0/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | budget | 381/300 | 0.5 | 2026-10-18T15:19:40.584Z | 2026-10-18T15:19:41.055Z |
| P1-14 | python | medium | 2 | 0/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | budget | 381/300 | 0.2 | 2026-10-18T15:19:41.062Z | 2026-10-18T15:19:41.275Z |

## Per-stage seconds (all tasks)

| stage | n | total s | p50 s | p95 s |
|---|---|---|---|---|
| generate | 8 | 0.016 | 0.0 | 0.01 |
| model_load | 1 | 0.0 | 0.0 | 0.0 |
| preflight | 1 | 0.703 | 0.703 | 0.703 |
| sandbox_public | 4 | 0.5 | 0.094 | 0.213 |
| task | 2 | 0.64 | 0.32 | 0.425 |

## Controller stages (seconds per task, from the controller's own timeline)

| stage | tasks | total s | p50 s | p95 s |
|---|---|---|---|---|
| commit_verify | 2 | 0.0 | 0.0 | 0.0 |
| discriminate | 2 | 0.0 | 0.0 | 0.0 |
| execute | 2 | 0.5 | 0.25 | 0.32 |
| generate | 2 | 0.032 | 0.016 | 0.016 |
| intake | 2 | 0.108 | 0.054 | 0.089 |
| rank | 2 | 0.0 | 0.0 | 0.0 |
| repair | 2 | 0.0 | 0.0 | 0.0 |

model seconds 0.016, sandbox seconds 0.5, model share 0.031; first event 2026-10-18T15:19:39.832Z, last event 2026-10-18T15:19:41.282Z, timeline wall 1.45 s

## GPU utilisation (nvidia-smi every 30 s)

{
 "per_gpu": {},
 "generate_busy_seconds_by_worker": {
  "w0": 0.0
 },
 "wall_seconds": 1.5,
 "generate_busy_fraction_by_worker": {
  "w0": 0.011
 }
}

## Per-step times (UTC)

**P1-13** (2026-10-18T15:19:40.584Z -> 2026-10-18T15:19:41.055Z): task 15:19:40.609Z..15:19:41.048Z (0.437s); generate 15:19:40.702Z..15:19:40.705Z (0.0s); generate 15:19:40.706Z..15:19:40.707Z (0.0s); sandbox_public 15:19:40.708Z..15:19:40.951Z (0.234s); sandbox_public 15:19:40.954Z..15:19:41.037Z (0.094s); generate 15:19:41.039Z..15:19:41.041Z (0.0s); generate 15:19:41.042Z..15:19:41.043Z (0.0s)
**P1-14** (2026-10-18T15:19:41.062Z -> 2026-10-18T15:19:41.275Z): task 15:19:41.063Z..15:19:41.265Z (0.203s); generate 15:19:41.080Z..15:19:41.082Z (0.0s); generate 15:19:41.082Z..15:19:41.084Z (0.016s); sandbox_public 15:19:41.084Z..15:19:41.165Z (0.078s); sandbox_public 15:19:41.167Z..15:19:41.254Z (0.094s); generate 15:19:41.256Z..15:19:41.258Z (0.0s); generate 15:19:41.259Z..15:19:41.260Z (0.0s)
