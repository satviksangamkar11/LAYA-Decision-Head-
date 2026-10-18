# BANK40 RESULTS

**development results (D1 waiver), public examples only; hidden grading NOT done**

thinking_policy: `{"ceiling": 4096, "model_glob": "local:qwen2.5-coder-1.5b-instruct", "policy": "model_variant", "profile_id": "qwen2.5-coder-1.5b-instruct"}`; scorer: none (no LAYA); tasks finished 3 / 40; created 2026-10-18T15:31:25Z
jail: {"status": "PASS", "java_home": null, "bwrap_swapped": null}; GPUs: 1; kernel wall seconds: {"pass1": 1.8, "pass2": null, "total": 1.8}
unfinished / errored: []; pending (time guard): []; resumed from earlier version: []
sandbox: mxc; NOT_TESTED (not run, never moved to another sandbox): {}

labels by language: {"python": {"BLOCKED": 3}}

| task | lang | diff | cand | public ex (best/total; cands all-pass) | L2 | clusters pub/ext | critic | label | stop | tokens gen/prompt | seconds | start UTC | end UTC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1-02 | python | easy | 4 | 0/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | budget | 381/300 | 0.6 | 2026-10-18T15:31:23.999Z | 2026-10-18T15:31:24.622Z |
| P1-03 | python | easy | 2 | 0/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | same_failure_twice | 315/250 | 0.2 | 2026-10-18T15:31:24.629Z | 2026-10-18T15:31:24.835Z |
| P1-13 | python | medium | 2 | 0/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | budget | 381/300 | 0.2 | 2026-10-18T15:31:24.842Z | 2026-10-18T15:31:25.066Z |

## Per-stage seconds (all tasks)

| stage | n | total s | p50 s | p95 s |
|---|---|---|---|---|
| generate | 11 | 0.031 | 0.0 | 0.015 |
| model_load | 1 | 0.016 | 0.016 | 0.016 |
| preflight | 1 | 0.672 | 0.672 | 0.672 |
| sandbox_public | 8 | 0.844 | 0.093 | 0.185 |
| task | 3 | 1.0 | 0.219 | 0.556 |

## Controller stages (seconds per task, from the controller's own timeline)

| stage | tasks | total s | p50 s | p95 s |
|---|---|---|---|---|
| commit_verify | 3 | 0.0 | 0.0 | 0.0 |
| discriminate | 3 | 0.0 | 0.0 | 0.0 |
| execute | 3 | 0.844 | 0.172 | 0.467 |
| generate | 3 | 0.015 | 0.0 | 0.013 |
| intake | 3 | 0.11 | 0.016 | 0.073 |
| rank | 3 | 0.0 | 0.0 | 0.0 |
| repair | 3 | 0.016 | 0.0 | 0.014 |

model seconds 0.031, sandbox seconds 0.844, model share 0.0354; first event 2026-10-18T15:31:23.272Z, last event 2026-10-18T15:31:25.073Z, timeline wall 1.801 s

## GPU utilisation (nvidia-smi every 30 s)

{
 "per_gpu": {},
 "generate_busy_seconds_by_worker": {
  "w0": 0.0
 },
 "wall_seconds": 1.8,
 "generate_busy_fraction_by_worker": {
  "w0": 0.017
 }
}

## Per-step times (UTC)

**P1-02** (2026-10-18T15:31:23.999Z -> 2026-10-18T15:31:24.622Z): task 15:31:24.018Z..15:31:24.613Z (0.594s); generate 15:31:24.107Z..15:31:24.109Z (0.0s); generate 15:31:24.110Z..15:31:24.111Z (0.0s); sandbox_public 15:31:24.112Z..15:31:24.341Z (0.234s); sandbox_public 15:31:24.343Z..15:31:24.426Z (0.094s); sandbox_public 15:31:24.429Z..15:31:24.514Z (0.078s); sandbox_public 15:31:24.517Z..15:31:24.601Z (0.094s); generate 15:31:24.603Z..15:31:24.605Z (0.0s); generate 15:31:24.606Z..15:31:24.607Z (0.0s)
**P1-03** (2026-10-18T15:31:24.629Z -> 2026-10-18T15:31:24.835Z): task 15:31:24.630Z..15:31:24.827Z (0.187s); generate 15:31:24.646Z..15:31:24.647Z (0.0s); generate 15:31:24.649Z..15:31:24.651Z (0.0s); sandbox_public 15:31:24.651Z..15:31:24.733Z (0.079s); generate 15:31:24.736Z..15:31:24.737Z (0.0s); sandbox_public 15:31:24.737Z..15:31:24.821Z (0.093s)
**P1-13** (2026-10-18T15:31:24.842Z -> 2026-10-18T15:31:25.066Z): task 15:31:24.843Z..15:31:25.057Z (0.219s); generate 15:31:24.861Z..15:31:24.863Z (0.0s); generate 15:31:24.864Z..15:31:24.866Z (0.015s); sandbox_public 15:31:24.867Z..15:31:24.961Z (0.094s); sandbox_public 15:31:24.963Z..15:31:25.046Z (0.078s); generate 15:31:25.049Z..15:31:25.050Z (0.0s); generate 15:31:25.051Z..15:31:25.053Z (0.016s)
