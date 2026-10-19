# BANK40 RESULTS

**development results (D1 waiver), public examples only; hidden grading NOT done**

thinking_policy: `{"ceiling": 4096, "model_glob": "local:qwen2.5-coder-1.5b-instruct", "policy": "model_variant", "profile_id": "qwen2.5-coder-1.5b-instruct"}`; scorer: none (no LAYA); tasks finished 1 / 40; created 2026-10-18T21:13:01Z
jail: {"status": "PASS", "java_home": null, "bwrap_swapped": null}; GPUs: 1; kernel wall seconds: {"pass1": 832.7, "pass2": null, "total": 832.7}
unfinished / errored: []; pending (time guard): []; resumed from earlier version: []
sandbox: mxc-wslc; NOT_TESTED (not run, never moved to another sandbox): {}

labels by language: {"java": {"SUPPORTED": 1}}

| task | lang | diff | cand | public ex (best/total; cands all-pass) | L2 | clusters pub/ext | critic | label | stop | tokens gen/prompt | seconds | start UTC | end UTC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1-17 | java | medium | 4 | 10/10; 2 | INSUFFICIENT | 1/2 | applied | **SUPPORTED** | decided | 980/1800 | 808.4 | 2026-10-18T20:59:33.053Z | 2026-10-18T21:13:01.446Z |

## Per-stage seconds (all tasks)

| stage | n | total s | p50 s | p95 s |
|---|---|---|---|---|
| engine_start | 1 | 7.485 | 7.485 | 7.485 |
| generate | 6 | 46.844 | 6.649 | 17.144 |
| glm_critic | 1 | 507.063 | 507.063 | 507.063 |
| model_load | 1 | 16.687 | 16.687 | 16.687 |
| preflight | 1 | 0.047 | 0.047 | 0.047 |
| sandbox_experiment | 26 | 141.781 | 5.844 | 6.184 |
| sandbox_public | 9 | 112.016 | 13.781 | 16.028 |
| task | 1 | 168.593 | 168.593 | 168.593 |
| task_after_critic | 1 | 132.546 | 132.546 | 132.546 |

## Controller stages (seconds per task, from the controller's own timeline)

| stage | tasks | total s | p50 s | p95 s |
|---|---|---|---|---|
| cluster | 1 | 70.828 | 70.828 | 70.828 |
| commit_verify | 1 | 13.828 | 13.828 | 13.828 |
| discriminate | 1 | 0.0 | 0.0 | 0.0 |
| execute | 1 | 47.844 | 47.844 | 47.844 |
| generate | 1 | 0.016 | 0.016 | 0.016 |
| intake | 1 | 0.015 | 0.015 | 0.015 |
| l2_generate | 1 | 0.0 | 0.0 | 0.0 |
| l2_validate | 1 | 0.015 | 0.015 | 0.015 |
| rank | 1 | 0.0 | 0.0 | 0.0 |

model seconds 46.844, sandbox seconds 253.797, model share 0.1558; first event 2026-10-18T20:59:08.783Z, last event 2026-10-18T21:13:01.455Z, timeline wall 832.671 s

## GPU utilisation (nvidia-smi every 30 s)

{
 "per_gpu": {
  "0": {
   "samples": 27,
   "mean_util_pct": 3.4,
   "p50": 0.0,
   "p95": 15.5,
   "idle_samples_pct": 88.9
  }
 },
 "generate_busy_seconds_by_worker": {
  "w0": 46.8
 },
 "wall_seconds": 832.7,
 "generate_busy_fraction_by_worker": {
  "w0": 0.056
 }
}

## Per-step times (UTC)

**J1-17** (2026-10-18T20:59:33.053Z -> 2026-10-18T21:13:01.446Z): task 20:59:33.217Z..21:02:21.812Z (168.593s); generate 20:59:33.315Z..20:59:46.609Z (13.281s); generate 20:59:46.610Z..21:00:02.639Z (16.032s); sandbox_public 21:00:02.640Z..21:00:19.873Z (17.234s); sandbox_public 21:00:19.876Z..21:00:25.692Z (5.812s); sandbox_public 21:00:25.695Z..21:00:39.237Z (13.547s); sandbox_public 21:00:39.241Z..21:00:53.029Z (13.781s); generate 21:00:53.032Z..21:01:10.562Z (17.515s); sandbox_experiment 21:01:10.564Z..21:01:16.645Z (6.078s); sandbox_experiment 21:01:16.758Z..21:01:23.075Z (6.312s); sandbox_experiment 21:01:23.077Z..21:01:29.034Z (5.969s); sandbox_experiment 21:01:29.035Z..21:01:35.000Z (5.953s); sandbox_experiment 21:01:35.002Z..21:01:40.884Z (5.875s); sandbox_experiment 21:01:40.885Z..21:01:46.641Z (5.75s); sandbox_experiment 21:01:46.642Z..21:01:52.459Z (5.813s); sandbox_experiment 21:01:52.460Z..21:01:58.314Z (5.859s); sandbox_experiment 21:01:58.315Z..21:02:04.160Z (5.844s); sandbox_experiment 21:02:04.161Z..21:02:10.001Z (5.844s); sandbox_experiment 21:02:10.002Z..21:02:15.819Z (5.812s); sandbox_experiment 21:02:15.820Z..21:02:21.671Z (5.844s); glm_critic 21:02:21.813Z..21:10:48.879Z (507.063s); task_after_critic 21:10:48.881Z..21:13:01.429Z (132.546s); generate 21:10:48.905Z..21:10:48.907Z (0.0s); generate 21:10:48.907Z..21:10:48.909Z (0.016s); sandbox_public 21:10:48.910Z..21:11:03.132Z (14.219s); sandbox_public 21:11:03.135Z..21:11:09.100Z (5.968s); sandbox_public 21:11:09.104Z..21:11:23.092Z (13.985s); sandbox_public 21:11:23.100Z..21:11:36.760Z (13.657s); generate 21:11:36.763Z..21:11:36.765Z (0.0s); sandbox_experiment 21:11:36.767Z..21:11:42.652Z (5.875s); sandbox_experiment 21:11:42.654Z..21:11:48.861Z (6.219s); sandbox_experiment 21:11:48.862Z..21:11:54.914Z (6.047s); sandbox_experiment 21:11:54.915Z..21:12:00.750Z (5.828s); sandbox_experiment 21:12:00.751Z..21:12:06.546Z (5.797s); sandbox_experiment 21:12:06.546Z..21:12:12.287Z (5.75s); sandbox_experiment 21:12:12.288Z..21:12:17.977Z (5.687s); sandbox_experiment 21:12:17.978Z..21:12:23.808Z (5.828s); sandbox_experiment 21:12:23.809Z..21:12:29.557Z (5.75s); sandbox_experiment 21:12:29.558Z..21:12:35.636Z (6.079s); sandbox_experiment 21:12:35.637Z..21:12:41.633Z (6.0s) ...
