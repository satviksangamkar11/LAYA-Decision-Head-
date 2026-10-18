# BANK40 RESULTS

**development results (D1 waiver), public examples only; hidden grading NOT done**

thinking_policy: `{"ceiling": 4096, "model_glob": "local:qwen2.5-coder-1.5b-instruct", "policy": "model_variant", "profile_id": "qwen2.5-coder-1.5b-instruct"}`; scorer: none (no LAYA); tasks finished 3 / 40; created 2026-10-18T15:32:40Z
jail: {"status": "PASS", "java_home": null, "bwrap_swapped": null}; GPUs: 1; kernel wall seconds: {"pass1": 55.7, "pass2": null, "total": 55.7}
unfinished / errored: []; pending (time guard): []; resumed from earlier version: []
sandbox: mxc; NOT_TESTED (not run, never moved to another sandbox): {}

labels by language: {"python": {"SUPPORTED": 3}}

| task | lang | diff | cand | public ex (best/total; cands all-pass) | L2 | clusters pub/ext | critic | label | stop | tokens gen/prompt | seconds | start UTC | end UTC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1-02 | python | easy | 4 | 10/10; 4 | VALIDATED | 1/3 | applied | **SUPPORTED** | decided | 315/250 | 49.4 | 2026-10-18T15:31:44.991Z | 2026-10-18T15:32:34.411Z |
| P1-03 | python | easy | 2 | 10/10; 1 | INSUFFICIENT | None/None | no | **SUPPORTED** | decided | 381/300 | 0.5 | 2026-10-18T15:32:34.425Z | 2026-10-18T15:32:34.971Z |
| P1-13 | python | medium | 2 | 10/10; 2 | INSUFFICIENT | 1/2 | applied | **SUPPORTED** | decided | 315/250 | 5.0 | 2026-10-18T15:32:34.983Z | 2026-10-18T15:32:40.011Z |

## Per-stage seconds (all tasks)

| stage | n | total s | p50 s | p95 s |
|---|---|---|---|---|
| generate | 16 | 0.046 | 0.0 | 0.015 |
| glm_critic | 2 | 0.14 | 0.07 | 0.105 |
| model_load | 1 | 0.0 | 0.0 | 0.0 |
| preflight | 1 | 0.625 | 0.625 | 0.625 |
| sandbox_experiment | 334 | 51.276 | 0.156 | 0.188 |
| sandbox_public | 17 | 2.672 | 0.156 | 0.194 |
| task | 3 | 26.062 | 2.359 | 21.091 |
| task_after_critic | 2 | 28.719 | 14.36 | 24.921 |

## Controller stages (seconds per task, from the controller's own timeline)

| stage | tasks | total s | p50 s | p95 s |
|---|---|---|---|---|
| cluster | 2 | 8.907 | 4.454 | 6.844 |
| commit_verify | 3 | 0.516 | 0.172 | 0.172 |
| discriminate | 3 | 1.046 | 0.312 | 0.692 |
| execute | 3 | 1.236 | 0.329 | 0.582 |
| generate | 3 | 0.015 | 0.0 | 0.013 |
| intake | 3 | 0.062 | 0.016 | 0.029 |
| l2_generate | 3 | 0.016 | 0.0 | 0.014 |
| l2_validate | 3 | 17.437 | 0.0 | 15.693 |
| rank | 3 | 0.0 | 0.0 | 0.0 |
| repair | 1 | 0.015 | 0.015 | 0.015 |

model seconds 0.046, sandbox seconds 53.948, model share 0.0009; first event 2026-10-18T15:31:44.319Z, last event 2026-10-18T15:32:40.027Z, timeline wall 55.707 s

## GPU utilisation (nvidia-smi every 30 s)

{
 "per_gpu": {},
 "generate_busy_seconds_by_worker": {
  "w0": 0.0
 },
 "wall_seconds": 55.7,
 "generate_busy_fraction_by_worker": {
  "w0": 0.001
 }
}

## Per-step times (UTC)

**P1-02** (2026-10-18T15:31:44.991Z -> 2026-10-18T15:32:34.411Z): task 15:31:45.012Z..15:32:08.185Z (23.172s); generate 15:31:45.100Z..15:31:45.102Z (0.0s); generate 15:31:45.103Z..15:31:45.104Z (0.0s); sandbox_public 15:31:45.104Z..15:31:45.392Z (0.281s); sandbox_public 15:31:45.394Z..15:31:45.533Z (0.14s); sandbox_public 15:31:45.536Z..15:31:45.669Z (0.125s); sandbox_public 15:31:45.671Z..15:31:45.816Z (0.141s); generate 15:31:45.817Z..15:31:45.820Z (0.0s); sandbox_experiment 15:31:45.820Z..15:31:45.961Z (0.141s); sandbox_experiment 15:31:45.963Z..15:31:46.102Z (0.141s); sandbox_experiment 15:31:46.103Z..15:31:46.243Z (0.14s); sandbox_experiment 15:31:46.244Z..15:31:46.395Z (0.141s); sandbox_experiment 15:31:46.397Z..15:31:46.532Z (0.125s); sandbox_experiment 15:31:46.533Z..15:31:46.667Z (0.141s); sandbox_experiment 15:31:46.668Z..15:31:46.812Z (0.141s); sandbox_experiment 15:31:46.813Z..15:31:46.959Z (0.156s); sandbox_experiment 15:31:46.959Z..15:31:47.104Z (0.141s); sandbox_experiment 15:31:47.105Z..15:31:47.246Z (0.14s); sandbox_experiment 15:31:47.247Z..15:31:47.391Z (0.141s); sandbox_experiment 15:31:47.392Z..15:31:47.545Z (0.156s); sandbox_experiment 15:31:47.546Z..15:31:47.685Z (0.141s); sandbox_experiment 15:31:47.686Z..15:31:47.820Z (0.14s); sandbox_experiment 15:31:47.821Z..15:31:47.963Z (0.141s); sandbox_experiment 15:31:47.964Z..15:31:48.105Z (0.141s); sandbox_experiment 15:31:48.106Z..15:31:48.249Z (0.14s); sandbox_experiment 15:31:48.250Z..15:31:48.391Z (0.141s); sandbox_experiment 15:31:48.393Z..15:31:48.531Z (0.14s); sandbox_experiment 15:31:48.532Z..15:31:48.678Z (0.157s); sandbox_experiment 15:31:48.681Z..15:31:48.825Z (0.14s); sandbox_experiment 15:31:48.826Z..15:31:48.969Z (0.141s); sandbox_experiment 15:31:48.971Z..15:31:49.118Z (0.156s); sandbox_experiment 15:31:49.119Z..15:31:49.264Z (0.141s); sandbox_experiment 15:31:49.265Z..15:31:49.471Z (0.203s); sandbox_experiment 15:31:49.473Z..15:31:49.620Z (0.156s); sandbox_experiment 15:31:49.621Z..15:31:49.762Z (0.141s); sandbox_experiment 15:31:49.763Z..15:31:49.906Z (0.14s); sandbox_experiment 15:31:49.907Z..15:31:50.057Z (0.157s); sandbox_experiment 15:31:50.058Z..15:31:50.198Z (0.14s); sandbox_experiment 15:31:50.199Z..15:31:50.341Z (0.141s); sandbox_experiment 15:31:50.343Z..15:31:50.480Z (0.141s) ...
**P1-03** (2026-10-18T15:32:34.425Z -> 2026-10-18T15:32:34.971Z): task 15:32:34.430Z..15:32:34.959Z (0.531s); generate 15:32:34.451Z..15:32:34.453Z (0.0s); generate 15:32:34.455Z..15:32:34.456Z (0.0s); sandbox_public 15:32:34.456Z..15:32:34.608Z (0.157s); generate 15:32:34.612Z..15:32:34.614Z (0.015s); sandbox_public 15:32:34.615Z..15:32:34.790Z (0.172s); generate 15:32:34.793Z..15:32:34.795Z (0.0s); sandbox_public 15:32:34.796Z..15:32:34.950Z (0.156s)
**P1-13** (2026-10-18T15:32:34.983Z -> 2026-10-18T15:32:40.011Z): task 15:32:34.987Z..15:32:37.346Z (2.359s); generate 15:32:35.008Z..15:32:35.010Z (0.0s); generate 15:32:35.011Z..15:32:35.013Z (0.0s); sandbox_public 15:32:35.014Z..15:32:35.168Z (0.156s); sandbox_public 15:32:35.171Z..15:32:35.325Z (0.156s); generate 15:32:35.329Z..15:32:35.331Z (0.0s); sandbox_experiment 15:32:35.332Z..15:32:35.547Z (0.219s); sandbox_experiment 15:32:35.548Z..15:32:35.714Z (0.172s); sandbox_experiment 15:32:35.715Z..15:32:35.873Z (0.156s); sandbox_experiment 15:32:35.874Z..15:32:36.031Z (0.156s); sandbox_experiment 15:32:36.032Z..15:32:36.195Z (0.172s); sandbox_experiment 15:32:36.196Z..15:32:36.356Z (0.157s); sandbox_experiment 15:32:36.357Z..15:32:36.514Z (0.156s); sandbox_experiment 15:32:36.515Z..15:32:36.671Z (0.156s); sandbox_experiment 15:32:36.673Z..15:32:36.888Z (0.219s); sandbox_experiment 15:32:36.889Z..15:32:37.034Z (0.14s); sandbox_experiment 15:32:37.035Z..15:32:37.183Z (0.157s); sandbox_experiment 15:32:37.184Z..15:32:37.333Z (0.156s); glm_critic 15:32:37.347Z..15:32:37.364Z (0.031s); task_after_critic 15:32:37.365Z..15:32:39.994Z (2.625s); generate 15:32:37.392Z..15:32:37.394Z (0.0s); generate 15:32:37.395Z..15:32:37.397Z (0.015s); sandbox_public 15:32:37.398Z..15:32:37.547Z (0.141s); sandbox_public 15:32:37.550Z..15:32:37.704Z (0.156s); generate 15:32:37.707Z..15:32:37.709Z (0.016s); sandbox_experiment 15:32:37.710Z..15:32:37.860Z (0.141s); sandbox_experiment 15:32:37.861Z..15:32:38.013Z (0.156s); sandbox_experiment 15:32:38.015Z..15:32:38.159Z (0.14s); sandbox_experiment 15:32:38.160Z..15:32:38.316Z (0.157s); sandbox_experiment 15:32:38.318Z..15:32:38.461Z (0.141s); sandbox_experiment 15:32:38.462Z..15:32:38.611Z (0.141s); sandbox_experiment 15:32:38.612Z..15:32:38.759Z (0.156s); sandbox_experiment 15:32:38.760Z..15:32:38.927Z (0.172s); sandbox_experiment 15:32:38.928Z..15:32:39.077Z (0.14s); sandbox_experiment 15:32:39.079Z..15:32:39.229Z (0.157s); sandbox_experiment 15:32:39.230Z..15:32:39.371Z (0.14s); sandbox_experiment 15:32:39.373Z..15:32:39.517Z (0.141s); sandbox_experiment 15:32:39.520Z..15:32:39.670Z (0.156s); sandbox_experiment 15:32:39.672Z..15:32:39.824Z (0.156s); sandbox_public 15:32:39.826Z..15:32:39.981Z (0.157s)
