# BANK40 RESULTS

**development results (D1 waiver), public examples only; hidden grading NOT done**

thinking_policy: `{"ceiling": 4096, "model_glob": "local:qwen2.5-coder-1.5b-instruct", "policy": "model_variant", "profile_id": "qwen2.5-coder-1.5b-instruct"}`; scorer: none (no LAYA); tasks finished 5 / 40; created 2026-10-18T21:49:21Z
jail: {"status": "PASS", "java_home": null, "bwrap_swapped": null}; GPUs: 1; kernel wall seconds: {"pass1": 1102.2, "pass2": null, "total": 1102.2}
unfinished / errored: []; pending (time guard): []; resumed from earlier version: []
sandbox: mxc-wslc; NOT_TESTED (not run, never moved to another sandbox): {}

labels by language: {"java": {"BLOCKED": 4, "SUPPORTED": 1}}

| task | lang | diff | cand | public ex (best/total; cands all-pass) | L2 | clusters pub/ext | critic | label | stop | tokens gen/prompt | seconds | start UTC | end UTC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1-31 | java | hard | 6 | 10/10; 1 | INSUFFICIENT | None/None | no | **SUPPORTED** | decided | 6248/3308 | 391.5 | 2026-10-18T21:31:26.611Z | 2026-10-18T21:37:58.154Z |
| J1-32 | java | hard | 6 | 1/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | budget | 1644/2304 | 124.8 | 2026-10-18T21:37:58.165Z | 2026-10-18T21:40:02.990Z |
| J1-33 | java | hard | 5 | 0/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | budget | 5746/3278 | 295.8 | 2026-10-18T21:40:02.999Z | 2026-10-18T21:44:58.874Z |
| J1-34 | java | hard | 5 | 2/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | budget | 1713/3153 | 124.4 | 2026-10-18T21:44:58.885Z | 2026-10-18T21:47:03.318Z |
| J1-35 | java | hard | 5 | 2/5; 0 | INSUFFICIENT | None/None | no | **BLOCKED** | budget | 2180/2708 | 138.1 | 2026-10-18T21:47:03.329Z | 2026-10-18T21:49:21.424Z |

## Per-stage seconds (all tasks)

| stage | n | total s | p50 s | p95 s |
|---|---|---|---|---|
| engine_start | 1 | 10.641 | 10.641 | 10.641 |
| generate | 21 | 842.641 | 19.828 | 200.593 |
| model_load | 1 | 16.562 | 16.562 | 16.562 |
| preflight | 1 | 0.063 | 0.063 | 0.063 |
| sandbox_public | 28 | 230.268 | 7.046 | 15.065 |
| task | 5 | 1073.874 | 137.937 | 372.188 |

## Controller stages (seconds per task, from the controller's own timeline)

| stage | tasks | total s | p50 s | p95 s |
|---|---|---|---|---|
| commit_verify | 5 | 17.796 | 0.125 | 13.866 |
| discriminate | 5 | 0.0 | 0.0 | 0.0 |
| execute | 5 | 213.189 | 39.657 | 61.572 |
| generate | 5 | 445.578 | 55.344 | 193.484 |
| intake | 5 | 0.233 | 0.031 | 0.093 |
| l2_generate | 1 | 204.063 | 204.063 | 204.063 |
| l2_validate | 1 | 0.0 | 0.0 | 0.0 |
| rank | 5 | 0.0 | 0.0 | 0.0 |
| repair | 5 | 193.015 | 35.859 | 48.2 |

model seconds 842.641, sandbox seconds 230.268, model share 0.7854; first event 2026-10-18T21:30:59.307Z, last event 2026-10-18T21:49:21.435Z, timeline wall 1102.128 s

## GPU utilisation (nvidia-smi every 30 s)

{
 "per_gpu": {
  "0": {
   "samples": 36,
   "mean_util_pct": 46.7,
   "p50": 54.0,
   "p95": 77.5,
   "idle_samples_pct": 25.0
  }
 },
 "generate_busy_seconds_by_worker": {
  "w0": 842.6
 },
 "wall_seconds": 1102.1,
 "generate_busy_fraction_by_worker": {
  "w0": 0.765
 }
}

## Per-step times (UTC)

**J1-31** (2026-10-18T21:31:26.611Z -> 2026-10-18T21:37:58.154Z): task 21:31:26.821Z..21:37:58.129Z (391.313s); generate 21:31:26.929Z..21:31:36.944Z (10.016s); generate 21:31:36.946Z..21:32:35.920Z (58.969s); sandbox_public 21:32:35.921Z..21:32:45.820Z (9.906s); sandbox_public 21:32:45.824Z..21:32:51.900Z (6.078s); sandbox_public 21:32:51.903Z..21:33:05.985Z (14.094s); sandbox_public 21:33:05.989Z..21:33:12.138Z (6.141s); generate 21:33:12.142Z..21:33:30.351Z (18.203s); sandbox_public 21:33:30.353Z..21:33:44.574Z (14.219s); generate 21:33:44.578Z..21:34:02.231Z (17.656s); sandbox_public 21:34:02.232Z..21:34:16.774Z (14.547s); generate 21:34:16.778Z..21:37:40.830Z (204.063s); sandbox_public 21:37:40.833Z..21:37:57.987Z (17.156s)
**J1-32** (2026-10-18T21:37:58.165Z -> 2026-10-18T21:40:02.990Z): task 21:37:58.295Z..21:40:02.967Z (124.672s); generate 21:37:58.319Z..21:38:10.548Z (12.234s); generate 21:38:10.549Z..21:38:44.726Z (34.172s); sandbox_public 21:38:44.727Z..21:38:54.041Z (9.313s); sandbox_public 21:38:54.044Z..21:39:02.747Z (8.703s); sandbox_public 21:39:02.750Z..21:39:05.794Z (3.047s); sandbox_public 21:39:05.796Z..21:39:14.721Z (8.937s); generate 21:39:14.723Z..21:39:27.627Z (12.907s); sandbox_public 21:39:27.629Z..21:39:36.508Z (8.875s); generate 21:39:36.511Z..21:39:53.856Z (17.343s); sandbox_public 21:39:53.857Z..21:40:02.833Z (8.985s)
**J1-33** (2026-10-18T21:40:02.999Z -> 2026-10-18T21:44:58.874Z): task 21:40:03.170Z..21:44:58.856Z (295.687s); generate 21:40:03.196Z..21:40:27.206Z (24.016s); generate 21:40:27.207Z..21:43:47.800Z (200.593s); sandbox_public 21:43:47.801Z..21:43:54.616Z (6.813s); sandbox_public 21:43:54.619Z..21:44:01.700Z (7.078s); sandbox_public 21:44:01.704Z..21:44:05.467Z (3.766s); generate 21:44:05.471Z..21:44:28.891Z (23.407s); sandbox_public 21:44:28.892Z..21:44:35.467Z (6.563s); generate 21:44:35.472Z..21:44:55.310Z (19.828s); sandbox_public 21:44:55.311Z..21:44:58.707Z (3.407s)
**J1-34** (2026-10-18T21:44:58.885Z -> 2026-10-18T21:47:03.318Z): task 21:44:59.044Z..21:47:03.302Z (124.265s); generate 21:44:59.070Z..21:45:15.584Z (16.516s); generate 21:45:15.584Z..21:45:49.309Z (33.718s); sandbox_public 21:45:49.310Z..21:45:53.109Z (3.797s); sandbox_public 21:45:53.114Z..21:45:57.168Z (4.047s); sandbox_public 21:45:57.172Z..21:46:03.741Z (6.578s); sandbox_public 21:46:03.745Z..21:46:13.618Z (9.875s); generate 21:46:13.623Z..21:46:31.042Z (17.422s); sandbox_public 21:46:31.043Z..21:46:46.380Z (15.344s); generate 21:46:46.383Z..21:47:03.179Z (16.796s)
**J1-35** (2026-10-18T21:47:03.329Z -> 2026-10-18T21:49:21.424Z): task 21:47:03.460Z..21:49:21.405Z (137.937s); generate 21:47:03.488Z..21:47:29.138Z (25.641s); generate 21:47:29.139Z..21:47:58.840Z (29.703s); sandbox_public 21:47:58.841Z..21:48:05.852Z (7.015s); sandbox_public 21:48:05.856Z..21:48:09.743Z (3.891s); sandbox_public 21:48:09.747Z..21:48:16.351Z (6.609s); sandbox_public 21:48:16.354Z..21:48:28.153Z (11.797s); generate 21:48:28.157Z..21:48:58.041Z (29.875s); sandbox_public 21:48:58.043Z..21:49:01.720Z (3.687s); generate 21:49:01.725Z..21:49:21.292Z (19.563s)
