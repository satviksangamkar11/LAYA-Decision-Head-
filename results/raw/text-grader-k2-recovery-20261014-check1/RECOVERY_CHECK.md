# K1/K2 recovery check — 2026-10-14

Claude was not used. No experiment, inference, training, upload, commit, push or merge was performed.

## Verified remote status

- `sangeetasangamkar/ctb-k1-inventory`: RUNNING on repeated Kaggle CLI checks. The output request returned no inventory files. Final K1 results and usable-label counts remain unverified; do not launch a duplicate.
- `sangeetasangamkar/text-grader-k2-v1`: COMPLETE. Downloaded `out/METRICS.json`, `out/REPORT_tables.md`, preparation metadata and run metadata into this new recovery directory.

## K2 integrity finding

The downloaded METRICS.json explicitly records `run_meta.mode = dryrun` and `run_meta.stages.score.note = DRYRUN: fake random scores`. It records 18,774 synthetic score entries and 523 seconds total runtime, with an empty workers list. These are analysis-mechanics results, not checkpoint performance. Do not use any reported metric to recommend fine-tuning or claim an encoder signal.

The separately pulled remote source at `remote-source/text-grader-k2-v1.py` also sets `MODE = "dryrun"`. Its dryrun branch constructs random logits and random encoder/head features without model inference. Its final `K2 FULL DONE` log text is unconditional for non-smoke modes and cannot certify real inference.

The preserved earlier local smoke report at `../text-grader-k2-smoke-v1-tmp/out/SMOKE_MECHANICS.json` records 44/44 scored items, finite outputs, zero score failures/retries, zero library truncation and probability parity difference 0.0000499425. That establishes mechanics only. The 44-versus-50 selection explanation remains to be checked.

## Recovery limits and next steps

The installed Kaggle CLI accepts a version suffix but its output implementation does not pass that version into its output-list request. Therefore the downloaded output cannot be treated as a complete audit of historical versions. The browser shows the private notebook as unavailable without a signed-in session. Inspect saved-version history after Kaggle browser login to determine whether genuine full K2 output exists in another version; recover it before any rerun.

Keep training HOLD. K1 final outputs, K2 genuine full evaluation, useful encoder signal and usable CodeTraceBench training-label count are not yet established.
