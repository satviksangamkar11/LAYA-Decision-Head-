# K2 genuine full evaluation — Kaggle version 3

User authorized only changing dryrun to full and running the unchanged registered evaluation on Kaggle T4. Training remains HOLD.

Kernel: `sangeetasangamkar/text-grader-k2-v1`. Kaggle CLI accepted version 3 on 14 October 2026.

Only execution-mode line changed in canonical and staged scripts: `MODE = "dryrun"` became `MODE = "full"`. Renderer, datasets/revisions, checkpoint/revision, questions, baseline implementation, thresholds and output paths are unchanged. The registered configuration was not edited.

Canonical and staged script SHA256: `B5C39CDE7FF4F33FBA0BC64824F334978D79E34307594071754C1653DF7B5E64`. Both passed py_compile. No script was executed locally.

Launch metadata: private Python script; GPU enabled; TPU disabled; internet enabled; machine `NvidiaTeslaT4`; input dataset `sangeetasangamkar/laya-align-v1`; no kernel sources.

Authenticated remote metadata/source confirmed current version 3, GPU true, TPU false, machine NvidiaTeslaT4 and MODE full. Repeated status checks show RUNNING. Completion heartbeat `finish-k2-full-evaluation-v3` is active in this chat every five minutes: remain quiet while running, recover/verify real results, report RootSE/APB comparisons, then pause. No completed inference/result claim is made at launch.

Historical dryrun evidence remains in its separate recovery directories and on the previously pushed Git branch. All version-3 downloaded results will be saved in this new directory. Kernel acceptance is not evidence of successful inference; verify final run mode, worker devices/counts, finite logits, probability parity, completed item counts, missing scores, failures/retries and real model-score output before reporting results.
