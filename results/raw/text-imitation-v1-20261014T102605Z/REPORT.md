# text-imitation v1 (run record)

- Pre-registration: `configs/thresholds_text_imitation_v1.toml` (sha256 50164361...82ba), written before upload, smoke and training.
- Upload: private Kaggle dataset `sangeetasangamkar/laya-text-imitation-v1` (pools.jsonl.gz sha256 95176e4e...; uncompressed 91016212...; 335 TRAIN + 46 VAL pools). Secret scan `SCAN_REPORT.json`: 8,172,481 characters, 0 hits, nothing masked. CAL/TEST not read, not uploaded.
- Kernel `sangeetasangamkar/text-imitation-v1`, script ti1-2026-10-14a, Tesla T4, 1 process, torch 2.11.0+cu128.
  - v1: smoke, aborted at once (Kaggle auto-decompressed pools.jsonl.gz; loader fixed to accept both forms and to check the content sha256). `smoke-v1/`
  - v2: smoke, all mechanical checks OK (`smoke-v2/out/SMOKE_MECHANICS.json`); no metric computed.
  - v3: full run, wall 3,163 s (CV 2,788 s, final 197 s, encoding 127 s), 0 failures, 0 exclusions. `full-v3/out/` (METRICS.json, REPORT.md, pool_scores, items_manifest, final head seed 0 fp16). Kernel log not saved locally (the CLI failed to write it: Windows charmap error).
- Gate: FAIL (head minus probe -17.4, CI -35.3 to +2.0; head minus options-only -42.4, CI -56.2 to -26.2). Imitation experiments stop; next step is verified-outcome labels, not encoder fine-tuning.
- Observation (not pre-registered): at lr 2e-5, 1-6 epochs, the head barely moved (train log loss 1.267 -> 1.259 in 6 epochs; train top-1 about 34%); the trained head equals the unchanged one within 0.1 point on CV and +2.2 on VAL. The run tests this pre-registered recipe, not head capacity in general.
