[LAYA-INT-1 · Aim: integration, not performance]

# Correction note: capture-code pins of the LAYA scorer (v1 -> v2)

**Problem.** `evidence_ladder/models/laya.py` pinned `tools/fable_capture.py` (+ `decision_head/fable_features.py`) as "the capture code" (table `EXPECTED_CAPTURE_CODE`, computed 2026-10-16).
**Finding.** The features the three adapters were trained on (TRAIN 123 + 214 pools) and evaluated on (53 VAL pools) were NOT produced by `tools/fable_capture.py`. `tools/fable_capture.py` was only run for a smoke test (results/raw/fable-capture-smoke-t4-20261009T231315Z) before the training captures existed; it was never run to produce training features. Never run for training = said plainly.

**What produced them (evidence).**
- `meta.json` of the three capture runs: results/raw/level12-run-v2-cancelled-20261011T175252Z/out/level1/meta.json, results/raw/v3-context-capture-run-20261012T065106Z/out/level1/meta.json, results/raw/val-capture-run-20261012T111837Z/out/level1/meta.json: Tesla T4, float16, torch 2.11.0+cu128, transformers 5.16.1, protocol_id fable_capture_v1.
- Code: `kaggle/level12_t4.py` (driver) + `capture/capture.py` + `capture/capture_resident.py`. The `pkg/capture/*.py` copies inside each of the three run directories are byte-identical to the repo files (asserted by tools/pack_el_laya_int1_v1.py at pack time; recorded in the bundle's BUNDLE_MANIFEST.json `capture_code_equals_run_pkg`).

**Hashes.**
| file | sha256 | role |
|---|---|---|
| capture/capture.py | d7e817d68fb6e0696ababd8add51ca929b92a2207d39885b11f0025db52710b8 | v2 required |
| capture/capture_resident.py | 8fb5e492480f6130366fc3bb235e991e5c3e777a74f32fc609d63cb15831106a | v2 required |
| decision_head/fable_features.py | fe2ae8ea295a2991353238502fcb0da0b5998590dde3c64bf754f61f4486c9e9 | v2 required (protocol constants, unchanged) |
| kaggle/level12_t4.py | 9c46095bf7ca42fc5dd5baed8baec4d7e5f04f68b099ae9acea76c475f3de133 | informational (v3 and VAL runs); the first run used b1becd8c... (results/raw/level12-bundle-20261011T175011Z/kernel/) |
| tools/fable_capture.py | 9d8dd3790b73503646243ea897ed4bb6f4a47cebab20463cb324aba7fc63bd72 | v1 pin, alternative implementation, superseded |

**Do the two implementations agree?** Measured on Kaggle T4 (receipt integration_run2/receipts/T2b_alternative_implementation.json, sha256 2b174b70e28fb5817c18615d83f0692ad9b79a8a1bb8be4f1264ad41a145e0d4) on the three smoke pools against the stored training features:
tools/fable_capture.py (bucket-padded packed forward) gives cosine >= 0.9999978 and max-abs <= 0.0625, within the T2 thresholds, NOT bit-identical.
capture/capture*.py (the training path) gives bit-identical features on all 65 sampled records (T2 receipt, sha256 9e9af31befb39b7aa659c9a4396908812ae110b4dce0934072fc86a778390c97).
So both are numerically compatible with the training features within the pre-registered tolerance, but only the training path reproduces them exactly; the scorer uses the training path.

**Correction (versioned, nothing silently edited).**
- New `configs/laya_capture_code_pins_v2.toml` (sha256 1c1e9d2ece043c02ebe35854439c48c686113cb291a74e5a725c6cbf5857d6a2).
- `laya.py`: the old table is kept as `EXPECTED_CAPTURE_CODE_V1`; `load_capture_pins("v1"|"v2")` reads the versioned pins (v2 is cross-checked against an embedded literal); `EXPECTED_CAPTURE_CODE` now points at v2. The S3 contract schema is unchanged (capture_code_path/capture_code_hash fields carry the new files).
- Consequence for other bundles: any bundle that runs the real `LayaScorer()` metadata check must now ship `capture/capture.py`, `capture/capture_resident.py` (and `configs/laya_capture_code_pins_v2.toml`, optional) instead of `tools/fable_capture.py`.
