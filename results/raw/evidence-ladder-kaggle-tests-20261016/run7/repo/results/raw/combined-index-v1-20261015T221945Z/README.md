# Combined index v1 (2026-10-16)

One manifest, three evaluation tracks, three scorers. **Never pool scores across tracks.**

| track | rows used for evaluation | what is scored |
|---|---|---|
| python | 40 (dev diagnostic only) | final program correctness (hidden tests) |
| java | 40 (dev diagnostic only) | final program correctness (hidden tests) |
| fable | 46 cleaned VAL pools (`fable-val:*`) | recorded-action agreement (Fable's action); never verified correctness |

The 335 frozen v3 Fable pools (`fable-v3:*`) are listed as exposed reference rows: the adapters were fine-tuned on 334 of them, so they cannot test LAYA.

Built without running any model. No retraining, no regenerated labels, no record moved between TRAIN, VAL or TEST. Run `tools/validate_combined_index_v1.py <this folder>` to check it.
Source hashes: SOURCES.json. Row schema: see any row in INDEX.jsonl.
