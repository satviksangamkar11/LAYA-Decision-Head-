# gpt-oss-20b verification (2026-10-13): FAILED, pilot stopped

Two calls on the canonical TRAIN request fable5:py-incident-cli:1 with the pre-registered settings (PREREG.json). Required on both: HTTP 200,
finish_reason stop, a tool action decodable by the v3t decoder.

| call | HTTP | seconds | finish | answer chars | reasoning chars | decodable |
|---|---|---|---|---|---|---|
| 1 | 504 | 302 | - | 0 | - | no |
| 2 | 200 | 4 | stop | 0 | 235 | no (empty response) |

Raw responses: responses/verify_1.json, responses/verify_2.json; send ledger: ledger.jsonl; summary: VERIFY.json. No VAL problem prompt was sent.
