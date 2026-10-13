# GLM-5.3 low-effort verification (2026-10-13): FAILED the registered rule, pilot stopped

Two calls on the canonical TRAIN request fable5:py-incident-cli:1 (PREREG.json; required on BOTH: HTTP 200, finish_reason stop, decodable tool action).

| call | key | HTTP | seconds | finish | answer | decodable |
|---|---|---|---|---|---|---|
| 1 | KEY_2 | 504 | 302 | - | none | no |
| 2 | KEY_2 | 200 | 185 | stop | 210 chars: Read test_incidentctl.py + Bash git log/status | yes |

Observation (not a re-run): in both this check and the gpt-oss-20b check, the two calls were sent at the same time on the same key, and the one that did not
finish first got HTTP 504 at about 302 s (the gateway limit) with no answer. In the GLM-max pilot all 25 requests returned 200 across keys 2-6.
