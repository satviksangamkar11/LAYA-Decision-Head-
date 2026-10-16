# Context diagnostic v1 on the Fable TRAIN pools: result and reading (2026-10-15)

Kernel `sangeetasangamkar/laya-context-diag-run-v1` v4 (full; v1-v3 were setup failures / smoke), pre-registration
`configs/thresholds_context_diag_v1.toml` (sha256 291c9b5d...). Gates passed: adapter reproduction 0.7826 / 0.8043 / 0.7609; no-context recapture parity
min cosine 0.9999993 on the 112 stored pools. 303 pools (the 304th is over the capture limit), 177 trajectories, chance 0.291.
Pool alignment: the pack is the cleaned v4 view; an order-preserving option map (`upload/v4_option_map.json`, 304/304 verified) was applied to the
uncleaned text pools. Qwen-alone prompts use the context without the [system] block (what the text pools contain). Qwen max prompt 23,539 tokens.

## Standalone scorers (correct of 303; 95% CI by trajectory)
| scorer | with context | without context |
|---|---|---|
| Qwen alone (2 orders averaged) | 111 = 0.366 [0.316, 0.419] | 87 = 0.287 [0.235, 0.341] |
| LAYA + adapter, 3-seed ensemble (IN-SAMPLE) | 281 = 0.927 [0.895, 0.955] | 245 = 0.809 [0.762, 0.851] |
| LAYA seeds s0 / s1 / s2 | 0.904 / 0.917 / 0.908 | 0.828 / 0.822 / 0.746 |
| probe, out-of-fold | 260 = 0.858 [0.814, 0.899] | 231 = 0.762 [0.711, 0.813] |
| probe, in-sample | 302 = 0.997 | - |
| options-only text, out-of-fold | 229 = 0.756 [0.701, 0.813] | same (no context by construction) |

## Context gain (correct with - correct without), W->R / R->W = wrong->right / right->wrong when context is added
| scorer / rule | gain | CI | W->R | R->W | picks changed |
|---|---|---|---|---|---|
| Qwen alone | +24 | [+0.018, +0.141] | 61 | 37 | 145 |
| LAYA ensemble | +36 | [+0.074, +0.164] | 43 | 7 | 56 |
| probe out-of-fold | +29 | [+0.051, +0.143] | 40 | 11 | 57 |
| loop C_avg (LAYA, probe) | +38 | [+0.079, +0.175] | 48 | 10 | 66 |
| loop S_vote | +24 | [+0.036, +0.120] | 34 | 10 | 49 |

## Loop vs standalone (with context): wrong->right / right->wrong, net pools [CI]
| rule | vs LAYA ens | vs probe OOF | vs options-only OOF |
|---|---|---|---|
| S_vote (280 correct) | 1 / 2, -1 [-0.014, +0.007] | 25 / 5, +20 | 61 / 10, +51 |
| C_avg (269) | 5 / 17, -12 [-0.073, -0.007] | 10 / 1, +9 [+0.007, +0.055] | 53 / 13, +40 |
| C_laya_first t=0.20 (272) | 4 / 13, -9 | 14 / 2, +12 | 56 / 13, +43 |
| C_avg3 with Qwen (269) | 5 / 17, -12 | 14 / 5, +9 [0.000, +0.062] | 52 / 12, +40 |
No rule beats the LAYA ensemble (best: S_vote -1 pool). Every rule beats Qwen alone and options-only because LAYA (in-sample) is the strongest input.

## The held-out-like check: fold 0 (52 pools; the adapters used it only to select a checkpoint)
| | fold 0 (52) | folds 1-4 (251, fitted) | fold 0 without context |
|---|---|---|---|
| LAYA ensemble | 44 = 0.846 | 237 = 0.944 | 39 = 0.750 |
| probe OOF | 44 = 0.846 | 216 = 0.861 | 43 = 0.827 |
| options-only OOF | 42 = 0.808 | 187 = 0.745 | 42 = 0.808 |
| Qwen alone | 18 = 0.346 | 93 = 0.371 | 13 = 0.250 |
| S_vote / C_avg3 | 0.827 / 0.865 | 0.944 / 0.892 | 0.827 / 0.808 |
Memorisation indicator (LAYA ens, folds 1-4 minus fold 0): +9.8 points, CI [+0.1, +20.6].

## Other numbers
- Qwen position bias: mean probability of letter A is 0.64 (with context); after averaging two orders its mean top-1 margin is 0.19 vs 0.57 for LAYA,
  so it barely moves any combination (C_avg3 and C_avg differ on 14 pools, net zero).
- Oracle with context (303 pools): LAYA ens or probe 287; any of LAYA/probe/Qwen 291; with options-only 296; LAYA ens alone 281.
  LAYA ens right and probe wrong: 27; probe right and LAYA ens wrong: 6.

## Reading against the four pre-registered readings
1. Context improves accuracy and fixes wrong decisions: MET on all 303 pools for every scorer, including the out-of-fold probe (+29, 40 fixed vs 11
   broken, CI above 0). Two inflators: LAYA is in-sample, and "without context" is test-time removal (models were fitted on real contexts, so the
   no-context features are also out of distribution). On fold 0 the gain is small: LAYA +5 pools, probe +1, Qwen +5 of 52.
2. Predictions change but accuracy does not improve: partly true of Qwen alone (145 of 303 picks change, net +24).
3. Without context equal or better: not for LAYA or the probe overall; on fold 0 the no-context scorers are at or below options-only (0.808).
4. Loop beats every standalone scorer: NOT MET (best rule trails the in-sample LAYA ensemble by 1 pool; the others by 9-12).
Scope: agreement with Fable's recorded action on TRAIN pools; LAYA numbers are partly memorisation; nothing here shows better real-world decisions.
