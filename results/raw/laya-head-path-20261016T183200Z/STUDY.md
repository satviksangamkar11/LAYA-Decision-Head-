# LAYA head path study (arm E, L0 scorer) - written 2026-10-16, no GPU compute run

"VERIFIED" = I read the file, or a hash/structure check ran during this study (the packer `tools/pack_el_laya_gen_v1.py` asserted the sha256s below).
"CLAIMED" = written in a doc or report and not independently re-run. Paths are relative to D:\local model.

## 1. The frozen system

| part | path | sha256 | status |
|---|---|---|---|
| Qwen feature provider | models/qwen3-4b-thinking-2507, rev 768f209d9ea81521153ed38c47d515654e938aea | shards 98178c1c..., 57b5bd1f..., e3cccb84... (pinned in tools/fable_capture.py PIN) | pinned in code; checked in-kernel only |
| pretrained LAYA head | models/coding-decision-agent/model.safetensors | e316f15e1be381aee51ebfd5b9ccb1987395a20c31c103a42053e0f97c72810a | VERIFIED |
| adapter final-s0 | results/raw/laya-finetune-all-v1-20261012T080145Z/adapters/final-s0.pt | 111a36161c00e1e7bbb499d913b5fa33c1eb4586c96fd452a2b1ae9a4d19d4f2 | VERIFIED |
| adapter final-s1 | .../final-s1.pt | af43e2517040d9184fd94cd17349482bfaf6032d12c80156af65ee7c1bdbb36e | VERIFIED |
| adapter final-s2 | .../final-s2.pt | 67ce37c29a1455d5ac3ee7a5d218f20aefc868c045c4da748b303da54e9a50d6 | VERIFIED |
| baseline freeze | results/raw/laya-baseline-v1-20261012T215242Z/MANIFEST.json | - | read |

Head = `decision_head/head.py` DecisionHead(hidden 2560, d 1024, 2 transformer layers, k_layers 3, scorer "mlp"). 31 tensors are the pretrained Laya `Coding_Decision_Agent`
tensors (head.layers.* -> ctx.layers, scorer.* -> mlp, type_emb.weight -> prim_emb), frozen. The trained adapter is 8 tensors / 5,246,982 parameters
(in_q, in_o, mix_q, mix_o, out_norm) plus the fitted feature scaler; `final-s*.pt` stores every non-pretrained tensor. kind_emb is zero, the abstain/escalate gate unused.
Each adapter is scored ALONE; the reported figure is the mean of the three top-1 values (MANIFEST "scoring_rule"). An ensemble of scores is a different model.
Training: adapter only, on CPU (`tools/laya_finetune_all_v1.py`; objective "recorded", unknown_weight 0.5, 300 steps) on 334 TRAIN pools (192 trajectories) of 3-5 options each.
Positive = Fable's recorded action; alternatives are UNKNOWN, never "wrong". These pools have NO NONE option; none is added.

## 2. How the training features were captured (VERIFIED from run metas and code)

Code that produced them: `kaggle/level12_t4.py` (level 1) + `capture/capture.py` + `capture/capture_resident.py`. The `pkg/` copies in the three run directories are byte-identical to the repo files (packer assertion).
- Protocol `fable_capture_v1` (`decision_head/fable_features.py`): depths [18, 24, 30] = residual stream after decoder layer d (before the final norm), hidden 2560.
- prefix = tokens(context_text + "\n\nNext action:"), add_special_tokens=False. hq = hidden state at the LAST prefix token, shape [3, 2560].
- option i = tokens(" " + option_i), an independent suffix of the SAME prefix (block attention mask, position ids restart at the prefix length). ho[i] = fp32 mean of the stream over the option's own tokens, shape [N, 3, 2560]. Both stored as fp16.
- Model: `AutoModelForCausalLM.from_pretrained(dir, dtype=float16).eval().to("cuda")`, layer list truncated to the first 30, lm_head never run, SDPA; GQA-in-SDPA switched off on sm_75.
- Limits (nothing clipped): prefix <= 24000, option <= 4096, total <= 24576, <= 8 options.
- Context text = fable_ctx_v1: "[system]\n<1242-char system block>\n\n[user]\n...[assistant] <tool-call JSON> ... [tool result: ...]". One system block in all 337 contexts (sha256 9bf0352e...). Options = compact tool-call JSON (ensure_ascii=False, arguments sorted).
- DEVICE CLASS (VERIFIED): meta.json of results/raw/level12-run-v2-cancelled-20261011T175252Z/out/level1 (123 pools), results/raw/v3-context-capture-run-20261012T065106Z/out/level1 (214 pools) and results/raw/val-capture-run-20261012T111837Z/out/level1 (53 VAL pools): device "Tesla T4", dtype float16, torch 2.11.0+cu128, transformers 5.16.1. Not TPU, not the RTX 3050.
  A local RTX 3050 recapture of 12 TRAIN pools matched (worst cosine 0.99999, top choices identical; results/raw/local-capture-parity-20261012T110722Z/PARITY.json); it is a different device path.
- 123 + 214 = 337 captured ids; 334 enter training. Stored as `B.pt` (variant B, real context). Variants A (neutral prefix), S, X are controls.
- Mismatch found (VERIFIED by reading): `evidence_ladder/models/laya.py` EXPECTED_CAPTURE_CODE pins `tools/fable_capture.py`, a device-agnostic re-implementation of the same protocol. The features the adapters saw came from `kaggle/level12_t4.py` + `capture/*`. Not edited (it is the S3 contract table).
- The v2 run directory says "cancelled" because its Level-2 generation was cancelled; Level 1 finished (LEVEL1_VERIFY.json: 123/123, no failures).

## 3. Receipts for the accuracy figures

| figure | file | status |
|---|---|---|
| head 79.2% (seeds 79.25 / 81.13 / 77.36) on 53 VAL pools, CI [0.679, 0.893], chance 0.293 | results/raw/val-eval-v1-20261012T082055Z/REPORT-20261012T111926Z.json question_1 | VERIFIED (per-pool hits stored) |
| linear probe 94.3% on the same pools, C = 0.01; probe - head = +15.1 pts, significant | same file, linear_probe | VERIFIED |
| no-context head 64.2%; options-only text 67.9% (head - text +11.3, not significant) | same file | VERIFIED |
| in-dataset 5-fold: head 80.2%, probe 86.2% (334 pools) | results/raw/laya-finetune-all-v1-20261012T080145Z/REPORT-20261012T080534Z.json | VERIFIED as written; in-dataset, not unbiased |
| 89.8% top-1 of s0 on all 334 pools | ADAPTER_VERIFICATION.json | VERIFIED; mostly training pools |
| coding task success | none | NOT MEASURED |
Caveats on record: VAL is a development set inspected many times; alternatives were written by other models (authorship shortcut); the probe beat the head on every test; one backbone only.

## 4. The linear-probe baseline

Logistic regression on standardised raw option states (ho flattened, 3 x 2560 = 7680 features), class_weight "balanced", max_iter 2000, positive = Fable's recorded action,
C from (1e-3, 1e-2, 1e-1) by 3-fold trajectory-grouped CV (VAL run: 0.01). Code `tools/fullaction_pilot_v1.py probe_scores` (MANIFEST names `kaggle/head_exp4_gpu.py`, sha256 7e30b63d...; same recipe).
No weights file exists: it is refitted at scoring time on the 334 TRAIN pools. The probe and the random head are reported controls, not gates (user, 2026-10-16).

## 5. Random-head control

Defined here (no earlier artifact): adapter final-s0 with the 31 head tensors replaced by a seeded fresh init (`laya_forward.build_head(random_head_seed=0)`). Because the adapter co-adapted with the pretrained head this is a mismatch control.

## 6. What was built before the integration order

`configs/laya_head_gate_v1.toml` (pre-registered thresholds, the T2 file), `evidence_ladder/models/laya_forward.py`, `evidence_ladder/models/qwen_v2b.py`, an optional `backend=` in `laya.py`,
`kaggle/evidence_ladder_laya_gen_v1.py` (+ parity-only copy), `tools/pack_el_laya_gen_v1.py`, bundle `results/raw/el-laya-gen-bundle-20261016T183050Z` (157.5 MB: TRAIN set v3 + 53 canonical-VAL pools, public task prompts, adapters, 31 head tensors; no CAL/TEST/LOCKED, no keys; packer asserted every label row has split train/val).
Recorded per-pool scores exist only for VAL (the TRAIN pools were fitted by the final adapters), so reproduction of recorded scores uses the 53 VAL pools.
