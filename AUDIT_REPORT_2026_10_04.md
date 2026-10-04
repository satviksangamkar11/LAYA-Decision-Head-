# Audit Report: 10-Point Checklist (2026-10-04)

Inspector brief: Audit qwen_proposals_v2.py, pool_assembly.py, gates.py, state_recapture.py against a 10-point checklist. Report FOUND-OK / MISSING / WRONG per item with line numbers.

## Checklist Status by File

### 1. ORIGINAL OUTPUT: Full raw text per sample must be kept (compressed)
- **qwen_proposals_v2.py (lines 145-147):** MISSING
  - Currently truncates: `answer_text[:1500]`, `thinking_head[:200]`
  - Must store: full raw text (gzipped if needed)
  - **Action:** Add `raw_text` field to output dict
  
- **pool_assembly.py:** N/A (does not generate)
- **gates.py:** N/A (does not generate)
- **state_recapture.py:** N/A (reads states, not proposals)

### 2. SCORING: Store cumulative logprob and mean token logprob per sample
- **qwen_proposals_v2.py (lines 81-84, VLLMEngine.generate):** MISSING
  - Currently does not request or store logprobs
  - Must add: `logprobs=0` in `SamplingParams`, extract `cumulative_logprob` from `CompletionOutput`
  - Must add to output: `cumulative_logprob`, mean token logprob (cumulative / token count)
  - **Action:** Modify VLLMEngine to request and store logprobs
  
- **pool_assembly.py:** N/A
- **gates.py:** N/A
- **state_recapture.py:** N/A

### 3. MATCHING: Use exact_hit and near_hit; too-strict canonical equality
- **qwen_proposals_v2.py (lines 198-200, smoke function):** WRONG
  - Currently: `x["action_text"] == canon(rows_by_id[x["decision_id"]]["gold"])`
  - Must: Add `exact_hit` and `near_hit` flags; use `norm()` from `tools/hybrid_options.py`
  - Near-equal = same tool, same file/command, up to whitespace/quoting/order
  - **Action:** Add norm() call, track exact_hit and near_hit in output
  
- **pool_assembly.py (lines 80-86):** WRONG
  - Currently: `is_hit = (gold_canon == cand_canon)` (canonical equality only)
  - Must: Add `exact_hit` and `near_hit` fields; use `norm()` for detection
  - **Action:** Import norm(), add exact_hit/near_hit detection, store both flags
  
- **gates.py:** N/A (does not do matching)
- **state_recapture.py:** N/A

### 4. STYLE LEAK: Render ALL options through ONE serializer matching gold style
- **qwen_proposals_v2.py (line 43, parse_action):** WRONG
  - Currently: Returns `canonical_json()` (compact, sort_keys=True)
  - Gold strings: default style (spaces after ',' and ':', original key order)
  - Problem: TF-IDF/char classifier can separate by format alone
  - Must: Use ONE serializer for all options that matches gold style
  - **Action:** Do NOT use compact canonical JSON in action_text; serialize all with gold style
  
- **pool_assembly.py (lines 21-23, canonical_json):** WRONG
  - Currently: Uses compact JSON (separators=(",", ":"))
  - Must: Match gold style (spaces, original order) or find unified serializer
  - **Action:** Change serializer to match gold style
  
- **gates.py:** N/A (does not serialize)
- **state_recapture.py:** N/A

### 5. IMPOSSIBLE OPTIONS: Validate tool names and arguments; drop deterministically
- **qwen_proposals_v2.py:** MISSING
  - No validation of tool name against task's tool list
  - No validation of arguments against tool schema
  - nebius has `tools` column; swesmith and fable have own tool sets
  - Must: Drop invalid tools deterministically, count drops
  - **Action:** Add tool validation step before returning proposals
  
- **pool_assembly.py:** MISSING
  - No tool validation
  - **Action:** Add validation of all candidates
  
- **gates.py:** MISSING
  - No tool validation
  - **Action:** Add validation check
  
- **state_recapture.py:** N/A

### 6. DIVERSITY: 4 samples often repeat; sample extra, keep 4 distinct
- **qwen_proposals_v2.py (lines 124-125, run_batch):** MISSING
  - Currently: Generates 4 samples (N_SAMPLES=4), no dedup check
  - 4 samples often repeat by canonical form
  - Must: Sample extra, keep 4 distinct by canonical form, record diversity count
  - **Action:** Add dedup logic, track diversity metric per decision
  
- **pool_assembly.py:** N/A
- **gates.py:** N/A
- **state_recapture.py:** N/A

### 7. REPRODUCIBILITY: Seed, model hash, vllm version, prompt hash, prompt template in manifest
- **qwen_proposals_v2.py (lines 234-235, manifest):** MISSING
  - Currently: Only `mode`, `worker`, `shards`, `decisions_total`
  - Must add: seed, model revision hash, vllm-tpu version, prompt text hash, prompt template
  - **Action:** Expand manifest with reproducibility fields
  
- **pool_assembly.py (lines 140-146, manifest):** MISSING
  - Must add: version, timestamp, reproducibility fields
  - **Action:** Expand manifest
  
- **gates.py:** N/A
- **state_recapture.py:** MISSING
  - Protocol version `pool_v2_kaggle_t4_fp16`; must include timestamp, version
  - **Action:** Add version and timestamp to shard header
  
### 8. MEMORISATION: Compare gold-hit rate on nebius/swesmith vs fable (public vs private)
- **qwen_proposals_v2.py (lines 195-200, smoke):** PARTIAL
  - Computes hit rate per source in smoke output
  - Must report in full run manifest
  - **Action:** Add per-source hit rate to final manifest (lines 234-235)
  
- **gates.py (lines 71-88, qwen_hit_rate):** FOUND-OK
  - Already computes and reports hit rate per source
  
- **pool_assembly.py:** N/A
- **state_recapture.py:** N/A

### 9. CONTEXT FIDELITY: Document what Qwen does not see (tool list, earlier steps)
- **qwen_proposals_v2.py (line 4, header):** FOUND-OK (partially)
  - Documents: "state_text only (never candidates, never gold)"
  - Must also document: tool list, earlier steps are NOT in state_text
  - **Action:** Add comment documenting what state_text does NOT contain
  
- **pool_assembly.py:** N/A
- **gates.py:** N/A
- **state_recapture.py:** N/A

### 10. SPLITS: Train/cal/test must be produced the same way; nothing from test used to tune
- **qwen_proposals_v2.py (lines 268, 175-176):** PARTIAL
  - Reads split from rows.jsonl
  - No explicit guarantee that all splits use identical settings
  - Must: Document that all splits use same mode/cap/parameters
  - **Action:** Add assertion or comment that splits are identical
  
- **pool_assembly.py:** MISSING
  - No check that train/cal/test are disjoint
  - **Action:** Add verification that splits don't overlap
  
- **gates.py (lines 92-99):** MISSING
  - Counts splits but doesn't verify disjointness
  - Must: Verify train/cal/test decision_ids never overlap
  - **Action:** Add disjointness check
  
- **state_recapture.py:** MISSING
  - Must ensure all splits captured same way (same dtype, depth, device)
  - **Action:** Add comment documenting split consistency

## Summary of Changes Required

### For qwen_proposals_v2.py (Inspector owns; report with failing test)
1. Store full raw text per sample
2. Capture logprobs from vLLM
3. Add exact_hit and near_hit flags
4. Fix style leak (use gold-style serializer)
5. Add tool validation
6. Track diversity metric
7. Add reproducibility to manifest (seed, model hash, vllm version, prompt hash)
8. Report per-source hit rate in manifest
9. Document what Qwen doesn't see
10. Assert splits are identical

### For pool_assembly.py (Fix now)
1. Add exact_hit and near_hit detection (use norm())
2. Fix style serialization
3. Add tool validation
4. Check split disjointness

### For gates.py (Fix now)
1. Already has per-source hit rate (FOUND-OK)
2. Add split disjointness verification

### For state_recapture.py (Fix now)
1. Update protocol label for TPU (pool_v2_kaggle_tpu_bf16)
2. Document split consistency
3. Add version/timestamp to manifest

## Blocking Issues for Production

1. **Style leak (qwen_proposals and pool_assembly):** Text-only baseline can separate by format → MUST FIX
2. **No logprobs:** Soft target training cannot work → MUST FIX
3. **No tool validation:** Invalid tools reach training data → MUST FIX
4. **Split overlap not checked:** Test data could leak to train → MUST FIX
