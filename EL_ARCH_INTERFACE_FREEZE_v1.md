# EL-ARCH-1 INTERFACE FREEZE v1 (coordinator, 2026-10-17): candidate filter <-> controller

Frozen BEFORE integration. A change needs a new file (v2) with a reason; owners message the coordinator, nobody edits this file.

## Function (Agent 1 implements in evidence_ladder/controller/candidate_filter.py; Agent 2 calls it)
`filter_options(options, files, *, max_options=5, path_scope=None) -> FilterResult`
- `options`: list of CandidateIn in GENERATION ORDER, each `{"tool": str, "arguments": dict, "origin": {"seed": int|None, "index": int|None}, "raw": str|None}`; `files`: dict workspace-relative path -> current text (what the controller already read); malformed items arrive as `{"malformed": True, "origin": {...}, "raw": str}` so a bad item is rejected on its own and siblings survive.
- Pure and deterministic: no I/O, no clock, no randomness, no model.

## Result (plain JSON-serialisable dicts; journalled as-is under kind "candidate_filter")
```
FilterResult = {
  "accepted": [ {"option_id": <sha256 hex>, "tool", "arguments", "action_key": <str>, "origin": {...}} ... ],
  "rejected": [ {"option_id": <sha256 hex or null>, "origin": {...}, "rule": <code>, "reason": <short str>} ... ],
  "abstain":  null | {"code": "no_options_proposed" | "all_options_rejected", "reason": <str>},
  "counts":   {"in": n, "accepted": n, "rejected": n}
}
```
- Stable identity: `option_id` = the SAME sha256 the journal already stores as `option_sha256` (hash of the canonical option text produced by the existing renderer). Same action -> same id in every run, so stored LAYA scores, replay and journals key on it. The filter never edits, reorders or renumbers accepted options; it only removes. The decision-local index (`decision_id#i`) is assigned by the controller over the accepted list in order.
- `action_key` = normalised (tool, canonical-JSON arguments with whitespace-normalised path) used by Agent 2's repeated-failure key.
- Never invent, merge or pad options to reach k=4. 0 accepted => `abstain` set, controller journals an abstain decision with that code and takes no action.
- Rule codes (closed set): `malformed_option`, `unknown_tool`, `bad_arguments`, `path_outside_scope`, `protected_path`, `no_op_edit`, `unchanged_diff`, `old_string_absent`, `old_string_not_unique`, `duplicate`, `sequential_plan`, `beyond_max_options`, and (added by the controller, same record shape) `repeated_failure`.
- Sequential plans: a pure rule inside the filter; the matching prompt template is a NEW versioned id (never edit qwen_actions_v1).

## Controller side (Agent 2)
- Calls `filter_options` once per decision before ranking/scoring; appends `repeated_failure` rejections itself in the same record shape; journals the whole FilterResult; passes only `accepted` (in order) to the scorer/ranker; scorer inputs and journal `option_sha256` list = the accepted option_ids.

## Two final paths (never mixed)
- PATH A, LOCAL LIVE: 0.6B generates -> filter -> controller. LAYA is used only if the options being scored are the exact accepted options of this decision (a verified capture of those exact options, as the parity-gated local 4B capture did); otherwise LAYA is explicitly UNAVAILABLE (journalled `scorer_unavailable`, id-order fallback). 0.6B features never enter the 4B adapter.
- PATH B, FROZEN 4B REPLAY: stored verified 4B scores (results/raw/el-arch1-accept-20261017T101325Z and earlier) replayed ONLY on the exact stored contexts and options they were computed for (match by context sha + option sha list). No 0.6B-generated candidate is ever paired with a stored 4B score; a mismatch is a refusal, not a fallback. The LAYA-top-loses-to-trusted-check conflict test lives on PATH B.
