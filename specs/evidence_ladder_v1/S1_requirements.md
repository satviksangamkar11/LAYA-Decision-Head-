# S1. Requirements contract (Evidence-Ladder v1, P0 + R3, 2026-10-16)

Schema: `schemas_v1.json#/$defs/requirement`. Source of truth for the architecture: `ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md` (R1, R2, 12c).

## 1. Three kinds of requirement

| kind | where it comes from | who creates it | may it gate approval? |
|---|---|---|---|
| `explicit` | an exact span of the user's request | Qwen proposes; **the controller checks that `quote` equals `request[char_start:char_end]`** | yes |
| `contract` | task metadata or repository artifacts: function signature, entry point, allowed files, permissions, existing public API | controller (deterministic) | yes |
| `derived_hypothesis` | inferred, e.g. "do not break password reset", "keep the public API" | Qwen or the critic, with a `rationale` and a source link | **no**, until `status = validated` |

- A proposed `explicit` requirement whose quote does not match the request is turned into a `derived_hypothesis` with `status = needs_validation`. It is never silently dropped and never silently promoted.
- A `derived_hypothesis` becomes `validated` only through one of:
  - trusted evidence: an existing test or the real environment shows the behaviour is relied on;
  - a `contract` source;
  - the user's answer.
- Until validated, it drives investigation only. It can reject a candidate only when trusted evidence shows the candidate breaks something that was observed to work before (a regression on an existing test).

## 2. Ambiguity and contradiction

- `ambiguous`: the request admits two readings that change the expected behaviour. The controller records both readings in `ambiguity_notes`. If no trusted check can tell which one is meant, the stop cause is `missing_information` and the user is asked. The model never picks a reading on its own.
- `contradicted`: two hard or functional requirements cannot both hold. `conflicts_with` lists both ids. The stop cause is `contradictory_requirements`, and the report includes both quotes.

## 3. Versioning

- Any change to a requirement's text, kind or priority creates `version + 1`.
- Evidence records carry `requirement_version` (S2). Evidence for an older version does not count.

## 4. Stage-1 shape (function-level coding)

- `contract` requirements come from the problem metadata: entry point / IO format, time and memory limits, language.
- `explicit` requirements come from the statement: examples and constraints quoted verbatim.
- `derived_hypothesis` requirements are model-inferred edge cases (empty input, limits). They rank candidates and guide repairs only.

## 5. Tests this contract must pass (implemented in the pilot's stage 0)

| id | case | expected |
|---|---|---|
| S1-T1 | explicit requirement with a quote not present in the request | becomes `derived_hypothesis` + `needs_validation` |
| S1-T2 | derived hypothesis used to approve | refused by the controller |
| S1-T3 | evidence for requirement version 1 after the requirement moves to version 2 | ignored for approval |
| S1-T4 | two contradictory hard requirements | stop cause `contradictory_requirements`, both quotes reported |
