# EL-ARCH-1 TRACEABILITY MATRIX (coordinator, 2026-10-17)

Every original architecture requirement -> owner -> acceptance test -> path. Sources: ARCHITECTURE_EVIDENCE_LADDER_2026_10_16.md sections 3-7, 12a (R1), 12b (R2), 12c (D5); EL_ARCH_1_CHECKLIST.md; the user's GO and COORDINATION CORRECTION (2026-10-17).
Rules: no unowned row; a row is PASS only with a real-path test (mock-only = LOGIC-ONLY, never "REAL PASS"); statistical rows are INSUFFICIENT_DATA with 3 examples; PATH A = local 0.6B live controller, PATH B = frozen 4B replay on exact stored contexts+options (never mixed).
"Prior" = what EL_ARCH_1_CHECKLIST.md claims as done earlier (commit shown); I have NOT re-verified it, so each row still needs its acceptance test run on the current tree.
Owners: A1 / A2 / A3 = Agents 1-3, C = coordinator. Overflow: Agent 2 carries the most rows; when Agent 1 finishes A1.x it takes rows I mark "A2->A1 overflow" as NEW TEST FILES only (no edits to Agent 2's files).

| # | Requirement (source) | Owner | Acceptance test (real path) | Path | Prior |
|---|---|---|---|---|---|
| **Invariants (sec. 3)** | | | | | |
| T01 | I1 authority: flag from evidence record, equals highest level reached; VERIFIED only with every hard/functional requirement VERIFIED (R1.1) | A2 | positive: ex2 passing run -> VERIFIED; negative: model text claims "fixed" with failing tests -> not VERIFIED | A | phase A |
| T02 | I2 monotone safety: no score reorders across evidence levels or accepts what L3-L5 rejects | A2 | negative: LAYA-top candidate with failing trusted check loses (conflict test) | B (stored 4B scores) + A (unavailable fallback) | phase C |
| T03 | I3 no self-authorisation: model/critic/tool text cannot change permissions, caps, levels, check library | A2 | negative: injection text in file/test output/GLM reply changes nothing | A | phase A |
| T04 | I4 unknown is not pass: UNKNOWN/TIMEOUT/ERROR/MISSING_EVIDENCE/INVALID_SPEC never pass (broken test tool -> ERROR) | A2 + A3 variant | exhaustive status-mapper test + broken-tool variant of ex2 | A | phase A/D1 |
| T05 | I5 bounded: tokens, rounds, steps, wall time, model calls ENFORCED; stop label names the exhausted limit (`step_cap`...) | A2 (enforce+label) + A3 (budget-exhaustion variant) | one test per cap hits it and stops; receipts show the specific label | A | phase A (enforced) / label open |
| T06 | I6 observe before commit: reversible in sandbox first; irreversible needs L4/L5 pre-checks + user confirmation; fixed action table | A2 | negative: unsafe/irreversible action cannot bypass authorization (unsafe-action variant) | A | phase B5 |
| T07 | I7 frozen scorers/check library hash-checked at start AND end; journal append-only | A2 + A3 (hashes in receipt) | negative: tampered scorer/check hash aborts; journal chain break detected | A | phase A |
| T08 | I8 sealed evaluation: eval-only oracles unreachable from runtime, incl. the proposal context | A2 (context) + A3 (deep scan) | negative: hidden/eval data placed in context is detected/absent; deep scan flags a held grader | A | D1 deep scan |
| **Evidence ladder (sec. 4, R1.1, R2.2)** | | | | | |
| T09 | L5 real observation (protected tests, real end state) may approve | A2 | ex2 trusted check pass on the real end state | A | |
| T10 | L4 certified checks approve only within stated scope | A2 | negative: out-of-scope L4 pass does not cover requirement | A | |
| T11 | L3 examples/constraints quoted from the request, deterministic | A2 | positive+negative on quoted-span filter | A | |
| T12 | L2 model-written checks validated (span, non-vacuous on mutants, consistent with L3/L5); never approve | A2 (A2->A1 overflow: new test file) | real 0.6B-written test on an ex task validated or rejected; negative: vacuous test rejected; cannot upgrade a row | A | phase B6 logic-only |
| T13 | L1 behaviour clustering of candidates | A2 (overflow ok) | two candidates with equal behaviour cluster; different behaviour split (real sandbox) | A | phase B6 |
| T14 | L0 scores (LAYA/probe/DeepConf/critic) never approve | A2 | negative: highest L0 cannot change a trusted rejection | A+B | |
| T15 | Evidence per requirement with scope + provenance; coverage table; answer flag = weakest row (R1.1, R2.2) | A2 | receipt has a coverage table; negative: one open requirement keeps flag below VERIFIED | A | verify |
| T16 | Output flags VERIFIED/SUPPORTED/PLAUSIBLE/UNVERIFIED/BLOCKED; report carries separate `outcome` and `process` fields | A2 + A3 (report) | receipt fields present; negative: process violation with passing outcome flagged | A | verify |
| **Router (sec. 5, R1.2)** | | | | | |
| T17 | O1 trusted executable oracle -> generate-execute-repair | A2 | ex1/ex2/ex3 route O1 | A | |
| T18 | O2 partial oracle (examples, regression) -> discriminate path | A3 variant + A2 | controlled variation of the 3 tasks routes O2 | A | |
| T19 | O3 no oracle: best effort, PLAUSIBLE/BLOCKED, blocking ambiguity asks the user (BLOCKED until answered) | A3 variant + A2 | variation with no tests routes O3; ambiguity variant BLOCKED | A | |
| T20 | Router re-run after every evidence-adding stage; upgrade only by trusted sources, never by model artifacts | A2 | test: repo tests found during INVESTIGATE upgrade a requirement; model-written test does not | A | verify |
| **Eight stages in ONE controller + one journal (sec. 6, R2.1)** | | | | | |
| T21 | INTAKE: contract skeleton, requirement list with quoted spans (phantom requirements dropped), caps, tier | A2 | negative: requirement without request span dropped | A | |
| T22 | INVESTIGATE: real snippets/paths/signatures/failing test output/constraints; truncation journalled | A2 | test asserts proposal prompt contains the real source lines of clamp.py; truncation record present | A | open (gap found) |
| T23 | GENERATE: real local 0.6B backend, registered prompt strategies, valid alternatives recorded separately | A1 (parse/filter/template) + A2 | PATH A real generation run; sequential plan / malformed variants handled | A | |
| T24 | CANDIDATE FILTER (frozen interface): no-op/unappliable/duplicate/plan/malformed rejected per option, abstain with reason, never invent options | A1 (+A2 wires) | phaseF1 tests + real-controller test of each rule code | A | open |
| T25 | EXECUTE in qualified WSL sandbox, actual outcomes recorded | A2 | real WSL run, hashes in receipt | A | |
| T26 | DISCRIMINATE: behavioural differences, validated L2 tests, blinded local GLM experiment | A2 + A3 | a decision with >=2 surviving clusters runs a discriminating experiment; negative: GLM reply cannot select | A | partial |
| T27 | RANK: evidence level first, then trusted checks before learned scores, LAYA at L0 only | A2 | negative: LAYA order cannot beat evidence level | A+B | |
| T28 | REPAIR: fixed template, real failure evidence fed back, progress tracked, regression detected, same-failure-twice / 2-no-progress stops | A2 | ex3 repair loop shows failure text in next prompt; negative: repeated identical failure blocked; legit retry after state change allowed | A | open |
| T29 | COMMIT & VERIFY: authorization, apply only permitted, inspect REAL end state, rerun trusted checks, diff scope | A2 | final state hash + test re-run in receipt | A | |
| T30 | Runs per action (intake -> check -> commit per tool call), task-level verify once at end (R2.1) | A2 | journal shows per-action intake/check/commit | A | verify |
| T31 | Approval = f(coverage, evidence, action risk) (R2.3) | A2 | reversible approved on trusted checks; irreversible needs confirmation | A | |
| **Models and scorers (sec. 7)** | | | | | |
| T32 | Missing LAYA scores journalled; explicit id-order fallback; no silent learned-score fallback | A2 | existing ex1 case + new variant with scorer removed | A | partial |
| T33 | LAYA ranks next actions (L0 ordering) only on exact scored options; never averaged with critic | A2 | PATH A: scored only if exact accepted options captured; else scorer_unavailable | A | |
| T34 | Frozen 4B LAYA replay: stored verified scores, exact context sha + option sha list, mismatch refused | A3 (harness) + A2 | PATH B replay matches stored scores; mismatched options refused | B | |
| T35 | 0.6B features never enter the frozen 4B adapter; provenance guard refuses | A3 | negative: 0.6B tensor refused (T7 cases re-run once) | A | done 437df8e5 |
| T36 | Fake/incompatible capture source rejected; live-capture provenance hardened | A3 | negative: fabricated tensors with valid metadata fail re-capture comparison | A+B | |
| T37 | Probe baseline + correctly constructed random-head control on the same compatible 4B pools, separate from LAYA, comparison-only | A3 | PATH B pools: probe and random-head scores journalled, selector=false | B | |
| T38 | Monotonicity gate: verification-effort ordering stays disabled unless registered criterion passes | A2 | 3 examples -> INSUFFICIENT_DATA, ordering disabled | A+B | phase C3 |
| T39 | No unregistered adapter ensemble; no untrained NONE probability | A2 | grep + negative test: unregistered ensemble/NONE refused | A | |
| **Critic (D5, sec. 5 stage 5)** | | | | | |
| T40 | Critic blinded (no LAYA pick, no generator reasoning, no hidden tests/labels) | A3 | negative: packet containing a forbidden field is refused | A | |
| T41 | Experiment types validated; only allowed operations executed via WSL; critic never authorises | A3 + A2 | negative: disallowed experiment type refused; label unchanged by critic | A | |
| T42 | GLM accounting reconciles with ledger (calls, sends, retries, tokens, errors); transport failures classified, retries bounded; no credentials logged; UTF-8 receipts (no mojibake) | A3 | reconcile test on the real local ledger; negative: non-ASCII round-trip stays intact; secret scan 0 | A | open (mojibake) |
| **Run integrity and reporting** | | | | | |
| T43 | Commit + secret scan BEFORE final run; receipts carry commit SHA, config hashes, model identity, source hashes, receipt + journal hashes; dirty tree refused | A3 (driver) + C (commit) | negative: dirty tree refused in final mode | A+B | |
| T44 | Final path uses real backends only; receipt names the backend; mock refused | A3 | negative: stub backend refused in final mode | A | D1 stub refusal |
| T45 | Replay of final decisions from committed code matches receipts (offline + WSL) | A3 + C | replay identical for every example | A (+B) | |
| T46 | Fault injection on the 3 tasks: malformed JSON, sequential plan, duplicates, empty pool, unsafe action, broken test tool, missing scorer, budget exhaustion, O2, O3, evidence conflict | A3 | each variant reaches its expected terminal state with the expected rule/stop code | A (+B for conflict) | D1 16x3 variants (re-run on new code) |
| T47 | Report gates and comparison arms; statistical gates INSUFFICIENT_DATA with 3 examples; dev tests marked unsealed | A3 | report shows INSUFFICIENT_DATA; no PASS from n=3 | A+B | phase C5/C6 |
| T48 | ARCHITECTURE_FINAL_REPORT.md: one row per row of THIS matrix and per invariant; PASS/FAIL/NOT_APPLICABLE/INSUFFICIENT_DATA | A3 (generator) + C (run) | generator refuses a PASS row without a real-test + negative-test + receipt reference | - | |
| T49 | Two paths kept apart in report and receipts; 0.6B candidates never paired with stored 4B scores | A3 + C | receipt sections separate; negative: pairing attempt refused | A,B | |
| T50 | Backbone profile (model-agnostic capture/generation/provenance) unchanged and still green | C | backbone suite 14 green on merged tree | A | 437df8e5 |
| **Explicitly out of scope** | stage-2/3 benchmarks, model accuracy, training, downloads, statistical generalisation, 70B/PEFT | - | - | - | - |

Open items needing a user decision: none required to proceed. Interpretation made (flag if wrong): on PATH A the existing parity-gated local 4B capture of the exact accepted options remains an allowed LAYA source (it scores THOSE options); the "unavailable-LAYA fallback" applies whenever that exact-option capture is not available/verified.
