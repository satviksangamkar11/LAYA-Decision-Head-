# P5 final readiness receipts (2026-10-20) — preflight complete, no model generation

Status: every preflight item authorized by the user is done. The final 105-task GPU generation is **NOT started** and needs the user's GO.

## 1. Hashes
| Item | sha256 |
|---|---|
| `P5_PROTOCOL_v2.md` (committed bytes) | `3c5326a4f6f0f9a677993be22ce15322e1c0b6bbda5e896a330cc6475c945dac` |
| `P5_PROTOCOL_v1.md` (committed bytes; kept) | `3d26566ca19504d32dfe252c2e84efe3a6581f62e307a6af4303b80a2695961b` |
| `configs/p5_final_lock_v2.json` | `71d197fb138c21a11c6185ec5913b011ef181dd140735adbf865d19171416e22` |
| `configs/p5_final_lock_v1.json` (kept; ids and order unchanged in v2) | `9f3a3cdbeb43ef9a2c45bbb57060e05563b320092f1977e0b1f03fe79ad38f8d` |
| locked LCB ids (85) | `a18cfc1dc62025ce2be1cc97634907e6691f6b25526e32173f4aac538b5c164e` |
| **primary** grading: blind `private/p5-eval-v3/cases_v3.json` (849 cases; content outside git and Kaggle) | `7b5b109f1c80f2312b62fbc87b75de72cc0590901cf0e2056487fef81d3e251b` |
| primary `CHECK_v3.json` / `SHA256SUMS_v3.txt` | `c9de7495…63bd` / `2df7a856…3f79` (full values in the lock) |
| secondary grading: public `results/raw/p5-eval-only-v1/hidden_cases.json` | `1d5b02482e2fc64fa29b2b1220f1c4b112b3a5fdd0ad613337acc8faa62c1ea0` |
| disclosed non-blind validation: `private/p5-eval-v2/cases_v2.json` (820 cases) | `0cdbb530df86af806f555c4e1ce7f4f94e1a31a685b2690fbc210f5e111018dc` |
| final Kaggle bundle (dataset `el-p5-bundle-v1`, latest version) | commit `d955ed40`, 152 files, `results/raw/el-p5-kaggle-final-bundle-20261020T171644Z/AUDIT.json` |

## 2. Protocol diff (v1 → v2)
Amended, with written reasons in `P5_PROTOCOL_v2.md`: execution environment MXC → Kaggle jail + trusted verifier; shards sequential, one jail user; size fixed at
85 LCB + 20 contract (ids/order of lock v1); contract grading primary = new private blind set, secondary = existing public set. One clarification (no criterion
change): the D-vs-C wall-time criterion is computed on wall-equivalent time because the shared generation cache makes raw arm time incomparable.
Unchanged: model and revision, sampling, arms A–D, G = 6,000, contracts and their semantics, success criteria, measures, uncertainty method.

## 3. Private grading — validation summary
- **Primary (blind v3):** authored by a fresh-context subagent (user-authorized exception to the no-subagent rule, this task only) with access to
  `STATEMENTS.jsonl` and the grading format only. Its report: `all_ok`, no unresolved cases, no accidental exposure, no git commands.
  **Count correction:** the author's report stated 853 cases; its own per-task table and the file contain **849**. Both the error and the correction are recorded
  here; the dataset was not regenerated. Coordinator checks (no case opened): 9/9 file hashes match; the author's validator re-run in a scratch copy →
  all_ok, generator byte-identical, cases file unchanged; cross-check of all 849 expected values against the separately written v2 reference → 0
  disagreements (verification only, no repair); every expected value passes the grader's strict comparator.
- **Non-blind v2 (820 cases):** preserved unchanged with its `PROVENANCE.md` (author's prior exposure disclosed); used for nothing but disclosed validation.
- **Grader fails closed** (`tools/el_p5_grade.py`): the contract cohort requires an explicit set name (primary | secondary | nonblind) and the file's sha256 must
  equal lock v2; no default, no fallback. Tested: no name → refused; unknown name → refused; primary → grades (counts only); a one-byte-modified primary copy →
  refused on hash; no output file written by a refusal.

## 4. Bundle audit (13/13 PASS)
manifest hashes; no eval-only file; no reference/generator line (shared idioms aside); no public hidden input with its expected value; bundled lock = repository
lock v2; **no case of either private set (1,669 scanned)**; no `private/` file; LCB rows = the 85 locked ids; LCB private tests empty; contract tasks public only;
no P5 grader; no key-shaped string; plan = 5 × 17 LCB + 1 × 20 contract shards. Unused grader material removed (Java pilot grader/support, pilot task lists);
the Python sandbox scripts stay (runtime: `strict_equal`, jail self-check).

## 5. CPU dry run on Kaggle (kernel `el-p5-final-dryrun-v1`, no GPU) — PASS as designed
| Run | Bundle | Jail self-check | Port qualification | Model sha256 | GPUs | Verdict | Shards started |
|---|---|---|---|---|---|---|---|
| v1 (17:17 UTC) | d955ed40 | PASS | 10/10 | ok | 0 | FAILED_NO_GPU (stop before generation) | 0 |
| v2 (17:26 UTC, cleanup fix) | d955ed40 | PASS | 10/10 | ok | 0 | FAILED_NO_GPU (stop before generation) | 0 |

Finding from v1: an early exit left the ~3 GB model and the bundle copy in the downloadable output; the local download was stopped at ~3.5 MB (approval rule
for >1 GB downloads). Fix: the kernel's common exit removes both (v2 output = 5 small files). The partial download stays in
`results/raw/p5-final-dryrun-v1/` (not committed; deletion of model files needs the user's yes).
Receipts: `results/raw/p5-final-dryrun-v2/` (RUN_SUMMARY, QUAL), `results/raw/p5-final-dryrun-v1/RUN_SUMMARY.json`.

## 6. Disclosures carried into the final report
Q03 diagnostic-contract effect; Kaggle smoke parallel-shard jail contention; compute estimate 7.2 h is a linear extrapolation (contract rate borrowed from bank
tasks); public secondary set; non-blind v2 set and its author's exposure; the 853→849 count correction.
