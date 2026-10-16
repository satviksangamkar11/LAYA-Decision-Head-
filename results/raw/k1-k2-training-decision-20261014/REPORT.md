# LAYA K1/K2 decision — 14 October 2026

## Decision: HOLD fine-tuning

**Trustworthy CodeTraceBench examples admitted under the required independent-alignment rule: 0. Evidence that pretrained LAYA beats simple coding-error baselines: not established. K3 is not justified and was not launched or packaged as a launch-ready job.**

K1 completed a real full inventory. The available K2 output is an analysis dry run with random logits and random features, not a full unchanged-checkpoint evaluation. This is an execution/evidence problem; it is not evidence of a weak encoder. The original reports, thresholds, scripts and checkpoints were preserved. No Claude, new experiment, parsing campaign, inference, training, upload, commit, push or merge was used for this review. Local activity was limited to editing, Kaggle operations, source inspection and reading summary reports; no trajectory or score-shard analysis was run locally.

## 1. Recovery and provenance

- K1: `sangeetasangamkar/ctb-k1-inventory`; live status COMPLETE; downloaded `INVENTORY.json`, `RUNS.jsonl` and log into `results/raw/ctb-k1-recovery-20261014-complete-check/`. Inventory mode `full`; 3,316 selected runs; processing counts cover all 3,316. Runtime 5,073.3 seconds (84.555 minutes).
- K2: `sangeetasangamkar/text-grader-k2-v1`; live status COMPLETE; summary reports recovered into `results/raw/text-grader-k2-recovery-20261014-check1/` and freshly rechecked in `results/raw/text-grader-k2-decision-recheck-20261014/`. `METRICS.json.run_meta.mode = dryrun`; score-stage note explicitly says `DRYRUN: fake random scores`; workers list empty. Runtime 523 seconds, preparation 145 seconds, analysis 364 seconds. There are 18,774 synthetic records and zero listed missing scores/failures; these are not 18,774 real model evaluations.
- Authenticated Kaggle metadata currently reports **version 2 for both kernels**. This is the current metadata version, not an independently embedded version identifier in the output. Script version for K2 is `k2-2026-10-14a`. The installed CLI output implementation ignores a requested version suffix when listing session output; historical output attribution must not be inferred from it. A version-specific GetKernel request returned HTTP 403. Historical versions therefore remain unverified, although current K2 source and current outputs both identify dryrun mode. No claim is made that a genuine full run cannot exist elsewhere.
- K1 dataset revision: `aa213b84ffb6690fc37ca15766d6ca174ec36d4d`; CodeTracer parser/backfill commit: `2d302191dd07e7c0c2da6f7a5e9451c7cbb62d34`. K1 recorded the dataset API head matching its pinned revision.
- Rules read: `TEXT_GRADER_PLAN_2026_10_14.md` including amendments and P1 findings; `configs/thresholds_text_grader_k2_v1.toml`. The registry includes UC1–UC6 before scoring and explicitly says no separate v1b file was needed. No registered rule was edited.

The GitHub branch's lack of these reports does not establish whether Kaggle ran. This report is local only because commit/push remains unapproved.

K2's current official manifest, preparation/run metadata, four score CSVs and 71,010,895-byte feature archive were also recovered into the fresh recheck folder. They were preserved without local analysis; the feature archive contains synthetic dryrun features. The CLI emitted a character-encoding error while handling the log, leaving a zero-byte local log; this does not invalidate the readable JSON/CSV/NPZ downloads, but no complete K2 log is claimed.

## 2. K1: training-data evidence

Full inventory totals: **3,316 trajectories, 17,701 incorrect flags and 1,468 unuseful flags = 19,169 labelled flags**. These are annotation counts, not independently aligned training examples. The inventory records 4,039 explicit manifest step-entry records with keys `action_ref,labels,observation_ref,step_id`; do not confuse records with the expanded flag count.

The table separates every evidence bucket. A reference-consistent run is not action-verified: its references were backfilled positionally.

| Agent / source format | Runs | Incorrect / unuseful flags | Independently action-verified | Reference-consistent | Count-only | Reference conflict | Missing source | Count mismatch | Early-stop BLOCKED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mini-SWE-agent / Terminal | 833 | 4,254 / 280 | 0 | 216 | 179 | 188 | 197 | 53 | 0 |
| mini-SWE-agent / SWE | 191 | 439 / 18 | 0 | 0 | 0 | 0 | 0 | 7 | 184 |
| OpenHands / Terminal | 889 | 5,606 / 373 | 0 | 0 | 0 | 0 | 102 | 92 | 695 |
| OpenHands / unknown | 25 | 120 / 9 | 0 | 0 | 0 | 0 | 25 | 0 | 0 |
| OpenHands / SWE | 328 | 232 / 19 | 0 | 0 | 0 | 0 | 55 | 273 | 0 |
| SWE-agent / SWE | 127 | 235 / 51 | 0 | 0 | 0 | 0 | 0 | 5 | 122 |
| Terminus2 / Terminal | 923 | 6,815 / 718 | 0 | 0 | 0 | 0 | 6 | 3 | 914 |
| Total | 3,316 | 17,701 / 1,468 | 0 | 216 | 179 | 188 | 385 | 486 | 1,915 |

BLOCKED means the registered per-format early-stop was applied: only cheap inventory statistics were collected after the trial found no independent anchor. It is not a newly observed reference conflict. The 216 reference-consistent mini-SWE Terminal runs contain 1,109 incorrect and 94 unuseful flags; **all 1,203 remain excluded from trustworthy training** under the user's stricter independent-alignment rule. Count-only runs have no negative flags in this inventory and are not verified weak positives.

Missing-source reasons: mini-SWE Terminal has 192 no-trajectory-file and 5 stub runs; OpenHands unknown has 25 absent archives; OpenHands Terminal has 102 no-trajectory-file runs; OpenHands SWE has 55; Terminus2 Terminal has 6 stubs. Conflict, missing-source, mismatch and early-stop buckets are excluded. Even reference-consistent and count-only buckets lack the independent evidence required for admission.

**Admitted incorrect = 0; admitted unuseful = 0; admitted weak-positive = 0; admitted total = 0.** This is a qualification conclusion, not a statement that all publisher labels are wrong or that the dataset contains no potentially recoverable examples. K1 did not produce a qualified rendered training manifest or a validated group-disjoint split. Consequently train/dev/test admitted counts are **0/0/0**, and group-integrity verification is not applicable to a nonexistent admitted set. The earlier planned 2,353/267/696 split is not a measured result.

Secret scan: 252,535 files inventoried; 252,526 text-scanned; 9 binary skipped; 8 scans truncated; 716 files and 651 runs with pattern hits; 788 runs with secret-like filenames. Pattern-hit counts: OpenAI-style 638, AWS 473, Hugging Face 244, GitHub 177, JWT 26, private-key-block 16, Google API 135. These are pattern matches, not proof of live credentials. No raw secrets are reproduced here. Any future admitted rendering needs sanitization and a fresh scan of the actual model inputs; eight truncated scans prevent a blanket clean-data claim.

Recorded processing failures = 0; extraction errors = 0; two files rejected by the size cap; downloaded-size and SHA mismatch runs = 0. K1 fetched 3,292 files totalling 462,305,519 bytes on Kaggle. These integrity checks do not establish label/action alignment.

### Prepared author request — draft only, not sent

> We are evaluating CodeTraceBench full at revision aa213b84ffb6690fc37ca15766d6ca174ec36d4d and CodeTracer at 2d302191dd07e7c0c2da6f7a5e9451c7cbb62d34. Our full 3,316-run inventory cannot independently map annotation step IDs to actions. Manifest action references appear to be generated by positional backfill, and count matches therefore do not resolve the annotation unit. Could you provide the original normalized steps.json files used during annotation and the original annotation-to-step mapping, keyed by traj_id? For each annotated step, we need its original step ID, raw event/message IDs, full action and immediate observation, label, and the normalizer version. Please clarify which actions are counted (including thought/edit/read/run/finish), indexing conventions, and whether compound actions span multiple messages. We also need the solved/unflagged run linkage to qualify weak positives without calling them verified correct. Mini-SWE Terminal currently has 216 positionally reference-consistent runs carrying 1,203 flags; other formats have conflicts, missing sources or mismatched units. An independent mapping or reproducible original annotation export would let us recover qualified examples without inventing a new numbering scheme. Please indicate a safe, sanitized way to share these files; our archives contain secret-like strings and filenames.

## 3. K2: what the available results establish

**No valid full-run LAYA performance figures are available in the recovered output.** Do not report the synthetic zero-shot scores, random-feature probes, their bootstrap intervals or Holm p-values as model measurements. Some text/rule diagnostics use genuine rendered inputs during the dryrun, but this does not make comparisons against random LAYA arms a valid pretrained-model evaluation. No model ranking is endorsed from this dryrun.

The registered coding cohort is 72 non-OpenHands runs; the other 30 OpenHands runs are secondary and include known overlap. Prepared dryrun subsets, per prediction time, are:

- Primary RootSE: 1,365 items, 72 run clusters, 71 eligible localization episodes.
- Secondary RootSE: 629 items, 30 run clusters, 27 eligible episodes.
- RootSE combined: 1,994 items before and 1,994 after action. Three secondary runs without earlier steps are listed as `rs027`, `rs028`, `rs038`; other decisive-item exclusions require the manifest/exclusion ledger, not assumptions. Earlier steps mean not-decisive, never verified good.
- APB: 7,393 items before and 7,393 after action; 200 task clusters; 418 eligible first-error episodes in the saved analysis. Raw label totals in preparation are 5,347 positive, 2,710 negative and 452 zero labels dropped. Thus 8,057 nonzero-labelled steps precede filtering and 664 are excluded from the retained set per prediction time. APB is general tool-use evidence, not coding evidence.

Requested RootSE before/after top-1, top-3, paired win-rate, percentile rank, baseline differences and confidence intervals are **unavailable as real model results**. The same applies to APB first-error and all-step comparisons. Bootstrap design is registered as 2,000 resamples over RootSE runs and APB tasks; a CI around fabricated scores cannot establish model ability.

### Input-quality and scoring audit

The real earlier mechanics trial completed **44/44 items**, not 50 attempted items with six failures. Its selector requests roughly 50: eight RootSE agent/model cells each contribute two steps at two prediction times = 32; APB contributes at most one eligible positive and negative per four configs at two prediction times = at most 16. The actual APB selection has six eligible label/config cells = 12 items, for 44 total. The selector skips assistant ordinal zero and unavailable eligible cells. This is selection-dependent coverage, not a fixed 50-item cap. Smoke metadata reports no leakage exclusions, failures or retries, so the missing six from the informal “~50” description are not failed inference. The precise identities of unavailable APB cells are not enumerated in the summary; do not invent them.

Trial clipping: action 6/44 = **13.64%**; task 32/44 = **72.73%**; observation 13/22 post-action items = **59.09%** (13/44 = 29.55% if expressing the mixed-time denominator). Library truncation 0/44; fixed-block overflow 0/44; state tokens 116–1,928 within budget 1,969; maximum full sequence 2,007 of 2,048 tokens. Finite outputs true; maximum probability difference from rounded library output **0.000049942543026482156**. Preparation took 37 seconds; score orchestration 40 seconds; each of two T4 workers processed 22 items in 13 seconds. Smoke performance metrics were intentionally not computed.

Full-preparation action/task/observation clipping percentages are not aggregated in the downloaded small summaries. They reside in per-item manifest flags; no local tokenizer or manifest-processing job was run to manufacture them. The full dryrun has explicit leakage exclusions; **do not claim zero full-run leakage hits**. IDs and reasons are preserved in `METRICS.json.run_meta.exclusions_leak` and `meta_prepare.json`, including answer-text matches and paired pre/post exclusions where one entry can have an empty own-reason list because its counterpart triggered exclusion. Retained items are filtered by the registered checks, but those heuristic string checks are not a proof that every possible temporal leak is absent. No real full inference means its parity/truncation worker checks are unavailable.

Weights and orientation: the real smoke inference loads pinned checkpoint `S4MPL3BI4S/Coding_Decision_Agent` at `fbc40c2e4b625519de1c1f617724449a636d08ff`; inspected library code uses model.eval and no-grad prediction. No optimizer or weight-update path exists in the scoring worker. This establishes forward-only intent and smoke behavior; no before/after weight-hash audit is recorded. Score orientation matches registration: higher badness = 1-P(continue), P(failure)+P(harmful), P(yes-review), or expected risk level 0–3. The native questions ask about traces/outcomes, not directly about the next coding action's correctness. Question mismatch is therefore a plausible evaluation-design limitation, not evidence already established by full model scores.

The same retained rows are used for all arms within each dataset/prediction-time block; pre/post exclusions are paired. Fitted baselines are diagnostics, not independent held-out tests. Outer folds use RootSE repo and APB task grouping. Source review found TF-IDF is fit on outer-training rows before inner C-selection, so its vocabulary/IDF sees inner validation text: strict inner-fold-only preprocessing is not established. Also the fixed observation-error rule can inspect a trailing 200-character observation segment not present in the renderer's leading 600-character observation. This limits its same-input comparison even though it was registered. These are documented design issues; no new controls or reruns were introduced.

## 4. K3 decision and fastest justified next step

**Case B applies: no independently aligned training examples are currently admitted. K2 has an evidence gap, not an established poor-performance result.** We cannot distinguish an encoder representation problem from question/context design effects using random scores. We cannot genuinely claim coding-error detection beyond simple baselines, nor conclude the encoder has no useful signal.

Best next step: recover a genuine full K2 version if one exists, and obtain original annotation mapping from the authors. Do not restart K1, develop additional parsers, change numbering, or launch new model work in this analysis task. The draft author request above is ready for review and was not sent. Historical K2 recovery is blocked by version-specific API access; the currently available data cannot fill that gap.

**No recommended K3 configuration is justified yet.** Preparing a launch-ready job on zero qualified labels would conceal the blocker. No training manifest, invented weak positives, optimizer settings, GPU-time promise or memory figure is presented as validated. When independent labels and genuine model results are available, revisit one frozen-encoder, head-only experiment using the approved renderer and separate prediction times; its concrete loss/split/runtime choices must follow those results. K3 still requires explicit training GO.

## 5. Checklist

- [x] Read saved plan/amendments and evaluation registry; preserve originals.
- [x] Recover and inspect full K1 inventory; record counts, runtime, exclusions and failures.
- [x] Recheck current K2 status/output and distinguish dryrun from real inference.
- [x] Record current version metadata and historical-access limitation.
- [x] Separate independently verified, count-only, reference-consistent, conflicting, missing and early-stop evidence.
- [x] State admitted labels and split counts without treating positional refs as proof.
- [x] Report secret-scan and extraction results without exposing secrets.
- [x] Audit real smoke mechanics, clipping, selection count, orientation and forward-only path.
- [x] Separate coding versus general-task scope and before/after prediction times.
- [x] Identify question/context/baseline design limits without new experiments.
- [x] Prepare the precise author evidence request; do not send it.
- [x] Make the fine-tuning decision: HOLD, with no K3 launch.
- [ ] Recover a real FULL K2 inference report, with version attribution, scored features and worker records — blocked/unverified.
- [ ] Supply real LAYA-versus-baseline metrics/CIs and full clipping percentages — unavailable in recovered model evidence.
- [ ] Independently qualify training labels and validate a group-disjoint manifest — missing author mapping.
- [ ] Prepare justified K3 configuration — conditional on the preceding evidence; not justified now.

Completed analysis must not be confused with completion of the unavailable evidence checks above. No training approval is being requested on these results.
