# Verification of three 2026 spec/check-trust claims, plus small-model evidence

Date of research: 2026-10-16. Web research only (WebSearch / WebFetch). No downloads, no code run.

Tag key: [read] = page text fetched with WebFetch (note: WebFetch answers are produced by a small model over the page; quotes are as returned, and they were checked against the page where the summary gave conflicting values). [snippet] = search result text only, not opened.

---

## A. CoVer

Identity found: the NeurIPS 2026 virtual page https://neurips.cc/virtual/2026/172888 [read] lists the paper "Not Self-Decidable: LLMs Cannot Draw the Boundary of What an Agent Verifier Can Check". CoVer = "corroborate-then-verify". Only author named in one fetch: Anthony Rhodes (second fetch summary did not name authors; [snippet] search text also named Rhodes).

- arXiv ID: NOT FOUND. Searches returned only the NeurIPS page. Two unrelated arXiv papers also use the name: "CoVer: Conflict-Aware Claim Verification" (arXiv 2609.00508) and "CoVeR: Coverage-Based Routing of Verifier Calls in Agentic Retrieval" (arXiv 2609.26086). [snippet]
- What it does (abstract, [read]): treats model unanimity "as a nomination". A predicate is admitted "only when the check synthesized for it survives intervention", reads fields the agent cannot write, and holds under deterministic rewording.
- Models: NOT NAMED. Abstract says "about 22,000 labels from four models built by three labs" over six corpora (incl. EU AI Act, FINRA guidance, a deployed credit agent). [read]
- Key finding (summary of abstract, [read]): agreement with a reference judge certifies nothing; agreement rose from 30% to 77% across calibration bands while the decidable share stayed flat.

### A claims

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| A1 | Validates check definitions synthesised from natural language | CONFIRMED (in part) | Quote: "the check synthesized for it survives intervention". [read] https://neurips.cc/virtual/2026/172888. The "from natural language" part is inferred from predicates drawn from regulatory text; the page does not use that exact phrase. |
| A2 | Paper says an intervention-resistant check can capture only part of the original obligation | NOT FOUND | No such statement. Closest: "at a cost in coverage we report rather than tune away." (cost is coverage of admitted predicates, not partial capture of an obligation). [read] https://neurips.cc/virtual/2026/172888. The phrase "intervention-resistant" does not appear. |
| A3 | Initial evaluation used a 73-predicate agent corpus | NOT FOUND | The number 73 is absent. Reported scale is ~22,000 labels, four models, three labs, six corpora. [read] https://neurips.cc/virtual/2026/172888 |

---

## B. VeriSpecGen

Identity: the paper is "Intent-aligned Formal Specification Synthesis via Traceable Refinement", arXiv 2604.10392 (submitted 12 Apr 2026, cs.LG), which names the method VeriSpecGen. [read] https://arxiv.org/abs/2604.10392. Authors: Zhe Ye, Aidan Z.H. Yang, Huangyuan Su, Zhenyu Liao, Samuel Tenka, Zhizhen Qin, Udaya Ghai, Dawn Song, Soonho Kong. Target language: Lean 4. A Microsoft Research listing exists: https://www.microsoft.com/en-us/research/?p=1168974 [snippet]. ICML 2026 listing: https://icml.cc/virtual/2026/82470 [snippet].

### B claims

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| B1 | Requirement-level traceability and targeted tests improve formal-spec generation | CONFIRMED | Abstract [read] https://arxiv.org/abs/2604.10392: "generates requirement-targeted tests with explicit traceability maps" and "traceability maps attribute failures to specific requirements". |
| B2 | 86.6% on VERINA SpecGen using Claude Opus 4.5 | CONFIRMED | Abstract [read] https://arxiv.org/abs/2604.10392: "VeriSpecGen achieve 86.6% on VERINA SpecGen task using Claude Opus 4.5." Table 2 [read] https://arxiv.org/html/2604.10392 gives Opus 4.5 baseline 59.0 -> VeriSpecGen 86.6 (+27.6). |

Small/open model results for VeriSpecGen (from full HTML, read via summary [read] https://arxiv.org/html/2604.10392):

- Models in Table 2: Claude Opus 4.5 (59.0 -> 86.6), Claude Sonnet 4.5 (50.7 -> 72.0), Qwen3-Coder-480B-A35B (36.7 -> 68.5), Qwen3-Coder-30B-A3B (29.3 -> 50.3; 30B total, 3B active), Qwen3-4B-Instruct-2507 (15.8 -> 23.0, +7.2, +45.6%).
- The smallest model is Qwen3-4B-Instruct-2507 (4B). Its base value is 14.7 in Table 4 (SpecGen pass@1), not 15.8; the paper does not explain the difference. Table 4 SFT No-Test figure for it is 30.3 per the summary. Treat the Table 4 values as unverified until the table is read directly.
- The "31.8 points" headline is the Qwen3-Coder-480B row (36.7 -> 68.5), not the Opus row.
- The "62-106% relative" range comes from fine-tuned Qwen3 rows: 61.9% (Qwen3-4B) to 106.1% (Qwen3-Coder-30B-A3B). [read, summary of abstract/intro/Table 4]
- No 7B, 8B, 14B, Llama, Gemma or DeepSeek model appears in the parts read. Verdict on "small (<=14B) open models": only the 4B Qwen row exists (15.8 -> 23.0). The 30B-A3B row is not small in memory terms.
- Note: the paper says the gains hold "across different model families and scales" [read]; the small-model numbers are the 4B row only.

---

## C. SpecSyn

Identity: "SpecSyn: LLM-based Synthesis and Refinement of Formal Specifications for Real-world Program Verification", arXiv 2604.21570 (v1 and v2 exist). Authors: Lezhi Ma, Shangqing Liu, Yi Li, Qiong Wu, Han Wang, Jiawan Wang, Lei Bu. [read] https://arxiv.org/abs/2604.21570

### C claims

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| C1 | Uses non-equivalent program mutations to test whether specs distinguish correct from altered behaviour | CONFIRMED | Abstract [read] https://arxiv.org/abs/2604.21570: "semantic-non-equivalent program mutations and variant discrimination". Filtering of equivalent mutants by compiling with optimisation flags, and the scoring rule, come from search snippets only [snippet] (https://www.alphaxiv.org/abs/2604.21570, https://www.emergentmind.com/papers/2604.21570), not from a read of the paper. |
| C2 | Verified 1,071 / 1,365 target properties | DIFFERENT (version-dependent) | v1 abstract [read] https://arxiv.org/abs/2604.21570v1: "SpecSyn successfully handles 1071 out of 1365 target properties for open-source programs." The v2 HTML [read] https://arxiv.org/html/2604.21570v2 says "SpecSyn successfully discharges 1,022 out of 1,421 verification targets" (Section 5.3) and "1022 out of 1421" in the intro. So 1,071/1,365 is the v1 figure; v2 reports 1,022/1,421. v2 wording is "discharges proof obligations", not "verified". |

Reported precision/recall (abstract, [read] v2 abs): precision "over 90%" and recall "over 75%"; the model these refer to is not stated in the parts read.

Models:

- SpecSyn generation models (v2 HTML, read via summary [read]): gpt-4-turbo-2024-04-09, gpt-5-2025-08-07, DeepSeek-R1. No small model is a SpecSyn generator in the parts read.
- Comparison baselines (v2 HTML [read]): Codex (GPT-5 backbone) 883 of 1,421; Preguss 510; AutoSpec 259. Fine-tuned baselines llama3.1-8b-instruct-fma and qwen2.5-coder-7b-instruct-fma (7B-8B). Their Table 1 averages: Llama 3.1 8B fma precision 5.25%, recall 1.63%; Qwen2.5-coder-7B fma precision 0.55%, recall 0.20%. Table 3: llama3.1-fma proves 29 of 1,421 targets.
- Small-model result for SpecSyn itself: NOT FOUND.

---

## D. Small open models on formal spec / verified code

Sparse evidence. Per-size numbers are only available for a few rows. Most benchmarks report frontier models only.

| Source (arXiv) | Benchmark / language | Model (size) | Number | Tag |
|---|---|---|---|---|
| 2604.10392 (VeriSpecGen), Table 2 | VERINA SpecGen, Lean, pass@1 | Qwen3-4B-Instruct-2507 (4B) | 15.8 baseline -> 23.0 VeriSpecGen | [read] |
| 2604.10392, Table 4 | VERINA SpecGen, Lean | Qwen3-4B (SFT) | 14.7 base; SFT No-Test 30.3 (conflicts with Table 2 base; unverified) | [read, summary] |
| 2604.10392, Table 2 | VERINA SpecGen | Qwen3-Coder-30B-A3B (30B MoE, 3B active) | 29.3 -> 50.3 | [read] |
| 2406.08467 (DafnyBench), Table 3 (n=10 attempts) | Dafny verification success | CodeLlama-7b-Instruct (7B) | 28.0 +/- 1.6 (no-LLM baseline 26.9; GPT-3.5 Turbo 44.0; GPT-4 Turbo 59.8; Claude 3 Opus 67.8) | [read] |
| 2512.10173 (ATLAS) abstract | DafnyBench | Qwen2.5-Coder-7B, fine-tuned | "improves performance from 32.4% to 56.9%" (Pass@1 label not stated in abstract; search snippet's 55.8 Pass@1 NOT verified) | [read] abstract; 55.8 [snippet] |
| 2604.21570 (SpecSyn) v2, Table 1 | Formal spec generation, open-source programs | Llama 3.1 8B fma (8B) baseline | precision 5.25%, recall 1.63% | [read, summary] |
| 2604.21570 v2, Table 1 | same | Qwen2.5-coder-7B fma (7B) baseline | precision 0.55%, recall 0.20% | [read, summary] |
| 2501.16207 (FM-ALPACA) | Multi-language proofs: Coq, Lean4, Dafny, ACSL, TLA+ (not Verus) | qwen2.5-coder-7B fma (7B) | text: 27.31%; Table 3 "Ave" column: 22.00 (conflict, unresolved) | [read] |
| 2412.06176 (AlphaVerus) abstract | Verus | LLaMA-3.1-70B (no finetune) | abstract only mentions 70B; claimed Llama 8B 11.8% -> 18.8% from snippet is NOT in the abstract | [read] abstract; 8B figures [snippet], unverified |
| 2505.23135 (VERINA) v2 | Lean SpecGen / CodeGen / ProofGen | o4-mini, Claude Sonnet 3.7, etc. | o4-mini 51.0% spec soundness+completeness (best); no small model in text read. ICLR version reportedly o3 52.3% [snippet] | [read] v2; ICLR [snippet] |
| 2605.08553 (VeriContest) abstract | Verifiable code gen (Verus per snippet) | ten unnamed models; sizes absent | spec gen 48.31%, proof gen 13.95%, end-to-end 5.29%, NL-to-code 92.18%. Qwen 3.6 at 4.86% and GPT-5.5 at 13.95% are [snippet] only, not in the abstract | [read] abstract |
| 2502.05714 (FVAPPS) | Lean 4 proofs on 406 theorems of 100 samples | Sonnet (size n/a) 30%; Gemini 18% | no small models | [snippet] |
| 2509.22908 (vericoding) | Lean / Verus / Dafny | off-the-shelf LLMs: 27% Lean, 44% Verus, 82% Dafny (no size breakdown) | [snippet] |
| 2602.09464 (AlgoVeri) | 77 classical algorithms, Dafny/Verus/Lean | Gemini-3 Flash (size n/a): 40.3% Dafny, 24.7% Verus, 7.8% Lean | [snippet] |
| 2310.17807 (Clover) | Dafny consistency checker, GPT-4 | GPT-4 (size n/a); accepts up to 87% of correct CloverBench programs, 0 false positives | [snippet] |

Summary for D: the only rows in the 4B-8B range with numbers read from a page are Qwen3-4B on VERINA SpecGen (Lean, 15.8 -> 23.0), CodeLlama-7B on DafnyBench (28.0%), fine-tuned Qwen2.5-Coder-7B on DafnyBench (32.4% -> 56.9%, abstract), and the 7B-8B fine-tuned baselines in SpecSyn (very low precision/recall). No source found gives a per-size curve for Qwen 4B-14B, Llama 8B-class or Gemma small on VERINA, Verus, or Lean spec tasks. Gemma small: NOT FOUND.

---

## Checklist

- [x] A1 CoVer validates synthesized checks: CONFIRMED (in part), [read]
- [x] A2 CoVer "intervention-resistant check captures only part of obligation": NOT FOUND, [read]
- [x] A3 CoVer 73-predicate corpus: NOT FOUND, [read]
- [x] A arXiv ID for CoVer: NOT FOUND (only NeurIPS 2026 page), [read]
- [x] B1 VeriSpecGen requirement-level traceability and targeted tests: CONFIRMED, [read]
- [x] B2 VeriSpecGen 86.6% VERINA SpecGen, Claude Opus 4.5: CONFIRMED, [read]
- [x] B small/open model results: PARTIAL (only Qwen3-4B 15.8 -> 23.0 and Qwen3-30B-A3B row), [read]; 7B-14B: NOT FOUND
- [x] C1 SpecSyn non-equivalent program mutations: CONFIRMED, [read]
- [x] C2 SpecSyn 1,071 / 1,365 verified: DIFFERENT (v1 says 1071/1365 "handles"; v2 says 1,022/1,421 "discharges"), [read]
- [x] C generator model: DIFFERENT / answered (gpt-4-turbo, gpt-5, DeepSeek-R1; not small), [read]
- [x] C small-model result for SpecSyn: NOT FOUND, [read]
- [x] D small open-model spec/verified-code numbers (4B-8B): PARTIAL (DafnyBench CodeLlama-7B 28.0; ATLAS Qwen-7B 32.4 -> 56.9; VeriSpecGen Qwen3-4B 15.8 -> 23.0), [read]
- [x] D Gemma small models: NOT FOUND
- [x] D AlphaVerus Llama-8B 11.8 -> 18.8: NOT VERIFIED (not in abstract), [snippet]
- [ ] Not done: full-text read of arXiv 2604.10392 Table 4 and rows beyond the first 100k characters; full text of 2604.21570 mutation method; full CoVer arXiv listing (not found in search).
