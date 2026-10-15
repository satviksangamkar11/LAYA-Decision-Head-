# FormalJudge: claim verification (research pass A)

Date: 2026-10-16. Method: WebSearch and WebFetch only. No downloads by me, no git clone, no code run.

Evidence tags:
- [read] = page fetched with WebFetch. WebFetch returns an answer written by a small model over the page text, so quoted sentences are that model's extraction. I could not check them byte-for-byte against the raw page. Treat quotes as close, not certified.
- [snippet] = search result text only.

WebFetch cached the arXiv PDF at `C:\Users\Satvik\.claude\projects\D--local-model\7392300c-4d44-483f-b905-b620ef00cddb\tool-results\webfetch-1792093704351-idldwc.pdf`. It is binary and could not be read here (no PDF renderer), so the PDF was not used as evidence.

## Identifiers

- Title: "FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight"
- arXiv 2602.11136, cs.AI. v1 submitted 11 Feb 2026, v2 revised 12 Feb 2026. PDF metadata says CC BY 4.0. [read] https://arxiv.org/abs/2602.11136
- Authors: Jiayi Zhou, Yang Sheng, Hantao Lou, Yaodong Yang, Jie Fu. The arXiv abstract page lists no affiliations. [read]. Affiliations Peking University, Shanghai AI Lab, Fudan University: [snippet] only.
- ICML 2026 poster page: https://icml.cc/virtual/2026/poster/61086. Poster session Thursday 9 July 2026, 10:30 to 12:15 KST, Hall A #301. The page does not say "accepted" in words; the poster listing implies it. [read]
- Repo: https://github.com/htlou/FormalJudge (page title matches the paper title; README rendered). [read]
- Do not confuse with "Certifying the Judge" / CertJudge, a different ICML 2026 paper on Lean 4 formal code. [snippet] https://icml.cc/virtual/2026/82750

## Claim verdicts

### 1. ICML 2026 paper; arXiv ID, authors, title
**CONFIRMED.** arXiv 2602.11136; title and five authors as above. Sources: [read] arxiv.org/abs/2602.11136, [read] icml.cc/virtual/2026/poster/61086.

### 2. Workflow: three LLM roles plus a formal executor (Dafny/Z3)
**PARTLY CONFIRMED; wording DIFFERENT.**
- README [read, via repo page]: three agents, one decomposes requirements, one translates them into formal specs, one extracts values from execution traces. An executor then gives YES/NO verdicts. Dafny and Z3 are named. Source: https://github.com/htlou/FormalJudge
- Paper [read, arxiv HTML v1]: four phases, not three roles:
  1. top-down intent decomposition (LLM splits intent into atomic yes/no facts)
  2. context-aware grounded extraction (structured fields parsed deterministically first, "without LLM involvement"; an LLM answers the remaining facts)
  3. formal proof synthesis (a generator produces Dafny code)
  4. SMT verification (Dafny compiled through Boogie to Z3; "entirely deterministic")
- Python as a translation target: NOT FOUND in any quote I could confirm. One [read] pass mentioned Dafny and Python as verification languages, with no quote.

### 3. Dafny verifier receives LLM-extracted facts; facts themselves not proven
**CONFIRMED.** [read, arXiv v1] The paper says the design "confines probabilistic reasoning to atomic semantic extraction while delegating all logical composition to deterministic solvers." The solver proves the composition of the facts. The paper does not claim the extracted facts are themselves verified. Source: https://arxiv.org/html/2602.11136v1

### 4. Authors name specification decomposition and semantic fact extraction as main remaining error sources
**CONFIRMED.** [read, arXiv v1, Section 5]
- "errors in decomposing user intent into atomic constraints during the top-down phase"
- "errors in extracting semantic information from agent trajectories during the grounded extraction phase"
- "it does not achieve perfect accuracy"
- Also [read]: remaining errors "stem primarily from the LLM components within our pipeline." No quantitative breakdown of the two error types was found.

### 5. Agent-SafetyBench refinement: Qwen 7B 30.8 to 49.4; Qwen 14B 34.3 to 74.2; models; rounds; metric
**NUMBERS CONFIRMED. Model family DIFFERENT (Qwen2.5, not Qwen3).**
- Quote [read, arXiv v1, Section 4.5]: "Qwen-7B improves from 30.8% to 49.4% and Qwen-14B from 34.3% to 74.2% with Dafny guidance"
- Models: Qwen-2.5 at 7B, 14B, 32B, 72B [read, Section 4.1: "four open-source (Qwen-2.5 7B/14B/32B/72B)"].
- Rounds: three. Table 11 gives max_iterations = 3 [read, appendix C.4.3]. Section 4.5 says "measuring safety classification accuracy over three refinement rounds". Round 0 is the initial response with no feedback [read, one pass]. Most passing samples finished within 2 iterations [read, appendix C.11].
- Metric: "safety classification accuracy" [read, Section 4.5].
- Benchmark: Agent-SafetyBench.
- Note: Section 4.5 lists the refined models as "Claude-4.5-Opus, GPT-4o, Qwen-7B, and Qwen-14B" [read]. Appendix C.4.4 lists the target models as "Claude-4.5-Opus, GPT-5, Gemini-3-Pro-Preview" [read]. The paper is inconsistent on this point.
- Separate README claim [read, repo page]: agents "improve from 70.7% to 99.8% accuracy over three rounds with Dafny specification feedback." Not found in the paper text I read. Do not use it as paper evidence until checked against the paper.

### 6. Strong model (Claude Opus?) wrote specs; which model refined; was the judge the same weak model?
**DIFFERENT / PARTLY SUPPORTED.**
- Spec synthesis: Table 12, "Hyperparameters for Agent-SafetyBench experiments", Verification group, rows "Decomposition model | Claude-4.5-Opus" and "Spec synthesis model | Claude-4.5-Opus" [read, two passes]. The two passes disagreed on whether this table covers the main pipeline or the refinement experiment. Neither pass found a sentence tying it to refinement. The refinement pipeline text says only "Feedback Generation: Create structured feedback with verification artifacts" [read, appendix C.4]. So the spec writer in the refinement experiment is NOT STATED.
- Refined model: Qwen-7B and Qwen-14B (the headline numbers). Claude-4.5-Opus and GPT-4o were also listed as refined models in Section 4.5 [read].
- Judge: "The judge model for verification is Claude-4.5-Opus-20251101" [read, appendix C.4, refinement section]. So the judge is a strong model, not the weak Qwen model. The question's premise that the judge was the same weak model is NOT supported.
- Sources: https://arxiv.org/html/2602.11136v1 (appendix C.4, C.8, Table 12)

### 7. README: AgentSafetyBench available; VitaBench and UpwardDeceivers forthcoming; licence
**PARTLY DIFFERENT.** [read, repo page github.com/htlou/FormalJudge; README rendered]
- AgentSafetyBench: status "Available". CONFIRMED.
- UpwardDeceivers: status "Coming soon". CONFIRMED.
- VitaBench: status links to "Experiment workflows and data examples". DIFFERENT: not labelled as forthcoming in the text I read.
- Licence: "This project is released under the MIT License." MIT. CONFIRMED [read].
- Caveat: the raw README URL (raw.githubusercontent.com) and the blob URL both returned HTTP 404 on my fetch, while the repo page itself rendered. The quotes come from a WebFetch pass over the repo page, not the raw file.

### 8. Small models (<= 14B) as spec writer or fact extractor, per the paper
**NOT FOUND.** Small Qwen-2.5 models (7B, 14B) appear only as target agents being refined [read]. No passage I read says a small model writes specs or extracts facts. The model used for LLM extraction is not named in any passage read.

### 9. Results on any coding task (not safety)
**NOT FOUND in the paper.** [read, arXiv v1] The abstract names three benchmarks: behavioral safety, multi-domain constraint adherence, and agentic upward deception detection. No coding-task result was found. The Deceivers benchmark has a "Software Eng." domain with 24 tasks [read]. That is a benchmark domain, not a reported coding result.
- Contradiction to flag: a search snippet [snippet] describes FinanceBench, PrimeBench and CodeBench across seven agent models. This does not match the arXiv abstract (three different benchmarks). Treat the snippet as unreliable. It was not confirmed on the paper.

## Headline result, comparison, and limitations

- Headline: "achieves an average improvement of 16.6% over LLM-as-a-Judge baselines" [read, arXiv abstract]. README gives the same figure across 3 benchmarks and 7 agent models [read].
- Second headline: a 7B judge gets "over 90% accuracy detecting deception from 72B agents" (weak-to-strong) [read, abstract].
- Third: "near-linear safety improvement through iterative refinement" [read, abstract and contributions]. Section 4.5 does not use the phrase "near-linear".
- Compared against: LLM-as-a-Judge baselines. A [snippet] description says six LLM-as-a-Judge setups; not verified in the paper.
- Limitations (stated, Section 5 [read]): errors in intent decomposition; errors in semantic extraction from trajectories; "it does not achieve perfect accuracy". No separate limitations section was found. No small-model or coding limitation is stated.

## Open items

- The PDF could not be read here. Quotes come from WebFetch passes that a small model answered. Verify key quotes (4.5 numbers, Table 12, judge sentence) against the PDF text before publishing.
- The paper's own inconsistency: Section 4.5 vs appendix C.4.4 on the refined model list.
- README "70.7% to 99.8%" is not found in the paper.
- The raw README returned 404. Check the repo directly (git clone is out of scope for this pass).

## Checklist

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | ICML 2026; arXiv ID, authors, title | CONFIRMED | [read] arxiv.org/abs/2602.11136; [read] icml.cc/virtual/2026/poster/61086 |
| 2 | Three LLM roles plus Dafny/Z3 executor | PARTLY; wording DIFFERENT (paper has 4 phases; Python not confirmed) | [read] README; [read] arxiv v1 |
| 3 | Dafny gets LLM-extracted facts, facts not proven | CONFIRMED | [read] arxiv v1 ("delegating all logical composition to deterministic solvers") |
| 4 | Decomposition and fact extraction as main error sources | CONFIRMED | [read] arxiv v1 Section 5 |
| 5 | Qwen 30.8 to 49.4, 34.3 to 74.2; models; rounds; metric | NUMBERS CONFIRMED; models DIFFERENT (Qwen2.5 not Qwen3); 3 rounds; metric = safety classification accuracy | [read] arxiv v1 Sec 4.1, 4.5, Table 11 |
| 6 | Claude Opus wrote specs; judge was weak model? | DIFFERENT/PARTLY: Table 12 lists Claude-4.5-Opus as spec synthesis model, not tied to refinement; judge = Claude-4.5-Opus-20251101 | [read] arxiv v1 appendix C.4, C.8 |
| 7 | README benchmark status; licence | AgentSafetyBench Available and UpwardDeceivers Coming soon CONFIRMED; VitaBench DIFFERENT (links to experiment workflows); MIT CONFIRMED | [read] github.com/htlou/FormalJudge (raw URL 404) |
| 8 | Small models as spec writer or fact extractor | NOT FOUND | [read] arxiv v1 (both halves) |
| 9 | Coding-task results | NOT FOUND in paper; snippet naming CodeBench contradicts abstract | [read] arxiv v1 abstract; [snippet] |
| - | Headline, comparison, limitations | 16.6% avg over LLM-as-a-Judge baselines; 7B judge >90% on 72B deception; Section 5 error sources | [read] arxiv abstract and Section 5 |
