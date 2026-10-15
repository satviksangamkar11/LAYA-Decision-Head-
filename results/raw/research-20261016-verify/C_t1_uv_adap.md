# Web verification of five method claims (A-E)

Date: 2026-10-16. Web research only (WebSearch / WebFetch). No downloads, no code run.

Markers:
- [read] = text returned by WebFetch from the primary page. WebFetch returns a model-summarised rendering, so quotes are as returned, not checked against raw HTML.
- [snippet] = seen only in WebSearch result text, not opened at the primary page.
- Verdicts: CONFIRMED / DIFFERENT / NOT FOUND. "Quote" lines are under 15 words.

---

## A. T1: Tool-integrated Verification (arXiv 2504.04718)

Authors: Minki Kang, Jongwon Jeong, Jaewoong Cho (KRAFTON / KAIST). Venue: ICLR 2026 poster, Apr 23 2026 [read: https://iclr.cc/virtual/2026/poster/10007001]. arXiv comment field says "ICLR 2026" [read: https://arxiv.org/abs/2504.04718]. v1 title was "Tool-integrated Self-verification"; later title "Tool-integrated Verification".

**(1) Offloading numerical/factual checks to tools improved small-model verification: CONFIRMED (qualitative).**
- Quote: "sLMs struggle with verification tasks requiring memorization, such as numerical calculations and fact-checking." [read: https://arxiv.org/abs/2504.04718]
- Quote: "offloading to external tools reduces the memorization burden on sLMs" [read: https://iclr.cc/virtual/2026/poster/10007001]
- Method: two stages. A tool-based filter (e.g. code interpreter) first filters candidates; then a small LM (PRM or GenRM/critic) does final verification [read: abs].
- Motivating arithmetic experiment (N three-digit numbers): code execution removes the accuracy drop [snippet: search summary].

**(2) Llama-3.2 1B verifier beats Llama-3.1 8B on MATH: CONFIRMED, with a setup caveat.**
- Quote: "a Llama-3.2 1B model under test-time scaling outperforms the significantly larger Llama-3.1 8B model." [read: https://arxiv.org/abs/2504.04718]
- Quote: "Notably, with ToolV, only using Llama 1B models outperforms the performance of the 8B model." [read: https://arxiv.org/html/2504.04718v1, Sec. 6.2]
- Setup [read: arXiv HTML v1, Sec. 6.1, Fig. 3 caption]:
  - The small model is the VERIFIER (PRM or GenRM). The paper says "whether sLMs can reliably verify the output candidates".
  - Candidates: "weighted best-of-N" with N = 64, from "a fixed generator". The generator and verifiers are the same base model family in each experiment.
  - The Llama-3.1 8B baseline is "N=1 greedy decoding" (a single run, not best-of-N and not a verifier). So the comparison is 1B verifier + ToolV at N=64 against 8B single greedy. The user's phrase "verifier surpassed a larger verifier" is DIFFERENT: the 8B is not a verifier in that comparison.
- GenRM setting [read: Sec. 6.3]: "on MATH500, 1B GenRM + ToolV outperforms 8B GenRM." Here both are verifiers of different sizes.
- Exact numbers: NOT FOUND in text. Figures 3 and 4 (PRM, GenRM) are images. A search snippet gives MATH500 47.4 (PRM alone) vs 53.2 (ToolV+PRM) for Llama-3.2-1B-Instruct [snippet, PDF-extracted table, UNVERIFIED]. The v2 PDF was binary and could not be read. Do not cite these figures without the figure source.

---

## B. Universal Verifier (arXiv 2604.06240, "The Art of Building Verifiers for Computer Use Agents")

Authors (from paper listing): Corby Rosset, Pratyusha Sharma, Andrew Zhao, Miguel Gonzalez-Fernandez, Ahmed Awadallah [read: https://arxiv.org/abs/2604.06240]. Date April 2026. Code and CUAVerifierBench: github microsoft/fara [snippet]. Microsoft Research affiliation: NOT CONFIRMED in fetched text; the Browserbase blog ties the work to Browserbase [snippet]. Some listings call it "The Universal Verifier for Agent Benchmarks" [snippet].

**(1) Separates task outcome from execution quality / process: CONFIRMED.**
- Quote: "process and outcome rewards that yield complementary signals" [read: abs]
- Process score is continuous, outcome is binary [snippet].
- Human agreement is lower for process than outcome: outcome kappa 0.58, process kappa 0.43; human-human ranges 0.53-0.57 (outcome) and 0.36-0.45 (process), Table 14 [read: https://arxiv.org/html/2604.06240].

**(2) Precise independent rubric criteria: CONFIRMED on the criteria, DIFFERENT on the word "independent".**
- Quote: "constructing rubrics with meaningful, non-overlapping criteria to reduce noise." [read: abs]
- The paper says "non-overlapping", not "independent". Use the paper's word.

**(3) Improved web-agent verification: CONFIRMED on the authors' own numbers (self-reported).**
- Table 2 [read: arXiv HTML]. kappa / FPR:
  - WebVoyager (GPT-4o): internal kappa 0.31, FPR 0.45; Browserbase kappa 0.13, FPR 0.60.
  - WebJudge (o4-mini): internal kappa 0.44, FPR 0.22; Browserbase kappa 0.26, FPR 0.40.
  - Universal Verifier (GPT-5.2): internal kappa 0.64, FPR 0.01; Browserbase kappa 0.58, FPR 0.08.
  - Internal UV outcome accuracy 0.81, F1 0.81.
- Quote: "a reduction in false positive rates to near zero" [read: abs].
- Upgrading baselines to GPT-5.2 lowers their FPR (WebVoyager internal 0.10, WebJudge internal 0.07), but "overall kappa improves only modestly" [read].
- Quote: "agrees with humans as often as humans agree with each other" [read: abs].

**Which model is the verifier: GPT-5.2 (closed, OpenAI) [read: Table 1].** Parameter count not stated.

**Usable with small open models: NOT FOUND (no evidence).**
- The paper says it "can be configured to use any json-capable multimodal LLM available as an endpoint" [read].
- All ablations use OpenAI models (GPT-4o, o4-mini, o3, GPT-5, GPT-5.1, GPT-5.2, GPT-5.4). No open-weight model is tested [read].

---

## C. Adaptive Generate-Rank-Verify (ADAP; arXiv 2605.17609)

Title: "Adaptive Generate-Rank-Verify: Inference-Time Search with Costly Verification". Authors: Shaddin Dughmi, Mahdi Haghifam, Yusuf Hakan Kalayci [read: abs]. arXiv v2 June 7 2026. ICML 2026 listings exist (icml.cc/virtual/2026/75197 and 82547) [snippet].

**(1) Studies adaptive allocation of generation, cheap ranking and expensive verification: CONFIRMED.**
- Framed as "generative active search": a cost-sensitive first-positive search. A policy samples candidates, observes cheap scores, and pays for verifier labels [snippet, search summary of abstract].
- ADAP "progressively increases the number of sampled responses and top-ranked verifications" [snippet].

**(2) Efficiency guarantee requires an informative score ordering: CONFIRMED, with the exact condition below.**
- Exact condition: monotonicity. Quote: "higher reward scores are no less likely to pass verification" [read: abs]. Formally, higher ranker score never lowers the probability of passing the verifier.
- Guarantee: quote "expected cost within a constant factor of the distribution-aware optimum" [read: abs]. No constant stated in the abstract.
- Necessity: learning-theoretic lower bounds via a "centered star number" show structural assumptions on the score-label relation are needed [read: abs]. The star-number definition was not in the fetched text.
- DIFFERENT: an agentpatterns.ai page says the ranker must be "calibrated" and gives rules of thumb (verifier ~10x generation cost, pool of about 16). That is secondary, not the paper. The paper's condition is monotonicity, not calibration. Do not use those thresholds.

**Numbers: NOT FOUND in fetched abstract.** Experiments on "mathematical reasoning and competitive programming" "validate the predicted advantage" over fixed and difficulty-adaptive baselines [read: abs]. No datasets, models, or percentages were in the fetched text. Full paper not read.

---

## D. Small LLM judges / critics that output a discriminating check

**Papers with the exact pattern (judge emits a distinguishing input or counterexample, not a verdict):**

- **S\* (arXiv 2502.14382), "S*: Test Time Scaling for Code Generation"**, Dacheng Li et al. (UC Berkeley). Findings of EMNLP 2025 [snippet: papers.cool].
  - Quote: "adaptively generates distinguishing inputs for pairwise comparison" [read: abs].
  - Quote: "enabling a 3B model to outperform GPT-4o-mini" [read: abs]. The 3B model is not named in the abstract and is not in Table 3.
  - Quote: "DeepSeek-R1-Distill-Qwen-32B with S* achieves 85.7% on LiveCodeBench" [read: abs].
  - Table 3 [read: arXiv HTML v1; LiveCodeBench v4, N=8, temp 0.7, Pass@1]:

    | Model | Public only | Generated tests | LLM judge | Adaptive input synthesis |
    |---|---|---|---|---|
    | Qwen-Coder-7B | 42.3 | 42.3 | 42.3 | 44.5 |
    | Qwen-Coder-32B | 54.6 | 57.8 | 55.5 | 58.3 |
    | GPT-4o mini | 54.1 | 55.2 | 56.3 | 57.3 |
    | QwQ-32B-Preview | 64.3 | 65.9 | 68.6 | 69.7 |
    | Avg. (as printed) | 53.8 | 53.1 | 55.6 | 57.5 |

  - Source inconsistency: the printed Generated-tests average is 53.1, but the mean of its four rows is 55.3. Flag this; the other averages match their rows.
  - Fraction of judge challenges that were real errors: NOT FOUND in abstract or Table 3.
  - Same-model vs different-model judge: NOT FOUND in S*.

- **CodeT (arXiv 2207.10397, ICLR 2023)** [snippet]. Model generates tests and candidates; selection by "dual execution agreement". HumanEval pass@1 65.8% [snippet, unverified]. It selects by agreement, not by a discriminating input.

- **Mokav** [snippet]: LLM writes inputs that make two program versions diverge; 81.7% of program pairs vs 4.9% for Pynguin. Not opened. Unverified.

- **ProbeGen (arXiv 2502.18473)** [snippet]: searches for counterexamples to equivalence; disproves 18% of samples that benchmark tests accepted. Not opened. Unverified.

- **BACE (arXiv 2603.28653)** [snippet]: LLM writes input generators that split candidate clusters; a Bayesian step filters bad tests. Not opened.

**Same-model / correlated-error judges:**

- **Kim et al., "Correlated Errors in Large Language Models" (arXiv 2506.07962), ICML 2025** [read: abs].
  - Quote: "including shared architectures and providers" [read: abs, "factors driving model correlation"].
  - Over 350 LLMs; "models agree 60% of the time when both models err" on one leaderboard [read: abs].
  - DIFFERENT: the claim that each judge inflates less-accurate models' accuracy was NOT in the fetched abstract [snippet only]. Not verified.

- **Vishwarupe, Shadbolt, Alwazae, "A Second Judge is Not a Second Opinion" (NeurIPS 2026 workshop poster)** [read: https://neurips.cc/virtual/2026/172889].
  - Quote: "verifier redundancy is worth only as much as verifier independence" [read].
  - Two open-weight judges kappa 0.879 with each other; human-human 0.565; judge-human about 0.19; open-vs-frontier about 0.10.
  - Domain: subjective labelling of 315 interview utterances, NOT code. Use as an analogy only.

**Fraction of judge challenges that were real errors (across D): NOT FOUND** in any source opened.

**CodeJudgeBench (arXiv 2507.10535)** [snippet]: 26 judges; small thinking models (Qwen3-8B) beat specially trained judges up to 70B; answer order changes accuracy. Not opened.

---

## E. Independent authorization vs compromised upstream validator (arXiv 2609.17648)

Title: "Trust propagation and structural containment in Multi-agent LLM pipelines". Authors: Tanzim Hossain Safin, Sharif Noor Zisad, Swakkhar Shatabda, Ragib Hasan. Submitted 15 Sep 2026 [read: https://arxiv.org/abs/2609.17648].

**Claim: independent authorization prevents unsafe execution even when an upstream validator is compromised: CONFIRMED (one preprint, self-reported).**
- Quote: "the Validator can remain compromised while execution is contained" [read].
- Quote: "with authorization enabled ... 100% JBR but 0% Unsafe Action Rate" [read]. Judgment Bypass Rate (JBR) = compromise at the attacked agent; Unsafe Action Rate = at execution.
- Without authorization, "memory poisoning reaches execution in every undefended trial" [read].
- Mechanism: "an independent authorization layer using task-bound signed tokens and a separately verified policy oracle" [read].
- Oracle containment holds against an attacker who holds the signing secret; an independently authored least-privilege policy preserves it [snippet].
- Observer layer: agent-hijacking false-positive rate 49% to 7%, "without weakening execution-level security" [read].
- Scale: 4-agent LangGraph pipeline (Supervisor, Researcher, Validator, Executor); 60 labelled tasks, 3 seeds [snippet].
- Models used: NOT FOUND.
- Peer review status: not verified.

**Related design paper:** arXiv 2605.05440, "Authorization Propagation in Multi-Agent AI Systems" [snippet]. Argues downstream agents should not inherit upstream trust; deny access at each hop. No numbers seen.

**Related IETF draft:** draft-tonyai-a2a-trust-00 (individual, informational) [snippet]. Proposal only.

---

## Caveats

- Most A-E figures come from one abstract or one HTML render. Numbers in figure images (T1) were not reachable.
- S* Table 3 has a printed-average inconsistency (53.1 vs row mean 55.3).
- UV numbers are self-reported by the authors; the verifier is GPT-5.2 (closed).
- ADAP has no numeric results in the abstract.
- E is a 60-task, 3-seed preprint.

---

## Checklist (end of task)

- [x] DONE: A(1) T1 offloading claim, quotes [read]. Evidence: arxiv.org/abs/2504.04718.
- [x] DONE: A(2) 1B vs 8B claim and setup (verifier = small model; 8B = N=1 greedy). Evidence: arXiv HTML v1 Sec. 6.1-6.3.
- [ ] NOT DONE: A exact MATH numbers. Only figure images and an unverified snippet (47.4 / 53.2). Needs figure extraction or the source tables.
- [x] DONE: B(1)-(3) claims with Table 2 numbers and GPT-5.2 verifier. Evidence: arxiv.org/html/2604.06240.
- [ ] NOT DONE: B Microsoft Research affiliation (not in fetched text). B small-open-model usability: NOT FOUND (only OpenAI models tested).
- [x] DONE: C(1)-(2) ADAP condition (monotonicity) and guarantee wording. Evidence: arxiv.org/abs/2605.17609.
- [ ] NOT DONE: C numbers and the constant in the guarantee (not in abstract; full paper not read).
- [x] DONE: D S* Table 3 read; judge-vs-generated-test comparison.
- [ ] NOT DONE: D fraction of judge challenges that were real errors: NOT FOUND in any opened source.
- [ ] NOT DONE: D Mokav, ProbeGen, BACE, CodeT, CodeJudgeBench numbers: snippet only, not opened.
- [x] DONE: D correlated-error evidence (Kim et al. 2506.07962, Vishwarupe et al.). Note: the second is not code.
- [x] DONE: E paper located and key numbers read (60 tasks, 3 seeds, self-reported).
- [ ] NOT DONE: E models used; peer-review status.
- [ ] BLOCKED: exact T1 numbers and the full ADAP guarantee constant need the full PDFs (PDF/figures not readable via WebFetch in this session). Not a user decision; a tool limit.
- WAITING ON USER: none. No downloads or code runs were needed or done.
