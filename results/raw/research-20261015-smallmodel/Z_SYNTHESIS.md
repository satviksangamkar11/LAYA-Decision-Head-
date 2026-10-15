# Synthesis: making small models answer as well as possible, with adapter + LAYA + loop engineering (2026-10-15)

Inputs: A_methods, B_coverage_gap, C_test_time_compute, D_scaffolds_memory_steering, E_self_improvement, F_capacity_routing_uncertainty,
G_unconventional, H_general_agents (this folder; Haiku 5.5 web research, mostly abstracts and snippets: every number below is the papers' own and
is marked in the source files as [abs]/[snip]/[unverified]). Our own measured results are cited with their results/raw folders.

## 1. Goal and judging criteria (user, 2026-10-15)
Best possible final answers from the small local model (Qwen family); the system includes the adapter (Qwen states -> LAYA), the pretrained LAYA
head and loop engineering; no strong model at runtime (offline use allowed). Judged by: measured final-answer quality (tests pass / correct answer),
runtime constraint, our hardware (6 GB GPU, 15 GB RAM, free Kaggle), evidence strength, and LAYA kept only where it beats simpler baselines.

## 2. What the evidence says
1. Untrained small models solve almost nothing at repository level (1.8-7% SWE-bench Verified, 7-32B zero-shot) [B]; trained 4B models reach
   27.6-40.7% [B]. Function-level coding is where a frozen 4B already succeeds (Qwen3-4B thinking: LiveCodeBench v5 54.2) [B].
2. Selection with real execution beats voting: S* lets a 3B model beat GPT-4o-mini on LiveCodeBench [C]; CodeT +18.8 points [C]; verifiers recover
   about 50-62% of the pass@1 -> pass@k gap at 32B [B]; public tests recover 69% at k=5, generated tests 23%, an LLM judge 30% [B].
3. Hidden-state probes predict correctness: AUC up to 0.83 (2607.05188) [F]; 0.91 on Qwen3-4B vs 0.75 length baseline (2606.14530, unverified) [D];
   probes on 18 tool-calling LLMs detect wrong-value calls (NeurIPS 2026 workshop) [H]; small probes reported to match PRMs up to 810x larger [H, unverified].
4. Repair helps only with real feedback (Self-Debugging with tests) [C]; open small models self-repair poorly (4.4% vs 28.9%) [A, snippet].
5. Compute should follow difficulty (Snell et al.; Inference Scaling Laws) [C]; confidence early stop (DeepConf) saves up to 84.7% tokens on math [C].
6. No small learned steering component (PRM, value head, reranker) beat simple baselines on SWE-bench [A]; SWE-Shepherd PRM 51% vs 57% baseline [A].
   Matches our own LAYA results (head swap, probe > head).
7. Debate / mixture-of-agents do not beat self-consistency at equal compute; agent-level test-time scaling unreliable; more thinking can hurt [C].
8. Training the small model is the largest measured lever: ~500 trajectories multiplied 7-32B success several times [B]; ToolRL 1.5B BFCL 19.4 -> 46.2 [H];
   When2Call abstain training cut tool hallucination 16% -> 1.9% at 4B [H]; self-training on verified own attempts (ReST/STaR) [E].
9. A stronger small base fits: Qwen3.5-9B at 4-bit in 6 GB (LiveCodeBench v6 65.6, official card) [F]; RAM (15.3 GB) is the binding limit [F].

## 3. Ranked options for adapter + LAYA + loop (Qwen frozen)
| rank | option | role of adapter + LAYA | evidence | cost | cheapest experiment |
|---|---|---|---|---|---|
| 1 | Correctness verifier over the small model's OWN candidates | score each candidate's P(correct) from Qwen states (LAYA native `outcome` question) | items 2, 3 | low-medium (Kaggle sampling + capture) | verifier-gap test (section 4) |
| 2 | Compute controller in the loop | decide stop / sample more / retry with feedback / abstain (LAYA native `action` question) | items 4, 5 | low once 1 exists | accuracy-vs-compute curve on the same data |
| 3 | Tool-call verdict for general agents | LAYA native `tool_call.call_verdict` (execute / fix_args / wrong_tool / abstain), labels from schema validation and real execution errors | item 3 (tool-call probes), item 8 (When2Call) | medium | BFCL-style subset: small model proposes calls, executor labels them |
| 4 | Portability: new adapter per Qwen model | train only a new adapter (head shared) for Qwen3.5-9B | item 9 | medium | repeat option 1 with Qwen3.5-9B |
| 5 | Scaffold hygiene around the loop | none (deterministic): observation masking (~half cost, same solve rate), Agentless localize -> sample -> validate, constrained decoding for syntax only with an abstain option | D | low | add to whichever loop is built |
Mandatory competitors for LAYA's slot in every experiment: linear probe on the same states, Qwen's own log-probability confidence (DeepConf-style),
self-consistency (majority vote), random-head adapter, and the oracle (best of k). LAYA keeps the slot only if it beats them at equal compute.

## 4. First experiment: verifier-gap test (function-level coding with tests)
- Problems: a public function-level set with tests (e.g. MBPP+/HumanEval+/LiveCodeBench subset; licence and size to check; small download).
- Qwen3-4B-Thinking-2507 (current backbone), k = 8 attempts per problem on Kaggle; Qwen states captured with the existing protocol.
- Every attempt executed against the tests (sandboxed subprocess with timeouts; location to decide) -> pass/fail label.
- Train adapter (head frozen) / random-head adapter / linear probe; grouped CV by problem; DeepConf-style and self-consistency baselines.
- Report: pass@1, pass@8 (ceiling), selected@8 per judge, share of the gap recovered, calibration; then the adaptive loop's accuracy vs compute.
- Gate (to pre-register): LAYA counts only if it recovers more of the gap than the probe and DeepConf with a trajectory/problem-clustered CI above 0.

## 5. Options that need the user's rule change (training Qwen itself)
Largest measured levers for answer quality: self-training on verified own attempts (ReST/STaR) [E], small offline distillation (~500-1k traces) [B, E],
execution-preference training (PLUM, CodeDPO) [G], RL with test rewards (ToolRL, SWE-Master) [H, A]. These conflict with "Qwen is never trained".

## 6. Watch list / not recommended
Watch: evolutionary search with an execution evaluator (ShinkaEvolve) as a heavy loop with LAYA as pre-filter; layer repetition at inference;
repository test-time training; hypernetwork adapters (Text-to-LoRA) [G]. Not recommended now: debate / mixture-of-agents, self-critique without
execution, unbounded thinking, steering for capability, model merging, PRMs without evidence [C, D, A].

## 7. Honest risks
The probe may again match LAYA (then the probe takes the slot); published verifier gains are for 32B-class models, not 4B; labels from tests reward
passing the given tests, not full correctness; executing model-written code needs isolation.
