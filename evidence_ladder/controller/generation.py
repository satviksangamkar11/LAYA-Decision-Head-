"""[EL-ARCH-1 B3] The controller's side of Qwen generation: it asks a generation backend (interfaces.GenerationBackend) for k candidates / next-action proposals using the FROZEN templates
(qwen_sample_v2, qwen_sample_v2b, the solution-format block of qwen.py; the next-action instruction of the T4 step kernel), and parses the structured output STRICTLY.
Pure standard library; importable without torch. The backends are interchangeable: HFQwenBackend (Kaggle), FixtureGenerationBackend (unit tests) and ReplayGenerationBackend (a Kaggle
generation run replayed on the PC, refusing any request whose prompt hash differs from what Kaggle actually generated).

parse_actions: the answer after the last </think> must be ONE JSON array of {"name","arguments"} objects (an optional single ```json fence around the whole answer is formatting, not repair).
Prose around the array, trailing text, a missing </think>, a non-array, an unknown tool, wrong argument keys or non-string values are REJECTED and counted; nothing is repaired or guessed."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evidence_ladder.common.interfaces import Generation, GenerationRequest, sha256_text
from evidence_ladder.models import qwen as Q
from evidence_ladder.models import qwen_v2b as V2B

ACTION_TEMPLATE_ID = "qwen_actions_v1"
# verbatim from kaggle/evidence_ladder_laya_t6_step_v1.py (INSTR / TOOLS); a test compares them with the kernel file
ACTION_TOOLS: Dict[str, frozenset] = {"Read": frozenset({"file_path"}), "Edit": frozenset({"file_path", "old_string", "new_string"}), "Bash": frozenset({"command"})}
ACTION_INSTR = ("Propose exactly 4 DISTINCT candidate next actions for the agent, as ONE JSON list. Each item is an object {\"name\": <tool>, \"arguments\": {...}} with tool one of "
                "Read {\"file_path\"}, Edit {\"file_path\",\"old_string\",\"new_string\"} or Bash {\"command\"}. Output only the JSON list.")
ACTION_TEMPLATE_HASH = sha256_text(ACTION_INSTR)
THINK_CLOSE = "</think>"
FENCE = "`" * 3


def render_call(name: str, args: Mapping[str, Any]) -> str:
    """Same text as decision_head.fable_pool.render_call (the option text LAYA was trained on): compact, sorted keys, ensure_ascii False."""
    return '{"name":%s,"arguments":%s}' % (json.dumps(name, ensure_ascii=False), json.dumps(args, ensure_ascii=False, separators=(",", ":"), sort_keys=True))


def render_action_prompt(state_text: str) -> Tuple[str, str]:
    p = state_text + "\n\n" + ACTION_INSTR
    return p, sha256_text(p)


def parse_actions(text: str, max_actions: int = 5) -> Dict[str, Any]:
    """-> {"status", "actions": [option text], "rejected": [{"index","reason"}], "answer_sha256"}. status: ok | no_think_close | not_json_array | no_valid_action."""
    i = text.rfind(THINK_CLOSE)
    if i < 0:
        return {"status": "no_think_close", "actions": [], "rejected": [{"index": None, "reason": "thinking never finished (no </think>)"}], "answer_sha256": sha256_text(text)}
    ans = text[i + len(THINK_CLOSE):].strip()
    if ans.startswith(FENCE) and ans.endswith(FENCE) and ans.count(FENCE) == 2:
        first, _, rest = ans[3:].partition("\n")
        if first.strip().lower() in ("", "json"):
            ans = rest[:-3].strip()
    out = {"status": "ok", "actions": [], "rejected": [], "answer_sha256": sha256_text(ans)}
    try:
        arr = json.loads(ans)
    except ValueError as e:
        return dict(out, status="not_json_array", rejected=[{"index": None, "reason": "answer is not exactly one JSON value: %s" % str(e)[:80]}])
    if not isinstance(arr, list):
        return dict(out, status="not_json_array", rejected=[{"index": None, "reason": "top-level JSON value is not an array"}])
    seen = set()
    for n, a in enumerate(arr):
        if not (isinstance(a, dict) and set(a) == {"name", "arguments"} and a["name"] in ACTION_TOOLS and isinstance(a["arguments"], dict)):
            out["rejected"].append({"index": n, "reason": "not an {name, arguments} object of a known tool"})
        elif set(a["arguments"]) != ACTION_TOOLS[a["name"]] or not all(isinstance(v, str) for v in a["arguments"].values()):
            out["rejected"].append({"index": n, "reason": "arguments of %s are not exactly %s with string values" % (a["name"], sorted(ACTION_TOOLS[a["name"]]))})
        else:
            s = render_call(a["name"], a["arguments"])
            if s in seen:
                out["rejected"].append({"index": n, "reason": "duplicate"})
            elif len(out["actions"]) >= max_actions:
                out["rejected"].append({"index": n, "reason": "beyond max_actions=%d" % max_actions})
            else:
                seen.add(s)
                out["actions"].append(s)
    if not out["actions"]:
        out["status"] = "no_valid_action"
    return out


class ActionProposer:
    """Next-action proposals for the per-action controller from any GenerationBackend. One Generation per seed; actions are merged in seed order (first seen wins, duplicates counted)."""

    def __init__(self, backend, *, max_new_tokens: int = 700, max_actions: int = 5, min_actions: int = 2):
        self.backend, self.max_new_tokens, self.max_actions, self.min_actions = backend, max_new_tokens, max_actions, min_actions

    def propose(self, task_id: str, step: int, state_text: str, seeds: Sequence[int] = (0,)) -> Dict[str, Any]:
        prompt, psha = render_action_prompt(state_text)
        req = GenerationRequest("%s:step%d" % (task_id, step), prompt, len(seeds), tuple(int(s) for s in seeds), self.max_new_tokens, "sample", True, ACTION_TEMPLATE_ID)
        gens = list(self.backend.generate(req))
        options: List[str] = []
        rejected: List[Dict[str, Any]] = []
        statuses = []
        for g in gens:
            pa = parse_actions(g.text, self.max_actions)
            statuses.append(pa["status"])
            for a in pa["actions"]:
                if a not in options and len(options) < self.max_actions:
                    options.append(a)
                else:
                    rejected.append({"seed": g.seed, "reason": "duplicate or beyond max_actions"})
            rejected += [dict(r, seed=g.seed) for r in pa["rejected"]]
        status = "ok" if len(options) >= self.min_actions else ("too_few_actions" if options else (statuses[0] if statuses else "no_generation"))
        return {"status": status, "options": options, "rejected": rejected, "prompt_sha256": psha, "statuses": statuses, "raw_sha256": [sha256_text(g.text) for g in gens],
                "qwen_meta": {"prompt_tokens": sum(g.tokens_prompt for g in gens), "new_tokens": sum(g.tokens_generated for g in gens), "kaggle_seconds": sum(g.wall_seconds for g in gens)},
                "generations": [{"seed": g.seed, "model": g.model, "model_revision": g.model_revision, "prompt_sha256": g.prompt_sha256, "template": g.prompt_template_id} for g in gens]}


# ------------------------------------------------------------------ function-level sampling plan (qwen_sample_v2 / v2b + solution-format block)
FROZEN_STRATEGIES = ("qwen_sample_v2", "qwen_sample_v2b")


def strategy_for(index: int, strategies: Sequence[str] = FROZEN_STRATEGIES) -> str:
    """sample index -> template id. The frozen pair keeps qwen_v2b.TEMPLATES_BY_SAMPLE (0-1 -> v2, 2-3 -> v2b, then repeats); any other tuple cycles one by one."""
    if tuple(strategies) == FROZEN_STRATEGIES:
        return V2B.TEMPLATES_BY_SAMPLE[index % len(V2B.TEMPLATES_BY_SAMPLE)]
    return strategies[index % len(strategies)]


def render_sample_prompt(template_id: str, task) -> Tuple[str, str]:
    fields = dict(task_prompt=task.prompt, examples=Q.format_examples(task.public_tests), solution_format=Q.SOLUTION_FORMATS[task.language], language=task.language)
    return V2B.render(template_id, **fields)


def plan_samples(task, n: int, seeds: Sequence[int], start_index: int = 0, strategies: Sequence[str] = FROZEN_STRATEGIES, max_new_tokens: int = 4096) -> List[GenerationRequest]:
    """k candidates -> one GenerationRequest per strategy run (consecutive samples of the same template share a request). The prompt carries the required solution format (F13)."""
    groups: List[Tuple[str, List[int]]] = []
    for j in range(n):
        tid = strategy_for(start_index + j, strategies)
        if groups and groups[-1][0] == tid:
            groups[-1][1].append(int(seeds[j]))
        else:
            groups.append((tid, [int(seeds[j])]))
    out = []
    for tid, sd in groups:
        prompt, _ = render_sample_prompt(tid, task)
        out.append(GenerationRequest(task.task_id, prompt, len(sd), tuple(sd), max_new_tokens, "sample", True, tid))
    return out


# ------------------------------------------------------------------ Kaggle -> PC hand-off
class PromptMismatch(RuntimeError):
    """The controller asked for a prompt other than the one the recorded generation was made from."""


class ReplayGenerationBackend:
    """Serves generations recorded by evidence_ladder.models.gen_entry.run_jobs (Kaggle output). Keyed by (task_id, seed, purpose); the request's prompt hash must equal the recorded one,
    otherwise PromptMismatch (a replay can never silently answer a different question)."""

    def __init__(self, records: Iterable[Mapping[str, Any]]):
        self._by: Dict[Tuple[str, int, str], Generation] = {}
        for r in records:
            g = Q.generation_from_dict(r["generation"])
            key = (r["task_id"], int(r["seed"]), r["purpose"])
            if key in self._by:
                raise ValueError("duplicate record %r" % (key,))
            self._by[key] = g

    @classmethod
    def from_jsonl(cls, path) -> "ReplayGenerationBackend":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.loads(l) for l in f if l.strip())

    def generate(self, request: GenerationRequest) -> List[Generation]:
        want = sha256_text(request.prompt)
        out = []
        for seed in list(request.seeds)[:request.n]:
            g = self._by.get((request.task_id, int(seed), request.purpose))
            if g is None:
                raise Q.FixtureMissing((request.task_id, int(seed), request.purpose))
            if g.prompt_sha256 != want:
                raise PromptMismatch("recorded prompt %s != requested %s for %s seed %s" % (g.prompt_sha256[:12], want[:12], request.task_id, seed))
            out.append(g)
        return out

    def model_info(self) -> Dict[str, Any]:
        return {"backend": "replay_of_kaggle_generation", "repo": Q.QWEN_REPO, "revision": Q.QWEN_REVISION, "records": len(self._by)}
