"""[EL-ARCH-1 B3] The controller's side of Qwen generation: it asks a generation backend (interfaces.GenerationBackend) for k candidates / next-action proposals using the FROZEN templates
(qwen_sample_v2, qwen_sample_v2b, the solution-format block of qwen.py; the next-action instruction of the T4 step kernel), and parses the structured output STRICTLY.
Pure standard library; importable without torch. The backends are interchangeable: HFQwenBackend (Kaggle), FixtureGenerationBackend (unit tests) and ReplayGenerationBackend (a Kaggle
generation run replayed on the PC, refusing any request whose prompt hash differs from what Kaggle actually generated).

parse_actions: the answer after the last </think> must be ONE JSON array of {"name","arguments"} objects (an optional single ```json fence around the whole answer is formatting, not repair).
Prose around the array, trailing text, a missing </think>, a non-array, an unknown tool, wrong argument keys or non-string values are REJECTED and counted; nothing is repaired or guessed."""
from __future__ import annotations

import json
import re
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


def render_action_prompt(state_text: str, instr: Optional[str] = None) -> Tuple[str, str]:
    p = state_text + "\n\n" + (ACTION_INSTR if instr is None else instr)
    return p, sha256_text(p)


def _answer_text(text: str) -> Optional[str]:
    """The answer after the last </think> (single ```json fence around the whole answer removed); None when thinking never finished."""
    i = text.rfind(THINK_CLOSE)
    if i < 0:
        return None
    ans = text[i + len(THINK_CLOSE):].strip()
    if ans.startswith(FENCE) and ans.endswith(FENCE) and ans.count(FENCE) == 2:
        first, _, rest = ans[3:].partition("\n")
        if first.strip().lower() in ("", "json"):
            ans = rest[:-3].strip()
    return ans


def _collect(items: Sequence[Tuple[int, Any]], max_actions: int, out: Dict[str, Any]) -> None:
    """Validate (index, object) pairs strictly into out["actions"] / out["rejected"]; shared by the strict and the salvaging parser."""
    seen = set()
    for n, a in items:
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


def parse_actions(text: str, max_actions: int = 5) -> Dict[str, Any]:
    """-> {"status", "actions": [option text], "rejected": [{"index","reason"}], "answer_sha256"}. status: ok | no_think_close | not_json_array | no_valid_action."""
    ans = _answer_text(text)
    if ans is None:
        return {"status": "no_think_close", "actions": [], "rejected": [{"index": None, "reason": "thinking never finished (no </think>)"}], "answer_sha256": sha256_text(text)}
    out = {"status": "ok", "actions": [], "rejected": [], "answer_sha256": sha256_text(ans)}
    try:
        arr = json.loads(ans)
    except ValueError as e:
        return dict(out, status="not_json_array", rejected=[{"index": None, "reason": "answer is not exactly one JSON value: %s" % str(e)[:80]}])
    if not isinstance(arr, list):
        return dict(out, status="not_json_array", rejected=[{"index": None, "reason": "top-level JSON value is not an array"}])
    _collect(list(enumerate(arr)), max_actions, out)
    return out


# ------------------------------------------------------------------ [EL-ARCH-1 F1] v2: per-option salvage, one independent next action per option
ACTION_TEMPLATE_ID_V2 = "qwen_actions_v2"
ACTION_INSTR_V2 = ("Propose exactly 4 DISTINCT ALTERNATIVE candidates for the SINGLE next action of the agent, as ONE JSON list. The candidates compete: exactly one will be executed now, "
                   "so each must be executable right now from the current state and must not depend on another candidate. Do NOT list successive steps of one plan. "
                   "Each item is an object {\"name\": <tool>, \"arguments\": {...}} with tool one of Read {\"file_path\"}, Edit {\"file_path\",\"old_string\",\"new_string\"} or Bash {\"command\"}. "
                   "Every double quote inside a string value must be escaped as \\\". Output only the JSON list.")
ACTION_TEMPLATE_HASH_V2 = sha256_text(ACTION_INSTR_V2)
_OBJ_START = re.compile(r'\{\s*"name"\s*:')
# ------------------------------------------------------------------ [EL-ARCH-1 brief-thinking] v3 = v2 + a brief-reasoning instruction; NO token cutoff on thinking (only a large technical safety ceiling set by the caller)
ACTION_TEMPLATE_ID_V3 = "qwen_actions_v3"
BRIEF_THINKING_INSTR = ("Think BRIEFLY: reason in a few short sentences only, do not re-verify or re-derive the same point, do not enumerate every possibility, and output the JSON list as soon as you have your candidates. ")
ACTION_INSTR_V3 = BRIEF_THINKING_INSTR + ACTION_INSTR_V2
ACTION_TEMPLATE_HASH_V3 = sha256_text(ACTION_INSTR_V3)



def parse_actions_v2(text: str, max_actions: int = 5) -> Dict[str, Any]:
    """Like parse_actions, plus per-option SALVAGE (A1.1): when the answer starts with '[' but the array is not valid JSON (e.g. one unescaped quote), each object that begins with {"name": is decoded on its own;
    objects that decode are validated as usual and kept, one that does not is rejected with its index and the decoder's reason. Nothing inside a broken object is repaired or guessed.
    Prose around the array is still rejected as a whole (salvage needs the answer to start with '['). Result gets "salvaged": bool."""
    strict = parse_actions(text, max_actions)
    ans = _answer_text(text)
    if strict["status"] != "not_json_array" or ans is None or not ans.startswith("["):
        return dict(strict, salvaged=False)
    out = {"status": "ok", "actions": [], "rejected": [], "answer_sha256": strict["answer_sha256"], "salvaged": True}
    dec, pos, n, items = json.JSONDecoder(), 1, 0, []
    while True:
        m = _OBJ_START.search(ans, pos)
        if m is None:
            break
        try:
            obj, end = dec.raw_decode(ans, m.start())
            items.append((n, obj))
            pos = end
        except ValueError as e:
            out["rejected"].append({"index": n, "reason": "option %d is not valid JSON (%s); rejected, not repaired" % (n, str(e)[:60])})
            pos = m.end()
        n += 1
    if n == 0:
        return dict(strict, salvaged=False)
    _collect(items, max_actions, out)
    out["rejected"].sort(key=lambda r: r["index"])
    return out


def parse_candidates_v2(text: str, seed: Any = None) -> Dict[str, Any]:
    """Parse one model answer into CandidateIn items for candidate_filter.filter_options WITHOUT judging them: -> {"status", "items"}. Items keep generation order; a decodable {"name","arguments"}
    object becomes {"tool","arguments","origin","raw"} (duplicates and bad tools/arguments included: the filter decides), anything else - an undecodable fragment of a broken array, a wrong-shaped
    entry - becomes {"malformed": True, "origin", "raw"}. Same entry rule as parse_actions_v2: JSON array, or salvage when the answer starts with '['. status: ok | no_think_close | not_json_array | no_valid_action."""
    ans = _answer_text(text)
    if ans is None:
        return {"status": "no_think_close", "items": []}
    raw_items: List[Tuple[Any, Optional[str]]] = []      # (decoded object or None, raw text)
    try:
        arr = json.loads(ans)
        if not isinstance(arr, list):
            return {"status": "not_json_array", "items": []}
        raw_items = [(a, json.dumps(a, ensure_ascii=False, sort_keys=True)) for a in arr]
    except ValueError:
        if not ans.startswith("["):
            return {"status": "not_json_array", "items": []}
        dec, pos = json.JSONDecoder(), 1
        starts = [m.start() for m in _OBJ_START.finditer(ans)]
        while True:
            m = _OBJ_START.search(ans, pos)
            if m is None:
                break
            nxt = next((s for s in starts if s > m.start()), len(ans))
            try:
                obj, end = dec.raw_decode(ans, m.start())
                raw_items.append((obj, ans[m.start():end]))
                pos = end
            except ValueError:
                raw_items.append((None, ans[m.start():nxt].rstrip(" ,]\n")))
                pos = m.end()
    items = []
    for k, (obj, raw) in enumerate(raw_items):
        origin = {"seed": seed, "index": k}
        if isinstance(obj, dict) and set(obj) == {"name", "arguments"} and isinstance(obj["name"], str):
            items.append({"tool": obj["name"], "arguments": obj["arguments"], "origin": origin, "raw": raw})
        else:
            items.append({"malformed": True, "origin": origin, "raw": raw})
    return {"status": "ok" if items else "no_valid_action", "items": items}


class ActionProposer:
    """Next-action proposals for the per-action controller from any GenerationBackend. One Generation per seed; actions are merged in seed order (first seen wins, duplicates counted).
    Frozen v1 behaviour (template qwen_actions_v1, strict parser); ActionProposerV2 below adds salvage, the one-next-action prompt and the candidate filter."""
    TEMPLATE_ID, INSTR = ACTION_TEMPLATE_ID, ACTION_INSTR

    def __init__(self, backend, *, max_new_tokens: int = 700, max_actions: int = 5, min_actions: int = 2):
        self.backend, self.max_new_tokens, self.max_actions, self.min_actions = backend, max_new_tokens, max_actions, min_actions

    def _parse(self, text: str, seed: Any = None, ctx: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return parse_actions(text, self.max_actions)

    def propose(self, task_id: str, step: int, state_text: str, seeds: Sequence[int] = (0,), **ctx: Any) -> Dict[str, Any]:
        prompt, psha = render_action_prompt(state_text, self.INSTR)
        req = GenerationRequest("%s:step%d" % (task_id, step), prompt, len(seeds), tuple(int(s) for s in seeds), self.max_new_tokens, "sample", True, self.TEMPLATE_ID)
        gens = list(self.backend.generate(req))
        options: List[str] = []
        rejected: List[Dict[str, Any]] = []
        statuses = []
        filters: List[Dict[str, Any]] = []
        cands_all: List[Dict[str, Any]] = []
        for g in gens:
            pa = self._parse(g.text, g.seed, ctx)
            statuses.append(pa["status"])
            if "filter" in pa:
                filters.append(pa["filter"])
            cands_all.extend(pa.get("items", []))
            for a in pa["actions"]:
                if a not in options and len(options) < self.max_actions:
                    options.append(a)
                else:
                    rejected.append({"seed": g.seed, "reason": "duplicate or beyond max_actions", "rule": "duplicate" if a in options else "beyond_max_options"})
            rejected += [dict(r, seed=g.seed) for r in pa["rejected"]]
        n_in = sum(f["counts"]["in"] for f in filters)
        if cands_all:      # prefilter=False: the raw items go to the controller's own single filter_options call; no option is judged here
            status = "ok"
        else:
            status = "ok" if len(options) >= self.min_actions else ("too_few_actions" if options else ("no_valid_action" if n_in else statuses[0] if statuses else "no_generation"))
        out = {"status": status, "options": options, "rejected": rejected, "prompt_sha256": psha, "statuses": statuses, "raw_sha256": [sha256_text(g.text) for g in gens],
               "qwen_meta": {"prompt_tokens": sum(g.tokens_prompt for g in gens), "new_tokens": sum(g.tokens_generated for g in gens), "kaggle_seconds": sum(g.wall_seconds for g in gens)},
               "generations": [{"seed": g.seed, "model": g.model, "model_revision": g.model_revision, "prompt_sha256": g.prompt_sha256, "template": g.prompt_template_id} for g in gens]}
        if cands_all:
            out["candidates"] = cands_all
        if filters:      # v2: the per-generation FilterResults (journalled as-is) and, when nothing survives, the explicit abstain (T24)
            out["filter_results"] = filters
            if not options:
                out["abstain"] = ({"code": "all_options_rejected", "reason": "no valid option after filtering: all %d options rejected" % n_in} if n_in
                                  else {"code": "no_options_proposed", "reason": "no options proposed"})
                out["abstain_reason"] = out["abstain"]["reason"]
        return out


class ActionProposerV2(ActionProposer):
    """[EL-ARCH-1 F1] qwen_actions_v2 (one independent next action per option) + per-option salvage (parse_actions_v2) + candidate_filter.filter_options.
    propose(..., files={path: text}, path_scope=None) - pass the current workspace file texts to enable the unappliable-edit checks. The filter runs once per generation (so the
    sequential-plan rule sees one model answer), then generations are merged in seed order with the usual dedupe."""
    TEMPLATE_ID, INSTR = ACTION_TEMPLATE_ID_V2, ACTION_INSTR_V2

    def __init__(self, backend, *, prefilter: bool = True, **kw: Any):
        """prefilter=True (default): filter_options runs here, once per generation. prefilter=False: filter_options is NOT called; propose() returns the parsed CandidateIn items (malformed ones
        and duplicates untouched, generation order) in out["candidates"] with options == [], so the controller's single filter_options call journals every rejection (frozen interface)."""
        super().__init__(backend, **kw)
        self.prefilter = bool(prefilter)

    def _parse(self, text: str, seed: Any = None, ctx: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        ctx = ctx or {}
        pc = parse_candidates_v2(text, seed)
        if not self.prefilter:
            return {"status": pc["status"], "actions": [], "rejected": [], "items": pc["items"]}
        from evidence_ladder.controller.candidate_filter import filter_options
        res = filter_options(pc["items"], ctx.get("files"), max_options=self.max_actions, path_scope=ctx.get("path_scope"))
        acts = [render_call(a["tool"], a["arguments"]) for a in res["accepted"]]
        rej = [dict(index=r["origin"].get("index"), reason=r["reason"], rule=r["rule"], stage="filter") for r in res["rejected"]]
        return {"status": pc["status"] if not pc["items"] else ("ok" if acts else "no_valid_action"), "actions": acts, "rejected": rej, "filter": res}


# ------------------------------------------------------------------ function-level sampling plan (qwen_sample_v2 / v2b + solution-format block)
FROZEN_STRATEGIES = ("qwen_sample_v2", "qwen_sample_v2b")


def strategy_for(index: int, strategies: Sequence[str] = FROZEN_STRATEGIES) -> str:
    """sample index -> template id. The frozen pair keeps qwen_v2b.TEMPLATES_BY_SAMPLE (0-1 -> v2, 2-3 -> v2b, then repeats); any other tuple cycles one by one."""
    if tuple(strategies) == FROZEN_STRATEGIES:
        return V2B.TEMPLATES_BY_SAMPLE[index % len(V2B.TEMPLATES_BY_SAMPLE)]
    return strategies[index % len(strategies)]


LCB_TEMPLATE_ID = "lcb_generic_v1"      # [P5] LiveCodeBench's generic prompt: the task prompt IS the rendered LiveCodeBench prompt (tools/lcb_tasks.render_prompt)


def render_sample_prompt(template_id: str, task) -> Tuple[str, str]:
    if template_id == LCB_TEMPLATE_ID:
        return task.prompt, LCB_TEMPLATE_ID
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
class ActionProposerV3(ActionProposerV2):
    """[EL-ARCH-1 brief-thinking] qwen_actions_v3: the v2 behaviour (salvage, one independent next action, filter) with a brief-reasoning instruction in the prompt. It does NOT cut thinking by a token
    count; the caller sets only a large technical safety ceiling (max_new_tokens), and a sample that hits it is a failure (no closed think block), never a success."""
    TEMPLATE_ID, INSTR = ACTION_TEMPLATE_ID_V3, ACTION_INSTR_V3


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
