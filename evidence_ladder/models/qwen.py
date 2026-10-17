"""Qwen3-4B-Thinking-2507 connectors: prompt templates, code extraction, fixture replay backend, GATED HF backend, token budget accounting.

HFQwenBackend refuses unless authorized=True AND /kaggle exists AND EVIDENCE_LADDER_ALLOW_INFERENCE=1 (its GPU path has still never run on Kaggle: LOGIC-ONLY until the first Kaggle receipt)."""
from __future__ import annotations

import dataclasses
import json
import os
import string
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evidence_ladder.common.interfaces import (Budget, BudgetExhausted, Generation, GenerationRequest, NotAuthorized, sha256_text)
from evidence_ladder.models._util import canonical_json

QWEN_REPO = "Qwen/Qwen3-4B-Thinking-2507"
QWEN_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

# ------------------------------------------------------------------ templates
TEMPLATES: Dict[str, str] = {
    "qwen_sample_v1": (
        "{task_prompt}\n\n"
        "Public examples (input and expected output):\n{examples}\n\n"
        "Think step by step, then give your final complete solution in {language} as ONE fenced code block. "
        "The code block must be the last thing in your answer."
    ),
    # qwen_sample_v2 (new, v1 untouched): v1 gave the model no solution format, so FC-3 samples were stdin programs (reason: configs/qwen_sample_v2_reason.md).
    # {solution_format} is SOLUTION_FORMATS[language], the task kit's SOLUTION_FORMAT.md rule restated for the model.
    "qwen_sample_v2": (
        "{task_prompt}\n\n"
        "Public examples (input and expected output):\n{examples}\n\n"
        "Required solution format:\n{solution_format}\n\n"
        "Think step by step, then give your final complete solution in {language} as ONE fenced code block. "
        "The code block must be the last thing in your answer."
    ),
    # Fixed repair template = architecture doc section 6, verbatim first block; the task and the previous source are appended because the model cannot repair code it cannot see.
    "qwen_repair_v1": (
        "Previous candidate: {candidate_id}.\n"
        "Verified failure (level {level}): {input} -> expected {expected}, observed {observed}.\n"
        "Requirement (quoted): \"{requirement}\".\n"
        "Unchanged constraints: {constraints}.\n"
        "Task: propose the smallest corrected change. Do not modify anything outside {scope}.\n\n"
        "Original task:\n{task_prompt}\n\n"
        "Previous candidate code:\n{source}\n\n"
        "Give the corrected complete solution in {language} as ONE fenced code block, last in your answer."
    ),
    "qwen_tests_v1": (
        "{task_prompt}\n\n"
        "Public examples (input and expected output):\n{examples}\n\n"
        "Write {n_tests} additional test cases that follow from the statement. Do not copy the public examples. "
        "Output ONE fenced json code block: a list of objects with the keys \"input\" and \"expected\"."
    ),
}
SOLUTION_FORMATS: Dict[str, str] = {
    "python": ("Write one Python file that defines `def solve(data):`. `data` is the task's input dictionary (the \"input\" value of an example). "
               "Return a JSON-serializable Python value. Do NOT read standard input and do NOT print the answer: the grader imports your file and calls solve(data) once per case."),
    "java": ("Write one complete Java 21 source file that defines exactly `import java.util.*;` and `public final class Solution { public static Object solve(Map<String,Object> data) { ... } }`. "
             "JSON numbers arrive as `Number` (use `((Number) x).intValue()` or `.longValue()`), arrays as `List<?>`, objects as `Map<?,?>`. Return only JSON-serializable values. "
             "Do NOT read standard input, print the answer, or write a `main` method: the grader calls Solution.solve(data) once per case. Standard library only."),
}
TEMPLATE_HASHES: Dict[str, str] = {k: sha256_text(v) for k, v in TEMPLATES.items()}


def template_hash(template_id: str) -> str:
    return TEMPLATE_HASHES[template_id]


def render(template_id: str, **fields: Any) -> Tuple[str, str]:
    """-> (prompt, sha256 of the rendered prompt). Fields must match the template exactly."""
    tpl = TEMPLATES[template_id]
    need = {f for _, f, _, _ in string.Formatter().parse(tpl) if f}
    if set(fields) != need:
        raise ValueError("template %s needs fields %s, got %s" % (template_id, sorted(need), sorted(fields)))
    prompt = tpl.format(**fields)
    return prompt, sha256_text(prompt)


def _s(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True, ensure_ascii=False)


def format_examples(public_tests: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join("Example %d\nInput:\n%s\nExpected output:\n%s" % (i + 1, _s(t["input"]), _s(t["expected"])) for i, t in enumerate(public_tests))


# ------------------------------------------------------------------ extraction
LANG_TAGS = {"python": {"", "python", "py", "python3"}, "java": {"", "java"}, "json": {"", "json"}}
THINK_CLOSE = "</think>"


def extract_code(text: str, language: str = "python") -> Tuple[str, str]:
    """Final fenced code block AFTER the last </think>. -> (code, "ok"|"no_code").
    no_code when: no </think> (thinking never finished), no closed block of the right language, or the last fence is unclosed (truncated)."""
    text = text.replace("\r\n", "\n")
    i = text.rfind(THINK_CLOSE)
    if i < 0:
        return "", "no_code"
    tags = LANG_TAGS[language]
    blocks: List[Tuple[str, str]] = []
    cur: Optional[List[str]] = None
    tag = ""
    for line in text[i + len(THINK_CLOSE):].split("\n"):
        s = line.strip()
        if cur is None:
            if s.startswith("```"):
                parts = s[3:].strip().lower().split()
                tag, cur = (parts[0] if parts else ""), []
        elif s == "```":
            blocks.append((tag, "\n".join(cur)))
            cur = None
        else:
            cur.append(line)
    if cur is not None:
        return "", "no_code"
    ok = [b for t, b in blocks if t in tags and b.strip()]
    return (ok[-1], "ok") if ok else ("", "no_code")


def make_generation(text: str, *, tokens_prompt: int, tokens_generated: int, seed: int, prompt_template_id: str, prompt_sha256: str,
                    language: str = "python", model: str = QWEN_REPO, model_revision: str = QWEN_REVISION, wall_seconds: float = 0.0,
                    logprob_summary: Optional[Dict[str, float]] = None) -> Generation:
    code, status = extract_code(text, language)
    return Generation(text, code, status, tokens_prompt, tokens_generated, seed, model, model_revision, prompt_template_id, prompt_sha256,
                      wall_seconds, logprob_summary)


# ------------------------------------------------------------------ serialization
_GEN_FIELDS = tuple(f.name for f in dataclasses.fields(Generation))


def generation_to_dict(g: Generation) -> Dict[str, Any]:
    return dataclasses.asdict(g)


def generation_from_dict(d: Mapping[str, Any]) -> Generation:
    extra = set(d) - set(_GEN_FIELDS)
    if extra:
        raise ValueError("unknown Generation fields %s" % sorted(extra))
    if d.get("extract_status") not in ("ok", "no_code"):
        raise ValueError("bad extract_status %r" % d.get("extract_status"))
    return Generation(**{k: d[k] for k in _GEN_FIELDS if k in d})


def generation_to_json(g: Generation) -> str:
    return canonical_json(generation_to_dict(g))


def generation_from_json(s: str) -> Generation:
    return generation_from_dict(json.loads(s))


# ------------------------------------------------------------------ fixture backend
class FixtureMissing(KeyError):
    pass


class FixtureGenerationBackend:
    """Deterministic replay. Records: {"task_id","seed","purpose","generation":{Generation fields}}, keyed by (task_id, seed, purpose)."""

    def __init__(self, records: Iterable[Mapping[str, Any]]):
        self._by_key: Dict[Tuple[str, int, str], Generation] = {}
        for r in records:
            key = (r["task_id"], int(r["seed"]), r["purpose"])
            if key in self._by_key:
                raise ValueError("duplicate fixture key %r" % (key,))
            g = generation_from_dict(r["generation"])
            if g.seed != key[1]:
                raise ValueError("fixture seed mismatch for %r" % (key,))
            self._by_key[key] = g

    @classmethod
    def from_file(cls, path: str) -> "FixtureGenerationBackend":
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        return cls(doc["records"])

    def generate(self, request: GenerationRequest) -> List[Generation]:
        seeds = list(request.seeds)
        if len(seeds) < request.n:
            raise ValueError("request needs %d seeds, got %d" % (request.n, len(seeds)))
        out = []
        for seed in seeds[:request.n]:
            key = (request.task_id, int(seed), request.purpose)
            if key not in self._by_key:
                raise FixtureMissing(key)
            out.append(self._by_key[key])
        return out

    def model_info(self) -> Dict[str, Any]:
        return {"backend": "fixture", "synthetic": True, "repo": QWEN_REPO, "revision": QWEN_REVISION, "dtype": "n/a", "records": len(self._by_key)}


# ------------------------------------------------------------------ gated HF backend
class SnapshotMismatch(RuntimeError):
    """[EL-ARCH-1 B3] a mounted model file does not match its pinned sha256."""


class HFQwenBackend:
    """Real Qwen3-4B-Thinking-2507 generation (fp16, thinking mode, T=0.6 top_p=0.95 top_k=20). Runs only on Kaggle.
    Gate (checked in __init__ and again in generate): authorized is True, path /kaggle exists, env EVIDENCE_LADDER_ALLOW_INFERENCE == "1".
    Logs one JSONL line per generation (revision, template id, prompt sha256, token counts, seed). Model files are expected pre-mounted at model_dir; pass `verify_files`
    ({relative file: sha256}) to refuse a snapshot that differs from the pinned one (the Kaggle entry does). [EL-ARCH-1 B3 review] Fixes by reading the never-executed original:
    tokenize with add_special_tokens=False (the chat template already carries every special token, as in the T4 kernels), stop on <|im_end|> as well as eos, one seed per
    sequence with the seed set before it, generation split into small functions so the non-GPU logic is unit-testable with a stub torch (tests/phaseB_tests.py)."""
    INFERENCE_ENV = "EVIDENCE_LADDER_ALLOW_INFERENCE"

    def __init__(self, *, authorized: bool = False, model_dir: Optional[str] = None, log_path: Optional[str] = None, language: str = "python",
                 env: Optional[Mapping[str, str]] = None, path_exists=os.path.exists, verify_files: Optional[Mapping[str, str]] = None):
        self._authorized = authorized is True
        self._env = os.environ if env is None else env
        self._exists = path_exists
        self.model_dir, self.log_path, self.language = model_dir, log_path, language
        self.verify_files = dict(verify_files or {})
        self._gate()
        self._model = None
        self._tok = None
        self.snapshot_verified = False

    def _gate(self) -> None:
        if not self._authorized:
            raise NotAuthorized("HFQwenBackend needs authorized=True")
        if not self._exists("/kaggle"):
            raise NotAuthorized("HFQwenBackend runs only on Kaggle (/kaggle not found)")
        if self._env.get(self.INFERENCE_ENV) != "1":
            raise NotAuthorized("set %s=1 to allow inference" % self.INFERENCE_ENV)

    def model_info(self) -> Dict[str, Any]:
        return {"backend": "hf", "repo": QWEN_REPO, "revision": QWEN_REVISION, "dtype": "fp16", "thinking": True, "sampling": dict(SAMPLING),
                "templates": dict(TEMPLATE_HASHES), "snapshot_verified": self.snapshot_verified, "pinned_files": sorted(self.verify_files)}

    def _verify_snapshot(self) -> None:
        import hashlib
        for rel, want in sorted(self.verify_files.items()):
            h = hashlib.sha256()
            with open(os.path.join(self.model_dir or "", rel), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != want:
                raise SnapshotMismatch("%s: sha256 %s != pinned %s" % (rel, h.hexdigest(), want))
        self.snapshot_verified = bool(self.verify_files)

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if self.verify_files:
                self._verify_snapshot()
            src = self.model_dir or QWEN_REPO
            rev = None if self.model_dir else QWEN_REVISION
            self._tok = AutoTokenizer.from_pretrained(src, revision=rev)
            self._model = AutoModelForCausalLM.from_pretrained(src, revision=rev, torch_dtype=torch.float16).to("cuda").eval()
        return self._model, self._tok

    def _generate_one(self, torch, model, tok, request: GenerationRequest, seed: int, prompt_sha: str, lang: str) -> Generation:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        chat = tok.apply_chat_template([{"role": "user", "content": request.prompt}], tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
        n_prompt = int(enc["input_ids"].shape[1])
        stops = sorted(i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} if isinstance(i, int) and i >= 0)
        t0 = time.time()
        with torch.no_grad():
            ids = model.generate(**enc, do_sample=True, max_new_tokens=request.max_new_tokens, pad_token_id=tok.eos_token_id, eos_token_id=stops, **SAMPLING)
        new = ids[0, n_prompt:]
        return make_generation(tok.decode(new, skip_special_tokens=True), tokens_prompt=n_prompt, tokens_generated=int(new.shape[0]), seed=seed,
                               prompt_template_id=request.prompt_template_id, prompt_sha256=prompt_sha, language=lang, wall_seconds=time.time() - t0)

    def generate(self, request: GenerationRequest) -> List[Generation]:
        self._gate()
        if not request.thinking:
            raise ValueError("Qwen3-4B-Thinking-2507 is thinking-only")
        if len(request.seeds) < request.n:
            raise ValueError("request needs %d seeds, got %d" % (request.n, len(request.seeds)))
        import torch
        model, tok = self._load()
        lang = "json" if request.purpose == "tests" else self.language
        prompt_sha = sha256_text(request.prompt)
        out = []
        for seed in list(request.seeds)[:request.n]:
            g = self._generate_one(torch, model, tok, request, int(seed), prompt_sha, lang)
            self._log(request, g)
            out.append(g)
        return out

    def _log(self, request: GenerationRequest, g: Generation) -> None:
        if self.log_path:
            rec = {"task_id": request.task_id, "purpose": request.purpose, "model": g.model, "model_revision": g.model_revision,
                   "template": g.prompt_template_id, "prompt_sha256": g.prompt_sha256, "tokens_prompt": g.tokens_prompt,
                   "tokens_generated": g.tokens_generated, "seed": g.seed, "extract_status": g.extract_status}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(canonical_json(rec) + chr(10))


# ------------------------------------------------------------------ budget accounting
def ensure_budget(budget: Budget) -> None:
    if budget.exhausted():
        raise BudgetExhausted("generation budget exhausted before the call")


def charge_generations(budget: Budget, gens: Iterable[Generation]) -> None:
    """Charges prefill then generated tokens for each generation; BudgetExhausted propagates from Budget.charge."""
    for g in gens:
        budget.charge("qwen_prefill", g.tokens_prompt)
        budget.charge("qwen_gen", g.tokens_generated)


def generate_and_charge(backend, request: GenerationRequest, budget: Budget) -> List[Generation]:
    ensure_budget(budget)
    gens = backend.generate(request)
    charge_generations(budget, gens)
    return gens


class TokenBudget:
    """Minimal Budget implementation. Cap counts generated tokens (qwen_gen + glm_out), as in S6. charge() records the tokens first, then raises once the cap is crossed."""
    KINDS = ("qwen_gen", "qwen_prefill", "glm_in", "glm_out")
    COUNTERS = ("glm_calls", "scorer_calls", "tool_runs", "gpu_seconds", "wall_seconds", "api_errors")

    def __init__(self, problem_id: str, arm: str, generated_cap: int):
        self.problem_id, self.arm, self.cap = problem_id, arm, generated_cap
        self.used = {k: 0 for k in self.KINDS}
        self.extra = {k: 0 for k in self.COUNTERS}

    def generated(self) -> int:
        return self.used["qwen_gen"] + self.used["glm_out"]

    def charge(self, kind: str, tokens: int) -> None:
        if kind not in self.KINDS:
            raise ValueError("unknown budget kind %r" % kind)
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("tokens must be a non-negative int")
        self.used[kind] += tokens
        if self.generated() > self.cap:
            raise BudgetExhausted("generated-token cap %d crossed (%d)" % (self.cap, self.generated()))

    def add(self, counter: str, amount: float = 1) -> None:
        self.extra[counter] += amount

    def exhausted(self) -> bool:
        return self.generated() >= self.cap

    def record(self) -> Dict[str, Any]:
        return {"problem_id": self.problem_id, "arm": self.arm, "qwen_gen_tokens": self.used["qwen_gen"], "qwen_prefill_tokens": self.used["qwen_prefill"],
                "glm_tokens_in": self.used["glm_in"], "glm_tokens_out": self.used["glm_out"], "glm_calls": int(self.extra["glm_calls"]),
                "scorer_calls": int(self.extra["scorer_calls"]), "tool_runs": int(self.extra["tool_runs"]), "gpu_seconds": float(self.extra["gpu_seconds"]),
                "wall_seconds": float(self.extra["wall_seconds"]), "api_errors": int(self.extra["api_errors"])}
