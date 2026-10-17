"""GLM-5.3-flash critic: blinded request validation, chat-completions payload builder, response parser. No network, no keys.

The reasoning-effort request parameter is PENDING feasibility check FC-2. GlmConfig.reasoning_params stays None until FC-2 records the real
parameter; building a payload with it unset raises ConfigError (never guess a name, never silently downgrade)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from evidence_ladder.common.contract_import import C
from evidence_ladder.models._util import canonical_json

GLM_MODEL = "z-ai/glm-5.3-flash"
PROMPT_ID = "glm_critic_v1"
STAGE1_KINDS = ("stdin_input", "function_call")


class ConfigError(Exception):
    pass


class BlindingError(Exception):
    pass


@dataclass(frozen=True)
class GlmConfig:
    # ponytail: default endpoint is NVIDIA's public chat-completions URL; the PC transport owns the real URL and keys.
    endpoint: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    max_tokens: int = 32768
    stream: bool = True
    reasoning_params: Optional[Mapping[str, Any]] = None   # PENDING (FC-2): extra top-level payload fields that select maximum reasoning
    temperature: Optional[float] = None
    allowed_kinds: Tuple[str, ...] = STAGE1_KINDS             # experiment kinds the prompt offers AND the parser accepts (step E: experiments.SUPPORTED_KINDS)


# ------------------------------------------------------------------ blinding
TOP_KEYS = {"request_id", "model", "decision_id", "candidate_ids", "requirement_quotes", "candidate_code", "observed_outputs", "question"}
ALLOWED_OUTPUT_KEYS = {"input", "observed", "status", "failure_class", "detail", "stderr_tail", "public_expected", "exit_code", "duration_s", "check_id"}
FORBIDDEN_KEY_RE = re.compile(r"laya|scor|probe|choice|pick|rank|reason|think|hidden|label|truth|oracle|secret|expected", re.I)
FORBIDDEN_VALUE_RES = (re.compile(r"</?think>", re.I), re.compile(r"\blaya\b", re.I), re.compile(r"hidden[ _-]?tests?", re.I))


def _scan_value(v: Any, where: str) -> None:
    if isinstance(v, str):
        for rx in FORBIDDEN_VALUE_RES:
            if rx.search(v):
                raise BlindingError("forbidden content (%s) in %s" % (rx.pattern, where))
    elif isinstance(v, dict):
        for k, x in v.items():
            _scan_value(k, where + " key")
            _scan_value(x, where + "." + str(k))
    elif isinstance(v, list):
        for x in v:
            _scan_value(x, where)


def _check_output_keys(node: Any, where: str) -> None:
    if isinstance(node, dict):
        for k, x in node.items():
            if FORBIDDEN_KEY_RE.search(k) and k != "public_expected":
                raise BlindingError("forbidden key %r in %s" % (k, where))
            if k not in ALLOWED_OUTPUT_KEYS:
                raise BlindingError("key %r in %s is not on the observed_outputs allow-list" % (k, where))
            _check_output_keys(x, where + "." + k)
    elif isinstance(node, list):
        for x in node:
            _check_output_keys(x, where)


def check_blinding(req: Mapping[str, Any]) -> None:
    """Allow-list check. Raises BlindingError. observed_outputs shape: {candidate_id: [ {input, observed, status, ...}, ... ]}."""
    for k in req:
        if k not in TOP_KEYS:
            why = "forbidden" if FORBIDDEN_KEY_RE.search(k) else "not on the allow-list"
            raise BlindingError("top-level key %r is %s" % (k, why))
    ids = req.get("candidate_ids", [])
    if set(req.get("candidate_code", {})) != set(ids):
        raise BlindingError("candidate_code keys must equal candidate_ids")
    oo = req.get("observed_outputs", {})
    if not isinstance(oo, dict):
        raise BlindingError("observed_outputs must be an object keyed by candidate id")
    for cid, node in oo.items():
        if cid not in ids:
            raise BlindingError("observed_outputs key %r is not a candidate id" % cid)
        _check_output_keys(node, "observed_outputs." + cid)
    _scan_value({k: v for k, v in req.items()}, "request")


def schema_error(name: str, obj: Any) -> Optional[str]:
    """None when valid, else the schema message. (jsonschema must be installed; ImportError is not swallowed.)"""
    from jsonschema import ValidationError
    try:
        C.validate(name, obj)
        return None
    except ValidationError as e:
        return e.message


def validate_request(req: Mapping[str, Any]) -> Dict[str, Any]:
    """Blinding first (clear errors), then critic_request schema. Returns a plain-dict copy."""
    check_blinding(req)
    msg = schema_error("critic_request", dict(req))
    if msg:
        raise ValueError("critic_request schema: " + msg)
    return json.loads(canonical_json(req))


# ------------------------------------------------------------------ payload
SYSTEM_PROMPT = ("You are a verification critic. You see requirements quoted from a task, candidate programs and their concrete outputs. "
                 "Propose ONE experiment (a stdin input or a function call) on which the candidates are expected to behave differently, or that challenges "
                 "a quoted requirement. Reply with a single JSON object and nothing else.")
RESERVED = {"model", "messages", "stream", "max_tokens", "temperature"}


EXEC_SYSTEM_PROMPT = ("You are a verification critic. You see requirements quoted from a task, candidate programs and their concrete outputs. "
                      "Propose ONE experiment that the sandbox can run on the candidate and that challenges a quoted requirement or exposes a defect. "
                      "Reply with a single JSON object and nothing else.")


def build_messages(req: Mapping[str, Any], kinds: Tuple[str, ...] = STAGE1_KINDS) -> List[Dict[str, str]]:
    parts = ["Question type: " + req["question"], "", "Requirements (verbatim quotes):"]
    parts += ["%d. %s" % (i + 1, q) for i, q in enumerate(req["requirement_quotes"])]
    for cid in req["candidate_ids"]:
        parts += ["", "### candidate " + cid, "```", req["candidate_code"][cid], "```"]
    parts += ["", "Observed outputs (JSON):", json.dumps(req["observed_outputs"], sort_keys=True, indent=2, ensure_ascii=False), ""]
    if tuple(kinds) != STAGE1_KINDS:      # step E: the prompt offers exactly the kinds the controller executor supports (evidence_ladder.models.experiments)
        from evidence_ladder.models import experiments as EXP
        parts += ["Allowed experiment kinds (payload is a string):"] + ["- %s: %s" % (k, EXP.PAYLOAD_HELP[k]) for k in kinds] + [
            'Reply with ONLY: {"experiment": {"kind": %s, "payload": "<string>", "expected_by_candidate": {"<candidate_id>": "<expected output>"}}, "implications": "<what each outcome would show>"}' % " or ".join('"%s"' % k for k in kinds)]
        return [{"role": "system", "content": EXEC_SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}]
    parts += [
              'Reply with ONLY: {"experiment": {"kind": "stdin_input" or "function_call", "payload": "<string>", '
              '"expected_by_candidate": {"<candidate_id>": "<expected output>"}}, "implications": "<what each outcome would show>"}']
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}]


def build_payload(req: Mapping[str, Any], cfg: GlmConfig) -> Dict[str, Any]:
    if not cfg.reasoning_params:
        raise ConfigError("GlmConfig.reasoning_params is unset: the maximum-reasoning request parameter is pending feasibility check FC-2")
    clash = RESERVED & set(cfg.reasoning_params)
    if clash:
        raise ConfigError("reasoning_params may not override %s" % sorted(clash))
    r = validate_request(req)
    payload: Dict[str, Any] = {"model": GLM_MODEL, "messages": build_messages(r, cfg.allowed_kinds), "stream": cfg.stream, "max_tokens": cfg.max_tokens}
    if cfg.temperature is not None:
        payload["temperature"] = cfg.temperature
    payload.update(dict(cfg.reasoning_params))
    return payload


# ------------------------------------------------------------------ response parsing
@dataclass
class ParsedCritic:
    status: str                       # "ok" | "malformed"
    reason: str = ""                  # malformed: empty_content | not_json | schema_invalid | unsupported_kind | unreadable_body
    response: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    reasoning: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    detail: str = ""


def _chunks(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("data:") or "\ndata:" in s:
            out = []
            for line in s.splitlines():
                line = line.strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    try:
                        out.append(json.loads(line[5:].strip()))
                    except ValueError:
                        pass
            return [c for c in out if isinstance(c, dict)]
        try:
            return _chunks(json.loads(s))
        except ValueError:
            return None
    if isinstance(raw, dict):
        if "choices" not in raw and "body" in raw:
            return _chunks(raw["body"])
        return [raw]
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]
    return None


def assemble(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalise a non-streamed body, a list of stream chunks, or SSE text to {content|None, reasoning, usage}. None if unreadable."""
    chunks = _chunks(raw)
    if chunks is None:
        return None
    content: List[str] = []
    reasoning: List[str] = []
    usage: Dict[str, Any] = {}
    for ch in chunks:
        if isinstance(ch.get("usage"), dict):
            usage = ch["usage"]
        for choice in ch.get("choices") or []:
            msg = choice.get("message") or choice.get("delta") or {}
            if isinstance(msg.get("content"), str):
                content.append(msg["content"])
            for rk in ("reasoning_content", "reasoning"):
                if isinstance(msg.get(rk), str):
                    reasoning.append(msg[rk])
    return {"content": "".join(content) if content else None, "reasoning": "".join(reasoning), "usage": usage}


def find_json_object(text: str) -> Optional[Dict[str, Any]]:
    """First JSON object in text that has an 'experiment' key (else the first object). Tolerates code fences and prose around it."""
    dec = json.JSONDecoder()
    first = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            if "experiment" in obj:
                return obj
            first = first or obj
    return first


def parse_critic_output(raw: Any, allowed_kinds: Tuple[str, ...] = STAGE1_KINDS) -> ParsedCritic:
    a = assemble(raw)
    if a is None:
        return ParsedCritic("malformed", "unreadable_body")
    u = a["usage"]
    tin = u.get("prompt_tokens") if isinstance(u.get("prompt_tokens"), int) else 0
    tout = u.get("completion_tokens") if isinstance(u.get("completion_tokens"), int) else 0
    base = dict(content=a["content"], reasoning=a["reasoning"], tokens_in=tin, tokens_out=tout)
    if a["content"] is None or not a["content"].strip():
        return ParsedCritic("malformed", "empty_content", **base)   # reasoning text is never parsed for the experiment
    obj = find_json_object(a["content"])
    if obj is None:
        return ParsedCritic("malformed", "not_json", **base)
    msg = schema_error("critic_response", obj)
    if msg:
        return ParsedCritic("malformed", "schema_invalid", detail=msg, **base)
    if obj["experiment"]["kind"] not in allowed_kinds:
        return ParsedCritic("malformed", "unsupported_kind", detail=obj["experiment"]["kind"], **base)
    return ParsedCritic("ok", response=obj, **base)
