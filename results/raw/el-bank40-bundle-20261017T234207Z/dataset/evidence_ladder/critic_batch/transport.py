"""Streamed HTTP transport for the critic kernel (the only network code of Step 4). transport(packet, key_label) -> (http_status | None, raw).

Bounded time: `read_timeout_s` is the socket timeout for connect and for every read (urlopen's timeout); `total_timeout_s` caps the whole stream
(checked per line; raises TimeoutError, which PcSender turns into a retryable failure). Streaming is required: non-streamed calls are cut at 300 s.
Key values live only in the `keys` dict handed in by the kernel and in the Authorization header; they are never logged, returned or put in an exception text.
`opener` is injectable so tests run with no network."""
from __future__ import annotations

import codecs
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from evidence_ladder.models._util import describe_error, redact_message

ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
# [EL-ARCH-1 A3.2] "â€”" is the UTF-8 byte sequence of an em dash (E2 80 94) read as Windows-1252. The wire bytes are decoded here as UTF-8 (incrementally, so a multi-byte character split across
# reads cannot break) and every receipt file is written as ASCII-escaped JSON, so the mojibake appears only when a READER (PowerShell 5.1 Get-Content, cmd, a text=True subprocess) uses the ANSI code page.
# The body therefore carries `encoding_check`; accounting/reconcile.py re-checks the saved files.
MOJIBAKE_RE = re.compile("â€|Ã[-¿]|Â[ -¿]")


def encoding_check(text: str) -> Dict[str, int]:
    return {"replacement_chars": text.count("�"), "mojibake_markers": len(MOJIBAKE_RE.findall(text))}


def read_stream(lines, *, clock: Callable[[], float] = time.monotonic, total_timeout_s: float = 1800.0):
    """Accumulate SSE lines into (non-stream-shaped body, completed). completed is False when the stream ended without finish_reason or [DONE]."""
    t0 = clock()
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    content, reasoning, usage, finish, done, err = [], [], {}, None, False, None
    for line in lines:
        if clock() - t0 > total_timeout_s:
            raise TimeoutError("total stream time exceeded")
        if isinstance(line, bytes):
            line = dec.decode(line)
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            done = True
            continue
        try:
            ev = json.loads(body)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("error"):
            err = ev["error"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if isinstance(d.get("content"), str):
                content.append(d["content"])
            r = d.get("reasoning_content") or d.get("reasoning")
            if isinstance(r, str):
                reasoning.append(r)
            finish = ch.get("finish_reason") or finish
        if isinstance(ev.get("usage"), dict):
            usage = ev["usage"]
    msg: Dict[str, Any] = {"role": "assistant", "content": "".join(content) if content else None}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    body = {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage, "encoding_check": encoding_check("".join(content) + "".join(reasoning))}
    if err:
        body["stream_error"] = err
    return body, (finish is not None or done)


def make_transport(keys: Mapping[str, str], *, endpoint: str = ENDPOINT, opener: Callable = urllib.request.urlopen, read_timeout_s: float = 120.0,
                   total_timeout_s: float = 1800.0, clock: Callable[[], float] = time.monotonic):
    def transport(packet: Dict[str, Any], key_label: str) -> Tuple[Optional[int], Any]:
        req = urllib.request.Request(endpoint, data=json.dumps(packet["payload"]).encode("utf-8"), method="POST",
                                     headers={"Authorization": "Bearer " + keys[key_label], "Content-Type": "application/json", "Accept": "text/event-stream"})
        phase, secrets = "connect", list(keys.values())
        try:
            with opener(req, timeout=read_timeout_s) as r:
                status = r.status
                phase = "stream"
                body, complete = read_stream(r, clock=clock, total_timeout_s=total_timeout_s)
        except urllib.error.HTTPError as e:
            try:
                text = redact_message(e.read().decode("utf-8", "replace"), secrets, 2000)
            except Exception:
                text = ""
            return e.code, {"error": text, "headers": {k: v for k, v in (e.headers.items() if e.headers else []) if k.lower() == "retry-after"}}
        except Exception as e:      # [EL-ARCH-1 A2] still raised (PcSender files it as a retryable transport failure); the phase travels with it and PcSender records describe_error()
            e.el_phase = phase
            raise
        if body.get("stream_error"):
            return 502, {"error": json.dumps(body["stream_error"], default=str)[:2000]}
        if not complete:
            return None, {"error": "truncated_stream"}
        return status, body
    return transport
