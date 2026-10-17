"""Tiny shared helpers: canonical JSON, hashing, ledger redaction."""
import hashlib
import json
import re


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


_SECRET_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]+|Bearer\s+[A-Za-z0-9._\-]+")


def redact(text: str) -> str:
    """Defence in depth: the ledger never stores credentials even if one slips into a response body."""
    return _SECRET_RE.sub("[REDACTED]", text)


# ---------------------------------------------------------------- [EL-ARCH-1 A2] transport error description (no credentials, ever)
_MSG_PATTERNS = (re.compile(r"(?i)(authorization|x-api-key|api[-_ ]?key|proxy-authorization|cookie|set-cookie)\s*[:=]\s*[^\r\n,;]*"),
                 re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]+"),
                 re.compile(r"nvapi-[A-Za-z0-9_\-]+|\bsk-[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,}|KGAT_[A-Za-z0-9]{6,}"),
                 re.compile(r"[A-Za-z0-9_\-+/=]{32,}"))                 # any long opaque token
_URL_QUERY = re.compile(r"(https?://[^\s\"'?#]*)\?[^\s\"']*")


def redact_message(text, secrets=(), limit: int = 200) -> str:
    """Short, credential-free text for an error message: header values, bearer tokens, key-shaped strings, long opaque tokens, URL query strings and any literal secret are removed."""
    t = str(text)
    for s in secrets:
        if s:
            t = t.replace(s, "[REDACTED]")
    t = _URL_QUERY.sub(r"\1?[QUERY-REDACTED]", t)
    for p in _MSG_PATTERNS:
        t = p.sub(lambda m: (m.group(1) + ": [REDACTED]") if m.re is _MSG_PATTERNS[0] else "[REDACTED]", t)
    return " ".join(t.split())[:limit]


def describe_error(exc: BaseException, phase: str = "", secrets=()) -> dict:
    """-> {"error_class": exception type name, "category": connect|read|tls|stream|truncated|timeout|http|other, "phase", "errno", "message"}. phase: where it happened ("connect"|"stream"|"")."""
    name, cat, en = type(exc).__name__, "other", getattr(exc, "errno", None)
    reason = getattr(exc, "reason", None)                  # urllib.error.URLError wraps the real exception
    inner = reason if isinstance(reason, BaseException) else exc
    phase = phase or getattr(exc, "el_phase", "")
    if any(c.__name__ in ("SSLError", "SSLCertVerificationError") for c in type(inner).__mro__):      # no ssl/socket import here: this package stays network-free (static test)
        cat = "tls"
    elif isinstance(inner, TimeoutError):          # socket.timeout is TimeoutError
        cat = "timeout"
    elif type(inner).__name__ in ("IncompleteRead", "RemoteDisconnected", "ChunkedEncodingError"):
        cat = "stream"
    elif isinstance(inner, (ConnectionError, BrokenPipeError, OSError)):
        cat = "connect" if phase == "connect" else "read"
        en = en if en is not None else getattr(inner, "errno", None)
    return {"error_class": name if inner is exc else "%s(%s)" % (name, type(inner).__name__), "category": cat, "phase": phase, "errno": en if isinstance(en, int) else None,
            "message": redact_message(exc, secrets)}


def error_info(status, raw, secrets=()):
    """Error description recorded next to every non-2xx / transport-failed ingest, or None for a 2xx. Old raws ({"error": "timeout"} ...) are classified too."""
    if status is not None and 200 <= status < 300:
        return None
    if status is None:
        r = raw if isinstance(raw, dict) else {}
        if isinstance(r.get("error_class"), dict):
            return dict(r["error_class"])
        e = r.get("error")
        if e == "truncated_stream":
            return {"error_class": "TruncatedStream", "category": "truncated", "phase": "stream", "errno": None, "message": "stream ended without finish_reason or [DONE]"}
        if e == "timeout":
            return {"error_class": "TimeoutError", "category": "timeout", "phase": "", "errno": None, "message": "timeout"}
        return {"error_class": str(r.get("type") or "unknown"), "category": "other", "phase": "", "errno": None, "message": redact_message(e or "", secrets)}
    text = raw.get("error") if isinstance(raw, dict) else raw
    return {"error_class": "HTTP%d" % status, "category": "http", "phase": "", "http_status_class": "%dxx" % (status // 100), "errno": None, "message": redact_message(text or "", secrets)}
