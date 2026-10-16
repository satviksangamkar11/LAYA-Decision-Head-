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
