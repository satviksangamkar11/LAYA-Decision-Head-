"""Shared helpers for the orchestrator. Pure stdlib, no model, no network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THRESHOLDS = REPO / "configs" / "thresholds_evidence_ladder_v1.toml"
DEV_LABEL = "DEVELOPMENT EVIDENCE (not a held-out claim)"
SYNTHETIC_LABEL = "SYNTHETIC DRY RUN: scripted fakes; nothing here is accuracy and nothing here is evidence"


class ConfigError(Exception):
    """A required value is pending/unset or malformed."""


class PoolingError(Exception):
    """Python, Java and Fable results must never be combined."""


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(o) -> str:
    return hashlib.sha256((o if isinstance(o, str) else canon(o)).encode("utf-8")).hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_bytes_lf_tolerant(b: bytes):
    """Both hashes of a text file: raw and with CRLF normalized to LF (git autocrlf checkouts). Tamper detection is unchanged."""
    return {sha_bytes(b), sha_bytes(b.replace(b"\r\n", b"\n"))}


def write_json(path: Path, obj) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return sha_bytes(text.encode("utf-8"))


def write_jsonl(path: Path, rows) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canon(r) + "\n" for r in rows)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return sha_bytes(text.encode("utf-8"))


def load_toml(path: Path = THRESHOLDS) -> dict:
    import tomllib
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))
