"""[P4] Host-controlled trust for task contracts. The ONLY place a contract becomes authoritative.

Registry file (written by the host, versioned, committed BEFORE candidates are generated):
  {"registry_version": "contract_registry_v1", "entries": {"<task_id>": {"path": "configs/contracts/...json", "sha256": "...", "authority": "authoritative"|"diagnostic",
                                                                         "basis": "...", "frozen_utc": "..."}}}
Rules: the contract file must match the pinned sha256 (else ContractError, fail closed); a "model:*" source can never be authoritative, whatever the registry says;
derived contracts come only from an ALLOWLISTED deterministic parser run here on the task text (never from a stored document)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from . import contract_v1 as CV

REGISTRY_VERSION = "contract_registry_v1"
PARSER_ALLOWLIST = {CV.PARSER_ID: CV.derive_from_text}


def load_registry(path: Path) -> Dict[str, Any]:
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    if reg.get("registry_version") != REGISTRY_VERSION:
        raise CV.ContractError("not a %s file" % REGISTRY_VERSION)
    return reg


def trusted_contract(task_id: str, registry: Optional[Dict[str, Any]], root: Path, task_prompt: Optional[str] = None,
                     parsers: Optional[list] = None) -> Optional[CV.Contract]:
    """-> the contract for a task with HOST-set authority, or None (no contract: uncertainty kept, no verdict).
    Order: a pinned registry entry; else an allowlisted parser on the task text (authoritative only for what it parses); else None."""
    e = (registry or {}).get("entries", {}).get(task_id)
    if e is not None:
        p = Path(root) / e["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != e["sha256"]:
            raise CV.ContractError("contract %s differs from its pinned sha256 in the registry" % e["path"])
        c = CV.load(json.loads(p.read_text(encoding="utf-8")), Path(root))
        model = c.source.split(":")[0] == "model"
        c.authority = "authoritative" if (e.get("authority") == "authoritative" and not model) else "diagnostic"
        return c
    for pid in (parsers or []):
        if pid not in PARSER_ALLOWLIST:
            raise CV.ContractError("parser %r is not allowlisted" % pid)
        doc = PARSER_ALLOWLIST[pid](task_id, task_prompt or "")
        if doc:
            c = CV.load(doc)
            c.authority = "authoritative"
            return c
    return None
