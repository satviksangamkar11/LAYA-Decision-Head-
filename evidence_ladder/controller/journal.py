"""Append-only, hash-chained evidence journal (invariant I7: immutable journal). Pure standard library.
Every entry is canonical JSON; entry N carries the hash of entry N-1. journal_hash() is the reproducibility receipt: identical inputs and seeds
give the same hash, so nothing nondeterministic (wall time, durations) may ever be written into an entry.
Evidence records are admitted only through add_evidence(), which uses C.Registry.accept (registered check + schema)."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from evidence_ladder.common.contract_import import C
from .components import MissingComponentError

GENESIS = C.sha256("evidence-journal-v1")


def canon(o: Any, default=None) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default)


class JournalError(Exception):
    pass


class JournalRegistryMissing(MissingComponentError, JournalError):
    """[EL-ARCH-1 A5] a journal without a registry cannot admit evidence: a missing component, and (for old callers) still a JournalError."""


class EvidenceJournal:
    def __init__(self, registry=None):
        self._registry = registry
        self._lines: List[str] = []
        self._head = GENESIS
        self._evidence: List[Dict[str, Any]] = []
        self._ev_ids = set()

    # ---- writing (the only mutation is append)
    def append(self, kind: str, data: Any) -> str:
        entry = {"seq": len(self._lines), "kind": kind, "data": data, "prev": self._head}
        line = canon(entry)          # raises TypeError on non-JSON data: nothing nondeterministic slips in silently
        self._head = C.sha256(line)
        self._lines.append(line)
        return self._head

    def add_evidence(self, ev: Dict[str, Any], outcome: Optional[Dict[str, Any]] = None) -> str:
        """Admit one evidence record. Every record is bound to artifact_hash, environment_revision, requirement_version, check_version and candidate_id
        (all are required by the schema). `outcome` is the trimmed, deterministic execution outcome it was derived from (kept for replay and for I4)."""
        if self._registry is None:
            raise JournalRegistryMissing("registry", "evidence cannot be admitted")
        if ev["evidence_id"] in self._ev_ids:
            raise JournalError("duplicate evidence id " + ev["evidence_id"])
        self._registry.accept(ev)    # KeyError for unregistered checks, ValidationError for schema violations
        self._ev_ids.add(ev["evidence_id"])
        self._evidence.append(copy.deepcopy(ev))
        return self.append("evidence", {"evidence": ev, "outcome": outcome or {}})

    # ---- reading (copies only)
    def entries(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        out = [json.loads(l) for l in self._lines]
        return [e for e in out if kind is None or e["kind"] == kind]

    def evidence(self, candidate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return copy.deepcopy([e for e in self._evidence if candidate_id is None or e["candidate_id"] == candidate_id])

    def journal_hash(self) -> str:
        return self._head

    def __len__(self) -> int:
        return len(self._lines)

    def verify_chain(self) -> bool:
        h = GENESIS
        for line in self._lines:
            if json.loads(line)["prev"] != h:
                return False
            h = C.sha256(line)
        return h == self._head
