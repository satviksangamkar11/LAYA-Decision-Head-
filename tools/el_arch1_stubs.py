"""[EL-ARCH-1 D1] Stand-ins for Qwen / LAYA / GLM used ONLY by unit tests, the --offline-rehearsal mode and the fault-injection variants. Every object here carries `stub = True`
(and records carry `"stub": true`), so a receipt can never be mistaken for a real one: the final acceptance mode refuses any stub (tools/el_arch1_acceptance.py no_stubs()) and the
conformance generator (tools/el_arch1_conformance.py) can never grade a component that touched one as REAL PASS. Pure standard library + the controller's own modules."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evidence_ladder.controller import generation as GEN
from evidence_ladder.models.broker import packet_id_of

STUB_NOTE = "STUB: stands in for a real model; never evidence for Qwen, LAYA or GLM"


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


class StubScorer:
    """Deterministic fake LAYA / probe / random head: a score is a hash of (kind, option text). Same call interface as evidence_ladder.models.laya scorers."""
    stub = True

    def __init__(self, kind: str = "laya", salt: str = ""):
        self.kind, self.salt = kind, salt
        self.scorer_id = "stub:" + kind
        self.scorer_def = {"weights_hash": "stub:" + kind, "status": "stub"}
        self._backend = None
        self.calls: List[int] = []

    def score(self, decision_type: str, pool: Any, exploratory: bool = False) -> List[float]:
        opts = pool["options"] if isinstance(pool, Mapping) else list(pool)
        self.calls.append(len(opts))
        return [round((_h(self.kind, self.salt, str(o)) % 1000) / 1000.0, 3) for o in opts]


class StubCritic:
    """Fake StagedCriticBroker for the controller's critic stage: serves one canned critic response for every packet (no ledger file, so accounting reports none)."""
    stub = True

    def __init__(self, response: Optional[Dict[str, Any]]):
        self.response, self.packets = response, {}

    def build_packet(self, req: Dict[str, Any]) -> Dict[str, Any]:
        pid = packet_id_of(req)
        self.packets.setdefault(pid, {"request": req, "state": "ok" if self.response is not None else "unavailable", "reason": "stub"})
        return {"packet_id": pid, "request": req, "created": 0.0, "new": True}

    def completed(self, pid: str):
        return self.response

    def terminal_state(self, pid: str):
        return None if self.response is not None else "unavailable"


def option(name: str, **args: str) -> str:
    """The option text exactly as the controller and LAYA see it (render_call)."""
    return GEN.render_call(name, args)


def raw_generation_text(options: Sequence[str]) -> str:
    """What a (stub) thinking model writes: a closed think block and ONE JSON array, the shape controller.generation.parse_actions accepts."""
    return "<think>\nstub reasoning\n</think>\n" + json.dumps([json.loads(o) for o in options], ensure_ascii=False)
