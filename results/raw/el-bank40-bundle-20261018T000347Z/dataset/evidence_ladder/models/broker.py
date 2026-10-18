"""StagedCriticBroker (decision D7 option a): packets are built and exported for the PC to send; responses are ingested later. Idempotent by packet_id.
Pure bookkeeping: no network, no key handling. State lives in an append-only JSONL ledger (replayed on start), which never stores credentials."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from evidence_ladder.common.interfaces import Budget, sha256_text
from evidence_ladder.models import glm
from evidence_ladder.models._util import canonical_json, error_info, redact

# Call-logic limits (user, FIXED NVIDIA CALL LOGIC). Configuration constants only; enforced by pc_sender.KeyScheduler.
LIMITS = {"max_in_flight_per_key": 10, "max_sends_per_key_per_minute": 39, "rate_limit_cooldown_s": 60}  # user 2026-10-16: 10 in flight per key (2026-10-11 measurement: at 10, about half got 429; 39/min and 60 s cooldown unchanged)
TERMINAL = ("ok", "malformed", "unavailable")
QUOTA_RE = re.compile(r"payment|billing|insufficient[ _-]*(credit|quota|balance|fund)|out of credits|quota[ _-]*(exceeded|exhausted)|exceeded.{0,40}quota|subscription", re.I)


class CriticStopped(Exception):
    """stop_all_critic_calls is set (payment/quota error): no further critic work, ever, in this ledger."""


def packet_id_of(request: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(request))


def retry_after_s(raw: Any) -> Optional[float]:
    if isinstance(raw, dict) and isinstance(raw.get("headers"), dict):
        for k, v in raw["headers"].items():
            if str(k).lower() == "retry-after":
                try:
                    x = float(str(v).strip())
                    return x if x >= 0 else None
                except ValueError:
                    return None
    return None


class StagedCriticBroker:
    """max_retries = re-sends allowed after the first send; the (max_retries+1)-th failure makes the packet "unavailable".
    Backoff is COMPUTED (stored as not_before), never slept. 429 honours Retry-After (default 60 s, capped at retry_after_cap_s)."""

    def __init__(self, ledger_path, *, config: Optional[glm.GlmConfig] = None, max_retries: int = 3, backoff_base_s: float = 2.0, backoff_cap_s: float = 300.0,
                 default_retry_after_s: float = 60.0, retry_after_cap_s: float = 3600.0, clock: Callable[[], float] = time.time, budget: Optional[Budget] = None):
        self.ledger_path = Path(ledger_path)
        self.cfg = config or glm.GlmConfig()
        self.max_retries, self.backoff_base_s, self.backoff_cap_s = max_retries, backoff_base_s, backoff_cap_s
        self.default_retry_after_s, self.retry_after_cap_s = default_retry_after_s, retry_after_cap_s
        self.clock, self.budget = clock, budget
        self.packets: Dict[str, Dict[str, Any]] = {}
        self.stop_all_critic_calls = False
        self.counters = {"ingested": 0, "duplicates": 0, "tokens_in": 0, "tokens_out": 0, "api_errors": 0, "malformed": 0, "sends": 0}
        self._seq = 0
        self._lock = threading.Lock()
        if self.ledger_path.exists():
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ev = json.loads(line)
                        self._seq = max(self._seq, ev["seq"])
                        self._apply(ev)
        if self._seq == 0:
            self._emit({"event": "config", "limits": LIMITS, "max_retries": max_retries, "model": glm.GLM_MODEL})

    # ---------------------------------------------------------------- ledger
    def _emit(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            ev = dict(ev, seq=self._seq, ts=self.clock())
            line = redact(canonical_json(ev))
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._apply(json.loads(line))

    def _apply(self, ev: Dict[str, Any]) -> None:
        kind, pid = ev["event"], ev.get("packet_id")
        if kind == "packet_built":
            self.packets[pid] = {"request": ev["request"], "state": "pending", "attempts": 0, "not_before": 0.0, "response": None, "reason": "", "created": ev["ts"]}
        elif kind == "exported":
            self.packets[pid]["state"] = "exported"
        elif kind == "requeued":     # exported but never answered (crash while in flight): back to pending, bounded by the caller's policy
            if self.packets[pid]["state"] == "exported":
                self.packets[pid]["state"] = "pending"
        elif kind == "send":
            self.counters["sends"] += 1
        elif kind == "duplicate_ingest":
            self.counters["duplicates"] += 1
        elif kind == "ingest":
            p = self.packets[pid]
            p.update(state=ev["state"], attempts=ev["attempts"], not_before=ev["not_before"], response=ev["response"], reason=ev["reason"])
            c = self.counters
            c["ingested"] += 1
            c["tokens_in"] += ev["tokens_in"]
            c["tokens_out"] += ev["tokens_out"]
            c["api_errors"] += 1 if ev["api_error"] else 0
            c["malformed"] += 1 if ev["state"] == "malformed" else 0
            if ev["stop_all"]:
                self.stop_all_critic_calls = True

    def _check_open(self) -> None:
        if self.stop_all_critic_calls:
            raise CriticStopped("stop_all_critic_calls is set (payment/quota error); no further critic calls")

    # ---------------------------------------------------------------- CriticBroker interface
    def build_packet(self, critic_request: Dict[str, Any]) -> Dict[str, Any]:
        self._check_open()
        req = glm.validate_request(critic_request)
        pid = packet_id_of(req)
        new = pid not in self.packets
        if new:
            self._emit({"event": "packet_built", "packet_id": pid, "request": req})
        p = self.packets[pid]
        return {"packet_id": pid, "request": p["request"], "created": p["created"], "new": new}

    def _ready(self, include_exported: bool) -> List[str]:
        now = self.clock()
        states = ("pending", "retry") + (("exported",) if include_exported else ())
        return [pid for pid, p in self.packets.items() if p["state"] in states and p["not_before"] <= now]

    def export_pending(self, limit: Optional[int] = None, include_exported: bool = False) -> List[Dict[str, Any]]:
        """What the PC must send now. Raises ConfigError (no state change) while the FC-2 reasoning parameter is unset."""
        self._check_open()
        ids = self._ready(include_exported)
        if limit is not None:
            ids = ids[:max(0, limit)]
        out = [{"packet_id": pid, "endpoint": self.cfg.endpoint, "payload": glm.build_payload(self.packets[pid]["request"], self.cfg),
                "attempt": self.packets[pid]["attempts"] + 1} for pid in ids]
        for pid in ids:
            self._emit({"event": "exported", "packet_id": pid})
        return out

    def completed(self, packet_id: str) -> Optional[Dict[str, Any]]:
        """Validated critic_response of a packet whose ingest state is "ok"; None for unknown, pending, retry, malformed or unavailable packets.
        State comes from the replayed ledger, so a new process reads what the PC ingested. Works after stop_all_critic_calls too (read-only)."""
        p = self.packets.get(packet_id)
        return p["response"] if p is not None and p["state"] == "ok" else None

    def terminal_state(self, packet_id: str) -> Optional[str]:
        """Read-only: "malformed" or "unavailable" when the packet ended in that state, else None (pending, retry, ok or unknown)."""
        p = self.packets.get(packet_id)
        return p["state"] if p is not None and p["state"] in ("malformed", "unavailable") else None

    def pending_summary(self) -> Dict[str, Any]:
        now = self.clock()
        return {"ready": len(self._ready(False)),
                "waiting_until": sorted(p["not_before"] for p in self.packets.values() if p["state"] == "retry" and p["not_before"] > now),
                "exported_unresolved": sum(1 for p in self.packets.values() if p["state"] == "exported")}

    def requeue_exported(self, packet_id: str) -> bool:
        """At-least-once recovery: an exported packet with no answer becomes sendable again (ledger event "requeued"). False if it is not exported."""
        if self.packets[packet_id]["state"] != "exported":
            return False
        self._emit({"event": "requeued", "packet_id": packet_id})
        return True

    def log_send(self, packet_id: str, key_label: str) -> None:
        """Audit line for every send; key_label is an opaque label (e.g. "k3"), never a credential."""
        self._emit({"event": "send", "packet_id": packet_id, "key_label": key_label})

    def _classify(self, status: Optional[int], raw: Any):
        """-> (kind, reason): kind in ok|retry|fatal|stop"""
        text = json.dumps(raw, default=str).lower() if raw is not None else ""
        if status is None:
            return "retry", "transport_failure"
        if status == 402:
            return "stop", "payment_required"
        if status >= 400 and QUOTA_RE.search(text):
            return "stop", "quota_or_payment_text"
        if 200 <= status < 300:
            return "ok", ""
        if status == 429:
            return "retry", "rate_limited"
        if status == 408 or 500 <= status < 600:
            return "retry", "server_error" if status != 408 else "timeout"
        return "fatal", "client_error_%d" % status

    def ingest(self, packet_id: str, http_status: Optional[int], raw_response: Any) -> Dict[str, Any]:
        p = self.packets[packet_id]          # KeyError for unknown ids
        if p["state"] in TERMINAL:
            self._emit({"event": "duplicate_ingest", "packet_id": packet_id, "http_status": http_status})
            return {"state": "duplicate", "response": p["response"], "reason": "already_" + p["state"]}
        kind, reason = self._classify(http_status, raw_response)
        parsed = glm.parse_critic_output(raw_response, tuple(self.cfg.allowed_kinds)) if kind == "ok" else None
        state, response, stop = None, None, False
        attempts, not_before, backoff, ra = p["attempts"], 0.0, None, retry_after_s(raw_response)
        tin = tout = 0
        if parsed is not None:
            tin, tout = parsed.tokens_in, parsed.tokens_out
            state, response, reason = ("ok", parsed.response, "") if parsed.status == "ok" else ("malformed", None, parsed.reason)
        elif kind == "stop":
            state, stop = "unavailable", True
        elif kind == "fatal":
            state = "unavailable"
        else:
            attempts += 1
            if attempts > self.max_retries:
                state, reason = "unavailable", "retries_exhausted:" + reason
            else:
                state = "retry"
                if reason == "rate_limited":
                    backoff = min(ra if ra is not None else self.default_retry_after_s, self.retry_after_cap_s)
                else:
                    backoff = min(self.backoff_cap_s, self.backoff_base_s * 2 ** (attempts - 1))
                not_before = self.clock() + backoff
        raw_text = redact(canonical_json(raw_response) if not isinstance(raw_response, str) else raw_response)
        api_error = not (200 <= (http_status or 0) < 300)
        err = error_info(http_status, raw_response)           # [EL-ARCH-1 A2] error_class/category/errno/redacted message; reason keeps "transport_failure" etc.
        self._emit({"event": "ingest", "packet_id": packet_id, "http_status": http_status, "state": state, "reason": reason, "attempts": attempts,
                    "not_before": not_before, "backoff_s": backoff, "retry_after_s": ra, "response": response, "tokens_in": tin, "tokens_out": tout,
                    "api_error": api_error, "error": err, "stop_all": stop, "raw_sha256": sha256_text(raw_text), "raw_head": raw_text[:50000]})
        if self.budget is not None and (tin or tout):
            self.budget.charge("glm_in", tin)
            self.budget.charge("glm_out", tout)
        return {"state": state, "response": response, "reason": reason, "backoff_s": backoff, "retry_after_s": ra, "not_before": not_before or None,
                "stop_all_critic_calls": self.stop_all_critic_calls, "tokens_in": tin, "tokens_out": tout, "error": err,
                "reasoning_text": parsed.reasoning if parsed else ""}
