"""[EL-ARCH-1 A1] GLM call / retry / token accounting computed from the REAL ledgers (never a hard-coded 0).
Sources: the broker ledger (send / ingest events) and, when present, the BatchRunner progress ledger (the raw answers). Per packet:
  calls            = sends (every request that left the machine, retries included)
  ok / retries     = ingests that ended ok / that were filed as "retry" (the next send follows)
  failed_attempts  = ingests that were not 2xx (transport failure, 429, 5xx, ...)
  tokens_in/out    = the provider's usage when it sent one; else ESTIMATED from text length (1 token ~ 4 chars of the request messages / of the streamed content+reasoning),
                     flagged estimated=true. Attempts that got no answer count 0 tokens (nothing is invented for them).
Pure standard library plus glm.assemble; reads files only."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evidence_ladder.models import glm
from evidence_ladder.models._util import error_info

CHARS_PER_TOKEN = 4


def _read(path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _toks(chars: int) -> int:
    return int(math.ceil(chars / CHARS_PER_TOKEN)) if chars > 0 else 0


def account(broker_ledger, progress_ledger=None, packet_ids: Optional[Iterable[str]] = None, allowed_kinds=None) -> Dict[str, Any]:
    from evidence_ladder.models import experiments as EXP
    kinds = tuple(allowed_kinds or EXP.SUPPORTED_KINDS)
    bl, pl = _read(broker_ledger), _read(progress_ledger) if progress_ledger else []
    want = set(packet_ids) if packet_ids is not None else None
    requests = {e["packet_id"]: e["request"] for e in bl if e["event"] == "packet_built"}
    results: Dict[str, List[Dict[str, Any]]] = {}
    for e in pl:
        if e["event"] == "result":
            results.setdefault(e["packet_id"], []).append(e)
    per: Dict[str, Dict[str, Any]] = {}
    for pid in requests:
        if want is not None and pid not in want:
            continue
        per[pid] = dict(calls=0, ok=0, retries=0, failed_attempts=0, tokens_in=0, tokens_out=0, estimated=False, attempts=[])
    in_est = {}
    for pid in per:
        try:
            in_est[pid] = _toks(sum(len(m["content"]) for m in glm.build_messages(requests[pid], kinds)))
        except Exception:      # noqa: BLE001  an unreadable request is not guessed at
            in_est[pid] = 0
    ing_n: Dict[str, int] = {}
    for e in bl:
        pid = e.get("packet_id")
        if pid not in per:
            continue
        P = per[pid]
        if e["event"] == "send":
            P["calls"] += 1
        elif e["event"] == "ingest":
            n = ing_n[pid] = ing_n.get(pid, 0) + 1
            r = results.get(pid, [])
            err = e.get("error") or (error_info(e["http_status"], r[n - 1].get("raw")) if e["api_error"] and len(r) >= n else None)      # ledgers written before A2 carry no class: derive it from the raw answer
            a = {"attempt": n, "http_status": e["http_status"], "state": e["state"], "reason": e["reason"], "error": err, "tokens_in": 0, "tokens_out": 0, "estimated": False}
            if e["state"] == "retry":
                P["retries"] += 1
            if e["state"] == "ok":
                P["ok"] += 1
            if e["api_error"]:
                P["failed_attempts"] += 1
            elif e["tokens_in"] or e["tokens_out"]:
                a.update(tokens_in=e["tokens_in"], tokens_out=e["tokens_out"])
            else:                                            # answered, but the provider reported no usage: estimate from text, flag it
                raw = r[n - 1]["raw"] if len(r) >= n and r[n - 1].get("raw") is not None else None
                asm = glm.assemble(raw) if raw is not None else None
                chars = (len((asm or {}).get("content") or "") + len((asm or {}).get("reasoning") or "")) if asm else len(e.get("raw_head") or "")
                usage = (asm or {}).get("usage") or {}
                if usage.get("prompt_tokens") or usage.get("completion_tokens"):
                    a.update(tokens_in=int(usage.get("prompt_tokens") or 0), tokens_out=int(usage.get("completion_tokens") or 0))
                else:
                    a.update(tokens_in=in_est[pid], tokens_out=_toks(chars), estimated=True)
            P["attempts"].append(a)
            P["tokens_in"] += a["tokens_in"]
            P["tokens_out"] += a["tokens_out"]
            P["estimated"] = P["estimated"] or a["estimated"]
    tot = {k: sum(p[k] for p in per.values()) for k in ("calls", "ok", "retries", "failed_attempts", "tokens_in", "tokens_out")}
    tot["estimated"] = any(p["estimated"] for p in per.values())
    tot["packets"] = len(per)
    return {"totals": tot, "per_packet": per}


def from_broker(broker, packet_ids=None) -> Optional[Dict[str, Any]]:
    """Accounting for a StagedCriticBroker (broker ledger + the sibling progress.jsonl of a BatchRunner out dir). None when the broker has no ledger file (a test double)."""
    lp = getattr(broker, "ledger_path", None)
    if lp is None or not Path(lp).exists():
        return None
    cfg = getattr(broker, "cfg", None)
    return account(lp, Path(lp).parent / "progress.jsonl", packet_ids, getattr(cfg, "allowed_kinds", None))
