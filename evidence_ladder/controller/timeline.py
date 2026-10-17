"""[EL-ARCH-1 F2b] OPTIONAL timeline side channel: real timestamps for every journal entry and every pipeline stage / model call / sandbox call, kept in a SIDE list.
Nothing here is ever written into a journal entry, a receipt field that is hashed, or a decision: journal hashes, replay and receipts are byte-identical with the timeline off or on.
A missing or failing clock only blanks the timestamps of that row (`clock_error`); a failing Timeline can never change a decision (every hook call is guarded, and the controllers
catch everything a Timeline raises).

    from evidence_ladder.controller.timeline import Timeline
    tl = Timeline()                                    # default clocks: real UTC (datetime.now(timezone.utc)) and time.monotonic; Timeline(utc=fn, mono=fn) for tests
    ctx = AgenticContext(sandbox=..., timeline=tl)     # agentic mode: stage map intake/investigate/generate/filter/check/rank/commit/observe/repair/discriminate/commit_verify
    ctl = EvidenceLadderController(timeline=tl)        # ladder mode: intake/generate/execute/l2/cluster/discriminate/rank/repair/commit_verify
    tl.attach(journal)                                 # (done by the controllers) every journal.append calls tl.on_entry(seq, kind)
    rows = tl.rows()        # JSON-serialisable list of dicts, in call order
    summ = tl.summary()     # {"stages": {stage: {count, total_s, p50, p95}}, "spans": {...}, "events": {kind: {count, total_s, p50, p95}}, "entries": n, "wall_s": ...}

Row shapes (all rows: row, seq_in_timeline, utc_iso, monotonic_s, stage, step, and `clock_error` when a clock raised):
  {"row": "entry", "seq": <journal seq>, "kind": <journal kind>}
  {"row": "stage", "name", "start_utc_iso", "start_monotonic_s", "end_utc_iso", "end_monotonic_s", "duration_s", ...meta}      (flat stage map: enter() closes the open stage)
  {"row": "span", "name", "start_*", "end_*", "duration_s", "parent_stage", ...meta}                                           (nested timed section, e.g. one sandbox run)
  {"row": "event", "kind": "model_call"|"sandbox"|..., "seconds", ...meta}   model_call: model, tokens_prompt, tokens_generated, seconds; sandbox: seconds, check_id, status
"""
from __future__ import annotations

import contextlib
import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(vals: List[float], p: float) -> Optional[float]:
    """nearest-rank percentile; None for an empty list"""
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(math.ceil(p / 100.0 * len(s))) - 1))]


def _agg(vals: List[float]) -> Dict[str, Any]:
    return {"count": len(vals), "total_s": round(sum(vals), 6), "p50": _pct(vals, 50), "p95": _pct(vals, 95) if len(vals) >= 2 else None}      # p95 only where it means something (n >= 2)


class Timeline:
    def __init__(self, utc: Optional[Callable[[], str]] = None, mono: Optional[Callable[[], float]] = None):
        self._utc = utc or _utc_now
        self._mono = mono or time.monotonic
        self._rows: List[Dict[str, Any]] = []
        self._open: Optional[Dict[str, Any]] = None
        self._stage: Optional[str] = None
        self._step: Optional[int] = None
        self._parent: List[str] = []

    # ---- clocks (never raise)
    def _now(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            out["utc_iso"] = self._utc()
        except Exception as ex:      # noqa: BLE001  a failing clock blanks this row only
            out["utc_iso"], out["clock_error"] = None, "utc: %s" % type(ex).__name__
        try:
            out["monotonic_s"] = float(self._mono())
        except Exception as ex:      # noqa: BLE001
            out["monotonic_s"] = None
            out["clock_error"] = (out.get("clock_error", "") + " mono: %s" % type(ex).__name__).strip()
        return out

    def _add(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["seq_in_timeline"] = len(self._rows)
        row.setdefault("stage", self._stage)
        row.setdefault("step", self._step)
        self._rows.append(row)
        return row

    @staticmethod
    def _dur(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None else round(max(0.0, b - a), 6)

    # ---- journal hook
    def attach(self, journal) -> None:
        journal.timeline = self

    def on_entry(self, seq: int, kind: str) -> None:
        self._add(dict({"row": "entry", "seq": seq, "kind": kind}, **self._now()))

    # ---- flat stage map
    def enter(self, name: str, step: Optional[int] = None, **meta: Any) -> None:
        self.leave()
        if step is not None:
            self._step = step
        self._stage = name
        t = self._now()
        self._open = {"row": "stage", "name": name, "start_utc_iso": t["utc_iso"], "start_monotonic_s": t["monotonic_s"], "meta": dict(meta), "clock_error": t.get("clock_error")}

    def leave(self) -> None:
        o, self._open = self._open, None
        if o is None:
            return
        t = self._now()
        row = {"row": "stage", "name": o["name"], "start_utc_iso": o["start_utc_iso"], "start_monotonic_s": o["start_monotonic_s"], "end_utc_iso": t["utc_iso"], "end_monotonic_s": t["monotonic_s"],
               "duration_s": self._dur(o["start_monotonic_s"], t["monotonic_s"]), "stage": o["name"], "step": self._step}
        row.update(o["meta"])
        errs = " ".join(x for x in (o.get("clock_error"), t.get("clock_error")) if x)
        if errs:
            row["clock_error"] = errs
        self._add(row)

    # ---- nested timed sections
    @contextlib.contextmanager
    def span(self, name: str, **meta: Any):
        a = self._now()
        self._parent.append(name)
        try:
            yield meta
        finally:
            self._parent.pop()
            b = self._now()
            row = {"row": "span", "name": name, "start_utc_iso": a["utc_iso"], "start_monotonic_s": a["monotonic_s"], "end_utc_iso": b["utc_iso"], "end_monotonic_s": b["monotonic_s"],
                   "duration_s": self._dur(a["monotonic_s"], b["monotonic_s"]), "parent_stage": self._stage}
            row.update(meta)
            errs = " ".join(x for x in (a.get("clock_error"), b.get("clock_error")) if x)
            if errs:
                row["clock_error"] = errs
            self._add(row)

    def event(self, kind: str, seconds: Optional[float] = None, **meta: Any) -> None:
        """a point record with a measured duration supplied by the caller (model seconds reported by the backend, a sandbox run timed by the caller)"""
        self._add(dict({"row": "event", "kind": kind, "seconds": None if seconds is None else round(float(seconds), 6)}, **meta, **self._now()))

    def model_call(self, model: str, seconds: Optional[float], tokens_prompt: int = 0, tokens_generated: int = 0, **meta: Any) -> None:
        self.event("model_call", seconds, model=model, tokens_prompt=int(tokens_prompt), tokens_generated=int(tokens_generated), **meta)

    def sandbox_call(self, seconds: Optional[float], check_id: Optional[str], status: Optional[str], **meta: Any) -> None:
        self.event("sandbox", seconds, check_id=check_id, status=status, **meta)

    def mark(self) -> Optional[float]:
        """the monotonic clock now (None if it fails); pair with finish_call"""
        return self._now()["monotonic_s"]

    def finish_call(self, kind: str, mark: Optional[float], status: Optional[str] = None, **meta: Any) -> None:
        self.event(kind, self._dur(mark, self._now()["monotonic_s"]), status=status, **meta)

    def timed_call(self, kind: str, fn: Callable[[], Any], **meta: Any):
        """run fn() (a model or tool call), record an event of `kind` with its real seconds, and return fn()'s result unchanged; fn's exceptions propagate (meta status 'error')."""
        a = self._now()
        try:
            res = fn()
        except BaseException:
            self.event(kind, self._dur(a["monotonic_s"], self._now()["monotonic_s"]), status="error", **meta)
            raise
        self.event(kind, self._dur(a["monotonic_s"], self._now()["monotonic_s"]), **meta)
        return res

    def timed_sandbox(self, check_id: Optional[str], fn: Callable[[], Any], status_of: Callable[[Any], Optional[str]], **meta: Any):
        """run fn() (a sandbox call), record a `sandbox` event with its real seconds, check_id and status, and return fn()'s result unchanged; fn's exceptions propagate (status 'error')."""
        a = self._now()
        try:
            res = fn()
        except BaseException:
            self.sandbox_call(self._dur(a["monotonic_s"], self._now()["monotonic_s"]), check_id, "error", **meta)
            raise
        try:
            st = status_of(res)
        except Exception:      # noqa: BLE001
            st = None
        self.sandbox_call(self._dur(a["monotonic_s"], self._now()["monotonic_s"]), check_id, st, **meta)
        return res

    # ---- output
    def rows(self) -> List[Dict[str, Any]]:
        self.leave()
        return [dict(r) for r in self._rows]

    def summary(self) -> Dict[str, Any]:
        rows = self.rows()
        by: Dict[str, Dict[str, List[float]]] = {"stages": {}, "spans": {}, "events": {}}
        for r in rows:
            if r["row"] == "stage" and r.get("duration_s") is not None:
                by["stages"].setdefault(r["name"], []).append(r["duration_s"])
            elif r["row"] == "span" and r.get("duration_s") is not None:
                by["spans"].setdefault(r["name"], []).append(r["duration_s"])
            elif r["row"] == "event" and r.get("seconds") is not None:
                by["events"].setdefault(r["kind"], []).append(r["seconds"])
        mono = [r["monotonic_s"] for r in rows if r.get("monotonic_s") is not None] + [r["start_monotonic_s"] for r in rows if r.get("start_monotonic_s") is not None]
        ends = [r["end_monotonic_s"] for r in rows if r.get("end_monotonic_s") is not None]
        allm = mono + ends
        return {"stages": {k: _agg(v) for k, v in sorted(by["stages"].items())}, "spans": {k: _agg(v) for k, v in sorted(by["spans"].items())},
                "events": {k: _agg(v) for k, v in sorted(by["events"].items())}, "entries": sum(1 for r in rows if r["row"] == "entry"),
                "wall_s": None if not allm else round(max(allm) - min(allm), 6), "clock_errors": sum(1 for r in rows if r.get("clock_error"))}
