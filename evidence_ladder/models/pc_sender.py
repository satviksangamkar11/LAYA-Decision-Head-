"""PC-side sender: a pure per-key scheduler plus a loop that drains the broker through an INJECTED transport.
No key handling and no key tooling here: the transport callable (owned by the PC operator) picks the credential for an opaque key label.
transport(packet, key_label) -> (http_status | None, raw_response). Limits: <= 10 in flight per key, <= 39 sends per key per rolling minute, 429 cools the key."""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from evidence_ladder.models.broker import LIMITS, StagedCriticBroker


class KeyScheduler:
    def __init__(self, key_labels: Sequence[str], *, max_in_flight: int = LIMITS["max_in_flight_per_key"],
                 per_minute: int = LIMITS["max_sends_per_key_per_minute"], cooldown_s: float = LIMITS["rate_limit_cooldown_s"], window_s: float = 60.0):
        if not key_labels:
            raise ValueError("at least one key label")
        self.keys = list(key_labels)
        self.max_in_flight, self.per_minute, self.cooldown_s, self.window_s = max_in_flight, per_minute, cooldown_s, window_s
        self.in_flight: Dict[str, int] = {k: 0 for k in self.keys}
        self.sends: Dict[str, Deque[float]] = {k: deque() for k in self.keys}
        self.cool_until: Dict[str, float] = {k: 0.0 for k in self.keys}
        self._rr = 0

    def _window(self, k: str, now: float) -> Deque[float]:
        d = self.sends[k]
        while d and d[0] <= now - self.window_s:
            d.popleft()
        return d

    def free_slots(self, k: str, now: float) -> int:
        if now < self.cool_until[k]:
            return 0
        return max(0, min(self.max_in_flight - self.in_flight[k], self.per_minute - len(self._window(k, now))))

    def capacity(self, now: float) -> int:
        return sum(self.free_slots(k, now) for k in self.keys)

    def acquire(self, now: float) -> Optional[str]:
        n = len(self.keys)
        for i in range(n):
            k = self.keys[(self._rr + i) % n]
            if self.free_slots(k, now) > 0:
                self._rr = (self._rr + i + 1) % n
                self.sends[k].append(now)
                self.in_flight[k] += 1
                return k
        return None

    def release(self, k: str, now: float, *, rate_limited: bool = False, retry_after: Optional[float] = None) -> None:
        self.in_flight[k] = max(0, self.in_flight[k] - 1)
        if rate_limited:
            self.cool_until[k] = max(self.cool_until[k], now + max(self.cooldown_s, retry_after or 0.0))

    def next_ready_time(self, now: float) -> float:
        """Earliest time any key has a free slot (ignores in-flight, which the caller releases before waiting)."""
        best = None
        for k in self.keys:
            t = max(now, self.cool_until[k])
            d = self._window(k, now)
            if len(d) >= self.per_minute:
                t = max(t, d[len(d) - self.per_minute] + self.window_s)
            best = t if best is None else min(best, t)
        return best


class PcSender:
    def __init__(self, broker: StagedCriticBroker, scheduler: KeyScheduler, *, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, executor=None, max_wait_s: float = 3600.0):
        """executor: optional object with .map(fn, items) (e.g. ThreadPoolExecutor) to run one wave concurrently; default is sequential."""
        self.broker, self.sched, self.clock, self.sleep, self.executor, self.max_wait_s = broker, scheduler, clock, sleep, executor, max_wait_s

    def send_pending(self, transport: Callable[[Dict[str, Any], str], Tuple[Optional[int], Any]], max_waves: int = 100000) -> Dict[str, Any]:
        states: Counter = Counter()
        sent = waves = 0
        while waves < max_waves and not self.broker.stop_all_critic_calls:
            now = self.clock()
            summ = self.broker.pending_summary()
            if summ["ready"] == 0 and not summ["waiting_until"]:
                break
            cap = self.sched.capacity(now)
            if cap == 0 or summ["ready"] == 0:
                wake = self.sched.next_ready_time(now) if summ["ready"] else summ["waiting_until"][0]
                self.sleep(min(self.max_wait_s, max(wake - now, 0.001)))
                continue
            packets = self.broker.export_pending(limit=cap)
            waves += 1
            wave = [(p, self.sched.acquire(now)) for p in packets]
            for p, k in wave:
                self.broker.log_send(p["packet_id"], k)

            def call(item):
                p, k = item
                try:
                    return transport(p, k)
                except TimeoutError:
                    return None, {"error": "timeout"}
                except Exception as e:   # transport failure is evidence of nothing; the broker retries it with backoff
                    return None, {"error": "transport_exception", "type": type(e).__name__}

            results = list(self.executor.map(call, wave)) if self.executor else [call(w) for w in wave]
            for (p, k), (status, raw) in zip(wave, results):
                res = self.broker.ingest(p["packet_id"], status, raw)
                self.sched.release(k, self.clock(), rate_limited=(status == 429), retry_after=res.get("retry_after_s"))
                states[res["state"]] += 1
                sent += 1
        return {"sent": sent, "waves": waves, "states": dict(states), "stopped": self.broker.stop_all_critic_calls}
