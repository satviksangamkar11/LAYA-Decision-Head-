"""Kernel-side batch loop (CP2/CP3) and PC-side importer. Pure orchestration over StagedCriticBroker + PcSender + an INJECTED transport: no network here.

DELIVERY IS AT-LEAST-ONCE, NOT EXACTLY-ONCE. Every answer is fsynced to progress.jsonl BEFORE it is ingested, so a completed packet is never resent.
A packet that was sent but not answered when the process died is UNCERTAIN (the provider may or may not have processed it): on restart it is
resent ONCE with the same packet_id; if it is uncertain a second time it becomes "unavailable" (uncertain_inflight_twice) and is not sent again.
Responses are keyed by packet_id, so a duplicate answer is dropped (broker "duplicate_ingest") and the bundle holds one record per packet."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evidence_ladder.critic_batch import validator as V
from evidence_ladder.critic_batch.config import scheduler_from_config
from evidence_ladder.models import glm
from evidence_ladder.models._util import canonical_json, redact
from evidence_ladder.models.broker import StagedCriticBroker, TERMINAL, packet_id_of
from evidence_ladder.models.pc_sender import PcSender

REASONING = {"reasoning_effort": "max"}      # FC-2 PASS (results/raw/evidence-ladder-critic-v1-run8-20261016/FC2.json); never lowered
UNCERTAIN_LIMIT = 1                           # resends allowed for a packet found in flight after a crash


class ProgressLedger:
    """Append-only JSONL, flush + fsync per line. Never holds key values (labels only); redact() strips credential-shaped text."""

    def __init__(self, path, clock: Callable[[], float] = time.time):
        self.path, self.clock, self.events = Path(path), clock, []
        torn = False
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.events.append(json.loads(line))
                        except ValueError:
                            torn = True                # torn last line from a crash: keep only the complete lines before it
                            break
            if torn:
                self._heal()

    def _heal(self) -> None:
        tmp = self.path.with_suffix(".healed")
        with open(tmp, "w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(canonical_json(ev) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def append(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        ev = dict(ev, seq=len(self.events) + 1, ts=self.clock())
        line = redact(canonical_json(ev))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.events.append(json.loads(line))
        return self.events[-1]

    def last(self, pid: str, kind: str) -> Optional[Dict[str, Any]]:
        for ev in reversed(self.events):
            if ev.get("packet_id") == pid and ev["event"] == kind:
                return ev
        return None

    def count(self, pid: str, kind: str) -> int:
        return sum(1 for ev in self.events if ev.get("packet_id") == pid and ev["event"] == kind)


class BatchRunner:
    def __init__(self, packets_dir, out_dir, transport, key_labels: Sequence[str], concurrency: Dict[str, Any], *, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, executor=None, max_retries: int = 3, poll_s: float = 0.05):
        self.packets_dir, self.out, self.transport, self.labels, self.cc = Path(packets_dir), Path(out_dir), transport, list(key_labels), concurrency
        self.clock, self.sleep, self.executor, self.max_retries, self.poll_s = clock, sleep, executor, max_retries, poll_s
        self.rejected: List[Dict[str, Any]] = []
        self.accepted: List[str] = []
        self.sched = None

    def _last_result(self, pid: str) -> Optional[Dict[str, Any]]:
        return self.ledger.last(pid, "result")

    def _recover(self) -> Dict[str, int]:
        rec = Counter()
        for pid in self.accepted:
            st = self.broker.packets[pid]["state"]
            r, ing = self.ledger.last(pid, "result"), self.ledger.last(pid, "ingested")
            if st not in TERMINAL and r is not None and (ing is None or ing["seq"] < r["seq"]):      # answered, but not yet ingested
                self.broker.ingest(pid, r["http_status"], r["raw"])
                self.ledger.append({"event": "ingested", "packet_id": pid, "state": self.broker.packets[pid]["state"], "recovered": True})
                rec["reingested"] += 1
        for pid in self.accepted:
            if self.broker.packets[pid]["state"] == "exported":                                       # sent, never answered: UNCERTAIN
                n = self.ledger.count(pid, "uncertain")
                self.ledger.append({"event": "uncertain", "packet_id": pid, "n": n + 1})
                if n >= UNCERTAIN_LIMIT:
                    self.ledger.append({"event": "result", "packet_id": pid, "key_label": None, "attempt": None, "http_status": 400,
                                        "raw": {"error": "uncertain_inflight_twice"}, "timing": None})
                    self.broker.ingest(pid, 400, {"error": "uncertain_inflight_twice"})
                    self.ledger.append({"event": "ingested", "packet_id": pid, "state": self.broker.packets[pid]["state"], "recovered": True})
                    rec["uncertain_exhausted"] += 1
                else:
                    self.broker.requeue_exported(pid)
                    self.ledger.append({"event": "resend_uncertain", "packet_id": pid})
                    rec["uncertain_resent"] += 1
        return dict(rec)

    def run(self) -> Dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        accepted, self.rejected = V.read_packets(self.packets_dir)
        (self.out / "rejected.json").write_text(json.dumps(self.rejected, indent=1, sort_keys=True), encoding="utf-8")
        self.ledger = ProgressLedger(self.out / "progress.jsonl", self.clock)
        self.broker = StagedCriticBroker(self.out / "broker_ledger.jsonl", config=glm.GlmConfig(reasoning_params=REASONING), max_retries=self.max_retries, clock=self.clock)
        if self.broker.stop_all_critic_calls:                  # an earlier run hit 402/quota: nothing new may be built or sent
            self.accepted = sorted(pid for pid, _ in accepted if pid in self.broker.packets)
        else:
            self.accepted = sorted(pid for pid, _ in accepted)
            for pid, req in accepted:
                assert self.broker.build_packet(req)["packet_id"] == pid
        self.ledger.append({"event": "run_start", "concurrency_sha256": self.cc["sha256"], "n_accepted": len(accepted), "n_rejected": len(self.rejected),
                            "key_labels": self.labels, "limits": self.cc["limits"], "ramp": self.cc["ramp"], "delivery": "at-least-once"})
        recovered = self._recover()
        self.sched = scheduler_from_config(self.labels, self.cc)
        sender = PcSender(self.broker, self.sched, clock=self.clock, sleep=self.sleep)
        own = self.executor is None
        ex = self.executor or ThreadPoolExecutor(max_workers=len(self.labels) * self.cc["limits"]["max_in_flight_per_key"])
        try:
            summary = sender.send_streaming(self.transport, ex, poll_s=self.poll_s, on_send=self._on_send, on_result=self._on_result, on_ingested=self._on_ingested)
        finally:
            if own:
                ex.shutdown(wait=True)
        manifest = self.write_bundle()
        self.ledger.append({"event": "run_end", "summary": summary, "recovered": recovered, "bundle_sha256": manifest["responses_sha256"]})
        return {"summary": summary, "recovered": recovered, "manifest": manifest}

    # ------------------------------------------------------------ hooks
    def _on_send(self, p, k):
        self.ledger.append({"event": "send", "packet_id": p["packet_id"], "key_label": k, "attempt": p["attempt"]})

    def _on_result(self, p, k, status, raw, timing):
        self.ledger.append({"event": "result", "packet_id": p["packet_id"], "key_label": k, "attempt": p["attempt"], "http_status": status, "raw": raw, "timing": timing})

    def _on_ingested(self, p, k, res):
        self.ledger.append({"event": "ingested", "packet_id": p["packet_id"], "state": res["state"], "key_label": k, "level": self.sched.level[k]})

    # ------------------------------------------------------------ bundle
    def write_bundle(self) -> Dict[str, Any]:
        recs, states = [], Counter()
        for pid in self.accepted:
            p = self.broker.packets[pid]
            r = self._last_result(pid) or {}
            asm = glm.assemble(r["raw"]) if r.get("raw") is not None else None
            usage = (asm or {}).get("usage") or {}
            state = p["state"] if p["state"] in TERMINAL else "not_completed"
            states[state] += 1
            recs.append({"packet_id": pid, "request_sha256": pid, "model": glm.GLM_MODEL, "state": state, "reason": p["reason"], "broker_state": p["state"],
                         "attempts_failed": p["attempts"], "attempt": r.get("attempt"), "key_label": r.get("key_label"), "http_status": r.get("http_status"),
                         "response": p["response"], "usage": {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")},
                         "reasoning_chars": len((asm or {}).get("reasoning") or ""), "timing": r.get("timing"), "raw": r.get("raw")})
        data = ("\n".join(canonical_json(x) for x in recs) + ("\n" if recs else "")).encode("utf-8")
        (self.out / "responses.jsonl").write_bytes(data)
        try:
            in_man = json.loads((self.packets_dir / "PACKETS_MANIFEST.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            in_man = {}
        man = {"bundle": V.RESPONSE_BUNDLE, "count": len(recs), "responses_sha256": V._sha(data), "states": dict(sorted(states.items())), "rejected": self.rejected,
               "model": glm.GLM_MODEL, "packets_sha256": in_man.get("packets_sha256"), "concurrency_sha256": self.cc["sha256"],
               "stop_all_critic_calls": self.broker.stop_all_critic_calls, "delivery": "at-least-once"}
        (self.out / "MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
        return man


# ---------------------------------------------------------------- PC side
def import_bundle(bundle_dir, broker: StagedCriticBroker) -> Dict[str, Any]:
    """Feed a verified response bundle into the PC broker. Rejects (records, does not ingest): packet ids unknown to the broker, id/hash mismatches,
    duplicates (inside the bundle or already terminal in the broker), and records that never completed. The kernel already used up its retries, so a
    non-answer in the bundle becomes "unavailable" here (max_retries is 0 for the duration of the import)."""
    d = Path(bundle_dir)
    try:
        man = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
        data = (d / "responses.jsonl").read_bytes()
    except (OSError, ValueError) as e:
        raise V.ManifestError("cannot read response bundle: %s" % type(e).__name__)
    if man.get("bundle") != V.RESPONSE_BUNDLE or man.get("responses_sha256") != V._sha(data):
        raise V.ManifestError("response bundle id or sha256 does not match")
    out: Dict[str, Any] = {"ingested": [], "rejected": []}
    seen = set()
    saved, broker.max_retries = broker.max_retries, 0
    try:
        for n, raw in enumerate(data.decode("utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
                pid = rec["packet_id"]
            except (ValueError, KeyError, TypeError):
                out["rejected"].append({"line": n, "packet_id": None, "class": "malformed"})
                continue

            def rej(cls):
                out["rejected"].append({"line": n, "packet_id": pid, "class": cls})
            if pid in seen:
                rej("duplicate_in_bundle")
                continue
            seen.add(pid)
            if pid not in broker.packets:
                rej("unknown_packet")
                continue
            if rec.get("request_sha256") != pid or packet_id_of(broker.packets[pid]["request"]) != pid:
                rej("id_hash_mismatch")
                continue
            if rec.get("state") not in TERMINAL:
                rej("not_completed")
                continue
            if broker.packets[pid]["state"] in TERMINAL:
                broker.ingest(pid, rec.get("http_status"), rec.get("raw"))          # logs duplicate_ingest, changes nothing
                rej("duplicate_already_terminal")
                continue
            res = broker.ingest(pid, rec.get("http_status"), rec.get("raw"))
            out["ingested"].append({"packet_id": pid, "state": res["state"], "bundle_state": rec["state"], "match": res["state"] == rec["state"]})
    finally:
        broker.max_retries = saved
    out["mismatched_states"] = [x["packet_id"] for x in out["ingested"] if not x["match"]]
    return out


def export_for_kernel(broker: StagedCriticBroker, out_dir) -> Dict[str, Any]:
    """PC side: take what the broker would send now (marks those packets exported in its ledger) and write the validated packet dataset directory."""
    ids = [e["packet_id"] for e in broker.export_pending()]
    return V.write_packets(out_dir, {pid: broker.packets[pid]["request"] for pid in ids})
