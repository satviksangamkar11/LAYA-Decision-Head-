"""[EL-ARCH-1 F2c / B1] Optional per-candidate hidden-state capture for the local bank run (RTX 3050 + a resident causal LM such as Qwen2.5-Coder-1.5B-Instruct: 28 layers, hidden 1536,
depths 14/19/23). One call per candidate, states streamed to disk as fp16 and freed at once (free RAM is about 1 GB: nothing accumulates in memory).

    from evidence_ladder.models.state_capture import StateCapture
    sc = StateCapture(capture_fn, out_root="<run dir>", profile_id="qwen25-coder-1.5b", depths=(14, 19, 23))      # capture_fn=None or enabled=False: every call returns None and does nothing
    path = sc.capture(task_id, candidate_id, context_text, candidate_text)                                          # -> path of tasks/<task_id>/states/<candidate_id>.pt, or None

`capture_fn(context_text, options) -> (hq, ho, info)` is the recipe of evidence_ladder.models.profile_capture.make_capture_fn (query state at the LAST prefix token after the cue, mean over
the option's own tokens; fp16): hq [depths, hidden], ho [n_options, depths, hidden]; info["status"] == "ok" on success. The hook calls it with ONE option (the candidate's source text), so
the stored file holds hq [D, H] and ho [D, H] for that candidate. File (torch.save of a dict): candidate_id, task_id, profile_id, depths, hq, ho (fp16), context_sha256, option_sha256,
features_sha256 (from info when present), seconds. A line per state is appended to tasks/<task_id>/states/index.jsonl.

Guarantees: never raises (every failure is counted in .errors and returns None); never writes into a journal entry, hash, receipt or record; never changes a decision (it only reads its arguments);
silent when disabled; resumable (an existing file is kept and returned); atomic write (temp file + replace). The states are DIAGNOSTIC material for tools/el_bank_state_probe.py.
The frozen 4B LAYA head cannot consume these states (hidden 1536 vs the head's 2560; the capture-provenance guard refuses a state of another profile, see tools/el_bank_state_probe.py)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class StateCapture:
    def __init__(self, capture_fn: Optional[Callable[[str, Sequence[str]], Any]], out_root, profile_id: str = "unknown", depths: Sequence[int] = (), enabled: bool = True):
        self.capture_fn, self.root, self.profile_id, self.depths = capture_fn, Path(out_root), profile_id, tuple(int(d) for d in depths)
        self.enabled = bool(enabled) and capture_fn is not None
        self.errors: List[Dict[str, str]] = []
        self.saved = 0
        self.skipped = 0

    def path_of(self, task_id: str, candidate_id: str) -> Path:
        return self.root / "tasks" / task_id / "states" / (candidate_id + ".pt")

    def capture(self, task_id: str, candidate_id: str, context_text: str, candidate_text: str) -> Optional[Path]:
        if not self.enabled:
            return None
        try:
            return self._capture(task_id, candidate_id, context_text, candidate_text)
        except BaseException as ex:      # noqa: BLE001  nothing here may change the run (KeyboardInterrupt is not swallowed)
            if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                raise
            self.errors.append({"task_id": task_id, "candidate_id": candidate_id, "error": "%s: %s" % (type(ex).__name__, str(ex)[:200])})
            return None

    def _capture(self, task_id: str, candidate_id: str, context_text: str, candidate_text: str) -> Optional[Path]:
        import torch
        p = self.path_of(task_id, candidate_id)
        if p.exists():
            self.skipped += 1
            return p
        t0 = time.time()
        hq, ho, info = self.capture_fn(context_text, [candidate_text])
        if hq is None or ho is None or (info or {}).get("status", "ok") != "ok":
            self.errors.append({"task_id": task_id, "candidate_id": candidate_id, "error": "capture status %s" % (info or {}).get("status")})
            return None
        hq, ho = hq.detach().to("cpu").half(), ho.detach().to("cpu").half()
        if ho.dim() == 3:                       # [1 option, D, H] -> [D, H]
            ho = ho[0]
        if hq.dim() != 2 or ho.dim() != 2 or hq.shape != ho.shape:
            raise ValueError("unexpected state shapes hq %s ho %s" % (tuple(hq.shape), tuple(ho.shape)))
        if self.depths and hq.shape[0] != len(self.depths):
            raise ValueError("state has %d depths, the profile says %d" % (hq.shape[0], len(self.depths)))
        if not (torch.isfinite(hq).all() and torch.isfinite(ho).all()):
            raise ValueError("non-finite state")
        rec: Dict[str, Any] = {"candidate_id": candidate_id, "task_id": task_id, "profile_id": self.profile_id, "depths": list(self.depths), "hq": hq.contiguous(), "ho": ho.contiguous(),
                               "context_sha256": _sha(context_text), "option_sha256": _sha(candidate_text), "features_sha256": (info or {}).get("features_sha256"),
                               "seconds": round(time.time() - t0, 3)}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".pt.tmp")
        torch.save(rec, tmp)
        os.replace(tmp, p)
        with open(p.parent / "index.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"candidate_id": candidate_id, "file": p.name, "profile_id": self.profile_id, "depths": list(self.depths), "hidden": int(hq.shape[1]),
                                 "context_sha256": rec["context_sha256"], "option_sha256": rec["option_sha256"], "features_sha256": rec["features_sha256"], "seconds": rec["seconds"]}, sort_keys=True) + "\n")
        self.saved += 1
        del hq, ho, rec                            # nothing accumulates: the next candidate starts from an empty slate
        return p

    def summary(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "saved": self.saved, "skipped_existing": self.skipped, "errors": len(self.errors), "profile_id": self.profile_id, "depths": list(self.depths)}


def load_state(path) -> Dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)
