"""[EL-ARCH-1 F2d / security evidence] Reads the WSLC security gate receipts (security_gates.json, written by the backend side) and REFUSES a final report if any gate is missing.

    python tools/el_wslc_security_check.py <security_gates.json> [--expect-backend mxc-wslc] [--expect-digest sha256:<64 hex>]     -> JSON on stdout; exit 0 only if every gate passed

security_gates.json = {"backend": "mxc-wslc", "image_digest": "sha256:<64 hex>", "utc": iso,
                       "gates": {"host_file_read_blocked": {"status": "PASS"|"FAIL", "evidence": {...what was attempted and observed...}, "utc": iso}, "network_blocked": {...}, "timeout_kill": {...},
                                 "output_cap": {...}, "limits": {...memory / pids / cpu observed...}}}
A gate passes only if it is present, status == "PASS", its evidence is non-empty and its utc parses. The receipt's backend must be the expected one (default `mxc-wslc`) and its image digest must be
sha256:<64 hex> (and equal to --expect-digest when given: the image the run actually used). A missing file, unparsable JSON, a missing / failed / evidence-less gate, a foreign backend or a digest mismatch
all make `ok` false; `require_gates()` raises SecurityGateMissing, which tools/el_wslc_final_report.py uses to refuse a report."""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REQUIRED_GATES = ("host_file_read_blocked", "network_blocked", "timeout_kill", "output_cap", "limits")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class SecurityGateMissing(Exception):
    pass


def _utc_ok(s: Any) -> bool:
    try:
        datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return bool(s)
    except ValueError:
        return False


def check(path, expect_backend: str = "mxc-wslc", expect_digest: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": str(path), "ok": False, "missing": [], "failed": [], "problems": [], "gates": {}}
    p = Path(path)
    if not p.exists():
        out["problems"].append("security_gates.json does not exist")
        out["missing"] = list(REQUIRED_GATES)
        return out
    try:
        d = json.loads(p.read_text("utf-8"))
    except ValueError as ex:
        out["problems"].append("unparsable security_gates.json: %s" % ex)
        out["missing"] = list(REQUIRED_GATES)
        return out
    if not str(d.get("backend", "")).startswith(expect_backend):
        out["problems"].append("receipt backend %r is not the expected %r" % (d.get("backend"), expect_backend))
    if not DIGEST.match(str(d.get("image_digest", ""))):
        out["problems"].append("receipt image_digest %r is not sha256:<64 hex>" % (d.get("image_digest"),))
    elif expect_digest and d["image_digest"] != expect_digest:
        out["problems"].append("receipt image_digest differs from the image the run used")
    gates = d.get("gates") or {}
    for g in REQUIRED_GATES:
        r = gates.get(g)
        if not isinstance(r, dict):
            out["missing"].append(g)
            continue
        why = []
        if r.get("status") != "PASS":
            why.append("status %r" % (r.get("status"),))
        if not r.get("evidence"):
            why.append("no evidence")
        if not _utc_ok(r.get("utc")):
            why.append("no parseable utc")
        out["gates"][g] = {"ok": not why, "why": why}
        if why:
            out["failed"].append("%s (%s)" % (g, ", ".join(why)))
    out["ok"] = not (out["missing"] or out["failed"] or out["problems"])
    return out


def require_gates(path, expect_backend: str = "mxc-wslc", expect_digest: Optional[str] = None) -> Dict[str, Any]:
    r = check(path, expect_backend, expect_digest)
    if not r["ok"]:
        raise SecurityGateMissing("security gates not satisfied: missing=%s failed=%s problems=%s" % (r["missing"], r["failed"], r["problems"]))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--expect-backend", default="mxc-wslc")
    ap.add_argument("--expect-digest")
    a = ap.parse_args(argv)
    r = check(a.path, a.expect_backend, a.expect_digest)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
