"""Authorization gate. Nothing is authorized by default; this module never records an approval the user has not given (callers pass them in)."""
from __future__ import annotations

import json
from pathlib import Path

from evidence_ladder.common.interfaces import NotAuthorized
from evidence_ladder.orchestrator.base import THRESHOLDS, sha, sha_bytes_lf_tolerant, write_json

APPROVALS = ("d1_waiver", "d7_critic_location", "nvidia_credits_confirmed", "pilot_go")
VERSION = "authorization_v1"
FILENAME = "authorization_manifest_v1.json"


def _thr_sha(thresholds_path) -> str:
    """sha256 of the thresholds toml, checked against its .sha256 file. Raises NotAuthorized on mismatch."""
    p = Path(thresholds_path)
    want = Path(str(p) + ".sha256").read_text(encoding="utf-8").split()[0]
    got = sha_bytes_lf_tolerant(p.read_bytes())
    if want not in got:
        raise NotAuthorized(f"thresholds toml sha256 {sorted(got)} != recorded {want}")
    return want


def _body(approvals: dict, thr: str) -> dict:
    return {"version": VERSION, "approvals": approvals, "thresholds_sha256": thr}


def write_authorization_manifest(approvals: dict = None, out_dir=".", thresholds_path=THRESHOLDS) -> Path:
    """Records exactly the approvals given (missing ones are False). Unknown names or non-bool values are rejected."""
    approvals = dict(approvals or {})
    bad = [k for k in approvals if k not in APPROVALS] + [k for k, v in approvals.items() if not isinstance(v, bool)]
    if bad:
        raise ValueError(f"invalid approvals: {bad}")
    full = {k: approvals.get(k, False) for k in APPROVALS}
    body = _body(full, _thr_sha(thresholds_path))
    p = Path(out_dir) / FILENAME
    write_json(p, dict(body, manifest_sha256=sha(body)))
    return p


def require_authorization(manifest_path, thresholds_path=THRESHOLDS) -> dict:
    p = Path(manifest_path) if manifest_path else None
    if p is None or not p.is_file():
        raise NotAuthorized("no authorization manifest: nothing is authorized")
    m = json.loads(p.read_text(encoding="utf-8"))
    try:
        body = _body(m["approvals"], m["thresholds_sha256"])
        ok = m["version"] == VERSION and m["manifest_sha256"] == sha(body)
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise NotAuthorized("authorization manifest is malformed or its hash does not match")
    if set(m["approvals"]) != set(APPROVALS):
        raise NotAuthorized("authorization manifest has the wrong approval keys")
    missing = [k for k in APPROVALS if m["approvals"][k] is not True]
    if missing:
        raise NotAuthorized(f"not approved: {missing}")
    if m["thresholds_sha256"] != _thr_sha(thresholds_path):
        raise NotAuthorized("thresholds toml changed since the approvals were recorded")
    return m
