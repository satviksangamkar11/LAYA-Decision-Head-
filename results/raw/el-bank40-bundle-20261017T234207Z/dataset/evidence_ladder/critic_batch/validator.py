"""Allow-list validator shared by the PC export and the kernel import. Pure; no network, no keys.
A critic packet line is exactly {"packet_id", "request"}. The request must pass glm.validate_request (top-level and observed_outputs key allow-lists,
critic_request schema) AND contain none of the forbidden content classes below. packet_id must be the content hash of the request."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from evidence_ladder.models import glm
from evidence_ladder.models._util import canonical_json
from evidence_ladder.models.broker import packet_id_of

PACKET_BUNDLE = "evidence_ladder_critic_packets_v1"
RESPONSE_BUNDLE = "evidence_ladder_critic_responses_v1"
LINE_KEYS = {"packet_id", "request"}
_I = re.I
FORBIDDEN_CLASSES: Dict[str, "re.Pattern"] = {
    "hidden_tests": re.compile(r"hidden[ _-]?(tests?|cases?)", _I),
    "protected_cases": re.compile(r"protected[ _-]?(tests?|cases?|inputs?|grader|evaluator)", _I),
    "reference_solution": re.compile(r"reference[ _-]?(solution|implementation|answer)|gold[ _-]?(solution|patch)", _I),
    "private_labels": re.compile(r"private[ _-]?labels?|ground[ _-]?truth", _I),
    "laya": re.compile(r"\blaya\b|laya[_ -]?(choice|score|head|pick)", _I),
    "qwen_private_reasoning": re.compile(r"</?think>", _I),
    "secret": re.compile(r"nvapi-[A-Za-z0-9_\-]{6,}|hf_[A-Za-z0-9]{16,}|KGAT_[A-Za-z0-9]{6,}|\bsk-[A-Za-z0-9_\-]{16,}|Bearer\s+[A-Za-z0-9._\-]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY"),
}


class PacketRejected(Exception):
    def __init__(self, cls: str, detail: str = ""):
        super().__init__("%s: %s" % (cls, detail))
        self.cls, self.detail = cls, detail


class ManifestError(Exception):
    pass


def forbidden_class_of(request: Any) -> str:
    """First forbidden content class found anywhere in the request (keys and values), else ''."""
    text = canonical_json(request)
    for name, rx in FORBIDDEN_CLASSES.items():
        if rx.search(text):
            return name
    return ""


def validate_packet(line: Any) -> Dict[str, Any]:
    """-> validated request. Raises PacketRejected(class, detail)."""
    if not isinstance(line, dict) or set(line) != LINE_KEYS:
        raise PacketRejected("unexpected_fields", "packet line keys must be exactly %s" % sorted(LINE_KEYS))
    pid, req = line["packet_id"], line["request"]
    if not isinstance(pid, str) or not isinstance(req, dict):
        raise PacketRejected("malformed", "packet_id must be a string and request an object")
    cls = forbidden_class_of(req)
    if cls:
        raise PacketRejected(cls, "forbidden content class in request")
    try:
        clean = glm.validate_request(req)
    except glm.BlindingError as e:
        raise PacketRejected("blinding", str(e)[:200])
    except ValueError as e:
        raise PacketRejected("schema", str(e)[:200])
    if packet_id_of(clean) != pid:
        raise PacketRejected("id_mismatch", "packet_id is not the content hash of the request")
    return clean


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_packets(out_dir, requests: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """PC side. requests: {packet_id: request}. Every packet is validated before anything is written (a bad one raises)."""
    out = Path(out_dir)
    lines = []
    for pid in sorted(requests):
        validate_packet({"packet_id": pid, "request": dict(requests[pid])})
        lines.append(canonical_json({"packet_id": pid, "request": requests[pid]}))
    out.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    (out / "packets.jsonl").write_bytes(data)
    man = {"bundle": PACKET_BUNDLE, "count": len(lines), "packets_sha256": _sha(data), "packet_ids": sorted(requests), "model": glm.GLM_MODEL}
    (out / "PACKETS_MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    return man


def read_packets(in_dir) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    """Kernel side. -> (accepted [(packet_id, request)], rejected [{line, packet_id, class, detail}]). ManifestError if the manifest or file hash is wrong."""
    d = Path(in_dir)
    try:
        man = json.loads((d / "PACKETS_MANIFEST.json").read_text(encoding="utf-8"))
        data = (d / "packets.jsonl").read_bytes()
    except (OSError, ValueError) as e:
        raise ManifestError("cannot read packet manifest/file: %s" % type(e).__name__)
    if man.get("bundle") != PACKET_BUNDLE or man.get("packets_sha256") != _sha(data):
        raise ManifestError("packet bundle id or sha256 does not match")
    accepted, rejected, seen = [], [], set()
    n_lines = 0
    for n, raw in enumerate(data.decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        n_lines += 1
        try:
            line = json.loads(raw)
        except ValueError:
            rejected.append({"line": n, "packet_id": None, "class": "malformed", "detail": "not JSON"})
            continue
        pid = line.get("packet_id") if isinstance(line, dict) else None
        try:
            req = validate_packet(line)
            if pid in seen:
                raise PacketRejected("duplicate_packet", pid)
        except PacketRejected as e:
            rejected.append({"line": n, "packet_id": pid if isinstance(pid, str) else None, "class": e.cls, "detail": e.detail})
            continue
        seen.add(pid)
        accepted.append((pid, req))
    if man.get("count") != n_lines:
        raise ManifestError("manifest count %r != lines read %d" % (man.get("count"), n_lines))
    return accepted, rejected


# ---------------------------------------------------------------- static checks on what the kernel is allowed to see
def check_kernel_metadata(meta: Mapping[str, Any], packet_dataset_id: str) -> None:
    """The critic kernel may mount ONLY the packet dataset (never test bundles or hidden data). Raises ValueError."""
    if meta.get("dataset_sources") != [packet_dataset_id]:
        raise ValueError("dataset_sources must be exactly [%r], got %r" % (packet_dataset_id, meta.get("dataset_sources")))
    for k in ("kernel_sources", "competition_sources", "model_sources"):
        if meta.get(k):
            raise ValueError("%s must be empty" % k)
    if "packets" not in packet_dataset_id or "test" in packet_dataset_id or "bundle" in packet_dataset_id:
        raise ValueError("dataset id does not look like a packet dataset: %r" % packet_dataset_id)
    if meta.get("enable_gpu") or not meta.get("enable_internet"):
        raise ValueError("critic kernel: internet on, GPU off")


DATASET_TOP_FILES = {"packets.jsonl", "PACKETS_MANIFEST.json", "CODE_MANIFEST.json", "dataset-metadata.json"}
DATASET_CODE_SUFFIXES = (".py", ".json", ".toml")
FORBIDDEN_PATH_RE = re.compile(r"hidden|protected|grader|\.s" r"ecrets|reference_solution|worker|results", re.I)


def check_dataset_dir(d) -> List[str]:
    """Allow-list of files in the packet dataset: the four top-level files plus code/** with .py/.json/.toml names. Returns the violations (empty = ok)."""
    root, bad = Path(d), []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        top_ok = rel in DATASET_TOP_FILES
        code_ok = rel.startswith("code/") and rel.endswith(DATASET_CODE_SUFFIXES) and "__pycache__" not in rel
        if not (top_ok or code_ok) or FORBIDDEN_PATH_RE.search(rel):
            bad.append(rel)
    return bad
