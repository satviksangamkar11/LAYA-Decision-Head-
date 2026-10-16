"""LAYA / probe / random-head scorer metadata and gates (S3 compatibility contract).

The forward pass runs only on Kaggle (evidence_ladder/models/laya_forward.py, kaggle/evidence_ladder_laya_gen_v1.py); here score() either refuses (unsupported decision type, disabled scorer) or raises NotAuthorized
because no feature extraction or head execution exists here. Nothing in this module loads or modifies weights, adapters or the pretrained head;
it only hashes files and reads the baseline MANIFEST JSON."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evidence_ladder.common.contract_import import C, REPO
from evidence_ladder.common.interfaces import NotAuthorized, UnsupportedDecisionType
from evidence_ladder.models._util import sha256_json

MANIFEST_REL = "results/raw/laya-baseline-v1-20261012T215242Z/MANIFEST.json"
QWEN_REPO = "Qwen/Qwen3-4B-Thinking-2507"
QWEN_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
# S3 table, computed 2026-10-16 (both hashes re-computed from the main checkout the same day).
EXPECTED_CAPTURE_CODE = (
    {"path": "tools/fable_capture.py", "sha256": "9d8dd3790b73503646243ea897ed4bb6f4a47cebab20463cb324aba7fc63bd72"},
    {"path": "decision_head/fable_features.py", "sha256": "fe2ae8ea295a2991353238502fcb0da0b5998590dde3c64bf754f61f4486c9e9"},
)


class ScorerDisabled(Exception):
    """The scorer's compatibility check failed (capture-code hash mismatch, missing file or manifest): disabled for the run (S3-T5)."""


def file_sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


@dataclass
class ScorerMeta:
    scorer_def: Dict[str, Any]
    capture_report: Dict[str, Any]
    disabled_reason: str = ""


def _manifest(root: Path, manifest_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    p = manifest_path or (root / MANIFEST_REL)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_meta(scorer_id: str, version: str, kind: str, *, repo_root=None, manifest_path=None, expected_capture=EXPECTED_CAPTURE_CODE) -> ScorerMeta:
    """kind: 'laya' | 'probe' | 'random_head'. Hashes the capture code at load time and compares with the pinned table; any problem -> status 'disabled'."""
    root = Path(repo_root) if repo_root else REPO
    problems: List[str] = []
    report: Dict[str, Any] = {}
    for e in expected_capture:
        h = file_sha256(root / e["path"])
        report[e["path"]] = {"sha256": h, "expected": e["sha256"], "match": h == e["sha256"]}
        if h != e["sha256"]:
            problems.append("capture code hash mismatch or missing: " + e["path"])
    man = _manifest(root, Path(manifest_path) if manifest_path else None)
    if man is None:
        problems.append("baseline MANIFEST missing or unreadable")
        weights = "MISSING"
    elif kind == "laya":
        try:
            weights = sha256_json({"head": man["pretrained_head"]["sha256"], "adapters": {k: man["adapters"][k]["sha256"] for k in ("final-s0", "final-s1", "final-s2")}})
        except (KeyError, TypeError):
            problems.append("MANIFEST lacks head/adapter hashes")
            weights = "MISSING"
    elif kind == "probe":
        m = re.search(r"sha256 ([0-9a-f]{64})", str(man.get("linear_probe", {}).get("code", "")))
        weights = "recipe:" + (m.group(1) if m else "unknown") + ";refit_at_scoring_time"
    else:
        weights = "none:random_init_head"
    sd = {
        "scorer_id": scorer_id, "version": version, "weights_hash": weights,
        "status": "disabled" if problems else ("validated" if kind != "random_head" else "transfer_test"),
        "decision_types_validated": ["next_action"],
        "compatibility": {
            "qwen_repo": QWEN_REPO, "qwen_revision": QWEN_REVISION, "dtype": "fp16", "device_class": "kaggle_t4", "depths": [18, 24, 30],
            "query_feature": "hidden state at last prefix token after cue '\\n\\nNext action:'",
            "option_feature": "mean over the option's own tokens, independent suffix of the shared prefix, option prefix ' '",
            # schema has one path/hash string: both files, ';'-joined in the order of EXPECTED_CAPTURE_CODE
            "capture_code_path": ";".join(e["path"] for e in expected_capture),
            "capture_code_hash": ";".join(str(report[e["path"]]["sha256"]) for e in expected_capture),
            "max_options": 8, "prefix_token_limit": 24000, "option_token_limit": 4096,
        },
    }
    C.validate("scorer_def", sd)
    return ScorerMeta(sd, report, "; ".join(problems))


class _BaseScorer:
    """Common gate. Subclasses set KIND/SCORER_ID/VERSION. Not trained here; no weights are opened."""
    KIND = ""
    SCORER_ID = ""
    VERSION = "v1"

    def __init__(self, *, repo_root=None, manifest_path=None, expected_capture=EXPECTED_CAPTURE_CODE, authorized: bool = False, backend=None):
        self.meta = build_meta(self.SCORER_ID, self.VERSION, self.KIND, repo_root=repo_root, manifest_path=manifest_path, expected_capture=expected_capture)
        self.scorer_id = self.SCORER_ID
        self._authorized = authorized is True
        self._backend = backend   # e.g. laya_forward.SavedScoreBackend: scores a Kaggle kernel already computed; L0 only, returns scores, never an approval

    @property
    def scorer_def(self) -> Dict[str, Any]:
        return self.meta.scorer_def

    @property
    def disabled(self) -> bool:
        return self.scorer_def["status"] == "disabled"

    def allowed(self, decision_type: str, exploratory: bool = False) -> bool:
        return C.scorer_allowed(self.scorer_def, decision_type, exploratory)

    def score(self, decision_type: str, pool: Sequence[Any], exploratory: bool = False) -> List[float]:
        if self.disabled:
            raise ScorerDisabled("%s disabled: %s" % (self.scorer_id, self.meta.disabled_reason))
        if not self.allowed(decision_type, exploratory):
            raise UnsupportedDecisionType("%s is not validated for %r (validated: %s); pass exploratory=True only for a flagged exploratory run"
                                          % (self.scorer_id, decision_type, self.scorer_def["decision_types_validated"]))
        if self._authorized and self._backend is not None:
            return [float(x) for x in self._backend.score(decision_type, pool)]   # the backend refuses anything that is not frozen-Qwen features (laya_runtime)
        if self._authorized:
            raise NotImplementedError("%s forward pass is not implemented in this build" % self.scorer_id)
        raise NotAuthorized("%s scoring needs a Kaggle run with an explicit authorization; no feature extraction exists in this build" % self.scorer_id)


class LayaScorer(_BaseScorer):
    """Fable adapter (final-s0..s2) + pretrained Coding_Decision_Agent head. Validated for decision type next_action ONLY."""
    KIND, SCORER_ID = "laya", "laya_adapter_v1"


class ProbeScorer(_BaseScorer):
    """Logistic probe on the same frozen features (mandatory baseline). Refit at scoring time; no weights file."""
    KIND, SCORER_ID = "probe", "probe_v1"


class RandomHeadScorer(_BaseScorer):
    """Control: the adapter with a random-initialised head, to test whether the pretrained head matters."""
    KIND, SCORER_ID = "random_head", "random_head_v1"
