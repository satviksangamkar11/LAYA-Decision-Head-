"""[EL-ARCH-1 local run] BACKBONE PROFILE: the one model-agnostic description of a frozen backbone (repo, pinned revision, hidden size, layers, capture depths, dtype, device policy, thinking setting,
recipe id, and which frozen adapter sets can consume its states). Everything else (capture, generation, provenance guard, receipts) READS the profile instead of hard-coding the 4B constants.
Source: configs/backbones_v1.toml. The 4B entry is also embedded here (like laya.py's capture pins) so an old bundle without the toml still works; when both exist they must agree.
Adding a model = one toml entry (+ an adapter set if head_compatible). Pure standard library."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_REL = "configs/backbones_v1.toml"
DEFAULT_ID = "qwen3-4b-thinking-2507"          # states of records written before profiles existed are, by definition, this profile's

_BUILTIN_4B: Dict[str, Any] = {
    "repo": "Qwen/Qwen3-4B-Thinking-2507", "revision": "768f209d9ea81521153ed38c47d515654e938aea", "local_path": "models/qwen3-4b-thinking-2507", "hidden_size": 2560, "num_layers": 36,
    "depths": [18, 24, 30], "dtype": "float16", "local_device_policy": "stream", "device_classes": ["Tesla T4"], "recipe_id": "fable_capture_v1", "query_cue": "\n\nNext action:",
    "option_prefix": " ", "head_compatible": ["laya_fable_adapters_v1"], "generation": {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}}


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    id: str
    repo: str
    revision: str
    local_path: str
    hidden_size: int
    num_layers: int
    depths: Tuple[int, ...]
    dtype: str
    local_device_policy: str
    device_classes: Tuple[str, ...]
    recipe_id: str
    query_cue: str
    option_prefix: str
    head_compatible: Tuple[str, ...]
    generation: Mapping[str, Any]
    enable_thinking: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "repo": self.repo, "revision": self.revision, "local_path": self.local_path, "hidden_size": self.hidden_size, "num_layers": self.num_layers, "depths": list(self.depths),
                "dtype": self.dtype, "local_device_policy": self.local_device_policy, "device_classes": list(self.device_classes), "recipe_id": self.recipe_id, "query_cue": self.query_cue,
                "option_prefix": self.option_prefix, "head_compatible": list(self.head_compatible), "generation": dict(self.generation), "enable_thinking": self.enable_thinking}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    @property
    def is_head_compatible(self) -> bool:
        return bool(self.head_compatible)

    def model_dir(self, root: Optional[Path] = None) -> Path:
        return Path(root or REPO_ROOT) / self.local_path


def from_dict(pid: str, d: Mapping[str, Any]) -> Profile:
    need = ("repo", "revision", "local_path", "hidden_size", "num_layers", "depths", "dtype", "local_device_policy", "device_classes", "recipe_id", "query_cue", "option_prefix", "head_compatible", "generation")
    miss = [k for k in need if k not in d]
    if miss:
        raise ProfileError("profile %r lacks %s" % (pid, miss))
    depths = tuple(int(x) for x in d["depths"])
    if not depths or list(depths) != sorted(set(depths)) or depths[0] < 1 or depths[-1] > int(d["num_layers"]):
        raise ProfileError("profile %r: depths %r must be increasing, within 1..%d" % (pid, depths, d["num_layers"]))
    if d["local_device_policy"] not in ("stream", "resident"):
        raise ProfileError("profile %r: local_device_policy must be stream|resident" % pid)
    return Profile(pid, d["repo"], d["revision"], d["local_path"], int(d["hidden_size"]), int(d["num_layers"]), depths, d["dtype"], d["local_device_policy"], tuple(d["device_classes"]), d["recipe_id"],
                   d["query_cue"], d["option_prefix"], tuple(d["head_compatible"]), dict(d["generation"]), d.get("enable_thinking"))


def load_all(path: Optional[Path] = None) -> Dict[str, Profile]:
    out = {DEFAULT_ID: from_dict(DEFAULT_ID, _BUILTIN_4B)}
    p = Path(path) if path else REPO_ROOT / CONFIG_REL
    if p.is_file():
        import tomllib
        for pid, d in tomllib.loads(p.read_text(encoding="utf-8")).get("backbone", {}).items():
            prof = from_dict(pid, d)
            if pid == DEFAULT_ID and prof != out[DEFAULT_ID]:
                raise ProfileError("%s differs from the 4B profile embedded in backbone.py" % CONFIG_REL)
            out[pid] = prof
    return out


_CACHE: Dict[str, Dict[str, Profile]] = {}


def get(pid: Optional[str] = None, path: Optional[Path] = None) -> Profile:
    key = str(path or "")
    if key not in _CACHE:
        _CACHE[key] = load_all(path)
    pid = pid or DEFAULT_ID
    if pid not in _CACHE[key]:
        raise ProfileError("unknown backbone profile %r (known: %s)" % (pid, sorted(_CACHE[key])))
    return _CACHE[key][pid]


def require_head_compatible(profile: Profile, what: str = "the frozen LAYA head") -> None:
    if not profile.is_head_compatible:
        raise ProfileError("states of backbone %r (hidden %d) are DIAGNOSTIC data: not consumable by %s (head_compatible is empty); they are data for a future adapter"
                           % (profile.id, profile.hidden_size, what))


# ------------------------------------------------------------------ extra (parity-gated) devices, per profile, per process
_EXTRA_DEVICES: Dict[str, List[str]] = {}


def register_device(profile: Profile, device_prefix: str, registration_file: Path) -> None:
    """Registers a device class for ONE profile for THIS PROCESS, only from a versioned file that names a PASSED parity result whose sha256 it carries (re-checked here)."""
    import tomllib
    reg = tomllib.loads(Path(registration_file).read_text(encoding="utf-8"))["registration"]
    res = Path(reg["parity_result"])
    res = res if res.is_absolute() else REPO_ROOT / res
    if hashlib.sha256(res.read_bytes()).hexdigest() != reg["parity_result_sha256"]:
        raise ProfileError("parity result file does not match the sha256 in the registration")
    doc = json.loads(res.read_text(encoding="utf-8"))
    if doc.get("verdict") != "PASS" or reg["profile_id"] != profile.id or not str(device_prefix).startswith(reg["device_prefix"]) or doc.get("device_prefix") != reg["device_prefix"]:
        raise ProfileError("registration does not match a PASSED parity result for this profile and device")
    _EXTRA_DEVICES.setdefault(profile.id, []).append(reg["device_prefix"])


def clear_registered_devices() -> None:
    _EXTRA_DEVICES.clear()


def allowed_devices(profile: Profile) -> Tuple[str, ...]:
    return tuple(profile.device_classes) + tuple(_EXTRA_DEVICES.get(profile.id, ()))
