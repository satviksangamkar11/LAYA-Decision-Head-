"""Data firewall for A5-v2 construction (spec: configs/a5_v2_spec.toml). Enforced in code, not by discipline.
load_dev() yields TRAIN rows only. cal_once() grants exactly one CAL evaluation (marker file). TEST and OOD can never be requested here."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"test", "ood"}
MARK = ROOT / "results/raw/a5v2-cal-evaluation-used.json"


def _rows(path, splits):
    assert not (set(splits) & FORBIDDEN), "test/ood are forbidden in A5-v2 construction"
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] in splits:
            yield r


def load_dev(path):
    """TRAIN rows only: the only data the sampler may be developed on."""
    return _rows(path, {"train"})


def cal_once(path, tag):
    """The one allowed CAL pass. Raises if it was ever used; records the use before returning the rows."""
    if MARK.exists():
        raise RuntimeError("the A5-v2 CAL evaluation was already used: " + MARK.read_text()[:200])
    with open(MARK, "x", encoding="utf-8") as f:
        json.dump({"used_at": datetime.now(timezone.utc).isoformat(), "tag": tag}, f)
    return _rows(path, {"cal"})


def load_cal_calibration_only(path):
    """CAL rows for temperature fitting and capture only (v3 role: calibration). Does not consume the one-time CAL evaluation marker; every use is logged."""
    LOG = ROOT / "results/raw/cal-access-log.jsonl"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "purpose": "v3 calibration/capture only"}) + "\n")
    return _rows(path, {"cal"})


def log_locked_access(purpose):
    """Every opening of the locked fresh file must call this first."""
    with open(ROOT / "results/raw/locked-access-log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "purpose": purpose}) + "\n")
