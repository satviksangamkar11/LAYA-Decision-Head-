"""Loads the validated P0 decision core (specs/evidence_ladder_v1/contract/contract_lib_v1.py) by file path. COORDINATOR-OWNED."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_contract():
    p = REPO / "specs" / "evidence_ladder_v1" / "contract" / "contract_lib_v1.py"
    spec = importlib.util.spec_from_file_location("contract_lib_v1", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["contract_lib_v1"] = m
    spec.loader.exec_module(m)
    return m


C = load_contract()
