"""Stage-1 check registry (S3). Hashed at run start and end (invariant I7). hidden_tests_v1 is registered as grader_only / not runtime visible so that
C.Registry.runtime_lookup refuses it (S3-T4); nothing in the controller ever looks it up."""
from __future__ import annotations

import json
from typing import List

from evidence_ladder.common.contract_import import C

CHECK_VERSION = "1"


def _chk(check_id, kind, trust, covers, vis=True, grader=False, limitations=()):
    return {"check_id": check_id, "version": CHECK_VERSION, "code_hash": C.sha256(check_id + "@" + CHECK_VERSION), "kind": kind, "trust_class": trust,
            "covers": list(covers), "runtime_visible": vis, "grader_only": grader, "limitations": list(limitations)}


def check_defs(example_req_ids: List[str]) -> List[dict]:
    return [
        _chk("parse_v1", "parse", "rejecting_only", ["syntax"], limitations=["says nothing about behaviour"]),
        _chk("public_examples_v1", "public_test", "authoritative_in_scope", example_req_ids,
             limitations=["examples are few: passing them does not verify the problem"]),
        _chk("limits_v1", "public_test", "rejecting_only", ["limits"], limitations=["hardware differs from the judge's"]),
        _chk("qwen_tests_v1", "model_test", "diagnostic", [], limitations=["may share the generator's misunderstanding"]),
        _chk("critic_experiment_v1", "critic_experiment", "diagnostic", [], limitations=["the critic's expected output is not truth"]),
        _chk("behaviour_cluster_v1", "behaviour_cluster", "diagnostic", [], limitations=["agreement is not correctness"]),
        _chk("hidden_tests_v1", "hidden_test", "authoritative", [], vis=False, grader=True, limitations=["evaluation only"]),
    ]


def build_registry(example_req_ids: List[str]):
    return C.Registry(check_defs(example_req_ids))


def digest(reg) -> str:
    """Recompute the registry hash from the live check table (same formula as C.Registry.hash), so in-place mutation is detected."""
    return C.sha256(json.dumps(list(reg.checks.values()), sort_keys=True))
