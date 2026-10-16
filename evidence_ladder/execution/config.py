"""Sandbox settings. Defaults equal the values of the existing receipted sandboxes (python: grade_sandboxed_v2.py / runner_v2.py; java: grade_sandboxed_java_v1.py).
Paths below point at the EXISTING, unmodified scripts and the public worker support files; nothing here names hidden-test data."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY_ROOT = REPO / "results" / "raw" / "codex-pilot1-20261016" / "fixed_v2"
JAVA_ROOT = REPO / "results" / "raw" / "codex-pilot1-java-20261016"
PY_GRADE_SCRIPT = PY_ROOT / "grader" / "sandbox" / "grade_sandboxed_v2.py"
PY_RUNNER = PY_ROOT / "grader" / "sandbox" / "runner_v2.py"
JAVA_GRADE_SCRIPT = JAVA_ROOT / "fixed_v1" / "grader" / "grade_sandboxed_java_v1.py"
JAVA_SUPPORT = JAVA_ROOT / "kit" / "Pilot1_Java" / "worker" / "support"      # Json.java, Runner.java (public worker files)
PY_TASKS = PY_ROOT / "worker" / "tasks.jsonl"
JAVA_TASKS = JAVA_ROOT / "kit" / "Pilot1_Java" / "worker" / "tasks.jsonl"


@dataclass(frozen=True)
class SandboxConfig:
    py_timeout: float = 10.0            # wall seconds for the whole case batch (existing default)
    java_run_timeout: float = 10.0
    java_compile_timeout: float = 60.0
    max_source_bytes: int = 131072      # existing OVERSIZED_SUBMISSION limit
    max_output_bytes: int = 4 << 20     # stdout cap; exceeding it kills the jail (invalid_output)
    max_observed_chars: int = 2000      # per observed/counterexample string
    jdk: str = os.environ.get("EL_KJAIL_JDK") or "/opt/toolchains/jdk-26.0.1"      # EL_KJAIL_JDK: the Kaggle system JDK (FC-1 jail)
    java_xmx: str = "-Xmx128m"
    # prlimit flags exactly as the existing java grader
    java_compile_prlimit: tuple = ("--nproc=256", "--fsize=4194304", "--cpu=40")
    java_run_prlimit: tuple = ("--nproc=256", "--fsize=1048576", "--nofile=256", "--cpu=30")
    # python RLIMITs (AS 1 GiB, FSIZE 1 MiB, NOFILE 64, NPROC 32, CORE 0) are fixed inside runner_v2.py and cannot change without editing that receipted file.
