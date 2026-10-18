"""[EL-WSLC] TOOLCHAIN REGISTRY of the bank execution engine (evidence_ladder/execution/provider_engine.py). One entry per language: file extension, version probe, compile command, run command, the runner
support files, and the runner KIND the engine implements for it. A language is RUNNABLE in a backend only if BOTH hold: (1) its probe commands pass INSIDE that backend (`probe_toolchains`), and
(2) the engine has an implemented runner / judge for it (`runner` is not None). Otherwise it is NOT_TESTED with the exact reason; nothing is ever claimed for a toolchain that is not installed and probed.

Implemented runners today: `python` (portable runner, direct `python3 runner.py <cap>` command, works on MXC and in a container) and `java` (javac + java inside the container, test input staged as a file and
redirected by a shell; verdicts through the same engine.judge_each / observe as the qualified WSL engine). c, cpp, javascript, go and rust are REGISTERED (commands + probes documented) with runner None:
they are never runnable until someone adds a runner and a judge.

HOW TO ADD A LANGUAGE: add a Toolchain entry below (probe commands must exit 0 inside the backend), implement a `_run_<lang>` in provider_engine.py that returns an engine.RunResult, set `runner="<lang>"`,
add a parity test against a qualified reference engine (tests/phaseF3d_tests.py pattern), and only then list the language in a manifest. Commands are POSIX `sh` strings; paths are relative to the workspace
(the backend runs the command with the workspace as its working directory)."""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Toolchain:
    language: str
    extension: str
    probe: Tuple[str, ...]                 # each command must exit 0 inside the backend; the first output line of the first one is recorded as the version
    compile: Optional[str]                 # template; {src} {out}; None for interpreted languages
    run: str                               # template; {src} {out}
    support_files: Tuple[str, ...]         # repo-relative files copied into the workspace next to the candidate
    runner: Optional[str]                  # implemented runner kind in provider_engine.py, None = registered only
    tool_label: str                        # NOT_TESTED_NO_<tool_label> status when the probe fails


REGISTRY: Dict[str, Toolchain] = {
    "python": Toolchain("python", ".py", ("python3 --version",), None, "python3 {src}", (), "python", "PYTHON"),
    "java": Toolchain("java", ".java", ("sh -c %s" % shlex.quote("timeout 30 javac -version 2>&1 && timeout 30 java -version 2>&1"),),
                      "javac -proc:none -encoding UTF-8 -d . {src}", "java -Xmx128m -cp . {out}",
                      ("results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker/support/Json.java", "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker/support/Runner.java"), "java", "JDK"),
    "c": Toolchain("c", ".c", ("gcc --version",), "gcc -O1 -o {out} {src}", "./{out}", (), None, "GCC"),
    "cpp": Toolchain("cpp", ".cpp", ("g++ --version",), "g++ -O1 -std=c++17 -o {out} {src}", "./{out}", (), None, "GPP"),
    "javascript": Toolchain("javascript", ".js", ("node --version",), None, "node {src}", (), None, "NODE"),
    "go": Toolchain("go", ".go", ("go version",), "go build -o {out} {src}", "./{out}", (), None, "GO"),
    "rust": Toolchain("rust", ".rs", ("rustc --version",), "rustc -O -o {out} {src}", "./{out}", (), None, "RUSTC"),
}


def get(language: str) -> Optional[Toolchain]:
    return REGISTRY.get(language)


def not_tested_status(language: str) -> str:
    tc = REGISTRY.get(language)
    return "NOT_TESTED_NO_%s" % (tc.tool_label if tc else "TOOLCHAIN")
