"""Deterministic hard policy (runbook 22 and 23): authoritative over every learned score.

ponytail: pattern lists, not a shell parser. They catch what they list and nothing else, so this layer is paired with a
sandbox and the protected-path rules; extend the lists when a miss is found, never loosen them to make a test pass.
"""
import os
import re
from typing import NamedTuple

from state.schema import NONE_ID

DESTRUCTIVE = [r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+(/|~|\*|\.\s|\.$)", r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r",
               r"\bgit\s+reset\s+--hard", r"\bgit\s+clean\s+-[a-z]*f", r"\bgit\s+push\s+.*(--force|-f\b)",
               r"\bgit\s+(checkout|restore)\s+(--\s+)?\.\s*$", r"\bdrop\s+(table|database|column)\b", r"\btruncate\s+table\b",
               r"\bmkfs\b", r"\bdd\s+if=.*\bof=/dev/", r":\(\)\s*\{\s*:\|:&\s*\};:", r"\bchmod\s+-R\s+777\s+/",
               r"\bformat\s+[a-z]:", r"\bdel\s+/[sq]\b", r"\bRemove-Item\b.*-Recurse.*-Force", r"\bshutdown\b|\breboot\b"]
SECRET_FILES = r"(\.env\b|id_rsa|id_ed25519|\.aws/credentials|\.ssh/|\.netrc|\.npmrc|secrets?\.(json|ya?ml|toml)|\.pypirc)"
NETWORK = r"\b(curl|wget|nc|ncat|scp|sftp|ftp|Invoke-WebRequest|iwr|Invoke-RestMethod)\b"
ENV_DUMP = r"\b(printenv|env|set)\b\s*(\||>)|\$\{?[A-Z_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*\}?"
WRITES = r"(>>?|\btee\b|\bsed\s+-i|\bmv\b|\bcp\b|\brm\b|\bgit\s+(commit|apply|checkout|merge)\b|\bchmod\b|\bmkdir\b|\btouch\b)"
WRITE_TOOLS = {"edit", "write_file", "apply_patch", "create_file", "delete_file"}


class Verdict(NamedTuple):
    allowed: bool
    reasons: tuple = ()


def _paths(c):
    out = []
    for src in (c.target, c.arguments):
        for k in ("file", "path", "target_file", "dest"):
            if isinstance(src.get(k), str):
                out.append(src[k])
    return out


class HardPolicy:
    def __init__(self, repo_root, tools, protected=("tests/", "evaluator/"), existing_tests=frozenset()):
        self.root = os.path.realpath(repo_root)
        self.tools = set(tools)
        self.protected = tuple(p.rstrip("/") + "/" for p in protected)
        self.existing_tests = {t.replace("\\", "/") for t in existing_tests}

    def _inside(self, p):
        full = os.path.realpath(os.path.join(self.root, p))
        inside = os.path.normcase(full) == os.path.normcase(self.root) or os.path.normcase(full).startswith(os.path.normcase(self.root) + os.sep)
        return inside, (os.path.relpath(full, self.root).replace("\\", "/") if inside else p)

    def _protected(self, rel):
        return rel in self.existing_tests or any(rel.startswith(p) for p in self.protected)

    def check(self, c):
        if c.candidate_id == NONE_ID:
            return Verdict(True)
        why = list(c.problems())
        if c.action_type in ("tool_call", "edit") and c.tool not in self.tools:
            why.append(f"unknown tool {c.tool!r}")
        if c.safety_class == "DESTRUCTIVE":
            why.append("destructive safety class is never executed automatically")
        writes = c.tool in WRITE_TOOLS or c.action_type == "edit"
        for p in _paths(c):
            ok, rel = self._inside(p)
            if not ok:
                why.append(f"path outside the repository: {p}")
            elif writes and self._protected(rel):
                why.append(f"write to protected path: {rel}")
        cmd = c.arguments.get("cmd", "") if c.tool == "shell" else ""
        if cmd:
            low = cmd.lower()
            if any(re.search(p, low if p.islower() else cmd, re.I) for p in DESTRUCTIVE):
                why.append("destructive shell pattern")
            if re.search(SECRET_FILES, cmd, re.I) and re.search(NETWORK, cmd, re.I):
                why.append("secret file read combined with a network tool")
            if re.search(ENV_DUMP, cmd) and re.search(NETWORK, cmd, re.I):
                why.append("environment or secret variable sent over the network")
            if re.search(WRITES, cmd) and any(re.search(re.escape(p), cmd) for p in self.protected):
                why.append("shell write touching a protected path")
            if c.safety_class == "READ_ONLY" and (re.search(WRITES, cmd) or re.search(NETWORK, cmd, re.I)):
                why.append("declared READ_ONLY but the command writes or uses the network")
        return Verdict(not why, tuple(why))

    def filter(self, cands):
        """(allowed, denied) where denied is [(candidate, verdict)]. NONE always survives."""
        allowed, denied = [], []
        for c in cands:
            v = self.check(c)
            (allowed.append(c) if v.allowed else denied.append((c, v)))
        return allowed, denied


def may_stop(state, require_tests=True):
    """Completion invariants: a learned completion score never overrides these (runbook 22, CLAUDE.md evidence order)."""
    why = []
    if any(t.get("status") != "pass" for t in state.tests):
        why.append("a test is failing or errored")
    if require_tests and not state.tests:
        why.append("no test has been run")
    if state.compiler.get("errors"):
        why.append("compiler errors present")
    if state.pending_actions:
        why.append("pending actions remain")
    if any(not a.get("met") for a in state.acceptance):
        why.append("an acceptance criterion is unmet")
    return Verdict(not why, tuple(why))
