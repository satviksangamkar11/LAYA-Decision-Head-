"""[EL-ARCH-1 B5] Fixed action-risk table (architecture I6, R2.3). Pure standard library. Replaces the regex deny-list of the LAYA-INT-1 driver: an action is CLASSIFIED by program/tool
(a fixed table below, never a model call), the class decides which authorizations are required, and `authorize_action` refuses before anything is executed.

Classes (weakest to strongest):  read_only < reversible_workspace_edit < irreversible | outside_workspace | network   (need user_confirmed AND a trusted pre-check)
                                  secret | protected_write | unknown                                                  (never authorized, not even with user_confirmed)
Unknown programs, unparsable commands and any shell substitution are `unknown` and DENIED. The WSL sandbox (no network, bubblewrap, protected files read-only) stays the real boundary
for code that is executed; this table classifies the SHAPE of the action so that dangerous shapes never reach the sandbox unauthorized.
Mapping to the decision core's risk enum: read_only and reversible_workspace_edit run in the sandbox ("sandboxed"); irreversible/outside/network are "irreversible" (decision.authorize then
needs every hard row VERIFIED AND user_confirmed)."""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex
from typing import Any, Dict, List, Optional, Sequence, Tuple

READ, EDIT, IRREV, OUTSIDE, NETWORK = "read_only", "reversible_workspace_edit", "irreversible", "outside_workspace", "network"
SECRET, PROT, UNKNOWN = "secret", "protected_write", "unknown"
ORDER = {READ: 0, EDIT: 1, IRREV: 2, OUTSIDE: 2, NETWORK: 2, SECRET: 3, PROT: 3, UNKNOWN: 3}
NEVER = (SECRET, PROT, UNKNOWN)
# class -> authorizations REQUIRED before execution (checked by authorize_action; "never" cannot be satisfied)
REQUIRES = {READ: [], EDIT: ["inside_workspace", "not_protected"], IRREV: ["user_confirmed", "trusted_precheck"], OUTSIDE: ["user_confirmed", "trusted_precheck"],
            NETWORK: ["user_confirmed", "trusted_precheck"], SECRET: ["never"], PROT: ["never"], UNKNOWN: ["never"]}
DECISION_RISK = {READ: "sandboxed", EDIT: "sandboxed", IRREV: "irreversible", OUTSIDE: "irreversible", NETWORK: "irreversible"}

# fixed program table: program -> class (arguments are analysed for paths / flags below)
PROGRAMS: Dict[str, str] = {}
for _p in "cat ls head tail grep egrep fgrep rg wc pwd echo printf diff sort uniq cut tr stat file which true false test expr basename dirname tree cd date".split():
    PROGRAMS[_p] = READ
for _p in "touch mkdir cp mv rm tee sed patch ln python python3 pytest".split():
    PROGRAMS[_p] = EDIT
for _p in "sudo su dd mkfs chmod chown chgrp shutdown reboot kill killall pkill truncate shred".split():
    PROGRAMS[_p] = IRREV
for _p in "curl wget ssh scp sftp nc ncat telnet ftp rsync apt apt-get npm pip pip3 yarn".split():
    PROGRAMS[_p] = NETWORK
GIT_READ = {"status", "diff", "log", "show", "blame", "branch", "ls-files", "rev-parse"}
GIT_EDIT = {"add", "commit", "restore", "mv", "rm"}
GIT_NET = {"push", "pull", "fetch", "clone", "remote"}
OPS = {";", "&&", "||", "|", "&", "\n"}
SECRET_RE = re.compile(r"(^|/)(\.secrets?|\.env(\..*)?|id_rsa\w*|id_ed25519\w*|\.ssh|\.aws|\.netrc|credentials(\.\w+)?|\.pem|\.git-credentials)(/|$)|\.(pem|key)$|api[_-]?key|token", re.I)
SECRET_WORD = re.compile(r"\.secrets?|\.env|id_rsa|id_ed25519|\.ssh|\.aws|\.netrc|credentials|\.pem|api[_-]?key|\.git-credentials", re.I)
NET_WORDS = re.compile(r"\b(urllib|requests|socket|http\.client|ftplib|smtplib|telnetlib|subprocess|os\.system|paramiko)\b|://")
SUBST = re.compile(r"\$\(|`|\$\{")

# the table is data: hashed into the journal so a change is visible (I7)
TABLE = {"classes": ORDER, "requires": REQUIRES, "decision_risk": DECISION_RISK, "programs": PROGRAMS,
         "git": {"read": sorted(GIT_READ), "edit": sorted(GIT_EDIT), "network": sorted(GIT_NET)}, "tools": {"Read": READ, "Edit": EDIT, "Bash": "by_command"}}


def table_hash() -> str:
    return hashlib.sha256(json.dumps(TABLE, sort_keys=True).encode()).hexdigest()


def _path_class(p: str, workspace: str, protected: Sequence[str]) -> Tuple[str, str]:
    """-> ("inside"|"outside"|"secret"|"protected", normalized). Paths are judged lexically (no filesystem access): '..' escapes and absolute paths outside the workspace are outside."""
    if SECRET_RE.search(p):
        return "secret", p
    if p.startswith("~") or p.startswith("$"):
        return "outside", p
    ws = workspace.replace("\\", "/").rstrip("/")
    q = p.replace("\\", "/")
    if q.startswith("/"):
        n = posixpath.normpath(q)
        if not (ws and (n == ws or n.startswith(ws + "/"))):
            return "outside", n
        q = posixpath.relpath(n, ws)
    n = posixpath.normpath(q)
    if n == ".." or n.startswith("../"):
        return "outside", n
    names = {posixpath.basename(x) for x in protected} | set(protected)
    if n in names or posixpath.basename(n) in names:
        return "protected", n
    return "inside", n


def _is_pathlike(t: str) -> bool:
    return "/" in t or t.startswith(("~", "$", ".")) or t == ".."


def _segments(command: str) -> Optional[List[List[str]]]:
    """Split a command line into segments of argv lists on control operators; redirection operators are kept as tokens. None when it cannot be parsed."""
    if not command.strip() or len(command) > 4000 or SUBST.search(command) or "\x00" in command:
        return None
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        return None
    segs, cur = [], []
    for t in toks:
        if t in OPS:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        segs.append(cur)
    return segs or None


def _classify_segment(argv: List[str], workspace: str, protected: Sequence[str]) -> Tuple[str, str]:
    """-> (class, reason) for ONE simple command."""
    args = list(argv)
    while args and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", args[0]):      # leading VAR=value assignments
        args.pop(0)
    if not args:
        return UNKNOWN, "assignment only"
    # redirections: the target is a write (or a read for <); fd duplication like 2>&1 is skipped
    cls, why = READ, "read-only program"
    plain: List[str] = []
    i = 0
    while i < len(args):
        t = args[i]
        if t in (">", ">>", ">|", "<", "<<", "&>"):
            tgt = args[i + 1] if i + 1 < len(args) else ""
            kind, _ = _path_class(tgt, workspace, protected)
            if kind == "secret":
                return SECRET, "redirection names a secret path"
            if kind == "outside" and tgt != "/dev/null":
                return OUTSIDE, "redirection outside the workspace"
            if t.startswith(">") or t == "&>":
                if kind == "protected":
                    return PROT, "redirects output into a protected file"
                if tgt != "/dev/null":
                    cls, why = EDIT, "redirects output into the workspace"
            i += 2
        elif t in (">&", "<&"):
            i += 2
        else:
            plain.append(t)
            i += 1
    if not plain:
        return cls, why
    prog = posixpath.basename(plain[0])
    rest = plain[1:]
    if prog not in PROGRAMS and prog not in ("git",):
        return UNKNOWN, "program %r is not in the action table" % prog
    base = READ if prog == "git" else PROGRAMS[prog]
    if prog == "git":
        sub = next((a for a in rest if not a.startswith("-")), "")
        if sub in GIT_READ:
            base = READ
        elif sub in GIT_EDIT:
            base = EDIT
        elif sub in GIT_NET:
            return NETWORK, "git %s uses the network" % sub
        else:
            return IRREV if sub else UNKNOWN, "git %s rewrites history / state or is unknown" % (sub or "?")
    flags = [a for a in rest if a.startswith("-")]
    if prog == "rm" and any(re.search(r"r|R", f) for f in flags if not f.startswith("--")) or prog == "rm" and "--recursive" in flags:
        return IRREV, "recursive delete"
    if prog == "sed":
        base = EDIT if any(f.startswith("-i") or f == "--in-place" for f in flags) else READ
    if prog in ("python", "python3"):
        code = []
        if "-c" in rest:
            j = rest.index("-c")
            code = rest[j + 1:j + 2]
        text = " ".join(code)
        if SECRET_WORD.search(text):
            return SECRET, "python -c touches a secret path"
        if NET_WORDS.search(text):
            return NETWORK, "python -c uses network/process modules"
        if "-m" in rest and rest[rest.index("-m") + 1:rest.index("-m") + 2] == ["pip"]:
            return NETWORK, "python -m pip"
    if prog in ("pip", "pip3") and rest[:1] == ["uninstall"]:
        return IRREV, "pip uninstall"
    if prog == "chmod" and any(a in ("777", "a+rwx") for a in rest):
        return IRREV, "chmod 777"
    cls = base if ORDER[base] > ORDER[cls] else cls
    why = "program %s" % prog if cls == base else why
    # path arguments
    skip_first = prog in ("grep", "egrep", "fgrep", "rg", "sed", "echo", "printf", "expr", "tr", "cut", "test") or (prog in ("python", "python3") and "-c" in rest)
    paths = [t for t in rest if not t.startswith("-") and _is_pathlike(t)]
    if prog in ("python", "python3", "pytest"):
        paths = [t for t in rest if not t.startswith("-") and (t.endswith(".py") or _is_pathlike(t)) and not (prog != "pytest" and "-c" in rest)]
    for p in paths:
        kind, _n = _path_class(p, workspace, protected)
        if kind == "secret":
            return SECRET, "argument %r names a secret path" % p
        if kind == "outside":
            return OUTSIDE, "argument %r is outside the workspace" % p
        if kind == "protected" and cls == EDIT and prog not in ("python", "python3", "pytest"):
            return PROT, "write to protected file %r" % p
    if cls == EDIT and prog in ("touch", "mkdir", "cp", "mv", "rm", "tee", "sed", "patch", "ln"):
        for a in rest:
            if a.startswith("-") or (skip_first and a == rest[0]):
                continue
            kind, _ = _path_class(a, workspace, protected)
            if kind == "protected":
                return PROT, "write to protected file %r" % a
            if kind == "outside" and _is_pathlike(a):
                return OUTSIDE, "argument %r is outside the workspace" % a
    # secret words anywhere in plain args (e.g. cat .secrets/key)
    for a in rest:
        if SECRET_RE.search(a) and not a.startswith("-"):
            return SECRET, "argument %r names a secret" % a
    return cls, why


def classify(action: Dict[str, Any], workspace: str = ".", protected: Sequence[str] = ()) -> Dict[str, Any]:
    """-> {"class", "reason", "tool"}. action = {"name", "arguments"}. Anything not in the fixed table is class `unknown`."""
    if not isinstance(action, dict) or not isinstance(action.get("name"), str) or not isinstance(action.get("arguments"), dict):
        return {"class": UNKNOWN, "reason": "not an action object", "tool": None}
    name, a = action["name"], action["arguments"]
    if name == "Read":
        k, _ = _path_class(str(a.get("file_path", "")), workspace, protected)
        c, why = (SECRET if k == "secret" else OUTSIDE if k == "outside" else READ), "Read of a %s path" % ("workspace" if k in ("inside", "protected") else k)
        return {"class": c, "reason": why, "tool": name}
    if name == "Edit":
        k, _ = _path_class(str(a.get("file_path", "")), workspace, protected)
        c = {"secret": SECRET, "outside": OUTSIDE, "protected": PROT}.get(k, EDIT)
        return {"class": c, "reason": "Edit of a %s path" % k, "tool": name}
    if name == "Bash":
        segs = _segments(str(a.get("command", "")))
        if segs is None:
            return {"class": UNKNOWN, "reason": "command cannot be parsed or contains substitution", "tool": name}
        worst, reason = READ, "read-only"
        for s in segs:
            c, why = _classify_segment(s, workspace, protected)
            if ORDER[c] > ORDER[worst] or (c == worst and worst == READ and reason == "read-only"):
                worst, reason = c, why
        return {"class": worst, "reason": reason, "tool": name}
    return {"class": UNKNOWN, "reason": "tool %r is not in the action table" % name, "tool": name}


def authorize_action(action: Dict[str, Any], workspace: str = ".", protected: Sequence[str] = (), user_confirmed: bool = False, trusted_precheck: bool = False) -> Dict[str, Any]:
    """The gate in front of every execution. -> {"allowed", "class", "requires", "missing", "reason", "decision_risk"}. `trusted_precheck` is True only when the caller holds L4/L5 evidence
    for exactly this action (decision.authorize on its candidate id). Classes in NEVER are refused whatever the flags say; irreversible/outside/network need BOTH flags."""
    c = classify(action, workspace, protected)
    cls = c["class"]
    req = list(REQUIRES[cls])
    have = {"user_confirmed": bool(user_confirmed), "trusted_precheck": bool(trusted_precheck), "inside_workspace": cls not in (OUTSIDE,), "not_protected": cls != PROT}
    missing = ["never"] if cls in NEVER else [r for r in req if not have.get(r, False)]
    return {"allowed": not missing, "class": cls, "requires": req, "missing": missing, "reason": c["reason"], "decision_risk": DECISION_RISK.get(cls, "irreversible"), "tool": c["tool"]}
