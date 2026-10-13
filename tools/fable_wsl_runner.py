"""OS-isolated replay runner for trace-reconstructed Fable decision states (WSL Ubuntu 24.04, run as root by the orchestrator;
trace code runs as the unprivileged `replay` user). Toolchains: /opt/toolchains (results/raw/wsl-toolchains-20261010/MANIFEST.md).

Per run: fresh copy of <state>/workspace in /var/lib/fable-runs/<id>/ws (owned by replay); systemd scope (MemoryMax 2G,
MemorySwapMax 0, TasksMax 256, CPUQuota 400%); bubblewrap with every namespace unshared (empty network namespace), only /usr,
/etc and /opt/toolchains read-only, the workspace read-write at /work, private /tmp, no /mnt /root /home; protected test files
bind-mounted read-only; cleared environment; hard KILL at the time limit. The recorded test command is run with bash -c.

  qualify   isolation probes (stop if any fails), then the untouched baseline twice for the proof decisions
            (3 per language, smallest sha256 of the decision id: outcome-blind) -> results/raw/fable-wsl-runner-qualification-<ts>/

Run (from Windows): wsl -d Ubuntu-24.04 -u root -- python3 "/mnt/d/local model/tools/fable_wsl_runner.py" qualify
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/d/local model")
RUNS = Path("/var/lib/fable-runs")
TC = "/opt/toolchains"
# 2026-10-13 (user: fix Go replay): GOPROXY points at a read-only, offline module cache under /opt/toolchains/gomodcache (gopkg.in/yaml.v3 v3.0.1,
# h1:fxVm/GzAzEWqLHuvctI91KS9hhNmmWOoWu0XTYJS7CA=, verified against sum.golang.org when downloaded); GOSUMDB off because the sandbox has no network.
# 2026-10-13 (user: fix C# / Rust): .NET SDK 10.0.302 and Rust 1.96.0 (cargo) in /opt/toolchains (official archives, sha512 / sha256 verified); NuGet packages
# restored by a dummy project into the read-only fallback folder /opt/toolchains/nuget-packages; NuGetAudit off (it needs network).
TIMEOUT = 300
ENV = {"PATH": TC + "/bin:/usr/bin:/bin", "HOME": "/tmp/home", "TMPDIR": "/tmp", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
       "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "JAVA_HOME": TC + "/jdk-26.0.1",
       "GOCACHE": "/tmp/gocache", "GOPATH": "/tmp/gopath", "GOPROXY": "file:///opt/toolchains/gomodcache/cache/download", "GOSUMDB": "off", "GOTOOLCHAIN": "local", "GOTELEMETRY": "off", "GOFLAGS": "-mod=mod",
       "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "NUGET_PACKAGES": "/tmp/nuget", "RestoreAdditionalProjectFallbackFolders": "/opt/toolchains/nuget-packages", "NuGetAudit": "false", "POWERSHELL_TELEMETRY_OPTOUT": "1", "NO_COLOR": "1", "CI": "1"}
TEST_RE = re.compile(r"(^|/)(test|tests|spec)[^/]*$|_test\.|\.test\.|\.spec\.|test_|Test[A-Z]?\w*\.(java|cs)$|\.Tests?\.ps1$", re.I)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def tree_hash(d):
    return sha(json.dumps({str(f.relative_to(d)): sha(f.read_bytes()) for f in sorted(d.rglob("*")) if f.is_file()}, sort_keys=True))


def sandbox_argv(ws, cmd, protected, timeout, tmp=None, state=None):
    argv = ["systemd-run", "--scope", "--quiet", "--collect", "-p", "MemoryMax=2G", "-p", "MemorySwapMax=0", "-p", "TasksMax=256",
            "-p", "CPUQuota=400%", "--uid=replay", "--gid=replay", "--",
            "timeout", "-s", "KILL", str(timeout),
            "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--uid", "999", "--gid", "999",
            "--ro-bind", "/usr", "/usr", "--tmpfs", "/usr/lib/wsl", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc", "--ro-bind", TC, TC,
            "--bind", str(ws), "/work", *(["--bind", str(tmp), "/tmp"] if tmp else ["--tmpfs", "/tmp", "--dir", "/tmp/home"]),
            *(["--bind", str(state), "/state"] if state else []), "--proc", "/proc", "--dev", "/dev", "--chdir", "/work", "--clearenv"]
    for p in protected:
        argv += ["--ro-bind", str(ws / p), "/work/" + p]
    argv += ["--remount-ro", "/usr/lib/wsl", "--remount-ro", "/"]  # the WSL mask and the sandbox root (bwrap tmpfs) read-only too; only /work, /tmp, /state stay writable
    for k, v in ENV.items():
        argv += ["--setenv", k, v]
    return argv + ["/bin/bash", "-c", cmd]


def run_once(workspace, cmd, protected=(), timeout=TIMEOUT, prepare=None):
    run = RUNS / uuid.uuid4().hex
    ws = run / "ws"
    shutil.copytree(workspace, ws)
    if prepare:
        prepare(ws)
    protected = [p for p in protected if (ws / p).is_file()]
    before = {p: sha((ws / p).read_bytes()) for p in protected}
    pre = tree_hash(ws)
    subprocess.run(["chown", "-R", "replay:replay", str(run)], check=True)
    t0 = time.monotonic()
    p = subprocess.run(sandbox_argv(ws, cmd, protected, timeout), capture_output=True)
    secs = round(time.monotonic() - t0, 3)
    after = {q: sha((ws / q).read_bytes()) for q in protected}
    res = dict(exit_code=p.returncode, killed=p.returncode in (137, -9), seconds=secs, stdout=p.stdout.decode("utf-8", "replace")[-20000:],
               stderr=p.stderr.decode("utf-8", "replace")[-20000:], pre_state_sha256=pre, protected=sorted(protected),
               protected_unchanged=before == after, post_state_sha256=tree_hash(ws))
    shutil.rmtree(run, ignore_errors=True)
    return res


# (name, command, expectation) - expectation: "fail" = must not succeed, "ok" = must succeed, or a callable on the result
PROBES = [
    ("network_tcp", "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 80), timeout=5)\"", "fail"),
    ("network_curl", "curl -sS -m 5 http://example.com -o /dev/null", "fail"),
    ("dns", "getent hosts example.com", "fail"),
    ("windows_drive_hidden", "ls /mnt/d", "fail"),
    ("secrets_unreachable", "cat '/mnt/d/local model/.secrets/nvidia_api_key_2.txt'", "fail"),
    ("root_home_hidden", "ls /root", "fail"),
    ("write_toolchains", "touch /opt/toolchains/x", "fail"),
    ("write_usr", "touch /usr/x", "fail"),
    ("write_etc", "touch /etc/x", "fail"),
    ("protected_test_readonly", "echo x >> test_protected.py", "fail"),
    ("sudo", "sudo -n true", "fail"),
    ("not_root", "test \"$(id -u)\" != 0", "ok"),
    ("env_has_no_secrets", "! env | grep -iE 'key|token|secret|nvidia|password'", "ok"),
    ("memory_cap", "python3 -c \"b = bytearray(3 * 1024**3); print(len(b))\"", "fail"),
    ("process_cap", "python3 -c \"import subprocess as s; ps=[s.Popen(['sleep','30']) for _ in range(400)]; print(len(ps))\"", "fail"),
    ("timeout_kill", "while :; do :; done", lambda r: r["killed"]),
    ("workspace_writable", "echo ok > out.txt && cat out.txt", "ok"),
    ("toolchains", "python3 --version && node --version && go version && java -version 2>&1 | head -1 && pwsh -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'", "ok"),
]


def qualify():
    out = ROOT / ("results/raw/fable-wsl-runner-qualification-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    src = RUNS / ("probe-src-" + uuid.uuid4().hex)
    src.mkdir()
    (src / "test_protected.py").write_text("print('protected')\n")
    probes = {}
    for name, cmd, expect in PROBES:
        r = run_once(src, cmd, protected=["test_protected.py"], timeout=10 if name == "timeout_kill" else 60)
        ok = expect(r) if callable(expect) else (r["exit_code"] == 0) == (expect == "ok")
        probes[name] = dict(qualified=bool(ok), exit_code=r["exit_code"], killed=r["killed"], stdout=r["stdout"][-200:], stderr=r["stderr"][-200:])
    shutil.rmtree(src, ignore_errors=True)
    qualified = all(p["qualified"] for p in probes.values())
    summary = dict(sandbox_qualified=qualified, probes=probes, isolation="systemd scope (cgroup limits) + bubblewrap namespaces, unprivileged uid 999, no network namespace devices")
    if not qualified:
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(probes, indent=1))
        sys.exit("sandbox NOT qualified: baseline not run")
    states_dir = sorted(ROOT.glob("results/raw/fable-replay-states-go-py-ts-strict-*"))[-1]
    decs = [json.loads(p.read_text("utf-8")) for p in sorted(states_dir.glob("*/decision.json"))]
    proof = []
    for lang in ("py", "ts", "go"):
        proof += sorted((d for d in decs if d["lang"] == lang), key=lambda d: sha(d["decision_id"]))[:3]
    base = []
    for d in proof:
        cmd = (d["later_test_commands"] or [None])[0]
        prot = [p for p in d["files"] if TEST_RE.search(p)]
        runs = [run_once(states_dir / d["key"] / "workspace", cmd, protected=prot) for _ in range(2)]
        same = runs[0]["exit_code"] == runs[1]["exit_code"] and sha(runs[0]["stdout"]) == sha(runs[1]["stdout"])
        status = ("timeout" if any(r["killed"] for r in runs) else "test_file_changed" if not all(r["protected_unchanged"] for r in runs)
                  else "deterministic" if same else "nondeterministic")
        base.append(dict(decision_id=d["decision_id"], lang=d["lang"], key=d["key"], test_command=cmd, protected=prot, status=status,
                         exit_codes=[r["exit_code"] for r in runs], seconds=[r["seconds"] for r in runs],
                         stdout_tail=runs[0]["stdout"][-600:], stderr_tail=runs[0]["stderr"][-600:]))
    summary.update(states_dir=str(states_dir.relative_to(ROOT)), proof_decisions=len(base), baseline_status=dict(Counter(b["status"] for b in base)),
                   baseline_by_lang={l: dict(Counter((b["status"], b["exit_codes"][0]) for b in base if b["lang"] == l).most_common()) for l in ("py", "ts", "go")})
    summary["baseline_by_lang"] = {l: {"%s/exit %s" % k: v for k, v in c.items()} for l, c in summary["baseline_by_lang"].items()}
    (out / "baseline.jsonl").write_text("".join(json.dumps(b) + "\n" for b in base))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "probes"}, indent=1))
    print("probes:", {k: v["qualified"] for k, v in probes.items()})
    print("written:", out.relative_to(ROOT))


# ------------------------------------------------------------------ trace-fidelity gate (language-independent)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SIG = re.compile(r"(?i)(\bpass(ed|es)?\b|\bfail(ed|s|ures?)?\b|\bok\b|\berror|\btests?\b|\bchecks?\b|panic|exception|✔|✖|✓|✗)")
WRAP = 'cd "$(cat /state/cwd)" 2>/dev/null || cd /work\nexec 2>&1\n{ %s\n}\n__rc=$?\npwd > /state/cwd\nexit $__rc\n'


def normalize(text):
    t = ANSI.sub("", (text or "").replace("\r", ""))
    t = re.sub(r"\b\d+(\.\d+)?\s?(ms|µs|us|ns|s|sec|seconds)\b", "<T>", t)
    t = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", t)
    t = re.sub(r"(?:/[\w.@+~-]+){2,}/?", "<PATH>", t)
    t = t.replace("(cached)", "")
    return [re.sub(r"[ \t]+", " ", l).strip() for l in t.split("\n") if l.strip()]


def signature(lines):
    return [l for l in lines if SIG.search(l)]


def compare(ours_exit, ours_out, rec_exit, rec_out):
    a, b = normalize(ours_out), normalize(rec_out)
    if re.search(r"truncated|\.\.\. \[\d+ (lines|chars)", rec_out or ""):
        a, b = a[:20], b[:20]
    exact = a == b
    sig = bool(signature(b)) and signature(a) == signature(b)
    return dict(exit_match=ours_exit == rec_exit, exact=exact, signature=sig, match=ours_exit == rec_exit and (exact or sig))


def compare_v2(ours_exit, ours_out, rec_exit, rec_out):
    """Fidelity rule v2 (added 2026-10-13, AFTER seeing C# / Rust replays; reported next to v1, never replacing it). Two differences from v1, both
    formatting only: (1) a token containing a normalised path collapses to <PATH> (a recorded relative `bin/Debug/x.dll` became `bin<PATH>`, ours
    absolute `/work/bin/...` became `<PATH>`); (2) pass/fail signature lines are compared as a multiset, because parallel test runners (cargo test)
    print the same results in a different order. Exit codes must still be equal."""
    col = lambda ls: [re.sub(r"\S*<PATH>\S*", "<PATH>", l) for l in ls]
    a, b = col(normalize(ours_out)), col(normalize(rec_out))
    if re.search(r"truncated|\.\.\. \[\d+ (lines|chars)", rec_out or ""):
        a, b = a[:20], b[:20]
    sig = bool(signature(b)) and sorted(signature(a)) == sorted(signature(b))
    return dict(exit_match=ours_exit == rec_exit, exact=a == b, signature=sig, match=ours_exit == rec_exit and (a == b or sig))


def _safe_rel(p):
    return bool(p) and not p.startswith("/") and ".." not in Path(p).parts


def replay(dec_row, workspace, timeout=TIMEOUT):
    """Replay the recorded steps from the reconstructed state in ONE persistent sandbox workspace (cwd and /tmp carried
    across shell steps, like the agent's shell). No pipefail: the replay must behave exactly like the original shell."""
    run = RUNS / uuid.uuid4().hex
    ws, tmp, state = run / "ws", run / "tmp", run / "state"
    shutil.copytree(workspace, ws)
    (tmp / "home").mkdir(parents=True)
    state.mkdir()
    (state / "cwd").write_text("/work\n")
    log, verdict = [], None
    for st in dec_row["steps"]:
        eff, a = st["effect"], st["arguments"] or {}
        if eff in ("write", "edit"):
            rel = re.sub(r"^\./", "", (a.get("file_path") or "").replace("\\", "/"))
            if not _safe_rel(rel):
                verdict = "UNREPLAYABLE:path"
                break
            f = ws / rel
            if eff == "write":
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(a.get("content") or "", encoding="utf-8", newline="")
                continue
            if not f.is_file():
                verdict = "MISMATCH:edit_target_missing"
                log.append(dict(step="Edit", file=rel))
                break
            with open(f, encoding="utf-8", newline="") as h:  # Path.read_text has no newline= before Python 3.13
                txt = h.read()
            o, n = a["old_string"], a["new_string"]
            k = txt.count(o)
            if k == 0 or (k > 1 and not a.get("replace_all")):
                verdict = "MISMATCH:edit_not_applicable"
                log.append(dict(step="Edit", file=rel, occurrences=k))
                break
            f.write_text(txt.replace(o, n) if a.get("replace_all") else txt.replace(o, n, 1), encoding="utf-8", newline="")
        elif eff == "shell":
            subprocess.run(["chown", "-R", "replay:replay", str(run)], check=True)
            cmd = a.get("command", "")
            p = subprocess.run(sandbox_argv(ws, WRAP % cmd, (), timeout, tmp=tmp, state=state), capture_output=True)
            out = p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")
            c = compare(p.returncode, out, st["recorded_exit"], st["recorded_output"])
            log.append(dict(step="Bash", is_test=st.get("is_test"), command=cmd[:300], ours_exit=p.returncode, recorded_exit=st["recorded_exit"],
                            ours_tail=out[-800:], recorded_tail=(st["recorded_output"] or "")[-800:], **c))
            if st.get("is_test"):
                verdict = "MATCH" if c["match"] else ("MISMATCH:exit" if not c["exit_match"] else "MISMATCH:output")
        elif eff.startswith("unreplayable"):
            verdict = "UNREPLAYABLE:" + eff.split(":", 1)[1]
            break
    shutil.rmtree(run, ignore_errors=True)
    shell_steps = [l for l in log if l.get("step") == "Bash" and not l.get("is_test")]
    return dict(verdict=verdict or "UNREPLAYABLE:no_test_reached", intermediate_shell_mismatches=sum(not l["match"] for l in shell_steps), log=log)


def fidelity(steps_dir):
    import gzip
    from concurrent.futures import ThreadPoolExecutor
    steps_dir = ROOT / steps_dir
    rows = [json.loads(l) for l in gzip.open(steps_dir / "steps.jsonl.gz", "rt", encoding="utf-8")]
    states_dir = ROOT / json.loads((steps_dir / "summary.json").read_text())["states_dir"].replace("\\", "/")
    out = ROOT / ("results/raw/fable-fidelity-gate-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    sink = open(out / "rows.jsonl", "a", encoding="utf-8")

    def one(r):
        try:
            res = replay(r, states_dir / r["key"] / "workspace")
        except Exception as e:  # one broken decision must not stop the run
            res = dict(verdict="ERROR:%s" % type(e).__name__, intermediate_shell_mismatches=None, log=[dict(error=str(e)[:500])])
        row = dict(decision_id=r["decision_id"], lang=r["lang"], key=r["key"], n_steps=r["n_steps"], **res)
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()  # every finished decision is on disk at once
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        done = list(pool.map(one, rows))
    sink.close()
    by = {}
    for r in done:
        by.setdefault(r["lang"], Counter())[r["verdict"]] += 1
    summary = dict(steps_dir=str(steps_dir.relative_to(ROOT)), states_dir=str(states_dir.relative_to(ROOT)), decisions=len(done),
                   verdicts=dict(Counter(r["verdict"] for r in done)), by_lang={k: dict(v) for k, v in by.items()},
                   match_with_clean_intermediate_steps=sum(r["verdict"] == "MATCH" and r["intermediate_shell_mismatches"] == 0 for r in done),
                   rule="MATCH = test command exit code equal to the trace AND (normalized output equal OR pass/fail signature lines equal)")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    if os.geteuid() != 0 or not sys.argv[1:] or sys.argv[1] not in ("qualify", "fidelity"):
        sys.exit("run as root in WSL: python3 fable_wsl_runner.py qualify | fidelity <steps_dir>")
    qualify() if sys.argv[1] == "qualify" else fidelity(sys.argv[2])
