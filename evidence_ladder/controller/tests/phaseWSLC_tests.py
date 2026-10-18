"""[WSLC backend] Tests of the WSLC provider (evidence_ladder/controller/wslc_sandbox.py, wslc_gates.py, sandbox_provider.py 'wslc').
Tests named *_wslc run commands in the REAL MXC WSLC container (pinned local image) on this host; they are SKIPPED AND LISTED (never counted as passed) when WSLC cannot run here. Others are pure logic.
No mock stands in for WSLC in any *_wslc test. WSLC allows ONE session at a time: do not run other WSLC work (verifier tests, smoke runs) while this suite runs.
Run from the repo root: python evidence_ladder/controller/tests/phaseWSLC_tests.py   (prints {"tests","passed","failed","skipped"}; exit 1 on failure)."""
import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evidence_ladder.controller import mxc_sandbox as M, sandbox_provider as P, wslc_gates as G, wslc_sandbox as W   # noqa: E402

TESTS, SKIPPED, _CAP = {}, {}, {}


class Skip(Exception):
    pass


def test(f):
    TESTS[f.__name__] = f
    return f


def need_wslc():
    if "c" not in _CAP:
        _CAP["c"] = W.capabilities(use_cache=True)
    if not _CAP["c"]["available"]:
        raise Skip("NOT_TESTED: WSLC unavailable: %s" % _CAP["c"]["reasons"])


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % (exc,))


def ws_of(files):
    d = Path(tempfile.mkdtemp(prefix="wslct-")) / "ws"
    d.mkdir(parents=True)
    for k, v in files.items():
        (d / k).write_bytes(v.encode("utf-8"))
    return d


def sbx(prot=(), **kw):
    return P.make_sandbox(Path(tempfile.mkdtemp(prefix="wslct-sb-")), list(prot), "wslc", **kw)


JAVA = "public class Hello { public static void main(String[] a) { System.out.println(\"hello \" + System.getProperty(\"java.version\")); } }\n"


# ====================================================================== logic only
@test
def wslc_is_an_explicit_backend_with_no_auto_and_no_fallback():
    assert "wslc" in P.BACKENDS and P.DEFAULT_BACKEND == "wsl"
    raises(M.SandboxUnavailable, P.selected_backend, "auto")
    assert P.selected_backend("wslc") == "wslc"
    # a host that cannot run it raises, it never switches backend (negative control: a node binary that does not exist)
    c = W.capabilities(node="Z:\\no\\such\\node.exe")
    assert not c["available"] and c["reasons"]
    raises(M.SandboxUnavailable, W.WslcSandbox, Path(tempfile.mkdtemp()), [], node="Z:\\no\\such\\node.exe")


@test
def unenforceable_or_invalid_policy_is_rejected():
    raises(M.UnsupportedPolicy, W.WslcSandbox, Path(tempfile.mkdtemp()), [], max_processes=256, check_capabilities=False)       # no pids field in WSLC V1: refuse, never ignore
    for kw in ({"cpu_count": 0}, {"memory_mb": -1}, {"max_output_bytes": True}):
        raises(M.UnsupportedPolicy, W.WslcSandbox, Path(tempfile.mkdtemp()), [], check_capabilities=False, **kw)
    raises(M.UnsupportedPolicy, W.WslcSandbox, Path(tempfile.mkdtemp()), ["../x"], check_capabilities=False)
    raises(M.UnsupportedPolicy, W.WslcSandbox, Path(tempfile.mkdtemp()), ["C:\\abs.py"], check_capabilities=False)


@test
def image_pin_is_checked_and_a_tampered_tar_is_refused():
    cfg = W.load_config()
    assert cfg["image"]["sha256"] and len(cfg["image"]["sha256"]) == 64 and cfg["image"]["tar"].endswith(".tar")
    if (ROOT / cfg["image"]["tar"]).is_file():
        info = W.verify_image(cfg)
        assert info["image_digest"] == "sha256:" + cfg["image"]["sha256"] and info["bytes"] == cfg["image"]["bytes"]
        bad = json.loads(json.dumps(cfg))
        bad["image"]["sha256"] = "0" * 64
        raises(M.SandboxUnavailable, W.verify_image, bad)                  # same tar, different pin -> refused
    gone = json.loads(json.dumps(cfg))
    gone["image"]["tar"] = "models/wslc-image/does_not_exist.tar"
    raises(M.SandboxUnavailable, W.verify_image, gone)


@test
def runner_metadata_scrub_and_contract_helpers():
    m = W.META_RE.search("some stderr\n@@LAYA_META rc=137 timed_out=1 capped=0 elapsed=4@@\n")
    assert m and m.groups() == ("137", "1", "0", "4") and W.META_RE.search("no meta here") is None
    ws, job = Path(tempfile.mkdtemp()) / "ws", Path(tempfile.mkdtemp()) / "job"
    txt = W._norm('File "%s/x.py", line 1\r\n%s' % (W._wsl_path(ws), W._wsl_path(job)), ws, job)
    assert txt == 'File "/work/x.py", line 1\n/job'
    assert W.RUNNER_SHA256 == hashlib.sha256(W.RUNNER.encode()).hexdigest() and "ulimit -f" in W.RUNNER and "timeout -s KILL" in W.RUNNER
    assert W.ENV_LABEL == "mxc-wslc" and W.WslcSandbox.env_label == "mxc-wslc"


# ====================================================================== contract, toolchains, failure kinds on the real WSLC
@test
def python_java_go_hello_and_provenance_wslc():
    need_wslc()
    ws = ws_of({"hello.py": "import sys; print('py', sys.version.split()[0])\n", "Hello.java": JAVA, "hello.go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"go ok\") }\n"})
    with sbx() as sb:
        a = P.validate_result(sb.run(ws, "python3 hello.py"))
        b = P.validate_result(sb.run(ws, "javac Hello.java && java -cp . Hello"))
        c = P.validate_result(sb.run(ws, "go run hello.go"))
        assert a["stdout"] == "py 3.13.11\n" and b["stdout"] == "hello 26.0.1\n" and c["stdout"] == "go ok\n"
        assert all(x["exit_code"] == 0 and not x["killed"] and x["stderr"] == "" and x["protected_unchanged"] for x in (a, b, c))
        prov = sb.provenance()
        assert prov["backend"] == "mxc-wslc" and prov["image_digest"] == "sha256:" + W.load_config()["image"]["sha256"]
        assert prov["toolchain"]["javac"] == "javac 26.0.1" and "26.0.1" in prov["toolchain"]["java"] and prov["toolchain"]["python"] == "Python 3.13.11" and "go1.26.4" in prov["toolchain"]["go"]
        d = sb.last_diagnostics
        assert d["backend"] == "mxc-wslc" and d["image_digest"] == prov["image_digest"] and d["memory_mb"] == 1536 and "wslc_call_ms" in d and "diagnostics" not in a


@test
def candidate_failures_are_results_and_infrastructure_failures_raise_wslc():
    need_wslc()
    ws = ws_of({"Bad.java": "public class Bad { void f( }\n", "Boom.java": "public class Boom { public static void main(String[] a) { throw new RuntimeException(\"boom\"); } }\n",
                "Wrong.java": "public class Wrong { public static void main(String[] a) { System.out.println(41); } }\n", "crash.py": "import os\nos.abort()\n", "t.py": "assert 1 == 2, 'bad output'\n"})
    with sbx() as sb:
        compile_err = sb.run(ws, "javac Bad.java")
        runtime = sb.run(ws, "javac Boom.java && java -cp . Boom")
        wrong = sb.run(ws, "javac Wrong.java && java -cp . Wrong")
        crash = sb.run(ws, "python3 crash.py")
        assertion = sb.run(ws, "python3 t.py")
        assert compile_err["exit_code"] == 1 and "error" in compile_err["stdout"] + compile_err["stderr"] and not compile_err["killed"]      # compile failure (candidate)
        assert runtime["exit_code"] == 1 and "RuntimeException: boom" in runtime["stderr"]                              # runtime exception (candidate)
        assert wrong["exit_code"] == 0 and wrong["stdout"] == "41\n"                                                    # bad output: a normal result the oracle must judge
        assert crash["exit_code"] == 134 and crash["killed"] is False                                                   # abort(): SIGABRT, not a sandbox failure
        assert assertion["exit_code"] == 1 and "AssertionError: bad output" in assertion["stderr"]
        # infrastructure failures raise
        raises(ValueError, sb.run, ws, "")
        raises(ValueError, sb.run, ws, "echo x", timeout=0)
        raises(M.SandboxIntegrityError, sb.run, ws / "missing", "echo x")
    # the image tar going missing/unpinned is an infrastructure error, never a result
    bad = W.load_config()
    bad["image"]["tar"] = "models/wslc-image/does_not_exist.tar"
    raises(M.SandboxUnavailable, W.verify_image, bad)


@test
def determinism_and_tree_hashes_wslc():
    need_wslc()
    ws = ws_of({"a.py": "print('x')\n", "test_a.py": "import a\n"})
    with sbx(["test_a.py"]) as sb:
        r1, r2 = sb.run(ws, "python3 a.py"), sb.run(ws, "python3 a.py")
        assert r1 == r2 and r1["pre_tree_sha256"] == r1["post_tree_sha256"] and len(r1["pre_tree_sha256"]) == 64
        w = sb.run(ws, "echo new > n.txt")
        assert w["pre_tree_sha256"] != w["post_tree_sha256"] and not (ws / "n.txt").exists()                      # copy-in/copy-out: the original is untouched
        h = hashlib.sha256()
        for p in sorted(x for x in ws.rglob("*") if x.is_file()):
            h.update(str(p.relative_to(ws)).replace("\\", "/").encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
        assert w["pre_tree_sha256"] == h.hexdigest()                                                                  # same algorithm as the WSL helper


@test
def persistence_and_rollback_wslc():
    need_wslc()
    ws = ws_of({"a.txt": "a\n", "test_a.py": "t\n"})
    before = {p.name: p.read_bytes() for p in ws.iterdir()}
    with sbx(["test_a.py"]) as sb:
        r = sb.run(ws, "echo scratch > s.txt", persist=False)
        assert not r["persisted"] and {p.name: p.read_bytes() for p in ws.iterdir()} == before
        r = sb.run(ws, "echo kept > k.txt; rm a.txt", persist=True)
        assert r["persisted"] and (ws / "k.txt").read_bytes().strip() == b"kept" and not (ws / "a.txt").exists() and (ws / "test_a.py").read_bytes() == before["test_a.py"]
        r = sb.run(ws, "echo late > late.txt; exit 4", persist=True)
        assert r["exit_code"] == 4 and r["persisted"] and (ws / "late.txt").exists()             # same rule as the WSL helper: a failing command still persists when not killed
        r = sb.run(ws, "echo timeout_victim > tv.txt; sleep 30", persist=True, timeout=3)
        assert r["killed"] and not r["persisted"] and not (ws / "tv.txt").exists()               # a killed run is never persisted


@test
def links_are_refused_and_workload_links_are_not_persisted_wslc():
    need_wslc()
    ws = ws_of({"a.txt": "a\n"})
    target = ws.parent / "hostfile.txt"
    target.write_text("HOST SECRET")
    import subprocess
    made = False
    try:
        os.symlink(target, ws / "link.txt")
        made = True
    except OSError:      # unprivileged Windows: use a junction (a reparse point, no privilege needed)
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(ws / "link.txt"), str(ws.parent)], capture_output=True, text=True).returncode == 0
    if not made:
        raise Skip("NOT_TESTED: this account cannot create a symlink or junction")
    with sbx() as sb:
        raises(M.SandboxIntegrityError, sb.run, ws, "echo hi")
        (os.rmdir if (ws / "link.txt").is_dir() else os.unlink)(ws / "link.txt")
        r = sb.run(ws, "ln -s /etc/passwd made_link; echo done", persist=True)
        assert not (ws / "made_link").exists() and r["persisted"] is False
        assert "links_created" in sb.last_diagnostics and sb.last_diagnostics["persist_refused"] == "links_created"


@test
def all_security_gates_pass_wslc():
    need_wslc()
    out = {}
    for name, fn in (("host_file_read_blocked", G.gate_host_isolation), ("network_blocked", G.gate_network), ("timeout_kill", G.gate_timeout), ("output_cap", G.gate_output_cap),
                     ("limits", G.gate_limits), ("secret_env_absent", G.gate_secret_env), ("protected_mutation_blocked", G.gate_protected_mutation)):
        g = fn()
        out[name] = g["status"]
        assert g["status"] == "PASS" and g["evidence"], (name, json.dumps(g)[:500])
    assert set(G.GATES) <= set(out)


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
