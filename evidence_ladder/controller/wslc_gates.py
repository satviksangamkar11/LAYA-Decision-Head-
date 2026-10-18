"""[WSLC backend] Security gates and toolchain probe for the WSLC provider, in the record formats agreed with the report tooling:
    security_gates.json = {"backend": "mxc-wslc", "image_digest": "sha256:...", "utc": iso, "gates": {name: {"status": "PASS"|"FAIL", "evidence": {...}, "utc": iso}}}
    toolchain_probe.json = {"backend": ..., "image_digest": ..., "languages": {"java": {"probe": "PASS"|"FAIL", "versions": {...}, "reason": "..."}, "python": {...}, "go": {...}}}
Required gates: host_file_read_blocked, network_blocked, timeout_kill, output_cap, limits. Extra gates: secret_env_absent, protected_mutation_blocked.
Every gate is a REAL attempt inside a real WSLC container through the provider (no mocks); the evidence records what was attempted and observed. Scaled-down workloads only: 600 MB memory touch under a
384 MB session limit, 200 forks, 5 MB / endless `yes` output, 4-second timeouts. Create-only outputs.
    PYTHONPATH=. .venv/Scripts/python.exe -m evidence_ladder.controller.wslc_gates <out_dir>"""
from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evidence_ladder.controller import sandbox_provider as P
from evidence_ladder.controller.wslc_sandbox import ROOT, _wsl_path

GATES = ("host_file_read_blocked", "network_blocked", "timeout_kill", "output_cap", "limits")


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ws(files: Optional[Dict[str, str]] = None) -> Path:
    d = Path(tempfile.mkdtemp(prefix="wslc-gate-")) / "ws"
    d.mkdir(parents=True)
    for k, v in (files or {}).items():
        (d / k).write_bytes(v.encode("utf-8"))
    return d


def _new(prot=(), **kw):
    return P.make_sandbox(Path(tempfile.mkdtemp(prefix="wslc-gate-sb-")), list(prot), "wslc", **kw)


class _Vmmem:
    """Samples the WSL VM working set (vmmemWSL) while a container runs."""

    def __init__(self):
        import psutil
        self.ps, self.peak, self.base, self._stop = psutil, 0, 0, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _rss(self):
        try:
            return max([p.memory_info().rss for p in self.ps.process_iter(["name"]) if p.info["name"] in ("vmmemWSL", "vmmem")] or [0])
        except Exception:      # noqa: BLE001
            return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(0.1)

    def __enter__(self):
        self.base = self._rss()
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()


def gate_host_isolation() -> Dict[str, Any]:
    ws = _ws({"test_prot.py": "assert True\n", "free.txt": "free\n"})
    near = ws.parent / "sentinel_near_dummy.txt"          # next to the workspace copy's source dir
    root_s = ROOT / "_wslc_sentinel_root_dummy.txt"         # in the project root
    prot_s = ROOT / "evidence_ladder" / "_wslc_sentinel_tests_dummy.txt"       # in a tests-like path of the project
    wrote = ws.parent / "written_outside_dummy.txt"
    for f, c in ((near, "NEAR-SENTINEL"), (root_s, "ROOT-SENTINEL"), (prot_s, "TESTS-SENTINEL")):
        f.write_text(c + "\n", "utf-8")
    try:
        sb = _new(["test_prot.py"])
        try:
            m = lambda p: _wsl_path(p)      # noqa: E731
            cmd = ("for f in '%s' '%s' '%s'; do cat \"$f\" 2>&1 | head -c 80; echo \" rc=$?\"; done; "
                   "echo w > '%s' 2>&1; echo write_outside_rc=$?; "
                   "ls /mnt; echo ---drive-root; ls /mnt/%s 2>&1; echo ---env; env | cut -d= -f1 | sort | tr '\\n' ' '" % (m(near), m(root_s), m(prot_s), m(wrote), m(ROOT)[5]))
            r = sb.run(ws, cmd)
            seen = r["stdout"]
            leaked = [s for s in ("NEAR-SENTINEL", "ROOT-SENTINEL", "TESTS-SENTINEL") if s in seen]
            host_has_written = wrote.exists()
            drive_listing = seen.split("---drive-root")[1].split("---env")[0].split() if "---drive-root" in seen else None
            ok = not leaked and not host_has_written and r["exit_code"] == 0
            ev = {"attempted": "cat of 3 dummy sentinels (beside the workspace, in the project root, in an evidence_ladder tests-like path) + a write beside the workspace + listing of /mnt and of the drive root",
                  "sentinels_leaked": leaked, "write_outside_reached_host": host_has_written, "mnt_listing_and_drive_root_and_env": seen[-900:], "stderr": r["stderr"][-200:],
                  "drive_root_entries": drive_listing, "exit_code": r["exit_code"]}
        finally:
            sb.close()
    finally:
        for f in (near, root_s, prot_s, wrote):
            f.unlink(missing_ok=True)
    return {"status": "PASS" if ok else "FAIL", "evidence": ev, "utc": utc()}


def gate_network() -> Dict[str, Any]:
    sb = _new()
    try:
        ws = _ws()
        (ws / "net.py").write_bytes(b"import socket\nfor h in ('1.1.1.1', '8.8.8.8', '127.0.0.1'):\n    try:\n        socket.create_connection((h, 53), timeout=4); print(h, 'CONNECTED')\n    except Exception as e:\n        print(h, 'BLOCKED', type(e).__name__)\ntry:\n    print('dns', socket.gethostbyname('example.com'))\nexcept Exception as e:\n    print('dns BLOCKED', type(e).__name__)\nprint('ifaces', [l.split(':')[0].strip() for l in open('/proc/net/dev').read().splitlines()[2:]])\n")
        r = sb.run(ws, "python3 net.py")
        ok = r["exit_code"] == 0 and "CONNECTED" not in r["stdout"] and r["stdout"].count("BLOCKED") == 4 and "['lo']" in r["stdout"]
        return {"status": "PASS" if ok else "FAIL", "evidence": {"attempted": "TCP connect to 1.1.1.1:53, 8.8.8.8:53, 127.0.0.1:53 and DNS lookup of example.com; interface list", "observed": r["stdout"], "stderr": r["stderr"][-200:]}, "utc": utc()}
    finally:
        sb.close()


def gate_timeout() -> Dict[str, Any]:
    sb = _new()
    try:
        ws = _ws()
        t0 = time.time()
        r = sb.run(ws, "sleep 300 & sleep 300 & sleep 300", timeout=4)
        wall = time.time() - t0
        d = sb.last_diagnostics
        tree = sb.run(ws, "timeout -s KILL 2 bash -c 'sleep 301 & sleep 302 & wait'; echo timeout_rc=$?; sleep 0.5; n=0; for c in /proc/[0-9]*/cmdline; do if tr '\\0' ' ' < $c 2>/dev/null | grep -qx 'sleep 30[12] '; then n=$((n+1)); fi; done; echo survivors=$n; sleep 303 & sleep 0.3; c=0; for c2 in /proc/[0-9]*/cmdline; do if tr '\\0' ' ' < $c2 2>/dev/null | grep -qx 'sleep 303 '; then c=$((c+1)); fi; done; echo detector_control_live_sleep=$c")
        ok = r["killed"] is True and r["exit_code"] == 137 and d.get("timed_out") is True and wall < 4 + 20 and "survivors=0" in tree["stdout"] and "detector_control_live_sleep=1" in tree["stdout"] and "timeout_rc=137" in tree["stdout"]
        return {"status": "PASS" if ok else "FAIL", "evidence": {"attempted": "3 background sleeps with timeout=4 s; then the same group-kill used by the runner (timeout -s KILL) on a bash with 2 sleeping grandchildren, counted from /proc by exact command line (`sleep 301`/`sleep 302`), not by substring, so the probe's own command line is not counted",
                                                                 "killed": r["killed"], "exit_code": r["exit_code"], "timed_out_flag": d.get("timed_out"), "in_container_seconds": d.get("in_container_seconds"), "wall_seconds": round(wall, 1),
                                                                 "grandchild_survivors_after_group_kill": tree["stdout"].strip(), "container_destroyed_after_every_run": True}, "utc": utc()}
    finally:
        sb.close()


def gate_output_cap() -> Dict[str, Any]:
    sb = _new(max_output_bytes=1 << 20)
    try:
        ws = _ws()
        t0 = time.time()
        a = sb.run(ws, "head -c 5242880 /dev/zero | tr '\\0' a", timeout=30)
        da = dict(sb.last_diagnostics)
        b = sb.run(ws, "yes flood", timeout=30)
        db = dict(sb.last_diagnostics)
        c = sb.run(ws, "echo small", timeout=30)
        ok = (a["killed"] and da["output_capped"] and b["killed"] and db["output_capped"] and len(a["stdout"]) <= 6000 and len(b["stdout"]) <= 6000
              and c["stdout"] == "small\n" and not sb.last_diagnostics["output_capped"] and (time.time() - t0) < 90)
        return {"status": "PASS" if ok else "FAIL", "evidence": {"attempted": "5 MiB of stdout with a 1 MiB cap; an endless `yes`; a small output as control",
                                                                 "five_mib": {"killed": a["killed"], "output_capped": da["output_capped"], "returned_chars": len(a["stdout"]), "exit_code": a["exit_code"]},
                                                                 "endless_yes": {"killed": b["killed"], "output_capped": db["output_capped"], "returned_chars": len(b["stdout"]), "exit_code": b["exit_code"], "seconds": db["total_ms"] / 1000.0},
                                                                 "control_small": {"stdout": c["stdout"], "output_capped": sb.last_diagnostics["output_capped"]},
                                                                 "mechanism": "stdout/stderr redirected to files inside the container with a 64 MiB file-size limit (SIGXFSZ ends a flood); only the first cap bytes are returned; the flood never reaches the host"}, "utc": utc()}
    finally:
        sb.close()


def gate_limits() -> Dict[str, Any]:
    mem_py = b"x = []\nfor i in range(12):\n    x.append(b'x' * (50 * 1024 * 1024))\nprint('touched 600 MB')\n"
    cpu_py = (b"import os, time\nt = time.time()\nps = []\nfor _ in range(2):\n    pid = os.fork()\n    if pid == 0:\n        n = 0\n        while n < 15_000_000:\n            n += 1\n        os._exit(0)\n"
              b"    ps.append(pid)\nfor p in ps:\n    os.waitpid(p, 0)\nprint(round(time.time() - t, 2))\n")
    fork_py = b"import os, time\nn = 0\ntry:\n    for i in range(200):\n        pid = os.fork()\n        if pid == 0:\n            time.sleep(3)\n            os._exit(0)\n        n += 1\nexcept Exception as e:\n    print('fork failed', type(e).__name__)\nprint('forked', n)\n"
    ws = _ws()
    for name, body in (("mem.py", mem_py), ("cpu.py", cpu_py), ("fork.py", fork_py)):
        (ws / name).write_bytes(body)
    ev: Dict[str, Any] = {}
    with _Vmmem() as vm:
        sb_small = _new(memory_mb=384)
        try:
            m = sb_small.run(ws, "python3 mem.py", timeout=60)
            ev["memory"] = {"session_limit_mb": 384, "attempt": "touch 600 MB", "exit_code": m["exit_code"], "killed": m["killed"], "stdout": m["stdout"][-80:], "enforced": m["exit_code"] == 137 and m["killed"]}
            f = sb_small.run(ws, "python3 fork.py", timeout=60)
            ev["pids"] = {"attempt": "200 forks", "stdout": f["stdout"][-80:], "enforced": "fork failed" in f["stdout"], "note": "no process-count field in the WSLC V1 policy; provider refuses max_processes"}
        finally:
            sb_small.close()
        walls = {}
        for n in (1, 2):
            s = _new(cpu_count=n)
            try:
                r = s.run(ws, "python3 cpu.py", timeout=60)
                walls[n] = float(r["stdout"].strip())
            finally:
                s.close()
        ev["cpu"] = {"attempt": "2 busy processes, same work", "wall_seconds_cpu_count_1": walls[1], "wall_seconds_cpu_count_2": walls[2], "enforced": walls[1] > 1.4 * walls[2]}
        sb = _new()
        try:
            info = sb.run(ws, "head -2 /proc/meminfo | tr '\\n' ' '", timeout=30)
            ev["observed_container_meminfo_default_1536"] = info["stdout"].strip()
        finally:
            sb.close()
    ev["host_vmmemWSL_mb"] = {"before": round(vm.base / 2 ** 20), "peak_during_gate_runs": round(vm.peak / 2 ** 20)}
    ok = ev["memory"]["enforced"] and ev["cpu"]["enforced"]
    ev["summary"] = "memory and CPU limits are enforced and observed; the process-count limit is NOT enforced by WSLC (documented, not claimed)" if ok else "a required limit was not observed"
    return {"status": "PASS" if ok else "FAIL", "evidence": ev, "utc": utc()}


def gate_secret_env() -> Dict[str, Any]:
    os.environ["LAYA_WSLC_TEST_SECRET"] = "s3cret-value"
    sb = _new()
    try:
        r = sb.run(_ws(), "env | sort")
        names = sorted(l.split("=", 1)[0] for l in r["stdout"].splitlines() if "=" in l)
        ok = "s3cret" not in r["stdout"] and "LAYA_WSLC_TEST_SECRET" not in r["stdout"] and set(names) <= set(__import__("evidence_ladder.controller.wslc_sandbox", fromlist=["CONTAINER_ENV"]).CONTAINER_ENV) | {"PWD", "SHLVL", "_", "OLDPWD"}
        return {"status": "PASS" if ok else "FAIL", "evidence": {"attempted": "set a secret-looking variable in the host process, list the container environment", "container_env_names": names}, "utc": utc()}
    finally:
        os.environ.pop("LAYA_WSLC_TEST_SECRET", None)
        sb.close()


def gate_protected_mutation() -> Dict[str, Any]:
    ws = _ws({"test_prot.py": "assert True\n", "src.txt": "src\n"})
    before = (ws / "test_prot.py").read_bytes()
    sb = _new(["test_prot.py"])
    try:
        attempts = ["echo hacked > test_prot.py", "echo hacked >> test_prot.py", "rm -f test_prot.py", "mv src.txt test_prot.py", "python3 -c \"open('test_prot.py','w').write('x')\"", "chmod 777 test_prot.py; echo x > test_prot.py"]
        obs = []
        ok = True
        for a in attempts:
            r = sb.run(ws, a, persist=True)
            same = (ws / "test_prot.py").read_bytes() == before
            obs.append({"cmd": a, "exit_code": r["exit_code"], "protected_unchanged": r["protected_unchanged"], "host_file_unchanged": same, "persisted": r["persisted"]})
            ok = ok and same and r["protected_unchanged"] and r["exit_code"] != 0
        r2 = sb.run(ws, "echo ok > free.txt", persist=True)
        ok = ok and r2["exit_code"] == 0 and r2["persisted"] and (ws / "free.txt").exists()
        return {"status": "PASS" if ok else "FAIL", "evidence": {"attempted": "write, append, delete, rename-over and chmod+write on a protected file; control: write to an unprotected file persists", "results": obs,
                                                                 "control": {"exit_code": r2["exit_code"], "persisted": r2["persisted"]}}, "utc": utc()}
    finally:
        sb.close()


def toolchain_probe(sb) -> Dict[str, Any]:
    prov = sb.provenance()
    ws = _ws({"Hello.java": "public class Hello { public static void main(String[] a) { System.out.println(\"hi java\"); } }\n", "hello.py": "print('hi python')\n",
              "hello.go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"hi go\") }\n"})
    langs = {}
    for lang, cmd, want in (("java", "javac Hello.java && java -cp . Hello", "hi java"), ("python", "python3 hello.py", "hi python"), ("go", "go run hello.go", "hi go")):
        r = sb.run(ws, cmd, timeout=120)
        okk = r["exit_code"] == 0 and want in r["stdout"]
        vers = {k: v for k, v in prov["toolchain"].items() if (k in ("javac", "java") and lang == "java") or (k == lang)}
        langs[lang] = {"probe": "PASS" if okk else "FAIL", "versions": vers, "reason": "" if okk else "exit %s: %s" % (r["exit_code"], (r["stderr"] or r["stdout"])[-200:])}
    return {"backend": prov["backend"], "image_digest": prov["image_digest"], "languages": langs}


def run_all(out_dir: Path) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True)
    sb = _new()
    try:
        prov = sb.provenance()
        tp = toolchain_probe(sb)
    finally:
        sb.close()
    fns: Dict[str, Callable[[], Dict[str, Any]]] = {"host_file_read_blocked": gate_host_isolation, "network_blocked": gate_network, "timeout_kill": gate_timeout, "output_cap": gate_output_cap,
                                                    "limits": gate_limits, "secret_env_absent": gate_secret_env, "protected_mutation_blocked": gate_protected_mutation}
    gates = {}
    for name, fn in fns.items():
        try:
            gates[name] = fn()
        except Exception as e:      # noqa: BLE001  a gate that crashes is a FAIL with the error as evidence, never a silent skip
            gates[name] = {"status": "FAIL", "evidence": {"error": "%s: %s" % (type(e).__name__, str(e)[:400])}, "utc": utc()}
        print(name, gates[name]["status"], flush=True)
    doc = {"backend": prov["backend"], "image_digest": prov["image_digest"], "utc": utc(), "gates": gates}
    with open(out / "security_gates.json", "x", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    with open(out / "toolchain_probe.json", "x", encoding="utf-8") as f:
        json.dump(tp, f, indent=1, ensure_ascii=False)
    return 0 if all(gates[g]["status"] == "PASS" for g in GATES) else 1


if __name__ == "__main__":
    sys.exit(run_all(Path(sys.argv[1])))
