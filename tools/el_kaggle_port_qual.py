"""[P5 Kaggle] Qualification of the Kaggle jail PORT for the trusted verifier, run INSIDE the Kaggle kernel before any generation (fail closed).
Checks (python): honest candidate passes; forged result frames / claim-then-crash never pass; a hang is killed (timeout verdict); socket() is denied inside the jail;
the host's /kaggle/input and /kaggle/working are unreadable; the expected values are never staged in the workspace; LcbRunner stdin + functional honest pass and
wrong answer fail through the same port. Writes <out>/QUAL.json; exit 0 only if every check holds. Usage: python tools/el_kaggle_port_qual.py <out dir>"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_ladder.execution import config as K, engine, trusted_verify as TV        # noqa: E402
from evidence_ladder.execution.kaggle_port import KaggleJailPort                         # noqa: E402
from evidence_ladder.execution.lcb_runner import LcbRunner                               # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
port = KaggleJailPort()
cfg = K.SandboxConfig()
CASES = [{"input": {"x": i}, "expected": i * 2} for i in (1, 2, 3)]
checks = {}


def tv_verdict(code, per_input=None):
    tv = TV.TrustedVerifier(port.run, cfg, scratch=Path(tempfile.mkdtemp(prefix="kq-")), per_input_timeout=per_input)
    res = tv.run("python", code.encode(), [c["input"] for c in CASES])
    v, each = engine.judge_each("python", res, CASES, cfg)
    return v, res


def check(name, fn):
    try:
        checks[name] = {"ok": bool(fn())}
    except Exception as e:      # noqa: BLE001
        checks[name] = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300])}


check("honest_pass", lambda: tv_verdict("def solve(d):\n    return d['x'] * 2\n")[0].status == "pass")
check("forged_frames_never_pass", lambda: tv_verdict("def solve(d):\n    print('%s V %%d' %% (d['x']*2))\n    return -1\n" % TV.TAG)[0].status != "pass")
check("claim_then_crash_never_pass", lambda: tv_verdict("import os\ndef solve(d):\n    print('%s V %%d' %% (d['x']*2), flush=True)\n    os._exit(3)\n" % TV.TAG)[0].status != "pass")
check("hang_is_killed", lambda: tv_verdict("import time\ndef solve(d):\n    time.sleep(60)\n", per_input=2.0)[0].status == "timeout")
SOCK = ("def solve(d):\n    import socket\n    try:\n        socket.socket(); return 'ALLOWED'\n    except OSError:\n        return 'denied'\n")
check("socket_denied", lambda: json.loads(tv_verdict(SOCK)[1].stdout.splitlines()[0])["value"] == "denied")
HOST = ("def solve(d):\n    import os\n    seen = []\n    for p in ('/kaggle/input', '/kaggle/working', '/proc/1/environ'):\n"
        "        try:\n            os.listdir(p) if not p.startswith('/proc') else open(p).read(10); seen.append(p)\n        except OSError:\n            pass\n    return seen\n")
check("host_paths_unreadable", lambda: json.loads(tv_verdict(HOST)[1].stdout.splitlines()[0])["value"] == [])
SCAN = ("def solve(d):\n    import os\n    hits = []\n    for dp, dn, fn in os.walk('/'):\n        for f in fn:\n            try:\n"
        "                if (b'EXPECTED-' + b'CANARY-77') in open(os.path.join(dp, f), 'rb').read(100000): hits.append(f)\n            except OSError:\n                pass\n    return hits\n")
def no_expected():
    tv = TV.TrustedVerifier(port.run, cfg, scratch=Path(tempfile.mkdtemp(prefix="kq-")), per_input_timeout=30)
    cases = [{"input": {"x": 1}, "expected": "EXPECTED-CANARY-77"}]
    res = tv.run("python", SCAN.encode(), [c["input"] for c in cases])
    return json.loads(res.stdout.splitlines()[0])["value"] == []
check("expected_values_never_staged", no_expected)
lr = LcbRunner(port.run, Path(tempfile.mkdtemp(prefix="kq-lcb-")))
check("lcb_stdin_honest_pass", lambda: lr.grade("a, b = map(int, input().split())\nprint(a + b)\n", [{"input": "2 3", "output": "5", "testtype": "stdin"}], None)["pass"])
check("lcb_stdin_wrong_fails", lambda: not lr.grade("print(0)\n", [{"input": "2 3", "output": "5", "testtype": "stdin"}], None)["pass"])
check("lcb_functional_honest_pass", lambda: lr.grade("class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b\n",
                                                      [{"input": "2\n3", "output": "5", "testtype": "functional"}], "add")["pass"])
ok = all(c["ok"] for c in checks.values())
(out / "QUAL.json").write_text(json.dumps({"port": "kaggle-jail", "verifier": TV.VERSION, "checks": checks, "all_ok": ok, "jail_calls": port.calls}, indent=1))
print(json.dumps({"all_ok": ok, "checks": {k: v["ok"] for k, v in checks.items()}}))
sys.exit(0 if ok else 1)
