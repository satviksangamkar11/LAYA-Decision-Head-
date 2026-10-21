"""[MULTILANG] Logic test of evidence_ladder/execution/trusted_native.py with a scripted fake sandbox port (no real sandbox, runs anywhere):
exit status / kill / capture-limit rules, compile outcomes, infrastructure errors, and that only the INPUT (never the expected value) is staged. Usage: PYTHONPATH=. python tools/test_trusted_native.py"""
import json
import sys
import tempfile
from pathlib import Path

from evidence_ladder.execution import config as K, engine, trusted_native as N
from evidence_ladder.execution.trusted_verify import TrustedVerifier, TrustedVerifyInfraError

CFG = K.SandboxConfig()
CASES = [{"input": "1 2\n", "expected": "3\n"}, {"input": "3 4\n", "expected": "7\n"}]
staged = []


def R(stdout="", exit_code=0, killed=False, persisted=False, stderr=""):
    return {"exit_code": exit_code, "killed": killed, "stdout": stdout, "stderr": stderr, "protected_unchanged": True, "pre_tree_sha256": "", "post_tree_sha256": "", "persisted": persisted}


def verdict(behaviour, lang="c", compile_result=None, cases=CASES):
    def port(ws, cmd, persist=False, timeout=60):
        for f in Path(ws).iterdir():
            staged.append(f.read_bytes())
        if "launch.py" in cmd and "--" == cmd.split()[-1]:
            return R()                                                  # launch-overhead calibration
        if "launch.py" not in cmd:                                      # the compile step
            r = compile_result or R(persisted=True)
            if r["exit_code"] == 0 and r["persisted"]:
                (Path(ws) / "prog").write_bytes(b"\x7fELF")
            return r
        inp = (Path(ws) / "in.txt").read_text()
        return behaviour(inp)
    tv = TrustedVerifier(port, CFG, scratch=Path(tempfile.mkdtemp(prefix="tn-")), launch_overhead=0.0)
    res = tv.run(lang, b"src", [c["input"] for c in cases])
    return engine.judge_each(lang, res, cases, CFG)[0], res


sums = lambda inp: R("%d\n" % sum(int(x) for x in inp.split()))
assert verdict(sums)[0].status == "pass"
assert verdict(lambda i: R("999\n"))[0].failure_class == "wrong_answer"
assert verdict(lambda i: R("%d\n" % sum(int(x) for x in i.split()), exit_code=3))[0].status != "pass", "claim-then-crash must not pass"
assert verdict(lambda i: R("3\n", killed=True, exit_code=-9))[0].status == "timeout"
assert verdict(lambda i: R("x" * 6000))[0].failure_class == "invalid_output", "an output at the capture limit is never compared"
assert verdict(lambda i: R("@@TV@@ V 3\n@@LCB@@ 3\n"))[0].status != "pass", "printed frames are just stdout"
assert verdict(sums, compile_result=R(exit_code=1, persisted=True, stderr="error: x"))[0].failure_class == "compile_error"
try:
    verdict(sums, compile_result=R(exit_code=127, persisted=True, stderr="gcc: not found"))
    raise AssertionError("a toolchain that cannot start must raise")
except TrustedVerifyInfraError:
    pass
try:
    verdict(sums, compile_result=R(exit_code=0, persisted=False))
    raise AssertionError("a compile that persisted nothing is infrastructure")
except TrustedVerifyInfraError:
    pass
assert verdict(sums, lang="javascript")[0].status == "pass"      # javascript has no compile step
assert all(b not in (b"3\n", b"7\n") for b in staged), "an expected value was staged"
print(json.dumps({"ok": True, "staged_files_seen": len(staged)}))
sys.exit(0)
