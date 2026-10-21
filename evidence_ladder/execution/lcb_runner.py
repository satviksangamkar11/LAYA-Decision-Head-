"""[P5 adapter] LiveCodeBench program execution over the SANDBOX PORT (sandbox.run(workspace, cmd, persist=False, timeout)), one fresh process per test case,
verdict on the host. Comparison follows LiveCodeBench's own grader (lcb_runner/evaluation/testing_util.py, read 2026-10-20):
  stdin       : strip the whole output, split on "\n", strip each line; line counts must match; each line equal as a string, else equal as a list of Decimal tokens
                (no float tolerance); anything else is a wrong answer.
  functional  : each line of the test input is json.loads'ed and passed positionally to Solution().<fn_name> (or a module-level function when there is no
                class Solution); a tuple result becomes a list; compared with == against json.loads(expected output).
  time limit  : 6 s per test case (LiveCodeBench's default); the kill is at 6 s + the measured launch overhead of the sandbox.
The candidate never sees expected outputs: they stay on the host. Infrastructure failures raise LcbInfraError (never a verdict)."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

TAG = "@@LCB@@"
TIME_LIMIT = 6.0
RESULT_KEYS = ("exit_code", "killed", "stdout", "stderr", "protected_unchanged", "pre_tree_sha256", "post_tree_sha256", "persisted")

STDIN_WRAPPER = '''import runpy, sys
sys.stdin = open("in.txt", "r", encoding="utf-8")
sys.setrecursionlimit(1 << 25)
runpy.run_path("cand.py", run_name="__main__")
'''

FUNC_WRAPPER = '''import json, sys
sys.setrecursionlimit(1 << 25)
import importlib.util
spec = importlib.util.spec_from_file_location("cand", "cand.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn_name = sys.argv[2]
obj = mod.Solution() if hasattr(mod, "Solution") else mod
fn = getattr(obj, fn_name)
args = [json.loads(line) for line in open("in.txt", encoding="utf-8").read().split("\\n")]
out = fn(*args)
if isinstance(out, tuple):
    out = list(out)
sys.stdout.write("\\n" + sys.argv[1] + " " + json.dumps(out) + "\\n")
'''

PRELUDE = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\n"
           "from math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\n"
           "from sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string, re, datetime, collections, heapq, bisect, copy, math, random, "
           "statistics, itertools, functools, operator, io, sys, json\nsys.setrecursionlimit(50000)\n")      # LiveCodeBench's import_string, prepended to functional code


class LcbInfraError(RuntimeError):
    infra = True


def stripped_lines(s: str) -> List[str]:
    return [l.strip() for l in s.strip().split("\n")]


def stdio_equal(pred: str, gt: str) -> bool:
    a, b = stripped_lines(pred), stripped_lines(gt)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if [Decimal(t) for t in x.split()] != [Decimal(t) for t in y.split()]:
                return False
        except (InvalidOperation, ValueError):
            return False
    return True


def functional_equal(pred: Any, gt_text: str) -> bool:
    return pred == json.loads(gt_text)


class LcbExecutionBackend:
    """[P5] ExecutionBackend for the Evidence-Ladder controller on LiveCodeBench tasks: run_public() on the PUBLIC tests only (the private tests are never given to it).
    meta: task_id -> {"format": "stdin"|"functional", "fn_name": str|None, "public": [LiveCodeBench public cases]}."""
    def __init__(self, runner: "LcbRunner", meta: Dict[str, Dict[str, Any]]):
        self.runner, self.meta = runner, meta

    def run_public(self, artifact, task) -> List[Dict[str, Any]]:
        import hashlib
        if hashlib.sha256(artifact.source.encode("utf-8")).hexdigest() != artifact.artifact_id:
            return [{"check_id": "parse_v1", "status": "error", "failure_class": "sandbox_error", "detail": "artifact integrity"}]
        m = self.meta[task.task_id]
        try:
            outs = [self.runner.run_case(artifact.source, c, m["fn_name"]) for c in m["public"]]
        except LcbInfraError as e:
            return [{"check_id": "parse_v1", "status": "error", "failure_class": "sandbox_error", "detail": str(e)[:200]}]
        res = []
        for i, o in enumerate(outs):
            st = "pass" if o["status"] == "pass" else ("timeout" if o["status"] == "timeout" else "fail")
            res.append({"check_id": "public_examples_v1", "example_index": i, "status": st, "failure_class": "" if st == "pass" else o["status"], "observed": o.get("observed", o["status"]),
                        "duration_s": o["seconds"]})
        if res and all(r["status"] == "pass" for r in res):
            res.insert(0, {"check_id": "parse_v1", "status": "pass", "detail": "the program loaded and ran every public test"})
        return res

    def backend_info(self) -> Dict[str, Any]:
        return {"engine": "lcb_runner_v1", "sandbox": "sandbox port (MXC process container in P5)", "time_limit_s": TIME_LIMIT,
                "comparison": "LiveCodeBench testing_util rules (stdin: stripped lines then Decimal tokens; functional: json ==)", "public_tests_only": True}

    def run_experiment(self, artifact, experiment):
        return {"check_id": "critic_experiment_v1", "status": "error", "failure_class": "sandbox_error", "detail": "experiments are not supported for LiveCodeBench tasks (critic off in P5)"}

    def run_inputs(self, artifact, task, inputs):
        raise RuntimeError("contract inputs are not supported for LiveCodeBench tasks (no trusted contract exists)")


class LcbRunner:
    def __init__(self, run: Callable[..., Dict[str, Any]], scratch: Path, launch_overhead: Optional[float] = None):
        self._run, self.scratch = run, Path(scratch).resolve()
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.overhead = launch_overhead
        self.calls = 0

    def _call(self, files: Dict[str, str], cmd: str, timeout: float) -> Dict[str, Any]:
        wd = self.scratch / uuid.uuid4().hex
        wd.mkdir(parents=True)
        try:
            for n, t in files.items():
                (wd / n).write_text(t, encoding="utf-8")
            try:
                r = self._run(str(wd), cmd, persist=False, timeout=int(max(1, round(timeout))))
            except KeyboardInterrupt:
                raise
            except BaseException as e:      # noqa: BLE001
                raise LcbInfraError("sandbox.run failed: %s: %s" % (type(e).__name__, str(e)[:300])) from e
            if any(k not in r for k in RESULT_KEYS):
                raise LcbInfraError("sandbox result lacks contract fields")
            self.calls += 1
            return r
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    def _oh(self) -> float:
        if self.overhead is None:
            ts = []
            for _ in range(2):
                t0 = time.time(); r = self._call({}, "python3 -c 0", 60); ts.append(time.time() - t0)
                if r["exit_code"] != 0:
                    raise LcbInfraError("launch calibration failed")
            self.overhead = min(ts)
        return self.overhead

    # ---------------------------------------------------------------- [MULTILANG 2026-10-21] stdin programs in c / cpp / javascript / go / rust (trusted_native.py); python is untouched
    def _native_case(self, code: str, case: Dict[str, Any], language: str) -> Dict[str, Any]:
        import hashlib
        from . import trusted_native as N
        if language not in N.NATIVE_LANGUAGES:
            raise LcbInfraError("language %r has no native runner" % language)
        if case["testtype"] != "stdin":
            raise LcbInfraError("functional (class Solution) LiveCodeBench tests are Python-only; %s supports stdin tests" % language)
        if getattr(self, "_nc", None) is None:
            def port(wd, cmd, persist, timeout, role):
                try:
                    r = self._run(str(wd), cmd, persist=persist, timeout=int(max(1, round(timeout))))
                except KeyboardInterrupt:
                    raise
                except BaseException as e:      # noqa: BLE001
                    raise LcbInfraError("sandbox.run failed: %s: %s" % (type(e).__name__, str(e)[:300])) from e
                if any(k not in r for k in RESULT_KEYS):
                    raise LcbInfraError("sandbox result lacks contract fields")
                self.calls += 1
                return r
            self._nc, self._compiled = N.NativeCore(port, self.scratch / "native", infra=LcbInfraError), {}
        key = (language, hashlib.sha256(code.encode("utf-8")).hexdigest())
        if key not in self._compiled:
            self._compiled[key] = self._nc.compile(language, code.encode("utf-8"))
        comp = self._compiled[key]
        if comp.status != "ok":
            return {"status": "timeout" if comp.status == "timeout" else "compile_error", "seconds": round(comp.seconds, 3), "detail": comp.stderr[-200:], "observed": "<%s>" % comp.status}
        kill = TIME_LIMIT + self._oh()
        r = self._nc.run_one(language, comp, case["input"], kill, "lcb_" + language)
        dt = round(r["seconds"], 3)
        kind, detail = N.classify(r, 4 << 20, int(getattr(self, "capture_limit", None) or N.DEFAULT_CAPTURE_LIMIT))
        if kind == "timeout":
            return {"status": "timeout", "seconds": dt, "detail": "killed at %.1fs" % kill}
        if kind == "crash":
            err = str(r.get("stderr", "")).strip().splitlines()
            return {"status": "runtime_error", "seconds": dt, "detail": str(r.get("stderr", ""))[-200:], "observed": "<error: %s>" % (err[-1][:200] if err else detail)}
        if kind != "ok":
            return {"status": "invalid_output", "seconds": dt, "detail": detail}
        out = str(r.get("stdout", ""))
        return {"status": "pass" if stdio_equal(out, case["output"]) else "wrong_answer", "seconds": dt, "detail": "", "observed": out.strip()[:300]}

    def run_case(self, code: str, case: Dict[str, Any], fn_name: Optional[str], language: str = "python") -> Dict[str, Any]:
        """-> {"status": pass|wrong_answer|runtime_error|timeout, "seconds", "detail"}; expected output is compared HERE, never passed in."""
        if language != "python":
            return self._native_case(code, case, language)
        kill = TIME_LIMIT + self._oh()
        t0 = time.time()
        if case["testtype"] == "stdin":
            r = self._call({"cand.py": PRELUDE + code, "in.txt": case["input"], "wrap.py": STDIN_WRAPPER}, "python3 wrap.py", kill)
        else:
            r = self._call({"cand.py": PRELUDE + code, "in.txt": case["input"], "wrap.py": FUNC_WRAPPER}, "python3 wrap.py %s %s" % (TAG, fn_name), kill)
        dt = time.time() - t0
        if r["killed"]:
            return {"status": "timeout", "seconds": round(dt, 3), "detail": "killed at %.1fs" % kill}
        if int(r["exit_code"]) != 0:
            err = str(r.get("stderr", "")).strip().splitlines()
            return {"status": "runtime_error", "seconds": round(dt, 3), "detail": str(r.get("stderr", ""))[-200:], "observed": "<error: %s>" % (err[-1][:200] if err else "nonzero exit")}
        out = str(r.get("stdout", ""))
        if case["testtype"] == "stdin":
            ok = stdio_equal(out, case["output"])
        else:
            frames = [l[len(TAG) + 1:] for l in out.splitlines() if l.startswith(TAG + " ")]
            if len(frames) != 1:
                return {"status": "wrong_answer", "seconds": round(dt, 3), "detail": "expected exactly one result frame, got %d" % len(frames)}
            try:
                ok = functional_equal(json.loads(frames[0]), case["output"])
            except ValueError:
                return {"status": "wrong_answer", "seconds": round(dt, 3), "detail": "unparsable result"}
        return {"status": "pass" if ok else "wrong_answer", "seconds": round(dt, 3), "detail": "", "observed": out.strip()[:300] if case["testtype"] == "stdin" else (frames[0][:300] if frames else "")}

    def grade(self, code: str, cases: List[Dict[str, Any]], fn_name: Optional[str], stop_at_first_fail: bool = True, language: str = "python") -> Dict[str, Any]:
        res = []
        for c in cases:
            res.append(self.run_case(code, c, fn_name, language))
            if stop_at_first_fail and res[-1]["status"] != "pass":
                break
        passed = sum(r["status"] == "pass" for r in res)
        return {"pass": passed == len(cases), "passed": passed, "total": len(cases), "first_fail": next((r for r in res if r["status"] != "pass"), None),
                "seconds": round(sum(r["seconds"] for r in res), 2)}
