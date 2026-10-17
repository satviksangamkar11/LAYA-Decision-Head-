"""[MXC replacement] Latency benchmark: the original WSL sandbox vs MXC (processcontainer), same machine, same workspaces, same commands, same isolation requirements (no network, scrubbed env,
protected tests read-only and hash-verified, timeout, resource limits). ABAB interleaved so drift hits both equally. Create-only output dir with BENCHMARK.json (+ raw samples).

    PYTHONPATH=. .venv/Scripts/python.exe tools/mxc_benchmark.py results/raw/mxc-benchmark-<UTC> [--n 30] [--cases a,b,c] [--task-runs 30]

Cases (n >= 30 each):
  cold_first_call     a NEW sandbox object, first `echo`: WSL = object + wsl helper start; MXC one-shot = object + capability probe + node start + container; MXC persistent = same + bridge stays up
  warm_echo           `echo hi` on an existing sandbox object (WSL: every call is a fresh helper; MXC persistent: bridge already running; MXC one-shot: node started per call)
  python_test         `python3 test_clamp.py` (failing protected test, the controller's most frequent call)
  python_test_pass    the same command on the fixed workspace (exit 0)
  workspace_persist   `echo x > out.txt` with persist=True on a 12-file workspace (copy + run + hash + write-back)
  sequential_cands    10 different one-file candidate programs executed back to back; per-iteration total (n iterations)
  full_task           the full controller task (AgenticRun: investigate, dry-runs, commit, observe, finalize) on slug/clamp/wordcount, scripted options, no model; seconds per task
Model generation time is excluded everywhere (no model is loaded). Breakdown for MXC comes from sandbox.diagnostics_log; WSL has no equivalent breakdown (reported as not available).
Peak memory: peak RSS of the MXC bridge process tree (psutil, sampled every 20 ms) and the growth of vmmemWSL for the WSL runs; CPU seconds likewise. Where a number is not measured it says so."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import psutil                                                                              # noqa: E402
from laya_int1_t6 import TASKS                                                              # noqa: E402
from evidence_ladder.controller import replay as REP, sandbox_provider as P                  # noqa: E402
from evidence_ladder.controller.agentic import AgenticContext, AgenticTask                   # noqa: E402
from evidence_ladder.controller.mxc_sandbox import Limits                                    # noqa: E402

CASES = ["cold_first_call", "warm_echo", "python_test", "python_test_pass", "workspace_persist", "sequential_cands", "full_task"]


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def stats(xs):
    return {"n": len(xs), "mean": round(statistics.mean(xs), 4), "p50": round(pct(xs, 0.5), 4), "p95": round(pct(xs, 0.95), 4), "min": round(min(xs), 4), "max": round(max(xs), 4),
            "stdev": round(statistics.pstdev(xs), 4)}


class Sampler:
    """Peak RSS / CPU seconds of this process's descendants + vmmemWSL working set, sampled in a thread."""

    def __init__(self):
        self.peak_rss, self.cpu, self._stop, self._seen = 0, {}, threading.Event(), {}
        self.vmmem_peak = 0
        self.t = threading.Thread(target=self._run, daemon=True)

    def _vmmem(self):
        try:
            return max([p.memory_info().rss for p in psutil.process_iter(["name"]) if p.info["name"] in ("vmmemWSL", "vmmem")] or [0])
        except psutil.Error:
            return 0

    def _run(self):
        me = psutil.Process()
        while not self._stop.is_set():
            try:
                kids = me.children(recursive=True)
                rss = 0
                for k in kids:
                    try:
                        rss += k.memory_info().rss
                        ct = k.cpu_times()
                        self._seen[k.pid] = max(self._seen.get(k.pid, 0.0), ct.user + ct.system)
                    except psutil.Error:
                        pass
                self.peak_rss = max(self.peak_rss, rss)
                self.vmmem_peak = max(self.vmmem_peak, self._vmmem())
            except psutil.Error:
                pass
            time.sleep(0.02)

    def __enter__(self):
        self.vmmem0 = self._vmmem()
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()

    def result(self):
        return {"peak_descendant_rss_mb": round(self.peak_rss / 2 ** 20, 1), "descendant_cpu_seconds_sampled": round(sum(self._seen.values()), 2),
                "vmmemWSL_mb_start": round(self.vmmem0 / 2 ** 20, 1), "vmmemWSL_mb_peak": round(self.vmmem_peak / 2 ** 20, 1)}


def mk_ws(task, extra_files=0, fixed=False):
    T = TASKS[task]
    d = Path(tempfile.mkdtemp(prefix="bench-")) / "ws"
    d.mkdir(parents=True)
    for fn, c in T["files"].items():
        (d / fn).write_bytes(c.encode("utf-8"))
    for i in range(extra_files):
        (d / ("extra_%02d.py" % i)).write_bytes(("VALUE_%d = %d\n" % (i, i) + "x = [%s]\n" % ",".join(map(str, range(200)))).encode())
    if fixed:
        import el_arch1_variants as V
        f, o, n = V.FIX[task]
        (d / f).write_bytes((d / f).read_bytes().replace(o.encode(), n.encode()))
    return d, T


def backends(n_scratch):
    """-> {label: factory(protected) -> sandbox}. mxc_persistent keeps the node bridge alive between calls (no candidate state is kept: every run copies a fresh workspace into a new run directory)."""
    def wsl(prot):
        return P.make_sandbox(Path(tempfile.mkdtemp(prefix="bench-sb-")), prot, "wsl")

    def mxc1(prot):
        return P.make_sandbox(Path(tempfile.mkdtemp(prefix="bench-sb-")), prot, "mxc", persistent=False)

    def mxcp(prot):
        return P.make_sandbox(Path(tempfile.mkdtemp(prefix="bench-sb-")), prot, "mxc", persistent=True)
    return {"wsl": wsl, "mxc_oneshot": mxc1, "mxc_persistent": mxcp}


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return time.perf_counter() - t, r


def breakdown(sb):
    log = getattr(sb, "diagnostics_log", None)
    if not log:
        return None
    keys = ("prepare_ms", "mxc_call_ms", "verify_ms", "persist_ms", "total_ms")
    return {k: round(statistics.mean(d[k] for d in log if k in d), 2) for k in keys if any(k in d for d in log)}


def case_cold(n):
    out = {k: [] for k in backends(0)}
    ws, T = mk_ws("clamp")
    for i in range(n):
        for label, f in backends(0).items():
            def go():
                sb = f(T["protected"])
                r = sb.run(ws, "echo hi")
                P.validate_result(r)
                return sb
            dt, sb = timed(go)
            out[label].append(dt)
            if hasattr(sb, "close"):
                sb.close()
    return out, {}


def case_loop(n, cmd, task="clamp", persist=False, fixed=False, extra=0):
    out = {k: [] for k in backends(0)}
    parts = {}
    ws, T = mk_ws(task, extra, fixed)
    sbs = {label: f(T["protected"]) for label, f in backends(0).items()}
    for label, sb in sbs.items():
        P.validate_result(sb.run(ws, cmd, persist=persist))      # warm-up call, not timed (first call of a persistent bridge / helper)
    for i in range(n):
        for label, sb in sbs.items():
            dt, r = timed(lambda sb=sb: sb.run(ws, cmd, persist=persist))
            P.validate_result(r)
            out[label].append(dt)
    for label, sb in sbs.items():
        parts[label] = breakdown(sb)
        if hasattr(sb, "close"):
            sb.close()
    return out, parts


def case_sequential(n):
    out = {k: [] for k in backends(0)}
    ws0, T = mk_ws("clamp")
    cands = [("c%d.py" % i, "def solve(a, b):\n    return (a * %d + b) %% 97\nimport sys\nprint(solve(int(sys.argv[1]) if len(sys.argv) > 1 else 3, %d))\n" % (i + 2, i)) for i in range(10)]
    sbs = {label: f(T["protected"]) for label, f in backends(0).items()}
    for label, sb in sbs.items():
        sb.run(ws0, "echo warm")
    for it in range(n):
        for label, sb in sbs.items():
            t = time.perf_counter()
            for name, src in cands:
                d = Path(tempfile.mkdtemp(prefix="bench-c-")) / "ws"
                d.mkdir(parents=True)
                (d / name).write_bytes(src.encode())
                for fn, c in T["files"].items():
                    (d / fn).write_bytes(c.encode())
                P.validate_result(sb.run(d, "python3 " + name))
                shutil.rmtree(d.parent, ignore_errors=True)
            out[label].append(time.perf_counter() - t)
    parts = {}
    for label, sb in sbs.items():
        parts[label] = breakdown(sb)
        if hasattr(sb, "close"):
            sb.close()
    return out, parts


def full_task(label, factory, task):
    import el_arch1_variants as V
    sys.path.insert(0, str(ROOT / "evidence_ladder" / "controller" / "tests"))
    import phaseB_tests as B
    T = TASKS[task]
    ws = Path(tempfile.mkdtemp(prefix="bench-t-")) / "ws"
    ws.mkdir(parents=True)
    for fn, c in T["files"].items():
        (ws / fn).write_bytes(c.encode("utf-8"))
    sb = factory(T["protected"])
    t = time.perf_counter()
    at = AgenticTask(task, T["user"], ws, T["test_cmd"], T["protected"], "system text", 3, [V.FILE[task]])
    rr = REP.RecordedRun(at, AgenticContext(sandbox=sb, scorer=B.Scorer(scores=[0.1, 0.2, 0.9, 0.3]), arm="E", config={}), lambda m: "ctx")
    rr.begin()
    rr.investigate()
    rr.step([V.option("Read", file_path=T["protected"][0]), V.fix(task), V.wrong(task), V.bash(T["test_cmd"])], {})
    res = rr.finish()
    dt = time.perf_counter() - t
    assert res.answer_label == "VERIFIED", (label, task, res.answer_label)
    calls = len(sb.diagnostics_log) if hasattr(sb, "diagnostics_log") else len(rr.sb.calls)
    sandbox_ms = sum(d["total_ms"] for d in sb.diagnostics_log) if hasattr(sb, "diagnostics_log") else None
    if hasattr(sb, "close"):
        sb.close()
    return dt, calls, sandbox_ms


def case_full(n):
    out = {k: [] for k in backends(0)}
    calls, sandbox_ms = {}, {}
    fs = backends(0)
    for it in range(n):
        for task in ("slug", "clamp", "wordcount"):
            for label, f in fs.items():
                dt, c, sm = full_task(label, f, task)
                out[label].append(dt)
                calls.setdefault(label, []).append(c)
                if sm is not None:
                    sandbox_ms.setdefault(label, []).append(sm / 1000.0)
    extra = {"sandbox_calls_per_task_mean": {k: round(statistics.mean(v), 1) for k, v in calls.items()},
             "sandbox_seconds_per_task_mean": {k: round(statistics.mean(v), 3) for k, v in sandbox_ms.items()}, "tasks": ["slug", "clamp", "wordcount"], "model_generation_seconds": 0.0}
    return out, extra


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--task-runs", type=int, default=10, help="full_task repetitions per task and backend (x3 tasks => samples)")
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        ap.error("output dir exists (create-only)")
    out.mkdir(parents=True)
    res = {"started_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "host": {"cpu_count": os.cpu_count(), "platform": sys.platform, "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1)},
           "n": a.n, "isolation_profile": {"network": "none", "env": "scrubbed", "protected": "read-only + hash-verified", "limits": Limits().as_dict(), "wsl_limits": "MemoryMax 2G, TasksMax 256, CPUQuota 400%"}, "cases": {}}
    for case in a.cases.split(","):
        with Sampler() as smp:
            t0 = time.time()
            if case == "cold_first_call":
                samples, extra = case_cold(a.n)
            elif case == "warm_echo":
                samples, extra = case_loop(a.n, "echo hi")
            elif case == "python_test":
                samples, extra = case_loop(a.n, TASKS["clamp"]["test_cmd"])
            elif case == "python_test_pass":
                samples, extra = case_loop(a.n, TASKS["clamp"]["test_cmd"], fixed=True)
            elif case == "workspace_persist":
                samples, extra = case_loop(a.n, "echo x > out.txt", persist=True, extra=12)
            elif case == "sequential_cands":
                samples, extra = case_sequential(a.n)
            elif case == "full_task":
                samples, extra = case_full(a.task_runs)
            else:
                continue
            res["cases"][case] = {"seconds_per_call_or_unit": {k: stats(v) for k, v in samples.items()}, "breakdown_ms_mean": extra if case != "full_task" else None,
                                  "extra": extra if case == "full_task" else None, "resources": smp.result(), "wall_seconds": round(time.time() - t0, 1)}
            with open(out / ("samples_%s.json" % case), "x") as f:
                json.dump(samples, f)
        print(case, json.dumps({k: (v["p50"], v["p95"]) for k, v in res["cases"][case]["seconds_per_call_or_unit"].items()}), flush=True)
    res["finished_utc"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(out / "BENCHMARK.json", "x") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
