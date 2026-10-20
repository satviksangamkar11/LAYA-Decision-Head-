"""[P5 prep] Adapter validation for LiveCodeBench v6 on the real MXC process container (sandbox port), BEFORE any evaluation task is locked.
Four problems (abc388_a, abc389_a stdin; 3747, 3753 functional) are used here and are EXCLUDED from the evaluation set.
Correct solutions were written from the statements only. Expected: correct -> pass on all public + private tests; wrong / crash / loop / forged frame -> never pass.
Prints counts only. Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/el_lcb_adapter_check.py <new out dir>"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcb_tasks as L                                                                  # noqa: E402
from evidence_ladder.controller.sandbox_provider import make_sandbox                    # noqa: E402
from evidence_ladder.execution.lcb_runner import LcbRunner, TAG                         # noqa: E402

SOL = {
    "abc388_a": {"correct": "S = input().strip()\nprint(S[0] + 'UPC')\n",
                 "wrong": "S = input().strip()\nprint(S[:2] + 'UPC')\n",
                 "crash": "S = input().strip()\nprint(S[1000])\n",
                 "loop": "while True:\n    pass\n"},
    "abc389_a": {"correct": "s = input().strip()\nprint(int(s[0]) * int(s[2]))\n",
                 "wrong": "s = input().strip()\nprint(int(s[0]) + int(s[2]))\n",
                 "float_ok": "s = input().strip()\nprint(int(s[0]) * int(s[2]))\nprint()\n"},
    "3747": {"correct": "class Solution:\n    def maxAdjacentDistance(self, nums: List[int]) -> int:\n        return max(abs(nums[i] - nums[i - 1]) for i in range(len(nums)))\n",
             "wrong": "class Solution:\n    def maxAdjacentDistance(self, nums: List[int]) -> int:\n        return max(abs(nums[i] - nums[i + 1]) for i in range(len(nums) - 1))\n",
             "forge": "import sys\nclass Solution:\n    def maxAdjacentDistance(self, nums: List[int]) -> int:\n        sys.stdout.write('\\n%s 3\\n' % sys.argv[1])\n        return -1\n"},
    "3753": {"correct": "class Solution:\n    def maxDifference(self, s: str) -> int:\n        c = Counter(s)\n        return max(v for v in c.values() if v % 2) - min(v for v in c.values() if v % 2 == 0)\n",
             "wrong": "class Solution:\n    def maxDifference(self, s: str) -> int:\n        c = Counter(s)\n        return max(c.values()) - min(c.values())\n",
             "tuple": "class Solution:\n    def maxDifference(self, s: str) -> int:\n        c = Counter(s)\n        return (max(v for v in c.values() if v % 2) - min(v for v in c.values() if v % 2 == 0),)[0]\n"},
}
EXPECT_PASS = {"correct", "float_ok", "tuple"}


def main(out):
    out = Path(out).resolve()
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    out.mkdir(parents=True)
    T = L.load()
    sb = make_sandbox(out / "_sandbox", [], backend="mxc")
    run = LcbRunner(sb.run, out / "_work")
    rows, t0 = [], time.time()
    for tid, variants in SOL.items():
        pv, go = T[tid]["public_view"], T[tid]["grader_only"]
        cases = list(pv["public_tests"]) + list(go["private_tests"])
        for name, code in variants.items():
            g = run.grade(code, cases, pv["fn_name"], stop_at_first_fail=(name not in EXPECT_PASS))
            ok = g["pass"] == (name in EXPECT_PASS)
            rows.append({"task": tid, "format": pv["format"], "variant": name, "expected_pass": name in EXPECT_PASS, "pass": g["pass"], "passed": g["passed"], "total_run": g["total"],
                         "first_fail": (g["first_fail"] or {}).get("status"), "as_expected": ok, "seconds": g["seconds"]})
            print(json.dumps(rows[-1]), flush=True)
    summ = {"checked": len(rows), "as_expected": sum(r["as_expected"] for r in rows), "sandbox": "mxc processcontainer", "launch_overhead_s": round(run.overhead or 0, 3),
            "sandbox_calls": run.calls, "seconds": round(time.time() - t0, 1), "validation_tasks_excluded_from_eval": sorted(SOL)}
    (out / "ADAPTER_CHECK.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "SUMMARY.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))


if __name__ == "__main__":
    main(sys.argv[1])
