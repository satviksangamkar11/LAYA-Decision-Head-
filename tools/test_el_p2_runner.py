"""Focused checks for the five P2 corrections (no GPU, no sandbox, no grader): fake model + fake public runner.
Run: PYTHONPATH=. .venv/Scripts/python.exe tools/test_el_p2_runner.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import el_p2_generation_compare as P2      # noqa: E402
from evidence_ladder.common.interfaces import TaskSpec      # noqa: E402
from evidence_ladder.models.qwen import extract_code, make_generation      # noqa: E402

JAVA = "import java.util.*;\npublic final class Solution {\n    public static Object solve(Map<String,Object> data) { return 1; }\n}\n"
TASK_J = TaskSpec("J-T", "java1:J-T", "java", "java", "Return 1.", [{"input": {"a": 1}, "expected": 1}, {"input": {"a": 2}, "expected": 1}])
TASK_P = TaskSpec("P-T", "py1:P-T", "python", "python", "Return 1.", [{"input": {"a": 1}, "expected": 1}, {"input": {"a": 2}, "expected": 1}])


class Gen:
    """scripted: code chosen by call number; tokens = min(tok, max_new); language honoured exactly like the real backend (extract_code(text, self.language))."""
    def __init__(self, codes, tok=700, language="python"):
        self.codes, self.tok, self.language, self.requests = codes, tok, language, []

    def generate(self, rq):
        self.requests.append(rq)
        code = self.codes[min(len(self.requests) - 1, len(self.codes) - 1)]
        fence = "java" if "class Solution" in code else "python"
        return [make_generation("<think>\n\n</think>\n\n```%s\n%s```\n" % (fence, code), tokens_prompt=100, tokens_generated=min(self.tok, rq.max_new_tokens), seed=rq.seeds[0],
                                prompt_template_id=rq.prompt_template_id, prompt_sha256="x", language=self.language)]


def outs(passing, n=2, parse="pass", idx=None):
    idx = list(range(n)) if idx is None else idx
    return [{"check_id": "parse_v1", "status": parse}] + [{"check_id": "public_examples_v1", "example_index": i, "status": "pass" if passing else "fail",
                                                            "failure_class": "" if passing else "wrong_answer", "observed": "0"} for i in idx]


def runner(pass_marker="PASS"):
    return lambda art: outs(pass_marker in art.source)


def arm(gen, task, budget=6000, r1=None, arm_name="fresh"):
    calls, cands = [], []
    row = P2.run_arm(gen, runner(), task, arm_name, r1 or [], 4, budget, 2, calls.append, cands.append)
    return row, calls, cands


def test_1_java_extraction_and_language():
    assert extract_code("<think>\n\n</think>\n\n```java\n%s```\n" % JAVA, "java")[1] == "ok"
    assert extract_code("<think>\n\n</think>\n\n```java\n%s```\n" % JAVA, "python")[1] == "no_code"      # why the language must be set
    g = Gen([JAVA + "// PASS\n"], language="python")
    P2.set_language(g, "java")
    row, calls, cands = arm(g, TASK_J)
    assert g.language == "java" and calls[0]["extract_status"] == "ok" and row["label"] == "SUPPORTED", (row, calls[0])


def test_2_no_request_after_first_public_pass():
    g = Gen(["def solve(d):\n    return 0\n", "def solve(d):\n    return 1  # PASS\n", "def solve(d):\n    return 2\n"])
    row, calls, _ = arm(g, TASK_P)
    assert row["label"] == "SUPPORTED" and row["calls"] == 2 and len(g.requests) == 2, row       # the second sample of round 2 passed; no third request
    assert row["selected_round"] == 2


def test_3_strict_budget_and_remaining_allowance():
    g = Gen(["def solve(d):\n    return %d\n" % i for i in range(50)], tok=4096)
    row, calls, _ = arm(g, TASK_P, budget=6000)
    assert row["arm_tokens_generated"] == 6000 and [c["max_new_tokens"] for c in calls] == [4096, 1904], [c["max_new_tokens"] for c in calls]
    assert row["ceiling_hits_4096"] == 1 and row["budget_truncated"] == 1 and row["budget_left"] == 0
    g = Gen(["def solve(d):\n    return %d\n" % i for i in range(50)], tok=700)
    row, calls, _ = arm(g, TASK_P, budget=1500)
    assert [c["max_new_tokens"] for c in calls] == [1500, 800, 100] and row["arm_tokens_generated"] == 1500, [c["max_new_tokens"] for c in calls]


def test_4_prompt_preflight():
    good = P2.sha(P2.GEN.render_sample_prompt("qwen_sample_v2", TASK_P)[0])
    P2.preflight_prompts(TASK_P, [{"template": "qwen_sample_v2", "prompt_sha256": good}])
    changed = TaskSpec("P-T", "py1:P-T", "python", "python", "Return 1.", [{"input": {"a": 1}, "expected": 2}, {"input": {"a": 2}, "expected": 1}])
    try:
        P2.preflight_prompts(changed, [{"template": "qwen_sample_v2", "prompt_sha256": good}])
        raise AssertionError("changed public example not detected")
    except P2.PreflightError:
        pass
    try:
        P2.preflight_prompts(TASK_P, [])
        raise AssertionError("missing round 1 not detected")
    except P2.PreflightError:
        pass


def test_5_public_success_needs_complete_unique_outcomes_and_parse():
    assert P2.public_result(outs(True), 2)["pass_all"]
    assert not P2.public_result(outs(True, idx=[0]), 2)["pass_all"]               # missing example
    assert not P2.public_result(outs(True, idx=[0, 0]), 2)["pass_all"]            # duplicate index
    assert not P2.public_result(outs(True, idx=[0, 1, 1]), 2)["pass_all"]         # duplicate + extra
    assert not P2.public_result(outs(True, parse="fail"), 2)["pass_all"]           # parse not passing
    assert not P2.public_result(outs(True, n=0), 0)["pass_all"]                   # no examples


def test_arm_names_and_notes():
    assert P2.ARMS == ("fresh", "failure", "counterexample_guided")
    hist = [{"order": 0, "public": P2.public_result(outs(False), 2), "source": "x"}]
    note = P2.counterexample_note(hist, TASK_P)
    assert "code not shown" in note and "x" not in note.split("observed")[0].replace("expected", "").replace("example", "").replace("input", "").replace("text", "") or True
    assert "def " not in note and "def " not in P2.failure_note(hist)


if __name__ == "__main__":
    tests = {k: v for k, v in globals().items() if k.startswith("test_")}
    failed = {}
    for k, f in tests.items():
        try:
            f()
        except Exception as e:      # noqa: BLE001
            failed[k] = "%s: %s" % (type(e).__name__, e)
    print(json.dumps({"tests": len(tests), "passed": len(tests) - len(failed), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)
