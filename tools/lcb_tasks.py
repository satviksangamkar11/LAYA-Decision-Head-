"""[P5] Load LiveCodeBench v6 tasks safely. The 'private' tests are base64(zlib(pickle(str))): decoded ONLY with an unpickler that refuses every class,
so a crafted payload cannot run code. Split into what generation may see (title, statement, starter code, PUBLIC tests) and what only the offline grader may see."""
import base64
import io
import json
import pickle
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "data/livecodebench/v6/test6.jsonl"


class _StringsOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("refused class %s.%s" % (module, name))


def _private(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return json.loads(_StringsOnly(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load())


def load(path: Path = V6):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        meta = json.loads(r.get("metadata") or "{}")
        out[r["question_id"]] = {
            "public_view": {"task_id": r["question_id"], "title": r["question_title"], "platform": r["platform"], "difficulty": r["difficulty"],
                            "contest_date": r["contest_date"], "statement": r["question_content"], "starter_code": r["starter_code"],
                            "fn_name": meta.get("func_name"), "format": "functional" if meta.get("func_name") else "stdin",
                            "public_tests": json.loads(r["public_test_cases"])},
            "grader_only": {"private_tests": _private(r["private_test_cases"])},
        }
    return out


SYSTEM_MESSAGE_GENERIC = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches "
                          "the specification and passes all tests.")
FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMATTING_WITHOUT_STARTER_CODE = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within "
                                   "delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")


def render_prompt(pv) -> str:
    """LiveCodeBench's get_generic_question_template_answer (lcb_runner/prompts/code_generation.py, read 2026-10-20), verbatim; the generic system message is
    placed first because our backend sends one user turn."""
    p = SYSTEM_MESSAGE_GENERIC + "\n\n" + "### Question:\n%s\n\n" % pv["statement"]
    if pv["starter_code"]:
        p += "### Format: %s\n" % FORMATTING_MESSAGE_WITH_STARTER_CODE
        p += "```python\n%s\n```\n\n" % pv["starter_code"]
    else:
        p += "### Format: %s\n" % FORMATTING_WITHOUT_STARTER_CODE
        p += "```python\n# YOUR CODE HERE\n```\n\n"
    p += "### Answer: (use the provided format with backticks)\n\n"
    return p
