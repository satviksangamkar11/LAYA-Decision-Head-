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
