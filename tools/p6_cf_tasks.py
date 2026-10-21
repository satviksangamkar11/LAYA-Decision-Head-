"""[P6] Codeforces cohort: public task view (statement + statement examples only; never a grading test) and the LiveCodeBench stdin prompt.
The statement text is composed deterministically from the dataset fields (description, Input, Output, Note) and rendered with the UNCHANGED LiveCodeBench
generic template (tools/lcb_tasks.render_prompt, no starter code -> the official stdin format). Public tests = the statement's own examples.
  build:  python tools/p6_cf_tasks.py <verifiable test parquet> <candidates.json> <out jsonl>   (writes the public file once, create-only)"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lcb_tasks as L      # noqa: E402

PUBLIC_FILE = ROOT / "configs/p6_cf_public_v1.jsonl"
FIELDS = ("id", "contest_id", "index", "title", "rating", "contest_start", "time_limit", "memory_limit", "description", "input_format", "output_format", "note", "examples")


def statement(r):
    parts = [r["title"], "", (r["description"] or "").strip(), "", "Input", (r["input_format"] or "").strip(), "", "Output", (r["output_format"] or "").strip()]
    for k, ex in enumerate(r["examples"] or []):
        parts += ["", "Example %d" % (k + 1), "Input", ex["input"].rstrip("\n"), "Output", ex["output"].rstrip("\n")]
    if (r.get("note") or "").strip():
        parts += ["", "Note", r["note"].strip()]
    return "\n".join(parts)


def prompt(r):
    return L.render_prompt({"statement": statement(r), "starter_code": ""})


def load(path=PUBLIC_FILE):
    return {json.loads(l)["id"]: json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()}


def task_spec(r):
    from evidence_ladder.common.interfaces import TaskSpec
    return TaskSpec(r["id"], "cf:" + r["id"], "python", "python", prompt(r), [{"input": e["input"], "expected": e["output"]} for e in r["examples"]], str(r.get("rating")))


def lcb_meta(r):
    return {"format": "stdin", "fn_name": None, "public": [{"input": e["input"], "output": e["output"], "testtype": "stdin"} for e in r["examples"]]}


def build(parquet, cands, out):
    import pyarrow.parquet as pq
    out = Path(out)
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    want = set(json.loads(Path(cands).read_text("utf-8")))
    rows = [{k: r.get(k) for k in FIELDS} for r in pq.read_table(parquet).to_pylist() if r["id"] in want]
    for r in rows:
        r["url"] = "https://codeforces.com/contest/%s/problem/%s" % (r["contest_id"], r["index"])
        r["examples"] = [{"input": e["input"], "output": e["output"]} for e in (r["examples"] or [])]
        r["source"] = "open-r1/codeforces@fbe3f6e9 verifiable/test (cc-by-4.0); statement and statement examples only"
    rows.sort(key=lambda r: r["id"])
    out.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    print(len(rows), "rows ->", out)


if __name__ == "__main__":
    build(*sys.argv[1:4])
