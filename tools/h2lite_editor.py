"""[H2-lite, H2LITE_PLAN_v1.md §4] Offline GLM editor: proposes prompt-template variants for the frozen 1.5B generator. NOT a selector: the winner is chosen only by
the pre-registered search (§5) on GPU runs; nothing here ranks variants.
Inputs sent to GLM: the two frozen template texts, the adapter rules, the SEARCH-split development tasks (statement + public examples) and, from round 2 on, the
runtime-visible outcome table of earlier variants (none exists yet). Never sent: confirmation-split tasks, P5/P6 tasks, hidden graders, grades, contracts.
Call cap: 6 sends in total for H2-lite (append-only ledger in the out dir; a completed request id is never resent). Keys: tools/laya_int1_stepE.load_keys
(in memory, never printed or saved). Model z-ai/glm-5.3-flash, reasoning_effort max, streamed.
Usage: PYTHONPATH=. python tools/h2lite_editor.py <out dir> --round 1 [--max-calls 2]"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from evidence_ladder import v2min as V                                    # noqa: E402
from evidence_ladder.controller import generation as GEN                 # noqa: E402
from evidence_ladder.critic_batch.transport import make_transport        # noqa: E402
from evidence_ladder.models import glm                                    # noqa: E402

CAP = 6
SEARCH = ["P1-%02d" % i for i in (2, 4, 6, 7, 9, 12, 29, 30, 35, 38, 39, 11, 21, 23, 26, 34, 16, 28, 37, 14, 20, 25)]
CONFIRM = ["P1-%02d" % i for i in (1, 3, 5, 8, 10, 13, 19, 24, 17, 27, 36, 40, 18, 31, 32, 33, 15, 22)]
assert len(SEARCH) == 22 and len(CONFIRM) == 18 and not set(SEARCH) & set(CONFIRM)
TASKS = ROOT / "results/raw/codex-pilot1-20261016/fixed_v2/worker/tasks.jsonl"

SYSTEM = ("You improve the prompt templates of a small (1.5B-parameter) code model. The model and its sampling are frozen; only the template text may change. "
          "Reply with ONE JSON object and nothing else.")

RULES = """Rules for every template you write:
- It must contain exactly these placeholders, each written as {name}: {task_prompt} {examples} {solution_format} {language}. No other braces.
- It must keep the line "Required solution format:" immediately followed by a newline and {solution_format}.
- It may be at most %d characters longer than the frozen template it replaces. Shorter is allowed; a 1.5B model follows short instructions better.
- Text must be TASK-INDEPENDENT: no facts, hints or algorithms for any particular task, no example solutions.
- The answer must still end with ONE fenced code block in {language} (the extractor takes the last fenced block).""" % V.MAX_EXTRA_CHARS

ASK = """Propose exactly TWO variants (Harness-style invariant / domain pair):
1. "general": a change to how the model approaches ANY task (planning, checking its own answer against the examples, staying concise).
2. "coding": a change specific to writing a Python solve(data) function (reading the input dictionary, edge cases, return types, avoiding printing).
Each variant gives full replacement texts for BOTH frozen templates.
JSON schema: {"variants": [{"name": "general", "qwen_sample_v2": "...", "qwen_sample_v2b": "...", "rationale": "one sentence"},
                           {"name": "coding",  "qwen_sample_v2": "...", "qwen_sample_v2b": "...", "rationale": "one sentence"}]}"""


def search_tasks():
    rows = {json.loads(l)["task_id"]: json.loads(l) for l in TASKS.read_text("utf-8").splitlines() if l.strip()}
    return [{"task_id": t, "statement": rows[t]["prompt"], "public_examples": rows[t]["public_tests"]} for t in SEARCH]


def user_message(round_no, outcomes, repair_errors=None):
    frozen = {tid: V.frozen_template(tid) for tid in GEN.FROZEN_STRATEGIES}
    parts = ["ROUND %d." % round_no, RULES, "FROZEN TEMPLATES (current text):\n" + json.dumps(frozen, indent=1),
             "DEVELOPMENT TASKS the variants will be tried on (statements and public examples only):\n" + json.dumps(search_tasks(), ensure_ascii=False),
             "OUTCOMES of earlier variants (runtime-visible only): " + (json.dumps(outcomes) if outcomes else "none yet (first round)."), ASK]
    if repair_errors:
        parts.append("Your previous reply violated the rules: %s. Return corrected JSON in the same schema." % json.dumps(repair_errors))
    return "\n\n".join(parts)


def payload(msg):
    p = {"model": glm.GLM_MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}], "max_tokens": 32768, "stream": True,
         "stream_options": {"include_usage": True}}
    p.update({"reasoning_effort": "max"})
    return p


def parse(content, round_no):
    t = (content or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    i, j = t.find("{"), t.rfind("}")
    doc = json.loads(t[i:j + 1])
    out, errors = [], []
    for v in doc.get("variants", []):
        try:
            ad = V.PromptAdapter("h2r%d_%s" % (round_no, v["name"]), {k: v[k] for k in GEN.FROZEN_STRATEGIES})
            out.append({"name": v["name"], "texts": ad.texts, "sha256": ad.sha256, "rationale": v.get("rationale", "")})
        except (KeyError, ValueError) as e:
            errors.append("%s: %s" % (v.get("name"), e))
    if len(doc.get("variants", [])) != 2:
        errors.append("expected exactly two variants")
    return out, errors


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--round", type=int, default=1); ap.add_argument("--max-calls", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ledger = out / "LEDGER.jsonl"
    sent = [json.loads(l) for l in ledger.read_text("utf-8").splitlines()] if ledger.exists() else []
    if any(e["round"] == a.round and e.get("valid") for e in sent):
        sys.exit("round %d already has a valid proposal set; nothing sent" % a.round)
    import laya_int1_stepE as SE
    keys = SE.load_keys()
    if not keys:
        sys.exit("no readable key file; zero calls made")
    send = make_transport(keys, read_timeout_s=120.0, total_timeout_s=1800.0)
    errors, calls = None, 0
    while calls < a.max_calls:
        if len(sent) >= CAP:
            sys.exit("H2-lite editor cap of %d sends reached" % CAP)
        rid = "h2lite-r%d-c%d" % (a.round, len([e for e in sent if e["round"] == a.round]) + 1)
        msg = user_message(a.round, None, errors)
        (out / ("%s.request.json" % rid)).write_text(json.dumps(payload(msg), indent=1, ensure_ascii=False), encoding="utf-8")
        t0 = time.time()
        label = list(keys)[len(sent) % len(keys)]
        try:
            status, body = send({"payload": payload(msg)}, label)
        except Exception as e:      # noqa: BLE001
            status, body = None, {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
        calls += 1
        rec = {"request_id": rid, "round": a.round, "key_label": label, "status": status, "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        valid = False
        if status == 200:
            content = body["choices"][0]["message"].get("content")
            (out / ("%s.response.json" % rid)).write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
            rec["usage"] = body.get("usage")
            try:
                props, errors = parse(content, a.round)
            except ValueError as e:
                props, errors = [], ["reply is not valid JSON: %s" % str(e)[:200]]
            valid = not errors and len(props) == 2
            rec["errors"] = errors
            if valid:
                (out / ("round%d_proposals.json" % a.round)).write_text(json.dumps({"round": a.round, "request_id": rid, "variants": props,
                                                                                    "note": "proposals only; the winner is chosen by the pre-registered search"}, indent=1, ensure_ascii=False), encoding="utf-8")
        else:
            rec["error"] = str(body)[:300]
        rec["valid"] = valid
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        sent.append(rec)
        print(json.dumps(rec))
        if valid or status is None:      # a send without status may have run: never retried automatically
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
