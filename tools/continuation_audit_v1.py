"""Option A, step 1: no-inference eligibility audit of the canonical TRAIN replay states (user GO, chat 2026-10-14). File reading only: no model, no
sandbox execution, no Kaggle. For every canonical TRAIN state that replayed exactly: snapshot check (every recorded file present with its recorded
sha256), candidate audit (Fable's recorded action + the 4 saved Qwen3-30B proposals): valid / truncated, tool, action type, executable in the sandbox,
duplicates, trivial moves. Writes results/raw/continuation-audit-v1-<ts>/ (AUDIT.json, STATES.jsonl).

  PYTHONPATH=. python tools/continuation_audit_v1.py
"""
import collections
import glob
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import mutation_pilot_v1 as M  # noqa: E402

STATES = ROOT / "results/raw/fable-replay-states-go-py-ts-strict-20261009T220928Z"
GATES = [ROOT / "results/raw/fable-fidelity-gate-20261009T221736Z", ROOT / "results/raw/fable-fidelity-gate-20261009T222250Z"]
TRIVIAL = re.compile(r"^\s*(pwd|ls|ls -la|ls -l|whoami|echo [^>|]*|true|cd \S+)\s*$")


def category(name, args):
    n = (name or "").lower()
    if n in ("write", "edit", "multiedit", "notebookedit"):
        return "edit"
    if n == "read":
        return "read"
    if n in ("grep", "glob"):
        return "search"
    if n in ("todowrite", "task"):
        return "plan"
    if n != "bash":
        return "other"
    c = (args.get("command") or "").strip()
    if TRIVIAL.match(c):
        return "trivial"
    if re.search(r"\bsed\s+-i|(^|[^2&])>\s*[\w./]|\btee\b|\bpatch\b", c):
        return "edit"
    if re.search(r"\b(pytest|unittest|tox)\b|node --test|go test|cargo test|npm (run )?test|python\S*\s+\S*test|pwsh .*test|bash \S*test", c):
        return "test"
    if re.match(r"(cd \S+ && )?(grep|rg|find|ls|tree)\b", c):
        return "search"
    if re.match(r"(cd \S+ && )?(cat|head|tail|sed -n|less|nl|wc)\b", c):
        return "read"
    if re.search(r"\b(python|python3|node|go run|go build|make|bash|sh|\./)", c):
        return "run"
    return "other"


EXECUTABLE = {"edit", "read", "search", "test", "run", "trivial", "other"}   # Bash / Write / Edit run in the sandbox; Read / Grep / Glob map to shell


def parse_proposal(p):
    if p.get("finish_reason") not in (None, "stop"):
        return None, "truncated_%s" % p.get("finish_reason")
    try:
        a = json.loads(p["action_text"])
    except ValueError:
        return None, "invalid_json"
    calls = a.get("calls") if isinstance(a, dict) and "calls" in a else [a] if isinstance(a, dict) and "name" in a else None
    if isinstance(a, dict) and a.get("reply") and not calls:
        return [("reply", {})], None
    if not calls:
        return None, "no_tool_call"
    return [(c.get("name"), c.get("arguments") or {}) for c in calls], None


def canon(calls):
    return json.dumps([(n, {k: v for k, v in sorted(a.items()) if k not in ("description", "timeout")}) for n, a in calls], sort_keys=True, ensure_ascii=False)


def main():
    out = ROOT / ("results/raw/continuation-audit-v1-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True)
    sp = M.canonical_train()
    verdict = [{json.loads(l)["decision_id"]: json.loads(l)["verdict"] for l in open(g / "rows.jsonl", encoding="utf-8")} for g in GATES]
    go_fixed = {json.loads(l)["decision_id"] for p in glob.glob(str(ROOT / "results/raw/go-mod-fidelity-v1-*/rows.jsonl")) for l in open(p, encoding="utf-8")
                if json.loads(l).get("verdicts") == ["MATCH", "MATCH"]}
    rows, why = [], collections.Counter()
    for dj in sorted(STATES.glob("*/decision.json")):
        d = json.loads(dj.read_text("utf-8"))
        did = d["decision_id"]
        if sp.get(did.split(":")[1]) != "train":
            continue
        replay = (verdict[0].get(did) == "MATCH" and verdict[1].get(did) == "MATCH") or did in go_fixed
        if not replay:
            continue
        ws = dj.parent / "workspace"
        snap_ok = all((ws / f).is_file() and hashlib.sha256((ws / f).read_bytes()).hexdigest() == meta["sha256"] for f, meta in d["files"].items())
        cands = [dict(source="fable", calls=[(c["name"], c["arguments"]) for c in d["recorded_calls"]], problem=None)]
        for p in d["proposals"]:
            calls, prob = parse_proposal(p)
            cands.append(dict(source="qwen30b_%s" % p["sample_idx"], calls=calls, problem=prob))
        seen, eligible = {}, []
        for c in cands:
            if c["problem"]:
                c["status"] = c["problem"]
                continue
            cats = [category(n, a) for n, a in c["calls"]]
            c["types"] = cats
            c["type"] = "edit" if "edit" in cats else cats[0]
            k = canon(c["calls"])
            if c["type"] == "reply":
                c["status"] = "ask_or_reply_not_compared"
            elif all(t == "trivial" for t in cats):
                c["status"] = "trivial"
            elif k in seen:
                c["status"] = "duplicate_of_%s" % seen[k]
            else:
                seen[k] = c["source"]
                c["status"] = "eligible"
                eligible.append(c)
        types = sorted({c["type"] for c in eligible})
        row = dict(decision_id=did, lang=d["lang"], trajectory=did.split(":")[1], snapshot_exact=snap_ok, replay_exact=True,
                   candidates=[dict(source=c["source"], status=c["status"], types=c.get("types"), tools=[n for n, _ in c["calls"]] if c.get("calls") else None)
                               for c in cands],
                   n_eligible=len(eligible), eligible_types=types, two_plus_distinct=len(eligible) >= 2, two_plus_types=len(types) >= 2)
        rows.append(row)
        for c in cands:
            why[c["status"].split("_of_")[0]] += 1
    (out / "STATES.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    ok = [r for r in rows if r["snapshot_exact"]]
    summary = dict(canonical_train_replay_states=len(rows), trajectories=len({r["trajectory"] for r in rows}), snapshot_exact=len(ok),
                   candidate_status=dict(why), eligible_per_state=dict(collections.Counter(r["n_eligible"] for r in ok)),
                   states_2plus_distinct_eligible=sum(r["two_plus_distinct"] for r in ok),
                   states_2plus_action_types=sum(r["two_plus_types"] for r in ok),
                   trajectories_2plus_action_types=len({r["trajectory"] for r in ok if r["two_plus_types"]}),
                   type_combinations_2plus=dict(collections.Counter("+".join(r["eligible_types"]) for r in ok if r["two_plus_types"]).most_common()),
                   languages_2plus_types=dict(collections.Counter(r["lang"] for r in ok if r["two_plus_types"])))
    (out / "AUDIT.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary, indent=1))
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
