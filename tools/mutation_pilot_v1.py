"""Mutation pilot v1 (user GO, chat 2026-10-13, step 2): execution-verified alternatives on 20 canonical TRAIN states, labelled by tests only.
Runs as root inside the qualified WSL sandbox (tools/fable_wsl_runner.py). No model, no generator, no judge, no training, no upload. Canonical
CAL / TEST / VAL states are never executed: the canonical split comes from the imitation set's split map, NOT the trace file's `split` field
(of the 118 trace-"train" replayable states only 72 are canonical TRAIN; 16 trace-"val" states are canonical TRAIN).

  wsl -d Ubuntu-24.04 -u root -e python3 "/mnt/d/local model/tools/mutation_pilot_v1.py" [<existing out dir to resume>]

Create-only; every finished screen / mutant is appended to disk at once and skipped on resume.
"""
import collections
import gzip
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fable_wsl_runner as W  # noqa: E402

STATES = ROOT / "results/raw/fable-replay-states-go-py-ts-strict-20261009T220928Z"
STEPS = ROOT / "results/raw/fable-fidelity-steps-20261009T221513Z"
GATES = [ROOT / "results/raw/fable-fidelity-gate-20261009T221736Z", ROOT / "results/raw/fable-fidelity-gate-20261009T222250Z"]
SPLIT_MAP = ROOT / "results/raw/fable-imitation-pools-v1.2.1-20261010T093223Z"
N_STATES, MAX_MUTANTS, MUT_TIMEOUT, WORKERS = 20, 8, 60, 3

PLAN = {
    "written": "2026-10-13, before any state was screened or any mutant was executed",
    "approved_by": "user, chat 2026-10-13: GO steps 0, 1, 2; step 2 = mutation pilot on 20 canonical TRAIN states in isolated WSL sandboxes on this PC",
    "purpose": "diagnostic of execution-verified, same-style alternatives. These states were trained on (imitation pools), so nothing here is evidence about unseen decisions.",
    "population": "replayable states (both fidelity runs MATCH) whose trajectory is canonical TRAIN in the imitation set's split map; never CAL / TEST / VAL",
    "selection (outcome-blind order)": "ascending sha256(decision_id); screen each: the recorded test on the untouched state (baseline) and with Fable's own action only "
                                       "(action_only), twice each. Selected iff both pairs are stable, Fable's own calls are all Write / Edit, no other file change sits "
                                       "between Fable's action and the recorded test, baseline differs from action_only, and action_only exits 0. Stop at %d." % N_STATES,
    "mutants": "single-site text-span replacement inside Fable's new text (Write content, Edit new_string); comment lines, trailing comments and quoted strings "
               "skipped. Operators: cmp (== != === !== and spaced < <= > >=), bool (True/False, true/false), logic (and/or, &&/||), num (integer +1), "
               "arith (spaced + / -). Sites ordered by sha256(decision_id|operator|offset), taken round-robin over operators, at most %d per state, "
               "duplicates dropped. No formatting change outside the site." % MAX_MUTANTS,
    "execution": "each mutant: Fable's calls with the mutated text, then the recorded test, twice, each from a fresh copy of the state; time limit %d s" % MUT_TIMEOUT,
    "labels (vs Fable's own action on the same state)": {
        "WORSE": "both runs agree, the mutant's test exits non-zero, Fable's action_only exits 0",
        "EQUAL_ON_TESTS": "both runs agree and match action_only (exit and normalised output); NOT proof of equivalence (weak tests, uncovered code)",
        "UNKNOWN": "unstable, killed / timed out, unreplayable, or anything else"},
    "flag": "crash = SyntaxError / IndentationError / compile error text in the test output: trivially detectable, counted separately",
    "quality measures (reported, no threshold moved later)": [
        "share of mutants WORSE / EQUAL_ON_TESTS / UNKNOWN, overall, by operator and by language",
        "share of WORSE that are crashes",
        "states with >= 2 non-crash WORSE mutants (enough for a pool of Fable's action + 2 alternatives)"],
    "usable_if (pre-registered)": "at least 15 of 20 states have >= 2 non-crash WORSE mutants AND crashes are at most half of all WORSE",
    "next (separate GO)": "pools of Fable's action + up to 4 non-crash WORSE mutants scored by the frozen baseline (laya-baseline-v1, each adapter alone), the linear "
                          "probe and an options-only control; needs a Kaggle feature capture and upload, which need the user's GO",
}

LINE_COMMENT = re.compile(r"^\s*(#|//|/\*|\*)")
QUOTED = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`')
TRAIL = re.compile(r"\s(#|//)")
OPS = [
    ("cmp", re.compile(r"(?<![=!<>])(==|!=)(?!=)"), {"==": "!=", "!=": "=="}),
    ("cmp", re.compile(r"(===|!==)"), {"===": "!==", "!==": "==="}),
    ("cmp", re.compile(r"(?<= )(<=|>=|<|>)(?= )"), {"<=": "<", ">=": ">", "<": "<=", ">": ">="}),
    ("bool", re.compile(r"\b(True|False|true|false)\b"), {"True": "False", "False": "True", "true": "false", "false": "true"}),
    ("logic", re.compile(r"(?<= )(and|or)(?= )|(&&|\|\|)"), {"and": "or", "or": "and", "&&": "||", "||": "&&"}),
    ("num", re.compile(r"(?<![\w.])(\d+)(?![\w.])"), None),
    ("arith", re.compile(r"(?<= )(\+|-)(?= )"), {"+": "-", "-": "+"}),
]
CRASH = re.compile(r"SyntaxError|IndentationError|TabError|error TS\d+|ERR_INVALID|Unexpected token|syntax error", re.I)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sites(text):
    """-> [(operator, start, end, replacement)] on code only."""
    out, pos = [], 0
    for line in text.split("\n"):
        if not LINE_COMMENT.match(line):
            masked = QUOTED.sub(lambda m: " " * len(m.group(0)), line)
            t = TRAIL.search(masked)
            code = masked[:t.start()] if t else masked
            for name, rx, table in OPS:
                for m in rx.finditer(code):
                    g = next(i for i in range(1, (m.lastindex or 0) + 1) if m.group(i) is not None)
                    old = m.group(g)
                    new = str(int(old) + 1) if table is None else table[old]
                    out.append((name, pos + m.start(g), pos + m.end(g), new))
        pos += len(line) + 1
    return out


def mutants(dec_id, own):
    """own: Fable's write/edit steps. -> [dict(id, op, step, before, after, steps)]"""
    pool = []
    for si, st in enumerate(own):
        key = "content" if st["effect"] == "write" else "new_string"
        text = (st["arguments"] or {}).get(key) or ""
        for op, a, b, new in sites(text):
            pool.append(dict(op=op, step=si, key=key, a=a, b=b, new=new, text=text, h=sha("%s|%s|%d|%d" % (dec_id, op, si, a))))
    by = collections.defaultdict(list)
    for p in sorted(pool, key=lambda p: p["h"]):
        by[p["op"]].append(p)
    order, seen, out = [], set(), []
    while any(by.values()):
        for op in sorted(by):
            if by[op]:
                order.append(by[op].pop(0))
    for p in order:
        if len(out) >= MAX_MUTANTS:
            break
        mt = p["text"][:p["a"]] + p["new"] + p["text"][p["b"]:]
        if mt in seen or mt == p["text"]:
            continue
        seen.add(mt)
        steps = [dict(s, arguments=dict(s["arguments"], **{p["key"]: mt})) if i == p["step"] else s for i, s in enumerate(own)]
        ctx = p["text"][max(0, p["a"] - 60):p["b"] + 60]
        out.append(dict(id="%s#m%s" % (dec_id, p["h"][:10]), op=p["op"], step=p["step"], offset=p["a"], before=p["text"][p["a"]:p["b"]], after=p["new"],
                        context_before=ctx, steps=steps))
    return out


def run(state_ws, steps, test, timeout=W.TIMEOUT):
    x = W.replay(dict(steps=steps + [test]), state_ws, timeout=timeout)
    t = x["log"][-1] if x["log"] else {}
    return dict(verdict=x["verdict"], exit=t.get("ours_exit"), tail=t.get("ours_tail", ""))


def same(p, q):
    return p["exit"] is not None and q["exit"] is not None and W.compare(p["exit"], p["tail"], q["exit"], q["tail"])["match"]


def canonical_train():
    sp = {}
    for f in ("private_labels.jsonl.gz", "excluded.jsonl.gz"):
        for l in gzip.open(SPLIT_MAP / f, "rt", encoding="utf-8"):
            r = json.loads(l)
            sp[r["trajectory_id"]] = r["split"]
    return sp


def main(resume=None):
    out = Path(resume) if resume else ROOT / ("results/raw/mutation-pilot-v1-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    if not resume:
        out.mkdir(parents=True)
        (out / "PLAN.json").write_text(json.dumps(PLAN, indent=1), "utf-8")
    W.RUNS.mkdir(parents=True, exist_ok=True)
    sp = canonical_train()
    rows = {json.loads(l)["decision_id"]: json.loads(l) for l in gzip.open(STEPS / "steps.jsonl.gz", "rt", encoding="utf-8")}
    verdict = [{json.loads(l)["decision_id"]: json.loads(l)["verdict"] for l in open(g / "rows.jsonl", encoding="utf-8")} for g in GATES]
    pop = sorted((d for d in rows if all(v.get(d) == "MATCH" for v in verdict) and sp.get(d.split(":")[1]) == "train"), key=sha)
    split_check = collections.Counter(sp.get(d.split(":")[1], "?") for d in rows if all(v.get(d) == "MATCH" for v in verdict))
    print("replayable by canonical split:", dict(split_check), "| canonical TRAIN population:", len(pop), flush=True)

    def done(name):
        p = out / name
        return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []

    screened = {s["decision_id"]: s for s in done("screen.jsonl")}
    sink = open(out / "screen.jsonl", "a", encoding="utf-8")
    for d in pop:
        if sum(s["selected"] for s in screened.values()) >= N_STATES:
            break
        if d in screened:
            continue
        r = rows[d]
        dec = json.loads((STATES / r["key"] / "decision.json").read_text("utf-8"))
        n_own = len(dec["recorded_calls"])
        own, test = r["steps"][:n_own], next(s for s in r["steps"] if s.get("is_test"))
        between = [s["effect"] for s in r["steps"][n_own:] if not s.get("is_test") and s["effect"] in ("write", "edit", "shell")]
        ws = STATES / r["key"] / "workspace"
        rec = dict(decision_id=d, lang=r["lang"], key=r["key"], split="train", own_effects=[s["effect"] for s in own], other_changes_before_test=len(between),
                   test_command=test["arguments"].get("command", "")[:300])
        if not all(s["effect"] in ("write", "edit") for s in own) or between:
            rec.update(selected=False, reason="not_only_fable_write_edit_before_test")
        else:
            b, a = [run(ws, [], test) for _ in range(2)], [run(ws, own, test) for _ in range(2)]
            stable = same(b[0], b[1]) and same(a[0], a[1])
            rec.update(baseline_exit=[x["exit"] for x in b], action_only_exit=[x["exit"] for x in a], stable=stable, useful=bool(stable and not same(b[0], a[0])),
                       action_only_tail=a[0]["tail"][-800:])
            rec["selected"] = bool(rec["useful"] and a[0]["exit"] == 0)
            rec["reason"] = "selected" if rec["selected"] else ("unstable" if not stable else "oracle_not_useful" if not rec["useful"] else "action_only_not_exit_0")
        screened[d] = rec
        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sink.flush()
        print("SCREEN", d, r["lang"], rec["reason"], flush=True)
    sink.close()
    chosen = [s for s in screened.values() if s["selected"]][:N_STATES]

    finished = {m["id"] for m in done("mutants.jsonl")}
    jobs = []
    for s in chosen:
        r = rows[s["decision_id"]]
        dec = json.loads((STATES / r["key"] / "decision.json").read_text("utf-8"))
        own, test = r["steps"][:len(dec["recorded_calls"])], next(x for x in r["steps"] if x.get("is_test"))
        ref = dict(exit=0, tail=s["action_only_tail"])
        for m in mutants(s["decision_id"], own):
            if m["id"] not in finished:
                jobs.append((s, r, test, ref, m))
    print("mutants to run:", len(jobs), flush=True)
    msink = open(out / "mutants.jsonl", "a", encoding="utf-8")

    def one(job):
        s, r, test, ref, m = job
        try:
            x = [run(STATES / r["key"] / "workspace", m["steps"], test, timeout=MUT_TIMEOUT) for _ in range(2)]
            stable = same(x[0], x[1])
            label = ("UNKNOWN" if not stable or any(y["verdict"].startswith(("UNREPLAYABLE", "ERROR")) for y in x)
                     else "EQUAL_ON_TESTS" if same(x[0], ref) else "WORSE" if x[0]["exit"] not in (0, None) else "UNKNOWN")
            res = dict(exits=[y["exit"] for y in x], stable=stable, label=label, crash=bool(CRASH.search(x[0]["tail"])), tail=x[0]["tail"][-600:])
        except Exception as e:  # one broken mutant must not stop the run
            res = dict(exits=None, stable=False, label="UNKNOWN", crash=False, tail="", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        row = dict(id=m["id"], decision_id=s["decision_id"], lang=s["lang"], op=m["op"], step=m["step"], offset=m["offset"], before=m["before"], after=m["after"],
                   context_before=m["context_before"], steps=m["steps"], **res)
        msink.write(json.dumps(row, ensure_ascii=False) + "\n")
        msink.flush()
        print("MUT", row["id"], row["op"], row["label"], "crash" if row["crash"] else "", row["exits"], flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, jobs))
    msink.close()

    ms = done("mutants.jsonl")
    tally = lambda xs: dict(collections.Counter(x["label"] for x in xs))
    worse = [m for m in ms if m["label"] == "WORSE"]
    per = collections.Counter(m["decision_id"] for m in worse if not m["crash"])
    ok_states = sum(per[s["decision_id"]] >= 2 for s in chosen)
    crash_share = round(sum(m["crash"] for m in worse) / len(worse), 4) if worse else None
    summary = dict(canonical_split_of_replayable_states=dict(split_check), canonical_train_population=len(pop), screened=len(screened),
                   screen_reasons=dict(collections.Counter(s["reason"] for s in screened.values())), selected=len(chosen),
                   selected_by_lang=dict(collections.Counter(s["lang"] for s in chosen)), mutants=len(ms), labels=tally(ms),
                   labels_by_operator={op: tally([m for m in ms if m["op"] == op]) for op in sorted({m["op"] for m in ms})},
                   labels_by_lang={l: tally([m for m in ms if m["lang"] == l]) for l in sorted({m["lang"] for m in ms})},
                   worse_crash_share=crash_share, non_crash_worse_per_state={s["decision_id"]: per[s["decision_id"]] for s in chosen},
                   states_with_2_or_more_non_crash_worse=ok_states,
                   usable=bool(len(chosen) == N_STATES and ok_states >= 15 and crash_share is not None and crash_share <= 0.5),
                   usable_rule=PLAN["usable_if (pre-registered)"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary, indent=1))
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main(*sys.argv[1:])
