"""Mutation pilot v1.1 (user GO, chat 2026-10-13): the SAME 20 states and the SAME mutants as v1, re-run with the audited evaluation commands.

Reason for the new version (written after v1, before v1.1 ran): v1 labelled WORSE by the exit code of the recorded test command. In 5 of the 8 TS states
that command ends in `2>&1 | tail -N`, so the exit code is tail's (always 0): 25 of 37 TS UNKNOWN mutants show assertion failures with exit 0, and Fable's
own action in ts-msgfmt:3 shows failures while exiting 0. Repair: the evaluation command from
results/raw/fable-fidelity-report-20261010/evaluation_command_audit.jsonl (status VERIFIED_DISPLAY_ONLY: only the display filter removed). States not in the
audit keep their recorded command only if it contains no pipe; otherwise the state is INVALID. Nothing else changes: same mutants (regenerated
deterministically and checked against v1 ids), same labels, same time limit, same usable rule. A state whose Fable action does not pass (stable, differs from
the baseline, exit 0) under the evaluation command is INVALID and still counts in the denominator of 20 (no replacement). v1 stays on record unchanged.

  wsl -d Ubuntu-24.04 -u root -e python3 "/mnt/d/local model/tools/mutation_pilot_v1_1.py" [<existing out dir to resume>]
"""
import collections
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fable_wsl_runner as W  # noqa: E402
import mutation_pilot_v1 as V  # noqa: E402

V1 = ROOT / "results/raw/mutation-pilot-v1-20261012T215655Z"
AUDIT = ROOT / "results/raw/fable-fidelity-report-20261010/evaluation_command_audit.jsonl"
PLAN = dict(V.PLAN, version="v1.1", written="2026-10-13, after v1 and before v1.1 ran", reason=__doc__.split("\n\n")[1],
            approved_by="user, chat 2026-10-13: GO mutation pilot v1.1; preserve v1; rerun the same mutants; verify Fable's action passes; an invalid state "
                        "stays invalid and is not replaced",
            states="the 20 states selected by v1 (v1 screen.jsonl), unchanged")


def main(resume=None):
    out = Path(resume) if resume else ROOT / ("results/raw/mutation-pilot-v1.1-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    if not resume:
        out.mkdir(parents=True)
        (out / "PLAN.json").write_text(json.dumps(PLAN, indent=1), "utf-8")
    W.RUNS.mkdir(parents=True, exist_ok=True)
    audit = {json.loads(l)["decision_id"]: json.loads(l) for l in open(AUDIT, encoding="utf-8")}
    rows = {json.loads(l)["decision_id"]: json.loads(l) for l in gzip.open(V.STEPS / "steps.jsonl.gz", "rt", encoding="utf-8")}
    chosen = [s for s in map(json.loads, open(V1 / "screen.jsonl", encoding="utf-8")) if s["selected"]]
    v1_ids = collections.defaultdict(set)
    for m in map(json.loads, open(V1 / "mutants.jsonl", encoding="utf-8")):
        v1_ids[m["decision_id"]].add(m["id"])
    assert len(chosen) == V.N_STATES

    def done(name):
        p = out / name
        return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []

    states, jobs = {s["decision_id"]: s for s in done("states.jsonl")}, []
    finished = {m["id"] for m in done("mutants.jsonl")}
    sink = open(out / "states.jsonl", "a", encoding="utf-8")
    for s in chosen:
        d = s["decision_id"]
        r = rows[d]
        dec = json.loads((V.STATES / r["key"] / "decision.json").read_text("utf-8"))
        own, test = r["steps"][:len(dec["recorded_calls"])], next(x for x in r["steps"] if x.get("is_test"))
        hist = test["arguments"].get("command", "")
        a = audit.get(d)
        cmd = a["evaluation_command"] if a and a["status"] in ("VERIFIED_DISPLAY_ONLY", "NO_FILTER") else (hist if "|" not in hist else None)
        muts = V.mutants(d, own)
        if d not in states:
            rec = dict(decision_id=d, lang=s["lang"], historical_command=hist, evaluation_command=cmd, audit_status=a["status"] if a else "NOT_IN_AUDIT",
                       same_mutants_as_v1={m["id"] for m in muts} == v1_ids[d], n_mutants=len(muts))
            if cmd is None:
                rec.update(valid=False, reason="unaudited_pipe")
            else:
                t = dict(test, arguments=dict(test["arguments"], command=cmd))
                ws = V.STATES / r["key"] / "workspace"
                b, x = [V.run(ws, [], t) for _ in range(2)], [V.run(ws, own, t) for _ in range(2)]
                stable = V.same(b[0], b[1]) and V.same(x[0], x[1])
                rec.update(baseline_exit=[y["exit"] for y in b], action_only_exit=[y["exit"] for y in x], stable=stable,
                           useful=bool(stable and not V.same(b[0], x[0])), action_only_tail=x[0]["tail"][-800:])
                rec["valid"] = bool(rec["useful"] and x[0]["exit"] == 0)
                rec["reason"] = "valid" if rec["valid"] else ("unstable" if not stable else "oracle_not_useful" if not rec["useful"] else "fable_action_fails_tests")
            states[d] = rec
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            print("STATE", d, rec["reason"], rec.get("action_only_exit"), "same mutants" if rec["same_mutants_as_v1"] else "MUTANTS DIFFER", flush=True)
        rec = states[d]
        if rec["valid"]:
            t = dict(test, arguments=dict(test["arguments"], command=rec["evaluation_command"]))
            jobs += [(s, r, t, dict(exit=0, tail=rec["action_only_tail"]), m) for m in muts if m["id"] not in finished]
    sink.close()
    print("mutants to run:", len(jobs), flush=True)
    msink = open(out / "mutants.jsonl", "a", encoding="utf-8")

    def one(job):
        s, r, test, ref, m = job
        try:
            x = [V.run(V.STATES / r["key"] / "workspace", m["steps"], test, timeout=V.MUT_TIMEOUT) for _ in range(2)]
            stable = V.same(x[0], x[1])
            label = ("UNKNOWN" if not stable or any(y["verdict"].startswith(("UNREPLAYABLE", "ERROR")) for y in x)
                     else "EQUAL_ON_TESTS" if V.same(x[0], ref) else "WORSE" if x[0]["exit"] not in (0, None) else "UNKNOWN")
            res = dict(exits=[y["exit"] for y in x], stable=stable, label=label, crash=bool(V.CRASH.search(x[0]["tail"])), tail=x[0]["tail"][-600:])
        except Exception as e:  # one broken mutant must not stop the run
            res = dict(exits=None, stable=False, label="UNKNOWN", crash=False, tail="", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        row = dict(id=m["id"], decision_id=s["decision_id"], lang=s["lang"], op=m["op"], step=m["step"], offset=m["offset"], before=m["before"], after=m["after"],
                   context_before=m["context_before"], steps=m["steps"], **res)
        msink.write(json.dumps(row, ensure_ascii=False) + "\n")
        msink.flush()
        print("MUT", row["id"], row["op"], row["label"], "crash" if row["crash"] else "", row["exits"], flush=True)

    with ThreadPoolExecutor(max_workers=V.WORKERS) as pool:
        list(pool.map(one, jobs))
    msink.close()

    ms = done("mutants.jsonl")
    tally = lambda xs: dict(collections.Counter(x["label"] for x in xs))
    worse = [m for m in ms if m["label"] == "WORSE"]
    per = collections.Counter(m["decision_id"] for m in worse if not m["crash"])
    ok = sum(per[d] >= 2 for d, s in states.items() if s["valid"])
    crash_share = round(sum(m["crash"] for m in worse) / len(worse), 4) if worse else None
    v1m = {m["id"]: m["label"] for m in map(json.loads, open(V1 / "mutants.jsonl", encoding="utf-8"))}
    summary = dict(version="v1.1", states=len(states), valid_states=sum(s["valid"] for s in states.values()),
                   invalid_states={d: s["reason"] for d, s in states.items() if not s["valid"]},
                   all_mutants_identical_to_v1=all(s["same_mutants_as_v1"] for s in states.values()),
                   mutants=len(ms), labels=tally(ms), labels_by_operator={op: tally([m for m in ms if m["op"] == op]) for op in sorted({m["op"] for m in ms})},
                   labels_by_lang={l: tally([m for m in ms if m["lang"] == l]) for l in sorted({m["lang"] for m in ms})},
                   label_change_from_v1=dict(collections.Counter("%s -> %s" % (v1m.get(m["id"]), m["label"]) for m in ms)),
                   worse_crash_share=crash_share, non_crash_worse_per_state={d: per[d] for d in states},
                   states_with_2_or_more_non_crash_worse=ok, denominator=V.N_STATES,
                   usable=bool(ok >= 15 and crash_share is not None and crash_share <= 0.5), usable_rule=V.PLAN["usable_if (pre-registered)"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary, indent=1))
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main(*sys.argv[1:])
