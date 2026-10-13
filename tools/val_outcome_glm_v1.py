"""VAL outcome pilot with GLM-5.3 as the candidate generator (user GO, chat 2026-10-13). Same pool, order, blind requests, decoder, sandbox execution and
outcome rules as tools/cal_outcome_v1.py (amendment 1: the 23 clean fail-to-pass VAL decisions); ONLY the generator changes (Qwen3-4B produced 0
code-change candidates on the same 5 pilot decisions). Nothing is trained. CAL / TEST untouched.

  gen     (Windows) 5 pilot decisions x 5 samples via the NVIDIA endpoint; writes <out>/gen-glm-<ts>/out/gen-0.jsonl in the format cal_outcome_v1 decode reads
  then:   python tools/cal_outcome_v1.py decode results/raw/val-outcome-v1-20261012T134540Z <gen dir>
          wsl -u root python3 tools/cal_outcome_v1.py execute <pilot dir>
"""
import gzip
import re
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


OUT = ROOT / "results/raw/val-outcome-v1-20261012T134540Z"
SAMPLES, INFLIGHT, RPM = 5, 5, 39
SETTINGS = dict(model="z-ai/glm-5.3", temperature=0.5, top_p=1.0, reasoning_effort="max", stream=True, max_tokens=65536)
PREREG = {
    "written": "2026-10-13, before any GLM-5.3 request for a VAL decision was sent",
    "approved_by": "user, chat 2026-10-13: GO GLM-5.3 generation for the five pilot problems (existing blind VAL prompts, full reasoning, five independent "
                   "candidates per problem; generator, prompt, sampling settings and budget fixed); WSL execution; Kaggle feature capture of the five problems' "
                   "contexts and candidates only; no automatic expansion beyond five",
    "pilot": "the 5 pilot decisions of MANIFEST.json (hash order, at most one per trajectory), unchanged from the Qwen pilot",
    "prompt": "requests_all_23_not_uploaded.jsonl.gz messages unchanged (system + user, protocol v3t blind next-turn request; Fable's action not in it)",
    "generation": dict(SETTINGS, samples=SAMPLES, why="the project's glm_max64 setting (tools/judge_qual_v1.py): full reasoning, streamed past the gateway's "
                       "300 s non-streaming limit, 65,536 tokens because 32,768 truncated 11 of 131 reasoning answers on 2026-10-11"),
    "call_logic": "user's fixed NVIDIA logic via judge_qual_v1.run_jobs: up to %d requests in flight per key, <= %d sends per key per minute, a rate-limit "
                  "response cools that key 60 s and re-queues the request; every send logged; a completed id is never resent" % (INFLIGHT, RPM),
    "budget": "%d samples (5 decisions x %d); a sample whose attempt gets no HTTP 200 is retried under a new attempt id (user: do not stop until done; hard stop 20 attempts); rate-limit rejections re-queued; a 200 answer is final; spend Rs 0" % (5 * SAMPLES, SAMPLES),
    "candidate / validity / execution / outcome": "exactly as tools/cal_outcome_v1.py PREREG (decoded by the v3t decoder from the answer text; "
                                                  "finish_reason other than stop = not finished = INVALID)",
    "selectors (fixed now; none chosen on VAL)": {
        "glm_first": "the candidate from sample 1",
        "laya_v4": "coding head + adapter, the 3 VAL-configuration runs of the head-swap experiment rebuilt with saved weights (pick-for-pick gate); mean "
                   "selection success over the 3 seeds",
        "random_head_v4": "the 9 VAL-configuration random-head runs rebuilt likewise; mean selection success over all 9 (no run chosen)",
        "laya_baseline_v1": "secondary: the 3 frozen Fable-trained adapters (334 v3 pools); mean over the 3",
        "linear_probe": "refitted deterministically on TRAIN pools only (the 303 v4 pools for the primary comparison)",
        "random_pick": "exact expectation of a uniform pick"},
    "report": "valid candidate rate, oracle rate (>= 1 passing candidate), first-candidate success, each selector's success; informative decisions (>= 1 pass and "
              ">= 1 fail) separately. Feasibility pilot: no expansion decision from accuracy on five problems",
}


def gen(settings=SETTINGS, pilot=None, prereg=PREREG, tag="glm", reqs=None, samples=SAMPLES):
    """Defaults reproduce the GLM-5.3 pilot exactly; tools/val_outcome_oss20b_v1.py passes its own pre-registered settings and problems."""
    import judge_qual_v1 as J
    reqs = reqs or {r["decision_id"]: r for r in (json.loads(l) for l in gzip.open(OUT / "requests_all_23_not_uploaded.jsonl.gz", "rt", encoding="utf-8"))}
    pilot = pilot or json.loads((OUT / "MANIFEST.json").read_text("utf-8"))["pilot"]
    g = OUT / ("gen-%s-%s" % (tag, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    (g / "out").mkdir(parents=True)
    (g / "PREREG.json").write_text(json.dumps(prereg, indent=1), "utf-8")
    jid = lambda d, s, a: "%s__s%d__a%d" % (d.replace(":", "_"), s, a)
    resp = lambda d, s, a: json.loads((g / "responses" / (jid(d, s, a) + ".json")).read_text("utf-8")) if (g / "responses" / (jid(d, s, a) + ".json")).exists() else None
    # user, chat 2026-10-13: do not stop until done. A sample whose attempt got no HTTP 200 (gateway error, timeout) gets a NEW attempt id; a 200 answer
    # (even one cut at the token limit) is final and never resent. ponytail: 20 attempts per sample as a hard stop against an endpoint that is down.
    attempt, stats = {(d, s): 1 for d in pilot for s in range(1, samples + 1)}, []
    while True:
        todo = []
        for k, a in attempt.items():
            r = resp(*k, a)
            if r is None:
                todo.append(k)
            elif r["status"] != 200 and a < 20:
                attempt[k] = a + 1
                todo.append(k)
        if not todo:
            break
        stats.append(J.run_jobs([dict(id=jid(d, s, attempt[(d, s)]), payload=dict(settings, messages=reqs[d]["messages"])) for d, s in todo], g,
                                inflight=INFLIGHT, rpm=RPM))
        print("round", len(stats), stats[-1], flush=True)
    rows = []
    for d in pilot:
        for s in range(1, samples + 1):
            r = resp(d, s, attempt[(d, s)])
            if not r or r["status"] != 200:
                rows.append(dict(decision_id=d, sample=s, status="http_%s" % (r or {}).get("status"), finished=False, seconds=(r or {}).get("seconds"), text=""))
                continue
            body = json.loads(r["raw"])
            ch = body["choices"][0]
            rows.append(dict(decision_id=d, sample=s, status="ok", finished=ch.get("finish_reason") == "stop", finish_reason=ch.get("finish_reason"),
                             seconds=r["seconds"], usage=body.get("usage"), reasoning_chars=ch["message"].get("reasoning_chars"), text=ch["message"].get("content") or ""))
    with open(g / "out" / "gen-0.jsonl", "x", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    print(json.dumps(stats))
    for x in rows:
        print(x["decision_id"], x["sample"], x["status"], x.get("finish_reason"), x.get("seconds"), len(x["text"]))
    print(str(g.relative_to(ROOT)).replace("\\", "/"))
    return g


AMENDMENT_2 = {
    "written": "2026-10-13, after decoding the GLM candidates and BEFORE any candidate was executed",
    "approved_by": "user, chat 2026-10-13: 'decode them, run them in the sandbox, then the Kaggle capture'",
    "reason": "the cal_outcome_v1 rule counts a candidate as a code change only if EVERY call is Write / Edit (written for single Qwen actions). All 17 "
              "valid GLM candidates bundle the fix with Bash calls, and every one of those Bash calls is a test run, a syntax check (bash -n) or "
              "chmod +x; under the old rule all 17 would be FAIL and the pilot uninformative",
    "rule": "each candidate's calls run IN ORDER inside the qualified sandbox (Write / Edit applied as before, Bash executed like a replay shell step), "
            "then the decision's recorded test command; twice, each from a fresh copy of the pre-action state. PASS = the recorded test exits 0 both "
            "times. A candidate with no Write / Edit is `not a code change` (FAIL). TAMPERING (FAIL): a Write / Edit to a protected file, or a Bash call "
            "naming a protected file together with a write operation (>, >>, tee, sed -i, mv, cp, rm, truncate). Unstable or not applicable = FAIL. "
            "Each problem's untouched state must reproduce the recorded failure (baseline exit non-zero, twice) or the problem is reported as invalid",
    "limit": "Bash side effects beyond the patterns above are not detected; PASS means the recorded test command passes, not a proven correct fix",
}
WRITE_OP = re.compile(r">|\btee\b|\bsed\s+-i|\bmv\b|\bcp\b|\brm\b|\btruncate\b")
AMENDMENT_3 = {
    "written": "2026-10-13, after the amendment-2 run flagged 10 candidates as tampering and BEFORE those candidates were executed",
    "reason": "two measurement bugs, not candidate behaviour: (1) cal_outcome_v1's protected list matches by SUBSTRING of the test command, so the module "
              "under test (permits.py in `python3 test_permits.py`, sitemap.py in `python3 test_sitemap.py`) was protected and every fix to it counted as "
              "tampering; (2) the Bash check fired on any `>`, including the stderr redirect `2>&1`",
    "rule": "protected = a file of the state that matches TESTISH, or whose path or base name is a WHOLE whitespace-separated word of the recorded test "
            "command. A Bash call is tampering only if a write operation targets a protected file directly: `>` / `>>` / `tee [-a]` followed by it, or a "
            "`sed -i` / `mv` / `cp` / `rm` / `truncate` command that names it as a word. Everything else is unchanged from amendment 2. Outcomes go to "
            "OUTCOMES_A3.json; OUTCOMES.json (amendment 2) stays on record",
}


def protected_a3(files, cmd):
    import cal_outcome_v1 as C
    words = set(cmd.split())
    return {p for p in files if C.TESTISH.search(p) or p in words or p.split("/")[-1] in words}


def tampers_a3(cmd, prot):
    for p in prot:
        b = re.escape(p.split("/")[-1])
        if re.search(r"(>>?|\btee(\s+-a)?)\s*[\"']?(\S*/)?" + b + r"\b", cmd) or re.search(r"\b(sed\s+-i\S*|mv|cp|rm|truncate)\b[^;&|]*\b" + b + r"\b", cmd):
            return True
    return False


def execute(res_arg, version="a2"):
    """WSL, root. Amendment 2: every distinct candidate's calls in order, then the recorded test command; twice. Create-only OUTCOMES.json (same schema as cal_outcome_v1)."""
    import fable_wsl_runner as W
    import cal_outcome_v1 as C
    res = ROOT / res_arg
    a3 = version == "a3"
    name = "OUTCOMES_A3.json" if a3 else "OUTCOMES.json"
    (res / ("AMENDMENT_3.json" if a3 else "AMENDMENT_2.json")).write_text(json.dumps(AMENDMENT_3 if a3 else AMENDMENT_2, indent=1), "utf-8")
    assert not (res / name).exists(), "create-only"
    decs = json.loads((res / "CANDIDATES.json").read_text("utf-8"))
    steps = {r["decision_id"]: r for r in (json.loads(l) for l in gzip.open(C.STEPS, "rt", encoding="utf-8"))}
    W.RUNS.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in decs:
        test = next(s for s in steps[d["decision_id"]]["steps"] if s.get("is_test"))
        ws = C.STATES / d["key"] / "workspace"

        def run(st):
            x = W.replay(dict(steps=st + [test]), ws)
            t = x["log"][-1] if x["log"] else {}
            return dict(verdict=x["verdict"], exit=t.get("ours_exit") if t.get("is_test") else None, tail=(t.get("ours_tail") or "")[-500:])

        base = [run([]) for _ in range(2)]
        rows.append(dict(decision_id=d["decision_id"], cid=None, kind="baseline", exits=[b["exit"] for b in base],
                         outcome="PASS" if all(b["exit"] == 0 for b in base) else "FAIL", reproduces_recorded_failure=all(b["exit"] not in (0, None) for b in base)))
        prot = protected_a3([*json.loads((C.STATES / d["key"] / "decision.json").read_text("utf-8"))["files"]], d["test_command"]) if a3 else set(d["protected_files"])
        for c in d["candidates"]:
            a = json.loads(c["action_text"])
            calls = (a.get("calls") if isinstance(a, dict) else a) or []
            st, code, tamper = [], False, False
            for k in calls:
                g = dict(k.get("arguments") or {})
                if k.get("name") in ("Write", "Edit"):
                    code = True
                    g["file_path"] = C._norm(g.get("file_path"))
                    tamper |= g["file_path"] in prot
                    st.append(dict(effect=k["name"].lower(), name=k["name"], arguments=g))
                elif k.get("name") == "Bash":
                    cmd = g.get("command", "")
                    tamper |= tampers_a3(cmd, prot) if a3 else bool(WRITE_OP.search(cmd)) and any(p in cmd or p.split("/")[-1] in cmd for p in prot)
                    st.append(dict(effect="shell", name="Bash", arguments=g, recorded_exit=None, recorded_output=""))
                else:
                    st.append(dict(effect="unreplayable:%s" % k.get("name"), name=k.get("name"), arguments=g))
            row = dict(decision_id=d["decision_id"], cid=c["cid"], first_sample=c["first_sample"], calls=[k.get("name") for k in calls], protected=sorted(prot))
            if not code:
                row.update(kind="not_a_code_change", outcome="FAIL", exits=None)
            elif tamper:
                row.update(kind="tampering", outcome="FAIL", exits=None)
            else:
                r = [run(st) for _ in range(2)]
                if any(x["exit"] is None for x in r):
                    row.update(kind="not_applicable", outcome="FAIL", exits=None, why=r[0]["verdict"])
                elif r[0]["exit"] != r[1]["exit"]:
                    row.update(kind="unstable", outcome="FAIL", exits=[x["exit"] for x in r])
                else:
                    row.update(kind="executed", outcome="PASS" if r[0]["exit"] == 0 else "FAIL", exits=[x["exit"] for x in r], tail=r[0]["tail"])
            rows.append(row)
            print(row["decision_id"], row["cid"], row["kind"], row["outcome"], row.get("exits"), row.get("why", ""), flush=True)
    with open(res / name, "x", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print("baselines", [(r["decision_id"], r["exits"], r["reproduces_recorded_failure"]) for r in rows if r["kind"] == "baseline"])


if __name__ == "__main__":
    {"gen": gen, "execute": execute}[sys.argv[1]](*sys.argv[2:])
