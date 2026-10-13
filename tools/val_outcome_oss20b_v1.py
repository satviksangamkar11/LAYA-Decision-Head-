"""VAL outcome pilot 2: openai/gpt-oss-20b as the candidate generator (user GO, chat 2026-10-13). A NEW, separately registered experiment, not a change
of the GLM pilot's generator. Same blind requests, decoder, sandbox and amendments 2 + 3 (from the start). Nothing is trained; no Kaggle; CAL / TEST untouched.

  verify   2 calls on a canonical TRAIN request (py-incident-cli:1, from the Level 2 TRAIN requests); checks HTTP 200, finish_reason stop, decodable
           tool action. Stops (exit 1) if either check fails.
  gen      the 5 registered VAL problems x 5 samples (tools/val_outcome_glm_v1.gen with these settings)
  then:    python tools/cal_outcome_v1.py decode results/raw/val-outcome-v1-20261012T134540Z <gen dir>
           wsl -u root python3 tools/val_outcome_glm_v1.py execute <pilot dir> a3
"""
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import val_outcome_glm_v1 as V  # noqa: E402

PROBLEMS = ["fable5:py-event-sched:1", "fable5:py-trie-suggest:1", "fable5:ps-shiftbrief:2", "fable5:ps-loadsheet:1", "fable5:ps-fleetpatch:1"]
SETTINGS = dict(model="openai/gpt-oss-20b", temperature=1.0, top_p=0.95, reasoning_effort="high", stream=True, max_tokens=32768)
L2 = ROOT / "results/raw/level12-bundle-20261012T104251Z"
VERIFY_ID = "fable5:py-incident-cli:1"
PREREG = {
    "written": "2026-10-13, before any gpt-oss-20b request was sent",
    "approved_by": "user, chat 2026-10-13: GO execute the GPT-OSS-20B pilot (two TRAIN verification calls, 5 registered problems x 5 samples, amendments 2 "
                   "and 3 from the beginning, local WSL execution, local Qwen capture only after a parity check, no Kaggle, expansion gate fixed)",
    "generator": dict(SETTINGS, samples=5, why="the project's gptoss20_high setting (tools/judge_qual_v1.py: temperature 1.0, top_p 0.95, reasoning_effort "
                      "high, streamed), max_tokens 32,768 (the project's streamed allowance); fixed before the first call"),
    "verification": "2 calls on the canonical TRAIN request %s (Level 2 TRAIN requests, same blind builder); required: HTTP 200, finish_reason stop and "
                    "a tool action decodable by the v3t decoder, on BOTH calls; otherwise stop and report" % VERIFY_ID,
    "problems": PROBLEMS, "problem_choice": "positions 6-10 of the 23-problem hash order in MANIFEST.json (the GLM pilot used 1-5)",
    "prompt": "requests_all_23_not_uploaded.jsonl.gz messages unchanged",
    "call_logic": "judge_qual_v1.run_jobs: <= 5 in flight per key, <= 39 sends per key per minute, a rate-limit response cools that key 60 s and re-queues; a "
                  "sample with no HTTP 200 gets a new attempt id (hard stop 20); a 200 answer is final; all 25 raw responses kept",
    "execution": "tools/val_outcome_glm_v1.py execute <pilot> a3 (amendments 2 and 3 from the start; baseline must reproduce the recorded failure twice)",
    "features": "local GPU capture with tools/val_capture_local.py (same representation as the Kaggle capture); used ONLY after its parity gate passes "
                "against the Kaggle capture of the GLM pilot pools (every hq / ho cosine >= 0.999 and identical top choices); otherwise scoring stops",
    "selectors": "laya_v4 (rebuilt coding head x3), random_head_v4 (rebuilt x9), laya_baseline_v1 (x3), linear_probe (refit on the 303 TRAIN pools), "
                 "first valid candidate, random among valid; raw sample 1 reported separately. End-to-end (invalid = fail) and valid-only reported apart",
    "expansion_gate": "continue to the remaining 13 VAL problems only if at least 2 of the 5 problems have >= 1 PASS and >= 1 FAIL among valid candidates; "
                      "otherwise stop. Generator and threshold are not changed after seeing results",
}


def verify():
    import judge_qual_v1 as J
    import fable_teacher_runner as RR
    req = next(r for r in (json.loads(l) for l in gzip.open(L2 / "upload/requests_level2.jsonl.gz", "rt", encoding="utf-8")) if r["decision_id"] == VERIFY_ID)
    out = V.OUT.parent / ("oss20b-verify-%s" % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir()
    (out / "PREREG.json").write_text(json.dumps(PREREG, indent=1), "utf-8")
    stats = J.run_jobs([dict(id="verify_%d" % i, payload=dict(SETTINGS, messages=req["messages"])) for i in (1, 2)], out, inflight=2, rpm=V.RPM)
    ctx = json.loads((L2 / "level2_requests/contexts" / (RR.key_of(VERIFY_ID) + ".json")).read_text("utf-8"))
    checks = []
    for i in (1, 2):
        r = json.loads((out / "responses" / ("verify_%d.json" % i)).read_text("utf-8"))
        ok200 = r["status"] == 200
        ch = json.loads(r["raw"])["choices"][0] if ok200 else {}
        answer = (ch.get("message") or {}).get("content") or ""
        saved = dict(status="completed", raw_response=json.dumps(dict(choices=[dict(index=0, message=dict(role="assistant", content=answer), finish_reason="stop")])))
        try:
            main, bad = RR.partition_saved(VERIFY_ID, "train", saved, RR.protocol_partition_v3t, ctx, strict_finish=True)
        except Exception as e:
            main, bad = [], [dict(reason="decoder_error:%s" % type(e).__name__)]
        a = main[0]["raw_action"] if main else {}
        checks.append(dict(call=i, http=r["status"], seconds=r["seconds"], finish_reason=ch.get("finish_reason"), answer_chars=len(answer),
                           reasoning_chars=(ch.get("message") or {}).get("reasoning_chars"), decodable=bool(main),
                           tools=[c["name"] for c in a.get("calls") or []] or (["reply"] if a.get("reply") else []),
                           why_not=(bad[0].get("reason") if bad and not main else None), answer_head=answer[:300]))
    passed = all(c["http"] == 200 and c["finish_reason"] == "stop" and c["decodable"] for c in checks)
    (out / "VERIFY.json").write_text(json.dumps(dict(passed=passed, checks=checks, stats=stats), indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps(dict(passed=passed, checks=checks), indent=1, ensure_ascii=False))
    sys.exit(0 if passed else 1)


def gen():
    V.gen(settings=SETTINGS, pilot=PROBLEMS, prereg=PREREG, tag="oss20b")


if __name__ == "__main__":
    {"verify": verify, "gen": gen}[sys.argv[1]]()
