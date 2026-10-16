"""VAL outcome pilot 3: GLM-5.3 at LOW reasoning effort as the candidate generator (user GO, chat 2026-10-13: "use glm itself with low accuracy").
A NEW, separately registered experiment (not a change of the GLM-max pilot's generator). Same problems as the stopped gpt-oss-20b pilot, same blind
requests, decoder, sandbox and amendments 2 + 3; local Qwen capture only after its parity gate; no Kaggle; nothing trained; CAL / TEST untouched.

  verify | gen      (Windows)  then the same decode / execute / capture / score steps as the earlier pilots
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import val_outcome_glm_v1 as V  # noqa: E402
import val_outcome_oss20b_v1 as O  # noqa: E402

SETTINGS = dict(model="z-ai/glm-5.3", temperature=1.0, top_p=1.0, reasoning_effort="low", stream=True, max_tokens=32768)
PREREG = dict(O.PREREG,
              written="2026-10-13, before any GLM-5.3 low-effort request was sent",
              approved_by="user, chat 2026-10-13: 'Can you use glm itself with low accuracy' -> agent plan -> 'Go complete report update'",
              generator=dict(SETTINGS, samples=5, why="a weaker setting of the endpoint that worked (GLM-5.3 answered 25 of 25 streamed requests in the "
                             "GLM-max pilot): reasoning_effort low (accepted by this endpoint on 2026-10-11, tools/judge_qual_v1.py 'glm'), temperature 1.0 "
                             "for more variety between the 5 samples, top_p 1.0; fixed before the first call"),
              aim="more pools with BOTH passing and failing valid candidates; all candidates of a pool share one author and one setting",
              limit="a positive selector result shows separation of passing from failing fixes among real model errors; it does not show help for a "
                    "small local model",
              problem_choice="the 5 problems registered for the stopped gpt-oss-20b pilot (positions 6-10 of the hash order); none was ever sent to a model")


PREREG_V2 = dict(PREREG,
                 written="2026-10-13, after the first low-effort verification failed (504 on the second of two simultaneous calls on KEY_2) and before "
                         "any v2 call",
                 approved_by="user, chat 2026-10-13: GO a new, separately registered low-effort GLM verification with sequential API calls; if both "
                             "pass, the already-defined five-problem pilot with unchanged settings",
                 verification="v2: call 1 alone and waited for; call 2 only after call 1 passed; required on each: HTTP 200, finish_reason stop, a "
                               "decodable tool action; any failure stops. Concurrency is a plausible, NOT confirmed, cause of the earlier 504s",
                 monitoring="every send's start, end, key, latency and HTTP status are reconstructed from the ledger, with the number of requests in flight "
                            "on that key at its start, so a recurrence of 504s during generation can be tied to concurrency or not")


PREREG_V3 = dict(PREREG_V2,
                 written="2026-10-13, after verification v2 (call 2 wrapped its action in a JSON list) and before any v3 generation",
                 approved_by="user, chat 2026-10-13: GO glmlow-v3: strict singleton-list normalization, verified on the saved responses, then the bounded "
                             "five-problem pilot; temperature unchanged; no repeated verification loops",
                 normalization="answer text that parses as a JSON list of EXACTLY one object whose keys are EXACTLY {reply, calls} is replaced by that "
                               "object; anything else is passed unchanged to the unchanged v3t decoder (which validates schema and tool safety). The raw "
                               "text is kept next to the normalized text. Applies to v3 only",
                 verification="OFFLINE: the saved responses of verification v2 (and v1 call 2) re-decoded with the normalization must all decode; no new API "
                              "verification calls. Transport failures (no HTTP 200) are retried by the generation loop; format failures are INVALID samples "
                              "and never stop the pilot")


def normalize(text):
    """Strict singleton-list unwrap (PREREG_V3). Returns (text, changed)."""
    try:
        obj = json.loads(text)
    except ValueError:
        return text, False
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict) and set(obj[0]) == {"reply", "calls"}:
        return json.dumps(obj[0], ensure_ascii=False), True
    return text, False


def verify():
    O.verify(settings=SETTINGS, prereg=PREREG, tag="glmlow")


def verify_offline():
    """v3 verification on saved responses only (no API call)."""
    import gzip
    import glob
    import fable_teacher_runner as RR
    ctx = json.loads((O.L2 / "level2_requests/contexts" / (RR.key_of(O.VERIFY_ID) + ".json")).read_text("utf-8"))
    files = sorted(glob.glob(str(V.OUT.parent / "glmlow-v2-verify-*/responses/*.json"))) + sorted(glob.glob(str(V.OUT.parent / "glmlow-verify-*/responses/verify_2.json")))
    res = []
    for f in files:
        r = json.loads(open(f, encoding="utf-8").read())
        answer = (json.loads(r["raw"])["choices"][0]["message"].get("content") or "") if r["status"] == 200 else ""
        text, changed = normalize(answer)
        saved = dict(status="completed", raw_response=json.dumps(dict(choices=[dict(index=0, message=dict(role="assistant", content=text), finish_reason="stop")])))
        main, bad = RR.partition_saved(O.VERIFY_ID, "train", saved, RR.protocol_partition_v3t, ctx, strict_finish=True)
        res.append(dict(file=str(Path(f).relative_to(ROOT)), http=r["status"], normalized=changed, decodable=bool(main), why_not=None if main else (bad[0].get("reason") if bad else None)))
        print(res[-1], flush=True)
    assert all(normalize(x)[1] is False for x in ('{"reply":null,"calls":[]}', '[{"reply":null}]', '[{"reply":null,"calls":[]},{"reply":null,"calls":[]}]', '[1]'))
    passed = bool(res) and all(x["decodable"] for x in res)
    print("OFFLINE VERIFY", "PASSED" if passed else "FAILED")
    return passed, res


def verify_seq():
    """v2: two calls strictly one after the other (one job per run_jobs call, one worker). Stops at the first failure."""
    import gzip
    import json
    from datetime import datetime, timezone
    import judge_qual_v1 as J
    import fable_teacher_runner as RR
    req = next(r for r in (json.loads(l) for l in gzip.open(O.L2 / "upload/requests_level2.jsonl.gz", "rt", encoding="utf-8")) if r["decision_id"] == O.VERIFY_ID)
    ctx = json.loads((O.L2 / "level2_requests/contexts" / (RR.key_of(O.VERIFY_ID) + ".json")).read_text("utf-8"))
    out = V.OUT.parent / ("glmlow-v2-verify-%s" % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir()
    (out / "PREREG.json").write_text(json.dumps(PREREG_V2, indent=1), "utf-8")
    checks = []
    for i in (1, 2):
        J.run_jobs([dict(id="verify_%d" % i, payload=dict(SETTINGS, messages=req["messages"]))], out, inflight=1, numbers=J.key_files()[0].name[15:-4], rpm=V.RPM)
        r = json.loads((out / "responses" / ("verify_%d.json" % i)).read_text("utf-8"))
        ch = json.loads(r["raw"])["choices"][0] if r["status"] == 200 else {}
        answer = (ch.get("message") or {}).get("content") or ""
        saved = dict(status="completed", raw_response=json.dumps(dict(choices=[dict(index=0, message=dict(role="assistant", content=answer), finish_reason="stop")])))
        try:
            main, bad = RR.partition_saved(O.VERIFY_ID, "train", saved, RR.protocol_partition_v3t, ctx, strict_finish=True)
        except Exception as e:
            main, bad = [], [dict(reason="decoder_error:%s" % type(e).__name__)]
        a = main[0]["raw_action"] if main else {}
        c = dict(call=i, alias=r["alias"], http=r["status"], seconds=r["seconds"], finish_reason=ch.get("finish_reason"), answer_chars=len(answer), decodable=bool(main),
                 tools=[x["name"] for x in a.get("calls") or []] or (["reply"] if a.get("reply") else []), why_not=(bad[0].get("reason") if bad and not main else None),
                 answer_head=answer[:300])
        c["passed"] = c["http"] == 200 and c["finish_reason"] == "stop" and c["decodable"]
        checks.append(c)
        print(json.dumps(c, ensure_ascii=False), flush=True)
        if not c["passed"]:
            break
    passed = len(checks) == 2 and all(c["passed"] for c in checks)
    (out / "VERIFY.json").write_text(json.dumps(dict(passed=passed, sequential=True, checks=checks), indent=1, ensure_ascii=False), "utf-8")
    print("VERIFY PASSED" if passed else "VERIFY FAILED", flush=True)
    sys.exit(0 if passed else 1)


def concurrency(ledger_path):
    """From a run_jobs ledger (ts = end, seconds = duration): every send with start, end, key, status and the requests in flight on its key at its start."""
    import json
    from datetime import datetime
    rows = [json.loads(l) for l in open(ledger_path, encoding="utf-8")]
    for r in rows:
        r["end"] = datetime.fromisoformat(r["ts"]).timestamp()
        r["start"] = r["end"] - r["seconds"]
    for r in rows:
        r["in_flight_on_key_at_start"] = sum(1 for x in rows if x["alias"] == r["alias"] and x["start"] <= r["start"] < x["end"])
    return [dict(id=r["id"], alias=r["alias"], status=r["status"], seconds=r["seconds"], in_flight_on_key_at_start=r["in_flight_on_key_at_start"]) for r in rows]


def gen():
    raise SystemExit("glmlow v1 stopped at verification; use gen_v3")


def gen_v3():
    ok, res = verify_offline()
    if not ok:
        raise SystemExit("offline verification failed")
    V.gen(settings=SETTINGS, pilot=O.PROBLEMS, prereg=dict(PREREG_V3, offline_verification=res), tag="glmlow3", normalize=normalize)


if __name__ == "__main__":
    {"verify": verify, "verify_seq": verify_seq, "gen": gen, "gen_v3": gen_v3}[sys.argv[1]]()
