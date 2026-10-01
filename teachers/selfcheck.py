"""Check the judge and conductor wrappers:  laya-audit/.venv/Scripts/python.exe -m teachers.selfcheck   (from the project root)

Uses the real checkpoints on the GPU, offline, pinned by sha256. The six routing cases are the examples printed on the conductor's
model card [author-reported]; the judge and continue cases are my own sanity cases. None of this is a quality measurement.
"""
from teachers.laya_teachers import CodeRelevanceTeacher, ConductorTeacher, JudgeTeacher
from state.schema import NONE_ID

cond, judge, code = ConductorTeacher(), JudgeTeacher(), CodeRelevanceTeacher()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    assert ok, name


# out of lane: no answer, still a valid record
for t, dtype in ((cond, "COMPLETION"), (judge, "ACTION"), (cond, "ACTION"), (code, "ACTION"), (code, "EVIDENCE"), (judge, "CODE_RELEVANCE"), (cond, "CODE_RELEVANCE")):
    o = t.decide("d", dtype)
    check(f"{t.name} is not applicable to {dtype}", not o.applicable and not o.problems() and o.probabilities is None)

# conductor effort: the six printed examples land on the card's tier, records are valid distributions without NONE
cases = [("rename variable x to y", "smol"), ("write a git commit message", "smol"), ("add OAuth login with Google", "default"),
         ("write unit test for parseDate", "default"), ("why does the server crash under load? investigate", "slow"),
         ("design a new multi-tenant billing system", "slow")]
hits = 0
for i, (req, tier) in enumerate(cases):
    o = cond.decide(f"e{i}", "ESCALATE", request=req)
    assert not o.problems() and NONE_ID not in o.candidate_ids, o.problems()
    hits += o.selected_candidate == tier
check("conductor effort: same tier as the card on all 6 printed examples", hits == len(cases), f"{hits}/{len(cases)}")

# conductor continue: an announced next step reads as continue, a finished report and a question read as stop
c1 = cond.decide("c1", "CONTINUE", last_response="I've updated auth.ts. Next I'll update the tests and then the docs.", task="refactor auth module")
c2 = cond.decide("c2", "CONTINUE", last_response="Done. All 14 tests pass and the refactor is complete.", task="refactor auth module")
c3 = cond.decide("c3", "CONTINUE", last_response="Which of the two approaches would you prefer?", task="refactor auth module")
check("conductor continue: announcement -> continue, finished and question -> stop",
      c1.selected_candidate == "continue" and c2.selected_candidate == "stop" and c3.selected_candidate == "stop",
      f"P(continue) {c1.probabilities[0]:.2f} / {c2.probabilities[0]:.2f} / {c3.probabilities[0]:.2f}")

# judge: announced-next-step turns score higher P(incomplete) than a direct answer
req = "Fix the expired-token failure in verifyToken and make sure the tests pass."
ann = judge.decide("j1", "COMPLETION", req, "Fixed the locator. Next I'll run the tests to confirm everything passes.")
ans = judge.decide("j2", "COMPLETION", req, "The failure came from comparing a local-time expiry against a UTC clock; the fix is in verifyToken and all 14 tests pass.")
check("judge: valid records over complete/incomplete", not ann.problems() and not ans.problems() and ann.candidate_ids == ["complete", "incomplete"])
check("judge: announcement scores higher P(incomplete) than a direct answer", ann.probabilities[1] > ans.probabilities[1] + 0.3,
      f"{ann.probabilities[1]:.2f} vs {ans.probabilities[1]:.2f}")
check("records carry revision, calibration label, latency and a raw reference",
      ann.model_revision == judge.revision and ann.calibration_revision == "shipped" and ann.latency_ms > 0 and len(ann.raw_ref) == 16)
# laya-code: independent scores, explicit truncation, lane CODE_RELEVANCE
task = "Fix the expired-token failure in verifyToken"
auth = {"id": "auth", "path": "src/auth.py", "start": 10, "end": 30, "code": "def verify_token(token):\n    claims = decode(token)\n    if claims['exp'] < now():\n        raise TokenExpired()\n    return claims"}
css = {"id": "css", "path": "web/theme.css", "start": 1, "end": 8, "code": ".btn { color: #333; padding: 4px; }\n.footer { margin-top: 12px; }"}
long = {"id": "long", "path": "src/big.py", "start": 1, "end": 900, "code": "x = 1\n" * 600}
one = code.decide("k1", "CODE_RELEVANCE", task, [auth])
many = code.decide("k2", "CODE_RELEVANCE", task, [auth, css, dict(css, id="css2"), dict(css, id="css3"), long])
check("laya-code returns valid INDEPENDENT_SCORES records", not one.problems() and not many.problems() and many.form == "INDEPENDENT_SCORES"
      and many.candidate_ids == [] and many.probabilities is None)
check("laya-code ranks the relevant chunk above unrelated ones", many.scores[0] > many.scores[1] + 0.1, f"auth {many.scores[0]:.2f} vs css {many.scores[1]:.2f}")
check("a chunk's score does not change when other chunks are added (no softmax, no batch coupling)",
      abs(one.scores[0] - many.scores[0]) < 0.02, f"alone {one.scores[0]:.4f} in a batch of 5 {many.scores[0]:.4f}")
check("scores are not normalised across chunks", abs(sum(many.scores) - 1.0) > 1e-3, f"sum {sum(many.scores):.3f}")
check("truncation is explicit and recorded", many.meta["truncated_items"] == ["long"] and one.meta["truncated_items"] == [] and many.meta["state_tokens"] == 128)
check("laya-code record carries revision, calibration label, latency, raw hash",
      many.model_revision == code.revision and many.calibration_revision == "shipped" and many.latency_ms > 0 and len(many.raw_ref) == 16)
print("ALL CHECKS PASSED")
