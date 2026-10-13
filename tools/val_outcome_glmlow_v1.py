"""VAL outcome pilot 3: GLM-5.3 at LOW reasoning effort as the candidate generator (user GO, chat 2026-10-13: "use glm itself with low accuracy").
A NEW, separately registered experiment (not a change of the GLM-max pilot's generator). Same problems as the stopped gpt-oss-20b pilot, same blind
requests, decoder, sandbox and amendments 2 + 3; local Qwen capture only after its parity gate; no Kaggle; nothing trained; CAL / TEST untouched.

  verify | gen      (Windows)  then the same decode / execute / capture / score steps as the earlier pilots
"""
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


def verify():
    O.verify(settings=SETTINGS, prereg=PREREG, tag="glmlow")


def gen():
    V.gen(settings=SETTINGS, pilot=O.PROBLEMS, prereg=PREREG, tag="glmlow")


if __name__ == "__main__":
    {"verify": verify, "gen": gen}[sys.argv[1]]()
