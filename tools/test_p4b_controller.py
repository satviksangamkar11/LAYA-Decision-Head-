"""[P4B] End-to-end controller test on the REAL MXC process container with a scripted model (no GPU, no network).
Task C14 (frozen contract cohort). Candidate order per request: (1) hard-codes the 4 public examples, returns -1 otherwise (passes public, violates the contract);
(2) correct bisect. With contracts ON (V1.5) the contract check must block (1) and the answer must be (2); with contracts OFF (V1) the answer is (1).
Usage: PYTHONPATH=. .venv/Scripts/python.exe tools/test_p4b_controller.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from evidence_ladder.common.interfaces import ControllerContext, TaskSpec          # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController        # noqa: E402
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend   # noqa: E402
from evidence_ladder.models import qwen as Q                                       # noqa: E402

HARDCODE = ("def solve(data):\n    nums, t = data['nums'], data['target']\n    table = {((), 5): 0, ((1, 3, 5, 6), 5): 2, ((1, 3, 5, 6), 2): 1, ((1, 3, 5, 6), 7): 4}\n"
            "    return table.get((tuple(nums), t), -1)\n")
CORRECT = "import bisect\ndef solve(data):\n    return bisect.bisect_left(data['nums'], data['target'])\n"


class Gen:
    def __init__(self):
        self.k = 0

    def generate(self, rq):
        out = []
        for s in list(rq.seeds)[:rq.n]:
            code = HARDCODE if self.k % 2 == 0 else CORRECT
            self.k += 1
            out.append(Q.make_generation("<think>\n\n</think>\n\n```python\n%s```\n" % code, tokens_prompt=100, tokens_generated=80, seed=s,
                                         prompt_template_id=rq.prompt_template_id, prompt_sha256="x", language="python"))
        return out

    def model_info(self):
        return {"backend": "scripted"}


def run(contracts):
    t = next(json.loads(l) for l in (ROOT / "configs/p5_contract_tasks_v1/tasks.jsonl").read_text().splitlines() if json.loads(l)["task_id"] == "C14")
    task = TaskSpec("C14", "p5:C14", "python", "python", t["prompt"], t["public_tests"], "medium")
    exe = ProviderExecutionBackend(backend="mxc", scratch=Path(tempfile.mkdtemp(prefix="p4b-")), trusted=True)
    cfg = {"mode": "ladder", "k": 2, "l2": False, "thinking": False, "discriminate": False, "critic": False, "max_new_tokens": 512, "max_rounds": 1}
    if contracts:
        cfg["contracts"] = {"registry": str(ROOT / "configs/p5_contract_tasks_v1/REGISTRY_v1.json"), "root": str(ROOT), "n_derived": 6}
    ctx = ControllerContext(Gen(), exe, Q.TokenBudget("C14", "D0", 100000), "D0", None, None, (1, 2, 3, 4), cfg)
    ctl = EvidenceLadderController()
    res = ctl.run_problem(task, ctx)
    j = ctl.last_journal
    notes = [e["data"] for e in j.entries("contract_check")] if hasattr(j, "entries") else []
    chosen = next((e["data"]["candidate_id"] for e in j.entries("decision")), None)
    src = {e["data"]["id"]: e["data"]["source"] for e in j.entries("candidate")}
    return {"label": res.answer_label, "chosen_is_correct_program": (src.get(chosen) or "").strip() == CORRECT.strip(), "chosen_is_hardcode": (src.get(chosen) or "").strip() == HARDCODE.strip(),
            "contract_checks": notes, "registry_has_contract_check": "contract_checks_v1" in ctl._registry.checks}


if __name__ == "__main__":
    v15, v1 = run(True), run(False)
    print(json.dumps({"V1.5": v15, "V1": v1}, indent=1))
    ok = (v15["label"] in ("SUPPORTED", "VERIFIED") and v15["chosen_is_correct_program"] and any(n["status"] == "fail" and n["authority"] == "authoritative" for n in v15["contract_checks"])
          and v15["registry_has_contract_check"] and v1["chosen_is_hardcode"] and not v1["registry_has_contract_check"] and not v1["contract_checks"])
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
