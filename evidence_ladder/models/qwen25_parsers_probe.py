"""[Qwen2.5-Coder-1.5B local run] Real-model check that the ladder's STRICT parsers accept this non-thinking model's typical output behind the backend's empty-think wrapper:
code extraction (qwen_sample_v2), L2 test array (qwen_tests_v1 -> l2.parse_checks) and action proposals (qwen_actions_v2 -> generation.parse_actions_v2 / parse_actions). Create-only receipt
results/raw/qwen25coder15b-parsers-<UTC>/RECEIPT.json. GPU use user-approved (chat, 2026-10-18)."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PID = "qwen2.5-coder-1.5b-instruct"
TASKS = ROOT / "results" / "raw" / "codex-pilot1-20261016" / "fixed_v2" / "worker" / "tasks.jsonl"


def main() -> int:
    from evidence_ladder.common.interfaces import GenerationRequest, TaskSpec
    from evidence_ladder.controller import generation as GEN, l2 as L2
    from evidence_ladder.models import qwen as Q
    out = ROOT / "results" / "raw" / ("qwen25coder15b-parsers-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True)
    rows = [json.loads(l) for l in TASKS.read_text("utf-8").splitlines() if l.strip()]
    res = {"probe": "qwen25coder15b_parsers_v1", "profile": PID, "tasks": []}
    be = Q.build_local_backend(PID, log_path=str(out / "gen_log.jsonl"))
    for tid in ("P1-03", "P1-11", "P1-15"):
        row = next(r for r in rows if r["task_id"] == tid)
        task = TaskSpec(tid, "py1:" + tid, "python", "python", row["prompt"] + "\n\nSignature: " + row["function_signature"], row["public_tests"])
        p_code, _ = GEN.render_sample_prompt("qwen_sample_v2", task)
        g_code = be.generate(GenerationRequest(tid, p_code, 1, (1,), 700, "sample", True, "qwen_sample_v2"))[0]
        p_tests, _ = Q.render("qwen_tests_v1", task_prompt=task.prompt, examples=Q.format_examples(task.public_tests), n_tests=6)
        g_tests = be.generate(GenerationRequest(tid, p_tests, 1, (1,), 700, "tests", True, "qwen_tests_v1"))[0]
        state = "File: solution.py\n```python\n" + g_code.code + "\n```\nTests failing: test_solution.py::test_x"
        p_act, _ = GEN.render_action_prompt(state, GEN.ACTION_INSTR_V2)
        g_act = be.generate(GenerationRequest(tid, p_act, 1, (1,), 500, "sample", True, GEN.ACTION_TEMPLATE_ID_V2))[0]
        pt, pa, pa2 = L2.parse_checks(g_tests.text, 6), GEN.parse_actions(g_act.text), GEN.parse_actions_v2(g_act.text)
        res["tasks"].append({"task": tid, "code": {"extract_status": g_code.extract_status, "tokens": g_code.tokens_generated, "code_head": g_code.code[:160]},
                             "tests": {"parse_status": pt["status"], "n_checks": len(pt["checks"]), "rejected": pt["rejected"][:3], "text_head": g_tests.text[len(Q.NONTHINK_PREFIX):][:300], "tokens": g_tests.tokens_generated},
                             "actions": {"strict_status": pa["status"], "strict_n": len(pa["actions"]), "v2_status": pa2["status"], "v2_n": len(pa2["actions"]), "v2_salvaged": pa2.get("salvaged"),
                                         "rejected": pa["rejected"][:3], "text_head": g_act.text[len(Q.NONTHINK_PREFIX):][:300], "tokens": g_act.tokens_generated}})
    res["summary"] = {"code_ok": sum(t["code"]["extract_status"] == "ok" for t in res["tasks"]), "tests_ok": sum(t["tests"]["parse_status"] == "ok" for t in res["tasks"]),
                      "actions_strict_ok": sum(t["actions"]["strict_status"] == "ok" for t in res["tasks"]), "actions_v2_ok": sum(t["actions"]["v2_status"] == "ok" for t in res["tasks"]), "n": len(res["tasks"])}
    res["model_info"] = be.model_info()
    with open(out / "RECEIPT.json", "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(res["summary"]), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
