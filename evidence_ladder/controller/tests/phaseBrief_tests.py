"""[brief-thinking] CPU tests: the v3 prompt adds a brief-reasoning instruction, leaves v1/v2 untouched, and the proposer uses it without any token cutoff of its own."""
import json, sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from evidence_ladder.controller import generation as G  # noqa: E402


def t_v3_adds_brief_instruction_and_v1_v2_are_frozen():
    assert G.ACTION_INSTR_V3 == G.BRIEF_THINKING_INSTR + G.ACTION_INSTR_V2 and "BRIEFLY" in G.ACTION_INSTR_V3
    assert G.ACTION_TEMPLATE_ID_V3 == "qwen_actions_v3" and G.ACTION_TEMPLATE_HASH_V3 != G.ACTION_TEMPLATE_HASH_V2
    assert G.ACTION_TEMPLATE_ID == "qwen_actions_v1" and G.ACTION_TEMPLATE_ID_V2 == "qwen_actions_v2" and "BRIEFLY" not in G.ACTION_INSTR_V2 and "BRIEFLY" not in G.ACTION_INSTR


def t_v3_proposer_renders_the_v3_prompt_and_has_no_thinking_cutoff():
    assert G.ActionProposerV3.TEMPLATE_ID == "qwen_actions_v3" and G.ActionProposerV3.INSTR == G.ACTION_INSTR_V3 and issubclass(G.ActionProposerV3, G.ActionProposerV2)
    prompt, _ = G.render_action_prompt("state text", G.ActionProposerV3.INSTR)
    assert prompt.startswith("state text") and "BRIEFLY" in prompt
    import inspect; src = inspect.getsource(G.ActionProposerV3)
    assert "think_budget" not in src and "FORCE_CLOSE" not in src          # no cutoff logic here: only the caller's safety ceiling


if __name__ == "__main__":
    n, failed = 0, {}
    for k, f in sorted(globals().items()):
        if k.startswith("t_") and callable(f):
            n += 1
            try: f()
            except BaseException as e: failed[k] = "".join(traceback.format_exception_only(type(e), e))[-300:]  # noqa: BLE001
    print(json.dumps({"tests": n, "passed": n - len(failed), "failed": failed}, indent=1)); sys.exit(1 if failed else 0)
