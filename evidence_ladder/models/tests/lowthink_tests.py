"""[lowthink] CPU tests for the thinking-budget path (no model): the force rule, the constructor guard, model_info, the kernel template and the packer. The real behaviour is the Kaggle run itself."""
import json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evidence_ladder.models import qwen as Q  # noqa: E402


def t_force_rule_positive_and_negative():
    assert Q.budget_needs_force("Okay, let's see... the function min(lo, max(x, hi))")           # budget ran out: no closed think block -> force
    assert not Q.budget_needs_force("Okay... so the fix is simple.\n</think>")                   # closed naturally -> nothing is appended
    assert "</think>" in Q.FORCE_CLOSE and Q.FORCE_CLOSE.endswith("\n\n")                        # the forcing text really closes the block, so the strict parser sees a normal answer


def t_constructor_guard_and_model_info():
    kw = dict(authorized=True, env={"EVIDENCE_LADDER_ALLOW_INFERENCE": "1"}, path_exists=lambda p: True)
    for bad in (0, 10, -5, "1024"):
        try:
            Q.HFQwenBackend(think_budget=bad, **kw); raise AssertionError("accepted a bad budget %r" % (bad,))
        except ValueError:
            pass
    assert "think_budget" not in Q.HFQwenBackend(**kw).model_info()                                 # unchanged runs keep their identity (no new keys)
    info = Q.HFQwenBackend(think_budget=1024, answer_tokens=700, **kw).model_info()
    assert info["think_budget"] == 1024 and info["answer_tokens"] == 700 and len(info["force_close_text_sha256"]) == 64


def t_kernel_template_and_packer_carry_the_budget_and_v1_is_untouched():
    low = (ROOT / "kaggle/el_arch1_generate_lowthink_v1.py").read_text("utf-8")
    v1 = (ROOT / "kaggle/el_arch1_generate_v1.py").read_text("utf-8")
    assert "think_budget=REQUEST.get(\"think_budget\")" in low and "__GEN_REQUEST_JSON__" in low
    assert "think_budget" not in v1                                                                 # the running kernel template is unchanged
    assert low.replace('RS["think_budget"] = REQUEST.get("think_budget"); RS["answer_tokens"] = REQUEST.get("answer_tokens", 700)\n    ', "").replace(
        ', think_budget=REQUEST.get("think_budget"), answer_tokens=int(REQUEST.get("answer_tokens", 700))', "") == v1                  # and the variant differs ONLY by the budget lines
    assert "el_arch1_generate_lowthink_v1.py" in (ROOT / "tools/pack_el_arch1_bundle_v1.py").read_text("utf-8")


if __name__ == "__main__":
    res, failed = 0, {}
    for n, f in sorted(globals().items()):
        if n.startswith("t_") and callable(f):
            res += 1
            try: f()
            except BaseException as e:      # noqa: BLE001
                failed[n] = "".join(traceback.format_exception_only(type(e), e))[-300:]
    print(json.dumps({"tests": res, "passed": res - len(failed), "failed": failed}, indent=1)); sys.exit(1 if failed else 0)
