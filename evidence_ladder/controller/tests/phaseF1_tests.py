"""[EL-ARCH-1 F1 / T23-T24] Plain-assert tests for the candidate layer against EL_ARCH_INTERFACE_FREEZE_v1: per-option salvage (A1.1), one independent next action per option and the
sequential-plan rule (A1.2), no-op / unappliable edits (A1.3), structure + scope + dedupe + count (A1.4), explicit abstain (A1.5), frozen result shape (A1.6). No model, no network; every rule
code has a positive (rejected) and a negative (accepted) case. Run from the repo root: python evidence_ladder/controller/tests/phaseF1_tests.py  (prints {"tests","passed","failed"})."""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evidence_ladder.common.interfaces import Generation, sha256_text                          # noqa: E402
from evidence_ladder.controller import candidate_filter as CF, generation as GEN               # noqa: E402
from evidence_ladder.models import qwen as Q                                                   # noqa: E402

TESTS = {}


def test(f):
    TESTS[f.__name__] = f
    return f


def act(tool, **a):
    return {"tool": tool, "arguments": a}


def cand(a, i=0, seed=0):
    return dict(a, origin={"seed": seed, "index": i}, raw=None)


def run(actions, files=None, **kw):
    return CF.filter_options([cand(a, i) for i, a in enumerate(actions)], files, **kw)


def rules(res):
    return [r["rule"] for r in res["rejected"]]


def oid(a):
    return sha256_text(GEN.render_call(a["tool"], a["arguments"]))


def txt(a):
    return GEN.render_call(a["tool"], a["arguments"])


READ = act("Read", file_path="src/clamp.py")
EDIT = act("Edit", file_path="src/clamp.py", old_string="return x", new_string="return max(lo, min(x, hi))")
TESTCMD = act("Bash", command="python -m pytest -q")
LS = act("Bash", command="ls")
FILES = {"src/clamp.py": "def clamp(x, lo, hi):\n    return x\n"}


# ------------------------------------------------------------------ A1.1 per-option salvage
BROKEN = '</think>[{"name":"Read","arguments":{"file_path":"a.py"}},{"name":"Edit","arguments":{"file_path":"a.py","old_string":"say "hi"","new_string":"x"}},{"name":"Bash","arguments":{"command":"ls"}}]'


@test
def a11_salvage_keeps_valid_siblings_and_rejects_the_broken_one():
    p = GEN.parse_actions_v2(BROKEN)
    assert p["salvaged"] and p["status"] == "ok"
    assert p["actions"] == [txt(act("Read", file_path="a.py")), txt(LS)]
    assert len(p["rejected"]) == 1 and p["rejected"][0]["index"] == 1 and "not valid JSON" in p["rejected"][0]["reason"]


@test
def a11_negative_strict_parser_and_prose_stay_strict():
    assert GEN.parse_actions(BROKEN)["actions"] == []                                           # the frozen strict parser is unchanged
    prose = GEN.parse_actions_v2("</think>Sure! " + json.dumps([{"name": "Read", "arguments": {"file_path": "a"}}]))
    assert prose["actions"] == [] and prose["status"] == "not_json_array" and not prose["salvaged"]
    assert GEN.parse_actions_v2('</think>[{"name":"Read","arguments":{"file_path":"a"b"}}]')["status"] == "no_valid_action"


@test
def a11_filter_rejects_a_malformed_item_alone():
    res = CF.filter_options([cand(READ, 0), {"malformed": True, "origin": {"seed": 0, "index": 1}, "raw": "{oops"}, cand(LS, 2)], None)
    assert [a["tool"] for a in res["accepted"]] == ["Read", "Bash"] and rules(res) == ["malformed_option"]
    assert res["rejected"][0]["option_id"] == sha256_text("{oops") and res["rejected"][0]["origin"]["index"] == 1


# ------------------------------------------------------------------ A1.2 one independent next action
@test
def a12_sequential_plan_keeps_first_step_only():
    res = run([READ, EDIT, TESTCMD, act("Bash", command="echo summary")], FILES)
    assert [a["tool"] for a in res["accepted"]] == ["Read"] and rules(res) == ["sequential_plan"] * 3


@test
def a12_negative_independent_alternatives_are_not_a_plan():
    res = run([EDIT, TESTCMD, act("Read", file_path="README.md"), READ], dict(FILES, **{"README.md": "x"}))      # Edit first: no Read -> Edit -> Bash chain
    assert res["counts"]["accepted"] == 4 and res["rejected"] == []
    assert run([READ, EDIT], FILES)["counts"]["accepted"] == 2                                      # Read -> Edit without a later Bash is not a plan


@test
def a12_new_prompt_template_is_versioned_and_v1_is_frozen():
    assert GEN.ACTION_TEMPLATE_ID == "qwen_actions_v1" and GEN.ACTION_TEMPLATE_ID_V2 == "qwen_actions_v2"
    assert "SINGLE next action" in GEN.ACTION_INSTR_V2 and "Do NOT list successive steps" in GEN.ACTION_INSTR_V2
    assert GEN.ACTION_INSTR.startswith("Propose exactly 4 DISTINCT candidate next actions")        # untouched
    assert GEN.render_action_prompt("S")[0] == "S\n\n" + GEN.ACTION_INSTR and GEN.render_action_prompt("S", GEN.ACTION_INSTR_V2)[0].endswith(GEN.ACTION_INSTR_V2)


# ------------------------------------------------------------------ A1.3 no-op / unappliable edits
@test
def a13_rejects_noop_unchanged_absent_missing_and_not_unique_edits():
    e = lambda f, o, n: act("Edit", file_path=f, old_string=o, new_string=n)       # noqa: E731
    res = run([e("src/clamp.py", "clamp@1", "clamp@1"), e("src/clamp.py", "a\r\nb", "a\nb"), e("src/clamp.py", "NOT THERE", "y"), e("src/nope.py", "a", "b"), e("d.py", "x", "y"), EDIT],
              dict(FILES, **{"d.py": "x x"}))
    assert [a["tool"] for a in res["accepted"]] == ["Edit"] and res["accepted"][0]["option_id"] == oid(EDIT)
    assert rules(res) == ["no_op_edit", "unchanged_diff", "old_string_absent", "old_string_absent", "old_string_not_unique"]


@test
def a13_negative_applicable_edit_kept_and_no_files_skips_only_applicability():
    assert run([EDIT], FILES)["counts"]["accepted"] == 1
    assert run([act("Edit", file_path="zz.py", old_string="a", new_string="b")], None)["counts"]["accepted"] == 1          # no snapshot: applicability unknown
    assert rules(run([act("Edit", file_path="zz.py", old_string="a", new_string="a")], None)) == ["no_op_edit"]            # but a no-op needs no file


# ------------------------------------------------------------------ A1.4 structure, scope, dedupe, count
@test
def a14_rejects_bad_structure_scope_and_dedupes():
    items = [{"tool": "Rm", "arguments": {"path": "/"}}, act("Read"), act("Read", file_path=3), act("Read", file_path="  "), act("Read", file_path="../../etc/passwd"),
             act("Read", file_path=".secrets/key.txt"), act("Edit", file_path="tests/test_x.py", old_string="a", new_string="b"), READ, READ]
    res = run(items, None, path_scope={"protected": ["tests/test_x.py"]})
    assert [a["option_id"] for a in res["accepted"]] == [oid(READ)]
    assert rules(res) == ["unknown_tool", "bad_arguments", "bad_arguments", "bad_arguments", "path_outside_scope", "path_outside_scope", "protected_path", "duplicate"]


@test
def a14_negative_risky_actions_are_accepted_for_the_authorization_gate():
    risky = [act("Bash", command="curl http://x"), act("Bash", command="rm -rf build"), act("Bash", command="frobnicate --now"), act("Bash", command="git push origin main")]
    res = run(risky)
    assert res["counts"]["accepted"] == 4 and res["rejected"] == []
    assert run([act("Bash", command="cat .secrets/key.txt")])["counts"]["accepted"] == 1                         # shell paths go to the policy gate, which journals the denial
    assert rules(run([act("Read", file_path="/etc/passwd")])) == ["path_outside_scope"]                           # structured file_path scope is the filter's job


@test
def a14_path_scope_prefixes_and_negative():
    res = run([act("Read", file_path="src/a.py"), act("Read", file_path="docs/a.md")], None, path_scope=["src"])
    assert [a["arguments"]["file_path"] for a in res["accepted"]] == ["src/a.py"] and rules(res) == ["path_outside_scope"]
    assert run([act("Read", file_path="src/a.py")], None, path_scope=["src"])["rejected"] == []


@test
def a14_count_cap_order_and_never_pads():
    many = [act("Read", file_path="f%d.py" % i) for i in range(8)]
    res = run(many, max_options=5)
    assert [a["arguments"]["file_path"] for a in res["accepted"]] == ["f%d.py" % i for i in range(5)] and rules(res) == ["beyond_max_options"] * 3
    one = run([READ])
    assert one["counts"] == {"in": 1, "accepted": 1, "rejected": 0}                                  # one valid option stays one: no invented alternatives up to k=4


# ------------------------------------------------------------------ A1.5 explicit abstain
@test
def a15_abstain_codes_and_negative():
    res = run([act("Edit", file_path="src/clamp.py", old_string="q", new_string="q"), {"tool": "Rm", "arguments": {}}], FILES)
    assert res["accepted"] == [] and res["abstain"]["code"] == "all_options_rejected"
    assert res["abstain"]["reason"] == "no valid option after filtering: 2 of 2 options rejected (no_op_edit=1, unknown_tool=1)", res["abstain"]
    empty = CF.filter_options([], FILES)
    assert empty["abstain"] == {"code": "no_options_proposed", "reason": "no options proposed"}      # exact string next_action.py journals for an empty pool
    assert run([READ])["abstain"] is None


@test
def a15_proposer_v2_end_to_end():
    def backend(text):
        g = Generation(text=text, code="", extract_status="no_code", tokens_prompt=5, tokens_generated=7, seed=0, model="m", model_revision="r", prompt_template_id="qwen_actions_v2",
                       prompt_sha256="x", wall_seconds=0.1)
        return Q.FixtureGenerationBackend([{"task_id": "t:step1", "seed": 0, "purpose": "sample", "generation": Q.generation_to_dict(g)}])

    def jd(acts):
        return "</think>" + json.dumps([{"name": a["tool"], "arguments": a["arguments"]} for a in acts])
    noop = act("Edit", file_path="src/clamp.py", old_string="clamp@1", new_string="clamp@1")
    out = GEN.ActionProposerV2(backend(jd([noop]))).propose("t", 1, "ctx", seeds=(0,), files=FILES)
    assert out["options"] == [] and out["status"] == "no_valid_action" and out["abstain"]["code"] == "all_options_rejected"
    assert out["rejected"][0]["rule"] == "no_op_edit" and out["generations"][0]["template"] == "qwen_actions_v2" and out["filter_results"][0]["counts"]["rejected"] == 1
    out = GEN.ActionProposerV2(backend(jd([LS, READ, EDIT, TESTCMD]))).propose("t", 1, "ctx", seeds=(0,), files=FILES)
    assert out["options"] == [txt(LS), txt(READ)] and out["status"] == "ok" and "abstain" not in out
    out = GEN.ActionProposerV2(backend(BROKEN)).propose("t", 1, "ctx", seeds=(0,))
    assert len(out["options"]) == 2 and any(r["rule"] == "malformed_option" for r in out["rejected"])
    out = GEN.ActionProposerV2(backend("</think>[]")).propose("t", 1, "ctx", seeds=(0,))
    assert out["options"] == [] and out["abstain"]["code"] == "no_options_proposed" and out["abstain_reason"] == "no options proposed"


def _be(text, seed=0):
    g = Generation(text=text, code="", extract_status="no_code", tokens_prompt=5, tokens_generated=7, seed=seed, model="m", model_revision="r", prompt_template_id="qwen_actions_v2",
                   prompt_sha256="x", wall_seconds=0.1)
    return Q.FixtureGenerationBackend([{"task_id": "t:step1", "seed": seed, "purpose": "sample", "generation": Q.generation_to_dict(g)}])


def _jd(acts):
    return "</think>" + json.dumps([{"name": a["tool"], "arguments": a["arguments"]} for a in acts])


DUP_BROKEN = '</think>[{"name":"Read","arguments":{"file_path":"a.py"}},{"name":"Read","arguments":{"file_path":"a.py"}},{"name":"Edit","arguments":{"file_path":"a.py","old_string":"say "hi"","new_string":"x"}},{"name":"Bash","arguments":{"command":"ls"}}]'


@test
def a17_duplicate_surfaces_as_duplicate_never_as_malformed_option():
    out = GEN.ActionProposerV2(_be(DUP_BROKEN)).propose("t", 1, "ctx", seeds=(0,))
    rules_ = [r["rule"] for r in out["rejected"]]
    assert rules_ == ["duplicate", "malformed_option"], rules_                                    # positive: the 2nd Read is `duplicate`; the unescaped quote is the only malformed one
    dup = [r for r in out["rejected"] if r["rule"] == "duplicate"][0]
    assert dup["index"] == 1 and "identical" in dup["reason"]
    assert [r["rule"] for r in out["rejected"] if r["index"] == 1] == ["duplicate"]       # negative: nothing with reason "duplicate" is tagged malformed_option
    out2 = GEN.ActionProposerV2(_be(_jd([READ, READ]))).propose("t", 1, "ctx", seeds=(0,))
    assert [r["rule"] for r in out2["rejected"]] == ["duplicate"] and out2["options"] == [txt(READ)]


@test
def a17_prefilter_false_hands_raw_items_through_untouched():
    real = CF.filter_options

    def boom(*a, **k):
        raise AssertionError("filter_options must not be called when prefilter=False")
    CF.filter_options = boom
    try:
        out = GEN.ActionProposerV2(_be(DUP_BROKEN), prefilter=False).propose("t", 1, "ctx", seeds=(0,), files=FILES)
    finally:
        CF.filter_options = real
    c = out["candidates"]
    assert out["options"] == [] and out["status"] == "ok" and "filter_results" not in out and "abstain" not in out and out["rejected"] == []
    assert [("malformed" in x) for x in c] == [False, False, True, False]                           # generation order; the duplicate is still there; the broken item is a malformed item
    assert c[0]["tool"] == c[1]["tool"] == "Read" and c[0]["arguments"] == c[1]["arguments"] and c[3]["arguments"] == {"command": "ls"}
    assert c[2]["malformed"] is True and c[2]["origin"] == {"seed": 0, "index": 2} and "say" in c[2]["raw"] and all(x["origin"]["seed"] == 0 for x in c)
    res = CF.filter_options(c, FILES)                                                                # the controller's single call then journals every rejection
    assert [r["rule"] for r in res["rejected"]] == ["duplicate", "malformed_option"] and res["counts"]["accepted"] == 2


@test
def a17_prefilter_true_still_filters_and_is_the_default():
    assert GEN.ActionProposerV2(_be("</think>[]")).prefilter is True
    out = GEN.ActionProposerV2(_be(_jd([READ, EDIT, TESTCMD, LS])), prefilter=True).propose("t", 1, "ctx", seeds=(0,), files=FILES)
    assert out["options"] == [txt(READ)] and "candidates" not in out and out["filter_results"][0]["counts"]["rejected"] == 3
    assert [r["rule"] for r in out["rejected"]] == ["sequential_plan"] * 3
    none = GEN.ActionProposerV2(_be("</think>no json"), prefilter=False).propose("t", 1, "ctx", seeds=(0,))
    assert none["options"] == [] and "candidates" not in none and none["status"] == "not_json_array"


# ------------------------------------------------------------------ A1.6 frozen result shape
@test
def a16_result_shape_ids_and_purity():
    import copy
    import inspect
    sig = inspect.signature(CF.filter_options)
    assert list(sig.parameters) == ["options", "files", "max_options", "path_scope"] and sig.parameters["max_options"].default == 5
    opts = [cand(READ, 0), cand(LS, 1)]
    before = copy.deepcopy(opts)
    r1, r2 = CF.filter_options(opts, FILES), CF.filter_options(opts, FILES)
    assert r1 == r2 and opts == before and json.loads(json.dumps(r1)) == r1                         # pure, deterministic, JSON-serialisable, inputs untouched
    assert set(r1) == {"accepted", "rejected", "abstain", "counts"} and set(r1["accepted"][0]) == {"option_id", "tool", "arguments", "action_key", "origin"}
    assert r1["accepted"][0]["option_id"] == oid(READ) and [a["option_id"] for a in r1["accepted"]] == [oid(READ), oid(LS)]
    assert run([act("Read", file_path="./src//a.py")])["accepted"][0]["action_key"] == run([act("Read", file_path="src/a.py")])["accepted"][0]["action_key"]
    assert set(CF.RULES) >= {"malformed_option", "unknown_tool", "bad_arguments", "path_outside_scope", "protected_path", "no_op_edit", "unchanged_diff", "old_string_absent",
                             "old_string_not_unique", "duplicate", "sequential_plan", "beyond_max_options", "repeated_failure"}


def main():
    failed = {}
    for name, f in TESTS.items():
        try:
            f()
        except Exception:      # noqa: BLE001
            failed[name] = traceback.format_exc()[-900:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
