"""[EL-ARCH-1 Phase B] Plain-assert tests for B1 (oracle tiers), B2 (investigate), B3 (generation interface, strict action parser, HF backend logic, Kaggle entry), B4 (scorer hand-off),
B5 (action-risk table), B6 (L2 checks, clustering, critic trigger), B7 (final verification) and B8 (journal records + replay). No network, no keys, no Qwen/LAYA/GLM model: every model is a
stub, so a test that uses one is LOGIC-ONLY (never a REAL PASS). Tests that call the real WSL sandbox are named *_wsl and are REAL for the sandbox/controller path only; they are skipped
(and reported) when `wsl` does not answer. Run from the repo root: python evidence_ladder/controller/tests/phaseB_tests.py  (prints {"tests","passed","failed","skipped"}; exit 1 on failure)."""
import ast
import json
import math
import re
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_tests as RT                                                                  # noqa: E402  controller fakes (FakeGen, FakeExec, FakeCritic, run, ...)
from laya_int1_t6 import TASKS                                                           # noqa: E402  the three toy repos (definitions only)
from evidence_ladder.common.contract_import import C                                     # noqa: E402
from evidence_ladder.common.interfaces import ControllerContext, Generation, GenerationRequest, NotAuthorized, TaskSpec, sha256_text   # noqa: E402
from evidence_ladder.controller import generation as GEN, investigate as INV, l2 as L2, replay as REP, risk as RISK, tiers as TIERS   # noqa: E402
from evidence_ladder.controller import invariants as inv                                 # noqa: E402
from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask, NoOracle   # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController               # noqa: E402
from evidence_ladder.controller.decision import authorize                                # noqa: E402
from evidence_ladder.controller.final_verify import diff_scope                           # noqa: E402
from evidence_ladder.controller.journal import EvidenceJournal                           # noqa: E402
from evidence_ladder.controller.wsl_sandbox import WslSandbox, available                 # noqa: E402
from evidence_ladder.models import gen_entry as GE, qwen as Q                            # noqa: E402

TESTS = {}
SKIPPED = {}
_WSL = {}


def test(f):
    TESTS[f.__name__] = f
    return f


class Skip(Exception):
    pass


def need_wsl():
    if "ok" not in _WSL:
        _WSL["ok"] = available()
    if not _WSL["ok"]:
        raise Skip("wsl does not answer: LOGIC-ONLY environment")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % (exc,))


# ------------------------------------------------------------------ workspaces and sandboxes
def make_ws(task, drop=(), user=None, extra=None):
    T = TASKS[task]
    d = Path(tempfile.mkdtemp(prefix="phaseB-")) / "ws"
    d.mkdir(parents=True)
    for fn, c in T["files"].items():
        if fn not in drop:
            (d / fn).write_text(c, "utf-8")
    for fn, c in (extra or {}).items():
        (d / fn).write_text(c, "utf-8")
    return d, T


def atask(task, ws, T, user=None, scope=None, max_steps=3):
    return AgenticTask(task, user or T["user"], ws, T["test_cmd"], T["protected"], "system text", max_steps, scope)


SLUG_FIX = {"name": "Edit", "arguments": {"file_path": "slug.py", "old_string": 'return re.sub(r"[^a-z0-9]+", "-", text)', "new_string": 'return re.sub(r"[^a-z0-9]+", "-", text).strip("-")'}}
SLUG_BAD = {"name": "Edit", "arguments": {"file_path": "slug.py", "old_string": "text = text.strip().lower()", "new_string": "text = text.strip().upper()"}}
READ_SLUG = {"name": "Read", "arguments": {"file_path": "slug.py"}}
BASH_TEST = {"name": "Bash", "arguments": {"command": "python3 test_slug.py"}}


def opt(a):
    return GEN.render_call(a["name"], a["arguments"])


class ToySandbox:
    """LOGIC-ONLY stand-in for the sandbox (no code is executed): test_slug.py 'passes' iff slug.py ends its return with .strip("-"). Counts calls."""
    def __init__(self, ws_protected=("test_slug.py",)):
        self.calls, self.prot = [], list(ws_protected)
        self.h0 = None

    def run(self, workspace, cmd, persist=False, timeout=60):
        self.calls.append((cmd, persist))
        ws = Path(workspace)
        cur = {p: (ws / p).read_bytes() if (ws / p).exists() else b"" for p in self.prot}
        if self.h0 is None:
            self.h0 = cur
        unchanged = cur == self.h0
        if cmd.strip() == "python3 test_slug.py":
            ok = (ws / "slug.py").is_file() and '.strip("-")' in (ws / "slug.py").read_text("utf-8")
            err = "" if ok else ('Traceback (most recent call last):\n  File "/sandbox/ws/test_slug.py", line 3, in <module>\n    assert slugify("Hello, World!") == "hello-world", slugify("Hello, World!")\n'
                                 'AssertionError: hello-world-\n')
            return dict(exit_code=0 if ok else 1, killed=False, stdout="all tests passed\n" if ok else "", stderr=err, protected_unchanged=unchanged)
        return dict(exit_code=0, killed=False, stdout="ok\n", stderr="", protected_unchanged=unchanged)


class Scorer:
    scorer_id = "fake_laya_logic_only"
    scorer_def = {"weights_hash": "logic-only"}

    def __init__(self, scores=None, error=None):
        self.scores, self.error, self.calls = scores, error, []

    def score(self, decision_type, pool, exploratory=False):
        self.calls.append((decision_type, pool))
        if self.error:
            raise self.error
        return list(self.scores if self.scores is not None else [0.0] * len(pool["options"]))


def run_agentic(task="slug", sandbox=None, scorer=None, config=None, options=None, steps=1, scope=None, drop=(), user=None, extra=None, begin=True):
    ws, T = make_ws(task, drop, user, extra)
    sb = sandbox or ToySandbox()
    r = AgenticRun(atask(task, ws, T, user, scope), AgenticContext(sandbox=sb, scorer=scorer, arm="E", config=dict(config or {})), lambda m: "ctx")
    if begin:
        r.begin()
    for _ in range(steps if options else 0):
        r.step(options, {"new_tokens": 5, "prompt_tokens": 10})
    return r, ws, sb


# ====================================================================== B1
@test
def t_b1_three_examples_route_to_O1():
    for k in TASKS:
        ws, T = make_ws(k)
        f = TIERS.observe_facts(ws, T["user"], T["test_cmd"], T["protected"], TIERS.snapshot(ws))
        r = TIERS.route_tier(f, T["user"], ws)
        assert r["tier"] == "O1" and r["ceiling"] == "VERIFIED" and r["loop"] == "generate-execute-repair" and not r["ask_user"], (k, r)


@test
def t_b1_fault_injection_variants_route_O2_and_O3():
    ws, T = make_ws("slug", drop=("test_slug.py",))                                       # remove the tests
    f = TIERS.observe_facts(ws, T["user"], T["test_cmd"], T["protected"], TIERS.snapshot(ws))
    assert TIERS.route_tier(f, T["user"], ws)["tier"] == "O3"                              # ... and the request has no examples -> O3
    with_ex = T["user"] + ' For example slugify("Hello, World!") == "hello-world" and slugify("  A--B  ") -> "a-b".'
    r = TIERS.route_tier(TIERS.observe_facts(ws, with_ex, T["test_cmd"], T["protected"], TIERS.snapshot(ws)), with_ex, ws)
    assert r["tier"] == "O2" and r["ceiling"] == "SUPPORTED" and "example" in r["reason"] and r["covered"] == ["examples"]
    ws2, T2 = make_ws("slug", extra={"mypy.ini": "[mypy]\n"})                               # tests present: O1 even with extras
    assert TIERS.route_tier(TIERS.observe_facts(ws2, T2["user"], T2["test_cmd"], T2["protected"], TIERS.snapshot(ws2)), T2["user"], ws2)["tier"] == "O1"
    ws3, T3 = make_ws("slug", drop=("test_slug.py",), extra={"mypy.ini": "[mypy]\n"})       # no tests, a type-check config -> O2
    r3 = TIERS.route_tier(TIERS.observe_facts(ws3, T3["user"], T3["test_cmd"], T3["protected"], TIERS.snapshot(ws3)), T3["user"], ws3)
    assert r3["tier"] == "O2" and "type/compile" in r3["reason"]
    ws4, T4 = make_ws("slug", drop=("test_slug.py",), extra={"test_other.py": "from slug import slugify\nassert slugify('a') == 'a'\n"})      # other tests = a regression suite -> O2
    r4 = TIERS.route_tier(TIERS.observe_facts(ws4, T4["user"], T4["test_cmd"], T4["protected"], TIERS.snapshot(ws4)), T4["user"], ws4)
    assert r4["tier"] == "O2" and "regression" in r4["reason"]
    ws5, T5 = make_ws("wordcount", drop=("test_wc.py",))                                    # remove tests AND examples -> O3
    r5 = TIERS.route_tier(TIERS.observe_facts(ws5, T5["user"], T5["test_cmd"], T5["protected"], TIERS.snapshot(ws5)), T5["user"], ws5)
    assert r5["tier"] == "O3" and r5["ceiling"] == "PLAUSIBLE" and r5["ask_user"] is True
    assert TIERS.o3_outcome("t", dict(r5, blocked_until_answered=False))["answer_label"] == "PLAUSIBLE"


@test
def t_b1_model_written_tests_never_upgrade_the_tier():
    ws, T = make_ws("slug", drop=("test_slug.py",))
    base = TIERS.snapshot(ws)
    (ws / "test_slug.py").write_text(TASKS["slug"]["files"]["test_slug.py"], "utf-8")      # the agent writes the test file during the run
    (ws / "test_mine.py").write_text("assert 1 == 1\n", "utf-8")
    f = TIERS.observe_facts(ws, T["user"], T["test_cmd"], T["protected"], base)
    assert not f["tests_given"] and f["regression_suite"] == [] and TIERS.classify(f)["tier"] == "O3"
    assert "test_slug.py" in f["model_artifacts_ignored"] and "test_mine.py" in f["model_artifacts_ignored"]
    ws2, T2 = make_ws("slug")                                                              # an O1 oracle that was edited during the run stops counting
    b2 = TIERS.snapshot(ws2)
    (ws2 / "test_slug.py").write_text("assert True\n", "utf-8")
    assert TIERS.classify(TIERS.observe_facts(ws2, T2["user"], T2["test_cmd"], T2["protected"], b2))["tier"] == "O3"


@test
def t_b1_ceiling_cap_and_journal_only_on_change():
    assert [TIERS.cap_label(l, "O2") for l in ("VERIFIED", "SUPPORTED", "PLAUSIBLE", "UNVERIFIED", "BLOCKED")] == ["SUPPORTED", "SUPPORTED", "PLAUSIBLE", "UNVERIFIED", "BLOCKED"]
    assert TIERS.cap_label("SUPPORTED", "O3") == "PLAUSIBLE" and TIERS.cap_label("VERIFIED", "O1") == "VERIFIED"
    assert TIERS.check_ceiling("VERIFIED", "O2") != "" and TIERS.check_ceiling("BLOCKED", "O3") == ""
    ws, T = make_ws("slug")
    j = EvidenceJournal(None)
    r = TIERS.route_tier(TIERS.observe_facts(ws, T["user"], T["test_cmd"], T["protected"], TIERS.snapshot(ws)), T["user"], ws)
    assert TIERS.journal_tier(j, "intake", r, None) is True and TIERS.journal_tier(j, "again", r, r) is False and len(j.entries("oracle_tier")) == 1


@test
def t_b1_I1_rejects_a_label_above_the_tier_ceiling():
    r, _ws, _sb = run_agentic(options=[opt(READ_SLUG), opt(BASH_TEST)], steps=1)
    res = r.finish()
    assert res.answer_label == "BLOCKED" and r.invariants_checked
    forged = types.SimpleNamespace(answer_label="VERIFIED", final_artifact_id=None, decision={}, stop_cause="decided")
    j = EvidenceJournal(r.registry)
    j.append("oracle_tier", {"tier": "O2"})
    assert inv.check_i1(forged, j) != ""


@test
def t_b1_agentic_o3_asks_the_user_and_takes_no_action():
    sb = ToySandbox()
    r, ws, _ = run_agentic(sandbox=sb, drop=("test_slug.py",), user="Make the text handling nicer somehow.")
    j = r.journal
    assert [e["data"]["tier"] for e in j.entries("oracle_tier")] == ["O3"] and j.entries("ask_user")
    amb = j.entries("ask_user")[0]["data"]["ambiguities"]
    assert any(a.startswith("target_unknown") for a in amb) and any(a.startswith("no_verifiable_criterion") for a in amb)
    raises(NoOracle, r.step, [opt(READ_SLUG)], {})
    res = r.finish()
    assert res.answer_label == "BLOCKED" and res.stop_cause == "missing_information" and "ask_user" in res.stop_details and sb.calls == [] and res.rounds == 0
    assert r.invariants_checked == ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]


@test
def t_b1_agentic_o2_without_test_oracle_points_to_the_ladder():
    with_ex = TASKS["slug"]["user"] + ' For example slugify("Hello, World!") == "hello-world".'
    r, _ws, sb = run_agentic(drop=("test_slug.py",), user=with_ex)
    assert r.journal.entries("oracle_tier")[0]["data"]["tier"] == "O2" and not r.journal.entries("ask_user")
    res = r.finish()
    assert res.stop_cause == "missing_information" and "ladder" in res.stop_details and sb.calls == []


@test
def t_b1_agentic_o1_is_journaled_once_and_stable():
    r, ws, _ = run_agentic(options=[opt(READ_SLUG), opt(BASH_TEST)], steps=2)
    t = r.journal.entries("oracle_tier")
    assert len(t) == 1 and t[0]["data"]["tier"] == "O1" and t[0]["data"]["ceiling"] == "VERIFIED" and t[0]["data"]["reason"].startswith("trusted executable oracle")
    (ws / "test_new.py").write_text("assert True\n", "utf-8")                                # a model-written test file appears: the router is re-run, nothing changes
    r.retier("after-edit")
    assert len(r.journal.entries("oracle_tier")) == 1


@test
def t_b1_ladder_journals_O2_with_examples_and_asks_on_O3():
    res, ctl, gen, _ = RT.run([["OK:a"]], {"k": 1})
    t = ctl.last_journal.entries("oracle_tier")
    assert t and t[0]["data"]["tier"] == "O2" and t[0]["data"]["ceiling"] == "SUPPORTED" and res.answer_label == "SUPPORTED"
    vague = TaskSpec("V1", "py1:V1", "python", "python", "Make it better somehow.", [])
    res, ctl, gen, _ = RT.run([["OK:a"]], {"k": 1}, task=vague)
    assert ctl.last_journal.entries("oracle_tier")[0]["data"]["tier"] == "O3" and ctl.last_journal.entries("ask_user")
    assert res.stop_cause == "missing_information" and res.answer_label == "BLOCKED" and gen.calls == []
    concrete = TaskSpec("V2", "py1:V2", "python", "python", "Write solve(data) that returns the sum of the list data['xs'] and handles the empty list", [])
    res, ctl, gen, _ = RT.run([["OK:a"]], {"k": 1}, task=concrete)
    assert ctl.last_journal.entries("oracle_tier")[0]["data"]["tier"] == "O3" and res.answer_label in ("UNVERIFIED", "BLOCKED") and len(gen.calls) >= 1      # best effort: runs, ceiling PLAUSIBLE


# ====================================================================== B2
@test
def t_b2_parse_failure_formats():
    files, prot = ["slug.py", "test_slug.py", "pkg/mod.py"], ["test_slug.py"]
    tb = 'Traceback (most recent call last):\n  File "/tmp/x/test_slug.py", line 3, in <module>\n    assert slugify("a") == "b"\n  File "/tmp/x/pkg/mod.py", line 9, in f\n    raise ValueError("boom")\nValueError: boom\n'
    f = INV.parse_failure(tb, 1, files, prot)
    assert f["kind"] == "exception" and f["exception"] == "ValueError" and f["message"] == "boom"
    assert f["test_location"]["file"] == "test_slug.py" and f["test_location"]["line"] == 3 and f["code_location"]["file"] == "pkg/mod.py" and f["code_location"]["function"] == "f"
    a = INV.parse_failure('Traceback (most recent call last):\n  File "test_slug.py", line 5, in <module>\n    assert 1 == 2, 1\nAssertionError: 1\n', 1, files, prot)
    assert a["kind"] == "assertion" and a["message"] == "1" and a["code_location"] is None
    s = INV.parse_failure('  File "slug.py", line 2\n    def (:\n        ^\nSyntaxError: invalid syntax\n', 1, files, prot)
    assert s["kind"] == "syntax" and s["code_location"]["file"] == "slug.py"
    p = INV.parse_failure("test_slug.py:12: AssertionError\nE   assert 1 == 2\n", 1, files, prot)
    assert p["kind"] == "assertion" and p["test_location"]["line"] == 12
    assert INV.parse_failure("all tests passed", 0, files, prot)["kind"] == "none"
    assert INV.parse_failure("Killed", 137, files, prot)["kind"] == "timeout" and INV.parse_failure("weird output", 1, files, prot)["kind"] == "unparsed"


@test
def t_b2_investigate_records_L5_observation_and_context_logic_only():
    sb = ToySandbox()
    r, ws, _ = run_agentic(sandbox=sb)
    res = INV.investigate(r)
    j = r.journal
    assert res["failure"]["kind"] == "assertion" and res["failure"]["test_location"]["line"] == 3 and res["targets"] == [{"function": "slugify", "file": "slug.py", "line": 4}]
    ev = [e for e in j.evidence() if e["check_id"] == "protected_tests_v1"]
    assert len(ev) == 1 and ev[0]["status"] == "fail" and ev[0]["source"] == "user_test" and ev[0]["independence"] == "user_provided" and ev[0]["counterexample"]
    out = [e["data"]["outcome"] for e in j.entries("evidence") if e["data"]["evidence"]["check_id"] == "protected_tests_v1"][0]
    assert out["detail"] == "assertion at test_slug.py:3"                                    # the parsed location is part of the L5 record
    assert [e["kind"] for e in j.entries()][-4:].count("investigate") == 1 and j.entries("investigate_sources")
    assert "code under test: slugify (slug.py:4)" in r.messages[-1]["content"] and "no model" in r.messages[-1]["content"]
    assert r.ctx.scorer is None and r.ctx.critic is None                                    # no model component took part


@test
def t_b2_investigate_when_tests_already_pass_logic_only():
    r, ws, _ = run_agentic()
    (ws / "slug.py").write_text((ws / "slug.py").read_text("utf-8").replace("return re.sub", "return re.sub").replace('text)\n', 'text).strip("-")\n'), "utf-8")
    res = INV.investigate(r)
    assert res["failure"]["kind"] == "none" and res["tests_pass"] and r.decided and "already pass" in r.messages[-1]["content"]


@test
def t_b2_investigate_three_toy_repos_wsl():
    need_wsl()
    want = {"slug": ("test_slug.py", 3, "slugify", "slug.py"), "clamp": ("test_clamp.py", 3, "clamp", "clamp.py"), "wordcount": ("test_wc.py", 5, "count_words", "wc.py")}
    hashes = {}
    for k, (tf, line, fn, sf) in want.items():
        for rep in (1, 2):
            ws, T = make_ws(k)
            scratch = Path(tempfile.mkdtemp(prefix="phaseB-sb-"))
            r = AgenticRun(atask(k, ws, T), AgenticContext(sandbox=WslSandbox(scratch, T["protected"]), arm="E"), lambda m: "ctx")
            r.begin()
            res = INV.investigate(r)
            f = res["failure"]
            assert f["kind"] == "assertion" and f["test_location"]["file"] == tf and f["test_location"]["line"] == line, (k, f)
            assert res["targets"] and res["targets"][0]["function"] == fn and res["targets"][0]["file"] == sf
            assert res["exit_code"] != 0 and not res["tests_pass"] and r.journal.evidence()[0]["status"] == "fail"
            hashes.setdefault(k, set()).add(r.journal.journal_hash())
            shutil.rmtree(scratch, ignore_errors=True)
    assert all(len(v) == 1 for v in hashes.values()), hashes                                # deterministic: two investigations give the same journal hash


# ====================================================================== B3
def _gen(text, seed=0, tp=11, tg=22, sha="x", tid="qwen_actions_v1"):
    return Generation(text=text, code="", extract_status="no_code", tokens_prompt=tp, tokens_generated=tg, seed=seed, model="m", model_revision="r", prompt_template_id=tid, prompt_sha256=sha,
                      wall_seconds=0.5)


ACTS = [READ_SLUG, SLUG_FIX, SLUG_BAD, BASH_TEST]


@test
def t_b3_parse_actions_is_strict_and_never_repairs():
    good = "thinking...</think>" + json.dumps(ACTS)
    p = GEN.parse_actions(good)
    assert p["status"] == "ok" and p["actions"] == [opt(a) for a in ACTS] and p["rejected"] == []
    fenced = "t</think>\n" + "`" * 3 + "json\n" + json.dumps(ACTS) + "\n" + "`" * 3
    assert GEN.parse_actions(fenced)["actions"] == p["actions"]
    for bad in ("t</think>Sure! Here you go: " + json.dumps(ACTS), "t</think>" + json.dumps(ACTS) + " Hope this helps.", "t</think>" + json.dumps(ACTS)[:-1],
                "t</think>{}", "t</think>", "no think close " + json.dumps(ACTS)):
        q = GEN.parse_actions(bad)
        assert q["actions"] == [] and q["status"] in ("not_json_array", "no_think_close", "no_valid_action"), (bad[:30], q)
    mixed = [READ_SLUG, {"name": "Rm", "arguments": {"path": "/"}}, {"name": "Edit", "arguments": {"file_path": "a"}}, {"name": "Read", "arguments": {"file_path": 3}}, READ_SLUG, BASH_TEST]
    q = GEN.parse_actions("</think>" + json.dumps(mixed))
    assert q["actions"] == [opt(READ_SLUG), opt(BASH_TEST)] and [r["index"] for r in q["rejected"]] == [1, 2, 3, 4]
    assert len(GEN.parse_actions("</think>" + json.dumps(ACTS * 2), max_actions=3)["actions"]) == 3


@test
def t_b3_prompt_pieces_are_the_frozen_ones():
    try:
        from decision_head import fable_pool as FP
        assert FP.render_call("Bash", {"command": "ls é"}) == GEN.render_call("Bash", {"command": "ls é"})
    except ImportError:
        pass
    src = (ROOT / "kaggle" / "evidence_ladder_laya_t6_step_v1.py").read_text("utf-8")
    tree = ast.parse(src)
    vals = {t.targets[0].id: ast.literal_eval(t.value) for t in tree.body if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name) and t.targets[0].id in ("INSTR",)}
    assert vals["INSTR"] == GEN.ACTION_INSTR
    from evidence_ladder.models import qwen_v2b as V2B
    assert V2B.TEMPLATES_BY_SAMPLE == ("qwen_sample_v2", "qwen_sample_v2", "qwen_sample_v2b", "qwen_sample_v2b")
    task = TaskSpec("T9", "py1:T9", "python", "python", "Double x.", [{"input": {"x": 1}, "expected": 2}])
    p2, h2 = GEN.render_sample_prompt("qwen_sample_v2", task)
    assert h2 == sha256_text(p2) and Q.SOLUTION_FORMATS["python"] in p2 and "Public examples" in p2
    p2b, _ = GEN.render_sample_prompt("qwen_sample_v2b", task)
    assert Q.SOLUTION_FORMATS["python"] in p2b and p2b != p2 and "edge cases" in p2b


@test
def t_b3_action_proposer_feeds_the_agentic_step_logic_only():
    text = "think</think>" + json.dumps(ACTS)
    recs = [{"task_id": "slug:step1", "seed": 7, "purpose": "sample", "generation": Q.generation_to_dict(_gen(text, seed=7))}]
    be = Q.FixtureGenerationBackend(recs)
    pr = GEN.ActionProposer(be)
    r, ws, sb = run_agentic()
    out = pr.propose("slug", 1, r.context_text(), seeds=(7,))
    assert out["status"] == "ok" and out["options"] == [opt(a) for a in ACTS] and out["qwen_meta"] == {"prompt_tokens": 11, "new_tokens": 22, "kaggle_seconds": 0.5}
    scorer = Scorer(scores=[0.1, 0.2, 0.9, 0.3])
    r.ctx.scorer = scorer
    rec = r.step(out["options"], out["qwen_meta"])
    assert rec["chosen_index"] == 1 and rec["tests_pass"] and rec["trusted_evidence_overrode_laya"] is True         # trusted dry-run evidence beats the (wrong) high LAYA score
    assert r.budget.record()["qwen_gen_tokens"] == 22
    r5, _, _ = run_agentic()
    rec5 = r5.step_from_generation(pr, seeds=(7,))                                           # the same path as one call: propose -> journal "proposals" -> step
    p5 = r5.journal.entries("proposals")[0]["data"]
    assert rec5["chosen_index"] == 1 and p5["status"] == "ok" and p5["rejected"] == [] and len(p5["option_sha256"]) == 4 and p5["generations"][0]["template"] == "qwen_actions_v1"
    bad5 = Q.FixtureGenerationBackend([{"task_id": "slug:step1", "seed": 3, "purpose": "sample", "generation": Q.generation_to_dict(_gen("</think>no json here", seed=3))}])
    r6, _, sb6 = run_agentic()
    rec6 = r6.step_from_generation(GEN.ActionProposer(bad5), seeds=(3,))
    assert rec6["chosen_index"] is None and r6.journal.entries("proposals")[0]["data"]["status"] == "not_json_array" and r6.journal.entries("abstain")
    # prose from the model is rejected, not repaired
    be2 = Q.FixtureGenerationBackend([{"task_id": "slug:step1", "seed": 1, "purpose": "sample", "generation": Q.generation_to_dict(_gen("</think>Here: " + json.dumps(ACTS), seed=1))}])
    o2 = GEN.ActionProposer(be2).propose("slug", 1, "ctx", seeds=(1,))
    assert o2["options"] == [] and o2["status"] == "not_json_array" and o2["rejected"]


class RecGen:
    def __init__(self, text):
        self.reqs, self.text = [], text

    def generate(self, req):
        self.reqs.append(req)
        return [Generation(text=self.text, code=self.text, extract_status="ok", tokens_prompt=1, tokens_generated=1, seed=s, model="m", model_revision="r",
                           prompt_template_id=req.prompt_template_id, prompt_sha256=sha256_text(req.prompt)) for s in req.seeds[:req.n]]

    def model_info(self):
        return {}


@test
def t_b3_ladder_uses_the_frozen_strategies_when_asked():
    task = TaskSpec("S1", "py1:S1", "python", "python", "Double the number. Define solve.", [{"input": "1", "expected": "2"}])
    reqs = GEN.plan_samples(task, 4, [1, 2, 3, 4])
    assert [r.prompt_template_id for r in reqs] == ["qwen_sample_v2", "qwen_sample_v2b"] and [r.seeds for r in reqs] == [(1, 2), (3, 4)] and [r.n for r in reqs] == [2, 2]
    assert all(Q.SOLUTION_FORMATS["python"] in r.prompt for r in reqs)
    assert [r.prompt_template_id for r in GEN.plan_samples(task, 3, [1, 2, 3], start_index=1)] == ["qwen_sample_v2", "qwen_sample_v2b"]
    exe, gen = RT.FakeExec(), RecGen("OK:a")
    ctl = EvidenceLadderController()
    ctl.run_problem(task, ControllerContext(gen, exe, RT.FakeBudget(), "D0", None, None, (1, 2, 3, 4), {"k": 4, "strategies": GEN.FROZEN_STRATEGIES}))
    assert [r.prompt_template_id for r in gen.reqs[:2]] == ["qwen_sample_v2", "qwen_sample_v2b"]
    gj = ctl.last_journal.entries("generation_request")[0]["data"]
    assert gj["templates"] == ["qwen_sample_v2", "qwen_sample_v2b"] and len(gj["prompt_sha"]) == 2
    legacy = RecGen("OK:a")                                                                 # default config keeps the legacy single-template behaviour
    ctl2 = EvidenceLadderController()
    ctl2.run_problem(task, ControllerContext(legacy, RT.FakeExec(), RT.FakeBudget(), "D0", None, None, (1, 2, 3, 4), {"k": 4}))
    assert [r.prompt_template_id for r in legacy.reqs[:1]] == ["qwen_sample_v2"] and legacy.reqs[0].n == 4


class _Arr:
    """Minimal tensor stand-in: supports shape, [0, n:] slicing and int()."""
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, ix):
        r, s = ix
        return _Arr([self.rows[r][s]]) if False else _Row(self.rows[r][s])


class _Row:
    def __init__(self, v):
        self.v, self.shape = v, (len(v),)


class _Enc(dict):
    def to(self, dev):
        return self


def _fake_torch_stack(log):
    torch = types.ModuleType("torch")
    torch.float16 = "fp16"
    torch.manual_seed = lambda s: log.append(("seed", s))
    torch.cuda = types.SimpleNamespace(manual_seed_all=lambda s: log.append(("cuda_seed", s)))

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False
    torch.no_grad = NoGrad
    tf = types.ModuleType("transformers")

    class Tok:
        eos_token_id = 2

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<chat>" + msgs[0]["content"] + "<think>\n"

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            log.append(("tok", add_special_tokens))
            return _Enc(input_ids=_Arr([list(range(5))]))

        def convert_tokens_to_ids(self, t):
            return 3

        def decode(self, row, skip_special_tokens=True):
            return "reasoning</think>\n" + "`" * 3 + "python\ndef solve(data):\n    return 1\n" + "`" * 3

        @staticmethod
        def from_pretrained(src, revision=None):
            log.append(("tok_load", src, revision))
            return Tok()

    class Model:
        device = "cuda"

        def to(self, d):
            return self

        def eval(self):
            return self

        def generate(self, **kw):
            log.append(("generate", kw["max_new_tokens"], kw["temperature"], kw["top_p"], kw["top_k"], tuple(kw["eos_token_id"]), kw["pad_token_id"]))
            return _Arr([list(range(5)) + [9, 9, 9]])

        @staticmethod
        def from_pretrained(src, revision=None, torch_dtype=None):
            log.append(("model_load", src, revision, torch_dtype))
            return Model()
    tf.AutoTokenizer, tf.AutoModelForCausalLM = Tok, Model
    return torch, tf


@test
def t_b3_hf_backend_gate_and_generation_logic_with_a_stub_torch_logic_only():
    raises(NotAuthorized, Q.HFQwenBackend)                                                  # not authorized
    raises(NotAuthorized, Q.HFQwenBackend, authorized=True, path_exists=lambda p: False)    # not on Kaggle
    raises(NotAuthorized, Q.HFQwenBackend, authorized=True, path_exists=lambda p: True, env={})   # env flag missing
    log = []
    torch, tf = _fake_torch_stack(log)
    saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
    sys.modules["torch"], sys.modules["transformers"] = torch, tf
    try:
        tmp = Path(tempfile.mkdtemp(prefix="phaseB-model-"))
        (tmp / "tokenizer.json").write_text("{}", "utf-8")
        be = Q.HFQwenBackend(authorized=True, model_dir=str(tmp), log_path=str(tmp / "log.jsonl"), env={Q.HFQwenBackend.INFERENCE_ENV: "1"}, path_exists=lambda p: True,
                             verify_files={"tokenizer.json": sha256_text("{}")})
        req = GenerationRequest("T", "PROMPT", 2, (5, 6), 123, "sample", True, "qwen_sample_v2")
        gens = be.generate(req)
        assert [g.seed for g in gens] == [5, 6] and all(g.extract_status == "ok" and g.code.strip().startswith("def solve") for g in gens)
        assert gens[0].tokens_prompt == 5 and gens[0].tokens_generated == 3 and gens[0].prompt_sha256 == sha256_text("PROMPT") and gens[0].model_revision == Q.QWEN_REVISION
        assert ("tok", False) in log and ("generate", 123, 0.6, 0.95, 20, (2, 3), 2) in log and log.index(("seed", 5)) < log.index(("seed", 6))
        assert be.snapshot_verified and len((tmp / "log.jsonl").read_text("utf-8").splitlines()) == 2 and be.model_info()["snapshot_verified"] is True
        raises(ValueError, be.generate, GenerationRequest("T", "p", 2, (1,), 10))              # fewer seeds than n
        raises(ValueError, be.generate, GenerationRequest("T", "p", 1, (1,), 10, thinking=False))
        bad = Q.HFQwenBackend(authorized=True, model_dir=str(tmp), env={Q.HFQwenBackend.INFERENCE_ENV: "1"}, path_exists=lambda p: True, verify_files={"tokenizer.json": "0" * 64})
        raises(Q.SnapshotMismatch, bad.generate, req)                                           # a snapshot that differs from the pin is refused
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@test
def t_b3_gen_entry_resume_prompt_pin_and_replay_backend():
    out = Path(tempfile.mkdtemp(prefix="phaseB-gen-")) / "records.jsonl"
    req = GenerationRequest("T1", "PROMPT-A", 2, (1, 2), 50, "sample", True, "qwen_sample_v2")
    job = GE.make_job(req, "j1")
    text = "x</think>" + "`" * 3 + "python\ndef solve(d):\n    return 1\n" + "`" * 3

    class Be:
        def __init__(self):
            self.n = 0

        def generate(self, rq):
            self.n += 1
            return [Generation(text=text, code="def solve(d):\n    return 1", extract_status="ok", tokens_prompt=3, tokens_generated=4, seed=rq.seeds[0], model="m", model_revision="r",
                               prompt_template_id=rq.prompt_template_id, prompt_sha256=sha256_text(rq.prompt))]

        def model_info(self):
            return {"backend": "stub"}
    be = Be()
    s1 = GE.run_jobs(be, [job], str(out))
    assert s1["generated"] == 2 and be.n == 2
    s2 = GE.run_jobs(be, [job], str(out))                                                    # resumable: finished (job, seed) pairs are never regenerated
    assert s2["generated"] == 0 and s2["skipped_already_done"] == 2 and be.n == 2
    raises(GE.JobError, GE.run_jobs, be, [dict(job, prompt="tampered")], str(out))           # the pinned hash catches a changed prompt
    rb = GEN.ReplayGenerationBackend.from_jsonl(out)
    g = rb.generate(req)
    assert [x.seed for x in g] == [1, 2] and g[0].code.startswith("def solve")
    raises(GEN.PromptMismatch, rb.generate, GenerationRequest("T1", "PROMPT-B", 1, (1,), 50, "sample", True, "qwen_sample_v2"))
    raises(Q.FixtureMissing, rb.generate, GenerationRequest("T1", "PROMPT-A", 1, (9,), 50, "sample", True, "qwen_sample_v2"))


@test
def t_b3_kaggle_entry_script_is_wellformed():
    src = (ROOT / "kaggle" / "el_arch1_generate_v1.py").read_text("utf-8")
    ast.parse(src)
    assert src.count("r'''__GEN_REQUEST_JSON__'''") == 1 and 'REQUEST.get("authorized") is not True' in src and "verify_files=QWEN_SHA" in src and "gen_records.jsonl" in src
    k = (ROOT / "kaggle" / "evidence_ladder_gpu_gen_v1.py").read_text("utf-8")
    shas = re.findall(r'"(model-0000\d-of-00003\.safetensors|tokenizer\.json)": "([0-9a-f]{64})"', src)
    for name, h in shas:
        assert '"%s": "%s"' % (name, h) in k, name                                           # the pins equal the ones the earlier generation kernel verified
    assert len(shas) == 4


# ====================================================================== B4
@test
def t_b4_scorer_handoff_through_laya_scorer_and_verified_saved_backend_logic_only():
    sys.path.insert(0, str(ROOT / "evidence_ladder" / "models" / "tests"))
    import laya_integration_tests as LI
    from evidence_ladder.models import laya
    opts = [opt(READ_SLUG), opt(SLUG_FIX), opt(SLUG_BAD), opt(BASH_TEST)]
    d = LI._kres()
    d.update(options=opts, option_sha256=[LI.LF.sha256_text(o) for o in opts], provenance=LI.RT.make_provenance(device=LI.DEV, context_text="ctx", options=opts),
             scores={"laya": {"score": [0.1, 0.2, 0.9, 0.3]}})
    be = LI.RT.VerifiedSavedScoreBackend("laya")
    assert be.add(d, dict(d), live_run="live@v1", verify_run="verify@v2", verify_mode="replay") is True
    sc = laya.LayaScorer(authorized=True, backend=be)
    r, ws, sb = run_agentic(scorer=sc)
    rec = r.step(opts, {})
    s = r.journal.entries("scorer")[0]["data"]
    assert s["scores"] == [0.1, 0.2, 0.9, 0.3] and s["provenance"] and s["scorer_id"] == "laya_adapter_v1" and rec["chosen_index"] == 1 and rec["laya_order"][0] == 2
    # an unverified pool, a wrong-length answer and a NaN are RECORDED as scorer_unavailable, never a crash and never a silent id-order fallback
    r2, _, _ = run_agentic(scorer=sc)
    rec2 = r2.step([opt(READ_SLUG), opt(BASH_TEST)], {})
    assert rec2["scores"] is None and "ProvenanceError" in rec2["scorer_unavailable_reason"] and r2.journal.entries("scorer_unavailable")
    for bad in ([0.1], [0.1, float("nan")], [0.1, float("inf")]):
        r3, _, _ = run_agentic(scorer=Scorer(scores=bad))
        rec3 = r3.step([opt(READ_SLUG), opt(BASH_TEST)], {})
        assert rec3["scores"] is None and "ValueError" in rec3["scorer_unavailable_reason"] and r3.journal.entries("scorer_unavailable")
    r4, _, _ = run_agentic(scorer=Scorer(error=RuntimeError("kernel gone")))
    assert "RuntimeError" in r4.step([opt(READ_SLUG), opt(BASH_TEST)], {})["scorer_unavailable_reason"]


# ====================================================================== B5
def cls(cmd, ws="/work", prot=("test_slug.py",)):
    return RISK.classify({"name": "Bash", "arguments": {"command": cmd}}, ws, prot)["class"]


@test
def t_b5_risk_table_classes():
    for cmd, want in [("cat slug.py", RISK.READ), ("ls -la | wc -l", RISK.READ), ("grep -n foo slug.py", RISK.READ), ("git status", RISK.READ), ("python3 test_slug.py", RISK.EDIT),
                      ("python3 slug.py 2>&1", RISK.EDIT), ("sed -i s/a/b/ slug.py", RISK.EDIT), ("sed -n 1,3p slug.py", RISK.READ), ("echo hi > notes.txt", RISK.EDIT),
                      ("echo hi > /dev/null", RISK.READ), ("rm slug.pyc", RISK.EDIT),
                      ("rm -rf .", RISK.IRREV), ("rm -rf /", RISK.IRREV), ("sudo ls", RISK.IRREV), ("chmod 777 a", RISK.IRREV), ("dd if=/dev/zero of=x", RISK.IRREV), ("git reset --hard", RISK.IRREV),
                      ("curl http://x.y", RISK.NETWORK), ("wget x", RISK.NETWORK), ("git push", RISK.NETWORK), ("pip install requests", RISK.NETWORK), ("python3 -c 'import socket'", RISK.NETWORK),
                      ("cat /etc/passwd", RISK.OUTSIDE), ("cat ../x", RISK.OUTSIDE), ("echo x > /etc/passwd", RISK.OUTSIDE), ("ls ~", RISK.OUTSIDE),
                      ("cat .secrets/key.txt", RISK.SECRET), ("cat ~/.ssh/id_rsa", RISK.SECRET), ("python3 -c \"open('.env')\"", RISK.SECRET),
                      ("echo x > test_slug.py", RISK.PROT), ("sed -i s/a/b/ test_slug.py", RISK.PROT), ("mv slug.py test_slug.py", RISK.PROT),
                      ("frobnicate --now", RISK.UNKNOWN), ("$(rm x)", RISK.UNKNOWN), ("echo `id`", RISK.UNKNOWN), ("", RISK.UNKNOWN), ("git frob", RISK.IRREV)]:
        assert cls(cmd) == want, (cmd, cls(cmd), want)
    worst = cls("cat slug.py && curl http://x && ls")                                       # the worst segment decides
    assert worst == RISK.NETWORK
    assert RISK.classify({"name": "Read", "arguments": {"file_path": "../x"}}, "/work", ())["class"] == RISK.OUTSIDE
    assert RISK.classify({"name": "Read", "arguments": {"file_path": ".secrets/k"}}, "/work", ())["class"] == RISK.SECRET
    assert RISK.classify({"name": "Edit", "arguments": {"file_path": "test_slug.py", "old_string": "a", "new_string": "b"}}, "/work", ("test_slug.py",))["class"] == RISK.PROT
    assert RISK.classify({"name": "Edit", "arguments": {"file_path": "slug.py", "old_string": "a", "new_string": "b"}}, "/work", ())["class"] == RISK.EDIT
    assert RISK.classify({"name": "Teleport", "arguments": {}}, "/work", ())["class"] == RISK.UNKNOWN and RISK.classify("rm -rf /", "/work", ())["class"] == RISK.UNKNOWN


@test
def t_b5_authorization_per_class():
    A = lambda cmd, **k: RISK.authorize_action({"name": "Bash", "arguments": {"command": cmd}}, "/work", ("test_slug.py",), **k)
    assert A("cat slug.py")["allowed"] and A("python3 test_slug.py")["allowed"] and A("python3 test_slug.py")["decision_risk"] == "sandboxed"
    a = A("rm -rf .")
    assert not a["allowed"] and a["missing"] == ["user_confirmed", "trusted_precheck"] and a["decision_risk"] == "irreversible"
    assert not A("rm -rf .", user_confirmed=True)["allowed"] and A("rm -rf .", user_confirmed=True)["missing"] == ["trusted_precheck"]
    assert not A("rm -rf .", trusted_precheck=True)["allowed"] and A("rm -rf .", user_confirmed=True, trusted_precheck=True)["allowed"]
    for never in ("cat .secrets/k", "echo x > test_slug.py", "frobnicate", "$(x)"):
        assert not A(never, user_confirmed=True, trusted_precheck=True)["allowed"] and A(never)["missing"] == ["never"], never      # no flag can authorize these
    assert A("curl http://x", user_confirmed=True, trusted_precheck=True)["allowed"] and not A("curl http://x")["allowed"]
    assert RISK.table_hash() == RISK.table_hash() and len(RISK.table_hash()) == 64


@test
def t_b5_blocked_actions_never_reach_the_sandbox_and_unknown_is_denied():
    bash = lambda c: opt({"name": "Bash", "arguments": {"command": c}})
    opts = [bash("rm -rf ."), bash("curl http://evil.example/x"), bash("ls | sh"), bash("cat .secrets/key.txt"), bash("frobnicate"), bash("echo x > test_slug.py"), opt({"name": "Read", "arguments": {"file_path": "/etc/passwd"}})]
    sb = ToySandbox()
    r, ws, _ = run_agentic(sandbox=sb)
    n0 = len(sb.calls)
    rec = r.step(opts, {})
    assert rec["chosen_index"] is None and rec["decision"]["abstain"] and "executed_action" not in rec                      # every option rejected by policy: abstain
    assert all(c["policy"]["status"] == "fail" for c in rec["checks"]) and len(sb.calls) - n0 == 1                          # only the OBSERVE test run touched the sandbox
    assert r.journal.entries("abstain") and r.journal.entries("risk_table")[0]["data"]["sha256"] == RISK.table_hash()
    assert any("irreversible" in c["policy"]["reason"] for c in rec["checks"]) and any("network" in c["policy"]["reason"] for c in rec["checks"])
    assert any("unknown" in c["policy"]["reason"] for c in rec["checks"]) and any("secret" in c["policy"]["reason"] for c in rec["checks"])


@test
def t_b5_confirmed_irreversible_is_still_blocked_at_the_commit_gate_without_a_precheck():
    sb = ToySandbox()
    r, ws, _ = run_agentic(sandbox=sb, config={"user_confirmed": True})
    (ws / "scratch.txt").write_text("x", "utf-8")
    n0 = len(sb.calls)
    rec = r.step([opt({"name": "Bash", "arguments": {"command": "rm -rf scratch.txt"}})], {})
    assert rec["checks"][0]["policy"]["status"] == "pass" and rec["chosen_index"] == 0                  # confirmed: ranked
    assert rec["commit_denied"] is True and rec["action_auth"]["allowed"] is False and rec["action_auth"]["missing"] == ["trusted_precheck"]
    assert (ws / "scratch.txt").exists() and len(sb.calls) - n0 == 1 and "executed_action" not in rec     # nothing executed (only the observation run)
    a = r.journal.entries("action_auth")[-1]["data"]
    assert a["class"] == "irreversible" and a["allowed"] is False
    sr = r.journal.entries("step_record")[-1]["data"]
    assert sr["action"]["denied"] is True and sr["action"]["executed"] is None
    assert inv.check_i6(r.journal) == ""
    forged = EvidenceJournal(r.registry)
    forged.append("step_record", {"step": 1, "action": {"executed": SLUG_BAD, "class": "irreversible", "allowed": False, "denied": False}})
    assert "did not authorize" in inv.check_i6(forged)
    forged2 = EvidenceJournal(r.registry)
    forged2.append("action_auth", {"class": "secret", "allowed": True})
    assert inv.check_i6(forged2) != ""
    assert not authorize(r.reqs, r.journal.evidence(r.last_state), r.last_state, r.last_state, r.env, r.registry, "irreversible", False)     # decision.authorize: irreversible needs confirmation too


@test
def t_b5_default_run_denies_irreversible_at_policy_without_user_confirmed():
    r, ws, sb = run_agentic()
    rec = r.step([opt({"name": "Bash", "arguments": {"command": "rm -rf ."}}), opt(READ_SLUG)], {})
    assert rec["checks"][0]["policy"]["status"] == "fail" and rec["chosen_index"] == 1 and rec["action_auth"]["allowed"]


# ====================================================================== B6
def solve_run(source, inp):
    ns = {}
    try:
        exec(source, ns)                                       # LOGIC-ONLY helper: runs tiny pure test functions defined in this file
        return {"status": "pass", "observed": ns["solve"](inp)}
    except Exception:       # noqa: BLE001
        return {"status": "error", "failure_class": "runtime_error"}


DOUBLE = "def solve(data):\n    return data['x'] * 2\n"
PLUS2 = "def solve(data):\n    return data['x'] + 2\n"
TWICE = "def solve(data):\n    return data['x'] + data['x']\n"


@test
def t_b6_mutants_inputs_and_strict_check_parse():
    ms = L2.mutants(DOUBLE)
    assert ms and all("def solve" in m["source"] and m["source"] != DOUBLE for m in ms) and ms == L2.mutants(DOUBLE)       # deterministic
    for m in ms:
        ast.parse(m["source"])
    assert len(L2.mutants("def solve(d):\n    return (d['a'] < 3 and d['b'] + 1 > 2) * 5\n", limit=3)) == 3
    assert L2.mutants("def (:") == []
    pub = [{"input": {"x": 2, "name": "ab"}, "expected": 4}]
    ins = L2.derive_inputs(pub, 6)
    assert ins == L2.derive_inputs(pub, 6) and len(ins) == 6 and {"x": 2, "name": "ab"} not in ins and len({L2.canon(i) for i in ins}) == 6
    assert any(i["x"] != 2 for i in ins) and any(i["name"] != "ab" for i in ins)                                          # both fields are varied
    ok = L2.parse_checks('t</think>[{"input": {"x": 5}, "expected": 10, "quote": "twice the value"}]')
    assert ok["status"] == "ok" and ok["checks"][0]["expected"] == 10
    for bad in ('t</think>Here are tests: [{"input": {"x": 5}, "expected": 10, "quote": "q"}]', 't</think>[{"input": 5, "expected": 10, "quote": "q"}]', "[]", 't</think>[{"input": {}, "expected": 1}]'):
        assert L2.parse_checks(bad)["checks"] == [], bad


REQ = "Return twice the value of x. Define solve(data).\n\nExample 1\nInput:\n{\"x\": 2}\nOutput:\n4"
PUB = [{"input": {"x": 2}, "expected": 4}]


@test
def t_b6_validate_checks_every_rule_has_a_reason():
    refs = [{"id": "ref-double", "source": DOUBLE}, {"id": "ref-twice", "source": TWICE}]
    mk = lambda i, inp, exp, q="twice the value": {"index": i, "input": inp, "expected": exp, "quote": q}
    checks = [mk(0, {"x": 5}, 10), mk(1, {"x": 5}, 10, "no such words in request"), mk(2, {"x": 2}, 4), mk(3, {"x": 2}, 5), mk(4, {"x": 5}, 11), mk(5, {"x": 7}, 14, "short")]
    v = L2.validate_checks(checks, REQ, PUB, refs, solve_run)
    assert [c["index"] for c in v["validated"]] == [0]
    why = {d["index"]: d["reason"] for d in v["dropped"]}
    assert "verbatim span" in why[1] and why[2] == "duplicate_of_public_example" and why[3].startswith("contradicts_L3") and "every trusted reference fails" in why[4] and "verbatim span" in why[5]
    ck = v["validated"][0]
    assert REQ[ck["char_start"]:ck["char_end"]] == "twice the value" and ck["kills_mutants"] and ck["reference"] == "ref-double" and ck["passed_by"] == ["ref-double", "ref-twice"]
    # vacuous: a mutant set that cannot fail the check
    real = L2.mutants
    L2.mutants = lambda src, limit=8: [{"id": "mx", "kind": "noop", "source": src.replace("* 2", "*  2")}]
    try:
        vv = L2.validate_checks([mk(0, {"x": 5}, 10)], REQ, PUB, refs, solve_run)
    finally:
        L2.mutants = real
    assert vv["validated"] == [] and "vacuous" in vv["dropped"][0]["reason"]
    assert L2.validate_checks([mk(0, {"x": 5}, 10)], REQ, PUB, [], solve_run)["dropped"][0]["reason"].startswith("no trusted reference")
    cl = L2.cluster({"a": [1], "b": [1], "c": [2]})
    assert cl["n"] == 2 and sorted(map(len, cl["clusters"].values())) == [1, 2]


class EvalExec(RT.FakeExec):
    """LOGIC-ONLY execution backend that evaluates tiny solve(data) sources in-process (public examples and experiments)."""
    def run_public(self, art, task):
        out = [{"check_id": "parse_v1", "status": "pass"}]
        for i, t in enumerate(task.public_tests):
            r = solve_run(art.source, t["input"])
            base = {"check_id": "public_examples_v1", "example_index": i}
            if r["status"] == "pass" and r["observed"] == t["expected"]:
                out.append(dict(base, status="pass", observed=r["observed"]))
            else:
                out.append(dict(base, status="fail", observed=r.get("observed", "error"), failure_class="wrong_answer"))
        return out

    def run_experiment(self, art, experiment):
        self.exp_calls.append(art.artifact_id)
        r = solve_run(art.source, json.loads(experiment["payload"]))
        return dict(r, check_id="critic_experiment_v1")


EVAL_TASK = TaskSpec("E1", "py1:E1", "python", "python", "Return twice the value of x. Define solve(data).", [{"input": {"x": 2}, "expected": 4}])
CHECKS_TEXT = "t</think>" + json.dumps([{"input": {"x": 5}, "expected": 10, "quote": "twice the value of x"}, {"input": {"x": 3}, "expected": 6, "quote": "not in the request at all"}])


def crit_id(task_id, codes):
    return "crit-" + C.sha256(task_id + "|" + "|".join(sorted(sha256_text(c) for c in codes)))[:16]


@test
def t_b6_ladder_l2_discrimination_clustering_and_critic_trigger_logic_only():
    codes = [PLUS2, DOUBLE]                                                                  # both pass the single public example (x=2), they differ elsewhere
    cfg = {"k": 2, "l2": True, "discriminate": True, "critic": True, "mode": "ladder"}
    resp = {"experiment": {"kind": "function_call", "payload": json.dumps({"x": -3})}, "implications": "outputs differ"}
    cfg["critic_responses"] = {crit_id("E1", codes): resp}
    ctl = EvidenceLadderController()
    gen = RT.FakeGen([codes, [CHECKS_TEXT]])
    ctx = ControllerContext(gen, EvalExec(), RT.FakeBudget(), "D0", RT.FakeCritic(), None, (1, 2, 3, 4), cfg)
    res = ctl.run_problem(EVAL_TASK, ctx)
    j = ctl.last_journal
    assert res.stop_cause == "decided" and res.final_artifact.source == DOUBLE and res.answer_label == "SUPPORTED"
    l2 = j.entries("l2_checks")[0]["data"]
    assert l2["parsed"] == 2 and [c["quote"] for c in l2["validated"]] == ["twice the value of x"] and l2["validated"][0]["kills_mutants"] and len(l2["dropped"]) == 1 and "verbatim span" in l2["dropped"][0]["reason"]
    bc = j.entries("behaviour_clusters")[0]["data"]
    assert bc["public_clusters"] == 1 and bc["extended_clusters"] == 2 and bc["n_inputs"] >= 5                    # one public cluster, two once derived inputs are used
    trig = j.entries("critic_trigger")[0]["data"]
    assert trig["triggered"] is True and trig["public_clusters"] == 1 and trig["extended_clusters"] == 2 and ctl.critic_trigger_rate() == 1.0
    ev = j.evidence(sha256_text(PLUS2))
    l2e = [e for e in ev if e["check_id"] == "qwen_tests_v1"]
    assert l2e and l2e[0]["status"] == "fail" and l2e[0]["independence"] == "model_derived" and l2e[0]["counterexample"]["expected"] == "10"       # L2 failure: counterexample, never a trusted failure
    good = [e for e in j.evidence(sha256_text(DOUBLE)) if e["check_id"] == "qwen_tests_v1"]
    assert good and good[0]["status"] == "pass" and good[0]["requirement_id"] == "R2"                    # allowed soft pass on the all-inputs row only
    assert res.decision["requirement_rows"] and not res.decision["authorized_to_execute"]               # nothing is ever VERIFIED here
    assert inv.assert_invariants(res, j, ControllerContext(None, None, None, "D0")) == ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"]
    cr = j.entries("critic")[-1]["data"]
    assert cr["state"] == "applied"
    # the critic is NOT triggered when everything agrees (rate logged)
    ctl2 = EvidenceLadderController()
    cfg2 = {"k": 2, "l2": False, "discriminate": True, "critic": True}
    ctl2.run_problem(EVAL_TASK, ControllerContext(RT.FakeGen([[DOUBLE, TWICE]]), EvalExec(), RT.FakeBudget(), "D0", RT.FakeCritic(), None, (1, 2, 3, 4), cfg2))
    t2 = ctl2.last_journal.entries("critic_trigger")[0]["data"]
    assert t2["extended_clusters"] == 1 and t2["triggered"] is False and ctl2.critic_trigger_rate() == 0.0


@test
def t_b6_l2_failure_feeds_the_repair_template_and_no_progress_stops_logic_only():
    cfg = {"k": 1, "l2": True, "max_rounds": 3}
    gen = RT.FakeGen([[PLUS2], [CHECKS_TEXT], [PLUS2.replace("+ 2", "+ 2 ")], [PLUS2.replace("+ 2", "+  2")]])
    ctl = EvidenceLadderController()
    res = ctl.run_problem(EVAL_TASK, ControllerContext(gen, EvalExec(), RT.FakeBudget(), "D0", None, None, (1, 2, 3, 4, 5, 6), cfg))
    j = ctl.last_journal
    assert [e["data"]["validated"] != [] for e in j.entries("l2_checks")] == [False]                    # the only reference (PLUS2) fails the check: no trusted reference passes it -> dropped
    assert res.stop_cause != "decided" or res.answer_label != "VERIFIED"
    # with a passing reference present the failing candidate is repaired from the L2 counterexample
    gen2 = RT.FakeGen([[PLUS2, DOUBLE], [CHECKS_TEXT], [DOUBLE.replace("2", "2")]])
    ctl2 = EvidenceLadderController()
    res2 = ctl2.run_problem(EVAL_TASK, ControllerContext(gen2, EvalExec(), RT.FakeBudget(), "D0", None, None, (1, 2, 3, 4, 5, 6), {"k": 2, "l2": True}))
    c = ctl2.last_run.cands[sha256_text(PLUS2)]
    assert c.l2_fail and c.l2_fail["level"] == "L2" and c.first_fail["level"] == "L2" and c.sigs
    assert res2.final_artifact.source == DOUBLE and res2.stop_cause == "decided"


# ====================================================================== B7
@test
def t_b7_diff_scope_and_final_report_logic_only():
    base = {"slug.py": "a", "test_slug.py": "t", "helper.py": "h"}
    assert diff_scope(base, dict(base), ["slug.py"], ["test_slug.py"])["ok"]
    d = diff_scope(base, {"slug.py": "A", "test_slug.py": "t", "helper.py": "h", "new.py": "n"}, ["slug.py"], ["test_slug.py"])
    assert d["modified"] == ["slug.py"] and d["added"] == ["new.py"] and d["violations"] == ["new.py (outside scope)"] and not d["ok"]
    d2 = diff_scope(base, {"slug.py": "a", "test_slug.py": "T", "helper.py": "h"}, ["*.py"], ["test_slug.py"])
    assert d2["violations"] == ["test_slug.py (protected)"]                                  # a protected file is a violation even when the scope pattern would allow it
    assert diff_scope(base, {"slug.py": "a", "helper.py": "h"}, None, ["test_slug.py"])["violations"] == ["test_slug.py (protected)"]      # removed protected file; no scope declared = only protection applies
    sb = ToySandbox()
    r, ws, _ = run_agentic(sandbox=sb, scope=["slug.py"], extra={"helper.py": "x = 1\n"})
    r.step([opt(SLUG_FIX), opt(READ_SLUG)], {})
    res = r.finish()
    fr = r.final_report
    assert res.answer_label == "VERIFIED" and res.stop_cause == "decided" and fr["outcome"]["tests_pass"] and fr["outcome"]["authorized_by_authorize"] and fr["outcome"]["label_ceiling"] == "VERIFIED"
    assert fr["outcome"]["trusted_checks_rerun_on_real_end_state"] and fr["process"]["protected_unchanged"] and fr["process"]["diff"]["modified"] == ["slug.py"] and fr["process"]["constraints_respected"]
    assert r.journal.entries("final_report")[0]["data"] == fr and {x["requirement_id"] for x in res.decision["requirement_rows"]} == {"R1", "R2", "R3"}
    assert any(e["check_id"] == "diff_scope_v1" and e["status"] == "pass" for e in r.journal.evidence(res.final_artifact_id))


@test
def t_b7_out_of_scope_edit_blocks_the_final_label_logic_only():
    helper_fix = {"name": "Edit", "arguments": {"file_path": "helper.py", "old_string": "x = 1", "new_string": "x = 2"}}
    r, ws, _ = run_agentic(scope=["slug.py"], extra={"helper.py": "x = 1\n"}, options=[opt(helper_fix)], steps=1)
    fix = r.step([opt(SLUG_FIX)], {})
    res = r.finish()
    fr = r.final_report
    assert res.answer_label == "BLOCKED" and not fr["process"]["diff_scope_ok"] and fr["process"]["diff_violations"] == ["helper.py (outside scope)"] and not fr["process"]["constraints_respected"]
    assert not res.decision["authorized_to_execute"] and fix["tests_pass"] is True                         # the tests pass, the label still refuses: R3 FAILED by a trusted certified check
    rows = {x["requirement_id"]: x["label"] for x in res.decision["requirement_rows"]}
    assert rows == {"R1": "VERIFIED", "R2": "VERIFIED", "R3": "FAILED"}


@test
def t_b7_protected_edit_attempt_and_no_scope_report_logic_only():
    r, ws, _ = run_agentic(options=[opt({"name": "Edit", "arguments": {"file_path": "test_slug.py", "old_string": "assert", "new_string": "pass #"}}), opt(READ_SLUG)], steps=1)
    r.step([opt(SLUG_FIX)], {})
    res = r.finish()
    assert res.answer_label == "VERIFIED" and r.final_report["process"]["scope_declared"] is None and r.final_report["process"]["actions_denied_by_risk_table"] == 0
    assert (ws / "test_slug.py").read_text("utf-8") == TASKS["slug"]["files"]["test_slug.py"]             # the protected edit was never applied


# ====================================================================== B7 + B8 on the real WSL sandbox
@test
def t_b7_b8_full_run_on_the_real_wsl_sandbox_and_replay_wsl():
    need_wsl()
    ws, T = make_ws("slug")
    scratch = Path(tempfile.mkdtemp(prefix="phaseB-sb-"))
    sc = Scorer(scores=[0.1, 0.2, 0.9, 0.3])                                                # LOGIC-ONLY scorer: the controller / sandbox / journal path is the real part
    rr = REP.RecordedRun(atask("slug", ws, T, scope=["slug.py"]), AgenticContext(sandbox=WslSandbox(scratch, T["protected"]), scorer=sc, arm="E", config={}), lambda m: "ctx")
    rr.begin()
    inv_res = rr.investigate()
    assert inv_res["failure"]["kind"] == "assertion" and inv_res["targets"][0]["function"] == "slugify"
    rec = rr.step([opt(READ_SLUG), opt(SLUG_FIX), opt(SLUG_BAD), opt(BASH_TEST)], {"new_tokens": 9, "prompt_tokens": 40})
    res = rr.finish()
    assert rec["chosen_index"] == 1 and res.answer_label == "VERIFIED" and res.stop_cause == "decided" and rr.run.final_report["outcome"]["tests_pass"]
    assert rr.run.final_report["process"]["diff"]["modified"] == ["slug.py"] and res.rounds == 1 and rr.run.invariants_checked
    assert (ws / "slug.py").read_text("utf-8").count('.strip("-")') == 1                          # the REAL workspace end state holds the fix
    kinds = [e["kind"] for e in rr.run.journal.entries()]
    for k in ("oracle_tier", "risk_table", "investigate", "scorer", "ranking", "action_decision", "action_auth", "step_record", "label", "final_report", "finalize", "decision", "stop"):
        assert k in kinds, k
    rec_ = rr.record()
    assert json.loads(json.dumps(rec_))["journal_hash"] == res.journal_hash
    off = REP.replay(rec_)                                                                   # OFFLINE: recorded sandbox results served in order
    assert off["identical"] and off["code_unchanged"] and off["first_divergence"] is None and off["mode"] == "offline_recorded_sandbox", off
    live = REP.replay(rec_, sandbox=WslSandbox(Path(tempfile.mkdtemp(prefix="phaseB-sb2-")), T["protected"]))     # LIVE: the real sandbox re-executes everything
    assert live["identical"] and live["replayed_journal_hash"] == res.journal_hash and live["mode"] == "live_sandbox", live
    # divergence is located
    bad = json.loads(json.dumps(rec_))
    bad["sandbox"][0]["result"]["exit_code"] = 0
    d = REP.replay(bad)
    assert not d["identical"] and d["first_divergence"] is not None
    bad2 = json.loads(json.dumps(rec_))
    bad2["code"]["evidence_ladder/controller/agentic.py"] = "0" * 64
    d2 = REP.replay(bad2)
    assert d2["identical"] and not d2["code_unchanged"] and d2["changed_code_files"] == ["evidence_ladder/controller/agentic.py"]
    bad3 = json.loads(json.dumps(rec_))
    bad3["sandbox"][0]["cmd"] = "something else"
    assert REP.replay(bad3)["error"].startswith("sandbox call #1")


@test
def t_b8_replay_offline_logic_only_and_step_records():
    sb = ToySandbox()
    ws, T = make_ws("slug")
    rr = REP.RecordedRun(atask("slug", ws, T), AgenticContext(sandbox=sb, scorer=Scorer(scores=[0.3, 0.1, 0.2]), arm="E", config={}), lambda m: "ctx")
    rr.begin()
    rr.investigate()
    rr.step([opt(READ_SLUG), opt(SLUG_FIX), opt(SLUG_BAD)], {"new_tokens": 3})
    rr.finish()
    j = rr.run.journal
    sr = j.entries("step_record")[0]["data"]
    assert set(sr) == {"step", "decision_id", "inputs", "decision", "action", "outcome"} and sr["action"]["executed"] == SLUG_FIX and sr["outcome"]["tests_pass"] and sr["decision"]["scorer_available"]
    assert j.verify_chain() and len(j.chain_hashes()) == len(j)
    off = REP.replay(rr.record())
    assert off["identical"] and off["entries"][0] == off["entries"][1]
    again = REP.replay(rr.record())
    assert again["replayed_journal_hash"] == off["replayed_journal_hash"]
    r2 = rr.record()
    r2["scorer"]["calls"][0]["scores"] = [0.9, 0.1, 0.2]
    d = REP.replay(r2)
    assert not d["identical"] and d["first_divergence"]["recorded_kind"] in ("scorer", "ranking", "action_decision")
    ce = rr.record()
    ce["events"].append({"e": "critic_stage"})
    raises(REP.ReplayError, REP.replay, ce)
    raises(REP.ReplayError, REP.replay, {"schema": "x"})


# ====================================================================== static
@test
def t_static_controller_imports_stay_clean_and_modules_import_without_torch():
    assert inv.static_no_grader_imports() == [], inv.static_no_grader_imports()
    assert inv.static_isolation_scan() == {"controller_import_names": [], "runtime_oracle_scan": []}, inv.static_isolation_scan()      # [D1/S0] oracle scan next to the import-name scan
    import subprocess
    code = "import sys; sys.modules['torch']=None; import evidence_ladder.controller.generation, evidence_ladder.models.qwen, evidence_ladder.models.gen_entry, evidence_ladder.controller.l2, evidence_ladder.controller.replay"
    p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, env=dict(__import__("os").environ, PYTHONPATH=str(ROOT)))
    assert p.returncode == 0, p.stderr[-400:]


def main():
    failed = {}
    for name, f in TESTS.items():
        try:
            f()
        except Skip as s:
            SKIPPED[name] = str(s)
        except Exception:      # noqa: BLE001
            failed[name] = traceback.format_exc()[-900:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed) - len(SKIPPED), "failed": failed, "skipped": SKIPPED}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
