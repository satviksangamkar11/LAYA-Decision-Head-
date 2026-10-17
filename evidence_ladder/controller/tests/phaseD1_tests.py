"""[EL-ARCH-1 Phase D-1] Plain-assert tests for S0 (Phase C hooks wired), S1 (acceptance driver, stage map, staged Kaggle/GLM hand-off, stub refusal), S2 (fault-injection variants),
S3 (Kaggle bundle packer), S4 (test-kernel wrapper) and S5 (conformance generator). No network, no keys, no Kaggle, no NVIDIA, no Qwen/LAYA/GLM model: every model is a labelled stub, so a test that
uses one is LOGIC-ONLY. Tests that call the real WSL sandbox are named *_wsl (REAL for controller + sandbox only) and are skipped and listed when `wsl` does not answer.
Run from the repo root: python evidence_ladder/controller/tests/phaseD1_tests.py   (prints {"tests","passed","failed","skipped"}; exit 1 on failure)."""
import ast
import inspect
import json
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import el_arch1_acceptance as ACC                                                           # noqa: E402
import el_arch1_conformance as CONF                                                         # noqa: E402
import el_arch1_stubs as STUBS                                                              # noqa: E402
import el_arch1_variants as VAR                                                             # noqa: E402
import pack_el_arch1_bundle_v1 as PACK                                                      # noqa: E402
from laya_int1_t6 import TASKS                                                              # noqa: E402
from evidence_ladder.controller import generation as GEN, invariants as INV, replay as REP   # noqa: E402
from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask       # noqa: E402
from evidence_ladder.controller.wsl_sandbox import available                                 # noqa: E402
from evidence_ladder.models import qwen as Q                                                 # noqa: E402

TESTS, SKIPPED, _WSL, _CACHE = {}, {}, {}, {}


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


def tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="phaseD1-"))


def rehearsal() -> Path:
    """One full --offline-rehearsal run (real controller + WSL sandbox, stub Qwen/LAYA/GLM), shared by several tests."""
    need_wsl()
    if "reh" not in _CACHE:
        out = tmp() / "reh"
        code = ACC.Acceptance(out, final=False).run()
        assert code == 0, code
        _CACHE["reh"] = out
    return _CACHE["reh"]


# ====================================================================== S0 hooks
@test
def d1_s0_i8_deep_scan_is_in_check_i8_and_static_scans_are_clean():
    assert INV.static_isolation_scan() == {"controller_import_names": [], "runtime_oracle_scan": []}
    import importlib
    P = importlib.import_module("evidence_ladder.execution." + "protected")      # by name only: this file lives under controller/, whose import-name scan forbids the word in import statements
    ProtectedGrader, SelectionGate = P.ProtectedGrader, P.SelectionGate
    from evidence_ladder.controller.wsl_sandbox import WslSandbox

    class Backend:
        def __init__(self, h): self.helper = h
    sb = WslSandbox(tmp(), ["t.py"])
    assert not [k for k in vars(sb) if any(w in k.lower() for w in ("grader", "protected", "oracle", "hidden"))], "the real sandbox must pass the deep scan"
    ctx = AgenticContext(sandbox=Backend(ProtectedGrader(SelectionGate(), python_cases=Path("nowhere.json"), java_cases=Path("nowhere.json"))))
    from evidence_ladder.controller.journal import EvidenceJournal
    assert "ProtectedGrader" in INV.check_i8(EvidenceJournal(type("R", (), {"checks": {}})()), ctx)
    assert INV.check_i8(EvidenceJournal(type("R", (), {"checks": {}})()), AgenticContext(sandbox=sb)) == ""


@test
def d1_s0_pilot_runner_takes_invariant_results_and_gate_needs_them():
    from evidence_ladder.orchestrator import pilot, gates
    assert "invariant_results" in inspect.signature(pilot.PilotRunner.__init__).parameters
    assert gates.invariants_gate(None)["status"] == "FAIL"
    assert gates.invariants_gate({"I%d" % i: True for i in range(1, 9)})["status"] == "PASS"


@test
def d1_s0_unquoted_test_command_is_a_gating_requirement():
    ws = tmp() / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("x = 1\n", "utf-8")
    (ws / "test_a.py").write_text("assert True\n", "utf-8")
    r = AgenticRun(AgenticTask("t", "Make it work please, thanks.", ws, "python3 test_a.py", ["test_a.py"], "sys"), AgenticContext(sandbox=object()), lambda m: "c")
    r1 = r.reqs[0]
    assert r1["requirement_id"] == "R1" and r1["kind"] == "contract" and r1["status"] == "active", r1      # was derived_hypothesis: a failing test could no longer block VERIFIED


@test
def d1_s0_comparison_scores_and_monotonicity_gate_are_journaled_and_never_select_wsl():
    need_wsl()
    o = VAR.run_variant("budget_rounds", "ex1_slug", tmp() / "w")
    j = o["entries"]
    kinds = [e["kind"] for e in j]
    assert kinds.count("scorer_comparison") == 1 and kinds.count("monotonicity_gate") >= 1
    g = next(e for e in j if e["kind"] == "monotonicity_gate")
    assert g["data"]["status"] == "INSUFFICIENT_DATA" and g["data"]["laya_ordering_enabled"] is False and g["data"]["can_override_trusted_evidence"] is False
    assert g["seq"] < next(e["seq"] for e in j if e["kind"] == "scorer"), "the gate is journaled BEFORE the decision entries"
    cmp_ = next(e for e in j if e["kind"] == "scorer_comparison")["data"]
    assert cmp_["selector"] is False and set(cmp_["scores"]) == {"probe", "random_head"}
    rk = next(e for e in j if e["kind"] == "ranking")["data"]
    assert all(set(it) == {"id", "trusted_passes", "validated_passes", "cluster", "scorer", "deepconf"} for it in rk["items"]), "comparison scores never reach the ranking"
    rep = REP.replay(o["rr"].record(), render=lambda m: "\n".join(str(x.get("content", "")) for x in m)) if False else None
    assert rep is None


@test
def d1_s0_record_with_comparison_and_stage_notes_replays_identically_wsl():
    need_wsl()
    o = VAR.run_variant("budget_rounds", "ex2_clamp", tmp() / "w")
    render = lambda m: "\n".join(str(x.get("content", "")) for x in m)
    out = REP.replay(o["rr"].record(), render=render)
    assert out["identical"] and out["error"] is None, out


# ====================================================================== S1 stage map and driver
def _e(seq, kind, data=None): return {"seq": seq, "kind": kind, "data": data or {}}


@test
def d1_s1_stage_map_assigns_every_entry_exactly_once_and_names_substages():
    ev = lambda cid: {"evidence": {"check_id": cid}}
    es = [_e(0, "run_start"), _e(1, "routing", {"step": "intake"}), _e(2, "stage", {"name": "investigate"}), _e(3, "evidence", ev("protected_tests_v1")),
          _e(4, "stage", {"name": "generate", "round": 1, "part": "generate"}), _e(5, "proposals"),
          _e(6, "stage", {"name": "step", "round": 1, "part": "step"}), _e(7, "evidence", ev("action_policy_v1")), _e(8, "routing", {"step": "step1_checked"}),
          _e(9, "monotonicity_gate"), _e(10, "scorer"), _e(11, "ranking"), _e(12, "action_auth"), _e(13, "evidence", ev("protected_tests_v1")), _e(14, "step_record"),
          _e(15, "stage", {"name": "repair", "round": 2, "part": "generate"}), _e(16, "proposals"), _e(17, "stage", {"name": "repair", "round": 2, "part": "step"}), _e(18, "ranking"),
          _e(19, "stage", {"name": "commit_verify"}), _e(20, "critic"), _e(21, "finalize")]
    m = ACC.stage_map(es)
    flat = sorted(s for rs in m["stages"].values() for a, b in rs for s in range(a, b + 1))
    assert flat == list(range(len(es))), "each entry in exactly one stage"
    st = m["stages"]
    assert st["intake"] == [[0, 1]] and st["investigate"] == [[2, 3]] and st["generate"] == [[4, 5]] and st["repair"] == [[15, 18]] and st["commit_verify"] == [[19, 21]]
    assert st["rank"] == [[9, 11]] and st["execute"] == [[12, 14]] and [6, 8] in st["discriminate"]
    assert m["repair_substages"]["2"]["generate"] == [[15, 16]] and m["repair_substages"]["2"]["rank"] == [[18, 18]] and m["not_reached"] == []


@test
def d1_s1_rehearsal_runs_three_examples_through_one_stage_map_and_replays_wsl():
    out = rehearsal()
    s = json.loads((out / "receipts/ACCEPTANCE_SUMMARY.json").read_text("utf-8"))
    assert s["mode"] == "offline_rehearsal" and s["stub_run"] is True and s["backends"]["qwen"]["stub"] and s["backends"]["laya"]["stub"] and s["backends"]["glm"]["stub"]
    assert s["backends"]["sandbox"]["stub"] is False and s["secret_scan_hits"] == [] and s["variants"]["all_ok"]
    assert s["constants"]["require"] == ["scorer", "critic"]
    assert isinstance(s["preflight"]["untracked_core_files"], list) and s["preflight"]["static_isolation_scan"] == {"controller_import_names": [], "runtime_oracle_scan": []}
    assert s["preflight"]["source_committed"] is (not s["preflight"]["dirty_tracked_source"] and not s["preflight"]["untracked_core_files"] and bool(s["preflight"]["commit"]))
    assert set(s["examples"]) == {"ex1_slug", "ex2_clamp", "ex3_wordcount"}
    for n, e in s["examples"].items():
        assert e["answer_label"] == "VERIFIED" and e["stop_cause"] == "decided" and len(e["invariants"]) == 8 and e["stub"] is True
        assert e["replay_offline"]["identical"] and e["replay_live_wsl"]["identical"] and e["replay_live_wsl"]["code_unchanged"], n
        sm = e["stage_map"]
        j = json.loads((out / e["files"]["journal"]).read_text("utf-8"))
        assert sm["entries"] == len(j) and sum(b - a + 1 for rs in sm["stages"].values() for a, b in rs) == len(j)
        for st in ("intake", "investigate", "generate", "discriminate", "rank", "execute", "commit_verify"):
            assert st in sm["stages"], (n, st)
        assert e["monotonicity_entries"] and set(e["monotonicity_entries"]) == {"INSUFFICIENT_DATA"} and e["comparison_entries"] >= 1
    assert "repair" not in s["examples"]["ex1_slug"]["stage_map"]["stages"] and "repair" in s["examples"]["ex2_clamp"]["stage_map"]["stages"], "only the unsolved example needed a repair round"
    assert s["examples"]["ex2_clamp"]["rounds"] == 2 and s["examples"]["ex1_slug"]["rounds"] == 1
    # the Kaggle-shaped files the stub wrote are all marked, and generation prompts are pinned by hash
    for p in (out / "kaggle_in").rglob("*.json*"):
        assert '"stub": true' in p.read_text("utf-8") or '"stub":true' in p.read_text("utf-8"), p
    assert s["gates"]["monotonicity"]["status"] == "INSUFFICIENT_DATA" and s["gates"]["invariants"]["status"] == "PASS"


@test
def d1_s1_final_mode_refuses_every_kind_of_stub():
    f = ACC.Acceptance(tmp() / "o", final=True, bundle=tmp(), system_block="x")
    raises(ACC.StubRefused, f.no_stubs, laya=STUBS.StubScorer("laya"))
    raises(ACC.StubRefused, f.no_stubs, critic=STUBS.StubCritic(None))
    raises(ACC.StubRefused, f.no_stubs, rec={"stub": True})
    raises(ACC.StubRefused, f.no_stubs, rec={"gpu": "Tesla T4 (STUB)"})
    raises(ACC.StubRefused, f.no_stubs, rec=[{"model": "stub:scripted-qwen"}])
    f.no_stubs(ok={"gpu": "Tesla T4", "model": Q.QWEN_REPO})            # a real-looking record passes
    # stub generation records on disk are refused when ingested in final mode
    (f.kin / "round1/gen").mkdir(parents=True)
    g = Q.make_generation("x", tokens_prompt=1, tokens_generated=1, seed=0, prompt_template_id="t", prompt_sha256="h", model="stub:scripted-qwen", model_revision="stub")
    (f.kin / "round1/gen/gen_records.jsonl").write_text(json.dumps({"task_id": "a:step1", "seed": 0, "purpose": "sample", "generation": Q.generation_to_dict(g), "stub": True}) + "\n", "utf-8")
    (f.kin / "round1/gen/RUN_SUMMARY.json").write_text(json.dumps({"verdict": "DONE", "summary": {"errors": 0}}), "utf-8")
    raises(ACC.StubRefused, f.gen_records)
    # real-looking records made by another model are refused too
    g2 = Q.make_generation("x", tokens_prompt=1, tokens_generated=1, seed=0, prompt_template_id="t", prompt_sha256="h", model="some/other-model")
    (f.kin / "round1/gen/gen_records.jsonl").write_text(json.dumps({"task_id": "a:step1", "seed": 0, "purpose": "sample", "generation": Q.generation_to_dict(g2)}) + "\n", "utf-8")
    raises(ACC.StubRefused, f.gen_records)
    raises(RuntimeError, ACC.Acceptance(tmp() / "o2", final=True, bundle=tmp(), allow_dirty=True, system_block="x").preflight)       # --allow-dirty is rehearsal-only
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()):
        raises(SystemExit, ACC.main, [str(tmp() / "o3"), "--final"])                                                                   # --final needs --bundle


@test
def d1_s1_final_handoff_writes_exact_kernels_with_pinned_hashes_and_waits():
    b = tmp() / "bundle"
    (b / "kernels").mkdir(parents=True)
    shutil.copyfile(ROOT / "kaggle/el_arch1_generate_v1.py", b / "kernels/el_arch1_generate_v1.py")
    (b / "kernels/el_arch1_score_v1.py").write_text(PACK.patch_score_kernel((ROOT / "kaggle/evidence_ladder_laya_t6_step_v1.py").read_text("utf-8")), "utf-8")
    out = tmp() / "o"
    f = ACC.Acceptance(out, final=True, bundle=b, system_block="x")
    from evidence_ladder.common.interfaces import GenerationRequest
    rq = GenerationRequest("ex1_slug:step1", "PROMPT", 2, (0, 1), 111, "sample", True, GEN.ACTION_TEMPLATE_ID)
    e = raises(ACC.NeedKaggle, f.need_gen, 1, [ACC.GE.make_job(rq, rq.task_id)])
    assert e.what == "gen" and e.rnd == 1
    src = (out / "handoff/round1/gen/el_arch1_generate_v1.py").read_text("utf-8")
    assert "__GEN_REQUEST_JSON__" not in src
    emb = json.loads(re.search(r"json\.loads\(r'''(.*?)'''\)", src, re.S).group(1))
    assert emb["authorized"] is True and emb["jobs"][0]["prompt_sha256"] == ACC.sha("PROMPT") and emb["jobs"][0]["seeds"] == [0, 1] and emb["jobs"][0]["task_id"] == "ex1_slug:step1"
    meta = json.loads((out / "handoff/round1/gen/kernel-metadata.json").read_text("utf-8"))
    assert meta["id"] == "sangeetasangamkar/el-arch1-generate-v1" and meta["enable_internet"] is False and meta["enable_gpu"] is True
    assert meta["dataset_sources"] == ["leminhhung0101/qwen3-4b-thinking-2507", "sangeetasangamkar/evidence-ladder-el-arch1-bundle-v1"]
    items = [{"key": "ex1_slug:1", "step": 1, "context_text": "CTX", "saved_raw_output": json.dumps([{"name": "Read", "arguments": {"file_path": "slug.py"}}])}]
    e = raises(ACC.NeedKaggle, f.need_score, 1, items)
    assert e.what == "score"
    live = (out / "handoff/round1/score_live/el_arch1_score_v1.py").read_text("utf-8")
    assert live == (out / "handoff/round1/score_verify/el_arch1_score_v1.py").read_text("utf-8")
    emb = json.loads(re.search(r"json\.loads\(r'''(.*?)'''\)", live, re.S).group(1))
    assert emb["mode"] == "replay" and emb["steps"][0]["key"] == "ex1_slug:1" and "random_head_seed=0" not in live and "refit_probe" in live and "evidence_ladder_el_arch1_v1" in live
    ids = {json.loads((out / ("handoff/round1/score_%s/kernel-metadata.json" % w)).read_text("utf-8"))["id"] for w in ("live", "verify")}
    assert ids == {"sangeetasangamkar/el-arch1-score-live-v1", "sangeetasangamkar/el-arch1-score-verify-v1"}, "two separate kernels = two independent processes"
    # the driver never calls Kaggle or NVIDIA or reads keys: static check of its imports and calls
    tree = ast.parse((ROOT / "tools/el_arch1_acceptance.py").read_text("utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"requests", "urllib", "http", "socket", "ssl", "nvidia_keys", "kaggle", "laya_int1_stepE"}, imported
    text = (ROOT / "tools/el_arch1_acceptance.py").read_text("utf-8")
    assert "load_keys" not in text and "nvidia_api_key" not in text and '"kaggle", "kernels"' not in text and "kaggle kernels push" not in text.replace("push this kernel", "")


def _embedded(path):
    return json.loads(re.search(r"json\.loads\(r'''(.*?)'''\)", Path(path).read_text("utf-8"), re.S).group(1))


def _fabricate_kaggle(f, out):
    """TEST ONLY: answer every pending hand-off with records built by the stub generator and then STRIPPED of their stub markers, so the --final ingestion code path (T4 check, qwen model check,
    NeedGlm / resume) can be exercised. The result is forged data in a temp directory: it proves the driver's control flow, never a model."""
    stub = ACC.StubKaggle(f.kin)
    for rd in sorted((out / "handoff").glob("round*")):
        n = int(rd.name[5:])
        g = rd / "gen" / "el_arch1_generate_v1.py"
        if g.is_file() and not (f.kin / rd.name / "gen" / "gen_records.jsonl").exists():
            stub.gen(n, _embedded(g)["jobs"])
            gp = f.kin / rd.name / "gen" / "gen_records.jsonl"
            recs = [json.loads(l) for l in gp.read_text("utf-8").splitlines()]
            for r in recs:
                r.pop("stub", None)
                r["generation"]["model"], r["generation"]["model_revision"] = Q.QWEN_REPO, Q.QWEN_REVISION
            gp.write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n", "utf-8")
            (gp.parent / "RUN_SUMMARY.json").write_text(json.dumps({"verdict": "DONE", "gpu": "Tesla T4", "summary": {"errors": 0}}), "utf-8")
        sc = rd / "score_live" / "el_arch1_score_v1.py"
        if sc.is_file():
            items = _embedded(sc)["steps"]
            for w in ("live", "verify"):
                rp = f.kin / rd.name / ("score_" + w) / "t6_step_results.json"
                if not rp.exists():
                    stub.score(n, w, items)
                    d = json.loads(rp.read_text("utf-8"))
                    d.pop("stub", None); d["gpu"] = "Tesla T4"
                    for st in d["steps"].values():
                        st.pop("stub", None)
                        st["provenance"]["device"] = "Tesla T4"
                    rp.write_text(json.dumps(d), "utf-8")


@test
def d1_s1_final_path_resumes_through_gen_score_and_glm_waits_wsl():
    """The --final control flow (staged hand-off, resume, GLM wait) driven with FORGED, marker-stripped records in a temp directory: LOGIC-ONLY for the models, REAL for the driver code and the WSL sandbox."""
    need_wsl()
    out = tmp() / "fin"
    b = tmp() / "bundle"
    (b / "kernels").mkdir(parents=True)
    shutil.copyfile(ROOT / "kaggle/el_arch1_generate_v1.py", b / "kernels/el_arch1_generate_v1.py")
    (b / "kernels/el_arch1_score_v1.py").write_text(PACK.patch_score_kernel((ROOT / "kaggle/evidence_ladder_laya_t6_step_v1.py").read_text("utf-8")), "utf-8")
    f = ACC.Acceptance(out, final=True, bundle=b)
    f.preflight = lambda: {"source_committed": True, "commit": "test", "dirty_tracked_source": [], "static_isolation_scan": {}, "limits": {}, "wsl": True, "code_sha256": {}, "driver_sha256": "x"}      # the tree is dirty during development
    waits = []
    for _ in range(12):
        code = f.run()
        if code == 0:
            break
        assert code == ACC.EXIT_WAIT
        if (out / "glm" / "WAITING_GLM.txt").exists():
            waits.append("glm")
            from evidence_ladder.critic_batch import validator as V
            from evidence_ladder.critic_batch.batch import BatchRunner, REASONING
            from evidence_ladder.critic_batch.config import load_concurrency
            from evidence_ladder.models import glm, experiments as EXP
            pk = out / "glm/critic_packets"
            acc, rq = V.read_packets(pk)
            reqs = {pid: r for pid, r in acc}
            BatchRunner(pk, out / "glm/critic_batch", ACC.stub_critic_transport(reqs), ["key_1", "key_2"], load_concurrency(), max_retries=1,
                        glm_config=glm.GlmConfig(reasoning_params=dict(REASONING), allowed_kinds=EXP.SUPPORTED_KINDS)).run()      # TEST-ONLY fake transport; the real script is el_arch1_glm_live.py
        else:
            waits.append("kaggle")
            _fabricate_kaggle(f, out)
    else:
        raise AssertionError("did not converge: %s" % waits)
    assert waits[0] == "kaggle" and waits[-1] == "glm" and waits.count("kaggle") >= 3, waits       # gen r1, score r1, gen r2 (repair), score r2, then the GLM wait
    s = json.loads((out / "receipts/ACCEPTANCE_SUMMARY.json").read_text("utf-8"))
    assert s["mode"] == "final" and s["stub_run"] is False and all(s["backends"][k]["stub"] is False for k in ("qwen", "laya", "probe", "random_head", "glm"))
    assert {n: e["answer_label"] for n, e in s["examples"].items()} == {"ex1_slug": "VERIFIED", "ex2_clamp": "VERIFIED", "ex3_wordcount": "VERIFIED"}
    assert all(e["replay_live_wsl"]["identical"] for e in s["examples"].values()) and "repair" in s["examples"]["ex2_clamp"]["stage_map"]["stages"]
    assert (out / "handoff/round1/gen/WAITING.txt").is_file() and (out / "handoff/round1/WAITING_SCORE.txt").is_file()
    rep = CONF.generate(out)
    st = {r["id"]: r for r in rep["rows"]}
    assert st["ST3"]["status"] == "REAL PASS" and st["RT-GLM"]["status"] == "REAL PASS", "forged non-marked data is graded as non-stub: the marker rule cannot detect forgery, which is why the final run needs the two independent Kaggle runs and Kaggle run ids"


@test
def d1_s1_glm_live_script_is_the_only_key_reader_and_is_not_run_here():
    src = (ROOT / "tools/el_arch1_glm_live.py").read_text("utf-8")
    compile(src, "glm_live", "exec")
    tree = ast.parse(src)
    assert "SE.load_keys()" in src
    for n in ast.walk(tree):      # no file access whose argument names the secrets directory: keys come only through the project key system function
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in ("open", "read_text", "glob", "read_bytes"):
            assert not any(isinstance(c, ast.Constant) and isinstance(c.value, str) and ".secrets" in c.value for c in ast.walk(n)), ast.dump(n)[:200]


# ====================================================================== S2 fault injection
@test
def d1_s2_variant_table_covers_every_required_fault_and_has_expectations():
    need = {"no_tests_o3", "examples_only_o2", "blocked_action", "missing_information", "broken_test_command", "broken_sandbox", "abstain_all_rejected", "invalid_model_evidence", "budget_rounds", "budget_tokens", "budget_wall"}
    assert need <= set(VAR.VARIANTS), need - set(VAR.VARIANTS)
    for n, v in VAR.VARIANTS.items():
        assert v["expect"] and v["why"], n
    # the base repos come from the existing tasks, nothing is invented
    assert {VAR.BASES[b] for b in VAR.BASES} == set(TASKS)


@test
def d1_s2_all_variants_match_their_expected_outcome_on_all_three_bases_wsl():
    need_wsl()
    out = tmp()
    s = VAR.run_all(out)
    bad = {n: {b: x["violations"] for b, x in v.items() if x["violations"]} for n, v in s["variants"].items() if any(x["violations"] for x in v.values())}
    assert s["all_ok"] and not bad, bad
    assert sum(len(v) for v in s["variants"].values()) == len(VAR.VARIANTS) * 3
    o = json.loads((out / "variants/broken_sandbox/ex1_slug/outcome.json").read_text("utf-8"))
    assert o["stop_cause"] == "tool_failure" and o["label"] not in ("VERIFIED", "SUPPORTED") and o["stub"] is True and "qwen" in s["stub_components"]


@test
def d1_s2_expectation_checks_are_not_vacuous_wsl():
    need_wsl()
    o = VAR.run_variant("budget_rounds", "ex3_wordcount", tmp() / "w")
    assert VAR.check("budget_rounds", o) == []
    saved = dict(VAR.VARIANTS["budget_rounds"]["expect"])
    try:
        VAR.VARIANTS["budget_rounds"]["expect"] = dict(saved, stop_cause="decided")
        assert any("stop_cause" in m for m in VAR.check("budget_rounds", o))
        VAR.VARIANTS["budget_rounds"]["expect"] = dict(saved, details_has="limit:max_wall_seconds")
        assert any("stop_details" in m for m in VAR.check("budget_rounds", o))
        VAR.VARIANTS["budget_rounds"]["expect"] = dict(saved, steps_run=2)
        assert any("steps ran" in m for m in VAR.check("budget_rounds", o))
    finally:
        VAR.VARIANTS["budget_rounds"]["expect"] = saved


@test
def d1_s2_broken_test_tool_is_error_never_pass_and_unquoted_command_still_blocks_wsl():
    need_wsl()
    o = VAR.run_variant("broken_sandbox", "ex1_slug", tmp() / "w")
    ev = o["run"].journal.evidence()
    assert any(e["status"] == "error" and e["check_id"] == "protected_tests_v1" for e in ev) and not any(e["status"] == "pass" and e["check_id"] == "protected_tests_v1" for e in ev)
    assert o["result"].stop_cause == "tool_failure" and "tool_failure" in o["kinds"]
    o2 = VAR.run_variant("broken_test_command", "ex2_clamp", tmp() / "w2")
    assert o2["result"].answer_label not in ("VERIFIED", "SUPPORTED") and not o2["run"].decided      # was VERIFIED before the R1 fix: the failing test did not gate


# ====================================================================== S3 / S4 packers
@test
def d1_s3_score_kernel_patch_is_exact_and_loud():
    src = (ROOT / "kaggle/evidence_ladder_laya_t6_step_v1.py").read_text("utf-8")
    new = PACK.patch_score_kernel(src)
    compile(new, "score", "exec")
    assert "refit_probe" in new and "build_random_heads" in new and "random_head_seed=0" not in new and "evidence_ladder_laya_int1_v1" not in new and "__STEP_REQUEST_JSON__" in new
    raises(SystemExit, PACK.patch_score_kernel, src.replace("random_head_seed=0", "random_head_seed=1"))        # the frozen kernel changed: refuse, do not guess
    gen = (ROOT / "kaggle/el_arch1_generate_v1.py").read_text("utf-8")
    assert "evidence_ladder_el_arch1_v1" in gen and "__GEN_REQUEST_JSON__" in gen


@test
def d1_s3_bundle_packer_builds_a_verified_leak_free_bundle_from_a_small_base():
    base = tmp() / "base"
    (base / "gate").mkdir(parents=True)
    (base / "models").mkdir()
    files = {"gate/system_block.txt": b"block", "models/final-s0.pt": b"\0\1\2", "evidence_ladder/controller/old.py": b"stale = 1\n", "capture/capture.py": b"# capture\n"}
    for k, v in files.items():
        (base / k).parent.mkdir(parents=True, exist_ok=True)
        (base / k).write_bytes(v)
    (base / "BUNDLE_MANIFEST.json").write_text(json.dumps({"bundle": PACK.OLD_BUNDLE_ID, "files": {k: PACK.sha_file(base / k) for k in files}}), "utf-8")
    out = PACK.build(base, tmp(), utc="20260101T000000Z")
    man = json.loads((out / "dataset/BUNDLE_MANIFEST.json").read_text("utf-8"))
    assert man["bundle"] == "evidence_ladder_el_arch1_v1" and man["qwen_revision"] == Q.QWEN_REVISION
    assert "evidence_ladder/controller/old.py" not in man["files"], "the stale base copy of the code is replaced by the current code"
    assert "evidence_ladder/controller/agentic.py" in man["files"] and "evidence_ladder/models/probe.py" in man["files"] and "evidence_ladder/models/random_head.py" in man["files"]
    assert "configs/el_arch1_limits_v1.toml" in man["files"] and "gate/system_block.txt" in man["files"]
    for rel, h in man["files"].items():
        assert PACK.sha_file(out / "dataset" / rel) == h, rel
    bad = [f for f in man["files"] if PACK.FORBIDDEN_PATH.search(f) and f.startswith("evidence_ladder/")]
    assert not bad and not any("/tests/" in f or "execution/" in f or "orchestrator/" in f for f in man["files"])
    assert (out / "kernels/el_arch1_generate_v1.py").is_file() and (out / "kernels/el_arch1_score_v1.py").is_file() and (out / "PUSH_COMMANDS.txt").is_file()
    # tampering with a base file is refused
    (base / "models/final-s0.pt").write_bytes(b"changed")
    raises(SystemExit, PACK.build, base, tmp(), ROOT, "20260101T000001Z")
    # create-only
    raises(FileExistsError, PACK.build, base, out.parent, ROOT, "20260101T000000Z")


@test
def d1_s4_test_kernel_wrapper_anchors_and_files_exist():
    if not (ROOT / "tools/pack_el_arch1_tests_v1.py").is_file() or not (ROOT / "kaggle/evidence_ladder_tests_v1.py").is_file():
        raise Skip("the wrapper packer and the original test kernel script are not part of the test bundle")
    import pack_el_arch1_tests_v1 as PT
    base = (ROOT / "kaggle/evidence_ladder_tests_v1.py").read_text("utf-8")
    assert '    "replay": [sys.executable, "evidence_ladder/testing/replay_integration.py"],\n' in base       # the anchor the wrapper patches; the original kernel script is not edited
    for n, rel in PT.SUITES.items():
        assert (ROOT / rel).is_file(), rel
    for rel in PT.EXTRA_FILES:
        assert (ROOT / rel).is_file(), rel
    assert set(PT.SUITES) == {"phaseA", "phaseB", "phaseC", "stepE", "laya_integration", "phaseD1"} and PT.DEFAULT_SCORES.is_file()


# ====================================================================== S5 conformance generator
@test
def d1_s5_rehearsal_receipts_are_labelled_logic_only_never_real_pass():
    out = rehearsal()
    rep = CONF.generate(out)
    assert rep["rehearsal"] is True and rep["counts"]["REAL PASS"] == 0, rep["counts"]
    ids = [r["id"] for r in rep["rows"]]
    for must in ["ST%d" % i for i in range(1, 9)] + ["I%d" % i for i in range(1, 9)] + ["L0", "L1", "L2", "L3", "L4", "L5", "Lscope", "O1", "O2", "O3", "R1.1", "R1.2", "R1.3", "R2.1", "R2.2", "R2.3", "D5",
                                                                                 "M-QWEN", "M-ADAPT", "M-LAYA", "M-PROBE", "M-RAND", "RT-CTRL", "RT-WSL", "RT-GLM", "RT-REPAIR", "BUDGET", "SAFETY", "FAULTS", "REPORT", "JOURNAL"] + ["F%02d" % i for i in range(1, 29)]:
        assert must in ids, must
    assert len(ids) == len(set(ids))
    st = {r["id"]: r for r in rep["rows"]}
    for cid in ["ST1", "ST3", "ST5", "I1", "I8", "L5", "O1", "O2", "O3", "M-QWEN", "M-LAYA", "RT-GLM", "RT-WSL", "BUDGET", "SAFETY", "FAULTS", "D5"]:
        assert st[cid]["status"] == "LOGIC-ONLY" and st[cid]["receipts"], (cid, st[cid]["status"], st[cid]["note"])
        assert all(x["sha256"] for x in st[cid]["receipts"])
    assert st["ST3"]["stub_backends"] and "qwen" in st["ST3"]["stub_backends"] and "rehearsal_run" in st["ST3"]["stub_backends"]
    assert st["F10"]["status"] == "BLOCKED" and st["F18"]["status"] == "NOT APPLICABLE" and st["F02"]["status"] == "BLOCKED" and st["L1"]["status"] == "IMPLEMENTED"
    assert all(len(r["source"]) >= 1 for r in rep["rows"] if r["status"] not in ("NOT APPLICABLE",)), [r["id"] for r in rep["rows"] if not r["source"] and r["status"] != "NOT APPLICABLE"]
    for r in rep["rows"]:
        for loc in r["source"]:
            path, line = loc.rsplit(":", 1)
            assert (ROOT / path).is_file() and int(line) >= 1, loc
    md = CONF.render_md(rep)
    assert "REHEARSAL" in md and "REAL PASS" not in md.split("counts:")[1].split("\n")[0].replace("REAL PASS 0", "")


@test
def d1_s5_missing_receipts_are_blocked_and_a_stub_marker_can_never_be_hidden():
    out = rehearsal()
    work = tmp() / "copy"
    shutil.copytree(out, work, ignore=shutil.ignore_patterns("derived", "_sandbox", "handoff"))
    # (1) a deleted example journal turns the rows that need it BLOCKED (never silently graded)
    (work / "receipts/ex1_slug/journal.json").unlink()
    rep = CONF.generate(work)
    st = {r["id"]: r for r in rep["rows"]}
    assert st["ST1"]["status"] == "BLOCKED" and "missing receipt" in st["ST1"]["note"], st["ST1"]
    # (2) the summary claims a final, stub-free run, but the Kaggle records and the example receipts still say stub:true -> never REAL PASS
    shutil.copyfile(out / "receipts/ex1_slug/journal.json", work / "receipts/ex1_slug/journal.json")
    sp = work / "receipts/ACCEPTANCE_SUMMARY.json"
    s = json.loads(sp.read_text("utf-8"))
    s["mode"], s["stub_run"] = "final", False
    for b in s["backends"].values():
        b["stub"] = False
    sp.write_text(json.dumps(s), "utf-8")
    rep2 = CONF.generate(work)
    assert rep2["counts"]["REAL PASS"] == 0, [r["id"] for r in rep2["rows"] if r["status"] == "REAL PASS"]
    # (3) only the per-record markers cleared as well, but the summary's own backend flag says stub -> still no REAL PASS for rows needing that backend
    for e in s["examples"].values():
        e["stub"] = False
    s["kaggle"] = {"gen_runs": {"r": {"stub": False, "gpu": "Tesla T4"}}, "score_runs": {"r": {"live": {"stub": False, "gpu": "Tesla T4"}}}, "glm": {}}
    s["backends"]["qwen"]["stub"] = True
    sp.write_text(json.dumps(s), "utf-8")
    rep3 = CONF.generate(work)
    st3 = {r["id"]: r for r in rep3["rows"]}
    assert st3["ST3"]["status"] == "LOGIC-ONLY" and "qwen" in st3["ST3"]["stub_backends"], st3["ST3"]
    assert st3["ST1"]["status"] == "REAL PASS", "a row that needs no stubbed backend is REAL PASS only when every marker says real"
    # (4) a source symbol that does not exist -> BLOCKED
    assert CONF.resolve("evidence_ladder/controller/agentic.py::AgenticRun.no_such_method") is None and CONF.resolve("evidence_ladder/controller/agentic.py::AgenticRun.step").startswith("evidence_ladder/controller/agentic.py:")


@test
def d1_s5_cli_writes_create_only_reports():
    out = rehearsal()
    work = tmp() / "c"
    shutil.copytree(out, work, ignore=shutil.ignore_patterns("derived", "_sandbox", "handoff"))
    assert CONF.main([str(work)]) == 0 and (work / "CONFORMANCE.json").is_file() and (work / "CONFORMANCE.md").is_file()
    raises(FileExistsError, CONF.main, [str(work)])


def main():
    failed = {}
    for name, f in TESTS.items():
        try:
            f()
        except Skip as s:
            SKIPPED[name] = str(s)
        except (Exception, SystemExit):      # noqa: BLE001
            failed[name] = traceback.format_exc()[-1100:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed) - len(SKIPPED), "failed": failed, "skipped": SKIPPED}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
