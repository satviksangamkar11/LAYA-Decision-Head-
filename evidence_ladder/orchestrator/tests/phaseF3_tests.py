"""[EL-ARCH-1 F3] Agent 3 tests: GLM accounting + UTF-8 (T42), critic blinding / experiment validation (T40/T41), run provenance + dirty-tree + secret scan (T43), mock-backend refusal (T44),
cross-path pairing refusal (T34/T49), fault-injection variants (T04/T05/T18/T19/T46) and the final-report generator (T47/T48). Test names carry their traceability id.
Real code paths only (real validator, real BatchRunner/PcSender/broker with an injected TRANSPORT, real WSL sandbox for the variants); mocks appear only where a test says so.
Run from the repo root:  PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe evidence_ladder/orchestrator/tests/phaseF3_tests.py   -> JSON {"tests","passed","failed"}"""
import copy
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

TESTS = []


def test(f):
    TESTS.append(f)
    return f


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % exc.__name__)


REAL_RUN = REPO / "results/raw/el-arch1-local-20261017T160410Z"      # the real local GLM receipt of 2026-10-17 (read-only here: tests copy it)


def _real_batch_copy(tmp: Path) -> Path:
    dst = tmp / "critic_batch"
    shutil.copytree(REAL_RUN / "glm/critic_batch", dst)
    return dst


def _first_real_request():
    line = next(l for l in (REAL_RUN / "glm/critic_packets/packets.jsonl").read_text("utf-8").splitlines() if l.strip())
    return json.loads(line)


def _run_batch(tmp: Path, transport, max_retries=1):
    """The REAL BatchRunner/PcSender/broker on a real packet with an injected transport."""
    from evidence_ladder.critic_batch import validator as V
    from evidence_ladder.critic_batch.batch import BatchRunner, REASONING
    from evidence_ladder.critic_batch.config import load_concurrency
    from evidence_ladder.models import glm, experiments as EXP
    pk = _first_real_request()
    pdir = tmp / "packets"
    V.write_packets(pdir, {pk["packet_id"]: pk["request"]})
    cfg = glm.GlmConfig(reasoning_params=dict(REASONING), allowed_kinds=EXP.SUPPORTED_KINDS)
    r = BatchRunner(pdir, tmp / "critic_batch", transport, ["key_1", "key_2"], load_concurrency(), max_retries=max_retries, glm_config=cfg)
    r.run()
    return tmp / "critic_batch", pk


# ================================================================== T42 GLM accounting, errors, retries, credentials, UTF-8
@test
def t_T42_reconcile_the_real_local_ledger():
    from evidence_ladder.critic_batch.reconcile import reconcile
    r = reconcile(REAL_RUN / "glm/critic_batch")
    assert r["ok"], r
    assert r["accounting_totals"]["calls"] == 3 and r["accounting_totals"]["ok"] == 3
    assert r["tokens"]["answers_estimated"] == 3 and r["accounting_totals"]["estimated"] is True      # the 2026-10-17 stream carried no usage: tokens are ESTIMATED and flagged, not presented as actual


@test
def t_T42_negative_tampered_ledgers_fail_reconciliation():
    from evidence_ladder.critic_batch.reconcile import reconcile
    tmp = Path(tempfile.mkdtemp())
    try:
        d = _real_batch_copy(tmp)
        bl = d / "broker_ledger.jsonl"
        lines = bl.read_text("utf-8").splitlines()
        send = next(l for l in lines if '"event":"send"' in l)
        bl.write_text("\n".join(lines + [send]) + "\n", "utf-8")                     # one extra send that was never answered
        r = reconcile(d)
        assert not r["ok"] and not r["checks"]["sends_match"]["ok"], r["checks"]["sends_match"]
        shutil.rmtree(d); d = _real_batch_copy(tmp)
        (d / "responses.jsonl").write_text("", "utf-8")                              # responses lost
        assert not reconcile(d)["checks"]["ok_match"]["ok"]
        shutil.rmtree(d); d = _real_batch_copy(tmp)
        (d / "progress.jsonl").write_text((d / "progress.jsonl").read_text("utf-8") + json.dumps({"event": "note", "text": "Authorization: " + "Bear" + "er " + "abcdefghijklmnop" + "qrstuvwxyz0123456789"}      # assembled at runtime: still key-shaped for reconcile, but no key-shaped literal in the source (secret scan)) + "\n", "utf-8")
        r = reconcile(d)
        assert not r["checks"]["no_credentials"]["ok"] and "progress.jsonl" in r["checks"]["no_credentials"]["detail"]
        shutil.rmtree(d); d = _real_batch_copy(tmp)
        (d / "responses.jsonl").write_text((d / "responses.jsonl").read_text("utf-8") + json.dumps({"event": "x", "t": "â€”"}) + "\n", "utf-8")
        assert not reconcile(d)["checks"]["utf8_clean"]["ok"]                          # the cp1252-mojibake of an em dash is detected in saved receipts
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_T42_transport_failures_are_classified_retries_bounded_no_key_in_files():
    """Real BatchRunner with a transport that always fails: HTTP 503 then a raised timeout. Calls == 1 + max_retries, every failure carries a category, nothing key-shaped on disk."""
    from evidence_ladder.critic_batch.reconcile import reconcile
    secret = "nvapi-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
    for kind in ("http503", "timeout"):
        tmp = Path(tempfile.mkdtemp())
        try:
            def transport(packet, key_label, _k=kind):
                if _k == "timeout":
                    e = TimeoutError("read timed out talking to Bearer " + secret); e.el_phase = "stream"; raise e
                return 503, {"error": "upstream unavailable; Authorization: Bearer " + secret}
            d, pk = _run_batch(tmp, transport, max_retries=1)
            r = reconcile(d, secrets=[secret])
            assert r["accounting_totals"]["calls"] == 2, r["accounting_totals"]               # first send + exactly one retry: bounded
            assert r["accounting_totals"]["failed_attempts"] == 2 and r["accounting_totals"]["ok"] == 0
            assert r["checks"]["failures_classified"]["ok"], r["checks"]["failures_classified"]
            assert r["checks"]["no_credentials"]["ok"], r["checks"]["no_credentials"]
            assert r["checks"]["retries_bounded"]["ok"], r["checks"]["retries_bounded"]
            assert set(r["attempt_error_categories"]) <= {"http", "timeout", "stream", "connect", "read"}, r["attempt_error_categories"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@test
def t_T42_success_run_through_real_runner_reconciles_with_actual_tokens():
    from evidence_ladder.critic_batch.reconcile import reconcile
    tmp = Path(tempfile.mkdtemp())
    try:
        def transport(packet, key_label):
            body = {"choices": [{"message": {"role": "assistant", "content": json.dumps({"experiment": {"kind": "command", "payload": "python3 test_slug.py", "expected_by_candidate": {}}, "implications": "re-run — the tests"}), "reasoning_content": "r"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1234, "completion_tokens": 567}}
            return 200, body
        d, pk = _run_batch(tmp, transport)
        r = reconcile(d)
        assert r["ok"], r["checks"]
        assert r["accounting_totals"]["tokens_in"] == 1234 and r["accounting_totals"]["tokens_out"] == 567 and r["accounting_totals"]["estimated"] is False
        assert r["tokens"]["answers_with_usage"] == 1
        assert "—" in (d / "responses.jsonl").read_text("utf-8").encode("utf-8").decode("utf-8") or "\\u2014" in (d / "responses.jsonl").read_text("utf-8")      # the em dash survived as one character
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_T42_utf8_stream_roundtrip_and_mojibake_negative():
    from evidence_ladder.critic_batch.transport import read_stream, encoding_check
    em = "—"
    ev = {"choices": [{"delta": {"content": "a %s b" % em}}]}
    line = ("data: " + json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8")
    body, done = read_stream([line, b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}).encode() + b"\n", b"data: [DONE]\n"])
    assert done and body["choices"][0]["message"]["content"] == "a %s b" % em and body["encoding_check"] == {"replacement_chars": 0, "mojibake_markers": 0}
    assert body["usage"] == {"prompt_tokens": 1, "completion_tokens": 2}
    # negative: the SAME bytes decoded with Windows-1252 are the "â€”" mojibake, and the detector sees it
    bad = em.encode("utf-8").decode("cp1252")
    assert bad == "â€”" and encoding_check("x " + bad)["mojibake_markers"] == 1
    assert encoding_check("plain ascii and éè and %s" % em)["mojibake_markers"] == 0


@test
def t_T42_payload_requests_provider_usage_and_flag_is_reserved():
    from evidence_ladder.models import glm
    req = _first_real_request()["request"]
    cfg = glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=("function_call", "command"))
    assert glm.build_payload(req, cfg)["stream_options"] == {"include_usage": True}
    off = glm.GlmConfig(reasoning_params={"reasoning_effort": "max"}, allowed_kinds=("function_call", "command"), include_usage=False)
    assert "stream_options" not in glm.build_payload(req, off)
    raises(glm.ConfigError, glm.build_payload, req, glm.GlmConfig(reasoning_params={"stream_options": {}}, allowed_kinds=("command",)))      # a reasoning param cannot override it


# ================================================================== T40 / T41 critic blinding, experiment validation, GLM never authorises
@test
def t_T40_packet_blinding_positive_and_negative():
    from evidence_ladder.critic_batch import validator as V
    pk = _first_real_request()
    V.validate_packet(pk)                                                            # positive: the real packet is clean
    from evidence_ladder.models.broker import packet_id_of
    for field, value in (("laya_choice", "cand-1"), ("hidden_tests", "assert x"), ("generator_reasoning", "<think>I will pick A</think>"), ("private_labels", {"a": 1})):
        req = copy.deepcopy(pk["request"]); req[field] = value
        e = raises(V.PacketRejected, V.validate_packet, {"packet_id": packet_id_of(req), "request": req})
        assert e.cls, field
    req = copy.deepcopy(pk["request"]); req["question"] = "which one did LAYA rank first?"
    raises(V.PacketRejected, V.validate_packet, {"packet_id": packet_id_of(req), "request": req})      # forbidden content in a VALUE
    raises(V.PacketRejected, V.validate_packet, {"packet_id": pk["packet_id"], "request": pk["request"], "extra": 1})


@test
def t_T41_experiment_types_validated_and_only_allowed_ops_planned():
    from evidence_ladder.models import experiments as EXP
    ws = Path(tempfile.mkdtemp())
    try:
        (ws / "slug.py").write_text("def slugify(t):\n    return t\n", "utf-8")
        (ws / "test_slug.py").write_text("print(1)\n", "utf-8")
        ok = EXP.plan({"kind": "command", "payload": "python3 test_slug.py", "expected_by_candidate": {}}, ws, ["test_slug.py"], "python3 test_slug.py")
        assert ok["kind"] == "command" and ok["cmd"]
        for bad in ({"kind": "network", "payload": "x"}, {"kind": "command", "payload": "python3 test_slug.py; cat /etc/passwd"}, {"kind": "command", "payload": "curl http://x"},
                    {"kind": "command", "payload": "rm -rf ."}, {"kind": "shell", "payload": "ls"}, {"kind": "command", "payload": 5}, {"payload": "python3 test_slug.py"}):
            raises(EXP.Rejected, EXP.plan, bad, ws, ["test_slug.py"], "python3 test_slug.py")
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@test
def t_T41_critic_never_authorises_label_unchanged():
    """Real AgenticRun + real WSL sandbox: an 'approve everything' critic that also tries a forbidden experiment changes no label and runs nothing outside the allow-list."""
    import el_arch1_variants as V
    for name in ("invalid_model_evidence", "unsafe_critic_experiment"):
        w = Path(tempfile.mkdtemp(prefix="elf3-"))
        try:
            o = V.run_variant(name, "ex2_clamp", w)
            assert V.check(name, o) == [], (name, V.check(name, o))
        finally:
            shutil.rmtree(w, ignore_errors=True)


# ================================================================== T43 / T44 provenance, dirty tree, secret scan, mock refusal
@test
def t_T43_secret_scan_counts_only_added_lines():
    import el_arch1_acceptance as A
    key = "nvapi-" + "ZZ11yy22XX33ww44VV55uu66"
    diff = "+++ b/a.py\n@@ -1 +1,2 @@\n-old = '%s'\n context\n+new = '%s'\n+safe = 1\n+++ b/b.py\n+x = 2\n" % (key, key)
    r = A.added_lines_secret_scan(diff_text=diff)
    assert r["hits"] == 1 and r["files"] == ["a.py"], r                              # the REMOVED line is not counted; only the added one is; no text is returned
    assert key not in json.dumps(r)
    assert A.added_lines_secret_scan(diff_text="+++ b/a.py\n+ok = 1\n")["hits"] == 0


@test
def t_T43_config_and_source_hashes_present_and_change_with_content():
    import el_arch1_acceptance as A
    h = A.config_hashes()
    assert "configs/backbones_v1.toml" in h and all(len(v) == 64 for v in h.values())
    assert A.sha("a") != A.sha("b")


@test
def t_T43_dirty_tree_is_refused_on_the_final_path_only():
    import el_arch1_acceptance as A
    from types import SimpleNamespace as NS
    real_run, real_av, real_scan = A.subprocess.run, A.available, A.added_lines_secret_scan

    def fake_run(cmd, *a, **k):
        out = b""
        if cmd[:2] == ["git", "rev-parse"]: out = b"deadbeef\n"
        elif cmd[:2] == ["git", "status"]: out = b" M evidence_ladder/controller/agentic.py\n"
        elif cmd[:2] == ["git", "ls-files"]: out = b""
        return NS(stdout=out.decode() if k.get("text") else out, returncode=0)
    A.subprocess.run, A.available, A.added_lines_secret_scan = fake_run, (lambda: True), (lambda base=None, diff_text=None: {"hits": 0, "files": [], "base_ref": base})
    try:
        tmp = Path(tempfile.mkdtemp())
        sysb = "x"
        e = raises(RuntimeError, A.Acceptance(tmp / "a", final=False, local=True, backbone="qwen3-0.6b", allow_dirty=False, system_block=sysb).preflight)
        assert "committed source" in str(e), e
        info = A.Acceptance(tmp / "b", final=False, local=True, backbone="qwen3-0.6b", allow_dirty=True, system_block=sysb).preflight()      # development run: allowed, and recorded as NOT the final path
        assert info["final_path"] is False and info["source_committed"] is False and info["config_sha256"]
        A.added_lines_secret_scan = lambda base=None, diff_text=None: {"hits": 2, "files": ["x.py"], "base_ref": base}
        A.subprocess.run = lambda cmd, *a, **k: NS(stdout=(b"deadbeef\n" if cmd[:2] == ["git", "rev-parse"] else b""), returncode=0)      # clean tree, but key-shaped added lines
        e = raises(RuntimeError, A.Acceptance(tmp / "c", final=False, local=True, backbone="qwen3-0.6b", allow_dirty=False, system_block=sysb).preflight)
        assert "secret scan" in str(e), e
    finally:
        A.subprocess.run, A.available, A.added_lines_secret_scan = real_run, real_av, real_scan
        shutil.rmtree(tmp, ignore_errors=True)


@test
def t_T44_mock_backends_refused_in_the_acceptance_path_and_real_ones_named():
    import el_arch1_acceptance as A
    from el_arch1_stubs import StubScorer, StubCritic
    from evidence_ladder.controller.wsl_sandbox import WslSandbox
    from unittest import mock
    for bad in (StubScorer("laya"), StubCritic(None), mock.MagicMock()):
        raises(A.StubRefused, A.refuse_mock_backends, backend=bad)
    acc = A.Acceptance(Path(tempfile.mkdtemp()), final=True, bundle=None, system_block="x")
    raises(A.StubRefused, acc.no_stubs, laya=StubScorer("laya"))                      # final mode: refused whatever flags it carries
    sb = WslSandbox(Path(tempfile.mkdtemp()), ["test_x.py"])
    A.refuse_mock_backends(sandbox=sb)                                                # positive: the real sandbox class passes
    ident = A.backend_identity(sandbox=sb, scorer=StubScorer())
    assert ident["sandbox"].endswith("wsl_sandbox.WslSandbox") and "el_arch1_stubs" in ident["scorer"]      # the receipt names the class that ran, which makes a stub visible


@test
def t_T34_T49_pool_admission_needs_exact_context_and_options_never_cross_paths():
    import el_arch1_acceptance as A
    from evidence_ladder.models import laya_runtime as RT
    ctx, opts = "context text", ['{"name":"Read"}', '{"name":"Bash"}']
    live = {"status": "ok", "context_sha256": RT.LF.sha256_text(ctx), "options": opts}
    ok, why = A.pool_admissible({"context_text": ctx}, live, dict(live), opts)
    assert ok, why
    ok, why = A.pool_admissible({"context_text": ctx + " changed"}, live, dict(live), opts)
    assert not ok and "context" in why                                                # same options, different context (e.g. 0.6B-generated state): refused
    ok, why = A.pool_admissible({"context_text": ctx}, live, dict(live), opts[::-1])
    assert not ok and "options" in why                                                # reordered
    ok, why = A.pool_admissible({"context_text": ctx}, live, dict(live), opts[:1])
    assert not ok                                                                     # fewer options (a different pool)
    be = RT.VerifiedSavedScoreBackend("laya")
    raises(RT.ProvenanceError, be.score, "next_action", {"context_text": ctx, "options": opts})      # nothing admitted -> no score, no fallback


# ================================================================== T04 / T05 / T18 / T19 / T46 fault-injection variants (real controller, real WSL sandbox, scripted model text)
def _variant(name, base="ex2_clamp"):
    import el_arch1_variants as V
    w = Path(tempfile.mkdtemp(prefix="elf3-"))
    try:
        o = V.run_variant(name, base, w)
        return o, V.check(name, o)
    finally:
        shutil.rmtree(w, ignore_errors=True)


@test
def t_T04_broken_test_tool_is_error_never_pass():
    for n in ("broken_test_command", "broken_sandbox"):
        o, bad = _variant(n)
        assert bad == [], (n, bad)
        assert o["result"].answer_label not in ("VERIFIED", "SUPPORTED")


@test
def t_T05_budget_variants_name_the_exhausted_limit():
    for n in ("budget_rounds", "budget_tokens", "budget_wall", "budget_actions"):
        o, bad = _variant(n)
        assert bad == [], (n, bad)
        assert "limit:" in (o["result"].stop_details or "") or o["result"].stop_cause.endswith("_cap") or o["result"].stop_cause in ("wall_time", "token_cap"), o["result"]


@test
def t_T18_O2_variant_routes_O2_and_takes_no_action():
    o, bad = _variant("examples_only_o2")
    assert bad == [] and o["tiers"][-1]["tier"] == "O2" and not o["cmds"]


@test
def t_T19_O3_variants_route_O3_and_ambiguity_asks_the_user():
    o, bad = _variant("no_tests_o3"); assert bad == [] and o["tiers"][-1]["tier"] == "O3"
    o, bad = _variant("missing_information"); assert bad == [] and "ask_user" in o["kinds"] and o["result"].answer_label == "BLOCKED"


@test
def t_T46_malformed_sequential_duplicate_empty_unsafe_missing_scorer_variants():
    for n in ("malformed_json", "sequential_plan", "duplicate_options", "empty_pool", "unsafe_action_filtered", "missing_scorer", "missing_scorer_required", "blocked_action", "scope_conflict"):
        o, bad = _variant(n)
        assert bad == [], (n, bad)


@test
def t_T46_negative_a_variant_that_does_not_reach_its_expected_state_is_reported():
    import el_arch1_variants as V
    o, _ = _variant("budget_rounds")
    saved = V.VARIANTS["budget_rounds"]["expect"]
    try:
        V.VARIANTS["budget_rounds"]["expect"] = dict(saved, stop_cause="decided", label="VERIFIED")
        assert V.check("budget_rounds", o), "check() accepted a wrong expectation"
    finally:
        V.VARIANTS["budget_rounds"]["expect"] = saved


@test
def t_T24_rejections_are_journalled_by_the_controller_and_the_scored_pool_is_the_accepted_one():
    """prefilter=False: no-op/duplicate/plan/malformed rejections appear under journal kind candidate_filter (the controller's single filter call); the driver's accepted_preview equals what the
    controller accepted, so LAYA scores exactly the accepted options."""
    import el_arch1_variants as V, el_arch1_acceptance as A
    for name, rule in (("duplicate_options", "duplicate"), ("sequential_plan", "sequential_plan_step"), ("malformed_json", "malformed_option")):
        o, bad = _variant(name)
        assert bad == [], (name, bad)
        codes = V.journal_filter_codes(o["run"])
        assert any(c.split(":")[0] == rule or c == rule for c in codes) or (rule == "sequential_plan_step" and "sequential_plan" in codes), (name, codes)
        assert o["proposals"][0]["options"] == [] and o["proposals"][0].get("candidates"), "prefilter=False must hand raw candidates to the controller"
    acc = A.Acceptance(Path(tempfile.mkdtemp()), final=False, local=True, backbone="qwen3-0.6b", allow_dirty=True, system_block="x")
    w = Path(tempfile.mkdtemp(prefix="elf3-"))
    try:
        import el_arch1_variants as V2
        from evidence_ladder.controller.agentic import AgenticTask
        o = V2.run_variant("duplicate_options", "ex2_clamp", w)
        step_opts = V2.options_from_proposal(o["proposals"][0])
        from types import SimpleNamespace as NS
        from laya_int1_t6 import TASKS
        T = TASKS["clamp"]
        fresh = NS(run=NS(files_text=lambda: dict(T["files"]), task=NS(protected=T["protected"])))      # the workspace as it was BEFORE the step (the run has since applied the fix)
        preview = acc.accepted_preview(fresh, step_opts)
        first = o["run"].journal.entries("candidate_filter")[0]["data"]
        assert len(preview) == first["counts"]["accepted"] == 2 and len(step_opts) == 4
    finally:
        shutil.rmtree(w, ignore_errors=True)


@test
def t_T48_one_row_per_traceability_id_and_per_invariant_placeholder_guard():
    import el_arch1_final_report as FR
    assert "budget_actions" in FR.REGISTRY["T05"]["vneg"]


@test
def t_T37_pathB_stored_4b_replay_laya_probe_random_head_exact_pools_and_refusals():
    """Positive: stored verified 4B scores replayed through the current controller on the exact stored contexts (LAYA = stored; probe + random head attached as comparison only, selector false).
    Negative: a pool the controller's filter changes is refused, and a pool with one option removed is refused by the backend."""
    import el_arch1_pathb_replay as PB
    out = Path(tempfile.mkdtemp()) / "pb"
    try:
        assert PB.run(out) == 0
        d = json.loads((out / "PATHB_REPLAY_RECEIPT.json").read_text("utf-8"))
        s = d["summary"]
        assert d["stub"] is False and s["replayed"] >= 2 and s["comparison_only"] and s["laya_probe_random_head_all_stored_and_equal"] and s["negative_controls_refused"], s
        for p in d["pools"].values():
            if p["verdict"] == "REPLAYED":
                assert p["rebuilt_context_sha256"] == p["stored_context_sha256"] and p["pool_is_exact_stored_pool"] and p["comparison"][0]["selector"] is False
            else:
                assert p["verdict"] == "REFUSED" and p["reason"]
    finally:
        shutil.rmtree(out.parent, ignore_errors=True)


# ================================================================== T47 / T48 report generator
@test
def t_T48_generator_refuses_pass_without_real_negative_and_receipt():
    import el_arch1_final_report as FR
    row = FR.verdict(dict(real=["x"], neg=[], receipt=["r"], conf_ok=True, stat=False, na=None, failed=[], stub=False))
    assert row[0] == "INSUFFICIENT_DATA", row                                        # negative test missing
    assert FR.verdict(dict(real=[], neg=["n"], receipt=["r"], conf_ok=True, stat=False, na=None, failed=[], stub=False))[0] == "INSUFFICIENT_DATA"
    assert FR.verdict(dict(real=["x"], neg=["n"], receipt=[], conf_ok=True, stat=False, na=None, failed=[], stub=False))[0] == "INSUFFICIENT_DATA"
    assert FR.verdict(dict(real=["x"], neg=["n"], receipt=["r"], conf_ok=True, stat=False, na=None, failed=[], stub=True))[0] == "INSUFFICIENT_DATA"      # mock-only is never a PASS
    assert FR.verdict(dict(real=["x"], neg=["n"], receipt=["r"], conf_ok=True, stat=True, na=None, failed=[], stub=False))[0] == "INSUFFICIENT_DATA"       # statistical gate on 3 examples
    assert FR.verdict(dict(real=["x"], neg=["n"], receipt=["r"], conf_ok=False, stat=False, na=None, failed=[], stub=False))[0] == "FAIL"
    assert FR.verdict(dict(real=["x"], neg=["n"], receipt=["r"], conf_ok=True, stat=False, na=None, failed=["t_bad"], stub=False))[0] == "FAIL"
    assert FR.verdict(dict(real=["x"], neg=["n"], receipt=["r"], conf_ok=True, stat=False, na=None, failed=[], stub=False))[0] == "PASS"
    assert FR.verdict(dict(real=[], neg=[], receipt=[], conf_ok=None, stat=False, na="out of scope", failed=[], stub=False))[0] == "NOT_APPLICABLE"


@test
def t_T48_one_row_per_traceability_id_and_per_invariant():
    import el_arch1_final_report as FR
    ids = FR.traceability_ids()
    assert ids[:2] == ["T01", "T02"] and len(ids) == len(set(ids)) >= 50, ids[-3:]
    reg = FR.REGISTRY
    assert set(ids) <= set(reg), sorted(set(ids) - set(reg))                        # no unregistered requirement: a row without an entry would be silently dropped
    assert {"T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08"} <= set(reg)       # I1-I8


@test
def t_T47_statistical_rows_stay_insufficient_with_three_examples():
    import el_arch1_final_report as FR
    assert FR.REGISTRY["T38"]["stat"] and FR.REGISTRY["T47"]["stat"]
    assert FR.MIN_EXAMPLES_FOR_STATISTICS > 3


if __name__ == "__main__":
    failed = {}
    for t in TESTS:
        try:
            t()
        except Exception:
            failed[t.__name__] = traceback.format_exc()[-1500:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)
