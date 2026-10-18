"""[EL-ARCH-1 F2c / Agent 2] PART A: L2 prompt/parser diagnosis on the REAL saved Kaggle 4B generations + the new versioned qwen_tests_v2 template and parse_checks_v2 (additive; v1 untouched).
PART B: hidden-state capture hook (models/state_capture.py, tested with a tiny fake torch model on CPU, no real model is run) and the diagnostic probe tool (tools/el_bank_state_probe.py, tested with
synthetic states). Run from the repo root:
    PYTHONPATH=. ATEN_CPU_CAPABILITY=avx2 .venv/Scripts/python.exe evidence_ladder/controller/tests/phaseF2c_tests.py
Tests that read the saved run skip (reported) when results/raw/el-bank40-pass1-out-20261018T1752 is absent. *_logic = real controller / parser / tool code with fakes; nothing here uses a GPU or a network."""
import glob
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phaseB_tests as B                                                                  # noqa: E402
import run_tests as RT                                                                     # noqa: E402
from evidence_ladder.common.interfaces import ControllerContext                            # noqa: E402
from evidence_ladder.controller import l2 as L2                                            # noqa: E402
from evidence_ladder.controller.controller import EvidenceLadderController                 # noqa: E402
from evidence_ladder.models import qwen as Q                                               # noqa: E402

TESTS, SKIPPED = {}, {}
SAVED = ROOT / "results/raw/el-bank40-pass1-out-20261018T1752"


def test(f):
    TESTS[f.__name__] = f
    return f


def saved_tests_generations():
    rows = []
    for f in sorted(glob.glob(str(SAVED / "tasks/*/gen_calls.json"))):
        for c in json.loads(Path(f).read_text("utf-8")):
            if c.get("purpose") == "tests":
                for g in c["generations"]:
                    rows.append((c["task_id"], c["prompt_template_id"], g))
    if not rows:
        raise B.Skip("saved bank run not on this machine: %s" % SAVED)
    return rows


# ====================================================================== PART A
@test
def a_saved_v1_texts_are_all_rejected_because_prompt_and_parser_disagree_logic():
    rows = saved_tests_generations()
    assert len(rows) == 26 and {t for _, t, _ in rows} == {"qwen_tests_v1"}
    usable = sum(bool(L2.parse_checks(g["text"], 6)["checks"]) for _, _, g in rows)
    assert usable == 0                                                                                # BEFORE: 0 / 26 generations yield one usable check (L2 INSUFFICIENT on every task)
    # the frozen v1 prompt asks for exactly the keys input and expected; the v1 parser demands a third key
    assert 'the keys "input" and "expected"' in Q.TEMPLATES["qwen_tests_v1"].replace('\\"', '"') and "quote" not in Q.TEMPLATES["qwen_tests_v1"]
    assert 'set(a) == {"input", "expected", "quote"}' in Path(L2.__file__).read_text("utf-8")                       # v1 parser (frozen, unchanged) still demands the third key


@test
def a_v2_parser_names_the_reason_for_every_saved_item_and_invents_nothing_logic():
    rows = saved_tests_generations()
    st, counts, items = {}, {}, 0
    for _, _, g in rows:
        pc = L2.parse_checks_v2(g["text"], 6, None)
        st[pc["status"]] = st.get(pc["status"], 0) + 1
        for k, v in pc["counts"].items():
            counts[k] = counts.get(k, 0) + v
        assert pc["checks"] == []                                                                     # the span rule is not relaxed: no quote -> no check, never an invented quote
    assert st == {"no_valid_check": 24, "no_think_close": 2} and counts == {"missing_quote": 144, "no_think_close": 2}, (st, counts)
    # AFTER on the same 26 texts: 24 / 26 are valid fenced JSON arrays whose 144 items are all well-formed except for the missing quote; the 2 others hit the 8192-token ceiling inside <think>
    for _, _, g in rows:
        if g["tokens_generated"] >= 8192:
            assert L2.parse_checks_v2(g["text"], 6)["status"] == "no_think_close"


@test
def a_saved_run_facts_behind_the_diagnosis_logic():
    rows = saved_tests_generations()
    toks = [g["tokens_generated"] for _, _, g in rows]
    assert sum(1 for t in toks if t >= 8192) == 2 and round(sum(toks) / len(toks)) == 4382 and all(m == 8192 for m in [8192])
    str_inputs = 0
    for _, _, g in rows:
        i = g["text"].rfind("</think>")
        if i < 0:
            continue
        ans = g["text"][i + 8:].strip()
        if ans.startswith("```"):
            ans = ans.split("\n", 1)[1].rsplit("```", 1)[0]
        for it in json.loads(ans):
            str_inputs += isinstance(it["input"], str)
            assert set(it) == {"input", "expected"}                                                   # every item has exactly the two requested keys
    assert str_inputs == 18                                                                           # model behaviour: 18 of 144 items gave a bare string instead of an object


@test
def a_v2_template_and_frozen_v1_logic():
    p, h = L2.render_tests_v2("Return twice x.", "x=2 -> 4", 6, ["x"])
    assert '"quote"' in p and 'exactly the keys "x"' in p and "at least 8 characters" in p and h == L2.sha256_text(p) and L2.TEMPLATE_V2_ID == "qwen_tests_v2"
    assert "qwen_tests_v2" not in Q.TEMPLATES and Q.TEMPLATE_HASHES["qwen_tests_v1"] == L2.sha256_text(Q.TEMPLATES["qwen_tests_v1"])      # the frozen table has no v2 and v1 is the same text


@test
def a_v2_parser_accepts_compliant_text_and_rejects_each_bad_shape_logic():
    good = [{"input": {"x": 5}, "expected": 10, "quote": "twice the value of x"}]
    t = "think</think>\n```json\n" + json.dumps(good) + "\n```"
    pc = L2.parse_checks_v2(t, 6, ["x"])
    assert pc["status"] == "ok" and pc["checks"][0]["quote"] == "twice the value of x" and pc["rejected"] == []
    assert L2.parse_checks_v2("```json\n" + json.dumps(good) + "\n```", 6, ["x"], require_think_close=False)["status"] == "ok"      # a non-thinking model has no </think>
    assert L2.parse_checks_v2("```json\n" + json.dumps(good) + "\n```", 6, ["x"], require_think_close=True)["status"] == "no_think_close"
    bad = [{"input": "x=5", "expected": 10, "quote": "twice the value of x"}, {"input": {"y": 1}, "expected": 2, "quote": "q" * 10}, {"input": {"x": 1}, "expected": 2},
           {"input": {"x": 1}, "expected": 2, "quote": 7}, {"input": {"x": 1}, "expected": 2, "quote": "q" * 10, "why": "z"}, 5] + good * 7
    pc = L2.parse_checks_v2("t</think>" + json.dumps(bad), 6, ["x"])
    assert pc["counts"] == {"input_not_object": 1, "input_keys_mismatch": 1, "missing_quote": 1, "quote_not_string": 1, "unexpected_keys": 1, "not_object": 1, "beyond_max_checks": 1} and len(pc["checks"]) == 6
    assert L2.parse_checks_v2("t</think>nonsense", 6)["status"] == "not_json_array" and L2.parse_checks_v2("t</think>{}", 6)["counts"] == {"not_array": 1}
    assert set(pc["counts"]) <= set(L2.REJECT_CODES)


def _ladder(l2_version, text, thinking=True):
    codes = [B.PLUS2, B.DOUBLE]
    cfg = {"k": 2, "l2": True, "discriminate": True, "mode": "ladder", "l2_version": l2_version, "thinking": thinking}
    gen = RT.FakeGen([codes, [text]])
    ctl = EvidenceLadderController()
    ctx = ControllerContext(gen, B.EvalExec(), RT.FakeBudget(), "D0", None, None, (1, 2, 3, 4), cfg)
    return ctl, ctl.run_problem(B.EVAL_TASK, ctx), gen


@test
def a_controller_uses_v2_prompt_and_parser_and_keeps_every_l2_invariant_logic():
    chk = [{"input": {"x": 5}, "expected": 10, "quote": "twice the value of x"}, {"input": {"x": 3}, "expected": 6, "quote": "not in the request at all"}]
    ctl, res, gen = _ladder(2, "t</think>" + json.dumps(chk))
    j = ctl.last_journal
    assert gen.calls[1].prompt_template_id == "qwen_tests_v2" and '"quote"' in gen.calls[1].prompt
    lp = j.entries("l2_parse")[0]["data"]
    assert lp["version"] == 2 and lp["status"] == "ok" and lp["n_checks"] == 2
    l2 = j.entries("l2_checks")[0]["data"]
    assert [c["quote"] for c in l2["validated"]] == ["twice the value of x"] and len(l2["dropped"]) == 1 and "verbatim span" in l2["dropped"][0]["reason"]      # span rule intact
    assert l2["validated"][0]["kills_mutants"]                                                                                          # mutant validation intact
    assert not res.decision["authorized_to_execute"] and res.answer_label != "VERIFIED"                                                  # L2 never approves
    assert all(e["independence"] == "model_derived" for e in j.evidence() if e["check_id"] == "qwen_tests_v1" or e["check_id"].startswith("qwen_tests"))
    # negative control: v1 (the default) writes no l2_parse entry and uses the frozen template id
    ctl1, res1, gen1 = _ladder(1, B.CHECKS_TEXT)
    assert not ctl1.last_journal.entries("l2_parse") and gen1.calls[1].prompt_template_id == "qwen_tests_v1"


@test
def a_controller_v2_without_quote_validates_nothing_and_non_thinking_text_is_parsed_logic():
    noquote = [{"input": {"x": 5}, "expected": 10}]
    ctl, res, _ = _ladder(2, "t</think>" + json.dumps(noquote))
    assert ctl.last_journal.entries("l2_parse")[0]["data"]["counts"] == {"missing_quote": 1} and ctl.last_journal.entries("l2_checks")[0]["data"]["validated"] == []
    good = [{"input": {"x": 5}, "expected": 10, "quote": "twice the value of x"}]
    ctl2, res2, _ = _ladder(2, "```json\n" + json.dumps(good) + "\n```", thinking=False)                      # a non-thinking model (no </think>)
    assert len(ctl2.last_journal.entries("l2_checks")[0]["data"]["validated"]) == 1 and res2.answer_label != "VERIFIED"
    ctl3, res3, _ = _ladder(2, "```json\n" + json.dumps(good) + "\n```", thinking=True)                      # negative: a thinking model whose think block never closed is refused
    assert ctl3.last_journal.entries("l2_parse")[0]["data"]["status"] == "no_think_close" and ctl3.last_journal.entries("l2_checks")[0]["data"]["validated"] == []


# ====================================================================== PART B1: capture hook
def _tiny():
    import torch
    torch.manual_seed(0)
    emb = torch.nn.Embedding(256, 16)
    layers = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(3)])
    DEP = (1, 2, 3)

    def encode(s):
        return list(s.encode("utf-8"))[:200]

    def capture_fn(context_text, options):
        pre = encode(context_text + "\nCUE")
        outs = []
        with torch.no_grad():
            for o in options:
                suf = encode(o)
                h = emb(torch.tensor(pre + suf)).float()
                hs = []
                for L in layers:
                    h = torch.tanh(L(h))
                    hs.append(h)
                qp = len(pre) - 1
                hq = torch.stack([hs[d - 1][qp] for d in DEP])
                outs.append(torch.stack([hs[d - 1][len(pre):].mean(0) for d in DEP]))
        return hq.half(), torch.stack(outs).half(), {"status": "ok", "profile_id": "tiny", "features_sha256": "f" * 8}
    return capture_fn, DEP


@test
def b1_capture_hook_streams_fp16_states_per_candidate_to_disk_logic():
    from evidence_ladder.models.state_capture import StateCapture, load_state
    fn, dep = _tiny()
    root = Path(tempfile.mkdtemp(prefix="f2c-"))
    sc = StateCapture(fn, root, profile_id="tiny", depths=dep)
    p = sc.capture("T1", "c" * 64, "task prompt", "def solve(data):\n    return 1\n")
    assert p == root / "tasks/T1/states" / ("c" * 64 + ".pt") and p.exists() and not list(p.parent.glob("*.tmp"))
    st = load_state(p)
    assert st["candidate_id"] == "c" * 64 and st["task_id"] == "T1" and st["profile_id"] == "tiny" and st["depths"] == [1, 2, 3]
    assert str(st["hq"].dtype) == "torch.float16" and tuple(st["hq"].shape) == (3, 16) and tuple(st["ho"].shape) == (3, 16)
    idx = [json.loads(l) for l in (p.parent / "index.jsonl").read_text("utf-8").splitlines()]
    assert len(idx) == 1 and idx[0]["hidden"] == 16 and idx[0]["candidate_id"] == "c" * 64
    assert sc.capture("T1", "c" * 64, "task prompt", "different text") == p and sc.skipped == 1 and sc.saved == 1                       # resumable: an existing state is kept
    import torch
    assert not any(isinstance(v, torch.Tensor) for v in vars(sc).values())                                                              # nothing accumulates in memory
    assert sc.summary()["saved"] == 1 and sc.summary()["errors"] == 0


@test
def b1_capture_hook_is_silent_when_disabled_and_never_raises_logic():
    from evidence_ladder.models.state_capture import StateCapture
    fn, dep = _tiny()
    root = Path(tempfile.mkdtemp(prefix="f2c-"))
    off = StateCapture(fn, root, depths=dep, enabled=False)
    assert off.capture("T1", "a", "ctx", "src") is None and not (root / "tasks").exists()
    assert StateCapture(None, root).capture("T1", "a", "ctx", "src") is None
    def boom(c, o):
        raise RuntimeError("cuda oom")
    bad = StateCapture(boom, root, depths=dep)
    assert bad.capture("T1", "a", "ctx", "src") is None and len(bad.errors) == 1 and "RuntimeError" in bad.errors[0]["error"] and not (root / "tasks").exists()
    import torch
    nonfinite = StateCapture(lambda c, o: (torch.full((3, 4), float("inf")).half(), torch.zeros(1, 3, 4).half(), {"status": "ok"}), root, depths=dep)
    assert nonfinite.capture("T1", "b", "ctx", "src") is None and "non-finite" in nonfinite.errors[0]["error"]
    wrong_depth = StateCapture(lambda c, o: (torch.zeros(2, 4).half(), torch.zeros(1, 2, 4).half(), {"status": "ok"}), root, depths=dep)
    assert wrong_depth.capture("T1", "c", "ctx", "src") is None and "depths" in wrong_depth.errors[0]["error"]
    notok = StateCapture(lambda c, o: (None, None, {"status": "excluded_over_limit"}), root, depths=dep)
    assert notok.capture("T1", "d", "ctx", "src") is None and "excluded_over_limit" in notok.errors[0]["error"]


@test
def b1_capture_never_touches_the_journal_or_a_decision_logic():
    from evidence_ladder.models.state_capture import StateCapture
    fn, dep = _tiny()
    ctl0, r0, _ = _ladder(2, "t</think>[]")
    ctl1, r1, _ = _ladder(2, "t</think>[]")
    sc = StateCapture(fn, Path(tempfile.mkdtemp(prefix="f2c-")), depths=dep)
    before = ctl1.last_journal.journal_hash()
    for c in list(ctl1.last_run.cands.values()):                                               # what the local runner does: once per candidate, after the run
        sc.capture("E1", c.id, B.EVAL_TASK.prompt, c.art.source)
    assert sc.saved == 2 and ctl1.last_journal.journal_hash() == before == r0.journal_hash == r1.journal_hash                  # the journal (and so every decision) is byte-identical
    assert sc.errors == [] and ctl1.last_journal.verify_chain()


# ====================================================================== PART B2: probe tool
def _journal(cands, passes, clusters):
    es, seq = [], 0
    for c in cands:
        es.append({"seq": seq, "kind": "candidate", "data": {"id": c}}); seq += 1
    for c in cands:
        es.append({"seq": seq, "kind": "evidence", "data": {"evidence": {"candidate_id": c, "check_id": "public_examples_v1", "status": "pass" if passes[c] else "fail", "requirement_id": "R1"}, "outcome": {}}}); seq += 1
    members = {}
    for c, s in zip(cands, clusters):
        members.setdefault(s, []).append(c)
    es.append({"seq": seq, "kind": "behaviour_clusters", "data": {"members": members, "extended_clusters": len(members)}})
    return es


def _make_run(n_tasks=14, n_cand=6, signal_depth=1, signal=3.0, hidden=24, seed=0, task_leak=False):
    import numpy as np
    import torch
    rng = np.random.RandomState(seed)
    root = Path(tempfile.mkdtemp(prefix="f2c-run-"))
    direction = rng.randn(hidden)
    direction /= np.linalg.norm(direction)
    for t in range(n_tasks):
        tid = "T%02d" % t
        cands = ["%s-c%d" % (tid, i) for i in range(n_cand)]
        y = {c: bool((i + t) % 2 == 0 or i == 1) for i, c in enumerate(cands)}
        if all(y.values()) or not any(y.values()):
            y[cands[0]] = not y[cands[0]]
        td = root / "tasks" / tid
        (td / "states").mkdir(parents=True)
        (td / "journal.jsonl").write_text("\n".join(json.dumps(e) for e in _journal(cands, y, [0 if y[c] else 1 for c in cands])) + "\n", "utf-8")
        for i, c in enumerate(cands):
            ho = rng.randn(3, hidden) * 0.5
            if task_leak:
                ho[:, t % hidden] += 8.0
            elif signal:
                ho[signal_depth] += (signal if y[c] else -signal) * direction
            torch.save({"candidate_id": c, "task_id": tid, "profile_id": "tiny", "depths": [14, 19, 23], "hq": torch.zeros(3, hidden).half(), "ho": torch.tensor(ho).half()}, td / "states" / (c + ".pt"))
    return root


@test
def b2_labels_come_from_execution_evidence_and_clusters_logic():
    import el_bank_state_probe as P
    es = _journal(["a", "b", "c"], {"a": True, "b": False, "c": True}, ["s1", "s2", "s1"])
    lab = P.labels_from_journal(es)
    assert lab["order"] == ["a", "b", "c"] and lab["pass_all"] == {"a": True, "b": False, "c": True} and lab["majority_cluster"] == {"a": True, "b": False, "c": True}
    # negative: a candidate with one failing example is not pass_all; a candidate without evidence has no label at all
    es2 = es + [{"seq": 99, "kind": "evidence", "data": {"evidence": {"candidate_id": "a", "check_id": "public_examples_v1", "status": "fail"}, "outcome": {}}}]
    assert P.labels_from_journal(es2)["pass_all"]["a"] is False and "d" not in P.labels_from_journal(es)["pass_all"]


@test
def b2_labels_from_a_real_saved_journal_logic():
    import el_bank_state_probe as P
    jp = SAVED / "tasks/J1-02/journal.jsonl"
    if not jp.exists():
        raise B.Skip("saved bank run not on this machine")
    lab = P.labels_from_journal([json.loads(l) for l in jp.read_text("utf-8").splitlines() if l.strip()])
    assert len(lab["order"]) == 4 and all(lab["pass_all"].values()) and len(lab["pass_all"]) == 4 and all(lab["majority_cluster"].values())


@test
def b2_probe_finds_a_planted_signal_at_the_right_depth_and_not_where_there_is_none_logic():
    import el_bank_state_probe as P
    root = _make_run(signal_depth=1)
    rows, info = P.load_rows(root, [0, 1, 2])
    assert info["states"] == 84 and info["tasks_with_states"] == 14 and info["hidden"] == [24]
    d1 = P.analyze(rows, "pass_all", 1, boot=200)
    d0 = P.analyze(rows, "pass_all", 0, boot=200)
    assert d1["status"] == "ok" and d1["auroc"] > 0.9 and d1["auroc_ci95"][0] > 0.7 and d1["top1"]["probe"] > d1["top1"]["random"] + 0.2 and d1["top1_ci95"]["probe_minus_random"][0] > 0
    assert d0["auroc"] < 0.75 and d0["auroc"] < d1["auroc"] - 0.2                                                    # the other depth carries no planted signal
    for d in (d0, d1):
        assert d["selector"] is False and d["can_approve"] is False and "exploratory" in d["note"] and "never overrides trusted evidence" in d["note"]
    null = P.analyze(P.load_rows(_make_run(signal=0.0, seed=3), [0, 1, 2])[0], "pass_all", "all", boot=200)
    assert null["auroc"] < 0.7 and null["auroc_ci95"][0] < 0.5 + 0.05                                                  # negative control: no signal, the interval reaches chance


@test
def b2_cross_validation_is_grouped_by_task_so_a_task_fingerprint_does_not_leak_logic():
    import numpy as np
    import el_bank_state_probe as P
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    rows = []
    for t in range(12):                                                    # every task has its own pass rate and an unmistakable fingerprint in its states; inside a task nothing predicts the label
        n_pass = 1 if t % 2 == 0 else 7
        for i in range(8):
            x = np.zeros(12)
            x[t] = 6.0
            x += np.random.RandomState(100 * t + i).randn(12) * 0.3
            rows.append({"task": "T%02d" % t, "order": i, "x": {0: x.tolist()}, "y": {"pass_all": i < n_pass}})
    r = P.analyze(rows, "pass_all", 0, boot=100)
    assert r["auroc"] < 0.62                                               # held-out TASKS have unseen fingerprints: nothing to exploit
    X = np.array([x["x"][0] for x in rows]); y = np.array([int(x["y"]["pass_all"]) for x in rows]); sc = np.zeros(len(y))
    for tr, te in KFold(4, shuffle=True, random_state=0).split(X):          # negative control: ungrouped CV sees every task in training and learns the fingerprint
        sc[te] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr]).decision_function(X[te])
    assert P.auroc(y.tolist(), sc.tolist()) > r["auroc"] + 0.1


@test
def b2_few_tasks_say_insufficient_data_and_report_carries_the_labels_logic():
    import el_bank_state_probe as P
    root = _make_run(n_tasks=4)
    rep = P.report(root, boot=50)
    d = rep["by_depth"]["depth_19"]["pass_all"]
    assert rep["label"].startswith("exploratory, few tasks") and d["status"] == "INSUFFICIENT_DATA" and d["n_informative_tasks"] <= 4 and rep["status"] == "ok"
    assert set(rep["by_depth"]) == {"depth_14", "depth_19", "depth_23", "all_depths"} and rep["frozen_4b_laya_head"]["can_consume_these_states"] is False
    empty = Path(tempfile.mkdtemp(prefix="f2c-empty-"))
    (empty / "tasks").mkdir()
    assert P.report(empty, boot=10)["status"] == "NO_STATES" and P.report(Path(tempfile.mkdtemp(prefix="f2c-none-")), boot=10)["status"] == "NO_STATES"


@test
def b2_frozen_4b_laya_head_refuses_the_1536_dim_states_logic():
    import torch
    import el_bank_state_probe as P
    from evidence_ladder.models import probe as PR
    st = {"candidate_id": "x", "ho": torch.zeros(3, 1536).half()}
    try:
        P.frozen_laya_guard(st)
        assert False, "the frozen head's row check accepted a 1536-dim state"
    except PR.ProbeFitError:
        pass
    ok = {"decision_id": "p", "ho": torch.zeros(2, 3, 2560), "pos": torch.tensor([True, False])}                      # negative control: the shape the head expects is accepted
    PR.check_rows([ok])


@test
def b2_cli_is_create_only_and_writes_a_report_logic():
    import el_bank_state_probe as P
    root = _make_run(n_tasks=3)
    out = Path(tempfile.mkdtemp(prefix="f2c-out-")) / "r.json"
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        rc1 = P.main([str(root), "--out", str(out), "--boot", "20"])
        rc2 = P.main([str(root), "--out", str(out), "--boot", "20"])
    assert rc1 == 0 and json.loads(out.read_text("utf-8"))["label"].startswith("exploratory")
    assert rc2 == 2                                                                                                      # never overwrites


# ====================================================================== CONFORMANCE tool (tools/el_bank_conformance.py)
def _conf():
    import el_bank_conformance as CF
    return CF


def _fixture(tasks=("J1-08", "J1-12")):
    """copy the REAL saved Kaggle outputs of the named tasks (+ RUN_SUMMARY, TIMELINE.jsonl) into a temp run dir"""
    import shutil
    if not SAVED.exists():
        raise B.Skip("saved bank run not on this machine: %s" % SAVED)
    run = Path(tempfile.mkdtemp(prefix="f2c-conf-"))
    for f in ("RUN_SUMMARY.json", "TIMELINE.jsonl"):
        shutil.copy(SAVED / f, run / f)
    for t in tasks:
        shutil.copytree(SAVED / "tasks" / t, run / "tasks" / t)
    return run


def _audit(run, **kw):
    kw.setdefault("expect", "kaggle-bwrap-jail")
    kw.setdefault("legacy", True)
    if "tasks_path" not in kw or kw["tasks_path"] is None:      # the fixture holds a subset of the 40 tasks: the expected list is the copied tasks (the full-run case is tested with an explicit list)
        ex = Path(run) / "expected_tasks.json"
        ex.write_text(json.dumps(sorted(p.name for p in (Path(run) / "tasks").iterdir())), "utf-8")
        kw["tasks_path"] = str(ex)
    return _conf().audit_run(run, **kw)


def _edit_json(p, fn):
    d = json.loads(p.read_text("utf-8"))
    fn(d)
    p.write_text(json.dumps(d), "utf-8")


@test
def c_real_saved_tasks_are_pipeline_ok_and_the_run_is_complete_for_them_logic():
    run = _fixture()
    rep = _audit(run, tasks_path=None)
    assert rep["verdict"] == "PIPELINE_COMPLETE" and rep["counts"]["executed"] == 2 and rep["counts"]["broken"] == 0, rep["broken"]
    r = rep["tasks"]["J1-08"]
    assert r["verdict"] == "PIPELINE_OK" and all(c["ok"] for c in r["checks"].values())
    assert r["checks"]["l2"]["info"].get("model_produced_nothing") is True                                               # J1-08 truly produced nothing for L2 (ceiling): the reason is shown
    assert r["checks"]["stages"]["info"]["investigate"]["status"].startswith("NOT_APPLICABLE") and r["checks"]["stages"]["info"]["intake"]["start_utc"]


@test
def c_the_real_kaggle_l2_parser_mismatch_is_caught_as_broken_logic():
    run = _fixture(("J1-02",))
    rep = _audit(run)
    assert rep["verdict"] == "PIPELINE_BROKEN" and "J1-02" in rep["broken"]
    msg = "; ".join(rep["broken"]["J1-02"])
    assert "no valid test parsed" in msg and "missing_quote" in msg and "parser/prompt mismatch" in msg
    assert [k for k, c in rep["tasks"]["J1-02"]["checks"].items() if not c["ok"]] == ["l2"]                              # nothing else is wrong with that task


@test
def c_missing_stage_broken_chain_wrong_sandbox_and_hidden_marker_are_each_caught_logic():
    assert _audit(_fixture())["verdict"] == "PIPELINE_COMPLETE"                                                          # control: the uncorrupted copy
    t = "J1-08"
    # 1. a missing stage (rank and commit_verify removed from the timeline)
    r1 = _fixture()
    _edit_json(r1 / "tasks" / t / "controller_timeline.json", lambda d: d.update(rows=[x for x in d["rows"] if not (x["row"] == "stage" and x["name"] in ("rank", "commit_verify"))]))
    rep = _audit(r1)
    assert rep["verdict"] == "PIPELINE_BROKEN" and any("stage rank missing" in p or "stage commit_verify missing" in p for p in rep["broken"][t])
    # 2. a broken hash chain (one journal entry edited)
    r2 = _fixture()
    jp = r2 / "tasks" / t / "journal.jsonl"
    lines = jp.read_text("utf-8").splitlines()
    e = json.loads(lines[3])
    e["data"] = {"tampered": True}
    lines[3] = json.dumps(e)
    jp.write_text("\n".join(lines) + "\n", "utf-8")
    assert any(p.startswith("chain: hash chain broken at entry") for p in _audit(r2)["broken"][t])
    # 3. the wrong sandbox label (and a WSL engine named inside the task result)
    r3 = _fixture()
    _edit_json(r3 / "tasks" / t / "result.json", lambda d: d.update(engine="wsl-bwrap"))
    msgs = _audit(r3)["broken"][t]
    assert any("not the expected" in p for p in msgs) and any("forbidden execution backend" in p for p in msgs)
    assert _audit(_fixture(), expect="mxc")["verdict"] == "PIPELINE_BROKEN"                                               # expecting mxc on a bwrap-jail run
    # 4. a hidden-data marker anywhere in the task files
    r4 = _fixture()
    gp = r4 / "tasks" / t / "gen_calls.json"
    gp.write_text(gp.read_text("utf-8").replace("{", '{"note": "see hidden_cases file", ', 1), "utf-8")
    assert any("hidden/protected marker 'hidden_cases'" in p for p in _audit(r4)["broken"][t])
    # 5. invariants not all recorded; journal entries without UTC rows
    r5 = _fixture()
    _edit_json(r5 / "tasks" / t / "result.json", lambda d: d.update(invariants={"checked": ["I1", "I2"], "failed": None}))
    assert any(p.startswith("invariants:") for p in _audit(r5)["broken"][t])
    r6 = _fixture()
    _edit_json(r6 / "tasks" / t / "controller_timeline.json", lambda d: d.update(rows=[x for x in d["rows"] if not (x["row"] == "entry" and x["seq"] == 5)]))
    assert any(p.startswith("utc_rows:") for p in _audit(r6)["broken"][t])


@test
def c_repair_run_may_enter_rank_before_l2_but_an_unrepaired_misorder_is_broken_logic():
    t = "J1-02"                                                                                                             # has real l2 / cluster stages (its only other defect is the l2 mismatch)

    def reorder(d, with_repair):
        rows = d["rows"]
        st = [r for r in rows if r["row"] == "stage"]
        pick = lambda n: [r for r in st if r["name"] == n][:1]
        mid = [r for r in st if r["name"] not in ("intake", "generate", "execute", "commit_verify", "l2_generate", "l2_validate", "cluster")]
        head = pick("intake") + pick("generate") + pick("execute")
        ordered = head + [r for r in st if r["name"] in ("discriminate", "rank")][:2] + ([dict(pick("generate")[0], name="repair")] if with_repair else []) + pick("l2_generate") + pick("l2_validate") + pick("cluster") + pick("commit_verify")
        d["rows"] = [r for r in rows if r["row"] != "stage"] + ordered
    ok = _fixture((t,))
    _edit_json(ok / "tasks" / t / "controller_timeline.json", lambda d: reorder(d, True))
    assert "stage order wrong" not in "; ".join(_audit(ok).get("broken", {}).get(t, []))                                    # repair round present: allowed
    bad = _fixture((t,))
    _edit_json(bad / "tasks" / t / "controller_timeline.json", lambda d: reorder(d, False))
    assert any("stage order wrong" in p for p in _audit(bad)["broken"][t])                                                  # same order without a repair round: BROKEN


@test
def c_strict_mode_needs_policy_profile_l2_version_and_exec_records_logic():
    run = _fixture()
    rep = _audit(run, legacy=False)
    msgs = "; ".join(rep["broken"]["J1-08"])
    assert "profile sha256 not recorded" in msgs and "l2_version not recorded" in msgs and "exec_results.jsonl missing" in msgs
    # recorded -> accepted; a record that lacks one of the 8 contract fields -> broken
    keys = _conf().RESULT_KEYS
    for t in ("J1-08", "J1-12"):
        _edit_json(run / "tasks" / t / "result.json", lambda d: d.update(engine="mxc", profile_sha256="a" * 64, l2_version=2))
        (run / "tasks" / t / "exec_results.jsonl").write_text(json.dumps({k: 0 for k in keys}) + "\n", "utf-8")
    ok = _audit(run, legacy=False, expect="mxc")
    assert ok["verdict"] == "PIPELINE_COMPLETE", ok["broken"]
    (run / "tasks" / "J1-08" / "exec_results.jsonl").write_text(json.dumps({k: 0 for k in keys if k != "post_tree_sha256"}) + "\n", "utf-8")
    assert any("lacks ['post_tree_sha256']" in p for p in _audit(run, legacy=False, expect="mxc")["broken"]["J1-08"])


@test
def c_no_task_is_silently_missing_not_tested_error_and_missing_are_listed_logic():
    run = _fixture()
    cfg = run / "tasks.json"
    cfg.write_text(json.dumps(["J1-08", "J1-12", "J1-99", "J1-98", "J1-97"]), "utf-8")
    nt = run / "tasks" / "J1-98"
    nt.mkdir()
    (nt / "NOT_TESTED").write_text("", "utf-8")
    (nt / "result.json").write_text(json.dumps({"status": "NOT_TESTED", "reason": "Java is not supported by the MXC sandbox"}), "utf-8")
    er = run / "tasks" / "J1-97"
    er.mkdir()
    (er / "result.json").write_text(json.dumps({"status": "ERROR"}), "utf-8")
    rep = _audit(run, tasks_path=str(cfg))
    assert rep["verdict"] == "PIPELINE_PARTIAL" and rep["missing"] == ["J1-99"] and rep["not_tested"] == {"J1-98": "Java is not supported by the MXC sandbox"}
    assert [e["task"] for e in rep["error"]] == ["J1-97"] and rep["counts"]["executed"] == 2
    md = _conf().to_markdown(rep)
    assert "MISSING" in md and "J1-99" in md and "Java is not supported" in md and "J1-97" in md
    (run / "tasks" / "J1-99").mkdir()                                                                                       # negative: a task dir without any marker is an ERROR entry, never dropped
    assert "J1-99" in [e["task"] for e in _audit(run, tasks_path=str(cfg))["error"]]


@test
def c_states_are_outside_the_journal_and_a_foreign_state_is_caught_logic():
    import torch
    run = _fixture(("J1-12",))
    td = run / "tasks" / "J1-12"
    cands = [json.loads(l)["data"]["id"] for l in (td / "journal.jsonl").read_text("utf-8").splitlines() if '"kind": "candidate"' in l]
    (td / "states").mkdir()
    before = _audit(run)["tasks"]["J1-12"]
    from evidence_ladder.models.state_capture import StateCapture
    sc = StateCapture(lambda c, o: (torch.zeros(3, 4).half(), torch.zeros(1, 3, 4).half(), {"status": "ok"}), run, depths=(1, 2, 3))
    for c in cands:
        sc.capture("J1-12", c, "ctx", "src")
    after = _audit(run)
    assert after["verdict"] == "PIPELINE_COMPLETE" and after["tasks"]["J1-12"]["checks"]["states"]["info"]["states"] == len(cands) and before["verdict"] == "PIPELINE_OK"
    sc.capture("J1-12", "f" * 64, "ctx", "src")                                                                             # a state of a candidate the journal never had
    assert any("does not belong to a journaled candidate" in p for p in _audit(run)["broken"]["J1-12"])


@test
def c_state_hook_is_hash_neutral_selftest_and_cli_is_create_only_logic():
    CF = _conf()
    st = CF.selftest()
    assert st["identical"] and st["states_saved"] == 2
    run = _fixture()
    import contextlib
    import io
    out = Path(tempfile.mkdtemp(prefix="f2c-confout-"))
    with contextlib.redirect_stdout(io.StringIO()):
        ex = run / "expected_tasks.json"
        ex.write_text(json.dumps(["J1-08", "J1-12"]), "utf-8")
        rc1 = CF.main([str(run), "--legacy-kaggle", "--expect-engine", "kaggle-bwrap-jail", "--tasks", str(ex), "--out", str(out)])
        rc2 = CF.main([str(run), "--legacy-kaggle", "--expect-engine", "kaggle-bwrap-jail", "--tasks", str(ex), "--out", str(out)])
    new = out / "does" / "not" / "exist"                                                                                      # --out that does not exist is created (files inside stay create-only)
    with contextlib.redirect_stdout(io.StringIO()):
        rc3 = CF.main([str(run), "--legacy", "--expect-engine", "kaggle-bwrap-jail", "--tasks", str(ex), "--out", str(new)])
        rc4 = CF.main([str(run), "--expect-engine", "kaggle-bwrap-jail", "--tasks", str(ex), "--no-write"])                 # default is strict: the old run lacks the provenance keys
    assert rc3 == 0 and (new / "CONFORMANCE.json").exists() and rc4 == 1
    assert rc1 == 0 and rc2 == 2 and json.loads((out / "CONFORMANCE.json").read_text("utf-8"))["verdict"] == "PIPELINE_COMPLETE" and "PIPELINE_COMPLETE" in (out / "CONFORMANCE.md").read_text("utf-8")


@test
def c_capture_hook_is_not_in_the_controller_decision_path_logic():
    src = {p.name: p.read_text("utf-8") for p in sorted((ROOT / "evidence_ladder/controller").glob("*.py"))}
    assert not [n for n, t in src.items() if "state_capture" in t or "StateCapture" in t]                                   # no controller module imports or calls the hook
    from evidence_ladder.models.state_capture import StateCapture
    off = StateCapture(lambda c, o: 1 / 0, tempfile.mkdtemp(prefix="f2c-off-"), enabled=False)                              # disabled: not even the capture function is called
    assert off.capture("T", "c", "x", "y") is None and off.errors == [] and off.saved == 0


# ====================================================================== runner
def main():
    passed, failed = [], {}
    for name, f in sorted(TESTS.items()):
        try:
            f()
            passed.append(name)
        except B.Skip as e:
            SKIPPED[name] = str(e)
        except Exception:      # noqa: BLE001
            failed[name] = traceback.format_exc()
    print(json.dumps({"tests": len(TESTS), "passed": len(passed), "failed": failed, "skipped": SKIPPED}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
