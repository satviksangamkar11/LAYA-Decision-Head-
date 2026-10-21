"""[P5 Step 2, post-hoc diagnostics — NOT a benchmark] Offline analysis of the SAVED P5 final run (results/raw/p5-final-v1). No model generation.
Hidden grading is used only to describe saved candidates; nothing here feeds back into runtime selection, contracts or the reported P5 score.
  A  contracts on B: the frozen runtime contract check (contract edges + l2.derive_inputs(public, 6), contract_v1.evaluate, authority from the host registry /
     decl_v1) applied to B's final answer and to every B public-passing candidate of the contract cohort; blind v3 grade of each.
  B  ladder abstention audit (LiveCodeBench): for the tasks where C (and D) gave no answer: stop causes; tasks with a public-passing ladder candidate;
     tasks where a saved ladder candidate passes hidden grading (selection failure) vs none (generation failure); same counts for B's pool on those tasks.
  C  equal-spend replay of B: B's saved calls in their real order (shared round 1 first), included while the cumulative generated tokens stay <= the tokens C
     actually reported for that task (whole candidates only); answer = first included candidate with a complete public pass (B's rule); graded.
     Sanity: the replay with no cap must reproduce B's actual answers.
  D  disagreement feasibility: per task, the distinct public-passing candidates of B, C and D are run on derived inputs (contract cohort: 12 derived + edges,
     contract-valid only; LiveCodeBench functional: l2 variants of the JSON arguments; stdin: integer-token variants, validity UNKNOWN); candidates are grouped
     by output signature; a disagreement is "authoritatively resolved" only when trusted contract evidence leaves exactly one class (LiveCodeBench has none).
Writes <out>/DIAG.json (create-only), counts and task ids only (no hidden cases). Usage: PYTHONPATH=. python tools/el_p5_diagnostics.py <new out dir>"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import el_p5_run as P5                                                            # noqa: E402
import lcb_tasks as L                                                             # noqa: E402
from evidence_ladder.common.interfaces import Artifact                           # noqa: E402
from evidence_ladder.contracts import contract_v1 as CV, registry as CR          # noqa: E402
from evidence_ladder.controller import l2                                         # noqa: E402
from evidence_ladder.controller.sandbox_provider import make_sandbox             # noqa: E402
from evidence_ladder.execution import config as K, engine                        # noqa: E402
from evidence_ladder.execution.lcb_runner import LcbRunner                       # noqa: E402
from evidence_ladder.execution.provider_engine import ProviderExecutionBackend   # noqa: E402

FIN = ROOT / "results/raw/p5-final-v1"
OUT = Path(sys.argv[1]).resolve()
if OUT.exists():
    sys.exit("refusing: %s exists" % OUT)
OUT.mkdir(parents=True)
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
LOCK = json.loads((ROOT / "configs/p5_final_lock_v2.json").read_text("utf-8"))
rows = lambda p: [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]
CLAIM = {"SUPPORTED", "VERIFIED"}

# ------------------------------------------------------------------ data
SHARDS = {"lcb": ["final_lcb_%d" % k for k in range(1, 6)], "contract": ["final_contract"]}


def shard_of(cohort):
    data = {"res": {}, "ans": {}, "calls": defaultdict(list), "p2calls": defaultdict(list), "p2cand": defaultdict(list), "journal": {}}
    for s in SHARDS[cohort]:
        d = FIN / ("p5_" + s)
        for r in rows(d / "RESULTS.jsonl"):
            data["res"][(r["task"], r["arm"])] = r
        for r in rows(d / "ANSWERS.jsonl"):
            data["ans"][(r["task"], r["arm"])] = r["source"]
        for r in rows(d / "CALLS.jsonl"):
            data["calls"][r["task"]].append(r)
        for r in rows(d / "P2CALLS.jsonl"):
            data["p2calls"][r["task"]].append(r)
        for r in rows(d / "P2CANDIDATES.jsonl"):
            data["p2cand"][r["task"]].append(r)
        for j in (d / "journals").glob("*.jsonl"):
            t, arm = j.stem.rsplit("_", 1)
            data["journal"][(t, arm)] = rows(j)
    return data


def ladder_candidates(journal, n_public):
    src = {e["data"]["id"]: e["data"]["source"] for e in journal if e["kind"] == "candidate"}
    st = defaultdict(list)
    for e in journal:
        if e["kind"] == "evidence" and e["data"]["evidence"]["check_id"] == "public_examples_v1":
            st[e["data"]["evidence"]["candidate_id"]].append(e["data"]["evidence"]["status"])
    return [{"id": i, "source": s, "public_pass": len(st[i]) >= n_public and all(x == "pass" for x in st[i])} for i, s in src.items()]


# ------------------------------------------------------------------ execution (MXC on the PC, absolute scratch paths)
cfg = K.SandboxConfig()
exe = ProviderExecutionBackend(backend="mxc", scratch=OUT / "_engine", config=cfg, trusted=True)
exe.start(languages=("python",))
runner = LcbRunner(make_sandbox(OUT / "_sandbox", [], backend="mxc").run, OUT / "_lcb")
T = L.load()
raw_v3 = (ROOT / "private/p5-eval-v3/cases_v3.json").read_bytes()
assert hashlib.sha256(raw_v3).hexdigest() == LOCK["grading"]["primary"]["cases_sha256"], "blind set differs from its lock"
CASES_V3 = json.loads(raw_v3.decode("utf-8"))
CTASKS = {t.task_id: t for t, _ in P5.load_tasks("contract", LOCK["contract_ids"])[0]}
REG = CR.load_registry(ROOT / "configs/p5_contract_tasks_v1/REGISTRY_v1.json")
CONTRACT = {tid: CR.trusted_contract(tid, REG, ROOT, CTASKS[tid].prompt, ["decl_v1"]) for tid in CTASKS}
_grade_cache = {}


def hidden_ok(cohort, tid, src):
    k = (cohort, tid, sha(src))
    if k not in _grade_cache:
        if cohort == "lcb":
            pv, go = T[tid]["public_view"], T[tid]["grader_only"]
            _grade_cache[k] = runner.grade(src, list(pv["public_tests"]) + list(go["private_tests"]), pv["fn_name"], stop_at_first_fail=True)["pass"]
        else:
            res = exe._engine_run("python", src.encode("utf-8"), [c["input"] for c in CASES_V3[tid]])
            _grade_cache[k] = engine.judge_each("python", res, CASES_V3[tid], cfg)[0].status == "pass"
    return _grade_cache[k]


def contract_verdict(tid, src, n_derived=6):
    """The frozen runtime check (controller.contract_check): 'fail' / 'pass' / 'no_contract' / 'tool_failure'."""
    ct = CONTRACT[tid]
    if ct is None or not ct.authoritative:
        return "no_contract"
    task = CTASKS[tid]
    inputs = list(ct.edges) + l2.derive_inputs(list(task.public_tests), n=n_derived)
    try:
        results, term = exe.run_inputs(Artifact.make(tid, "python", src), task, inputs)
    except Exception:      # noqa: BLE001
        return "tool_failure"
    return "fail" if CV.evaluate(ct, inputs, results, whole_run_crashed=(term == "crash"))["hard"] else "pass"


# ------------------------------------------------------------------ A. contracts on B (contract cohort)
def diag_a(C):
    out = {"final_answers": [], "public_passing_candidates": defaultdict(int)}
    for tid in LOCK["contract_ids"]:
        src = C["ans"].get((tid, "B"))
        if src:
            v, ok = contract_verdict(tid, src), hidden_ok("contract", tid, src)
            out["final_answers"].append({"task": tid, "contract": v, "blind_correct": ok})
        for c in C["p2cand"][tid]:
            if c["public_pass"]:
                v, ok = contract_verdict(tid, c["source"]), hidden_ok("contract", tid, c["source"])
                out["public_passing_candidates"]["%s/%s" % ("correct" if ok else "wrong", v)] += 1
    fa = out["final_answers"]
    out["summary"] = {"B_answers": len(fa), "correct_rejected": [r["task"] for r in fa if r["blind_correct"] and r["contract"] == "fail"],
                      "wrong_rejected": [r["task"] for r in fa if not r["blind_correct"] and r["contract"] == "fail"],
                      "wrong_passed": [r["task"] for r in fa if not r["blind_correct"] and r["contract"] != "fail"],
                      "correct_passed": sum(1 for r in fa if r["blind_correct"] and r["contract"] != "fail"),
                      "tool_failures": [r["task"] for r in fa if r["contract"] == "tool_failure"],
                      "B_plus_D_solved_if_rejection_means_no_answer": sum(1 for r in fa if r["blind_correct"] and r["contract"] != "fail")}
    out["public_passing_candidates"] = dict(out["public_passing_candidates"])
    return out


# ------------------------------------------------------------------ B. ladder abstention audit (LiveCodeBench)
def diag_b(Lc):
    abst = [t for t in LOCK["lcb_ids"] if Lc["res"][(t, "C")]["label"] not in CLAIM]
    same_d = all((Lc["res"][(t, "D")]["label"] in CLAIM) == (Lc["res"][(t, "C")]["label"] in CLAIM) for t in LOCK["lcb_ids"])
    stops = defaultdict(int)
    per = []
    for t in abst:
        stops[Lc["res"][(t, "C")].get("stop")] += 1
        npub = len(T[t]["public_view"]["public_tests"])
        lad = ladder_candidates(Lc["journal"][(t, "C")], npub)
        lad_pub = [c for c in lad if c["public_pass"]]
        lad_hidden = [c for c in lad_pub if hidden_ok("lcb", t, c["source"])]
        b_pub = [c for c in Lc["p2cand"][t] if c["public_pass"]]
        b_hidden = [c for c in b_pub if hidden_ok("lcb", t, c["source"])]
        per.append({"task": t, "stop": Lc["res"][(t, "C")].get("stop"), "ladder_candidates": len(lad), "ladder_public_pass": len(lad_pub),
                    "ladder_hidden_pass": len(lad_hidden), "B_public_pass": len(b_pub), "B_hidden_pass": len(b_hidden)})
    s = {"abstentions": len(abst), "D_same_as_C": same_d, "stop_causes": dict(stops),
         "tasks_with_ladder_public_pass": sum(1 for p in per if p["ladder_public_pass"]),
         "tasks_with_ladder_hidden_pass_selection_failure": [p["task"] for p in per if p["ladder_hidden_pass"]],
         "tasks_without_any_ladder_public_pass_generation_failure": sum(1 for p in per if not p["ladder_public_pass"]),
         "of_those_B_found_hidden_pass": [p["task"] for p in per if not p["ladder_public_pass"] and p["B_hidden_pass"]],
         "tasks_where_B_solved_and_ladder_abstained": [p["task"] for p in per if p["B_hidden_pass"]]}
    return {"summary": s, "per_task": per}


# ------------------------------------------------------------------ C. equal-spend replay of B
def b_sequence(D, t):
    cand = {c["seed"]: c for c in D["p2cand"][t]}
    r1 = sorted((c for c in D["calls"][t] if c["arm"] == "round1"), key=lambda c: c["seed"])
    seq = [(c["seed"], c["tokens_generated"]) for c in r1] + [(c["seed"], c["tokens_generated"]) for c in D["p2calls"][t]]
    return [(s, tok, cand.get(s)) for s, tok in seq]


def replay(D, t, cap):
    cum = 0
    for s, tok, c in b_sequence(D, t):
        cum += tok
        if cap is not None and cum > cap:
            return None, cum - tok
        if c and c["public_pass"]:
            return c["source"], cum
    return None, cum


def diag_c(cohort, D, ids):
    per, match = [], 0
    for t in ids:
        full, _ = replay(D, t, None)
        match += (full or None) == (D["ans"].get((t, "B")) or None)
        cap = D["res"][(t, "C")].get("tokens_generated") or 0
        src, spent = replay(D, t, cap)
        per.append({"task": t, "cap_C_tokens": cap, "spent": spent, "answered": src is not None,
                    "correct": bool(src) and hidden_ok(cohort, t, src), "B_actual_tokens": D["res"][(t, "B")].get("tokens_generated")})
    return {"sanity_uncapped_replay_matches_B_answers": "%d/%d" % (match, len(ids)),
            "B_at_C_spend": {"solved": sum(p["correct"] for p in per), "answered": sum(p["answered"] for p in per),
                             "false_supported": sum(p["answered"] and not p["correct"] for p in per), "tokens": sum(p["spent"] for p in per)},
            "C_tokens": sum(p["cap_C_tokens"] for p in per), "per_task": per}


# ------------------------------------------------------------------ D. disagreement feasibility
def lcb_inputs(t, n=12):
    pv = T[t]["public_view"]
    pub = pv["public_tests"]
    tt = pub[0]["testtype"]
    if tt == "functional":
        dicts = []
        for p in pub:
            try:
                args = [json.loads(x) for x in p["input"].split("\n") if x.strip()]
            except ValueError:
                return tt, []
            dicts.append({"input": {"a%02d" % k: v for k, v in enumerate(args)}})
        return tt, ["\n".join(json.dumps(d[k]) for k in sorted(d)) for d in l2.derive_inputs(dicts, n=n)]
    out, seen = [], {p["input"] for p in pub}
    for p in pub:
        toks = list(re.finditer(r"-?\d+", p["input"]))
        for m in toks:
            for nv in (0, 1, int(m.group()) + 1):
                x = p["input"][:m.start()] + str(nv) + p["input"][m.end():]
                if x not in seen and len(out) < n:
                    seen.add(x); out.append(x)
    return tt, out


def signature_lcb(t, src, inputs, tt):
    fn = T[t]["public_view"]["fn_name"]
    sig = []
    for x in inputs:
        o = runner.run_case(src, {"input": x, "output": "null" if tt == "functional" else "", "testtype": tt}, fn)
        sig.append(o["status"] if o["status"] in ("timeout", "runtime_error") else o.get("observed", ""))
    return tuple(sig)


def signature_contract(tid, src, inputs):
    results, term = exe.run_inputs(Artifact.make(tid, "python", src), CTASKS[tid], inputs)
    return tuple(json.dumps(r, sort_keys=True, default=str) if r else "none" for r in results)


def diag_d(cohort, D, ids):
    per = []
    for t in ids:
        n_public = len(T[t]["public_view"]["public_tests"]) if cohort == "lcb" else len(CTASKS[t].public_tests)
        pool = {}
        for c in D["p2cand"][t]:
            if c["public_pass"]:
                pool.setdefault(sha(c["source"]), c["source"])
        for arm in ("C", "D"):
            for c in ladder_candidates(D["journal"].get((t, arm), []), n_public):
                if c["public_pass"]:
                    pool.setdefault(sha(c["source"]), c["source"])
        rec = {"task": t, "distinct_public_passing": len(pool)}
        if len(pool) >= 2:
            if cohort == "lcb":
                tt, inputs = lcb_inputs(t)
                rec["input_kind"] = tt + (" (validity unknown)" if tt == "stdin" else "")
                sigs = {h: signature_lcb(t, s, inputs, tt) for h, s in pool.items()} if inputs else {}
            else:
                ct = CONTRACT[t]
                cand_in = list(ct.edges) + l2.derive_inputs(list(CTASKS[t].public_tests), n=12) if ct else l2.derive_inputs(list(CTASKS[t].public_tests), n=12)
                inputs = [x for x in cand_in if ct is None or ct.valid(x) is True]
                rec["input_kind"] = "contract-valid derived"
                sigs = {h: signature_contract(t, s, inputs) for h, s in pool.items()} if inputs else {}
            rec["inputs"] = len(inputs)
            classes = defaultdict(list)
            for h, sg in sigs.items():
                classes[sg].append(h)
            rec["classes"] = len(classes)
            rec["disagreement"] = len(classes) >= 2
            correct = {h for h, s in pool.items() if hidden_ok(cohort, t, s)}
            rec["classes_with_a_correct_candidate"] = sum(1 for hs in classes.values() if set(hs) & correct)
            big = max(classes.values(), key=len) if classes else []
            rec["plurality_class_has_correct"] = bool(set(big) & correct)          # exploratory: output voting, not authoritative
            if cohort == "contract" and rec["disagreement"]:
                surv = [hs for hs in classes.values() if any(contract_verdict(t, pool[h]) not in ("fail",) for h in hs)]
                rec["surviving_classes_after_trusted_contract"] = len(surv)
                rec["authoritatively_resolved"] = len(surv) == 1
                rec["resolved_to_correct"] = len(surv) == 1 and bool(set(surv[0]) & correct)
            else:
                rec["authoritatively_resolved"] = False
        per.append(rec)
    dis = [p for p in per if p.get("disagreement")]
    return {"tasks_with_2plus_public_passing": sum(1 for p in per if p["distinct_public_passing"] >= 2), "tasks_with_disagreement": len(dis),
            "authoritatively_resolved": sum(1 for p in dis if p.get("authoritatively_resolved")),
            "resolved_to_correct": sum(1 for p in dis if p.get("resolved_to_correct")),
            "disagreements_with_a_correct_class": sum(1 for p in dis if p["classes_with_a_correct_candidate"]),
            "plurality_class_correct_exploratory": sum(1 for p in dis if p["plurality_class_has_correct"]), "per_task": per}


def main():
    Lc, Cc = shard_of("lcb"), shard_of("contract")
    rep = {"note": "post-hoc diagnostics on the saved P5 final run; hidden grades describe saved candidates only; not a benchmark and not a P5 score",
           "A_contracts_on_B": diag_a(Cc), "B_ladder_abstention_audit_lcb": diag_b(Lc),
           "C_equal_spend_replay": {"lcb": diag_c("lcb", Lc, LOCK["lcb_ids"]), "contract": diag_c("contract", Cc, LOCK["contract_ids"])},
           "D_disagreement": {"lcb": diag_d("lcb", Lc, LOCK["lcb_ids"]), "contract": diag_d("contract", Cc, LOCK["contract_ids"])}}
    (OUT / "DIAG.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    brief = {"A": rep["A_contracts_on_B"]["summary"], "A_candidates": rep["A_contracts_on_B"]["public_passing_candidates"],
             "B": rep["B_ladder_abstention_audit_lcb"]["summary"],
             "C": {k: {kk: vv for kk, vv in v.items() if kk != "per_task"} for k, v in rep["C_equal_spend_replay"].items()},
             "D": {k: {kk: vv for kk, vv in v.items() if kk != "per_task"} for k, v in rep["D_disagreement"].items()}}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
