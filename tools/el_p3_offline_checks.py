"""[P3] Offline verification experiment, DECISION step (no grader access). Run INSIDE WSL (reference engine: engine.run, bubblewrap).
For every stored candidate that passed all public examples (P1 runs, P2 local Python, P2 Kaggle), execute: the spec's explicitly permitted edge inputs and l2.derive_inputs() of the
public examples; each input is classified by the task spec as valid / invalid / uncertain (tools/p3_specs_v1.py, written from task text and public examples only).
  HARD failure (would change the label): a crash on a VALID input, or a stated property violated on a VALID input.
  UNCERTAIN: anything on invalid/uncertain inputs, timeouts or abnormal run ends, unparsable output.
  SHADOW (recorded only): disagreement between public-passing candidates of the same pool on VALID inputs.
Writes <out>/DECISIONS.jsonl, POOLS.jsonl, MANIFEST.json (sha256 of specs, this tool, decisions). The grader is not imported; scoring is a separate step.
Usage (inside WSL): python3 tools/el_p3_offline_checks.py <new out dir>"""
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from evidence_ladder.execution import config as K, engine          # noqa: E402
from evidence_ladder.controller import l2                           # noqa: E402
import p3_specs_v1 as S                                              # noqa: E402

P1_RUNS = ["results/raw/el-bank-local-20261018T2106", "results/raw/el-bank-java5hard-wslc-20261018T213058", "results/raw/el-bank-java20-wslc-20261018T211547"]
P2_LOCAL = "results/raw/p2-generation-compare-v2-20261019T215005Z"          # Python rows only (Java rows invalid: verifier v2)
P2_KAGGLE = sorted(glob.glob(str(ROOT / "results/raw/p2-kaggle-out-*/p2_shard*")))


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tasks_by_id():
    man = json.loads((ROOT / "configs/el_bank80_rest_v1.json").read_text("utf-8"))
    by = {}
    for lang in ("python", "java"):
        for ln in (ROOT / man["sources"][lang]["file"]).read_text("utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln); by[d.get("task_id") or d["id"]] = {"language": lang, "public_tests": d["public_tests"]}
    return by


def pools():
    """-> list of pools: {pool, task, candidates: [{id, source, public_pass, selected}] in original order}"""
    out = []
    for run in P1_RUNS:
        for td in sorted((ROOT / run / "tasks").iterdir()):
            if not (td / "DONE").exists() or not (td / "journal.jsonl").exists():
                continue
            cands, pub, sel = [], {}, None
            for line in (td / "journal.jsonl").read_text("utf-8").splitlines():
                r = json.loads(line); k, d = r["kind"], r["data"]
                if k == "candidate": cands.append({"id": d["id"], "source": d["source"]})
                elif k == "evidence" and d["evidence"].get("check_id") == "public_examples_v1": pub.setdefault(d["evidence"]["candidate_id"], []).append(d["evidence"]["status"])
                elif k == "decision": sel = d
            res = json.loads((td / "result.json").read_text("utf-8"))
            answered = res.get("answer_label") in ("SUPPORTED", "VERIFIED")
            for c in cands:
                p = pub.get(c["id"], [])
                c["public_pass"] = bool(p) and all(s == "pass" for s in p)
                c["selected"] = answered and sel is not None and c["id"] == sel.get("candidate_id")
            out.append({"pool": "P1:%s:%s" % (Path(run).name, td.name), "task": td.name, "candidates": cands})
    def p2(d, label, skip_java):
        d = Path(d)
        R = [json.loads(l) for l in (d / "RESULTS.jsonl").read_text("utf-8").splitlines()]
        C = [json.loads(l) for l in (d / "CANDIDATES.jsonl").read_text("utf-8").splitlines()]
        for r in R:
            if skip_java and r["task"].startswith("J"):
                continue
            seen, cands = set(), []
            for c in C:
                if c["task"] == r["task"] and c["arm"] == r["arm"] and c["id"] not in seen:
                    seen.add(c["id"]); cands.append({"id": c["id"], "source": c["source"], "public_pass": bool(c["public_pass"]), "selected": c["id"] == r["selected"]})
            out.append({"pool": "%s:%s:%s:%s" % (label, d.name, r["task"], r["arm"]), "task": r["task"], "candidates": cands})
    p2(ROOT / P2_LOCAL, "P2local", True)
    for d in P2_KAGGLE:
        p2(d, "P2kaggle", False)
    return out


def run_inputs(lang, src, inputs, cfg):
    res = engine.run(lang, src.encode("utf-8"), inputs, cfg)
    items, term = engine._parse(lang, res)
    return res, items, term


def main(out):
    out = Path(out)
    if out.exists():
        sys.exit("refusing: %s exists" % out)
    out.mkdir(parents=True)
    cfg, tasks = K.SandboxConfig(), tasks_by_id()
    t0 = time.time()
    dec_f, pool_f = open(out / "DECISIONS.jsonl", "a"), open(out / "POOLS.jsonl", "a")
    calls = 0
    for pl in pools():
        tid, sp = pl["task"], S.SPECS.get(pl["task"])
        pp = [c for c in pl["candidates"] if c["public_pass"]]
        if not pp:
            continue
        lang, pub = tasks[tid]["language"], tasks[tid]["public_tests"]
        if sp is None:
            raise SystemExit("no spec for task %s" % tid)
        inputs = [("edge", x) for x in sp.edges] + [("derived", x) for x in l2.derive_inputs(pub, n=6)]
        klass = [("valid" if sp.valid(x) is True else ("invalid" if sp.valid(x) is False else "uncertain")) for _, x in inputs]
        outs = {}
        for c in pp:
            ts = time.time()
            res, items, term = run_inputs(lang, c["source"], [x for _, x in inputs], cfg)
            calls += 1
            hard, unc, per = [], [], []
            for k, ((kind, x), cl) in enumerate(zip(inputs, klass)):
                if k >= len(items):
                    per.append(None); unc.append("no result for input %d (%s)" % (k, term)); continue
                ok, val = items[k]
                per.append(json.dumps(val, sort_keys=True) if ok else "<crash>")
                if cl != "valid":
                    continue
                if not ok:
                    hard.append({"input": k, "kind": kind, "check": "crash_on_valid_input", "detail": str(val)[:120]})
                    continue
                for name, good in sp.props(x, val):
                    if good is False:
                        hard.append({"input": k, "kind": kind, "check": "property:" + name})
            if term in ("timeout", "overflow", "invalid_line", "crash") or res.phase_status not in ("ran", "ok"):
                unc.append("run ended abnormally: phase=%s term=%s" % (res.phase_status, term))
            outs[c["id"]] = per
            dec_f.write(json.dumps({"pool": pl["pool"], "task": tid, "candidate": c["id"], "selected": c["selected"], "hard_failures": hard, "uncertain": unc,
                                    "inputs": len(inputs), "valid_inputs": klass.count("valid"), "seconds": round(time.time() - ts, 3)}) + "\n"); dec_f.flush()
        # shadow disagreement on VALID inputs among the pool's public-passing candidates
        vidx = [k for k, cl in enumerate(klass) if cl == "valid"]
        sigs = {cid: tuple(per[k] for k in vidx) for cid, per in outs.items()}
        order = [c["id"] for c in pl["candidates"]]
        orig = next((c["id"] for c in pl["candidates"] if c["selected"]), None)
        hard_ids = set()
        for line in (out / "DECISIONS.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d["pool"] == pl["pool"] and d["hard_failures"]:
                hard_ids.add(d["candidate"])
        # P3 answer: the original answer if it has no hard failure, else the first other public-passing candidate (original order) without one, else none (BLOCKED)
        new = orig if (orig is not None and orig not in hard_ids) else next((cid for cid in order if cid in outs and cid not in hard_ids), None)
        pool_f.write(json.dumps({"pool": pl["pool"], "task": tid, "public_passing": len(pp), "original_answer": orig,
                                 "p3_answer": new if orig is not None or new is not None else None, "hard_failed": sorted(hard_ids),
                                 "shadow_disagreement": len(set(sigs.values())) > 1, "distinct_behaviours_on_valid": len(set(sigs.values())),
                                 "inputs": len(inputs), "valid_inputs": len(vidx), "invalid_inputs": klass.count("invalid"), "uncertain_inputs": klass.count("uncertain")}) + "\n"); pool_f.flush()
    dec_f.close(); pool_f.close()
    man = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "engine": "reference engine (WSL bubblewrap, engine.run)", "sandbox_runs": calls, "seconds": round(time.time() - t0, 1),
           "specs_sha256": sha_file(ROOT / "tools/p3_specs_v1.py"), "tool_sha256": sha_file(__file__), "decisions_sha256": sha_file(out / "DECISIONS.jsonl"), "pools_sha256": sha_file(out / "POOLS.jsonl"),
           "grader_accessed": False}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
