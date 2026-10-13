"""Scoring of the mutation pilot v1.1 pools (user, chat 2026-10-13: "complete scoring first ... finish it first"). Step 2 of the approved plan ended with
"score the frozen scorers on the execution-verified pools"; the labels exist (results/raw/mutation-pilot-v1.1-20261012T220733Z), the scoring was never
done. File work only here: builds the Kaggle upload (contexts + options, NO labels) and the local private key. Capture and scoring run on Kaggle.

  PYTHONPATH=. python tools/mutation_score_pack_v1.py <out dir>          then upload + kernels (see MANIFEST.json)
  PYTHONPATH=. python tools/mutation_score_pack_v1.py join <out dir> <scores dir>    (lookup only: picks x outcomes)
"""
import collections
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from decision_head import fable_imitation as FI  # noqa: E402
from decision_head import fable_imitation_v1_2 as V  # noqa: E402
from decision_head import fable_pool as FP  # noqa: E402

R = ROOT / "results/raw"
MUT = R / "mutation-pilot-v1.1-20261012T220733Z"
IMIT = R / "fable-imitation-pools-v1.2.1-20261010T093223Z"
CODE = R / "level12-bundle-20261012T104251Z/upload"
CONFIG = R / "val-outcome-v1-20261012T134540Z/pilot-20261012T224831Z/capture_bundle/upload/level12_config.json"
MAX_WORSE = 4
SLUG = "sangeetasangamkar/laya-mutation-score-v1"
PLAN = {
    "written": "2026-10-13, before any feature of these pools was captured or scored",
    "approved_by": "user, chat 2026-10-13: 'complete scoring first then move ahead as no shortcuts and scoring will tell us ... finish it first'; the "
                   "mutation diagnostic itself was approved in the user's pasted plan (frozen scorers, no-context and option-only baselines, per-state "
                   "reporting, TRAIN diagnostic only); all computation on Kaggle (standing rule)",
    "states": "the 19 valid states of mutation pilot v1.1 (canonical TRAIN; ts-msgfmt:3 invalid and excluded)",
    "pool": "Fable's recorded action (verified PASS: the test fails before it and passes after it, stable) + up to %d mutants labelled WORSE in v1.1 "
            "with no crash flag, chosen by ascending sha256(mutant id); EQUAL_ON_TESTS and UNKNOWN mutants are excluded" % MAX_WORSE,
    "presentation": "context = FP.select_context(trace prefix, V.MAX_CONTEXT_CHARS), options = V.render(calls, None): the imitation builder's own functions; "
                    "a mutant differs from Fable's action only at its mutation site. Checked: Fable's rendered option equals the imitation set's option "
                    "text for every state that is in the imitation set. Option order: ascending sha256(decision_id|option text). No label is uploaded",
    "scorers (frozen)": "laya_v4 (rebuilt coding head x3), random_head_v4 (rebuilt x9), laya_baseline_v1 (x3), linear_probe (refit on the 303 TRAIN "
                        "pools), each with the real context (B) and with a neutral prefix (A, no context); option-only baselines computed locally from "
                        "the option texts: shortest, longest, uniform random",
    "measure": "per state: 1 if the scorer's top choice is Fable's (passing) action; mean over a family's runs; reported over all states and over the "
               "states NOT in any scorer's training pools; context effect = B minus A per state; paired comparisons by state (19 states, not mutants)",
    "limits": "TRAIN states: Fable's action was a positive training example for the states in the imitation pools (memorisation possible); mutants are "
              "synthetic single-site changes (not natural agent errors); 19 states only",
}


def option_of(calls):
    return V.render([(n, a) for n, a in calls], None)


def build(out_arg):
    out = ROOT / out_arg
    (out / "upload").mkdir(parents=True)
    states = [s for s in map(json.loads, open(MUT / "states.jsonl", encoding="utf-8")) if s["valid"]]
    want = {s["decision_id"] for s in states}
    muts = collections.defaultdict(list)
    for m in map(json.loads, open(MUT / "mutants.jsonl", encoding="utf-8")):
        if m["label"] == "WORSE" and not m["crash"] and m["decision_id"] in want:
            muts[m["decision_id"]].append(m)
    imit = {r["decision_id"]: r for r in FP.read_gz(IMIT / "requests.jsonl.gz")}
    lab = {r["decision_id"]: r for r in FP.read_gz(IMIT / "private_labels.jsonl.gz")}
    meta = {(r["trajectory_id"], r["step"] + 1): r for r in FP.read_gz(FI.METADATA) if r["source"] == "fable5"}
    rows, key, checks = [], [], []
    with open(FI.TRACES, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = meta.get((row["source_trajectory_id"], row["assistant_step"]))
            if m is None or m["decision_id"] not in want:
                continue
            did = m["decision_id"]
            msgs, ti = row["messages"], row["target_message_index"]
            ctx, info = FP.select_context(msgs[:ti], V.MAX_CONTEXT_CHARS)
            assert ctx is not None and not info.get("n_omitted"), did
            calls = [FP.parse_call(tc) for tc in msgs[ti]["tool_calls"]]
            fable = option_of(calls)
            if did in imit:
                ref = imit[did]["options"][int(lab[did]["reference_index"])]
                checks.append(dict(decision_id=did, fable_option_equals_imitation=fable == ref, context_equals_imitation=ctx == imit[did]["context_text"]))
            chosen = sorted(muts[did], key=lambda x: hashlib.sha256(x["id"].encode()).hexdigest())[:MAX_WORSE]
            opts = [("FABLE", fable)]
            for mm in chosen:
                st = mm["steps"][mm["step"]]
                k = "content" if st["effect"] == "write" else "new_string"
                c2 = [(n, dict(a, **{k: st["arguments"][k]}) if i == mm["step"] else a) for i, (n, a) in enumerate(calls)]
                t = option_of(c2)
                assert t != fable, (did, mm["id"])
                opts.append((mm["id"], t))
            opts.sort(key=lambda o: hashlib.sha256(("%s|%s" % (did, o[1])).encode()).hexdigest())
            texts = [t for _, t in opts]
            assert len(set(texts)) == len(texts), did
            rows.append(dict(decision_id=did, context_text=ctx, options=texts, same_trajectory_context="", other_trajectory_from=did, **FP.request_hashes(ctx, texts)))
            key.append(dict(decision_id=did, order=[o for o, _ in opts], fable_index=[o for o, _ in opts].index("FABLE"), in_imitation_pools=did in imit,
                            n_options=len(texts)))
    assert {r["decision_id"] for r in rows} == want, want - {r["decision_id"] for r in rows}
    with gzip.open(out / "upload/requests_level1.jsonl.gz", "xt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    for n in ("capture.py", "capture_resident.py", "level12_worker.py"):
        shutil.copy(CODE / n, out / "upload" / n)
    shutil.copy(CONFIG, out / "upload/level12_config.json")
    (out / "upload/dataset-metadata.json").write_text(json.dumps(dict(title="laya-mutation-score-v1", id=SLUG, licenses=[dict(name="other")])), "utf-8")
    (out / "PRIVATE_KEY.json").write_text(json.dumps(key, indent=1), "utf-8")   # labels: local only, never uploaded
    (out / "PLAN.json").write_text(json.dumps(PLAN, indent=1), "utf-8")
    man = dict(states=len(rows), options=[r["n_options"] for r in key], fable_position=collections.Counter(k["fable_index"] for k in key),
               in_imitation_pools=sum(k["in_imitation_pools"] for k in key), rendering_checks=checks,
               all_checks_pass=all(c["fable_option_equals_imitation"] and c["context_equals_imitation"] for c in checks), labels_uploaded=False)
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str), "utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "rendering_checks"}, indent=1, default=str))


def join(out_arg, scores_arg):
    import numpy as np
    out, S = ROOT / out_arg, json.loads((ROOT / scores_arg / "SCORES.json").read_text("utf-8"))
    key = {k["decision_id"]: k for k in json.loads((out / "PRIVATE_KEY.json").read_text("utf-8"))}
    reqs = {r["decision_id"]: r for r in FP.read_gz(out / "upload/requests_level1.jsonl.gz")}
    fams = sorted({k.split("/")[0] for k in S["runs"]})
    per = []
    for i, p in enumerate(S["pools"]):
        did, k = p["decision_id"], key[p["decision_id"]]
        n, f = k["n_options"], k["fable_index"]
        lens = [len(t) for t in reqs[did]["options"]]
        r = dict(decision_id=did, n_options=n, in_imitation_pools=k["in_imitation_pools"], random=1 / n,
                 shortest=float(int(np.argmin(lens)) == f), longest=float(int(np.argmax(lens)) == f))
        for fam in fams:
            for v, tag in (("B", ""), ("A", "_no_context")):
                ks = [x for x in S["runs"] if x.startswith(fam + "/")]
                r[fam + tag] = float(np.mean([int(np.argmax(S["runs"][x][v][i])) == f for x in ks]))
        per.append(r)
    names = ["random", "shortest", "longest"] + [fam + t for fam in fams for t in ("", "_no_context")]
    mean = lambda rs, x: round(float(np.mean([r[x] for r in rs])), 3) if rs else None
    unseen = [r for r in per if not r["in_imitation_pools"]]
    summary = dict(states=len(per), states_not_in_any_training_pool=len(unseen), captured_pools=len(S["pools"]),
                   all_states={x: mean(per, x) for x in names}, unseen_states={x: mean(unseen, x) for x in names},
                   context_effect_all={fam: round(mean(per, fam) - mean(per, fam + "_no_context"), 3) for fam in fams},
                   integrity_rebuilt=S["all_rebuilt_reproduce"])
    (out / "RESULT.json").write_text(json.dumps(dict(summary=summary, per_state=per), indent=1), "utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    join(*sys.argv[2:]) if sys.argv[1] == "join" else build(sys.argv[1])
