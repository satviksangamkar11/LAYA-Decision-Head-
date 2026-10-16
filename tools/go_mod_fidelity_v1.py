"""Go replay repair v1 (user approval, chat 2026-10-13: "fix Go / C# / Rust"). Diagnosis: every Go replay failed with `pattern ./...: directory prefix .
does not contain main module`: the reconstructed workspace has no go.mod (the trace never showed it). Repair: a go.mod is written into the replay COPY
only when the recorded test output names the module (`ok  <module>  <time>`), with `go 1.26` (the sandbox toolchain; the original directive is unknown).
Nothing else is added: a workspace whose other source files were never shown stays unreplayable. The SAME fidelity rule then decides (MATCH = exit code
equal to the trace AND normalised output or pass/fail signature equal). Canonical TRAIN and VAL states only (the imitation set's split map); CAL / TEST
are never executed. Run twice, like the original gate.

  wsl -d Ubuntu-24.04 -u root -e python3 "/mnt/d/local model/tools/go_mod_fidelity_v1.py"
"""
import collections
import gzip
import json
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fable_wsl_runner as W  # noqa: E402
import mutation_pilot_v1 as M  # noqa: E402

SETS = [ROOT / "results/raw/fable-fidelity-steps-20261009T221513Z", ROOT / "results/raw/fable-fidelity-steps-20261012T133720Z"]
ONLY = set(sys.argv[1:])   # optional: recheck only these decision ids
MOD = re.compile(r"^ok\s+(\S+)\s+[\d.]+s", re.M)


def main():
    out = ROOT / ("results/raw/go-mod-fidelity-v1-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True)
    W.RUNS.mkdir(parents=True, exist_ok=True)
    sp, jobs = M.canonical_train(), []
    for sd in SETS:
        states = ROOT / json.loads((sd / "summary.json").read_text())["states_dir"].replace("\\", "/")
        for r in (json.loads(l) for l in gzip.open(sd / "steps.jsonl.gz", "rt", encoding="utf-8")):
            split = sp.get(r["decision_id"].split(":")[1])
            if r["lang"] == "go" and split in ("train", "val") and (not ONLY or r["decision_id"] in ONLY):
                jobs.append((sd, states, r, split))
    print("Go states, canonical TRAIN + VAL:", len(jobs), flush=True)
    sink, lock = open(out / "rows.jsonl", "a", encoding="utf-8"), threading.Lock()

    def one(job):
        sd, states, r, split = job
        test = next((s for s in r["steps"] if s.get("is_test")), None)
        ws = states / r["key"] / "workspace"
        mods = sorted(set(MOD.findall((test or {}).get("recorded_output") or "")))
        row = dict(decision_id=r["decision_id"], split=split, steps_dir=sd.name, had_go_mod=(ws / "go.mod").exists(), module=mods)
        if row["had_go_mod"] or len(mods) != 1 or "/" in mods[0]:
            row.update(status="not_repaired", why="go.mod already present" if row["had_go_mod"] else "module name not provable from the recorded output")
        else:
            tmp = W.RUNS / ("gomod-" + uuid.uuid4().hex)
            shutil.copytree(ws, tmp)
            (tmp / "go.mod").write_text("module %s\n\ngo 1.26\n" % mods[0])
            v = [W.replay(r, tmp) for _ in range(2)]
            shutil.rmtree(tmp, ignore_errors=True)
            t = [x["log"][-1] if x["log"] else {} for x in v]
            row.update(status="repaired_replay", verdicts=[x["verdict"] for x in v], ours_tail=t[0].get("ours_tail", "")[-300:],
                       recorded_tail=t[0].get("recorded_tail", "")[-200:])
        with lock:
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(row["decision_id"], row["split"], row["status"], row.get("verdicts"), row.get("why", ""), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:   # ponytail: 4 parallel sandboxes (each capped at 4 CPUs); no shared Go build cache, by isolation
        res = list(pool.map(one, jobs))
    sink.close()
    rep = [x for x in res if x["status"] == "repaired_replay"]
    summary = dict(go_states_train_val=len(res), repaired=len(rep), match_both_runs=sum(x["verdicts"] == ["MATCH", "MATCH"] for x in rep),
                   by_split={s: sum(x["verdicts"] == ["MATCH", "MATCH"] for x in rep if x["split"] == s) for s in ("train", "val")},
                   verdicts=dict(collections.Counter("/".join(x["verdicts"]) for x in rep)), not_repaired=dict(collections.Counter(x["why"] for x in res if x["status"] != "repaired_replay")))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary, indent=1))
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
