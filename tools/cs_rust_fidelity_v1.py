"""C# / Rust replay repair v1 (user approval, chat 2026-10-13). Diagnosis: C# projects target net10.0 (the sandbox had SDK 8) and need NuGet packages
(no network); Rust had no cargo, and the one Rust workspace has no Cargo.toml (the trace never showed it). Repair: .NET SDK 10.0.302, Rust 1.96.0 and an
offline NuGet fallback folder in /opt/toolchains (see the comment in fable_wsl_runner.py). For Rust only, a minimal Cargo.toml is written into the replay
COPY when absent and the crate name is proven by the recorded output (`Doc-tests <crate>`) and the tests' own `use <crate>::`: package name = crate name,
version 0.1.0, edition 2021 (the original edition is unknown), no dependencies. A C# workspace without a project file is NOT repaired (its test packages
cannot be proven). Same fidelity rule as the original gate, run twice. Canonical TRAIN / VAL replay states only; CAL / TEST never executed.

  wsl -d Ubuntu-24.04 -u root -e python3 "/mnt/d/local model/tools/cs_rust_fidelity_v1.py"
"""
import collections
import gzip
import json
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fable_wsl_runner as W  # noqa: E402
import mutation_pilot_v1 as M  # noqa: E402

STEPS = ROOT / "results/raw/fable-fidelity-steps-20261012T133720Z"   # VAL; the TRAIN replay set has no C# / Rust states
DOC = re.compile(r"Doc-tests (\w+)")
ORIG = W.compare


def main():
    out = ROOT / ("results/raw/cs-rust-fidelity-v1-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True)
    W.RUNS.mkdir(parents=True, exist_ok=True)
    sp = M.canonical_train()
    states = ROOT / json.loads((STEPS / "summary.json").read_text())["states_dir"].replace("\\", "/")
    sink, res = open(out / "rows.jsonl", "a", encoding="utf-8"), []
    for r in (json.loads(l) for l in gzip.open(STEPS / "steps.jsonl.gz", "rt", encoding="utf-8")):
        split = sp.get(r["decision_id"].split(":")[1])
        if r["lang"] not in ("csharp", "rust") or split not in ("train", "val"):
            continue
        ws = states / r["key"] / "workspace"
        test = next((s for s in r["steps"] if s.get("is_test")), {})
        row = dict(decision_id=r["decision_id"], lang=r["lang"], split=split)
        tmp = W.RUNS / ("csrs-" + uuid.uuid4().hex)
        shutil.copytree(ws, tmp)
        if r["lang"] == "rust" and not (tmp / "Cargo.toml").exists():
            doc = set(DOC.findall(test.get("recorded_output") or ""))
            used = set(re.findall(r"\buse (\w+)::", "\n".join(p.read_text("utf-8", "replace") for p in tmp.rglob("tests/*.rs"))))
            if len(doc) == 1 and doc <= used:
                (tmp / "Cargo.toml").write_text('[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n' % doc.pop())
                row["repair"] = "Cargo.toml written"
            else:
                row["repair"] = "none: crate name not provable"
        elif r["lang"] == "csharp" and not list(tmp.glob("*.csproj")) and not list(tmp.glob("*.sln")):
            row["repair"] = "none: no project file in the workspace (not repairable)"
        else:
            row["repair"] = "toolchain only"
        v, v2 = [], []
        for _ in range(2):
            got = []
            W.compare = lambda *a: (got.append(W.compare_v2(*a)["match"]), ORIG(*a))[1]
            v.append(W.replay(r, tmp))
            W.compare = ORIG
            v2.append("MATCH" if got and got[-1] else "MISMATCH")   # the test step is the last comparison
        shutil.rmtree(tmp, ignore_errors=True)
        t = v[0]["log"][-1] if v[0]["log"] else {}
        row.update(verdicts=[x["verdict"] for x in v], verdicts_rule_v2=v2, ours_tail=t.get("ours_tail", "")[-400:], recorded_tail=t.get("recorded_tail", "")[-200:])
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        res.append(row)
        print(row["decision_id"], row["repair"], row["verdicts"], "rule v2:", v2, flush=True)
    sink.close()
    ok = lambda x: x["verdicts"] == ["MATCH", "MATCH"]
    summary = dict(states=len(res), match_both_runs=sum(map(ok, res)), by_lang={l: dict(match=sum(ok(x) for x in res if x["lang"] == l), states=sum(x["lang"] == l for x in res))
                                                                              for l in ("csharp", "rust")}, verdicts=dict(collections.Counter("/".join(x["verdicts"]) for x in res)),
                   match_both_runs_rule_v2=sum(x["verdicts_rule_v2"] == ["MATCH", "MATCH"] for x in res))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary, indent=1))
    for x in res:
        if not ok(x):
            print("NOT MATCHED", x["decision_id"], repr(x["ours_tail"][-300:]))
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
