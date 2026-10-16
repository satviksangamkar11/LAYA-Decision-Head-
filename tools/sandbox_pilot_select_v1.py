"""Sandbox label pilot v1: continue the seeded v2 walk (configs/thresholds_sandbox_pilot_v1.toml [scope].walk).
Reuses tools/wsl_feasibility_select_v2.py verbatim: its universe/exclusion/check code (everything before the walk loop) is executed as is,
then the walk resumes after the ids v2 already visited. Ids accepted by v2 that are not used by the pilot (dwave reserve) come first.

  uv run python tools/sandbox_pilot_select_v1.py <pilot_out_dir> <max_new_ids> [<accepted_needed>]
Writes <out>/walk/fetched.jsonl, task-*.json, prefix-*.json, walk.json (accepted in walk order)."""
import json, sys, time
from pathlib import Path

V2 = Path("results/raw/wsl-feasibility-v2-20261014T151858Z")
V1 = Path("results/raw/wsl-feasibility-v1-20261014T120445Z")
out = Path(sys.argv[1]) / "walk"
max_new, need = int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 99
out.mkdir(parents=True, exist_ok=True)

src = Path("tools/wsl_feasibility_select_v2.py").read_text(encoding="utf-8")
head = src[:src.index("v1_reject = ")]  # definitions + universe + seeded shuffle, unchanged
sys.argv = [sys.argv[0], str(out), str(V1)]
g = {"__file__": str(Path("tools/wsl_feasibility_select_v2.py").resolve()), "__name__": "select_v2_head"}
exec(compile(head, "wsl_feasibility_select_v2.py(head)", "exec"), g)
elig, repo_of = g["elig"], g["repo_of"]
_fetch = g["fetch"]


def patient_fetch(i):  # datasets-server answered HTTP 429 / 504 for minutes on 2026-10-14: up to 8 outer tries, 5 min apart (row content unchanged)
    for k in range(8):
        try:
            return _fetch(i)
        except OSError as e:
            print("fetch %s failed (%s); waiting 300 s" % (i, e), flush=True)
            time.sleep(300)
    raise RuntimeError("datasets-server unavailable for " + i)
g["fetch"] = patient_fetch

visited = [json.loads(l) for l in open(V2 / "fetched_v2.jsonl", encoding="utf-8")]
seen = {r["instance_id"] for r in visited}
v2_acc = [r["instance_id"] for r in visited if r["reject"] is None]
taken = {repo_of[i] for i in v2_acc}                     # one task per repo (v2 accepted repos stay taken)
last = max(elig.index(i) for i in seen)
assert all(elig.index(i) <= last for i in seen) and elig[last] == "dwavesystems__dwave-system-373", "v2 walk order not reproduced"
print("v2 walk reproduced: %d visited, last at position %d = %s" % (len(seen), last, elig[last]))

prev = [json.loads(l) for l in open(out / "fetched.jsonl", encoding="utf-8")] if (out / "fetched.jsonl").exists() else []  # resume after an HTTP 429 abort
acc, n_new = [r["instance_id"] for r in prev if r["reject"] is None], len(prev)
for r in prev:
    if r["reject"] is None and r["source"] != "v2_accepted_reserve":
        taken.add(r["repo"])
if prev:
    print("resuming after %d logged ids" % n_new)
with open(out / "fetched.jsonl", "a", encoding="utf-8") as log:
    # 1. v2 reserve dwave (counts as id 1 of the cap)
    i = "dwavesystems__dwave-system-373"
    if not prev:
      n_new += 1
      log.write(json.dumps(dict(instance_id=i, repo=repo_of[i], source="v2_accepted_reserve", reject=None, n=n_new)) + "\n")
    if not prev:
      (out / ("task-%s.json" % i)).write_text((V2 / ("task-%s.json" % i)).read_text(encoding="utf-8"), encoding="utf-8")
      (out / ("prefix-%s.json" % i)).write_text((V2 / ("prefix-%s.json" % i)).read_text(encoding="utf-8"), encoding="utf-8")
      acc.append(i)
    # 2. next walk ids
    for i in elig[last + 1:]:
        if n_new >= max_new or len(acc) >= need:
            break
        if repo_of[i] in taken or i in {r["instance_id"] for r in prev}:
            continue
        n_new += 1
        rec = dict(instance_id=i, repo=repo_of[i], n=n_new)
        split, row = g["fetch"](i)
        rec["source"] = "fetched_pilot:%s" % split
        why = "not_found" if row is None else g["check_v1"](row, taken)
        stage = "v1" if why else None
        if why is None:
            why = g["check_x1"](row); stage = "x1" if why else None
        if why is None:
            tid = g["run_of"](i)
            steps, dec, ws = g["prefix_of"](tid, i)
            net = [s["arguments"].get("command") for s in steps if s["name"] == "execute_bash" and g["NET"].search(s["arguments"].get("command") or "")]
            why = "no_decision_point" if dec is None else ("x2_network_prefix:%s" % [n[:100] for n in net]) if net else None
            stage = "x2" if why else None
            rec.update(trajectory_id=tid, prefix_steps=len(steps), decision=dec and (dec["kind"], dec["index"]))
        rec.update(reject=why, stage=stage, install_config=(row or {}).get("install_config"), pinned=len(g["reqs_of"](row)) if row else None)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(n_new, i, why and why[:150])
        time.sleep(0.5)
        if why is None:
            acc.append(i); taken.add(row["repo"])
            (out / ("task-%s.json" % i)).write_text(json.dumps(row), encoding="utf-8")
            (out / ("prefix-%s.json" % i)).write_text(json.dumps(dict(instance_id=i, trajectory_id=tid, workspace_dir=ws, prefix=steps, decision=dec, task=row)), encoding="utf-8")
(out / "walk.json").write_text(json.dumps(dict(accepted=acc, ids_counted=n_new, cap=max_new), indent=1))
print(json.dumps(dict(accepted=acc, ids_counted=n_new)))
