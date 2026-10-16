"""Sandbox pilot v2 prep (Windows, uv): late decision point (configs/thresholds_sandbox_pilot_v2.toml [decision_point]).
  uv run python tools/sandbox_pilot_prep_v2.py <out_dir> <iid> [<task_json>]
Writes <out>/prefix-<iid>.json (prefix, decision, remaining, task) and <out>/heads/<iid>.json (recorded system prompt + first user message).
The run is the seeded resolved run (same rule as v1: random.Random(20261014).choice(sorted resolved ids))."""
import glob, json, random, sys
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from wsl_feasibility_traj_v1 import load, P  # noqa: E402

out, iid = Path(sys.argv[1]), sys.argv[2]
if len(sys.argv) > 3:
    task_row = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
else:
    task_row = json.loads(Path(sorted(glob.glob("results/raw/wsl-feasibility-v2-*/prefix-%s.json" % iid))[0]).read_text(encoding="utf-8"))["task"]
rid = pq.read_table(P, columns=["trajectory_id", "resolved"], filters=[("instance_id", "=", iid), ("resolved", "=", 1)])["trajectory_id"].to_pylist()
tid = random.Random(20261014).choice(sorted(rid))
row = load(tid)
assert row["instance_id"] == iid and row["resolved"] == 1
tr = row["trajectory"]
resp = {m["tool_call_id"]: m["content"] for m in tr if m["role"] == "tool"}
acts = []
for m in tr:
    for c in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
        acts.append(dict(index=len(acts), name=c["function"]["name"], arguments=json.loads(c["function"]["arguments"] or "{}"), recorded=resp.get(c["id"])))
n = len(acts)
assert n >= 20, n
d = n - 10
import re  # noqa: E402
user0 = next(m["content"] for m in tr if m["role"] == "user")
ws = re.search(r"<uploaded_files>\s*(\S+)\s*</uploaded_files>", user0).group(1)
dec = dict(acts[d])
dec.pop("recorded")
rec = dict(instance_id=iid, trajectory_id=tid, workspace_dir=ws, n_actions=n, decision_position=d, prefix=acts[:d], decision=dec,
           remaining=acts[d:], task=task_row, rule="thresholds_sandbox_pilot_v2.toml [decision_point]")
out.mkdir(parents=True, exist_ok=True)
(out / "heads").mkdir(exist_ok=True)
(out / ("prefix-%s.json" % iid)).write_text(json.dumps(rec), encoding="utf-8")
sysmsg = next(m["content"] for m in tr if m["role"] == "system")
(out / "heads" / ("%s.json" % iid)).write_text(json.dumps(dict(instance_id=iid, trajectory_id=tid, system=sysmsg, user=user0)), encoding="utf-8")
print(iid, tid, "n_actions", n, "decision", d, dec["name"], (dec["arguments"].get("command") or "")[:60], dec["arguments"].get("path"))
