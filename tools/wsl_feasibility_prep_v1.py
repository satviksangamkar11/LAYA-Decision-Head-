"""WSL feasibility v1: extract the pre-registered prefix of the selected run (Windows side, pyarrow streaming). Usage: prep_v1.py <out_dir> <work_dir>"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from wsl_feasibility_traj_v1 import load

TEST = re.compile(r"pytest|py\.test|unittest|\btox\b|\bnox\b|runtests|run_tests|manage\.py test|make test|python[0-9.]* \S*test\S*\.py")
EDIT = {"create", "str_replace", "insert", "undo_edit"}
out, work = Path(sys.argv[1]), Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
sel = json.loads((out / "selection.json").read_text())
for iid in sel["accepted"] + sel["reserves"]:
    row = load(sel["run"][iid])
    assert row["instance_id"] == iid and row["resolved"] == 1
    tr = row["trajectory"]
    resp = {m["tool_call_id"]: m["content"] for m in tr if m["role"] == "tool"}
    steps, decision = [], None
    for m in tr:
        for c in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
            name, args = c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")
            is_edit = name == "str_replace_editor" and args.get("command") in EDIT
            is_test = name == "execute_bash" and bool(TEST.search(args.get("command") or ""))
            if is_edit or is_test:
                decision = dict(index=len(steps), name=name, arguments=args, kind="edit" if is_edit else "test")
                break
            steps.append(dict(index=len(steps), name=name, arguments=args, recorded=resp.get(c["id"])))
        if decision:
            break
    ws_dir = re.search(r"<uploaded_files>\s*(\S+)\s*</uploaded_files>", next(m["content"] for m in tr if m["role"] == "user")).group(1)
    task = json.loads((out / ("task-%s.json" % iid)).read_text(encoding="utf-8"))
    (work / (iid + ".json")).write_text(json.dumps(dict(instance_id=iid, trajectory_id=sel["run"][iid], workspace_dir=ws_dir, prefix=steps,
                                                        decision=decision, task=task)), encoding="utf-8")
    print(iid, ws_dir, "prefix", len(steps), "decision", decision and (decision["kind"], decision["index"]),
          [s["name"] + ":" + (s["arguments"].get("command") or "")[:60] for s in steps])
