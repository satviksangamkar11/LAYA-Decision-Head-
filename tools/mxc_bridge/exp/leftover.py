import json, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evidence_ladder.controller import mxc_sandbox as M

scratch = Path(tempfile.mkdtemp(prefix="mxc-left-"))
ws = scratch / "src"
ws.mkdir()
(ws / "bg.py").write_text(
    "import subprocess, sys\n"
    "c = subprocess.Popen([sys.executable, '-c', 'import time,os\\ntime.sleep(3)\\nopen(\"late_write.txt\",\"w\").write(\"x\")\\ntime.sleep(60)'], creationflags=0x00000008)\n"
    "print('CHILD', c.pid, flush=True)\n")
with M.MxcSandbox(scratch / "sb", [], persistent=True, limits=M.Limits()) as sb:
    r = sb.run(ws, "python3 bg.py", persist=True)
    print(json.dumps({k: r[k] for k in ("exit_code", "killed", "persisted")}), r["stdout"].strip())
    pid = [l.split()[1] for l in r["stdout"].splitlines() if l.startswith("CHILD")][0]
    time.sleep(1.0)
    out = subprocess.run(["tasklist", "/FI", "PID eq " + pid, "/NH"], capture_output=True, text=True).stdout
    print("child alive after normal exit:", pid in out)
    time.sleep(4)
    print("late_write in real workspace:", (ws / "late_write.txt").exists())
    out = subprocess.run(["tasklist", "/FI", "PID eq " + pid, "/NH"], capture_output=True, text=True).stdout
    print("child alive 5s later:", pid in out)
    r2 = sb.run(ws, "ls")
    print("next run sees:", r2["stdout"].split())
