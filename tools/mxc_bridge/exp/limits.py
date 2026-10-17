import json, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evidence_ladder.controller import mxc_sandbox as M

scratch = Path(tempfile.mkdtemp(prefix="mxc-lim-"))
ws = scratch / "src"
ws.mkdir()
(ws / "mem.py").write_text("x=[]\ntry:\n    for i in range(100):\n        x.append(bytearray(50*1024*1024)); print('alloc', (i+1)*50, 'MB', flush=True)\nexcept MemoryError:\n    print('MemoryError')\n")
(ws / "fork.py").write_text("import subprocess,sys\nps=[]\ntry:\n    for i in range(60):\n        ps.append(subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)']))\n    print('spawned', len(ps))\nexcept Exception as e:\n    print('spawn failed after', len(ps), type(e).__name__, str(e)[:60])\nfor p in ps: p.kill()\n")
(ws / "cpu.py").write_text("import time\nt=time.time();n=0\nwhile time.time()-t<3: n+=1\nprint('iters',n)\n")
for label, lim in (("no limits", None), ("mem300 proc20", M.Limits(memory_mb=300, max_processes=20, cpu_percent=None))):
    sb = M.MxcSandbox(scratch / ("sb-" + label.split()[0]), [], persistent=False, limits=lim)
    print("=====", label, "limits_enforced", sb.limits_enforced)
    for f in ("mem.py", "fork.py"):
        t = time.time()
        try:
            r = sb.run(ws, "python3 " + f, timeout=40)
            print(f, "exit", r["exit_code"], "killed", r["killed"], round(time.time() - t, 1), "s", json.dumps(r["stdout"][-200:]), json.dumps(r["stderr"][-200:]))
        except Exception as e:
            print(f, "RAISED", type(e).__name__, str(e)[:200], round(time.time() - t, 1), "s")
    sb.close()
