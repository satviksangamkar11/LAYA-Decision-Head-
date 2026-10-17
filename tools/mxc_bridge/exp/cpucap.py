import sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evidence_ladder.controller import mxc_sandbox as M

scratch = Path(tempfile.mkdtemp(prefix="mxc-cpu-"))
ws = scratch / "src"
ws.mkdir()
(ws / "work.py").write_text("import time\nt=time.process_time()\nn=0\nwhile n<20_000_000: n+=1\nprint(round(time.process_time()-t,2))\n")
(ws / "par.py").write_text("import subprocess, sys, time\nt=time.time()\nps=[subprocess.Popen([sys.executable,'work.py'],stdout=subprocess.PIPE,text=True) for _ in range(4)]\ncpu=[float(p.communicate()[0]) for p in ps]\nprint('WALL', round(time.time()-t,2), 'CPU_SUM', round(sum(cpu),2))\n")
for label, lim in (("no cpu cap", M.Limits(memory_mb=2048, max_processes=64, cpu_percent=None)), ("cpu cap 100% (1 core)", M.Limits(memory_mb=2048, max_processes=64, cpu_percent=100))):
    with M.MxcSandbox(scratch / "sb", [], limits=lim, persistent=True) as sb:
        r = sb.run(ws, "python3 par.py", timeout=120)
        print(label, "->", r["stdout"].strip(), "exit", r["exit_code"])
