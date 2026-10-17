import json, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from laya_int1_t6 import TASKS
from evidence_ladder.controller import mxc_sandbox as M

t0 = time.time()
cap = M.capabilities()
print("capabilities available:", cap["available"], cap["reasons"], "tier", cap.get("probe_tier"), round(time.time() - t0, 2), "s")
scratch = Path(tempfile.mkdtemp(prefix="mxc-smoke-"))
T = TASKS["clamp"]
ws = scratch / "src"
ws.mkdir()
for n, c in T["files"].items():
    (ws / n).write_text(c, "utf-8")
sb = M.MxcSandbox(scratch / "sb", T["protected"], persistent=True)
for cmd in [T["test_cmd"], "python3 -c \"print('a b')\""]:
    t = time.time()
    r = sb.run(ws, cmd)
    print(cmd, "->", json.dumps({k: (v if k not in ("pre_tree_sha256", "post_tree_sha256") else v[:10]) for k, v in r.items()}), round(time.time() - t, 3), "s")
print(json.dumps(sb.last_diagnostics))
sb.close()
