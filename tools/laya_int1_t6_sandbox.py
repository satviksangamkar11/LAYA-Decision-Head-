"""[LAYA-INT-1] WSL-side helper for T6 (run by tools/laya_int1_t6.py as: wsl -d Ubuntu-24.04 -u root -- python3 "/mnt/d/local model/tools/laya_int1_t6_sandbox.py" '<json>').
Runs ONE shell command in the qualified isolation of tools/fable_wsl_runner.py (systemd scope + bubblewrap, no network, protected test files bind-mounted read-only, hard timeout)
on a COPY of the workspace. persist=true copies the copy's files back into the workspace afterwards, but only when the protected files are unchanged and the run was not killed.
Input: a JSON file path (preferred; wsl.exe mangles quotes) or a JSON string. Fields: {"workspace": "/mnt/d/...", "cmd": "...", "protected": ["test_x.py"], "timeout": 60, "persist": false}. Output: one JSON object on stdout."""
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, "/mnt/d/local model/tools")
import fable_wsl_runner as W  # noqa: E402


def tree(ws):
    h = hashlib.sha256()
    for p in sorted(x for x in Path(ws).rglob("*") if x.is_file()):
        h.update(str(p.relative_to(ws)).replace("\\", "/").encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def main(a):
    W.RUNS.mkdir(parents=True, exist_ok=True)
    run = W.RUNS / uuid.uuid4().hex
    ws = run / "ws"
    shutil.copytree(a["workspace"], ws)
    prot = [p for p in a.get("protected", []) if (ws / p).is_file()]
    before = {p: hashlib.sha256((ws / p).read_bytes()).hexdigest() for p in prot}
    pre = tree(ws)
    subprocess.run(["chown", "-R", "replay:replay", str(run)], check=True)
    p = subprocess.run(W.sandbox_argv(ws, a["cmd"], prot, int(a.get("timeout", 60))), capture_output=True)
    after = {q: hashlib.sha256((ws / q).read_bytes()).hexdigest() for q in prot}
    killed = p.returncode in (137, -9)
    res = dict(exit_code=p.returncode, killed=killed, stdout=p.stdout.decode("utf-8", "replace")[-6000:], stderr=p.stderr.decode("utf-8", "replace")[-6000:],
               protected_unchanged=before == after, pre_tree_sha256=pre, post_tree_sha256=tree(ws), persisted=False)
    if a.get("persist") and res["protected_unchanged"] and not killed:
        src = Path(a["workspace"])
        for q in [x for x in src.rglob("*") if x.is_file() and not (ws / x.relative_to(src)).exists()]:
            q.unlink()
        for q in [x for x in ws.rglob("*") if x.is_file() and "__pycache__" not in x.parts]:
            d = src / q.relative_to(ws)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q, d)
        res["persisted"] = True
    shutil.rmtree(run, ignore_errors=True)
    print(json.dumps(res))


if __name__ == "__main__":
    arg = sys.argv[1]
    main(json.loads(Path(arg).read_text("utf-8")) if arg.endswith(".json") else json.loads(arg))
