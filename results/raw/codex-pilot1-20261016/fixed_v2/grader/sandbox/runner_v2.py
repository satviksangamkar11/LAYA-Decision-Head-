# Runs INSIDE the bubblewrap jail. argv[1] = candidate path. stdin = one JSON input per line. Results go to a private dup of fd 1;
# anything the candidate prints (stdout or fd 1) is redirected to stderr, which the parent only logs.
import importlib.util, json, os, resource, sys

real_out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr
for lim, val in ((resource.RLIMIT_AS, 1 << 30), (resource.RLIMIT_FSIZE, 1 << 20), (resource.RLIMIT_NOFILE, 64), (resource.RLIMIT_NPROC, 32), (resource.RLIMIT_CORE, 0)):
    try:
        resource.setrlimit(lim, (val, val))
    except (ValueError, OSError):
        pass
try:
    spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.solve
except BaseException as e:
    real_out.write(json.dumps({"fatal": type(e).__name__, "message": str(e)[:200]}) + "\n")
    sys.exit(1)
for line in sys.stdin:
    try:
        ans = fn(json.loads(line))
        real_out.write(json.dumps({"ok": True, "value": ans}, ensure_ascii=False) + "\n")
    except BaseException as e:
        real_out.write(json.dumps({"ok": False, "error": type(e).__name__}) + "\n")
