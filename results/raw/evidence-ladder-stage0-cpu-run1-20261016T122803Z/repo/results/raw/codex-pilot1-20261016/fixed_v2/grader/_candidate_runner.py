# Executes untrusted Python. DO NOT USE without a real process sandbox for protected/hostile grading.
# Used by grader only in explicit unsafe-development mode.
import importlib.util,json,sys,traceback
fname=sys.argv[1]
spec=importlib.util.spec_from_file_location('pilot1_candidate',fname)
mod=importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    fn=mod.solve
except BaseException as e:
    print(json.dumps({'fatal':type(e).__name__,'message':str(e)[:200]}));sys.exit(1)
for line in sys.stdin:
    try:
        v=json.loads(line)
        ans=fn(v)
        print(json.dumps({'ok':True,'value':ans},ensure_ascii=False),flush=True)
    except BaseException as e:
        print(json.dumps({'ok':False,'error':type(e).__name__,'message':str(e)[:200]}),flush=True)
