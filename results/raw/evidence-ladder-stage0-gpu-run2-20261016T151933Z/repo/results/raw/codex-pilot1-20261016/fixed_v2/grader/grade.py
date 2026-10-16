#!/usr/bin/env python3
"""Development-only grader. Running model code is UNSAFE without OS isolation."""
import argparse,json,subprocess,sys,os,time,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Pilot 1 local development grader. Not a secure sandbox.')
    p.add_argument('--solutions',type=Path,required=True,help='Folder with P1-01.py ... P1-40.py')
    p.add_argument('--out',type=Path,default=Path('result.json'))
    p.add_argument('--arm',default='unknown')
    p.add_argument('--unsafe-local',action='store_true',help='EXPLICITLY authorize local execution of untrusted submissions (unsafe).')
    p.add_argument('--timeout',type=float,default=5.0)
    a=p.parse_args()
    if not a.unsafe_local:
        p.error('Fail closed: no OS isolation provided. For trusted personal DEV submissions ONLY, add --unsafe-local. For real protected grading, integrate a genuine container/VM executor.')
    folder=Path(__file__).parent
    cases=json.loads((folder/'protected_cases.json').read_text())
    results=[];start=time.monotonic()
    for task in cases:
        name=task['task_id'];file=a.solutions/(name+'.py');data=task['cases'];fail=None;answers=[]
        if not file.exists():
            fail='MISSING_SUBMISSION'
        elif file.stat().st_size>131072:
            fail='OVERSIZED_SUBMISSION'
        else:
            env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONHASHSEED':'0'}
            # -I and -S reduce accidental coupling but DO NOT isolate filesystem, network, secrets or privileges.
            cmd=[sys.executable,'-I','-S',str(folder/'_candidate_runner.py'),str(file.resolve())]
            try:
                raw=''.join(json.dumps(c['input'],ensure_ascii=False)+'\n' for c in data)
                res=subprocess.run(cmd,input=raw,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=a.timeout,cwd='/tmp',env=env)
                if res.returncode!=0:fail='PROCESS_ERROR'
                else:
                    for line in res.stdout.splitlines():
                        try:answers.append(json.loads(line))
                        except (ValueError,TypeError):fail='INVALID_OUTPUT';break
                    if len(answers)!=len(data):fail='MISSING_OUTPUT'
            except subprocess.TimeoutExpired:fail='TIMEOUT'
            except OSError:fail='PROCESS_ERROR'
        ok=0
        if fail is None:
            for expected,received in zip(data,answers):
                if received.get('ok') is True and type(received.get('value')) is type(expected['expected']) and received['value']==expected['expected']:
                    ok+=1
                # JSON numbers bool/int require type sensitivity; list/dict/string handled above.
            if ok!=len(data):fail='WRONG_ANSWER_OR_RUNTIME_ERROR'
        results.append({'task_id':name,'passed':fail is None,'passed_tests':ok,'total_tests':len(data),'failure':fail,'sha256':hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None})
    out={'arm':a.arm,'status':'DEVELOPMENT_UNSAFE_EXECUTION','tasks':len(results),'solved':sum(x['passed'] for x in results),'solve_rate':sum(x['passed'] for x in results)/len(results),'duration_seconds':round(time.monotonic()-start,2),'results':results,'security_note':'This runner is not isolated; never expose trusted hidden test files to candidate execution in a real holdout.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2))
    print(json.dumps({'arm':a.arm,'solved':out['solved'],'tasks':out['tasks'],'out':str(a.out)}))
if __name__=='__main__':main()
