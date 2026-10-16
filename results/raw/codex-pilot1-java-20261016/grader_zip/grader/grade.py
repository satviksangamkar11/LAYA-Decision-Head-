#!/usr/bin/env python3
"""UNSAFE DEVELOPMENT ONLY. Java candidate code can access grading host filesystem."""
import argparse, json, pathlib, subprocess, tempfile, shutil, time, sys
HERE=pathlib.Path(__file__).resolve().parent

def same(a,b):
    if isinstance(a,bool) or isinstance(b,bool):return type(a) is type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return a==b
    if type(a) is not type(b):return False
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    return a==b

def grade_task(task_id,source,cases,compile_limit=12,run_limit=10):
    if not source.is_file():return {'status':'MISSING','passed':0,'total':len(cases),'error':'No submission'}
    with tempfile.TemporaryDirectory(prefix='java-pilot-dev-') as dirname:
        w=pathlib.Path(dirname)
        shutil.copy2(source,w/'Solution.java')
        shutil.copy2(HERE/'support'/'Runner.java',w/'Runner.java')
        shutil.copy2(HERE/'support'/'Json.java',w/'Json.java')
        try:
            p=subprocess.run(['javac','-encoding','UTF-8','-d',str(w),str(w/'Json.java'),str(w/'Runner.java'),str(w/'Solution.java')],capture_output=True,text=True,timeout=compile_limit)
        except subprocess.TimeoutExpired:
            return {'status':'COMPILE_TIMEOUT','passed':0,'total':len(cases)}
        if p.returncode:
            return {'status':'COMPILE_ERROR','passed':0,'total':len(cases),'error':p.stderr[-1200:]}
        ins='\n'.join(json.dumps(c['input'],ensure_ascii=False,separators=(',',':')) for c in cases)+'\n'
        try:
            p=subprocess.run(['java','-Xmx128m','-cp',str(w),'Runner'],input=ins,capture_output=True,text=True,timeout=run_limit)
        except subprocess.TimeoutExpired:
            return {'status':'RUNTIME_TIMEOUT','passed':0,'total':len(cases)}
        outputs=p.stdout.splitlines()
        if p.returncode or len(outputs)!=len(cases):
            return {'status':'RUNTIME_ERROR','passed':0,'total':len(cases),'error':p.stderr[-900:],'outputs_provided':len(outputs)}
        passed=0
        mismatches=[]
        for n,(line,case) in enumerate(zip(outputs,cases)):
            try:
                v=json.loads(line)
            except json.JSONDecodeError:
                mismatches.append(n)
                continue
            if same(v,case['expected']):passed+=1
            else:mismatches.append(n)
        return {'status':'PASS' if passed==len(cases) else 'FAIL','passed':passed,'total':len(cases),'mismatch_indices':mismatches[:8]}

def main():
    p=argparse.ArgumentParser(description='UNSAFE local development grading of Java Pilot 1 solutions')
    p.add_argument('--solutions',required=True,help='Directory with J1-01.java .. J1-40.java')
    p.add_argument('--arm',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--unsafe-local',action='store_true',help='Acknowledges this is NOT a code security sandbox')
    p.add_argument('--public-only',action='store_true',help='Run only public examples')
    args=p.parse_args()
    if not args.unsafe_local:p.error('Refusing to execute untrusted Java without --unsafe-local; use isolated grader for actual runs.')
    protected=json.loads((HERE/'protected_cases.json').read_text(encoding='utf-8'))
    public=json.loads((HERE/'public_cases_expected.json').read_text(encoding='utf-8'))
    rows=[]; start=time.time()
    for item,pub in zip(protected,public):
        assert item['id']==pub['id']
        tests=pub['cases'] if args.public_only else pub['cases']+item['cases']
        result=grade_task(item['id'],pathlib.Path(args.solutions)/(item['id']+'.java'),tests)
        rows.append({'task_id':item['id'],**result})
        print(item['id'],result['status'],str(result['passed'])+'/'+str(result['total']),flush=True)
    summary={'arm':args.arm,'tasks':len(rows),'solved':sum(1 for row in rows if row['status']=='PASS'), 'total_cases_passed':sum(x['passed'] for x in rows),'total_cases':sum(x['total'] for x in rows),'seconds':round(time.time()-start,2),'unsafe_local':True,'results':rows}
    out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Solved',summary['solved'],'of',summary['tasks'], 'tasks | Cases',summary['total_cases_passed'],'/',summary['total_cases'])
if __name__=='__main__':main()
