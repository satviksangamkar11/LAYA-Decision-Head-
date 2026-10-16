#!/usr/bin/env python3
import argparse,random,json,statistics,math
from pathlib import Path

def paired_bootstrap(a,b,reps=4000,seed=20261016):
    if not a:return [None,None]
    r=random.Random(seed);n=len(a);xs=[]
    for _ in range(reps):
        ids=[r.randrange(n) for _ in range(n)]
        xs.append(sum(b[i]-a[i] for i in ids)/n)
    xs.sort();return [round(xs[int(.025*reps)]*100,2),round(xs[int(.975*reps)-1]*100,2)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('reports',nargs='+',type=Path)
    p.add_argument('--out',type=Path,default=Path('comparison.json'))
    z=p.parse_args()
    runs={}
    for file in z.reports:
        d=json.loads(file.read_text());runs[d['arm']]=d
    if 'C' not in runs or 'D0' not in runs: print('WARNING: C vs D0 unavailable; controller value unmeasured.')
    pairs=[('A','B'),('B','C'),('C','D0'),('D0','D1'),('D1','E')]
    out={'scores':{k:{'solved':v['solved'],'total':v['tasks'],'percent':round(100*v['solve_rate'],2)} for k,v in runs.items()},'paired':[]}
    for left,right in pairs:
        if left not in runs or right not in runs:continue
        a={x['task_id']:int(x['passed']) for x in runs[left]['results']}
        b={x['task_id']:int(x['passed']) for x in runs[right]['results']}
        ids=sorted(set(a)&set(b));x=[a[i] for i in ids];y=[b[i] for i in ids]
        out['paired'].append({'comparison':f'{right} minus {left}','paired_tasks':len(ids),'gain_points':round(100*(sum(y)-sum(x))/len(ids),2),'bootstrap_95pct_CI_points':paired_bootstrap(x,y),'gained':sum(c==0 and d==1 for c,d in zip(x,y)),'lost':sum(c==1 and d==0 for c,d in zip(x,y))})
    z.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
