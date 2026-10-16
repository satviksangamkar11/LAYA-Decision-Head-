#!/usr/bin/env python3
"""Paired development-score report. Does not establish statistical significance."""
import argparse,json,pathlib,random,collections,math
p=argparse.ArgumentParser();p.add_argument('reports',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
reps=[json.loads(pathlib.Path(x).read_text()) for x in a.reports]
for r in reps:assert len(r['results'])==40
ids=[x['task_id'] for x in reps[0]['results']]
for r in reps:assert [x['task_id'] for x in r['results']]==ids
rng=random.Random(20261016)
base=[int(x['status']=='PASS') for x in reps[0]['results']]
results=[]
for r in reps:
 score=[int(x['status']=='PASS') for x in r['results']]
 diffs=[a-b for a,b in zip(score,base)]
 bootstrap=[sum(rng.choice(diffs) for _ in range(len(diffs)))/len(diffs) for _ in range(3000)]
 bootstrap.sort()
 results.append({'arm':r['arm'],'solved':sum(score),'success_rate':round(sum(score)/len(score),4),'delta_vs_first_pp':round(100*sum(diffs)/len(diffs),2),'bootstrap_95pp':[round(100*bootstrap[74],2),round(100*bootstrap[2924],2)]})
summary={'note':'Exploratory paired task bootstrap. 40 original tasks are a development set; do not claim generalization or independence.','arms':results}
pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
