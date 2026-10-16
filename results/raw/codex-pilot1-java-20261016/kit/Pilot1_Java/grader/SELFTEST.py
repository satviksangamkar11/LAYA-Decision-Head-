#!/usr/bin/env python3
"""Audit of a fixed development dataset; does not execute model generated programs."""
import json,subprocess,pathlib,tempfile
HERE=pathlib.Path(__file__).resolve().parent
pub=json.loads((HERE/'public_cases_expected.json').read_text(encoding='utf-8'))
priv=json.loads((HERE/'protected_cases.json').read_text(encoding='utf-8'))
assert len(pub)==len(priv)==40
assert all(a['id']==b['id'] and len(a['cases'])==4 and len(b['cases'])==40 for a,b in zip(pub,priv))
with tempfile.TemporaryDirectory() as tmp:
  c=subprocess.run(['javac','-encoding','UTF-8','-d',tmp,str(HERE/'support'/'Json.java'),str(HERE/'ReferenceRunner.java'),str(HERE/'ReferenceSolutions.java')],capture_output=True,text=True,timeout=25)
  assert c.returncode==0,c.stderr
  requests=[];expected=[]
  for i,(a,b) in enumerate(zip(pub,priv),1):
    for case in a['cases']+b['cases']:
      requests.append({'task_number':i,'input':case['input']});expected.append(case['expected'])
  payload='\n'.join(json.dumps(x,separators=(',',':'),ensure_ascii=False) for x in requests)+'\n'
  p=subprocess.run(['java','-Xmx256m','-cp',tmp,'ReferenceRunner'],input=payload,text=True,capture_output=True,timeout=80)
  assert p.returncode==0,p.stderr[:2000]
  got=[json.loads(x) for x in p.stdout.splitlines()]
  assert len(got)==len(expected)==1760,(len(got),len(expected))
  import sys
  sys.path.insert(0,str(HERE))
  from grade import same
  differences=[j for j,(x,y) in enumerate(zip(got,expected)) if not same(x,y)]
  assert not differences,'Reference mismatch at '+repr(differences[:5])
  assert all(any(case['expected'] is not None for case in a['cases']+b['cases']) for a,b in zip(pub,priv)),'Null mutant survived some task'
  assert all(len(set(json.dumps(case['input'],sort_keys=True,ensure_ascii=False) for case in a['cases']+b['cases']))==44 for a,b in zip(pub,priv))
print('PASS: 40 Java references compiled; 1760/1760 case outputs match; 40/40 constant-null mutants rejected; all inputs unique per task')
