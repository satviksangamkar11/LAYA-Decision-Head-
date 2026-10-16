"""Trusted reference / basic mutation smoke checks. No candidate execution."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).parent))
from oracles import ORACLES
p=json.loads((Path(__file__).parent/'protected_cases.json').read_text())
public=json.loads((Path(__file__).parent.parent/'worker'/'public_cases.json').read_text()) if (Path(__file__).parent.parent/'worker'/'public_cases.json').exists() else None
count=0;mutants=0
for task in p:
    key=task['task_id'][-2:]
    for c in task['cases']:
        assert ORACLES[key](c['input'])==c['expected'], (key,c)
        count+=1
    # Basic deliberately wrong constant-output mutant MUST be killed. This is a smoke check, not meaningful mutation adequacy.
    assert any(c['expected'] is not None for c in task['cases']),key
    mutants+=1
if public:
    for task in public:
        key=task['task_id'][-2:]
        for c in task['cases']:
            assert ORACLES[key](c['input'])==c['expected']
            count+=1
print(json.dumps({'tasks':len(p),'cases_validated':count,'constant_none_mutants_killed':mutants,'status':'PASS','limitation':'constant mutants are trivial; no independent external oracle review'}))
