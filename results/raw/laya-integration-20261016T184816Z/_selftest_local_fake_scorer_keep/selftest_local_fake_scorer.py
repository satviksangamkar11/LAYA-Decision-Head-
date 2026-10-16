import sys, json, shutil
sys.path.insert(0,"tools")
from pathlib import Path
import laya_int1_e2e as E
from laya_int1_t6 import TASKS
from evidence_ladder.controller.agentic import AgenticContext, AgenticRun, AgenticTask
out = Path("results/raw/laya-integration-20261016T184816Z/_selftest_local_fake_scorer")
if out.exists(): shutil.rmtree(out)
T = TASKS["slug"]; ws = out/"ws"
for fn,c in T["files"].items(): E.write_new(ws/fn, c)
class FakeScorer:
    scorer_id="fake"
    def score(self, dt, pool, exploratory=False):
        n=len(pool["options"]); return [float(n-i) for i in range(n)]   # prefers the first option
at = AgenticTask("selftest", T["user"], ws, T["test_cmd"], T["protected"], E.system_block(), 2)
r = AgenticRun(at, AgenticContext(sandbox=E.WslSandbox(out/"_sb", T["protected"]), scorer=FakeScorer(), arm="E"), E.render); r.begin()
fix = json.dumps({"name":"Edit","arguments":{"file_path":"slug.py","old_string":'return re.sub(r"[^a-z0-9]+", "-", text)',"new_string":'return re.sub(r"[^a-z0-9]+", "-", text).strip("-")'}})
wrong = json.dumps({"name":"Edit","arguments":{"file_path":"slug.py","old_string":"text.strip()","new_string":"text"}})
prot = json.dumps({"name":"Edit","arguments":{"file_path":"test_slug.py","old_string":"hello-world","new_string":"hello-world-"}})
rm = json.dumps({"name":"Bash","arguments":{"command":"rm -rf /"}})
rd = json.dumps({"name":"Read","arguments":{"file_path":"slug.py"}})
s1 = r.step([rd, wrong, prot, rm, fix])
print({k:s1[k] for k in ("laya_order","allowed_set","id_order_choice","chosen_index","trusted","laya_changed_the_choice_vs_id_order","trusted_evidence_overrode_laya","tests_pass","label","check_i2")})
res = r.finish()
print(res.answer_label, res.stop_cause, res.journal_hash[:12], r.invariants_checked)
