"""[MULTILANG] Merge the per-backend QUAL_<backend>.json files of one results directory into QUAL.json (language x backend table). `bare` is the negative control and is listed apart, never as a qualification.
A cell is `qualified` only if every check passed; kaggle-jail results produced on the WSL root proxy are labelled `qualified_on_proxy_only` (no Kaggle kernel was run). Usage: python tools/el_multilang_merge.py <dir>"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
out = d / "QUAL.json"
if out.exists():
    sys.exit("refusing to overwrite %s" % out)
recs = {p.stem[5:]: json.loads(p.read_text()) for p in sorted(d.glob("QUAL_*.json"))}
langs = ["c", "cpp", "javascript", "go", "rust"]
table, detail = {}, {}
for l in langs:
    table[l], detail[l] = {}, {}
    for b, r in recs.items():
        if b == "bare":
            continue
        c = r["languages"].get(l, {"state": "not_run"})
        st = c["state"]
        if b == "kaggle-jail" and st == "qualified":
            st = "qualified_on_proxy_only"
        table[l][b] = st
        detail[l][b] = {"state": c["state"], "reason": c.get("reason"), "version": c.get("version"), "failed_checks": [k for k, v in c.get("checks", {}).items() if not v["ok"]],
                        "checks": {k: v["ok"] for k, v in c.get("checks", {}).items()}}
control = {l: {"state": c["state"], "failed_checks": [k for k, v in c.get("checks", {}).items() if not v["ok"]]} for l, c in recs.get("bare", {"languages": {}})["languages"].items()}
doc = {"table": table, "detail": detail, "negative_control_bare": {"note": "no isolation; socket_denied and host_paths_unreadable MUST fail here (they did), proving the checks can detect a leak", "result": control},
       "backends": {b: {"note": r.get("note"), "backend_error": r.get("backend_error"), "capture_limit": r.get("capture_limit"), "utc": r.get("utc"), "listener_positive_control": r.get("listener_positive_control"),
                        "sandbox_calls": r.get("sandbox_calls"), "spy_expected_hits": r.get("spy_expected_hits")} for b, r in recs.items()}}
out.write_text(json.dumps(doc, indent=1))
print(json.dumps(table, indent=1))
