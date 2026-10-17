"""[MXC replacement] Stage ONLY the hunk of evidence_ladder/controller/agentic.py that contains 'env_label' (the file also holds another agent's uncommitted timeline work). Usage: python tools/mxc_stage_hunk.py"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = "evidence_ladder/controller/agentic.py"
diff = subprocess.run(["git", "diff", "-U0", "--", path], cwd=root, capture_output=True, text=True, encoding="utf-8").stdout
head, *hunks = re.split(r"(?m)^(?=@@ )", diff)
keep = [h for h in hunks if "env_label" in h]
if len(keep) != 1:
    sys.exit("expected exactly one hunk with env_label, found %d" % len(keep))
patch = head + keep[0]
f = Path(tempfile.gettempdir()) / "mxc_agentic_hunk.patch"
f.write_bytes(patch.encode("utf-8"))
r = subprocess.run(["git", "apply", "--cached", "--unidiff-zero", str(f)], cwd=root, capture_output=True, text=True)
print(r.returncode, r.stderr.strip()[:300])
sys.exit(r.returncode)
