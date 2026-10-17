"""[MXC replacement] Build the exact commit file list, secret-scan a copy of it (tools/secret_scan.py), print the list. Writes nothing into the repo."""
import shutil
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "tools"))
import secret_scan  # noqa: E402

files = ["MXC_MIGRATION.md", "MXC_REPLACEMENT_REPORT.md", "research/EL_SANDBOX_EXPLAINED.md", "evidence_ladder/controller/sandbox_provider.py", "evidence_ladder/controller/mxc_sandbox.py",
         "evidence_ladder/controller/win_job.py", "evidence_ladder/controller/mxc_dialect/__init__.py", "evidence_ladder/controller/mxc_dialect/posix_lite.py",
         "evidence_ladder/controller/tests/phaseMXC_tests.py", "kaggle/mxc_probe_v1.py", "kaggle/mxc_probe_v1/mxc_probe_v1.py", "kaggle/mxc_probe_v1/kernel-metadata.json",
         "tools/mxc_benchmark.py", "tools/mxc_suite_runner.py", "tools/mxc_commit_prep.py", "tools/mxc_bridge/bridge.mjs", "tools/mxc_bridge/package.json", "tools/mxc_bridge/package-lock.json",
         "tools/mxc_bridge/.gitignore", "tools/mxc_bridge/probe.mjs"]
files += sorted(p.relative_to(root).as_posix() for p in (root / "tools/mxc_bridge/exp").glob("*") if p.is_file())
for d in ["mxc-baseline-wsl-20261017T224021Z", "mxc-benchmark-20261017T230215Z", "mxc-benchmark-cold-after-capcache-20261018", "mxc-final-mxc-20261017T231801Z", "mxc-final-wsl-20261017T231801Z",
          "mxc-final-phaseMXC-20261018"]:
    files += sorted(p.relative_to(root).as_posix() for p in (root / "results/raw" / d).rglob("*") if p.is_file())
tmp = Path(tempfile.mkdtemp(prefix="mxc-commit-"))
for f in files:
    t = tmp / f
    t.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / f, t)
hits, fl, aud = secret_scan.scan_tree(tmp)
(root / ".git").exists() and Path(tempfile.gettempdir(), "mxc_files.txt").write_text("\n".join(files), "utf-8")
print("files:", len(files), "secret hits:", len(hits))
for h in hits[:20]:
    print("HIT", h)
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if hits else 0)
