"""[EL-ARCH-1 D1/S3] Pack the Kaggle bundle for the acceptance run (local packing only; NOTHING is pushed, no model/torch is run). Create-only.
  python tools/pack_el_arch1_bundle_v1.py [--base <results/raw/el-laya-int1-bundle-*>] [--out-root <dir>]
Creates <out-root>/el-arch1-bundle-<UTC>/
  dataset/                 the Kaggle dataset `sangeetasangamkar/evidence-ladder-el-arch1-bundle-v1` (BUNDLE_MANIFEST.json bundle == evidence_ladder_el_arch1_v1, sha256 of every file)
  kernels/el_arch1_generate_v1.py   the generation kernel TEMPLATE (kaggle/el_arch1_generate_v1.py, unchanged copy)
  kernels/el_arch1_score_v1.py      the capture + LAYA + probe + random-head kernel TEMPLATE = kaggle/evidence_ladder_laya_t6_step_v1.py (frozen file, NOT edited) with three exact string
                                    replacements applied here: the bundle name, the probe (-> evidence_ladder.models.probe.refit_probe on exactly the 334 TRAIN pools, fingerprinted) and the random head
                                    (-> the proper untrained seeded heads of evidence_ladder.models.random_head instead of the old trained-adapter+random-head MISMATCH control)
  PUSH_COMMANDS.txt, PACK_REPORT.json
Dataset content = the frozen LAYA-INT-1 bundle (adapters, 31 pretrained tensors, stored TRAIN/VAL features, request texts, label projection, capture code; every file re-verified against its
manifest before it is copied) + the CURRENT evidence_ladder code (common, models, controller, critic_batch; no tests, no execution/protected grader, no orchestrator) + configs/el_arch1_*.toml.
NOT included (asserted): protected or hidden cases, grader/oracle modules, tests, keys or secret-shaped text, CAL/TEST/LOCKED data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OWNER = "sangeetasangamkar"
BUNDLE_ID = "evidence_ladder_el_arch1_v1"
OLD_BUNDLE_ID = "evidence_ladder_laya_int1_v1"
CODE_DIRS = ("common", "models", "controller", "critic_batch")
EXTRA = ("configs/el_arch1_limits_v1.toml", "configs/el_arch1_gates_v1.toml", "configs/critic_concurrency_v1.toml")
KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}|Bearer\s+[A-Za-z0-9._\-]{20,}|sk-[A-Za-z0-9_\-]{20,}|hf_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{10,}")      # key VALUES: never allowed anywhere
PATH_RE = re.compile(r"\.secrets/|kaggle\.json")      # a MENTION of a secret path: only allowed where it is a deny pattern
PATH_MENTION_ALLOWED = {"evidence_ladder/controller/risk.py": "the fixed action-risk table lists secret path patterns to REFUSE them"}
FORBIDDEN_PATH = re.compile(r"(^|/)(tests?|testing|execution|orchestrator)(/|$)|cases|oracle|grader|protected\.py")
SCORE_REPLACEMENTS = [      # (old, new): each must occur EXACTLY once in kaggle/evidence_ladder_laya_t6_step_v1.py
    ('"evidence_ladder_laya_int1_v1"', '"%s"' % BUNDLE_ID),
    ('probe = LF.fit_probe(rows, C=0.01); rand_head = LF.build_head(PRE, G("models/final-s0.pt"), random_head_seed=0); RS["probe_train_pools"] = len(rows)',
     'from evidence_ladder.models import probe as PR, random_head as RH\n'
     '        fit = PR.refit_probe(rows, expect_pools=PR.TRAIN_POOLS); probe = fit["probe"]; RS["probe_train_pools"] = len(rows); RS["probe_fingerprint"] = fit["fingerprint"]; RS["probe_order_sha256"] = fit["order_sha256"]\n'
     '        rand_head = RH.build_random_heads(scaler_from=G("models/final-s0.pt")); RS["random_head_state_sha256"] = {n: RH.state_sha256(h) for n, h in rand_head.items()}; RS["random_head_role"] = RH.ROLE')]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


def latest_base() -> Path:
    c = [p for p in sorted((ROOT / "results/raw").glob("el-laya-int1-bundle-*")) if (p / "BUNDLE_MANIFEST.json").is_file()]
    if not c: raise SystemExit("no el-laya-int1-bundle-* with a BUNDLE_MANIFEST.json found")
    return c[-1]


def patch_score_kernel(src: str) -> str:
    for old, new in SCORE_REPLACEMENTS:
        if src.count(old) != 1: raise SystemExit("score-kernel patch: %r occurs %d times (expected 1): the frozen t6 kernel changed" % (old[:50], src.count(old)))
        src = src.replace(old, new)
    return src


def build(base: Path, out_root: Path, repo: Path = ROOT, utc: str = None) -> Path:
    utc = utc or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bm = json.loads((base / "BUNDLE_MANIFEST.json").read_text("utf-8"))
    if bm.get("bundle") != OLD_BUNDLE_ID: raise SystemExit("base bundle is %r, expected %s" % (bm.get("bundle"), OLD_BUNDLE_ID))
    out = out_root / ("el-arch1-bundle-" + utc)
    ds, kn = out / "dataset", out / "kernels"
    ds.mkdir(parents=True, exist_ok=False); kn.mkdir()
    files = {}

    def put(rel: str, src: Path):
        d = ds / rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, d); files[rel] = sha_file(d)
    for rel, h in sorted(bm["files"].items()):                                   # 1. the frozen base content, each file verified first
        if rel.startswith("evidence_ladder/"): continue                           # replaced by the current code below
        s = base / rel
        if not s.is_file() or sha_file(s) != h: raise SystemExit("base bundle file %s missing or differs from its manifest" % rel)
        put(rel, s)
    for d in CODE_DIRS:                                                           # 2. current code
        for p in sorted((repo / "evidence_ladder" / d).glob("*.py")): put("evidence_ladder/%s/%s" % (d, p.name), p)
    put("evidence_ladder/__init__.py", repo / "evidence_ladder/__init__.py")
    for rel in EXTRA: put(rel, repo / rel)
    bad = [f for f in files if f.startswith("evidence_ladder/") and FORBIDDEN_PATH.search(f)]
    if bad: raise SystemExit("forbidden material in the bundle: %s" % bad)
    leaks = [f for f in files if f.endswith((".py", ".json", ".toml", ".txt", ".jsonl")) and (lambda t: KEY_RE.search(t) or (PATH_RE.search(t) and f not in PATH_MENTION_ALLOWED))((ds / f).read_text("utf-8", "replace"))]
    if leaks: raise SystemExit("secret-shaped text in: %s" % leaks)
    from evidence_ladder.models import qwen as Q
    man = {"bundle": BUNDLE_ID, "qwen_revision": Q.QWEN_REVISION, "created_utc": utc, "base_bundle": base.name, "base_manifest_sha256": sha_file(base / "BUNDLE_MANIFEST.json"), "files": files,
           "note": "frozen LAYA-INT-1 content + current Evidence-Ladder code. No protected/hidden cases, no grader, no tests, no keys."}
    (ds / "BUNDLE_MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True), "utf-8")
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "evidence-ladder-el-arch1-bundle-v1", "id": OWNER + "/evidence-ladder-el-arch1-bundle-v1", "licenses": [{"name": "other"}]}, indent=2), "utf-8")
    gen = (repo / "kaggle/el_arch1_generate_v1.py").read_text("utf-8")
    assert "__GEN_REQUEST_JSON__" in gen and BUNDLE_ID in gen
    (kn / "el_arch1_generate_v1.py").write_text(gen, "utf-8")
    score = patch_score_kernel((repo / "kaggle/evidence_ladder_laya_t6_step_v1.py").read_text("utf-8"))
    assert "__STEP_REQUEST_JSON__" in score and BUNDLE_ID in score and OLD_BUNDLE_ID not in score
    (kn / "el_arch1_score_v1.py").write_text(score, "utf-8")
    cmds = ["# run from D:/local model with the DEFAULT Kaggle account (sangeetasangamkar); NOTHING below has been run",
            "kaggle datasets create -p \"%s\" --dir-mode zip      # first time; later: kaggle datasets version -p ... -m \"msg\" --dir-mode zip" % ds.as_posix(),
            "# the acceptance driver writes the per-round kernel directories (handoff/round<k>/...) and prints the exact push/output commands; kernels: el-arch1-generate-v1, el-arch1-score-live-v1, el-arch1-score-verify-v1"]
    (out / "PUSH_COMMANDS.txt").write_text("\n".join(cmds) + "\n", "utf-8")
    (out / "PACK_REPORT.json").write_text(json.dumps({"out": str(out), "files": len(files), "bytes": sum((ds / f).stat().st_size for f in files), "bundle_manifest_sha256": sha_file(ds / "BUNDLE_MANIFEST.json"),
                                                      "score_kernel_sha256": sha_file(kn / "el_arch1_score_v1.py"), "gen_kernel_sha256": sha_file(kn / "el_arch1_generate_v1.py")}, indent=1), "utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base"); ap.add_argument("--out-root", default=str(ROOT / "results/raw"))
    a = ap.parse_args()
    o = build(Path(a.base) if a.base else latest_base(), Path(a.out_root))
    print(json.dumps(json.loads((o / "PACK_REPORT.json").read_text("utf-8"))))
