"""[EL-ARCH-1 D1/S4] Pack ONE Kaggle CPU test kernel that runs ALL Evidence-Ladder suites: the existing six (contract, controller, orchestrator, models, critic_batch, replay) plus the new ones
(phaseA, phaseB, phaseC, stepE, laya_integration with T4_scores.json, phaseD1). Local packing only; nothing is pushed (the coordinator pushes it later). Stdlib. Create-only.
  python tools/pack_el_arch1_tests_v1.py [T4_scores.json]        (default: the integration_run2 scores of the frozen LAYA-INT-1 receipts)
Approach = tools/pack_laya_tests_v1.py: run tools/pack_evidence_ladder_tests_v1.py UNCHANGED (it copies the code tree and writes dataset + kernel), then patch the generated COPY of the kernel script
(kaggle/evidence_ladder_tests_v1.py itself is not edited) and add the files the new suites import. Own ids: dataset sangeetasangamkar/evidence-ladder-el-arch1-tests-bundle-v1, kernel
sangeetasangamkar/evidence-ladder-el-arch1-tests-run-v1. No WSL on Kaggle: the *_wsl tests skip and are listed under `skipped` (LOGIC-ONLY there); the WSL suites run on the PC."""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = ROOT / "results/raw/laya-integration-20261016T184816Z/integration_run2/T4_scores.json"
SUITES = {      # name -> script relative to the repo root (+ optional extra argument)
    "phaseA": "evidence_ladder/controller/tests/phaseA_tests.py",
    "phaseB": "evidence_ladder/controller/tests/phaseB_tests.py",
    "phaseC": "evidence_ladder/orchestrator/tests/phaseC_tests.py",
    "stepE": "evidence_ladder/critic_batch/tests/stepE_tests.py",
    "laya_integration": "evidence_ladder/models/tests/laya_integration_tests.py",
    "phaseD1": "evidence_ladder/controller/tests/phaseD1_tests.py"}
EXTRA_FILES = ["tools/laya_int1_t6.py", "tools/laya_int1_t6_sandbox.py", "tools/fable_wsl_runner.py", "tools/el_arch1_stubs.py", "tools/el_arch1_variants.py", "tools/el_arch1_acceptance.py",
               "tools/el_arch1_conformance.py", "tools/pack_el_arch1_bundle_v1.py", "tools/laya_int1_stepE.py",
               "configs/el_arch1_limits_v1.toml", "configs/el_arch1_gates_v1.toml", "configs/laya_capture_code_pins_v2.toml", "configs/laya_head_gate_v1.toml", "configs/laya_int1_thresholds_v1.toml",
               "capture/capture.py", "capture/capture_resident.py", "decision_head/head.py", "decision_head/fable_features.py", "decision_head/fable_pool.py",
               "decision_head/fable_train.py", "tools/fable_capture.py", "kaggle/el_arch1_generate_v1.py", "kaggle/evidence_ladder_gpu_gen_v1.py", "tools/el_arch1_glm_live.py", "kaggle/evidence_ladder_laya_t6_step_v1.py"]


# receipts the suites read (READ-ONLY inputs of phaseA / stepE / phaseC): small, text; copied byte for byte
EXTRA_TREES = ["results/raw/laya-integration-20261016T184816Z/stepE_20261017T070508Z", "results/raw/laya-integration-20261016T184816Z/e2e_pass2"]
EXTRA_GLOBS = ["results/raw/el-laya-int1-bundle-*/gate/system_block.txt"]


def pack(scores: Path = DEFAULT_SCORES, root: Path = ROOT, out_root: Path = None):
    cmd = [sys.executable, str(root / "tools/pack_evidence_ladder_tests_v1.py")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    out = Path(json.loads(r.stdout.strip().splitlines()[-1])["out"])
    ds, kn = out / "dataset", out / "kernel"
    man = json.loads((ds / "MANIFEST.json").read_text("utf-8"))
    for suite, rel in SUITES.items():
        assert rel in man["files"] or (root / rel).is_file(), "suite file missing: " + rel
    for rel in list(SUITES.values()) + EXTRA_FILES + ["results/raw/laya-baseline-v1-20261012T215242Z/MANIFEST.json"]:
        src = root / rel
        if not src.is_file(): raise SystemExit("file for the test bundle missing: " + rel)
        (ds / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, ds / rel)
        man["files"][rel] = hashlib.sha256((ds / rel).read_bytes()).hexdigest()
    extra = [p for t in EXTRA_TREES for p in sorted((root / t).rglob("*")) if p.is_file()] + [p for g in EXTRA_GLOBS for p in sorted(root.glob(g))[-1:]]
    for src in extra:
        rel = src.relative_to(root).as_posix()
        (ds / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, ds / rel)
        man["files"][rel] = hashlib.sha256((ds / rel).read_bytes()).hexdigest()
    shutil.copyfile(scores, ds / "laya_t4_scores.json")
    man["files"]["laya_t4_scores.json"] = hashlib.sha256((ds / "laya_t4_scores.json").read_bytes()).hexdigest()
    (ds / "MANIFEST.json").write_text(json.dumps(man, indent=1), "utf-8")
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "evidence-ladder-el-arch1-tests-bundle-v1", "id": "sangeetasangamkar/evidence-ladder-el-arch1-tests-bundle-v1", "licenses": [{"name": "other"}]}, indent=1))
    k = kn / "evidence-ladder-tests-run-v1.py"
    src = k.read_text("utf-8")
    line = '    "replay": [sys.executable, "evidence_ladder/testing/replay_integration.py"],\n'
    assert line in src, "kaggle/evidence_ladder_tests_v1.py changed shape"
    add = "".join('    "%s": [sys.executable, "%s"%s],\n' % (n, rel, ', "laya_t4_scores.json"' if n == "laya_integration" else "") for n, rel in SUITES.items())
    src = src.replace(line, line + add)
    (kn / "evidence-ladder-el-arch1-tests-run-v1.py").write_text(src, "utf-8"); k.unlink()
    meta = json.loads((kn / "kernel-metadata.json").read_text("utf-8"))
    meta.update({"id": "sangeetasangamkar/evidence-ladder-el-arch1-tests-run-v1", "title": "evidence-ladder-el-arch1-tests-run-v1", "code_file": "evidence-ladder-el-arch1-tests-run-v1.py",
                 "dataset_sources": ["sangeetasangamkar/evidence-ladder-el-arch1-tests-bundle-v1"]})
    (kn / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), "utf-8")
    (out / "PUSH_COMMANDS.txt").write_text("\n".join([
        "# DEFAULT Kaggle account sangeetasangamkar; NOTHING below has been run",
        "kaggle datasets create -p \"%s\" --dir-mode zip" % ds.as_posix(),
        "kaggle kernels push -p \"%s\"" % kn.as_posix(),
        "kaggle kernels status sangeetasangamkar/evidence-ladder-el-arch1-tests-run-v1",
        "kaggle kernels output sangeetasangamkar/evidence-ladder-el-arch1-tests-run-v1 -p <results dir>      # TEST_RESULTS.json"]) + "\n", "utf-8")
    return {"out": str(out), "dataset": str(ds), "kernel": str(kn), "files": len(man["files"]), "suites": ["contract", "controller", "orchestrator", "models", "critic_batch", "replay"] + list(SUITES)}


if __name__ == "__main__":
    print(json.dumps(pack(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SCORES)))
