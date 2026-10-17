"""[EL-BANK40] Pack the 40-task Kaggle run (local packing only; NOTHING is uploaded or pushed here). Create-only. Stdlib + the repo's own bank40 helpers.

  PYTHONPATH=. .venv/Scripts/python.exe tools/pack_el_bank40_v1.py --thinking-policy budget --think-budget 512 --answer-tokens 1500
  PYTHONPATH=. .venv/Scripts/python.exe tools/pack_el_bank40_v1.py --thinking-policy prompt_brief --brief "Reason briefly, then answer." --ceiling 8192
  PYTHONPATH=. .venv/Scripts/python.exe tools/pack_el_bank40_v1.py --thinking-policy model_variant --profile-id <id> --model-glob <dir fragment> --ceiling 8192
  (the thinking policy is REQUIRED: no default; see evidence_ladder/bank40.check_policy)

Output results/raw/el-bank40-bundle-<UTC>/: dataset/ (the code + 40 PUBLIC task records + sandbox support files + run_config.json + BUNDLE_MANIFEST.json), kernel_pass1/ (GPU kernel, T4 x2),
kernel_pass2/ (CPU kernel that applies the critic responses with replayed generations), PACK.json.
The bundle holds ONLY: evidence_ladder/** (no tests, no execution/protected.py), the four existing sandbox scripts + the two Java support files at the paths evidence_ladder/execution/config.py expects
(they read protected cases only when run as scripts with a --cases path; none is in the bundle), the thresholds/limits/critic/backbone configs, and the 40 tasks with prompt + public tests.
Asserts: no hidden/protected/grader/oracle path part, no held-case byte marker (taken from the real protected_cases.json / oracles.py, which are NOT bundled), no key-shaped string, task records have exactly
the public keys, the task sources match the sha256 pinned in configs/el_bank40_v1.json, and the bundle imports (controller, backend, jail, bank40) in a clean interpreter."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_ladder import bank40 as B   # noqa: E402

OWNER = "sangeetasangamkar"
BUNDLE_DS = "%s/el-bank40-bundle-v1" % OWNER
QWEN_DS = "leminhhung0101/qwen3-4b-thinking-2507"
SANDBOX_FILES = ["results/raw/codex-pilot1-20261016/fixed_v2/grader/sandbox/grade_sandboxed_v2.py", "results/raw/codex-pilot1-20261016/fixed_v2/grader/sandbox/runner_v2.py",
                 "results/raw/codex-pilot1-java-20261016/fixed_v1/grader/grade_sandboxed_java_v1.py",
                 "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker/support/Json.java", "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/worker/support/Runner.java"]
CONFIGS = ["configs/el_bank40_v1.json", "configs/el_arch1_limits_v1.toml", "configs/el_arch1_limits_v2.toml", "configs/el_arch1_gates_v1.toml", "configs/critic_concurrency_v1.toml", "configs/backbones_v1.toml",
           "configs/thresholds_evidence_ladder_v1.toml", "configs/thresholds_evidence_ladder_v1.toml.sha256"]
SPECS = ["specs/evidence_ladder_v1/schemas_v1.json", "specs/evidence_ladder_v1/contract/contract_lib_v1.py"]
EXCLUDE_CODE = {"evidence_ladder/execution/protected.py"}
MARKER_SOURCES = ["results/raw/codex-pilot1-20261016/fixed_v2/grader/protected_cases.json", "results/raw/codex-pilot1-20261016/fixed_v2/grader/oracles.py",
                  "results/raw/codex-pilot1-java-20261016/fixed_v1/grader/protected_cases.json", "results/raw/codex-pilot1-java-20261016/kit/Pilot1_Java/grader/protected_cases.json"]


def sha(b) -> str: return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def code_files(root: Path = ROOT):
    out = {}
    for p in sorted((root / "evidence_ladder").rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and "__pycache__" not in p.parts and "tests" not in p.parts and p.suffix in (".py", ".json", ".toml") and rel not in EXCLUDE_CODE and "/fixtures/" not in "/" + rel:
            out[rel] = p.read_bytes()
    return out


def load_bank(root: Path = ROOT):
    cfg = json.loads((root / "configs/el_bank40_v1.json").read_text("utf-8"))
    tasks = []
    for lang in ("python", "java"):
        src = cfg["sources"][lang]
        raw = (root / src["file"]).read_bytes()
        assert sha(raw) == src["sha256"], "task source %s differs from the sha256 pinned in configs/el_bank40_v1.json" % src["file"]
        by = {}
        for ln in raw.decode("utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln); by[d.get("task_id") or d["id"]] = d
        for tid in cfg[lang]["tasks"]:
            d = by[tid]; pt = d["public_tests"]
            assert pt and all(set(t) == {"input", "expected"} for t in pt), tid
            tasks.append({"task_id": tid, "language": lang, "difficulty": d.get("difficulty", ""), "prompt": d["prompt"], "public_tests": pt})
    assert len(tasks) == cfg["total"] == 40 and len({t["task_id"] for t in tasks}) == 40
    return tasks


def held_markers(root: Path = ROOT, per_file: int = 4, width: int = 48):
    ms = []
    for rel in MARKER_SOURCES:
        p = root / rel
        if p.is_file():
            b = p.read_bytes()
            for k in range(1, per_file + 1):
                i = len(b) * k // (per_file + 1); ms.append(b[i:i + width])
    return [m for m in ms if len(m) >= 32]


def policy_from_args(a) -> dict:
    d = {"policy": a.thinking_policy}
    if a.thinking_policy == "budget": d.update(think_budget=a.think_budget, answer_tokens=a.answer_tokens)
    elif a.thinking_policy == "prompt_brief": d.update(brief_instruction=a.brief, ceiling=a.ceiling)
    else: d.update(profile_id=a.profile_id, model_glob=a.model_glob, ceiling=a.ceiling)
    return B.check_policy(d)          # refuses a missing / malformed policy BEFORE anything is written


def build_files(policy: dict, root: Path = ROOT, commit: str = "") -> dict:
    files = {k: v for k, v in code_files(root).items()}
    for rel in SANDBOX_FILES + CONFIGS + SPECS: files[rel] = (root / rel).read_bytes()
    tasks = load_bank(root)
    files["bank40_tasks_public.jsonl"] = ("\n".join(json.dumps(t, sort_keys=True, ensure_ascii=False) for t in tasks) + "\n").encode("utf-8")
    files["run_config.json"] = json.dumps({"thinking_policy": policy, "seeds": [1, 2, 3, 4, 5, 6, 7, 8], "controller": {"mode": "ladder", "k": 4, "l2": True, "l2_n": 6, "discriminate": True, "critic": True,
                                           "max_rounds": 2, "mutant_limit": 8}, "scorer": "none", "label": "development results (D1 waiver), public examples only; hidden grading NOT done"}, indent=1, sort_keys=True).encode("utf-8")
    B.assert_public_bundle(files, held_markers(root))
    files["BUNDLE_MANIFEST.json"] = json.dumps({"bundle": "el_bank40_v1", "tasks": len(tasks), "commit": commit, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                "files": {k: sha(v) for k, v in files.items()}, "note": "code + public task prompts/examples only; no hidden cases, graders, oracles, keys or models"}, indent=1, sort_keys=True).encode("utf-8")
    return files


def kernel_meta(slug: str, code: str, gpu: bool, sources: list, kernel_sources: list) -> dict:
    return {"id": "%s/%s" % (OWNER, slug), "title": slug, "code_file": code, "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": sources, "kernel_sources": kernel_sources, "competition_sources": [], "model_sources": []}


def import_smoke(bundle_dir: Path) -> dict:
    code = ("import sys; sys.path.insert(0, %r)\nfrom evidence_ladder import bank40\nfrom evidence_ladder.controller.controller import EvidenceLadderController\n"
            "from evidence_ladder.execution.backend import SandboxExecutionBackend\nfrom evidence_ladder.execution import engine\nimport os\nif os.name == 'posix':\n    from evidence_ladder.execution import kaggle_jail\nfrom evidence_ladder.models.broker import StagedCriticBroker\n"
            "from evidence_ladder.orchestrator import isolation\nprint('IMPORT_OK')" % str(bundle_dir))
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(bundle_dir))
    return {"ok": "IMPORT_OK" in r.stdout, "stderr_tail": r.stderr[-600:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinking-policy", required=True, choices=B.POLICIES)
    ap.add_argument("--think-budget", type=int); ap.add_argument("--answer-tokens", type=int)
    ap.add_argument("--brief"); ap.add_argument("--ceiling", type=int)
    ap.add_argument("--profile-id"); ap.add_argument("--model-glob")
    ap.add_argument("--resume-dataset", help="slug of a dataset holding the previous pass-1 output (tasks/*/DONE) to resume from")
    ap.add_argument("--out-root", default=str(ROOT / "results/raw"))
    a = ap.parse_args(argv)
    policy = policy_from_args(a)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    files = build_files(policy, ROOT, commit)
    out = Path(a.out_root) / ("el-bank40-bundle-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ds, k1, k2 = out / "dataset", out / "kernel_pass1", out / "kernel_pass2"
    for d in (ds, k1, k2): d.mkdir(parents=True, exist_ok=False)
    for rel, data in files.items():
        p = ds / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "el-bank40-bundle-v1", "id": BUNDLE_DS, "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    smoke = import_smoke(ds)
    assert smoke["ok"], "the bundle does not import in a clean interpreter: " + smoke["stderr_tail"]
    tpl = (ROOT / "kaggle/el_bank40_run_v1.py").read_text("utf-8")
    assert 'RUN_MODE = "pass1"' in tpl
    for kd, mode, slug, gpu, srcs, ksrc in ((k1, "pass1", "el-bank40-run-pass1-v1", True, [BUNDLE_DS, QWEN_DS] + ([a.resume_dataset] if a.resume_dataset else []), []),
                                            (k2, "pass2", "el-bank40-run-pass2-v1", False, [BUNDLE_DS, "%s/el-bank40-pass1-out-v1" % OWNER, "%s/el-bank40-critic-responses-v1" % OWNER], [])):
        code = "el_bank40_run_%s.py" % mode
        (kd / code).write_text(tpl.replace('RUN_MODE = "pass1"', 'RUN_MODE = "%s"' % mode, 1), encoding="utf-8")
        (kd / "kernel-metadata.json").write_text(json.dumps(kernel_meta(slug, code, gpu, srcs, ksrc), indent=1), encoding="utf-8")
    pack = {"out": str(out), "files": len(files), "bytes": sum(len(v) for v in files.values()), "thinking_policy": policy, "commit": commit, "import_smoke": smoke["ok"], "held_markers_checked": len(held_markers()),
            "kernels": {"pass1": str(k1), "pass2": str(k2)}, "dataset": str(ds), "manifest_sha256": sha(files["BUNDLE_MANIFEST.json"])}
    (out / "PACK.json").write_text(json.dumps(pack, indent=1), encoding="utf-8")
    print(json.dumps(pack, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
