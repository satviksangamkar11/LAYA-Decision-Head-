"""Freeze-and-hash before any LOCKED action (amendment e10 of configs/a5_v3_spec.toml). Create-only; it never touches the LOCKED data and refuses to freeze if LOCKED has already been captured.
    freeze : PYTHONPATH=. .venv/Scripts/python.exe tools/freeze_head_v3.py freeze --selection results/raw/selection-v3-*.json --screen results/raw/screen-v3-*.json \
                 --checkpoints <final head .pt files> --temperature <CAL temperature json> [--allow-dirty] [--allow-code-change-since-screen AUDIT.json]
    verify : PYTHONPATH=. .venv/Scripts/python.exe tools/freeze_head_v3.py verify results/raw/freeze-v3-<time>.json     (exit 0 only if every recorded hash still matches and LOCKED is still untouched)
Records: git commit and dirty state, sha256 of the head code (decision_head/v3_*.py, tools/head_v3_worker.py, compiler, capture), the three configs, the TRAIN/CAL manifests, the capture and text-embedding verification records,
the frozen sampler artifact, the selection and screen results, every final checkpoint and the CAL temperature file, the selected cell, the seeds, the registered thresholds, and the locked-access-log length.
Refusals: screen PRE_LOCKED_STOP true; a missing checkpoint or temperature file; a dirty tree (unless --allow-dirty, which records the diff hash); head code that differs from the code hashed in the screen's provenance
(unless --allow-code-change-since-screen names an audit file proving equivalence); any LOCKED capture file or LOCKED text-embedding output already present locally.
The commit itself is the user's action: this tool records the commit it ran at and says if the tree is dirty."""
import argparse, datetime, glob, hashlib, json, subprocess, sys, tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


CODE_GLOBS = ["decision_head/v3_*.py", "tools/head_v3_worker.py", "tools/freeze_head_v3.py", "compiler/*.py", "capture/*.py", "state/*.py"]
FIXED = ["configs/a5_v2_spec.toml", "configs/a5_v3_spec.toml", "configs/thresholds.toml", "configs/phase2_device_rule_v1.toml", "data/v3/manifest-train.json", "data/v3/manifest-cal.json", "data/v3/manifest-locked.json",
         "data/a5v2/odds-a5v2-odds-1.pt", "results/raw/qwen3-4b-thinking-2507.manifest.json", "results/raw/phase2-capture-verify-20261003-145647.json", "results/raw/textemb-t4-verify-20261003-155540.json"]
LOCKED_OUTPUTS = ["data/v3/capture_t4/locked", "data/v3/textemb_t4_locked", "data/v3/capture/locked"]


def locked_untouched():
    present = [p for p in LOCKED_OUTPUTS if (ROOT / p).exists()]
    return present, len((ROOT / "results/raw/locked-access-log.jsonl").read_text().splitlines())


def freeze(a):
    sel, scr = json.load(open(a.selection)), json.load(open(a.screen))
    problems = []
    if scr.get("pre_locked_stop") is not False:
        problems.append("the screen did not clear (pre_locked_stop is not false): LOCKED must not be touched")
    present, n_log = locked_untouched()
    if present:
        problems.append(f"LOCKED outputs already exist locally: {present}")
    files = {}
    for g in CODE_GLOBS:
        for f in sorted(ROOT.glob(g)):
            files[f.relative_to(ROOT).as_posix()] = sha(f)
    for f in FIXED:
        files[f] = sha(ROOT / f)
    for name, p in (("selection", a.selection), ("screen", a.screen), ("temperature", a.temperature)):
        if not Path(p).exists():
            problems.append(f"missing {name} file {p}")
        else:
            files[Path(p).resolve().relative_to(ROOT).as_posix()] = sha(p)
    ck = [Path(p) for g in a.checkpoints for p in (glob.glob(g) or [g])]
    if not ck or not all(p.exists() for p in ck):
        problems.append("final head checkpoints are missing")
    for p in ck:
        if p.exists():
            files[p.resolve().relative_to(ROOT).as_posix()] = sha(p)
    dirty = git("status", "--porcelain", "--", "decision_head", "tools", "configs", "compiler", "capture", "state")
    if dirty and not a.allow_dirty:
        problems.append("the tree is dirty in code or configs: commit first (the user's action) or pass --allow-dirty, which records the diff hash")
    trainer = (scr.get("provenance") or {}).get("trainer_sha256", {})
    changed = [m for m, h in trainer.items() if files.get(f"decision_head/{m}.py") != h]
    if changed and not a.allow_code_change_since_screen:
        problems.append(f"head code changed since the screen: {changed}; pass --allow-code-change-since-screen AUDIT.json with an equivalence audit")
    if problems:
        print("REFUSED:\n- " + "\n- ".join(problems)); sys.exit(1)
    th = tomllib.load(open(ROOT / "configs/thresholds.toml", "rb"))
    rec = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), git=dict(commit=git("rev-parse", "HEAD"), dirty=bool(dirty), dirty_status=dirty.splitlines(),
               dirty_diff_sha256=hashlib.sha256(git("diff", "HEAD", "--", "decision_head", "tools", "configs", "compiler", "capture", "state").encode()).hexdigest()),
               selected_cell=sel["chosen"], selection_seeds=sel["seeds"], final_seeds=sel["final_seeds"], screen=dict(point=scr["point"], ci95=scr["ci95"], pre_locked_stop=scr["pre_locked_stop"]),
               thresholds_amendment_e=th["amendment_e_thresholds"], locked_access_log_lines=n_log, locked_outputs_present=[], files=files,
               code_changed_since_screen=changed, equivalence_audit=(str(a.allow_code_change_since_screen) if changed else None),
               note="Frozen BEFORE any LOCKED capture upload or LOCKED accuracy. A new head after the LOCKED read needs a fresh LOCKED-2. The commit that contains this state is the user's action.")
    p = Path(a.out) if a.out else ROOT / f"results/raw/freeze-v3-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    assert not p.exists(), f"{p} exists (create-only)"
    json.dump(rec, open(p, "w"), indent=1)
    print(f"frozen {len(files)} files, cell {sel['chosen']}, commit {rec['git']['commit'][:8]} dirty={rec['git']['dirty']} -> {p}\nfreeze file sha256 {sha(p)}")


def verify(a):
    rec = json.load(open(a.freeze))
    bad = [f for f, h in rec["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    present, n_log = locked_untouched()
    ok = not bad and not present
    print(json.dumps(dict(files=len(rec["files"]), mismatches=bad[:20], locked_outputs_present=present, locked_access_log_lines_then=rec["locked_access_log_lines"], now=n_log, verdict="MATCH" if ok else "DIFFERS"), indent=1))
    sys.exit(0 if ok else 1)


ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
f = sp.add_parser("freeze"); f.add_argument("--selection", required=True); f.add_argument("--screen", required=True); f.add_argument("--checkpoints", nargs="+", required=True)
f.add_argument("--temperature", required=True); f.add_argument("--allow-dirty", action="store_true"); f.add_argument("--out", default=None, help="test runs only; default results/raw/freeze-v3-<time>.json"); f.add_argument("--allow-code-change-since-screen", default=None)
v = sp.add_parser("verify"); v.add_argument("freeze")
a = ap.parse_args()
freeze(a) if a.cmd == "freeze" else verify(a)
