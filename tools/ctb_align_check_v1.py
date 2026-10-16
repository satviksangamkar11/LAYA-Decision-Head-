"""P1 (user GO, chat 2026-10-14): deterministic action-to-label alignment check of CodeTraceBench samples using CodeTracer's own parsers
(NJU-LINK/CodeTracer @ 2d302191, MIT). Local CPU exception approved for the 7 sample archives only. Parses files; executes nothing from the archives.

Per sample run: rebuilt step count vs manifest `step_count`; for every flagged step that carries a manifest `action_ref`, does the rebuilt step with the
same id contain that action's command? Evidence levels: ACTION_VERIFIED (count matches and every flagged step has a matching ref),
COUNT_MATCH_ONLY, REF_CONFLICT, MISMATCH_OR_MISSING; only ACTION_VERIFIED is admissible. Caveat: upstream refs were assigned positionally
(backfill_step_refs.py), so even a ref match is not fully independent evidence.

  python -I tools/ctb_align_check_v1.py <codetracer_src> <sample_root> <manifest_parquet> <out_json>
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pyarrow.parquet as pq


def load_upstream(src):
    pkg = types.ModuleType("codetracer")
    pkg.__path__ = []
    sys.modules["codetracer"] = pkg
    spec = importlib.util.spec_from_file_location("codetracer.models", Path(src) / "models/trajectory.py")
    models = importlib.util.module_from_spec(spec)
    sys.modules["codetracer.models"] = models
    spec.loader.exec_module(models)
    parsers = {}
    for name in ("miniswe", "openhands", "terminus2"):
        s = importlib.util.spec_from_file_location("ct_%s" % name, Path(src) / "skills/seed" / name / "parser.py")
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        parsers[name] = m.parser
    return parsers


def swe_steps(run_dir):
    """SWE-style raw runs (no upstream parser): count actions per format. Returns (format, [action texts])."""
    for p in run_dir.rglob("*.traj.json"):
        msgs = json.loads(p.read_text("utf-8", errors="replace")).get("messages") or []
        return "miniswe_traj_json", [m["content"] for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    for p in run_dir.rglob("*.traj"):
        t = json.loads(p.read_text("utf-8", errors="replace")).get("trajectory") or []
        return "sweagent_traj", [s.get("action") or "" for s in t]
    calls = sorted(run_dir.rglob("tensorblock__*.json"), key=lambda p: float(p.stem.split("-")[-1]))
    if calls:
        return "openhands_llm_calls", [p.read_text("utf-8", errors="replace")[-2000:] for p in calls]
    return None, []


def ref_command(ref):
    c = (ref or {}).get("content") or ""
    try:
        d = json.loads(c)
        return ((d.get("args") or {}).get("command") or d.get("message") or "").strip()
    except ValueError:
        return c.strip()


def main(src, sample_root, manifest, out):
    parsers = load_upstream(src)
    rows = {r["traj_id"]: r for r in pq.read_table(manifest).to_pylist()}
    report = []
    for arch in sorted(Path(sample_root).glob("x_*")):
        traj_id = arch.name[2:]
        r = rows[traj_id]
        stages = r["incorrect_stages"]
        stages = json.loads(stages) if isinstance(stages, str) else stages or []
        flagged = [s for g in stages for s in (g.get("steps") or [])]
        rec = dict(traj_id=traj_id, agent=r["agent"], manifest_step_count=r["step_count"], flagged=len(flagged))
        trial = [d for d in arch.rglob("*") if d.is_dir() and ((d / "agent-logs").is_dir() or (d / "sessions").is_dir())]
        if trial:
            run = trial[0]
            name = {"mini-SWE-agent": "miniswe", "OpenHands": "openhands", "Terminus2": "terminus2"}[r["agent"]]
            ok = parsers[name].can_parse(run)
            steps = parsers[name].parse(run).steps if ok else []
            rec.update(format="upstream_" + name, can_parse=ok, rebuilt=len(steps))
            actions = {s.step_id: s.action for s in steps}
        else:
            fmt, acts = swe_steps(arch)
            rec.update(format=fmt, rebuilt=len(acts))
            actions = {i + 1: a for i, a in enumerate(acts)}
        checks = []
        for s in flagged:
            cmd = ref_command(s.get("action_ref"))
            if not cmd:
                continue
            got = actions.get(s["step_id"], "")
            checks.append(dict(step_id=s["step_id"], match=cmd[:200] in got or got.strip()[:200] in cmd))
        rec.update(count_match=rec["rebuilt"] == r["step_count"], ref_checks=len(checks), ref_matches=sum(c["match"] for c in checks),
                   unverifiable_refs=len(flagged) - len(checks), ref_detail=checks)
        # evidence levels; count-only is never training-admissible (user review of f453d30: 0 == 0 passed before)
        if not rec["count_match"]:
            rec["evidence"] = "MISMATCH_OR_MISSING"
        elif rec["ref_checks"] and rec["ref_matches"] < rec["ref_checks"]:
            rec["evidence"] = "REF_CONFLICT"
        elif rec["ref_checks"] and not rec["unverifiable_refs"]:
            rec["evidence"] = "ACTION_VERIFIED"
        else:
            rec["evidence"] = "COUNT_MATCH_ONLY"
        rec["admissible"] = rec["evidence"] == "ACTION_VERIFIED"
        report.append(rec)
        print({k: v for k, v in rec.items() if k != "ref_detail"})
    Path(out).write_text(json.dumps(dict(codetracer_commit="2d302191dd07e7c0c2da6f7a5e9451c7cbb62d34", runs=report), indent=1), "utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
