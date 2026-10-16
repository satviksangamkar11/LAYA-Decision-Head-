"""Nebius branching feasibility count (user GO, chat 2026-10-13). Kaggle CPU. No model, no training, no generation. Counts only.

Input (mounted, private dataset laya-nebius-branch-v1): nebius_mixed.parquet (the 1,741 tasks of nebius/SWE-rebench-openhands-trajectories with >= 1 resolved
and >= 1 unresolved run; CC-BY-4.0), laya_pretraining_overlap_repos.json (repos of LAYA's pretraining source). Output /kaggle/working/BRANCH_REPORT.json,
BRANCH_EXAMPLES.json. Stage 1 checks parsing on 50 tasks; stage 2 counts all tasks only if stage 1 passes its gate.
"""
import collections
import glob
import hashlib
import json
import os
import random
import re
import time

import pyarrow.parquet as pq

IN, OUT = "/kaggle/input", "/kaggle/working"
find = lambda pat: sorted(glob.glob(os.path.join(IN, "**", pat), recursive=True))
DROP_ARGS = {"is_input", "timeout", "security_risk", "thought"}
PLAN = {
    "written": "2026-10-13, before any branch was counted",
    "approved_by": "user, chat 2026-10-13: GO bounded nebius branching feasibility count on Kaggle CPU; frozen branch definition; small parse check first; "
                   "300 branchpoints / 100 repos is a discovery threshold, not a training gate; stop after the report",
    "state": "same task (same system and user prompt) and the same sequence of executed tool actions with the same observed tool outputs before the "
             "decision. Free-text reasoning (assistant prose and `think` calls) is excluded: it differs between every run and does not change the "
             "workspace. Two matching levels reported: exact outputs, and normalised outputs (ANSI codes, timings, memory addresses, temp paths, "
             "timestamps and whitespace normalised). Three levels, never combined: (1) EXACT observed-history matches = the headline; (2) NORMALISED "
             "matches = exploratory; (3) verified identical WORKSPACE states = 0 available: the dataset has no workspace snapshots, so no branch is a "
             "verified identical environment (same calls and outputs can still hide different unobserved file contents). Task instructions (system and "
             "user prompt) are part of every state",
    "action": "a tool call (name + canonical JSON arguments without is_input / timeout / security_risk / thought; bash commands whitespace-collapsed). "
              "Branchpoint: a state reached by >= 2 runs whose next actions are >= 2 different actions",
    "action_types": "read (editor view; cat / head / tail / sed -n / less / nl), search (grep / rg / find / ls / tree), test (pytest / unittest / tox / "
                    "test scripts), run (other python / node / go / make / script execution), edit (editor create / str_replace / insert / undo; "
                    "sed -i, redirections, patch), vcs (git), env (pip / apt / npm install / conda), stop (finish), plan (task_tracker), other",
    "outcome": "each child action's runs and their final `resolved` flags: OBSERVATIONAL evidence (later steps also matter), never a verified label",
    "evidence tiers": {"single": ">= 2 children with different resolve rates (any run counts)", "repeated": ">= 2 children with >= 2 runs each and "
                       "different resolve rates", "strong": ">= 2 children with >= 3 runs each and resolve rates differing by >= 0.3"},
    "initial_vs_later": "depth 0 (the first executed action after the task prompt) vs depth >= 1",
    "independence": "unique tasks, repos and trajectories counted at every funnel stage; branchpoints inside one task are not independent",
    "overlap": "858 = size of the source-side repo list (coding-decision-cases, nebius/SWE-agent rows); 618 = its recorded overlap with the full nebius "
               "set; the filtered 859 repos are checked against the list here. Overlap is a contamination RISK, not proof an example was in pretraining",
    "stage1_gate": ">= 95% of runs parse and >= 95% of tool calls are matched to a tool output, on the first 50 tasks (sorted by instance id)",
    "discovery_threshold": "EXACT matches: >= 300 branchpoints with >= 2 action TYPES and single-tier outcome evidence, from >= 100 repos (not a training gate)",
}
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def norm_obs(t):
    t = ANSI.sub("", (t or "").replace("\r", ""))
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?\S*", "<TS>", t)
    t = re.sub(r"\b\d+(\.\d+)?\s?(ms|µs|us|ns|s|sec|seconds)\b", "<T>", t)
    t = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", t)
    t = re.sub(r"/tmp/[\w./-]+", "<TMP>", t)
    return re.sub(r"\s+", " ", t).strip()


def canon_action(name, args):
    a = {k: v for k, v in args.items() if k not in DROP_ARGS}
    if name == "execute_bash" and isinstance(a.get("command"), str):
        a["command"] = re.sub(r"\s+", " ", a["command"]).strip()
    return name + " " + json.dumps(a, sort_keys=True, ensure_ascii=False)


def category(name, args):
    if name == "finish":
        return "stop"
    if name == "task_tracker":
        return "plan"
    if name == "str_replace_editor":
        return "read" if args.get("command") == "view" else "edit"
    if name != "execute_bash":
        return "other"
    c = (args.get("command") or "").strip()
    first = c.split()[0] if c.split() else ""
    if re.search(r"\bsed\s+-i|(^|[^2&])>\s*[\w./]|\btee\b|\bpatch\b|git apply", c):
        return "edit"
    if re.search(r"\b(pytest|unittest|tox|nosetests)\b|python\S*\s+\S*test|go test|cargo test|npm (run )?test|make test", c):
        return "test"
    if first in ("git",):
        return "vcs"
    if re.search(r"\b(pip|pip3|apt|apt-get|conda|npm install|yarn add|poetry)\b", c):
        return "env"
    if first in ("grep", "rg", "find", "ls", "tree", "ag", "locate") or re.match(r"(cd \S+ && )?(grep|rg|find|ls)\b", c):
        return "search"
    if first in ("cat", "head", "tail", "less", "more", "nl", "wc") or re.match(r"(cd \S+ && )?(cat|head|tail|sed -n)\b", c):
        return "read"
    if re.search(r"\b(python|python3|node|go run|make|bash|sh|\./)", c):
        return "run"
    return "other"


def parse_run(msgs):
    """-> (root key, steps [(action key, short action, category, obs exact hash, obs norm hash, obs excerpt)], tool-call stats) or (None, reason, stats)."""
    if len(msgs) < 2 or msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
        return None, "no_system_user_prefix", (0, 0)
    root = hashlib.sha256((msgs[0]["content"] or "").encode() + b"\x00" + (msgs[1]["content"] or "").encode()).hexdigest()
    outputs = {m.get("tool_call_id"): (m.get("content") or "") for m in msgs if m["role"] == "tool"}
    steps, calls, matched = [], 0, 0
    for m in msgs[2:]:
        if m["role"] != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            name = c["function"]["name"]
            if name == "think":
                continue
            calls += 1
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:
                return None, "bad_arguments", (calls, matched)
            obs = outputs.get(c.get("id"))
            matched += obs is not None
            obs = obs or ""
            steps.append((canon_action(name, args), canon_action(name, args)[:240], category(name, args),
                          hashlib.sha256(obs.encode()).hexdigest()[:16], hashlib.sha256(norm_obs(obs).encode()).hexdigest()[:16], obs[:240]))
    return root, steps, (calls, matched)


def count(tasks, overlap):
    """tasks: {instance: [(run id, repo, resolved, root, steps)]}. -> funnel + branchpoints per matching level."""
    out = {}
    for level, oi in (("exact", 3), ("normalised", 4)):
        bps = []
        for inst, runs in tasks.items():
            nodes = collections.defaultdict(lambda: collections.defaultdict(list))   # state key -> action -> [run index]
            meta = {}
            for ri, (rid, repo, res, root, steps) in enumerate(runs):
                key = root
                for d, st in enumerate(steps):
                    nodes[key][st[0]].append(ri)
                    meta.setdefault(key, (d, steps[max(0, d - 2):d]))
                    key = hashlib.sha256((key + "|" + st[0] + "|" + st[oi]).encode()).hexdigest()
            for key, acts in nodes.items():
                if len(acts) < 2 or sum(map(len, acts.values())) < 2:
                    continue
                d, hist = meta[key]
                ch = []
                for a, rs in acts.items():
                    cat = next(st[2] for st in runs[rs[0]][4] if st[0] == a)
                    ch.append(dict(action=a[:240], category=cat, runs=len(rs), resolved=sum(runs[r][2] for r in rs), run_ids=[runs[r][0] for r in rs]))
                rates = [c["resolved"] / c["runs"] for c in ch]
                many = lambda k: [c for c in ch if c["runs"] >= k]
                rate_spread = lambda cs: (max(c["resolved"] / c["runs"] for c in cs) - min(c["resolved"] / c["runs"] for c in cs)) if len(cs) >= 2 else 0.0
                bps.append(dict(instance=inst, repo=runs[0][1], depth=d, n_runs=sum(c["runs"] for c in ch), children=ch,
                                types=sorted({c["category"] for c in ch}), multi_type=len({c["category"] for c in ch}) >= 2,
                                single=len(set(rates)) >= 2, repeated=len(many(2)) >= 2 and rate_spread(many(2)) > 0,
                                strong=len(many(3)) >= 2 and rate_spread(many(3)) >= 0.3, overlap=runs[0][1] in overlap,
                                history_tail=[dict(action=h[1], obs=h[5]) for h in hist]))
        u = lambda xs: dict(branchpoints=len(xs), tasks=len({x["instance"] for x in xs}), repos=len({x["repo"] for x in xs}),
                            trajectories=len({r for x in xs for c in x["children"] for r in c["run_ids"]}))
        sel = {"all": bps, "multi_type": [b for b in bps if b["multi_type"]], "same_type": [b for b in bps if not b["multi_type"]]}
        f = {}
        for name, xs in sel.items():
            f[name] = dict(any=u(xs), initial=u([x for x in xs if x["depth"] == 0]), later=u([x for x in xs if x["depth"] > 0]),
                           single=u([x for x in xs if x["single"]]), repeated=u([x for x in xs if x["repeated"]]), strong=u([x for x in xs if x["strong"]]),
                           single_no_laya_overlap=u([x for x in xs if x["single"] and not x["overlap"]]),
                           strong_no_laya_overlap=u([x for x in xs if x["strong"] and not x["overlap"]]))
        f["type_pairs_multi_type_single"] = dict(collections.Counter("+".join(b["types"]) for b in bps if b["multi_type"] and b["single"]).most_common(15))
        f["depth_hist_multi_type_single"] = dict(collections.Counter(min(b["depth"], 10) for b in bps if b["multi_type"] and b["single"]))
        out[level] = dict(funnel=f, _bps=bps)
    return out


def load(limit=None):
    f = pq.read_table(find("nebius_mixed.parquet")[0])
    ids = f.column("instance_id").to_pylist()
    order = sorted(set(ids))
    keep = set(order[:limit]) if limit else set(order)
    tasks, fails, calls, matched, nrun = collections.defaultdict(list), collections.Counter(), 0, 0, 0
    for i in range(f.num_rows):
        if ids[i] not in keep:
            continue
        nrun += 1
        root, steps, (c, m) = parse_run(f.column("trajectory")[i].as_py())
        calls, matched = calls + c, matched + m
        if root is None:
            fails[steps] += 1
            continue
        tasks[ids[i]].append((f.column("trajectory_id")[i].as_py(), f.column("repo")[i].as_py(), bool(f.column("resolved")[i].as_py()), root, steps))
    return tasks, dict(runs=nrun, parsed=nrun - sum(fails.values()), parse_failures=dict(fails), tool_calls=calls, tool_calls_matched=matched,
                       tasks=len(keep), tasks_with_2_parsed_runs=sum(len(v) >= 2 for v in tasks.values()),
                       tasks_with_2_runs_same_prompt=sum(max(collections.Counter(r[3] for r in v).values()) >= 2 for v in tasks.values() if v))


def main():
    t0 = time.time()
    overlap = set(json.load(open(find("laya_pretraining_overlap_repos.json")[0]))["repos"])
    rep = dict(plan=PLAN)
    tasks, integ = load(50)
    integ["parse_rate"] = round(integ["parsed"] / max(integ["runs"], 1), 4)
    integ["tool_call_match_rate"] = round(integ["tool_calls_matched"] / max(integ["tool_calls"], 1), 4)
    integ["gate_passed"] = integ["parse_rate"] >= 0.95 and integ["tool_call_match_rate"] >= 0.95
    c = count(tasks, overlap)
    integ["sample_funnel_exact"] = c["exact"]["funnel"]["multi_type"]["single"]
    rep["stage1"] = integ
    print("STAGE1", json.dumps(integ), flush=True)
    if not integ["gate_passed"]:
        rep["stopped"] = "stage-1 parsing gate failed"
        json.dump(rep, open(os.path.join(OUT, "BRANCH_REPORT.json"), "w"), indent=1)
        return
    tasks, integ2 = load(None)
    integ2["parse_rate"] = round(integ2["parsed"] / max(integ2["runs"], 1), 4)
    integ2["tool_call_match_rate"] = round(integ2["tool_calls_matched"] / max(integ2["tool_calls"], 1), 4)
    rep["stage2_integrity"] = integ2
    c = count(tasks, overlap)
    rep["funnel"] = {lvl: c[lvl]["funnel"] for lvl in c}
    mt = rep["funnel"]["exact"]["multi_type"]["single"]
    rep["verified_identical_workspace_branchpoints"] = 0
    rep["verified_identical_workspace_note"] = "no workspace snapshots in the dataset"
    rep["discovery_threshold_met"] = mt["branchpoints"] >= 300 and mt["repos"] >= 100
    bps = c["exact"]["_bps"]
    rng = random.Random(13)
    pick = lambda pred, k: rng.sample([b for b in bps if pred(b)], min(k, sum(map(pred, bps))))
    ex = (pick(lambda b: b["multi_type"] and b["strong"], 8) + pick(lambda b: not b["multi_type"] and b["repeated"], 5)
          + pick(lambda b: b["multi_type"] and b["single"] and b["depth"] == 0, 6) + pick(lambda b: b["multi_type"] and b["single"] and b["depth"] > 0, 6))
    for b in ex:
        for ch in b["children"]:
            ch.pop("run_ids", None)
    json.dump(ex, open(os.path.join(OUT, "BRANCH_EXAMPLES.json"), "w"), indent=1, ensure_ascii=False)
    rep["examples"] = len(ex)
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(OUT, "BRANCH_REPORT.json"), "w"), indent=1)
    print("FUNNEL multi_type EXACT", json.dumps(rep["funnel"]["exact"]["multi_type"]), flush=True)
    print("FUNNEL multi_type normalised (exploratory)", json.dumps(rep["funnel"]["normalised"]["multi_type"]), flush=True)
    print("THRESHOLD", rep["discovery_threshold_met"], "DONE", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
