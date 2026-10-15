"""WSL feasibility check v2: task selection + prefix extraction (rule: results/raw/wsl-feasibility-v2-*/SELECTION_v2.md).
Usage: select_v2.py <v2_out_dir> <v1_out_dir>. Writes fetched_v2.jsonl, task-*.json, prefix-*.json, selection_v2.json."""
import glob, json, random, re, sys, time, urllib.parse, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).parent))
from wsl_feasibility_traj_v1 import load

SEED = 20261014
P = "data/nebius_swe_rebench/trajectories.parquet"
out, v1 = Path(sys.argv[1]), Path(sys.argv[2])

# ---------------- universe + repo exclusions (verbatim v1)
t = pq.read_table(P, columns=["instance_id", "repo", "resolved"])
runs, repo_of = defaultdict(set), {}
for i, r, s in zip(*(t[c].to_pylist() for c in ("instance_id", "repo", "resolved"))):
    runs[i].add(s); repo_of[i] = r
mixed = sorted(i for i, v in runs.items() if v == {0, 1})
rootse = " ".join(p.lower() for p in glob.glob("C:/Users/Satvik/AppData/Local/Temp/rootse_apb/rootse/data/**/*", recursive=True))
ctb = " ".join((json.loads(l).get("traj_id", "") + " " + str(json.loads(l).get("artifact_path", ""))).lower()
               for l in open("results/raw/ctb-k1-inventory-20261014T2246Z-full/RUNS.jsonl", encoding="utf-8"))
laya = set()
for f in glob.glob("data/coding-decision-cases/data/*.parquet"):
    d = pq.read_table(f, columns=["source", "group"])
    laya |= {g.lower() for g in d.filter(pc.match_substring(d["source"], "nebius/SWE-agent"))["group"].to_pylist() if g}
def excl(i):
    k = repo_of[i].replace("/", "__").lower()
    return "rootse" if k in rootse else "codetracebench" if k in ctb else "laya" if k in laya else None
ex = {i: excl(i) for i in mixed}
elig = [i for i in mixed if not ex[i]]
random.Random(SEED).shuffle(elig)

# ---------------- v1 row checks (verbatim) + v2 exclusions
TEXT = re.compile(r"\.(py|rst|md|txt|cfg|toml|ini|in)$")
def check_v1(row, taken):
    ic = row.get("install_config") or {}
    files = re.findall(r"^diff --git a/(\S+)", (row.get("patch") or "") + "\n" + (row.get("test_patch") or ""), re.M)
    if row["repo"] in taken: return "repo_taken"
    if not ic.get("python"): return "no_python_version"
    if ic.get("pre_install"): return "pre_install:%s" % ic["pre_install"][:3]
    if ic.get("env_yml_path"): return "env_yml"
    if ic.get("packages") not in (None, "", "requirements.txt"): return "packages:%s" % ic["packages"]
    inst = ic.get("install") or ""
    if re.search(r"\b(apt|apt-get|conda|make|docker|sudo|yum|brew)\b", inst) or not re.match(r"^\s*(python[0-9.]* -m pip|pip)\b", inst): return "install:%s" % inst[:120]
    if not files or not all(TEXT.search(f) for f in files): return "non_python_files:%s" % [f for f in files if not TEXT.search(f)][:5]
    if not row.get("FAIL_TO_PASS"): return "no_fail_to_pass"
    return None

HEAVY = re.compile(r"(?i)(^|[\s\[,'\"])(torch|torchvision|torchaudio|tensorflow[\w-]*|tf-nightly|jax|jaxlib|nvidia-[\w-]+|cuda[\w-]*|cupy[\w-]*)\s*([=<>~!@\[;,'\"]|$)|\+cu\d{2,3}\b")
def reqs_of(row):  # same filter as the v1 runner
    return [l for l in (row.get("requirements") or "").splitlines() if l.strip() and not re.search(r"@ file:|^-e |git\+|^#", l)]
def check_x1(row):
    ic = row.get("install_config") or {}
    req = reqs_of(row)
    hits = [x for x in req + [ic.get("install") or ""] + list(ic.get("pip_packages") or []) if HEAVY.search(x)]
    if hits: return "x1_heavy:%s" % hits[:4]
    if len(req) > 80: return "x1_over_80:%d" % len(req)
    return None

NET = re.compile(r"\b(pip3?|uv|poetry|conda|mamba)\b[^\n;&|]*\b(install|sync|add)\b|\bcurl\b|\bwget\b|\bgit\s+(clone|fetch|pull)\b|\bapt(-get)?\b")
TEST = re.compile(r"pytest|py\.test|unittest|\btox\b|\bnox\b|runtests|run_tests|manage\.py test|make test|python[0-9.]* \S*test\S*\.py")
EDIT = {"create", "str_replace", "insert", "undo_edit"}
def prefix_of(tid, iid):  # verbatim v1 prep rule
    row = load(tid)
    assert row["instance_id"] == iid and row["resolved"] == 1
    tr = row["trajectory"]
    resp = {m["tool_call_id"]: m["content"] for m in tr if m["role"] == "tool"}
    steps, decision = [], None
    for m in tr:
        for c in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
            name, args = c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")
            is_edit = name == "str_replace_editor" and args.get("command") in EDIT
            is_test = name == "execute_bash" and bool(TEST.search(args.get("command") or ""))
            if is_edit or is_test:
                decision = dict(index=len(steps), name=name, arguments=args, kind="edit" if is_edit else "test")
                break
            steps.append(dict(index=len(steps), name=name, arguments=args, recorded=resp.get(c["id"])))
        if decision:
            break
    ws_dir = re.search(r"<uploaded_files>\s*(\S+)\s*</uploaded_files>", next(m["content"] for m in tr if m["role"] == "user")).group(1)
    return steps, decision, ws_dir
def run_of(i):
    rid = pq.read_table(P, columns=["trajectory_id", "resolved"], filters=[("instance_id", "=", i), ("resolved", "=", 1)])["trajectory_id"].to_pylist()
    return random.Random(SEED).choice(sorted(rid))

def fetch(i):
    for split in ("test", "filtered"):
        q = urllib.parse.urlencode(dict(dataset="nebius/SWE-rebench", config="default", split=split, where="\"instance_id\"='%s'" % i, length=1))
        for k in range(8):
            try:
                rows = json.load(urllib.request.urlopen("https://datasets-server.huggingface.co/filter?" + q, timeout=180))["rows"]
                break
            except OSError:
                if k == 7: raise
                time.sleep(30)
        if rows:
            return split, rows[0]["row"]
    return None, None

v1_reject = {json.loads(l)["instance_id"]: json.loads(l)["reject"] for l in open(v1 / "fetched.jsonl", encoding="utf-8")}
v1_rows = {json.loads(p.read_text(encoding="utf-8"))["instance_id"]: json.loads(p.read_text(encoding="utf-8")) for p in v1.glob("task-*.json")}
acc, taken, counts = [], set(), Counter()
with open(out / "fetched_v2.jsonl", "a", encoding="utf-8") as log:
    for i in elig:
        if repo_of[i] in taken: continue
        rec = dict(instance_id=i, repo=repo_of[i])
        if i in v1_rows:
            row, rec["source"] = v1_rows[i], "v1_full_row"
            why = check_v1(row, taken)
        elif i in v1_reject and v1_reject[i]:
            row, why, rec["source"] = None, v1_reject[i], "v1_reject_reused"
        else:
            split, row = fetch(i)
            rec["source"] = "fetched_v2:%s" % split
            why = "not_found" if row is None else check_v1(row, taken)
            time.sleep(0.5)
        stage = "v1" if why else None
        if why is None:
            why = check_x1(row); stage = "x1" if why else None
        if why is None:
            tid = run_of(i)
            steps, dec, ws = prefix_of(tid, i)
            net = [s["arguments"].get("command") for s in steps if s["name"] == "execute_bash" and NET.search(s["arguments"].get("command") or "")]
            why = "no_decision_point" if dec is None else ("x2_network_prefix:%s" % [n[:100] for n in net]) if net else None
            stage = "x2" if why else None
            rec.update(trajectory_id=tid, prefix_steps=len(steps), decision=dec and (dec["kind"], dec["index"]))
        key = "accepted" if why is None else ("docker_or_system_packages" if stage == "v1" and re.match(r"pre_install|install:", why) else
                                              "v1_other" if stage == "v1" else stage)
        counts[key] += 1
        rec.update(reject=why, stage=stage, install_config=(row or {}).get("install_config"), pinned=len(reqs_of(row)) if row else None)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(i, key, (why or "")[:150])
        if why is None:
            acc.append(i); taken.add(row["repo"])
            (out / ("task-%s.json" % i)).write_text(json.dumps(row), encoding="utf-8")
            (out / ("prefix-%s.json" % i)).write_text(json.dumps(dict(instance_id=i, trajectory_id=tid, workspace_dir=ws, prefix=steps, decision=dec, task=row)), encoding="utf-8")
            if len(acc) == 4: break
sel = dict(mixed=len(mixed), excluded_repo={k: sum(v == k for v in ex.values()) for k in ("rootse", "codetracebench", "laya")}, eligible=len(elig),
           visited=sum(counts.values()), walk_counts=dict(counts), accepted=acc[:2], reserves=acc[2:], seed=SEED)
(out / "selection_v2.json").write_text(json.dumps(sel, indent=1))
print(json.dumps(sel, indent=1))
