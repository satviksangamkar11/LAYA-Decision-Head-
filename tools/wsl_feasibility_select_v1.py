"""WSL feasibility check v1: task selection (rule in results/raw/wsl-feasibility-v1-*/SELECTION.md). Usage: select_v1.py <out_dir>"""
import glob, json, random, re, sys, time, urllib.parse, urllib.request
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

SEED = 20261014
out = Path(sys.argv[1])
t = pq.read_table("data/nebius_swe_rebench/trajectories.parquet", columns=["instance_id", "repo", "resolved"])
runs = defaultdict(set)
repo_of = {}
for i, r, s in zip(*(t[c].to_pylist() for c in ("instance_id", "repo", "resolved"))):
    runs[i].add(s); repo_of[i] = r
mixed = sorted(i for i, v in runs.items() if v == {0, 1})
rootse = " ".join(p.lower() for p in glob.glob("C:/Users/Satvik/AppData/Local/Temp/rootse_apb/rootse/data/**/*", recursive=True))
ctb = " ".join((json.loads(l).get("traj_id", "") + " " + str(json.loads(l).get("artifact_path", ""))).lower()
               for l in open("results/raw/ctb-k1-inventory-20261014T2246Z-full/RUNS.jsonl", encoding="utf-8"))
laya = set()
for f in glob.glob("data/coding-decision-cases/data/*.parquet"):
    d = pq.read_table(f, columns=["source", "group"])
    m = pc.match_substring(d["source"], "nebius/SWE-agent")
    laya |= {g.lower() for g in d.filter(m)["group"].to_pylist() if g}
def excl(i):
    k = repo_of[i].replace("/", "__").lower()
    return "rootse" if k in rootse else "codetracebench" if k in ctb else "laya" if k in laya else None
ex = {i: excl(i) for i in mixed}
elig = [i for i in mixed if not ex[i]]
random.Random(SEED).shuffle(elig)
stats = dict(mixed=len(mixed), excluded={k: sum(v == k for v in ex.values()) for k in ("rootse", "codetracebench", "laya")}, eligible=len(elig),
             rootse_names=rootse.count("instance") + rootse.count(".json"), laya_groups=len(laya))

def fetch(i):
    for split in ("test", "filtered"):
        q = urllib.parse.urlencode(dict(dataset="nebius/SWE-rebench", config="default", split=split, where="\"instance_id\"='%s'" % i, length=1))
        for k in range(8):  # the filter endpoint sometimes times out: retry
            try:
                rows = json.load(urllib.request.urlopen("https://datasets-server.huggingface.co/filter?" + q, timeout=180))["rows"]
                break
            except OSError:
                if k == 7: raise
                time.sleep(30)
        if rows:
            return split, rows[0]["row"]
    return None, None

TEXT = re.compile(r"\.(py|rst|md|txt|cfg|toml|ini|in)$")
def check(row, taken):
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

acc, taken = [], set()
with open(out / "fetched.jsonl", "a", encoding="utf-8") as log:
    for i in elig:
        if repo_of[i] in taken: continue  # same repo as an accepted task: not fetched
        split, row = fetch(i)
        why = "not_found" if row is None else check(row, taken)
        log.write(json.dumps(dict(instance_id=i, repo=repo_of[i], split=split, reject=why, install_config=(row or {}).get("install_config"))) + "\n")
        time.sleep(0.5)
        if why is None:
            acc.append(i); taken.add(row["repo"])
            (out / ("task-%s.json" % i)).write_text(json.dumps(dict(split=split, **row)), encoding="utf-8")
            if len(acc) == 4: break
sel = dict(stats, accepted=acc[:2], reserves=acc[2:], seed=SEED)
for i in acc:
    rid = pq.read_table("data/nebius_swe_rebench/trajectories.parquet", columns=["trajectory_id", "resolved"],
                        filters=[("instance_id", "=", i), ("resolved", "=", 1)])["trajectory_id"].to_pylist()
    sel.setdefault("run", {})[i] = random.Random(SEED).choice(sorted(rid))
(out / "selection.json").write_text(json.dumps(sel, indent=1))
print(json.dumps(sel, indent=1))
