"""Pack the arm-E LAYA kernels + their bundle dataset (local packing only; NOTHING is uploaded, no model or torch is run). Stdlib only. Create-only.
Usage: python tools/pack_el_laya_gen_v1.py
Outputs:  kaggle/evidence_ladder_laya_parity_v1/  (gate only)   kaggle/evidence_ladder_laya_gen_v1/  (gate, then generate + capture + score)
          results/raw/el-laya-gen-bundle-<UTC>/  (dataset: code, 3 adapters, 31 pretrained tensors, stored TRAIN/VAL features, their request texts, label projection, gate thresholds)
Bundle content rule: TRAIN pools of laya-fullaction-set-v3 and the 53 canonical-VAL pools only (both were already uploaded to Kaggle for the original captures), public task prompts,
Apache-2.0 head tensors, adapters. NOT included: CAL / TEST / LOCKED data, protected cases, hidden grades, keys, the S/X context variants, judge reviews."""
import gzip, hashlib, json, re, shutil, struct, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results/raw"
UTC = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
BD = R / ("el-laya-gen-bundle-" + UTC)
KG, KP = ROOT / "kaggle/evidence_ladder_laya_gen_v1", ROOT / "kaggle/evidence_ladder_laya_parity_v1"
BD.mkdir(exist_ok=False); KG.mkdir(exist_ok=False); KP.mkdir(exist_ok=False)
OWNER = "sangeetasangamkar"   # the default Kaggle CLI account; never satviksangamkar54321


def sha(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    x = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""): x.update(ch)
    return x.hexdigest()
def read_gz(p): return [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8") if l.strip()]


files = {}   # published path -> bytes (small) or Path (large, copied)
def put(path, src):
    files[path] = Path(src) if isinstance(src, (Path, str)) and not isinstance(src, bytes) else src

BL = json.loads((R / "laya-baseline-v1-20261012T215242Z/MANIFEST.json").read_text("utf-8"))
# ---- code (all hash-pinned in the manifest)
for k in ("evidence_ladder/__init__.py", "evidence_ladder/common/__init__.py", "evidence_ladder/common/interfaces.py", "evidence_ladder/models/__init__.py",
          "evidence_ladder/models/_util.py", "evidence_ladder/models/qwen.py", "evidence_ladder/models/qwen_v2b.py", "evidence_ladder/models/laya_forward.py",
          "capture/capture.py", "capture/capture_resident.py", "decision_head/head.py", "configs/laya_head_gate_v1.toml"):
    put(k, ROOT / k)
put("capture/__init__.py", b""); put("decision_head/__init__.py", b"")
assert sha_file(ROOT / "decision_head/head.py") == BL["code"]["decision_head/head.py"], "head.py differs from the frozen baseline"
used_pkg = {}
for run in ("level12-run-v2-cancelled-20261011T175252Z", "v3-context-capture-run-20261012T065106Z", "val-capture-run-20261012T111837Z"):
    for n in ("capture.py", "capture_resident.py"):
        used_pkg["%s/%s" % (run, n)] = sha_file(R / run / "pkg/capture" / n) == sha_file(ROOT / "capture" / n)
assert all(used_pkg.values()), used_pkg   # the repo capture code IS the code that produced the TRAIN and VAL features

# ---- public tasks (copy of the existing public bundle; its own manifest hash is checked)
GB = sorted(R.glob("el-gpu-gen-bundle-*"))[-1]; gman = json.loads((GB / "BUNDLE_MANIFEST.json").read_text("utf-8"))
assert sha_file(GB / "tasks_public.jsonl") == gman["files"]["tasks_public.jsonl"]
put("tasks_public.jsonl", GB / "tasks_public.jsonl")
for ln in (GB / "tasks_public.jsonl").read_text("utf-8").splitlines():
    assert set(json.loads(ln)) == {"task_id", "language", "prompt", "public_tests", "difficulty"}, "unexpected task field"

# ---- adapters + pretrained subset (31 tensors, byte-copied, dtype untouched)
for n, a in BL["adapters"].items():
    if isinstance(a, dict) and "path" in a:
        assert sha_file(ROOT / a["path"]) == a["sha256"], n
        put("models/%s.pt" % n, ROOT / a["path"])
src = ROOT / "models/coding-decision-agent/model.safetensors"
assert sha_file(src) == BL["pretrained_head"]["sha256"]
with open(src, "rb") as f:
    (hl,) = struct.unpack("<Q", f.read(8)); hdr = json.loads(f.read(hl)); base = 8 + hl
    keep = sorted(k for k in hdr if k != "__metadata__" and (k.startswith("head.layers.") or k.startswith("scorer.") or k == "type_emb.weight"))
    assert len(keep) == 31, len(keep)
    new, blobs, off, per = {}, [], 0, {}
    for k in keep:
        s, e = hdr[k]["data_offsets"]; f.seek(base + s); b = f.read(e - s); assert len(b) == e - s
        new[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"], "data_offsets": [off, off + len(b)]}; off += len(b); blobs.append(b); per[k] = sha(b)
    h2 = json.dumps(new, separators=(",", ":")).encode(); h2 += b" " * ((8 - len(h2) % 8) % 8)
    put("models/pretrained_31.safetensors", struct.pack("<Q", len(h2)) + h2 + b"".join(blobs))
# ---- stored features and the request texts they were captured from
RUNS = {"v2": ("level12-run-v2-cancelled-20261011T175252Z", "level12-bundle-20261011T175011Z"), "v3": ("v3-context-capture-run-20261012T065106Z", "level12-bundle-20261012T060047Z"),
        "val": ("val-capture-run-20261012T111837Z", "level12-bundle-20261012T104251Z")}
for nm, (run, bun) in RUNS.items():
    put("gate/feat_%s_B.pt" % nm, R / run / "out/level1/B.pt")
    reqs = read_gz(R / bun / "upload/requests_level1.jsonl.gz")
    put("gate/req_%s.jsonl" % nm, ("\n".join(json.dumps({k: r[k] for k in ("decision_id", "context_text", "options", "context_sha256", "option_sha256", "order_sha256")}, ensure_ascii=False)
                                           for r in reqs) + "\n").encode("utf-8"))
    if nm == "v3": sysblock = reqs[0]["context_text"].split("\n\n[user]\n")[0]
put("gate/system_block.txt", sysblock.encode("utf-8"))
assert sha(sysblock.encode("utf-8")) == "9bf0352e5debcef4997f0fe2b514a408b072c61e83a87249c7aeb4fdaa5d7250"
# ---- label projection (trajectory, reference index, status, option order) -- TRAIN set v3 and the VAL set only
lab = {"v3": {}, "val": {}}
for key, d, want in (("v3", R / "laya-fullaction-set-v3-20261012T055420Z", "train"), ("val", R / "laya-fullaction-val-v1-20261012T104239Z", "val")):
    rq = {r["decision_id"]: r["options"] for r in read_gz(d / "requests.jsonl.gz")}
    for p in read_gz(d / "private_labels.jsonl.gz"):
        assert p["split"] == want, ("unexpected split", key, p["decision_id"], p["split"])   # no CAL / TEST row can enter
        lab[key][p["decision_id"]] = {"trajectory": p["trajectory_id"], "reference_index": p["reference_index"], "status": p["status"], "options": rq[p["decision_id"]]}
put("gate/labels.json", json.dumps(lab, ensure_ascii=False).encode("utf-8"))
vr = json.loads(next((R / "val-eval-v1-20261012T082055Z").glob("REPORT-*.json")).read_text("utf-8"))
rec = {"decision_id": vr["per_pool"]["decision_id"], "head_real": vr["per_pool"]["head_real"], "probe_real": vr["per_pool"]["probe_real"], "head_top1": vr["question_1"]["head_real_context"]["top1"],
       "seed_top1": vr["question_1"]["each_seed"], "probe_top1": vr["linear_probe"]["real_context"]["top1"], "probe_C": vr["probe_C"], "train_pools": vr["train_pools"],
       "source": "results/raw/val-eval-v1-20261012T082055Z/" + next((R / "val-eval-v1-20261012T082055Z").glob("REPORT-*.json")).name}
assert len(rec["decision_id"]) == 53 and set(rec["decision_id"]) <= set(lab["val"]) and rec["train_pools"] == 334
put("gate/recorded.json", json.dumps(rec).encode("utf-8"))

# ---- manifest, leak scan, write
sys.path.insert(0, str(ROOT))
from evidence_ladder.models import qwen as Q, qwen_v2b as V
man_files = {}
for k, v in files.items():
    data = v if isinstance(v, bytes) else None
    man_files[k] = sha(data) if data is not None else sha_file(v)
    p = BD / k; p.parent.mkdir(parents=True, exist_ok=True)
    if data is not None: p.write_bytes(data)
    else: shutil.copyfile(v, p)
    if k.endswith((".py", ".json", ".jsonl", ".toml", ".txt")):
        t = p.read_text("utf-8", errors="replace")
        assert not re.search(r"nvapi-[A-Za-z0-9_-]{20,}|\.secrets/|kaggle\.json", t), "secret-like text in " + k
manifest = {"bundle": "evidence_ladder_laya_gen_v1", "qwen_revision": Q.QWEN_REVISION, "template_hash_qwen_sample_v2": Q.TEMPLATE_HASHES["qwen_sample_v2"],
            "template_hash_qwen_sample_v2b": V.TEMPLATE_HASH, "smoke_prompt_sha256": gman["smoke_prompt_sha256"], "tasks": 80, "files": man_files,
            "pretrained_31_tensor_sha256": per, "pretrained_source_sha256": BL["pretrained_head"]["sha256"], "capture_code_equals_run_pkg": used_pkg,
            "adapter_sha256": {n: a["sha256"] for n, a in BL["adapters"].items() if isinstance(a, dict) and "sha256" in a},
            "note": "TRAIN (set v3) and canonical-VAL Fable pools, public task prompts, adapters, head tensors. No CAL/TEST/LOCKED, no protected cases, no keys."}
(BD / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), "utf-8")
(BD / "dataset-metadata.json").write_text(json.dumps({"title": "evidence-ladder-laya-gen-bundle-v1", "id": OWNER + "/evidence-ladder-laya-gen-bundle-v1", "licenses": [{"name": "other"}]}, indent=2), "utf-8")

script = (ROOT / "kaggle/evidence_ladder_laya_gen_v1.py").read_text("utf-8")
assert script.count('MODE = "full"') == 1
srcs = ["leminhhung0101/qwen3-4b-thinking-2507", OWNER + "/evidence-ladder-laya-gen-bundle-v1"]
for kd, name, text in ((KG, "evidence-ladder-laya-gen-v1", script), (KP, "evidence-ladder-laya-parity-v1", script.replace('MODE = "full"', 'MODE = "parity"'))):
    code = name.replace("-", "_") + ".py"
    (kd / code).write_text(text, "utf-8")
    (kd / "kernel-metadata.json").write_text(json.dumps({"id": OWNER + "/" + name, "title": name, "code_file": code, "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": srcs, "kernel_sources": [], "competition_sources": []}, indent=2), "utf-8")
cmds = ["# run from D:/local model with the DEFAULT Kaggle account (sangeetasangamkar); every call under timeout; NOTHING below has been run",
        "timeout 600 kaggle datasets create -p %s --dir-mode zip" % BD.relative_to(ROOT).as_posix(),
        "timeout 120 kaggle kernels push -p kaggle/evidence_ladder_laya_parity_v1    # parity gate only (run this first)",
        "timeout 120 kaggle kernels push -p kaggle/evidence_ladder_laya_gen_v1       # shard 0/1; for two shards set EL_TASK_SHARD in a copy of the script (0/2 and 1/2)",
        "timeout 60 kaggle kernels status %s/evidence-ladder-laya-parity-v1" % OWNER]
(BD / "PUSH_COMMANDS.txt").write_text("\n".join(cmds) + "\n", "utf-8")
print(json.dumps({"bundle": str(BD), "files": {k: (BD / k).stat().st_size for k in sorted(files)}, "total_mb": round(sum((BD / k).stat().st_size for k in files) / 1e6, 1),
                  "kernels": [str(KG), str(KP)], "commands": cmds, "capture_code_equals_run_pkg": used_pkg}, indent=1))
