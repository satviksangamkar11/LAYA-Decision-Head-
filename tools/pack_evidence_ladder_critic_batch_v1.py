"""Pack the critic BATCH kernel and its packet dataset locally. NOTHING is uploaded or pushed here (a push needs the user's go). Create-only.
Usage: python tools/pack_evidence_ladder_critic_batch_v1.py <packet_dir>       (a dir written by critic_batch.batch.export_for_kernel / validator.write_packets)
       python tools/pack_evidence_ladder_critic_batch_v1.py --synthetic         (one synthetic fixture packet, to check the packaging only)
Output: results/raw/evidence-ladder-critic-batch-pack-<UTC>/{dataset/, kernel/, PACK.json}. The dataset holds ONLY: packets.jsonl, PACKETS_MANIFEST.json,
CODE_MANIFEST.json, dataset-metadata.json and a fixed list of code files (no tests, no execution/grading code, no hidden or protected data, no keys)."""
import hashlib, json, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_ladder.critic_batch import validator as V   # noqa: E402
from evidence_ladder.models.broker import packet_id_of    # noqa: E402

DATASET_ID = "sangeetasangamkar/evidence-ladder-critic-packets-v1"
CODE = ["evidence_ladder/__init__.py", "evidence_ladder/common/__init__.py", "evidence_ladder/common/contract_import.py", "evidence_ladder/common/interfaces.py",
        "evidence_ladder/models/__init__.py", "evidence_ladder/models/_util.py", "evidence_ladder/models/broker.py", "evidence_ladder/models/glm.py",
        "evidence_ladder/models/pc_sender.py", "evidence_ladder/models/kaggle_keys.py",
        "evidence_ladder/critic_batch/__init__.py", "evidence_ladder/critic_batch/validator.py", "evidence_ladder/critic_batch/config.py",
        "evidence_ladder/critic_batch/batch.py", "evidence_ladder/critic_batch/transport.py",
        "configs/critic_concurrency_v1.toml", "specs/evidence_ladder_v1/schemas_v1.json", "specs/evidence_ladder_v1/contract/contract_lib_v1.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if not arg:
        sys.exit(__doc__)
    out = ROOT / "results/raw" / ("evidence-ladder-critic-batch-pack-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ds, kn = out / "dataset", out / "kernel"
    ds.mkdir(parents=True, exist_ok=False)
    kn.mkdir()
    if arg == "--synthetic":
        req = json.loads((ROOT / "evidence_ladder/models/fixtures/glm_responses.json").read_text(encoding="utf-8"))["request"]
        V.write_packets(ds, {packet_id_of(req): req})
    else:
        shutil.copyfile(Path(arg) / "packets.jsonl", ds / "packets.jsonl")
        shutil.copyfile(Path(arg) / "PACKETS_MANIFEST.json", ds / "PACKETS_MANIFEST.json")
    acc, rej = V.read_packets(ds)                                   # re-validates every packet and the manifest hash
    if rej:
        shutil.rmtree(out)
        sys.exit("refusing to pack: %d rejected packets: %s" % (len(rej), [(r["packet_id"], r["class"]) for r in rej]))
    files = {}
    for rel in CODE:
        d = ds / "code" / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, d)
        files["code/" + rel] = sha(d)
    (ds / "CODE_MANIFEST.json").write_text(json.dumps({"bundle": "evidence_ladder_critic_code_v1", "files": files}, indent=1, sort_keys=True), encoding="utf-8")
    (ds / "dataset-metadata.json").write_text(json.dumps({"title": "evidence-ladder-critic-packets-v1", "id": DATASET_ID, "licenses": [{"name": "other"}]}, indent=1))
    bad = V.check_dataset_dir(ds)
    meta = json.loads((ROOT / "kaggle/evidence_ladder_critic_v2/kernel-metadata.json").read_text(encoding="utf-8"))
    V.check_kernel_metadata(meta, DATASET_ID)
    if bad:
        shutil.rmtree(out)
        sys.exit("dataset allow-list violations: %s" % bad)
    shutil.copyfile(ROOT / "kaggle/evidence_ladder_critic_v2/evidence-ladder-critic-v1.py", kn / meta["code_file"])
    (kn / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    pack = {"out": str(out), "packets": len(acc), "dataset_id": DATASET_ID, "kernel_id": meta["id"], "dataset_sources": meta["dataset_sources"],
            "code_manifest_sha256": sha(ds / "CODE_MANIFEST.json"), "packets_manifest_sha256": sha(ds / "PACKETS_MANIFEST.json"),
            "kernel_script_sha256": sha(kn / meta["code_file"]), "pushed": False}
    (out / "PACK.json").write_text(json.dumps(pack, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(pack))


if __name__ == "__main__":
    main()
