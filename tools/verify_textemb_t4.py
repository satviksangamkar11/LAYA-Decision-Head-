"""Verification of the T4 state-free text embeddings after download (amendment e1/e2/e6) + informational comparison with the 3050 bf16 embeddings for the same texts.
    PYTHONPATH=. .venv/Scripts/python.exe tools/verify_textemb_t4.py [--t4 data/v3/textemb_t4/textemb_t4] [--texts kaggle_bundle/textemb/data/texts_trainval.json] [--ref data/v3/textemb]
Create-only: writes results/raw/textemb-t4-verify-<timestamp>.json. No threshold beyond the registered ones (the 0.9999 batch-vs-alone gate ran on Kaggle; cosine vs 3050 is informational)."""
import os, json, glob, hashlib, argparse, datetime
os.environ.setdefault("ATEN_CPU_CAPABILITY", "avx2")
import torch
import torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("--t4", default="data/v3/textemb_t4/textemb_t4")
ap.add_argument("--texts", default="kaggle_bundle/textemb/data/texts_trainval.json")
ap.add_argument("--ref", default="data/v3/textemb")
args = ap.parse_args()
texts = json.load(open(args.texts, encoding="utf-8"))
dev = json.load(open(os.path.join(args.t4, "textemb-device.json")))
shards = sorted(glob.glob(os.path.join(args.t4, "shard-*.pt")))
emb, bad, hashes = {}, [], {}
for f in shards:
    hashes[os.path.basename(f)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    d = torch.load(f)
    for k, v in d.items():
        if v.shape != (3, 2560) or v.dtype != torch.float16 or not torch.isfinite(v).all() or v.float().abs().max() >= 60000:
            bad.append(k[:40])
        emb[k] = v
out = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), device_sidecar=dev, shards=len(shards), texts_embedded=len(emb), texts_expected=len(texts),
           missing=len(set(texts) - set(emb)), extra=len(set(emb) - set(texts)), bad_vectors=len(bad),
           texts_sha256_matches_sidecar=hashlib.sha256(json.dumps(texts).encode()).hexdigest() == dev["texts_sha256"], shard_sha256=hashes,
           lock_ok=dict(torch=dev["torch"] == "2.10.0+cu128", transformers=dev["transformers"] == "5.17.0", device="T4" in dev["device"], dtype=dev["dtype"] == "fp16"))
ref = {}
for f in sorted(glob.glob(os.path.join(args.ref, "shard-*.pt"))):
    ref.update(torch.load(f, map_location="cpu"))
common = [t for t in texts if t in ref and t in emb]
cs = torch.stack([F.cosine_similarity(emb[t].double(), ref[t].double(), dim=-1) for t in common])      # [N, 3]
below = (cs < 0.999).sum(0).tolist()
out["vs_3050_bf16"] = dict(texts_compared=len(common), per_depth_min=[round(x, 6) for x in cs.min(0).values.tolist()], per_depth_mean=[round(x, 6) for x in cs.mean(0).tolist()],
                           below_0p999_per_depth=below, note="informational; no cross-device threshold is registered")
p = f"results/raw/textemb-t4-verify-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
assert not os.path.exists(p)
json.dump(out, open(p, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "shard_sha256"}, indent=1)); print("wrote", p)
